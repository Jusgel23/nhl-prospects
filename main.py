"""
NHL Prospect Prediction System - CLI entry point.

Usage:
    python main.py collect [--seasons 2023-2024 2024-2025]
    python main.py process
    python main.py train
    python main.py rank [--draft-year 2025] [--top 50]
    python main.py report --player "Player Name"
    python main.py pipeline  # run collect -> process -> train -> rank in one shot
"""
import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich import print as rprint

# ── logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("nhl_prospects")
console = Console()

# ── imports ───────────────────────────────────────────────────────────────────
from src.data.database import init_db, upsert_players, upsert_seasons, upsert_outcomes
from src.data.database import load_players, load_seasons, load_outcomes, load_predictions
from src.data.database import upsert_predictions
from src.data.scrapers.eliteprospects import scrape_draft_class
from src.data.scrapers.uscho import scrape_ncaa_multiple_seasons
from src.data.loaders.historical import load_draft_outcomes
from src.data.scrapers.hockey_reference import (
    build_hr_id_mapping, scrape_player_career,
)
from src.data.scrapers.nhl_api import (
    enrich_player as nhl_enrich_player, load_id_map as nhl_load_id_map,
    save_id_map as nhl_save_id_map,
)
from src.models.nhle import apply_nhle_to_seasons, build_development_arc, is_excluded_league
from src.models.features import build_feature_matrix
from src.models.predictor import ProspectPredictor
from src.comparables.similarity import build_comparable_index
from src.rankings.ranker import rank_prospects, format_rankings_table, export_csv
from src.rankings.report import generate_report


DEFAULT_SEASONS = ["2022-2023", "2023-2024", "2024-2025"]
RANKINGS_CSV = Path("data/processed/rankings.csv")


def cmd_collect(args):
    seasons = args.seasons or DEFAULT_SEASONS
    league_filter = (getattr(args, "league", None) or "").upper().strip() or None

    console.rule("[bold green]Collecting prospect data")
    if league_filter:
        console.print(f"[yellow]League filter:[/] {league_filter} only")

    init_db()

    # CHL via EliteProspects — skip entirely if filtering to a non-CHL league
    CHL_LEAGUES = {"OHL", "WHL", "QMJHL", "CHL"}
    if league_filter is None or league_filter in CHL_LEAGUES:
        console.print(f"Scraping CHL data for seasons: {seasons}")
        players_df, seasons_df = scrape_draft_class(seasons)
        if league_filter and league_filter != "CHL":
            seasons_df = seasons_df[seasons_df["league"].str.upper() == league_filter]
            keep_ids = set(seasons_df["player_id"])
            players_df = players_df[players_df["player_id"].isin(keep_ids)]
        if not players_df.empty:
            upsert_players(players_df)
            console.print(f"  [green]✓[/] {len(players_df)} players stored")
        if not seasons_df.empty:
            upsert_seasons(seasons_df)
            console.print(f"  [green]✓[/] {len(seasons_df)} season rows stored")

    # NCAA via USCHO — skip if filtering to a non-NCAA league
    if league_filter is None or league_filter == "NCAA":
        console.print("Scraping NCAA data from USCHO...")
        ncaa_df = scrape_ncaa_multiple_seasons(seasons)
        if not ncaa_df.empty:
            upsert_seasons(ncaa_df)
            console.print(f"  [green]✓[/] {len(ncaa_df)} NCAA season rows stored")

    # Historical outcomes — scraped from Hockey Reference, cached locally
    console.print("Fetching historical draft outcomes (1995-2019) from Hockey Reference...")
    console.print("  [dim](~3 min first run, cached after that)[/]")
    outcomes_df = load_draft_outcomes()
    if not outcomes_df.empty:
        upsert_outcomes(outcomes_df)
        console.print(f"  [green]✓[/] {len(outcomes_df)} historical outcomes stored")

    console.print("[bold green]Collection complete.[/]")


def cmd_process(args):
    console.rule("[bold blue]Processing & normalizing data")

    players_df = load_players()
    seasons_df = load_seasons()
    outcomes_df = load_outcomes()

    if seasons_df.empty:
        console.print("[red]No season data found. Run `collect` first.[/]")
        return

    console.print(f"Applying NHLe to {len(seasons_df)} season rows...")
    seasons_df = apply_nhle_to_seasons(seasons_df, players_df)

    if not outcomes_df.empty:
        seasons_df = build_development_arc(seasons_df, outcomes_df)

    # Save processed seasons back. We used to do `if_exists="replace"` here
    # which dropped the schema (UNIQUE constraint + new derived columns after
    # each run), causing duplicate rows and _x/_y column drift. Instead:
    # DELETE all rows then upsert so the CREATE TABLE schema (now extended
    # to carry the derived columns) is preserved.
    from src.data.database import get_conn
    with get_conn() as conn:
        conn.execute("DELETE FROM seasons")
        conn.commit()
    upsert_seasons(seasons_df)

    console.print("[bold blue]Processing complete.[/]")


def cmd_train(args):
    console.rule("[bold magenta]Training prediction models")

    players_df  = load_players()
    seasons_df  = load_seasons()
    outcomes_df = load_outcomes()

    if outcomes_df.empty:
        console.print("[yellow]No cached outcomes — will scrape Hockey Reference. This takes ~3 min...[/]")
        from src.data.loaders.historical import load_draft_outcomes
        outcomes_df = load_draft_outcomes()
        if outcomes_df.empty:
            console.print("[red]Could not retrieve historical outcomes. Check your internet connection.[/]")
            return
        from src.data.database import upsert_outcomes
        upsert_outcomes(outcomes_df)

    features_df = build_feature_matrix(players_df, seasons_df, outcomes_df)
    labeled = features_df[features_df["is_nhler"].notna()]

    # Exclude draft classes whose NHL careers are too fresh to label reliably.
    # A player with only 3 years of elapsed NHL time hasn't had the chance to
    # hit 200 GP even if they're trending toward it — treating them as
    # "not NHLer" would be a false-negative label. Keep the 5-year-minimum
    # cutoff so we train only on settled outcomes.
    from datetime import date
    label_cutoff = date.today().year - 5
    if "draft_year" in labeled.columns:
        before = len(labeled)
        labeled = labeled[
            labeled["draft_year"].fillna(0).astype(int) <= label_cutoff
        ]
        console.print(
            f"[dim]Filtered to settled outcomes: draft_year ≤ {label_cutoff} → "
            f"{len(labeled)} of {before} labeled rows (drops recent drafts whose "
            f"NHL careers are still too young to label).[/]"
        )
    console.print(f"Training on {len(labeled)} labeled historical prospects...")

    predictor = ProspectPredictor()
    predictor.train(labeled)

    console.print("Running walk-forward validation (2005–2015)...")
    predictor.train_walk_forward(labeled, start_year=2005, end_year=2015)

    console.print("[bold magenta]Training complete.[/]")


def cmd_rank(args):
    console.rule("[bold yellow]Generating prospect rankings")
    league_filter = (getattr(args, "league", None) or "").upper().strip() or None

    players_df  = load_players()
    seasons_df  = load_seasons()
    outcomes_df = load_outcomes()

    if players_df.empty:
        console.print("[red]No player data. Run `collect` first.[/]")
        return

    draft_year = args.draft_year
    features_df = build_feature_matrix(
        players_df, seasons_df, outcomes_df if not outcomes_df.empty else None,
        draft_year=draft_year,
    )

    # Narrow to prospects with at least one season in the target league
    if league_filter:
        lg_players = seasons_df[
            seasons_df["league"].str.upper() == league_filter
        ]["player_id"].unique()
        before = len(features_df)
        features_df = features_df[features_df["player_id"].isin(lg_players)]
        console.print(
            f"[yellow]League filter:[/] {league_filter} → {len(features_df)} of {before} prospects retained."
        )
        if features_df.empty:
            console.print(f"[red]No prospects found with {league_filter} seasons.[/]")
            return

    predictor = ProspectPredictor()
    predictions_df = predictor.predict(features_df)

    # Build comparables index if historical data available
    comp_index = None
    if not outcomes_df.empty:
        hist_features = build_feature_matrix(players_df, seasons_df, outcomes_df)
        labeled = hist_features[hist_features["is_nhler"].notna()]
        if not labeled.empty:
            comp_index = build_comparable_index(labeled)

    top_n = args.top or 50
    ranked = rank_prospects(features_df, predictions_df, comp_index)
    ranked = ranked.head(top_n)

    # Save predictions to DB
    if not predictions_df.empty:
        upsert_predictions(predictions_df)

    # Export CSV
    RANKINGS_CSV.parent.mkdir(parents=True, exist_ok=True)
    export_csv(ranked, str(RANKINGS_CSV))

    # Display table
    console.print(format_rankings_table(ranked))
    console.print(f"\n[green]Top {top_n} rankings saved to {RANKINGS_CSV}[/]")


def cmd_report(args):
    if not args.player:
        console.print("[red]Provide a player name: --player 'First Last'[/]")
        return

    console.rule(f"[bold cyan]Report: {args.player}")

    players_df  = load_players()
    seasons_df  = load_seasons()
    outcomes_df = load_outcomes()
    predictions_df = load_predictions()

    if players_df.empty:
        console.print("[red]No data. Run `collect` first.[/]")
        return

    # Find player by name (case-insensitive partial match)
    mask = players_df["name"].str.lower().str.contains(args.player.lower(), na=False)
    matches = players_df[mask]
    if matches.empty:
        console.print(f"[red]Player '{args.player}' not found.[/]")
        return
    if len(matches) > 1:
        console.print(f"[yellow]Multiple matches:[/] {matches['name'].tolist()}")
        console.print("Use a more specific name.")
        return

    player_id = matches.iloc[0]["player_id"]
    features_df = build_feature_matrix(players_df, seasons_df)

    comp_index = None
    if not outcomes_df.empty:
        hist_features = build_feature_matrix(players_df, seasons_df, outcomes_df)
        labeled = hist_features[hist_features.get("is_nhler", pd.Series()).notna()]
        if not labeled.empty:
            comp_index = build_comparable_index(labeled)

    report = generate_report(
        player_id, features_df, seasons_df, predictions_df, comp_index
    )
    console.print(report)


def cmd_backfill_careers(args):
    """
    One-time: scrape Hockey Reference player career pages for every
    historical draftee we've already seen, upserting full bios +
    multi-season career into the DB. Feeds model training with richer
    features (DOB, D-3/D-2/D-1/D0/D+1 trajectory, proper age adjustment).
    """
    console.rule("[bold cyan]Backfilling HR player careers")
    init_db()

    outcomes_df = load_outcomes()
    if outcomes_df.empty:
        console.print("[red]No historical outcomes found. Run `collect` first.[/]")
        return

    # Build (or load from cache) the synthetic_id → hr_player_id mapping.
    start = int(args.start or 1995)
    end = int(args.end or 2019)
    mapping = build_hr_id_mapping(start=start, end=end, force_refresh=args.refresh_mapping)
    console.print(f"HR ID mapping: {len(mapping)} entries")

    # Select which synthetic IDs to scrape careers for
    target_ids = outcomes_df["player_id"].tolist()
    if args.sample:
        import random
        random.seed(42)
        target_ids = random.sample(target_ids, min(args.sample, len(target_ids)))
        console.print(f"[yellow]Sampling[/] {len(target_ids)} of {len(outcomes_df)} players")
    if args.nhlers_only:
        nhler_ids = set(outcomes_df[outcomes_df["is_nhler"] == 1]["player_id"])
        target_ids = [p for p in target_ids if p in nhler_ids]
        console.print(f"[yellow]NHLers only:[/] {len(target_ids)} players")

    # Translate to HR IDs (drop any we couldn't map)
    hr_ids = [mapping.get(p) for p in target_ids]
    hr_ids = [h for h in hr_ids if h]
    missing = len(target_ids) - len(hr_ids)
    if missing:
        console.print(f"[yellow]Could not map[/] {missing} players to HR IDs (rare names / alt spellings)")

    console.print(f"Scraping {len(hr_ids)} HR career pages "
                  f"(~{len(hr_ids) * 3.5 / 60:.0f} min at 3.5s/player, resumable via cache)")

    # Loop and upsert per-player so crashes preserve progress
    scraped = 0
    import pandas as pd
    for i, hr_id in enumerate(hr_ids, 1):
        career = scrape_player_career(hr_id)
        if career is None:
            continue

        # Upsert bio row — use hr_player_id as the canonical player_id for
        # historical data. This lines up with outcomes table keys.
        syn_id = next((s for s, h in mapping.items() if h == hr_id), hr_id)
        bio_row = {
            "player_id":  syn_id,            # matches outcomes table
            "name":       career.get("name"),
            "dob":        career.get("dob"),
            "nationality":career.get("nationality"),
            "position":   career.get("position"),
            "height_cm":  career.get("height_cm"),
            "weight_kg":  career.get("weight_kg"),
            "shoots":     career.get("shoots"),
            "source":     "hockey_reference",
            "draft_year": career.get("draft_year"),
            "draft_round":career.get("draft_round"),
            "draft_pick": career.get("draft_pick"),
            "draft_team": career.get("draft_team"),
        }
        upsert_players(pd.DataFrame([bio_row]))

        # Upsert seasons (dropping Tier-4 youth/prep rows at ingest — they
        # have no predictive value and would pollute feature aggregation).
        seasons = career.get("seasons", [])
        if seasons:
            s_df = pd.DataFrame(seasons)
            s_df["player_id"] = syn_id
            if "league" in s_df.columns:
                s_df = s_df[~s_df["league"].apply(is_excluded_league)].copy()
            if not s_df.empty:
                upsert_seasons(s_df)

        scraped += 1
        if i % 50 == 0 or i == len(hr_ids):
            console.print(f"  {i}/{len(hr_ids)} processed ({scraped} upserted)")

    console.print(f"[bold cyan]Backfill complete.[/] {scraped} career records stored.")


def cmd_enrich_nhl_bios(args):
    """
    For every EP-scraped player in the DB, look them up in the NHL API and
    upsert their bio (DOB, height, weight, shoots, draft info) + full career
    (NHL/AHL/juniors/international). Free, fast — typically 10-30 min for
    our entire prospect pool. Complements `backfill-careers` (Hockey
    Reference) which handles historical draftees.
    """
    console.rule("[bold magenta]Enriching bios from NHL API")
    init_db()

    import pandas as pd
    players_df = load_players()
    if players_df.empty:
        console.print("[red]No players in DB — run `collect` first.[/]")
        return

    # Target = EP-scraped prospects that don't already have an NHL API bio.
    target = players_df[players_df["source"] == "eliteprospects"].copy()
    if args.limit:
        target = target.head(int(args.limit))
    if args.missing_dob_only:
        target = target[target["dob"].isna()]

    console.print(f"Targeting {len(target)} EP-scraped players "
                  f"(of {len(players_df)} total).")

    id_map = nhl_load_id_map()
    hits, misses = 0, 0

    for i, row in enumerate(target.itertuples(index=False), 1):
        result = nhl_enrich_player(row.name, row.player_id, id_map=id_map)
        if result is None:
            misses += 1
        else:
            hits += 1
            upsert_players(pd.DataFrame([result["bio"]]))
            if result["seasons"]:
                s_df = pd.DataFrame(result["seasons"])
                # Drop Tier-4 youth/prep rows (WSI, USHS, CSSHL, U10-U17, etc.)
                if "league" in s_df.columns:
                    s_df = s_df[~s_df["league"].apply(is_excluded_league)].copy()
                if not s_df.empty:
                    upsert_seasons(s_df)

        if i % 25 == 0 or i == len(target):
            console.print(f"  {i}/{len(target)} processed "
                          f"(hits={hits}, misses={misses})")
            nhl_save_id_map(id_map)

    nhl_save_id_map(id_map)
    console.print(f"[bold magenta]Done.[/] {hits} players enriched, "
                  f"{misses} not found in NHL search.")


def cmd_pipeline(args):
    """Full pipeline: collect → process → train → rank."""
    cmd_collect(args)
    cmd_process(args)
    cmd_train(args)
    cmd_rank(args)


# ── CLI wiring ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NHL Prospect Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    # collect
    p_collect = sub.add_parser("collect", help="Scrape CHL/NCAA data")
    p_collect.add_argument("--seasons", nargs="+", default=None,
                            help="Seasons to scrape e.g. 2023-2024 2024-2025")
    p_collect.add_argument("--league", type=str, default=None,
                            help="Restrict scrape to a single league (e.g. OHL, WHL, QMJHL, NCAA)")

    # process
    sub.add_parser("process", help="Normalize data, compute NHLe")

    # train
    sub.add_parser("train", help="Train XGBoost models on historical data")

    # rank
    p_rank = sub.add_parser("rank", help="Generate prospect rankings")
    p_rank.add_argument("--draft-year", type=int, default=None)
    p_rank.add_argument("--top", type=int, default=50)
    p_rank.add_argument("--league", type=str, default=None,
                        help="Restrict rankings to prospects with seasons in this league")

    # report
    p_report = sub.add_parser("report", help="Generate per-player scouting report")
    p_report.add_argument("--player", type=str, required=True)

    # backfill-careers
    p_bf = sub.add_parser("backfill-careers",
                           help="Scrape Hockey Reference player career pages (bios + multi-season history)")
    p_bf.add_argument("--start", type=int, default=1995,
                       help="First draft year to include (default: 1995)")
    p_bf.add_argument("--end", type=int, default=2026,
                       help="Last draft year to include (default: 2026)")
    p_bf.add_argument("--sample", type=int, default=None,
                       help="Scrape a random sample of N players instead of all")
    p_bf.add_argument("--nhlers-only", action="store_true",
                       help="Only scrape players with is_nhler=1 (faster, less balanced)")
    p_bf.add_argument("--refresh-mapping", action="store_true",
                       help="Force re-scrape of the HR ID mapping from draft pages")

    # enrich-nhl-bios
    p_nhl = sub.add_parser("enrich-nhl-bios",
                            help="Use NHL.com API to fill bios + careers for EP-scraped prospects")
    p_nhl.add_argument("--limit", type=int, default=None,
                       help="Process only the first N players (for testing)")
    p_nhl.add_argument("--missing-dob-only", action="store_true",
                       help="Only enrich players whose DOB is currently missing")

    # pipeline
    p_pipeline = sub.add_parser("pipeline", help="Run full collect→train→rank pipeline")
    p_pipeline.add_argument("--seasons", nargs="+", default=None)
    p_pipeline.add_argument("--draft-year", type=int, default=None)
    p_pipeline.add_argument("--top", type=int, default=50)
    p_pipeline.add_argument("--league", type=str, default=None,
                            help="Restrict the pipeline to a single league (e.g. OHL)")

    args = parser.parse_args()
    dispatch = {
        "collect":           cmd_collect,
        "process":           cmd_process,
        "train":             cmd_train,
        "rank":              cmd_rank,
        "report":            cmd_report,
        "pipeline":          cmd_pipeline,
        "backfill-careers":  cmd_backfill_careers,
        "enrich-nhl-bios":   cmd_enrich_nhl_bios,
    }

    if not args.command:
        parser.print_help()
        sys.exit(0)

    dispatch[args.command](args)


if __name__ == "__main__":
    import pandas as pd
    main()
