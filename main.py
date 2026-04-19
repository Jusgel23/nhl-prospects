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
from src.models.nhle import apply_nhle_to_seasons, build_development_arc
from src.models.features import build_feature_matrix
from src.models.predictor import ProspectPredictor
from src.comparables.similarity import build_comparable_index
from src.rankings.ranker import rank_prospects, format_rankings_table, export_csv
from src.rankings.report import generate_report


DEFAULT_SEASONS = ["2022-2023", "2023-2024", "2024-2025"]
RANKINGS_CSV = Path("data/processed/rankings.csv")


def cmd_collect(args):
    seasons = args.seasons or DEFAULT_SEASONS
    console.rule("[bold green]Collecting prospect data")

    init_db()

    # CHL via EliteProspects
    console.print(f"Scraping CHL data for seasons: {seasons}")
    players_df, seasons_df = scrape_draft_class(seasons)
    if not players_df.empty:
        upsert_players(players_df)
        console.print(f"  [green]✓[/] {len(players_df)} players stored")
    if not seasons_df.empty:
        upsert_seasons(seasons_df)
        console.print(f"  [green]✓[/] {len(seasons_df)} season rows stored")

    # NCAA via USCHO
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

    # Save processed seasons back (overwrite)
    from src.data.database import get_conn
    import pandas as pd
    with get_conn() as conn:
        seasons_df.to_sql("seasons", conn, if_exists="replace", index=False)

    console.print("[bold blue]Processing complete.[/]")


def cmd_train(args):
    console.rule("[bold magenta]Training prediction models")

    players_df  = load_players()
    seasons_df  = load_seasons()
    outcomes_df = load_outcomes()

    if outcomes_df.empty:
        console.print("[yellow]No cached outcomes — will scrape Hockey Reference (1995-2019). This takes ~3 min...[/]")
        from src.data.loaders.historical import load_draft_outcomes
        outcomes_df = load_draft_outcomes()
        if outcomes_df.empty:
            console.print("[red]Could not retrieve historical outcomes. Check your internet connection.[/]")
            return
        from src.data.database import upsert_outcomes
        upsert_outcomes(outcomes_df)

    features_df = build_feature_matrix(players_df, seasons_df, outcomes_df)
    labeled = features_df[features_df["is_nhler"].notna()]
    console.print(f"Training on {len(labeled)} labeled historical prospects...")

    predictor = ProspectPredictor()
    predictor.train(labeled)

    console.print("Running walk-forward validation (2005–2015)...")
    predictor.train_walk_forward(labeled, start_year=2005, end_year=2015)

    console.print("[bold magenta]Training complete.[/]")


def cmd_rank(args):
    console.rule("[bold yellow]Generating prospect rankings")

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

    # process
    sub.add_parser("process", help="Normalize data, compute NHLe")

    # train
    sub.add_parser("train", help="Train XGBoost models on historical data")

    # rank
    p_rank = sub.add_parser("rank", help="Generate prospect rankings")
    p_rank.add_argument("--draft-year", type=int, default=None)
    p_rank.add_argument("--top", type=int, default=50)

    # report
    p_report = sub.add_parser("report", help="Generate per-player scouting report")
    p_report.add_argument("--player", type=str, required=True)

    # pipeline
    p_pipeline = sub.add_parser("pipeline", help="Run full collect→train→rank pipeline")
    p_pipeline.add_argument("--seasons", nargs="+", default=None)
    p_pipeline.add_argument("--draft-year", type=int, default=None)
    p_pipeline.add_argument("--top", type=int, default=50)

    args = parser.parse_args()
    dispatch = {
        "collect":  cmd_collect,
        "process":  cmd_process,
        "train":    cmd_train,
        "rank":     cmd_rank,
        "report":   cmd_report,
        "pipeline": cmd_pipeline,
    }

    if not args.command:
        parser.print_help()
        sys.exit(0)

    dispatch[args.command](args)


if __name__ == "__main__":
    import pandas as pd
    main()
