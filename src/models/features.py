from typing import Optional
"""
Feature engineering pipeline.
Builds the feature matrix used for training and inference.
"""
import logging
import re
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "nhle_ppg",          # NHLe-adjusted PPG (best single predictor)
    "nhle_ppg_d_minus1", # Prior season NHLe PPG
    "nhle_ppg_d_minus2", # Two seasons prior
    "ppg_delta",         # Avg year-over-year NHLe improvement
    "nhle_gpg",          # NHLe-adjusted goals per game
    "age_at_draft",      # Age on June 1 of draft year
    "birth_quarter",     # 1–4 (relative age effect)
    "height_cm",         # Physical: height
    "weight_kg",         # Physical: weight
    "gp_rate",           # Games played / team games (usage/health)
    "pim_per_game",      # Physicality proxy
    "pp_pts_pct",        # % of points on power play (5v5 skill signal)
    "is_forward",        # 1=F, 0=D (position encoded)
    "league_enc",        # Ordinal league strength encoding
]

LEAGUE_STRENGTH = {
    "OHL":   3,
    "WHL":   3,
    "QMJHL": 2,
    "NCAA":  3,
    "USHL":  1,
    "OTHER": 0,
}


def build_feature_matrix(players_df: pd.DataFrame,
                          seasons_df: pd.DataFrame,
                          outcomes_df: Optional[pd.DataFrame] = None,
                          draft_year: Optional[int] = None) -> pd.DataFrame:
    """
    Build one row per player with all features.
    If outcomes_df provided, attaches is_nhler / is_star labels.
    If draft_year provided, filters to that cohort.
    """
    # Pivot seasons to one row per player
    pivot = _pivot_seasons(seasons_df)

    # Merge bio
    bio = players_df[["player_id", "name", "dob", "position",
                       "height_cm", "weight_kg"]].drop_duplicates("player_id")
    df = pivot.merge(bio, on="player_id", how="left")

    if outcomes_df is not None and not outcomes_df.empty:
        # Hockey Reference outcomes use their own player_id format, which
        # doesn't match EliteProspects scraped player_ids. Remap via
        # normalized name so the left-merge actually finds labels.
        outcomes_df = _remap_outcomes_to_players(players_df, outcomes_df)

        out_cols = ["player_id", "draft_year", "draft_round", "draft_pick",
                    "is_nhler", "is_star", "nhl_gp", "nhl_points"]
        out_cols = [c for c in out_cols if c in outcomes_df.columns]
        df = df.merge(outcomes_df[out_cols], on="player_id", how="left")
        if draft_year:
            df = df[df["draft_year"] == draft_year]

    df = _engineer(df)
    return df


def _pivot_seasons(seasons_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse multi-season rows into one row per player."""
    if seasons_df.empty:
        return pd.DataFrame(columns=["player_id"])

    df = seasons_df.copy()

    # Best/most recent draft-eligible season (D0 = draft year season)
    d0 = df[df.get("draft_label", pd.Series(["unknown"] * len(df))) == "D0"]
    dm1 = df[df.get("draft_label", pd.Series(["unknown"] * len(df))) == "D-1"]
    dm2 = df[df.get("draft_label", pd.Series(["unknown"] * len(df))) == "D-2"]

    def best_season(subset: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
        if subset.empty:
            return pd.DataFrame(columns=["player_id"])
        agg = subset.groupby("player_id").agg(
            nhle_ppg=(f"nhle_ppg", "max"),
            nhle_gpg=(f"nhle_gpg", "max"),
            league=("league", "first"),
            gp=("gp", "sum"),
            pim=("pim", "sum"),
            pp_goals=("pp_goals", "sum"),
            pp_assists=("pp_assists", "sum"),
            points=("points", "sum"),
        ).reset_index()
        if suffix:
            agg = agg.rename(columns={
                "nhle_ppg": f"nhle_ppg_{suffix}",
                "nhle_gpg": f"nhle_gpg_{suffix}",
            })
        return agg

    base = best_season(d0 if not d0.empty else df)
    prev1 = best_season(dm1, "d_minus1") if not dm1.empty else pd.DataFrame()
    prev2 = best_season(dm2, "d_minus2") if not dm2.empty else pd.DataFrame()

    result = base
    if not prev1.empty:
        result = result.merge(
            prev1[["player_id", "nhle_ppg_d_minus1"]], on="player_id", how="left"
        )
    else:
        result["nhle_ppg_d_minus1"] = np.nan

    if not prev2.empty:
        result = result.merge(
            prev2[["player_id", "nhle_ppg_d_minus2"]], on="player_id", how="left"
        )
    else:
        result["nhle_ppg_d_minus2"] = np.nan

    # Development delta: average improvement across available seasons
    result["ppg_delta"] = _compute_ppg_deltas(seasons_df)
    return result


def _compute_ppg_deltas(seasons_df: pd.DataFrame) -> pd.Series:
    """Per-player average NHLe PPG improvement."""
    if "nhle_ppg" not in seasons_df.columns or "player_id" not in seasons_df.columns:
        return pd.Series(dtype=float)

    def delta(grp):
        s = grp.sort_values("season")["nhle_ppg"].dropna()
        if len(s) < 2:
            return 0.0
        return float(s.diff().dropna().mean())

    return (
        seasons_df.groupby("player_id")
        .apply(delta)
        .rename("ppg_delta")
    )


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Position encoding
    df["is_forward"] = (df["position"].fillna("F").str.upper().str[0] == "F").astype(int)

    # Physical defaults (median imputation)
    df["height_cm"] = pd.to_numeric(df.get("height_cm"), errors="coerce")
    df["weight_kg"] = pd.to_numeric(df.get("weight_kg"), errors="coerce")
    df["height_cm"] = df["height_cm"].fillna(df["height_cm"].median())
    df["weight_kg"] = df["weight_kg"].fillna(df["weight_kg"].median())

    # Age at draft
    df["age_at_draft"] = df.apply(_age_at_draft, axis=1)

    # Birth quarter (relative age effect: Q4 = undervalued)
    df["birth_quarter"] = df["dob"].apply(_birth_quarter)

    # Power-play % of points
    pp_pts = df.get("pp_goals", pd.Series(0)) + df.get("pp_assists", pd.Series(0))
    total_pts = df.get("points", pd.Series(1)).replace(0, 1)
    df["pp_pts_pct"] = (pp_pts / total_pts).fillna(0).round(3)

    # PIM per game
    gp = df.get("gp", pd.Series(1)).replace(0, 1)
    df["pim_per_game"] = (df.get("pim", pd.Series(0)) / gp).fillna(0).round(3)

    # GP rate (proxy for team usage / health) — filled with 1.0 if unknown.
    # _pivot_seasons drops this column, so the pivoted df may lack it entirely.
    if "gp_rate" in df.columns:
        df["gp_rate"] = pd.to_numeric(df["gp_rate"], errors="coerce").fillna(1.0)
    else:
        df["gp_rate"] = 1.0

    # League encoding
    df["league_enc"] = df.get("league", pd.Series("OTHER")).apply(
        lambda x: LEAGUE_STRENGTH.get(str(x).upper(), 0)
    )

    # Fill NaN features with 0
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0.0

    return df


def _age_at_draft(row) -> float:
    try:
        dob = datetime.strptime(row["dob"], "%Y-%m-%d").date()
        draft_yr = int(row.get("draft_year", datetime.now().year))
        draft_date = date(draft_yr, 6, 1)
        return round((draft_date - dob).days / 365.25, 2)
    except Exception:
        return 18.5  # median assumption


def _birth_quarter(dob: str) -> int:
    try:
        month = datetime.strptime(dob, "%Y-%m-%d").month
        return (month - 1) // 3 + 1  # 1–4
    except Exception:
        return 2


_HR_CACHE_PATH = (
    Path(__file__).parents[2] / "data" / "historical" / "draft_outcomes_cache.csv"
)


def _normalize_name(s) -> str:
    """Lowercase, strip anything non-alpha. Good enough for 'Connor McDavid'
    vs 'Connor M. McDavid' vs 'connor mcdavid'."""
    return re.sub(r"[^a-z]", "", str(s).lower())


def _remap_outcomes_to_players(players_df: pd.DataFrame,
                                outcomes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join historical outcomes to scraped players via normalized name.
    `nhl_outcomes` only stores Hockey Reference player_ids, which never match
    the EliteProspects player_ids used in the `players` table. We recover the
    match by loading the HR cache CSV (which still has names) and matching on
    normalized name.
    Returns an outcomes DataFrame whose `player_id` values are replaced with
    the matching EP player_ids. Unmatched outcomes are dropped.
    """
    if players_df.empty or "name" not in players_df.columns:
        return outcomes_df

    # The DB-stored outcomes table lost the `name` column via the defensive
    # upsert filter (it's not in the schema). Recover names from the raw cache.
    if not _HR_CACHE_PATH.exists():
        logger.warning("Historical outcomes cache missing — cannot remap by name.")
        return outcomes_df

    hr = pd.read_csv(_HR_CACHE_PATH, usecols=["player_id", "name"])
    with_name = outcomes_df.merge(hr, on="player_id", how="left")
    with_name["_norm"] = with_name["name"].apply(_normalize_name)

    players_norm = players_df[["player_id", "name"]].copy()
    players_norm["_norm"] = players_norm["name"].apply(_normalize_name)
    # De-dup normalized names on both sides so a merge can't explode rows
    players_norm = players_norm.drop_duplicates("_norm")
    with_name = with_name.drop_duplicates("_norm")

    matched = with_name.merge(
        players_norm[["player_id", "_norm"]].rename(columns={"player_id": "ep_player_id"}),
        on="_norm", how="inner",
    )
    matched["player_id"] = matched["ep_player_id"]
    matched = matched.drop(columns=["_norm", "ep_player_id", "name"])
    logger.info(f"Historical outcomes remapped by name: {len(matched)} matches "
                f"(from {len(outcomes_df)} HR records and {len(players_df)} scraped players).")
    return matched
