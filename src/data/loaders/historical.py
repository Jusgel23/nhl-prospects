"""
Load and clean historical Kaggle datasets:
  - NHL Draft 1963-2022 (draft outcomes)
  - Elite Prospects career data
"""
import logging
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parents[3] / "data" / "historical"

DRAFT_CSV       = DATA_DIR / "nhl_draft_1963_2022.csv"
EP_CAREERS_CSV  = DATA_DIR / "ep_career_stats.csv"


def load_draft_outcomes() -> pd.DataFrame:
    """
    Load historical draft outcomes.
    Expected columns (flexible): player, year/draft_year, round, pick,
    team, position, nhl_gp, nhl_goals, nhl_assists, nhl_points, etc.
    Returns cleaned DataFrame with canonical column names.
    """
    if not DRAFT_CSV.exists():
        logger.warning(f"Draft CSV not found at {DRAFT_CSV}. Download from Kaggle.")
        return _empty_outcomes()

    df = pd.read_csv(DRAFT_CSV, low_memory=False)
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

    # Normalize column names across different Kaggle file formats
    rename = {
        "overall_pick": "draft_pick",
        "overall":      "draft_pick",
        "pick":         "draft_pick",
        "yr":           "draft_year",
        "year":         "draft_year",
        "round_number": "draft_round",
        "rd":           "draft_round",
        "player_name":  "name",
        "full_name":    "name",
        "pos":          "position",
        "gp":           "nhl_gp",
        "g":            "nhl_goals",
        "a":            "nhl_assists",
        "pts":          "nhl_points",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    required = ["name", "draft_year", "draft_round", "draft_pick"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error(f"Draft CSV missing columns: {missing}")
        return _empty_outcomes()

    for col in ["nhl_gp", "nhl_goals", "nhl_assists", "nhl_points"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["is_nhler"] = (df["nhl_gp"] >= 200).astype(int)
    df["is_star"]  = (
        (df["nhl_gp"] >= 200) &
        (df["nhl_points"] / df["nhl_gp"].replace(0, np.nan) >= 0.4)
    ).fillna(False).astype(int)

    df["player_id"] = (
        df["name"].str.lower().str.replace(r"[^a-z0-9]", "_", regex=True)
        + "_" + df["draft_year"].astype(str)
    )

    df["position"] = df.get("position", pd.Series(["F"] * len(df))).fillna("F")
    df["position"] = df["position"].apply(_normalize_position)

    logger.info(f"Loaded {len(df)} historical draft picks. "
                f"NHLers: {df['is_nhler'].sum()} ({df['is_nhler'].mean():.1%})")
    return df


def load_ep_careers() -> pd.DataFrame:
    """Load EliteProspects career arc data for historical comparables."""
    if not EP_CAREERS_CSV.exists():
        logger.warning(f"EP careers CSV not found at {EP_CAREERS_CSV}.")
        return pd.DataFrame()

    df = pd.read_csv(EP_CAREERS_CSV, low_memory=False)
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
    return df


def _normalize_position(pos: str) -> str:
    pos = str(pos).upper().strip()
    if any(x in pos for x in ["C", "LW", "RW", "W", "F"]):
        return "F"
    if "D" in pos or "DEF" in pos:
        return "D"
    if "G" in pos:
        return "G"
    return "F"


def _empty_outcomes() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "player_id", "name", "draft_year", "draft_round", "draft_pick",
        "draft_team", "position", "nhl_gp", "nhl_goals", "nhl_assists",
        "nhl_points", "is_nhler", "is_star"
    ])
