from typing import Optional
"""
NHL Equivalency (NHLe) normalization.

Converts raw scoring rates from junior/college leagues to an NHL-equivalent
points-per-game figure, adjusted for age and relative draft year.
"""
import logging
from datetime import date, datetime

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Conversion factors from established research (TopDownHockey / Vollman),
# extended with additional leagues observed in HR + NHL-API career data.
NHLE_FACTORS: dict[str, float] = {
    # Core leagues
    "NHL":   1.00,
    "AHL":   0.44,
    "OHL":   0.30,
    "WHL":   0.29,
    "QMJHL": 0.27,
    "NCAA":  0.32,
    "USHL":  0.18,
    "ECHL":  0.24,
    "KHL":   0.82,
    "SHL":   0.68,
    "LIIGA": 0.66,
    "NLA":   0.60,
    "DEL":   0.57,

    # Junior A / Tier-2 feeder leagues
    "BCHL":  0.22,   # British Columbia Hockey League
    "OJHL":  0.20,   # Ontario Junior Hockey League
    "AJHL":  0.20,   # Alberta Junior Hockey League
    "NTDP":  0.25,   # USA Hockey National Team Development Program
    "MHL":   0.22,   # Russian U20 junior (top junior level in Russia)

    # Russian / European secondary pro
    "VHL":   0.55,   # Russian Tier-2 pro (one step below KHL)
    "RUSSIA": 0.40,  # Generic Russian senior amateur/semi-pro
    "IHL":   0.30,   # Historical International Hockey League (defunct US minor)

    # International tournaments (short-sample events — assigned a factor
    # but contribute modestly to aggregates because games-played is small)
    "WJC-20":               0.50,  # World Juniors (U20)
    "WJC-18":               0.40,  # U18 Worlds
    "WHC-17":               0.30,  # U17 Worlds
    "HLINKA GRETZKY CUP":   0.35,  # U18 August tournament
    "IVAN HLINKA MEMORIAL": 0.35,  # Same event, older name
    "WC":                   0.75,  # Senior Men's World Championship
    "WC-A":                 0.55,  # Worlds Division I-A
    "WJC-A":                0.25,  # World Jr A Challenge (Tier 2 nations)
    "WJAC-19":              0.25,  # Variant spelling of WJC-A
    "MEMORIAL CUP":         0.32,  # CHL post-season tournament
    "CHAMPIONS HL":         0.60,  # European Champions Hockey League
}


# Patterns for leagues that should NEVER be included in NHLe features.
# These are pre-draft youth / prep-school events where scoring has no
# predictive value for NHL projection (ages 11-16, tiny sample, uneven
# competition). Matching is case-insensitive against the uppercased league
# string; any match excludes the row from `apply_nhle_to_seasons` output.
EXCLUDED_LEAGUE_PATTERNS = [
    r"^WSI\b",           # World Selects Invitational (all youth ages)
    r"^USHS-",           # US high school variants
    r"^USA-S\d",         # US development squads by age
    r"^CSSHL",           # Canadian Sport School Hockey League
    r"\bU1[0-7]\b",      # Any "U10".."U17" tag
    r"^QAAA$", r"^QMAAA\b",
    r"^AMBHL$", r"^AMHL$", r"^SAAHL\b",
    r"^HEO\b", r"^GTHL\b", r"^ALLIANCE\b", r"^WAAA\b", r"^ETAHL\b", r"^T1EHL\b",
    r"^QC INT",          # Quebec International Pee-Wee
    r"BRICK",            # Brick Invitational (U10)
    r"\bPEE ?WEE\b",
    r"\bBANTAM\b",
    r"\bMIDGET\b",
    r"^MN HIGH",         # Minnesota high school
]


def is_excluded_league(league: Optional[str]) -> bool:
    """True if `league` matches a Tier-4 youth/prep pattern and should not
    contribute to NHLe feature aggregation."""
    if not league:
        return False
    import re as _re
    s = str(league).upper().strip()
    for pat in EXCLUDED_LEAGUE_PATTERNS:
        if _re.search(pat, s):
            return True
    return False

# Age adjustments — applied multiplicatively to NHLe PPG
# Younger players at equivalent production have higher ceilings
AGE_ADJUSTMENTS: dict[int, float] = {
    16: 1.20,
    17: 1.15,
    18: 1.08,
    19: 1.00,
    20: 0.95,
    21: 0.90,
    22: 0.85,
    23: 0.80,
}
DEFAULT_AGE_ADJ = 0.78  # 24+


def get_nhle_factor(league: str) -> float:
    return NHLE_FACTORS.get(league.upper().strip(), 0.25)


def age_at_season_midpoint(dob: str, season: str) -> Optional[int]:
    """
    Compute player age at ~Jan 1 of the given season.
    season: '2024-2025'
    dob: 'YYYY-MM-DD'
    """
    try:
        birth = datetime.strptime(dob, "%Y-%m-%d").date()
        mid_year = int(season.split("-")[0]) + 1  # Jan of second half
        mid_date = date(mid_year, 1, 1)
        return (mid_date - birth).days // 365
    except Exception:
        return None


def age_adjustment(age: Optional[int]) -> float:
    # Treat missing age as "unknown, don't penalize" — otherwise NaN would
    # fall through the dict lookup to DEFAULT_AGE_ADJ (0.78, the 24+ rate),
    # unfairly deflating NHLe for any player without a DOB.
    if age is None or pd.isna(age):
        return 1.0
    try:
        age_int = int(age)
    except (TypeError, ValueError):
        return 1.0
    return AGE_ADJUSTMENTS.get(age_int, DEFAULT_AGE_ADJ)


def compute_nhle_ppg(ppg: float, league: str, age: Optional[int] = None,
                     apply_age_adj: bool = True) -> float:
    """
    Convert raw PPG to NHLe PPG.
    NHLe_PPG = raw_PPG × league_factor × age_adjustment
    """
    factor = get_nhle_factor(league)
    base = ppg * factor
    if apply_age_adj:
        base *= age_adjustment(age)
    return round(base, 4)


def apply_nhle_to_seasons(seasons_df: pd.DataFrame,
                           players_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich a seasons DataFrame with nhle_ppg and age columns.
    seasons_df must have: player_id, season, league, (ppg OR points+gp)
    players_df must have: player_id, dob
    Idempotent: drops prior derived columns so re-runs don't stack _x/_y
    suffixes from repeated merges.
    """
    df = seasons_df.copy()

    # Drop Tier-4 youth/prep rows (WSI, USHS, CSSHL, U10-U17, etc). If they
    # somehow slipped into the seasons table from an older ingestion path,
    # don't let them pollute NHLe features. The cleanup script purges them
    # from the DB; this is a safety net for new ingestions.
    if "league" in df.columns:
        pre = len(df)
        df = df[~df["league"].astype(str).apply(is_excluded_league)].copy()
        dropped = pre - len(df)
        if dropped:
            logger.info(f"Dropped {dropped} excluded-league rows during NHLe apply.")

    # Drop previously-derived columns so the merge can recreate them cleanly.
    for col in ("dob", "age", "nhle_factor", "age_adj", "nhle_ppg", "nhle_gpg"):
        if col in df.columns:
            df = df.drop(columns=[col])

    # HR-sourced rows arrive without a ppg column — derive it.
    if "ppg" not in df.columns or df["ppg"].isna().all():
        df["ppg"] = 0.0
    missing_ppg = df["ppg"].isna() | (df["ppg"] == 0)
    if "points" in df.columns and "gp" in df.columns:
        derived = (
            pd.to_numeric(df["points"], errors="coerce")
            / pd.to_numeric(df["gp"], errors="coerce").replace(0, np.nan)
        ).round(3)
        df.loc[missing_ppg, "ppg"] = derived[missing_ppg].fillna(0.0)

    bio = players_df[["player_id", "dob"]].drop_duplicates("player_id")
    df = df.merge(bio, on="player_id", how="left")

    df["age"] = df.apply(
        lambda r: age_at_season_midpoint(r.get("dob", ""), r.get("season", ""))
        if pd.notna(r.get("dob")) else None,
        axis=1,
    )

    df["nhle_factor"] = df["league"].apply(get_nhle_factor)
    df["age_adj"]     = df["age"].apply(age_adjustment)
    df["nhle_ppg"]    = (df["ppg"] * df["nhle_factor"] * df["age_adj"]).round(4)

    # Goals NHLe (rough: assume 45% of points are goals in junior)
    df["nhle_gpg"] = (df["goals"] / df["gp"].replace(0, np.nan)
                      * df["nhle_factor"] * df["age_adj"]).round(4)

    logger.debug(f"NHLe computed for {len(df)} season rows.")
    return df


def build_development_arc(nhle_df: pd.DataFrame,
                           outcomes_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each player, compute multi-season NHLe trajectory relative to draft year.
    Labels seasons as D-2, D-1, D0, D+1 etc.

    outcomes_df must have: player_id, draft_year
    If a player has no draft_year in outcomes but has a DOB on their season
    row, we infer draft_year as (birth_year + 18) — standard NHL eligibility.
    This lets the D-relative labels populate for current undrafted prospects.
    """
    df = nhle_df.copy()

    # Drop previously-derived columns so the merge doesn't stack _x/_y.
    for col in ("draft_year", "season_start_yr", "draft_delta", "draft_label"):
        if col in df.columns:
            df = df.drop(columns=[col])

    if outcomes_df is None or outcomes_df.empty or "draft_year" not in outcomes_df.columns:
        logger.warning("No outcomes — falling back to DOB-inferred draft years only.")
    else:
        draft_years = outcomes_df[["player_id", "draft_year"]].drop_duplicates("player_id")
        df = df.merge(draft_years, on="player_id", how="left")

    if "draft_year" not in df.columns:
        df["draft_year"] = np.nan

    # Infer draft_year from DOB where missing
    def _infer_draft_year(row):
        dy = row.get("draft_year")
        if pd.notna(dy) and dy not in (None, 0):
            return dy
        dob = row.get("dob")
        if pd.notna(dob):
            try:
                return int(str(dob)[:4]) + 18
            except Exception:
                return np.nan
        return np.nan

    df["draft_year"] = df.apply(_infer_draft_year, axis=1)

    def season_year(s: str) -> int:
        try:
            return int(str(s).split("-")[0])
        except Exception:
            return 0

    df["season_start_yr"] = df["season"].apply(season_year)
    df["draft_delta"] = df.apply(
        lambda r: (r["season_start_yr"] - r["draft_year"])
        if pd.notna(r.get("draft_year")) else np.nan,
        axis=1,
    )
    df["draft_label"] = df["draft_delta"].apply(
        lambda d: f"D{int(d):+d}" if pd.notna(d) else "unknown"
    )
    return df


def ppg_delta(player_seasons: pd.DataFrame) -> float:
    """
    Year-over-year NHLe PPG improvement for a player.
    Higher is better — strong development signal.
    """
    s = player_seasons.sort_values("season")["nhle_ppg"].dropna()
    if len(s) < 2:
        return 0.0
    deltas = s.diff().dropna()
    return round(float(deltas.mean()), 4)
