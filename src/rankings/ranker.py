from typing import Optional
"""
Prospect ranking engine.
Combines model outputs into a final rank score and sorted list.
"""
import logging

import pandas as pd
import numpy as np

from src.comparables.similarity import ComparableIndex

logger = logging.getLogger(__name__)

# Ranking weight formula:
#   rank_score = nhler_prob×0.50 + norm_nhle_ppg×0.25 + ppg_delta_score×0.15 + age_bonus×0.10
WEIGHTS = {
    "nhler_probability": 0.50,
    "nhle_ppg_norm":     0.25,
    "ppg_delta_norm":    0.15,
    "age_bonus":         0.10,
}


def rank_prospects(features_df: pd.DataFrame,
                   predictions_df: pd.DataFrame,
                   comp_index: Optional["ComparableIndex"] = None) -> pd.DataFrame:
    """
    Merge features + predictions, compute rank score, sort, attach top comparable.
    Returns ranked DataFrame ready for display / CSV export.
    """
    df = features_df.merge(predictions_df, on="player_id", how="left")

    df["nhle_ppg_norm"]  = _minmax(df["nhle_ppg"])
    df["ppg_delta_norm"] = _minmax(df["ppg_delta"].clip(-0.5, 1.0))
    df["age_bonus"]      = _age_bonus(df["age_at_draft"])

    df["rank_score"] = (
        df["nhler_probability"].fillna(0)   * WEIGHTS["nhler_probability"] +
        df["nhle_ppg_norm"].fillna(0)        * WEIGHTS["nhle_ppg_norm"] +
        df["ppg_delta_norm"].fillna(0)       * WEIGHTS["ppg_delta_norm"] +
        df["age_bonus"]                      * WEIGHTS["age_bonus"]
    ).round(4)

    df = df.sort_values("rank_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    if comp_index is not None:
        df["top_comparable"] = df.apply(
            lambda row: comp_index.top_comparable_str(row), axis=1
        )
    else:
        df["top_comparable"] = "N/A"

    display_cols = [
        "rank", "name", "position", "league", "age_at_draft",
        "nhle_ppg", "nhler_probability", "star_probability",
        "projected_career_pts", "rank_score", "top_comparable",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    return df[display_cols + ["player_id"]]


def format_rankings_table(ranked_df: pd.DataFrame) -> str:
    """Return a Rich-ready string table for CLI display."""
    lines = [
        f"{'Rank':<5} {'Name':<25} {'Pos':<4} {'League':<7} {'Age':<5} "
        f"{'NHLe PPG':<10} {'NHLer%':<9} {'Star%':<7} {'Top Comparable'}",
        "-" * 110,
    ]
    for _, row in ranked_df.iterrows():
        lines.append(
            f"{int(row.get('rank', 0)):<5} "
            f"{str(row.get('name', '')):<25} "
            f"{str(row.get('position', '')):<4} "
            f"{str(row.get('league', '')):<7} "
            f"{float(row.get('age_at_draft', 0)):<5.1f} "
            f"{float(row.get('nhle_ppg', 0)):<10.3f} "
            f"{float(row.get('nhler_probability', 0))*100:<9.1f} "
            f"{float(row.get('star_probability', 0))*100:<7.1f} "
            f"{str(row.get('top_comparable', 'N/A'))}"
        )
    return "\n".join(lines)


def export_csv(ranked_df: pd.DataFrame, path: str):
    ranked_df.to_csv(path, index=False)
    logger.info(f"Rankings exported to {path}")


# ── helpers ───────────────────────────────────────────────────────────────────

def _minmax(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)


def _age_bonus(age_series: pd.Series) -> pd.Series:
    """
    Younger prospects get a bonus.
    17 → 1.0, 18 → 0.75, 19 → 0.40, 20+ → 0.10
    """
    mapping = {17: 1.0, 18: 0.75, 19: 0.40, 20: 0.20}
    return age_series.apply(
        lambda a: mapping.get(int(a) if pd.notna(a) else 19, 0.10)
    )
