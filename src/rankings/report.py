from typing import Optional
"""
Per-player scouting report generator.
Outputs a rich Markdown report with stats, NHLe trajectory, comparables, and risk flags.
"""
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

from src.comparables.similarity import ComparableIndex

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parents[2] / "data" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_report(player_id: str,
                     features_df: pd.DataFrame,
                     seasons_df: pd.DataFrame,
                     predictions_df: pd.DataFrame,
                     comp_index: Optional["ComparableIndex"],
                     save: bool = True) -> str:
    """
    Generate a full Markdown scouting report for one player.
    Returns the report as a string.
    """
    player_row = features_df[features_df["player_id"] == player_id]
    if player_row.empty:
        return f"Player {player_id} not found."
    player_row = player_row.iloc[0]

    pred_row = predictions_df[predictions_df["player_id"] == player_id]
    pred = pred_row.iloc[0] if not pred_row.empty else pd.Series()

    player_seasons = seasons_df[seasons_df["player_id"] == player_id].sort_values("season")

    name     = player_row.get("name", player_id)
    position = player_row.get("position", "?")
    league   = player_row.get("league", "?")
    age      = player_row.get("age_at_draft", "?")
    height   = player_row.get("height_cm", "?")
    weight   = player_row.get("weight_kg", "?")

    nhler_pct = float(pred.get("nhler_probability", 0)) * 100
    star_pct  = float(pred.get("star_probability", 0)) * 100
    proj_pts  = int(pred.get("projected_career_pts", 0))
    nhle_ppg  = float(player_row.get("nhle_ppg", 0))

    lines = [
        f"# Scouting Report: {name}",
        f"*Generated {datetime.now().strftime('%Y-%m-%d')}*",
        "",
        "## Bio",
        f"| Attribute | Value |",
        f"|-----------|-------|",
        f"| Position  | {position} |",
        f"| League    | {league} |",
        f"| Age (draft eligible) | {age} |",
        f"| Height    | {height} cm |",
        f"| Weight    | {weight} kg |",
        "",
        "## Projections",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| NHLe PPG | **{nhle_ppg:.3f}** |",
        f"| NHLer Probability (200+ GP) | **{nhler_pct:.1f}%** |",
        f"| Star Probability | **{star_pct:.1f}%** |",
        f"| Projected Career Points | **{proj_pts}** |",
        "",
        "## Season-by-Season Stats",
        _season_table(player_seasons),
        "",
        "## NHLe Trajectory",
        _nhle_trajectory(player_seasons),
        "",
    ]

    if comp_index is not None:
        comps = comp_index.find_comparables(player_row, n=5)
        lines += [
            "## Top 5 Player Comparables",
            _comparables_table(comps),
            "",
        ]

    flags = _risk_flags(player_row, player_seasons, pred)
    if flags:
        lines += ["## Flags & Notes", *[f"- {f}" for f in flags], ""]

    report = "\n".join(lines)

    if save:
        slug = str(name).lower().replace(" ", "_").replace(".", "")
        out_path = REPORTS_DIR / f"{slug}.md"
        out_path.write_text(report, encoding="utf-8")
        logger.info(f"Report saved to {out_path}")

    return report


def _season_table(seasons: pd.DataFrame) -> str:
    if seasons.empty:
        return "*No season data.*"
    cols = ["season", "league", "team", "gp", "goals", "assists",
            "points", "ppg", "nhle_ppg"]
    cols = [c for c in cols if c in seasons.columns]
    rows = ["| " + " | ".join(cols) + " |",
            "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in seasons.iterrows():
        rows.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    return "\n".join(rows)


def _nhle_trajectory(seasons: pd.DataFrame) -> str:
    if seasons.empty or "nhle_ppg" not in seasons.columns:
        return "*No NHLe data.*"
    lines = []
    for _, row in seasons.sort_values("season").iterrows():
        val = float(row.get("nhle_ppg", 0))
        bar = "█" * int(val * 20)
        lines.append(f"`{row.get('season','')}` {bar} {val:.3f}")
    return "\n".join(lines)


def _comparables_table(comps: pd.DataFrame) -> str:
    if comps.empty:
        return "*No comparables found.*"
    rows = ["| # | Name | Draft Year | Sim | NHL GP | NHL Pts | Outcome |",
            "|---|------|-----------|-----|--------|---------|---------|"]
    for i, row in comps.iterrows():
        rows.append(
            f"| {i+1} | {row.get('name','')} | {row.get('draft_year','')} | "
            f"{row.get('similarity_score',0)} | {row.get('nhl_gp',0)} | "
            f"{row.get('nhl_points',0)} | {row.get('outcome_label','')} |"
        )
    return "\n".join(rows)


def _risk_flags(player_row: pd.Series,
                seasons: pd.DataFrame,
                pred: pd.Series) -> list[str]:
    flags = []
    age = float(player_row.get("age_at_draft", 18))
    if age >= 20:
        flags.append(f"Overage prospect (age {age:.1f}) — lower ceiling signal")

    if len(seasons) < 2:
        flags.append("Limited data: fewer than 2 seasons available — projections less reliable")

    ppg_delta = float(player_row.get("ppg_delta", 0))
    if ppg_delta < -0.05:
        flags.append(f"Declining production trend (NHLe PPG delta: {ppg_delta:+.3f})")

    if float(player_row.get("pp_pts_pct", 0)) > 0.50:
        flags.append("High PP dependency (>50% of points on power play) — 5v5 impact uncertain")

    gp_rate = float(player_row.get("gp_rate", 1.0))
    if gp_rate < 0.75:
        flags.append(f"Low games played rate ({gp_rate:.0%}) — injury or usage concerns")

    nhler_pct = float(pred.get("nhler_probability", 0))
    if nhler_pct < 0.25:
        flags.append("Low NHLer probability — likely requires significant development")

    bq = int(player_row.get("birth_quarter", 2))
    if bq == 4:
        flags.append("Q4 birth (relative age effect) — likely undervalued, upside may be understated")

    return flags
