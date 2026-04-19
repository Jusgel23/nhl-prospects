"""
Hockey Reference draft history scraper.
Pulls historical NHL draft picks + career stats (no API key, no manual download).
Covers draft years 1990-2022 by default.
"""
import time
import logging
import re

import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

HR_BASE = "https://www.hockey-reference.com"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def scrape_draft_year(year: int) -> pd.DataFrame:
    """
    Scrape one NHL draft year from Hockey Reference.
    Returns rows with: name, draft_year, draft_round, draft_pick,
    draft_team, position, nhl_gp, nhl_goals, nhl_assists, nhl_points.
    """
    url = f"{HR_BASE}/draft/NHL_{year}_entry.html"
    logger.info(f"Scraping draft {year}: {url}")

    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"Failed to fetch draft {year}: {e}")
        return pd.DataFrame()

    soup = BeautifulSoup(resp.text, "lxml")

    # HR uses a commented-out table for draft data — parse via pandas
    try:
        from io import StringIO
        tables = pd.read_html(StringIO(resp.text), header=1)
        if not tables:
            return pd.DataFrame()
        df = tables[0]
        # HR uses multi-level header — flatten if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [" ".join(str(c) for c in col).strip() for col in df.columns]
    except Exception as e:
        logger.warning(f"Could not parse table for {year}: {e}")
        return pd.DataFrame()

    df.columns = [str(c).lower().strip().replace(" ", "_") for c in df.columns]

    # Drop header repeat rows
    df = df[df.get("rk", pd.Series(["1"] * len(df))).astype(str) != "Rk"]
    df = df[df.get("player", df.get("name", pd.Series(["x"] * len(df)))).astype(str) != "Player"]

    rename = {
        "rd":       "draft_round",
        "pick":     "draft_pick",
        "#":        "draft_pick",
        "team":     "draft_team",
        "player":   "name",
        "pos":      "position",
        "gp":       "nhl_gp",
        "g":        "nhl_goals",
        "a":        "nhl_assists",
        "pts":      "nhl_points",
        "nat":      "nationality",
        "to":       "last_season",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    df["draft_year"] = year

    for col in ["draft_round", "draft_pick", "nhl_gp", "nhl_goals", "nhl_assists", "nhl_points"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    if "name" not in df.columns:
        logger.warning(f"No name column found for {year}")
        return pd.DataFrame()

    df = df[df["name"].astype(str).str.strip() != ""]
    df = df[df["name"].astype(str) != "nan"]

    df["is_nhler"] = (df["nhl_gp"] >= 200).astype(int)
    df["is_star"] = (
        (df["nhl_gp"] >= 200) &
        (df["nhl_points"] / df["nhl_gp"].replace(0, np.nan) >= 0.40)
    ).fillna(False).astype(int)

    df["player_id"] = (
        df["name"].str.lower().str.replace(r"[^a-z0-9]", "_", regex=True)
        + "_" + df["draft_year"].astype(str)
    )

    df["position"] = df.get("position", pd.Series(["F"] * len(df))).fillna("F")
    df["position"] = df["position"].apply(_normalize_position)

    keep = ["player_id", "name", "draft_year", "draft_round", "draft_pick",
            "draft_team", "position", "nhl_gp", "nhl_goals", "nhl_assists",
            "nhl_points", "is_nhler", "is_star"]
    keep = [c for c in keep if c in df.columns]
    return df[keep]


def scrape_draft_history(start: int = 1995, end: int = 2019,
                          delay: float = 3.5) -> pd.DataFrame:
    """
    Scrape multiple draft years and return combined DataFrame.
    Uses a polite delay between requests to respect Hockey Reference rate limits.
    Years up to ~2019 have meaningful NHL career outcomes for training.
    """
    frames = []
    for year in range(start, end + 1):
        df = scrape_draft_year(year)
        if not df.empty:
            frames.append(df)
            logger.info(f"  {year}: {len(df)} picks, "
                        f"{df['is_nhler'].sum()} NHLers ({df['is_nhler'].mean():.1%})")
        time.sleep(delay)  # HR asks for ~3-4s between requests

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Total: {len(combined)} picks across {end-start+1} drafts. "
                f"NHLers: {combined['is_nhler'].sum()} ({combined['is_nhler'].mean():.1%})")
    return combined


def _normalize_position(pos: str) -> str:
    pos = str(pos).upper().strip()
    if any(x in pos for x in ["C", "LW", "RW", "W", "F"]):
        return "F"
    if "D" in pos or "DEF" in pos:
        return "D"
    if "G" in pos:
        return "G"
    return "F"
