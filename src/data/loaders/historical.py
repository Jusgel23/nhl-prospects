"""
Historical draft outcomes loader.
Scrapes from Hockey Reference automatically — no manual downloads required.
Caches results locally to avoid re-scraping on every run.
"""
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

CACHE_PATH = Path(__file__).parents[3] / "data" / "historical" / "draft_outcomes_cache.csv"


def load_draft_outcomes(start: int = 1996, end: int = 2026,
                         force_refresh: bool = False) -> pd.DataFrame:
    """
    Return historical draft outcomes, scraping Hockey Reference if not cached.
    Results are cached to data/historical/draft_outcomes_cache.csv.
    Set force_refresh=True to re-scrape.
    """
    if CACHE_PATH.exists() and not force_refresh:
        logger.info(f"Loading draft outcomes from cache: {CACHE_PATH}")
        df = pd.read_csv(CACHE_PATH)
        logger.info(f"Loaded {len(df)} picks from cache. "
                    f"NHLers: {df['is_nhler'].sum()} ({df['is_nhler'].mean():.1%})")
        return df

    logger.info(f"Cache not found — scraping Hockey Reference ({start}-{end})...")
    from src.data.scrapers.hockey_reference import scrape_draft_history
    df = scrape_draft_history(start=start, end=end)

    if df.empty:
        logger.error("No draft data scraped.")
        return _empty_outcomes()

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_PATH, index=False)
    logger.info(f"Cached {len(df)} picks to {CACHE_PATH}")
    return df


def _empty_outcomes() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "player_id", "name", "draft_year", "draft_round", "draft_pick",
        "draft_team", "position", "nhl_gp", "nhl_goals", "nhl_assists",
        "nhl_points", "is_nhler", "is_star"
    ])
