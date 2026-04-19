"""
USCHO.com NCAA hockey stats scraper (free, no API key required).
"""
import time
import logging
import re

import requests
import pandas as pd
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

USCHO_BASE = "https://www.uscho.com"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def _season_to_uscho(season: str) -> str:
    """'2024-2025' → '20242025'"""
    return season.replace("-", "")


def scrape_ncaa_stats(season: str, division: str = "d1") -> pd.DataFrame:
    """
    Scrape NCAA individual scoring stats from USCHO.com.
    Returns DataFrame with standard season schema columns.
    """
    season_code = _season_to_uscho(season)
    url = f"{USCHO_BASE}/stats/{division}/men/{season_code}/8/all/skaters/g"
    logger.info(f"Scraping NCAA {division.upper()} {season} from USCHO: {url}")

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"USCHO request failed: {e}")
        return pd.DataFrame()

    soup = BeautifulSoup(resp.text, "lxml")
    table = soup.find("table", id=re.compile(r"stats|skater", re.I))
    if table is None:
        table = soup.find("table")

    if table is None:
        logger.warning(f"No table found on USCHO for {season}")
        return pd.DataFrame()

    rows = []
    thead = table.find("thead")
    col_names = []
    if thead:
        col_names = [th.get_text(strip=True).lower() for th in thead.find_all("th")]

    tbody = table.find("tbody")
    if tbody is None:
        return pd.DataFrame()

    for tr in tbody.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if len(cells) < 6:
            continue

        player_link = tr.find("a", href=re.compile(r"/players?/"))
        if not player_link:
            continue

        name = player_link.get_text(strip=True)
        href = player_link.get("href", "")
        player_id = f"uscho_{_slugify(name)}"

        texts = [c.get_text(strip=True) for c in cells]

        def get_col(*aliases):
            for alias in aliases:
                for i, col in enumerate(col_names):
                    if alias in col and i < len(texts):
                        return texts[i]
            return "0"

        def safe_int(v):
            try:
                return int(str(v).replace("-", "0"))
            except (ValueError, TypeError):
                return 0

        gp     = safe_int(get_col("gp", "games"))
        goals  = safe_int(get_col("g", "goal"))
        assists= safe_int(get_col("a", "assist"))
        points = safe_int(get_col("pts", "point"))
        pim    = safe_int(get_col("pim", "pen"))
        pm     = safe_int(get_col("+/-", "pm"))

        team_cell = tr.find("td", class_=re.compile(r"team"))
        team = team_cell.get_text(strip=True) if team_cell else ""

        pos_cell = tr.find("td", class_=re.compile(r"pos"))
        raw_pos = pos_cell.get_text(strip=True) if pos_cell else ""
        position = _normalize_position(raw_pos)

        gp = gp or 1
        rows.append({
            "player_id":   player_id,
            "name":        name,
            "season":      season,
            "league":      "NCAA",
            "team":        team,
            "position":    position,
            "gp":          gp,
            "goals":       goals,
            "assists":     assists,
            "points":      points,
            "pim":         pim,
            "plus_minus":  pm,
            "pp_goals":    0,
            "pp_assists":  0,
            "ppg":         round(points / gp, 3),
            "gp_rate":     None,
        })
        time.sleep(0.02)

    df = pd.DataFrame(rows)
    logger.info(f"  → {len(df)} NCAA players scraped for {season}")
    return df


def scrape_ncaa_multiple_seasons(seasons: list[str]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        df = scrape_ncaa_stats(season)
        if not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "_", name.lower().strip())


def _normalize_position(pos: str) -> str:
    pos = pos.upper().strip()
    if any(x in pos for x in ["C", "LW", "RW", "W", "F"]):
        return "F"
    if "D" in pos:
        return "D"
    if "G" in pos:
        return "G"
    return "F"
