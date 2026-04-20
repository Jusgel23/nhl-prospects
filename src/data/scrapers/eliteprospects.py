"""
EliteProspects scraper.
Uses the eliteprospect-scraper PyPI package where available,
falls back to direct HTTP requests against the public EP website.
"""
import json
import time
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
import pandas as pd
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

EP_BASE = "https://www.eliteprospects.com"

# Raw scrape cache — makes re-runs nearly instant for already-fetched data.
# Both directories are gitignored via data/raw/ in .gitignore.
RAW_DIR = Path(__file__).parents[3] / "data" / "raw"
LEAGUE_STATS_CACHE = RAW_DIR / "league_stats"
BIOS_CACHE = RAW_DIR / "bios"


def _league_stats_cache_path(league: str, season: str) -> Path:
    return LEAGUE_STATS_CACHE / f"{league.upper()}_{season}.csv"


def _bio_cache_path(player_id: str) -> Path:
    return BIOS_CACHE / f"{player_id}.json"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

CHL_LEAGUES = ["OHL", "WHL", "QMJHL"]
NCAA_LEAGUES = ["NCAA"]
TARGET_LEAGUES = CHL_LEAGUES + NCAA_LEAGUES

LEAGUE_SLUGS = {
    "OHL":   "ohl",
    "WHL":   "whl",
    "QMJHL": "qmjhl",
    "NCAA":  "ncaa",
}


def _get(url: str, params: dict = None, retries: int = 3) -> requests.Response:
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def scrape_league_stats(league: str, season: str,
                         use_cache: bool = True) -> pd.DataFrame:
    """
    Scrape player stats for a given league and season from EliteProspects.
    season format: '2024-2025'
    Returns DataFrame with columns matching the seasons table schema.
    If use_cache=True and a local CSV cache exists, loads from it instead.
    """
    slug = LEAGUE_SLUGS.get(league.upper())
    if not slug:
        raise ValueError(f"Unknown league: {league}")

    cache = _league_stats_cache_path(league, season)
    if use_cache and cache.exists():
        logger.info(f"[cache] Loading {league} {season} from {cache.name}")
        df = pd.read_csv(cache)
        return _clean_stats_df(df)

    url = f"{EP_BASE}/league/{slug}/stats/{season}"
    logger.info(f"Scraping {league} {season} from {url}")

    resp = _get(url)
    soup = BeautifulSoup(resp.text, "lxml")

    rows = []
    table = soup.find("table", class_=re.compile(r"standings|player-stats|stats"))
    if table is None:
        # Try alternate table structure
        table = soup.find("table")

    if table is None:
        logger.warning(f"No stats table found for {league} {season}")
        return pd.DataFrame()

    headers_row = table.find("thead")
    col_names = []
    if headers_row:
        col_names = [th.get_text(strip=True).lower() for th in headers_row.find_all("th")]

    for tr in table.find("tbody").find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if len(cells) < 5:
            continue

        player_link = tr.find("a", href=re.compile(r"/player/"))
        if not player_link:
            continue

        href = player_link["href"]
        player_id = _extract_player_id(href)
        name = player_link.get_text(strip=True)

        cell_texts = [c.get_text(strip=True) for c in cells]

        row = {
            "player_id": player_id,
            "name":      name,
            "season":    season,
            "league":    league.upper(),
            "ep_url":    EP_BASE + href,
        }
        row.update(_parse_stat_cells(cell_texts, col_names))
        rows.append(row)
        time.sleep(0.05)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = _clean_stats_df(df)
    logger.info(f"  → {len(df)} players scraped for {league} {season}")

    # Persist raw scrape to cache so future runs skip this request
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)

    return df


def scrape_player_bio(player_id: str, ep_url: str,
                       use_cache: bool = True) -> dict:
    """Scrape biographical data for a single player.
    If use_cache=True and a local JSON cache exists, loads from it instead —
    this is where the bulk of the scrape time is spent (HTTP + polite sleep)."""
    cache = _bio_cache_path(player_id)
    if use_cache and cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Bio cache read failed for {player_id}: {e}; re-fetching.")

    logger.debug(f"Scraping bio: {ep_url}")
    resp = _get(ep_url)
    soup = BeautifulSoup(resp.text, "lxml")

    bio = {"player_id": player_id, "source": "eliteprospects"}

    name_tag = soup.find("h1", class_=re.compile(r"player"))
    if name_tag:
        bio["name"] = name_tag.get_text(strip=True)

    info_div = soup.find("div", class_=re.compile(r"player-details|bio|info"))
    if info_div:
        text = info_div.get_text(" ", strip=True)

        dob_match = re.search(r"(\d{4}-\d{2}-\d{2})", text)
        if dob_match:
            bio["dob"] = dob_match.group(1)

        nat_match = re.search(r"Nationality[:\s]+([A-Za-z ]+)", text)
        if nat_match:
            bio["nationality"] = nat_match.group(1).strip()

        pos_match = re.search(r"Position[:\s]+([A-Za-z/]+)", text)
        if pos_match:
            bio["position"] = _normalize_position(pos_match.group(1))

        ht_match = re.search(r"(\d{3})\s*cm", text)
        if ht_match:
            bio["height_cm"] = float(ht_match.group(1))

        wt_match = re.search(r"(\d{2,3})\s*kg", text)
        if wt_match:
            bio["weight_kg"] = float(wt_match.group(1))

        shoots_match = re.search(r"Shoots[:\s]+(Left|Right)", text, re.IGNORECASE)
        if shoots_match:
            bio["shoots"] = shoots_match.group(1).capitalize()

    # Persist bio to cache before the polite delay
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(bio), encoding="utf-8")
    except Exception as e:
        logger.warning(f"Bio cache write failed for {player_id}: {e}")

    time.sleep(0.3)
    return bio


def scrape_draft_class(seasons: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scrape all CHL + NCAA players across given seasons.
    Returns (players_df, seasons_df).
    """
    all_seasons = []
    all_bios = []

    for season in seasons:
        for league in TARGET_LEAGUES:
            try:
                df = scrape_league_stats(league, season)
                if df.empty:
                    continue

                # Seasons table has no `name` column — name lives in `players`
                all_seasons.append(df[["player_id", "season", "league",
                                       "team", "gp", "goals", "assists", "points",
                                       "pim", "plus_minus", "pp_goals", "pp_assists",
                                       "gp_rate", "ppg"]])

                # Collect bios for new players
                for _, row in df.iterrows():
                    if "ep_url" in row and pd.notna(row.get("ep_url")):
                        try:
                            bio = scrape_player_bio(row["player_id"], row["ep_url"])
                            bio.setdefault("name", row["name"])
                            # Fallback position from the stats-table name suffix
                            # (bio scraper can't currently extract from EP's
                            # redesigned pages).
                            stats_pos = row.get("position")
                            if not bio.get("position") and pd.notna(stats_pos) and stats_pos:
                                bio["position"] = stats_pos
                            all_bios.append(bio)
                        except Exception as e:
                            logger.warning(f"Bio scrape failed for {row['name']}: {e}")

            except Exception as e:
                logger.error(f"Failed scraping {league} {season}: {e}")

    players_df = pd.DataFrame(all_bios).drop_duplicates("player_id") if all_bios else pd.DataFrame()
    seasons_df = pd.concat(all_seasons, ignore_index=True) if all_seasons else pd.DataFrame()
    return players_df, seasons_df


# ── helpers ──────────────────────────────────────────────────────────────────

def _extract_player_id(href: str) -> str:
    match = re.search(r"/player/(\d+)/", href)
    return match.group(1) if match else href.split("/")[-1]


def _normalize_position(pos: str) -> str:
    pos = pos.upper().strip()
    if any(x in pos for x in ["C", "LW", "RW", "F"]):
        return "F"
    if "D" in pos or "DEF" in pos:
        return "D"
    if "G" in pos:
        return "G"
    return pos


def _parse_stat_cells(cells: list[str], col_names: list[str]) -> dict:
    def safe_int(v):
        try:
            return int(str(v).replace("-", "0").strip())
        except (ValueError, TypeError):
            return 0

    def safe_float(v):
        try:
            return float(str(v).replace("-", "0").strip())
        except (ValueError, TypeError):
            return 0.0

    col_map = {c: i for i, c in enumerate(col_names)}

    def get(col_aliases):
        for alias in col_aliases:
            if alias in col_map and col_map[alias] < len(cells):
                return cells[col_map[alias]]
        return "0"

    team_val = get(["team", "tm"])
    gp_val   = get(["gp", "games"])
    g_val    = get(["g", "goals"])
    a_val    = get(["a", "assists", "ast"])
    pts_val  = get(["pts", "points", "tp"])
    pim_val  = get(["pim"])
    pm_val   = get(["+/-", "pm", "plusminus"])
    ppg_val  = get(["ppg", "pp g", "pp goals"])
    ppa_val  = get(["ppa", "pp a", "pp assists"])

    gp = safe_int(gp_val) or 1
    pts = safe_int(pts_val)

    return {
        "team":       team_val,
        "gp":         gp,
        "goals":      safe_int(g_val),
        "assists":    safe_int(a_val),
        "points":     pts,
        "pim":        safe_int(pim_val),
        "plus_minus": safe_int(pm_val),
        "pp_goals":   safe_int(ppg_val),
        "pp_assists": safe_int(ppa_val),
        "ppg":        round(pts / gp, 3),
        "gp_rate":    None,  # filled later once team GP is known
    }


def _clean_stats_df(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = ["gp", "goals", "assists", "points", "pim",
                    "plus_minus", "pp_goals", "pp_assists", "ppg"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Extract position from the name suffix BEFORE stripping it,
    # e.g. "David Goyette(C/LW)" -> name="David Goyette", position="F"
    # This is our only source of position data because the bio scraper
    # currently can't match EP's updated page layout.
    if "name" in df.columns:
        name_series = df["name"].astype(str)
        raw_pos = name_series.str.extract(r"\(([^)]+)\)\s*$", expand=False)
        df["position"] = raw_pos.apply(
            lambda x: _normalize_position(x) if pd.notna(x) else None
        )
        df["name"] = (
            name_series
            .str.replace(r"\s*\([^)]*\)\s*$", "", regex=True)
            .str.strip()
        )

    return df
