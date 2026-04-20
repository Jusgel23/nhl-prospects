"""
Hockey Reference draft history + career scraper.
Pulls historical NHL draft picks + per-player multi-season career histories.
No API key, no manual download.
"""
import json
import time
import logging
import re
from pathlib import Path
from typing import Optional

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

# Cache directories for career + ID-mapping data.
_RAW_ROOT = Path(__file__).parents[3] / "data" / "raw"
HR_CAREER_CACHE = _RAW_ROOT / "hr_careers"
HR_ID_MAPPING_CACHE = Path(__file__).parents[3] / "data" / "historical" / "hr_id_mapping.json"

# Polite delay for HR career pages — robots.txt asks for no more than
# 20 requests per minute, so ~3.5s is a safe floor. HR in practice
# starts returning 429s at around 18 req/min, so 4-5s is safer.
HR_CAREER_DELAY = 4.5
HR_RATE_LIMIT_BACKOFF = 120.0  # seconds to wait after a 429


def _hr_get(url: str, timeout: int = 20, max_retries: int = 3) -> Optional[requests.Response]:
    """GET with 429 backoff. Returns Response on success, None after final failure."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout)
            if resp.status_code == 429:
                wait = HR_RATE_LIMIT_BACKOFF * (attempt + 1)
                logger.warning(f"429 from HR on attempt {attempt+1}; backing off {wait:.0f}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                logger.warning(f"HR fetch failed for {url}: {e}")
                return None
            time.sleep(5 * (attempt + 1))
    return None


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


# ─────────────────────────────────────────────────────────────────────────────
# HR ID mapping — bridges synthetic "{name}_{draft_year}" IDs to real HR IDs
# ─────────────────────────────────────────────────────────────────────────────

def _synthetic_id(name: str, draft_year: int) -> str:
    """Match the synthetic player_id format already used in scrape_draft_year."""
    return (
        re.sub(r"[^a-z0-9]", "_", name.lower())
        + "_" + str(int(draft_year))
    )


def _extract_hr_ids_for_year(year: int) -> dict[str, str]:
    """
    Return {synthetic_player_id: hr_player_id} for one draft year.
    Re-parses the HR draft page with BeautifulSoup to capture anchor hrefs
    (pd.read_html drops them).
    """
    url = f"{HR_BASE}/draft/NHL_{year}_entry.html"
    resp = _hr_get(url)
    if resp is None:
        return {}

    # HR often wraps draft tables in HTML comments; strip them so BS4 sees the table.
    html = resp.text.replace("<!--", "").replace("-->", "")
    soup = BeautifulSoup(html, "lxml")

    mapping: dict[str, str] = {}
    for a in soup.select("table tbody tr td a[href^='/players/']"):
        href = a.get("href", "")
        m = re.search(r"/players/[a-z]/([^.]+)\.html", href)
        if not m:
            continue
        hr_id = m.group(1)
        name = a.get_text(strip=True)
        if not name:
            continue
        mapping[_synthetic_id(name, year)] = hr_id

    return mapping


def build_hr_id_mapping(start: int = 1995, end: int = 2019,
                        force_refresh: bool = False) -> dict[str, str]:
    """
    Build (or load from cache) {synthetic_id: hr_player_id} for every draft
    year in range. Cached to data/historical/hr_id_mapping.json.
    """
    if HR_ID_MAPPING_CACHE.exists() and not force_refresh:
        logger.info(f"Loading HR ID mapping from cache: {HR_ID_MAPPING_CACHE}")
        return json.loads(HR_ID_MAPPING_CACHE.read_text(encoding="utf-8"))

    logger.info(f"Building HR ID mapping for drafts {start}-{end}...")
    mapping: dict[str, str] = {}
    for year in range(start, end + 1):
        year_map = _extract_hr_ids_for_year(year)
        mapping.update(year_map)
        logger.info(f"  {year}: {len(year_map)} IDs extracted (total: {len(mapping)})")
        time.sleep(HR_CAREER_DELAY)

    HR_ID_MAPPING_CACHE.parent.mkdir(parents=True, exist_ok=True)
    HR_ID_MAPPING_CACHE.write_text(json.dumps(mapping), encoding="utf-8")
    logger.info(f"Cached {len(mapping)} HR ID mappings to {HR_ID_MAPPING_CACHE}")
    return mapping


# ─────────────────────────────────────────────────────────────────────────────
# Player career scraper
# ─────────────────────────────────────────────────────────────────────────────

def _hr_career_cache_path(hr_player_id: str) -> Path:
    return HR_CAREER_CACHE / f"{hr_player_id}.json"


def _parse_hr_bio(soup: BeautifulSoup) -> dict:
    """
    Extract DOB, height, weight, shoots, position, nationality, and draft
    info from HR's #meta section. Format is inconsistent (colons may have
    newlines around them, unicode non-breaking spaces, etc.) so the regexes
    below are deliberately permissive.
    """
    bio = {}
    meta = soup.find("div", id="meta")
    if not meta:
        return bio

    text = meta.get_text(" ", strip=True)

    # DOB — HR usually exposes it as data-birth="YYYY-MM-DD"
    dob_span = meta.find("span", {"data-birth": True})
    if dob_span:
        bio["dob"] = dob_span["data-birth"]
    elif meta.find(id="necro-birth"):
        # Some pages use <span id="necro-birth" data-birth="...">
        el = meta.find(id="necro-birth")
        if el.get("data-birth"):
            bio["dob"] = el["data-birth"]

    if "dob" not in bio:
        m = re.search(r"([A-Z][a-z]+)\s+(\d{1,2}),\s+(\d{4})", text)
        if m:
            try:
                from datetime import datetime
                bio["dob"] = datetime.strptime(
                    f"{m.group(1)} {m.group(2)} {m.group(3)}", "%B %d %Y"
                ).strftime("%Y-%m-%d")
            except Exception:
                pass

    # Height/weight — HR shows "6-2, 214lb (188cm, 97kg)"
    h_match = re.search(r"(\d{3})\s*cm", text)
    if h_match:
        bio["height_cm"] = float(h_match.group(1))

    w_match = re.search(r"(\d{2,3})\s*kg", text)
    if w_match:
        bio["weight_kg"] = float(w_match.group(1))

    # Position / Shoots — permissive: HR inserts "\n: " between label and value.
    # Accept any whitespace/colon separator and capture letters+slashes.
    shoots_match = re.search(r"Shoots?\s*:?\s*(Left|Right)", text, re.IGNORECASE)
    if shoots_match:
        bio["shoots"] = shoots_match.group(1).capitalize()

    pos_match = re.search(r"Position\s*:?\s*([A-Z/]+)", text)
    if pos_match:
        bio["position"] = _normalize_position(pos_match.group(1))

    # Nationality: HR writes "Born: ... in City, Province/State CountryCode"
    # The last 2-letter lowercase token is the country code (ca, us, ru, se).
    nat_match = re.search(r"in\s+[^,]+,\s+[^,]+?\s+([a-z]{2,3})\b", text)
    if nat_match:
        bio["nationality"] = nat_match.group(1).upper()

    # Draft info: "Draft: Detroit, 2nd round (46th overall), 2016 NHL Entry"
    draft_match = re.search(
        r"Draft\s*:?\s*([A-Za-z .]+),\s*(\d+)(?:st|nd|rd|th)?\s*round\s*\((\d+)(?:st|nd|rd|th)?\s*overall\),\s*(\d{4})",
        text,
    )
    if draft_match:
        bio["draft_team"] = draft_match.group(1).strip()
        bio["draft_round"] = int(draft_match.group(2))
        bio["draft_pick"] = int(draft_match.group(3))
        bio["draft_year"] = int(draft_match.group(4))

    return bio


def _parse_career_table(soup: BeautifulSoup, table_id: str) -> list[dict]:
    """
    Parse one of HR's career tables. The two IDs that matter:
      - `player_stats`               → NHL regular-season rows
      - `stats_basic_minus_other`    → amateur/junior/non-NHL rows
    Returns list of per-season dicts ready for the seasons table.
    """
    table = soup.find("table", id=table_id)
    if table is None:
        return []

    body = table.find("tbody")
    if body is None:
        return []

    rows: list[dict] = []
    for tr in body.find_all("tr"):
        klass = tr.get("class") or []
        if "thead" in klass or "spacer" in klass:
            continue

        cells = {
            c.get("data-stat", ""): c.get_text(strip=True)
            for c in tr.find_all(["th", "td"])
            if c.get("data-stat")
        }

        # "season" in stats_basic_minus_other, "year_id" in player_stats
        season = cells.get("season") or cells.get("year_id")
        if not season or len(season) < 4:
            continue
        # HR sometimes puts career-summary rows at the bottom with season like "7 seasons"
        if not re.match(r"^\d{4}-\d{2,4}$", season):
            continue

        # Column names differ between NHL and amateur tables:
        gp = cells.get("games") or cells.get("games_played")
        league = (cells.get("lg_id") or cells.get("comp_name_abbr") or "").upper()
        team = cells.get("team_id") or cells.get("team_name_abbr") or cells.get("team_name") or ""

        # Skip mid-season trade summary rows (team "2TM", "3TM" etc.) — HR
        # already lists the individual team rows separately. Keeping the
        # summary would double-count the player's stats.
        if re.match(r"^\d+TM$", team):
            continue

        rows.append({
            "season":     season,
            "age":        _safe_int(cells.get("age")),
            "team":       team,
            "league":     league,
            "gp":         _safe_int(gp),
            "goals":      _safe_int(cells.get("goals")),
            "assists":    _safe_int(cells.get("assists")),
            "points":     _safe_int(cells.get("points")),
            "pim":        _safe_int(cells.get("pen_min")),
            "plus_minus": _safe_int(cells.get("plus_minus")),
            "pp_goals":   _safe_int(cells.get("goals_pp")),
            "pp_assists": _safe_int(cells.get("assists_pp")),
        })
    return rows


def _safe_int(v) -> int:
    try:
        return int(str(v).strip() or 0)
    except (ValueError, TypeError):
        return 0


def scrape_player_career(hr_player_id: str,
                          use_cache: bool = True) -> Optional[dict]:
    """
    Fetch one Hockey Reference player page and return a dict with bio +
    full season-by-season career.

    Structure:
        {
            "hr_player_id": "mcdavco01",
            "name": "Connor McDavid",
            "dob": "1997-01-13",
            "height_cm": 185, "weight_kg": 88,
            "shoots": "Left", "position": "F", "nationality": "CAN",
            "seasons": [
                {"season": "2012-13", "league": "OHL", "team": "Erie",
                 "gp": 25, "goals": 3, "assists": 22, "points": 25, ...},
                ...
            ],
        }

    Cached to data/raw/hr_careers/{hr_player_id}.json on first fetch.
    """
    cache = _hr_career_cache_path(hr_player_id)
    if use_cache and cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Corrupt career cache for {hr_player_id}: {e}; refetching.")

    first_letter = hr_player_id[0].lower()
    url = f"{HR_BASE}/players/{first_letter}/{hr_player_id}.html"
    resp = _hr_get(url)
    if resp is None:
        return None

    # HR hides many tables inside HTML comments. Strip comment markers BEFORE
    # parsing so BS4 sees them as real tables. (Safe because HR never wraps
    # bio content in comments.)
    html = resp.text.replace("<!--", "").replace("-->", "")
    soup = BeautifulSoup(html, "lxml")

    name_tag = soup.find("h1")
    name = name_tag.get_text(strip=True) if name_tag else ""

    bio = _parse_hr_bio(soup)

    # Two tables carry every stat we care about:
    # - player_stats            → NHL regular-season rows
    # - stats_basic_minus_other → amateur / junior / minor / European rows
    seasons: list[dict] = []
    for table_id in ("player_stats", "stats_basic_minus_other"):
        seasons.extend(_parse_career_table(soup, table_id))

    # De-dup on (season, league, team) — same season could be listed twice
    seen = set()
    unique_seasons = []
    for s in seasons:
        key = (s["season"], s["league"], s["team"])
        if key in seen:
            continue
        seen.add(key)
        unique_seasons.append(s)

    result = {
        "hr_player_id": hr_player_id,
        "name": name,
        **bio,
        "seasons": unique_seasons,
    }

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result), encoding="utf-8")
    return result


def scrape_all_careers(hr_player_ids: list[str],
                        delay: float = HR_CAREER_DELAY,
                        progress_every: int = 50) -> list[dict]:
    """
    Loop over HR player IDs, fetching each player's career. Respects per-player
    JSON cache so crashes only lose the currently-in-flight player.
    """
    results: list[dict] = []
    total = len(hr_player_ids)
    misses = 0
    logger.info(f"Scraping careers for {total} players (~{total * delay / 60:.0f} min at {delay}s/player)...")

    for i, pid in enumerate(hr_player_ids, 1):
        already_cached = _hr_career_cache_path(pid).exists()
        data = scrape_player_career(pid)
        if data is None:
            misses += 1
        else:
            results.append(data)

        if not already_cached:
            time.sleep(delay)

        if i % progress_every == 0 or i == total:
            logger.info(f"  {i}/{total} processed ({misses} misses, {len(results)} hits)")

    logger.info(f"Done. {len(results)} careers scraped/loaded, {misses} misses.")
    return results
