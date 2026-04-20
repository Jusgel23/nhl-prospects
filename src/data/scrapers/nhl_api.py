"""
NHL.com API integration. Pulls bios + career stats for any player in the
NHL system (drafted or signed) — our cleanest source for current prospect
data that the EliteProspects bio scraper can't get anymore.

Flow per prospect:
  1. Search `search.d3.nhle.com/api/v1/search/player?q=<name>` → NHL playerId
  2. Fetch `NHLClient().stats.player_career_stats(playerId)` → full bio +
     seasonTotals (all leagues: NHL, AHL, OHL, NCAA, KHL, etc.)
  3. Map into our players + seasons schema.

Per-player JSON is cached to data/raw/nhl_careers/{nhl_id}.json.
"""
from __future__ import annotations

import json
import logging
import re
import time
import urllib.parse
from pathlib import Path
from typing import Optional

import requests
from nhlpy import NHLClient

logger = logging.getLogger(__name__)

NHL_SEARCH_URL = "https://search.d3.nhle.com/api/v1/search/player"
CACHE_DIR = Path(__file__).parents[3] / "data" / "raw" / "nhl_careers"
ID_MAP_PATH = Path(__file__).parents[3] / "data" / "historical" / "nhl_id_map.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# NHL API is generous — 0.3s between requests is plenty
NHL_DELAY = 0.3

_client: Optional[NHLClient] = None


def _nhl() -> NHLClient:
    global _client
    if _client is None:
        _client = NHLClient()
    return _client


def _normalize_name_for_search(name: str) -> str:
    """Strip the position suffix and any unicode accents for the search query."""
    name = re.sub(r"\s*\([^)]*\)\s*$", "", name).strip()
    # Strip diacritics so "Tomáš Žižka" → "Tomas Zizka"
    import unicodedata
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    return name


def search_nhl_id(name: str, include_inactive: bool = True) -> Optional[dict]:
    """
    Look up an NHL playerId from a name. Returns the top-matching result as a
    dict (with playerId, birthCountry, positionCode, etc.) or None.
    Tries active-only first, then includes inactive if no match.
    """
    q = _normalize_name_for_search(name)
    if not q:
        return None

    for active in (True, False) if include_inactive else (True,):
        url = (
            f"{NHL_SEARCH_URL}?culture=en-us&limit=5&q={urllib.parse.quote(q)}"
            f"&active={'true' if active else 'false'}"
        )
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            hits = resp.json()
        except requests.RequestException as e:
            logger.warning(f"NHL search failed for {name!r}: {e}")
            continue

        if hits:
            return hits[0]

    return None


def _cache_path(nhl_id: str) -> Path:
    return CACHE_DIR / f"{nhl_id}.json"


def fetch_career(nhl_id: str, use_cache: bool = True) -> Optional[dict]:
    """Fetch full player record (bio + seasonTotals) from the NHL API."""
    cache = _cache_path(nhl_id)
    if use_cache and cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except Exception:
            pass  # corrupt cache → refetch

    try:
        data = _nhl().stats.player_career_stats(str(nhl_id))
    except Exception as e:
        logger.warning(f"NHL career fetch failed for {nhl_id}: {e}")
        return None

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data, default=str), encoding="utf-8")
    return data


# ── Transformers into our schema ─────────────────────────────────────────────

def _text(v) -> Optional[str]:
    """NHL API sometimes returns {default: ..., fr: ...} dicts for text."""
    if isinstance(v, dict):
        return v.get("default")
    return v


def _shoots(code: Optional[str]) -> Optional[str]:
    if not code:
        return None
    c = code.upper().strip()
    if c.startswith("L"): return "Left"
    if c.startswith("R"): return "Right"
    return None


def _position(code: Optional[str]) -> Optional[str]:
    if not code:
        return None
    c = code.upper().strip()
    if c in ("C", "LW", "RW", "W", "F"): return "F"
    if c == "D": return "D"
    if c == "G": return "G"
    return None


def to_bio_row(career: dict, player_id_for_db: str) -> dict:
    """Map a NHL API career dict into our players-table row."""
    draft = career.get("draftDetails") or {}
    return {
        "player_id":   player_id_for_db,
        "name":        f"{_text(career.get('firstName'))} {_text(career.get('lastName'))}".strip(),
        "dob":         career.get("birthDate"),
        "nationality": career.get("birthCountry"),
        "position":    _position(career.get("position")),
        "height_cm":   career.get("heightInCentimeters"),
        "weight_kg":   career.get("weightInKilograms"),
        "shoots":      _shoots(career.get("shootsCatches")),
        "source":      "nhl_api",
        "draft_year":  draft.get("year"),
        "draft_round": draft.get("round"),
        "draft_pick":  draft.get("overallPick"),
        "draft_team":  draft.get("teamAbbrev"),
    }


def to_season_rows(career: dict, player_id_for_db: str) -> list[dict]:
    """Map seasonTotals into our seasons-table rows.
    NHL API returns one entry per (season, league, team). Skip playoff rows
    (gameTypeId==3) — we only train on regular-season for now."""
    rows = []
    for s in career.get("seasonTotals", []) or []:
        if s.get("gameTypeId") == 3:
            continue
        season_raw = str(s.get("season", ""))
        season = (f"{season_raw[:4]}-{season_raw[4:]}"
                  if len(season_raw) == 8 else season_raw)
        rows.append({
            "player_id":  player_id_for_db,
            "season":     season,
            "league":     (s.get("leagueAbbrev") or "").upper(),
            "team":       _text(s.get("teamName")) or s.get("teamName", ""),
            "gp":         s.get("gamesPlayed") or 0,
            "goals":      s.get("goals") or 0,
            "assists":    s.get("assists") or 0,
            "points":     s.get("points") or 0,
            "pim":        s.get("pim") or 0,
            "plus_minus": s.get("plusMinus") or 0,
            "pp_goals":   s.get("powerPlayGoals") or 0,
            "pp_assists": 0,  # NHL API doesn't split PP assists separately
        })
    return rows


# ── Orchestrator ─────────────────────────────────────────────────────────────

def enrich_player(name: str, player_id_for_db: str,
                   id_map: Optional[dict] = None) -> Optional[dict]:
    """Full pipeline: search → fetch → return (bio, seasons) rows."""
    nhl_id = (id_map or {}).get(player_id_for_db)
    if not nhl_id:
        hit = search_nhl_id(name)
        time.sleep(NHL_DELAY)
        if not hit:
            return None
        nhl_id = str(hit["playerId"])
        if id_map is not None:
            id_map[player_id_for_db] = nhl_id

    career = fetch_career(nhl_id)
    if career is None:
        return None

    return {
        "bio":     to_bio_row(career, player_id_for_db),
        "seasons": to_season_rows(career, player_id_for_db),
        "nhl_id":  nhl_id,
    }


def load_id_map() -> dict:
    if ID_MAP_PATH.exists():
        return json.loads(ID_MAP_PATH.read_text(encoding="utf-8"))
    return {}


def save_id_map(id_map: dict) -> None:
    ID_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    ID_MAP_PATH.write_text(json.dumps(id_map, indent=2), encoding="utf-8")
