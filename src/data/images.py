"""
NHL headshot + placeholder helpers.

The enrich-nhl-bios CLI subcommand populates `data/historical/nhl_id_map.json`
mapping each local player_id (EP or HR synthetic) to the canonical NHL API
playerId. This module reads that map and exposes the deterministic headshot
URL for each player, with a silhouette-SVG fallback for the handful of
players we couldn't match in the NHL API.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

_ID_MAP_PATH = Path(__file__).parents[2] / "data" / "historical" / "nhl_id_map.json"
_PLACEHOLDER_PATH = Path(__file__).parents[2] / "assets" / "player_placeholder.svg"

# Deterministic NHL asset pattern — /latest/ handles retired/traded players
# gracefully and falls back to a generic placeholder on NHL's side if the
# mug is missing for a given player.
_HEADSHOT_TEMPLATE = "https://assets.nhle.com/mugs/nhl/latest/{nhl_id}.png"


@lru_cache(maxsize=1)
def _load_id_map() -> dict[str, str]:
    if not _ID_MAP_PATH.exists():
        return {}
    try:
        return json.loads(_ID_MAP_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


@lru_cache(maxsize=1)
def _placeholder_data_url() -> str:
    """Return a data: URL for the bundled silhouette SVG (or a minimal inline
    fallback if the asset file doesn't exist)."""
    if _PLACEHOLDER_PATH.exists():
        import base64
        svg = _PLACEHOLDER_PATH.read_bytes()
        return "data:image/svg+xml;base64," + base64.b64encode(svg).decode()
    # Minimal inline fallback (grey circle + gender-neutral silhouette)
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
        '<rect width="64" height="64" fill="#D5DBDB"/>'
        '<circle cx="32" cy="24" r="10" fill="#7B7D7D"/>'
        '<path d="M12 56 Q12 38 32 38 Q52 38 52 56 Z" fill="#7B7D7D"/>'
        '</svg>'
    )
    import base64
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def nhl_id_for(player_id: str) -> Optional[str]:
    """Return the NHL playerId for a local player_id, or None if unmapped."""
    return _load_id_map().get(str(player_id))


def headshot_url(player_id: str) -> str:
    """Return the NHL headshot URL for a player, or a silhouette placeholder
    if we don't have an NHL ID for them. Always returns a renderable URL so
    Streamlit's ImageColumn never shows a broken-image icon."""
    nhl_id = nhl_id_for(player_id)
    if nhl_id:
        return _HEADSHOT_TEMPLATE.format(nhl_id=nhl_id)
    return _placeholder_data_url()
