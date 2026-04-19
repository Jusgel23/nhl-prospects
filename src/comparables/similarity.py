"""
Player comparables engine.
Finds the top-N most similar historical prospects at the same development stage.
Uses z-score normalized Euclidean distance.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)

# Features used for similarity — nhle_ppg weighted 2× by duplicating
SIMILARITY_FEATURES = [
    "nhle_ppg",           # ×2 weight via duplication below
    "nhle_ppg",
    "ppg_delta",
    "nhle_gpg",
    "age_at_draft",
    "birth_quarter",
    "height_cm",
    "weight_kg",
    "gp_rate",
    "pp_pts_pct",
    "pim_per_game",
]

OUTCOME_LABELS = {
    (True, True):  "Elite (Star NHLer)",
    (True, False): "NHLer (Role Player)",
    (False, False):"Did Not Reach NHL",
}


def build_comparable_index(historical_df: pd.DataFrame) -> "ComparableIndex":
    """
    Build a per-position nearest-neighbor index from historical draft class data.
    historical_df: feature matrix with SIMILARITY_FEATURES + is_forward + outcome cols.
    """
    return ComparableIndex(historical_df)


class ComparableIndex:
    def __init__(self, historical_df: pd.DataFrame):
        self._data = historical_df.copy()
        self._models: dict[str, tuple[NearestNeighbors, StandardScaler, pd.DataFrame]] = {}
        self._build()

    def _build(self):
        for pos, pos_val in [("F", 1), ("D", 0)]:
            sub = self._data[self._data["is_forward"] == pos_val].copy()
            if len(sub) < 10:
                logger.warning(f"Not enough {pos} comps ({len(sub)}) to build index.")
                continue

            X = _extract_sim_features(sub)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            nn = NearestNeighbors(n_neighbors=min(10, len(sub)),
                                  metric="euclidean", algorithm="ball_tree")
            nn.fit(X_scaled)
            self._models[pos] = (nn, scaler, sub.reset_index(drop=True))
            logger.info(f"Comparables index built for {pos}: {len(sub)} players.")

    def find_comparables(self, prospect_row: pd.Series,
                          n: int = 5) -> pd.DataFrame:
        """
        Return top-n comparables for a single prospect.
        Returns DataFrame with columns: name, draft_year, similarity_score,
        nhler_probability (from outcomes), nhl_outcome_label, nhl_gp, nhl_points.
        """
        pos = "F" if prospect_row.get("is_forward", 1) == 1 else "D"
        if pos not in self._models:
            logger.warning(f"No comparable index for position {pos}.")
            return pd.DataFrame()

        nn, scaler, hist_df = self._models[pos]

        x = _extract_sim_features(prospect_row.to_frame().T)
        x_scaled = scaler.transform(x)

        dists, indices = nn.kneighbors(x_scaled, n_neighbors=min(n + 1, len(hist_df)))
        dists = dists[0]
        indices = indices[0]

        rows = []
        for dist, idx in zip(dists, indices):
            comp = hist_df.iloc[idx]

            # Skip if same player (e.g. prospect is in historical set)
            if comp.get("player_id") == prospect_row.get("player_id"):
                continue

            sim_score = _dist_to_similarity(dist)
            is_nhler = bool(comp.get("is_nhler", 0))
            is_star  = bool(comp.get("is_star", 0))
            outcome  = OUTCOME_LABELS.get((is_nhler, is_star), "Unknown")

            rows.append({
                "name":             comp.get("name", "Unknown"),
                "player_id":        comp.get("player_id", ""),
                "draft_year":       comp.get("draft_year", ""),
                "similarity_score": sim_score,
                "nhl_gp":           int(comp.get("nhl_gp", 0)),
                "nhl_points":       int(comp.get("nhl_points", 0)),
                "is_nhler":         is_nhler,
                "is_star":          is_star,
                "outcome_label":    outcome,
            })
            if len(rows) == n:
                break

        return pd.DataFrame(rows)

    def bulk_comparables(self, prospects_df: pd.DataFrame,
                          n: int = 5) -> dict[str, pd.DataFrame]:
        """Return {player_id: comparables_df} for all prospects."""
        result = {}
        for _, row in prospects_df.iterrows():
            pid = row.get("player_id", str(_))
            result[pid] = self.find_comparables(row, n=n)
        return result

    def top_comparable_str(self, prospect_row: pd.Series) -> str:
        comps = self.find_comparables(prospect_row, n=1)
        if comps.empty:
            return "N/A"
        c = comps.iloc[0]
        return f"{c['name']} ({c['draft_year']}) — sim {c['similarity_score']}"


# ── helpers ───────────────────────────────────────────────────────────────────

def _extract_sim_features(df: pd.DataFrame) -> np.ndarray:
    cols = SIMILARITY_FEATURES
    out = []
    for col in cols:
        if col in df.columns:
            out.append(pd.to_numeric(df[col], errors="coerce").fillna(0).values)
        else:
            out.append(np.zeros(len(df)))
    return np.column_stack(out)


def _dist_to_similarity(dist: float, scale: float = 3.0) -> int:
    """Convert Euclidean distance to a 0–100 similarity score."""
    sim = 100 * np.exp(-dist / scale)
    return max(0, min(100, int(round(sim))))
