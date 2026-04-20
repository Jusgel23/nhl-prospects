"""
One-shot dedup utility.

EP's stats-page scrape sometimes emits multiple player_ids for the same
human (same first/last/DOB). When we then ran NHL API enrichment, each
duplicate EP_id got written as its own row, so top-10 rankings showed
the same person multiple times (Patrick Brown x9, Lane Hutson x3, etc.).

This script collapses duplicates in-place:
  1. Find clusters of (name, dob) with >1 player_id
  2. Keep the "canonical" row (lowest player_id alphabetically)
  3. Rewrite matching rows in `seasons` and `predictions` to use the canonical id
  4. Delete the now-orphaned player rows

Run from the repo root:
    python scripts/dedupe_players.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.data.database import get_conn


def main() -> None:
    with get_conn() as conn:
        cur = conn.cursor()

        # Find dup groups keyed by (name, dob). Pick canonical as min player_id.
        cur.execute("""
            SELECT name, dob, MIN(player_id) AS canonical, COUNT(*) AS n
            FROM players
            WHERE dob IS NOT NULL AND name IS NOT NULL
            GROUP BY name, dob
            HAVING n > 1
        """)
        groups = cur.fetchall()
        print(f"Found {len(groups)} duplicate clusters "
              f"({sum(g[3] for g in groups)} rows total, "
              f"{sum(g[3] - 1 for g in groups)} to delete).")

        total_player_deletes = 0
        total_season_rewrites = 0
        total_pred_deletes = 0

        for name, dob, canonical, n in groups:
            # Collect all player_ids in the cluster
            cur.execute(
                "SELECT player_id FROM players WHERE name = ? AND dob = ?",
                (name, dob),
            )
            ids = [r[0] for r in cur.fetchall()]
            duplicates = [pid for pid in ids if pid != canonical]

            for dup in duplicates:
                # Rewrite seasons to point at canonical (keep UNIQUE constraint safe)
                cur.execute("""
                    INSERT OR IGNORE INTO seasons
                        (player_id, season, league, team, gp, goals, assists,
                         points, pim, plus_minus, pp_goals, pp_assists, gp_rate,
                         ppg, age, nhle_ppg, nhle_gpg, draft_year, draft_label)
                    SELECT ?, season, league, team, gp, goals, assists,
                           points, pim, plus_minus, pp_goals, pp_assists, gp_rate,
                           ppg, age, nhle_ppg, nhle_gpg, draft_year, draft_label
                    FROM seasons WHERE player_id = ?
                """, (canonical, dup))
                cur.execute("DELETE FROM seasons WHERE player_id = ?", (dup,))
                total_season_rewrites += cur.rowcount

                # Drop duplicate predictions (canonical's prediction stays)
                cur.execute("DELETE FROM predictions WHERE player_id = ?", (dup,))
                total_pred_deletes += cur.rowcount

                # Drop the duplicate player row
                cur.execute("DELETE FROM players WHERE player_id = ?", (dup,))
                total_player_deletes += cur.rowcount

        conn.commit()

    print(f"Deleted {total_player_deletes} duplicate player rows.")
    print(f"Rewrote/consolidated {total_season_rewrites} season rows.")
    print(f"Deleted {total_pred_deletes} duplicate prediction rows.")


if __name__ == "__main__":
    main()
