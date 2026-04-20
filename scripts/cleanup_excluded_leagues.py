"""
One-shot cleanup: delete seasons table rows belonging to Tier-4 leagues.

Tier-4 = pre-draft youth / prep-school events (ages 11-16, tiny sample,
uneven competition). See `EXCLUDED_LEAGUE_PATTERNS` in src/models/nhle.py
for the patterns. These rows have no predictive value and pollute
features like peak_nhle_ppg and league_count.

After a successful first run, new ingestions (backfill-careers +
enrich-nhl-bios) filter these out at upsert time, so this script only
needs to run once to purge whatever's already landed in the DB.

Run from the repo root:
    python scripts/cleanup_excluded_leagues.py

Dry-run:
    python scripts/cleanup_excluded_leagues.py --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.data.database import get_conn
from src.models.nhle import is_excluded_league


def main(dry_run: bool = False) -> None:
    with get_conn() as conn:
        # Pull every distinct league currently in the seasons table.
        cur = conn.cursor()
        cur.execute(
            "SELECT league, COUNT(*) AS n "
            "FROM seasons "
            "WHERE league IS NOT NULL "
            "GROUP BY league"
        )
        rows = cur.fetchall()

        excluded_leagues = [
            (league, n) for league, n in rows if is_excluded_league(league)
        ]
        excluded_leagues.sort(key=lambda x: -x[1])

        total_rows = sum(n for _, n in excluded_leagues)
        total_leagues = len(excluded_leagues)
        print(f"Found {total_leagues} Tier-4 league codes "
              f"representing {total_rows} season rows in the DB.")
        print()
        print("Top 20 by row count:")
        for league, n in excluded_leagues[:20]:
            print(f"  {league:<30}{n:>6}")
        if len(excluded_leagues) > 20:
            print(f"  ... and {len(excluded_leagues) - 20} more.")
        print()

        if dry_run:
            print("[DRY-RUN] No changes made.")
            return

        # Use the intersection of excluded codes + present codes to avoid
        # building a massive "WHERE league IN (...)" of patterns.
        if not excluded_leagues:
            print("Nothing to delete.")
            return

        placeholders = ",".join("?" * len(excluded_leagues))
        leagues_only = [lg for lg, _ in excluded_leagues]
        cur.execute(
            f"DELETE FROM seasons WHERE league IN ({placeholders})",
            leagues_only,
        )
        deleted = cur.rowcount
        conn.commit()

    print(f"Deleted {deleted} season rows from Tier-4 leagues.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be deleted without making changes.")
    args = ap.parse_args()
    main(dry_run=args.dry_run)
