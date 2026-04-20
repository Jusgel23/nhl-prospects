# Changelog

A running narrative of what changed in this repo, newest entries at the top.
See `git log` for the full technical history.

## 2026-04-19 — (scrape-cache)
**What changed:** EliteProspects scraper now persists each league-season
stats table to `data/raw/league_stats/{league}_{season}.csv` and each
player bio to `data/raw/bios/{player_id}.json`. Both paths short-circuit
the HTTP fetch when a cache hit is found.
**Why:** The previous 40-minute scrape was lost entirely when the final
DB write crashed. With caching, a future crash only loses the current
league-season; re-runs replay almost instantly.
**Files:** `src/data/scrapers/eliteprospects.py`

## 2026-04-19 — (db-fix)
**What changed:** `_upsert_method` in `src/data/database.py` now accepts either
a `sqlite3.Cursor` (pandas 2.x) or `sqlite3.Connection` (older pandas) as the
`conn` argument passed to pandas' `to_sql` callback.
**Why:** Pipeline crashed with `AttributeError: 'sqlite3.Cursor' object has
no attribute 'cursor'` after a ~40 minute scrape, losing all in-memory data.
**Files:** `src/data/database.py`

## 2026-04-19 — 3fd9fda
**What changed:** Wrapped NHLer and Star XGBoost classifiers in
`CalibratedClassifierCV` (isotonic when ≥50 positives, Platt otherwise);
added `scale_pos_weight` to `train_walk_forward`; Brier score now logged
alongside AUC.
**Why:** Raw logistic outputs were distorted by `scale_pos_weight`, so the
dashboard's 99% / 1% probability circles were misleading. Calibration makes
the numbers reflect actual historical hit rates.
**Files:** `src/models/predictor.py`

## 2026-04-19 — aa8c5d5
**What changed:** Added `--league` flag to `collect`, `rank`, and `pipeline`
subcommands; matching League multi-select in the Streamlit sidebar.
**Why:** Sanity-check the full pipeline on one league (OHL) before expanding
scrapes to WHL, QMJHL, and NCAA.
**Files:** `main.py`, `app.py`, `README.md`

## 2026-04-19 — 53cf275
**What changed:** New Streamlit frontend (`app.py`) renders side-by-side
prospect cards with bio, NHLe development bars, Star/NHLer probability
progressions, and DY/Full comparables — with placeholders when data is
missing. Added `same_draft_year_only` filter to `similarity.find_comparables`.
Project-level `README.md` added.
**Why:** Give the prediction system a visual interface modeled on the
Andy & Rono prospect-card template, and back the new DY Comps view.
**Files:** `README.md`, `app.py`, `requirements.txt`, `src/comparables/similarity.py`, `.gitignore`

## 2026-04-19 — 6986d90
**What changed:** Replaced the manually-downloaded Kaggle CSV with an
auto-scraper for historical draft outcomes.
**Why:** Remove a manual prerequisite from the pipeline; every run now
bootstraps its own historical labels.
**Files:** scraper + loader modules in `src/data/`

## 2026-04-19 — 4891353
**What changed:** Initial commit of the NHL Prospect Prediction System —
scraping, NHLe normalization, XGBoost models (F/D), ranker, per-player
scouting reports, CLI.
**Why:** Project inception.
**Files:** full initial tree
