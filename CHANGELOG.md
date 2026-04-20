# Changelog

A running narrative of what changed in this repo, newest entries at the top.
See `git log` for the full technical history.

## 2026-04-20 — (pipeline-schema-hardening)
**What changed:** Several follow-up fixes discovered while validating the
HR career integration end-to-end:
- `apply_nhle_to_seasons` drops derived columns (dob/age/nhle_factor/etc.)
  before merging and computes `ppg = points/gp` on-the-fly when missing
  (HR rows arrive without a pre-computed ppg).
- `build_development_arc` drops draft_year/draft_label before re-merging,
  preventing `_x`/`_y` suffix stacking on repeated process runs.
- `_pivot_seasons` now falls back to "best overall career season" for
  players who don't have a D+0 labeled row, so the feature matrix stops
  collapsing to just the HR-labeled subset (went from 1700+ rows to 19
  after the first HR ingest).
- `cmd_process` no longer uses `if_exists="replace"` (which nuked the
  UNIQUE constraint and let duplicates in). It DELETEs then upserts via
  the defensive method, preserving the CREATE TABLE schema.
- Extended `seasons` schema with `age`, `nhle_ppg`, `nhle_gpg`,
  `draft_year`, `draft_label` columns, with idempotent ALTER TABLE
  migration.
- HR scraper: shared `_hr_get` helper with 429 backoff (120s×attempts)
  and 4.5s default inter-request delay — robots.txt allows ~20 req/min
  but HR throttles at ~18 req/min in practice.

**Why:** The initial end-to-end validation surfaced five interlinked
bugs that only manifested with the richer HR data. None blocked the
architecture; they were stacking under repeated pipeline runs.

**Files:** `main.py`, `src/data/database.py`, `src/models/features.py`,
`src/models/nhle.py`, `src/data/scrapers/hockey_reference.py`

## 2026-04-20 — (hr-career-scraper)
**What changed:** Added player-centric career scraping from Hockey Reference.
New functions in `src/data/scrapers/hockey_reference.py`:
`scrape_player_career`, `scrape_all_careers`, `build_hr_id_mapping`. HR ID
mapping (synthetic → real `mcdavco01`-style IDs) cached to
`data/historical/hr_id_mapping.json`. Per-player careers cached to
`data/raw/hr_careers/{hr_id}.json`.

New CLI subcommand `python main.py backfill-careers [--start YEAR] [--end YEAR]
[--sample N] [--nhlers-only]` orchestrates HR ID mapping + career scraping
and upserts to DB.

DB schema: added `draft_year`, `draft_round`, `draft_pick`, `draft_team`
columns to `players` via idempotent PRAGMA migration in `init_db`.

Feature engineering:
- `_pivot_seasons` now extracts D-3 and D+1 NHLe in addition to D-1/D-2.
- `FEATURE_COLS` gains `nhle_ppg_d_minus3`, `nhle_ppg_d_plus1`,
  `peak_nhle_ppg`, `league_count` (18 cols total, up from 14).
- `_age_at_draft` drops `datetime.now()` fallback; infers draft year as
  (birth_year + 18) when missing.
- `build_development_arc` in `src/models/nhle.py` now infers draft year
  from DOB when `outcomes_df` doesn't have it, so current undrafted
  prospects get D-relative labels.

**Why:** Model was underfitting with only 371 labeled samples and Parekh
was ranked #48/94 D-men despite elite production. Root causes: no DOB
meant age adjustments defaulted; D-3/D+1 trajectory features were absent;
bio scraper was broken. HR has everything we need in a server-rendered
HTML page that's trivial to scrape, and it's already the source of our
draft outcomes.

**Files:** `src/data/scrapers/hockey_reference.py`, `src/data/database.py`,
`src/models/features.py`, `src/models/nhle.py`, `main.py`

## 2026-04-20 — (hr-position-backfill)
**What changed:** `build_feature_matrix` now backfills missing player
positions from the Hockey Reference draft outcomes CSV (matched by
normalized name). HR position values (C/LW/RW/D/G) are normalized to
the F/D/G buckets the model expects.
**Why:** Older EP stats pages (2016-2019) don't carry the "(C/LW)"
suffix the scraper relies on, so ~1,083 of 1,707 players had
position=None. `_engineer` then `fillna("F")`'d all of them,
mis-classifying historical defensemen as forwards and leaving the D
training set empty. Backfill lifts labeled D count from 0 to 88.
**Files:** `src/models/features.py`

## 2026-04-20 — (extract-position)
**What changed:** `_clean_stats_df` now extracts position from the EP
name suffix ("David Goyette(C/LW)" → position="F") before stripping
the suffix, and `scrape_draft_class` uses that as a fallback when the
bio scrape doesn't provide one.
**Why:** Every scraped player had `position=None` because the bio
scraper can't match EP's redesigned page layout. `_engineer` then
`fillna("F")`'d all of them, classifying every prospect as a forward,
so the Defenseman model had zero labeled rows and only `model_F.pkl`
was produced. Predictions for D-men came out as NaN; comparables index
was skewed.
**Files:** `src/data/scrapers/eliteprospects.py`

## 2026-04-20 — (remap-outcomes)
**What changed:** `build_feature_matrix` now remaps Hockey Reference
historical outcomes onto EliteProspects scraped players via normalized
name before merging. New helper `_remap_outcomes_to_players` loads names
from the HR CSV cache and joins on a lowercase alpha-only version of the
name.
**Why:** Previously the outcomes player_id (HR format like
`bryan_berard_1995`) never matched the scraped player_id (EP numeric),
so every merge produced zero labeled training rows. The pipeline
"trained" on zero samples, leaving `data/models/` empty and every
prediction as None. Now 365 historical prospects (103 NHLers, 51 stars)
are correctly labeled.
**Files:** `src/models/features.py`

## 2026-04-19 — (gp_rate-guard)
**What changed:** `_engineer` in `src/models/features.py` now checks
whether `gp_rate` is present in the DataFrame before calling
`pd.to_numeric`, defaulting to 1.0 if absent.
**Why:** `_pivot_seasons` aggregation doesn't include `gp_rate`, so the
pivoted DataFrame lacks that column. `df.get("gp_rate")` returned None,
and `pd.to_numeric(None).fillna()` crashed. Broke both `train` and the
Streamlit comparables-index build.
**Files:** `src/models/features.py`

## 2026-04-19 — (defensive-upsert)
**What changed:** `_upsert_method` now queries the target table's schema
via `PRAGMA table_info` and filters the insert to only the columns that
actually exist. Extra DataFrame columns are silently dropped instead of
triggering a SQLite "no such column" error.
**Why:** Third consecutive schema-mismatch crash — this time `upsert_outcomes`
failed because the historical-outcomes cache carries `name` and `position`
columns that `nhl_outcomes` doesn't have. Defensive filtering kills the
entire class of bug going forward.
**Files:** `src/data/database.py`

## 2026-04-19 — (seasons-schema-fix)
**What changed:** Removed the `name` column from the seasons DataFrame
projection in `scrape_draft_class`. Also strip position suffixes like
"(C/LW)" from player names in `_clean_stats_df` so both fresh scrapes
and cache-loaded rows produce clean names.
**Why:** `upsert_seasons` crashed after 155 players were stored because
the DataFrame included a `name` column that doesn't exist in the
`seasons` table schema. The suffix cleanup is a related data-quality
fix for names extracted from EP's stats tables.
**Files:** `src/data/scrapers/eliteprospects.py`

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
