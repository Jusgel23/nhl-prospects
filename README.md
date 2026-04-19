# NHL Prospect Prediction System

A Python pipeline that scrapes junior/college hockey stats, projects prospects
to the NHL, ranks them, finds historical comparables, and visualizes the
results in a Streamlit dashboard.

---

## What the code does

### 1. Data collection (`src/data/`)
- **`scrapers/eliteprospects.py`** — scrapes CHL (OHL / WHL / QMJHL) skater stats
  from EliteProspects for the configured draft classes.
- **`scrapers/uscho.py`** — scrapes NCAA men's hockey stats from USCHO.
- **`loaders/historical.py`** — pulls realized NHL career outcomes (1995–2019)
  from Hockey Reference to use as ground-truth labels.
- **`database.py`** — SQLite persistence layer. Four tables:
  `players`, `seasons`, `nhl_outcomes`, `predictions`.

### 2. Feature engineering (`src/models/`)
- **`nhle.py`** — converts raw league points-per-game to **NHL Equivalency (NHLe)**
  using league strength factors (OHL 0.30, NCAA 0.32, KHL 0.82, AHL 0.44, …)
  and age adjustments (younger = higher ceiling). Tags each season with a
  development label: `D-2`, `D-1`, `D0` (draft year), `D+1`, etc.
- **`features.py`** — pivots a player's multi-season history into one feature
  row: current/prior NHLe PPG, year-over-year improvement (`ppg_delta`),
  goals-per-game, age at draft, birth quarter, height/weight, games-played
  rate, PIM rate, power-play point share, league strength.

### 3. Model training (`src/models/predictor.py`)
- **Two XGBoost classifiers per position** (forwards and defensemen):
  - `nhler_probability` — likelihood of playing 200+ NHL games.
  - `star_probability` — likelihood of averaging 0.40+ points per NHL game.
- Also produces a regression for `projected_career_pts`.
- Walk-forward validation across 2005–2015 draft classes.

### 4. Comparables (`src/comparables/similarity.py`)
- Nearest-neighbor search over z-score normalized feature space (NHLe PPG
  weighted 2×). Returns each prospect's closest historical matches with
  their realized NHL outcomes (`Elite / NHLer / Did Not Reach`).
- Supports two modes: **full historical** comps and **same-draft-year** comps.

### 5. Rankings & reports (`src/rankings/`)
- **`ranker.py`** — blends model outputs into a single rank score:
  `nhler_prob × 0.50 + norm_nhle_ppg × 0.25 + ppg_delta × 0.15 + age_bonus × 0.10`.
- **`report.py`** — per-player markdown scouting report (bio, season table,
  NHLe trajectory, top 5 comps, risk flags).

### 6. CLI (`main.py`)
Orchestrates the whole pipeline via `argparse` subcommands: `collect`,
`process`, `train`, `rank`, `report`, `pipeline`.

### 7. Frontend (`app.py`)
Streamlit dashboard rendering side-by-side prospect cards modeled on the
Andy & Rono scouting-card layout:
- Header: name, flag, status (Superstar → Bust), position
- Bio block: GP / A / Pts / PPG / DOB / Age / H / W / Shoots / Draft Year / Round / Pick
- **NHLe bar chart** across `D-1`..`D+3` (points per 82-game pace)
- **Star & NHLer probability circles** across `D0`..`D3`
- **DY Comps** (same draft year) and **Full Comps** (all eras), color-coded by outcome
- Outcome legend at the bottom

Placeholders render when data is missing — the cards work even before the
full pipeline has been run.

---

## How to run

### Prerequisites
- Python 3.10+
- Git

### Install

```bash
git clone https://github.com/Jusgel23/nhl-prospects.git
cd nhl-prospects
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

### Populate the database (one-shot)

```bash
python main.py pipeline
```

This runs `collect → process → train → rank` end-to-end. The first run takes
~3 minutes because it scrapes historical NHL outcomes from Hockey Reference
(results are cached locally afterward).

### Or run steps individually

```bash
python main.py collect --seasons 2023-2024 2024-2025  # scrape raw stats
python main.py process                                 # compute NHLe + dev arcs
python main.py train                                   # fit XGBoost models
python main.py rank --draft-year 2025 --top 50         # generate rankings CSV
python main.py report --player "Connor Bedard"         # per-player markdown report
```

### Narrow to a single league (e.g. OHL) for validation

Use `--league` on `collect`, `rank`, or `pipeline` to restrict the run to one
league. Helpful for sanity-checking the full pipeline end-to-end before
expanding scrapes:

```bash
python main.py pipeline --league OHL --seasons 2023-2024 2024-2025
python main.py rank --league OHL --top 25
```

The Streamlit sidebar also has a **League** multi-select that filters the
player picker.

Outputs land in `data/processed/rankings.csv` and `data/reports/*.md`.

### Launch the Streamlit frontend

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser. Use the sidebar to pick up
to two prospects and compare their cards side-by-side. Filter by draft year
or position to narrow the list.

---

## Project layout

```
nhl-prospects/
├── app.py                    # Streamlit frontend
├── main.py                   # CLI entry point
├── requirements.txt
├── setup.py
├── data/                     # SQLite DB, raw scrapes, reports (gitignored)
├── notebooks/
└── src/
    ├── comparables/
    │   └── similarity.py     # Nearest-neighbor comp finder
    ├── data/
    │   ├── database.py       # SQLite schema + upserts
    │   ├── scrapers/         # EliteProspects, USCHO
    │   └── loaders/          # Hockey Reference historical outcomes
    ├── models/
    │   ├── nhle.py           # League + age equivalencies
    │   ├── features.py       # Feature engineering
    │   └── predictor.py      # XGBoost training + inference
    └── rankings/
        ├── ranker.py         # Combined rank score
        └── report.py         # Per-player markdown reports
```
