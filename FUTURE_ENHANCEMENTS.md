# Future Enhancements

A prioritized backlog of improvements to the NHL Prospect Prediction System.
Items are grouped by effort and tagged by theme: **[data]**, **[model]**,
**[ui]**, **[infra]**.

---

## Planned probability-calibration tuning pass **[model]**

**Problem:** The current board's top-5 prospects all hit 100% NHLer and
70-99% Star probability. That's too generous — historically only ~30-40%
of #1 overall picks become "stars" by the 200-GP / 0.40-PPG definition,
so the top of the distribution is compressed at the ceiling. The
predictions *rank* correctly (Celebrini #1, Parekh high among D, etc.)
but the absolute magnitudes read as overconfident.

**Root causes** (from `bce1baa`, `6066386`, and training on ~330 labeled
rows):
1. Thin training set pushes the model toward extremes.
2. `scale_pos_weight = neg/pos` (≈5×) aggressively boosts positive-class
   outputs; calibration only partially undoes it on small samples.
3. Monotonic `+1` constraints on every NHLe-variant feature cause elite
   prospects to pin at prob = 1.0 with no headroom.

**When to tune:** After the full HR career backfill finishes and the
labeled training set grows from ~330 to the ~2,000-5,000 range. More
data naturally regularizes the probability spread; tuning now would
just produce different noise.

**Tuning levers, in order of impact:**

| Lever | File | Effect | Effort |
|---|---|---|---|
| `scale_pos_weight = sqrt(neg/pos)` instead of `neg/pos` | `src/models/predictor.py:207` (the `_pos_weight` helper) | Less aggressive positive-class push → more realistic probabilities | 1 line |
| Remove monotonic constraint on `peak_nhle_ppg` (keep on `nhle_ppg` only) | `src/models/predictor.py:27` (the `_MONOTONE_CONSTRAINTS_BY_COL` dict) | Stops prospects with one freak season from pinning at 100% | 1 line |
| Widen isotonic calibration; `cv=10` on nhler_prob | `src/models/predictor.py:234` (the `_fit_calibrated_clf` helper) | Probabilities hew closer to historical base rates | 5 min |
| Add Bayesian shrinkage toward class base rate | `src/models/predictor.py` — new wrapper around `predict_proba` | Shrinks predictions toward 21% NHLer / 5% star when data is thin | ~30 LOC |
| Display "percentile within draft class" instead of raw prob on the board | `app.py` — new column in `build_board()` | "Top 1%" reads more honestly than "99%" for scouting | ~10 LOC |

**Recommended sequence:**
1. Finish HR backfill + retrain (no code changes) → observe new spread
2. If still compressed: apply levers 1 + 2 together (5 min)
3. If probabilities still feel wrong: add lever 5 (percentile view) —
   scouts intuit percentile rankings better than absolute probability
4. Defer levers 3 + 4 unless a backtest page reveals systemic miscalibration

**Verification:**
- Historical backtest page (separate FE item) — replay 2015 draft class,
  check predicted top-10 against known NHL outcomes. If calibration is
  working, Connor McDavid should sit near the observed ~70% star-prob
  for #1 picks historically, not 99%.
- Calibration plot: predicted probability buckets (0-10%, 10-20%, …)
  should match observed hit rate within each bucket.

---

## Quick wins (1–2 hrs each)

### Add playoff statistics **[data] [model]**
Regular-season PPG only tells half the story. Playoff performance under
pressure against top opponents is one of the strongest projection signals.
EliteProspects exposes playoff stats alongside regular season — minor scraper
extension, one new feature column (`playoff_nhle_ppg`) in the feature matrix.

### Add shot rates (SOG, shooting %) **[data] [model]**
Shots per 60 is more stable and predictive than goals for junior-age samples
(<100 GP per season). Expected lift: ~3–5% AUC, especially for forwards
drafted outside the first round.

### SHAP feature importance per player **[ui]**
Surface `shap.TreeExplainer` output on each player card — shows *why* the
model ranks a prospect where it does (e.g., "+12% from NHLe D-1, -5% from
age-at-draft"). Builds trust in the predictions without any retraining.
Adds `shap` dependency.

### Historical backtest dashboard page **[ui]**
A new Streamlit page that replays the model against, say, the 2015 draft class
and shows its predicted rankings alongside realized outcomes. Critical for
earning scout trust — without this, the model's numbers are unfalsifiable.

### Data-sufficiency confidence tag **[model] [ui]**
For each prospect, count how many historical comparables sit within a small
feature-space radius. "Based on 42 similar prospects" (high confidence) vs.
"Based on 3 similar prospects — unusual profile" (low confidence). Free to
add since the comparables index already exists.

---

## Medium (half-day each)

### Bootstrap/bagging confidence intervals **[model] [ui]**
Train 30–50 XGBoost models on bootstrap resamples; the spread of their
predictions is the uncertainty band. Show as a shaded range under each
probability circle on the player card. Honest uncertainty > spurious
precision. Combines well with the data-sufficiency tag above.

### Template-based player summaries **[ui]**
Auto-generated 2–3 sentence narrative per prospect, blending role
projection + key strength + risk flag + top comparable. Lives alongside
the existing `src/rankings/report.py` markdown generator. Rule-based to
start (zero API cost, deterministic). Example output:

> "Elite offensive profile (OHL, C). NHLe PPG (1.82) ranks in the 98th
> percentile of historical CHL forwards at age 17. Top comparables include
> Mitch Marner (2015) and Steven Stamkos (2008). Model projects 72%
> NHLer, 45% star. Slight concern around gp_rate (81%) — monitor for injury."

### International / U20 tournament stats **[data] [model]**
U18 Worlds, World Juniors, Hlinka. Performance against best peers is a
stronger signal than league-level dominance. Scrapable from EliteProspects.

### Scouting consensus rank **[data] [model]**
Central Scouting + TSN + McKeen's + Future Considerations → aggregate
consensus rank as a new feature. Huge signal; easy to scrape annually.
Functions as a "wisdom of the crowd" prior the model can defer to.

### Per-development-stage models **[model]**
Currently the dashboard's D0→D3 probability circles are mostly placeholders —
only the prospect's current stage has a real model output. Fix by training
separate classifiers on D-1, D0, D+1, D+2 snapshots. Unlocks the full visual
of the Andy & Rono template and matches how scouts think about risk
flattening over time.

### Optuna hyperparameter tuning **[model]**
Systematic search over `max_depth`, `learning_rate`, `n_estimators`,
`subsample`, `colsample_bytree`. Usually a 3–8% AUC lift on top of hand-tuned
defaults. Add as a `tune` subcommand in the CLI.

---

## Bigger (1+ day)

### Multi-class outcome target **[model]**
Replace binary `is_nhler` / `is_star` with a four-class target that better
matches scout mental models:

- **Superstar** (PPG ≥ 0.85, 500+ GP)
- **Top-6** (PPG ≥ 0.55, 400+ GP)
- **Bottom-6 / Depth** (200+ GP)
- **Did not reach NHL**

Allows a softmax distribution over tiers instead of two near-independent
probabilities. More interpretable on the card, more honest statistically.

### Goalie pipeline **[data] [model]**
Goalies are currently excluded entirely (features are skater-specific). A
parallel pipeline — scraping SV%, GAA, starts; training on a separate
outcome target (100+ NHL starts / starter-tier career) — is roughly 300
lines of new code. Worth it if draft classes with notable goalies are
being analyzed.

### Hybrid LLM-polished summaries **[ui]**
Pass the rule-based summary through Claude/GPT for prose polish. Grounding
text in the structured report prevents hallucinations. Cost: ~$0.01–0.05
per prospect × top-50 ≈ under $3 per refresh. Upgrade path from the
template-based summary, not a rewrite.

### Draft pick value distribution **[ui]**
For each draft slot (1–224), show the expected distribution of model-projected
career-points across history. Helps a front office understand "at pick 12,
how much surplus value is this model saying we'd get by taking Player X vs
the historical pick-12 median?"

### Weekly auto-refresh cron **[infra]**
Scheduled task (GitHub Actions or local Windows Task Scheduler) that runs
`collect → process → rank` nightly during the season. Writes predictions
to the DB with a timestamp so the dashboard always shows current-season
performance, not last-November's snapshot.

### Conformal prediction for prediction intervals **[model]**
Distribution-free, theoretically calibrated prediction intervals. More
principled than bagging but implementation is non-trivial. Consider only
if the bagging approach proves unsatisfactory in backtesting.

---

## Infrastructure & quality

### Unit tests **[infra]**
No tests currently. Start with `src/models/nhle.py` (pure functions, easy
to test) and `src/comparables/similarity.py`. pytest + ~200 LOC gets a
solid baseline.

### Schema migrations **[infra]**
No migration framework — schema changes today require dropping `prospects.db`.
Add Alembic or a simple numbered-migration pattern. Low priority until the
schema actually needs to change.

### Prediction drift monitoring **[infra]**
Log every `predictions` row with a timestamp; compare week-over-week to flag
prospects whose projection has shifted materially. Useful during the season
as performance data accumulates.

### Incremental collect (diff vs DB) **[infra]**
`python main.py collect` currently re-scrapes every run. A diff mode that
only fetches new/changed rows would be faster and kinder to upstream sites.

---

## Recommended next sprint

If picking a small set with compounding value:

1. **Playoff stats + shot rates** — biggest feature wins per hour.
2. **Bootstrap confidence intervals + data-sufficiency tag** — makes the
   existing dashboard honest about what it does and doesn't know.
3. **Template-based player summaries** — closes the gap between "a
   dashboard of numbers" and "something a scout wants to open every morning."
4. **Historical backtest page** — without this, nothing else matters,
   because there's no way to know if the model is any good.

One developer day.
