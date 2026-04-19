"""
XGBoost prediction model.
Trains separate classifiers for Forwards and Defensemen.
Outputs: NHLer probability, star probability, projected career points.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier, XGBRegressor

from src.models.features import FEATURE_COLS

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parents[2] / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

NHLER_THRESHOLD = 200   # NHL games played to qualify as "NHLer"
STAR_THRESHOLD  = 0.40  # pts/game rate in NHL


class ProspectPredictor:
    """
    Wraps two XGBoost models (Forward / Defense) for:
      - NHLer classification (binary)
      - Star classification (binary)
      - Career points regression
    """

    def __init__(self):
        self.models: dict[str, dict] = {
            "F": {"nhler": None, "star": None, "pts": None, "scaler": None},
            "D": {"nhler": None, "star": None, "pts": None, "scaler": None},
        }
        self._try_load()

    # ── Training ─────────────────────────────────────────────────────────────

    def train(self, features_df: pd.DataFrame):
        """
        Train on historical draft class data.
        features_df must include all FEATURE_COLS plus:
          is_forward, is_nhler, is_star, nhl_points, nhl_gp
        """
        for pos, label in [("F", "Forwards"), ("D", "Defensemen")]:
            pos_df = features_df[features_df["is_forward"] == (1 if pos == "F" else 0)].copy()
            if len(pos_df) < 50:
                logger.warning(f"Only {len(pos_df)} {label} in training set — skipping.")
                continue

            X = pos_df[FEATURE_COLS].values
            y_nhler = pos_df["is_nhler"].values
            y_star  = pos_df["is_star"].values
            y_pts   = pos_df.get("nhl_points", pd.Series(0, index=pos_df.index)).values

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            nhler_model = XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=_pos_weight(y_nhler),
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
            )
            nhler_model.fit(X_scaled, y_nhler)

            star_model = XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=_pos_weight(y_star),
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
            )
            star_model.fit(X_scaled, y_star)

            pts_model = XGBRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
            )
            pts_model.fit(X_scaled, y_pts)

            self.models[pos] = {
                "nhler":  nhler_model,
                "star":   star_model,
                "pts":    pts_model,
                "scaler": scaler,
            }

            # Cross-validation AUC
            cv_scores = cross_val_score(nhler_model, X_scaled, y_nhler,
                                        cv=5, scoring="roc_auc")
            logger.info(f"{label} NHLer model — CV AUC: {cv_scores.mean():.3f} "
                        f"(±{cv_scores.std():.3f})")

        self._save()
        logger.info("Models trained and saved.")

    def train_walk_forward(self, features_df: pd.DataFrame,
                           start_year: int = 2005, end_year: int = 2015):
        """
        Walk-forward validation: train on years [start, year-1], evaluate on year.
        Reports mean AUC across folds.
        """
        if "draft_year" not in features_df.columns:
            logger.error("features_df missing draft_year column.")
            return

        aucs = []
        for test_year in range(start_year, end_year + 1):
            train = features_df[features_df["draft_year"] < test_year]
            test  = features_df[features_df["draft_year"] == test_year]
            if train.empty or test.empty or "is_nhler" not in test.columns:
                continue

            model = XGBClassifier(n_estimators=200, max_depth=4,
                                   learning_rate=0.05, random_state=42,
                                   use_label_encoder=False, eval_metric="logloss")
            X_tr = train[FEATURE_COLS].fillna(0).values
            y_tr = train["is_nhler"].values
            X_te = test[FEATURE_COLS].fillna(0).values
            y_te = test["is_nhler"].values

            scaler = StandardScaler()
            model.fit(scaler.fit_transform(X_tr), y_tr)
            probs = model.predict_proba(scaler.transform(X_te))[:, 1]

            if len(np.unique(y_te)) > 1:
                auc = roc_auc_score(y_te, probs)
                aucs.append(auc)
                logger.info(f"  Year {test_year}: AUC={auc:.3f}  n={len(test)}")

        if aucs:
            logger.info(f"Walk-forward mean AUC: {np.mean(aucs):.3f}")

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run inference on a prospect DataFrame.
        Returns DataFrame with player_id + prediction columns.
        """
        results = []
        for pos, pos_label in [("F", 1), ("D", 0)]:
            sub = features_df[features_df["is_forward"] == pos_label].copy()
            if sub.empty:
                continue

            m = self.models[pos]
            if m["nhler"] is None:
                logger.warning(f"No trained model for {pos}. Run train() first.")
                sub["nhler_probability"] = np.nan
                sub["star_probability"]  = np.nan
                sub["projected_career_pts"] = np.nan
                results.append(sub[["player_id", "nhler_probability",
                                     "star_probability", "projected_career_pts"]])
                continue

            X = m["scaler"].transform(sub[FEATURE_COLS].fillna(0).values)
            sub["nhler_probability"]     = m["nhler"].predict_proba(X)[:, 1].round(3)
            sub["star_probability"]      = m["star"].predict_proba(X)[:, 1].round(3)
            sub["projected_career_pts"]  = np.maximum(
                0, m["pts"].predict(X).round(0).astype(int)
            )
            results.append(sub[["player_id", "nhler_probability",
                                  "star_probability", "projected_career_pts"]])

        if not results:
            return pd.DataFrame()
        return pd.concat(results, ignore_index=True)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self):
        for pos, m in self.models.items():
            if m["nhler"] is not None:
                joblib.dump(m, MODEL_DIR / f"model_{pos}.pkl")

    def _try_load(self):
        for pos in ["F", "D"]:
            path = MODEL_DIR / f"model_{pos}.pkl"
            if path.exists():
                self.models[pos] = joblib.load(path)
                logger.info(f"Loaded model for {pos} from {path}")


# ── helpers ───────────────────────────────────────────────────────────────────

def _pos_weight(y: np.ndarray) -> float:
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    return float(neg / pos) if pos > 0 else 1.0
