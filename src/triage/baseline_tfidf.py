"""Baseline ticket triage model using isolated sklearn Pipeline (TF-IDF + LightGBM)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline


class BaselineTriageModel:
    """Dual-pipeline TF-IDF + LightGBM baseline for Category and Urgency prediction.

    Strictly fits `TfidfVectorizer` and `LGBMClassifier` inside `sklearn.pipeline.Pipeline`
    exclusively on `train_tickets.csv` to guarantee zero preprocessing data leakage.
    """

    def __init__(
        self,
        categories: List[str],
        urgencies: List[str],
        seed: int = 42,
    ) -> None:
        self.categories = categories
        self.urgencies = urgencies
        self.seed = seed

        self.cat_to_idx = {c: i for i, c in enumerate(categories)}
        self.idx_to_cat = {i: c for i, c in enumerate(categories)}
        self.urg_to_idx = {u: i for i, u in enumerate(urgencies)}
        self.idx_to_urg = {i: u for i, u in enumerate(urgencies)}

        self.category_pipeline = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=90,
                        ngram_range=(1, 1),
                        min_df=4,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "clf",
                    LGBMClassifier(
                        n_estimators=30,
                        max_depth=3,
                        num_leaves=6,
                        learning_rate=0.08,
                        random_state=seed,
                        verbose=-1,
                    ),
                ),
            ]
        )

        self.urgency_pipeline = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=85,
                        ngram_range=(1, 1),
                        min_df=4,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "clf",
                    LGBMClassifier(
                        n_estimators=28,
                        max_depth=3,
                        num_leaves=6,
                        learning_rate=0.08,
                        random_state=seed,
                        verbose=-1,
                    ),
                ),
            ]
        )

    def fit(self, train_df: pd.DataFrame) -> "BaselineTriageModel":
        """Fit both TF-IDF + LightGBM pipelines strictly on the training split.

        Args:
            train_df: Training split DataFrame (`train_tickets.csv`).

        Returns:
            Self after fitting.
        """
        texts = train_df["sanitized_message"].fillna("").astype(str).tolist()
        y_cat = np.array([self.cat_to_idx[c] for c in train_df["category"]], dtype=np.int64)
        y_urg = np.array([self.urg_to_idx[u] for u in train_df["urgency"]], dtype=np.int64)

        self.category_pipeline.fit(texts, y_cat)
        self.urgency_pipeline.fit(texts, y_urg)
        return self

    def predict_proba(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Predict class probabilities for category and urgency.

        Args:
            texts: List of PII-sanitized ticket messages.

        Returns:
            Tuple of `(category_probs, urgency_probs)` arrays of shape `(N, 4)`.
        """
        cat_probs = self.category_pipeline.predict_proba(texts)
        urg_probs = self.urgency_pipeline.predict_proba(texts)
        return np.asarray(cat_probs, dtype=np.float64), np.asarray(urg_probs, dtype=np.float64)

    def evaluate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute offline evaluation metrics on a validation or test split.

        Args:
            df: Evaluation split DataFrame.

        Returns:
            Dictionary of Category Macro-F1, Urgency Macro-F1, Critical PR-AUC,
            and per-class precision/recall/F1.
        """
        texts = df["sanitized_message"].fillna("").astype(str).tolist()
        y_cat_true = np.array([self.cat_to_idx[c] for c in df["category"]], dtype=np.int64)
        y_urg_true = np.array([self.urg_to_idx[u] for u in df["urgency"]], dtype=np.int64)

        cat_probs, urg_probs = self.predict_proba(texts)
        cat_preds = np.argmax(cat_probs, axis=1)
        urg_preds = np.argmax(urg_probs, axis=1)

        cat_macro_f1 = float(f1_score(y_cat_true, cat_preds, average="macro"))
        urg_macro_f1 = float(f1_score(y_urg_true, urg_preds, average="macro"))

        crit_idx = self.urg_to_idx["Critical"]
        y_crit_bin = (y_urg_true == crit_idx).astype(np.int32)
        crit_pr_auc = float(average_precision_score(y_crit_bin, urg_probs[:, crit_idx]))

        prec, rec, f1_per_cls, _ = precision_recall_fscore_support(
            y_urg_true, urg_preds, labels=list(range(len(self.urgencies))), zero_division=0
        )

        return {
            "category_macro_f1": round(cat_macro_f1, 4),
            "urgency_macro_f1": round(urg_macro_f1, 4),
            "critical_pr_auc": round(crit_pr_auc, 4),
            "critical_precision": round(float(prec[crit_idx]), 4),
            "critical_recall": round(float(rec[crit_idx]), 4),
            "critical_f1": round(float(f1_per_cls[crit_idx]), 4),
            "cat_preds": cat_preds.tolist(),
            "urg_preds": urg_preds.tolist(),
            "cat_probs": cat_probs.tolist(),
            "urg_probs": urg_probs.tolist(),
        }

    def save(self, path: str | Path) -> None:
        """Persist trained pipeline to disk via joblib."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, out_path)

    @classmethod
    def load(cls, path: str | Path) -> "BaselineTriageModel":
        """Load persisted baseline pipeline from disk."""
        return joblib.load(Path(path))


def train_and_evaluate_baseline(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Train the TF-IDF + LightGBM baseline on `train_tickets.csv` and evaluate on `test_tickets.csv`."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_df = pd.read_csv(cfg["paths"]["train_tickets"])
    val_df = pd.read_csv(cfg["paths"]["val_tickets"])
    test_df = pd.read_csv(cfg["paths"]["test_tickets"])

    model = BaselineTriageModel(
        categories=cfg["triage"]["categories"],
        urgencies=cfg["triage"]["urgencies"],
        seed=int(cfg["project"]["seed"]),
    )
    model.fit(train_df)
    model.save(cfg["paths"]["baseline_model_path"])

    val_metrics = model.evaluate(val_df)
    test_metrics = model.evaluate(test_df)
    return {"val": val_metrics, "test": test_metrics}


if __name__ == "__main__":
    results = train_and_evaluate_baseline()
    print("=== BASELINE TF-IDF + LIGHTGBM METRICS (TEST SPLIT) ===")
    for k, v in results["test"].items():
        if not isinstance(v, list):
            print(f"- {k}: {v}")
