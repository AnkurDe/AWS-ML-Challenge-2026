"""Pairwise matching model training."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


class PairMatcher:
    """Thin wrapper around a numeric binary classifier."""

    def __init__(self) -> None:
        self.model = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=220,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42,
        )
        self.feature_columns: list[str] = []

    def fit(self, features: pd.DataFrame, labels: list[int]) -> "PairMatcher":
        if features.empty:
            raise ValueError("No training pairs were generated.")
        if len(set(labels)) < 2:
            raise ValueError("Training data must contain both positive and negative examples.")
        self.feature_columns = list(features.columns)
        self.model.fit(features[self.feature_columns], labels)
        return self

    def predict_proba(self, features: pd.DataFrame):
        if not self.feature_columns:
            raise RuntimeError("PairMatcher has not been fitted.")
        if features.empty:
            return []
        aligned = features.reindex(columns=self.feature_columns, fill_value=0.0)
        return self.model.predict_proba(aligned)[:, 1]

    def save(self, path: str | Path) -> None:
        joblib.dump(
            {"model": self.model, "feature_columns": self.feature_columns},
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "PairMatcher":
        payload = joblib.load(path)
        matcher = cls()
        matcher.model = payload["model"]
        matcher.feature_columns = payload["feature_columns"]
        return matcher
