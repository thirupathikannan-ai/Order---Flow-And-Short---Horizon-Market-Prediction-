from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "book_imbalance",
    "trade_pressure",
    "order_flow_imbalance",
    "spread",
    "depth",
    "short_term_return",
    "rolling_volatility",
    "ofi_zscore",
    "book_imbalance_zscore",
]


@dataclass
class ModelBundle:
    name: str
    model: object


def build_models(seed: int = 42) -> list[ModelBundle]:

    logistic = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=seed,
                ),
            ),
        ]
    )

    forest = RandomForestClassifier(
        n_estimators=250,
        max_depth=8,
        min_samples_leaf=10,
        random_state=seed,
        n_jobs=-1,
    )

    return [
        ModelBundle(
            name="Logistic Regression",
            model=logistic,
        ),
        ModelBundle(
            name="Random Forest",
            model=forest,
        ),
    ]
