from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier

from .backtest import generate_strategy_returns
from .data_generator import generate_synthetic_lob
from .evaluation import (
    classification_metrics,
    trading_metrics,
)
from .features import add_microstructure_features
from .labeling import create_forward_labels
from .models import FEATURES, build_models


def run_experiment(
    n_samples: int = 20000,
    horizon: int = 10,
    threshold: float = 0.0001,
    seed: int = 42,
    output_dir: str = "results",
):

    output_path = Path(output_dir)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = generate_synthetic_lob(
        n=n_samples,
        seed=seed,
    )

    df = add_microstructure_features(df)

    df = create_forward_labels(
        df,
        horizon=horizon,
        threshold=threshold,
    )

    df = df.dropna(
        subset=FEATURES + ["target"]
    ).reset_index(drop=True)

    split_1 = int(
        len(df) * 0.70
    )

    split_2 = int(
        len(df) * 0.85
    )

    train = df.iloc[:split_1].copy()
    validation = df.iloc[split_1:split_2].copy()
    test = df.iloc[split_2:].copy()

    X_train = train[FEATURES]
    y_train = train["target"]

    X_test = test[FEATURES]
    y_test = test["target"]

    results = []

    baseline = DummyClassifier(
        strategy="most_frequent"
    )

    baseline.fit(
        X_train,
        y_train,
    )

    baseline_pred = baseline.predict(
        X_test
    )

    baseline_metrics = classification_metrics(
        y_test,
        baseline_pred,
    )

    baseline_metrics["model"] = (
        "Majority Baseline"
    )

    results.append(
        baseline_metrics
    )

    for bundle in build_models(seed):

        bundle.model.fit(
            X_train,
            y_train,
        )

        predictions = bundle.model.predict(
            X_test
        )

        probabilities = (
            bundle.model.predict_proba(
                X_test
            )
        )

        cls_metrics = classification_metrics(
            y_test,
            predictions,
        )

        strategy_returns = (
            generate_strategy_returns(
                future_returns=test[
                    "future_return"
                ],
                probabilities=probabilities,
                classes=bundle.model.classes_,
            )
        )

        trade_metrics = trading_metrics(
            strategy_returns
        )

        row = {
            "model": bundle.name,
            **cls_metrics,
            **trade_metrics,
        }

        results.append(row)

        if bundle.name == "Random Forest":

            importance = pd.Series(
                bundle.model.feature_importances_,
                index=FEATURES,
            ).sort_values(
                ascending=False
            )

            importance.to_csv(
                output_path
                / "feature_importance.csv"
            )

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        output_path
        / "model_results.csv",
        index=False,
    )

    metadata = {
        "samples": n_samples,
        "horizon": horizon,
        "threshold": threshold,
        "seed": seed,
        "train_size": len(train),
        "validation_size": len(validation),
        "test_size": len(test),
        "features": FEATURES,
    }

    with open(
        output_path / "experiment_config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    return results_df


def make_plots(
    results_path: str = "results",
):

    path = Path(results_path)

    results_file = (
        path / "model_results.csv"
    )

    if not results_file.exists():
        return

    results = pd.read_csv(
        results_file
    )

    if "f1_macro" in results:

        plt.figure(
            figsize=(8, 5)
        )

        plt.bar(
            results["model"],
            results["f1_macro"],
        )

        plt.ylabel(
            "Macro F1"
        )

        plt.title(
            "Short-Horizon Prediction Performance"
        )

        plt.xticks(
            rotation=20
        )

        plt.tight_layout()

        plt.savefig(
            path / "model_f1.png",
            dpi=150,
        )

        plt.close()

    if "sharpe" in results:

        plt.figure(
            figsize=(8, 5)
        )

        plt.bar(
            results["model"],
            results["sharpe"],
        )

        plt.ylabel(
            "Sharpe Ratio"
        )

        plt.title(
            "Strategy Sharpe Ratio"
        )

        plt.xticks(
            rotation=20
        )

        plt.tight_layout()

        plt.savefig(
            path / "model_sharpe.png",
            dpi=150,
        )

        plt.close()
