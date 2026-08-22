from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def classification_metrics(
    y_true,
    y_pred,
) -> dict:

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),
        "precision_macro": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
    }


def trading_metrics(
    returns: pd.Series,
) -> dict:

    returns = pd.Series(
        returns
    ).fillna(0.0)

    equity = (1.0 + returns).cumprod()

    cumulative_return = (
        equity.iloc[-1] - 1.0
    )

    volatility = returns.std()

    if volatility > 0:
        sharpe = (
            returns.mean()
            / volatility
            * np.sqrt(252)
        )
    else:
        sharpe = 0.0

    running_max = equity.cummax()

    drawdown = (
        equity / running_max
    ) - 1.0

    max_drawdown = drawdown.min()

    non_zero = returns[
        returns != 0
    ]

    win_rate = (
        (non_zero > 0).mean()
        if len(non_zero) > 0
        else 0.0
    )

    return {
        "cumulative_return": cumulative_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "trades": int(len(non_zero)),
    }
