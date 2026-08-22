from __future__ import annotations

import numpy as np
import pandas as pd


def generate_strategy_returns(
    future_returns: pd.Series,
    probabilities: np.ndarray,
    classes: np.ndarray,
    confidence_threshold: float = 0.55,
    transaction_cost: float = 0.00005,
) -> pd.Series:

    class_to_index = {
        int(cls): i
        for i, cls in enumerate(classes)
    }

    up_index = class_to_index.get(1)
    down_index = class_to_index.get(-1)

    if up_index is None or down_index is None:
        raise ValueError(
            "Model must contain both UP and DOWN classes."
        )

    up_probability = probabilities[:, up_index]
    down_probability = probabilities[:, down_index]

    signal = np.zeros(
        len(probabilities)
    )

    signal[
        up_probability >= confidence_threshold
    ] = 1.0

    signal[
        down_probability >= confidence_threshold
    ] = -1.0

    raw_returns = (
        signal
        * future_returns.to_numpy()
    )

    position_change = np.abs(
        np.diff(
            np.concatenate(
                [[0.0], signal]
            )
        )
    )

    costs = (
        position_change
        * transaction_cost
    )

    strategy_returns = (
        raw_returns - costs
    )

    return pd.Series(
        strategy_returns,
        index=future_returns.index,
        name="strategy_return",
    )
