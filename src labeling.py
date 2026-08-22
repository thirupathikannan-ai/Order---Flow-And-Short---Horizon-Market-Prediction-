from __future__ import annotations

import numpy as np
import pandas as pd


def create_forward_labels(
    df: pd.DataFrame,
    horizon: int = 10,
    threshold: float = 0.0001,
) -> pd.DataFrame:

    data = df.copy()

    future_mid = data["mid_price"].shift(-horizon)

    data["future_return"] = (
        future_mid - data["mid_price"]
    ) / data["mid_price"]

    data["target"] = np.select(
        [
            data["future_return"] > threshold,
            data["future_return"] < -threshold,
        ],
        [
            1,
            -1,
        ],
        default=0,
    )

    data = data.dropna(
        subset=["future_return"]
    ).reset_index(drop=True)

    return data
