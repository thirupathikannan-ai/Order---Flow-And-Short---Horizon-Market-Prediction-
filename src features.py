from __future__ import annotations

import numpy as np
import pandas as pd


def add_microstructure_features(
    df: pd.DataFrame,
    rolling_window: int = 50,
) -> pd.DataFrame:

    data = df.copy()

    data["mid_price"] = (
        data["bid_price"] + data["ask_price"]
    ) / 2.0

    data["spread"] = (
        data["ask_price"] - data["bid_price"]
    )

    data["book_imbalance"] = (
        (data["bid_size"] - data["ask_size"])
        / (data["bid_size"] + data["ask_size"] + 1e-12)
    )

    data["trade_pressure"] = (
        (data["buy_volume"] - data["sell_volume"])
        / (
            data["buy_volume"]
            + data["sell_volume"]
            + 1e-12
        )
    )

    bid_change = data["bid_size"].diff().fillna(0.0)
    ask_change = data["ask_size"].diff().fillna(0.0)

    data["order_flow_imbalance"] = bid_change - ask_change

    data["depth"] = (
        data["bid_size"] + data["ask_size"]
    )

    returns = data["mid_price"].pct_change()

    data["short_term_return"] = returns

    data["rolling_volatility"] = (
        returns
        .rolling(rolling_window)
        .std()
        .fillna(0.0)
    )

    data["ofi_zscore"] = (
        (
            data["order_flow_imbalance"]
            - data["order_flow_imbalance"]
            .rolling(rolling_window)
            .mean()
        )
        /
        (
            data["order_flow_imbalance"]
            .rolling(rolling_window)
            .std()
            + 1e-12
        )
    )

    data["book_imbalance_zscore"] = (
        (
            data["book_imbalance"]
            - data["book_imbalance"]
            .rolling(rolling_window)
            .mean()
        )
        /
        (
            data["book_imbalance"]
            .rolling(rolling_window)
            .std()
            + 1e-12
        )
    )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return data
