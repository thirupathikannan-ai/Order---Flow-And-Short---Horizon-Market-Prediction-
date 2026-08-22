from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_lob(
    n: int = 20000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic limit-order-book observations.

    The generator is designed for software validation and research
    pipeline testing, not for realistic market simulation.
    """

    rng = np.random.default_rng(seed)

    timestamps = pd.date_range(
        start="2026-01-01",
        periods=n,
        freq="s",
    )

    price = np.empty(n)
    price[0] = 100.0

    bid_size = rng.lognormal(mean=3.2, sigma=0.45, size=n)
    ask_size = rng.lognormal(mean=3.2, sigma=0.45, size=n)

    buy_volume = rng.lognormal(mean=2.5, sigma=0.65, size=n)
    sell_volume = rng.lognormal(mean=2.5, sigma=0.65, size=n)

    latent_pressure = np.zeros(n)

    for i in range(1, n):
        pressure = (
            (bid_size[i] - ask_size[i])
            / (bid_size[i] + ask_size[i])
        )

        trade_pressure = (
            (buy_volume[i] - sell_volume[i])
            / (buy_volume[i] + sell_volume[i])
        )

        latent_pressure[i] = (
            0.70 * pressure
            + 0.30 * trade_pressure
        )

        noise = rng.normal(0.0, 0.02)

        price[i] = (
            price[i - 1]
            * (1.0 + 0.0005 * latent_pressure[i] + noise * 0.001)
        )

    spread = 0.01 + 0.02 * rng.random(n)

    bid_price = price - spread / 2.0
    ask_price = price + spread / 2.0

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "bid_price": bid_price,
            "ask_price": ask_price,
            "bid_size": bid_size,
            "ask_size": ask_size,
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
        }
    )
