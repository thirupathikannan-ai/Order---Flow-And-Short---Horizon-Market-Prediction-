import pandas as pd

from src.features import (
    add_microstructure_features,
)


def test_feature_generation():

    df = pd.DataFrame(
        {
            "bid_price": [100.0, 100.1, 100.2],
            "ask_price": [100.2, 100.3, 100.4],
            "bid_size": [10.0, 12.0, 14.0],
            "ask_size": [12.0, 11.0, 10.0],
            "buy_volume": [5.0, 6.0, 8.0],
            "sell_volume": [5.0, 4.0, 3.0],
        }
    )

    result = add_microstructure_features(
        df,
        rolling_window=2,
    )

    assert "mid_price" in result
    assert "spread" in result
    assert "book_imbalance" in result
    assert "trade_pressure" in result
    assert "order_flow_imbalance" in result
