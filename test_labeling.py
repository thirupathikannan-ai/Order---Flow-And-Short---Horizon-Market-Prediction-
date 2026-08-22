import pandas as pd

from src.labeling import (
    create_forward_labels,
)


def test_label_generation():

    df = pd.DataFrame(
        {
            "mid_price": [
                100.0,
                100.1,
                100.2,
                100.3,
                100.4,
            ]
        }
    )

    result = create_forward_labels(
        df,
        horizon=1,
        threshold=0.0001,
    )

    assert "future_return" in result
    assert "target" in result

    assert set(
        result["target"].unique()
    ).issubset(
        {-1, 0, 1}
    )
