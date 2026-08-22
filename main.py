from pathlib import Path

from src.experiment import (
    make_plots,
    run_experiment,
)


def main():

    print("=" * 60)
    print(
        "ORDER FLOW & SHORT-HORIZON MARKET PREDICTION"
    )
    print("=" * 60)

    results = run_experiment(
        n_samples=20000,
        horizon=10,
        threshold=0.0001,
        seed=42,
        output_dir="results",
    )

    make_plots(
        results_path="results"
    )

    print("\nClassification & Trading Results")
    print("-" * 60)

    print(
        results.to_string(
            index=False
        )
    )

    print("\nResults saved to:")
    print(
        Path("results").resolve()
    )


if __name__ == "__main__":
    main()
