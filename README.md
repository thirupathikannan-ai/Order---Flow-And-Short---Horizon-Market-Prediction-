#Order Flow & Short-Horizon Market Prediction

A market-microstructure research framework for studying whether order-flow information can predict short-horizon movements in the mid-price.

The project investigates the relationship between:

- Bid/ask order-flow imbalance
- Order-book imbalance
- Trade pressure
- Market depth
- Spread
- Short-horizon returns
- Predictive classification
- Confidence-aware trading signals

The objective is not to build a production trading system. It is to create a reproducible research pipeline for testing whether information contained in the limit order book and order flow has statistically significant predictive power over short horizons.

---

Research Question

«Can order-flow and limit-order-book information provide useful predictive signals for short-horizon market movements?»

The project evaluates this question using engineered market-microstructure features and compares several models against simple baselines.

The research is motivated by market-microstructure literature showing that order-flow imbalance is related to short-interval price changes and by research demonstrating that stationary order-flow representations can be effective for high-frequency return prediction.

---

Research Hypothesis

The main hypothesis is:

«Strong positive order-flow imbalance should increase the probability of an upward short-horizon price movement, while strong negative order-flow imbalance should increase the probability of a downward movement.»

A secondary hypothesis is:

«Combining order-flow imbalance with spread, depth and trade-pressure information should provide more useful predictions than using a single imbalance variable.»

---

Research Pipeline

Market / LOB Data
       │
       ▼
Data Cleaning
       │
       ▼
Microstructure Feature Engineering
       │
       ├── Order Flow Imbalance
       ├── Book Imbalance
       ├── Trade Pressure
       ├── Spread
       ├── Depth
       └── Volatility
       │
       ▼
Forward Mid-Price
       │
       ▼
Short-Horizon Labels
       │
       ├── DOWN
       ├── FLAT
       └── UP
       │
       ▼
Train / Validation / Test
       │
       ▼
Baseline Models
       │
       ├── Majority Class
       ├── Logistic Regression
       └── Random Forest
       │
       ▼
Prediction
       │
       ▼
Signal Generation
       │
       ▼
Transaction-Cost-Aware Backtest
       │
       ▼
Research Metrics

---

Features

1. Order Flow Imbalance

A simplified order-flow imbalance measure is calculated from changes in bid and ask liquidity.

The feature attempts to measure whether liquidity pressure is predominantly on the buying or selling side.

Positive values indicate stronger buying-side pressure.

Negative values indicate stronger selling-side pressure.

---

2. Order Book Imbalance

The top-of-book imbalance is defined as:

Bid Size - Ask Size
-------------------
Bid Size + Ask Size

Values close to:

+1 → strong bid-side dominance
 0 → balanced book
-1 → strong ask-side dominance

---

3. Trade Pressure

Trade pressure approximates whether recent executed volume is dominated by aggressive buying or selling.

Buy Volume - Sell Volume
------------------------
Buy Volume + Sell Volume

---

4. Bid-Ask Spread

Ask Price - Bid Price

The spread provides information about liquidity and execution cost.

---

5. Market Depth

The model incorporates available liquidity near the best bid and ask.

---

6. Short-Term Volatility

Rolling volatility is included to distinguish calm market states from highly active periods.

---

Target Definition

The target is based on the future mid-price.

mid_price = (best_bid + best_ask) / 2

For a prediction horizon "H":

future_return =
    (mid_price[t + H] - mid_price[t])
    / mid_price[t]

The return is converted into three classes:

DOWN
FLAT
UP

using a configurable threshold.

This prevents extremely small price movements from being treated as meaningful directional predictions.

---

Models

The framework intentionally begins with interpretable models.

Baseline 1 — Majority Class

Predicts the most common class.

This establishes the minimum performance level.

Baseline 2 — Logistic Regression

Provides an interpretable linear relationship between microstructure features and short-horizon direction.

Baseline 3 — Random Forest

Captures nonlinear relationships between order-flow variables.

---

Why Start With Simple Models?

The goal is not to maximize model complexity.

A useful quantitative research process should first establish:

1. Whether the signal exists.
2. Whether the signal survives out-of-sample testing.
3. Whether the signal is economically meaningful.
4. Whether transaction costs eliminate the apparent edge.
5. Whether additional model complexity improves the result.

Only after establishing a robust baseline should more complex models such as neural networks or Transformers be considered.

---

Experimental Design

The dataset is divided chronologically.

Historical Data
      │
      ├───────────────┐
      │               │
   Training       Validation
      │               │
      └───────┬───────┘
              │
           Test Set

Random shuffling is avoided for the final time-series evaluation because future information must not leak into the training set.

---

Evaluation Metrics

The project reports:

Classification

- Accuracy
- Balanced Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Trading

- Number of trades
- Cumulative return
- Mean strategy return
- Volatility
- Sharpe ratio
- Maximum drawdown
- Win rate
- Turnover

Research Diagnostics

- Feature importance
- Prediction confidence
- Signal distribution
- Performance by market regime

---

Trading Signal

The classifier produces probabilities:

P(DOWN)
P(FLAT)
P(UP)

A simple confidence-aware strategy is used.

If P(UP) > threshold:
    LONG

If P(DOWN) > threshold:
    SHORT

Otherwise:
    FLAT

Transaction costs are included in the backtest.

This is important because a predictive model can have statistical accuracy while remaining economically unprofitable after trading costs.

---

Leakage Prevention

This project explicitly considers several sources of look-ahead bias.

Chronological splitting

Training data always precedes test data.

Forward labels

Future prices are used only for target construction, never as model features.

Rolling statistics

Rolling features use historical information only.

No random train/test mixing

Time-series observations are not randomly shuffled for the final evaluation.

---

Synthetic Data Mode

The repository includes a synthetic market-data generator so that the complete research pipeline can be executed without requiring proprietary market data.

The generator creates:

- Bid prices
- Ask prices
- Bid sizes
- Ask sizes
- Buy volume
- Sell volume
- Mid-price
- Timestamp

The synthetic data is intended for software validation and demonstration.

It should not be interpreted as real-market performance evidence.

---

Running the Project

Clone the repository:

git clone https://github.com/YOUR_USERNAME/order-flow-short-horizon-market-prediction.git
cd order-flow-short-horizon-market-prediction

Create a virtual environment:

python -m venv .venv

Activate it on Linux/macOS:

source .venv/bin/activate

Windows:

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Run the research experiment:

python main.py

---

Expected Output

The experiment prints a research summary similar to:

============================================================
ORDER FLOW & SHORT-HORIZON MARKET PREDICTION
============================================================

Dataset
-------
Observations: ...
Features: ...

Prediction Horizon
------------------
Horizon: ...

Classification Results
----------------------
Model                Accuracy     F1
Majority Baseline    ...          ...
Logistic Regression  ...          ...
Random Forest        ...          ...

Trading Results
---------------
Strategy Return:     ...
Sharpe Ratio:        ...
Maximum Drawdown:    ...
Win Rate:            ...
Number of Trades:    ...

Feature Importance
------------------
OFI                  ...
Book Imbalance       ...
Trade Pressure       ...
Spread               ...
Depth                ...
Volatility           ...

============================================================

The exact values depend on the generated dataset and experiment configuration.

---

Example Research Interpretation

A successful experiment does not simply mean:

«"The model achieved high accuracy."»

The more important questions are:

- Does the model outperform a naive baseline?
- Does the signal remain useful out-of-sample?
- Which features drive the predictions?
- Does predictive power decay as the horizon increases?
- Does the signal survive transaction costs?
- Is the strategy stable across different market regimes?

These questions make the project a market-microstructure research study rather than a generic machine-learning classification exercise.

---

Future Research

Possible extensions include:

Multi-Level OFI

Extend the feature set from the best bid/ask to multiple levels of the limit order book.

DeepLOB

Implement a CNN + Inception + LSTM architecture for raw LOB prediction.

Transformer Model

Compare sequential attention-based models with traditional models.

Hawkes Processes

Model the arrival dynamics of market, limit and cancellation events.

Cross-Asset Order Flow

Study whether lagged order-flow information from related assets improves prediction.

Probability Calibration

Use calibrated probabilities instead of raw classifier outputs.

Regime Detection

Evaluate model performance separately during:

- High volatility
- Low volatility
- Wide spread
- Tight spread
- High volume
- Low volume

---

Research Limitations

This project has several important limitations.

1. Synthetic data does not represent real exchange microstructure perfectly.
2. Backtest results are not evidence of live profitability.
3. Transaction-cost assumptions are simplified.
4. Real limit-order-book data contains additional event types and complexities.
5. Market impact and queue position are not fully modeled.
6. Latency is not explicitly simulated.
7. The model is not designed for live trading.

Therefore, results should be interpreted as research evidence, not investment advice.

---

References

Cont, Kukanov & Stoikov

The Price Impact of Order Book Events.

The study investigates the relationship between order-flow imbalance and short-interval price changes.

Zhang, Zohren & Roberts

DeepLOB: Deep Convolutional Neural Networks for Limit Order Books.

A deep-learning architecture for predicting short-term price movements from limit-order-book data.

Kolm, Turiel & Westray

Deep Order Flow Imbalance: Extracting Alpha at Multiple Horizons from the Limit Order Book.

Research investigating high-frequency return prediction using stationary order-flow representations.

---

Project Objective

This repository demonstrates a complete quantitative research workflow:

Market Microstructure
        +
Statistical Feature Engineering
        +
Machine Learning
        +
Time-Series Evaluation
        +
Transaction-Cost-Aware Backtesting
        =
Short-Horizon Market Prediction Research

The primary objective is to understand whether order flow contains predictive information about short-horizon price movements, and under what conditions that information remains useful.

---

Author

Thirupathi Kannan K

B.E. Electronics & Communication Engineering

Research interests:

- Quantitative Trading
- Market Microstructure
- Algorithmic Trading
- Statistical Learning
- High-Frequency Data
- Financial Machine Learning

---

Disclaimer

This project is for educational and research purposes only.

It does not constitute financial advice and does not provide investment recommendations.
