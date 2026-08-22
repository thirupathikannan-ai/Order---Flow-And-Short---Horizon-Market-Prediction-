# Order Flow & Short-Horizon Market Prediction

A quantitative research framework for studying whether **order-flow information contains predictive signals for short-horizon price movements** using order-flow imbalance, lagged microstructure features, statistical classification, multi-horizon prediction, out-of-sample evaluation, and baseline comparison.

---

## Overview

Short-horizon price movements are influenced by the interaction between market participants, liquidity, executed trades, and buying and selling pressure.

Order-flow information provides a way to quantify this short-term market pressure.

This project develops a research-oriented prediction framework that studies whether order-flow information can be used to predict the direction of future price movement over short horizons.

The system focuses on:

- Order-flow imbalance
- Bid and ask pressure
- Lagged order-flow signals
- Short-term returns
- Feature engineering
- Binary price-direction prediction
- Multiple prediction horizons
- Logistic regression
- Out-of-sample evaluation
- Baseline comparison
- ROC-AUC analysis
- Log-loss evaluation
- Reproducible experiments

The objective is not to assume that order flow predicts prices.

Instead, the project experimentally evaluates:

```text
Order Flow
     +
Short-Term Market Dynamics
     +
Statistical Modeling
     ↓
Short-Horizon Price Prediction
     ↓
Out-of-Sample Evaluation
     ↓
Research Conclusion
Research Question
Can order-flow information provide statistically useful predictive information about short-horizon future price direction?
The project evaluates this question using a controlled and reproducible experimental framework.
Research Objectives
Objective 1 — Model Order Flow
Construct a synthetic order-flow process representing short-term buying and selling pressure.
Objective 2 — Calculate Order-Flow Imbalance
Transform bid-side and ask-side activity into a normalized order-flow signal.
Objective 3 — Engineer Microstructure Features
Create lagged order-flow and short-term return features to capture temporal dependencies.
Objective 4 — Predict Future Price Direction
Train a statistical classification model to predict whether the future price will move upward or downward.
Objective 5 — Compare Prediction Horizons
Evaluate prediction performance at multiple short horizons.
Objective 6 — Perform Out-of-Sample Evaluation
Separate training and testing data to measure predictive performance on unseen observations.
Objective 7 — Compare Against a Baseline
Evaluate whether the model performs better than a simple majority-class prediction strategy.
System Architecture
                    ┌──────────────────────┐
                    │   Synthetic Market   │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │ Market Price         │
                    │ Bid / Ask Activity   │
                    │ Buy / Sell Pressure  │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │ Order-Flow           │
                    │ Imbalance Calculation│
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │ Feature Engineering  │
                    │                      │
                    │ Lagged OFI            │
                    │ Short-Term Returns    │
                    │ Rolling Statistics    │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │ Prediction Target    │
                    │                      │
                    │ Future Price > Price │
                    │       ↓              │
                    │     UP / DOWN        │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │ Logistic Regression  │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │ Out-of-Sample        │
                    │ Prediction            │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │ Performance Metrics  │
                    │                      │
                    │ Accuracy             │
                    │ ROC-AUC              │
                    │ Log Loss             │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │ Multi-Horizon        │
                    │ Research Evaluation  │


               └──────────────────────┘


Mathematical Model
1. Order-Flow Imbalance
Order-flow imbalance measures the relative difference between buying and selling pressure.
Let:
Bid Quantity = B
Ask Quantity = A
The normalized imbalance is:
OFI =
(B - A)
/
(B + A)
The resulting signal is bounded between:
-1 ≤ OFI ≤ +1
Interpretation:
OFI > 0
    ↓
Greater buying pressure

OFI < 0
    ↓
Greater selling pressure

OFI ≈ 0
    ↓
Balanced order flow


Lagged Order-Flow Features
Market prediction cannot use future information.
Therefore, the model uses historical order-flow observations:
OFI(t-1)
OFI(t-2)
OFI(t-3)
...
These lagged signals allow the model to identify short-term temporal relationships.
Example:
Current Event
     │
     ├── OFI(t-1)
     ├── OFI(t-2)
     ├── OFI(t-3)
     ├── Return(t-1)
     └── Return(t-2)
3. Short-Term Returns
The system calculates short-term returns:
Return(t) =
Price(t) / Price(t-1) - 1
Lagged returns are included as additional predictive variables.
These features allow the model to distinguish between:
Persistent Short-Term Momentum
and:
Short-Term Mean Reversion
4. Prediction Target
The target is future price direction.
For a prediction horizon h:
Future Return =
Price(t+h) / Price(t) - 1
The classification target is:
Target = 1
if Future Return > 0

Target = 0
if Future Return ≤ 0
The project evaluates multiple horizons:
1 Tick
5 Ticks
10 Ticks
5. Logistic Regression
The primary prediction model is logistic regression.
The probability of an upward price movement is modeled as:
P(Y = 1 | X) =
1 /
(1 + exp(-(β₀ + βX)))
where:
Y = Future Price Direction

X = Order-Flow Features

β₀ = Intercept

β = Model Coefficients
The model therefore converts order-flow features into a probability of future upward movement.
6. Prediction Rule
The model produces:
P(UP)
The classification rule is:
P(UP) ≥ 0.50
        ↓
       UP

P(UP) < 0.50
        ↓
      DOWN
The probability itself can also be used for future confidence analysis.
7. Baseline Model
A simple majority-class baseline is used for comparison.
The baseline always predicts the most common class in the training dataset.
This is important because raw accuracy alone can be misleading.
The research question becomes:
Does the order-flow model
perform better than a simple baseline?
rather than:
Is the model accurate?
Experimental Design
A single train/test experiment can depend heavily on the generated market path.
Therefore, the project is designed around a reproducible experimental setup.
Synthetic Market
       ↓
50,000 Events
       ↓
Feature Construction
       ↓
Time-Ordered Train/Test Split
       ↓
70% Training
30% Testing
       ↓
Model Training
       ↓
Out-of-Sample Prediction
       ↓
Multiple Prediction Horizons
       ↓
Statistical Evaluation
The experiment uses:
Random Seed = 42
to ensure reproducibility.


Why Time-Ordered Testing?
Financial time-series data should not normally be randomly shuffled when evaluating predictive performance.
Random shuffling can introduce information leakage between training and testing observations.
Therefore:
Past Data
    ↓
Training

Future Data
    ↓
Testing
This provides a more realistic evaluation of short-horizon prediction.
Actual Research Results
The current implementation was evaluated using:
Random Seed        : 42
Total Events       : 50,000
Training Data      : 70%
Testing Data       : 30%
Prediction Model   : Logistic Regression
Features           : OFI + Lagged OFI + Short-Term Returns
Prediction Horizons: 1, 5, 10 ticks


1-Tick Prediction
Accuracy       : 54.12%
ROC-AUC        : 0.5522
Log Loss       : 0.6889
Baseline       : 51.23%
The model produces a modest improvement over the majority-class baseline.
This suggests that the order-flow features contain some short-term directional information.
However, the relatively small ROC-AUC improvement indicates that the signal is weak.
5-Tick Prediction
Accuracy       : 55.44%
ROC-AUC        : 0.5732
Log Loss       : 0.6851
Baseline       : 51.91%
The 5-tick horizon produces the strongest performance in the experiment.
The improvement over baseline accuracy is:
55.44% - 51.91%
= 3.53 percentage points
This makes the 5-tick horizon the most promising prediction horizon among the tested horizons.
10-Tick Prediction
Accuracy       : 54.19%
ROC-AUC        : 0.5545
Log Loss       : 0.6891
Baseline       : 47.04%
The model remains above the baseline.
However, the predictive signal is weaker than at the 5-tick horizon when measured using ROC-AUC.
This demonstrates that predictive performance can depend significantly on the forecast horizon.


Prediction Horizon Comparison
                 Accuracy

1 Tick            54.12%
5 Ticks           55.44%
10 Ticks          54.19%
The highest accuracy occurs at:
5 Ticks
with:
55.44%
The results therefore suggest that order-flow information may have a stronger relationship with price movement at a particular short horizon rather than uniformly across all horizons.

Result Interpretation
The experiment does not demonstrate that order flow can reliably predict future prices.
Instead, it finds:
Order Flow
     ↓
Weak Short-Horizon Signal
     ↓
Out-of-Sample Predictive Information
     ↓
Accuracy Above Baseline
The strongest result is:
5-Tick Horizon

Accuracy : 55.44%
ROC-AUC  : 0.5732
compared with:
Baseline Accuracy : 51.91%
This represents a modest predictive edge in the synthetic environment.

Key Research Finding
The central finding is:
Order-flow features contain weak but measurable short-horizon predictive information in the synthetic market environment.
However:
Predictive Edge
      ≠
Trading Profitability
A prediction model may achieve accuracy above 50% but still fail to generate profitable trading returns after considering:
Bid-Ask Spread
Transaction Costs
Slippage
Latency
Market Impact
Execution Probability
Adverse Selection
Therefore, this project treats prediction as a research problem, not as a claimed profitable trading strategy.
Visualizations
The research pipeline generates several visualizations.
1. Order-Flow Imbalance
outputs/order_flow_imbalance.png
Shows the evolution of buying and selling pressure.
2. Price and Order Flow
outputs/price_order_flow.png
Compares short-term price movement with order-flow imbalance.
3. Prediction Probability
outputs/prediction_probability.png
Shows the predicted probability of upward price movement.
4. Prediction Accuracy by Horizon
outputs/horizon_comparison.png
Compares model performance across prediction horizons.
5. Confusion Matrix
outputs/confusion_matrix.png
Shows the distribution of correct and incorrect directional predictions.
6. ROC Curve
outputs/roc_curve.png
Shows the classification performance across probability thresholds.
Model Evaluation
The system evaluates the prediction model using:
Accuracy
ROC-AUC
Log Loss
Baseline Accuracy
Confusion Matrix
Prediction Probability
Horizon Comparison
Why ROC-AUC?
Accuracy depends on a selected probability threshold.
ROC-AUC evaluates how well the model ranks positive and negative outcomes across different thresholds.
Interpretation:
ROC-AUC ≈ 0.50
        ↓
No meaningful discrimination

ROC-AUC > 0.50
        ↓
Some predictive information

ROC-AUC → 1.00
        ↓
Strong discrimination
The observed ROC-AUC values are:
1 Tick   : 0.5522
5 Ticks  : 0.5732
10 Ticks : 0.5545
These values indicate weak predictive discrimination rather than a highly predictive model.
Why Log Loss?
Log loss evaluates the quality of predicted probabilities.
Unlike accuracy, it penalizes predictions that are confidently wrong.
The model therefore needs to produce not only correct classifications but also reasonably calibrated probabilities.
Research Workflow
1. Generate Market Data
        ↓
2. Generate Bid/Ask Activity
        ↓
3. Calculate Order-Flow Imbalance
        ↓
4. Construct Lagged Features
        ↓
5. Calculate Short-Term Returns
        ↓
6. Generate Future Price Labels
        ↓
7. Split Data Chronologically
        ↓
8. Train Logistic Regression
        ↓
9. Generate Out-of-Sample Predictions
        ↓
10. Evaluate Accuracy / AUC / Log Loss
        ↓
11. Compare Prediction Horizons
        ↓
12. Interpret Results
Reproducibility
The experiment uses a fixed random seed:
seed = 42
The dataset contains:
50,000 Events
and uses:
70% Training
30% Testing
The chronological split ensures that future observations are not used to train the model.
The experiment can therefore be reproduced by running:
python main.py

Project Structure
Order-Flow-Short-Horizon-Market-Prediction/
│
├── README.md
│
├── main.py
│
├── data_generator.py
│
├── features.py
│
├── models.py
│
├── evaluation.py
│
├── experiment.py
│
├── requirements.txt
│
├── .gitignore
│
└── outputs/
    │
    ├── dataset.csv
    ├── prediction_results.csv
    ├── experiment_results.csv
    │
    ├── order_flow_imbalance.png
    ├── price_order_flow.png
    ├── prediction_probability.png
    ├── horizon_comparison.png
    ├── confusion_matrix.png
    └── roc_curve.png
Installation
Clone the repository:
git clone https://github.com/YOUR_USERNAME/Order-Flow-Short-Horizon-Market-Prediction.git
Enter the project directory:
cd Order-Flow-Short-Horizon-Market-Prediction
Install dependencies:
pip install -r requirements.txt
Requirements
The project uses:
Python >= 3.9

NumPy
Pandas
Matplotlib
Scikit-learn
Install all dependencies using:
pip install -r requirements.txt
Run the Complete Research Experiment
Run:
python main.py
The program generates:
outputs/
containing:
dataset.csv
prediction_results.csv
experiment_results.csv
and the research visualizations.

Terminal Output
============================================================
ORDER FLOW & SHORT-HORIZON MARKET PREDICTION
============================================================

Random Seed             : 42
Total Events            : 50,000
Training Data           : 70%
Testing Data            : 30%
Model                   : Logistic Regression

============================================================
OUT-OF-SAMPLE RESULTS
============================================================

Horizon     Accuracy    ROC-AUC    Log Loss    Baseline
--------------------------------------------------------
1 Tick      54.12%      0.5522     0.6889      51.23%
5 Ticks     55.44%      0.5732     0.6851      51.91%
10 Ticks    54.19%      0.5545     0.6891      47.04%

============================================================
BEST HORIZON
============================================================

Horizon     : 5 Ticks
Accuracy    : 55.44%
ROC-AUC     : 0.5732

============================================================

Future Research
1. Real Level-2 Order Book Data
Replace synthetic order-flow generation with historical market data containing:
Timestamp
Bid Price
Ask Price
Bid Quantity
Ask Quantity
Trades
Trade Direction
Order Updates
2. Advanced Order-Flow Features
Potential features include:
Order-Flow Imbalance
Trade Imbalance
Volume Imbalance
Depth Imbalance
Queue Imbalance
Spread
Mid-Price Movement
Microprice
Order Arrival Rate
Order Cancellation Rate
3. Microprice Prediction
The model can be extended to predict the future microprice:
Microprice =
(Ask × Bid Volume + Bid × Ask Volume)
/
(Bid Volume + Ask Volume)
This can provide a more microstructure-oriented prediction target.
4. Machine Learning Models
Future experiments can compare:
Logistic Regression
        ↓
Random Forest
        ↓
Gradient Boosting
        ↓
XGBoost
        ↓
LightGBM
        ↓
LSTM
        ↓
Temporal CNN
        ↓
Transformer
The purpose would be to determine whether more complex models provide statistically significant improvements over the simple baseline.
5. Walk-Forward Validation
A more realistic evaluation framework is:
Training Window
       ↓
Validation
       ↓
Testing
       ↓
Move Forward
       ↓
Retrain
       ↓
New Testing Window
This reduces dependence on a single train/test split.
6. Transaction-Cost-Aware Evaluation
Prediction performance should eventually be connected to economic performance.
Prediction
    ↓
Trading Signal
    ↓
Execution
    ↓
Spread
    ↓
Fees
    ↓
Slippage
    ↓
Net P&L
This would determine whether the predictive signal has practical economic value.
7. Statistical Significance
Future research should evaluate:
Confidence intervals
Bootstrap testing
Permutation tests
Diebold-Mariano tests
Stability across market regimes
Multiple-seed experiments
The goal is to distinguish:
Real Predictive Signal
from:
Random Statistical Noise

Final Research Conclusion
The experiment provides evidence that order-flow information contains weak but measurable short-horizon predictive information in the synthetic market environment.
The strongest observed result is:
Prediction Horizon : 5 Ticks
Accuracy            : 55.44%
ROC-AUC             : 0.5732
Baseline Accuracy   : 51.91%
The result suggests that order-flow features may contain useful information about near-term price direction.
However:
Small Predictive Edge
        ≠
Profitable Trading Strategy
Further research using real Level-2 order-book data, walk-forward validation, transaction costs, execution modeling, and statistical significance testing is required before making any claim about real-world trading performance.
Disclaimer
This project is intended for educational and quantitative research purposes only.
It does not constitute financial advice, investment advice, or a live trading strategy.
The reported results are generated from a synthetic market environment and should not be interpreted as evidence of real-world trading profitability.
Author
Thirupathi Kannan K
GitHub:
https://github.com/thirupathikannan-ai?
