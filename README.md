# MLFA: Fundamentals-Driven Equity Modeling Pipeline

**MLFA** is a professional-grade, end-to-end machine learning pipeline designed to predict the **probability distribution** of future stock returns using fundamental accounting data. Unlike traditional "black box" models that output a single price target, MLFA uses **Quantile Regression** to forecast the full range of risks and opportunities (downside, median, and upside scenarios).

This documentation provides a "whitepaper-level" deep dive into the mathematics, engineering, and financial theory behind the system.

---

## 📚 Table of Contents
1.  [The Core Philosophy](#-the-core-philosophy)
2.  [Mathematical Foundations](#-mathematical-foundations)
    *   [The Data Problem (Normalization)](#1-the-data-problem-normalization--scale-invariance)
    *   [The Time Problem (Leakage)](#2-the-time-problem-leakage-safety)
    *   [The Prediction Problem (Quantile Regression)](#3-the-prediction-problem-quantile-regression--pinball-loss)
3.  [Feature Engineering Deep Dive](#-feature-engineering-deep-dive)
4.  [System Architecture](#-system-architecture)
    *   [Layer 1: Lossless Ingestion](#layer-1-lossless-ingestion-data-acq)
    *   [Layer 2: Advanced Modeling](#layer-2-advanced-modeling-modelsmodel-b)
    *   [Layer 3: Visualization](#layer-3-visualization-dashboard)
5.  [Usage Guide](#-usage-guide)

---

## 🧠 The Core Philosophy

The project is built on three axioms:
1.  **Stock prices are noisy, but business fundamentals are the signal.** Short-term price action is random walk; long-term returns are driven by Return on Invested Capital (ROIC) and earnings growth.
2.  **Uncertainty is the only certainty.** Predicting a specific price (e.g., "$150.00") is mathematically arrogant. A robust model must predict a *distribution* (e.g., "90% chance price > $100").
3.  **Information Asymmetry is fatal.** If a model trains on data that wasn't public at the time (Look-Ahead Bias), it is worthless. We must rigorously simulate the "information lag" of real-world reporting.

---

## 📐 Mathematical Foundations

### 1. The Data Problem: Normalization & Scale Invariance
**The Challenge:** Machine learning models struggle with unnormalized financial data.
*   **Scale Variance:** Apple (\$380B revenue) vs. a Small Cap (\$100M revenue). If fed raw numbers, the model learns "Big Number = Good," which is false (large companies often grow slower).
*   **Heavy Tails:** Financial data follows a Power Law (Pareto) distribution, not a Bell Curve (Gaussian). Outliers (e.g., Amazon's assets) can be 10,000x the median, destroying gradient descent convergence.

**The Solutions:**

#### A. Ratio Normalization (The "Common Size" Approach)
We convert absolute accounting values into relative ratios. This makes the data **scale-invariant**.
*   Instead of `NetIncome` ($), we use `NetMargin` (%):
    $$NetMargin = \frac{NetIncome}{TotalRevenue}$$
*   Instead of `Debt` ($), we use `DebtToAssets` (%):
    $$DebtToAssets = \frac{LongTermDebt}{TotalAssets}$$

#### B. Signed Logarithmic Compression
For variables that *must* remain absolute (like `TotalAssets` to represent size, or `Revenue` to represent scale), we apply a transformation that compresses the heavy tails while preserving the sign (positive/negative) and zero values.
$$f(x) = \text{sign}(x) \cdot \ln(1 + |x|)$$
*   **Why not standard log?** Standard $\ln(x)$ is undefined for $x \le 0$. Companies can have negative equity or income.
*   **Effect:** Reduces the range from $[0, 10^{12}]$ to $[0, \approx 28]$, making it digestible for neural nets and tree-based models.

#### C. Seasonality Removal (YoY Deltas)
Most businesses have seasonal cycles (e.g., Retail in Q4). Comparing Q4 to Q3 is misleading. We use **Year-over-Year (YoY) Deltas**:
$$\Delta_{YoY} X_t = \frac{X_t - X_{t-4}}{X_{t-4}}$$
*   *Why:* This isolates organic growth from seasonal noise.

---

### 2. The Time Problem: Leakage Safety
**The Challenge:** **Look-Ahead Bias**.
If you train a model to predict returns starting Jan 1st using Q4 financial data (ending Dec 31st), you are cheating.
*   *Reality:* Companies don't file their 10-K/10-Q until mid-February (45+ days later).
*   *Result:* Your model "knows" the earnings surprise before the market does. It will show amazing backtest results but fail in production.

**The Solution: The 45-Day Lag Rule**
We rigorously define the "As-Of Date" (the first moment data is usable):
$$\text{AsOfDate} = \text{FiscalPeriodEnd} + \text{PUBLICATION\_LAG\_DAYS} (45)$$
*   **Training:** The target label (future return) is calculated starting from `AsOfDate`.
*   **Prediction:** We only generate predictions if `Today >= AsOfDate`.

---

### 3. The Prediction Problem: Quantile Regression & Pinball Loss
**The Challenge:** Standard regression (Least Squares) predicts the **conditional mean** ($E[y|x]$).
*   *Flaw:* In finance, the mean is boring. We care about the **tails** (Risk vs. Moonshots). We want to know: "What is the worst-case scenario?" (Value at Risk).

**The Solution: Quantile Regression**
We train the model to minimize the **Pinball Loss** (also known as Tilted Absolute Loss).
For a target quantile $\tau \in (0, 1)$ (e.g., 0.05):

$$L_\tau(y, \hat{y}) = \begin{cases} \tau(y - \hat{y}) & \text{if } y \ge \hat{y} \quad (\text{Underprediction}) \\ (\tau - 1)(y - \hat{y}) & \text{if } y < \hat{y} \quad (\text{Overprediction}) \end{cases}$$

**Intuition:**
*   **Case $\tau = 0.05$ (Downside Risk):**
    *   If the model predicts too high ($y < \hat{y}$), the penalty is small ($(0.05-1) \approx -0.95$).
    *   If the model predicts too low ($y > \hat{y}$), the penalty is tiny ($0.05$).
    *   *Wait, actually:* The logic is that to minimize loss for a low quantile, the model is pushed to prediction values where 95% of actual $y$ values are *above* it. It effectively finds the "floor."
*   **Case $\tau = 0.95$ (Upside Potential):**
    *   The model is heavily penalized for underpredicting. It is forced to push the prediction up until 95% of data is below it.

We train **5 Independent LightGBM Models** for $\tau \in \{0.05, 0.25, 0.50, 0.75, 0.95\}$.
This gives us a full probability distribution:
*   **Bear Case:** $q_{05}$
*   **Typical Range:** $[q_{25}, q_{75}]$ (Interquartile Range)
*   **Bull Case:** $q_{95}$

---

## 🧪 Feature Engineering Deep Dive

The `model_b.py` pipeline engineers features based on academic financial literature.

### 1. Quality of Earnings (Sloan Accruals)
**Formula:**
$$\text{AccrualsRatio} = \frac{\text{NetIncome} - \text{OperatingCashFlow}}{\text{TotalAssets}}$$
**Theory:** (Richard Sloan, 1996)
*   Earnings driven by **cash** are persistent.
*   Earnings driven by **accruals** (accounting adjustments, e.g., "Accounts Receivable") are temporary and mean-reverting.
*   **Signal:** High Accruals = **Short Signal** (Earnings quality is low).

### 2. Rolling Volatility (Stability Premium)
**Formula:**
$$\sigma_{margin} = \text{StdDev}(\text{Margin}_{t}, \text{Margin}_{t-1}, \dots, \text{Margin}_{t-7})$$
**Theory:**
*   Investors pay a higher multiple (P/E) for certainty.
*   A company with a stable 20% margin is more valuable than one swinging between 0% and 40%.
*   **Signal:** Low $\sigma$ = **Long Signal**.

### 3. Efficiency (Asset Turnover)
**Formula:**
$$\text{AssetTurnover} = \frac{\text{TotalRevenue}}{\text{TotalAssets}}$$
**Theory:** (DuPont Analysis)
*   Measures how many dollars of sales the company generates for every dollar of assets.
*   **Signal:** Rising Turnover = Improving Efficiency.

### 4. Momentum & Technicals
**Formula:**
$$Mom_{12m} = \frac{Price_t}{Price_{t-252}} - 1$$
**Theory:**
*   **Jegadeesh & Titman (1993):** Winners tend to keep winning (Momentum Anomaly).
*   Fundamentals don't matter if the market hates the stock. We combine "Value" (Fundamentals) with "Catalyst" (Momentum).

---

## 🏗 System Architecture

### Layer 1: Lossless Ingestion (`data-acq/`)
*   **Design Pattern:** "Bronze Layer" (Raw Data Lake).
*   **Objective:** Never lose data. Never break on API errors.
*   **Logic:**
    1.  **Dual-Provider Fallback:**
        *   Primary: **Financial Modeling Prep (FMP)** (High quality, throttled).
        *   Secondary: **Alpha Vantage (AV)** (Fallback, rate-limited).
    2.  **Envelope Storage:**
        *   Saves `data/raw/<SNAPSHOT>/<TICKER>/income_statement.json`.
        *   Content: `{"_metadata": {timestamp, provider, status}, "response": {...}}`.
        *   *Benefit:* Complete audit trail. If the model breaks, we can prove if it was bad code or bad data.

### Layer 2: Advanced Modeling (`models/model-B/`)
*   **Design Pattern:** "Silver/Gold Layer" (Feature Store & Inference).
*   **Algorithm:** LightGBM (Gradient Boosting Decision Trees).
    *   *Why:* Handles tabular data, missing values, and non-linear interactions better than Deep Learning for this scale of data.
*   **Validation Strategy:**
    *   **Time-Series Split:** Train on [2010-2018], Validate on [2019]. *Never shuffle time.*
    *   **Monotonicity Enforcement:** Post-processing ensures $q_{05} \le q_{25} \le q_{50} \dots$ (prevents crossing quantiles).

### Layer 3: Visualization (`dashboard/`)
*   **Design Pattern:** Interactive BI Tool.
*   **Tech:** Streamlit + Plotly.
*   **Key Components:**
    *   **Fan Chart:** Visualizes the "Cone of Uncertainty."
    *   **TreeSHAP (XAI):** Explains *why*.
        *   *Example:* "This prediction is bullish (+15%) primarily because `NetMarginYoY` is +5% and `DebtToAssets` decreased."

---

## 💻 Usage Guide

### 1. Setup Environment
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. Ingest Data (Layer 1)
Pull raw data. The script handles API limits and retries automatically.
```bash
# Get data for Tech Giants
python data-acq/data-acquisition.py --tickers AAPL MSFT GOOG NVDA --snapshot 20251029
```

### 3. Build Dataset (Layer 2a)
Transform raw JSONs into a clean Parquet file with engineered features and lagged labels.
```bash
python models/model-B/model_b.py build-dataset --snapshot 20251029 --period quarterly
```
*   *Output:* `data/structured/20251029/dataset_model_b.parquet`

### 4. Train Models (Layer 2b)
Train the 5 Quantile LightGBM models.
```bash
python models/model-B/model_b.py train --snapshot 20251029
```
*   *Output:* `models/model-B/20251029/*.txt` (Saved Boosters)

### 5. Predict & Visualize (Layer 3)
Generate predictions and launch the dashboard.
```bash
# Generate predictions for specific tickers
python models/model-B/model_b.py predict --snapshot 20251029 --tickers AAPL MSFT

# Run the UI
streamlit run dashboard/dashboard.py
```

---

## ⚠️ Risk & Limitations
1.  **Regime Change:** Models trained on a Bull Market (2010-2020) may fail in a Stagflationary environment.
2.  **Data Quality:** "Garbage In, Garbage Out." If FMP/AV provides bad data, the model will hallucinate.
3.  **Not Investment Advice:** This is a statistical research tool. It estimates probability, not certainty.
