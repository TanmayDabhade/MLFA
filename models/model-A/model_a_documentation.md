# Model A — Quantile Gradient Boosting on Fundamentals (LightGBM)

A **leakage-safe**, fundamentals-first forecasting pipeline that:

* **Builds** a clean dataset from raw financial statements (income, balance sheet, cashflow) you’ve already ingested.
* **Labels** each row with the **12-month forward total return** (via Yahoo Finance *Adjusted Close*).
* **Trains** five **quantile regression** models (5th, 25th, 50th, 75th, 95th percentiles) using **LightGBM**.
* **Predicts** a **range** of plausible future returns per ticker (not just one number), which is easier to reason about and safer to use.

> **Audience:** This README is written so **non-ML** readers can follow along. When a term sounds statistical, there’s a plain-English explanation and a quick “why it matters.”

---

## TL;DR

* You point the script at your **raw fundamentals** folder for a given **snapshot** date.
* It merges statements, engineers intuitive ratios (margins, current ratio, leverage, FCF, YoY changes), and sets an **“as-of”** date = fiscal period end + **45-day publication lag** (prevents cheating/leakage).
* It then grabs **prices** from Yahoo, computes each row’s **12-month forward return**, and stores a dataset.
* It fits **5 LightGBM models**, each predicting a **different percentile** of next-year returns (from downside to upside).
* For the latest fundamentals of a ticker, it returns a **quantile band** (q05–q95) that frames **downside/typical/upside** scenarios.

---

## Why this project exists

* Many stock “prediction” demos use price charts only. **Real analysts look at fundamentals.**
* Instead of claiming “the price will be X,” we present a **distribution** of outcomes:

  * **q05** ≈ bearish case
  * **q50** ≈ typical/median case
  * **q95** ≈ bullish case
* That range is more honest and more **useful for decision-making** than a single point guess.

---

## Core ideas (no ML background required)

### 1) Leakage safety (what is it, and why it matters?)

If you train on information that **wasn’t public yet** at the time, the model “cheats.”
We avoid this by shifting each fiscal report’s usable date to **45 days after fiscal period end**. The model **never** sees data earlier than investors could have.

### 2) Quantiles vs. a single prediction

* A single prediction says “+8% next year.” That’s misleading—**markets are uncertain**.
* Quantiles give you **bounds**: “downside −20% (q05), typical +8% (q50), upside +35% (q95).”
* You can make better decisions with a **range** (position sizing, risk budgeting).

### 3) Adjusted Close (total-return proxy)

* Yahoo’s *Adjusted Close* accounts for **splits and dividends**.
* Using it is a decent approximation to **total return**, which is what investors care about.

---

## What the code expects

```
data-acq/
  data/
    raw/
      <SNAPSHOT>/                 # e.g., 20251022
        AAPL/
          income_statement.json
          balance_sheet.json
          cashflow_statement.json
          ratios.json              # optional (ignored for now)
          key_metrics.json         # optional (ignored for now)
          enterprise_values.json   # optional (ignored for now)
        MSFT/
          ...
```

> If your ingestion used `data-acq/data/raw/snapshot/<SNAPSHOT>/...`, that’s also supported.

You can override the raw-data root via env var **`RAW_DIR`** (default: `data-acq/data/raw`).

---

## Install & prerequisites

* **Python**: 3.9–3.12 recommended (3.13 works but some wheels may lag).
* **Packages**:

  * `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `yfinance`, `joblib`, `python-dateutil`

```bash
# Always install into the same Python you use to run the script:
python -m pip install -U pandas numpy scikit-learn lightgbm yfinance joblib python-dateutil
```

> **macOS tip (LightGBM):** If you hit build errors on Intel macs:
>
> * Ensure Command Line Tools installed: `xcode-select --install`
> * Try: `python -m pip install --no-binary lightgbm lightgbm`

---

## Quick start

```bash
# 1) Build dataset (merges fundamentals, computes features, fetches prices, builds labels)
python model_a.py build-dataset --snapshot 20251022 --period quarterly

# 2) Train 5 quantile models (q05, q25, q50, q75, q95)
python model_a.py train --snapshot 20251022

# 3) Predict quantiles for latest fundamentals for a few tickers
python model_a.py predict --snapshot 20251022 --period quarterly --tickers AAPL MSFT TSLA
```

Artifacts are written under:

```
data/structured/<SNAPSHOT>/dataset_model_a.parquet
models/model_a/<SNAPSHOT>/
  ├─ lgbm_q5.txt
  ├─ lgbm_q25.txt
  ├─ lgbm_q50.txt
  ├─ lgbm_q75.txt
  ├─ lgbm_q95.txt
  ├─ preprocessor.joblib
  └─ metadata.json
```

---

## How it works (step-by-step)

### Step 1 — Load raw statements

For each ticker, we load three endpoints:

* **Income statement** (revenue, gross profit, operating income, net income, etc.)
* **Balance sheet** (assets, liabilities, equity, cash, inventories, etc.)
* **Cashflow statement** (operating cash flow, capex, investing/financing flows, dividends)

The loader supports both **Alpha Vantage** shapes and **FMP** shapes (the code maps field names where they differ).

### Step 2 — Build features that make intuitive sense

For each fiscal period we compute:

* **Margins**

  * `grossMargin = grossProfit / totalRevenue`
  * `operatingMargin = operatingIncome / totalRevenue`
  * `netMargin = netIncome / totalRevenue`
* **Liquidity & leverage**

  * `currentRatio = totalCurrentAssets / totalCurrentLiabilities`
  * `debtToAssets = longTermDebt / totalAssets`
* **Cash & efficiency**

  * `freeCashFlow = operatingCashflow - capitalExpenditures`
  * `fcfMargin = freeCashFlow / totalRevenue`
  * `cashPctAssets = (cash + shortTermInvestments) / totalAssets`
* **Growth & changes (YoY for quarterly using lag of 4)**

  * `revYoY = pct_change_4q(totalRevenue)`
  * `netIncomeYoY = pct_change_4q(netIncome)`
  * `opMarginYoY = diff_4q(operatingMargin)`
  * `netMarginYoY = diff_4q(netMargin)`

> **Why these?** They’re the **core lenses** analysts use:
> profitability (margins), solvency (liquidity/leverage), cash generation, and growth.

### Step 3 — Enforce publication lag to avoid leakage

We set `asof_date = fiscalDateEnding + 45 days`.
Training and labeling treat **this** as the first day the information is usable.

### Step 4 — Label each row with 12-month forward total return

* Pull **Adjusted Close** from Yahoo for each ticker across the entire date range needed.
* For each row:

  1. Move to the **next trading day** on or after `asof_date`.
  2. Jump forward **252 trading days** (~12 months).
  3. Compute return = `AdjClose_t+252 / AdjClose_t - 1`.

Rows that can’t be labeled (missing prices, out-of-range) are dropped.

### Step 5 — Train five quantile models (LightGBM)

* We impute missing values with **median** (tree models don’t need scaling).
* We fit **five** LightGBM models with objective **`quantile`** at α ∈ {0.05, 0.25, 0.50, 0.75, 0.95}.
* Each model learns to estimate a **different percentile** of the future return.

> **What is LightGBM?** A fast, industry-standard algorithm that builds an **ensemble of decision trees** to capture non-linear patterns efficiently.

### Step 6 — Predict for latest fundamentals

For requested tickers, we:

* Build **the latest** feature row (based on the most recent `asof_date` available in the snapshot).
* Run it through the preprocessor + five models.
* Return a compact table with `ticker`, `asof_date`, and columns `q5, q25, q50, q75, q95`.

---

## Interpreting the predictions

Sample (illustrative):

| ticker |  asof_date |   q05 |   q25 |  q50 |  q75 |  q95 |
| :----: | :--------: | ----: | ----: | ---: | ---: | ---: |
|  AAPL  | 2025-08-15 | -0.22 | -0.05 | 0.06 | 0.18 | 0.36 |

* **q05 = −22%**: severe downside case over the next 12 months.
* **q50 = +6%**: typical case (median).
* **q95 = +36%**: strong upside case.

How to use:

* **Risk framing:** If you only tolerate −10% downside, AAPL’s q05=−22% may be too risky for your mandate.
* **Position sizing:** Allocate less when downside quantiles are ugly; more when the **whole band** shifts upward.
* **Ranking:** Compare **q50** or **q75** across tickers to prioritize research.

> **Not investment advice.** Treat this as a **screening** / research tool, not a trading signal by itself.

---

## Configuration knobs (no code changes required)

* `RAW_DIR` (env var): base folder for raw JSON
  Default: `data-acq/data/raw`
* `LABEL_HORIZON_TRADING_DAYS` (in code): 252 (~12 months)
* `PUBLICATION_LAG_DAYS` (in code): 45 (conservative)

---

## Data dictionary (features)

| Feature         | Meaning (plain English)                                                    |
| --------------- | -------------------------------------------------------------------------- |
| totalRevenue    | Sales for the period                                                       |
| grossProfit     | Revenue − cost of goods sold                                               |
| operatingIncome | Profit from core business before interest/taxes                            |
| netIncome       | Bottom-line profit                                                         |
| grossMargin     | GrossProfit / Revenue                                                      |
| operatingMargin | OperatingIncome / Revenue                                                  |
| netMargin       | NetIncome / Revenue                                                        |
| currentRatio    | Ability to pay short-term obligations (CurrentAssets / CurrentLiabilities) |
| debtToAssets    | Long-term debt burden relative to asset base                               |
| freeCashFlow    | Cash from operations minus capital spending (what’s left over)             |
| fcfMargin       | Free cash flow as a % of revenue                                           |
| cashPctAssets   | Cash + short-term investments relative to assets                           |
| revYoY          | Revenue growth vs. same quarter last year                                  |
| netIncomeYoY    | Net income growth vs. same quarter last year                               |
| opMarginYoY     | Change in operating margin vs. same quarter last year                      |
| netMarginYoY    | Change in net margin vs. same quarter last year                            |
| y_12m (target)  | 12-month forward total return (Adj Close proxy)                            |

---

## Reproducibility & determinism

* `RANDOM_STATE = 42` is set for LightGBM’s seed.
* Results can still vary slightly across CPU/OS and threading. For stricter runs, consider:

  * Fixing LightGBM threads and disabling some stochastic params.
  * Training on the same Python & package versions.

---

## Extending the project

### Add more features

Edit `_feature_frame()`:

* Pull additional fields from the dataframes and create more ratios (e.g., ROE, ROIC, asset turnover).
* Keep new features **intuitive** (accounting 101 ratios tend to be robust).

Update training automatically:

* The pipeline picks up **all columns except** identifiers and the target.
* Metadata stores the feature list used at train time for consistent predict-time columns.

### Change the horizon or quantiles

* To forecast **6 months**: set `LABEL_HORIZON_TRADING_DAYS = 126`.
* To add **q10** or **q90**: append to `QUANTILES = [...]` and retrain.

### Use annuals instead of quarterlies

* Pass `--period annual` at dataset build / predict time.
* Consider adjusting YoY logic (lag 1 instead of 4).

---

## Troubleshooting (quick fixes)

**1) `ModuleNotFoundError: No module named 'yfinance'`**
You likely installed into a different Python than the one running the script.

* Always install with **the same interpreter**:

  ```bash
  python -m pip install yfinance
  ```
* If you have multiple Pythons, prefer `pyenv` or a virtualenv and be consistent.

**2) `Dataset not found: data/structured/<SNAPSHOT>/dataset_model_a.parquet`**
You must run **build-dataset** before **train** / **predict**:

```bash
python model_a.py build-dataset --snapshot 20251022 --period quarterly
```

**3) `No tickers found under <...>`**
Check your folder layout. It must be:

```
<RAW_DIR>/<SNAPSHOT>/<TICKER>/{income_statement,balance_sheet,cashflow_statement}.json
```

or

```
<RAW_DIR>/snapshot/<SNAPSHOT>/<TICKER>/...
```

**4) `Failed to load Yahoo prices.`**

* Ensure tickers are valid (e.g., `BRK-B` should be `BRK-B` on Yahoo; some require special suffixes).
* Check internet connectivity / rate limiting. Re-run.

**5) LightGBM build/install errors on macOS**

* Install Xcode CLT: `xcode-select --install`
* Try: `python -m pip install --no-binary lightgbm lightgbm`

**6) Sparse fundamentals leading to many NaNs**

* The pipeline uses **median imputation**.
* You can drop thin tickers or require a minimum count of non-missing features before training.

---

## Frequently asked (plain English)

**Q: Is this “AI that beats the market”?**
A: No promise. It’s a **research tool** to quantify how fundamentals relate to **future return distributions**.

**Q: Why not include valuation ratios (P/E, EV/EBITDA)?**
A: Easy to add later. We started with **pure accounting** signals to keep it transparent and vendor-neutral.

**Q: Why 45 days of lag?**
A: Conservative buffer so the model **never sees** data that the market didn’t. You can tune it, but shorter lags risk leakage.

**Q: Why 12 months?**
A: Many fundamentals resolve over a year. You can set other horizons if you have use-cases for them.

**Q: Can I use this for ranking stocks?**
A: Yes. Sort by **q50** (median) or a composite like **0.5·q75 + 0.5·q50**, then sanity-check outliers.

---

## Project structure (what lives where)

* `model_a.py` — the entire CLI: build dataset, train, predict.
* `data/structured/<SNAPSHOT>/dataset_model_a.parquet` — engineered features + labels.
* `models/model_a/<SNAPSHOT>/` — preprocessor, metadata, and five quantile models.

Key functions inside `model_a.py` (human-readable summary):

* `_load_statements(...)`: reads each JSON and normalizes Alpha Vantage vs. FMP shapes.
* `_feature_frame(...)`: merges statements and builds accounting ratios.
* `build_dataset(...)`: constructs the training dataset and labels it with future returns.
* `train_models(...)`: imputes, trains five LightGBM quantile models, saves artifacts.
* `predict_latest(...)`: builds the latest feature row per ticker and outputs quantiles.

---

## Risk, ethics, and limitations

* **Not investment advice.** Treat outputs as **inputs to research**, not as trade signals.
* **Data quality matters.** Bad or stale fundamentals produce noisy estimates.
* **Regime risk.** Relationships can change (rate cycles, macro shocks). Retrain periodically.
* **Coverage bias.** Small/micro caps with thin filings or ticker mapping quirks may be underrepresented.

---

## Maintenance checklist

* Re-ingest fundamentals for a **new snapshot** (e.g., monthly).
* Re-run `build-dataset` → `train` to keep models current.
* Track dataset size in `metadata.json` (`n_rows`). If too small, expect volatility in predictions.
* Consider storing **train/test splits** and tracking out-of-sample error over time.

---

## License and attribution

* You own your ingestion and model artifacts.
* Libraries used: LightGBM, scikit-learn, pandas, numpy, yfinance (see their respective licenses).

---

## Changelog (suggested to maintain)

* **v0.1**: First public quantile GBM on fundamentals (q05–q95), leakage-safe as-of logic, 12m horizon.

---

### Appendix — How quantile loss works (intuitive)

For a chosen percentile **α** (say 0.75), the model is penalized more when its prediction is **below** outcomes that are supposed to lie under the **75th percentile**. In practice, it learns the curve that says, “Given these fundamentals, **three-quarters** of the time the return should be at or **below** this value.” Training five such models gives you a **fan chart** of plausible futures.

