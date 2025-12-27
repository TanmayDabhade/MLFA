# Running the MLFA Pipeline

This guide provides step-by-step instructions to set up and run the **MLFA (Fundamentals-Driven Equity Modeling Pipeline)**.

## 📋 Prerequisites

- **Python 3.8+**
- **Virtual Environment** (recommended)

---

## 🚀 1. Setup Environment

First, clone the repository and set up your Python environment.

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install dependencies
python -m pip install -r requirements.txt
```

---

## 🔑 2. API Configuration

The pipeline uses **Financial Modeling Prep (FMP)** as the primary data source and **Alpha Vantage (AV)** as a fallback.

1.  Create a `.env` file in the `data-acq/` directory (or the root directory).
2.  Add your API keys:

```env
FMP_API_KEY=your_fmp_api_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here
```

> [!TIP]
> You can get a free API key from [Financial Modeling Prep](https://site.financialmodelingprep.com/developer/docs/) and [Alpha Vantage](https://www.alphavantage.co/support/#api-key).

---

## 📥 3. Data Acquisition (Layer 1)

Fetch raw financial data for your target tickers. The script handles rate limiting and provider fallback automatically.

```bash
# Example: Fetch data for Tech Giants
python data-acq/data-acquisition.py --tickers AAPL MSFT GOOG NVDA --snapshot 20251029
```

- `--tickers`: List of stock tickers.
- `--snapshot`: A folder name for this data pull (usually YYYYMMDD).
- `--period`: `quarter` (default) or `annual`.

---

## 🧠 4. Modeling Pipeline (Layer 2)

### A. Build Dataset
Transform raw JSON data into structured features for machine learning.

```bash
python models/model-B/model_b.py build-dataset --snapshot 20251029 --period quarterly
```

### B. Train Models
Train the 5 Quantile LightGBM models ($\tau \in \{0.05, 0.25, 0.50, 0.75, 0.95\}$).

```bash
python models/model-B/model_b.py train --snapshot 20251029
```

### C. Generate Predictions
Generate predictions for specific tickers to be used in the dashboard.

```bash
python models/model-B/model_b.py predict --snapshot 20251029 --tickers AAPL MSFT
```

---

## 📊 5. Run the Dashboard (Layer 3)

Launch the interactive Streamlit dashboard to visualize the "Cone of Uncertainty" and model explanations.

```bash
streamlit run dashboard/dashboard.py
```

- **Fan Chart**: Visualizes the predicted probability distribution of future returns.
- **TreeSHAP**: Explains which fundamental factors (e.g., Net Margin, Debt/Assets) are driving the prediction.

---

## 🛠 Troubleshooting

- **API Limits**: If you are using free tiers, you may encounter rate limits. The scripts include sleep timers, but for large batches, consider upgrading your plan or reducing the number of tickers.
- **Missing Data**: If a ticker is missing fundamental data on the providers' end, the script will log a warning and skip that ticker.
- **Environment**: Ensure you are running commands from the root directory of the project.

---

## 📚 Further Reading

For a deep dive into the mathematics and financial theory behind this pipeline, refer to the [README.md](file:///Users/tanmay/Desktop/fundamental-ML/README.md).
