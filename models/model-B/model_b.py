#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model A: Quantile Gradient Boosting (LightGBM)
- Builds a leakage-safe dataset from raw fundamentals JSON (FMP / AlphaVantage).
- Labels with 12m forward total return via Yahoo Finance (Adj Close).
- Trains 5 quantile models (5/25/50/75/95th).
- Predicts quantiles for latest fundamentals per ticker.

Folder assumptions (from your ingestion):
  data/raw/<SNAPSHOT>/<TICKER>/{income_statement,balance_sheet,cashflow_statement,ratios,key_metrics,enterprise_values}.json

Commands:
  python model_a.py build-dataset --snapshot 20251022 --period quarterly
  python model_a.py train --snapshot 20251022
  python model_a.py predict --snapshot 20251022 --tickers AAPL MSFT TSLA
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from joblib import dump, load
from datetime import datetime, timedelta
from dateutil.parser import parse as dtparse

import lightgbm as lgb


# ------------------------------
# Config
# ------------------------------
RAW_DIR = Path(os.environ.get("RAW_DIR", "data-acq/data/raw"))
OUT_DIR = Path("data/structured")
MODELS_DIR = Path("models/model_a")
LABEL_HORIZON_TRADING_DAYS = 252      # ~12 months
PUBLICATION_LAG_DAYS = 45             # conservative lag to avoid leakage
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]
RANDOM_STATE = 42

INCOME_KEYS_AV = [
    "totalRevenue","grossProfit","operatingIncome","netIncome",
    "researchAndDevelopment","sellingGeneralAndAdministrative",
    "depreciationAndAmortization","costOfRevenue","operatingExpenses"
]
BAL_KEYS_AV = [
    "totalAssets","totalLiabilities","totalShareholderEquity",
    "cashAndCashEquivalentsAtCarryingValue","shortTermInvestments",
    "longTermDebt","totalCurrentAssets","totalCurrentLiabilities","inventory"
]
CASH_KEYS_AV = [
    "operatingCashflow","capitalExpenditures","cashflowFromInvestment",
    "cashflowFromFinancing","dividendPayout"
]

# FMP stable likely uses slightly different field names; map best-effort:
FMP_MAP_INCOME = {
    "date":"fiscalDateEnding",
    "revenue":"totalRevenue",
    "grossProfit":"grossProfit",
    "operatingIncome":"operatingIncome",
    "netIncome":"netIncome",
    "researchAndDevelopment":"researchAndDevelopment",
    "sellingGeneralAndAdministrative":"sellingGeneralAndAdministrative",
    "depreciationAndAmortization":"depreciationAndAmortization",
    "costOfRevenue":"costOfRevenue",
    "operatingExpenses":"operatingExpenses",
}
FMP_MAP_BAL = {
    "date":"fiscalDateEnding",
    "totalAssets":"totalAssets",
    "totalLiabilities":"totalLiabilities",
    "totalEquity":"totalShareholderEquity",
    "cashAndCashEquivalents":"cashAndCashEquivalentsAtCarryingValue",
    "shortTermInvestments":"shortTermInvestments",
    "longTermDebt":"longTermDebt",
    "totalCurrentAssets":"totalCurrentAssets",
    "totalCurrentLiabilities":"totalCurrentLiabilities",
    "inventory":"inventory",
}
FMP_MAP_CASH = {
    "date":"fiscalDateEnding",
    "netCashProvidedByOperatingActivities":"operatingCashflow",
    "capitalExpenditure":"capitalExpenditures",
    "netCashUsedForInvestingActivites":"cashflowFromInvestment",
    "netCashUsedProvidedByFinancingActivities":"cashflowFromFinancing",
    "dividendsPaid":"dividendPayout"
}

def _slog1p(x: pd.Series) -> pd.Series:
    # signed log1p: handles negatives and heavy tails
    return np.sign(x) * np.log1p(np.abs(x))

def _safe_div(n: pd.Series, d: pd.Series) -> pd.Series:
    return n.astype(float) / d.replace({0.0: np.nan}).astype(float)

def _to_float(x):
    if x is None or x == "" or str(x).lower() == "none":
        return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan

def _read_json(path: Path) -> Optional[dict]:
    if not path.exists(): return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def _extract_response(doc: dict) -> Any:
    # Our ingestion envelope is {"_metadata":..., "response": ...}
    if not doc: return None
    return doc.get("response")

def _parse_statement(endpoint: str, resp: Any, period: str) -> pd.DataFrame:
    """
    endpoint in {"income_statement","balance_sheet","cashflow_statement"}
    Handles Alpha Vantage (dict with quarterlyReports/annualReports) and FMP (list of dicts).
    Returns DataFrame with 'fiscalDateEnding' + numeric columns.
    """
    period_key = "quarterlyReports" if period == "quarterly" else "annualReports"

    # Alpha Vantage shape: dict with arrays
    if isinstance(resp, dict) and period_key in resp:
        rows = resp.get(period_key, [])
        df = pd.DataFrame(rows)
        if "fiscalDateEnding" in df.columns:
            df["fiscalDateEnding"] = pd.to_datetime(df["fiscalDateEnding"], errors="coerce")
        # Keep numeric
        for c in df.columns:
            if c != "fiscalDateEnding":
                df[c] = df[c].map(_to_float)
        return df.sort_values("fiscalDateEnding")

    # FMP shape: list[dict]
    if isinstance(resp, list):
        df = pd.DataFrame(resp)
        # map fields to canonical names
        if endpoint == "income_statement":
            for src, dst in FMP_MAP_INCOME.items():
                if src in df.columns and dst not in df.columns:
                    df[dst] = df[src]
        elif endpoint == "balance_sheet":
            for src, dst in FMP_MAP_BAL.items():
                if src in df.columns and dst not in df.columns:
                    df[dst] = df[src]
        elif endpoint == "cashflow_statement":
            for src, dst in FMP_MAP_CASH.items():
                if src in df.columns and dst not in df.columns:
                    df[dst] = df[src]

        # coerce date
        date_col = "fiscalDateEnding" if "fiscalDateEnding" in df.columns else "date"
        if date_col in df.columns:
            df["fiscalDateEnding"] = pd.to_datetime(df[date_col], errors="coerce")
        else:
            df["fiscalDateEnding"] = pd.NaT
        # numeric coercion
        for c in df.columns:
            if c != "fiscalDateEnding":
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.sort_values("fiscalDateEnding")

    return pd.DataFrame(columns=["fiscalDateEnding"])

def _load_statements(raw_root: Path, snapshot: str, ticker: str, period: str) -> Dict[str, pd.DataFrame]:
    # Support two common layouts:
    # 1) raw_root/<snapshot>/<ticker>/...
    # 2) raw_root/snapshot/<snapshot>/<ticker>/...
    base = raw_root / snapshot / ticker
    if not base.exists():
        alt = raw_root / "snapshot" / snapshot / ticker
        if alt.exists():
            base = alt
    out = {}
    for ep in ["income_statement", "balance_sheet", "cashflow_statement"]:
        doc = _read_json(base / f"{ep}.json")
        resp = _extract_response(doc) if doc else None
        out[ep] = _parse_statement(ep, resp, period)
    return out

def _feature_frame(ticker: str, stmts: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    inc = stmts["income_statement"].copy()
    bal = stmts["balance_sheet"].copy()
    csh = stmts["cashflow_statement"].copy()

    if inc.empty:
        return pd.DataFrame()

    # Merge on fiscalDateEnding (outer to keep maximal coverage)
    df = inc.merge(bal, on="fiscalDateEnding", how="outer", suffixes=("","_bal"))\
            .merge(csh, on="fiscalDateEnding", how="outer", suffixes=("","_cash"))

       # --- Basic margins (existing) ---
    df["grossMargin"]      = _safe_div(df["grossProfit"], df["totalRevenue"])
    df["operatingMargin"]  = _safe_div(df["operatingIncome"], df["totalRevenue"])
    df["netMargin"]        = _safe_div(df["netIncome"], df["totalRevenue"])

    # --- Liquidity / leverage (existing+safer) ---
    df["currentRatio"]     = _safe_div(df["totalCurrentAssets"], df["totalCurrentLiabilities"])
    df["debtToAssets"]     = _safe_div(df["longTermDebt"], df["totalAssets"])
    df["debtToEquity"]     = _safe_div(df["longTermDebt"], df.get("totalShareholderEquity", np.nan))

    # --- Cash & FCF (existing) ---
    df["freeCashFlow"]     = df["operatingCashflow"] - df["capitalExpenditures"]
    df["fcfMargin"]        = _safe_div(df["freeCashFlow"], df["totalRevenue"])
    cash_like = (df.get("cashAndCashEquivalentsAtCarryingValue", 0.0).fillna(0.0)
                 + df.get("shortTermInvestments", 0.0).fillna(0.0))
    df["cashPctAssets"]    = _safe_div(cash_like, df["totalAssets"])

    # --- Size / profitability / quality / investment (NEW) ---
    df["logRevenue"]       = _slog1p(df["totalRevenue"])
    df["logAssets"]        = _slog1p(df["totalAssets"])
    df["assetTurnover"]    = _safe_div(df["totalRevenue"], df["totalAssets"])          # efficiency
    df["ROA"]              = _safe_div(df["netIncome"], df["totalAssets"])             # profitability
    df["OCFtoNI"]          = _safe_div(df["operatingCashflow"], df["netIncome"])       # earnings quality
    df["AccrualsTA"]       = _safe_div((df["netIncome"] - df["operatingCashflow"]), df["totalAssets"])  # Sloan accruals
    df["AssetGrowthYoY"]   = df["totalAssets"].pct_change(4)                           # investment
    df["DebtChangeYoY"]    = df["longTermDebt"].pct_change(4)

    # --- Fundamental momentum (YoY/QoQ trends; existing + NEW) ---
    df = df.sort_values("fiscalDateEnding")
    df["revYoY"]           = df["totalRevenue"].pct_change(4)
    df["netIncomeYoY"]     = df["netIncome"].pct_change(4)
    df["opMarginYoY"]      = df["operatingMargin"].diff(4)
    df["netMarginYoY"]     = df["netMargin"].diff(4)
    df["grossMarginYoY"]   = df["grossMargin"].diff(4)           # NEW
    df["ROA_YoY"]          = df["ROA"].diff(4)                    # NEW

    # --- Rolling stability/volatility (per-ticker trailing; NEW) ---
    df = df.set_index("fiscalDateEnding")
    for col in ["grossMargin","operatingMargin","netMargin","ROA","fcfMargin","assetTurnover","debtToAssets","debtToEquity"]:
        r = df[col].rolling(8, min_periods=4)                     # ~2 years for quarterly
        df[f"{col}_roll8_mean"] = r.mean()
        df[f"{col}_roll8_std"]  = r.std()
    df = df.reset_index()

    # --- Calendar features (NEW) ---
    df["fiscalQuarter"]    = df["fiscalDateEnding"].dt.quarter
    df["fiscalYear"]       = df["fiscalDateEnding"].dt.year

    # --- Publication lag (existing) ---
    df["asof_date"]        = df["fiscalDateEnding"] + pd.to_timedelta(PUBLICATION_LAG_DAYS, unit="D")
    df["ticker"]           = ticker

    # Keep updated list
    keep = [
        "ticker","fiscalDateEnding","asof_date",
        # scale/size
        "totalRevenue","grossProfit","operatingIncome","netIncome","totalAssets","longTermDebt",
        "logRevenue","logAssets",
        # margins / liquidity / leverage
        "grossMargin","operatingMargin","netMargin","currentRatio","debtToAssets","debtToEquity",
        # cash / quality / investment / efficiency
        "freeCashFlow","fcfMargin","cashPctAssets","assetTurnover","ROA","OCFtoNI","AccrualsTA","AssetGrowthYoY","DebtChangeYoY",
        # momentum YoY
        "revYoY","netIncomeYoY","opMarginYoY","netMarginYoY","grossMarginYoY","ROA_YoY",
        # rolling stability
        "grossMargin_roll8_mean","grossMargin_roll8_std",
        "operatingMargin_roll8_mean","operatingMargin_roll8_std",
        "netMargin_roll8_mean","netMargin_roll8_std",
        "ROA_roll8_mean","ROA_roll8_std",
        "fcfMargin_roll8_mean","fcfMargin_roll8_std",
        "assetTurnover_roll8_mean","assetTurnover_roll8_std",
        "debtToAssets_roll8_mean","debtToAssets_roll8_std",
        "debtToEquity_roll8_mean","debtToEquity_roll8_std",
        # calendar
        "fiscalQuarter","fiscalYear",
    ]
    return df[keep]


def _list_tickers(raw_root: Path, snapshot: str) -> List[str]:
    # Primary layout: <raw_root>/<snapshot>/<TICKER>/...
    base = raw_root / snapshot
    # Fallback layout: <raw_root>/snapshot/<snapshot>/<TICKER>/... (some ingest paths use an extra 'snapshot' folder)
    if not base.exists():
        alt = raw_root / "snapshot" / snapshot
        if alt.exists():
            base = alt
    if not base.exists():
        return []
    out = []
    for d in base.iterdir():
        if d.is_dir() and any((d / f"{name}.json").exists() for name in ["income_statement","balance_sheet","cashflow_statement"]):
            out.append(d.name)
    return sorted(out)

def _load_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    data = yf.download(tickers=tickers, start=start, end=end, auto_adjust=False, progress=False, threads=True)

    if isinstance(data, pd.DataFrame) and isinstance(data.columns, pd.MultiIndex):
        adj = data["Adj Close"].copy()
    else:
        # Single ticker -> Series
        if isinstance(data, pd.Series):
            adj = data.to_frame(name=tickers[0])
        else:
            # Sometimes Yahoo still returns a DF; pull column safely
            col = "Adj Close" if "Adj Close" in data.columns else data.columns[0]
            adj = data[[col]].copy()
            adj.columns = [tickers[0]]

    return adj.dropna(how="all")

    """
    Returns a DataFrame with business-day index and columns = tickers, values = Adj Close.
    """
    if not tickers:
        return pd.DataFrame()
    data = yf.download(tickers=tickers, start=start, end=end, auto_adjust=False, progress=False, threads=True)
    # Yahoo returns multi-index if multiple tickers
    if isinstance(data.columns, pd.MultiIndex):
        adj = data["Adj Close"].copy()
    else:
        adj = data.to_frame()["Adj Close"].copy()
        adj.columns = [tickers[0]]
    adj = adj.dropna(how="all")
    return adj

def _next_trading_day(idx: pd.DatetimeIndex, dt: pd.Timestamp) -> Optional[pd.Timestamp]:
    # find first index >= dt
    loc = idx.searchsorted(dt)
    if loc >= len(idx): return None
    return idx[loc]

def _offset_trading_day(idx: pd.DatetimeIndex, dt: pd.Timestamp, offset: int) -> Optional[pd.Timestamp]:
    loc = idx.searchsorted(dt)
    tgt = loc + offset
    if tgt >= len(idx): return None
    return idx[tgt]

def build_dataset(snapshot: str, period: str) -> Path:
    """
    Builds dataset with features and 12m forward return label.
    Saves to OUT_DIR/<snapshot>/dataset_model_a.parquet
    """
    raw_root = RAW_DIR
    tickers = _list_tickers(raw_root, snapshot)
    if not tickers:
        raise RuntimeError(f"No tickers found under {raw_root / snapshot}")

    # 1) Build feature rows from fundamentals
    frames = []
    for t in tickers:
        stmts = _load_statements(raw_root, snapshot, t, period=period)
        fx = _feature_frame(t, stmts)
        if not fx.empty:
            frames.append(fx)
    feats = pd.concat(frames, ignore_index=True).dropna(subset=["asof_date"]).sort_values(["ticker","asof_date"])

    # 2) Load prices spanning full window (min asof to max asof + 400d)
    start = (feats["asof_date"].min() - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    end   = (feats["asof_date"].max() + pd.Timedelta(days=LABEL_HORIZON_TRADING_DAYS+7)).strftime("%Y-%m-%d")
    prices = _load_prices(sorted(feats["ticker"].unique()), start, end)
    if prices.empty:
        raise RuntimeError("Failed to load Yahoo prices.")
    


        # --- Price-based momentum & volatility up to as-of date (leakage-safe) ---
    # daily returns
    rets = prices.pct_change().dropna(how="all")

    def past_return(ticker: str, asof_dt: pd.Timestamp, days: int) -> float:
        t0 = _next_trading_day(prices.index, asof_dt)
        if t0 is None: return np.nan
        t_start = _offset_trading_day(prices.index, t0, -days)
        if t_start is None: return np.nan
        p0 = prices.at[t_start, ticker]
        p1 = prices.at[t0, ticker]
        if pd.isna(p0) or pd.isna(p1): return np.nan
        return float(p1) / float(p0) - 1.0

    def past_vol(ticker: str, asof_dt: pd.Timestamp, days: int) -> float:
        t0 = _next_trading_day(rets.index, asof_dt)
        if t0 is None: return np.nan
        t_start = _offset_trading_day(rets.index, t0, -days)
        if t_start is None: return np.nan
        r = rets.loc[t_start:t0, ticker].dropna()
        return float(r.std()) if len(r) else np.nan

    feats["mom_3m"]  = [past_return(t, a, 63)  for t, a in zip(feats["ticker"], feats["asof_date"])]
    feats["mom_6m"]  = [past_return(t, a, 126) for t, a in zip(feats["ticker"], feats["asof_date"])]
    feats["mom_12m"] = [past_return(t, a, 252) for t, a in zip(feats["ticker"], feats["asof_date"])]
    feats["vol_3m"]  = [past_vol(t, a, 63)     for t, a in zip(feats["ticker"], feats["asof_date"])]
    feats["vol_6m"]  = [past_vol(t, a, 126)    for t, a in zip(feats["ticker"], feats["asof_date"])]

    # 3) Label each row with 12m forward total return using Adjusted Close (total-return proxy)
    labels = []
    trade_index = prices.index
    for i, row in feats.iterrows():
        t = row["ticker"]
        asof = pd.Timestamp(row["asof_date"]).tz_localize(None)
        t0 = _next_trading_day(trade_index, asof)
        if t0 is None:
            labels.append(np.nan); continue
        t1 = _offset_trading_day(trade_index, t0, LABEL_HORIZON_TRADING_DAYS)
        if t1 is None:
            labels.append(np.nan); continue
        p0 = prices.at[t0, t]
        p1 = prices.at[t1, t]
        if pd.isna(p0) or pd.isna(p1):
            labels.append(np.nan); continue
        r = float(p1) / float(p0) - 1.0
        labels.append(r)


    # Light winsorization to tame rare outliers (keeps quantile objective stable)
    def winsorize(s: pd.Series, p=0.005):
        lo, hi = s.quantile(p), s.quantile(1-p)
        return s.clip(lo, hi)

    for col in ["ROA","AccrualsTA","AssetGrowthYoY","DebtChangeYoY","fcfMargin","currentRatio",
                "debtToAssets","debtToEquity","mom_12m","mom_6m","mom_3m","vol_6m","vol_3m"]:
        if col in feats.columns:
            feats[col] = winsorize(feats[col])


    feats["y_12m"] = labels
    feats = feats.dropna(subset=["y_12m"]).reset_index(drop=True)

    out_dir = OUT_DIR / snapshot
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dataset_model_a.parquet"
    feats.to_parquet(out_path, index=False)
    return out_path

def _feature_target_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    target = df["y_12m"].astype(float)
    # Feature columns (exclude IDs and target)
    drop = {"ticker","fiscalDateEnding","asof_date","y_12m"}
    X = df[[c for c in df.columns if c not in drop]].copy()
    return X, target


def _clean_input_df(X: pd.DataFrame) -> pd.DataFrame:
    """Sanitize input feature frame before passing to sklearn:
    - replace +/-inf with NaN
    - coerce object columns that look numeric to numeric (safe)
    - mask extremely large values (>1e300) to NaN to avoid float overflow in validation
    """
    Xc = X.copy()
    # Replace infinities in numeric columns
    num_cols = Xc.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        Xc[num_cols] = Xc[num_cols].replace([np.inf, -np.inf], np.nan)
        # Mask extremely large magnitudes (rare) which can break float checks
        too_large = Xc[num_cols].abs() > 1e300
        if too_large.any().any():
            Xc.loc[:, num_cols] = Xc.loc[:, num_cols].mask(too_large, np.nan)

    # Try to coerce object columns that hold numeric strings
    obj_cols = Xc.select_dtypes(include=["object"]).columns.tolist()
    for c in obj_cols:
        coerced = pd.to_numeric(Xc[c], errors="coerce")
        # If coercion produced some non-NaN values, assume column is numeric-like
        if coerced.notna().any():
            Xc[c] = coerced

    return Xc

def _make_pipeline(feature_names: List[str]) -> Pipeline:
    # SimpleImputer(median) is sufficient for tree models; keep raw scale.
    pre = ColumnTransformer(
        transformers=[("num", SimpleImputer(strategy="median"), feature_names)],
        remainder="drop",
    )
    pipe = Pipeline([("pre", pre)])  # lightgbm will be fit separately on transformed arrays
    return pipe

def train_models(snapshot: str) -> Dict[str, str]:
    ds_path = OUT_DIR / snapshot / "dataset_model_a.parquet"
    if not ds_path.exists():
        raise RuntimeError(f"Dataset not found: {ds_path}. Run build-dataset first.")
    df = pd.read_parquet(ds_path)
    if df.shape[0] < 200:
        print(f"[WARN] Very small dataset ({df.shape[0]} rows). Models will be noisy but we proceed.")

    # Chronological split
    df = df.sort_values("asof_date").reset_index(drop=True)
    X, y = _feature_target_split(df)
    feature_names = list(X.columns)

    pre = _make_pipeline(feature_names)
    # Clean input (replace inf/-inf, mask absurdly large values, coerce numeric-like objects)
    X_clean = _clean_input_df(X)
    # Report small diagnostics for debugging
    # (counts of non-finite values per column)
    try:
        nonfinite_counts = X_clean.select_dtypes(include=[np.number]).isna().sum()
        bad = nonfinite_counts[nonfinite_counts > 0]
        if not bad.empty:
            print(f"[INFO] Columns with missing/non-finite after cleaning (showing up to 10): {bad.head(10).to_dict()}")
    except Exception:
        pass

    Xn = pre.fit_transform(X_clean)

    cutoff = int(0.8 * Xn.shape[0])
    X_tr, y_tr = Xn[:cutoff], y.values[:cutoff].astype(float)
    X_va, y_va = Xn[cutoff:], y.values[cutoff:].astype(float)

    models_paths = {}
    save_dir = MODELS_DIR / snapshot
    save_dir.mkdir(parents=True, exist_ok=True)

    base_params = {
        "objective": "quantile",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "min_data_in_leaf": 128,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 1.0,
        "lambda_l2": 5.0,
        "verbosity": -1,
        "seed": RANDOM_STATE,
        "metric": "quantile",
    }

    for tau in QUANTILES:
        params = dict(base_params, alpha=float(tau))
        dtr = lgb.Dataset(X_tr, label=y_tr)
        dva = lgb.Dataset(X_va, label=y_va, reference=dtr)

        booster = lgb.train(
            params, dtr, num_boost_round=5000,
            valid_sets=[dtr, dva], valid_names=["train","valid"],
            callbacks=[lgb.early_stopping(stopping_rounds=300), lgb.log_evaluation(period=0)],
        )
        best_rounds = booster.best_iteration or 1200

        # Refit on full
        dfull = lgb.Dataset(Xn, label=y.values.astype(float))
        model = lgb.train(params, dfull, num_boost_round=best_rounds, callbacks=[lgb.log_evaluation(period=0)])

        out_path = save_dir / f"lgbm_q{int(tau*100)}.txt"
        model.save_model(str(out_path))
        models_paths[f"{tau:.2f}"] = str(out_path)

    dump(pre, save_dir / "preprocessor.joblib")
    meta = {
        "snapshot": snapshot,
        "created_at": datetime.utcnow().isoformat(),
        "label_horizon_trading_days": LABEL_HORIZON_TRADING_DAYS,
        "publication_lag_days": PUBLICATION_LAG_DAYS,
        "quantiles": QUANTILES,
        "n_rows": int(df.shape[0]),
        "features": feature_names,
    }
    (save_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved models to {save_dir}")
    return models_paths

    """
    Train 5 LightGBM quantile models. Saves models and metadata.
    Returns dict of tau->model_path.
    """
    ds_path = OUT_DIR / snapshot / "dataset_model_a.parquet"
    if not ds_path.exists():
        raise RuntimeError(f"Dataset not found: {ds_path}. Run build-dataset first.")
    df = pd.read_parquet(ds_path)
    if df.shape[0] < 200:
        print(f"[WARN] Very small dataset ({df.shape[0]} rows). Models will be noisy but we proceed.")

    # Time split: use 'asof_date' for chronological split
    df = df.sort_values("asof_date").reset_index(drop=True)
    X, y = _feature_target_split(df)
    feature_names = list(X.columns)

    # Preprocess (impute)
    pre = _make_pipeline(feature_names)
    Xn = pre.fit_transform(X)

    # simple time split: last 20% as validation (we still train models on full after basic sanity)
    cutoff = int(0.8 * Xn.shape[0])
    X_train, y_train = Xn[:cutoff], y[:cutoff]
    X_full,  y_full  = Xn, y

    models_paths = {}
    save_dir = MODELS_DIR / snapshot
    save_dir.mkdir(parents=True, exist_ok=True)

        # Time-aware split on asof_date (last 20% = validation)
    df = df.sort_values("asof_date").reset_index(drop=True)
    X, y = _feature_target_split(df)
    feature_names = list(X.columns)

    # Preprocess (impute)
    pre = _make_pipeline(feature_names)
    Xn = pre.fit_transform(X)

    cutoff = int(0.8 * Xn.shape[0])
    X_tr, y_tr = Xn[:cutoff], y.values[:cutoff].astype(float)
    X_va, y_va = Xn[cutoff:], y.values[cutoff:].astype(float)

    models_paths = {}
    save_dir = MODELS_DIR / snapshot
    save_dir.mkdir(parents=True, exist_ok=True)

    base_params = {
        "objective": "quantile",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "min_data_in_leaf": 128,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 1.0,
        "lambda_l2": 5.0,
        "verbosity": -1,
        "seed": RANDOM_STATE,
        "metric": "quantile",
    }

    for tau in QUANTILES:
        params = dict(base_params)
        params["alpha"] = float(tau)

        dtr = lgb.Dataset(X_tr, label=y_tr)
        dva = lgb.Dataset(X_va, label=y_va, reference=dtr)

        # Train with early stopping on the time-aware validation
        booster = lgb.train(
            params, dtr, num_boost_round=5000,
            valid_sets=[dtr, dva],
            valid_names=["train","valid"],
            callbacks=[lgb.early_stopping(stopping_rounds=300), lgb.log_evaluation(period=0)],
        )
        best_rounds = booster.best_iteration or 1200

        # Refit on full data with best rounds
        dfull = lgb.Dataset(Xn, label=y.values.astype(float))
        model = lgb.train(params, dfull, num_boost_round=best_rounds, callbacks=[lgb.log_evaluation(period=0)])

        out_path = save_dir / f"lgbm_q{int(tau*100)}.txt"
        model.save_model(str(out_path))
        models_paths[f"{tau:.2f}"] = str(out_path)



    # Train each quantile model
    for tau in QUANTILES:
        params = {
            "objective": "quantile",
            "alpha": tau,
            "metric": "quantile",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "verbosity": -1,
            "seed": RANDOM_STATE,
        }
        dtrain = lgb.Dataset(X_full, label=y_full.values.astype(float))
        model = lgb.train(params, dtrain, num_boost_round=1200)
        out_path = save_dir / f"lgbm_q{int(tau*100)}.txt"
        model.save_model(str(out_path))
        models_paths[f"{tau:.2f}"] = str(out_path)

    # Save preprocessing and metadata
    dump(pre, save_dir / "preprocessor.joblib")
    meta = {
        "snapshot": snapshot,
        "created_at": datetime.utcnow().isoformat(),
        "label_horizon_trading_days": LABEL_HORIZON_TRADING_DAYS,
        "publication_lag_days": PUBLICATION_LAG_DAYS,
        "quantiles": QUANTILES,
        "n_rows": int(df.shape[0]),
        "features": feature_names,
    }
    (save_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved models to {save_dir}")
    return models_paths

def _enforce_monotonic_quantiles(df: pd.DataFrame, qcols=("q5","q25","q50","q75","q95")) -> pd.DataFrame:
    # Ensure q5 ≤ q25 ≤ q50 ≤ q75 ≤ q95 row-wise
    qarr = df.loc[:, qcols].to_numpy(copy=True)
    qarr.sort(axis=1)
    df.loc[:, qcols] = qarr
    return df

def predict_latest(snapshot: str, tickers: List[str], period: str) -> pd.DataFrame:
    """
    Build features for the latest quarter per ticker (in the snapshot),
    load models, and emit predicted quantiles (+ helpful uncertainty bands).
    """
    save_dir = MODELS_DIR / snapshot
    meta_path = save_dir / "metadata.json"
    pre_path  = save_dir / "preprocessor.joblib"
    if not meta_path.exists() or not pre_path.exists():
        raise RuntimeError(f"Models for snapshot {snapshot} not found. Train first.")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    pre = load(pre_path)
    feature_names: List[str] = meta["features"]

    # Build last-row feature set per ticker
    rows = []
    for t in tickers:
        stmts = _load_statements(RAW_DIR, snapshot, t, period)
        fx = _feature_frame(t, stmts)
        if fx.empty:
            continue
        last = fx.sort_values("asof_date").iloc[-1]
        rows.append(last)

    if not rows:
        raise RuntimeError("No features available for requested tickers.")

    F = pd.DataFrame(rows).reset_index(drop=True)

    # --- Add price-derived features (moments & vol) to match training dataset ---
    try:
        # Load prices covering the needed lookback window for all asof dates
        start = (F["asof_date"].min() - pd.Timedelta(days=300)).strftime("%Y-%m-%d")
        end = (F["asof_date"].max() + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        prices = _load_prices(sorted(F["ticker"].unique()), start, end)
        if prices.empty:
            # If we couldn't fetch prices, create NaN columns so pipeline can impute
            for c in ["mom_3m", "mom_6m", "mom_12m", "vol_3m", "vol_6m"]:
                F[c] = np.nan
        else:
            rets = prices.pct_change().dropna(how="all")

            def past_return(ticker: str, asof_dt: pd.Timestamp, days: int) -> float:
                t0 = _next_trading_day(prices.index, asof_dt)
                if t0 is None: return np.nan
                t_start = _offset_trading_day(prices.index, t0, -days)
                if t_start is None: return np.nan
                p0 = prices.at[t_start, ticker] if ticker in prices.columns else np.nan
                p1 = prices.at[t0, ticker] if ticker in prices.columns else np.nan
                if pd.isna(p0) or pd.isna(p1): return np.nan
                return float(p1) / float(p0) - 1.0

            def past_vol(ticker: str, asof_dt: pd.Timestamp, days: int) -> float:
                t0 = _next_trading_day(rets.index, asof_dt)
                if t0 is None: return np.nan
                t_start = _offset_trading_day(rets.index, t0, -days)
                if t_start is None: return np.nan
                r = rets.loc[t_start:t0, ticker].dropna() if ticker in rets.columns else pd.Series(dtype=float)
                return float(r.std()) if len(r) else np.nan

            F["mom_3m"] = [past_return(t, a, 63) for t, a in zip(F["ticker"], F["asof_date"]) ]
            F["mom_6m"] = [past_return(t, a, 126) for t, a in zip(F["ticker"], F["asof_date"]) ]
            F["mom_12m"] = [past_return(t, a, 252) for t, a in zip(F["ticker"], F["asof_date"]) ]
            F["vol_3m"] = [past_vol(t, a, 63) for t, a in zip(F["ticker"], F["asof_date"]) ]
            F["vol_6m"] = [past_vol(t, a, 126) for t, a in zip(F["ticker"], F["asof_date"]) ]
    except Exception:
        # On any failure, add the columns as NaN so preprocessor can still run
        for c in ["mom_3m", "mom_6m", "mom_12m", "vol_3m", "vol_6m"]:
            if c not in F.columns:
                F[c] = np.nan

    # Ensure feature order exactly matches training
    missing = [c for c in feature_names if c not in F.columns]
    if missing:
        raise RuntimeError(f"Missing required features in latest frame: {missing[:10]}{'...' if len(missing)>10 else ''}")

    X = F[feature_names].copy()
    X_clean = _clean_input_df(X)
    # If any non-finite remain in numeric columns, show counts to help debugging
    try:
        nf = (~np.isfinite(X_clean.select_dtypes(include=[np.number]).to_numpy())).sum(axis=0)
        cols = X_clean.select_dtypes(include=[np.number]).columns
        bad = {c: int(n) for c, n in zip(cols, nf) if n > 0}
        if bad:
            print(f"[WARN] Non-finite values detected in latest features prior to transform: {dict(list(bad.items())[:10])}")
    except Exception:
        pass

    Xn = pre.transform(X_clean)

    # Load each quantile model and predict
    preds: Dict[str, np.ndarray] = {}
    needed = {5, 25, 50, 75, 95}
    for q in sorted(needed):
        model_path = save_dir / f"lgbm_q{q}.txt"
        if not model_path.exists():
            raise RuntimeError(f"Missing model file: {model_path.name}")
        model = lgb.Booster(model_file=str(model_path))
        preds[f"q{q}"] = model.predict(Xn)

    out = F[["ticker","asof_date"]].copy()
    # Attach in canonical order
    for k in ["q5","q25","q50","q75","q95"]:
        out[k] = preds[k]

    # Enforce monotonic quantiles (prevents crossings from separate models)
    out = _enforce_monotonic_quantiles(out, ("q5","q25","q50","q75","q95"))

    # Add uncertainty bands for quick inspection
    out["iqr_width"]   = out["q75"] - out["q25"]         # 50% interval
    out["p90_width"]   = out["q95"] - out["q5"]          # 90% interval
    out["median"]      = out["q50"]                      # alias
    out["asof_date"]   = pd.to_datetime(out["asof_date"])  # ensure dtype

    # Optional: clip if your target is bounded (example: returns ≥ -1)
    # out[["q5","q25","q50","q75","q95"]] = out[["q5","q25","q50","q75","q95"]].clip(lower=-1.0)

    print((np.diff(out.loc[out["ticker"]=="MSFT", ["q5","q25","q50","q75","q95"]].values) >= 0).all())


    return out[["ticker","asof_date","q5","q25","q50","q75","q95","iqr_width","p90_width","median"]]


# ------------------------------
# CLI
# ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Model A (Quantile GBM): build dataset, train, predict.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-dataset", help="Build dataset from raw fundamentals + Yahoo labels")
    b.add_argument("--snapshot", required=True, help="Snapshot folder under data/raw/")
    b.add_argument("--period", choices=["quarterly","annual"], default="quarterly")

    t = sub.add_parser("train", help="Train quantile GBM models on built dataset")
    t.add_argument("--snapshot", required=True)

    p = sub.add_parser("predict", help="Predict latest quantiles for tickers")
    p.add_argument("--snapshot", required=True)
    p.add_argument("--period", choices=["quarterly","annual"], default="quarterly")
    p.add_argument("--tickers", nargs="+", required=True)

    args = ap.parse_args()

    if args.cmd == "build-dataset":
        out = build_dataset(snapshot=args.snapshot, period=args.period)
        print(f"Dataset written: {out}")

    elif args.cmd == "train":
        train_models(snapshot=args.snapshot)

    elif args.cmd == "predict":
        df = predict_latest(snapshot=args.snapshot, tickers=[t.upper() for t in args.tickers], period=args.period)
        print(df.to_string(index=False))

if __name__ == "__main__":
    main()
