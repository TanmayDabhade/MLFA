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

    # Basic margins
    df["grossMargin"]      = df["grossProfit"] / df["totalRevenue"]
    df["operatingMargin"]  = df["operatingIncome"] / df["totalRevenue"]
    df["netMargin"]        = df["netIncome"] / df["totalRevenue"]

    # Liquidity / leverage
    df["currentRatio"]     = df["totalCurrentAssets"] / df["totalCurrentLiabilities"]
    df["debtToAssets"]     = df["longTermDebt"] / df["totalAssets"]

    # Cash & FCF
    df["freeCashFlow"]     = df["operatingCashflow"] - df["capitalExpenditures"]
    df["fcfMargin"]        = df["freeCashFlow"] / df["totalRevenue"]
    df["cashPctAssets"]    = (df["cashAndCashEquivalentsAtCarryingValue"].fillna(0.0) + df["shortTermInvestments"].fillna(0.0)) / df["totalAssets"]

    # YoY changes (quarterly assumes lag 4)
    df = df.sort_values("fiscalDateEnding")
    df["revYoY"]           = df["totalRevenue"] .pct_change(4)
    df["netIncomeYoY"]     = df["netIncome"]    .pct_change(4)
    df["opMarginYoY"]      = df["operatingMargin"] .diff(4)
    df["netMarginYoY"]     = df["netMargin"]       .diff(4)

    # Publication (as-of) date with lag to avoid leakage
    df["asof_date"]        = df["fiscalDateEnding"] + pd.to_timedelta(PUBLICATION_LAG_DAYS, unit="D")
    df["ticker"]           = ticker

    # Keep only columns we want as features + keys
    keep = [
        "ticker","fiscalDateEnding","asof_date",
        "totalRevenue","grossProfit","operatingIncome","netIncome",
        "grossMargin","operatingMargin","netMargin",
        "currentRatio","debtToAssets",
        "freeCashFlow","fcfMargin","cashPctAssets",
        "revYoY","netIncomeYoY","opMarginYoY","netMarginYoY",
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

def _make_pipeline(feature_names: List[str]) -> Pipeline:
    # SimpleImputer(median) is sufficient for tree models; keep raw scale.
    pre = ColumnTransformer(
        transformers=[("num", SimpleImputer(strategy="median"), feature_names)],
        remainder="drop",
    )
    pipe = Pipeline([("pre", pre)])  # lightgbm will be fit separately on transformed arrays
    return pipe

def train_models(snapshot: str) -> Dict[str, str]:
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

    # Ensure feature order exactly matches training
    missing = [c for c in feature_names if c not in F.columns]
    if missing:
        raise RuntimeError(f"Missing required features in latest frame: {missing[:10]}{'...' if len(missing)>10 else ''}")

    X = F[feature_names].copy()
    Xn = pre.transform(X)

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
