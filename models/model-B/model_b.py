#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model B: Quantile Gradient Boosting (LightGBM) with richer features + diagnostics
- Builds a leakage-safe dataset from raw fundamentals JSON (FMP / AlphaVantage) + price momentum/vol.
- Labels with 12m forward total return via Yahoo Finance (Adj Close).
- Trains 5 quantile models (5/25/50/75/95th) and emits training diagnostics/reports.
- Predicts quantiles for latest fundamentals per ticker.

Folder assumptions (from your ingestion):
  data/raw/<SNAPSHOT>/<TICKER>/{income_statement,balance_sheet,cashflow_statement,ratios,key_metrics,enterprise_values}.json

Commands:
  python model_b.py build-dataset --snapshot 20251022 --period quarterly
  python model_b.py train --snapshot 20251022
  python model_b.py predict --snapshot 20251022 --tickers AAPL MSFT TSLA
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


# -------- Pretty printing helpers --------
def _print_rule(char="─", width=72):
    print(char * width)

def _print_title(title: str, width=72):
    _print_rule("═", width)
    print(f"{title}".center(width))
    _print_rule("═", width)

def _print_subtitle(title: str, width=72):
    print(f"\n{title}")
    _print_rule("─", width)

def _to_table(df: pd.DataFrame, max_rows=12, max_cols=8):
    with pd.option_context("display.max_rows", max_rows, "display.max_columns", max_cols, "display.width", 140):
        print(df.to_string(index=False))

def _safe_spearman(df: pd.DataFrame, target_col: str, cols: list[str], top_k=12):
    out = []
    y = df[target_col].astype(float)
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() > 20:
            rho = s.corr(y, method="spearman")
            if pd.notna(rho):
                out.append((c, float(rho)))
    res = (pd.DataFrame(out, columns=["feature","spearman_rho"])
             .sort_values("spearman_rho", key=lambda s: s.abs(), ascending=False)
             .head(top_k))
    return res

def _ensure_reports_dir(snapshot: str) -> Path:
    rep = MODELS_DIR / snapshot / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    return rep

# Pinball loss for quantile tau
def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> float:
    e = y_true - y_pred
    return float(np.mean(np.maximum(tau*e, (tau-1)*e)))


# ------------------------------
# Config
# ------------------------------
RAW_DIR = Path(os.environ.get("RAW_DIR", "data/raw"))
OUT_DIR = Path("data/structured")
MODELS_DIR = Path("models/model-B")
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

    # Ensure critical columns exist
    needed = [
        "totalRevenue","grossProfit","operatingIncome","netIncome","operatingExpenses",
        "totalAssets","totalLiabilities","totalShareholderEquity","totalCurrentAssets","totalCurrentLiabilities","longTermDebt",
        "operatingCashflow","capitalExpenditures"
    ]
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan


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
    Saves to OUT_DIR/<snapshot>/dataset_model_b.parquet
    Also prints dataset diagnostics and writes CSV reports.
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

    # 3) Label with 12m forward return
    labels = []
    trade_index = prices.index
    for _, row in feats.iterrows():
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
        labels.append(float(p1) / float(p0) - 1.0)

    # Light winsorization
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
    out_path = out_dir / "dataset_model_b.parquet"
    feats.to_parquet(out_path, index=False)

    # --------- Diagnostics (console + CSV) ----------
    rep = _ensure_reports_dir(snapshot)

    _print_title(f"DATASET SUMMARY — snapshot {snapshot}")
    # Table 1: basic snapshot stats
    snap_tbl = pd.DataFrame([{
        "rows": int(len(feats)),
        "tickers": int(feats["ticker"].nunique()),
        "asof_min": feats["asof_date"].min().date(),
        "asof_max": feats["asof_date"].max().date(),
        "period": period,
        "label_horizon_days": LABEL_HORIZON_TRADING_DAYS
    }])
    _print_subtitle("Snapshot")
    _to_table(snap_tbl)
    snap_tbl.to_csv(rep / "dataset_snapshot.csv", index=False)

    # Table 2: label distribution
    y = feats["y_12m"].astype(float)
    label_tbl = pd.DataFrame([{
        "mean": y.mean(), "std": y.std(), "median": y.median(),
        "p5": y.quantile(0.05), "p25": y.quantile(0.25), "p75": y.quantile(0.75), "p95": y.quantile(0.95),
        "share_neg": (y < 0).mean(), "share_<=-20%": (y <= -0.20).mean(), "share_>=+20%": (y >= 0.20).mean()
    }])
    _print_subtitle("12-Month Forward Return (Label) — Distribution")
    _to_table(label_tbl)
    label_tbl.to_csv(rep / "label_distribution.csv", index=False)

    # Table 3: top missing features
    miss = feats.drop(columns=["ticker","fiscalDateEnding","asof_date","y_12m"]).isna().mean().sort_values(ascending=False)
    miss_tbl = miss.head(20).reset_index()
    miss_tbl.columns = ["feature","missing_rate"]
    _print_subtitle("Top Missing Features")
    _to_table(miss_tbl)
    miss_tbl.to_csv(rep / "feature_missing_rates.csv", index=False)

    # Table 4: quick correlations (Spearman) of select features vs label
    cand = [c for c in feats.columns if c not in {"ticker","fiscalDateEnding","asof_date","y_12m"}]
    corr_tbl = _safe_spearman(feats, "y_12m", cand, top_k=15)
    _print_subtitle("Top |Spearman| Correlations to Target")
    _to_table(corr_tbl)
    corr_tbl.to_csv(rep / "spearman_target_corr.csv", index=False)

    _print_rule("═")
    print(f"Dataset written: {out_path}\nReports: {rep}")
    _print_rule("═")

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
    ds_path = OUT_DIR / snapshot / "dataset_model_b.parquet"
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
    X_clean = _clean_input_df(X)
    Xn = pre.fit_transform(X_clean)

    cutoff = int(0.8 * Xn.shape[0])
    X_tr, y_tr = Xn[:cutoff], y.values[:cutoff].astype(float)
    X_va, y_va = Xn[cutoff:], y.values[cutoff:].astype(float)

    models_paths = {}
    save_dir = MODELS_DIR / snapshot
    save_dir.mkdir(parents=True, exist_ok=True)
    rep = _ensure_reports_dir(snapshot)

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

    # We’ll store validation predictions from early-stopped boosters to compute metrics
    val_preds = {}

    _print_title(f"TRAINING — snapshot {snapshot}")
    for tau in QUANTILES:
        params = dict(base_params, alpha=float(tau))
        dtr = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names)
        dva = lgb.Dataset(X_va, label=y_va, reference=dtr, feature_name=feature_names)

        booster = lgb.train(
            params, dtr, num_boost_round=5000,
            valid_sets=[dtr, dva], valid_names=["train","valid"],
            callbacks=[lgb.early_stopping(stopping_rounds=300), lgb.log_evaluation(period=0)],
        )
        best_rounds = booster.best_iteration or 1200

        # Validation prediction from early-stopped model
        val_preds[tau] = booster.predict(X_va)

        # Refit on full
        dfull = lgb.Dataset(Xn, label=y.values.astype(float), feature_name=feature_names)
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

    # ---------- Validation Metrics (Tables) ----------
    # 1) Pinball losses
    rows = []
    for tau in QUANTILES:
        pl = _pinball_loss(y_va, val_preds[tau], float(tau))
        rows.append({"quantile": tau, "pinball_loss": pl})
    pin_tbl = pd.DataFrame(rows).sort_values("quantile")
    _print_subtitle("Validation — Pinball Loss by Quantile")
    _to_table(pin_tbl)
    pin_tbl.to_csv(rep / "validation_pinball_loss.csv", index=False)

    # 2) Interval coverage
    # Build q5..q95 arrays from collected preds
    qmap = {int(t*100): val_preds[t] for t in QUANTILES}
    cov_50 = np.mean((y_va >= qmap[25]) & (y_va <= qmap[75])) if 25 in qmap and 75 in qmap else np.nan
    cov_90 = np.mean((y_va >= qmap[5])  & (y_va <= qmap[95])) if 5 in qmap  and 95 in qmap else np.nan
    cov_tbl = pd.DataFrame([{"interval":"[25,75] (50%)", "coverage": cov_50},
                            {"interval":"[5,95] (90%)",  "coverage": cov_90}])
    _print_subtitle("Validation — Interval Coverage")
    _to_table(cov_tbl)
    cov_tbl.to_csv(rep / "validation_interval_coverage.csv", index=False)

    # 3) Feature importance (using full models)
    imp_rows = []
    for q in [5,25,50,75,95]:
        model = lgb.Booster(model_file=str(save_dir / f"lgbm_q{q}.txt"))
        gains = model.feature_importance(importance_type="gain")
        imp_df = pd.DataFrame({"feature": feature_names, "gain": gains})
        imp_df["quantile"] = q
        imp_rows.append(imp_df)
    imp_all = pd.concat(imp_rows, ignore_index=True)
    imp_agg = (imp_all.groupby("feature", as_index=False)
                      .agg(total_gain=("gain","sum"), avg_gain=("gain","mean"),
                           q_count=("gain","count"))
                      .sort_values("total_gain", ascending=False))
    top_imp = imp_agg.head(25)
    _print_subtitle("Feature Importance (Gain) — Aggregated Across Quantiles (Top 25)")
    _to_table(top_imp[["feature","total_gain","avg_gain","q_count"]])
    imp_all.to_csv(rep / "feature_importance_all.csv", index=False)
    imp_agg.to_csv(rep / "feature_importance_agg.csv", index=False)

    _print_rule("═")
    print(f"Saved models to {save_dir}\nReports: {rep}")
    _print_rule("═")
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
    load models, and emit predicted quantiles (+ uncertainty bands).
    Prints per-ticker table and ranked summary; saves CSVs.
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

    # --- Add price-derived features to match training ---
    try:
        start = (F["asof_date"].min() - pd.Timedelta(days=300)).strftime("%Y-%m-%d")
        end = (F["asof_date"].max() + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        prices = _load_prices(sorted(F["ticker"].unique()), start, end)
        if prices.empty:
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

            F["mom_3m"]  = [past_return(t, a, 63)  for t, a in zip(F["ticker"], F["asof_date"])]
            F["mom_6m"]  = [past_return(t, a, 126) for t, a in zip(F["ticker"], F["asof_date"])]
            F["mom_12m"] = [past_return(t, a, 252) for t, a in zip(F["ticker"], F["asof_date"])]
            F["vol_3m"]  = [past_vol(t, a, 63)     for t, a in zip(F["ticker"], F["asof_date"])]
            F["vol_6m"]  = [past_vol(t, a, 126)    for t, a in zip(F["ticker"], F["asof_date"])]
    except Exception:
        for c in ["mom_3m", "mom_6m", "mom_12m", "vol_3m", "vol_6m"]:
            if c not in F.columns:
                F[c] = np.nan

    missing = [c for c in feature_names if c not in F.columns]
    if missing:
        raise RuntimeError(f"Missing required features in latest frame: {missing[:10]}{'...' if len(missing)>10 else ''}")

    X = F[feature_names].copy()
    X_clean = _clean_input_df(X)
    Xn = pre.transform(X_clean)

    # Load quantile models and predict
    preds: Dict[str, np.ndarray] = {}
    needed = {5, 25, 50, 75, 95}
    for q in sorted(needed):
        model_path = save_dir / f"lgbm_q{q}.txt"
        if not model_path.exists():
            raise RuntimeError(f"Missing model file: {model_path.name}")
        model = lgb.Booster(model_file=str(model_path))
        preds[f"q{q}"] = model.predict(Xn)

    out = F[["ticker","asof_date"]].copy()
    for k in ["q5","q25","q50","q75","q95"]:
        out[k] = preds[k]
    out = _enforce_monotonic_quantiles(out, ("q5","q25","q50","q75","q95"))
    out["iqr_width"] = out["q75"] - out["q25"]
    out["p90_width"] = out["q95"] - out["q5"]
    out["median"]    = out["q50"]
    out["asof_date"] = pd.to_datetime(out["asof_date"])

    # Console tables
    _print_title(f"PREDICTIONS — snapshot {snapshot}")
    _print_subtitle("Per-Ticker Quantile Prediction (12m Ret)")
    show_cols = ["ticker","asof_date","q5","q25","q50","q75","q95","iqr_width","p90_width"]
    _to_table(out[show_cols].sort_values("ticker"))

    ranked = (out[["ticker","median","iqr_width","p90_width"]]
                .sort_values(["median","iqr_width"], ascending=[False, True])
                .reset_index(drop=True))

    _print_rule("─")
    _print_subtitle("Top 5 Ranked by Median Return (Risk-Adjusted)")
    _to_table(ranked.head(5))
    
    # Save
    out_path = save_dir / "predictions_latest.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved predictions to: {out_path}")
    return out


# ==============================================================================
# Public API for Dashboard / Simulations (Cached Loading & Inference)
# ==============================================================================

def load_resources(snapshot: str) -> Dict[str, Any]:
    """
    Loads all model artifacts once for high-performance inference in Streamlit.
    Returns a dict with: 'meta', 'preprocessor', 'models' (dict of boosters).
    """
    save_dir = MODELS_DIR / snapshot
    meta_path = save_dir / "metadata.json"
    pre_path  = save_dir / "preprocessor.joblib"
    
    if not meta_path.exists() or not pre_path.exists():
        raise RuntimeError(f"Artifacts not found for snapshot {snapshot}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    pre = load(pre_path)
    
    models = {}
    needed = {5, 25, 50, 75, 95}
    for q in needed:
        model_path = save_dir / f"lgbm_q{q}.txt"
        if not model_path.exists():
            continue
        models[q] = lgb.Booster(model_file=str(model_path))
        
    return {
        "meta": meta,
        "preprocessor": pre,
        "models": models,
        "feature_names": meta["features"]
    }

def predict_on_features(resources: Dict[str, Any], X: pd.DataFrame) -> pd.DataFrame:
    """
    Run inference on a custom feature set X using loaded resources.
    X must contain columns matching resources['feature_names'].
    Returns DataFrame with q5, q25, q50, q75, q95 columns.
    """
    pre = resources["preprocessor"]
    models = resources["models"]
    feats = resources["feature_names"]
    
    # Ensure alignment
    # If X has extra columns, ignore them. If missing, this will error (correctly).
    X_in = X[feats].copy()
    
    # Preprocessing (clean + transform)
    X_clean = _clean_input_df(X_in)
    Xn = pre.transform(X_clean)
    
    # Predict
    out = pd.DataFrame(index=X.index)
    for q, booster in models.items():
        out[f"q{q}"] = booster.predict(Xn)
        
    # Enforce monotonicity
    cols = [f"q{q}" for q in sorted(models.keys())]
    if len(cols) > 1:
        arr = out[cols].values
        arr.sort(axis=1)
        out[cols] = arr
        
    return out

    # Quality flags
    iq_med = ranked["iqr_width"].median()
    ranked["uncertainty_flag"] = np.where(ranked["iqr_width"] <= 0.5*iq_med, "narrow",
                                 np.where(ranked["iqr_width"] >= 1.5*iq_med, "wide", "normal"))
    _print_subtitle("Ranked Summary (by median; narrower IQR preferred)")
    _to_table(ranked.head(20))

    # Save CSVs
    rep = _ensure_reports_dir(snapshot)
    out.to_csv(rep / "predict_latest_full.csv", index=False)
    ranked.to_csv(rep / "predict_ranked.csv", index=False)

    # quick monotonicity sanity print for MSFT if present
    try:
        print("\nMonotone check (MSFT):", (np.diff(out.loc[out["ticker"]=="MSFT", ["q5","q25","q50","q75","q95"]].values) >= 0).all())
    except Exception:
        pass

    return out[["ticker","asof_date","q5","q25","q50","q75","q95","iqr_width","p90_width","median"]]


# ------------------------------
# CLI
# ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Model B (Quantile GBM + diagnostics): build dataset, train, predict.")
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
