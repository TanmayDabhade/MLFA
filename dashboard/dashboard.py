# streamlit_app.py
# V1 finance fundamentals viewer for AAPL/MSFT/TSLA (or any tickers you’ve collected)
# Directory layout expected: data/raw/<SNAPSHOT>/<TICKER>/<endpoint>.json
# Endpoints: income_statement.json, balance_sheet.json, cashflow_statement.json,
#            enterprise_values.json, key_metrics.json, ratios.json

import json, os, re, glob
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import importlib.util
from joblib import load as joblib_load
import lightgbm as lgb
import simulations  # Import sibling module

# ----------------------------
# Config & small utilities
# ----------------------------
APP_TITLE = "Fundamentals V1 – Multi-Ticker Dashboard"
DEFAULT_DATA_ROOT = "data/raw"       # change if your raw folder lives elsewhere
SNAPSHOT_REGEX = re.compile(r"^\d{8}$")

def _latest_snapshot(data_root: str) -> Optional[str]:
    if not os.path.isdir(data_root):
        return None
    candidates = [d for d in os.listdir(data_root) if SNAPSHOT_REGEX.match(d)]
    if not candidates:
        return None
    return sorted(candidates)[-1]

def _safe_int(x):
    # Alpha Vantage often uses strings and "None"
    if x in (None, "None", ""):
        return np.nan
    try:
        return int(x)
    except Exception:
        try:
            return float(x)
        except Exception:
            return np.nan

def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def _endpoint_path(data_root: str, snapshot: str, ticker: str, endpoint: str) -> str:
    # endpoint should be e.g. "income_statement", "balance_sheet"
    return os.path.join(data_root, snapshot, ticker, f"{endpoint}.json")

def _list_tickers(data_root: str, snapshot: str) -> List[str]:
    base = os.path.join(data_root, snapshot)
    if not os.path.isdir(base):
        return []
    # tickers are dirs that contain at least one of our endpoint files
    tickers = []
    for d in sorted(os.listdir(base)):
        p = os.path.join(base, d)
        if os.path.isdir(p):
            if any(os.path.exists(os.path.join(p, f"{e}.json")) for e in [
                "income_statement","balance_sheet","cashflow_statement",
                "enterprise_values","key_metrics","ratios"
            ]):
                tickers.append(d)
    return tickers

def _reports_to_df(reports: List[dict], keep: List[str]) -> pd.DataFrame:
    if not reports:
        return pd.DataFrame()
    rows = []
    for r in reports:
        row = {"fiscalDateEnding": r.get("fiscalDateEnding")}
        for k in keep:
            row[k] = _safe_int(r.get(k))
        rows.append(row)
    df = pd.DataFrame(rows)
    if "fiscalDateEnding" in df.columns:
        df["fiscalDateEnding"] = pd.to_datetime(df["fiscalDateEnding"], errors="coerce")
        df = df.sort_values("fiscalDateEnding")
    return df

def _load_statements(data_root: str, snapshot: str, ticker: str, period: str) -> Dict[str, pd.DataFrame]:
    """
    period: 'annual' or 'quarterly'
    returns dict with keys: income, balance, cashflow
    """
    period_key = "annualReports" if period == "annual" else "quarterlyReports"
    out = {}

    # Income Statement
    ipath = _endpoint_path(data_root, snapshot, ticker, "income_statement")
    idoc = _read_json(ipath)
    if idoc and isinstance(idoc.get("response"), dict):
        ireps = idoc["response"].get(period_key, [])
        ikeep = [
            "totalRevenue","grossProfit","operatingIncome","netIncome",
            "researchAndDevelopment","sellingGeneralAndAdministrative",
            "depreciationAndAmortization","costOfRevenue","operatingExpenses"
        ]
        out["income"] = _reports_to_df(ireps, ikeep)
    else:
        out["income"] = pd.DataFrame()

    # Balance Sheet
    bpath = _endpoint_path(data_root, snapshot, ticker, "balance_sheet")
    bdoc = _read_json(bpath)
    if bdoc and isinstance(bdoc.get("response"), dict):
        breps = bdoc["response"].get(period_key, [])
        bkeep = [
            "totalAssets","totalLiabilities","totalShareholderEquity",
            "cashAndCashEquivalentsAtCarryingValue","shortTermInvestments",
            "longTermDebt","totalCurrentAssets","totalCurrentLiabilities","inventory"
        ]
        out["balance"] = _reports_to_df(breps, bkeep)
    else:
        out["balance"] = pd.DataFrame()

    # Cash Flow
    cpath = _endpoint_path(data_root, snapshot, ticker, "cashflow_statement")
    cdoc = _read_json(cpath)
    if cdoc and isinstance(cdoc.get("response"), dict):
        creps = cdoc["response"].get(period_key, [])
        ckeep = [
            "operatingCashflow","capitalExpenditures",
            "cashflowFromInvestment","cashflowFromFinancing","dividendPayout"
        ]
        out["cashflow"] = _reports_to_df(creps, ckeep)
        if not out["cashflow"].empty:
            out["cashflow"]["freeCashFlow"] = (
                out["cashflow"]["operatingCashflow"] - out["cashflow"]["capitalExpenditures"]
            )
    else:
        out["cashflow"] = pd.DataFrame()

    return out

def _load_overview_like(data_root: str, snapshot: str, ticker: str) -> pd.Series:
    """
    enterprise_values.json / key_metrics.json / ratios.json in your run
    appear to be AlphaVantage OVERVIEW mirror (flat dict in response).
    We read first one that exists and return a flattened Series.
    """
    for ep in ["enterprise_values","key_metrics","ratios"]:
        path = _endpoint_path(data_root, snapshot, ticker, ep)
        doc = _read_json(path)
        if doc and isinstance(doc.get("response"), dict) and len(doc["response"]) > 0:
            return pd.Series(doc["response"])
    return pd.Series(dtype="object")

def _merge_for_display(ticker: str, income: pd.DataFrame, cash: pd.DataFrame) -> pd.DataFrame:
    # for charts that need revenue & netIncome + FCF in one frame
    df = pd.DataFrame()
    if not income.empty:
        df = income[["fiscalDateEnding","totalRevenue","grossProfit","operatingIncome","netIncome"]].copy()
    if not cash.empty:
        if df.empty:
            df = cash[["fiscalDateEnding"]].copy()
        df = df.merge(cash[["fiscalDateEnding","operatingCashflow","capitalExpenditures","freeCashFlow"]],
                      on="fiscalDateEnding", how="outer")
    if df.empty:
        return df
    if "fiscalDateEnding" not in df.columns:
        return pd.DataFrame()
    df["ticker"] = ticker
    return df.sort_values("fiscalDateEnding")

# ----------------------------
# Model A utilities (import on demand)
# ----------------------------
def _load_model_a_module() -> Optional[object]:
    path = Path("models/model-A/model_a.py")
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("model_a_dash", path)
    if not spec or not spec.loader:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

@st.cache_data(show_spinner=False)
def _load_model_a_predictions(data_root: str, snapshot: str, tickers: Tuple[str, ...], period: str):
    mod = _load_model_a_module()
    if not mod:
        raise RuntimeError("Model A file not found.")
    os.environ["RAW_DIR"] = data_root
    df = mod.predict_latest(snapshot=snapshot, tickers=list(tickers), period=period)
    return df

def _load_model_a_drivers(data_root: str, snapshot: str, ticker: str, period: str) -> Optional[pd.DataFrame]:
    """
    Returns top feature contributions for the ticker's latest row (q50 model).
    Uses LightGBM pred_contrib (TreeSHAP) for transparency.
    """
    mod = _load_model_a_module()
    if not mod:
        return None
    meta_path = Path("models/model_a") / snapshot / "metadata.json"
    pre_path = Path("models/model_a") / snapshot / "preprocessor.joblib"
    model_path = Path("models/model_a") / snapshot / "lgbm_q50.txt"
    if not (meta_path.exists() and pre_path.exists() and model_path.exists()):
        return None

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    feature_names: List[str] = meta.get("features", [])
    pre = joblib_load(pre_path)
    booster = lgb.Booster(model_file=str(model_path))

    # Build latest feature row for the ticker (reuse model_a helpers)
    os.environ["RAW_DIR"] = data_root
    stmts = mod._load_statements(Path(data_root), snapshot, ticker, period)
    fx = mod._feature_frame(ticker, stmts)
    if fx.empty:
        return None
    row = fx.sort_values("asof_date").iloc[-1:]
    missing = [c for c in feature_names if c not in row.columns]
    if missing:
        return None
    X = row[feature_names].copy()
    Xn = pre.transform(X)
    contrib = booster.predict(Xn, pred_contrib=True)
    if contrib is None or len(contrib) == 0:
        return None
    # Last column is bias term
    vals = contrib[0]
    feat_contrib = pd.DataFrame({
        "feature": feature_names + ["bias"],
        "contribution": vals
    })
    feat_contrib["abs"] = feat_contrib["contribution"].abs()
    top = feat_contrib.sort_values("abs", ascending=False).head(6)
    return top[["feature","contribution"]]

@st.cache_data(show_spinner=False)
def load_all(data_root: str, snapshot: str, tickers: Tuple[str, ...], period: str):
    bundles = {}
    for t in tickers:
        stmts = _load_statements(data_root, snapshot, t, period)
        overview = _load_overview_like(data_root, snapshot, t)
        bundles[t] = {"stmts": stmts, "overview": overview}
    return bundles

# ----------------------------
# Model B utilities (Simulation)
# ----------------------------
def _load_model_b_module() -> Optional[object]:
    path = Path("models/model-B/model_b.py")
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("model_b_dash", path)
    if not spec or not spec.loader:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

@st.cache_resource(show_spinner=False)
def _load_model_b_resources(data_root: str, snapshot: str):
    mod = _load_model_b_module()
    if not mod:
        return None
    # Model B needs RAW_DIR env to be correct if it uses it internally, 
    # though load_resources mostly reads from cache. 
    # But just in case:
    os.environ["RAW_DIR"] = data_root
    try:
        return mod.load_resources(snapshot)
    except Exception as e:
        st.error(f"Failed to load Model B resources: {e}")
        return None

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.markdown("**Data Source**")
    data_root = st.text_input("Data root", value=DEFAULT_DATA_ROOT)
    snap_default = _latest_snapshot(data_root) or ""
    snapshot = st.text_input("Snapshot (YYYYMMDD or folder)", value=snap_default, key="snapshot_input_widget")
    period = st.radio("Period", options=("annual","quarterly"), index=1, horizontal=True)

    # Mode Selector
    view_mode = st.radio("Mode", options=("Standard Dashboard", "🔬 Simulation Lab"), index=0)

    # Tickerview
    available = _list_tickers(data_root, snapshot) if snapshot else []
    default_sel = ["AAPL","MSFT","TSLA"]
    # fall back to whatever exists
    preselect = [t for t in default_sel if t in available] or available[:3]
    tickers = st.multiselect("Tickers", options=available, default=preselect)

    st.markdown("---")
    st.caption("Tip: Put your files under `data/raw/<SNAPSHOT>/<TICKER>/*.json`")

if not snapshot:
    st.warning("No snapshot selected. Enter a snapshot folder name (e.g., 20251022).")
    st.stop()
if not tickers:
    st.warning("No tickers found/selected in this snapshot.")
    st.stop()

bundles = load_all(data_root, snapshot, tuple(tickers), period)

# ---------------------------------
# Dispatcher
# ---------------------------------
if view_mode == "🔬 Simulation Lab":
    st.subheader("Interactive Simulation Lab")
    
    # Check Model B
    mod_b = _load_model_b_module()
    if not mod_b:
        st.error("Model B (models/model-B/model_b.py) not found. Cannot run simulations.")
        st.stop()
        
    # Load Resources
    with st.spinner("Loading Model B artifacts..."):
        res = _load_model_b_resources(data_root, snapshot)
        
    if not res:
        st.warning(f"Could not load Model B artifacts for snapshot '{snapshot}'. Train the model first.")
        st.info("Run: `python models/model-B/model_b.py train --snapshot <SNAPSHOT>`")
        st.stop()
        
    # Selector for single ticker simulation
    sim_ticker = st.selectbox("Select Ticker for Simulation", options=tickers)
    
    if sim_ticker:
        simulations.render_what_if_analysis(mod_b, res, bundles, sim_ticker)
        
    st.stop()  # Stop here so we don't render standard dashboard below

# ---------------------------------
# Overview cards (Market Cap, P/E)
# ---------------------------------
st.subheader("Overview (from overview-like endpoints)")
cols = st.columns(min(4, max(1, len(tickers))))
for i, t in enumerate(tickers):
    ov = bundles[t]["overview"]
    with cols[i % len(cols)]:
        st.markdown(f"#### {t}")
        if ov.empty:
            st.info("No overview data.")
        else:
            cap = ov.get("MarketCapitalization")
            pe = ov.get("PERatio") or ov.get("TrailingPE")
            ev_ebitda = ov.get("EVToEBITDA")
            beta = ov.get("Beta")
            st.metric("Market Cap", f"${float(cap)/1e12:.2f}T" if cap else "—")
            c1, c2 = st.columns(2)
            c1.metric("P/E", f"{float(pe):.2f}" if pe else "—")
            c2.metric("EV/EBITDA", f"{float(ev_ebitda):.2f}" if ev_ebitda else "—")
            st.caption(f"Beta: {beta if beta else '—'}")

# ---------------------------------
# Revenue & Income chart
# ---------------------------------
st.subheader(f"Revenue & Profitability ({period.title()})")
combined = []
for t in tickers:
    stmts = bundles[t]["stmts"]
    merged = _merge_for_display(t, stmts["income"], stmts["cashflow"])
    if not merged.empty:
        combined.append(merged)
if not combined:
    st.warning("No statement data to display.")
    st.stop()

df_all = pd.concat(combined, ignore_index=True).sort_values(["ticker","fiscalDateEnding"])
show_net = st.checkbox("Show Net Income", value=True)
show_gp = st.checkbox("Show Gross Profit", value=False)
show_oper = st.checkbox("Show Operating Income", value=False)

def make_rev_chart(df):
    base = df.dropna(subset=["totalRevenue"])
    if base.empty:
        return go.Figure()
    fig = go.Figure()
    for t in base["ticker"].unique():
        dd = base[base["ticker"]==t]
        fig.add_trace(go.Scatter(
            x=dd["fiscalDateEnding"], y=dd["totalRevenue"], mode="lines+markers", name=f"{t} – Revenue"
        ))
        if show_net and "netIncome" in dd.columns and dd["netIncome"].notna().any():
            fig.add_trace(go.Scatter(
                x=dd["fiscalDateEnding"], y=dd["netIncome"], mode="lines+markers", name=f"{t} – Net Income"
            ))
        if show_gp and "grossProfit" in dd.columns and dd["grossProfit"].notna().any():
            fig.add_trace(go.Scatter(
                x=dd["fiscalDateEnding"], y=dd["grossProfit"], mode="lines+markers", name=f"{t} – Gross Profit"
            ))
        if show_oper and "operatingIncome" in dd.columns and dd["operatingIncome"].notna().any():
            fig.add_trace(go.Scatter(
                x=dd["fiscalDateEnding"], y=dd["operatingIncome"], mode="lines+markers", name=f"{t} – Operating Income"
            ))
    fig.update_layout(
        height=460, margin=dict(l=10,r=10,t=40,b=10),
        yaxis_title="USD", xaxis_title="Fiscal Date",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    return fig

st.plotly_chart(make_rev_chart(df_all), use_container_width=True)

# ---------------------------------
# Margins (computed)
# ---------------------------------
st.subheader(f"Margins ({period.title()})")
def compute_margins(df):
    out = []
    for t in df["ticker"].unique():
        dd = df[df["ticker"]==t].copy()
        # Need revenue to compute margins
        dd = dd.dropna(subset=["totalRevenue"])
        if dd.empty: 
            continue
        dd["grossMargin"] = dd["grossProfit"] / dd["totalRevenue"]
        dd["operatingMargin"] = dd["operatingIncome"] / dd["totalRevenue"]
        dd["netMargin"] = dd["netIncome"] / dd["totalRevenue"]
        dd["ticker"] = t
        out.append(dd[["fiscalDateEnding","ticker","grossMargin","operatingMargin","netMargin"]])
    return pd.concat(out) if out else pd.DataFrame()

marg = compute_margins(df_all)
if marg.empty:
    st.info("Margins require revenue plus profit lines; not enough data.")
else:
    tab1, tab2, tab3 = st.tabs(["Gross Margin","Operating Margin","Net Margin"])
    for name, col, tab in [
        ("Gross Margin", "grossMargin", tab1),
        ("Operating Margin", "operatingMargin", tab2),
        ("Net Margin", "netMargin", tab3),
    ]:
        with tab:
            fig = go.Figure()
            for t in marg["ticker"].unique():
                dd = marg[marg["ticker"]==t].dropna(subset=[col])
                if dd.empty: 
                    continue
                fig.add_trace(go.Scatter(x=dd["fiscalDateEnding"], y=dd[col],
                                         mode="lines+markers", name=t))
            fig.update_layout(
                height=400, margin=dict(l=10,r=10,t=40,b=10),
                yaxis_title=name, xaxis_title="Fiscal Date", yaxis_tickformat=".0%",
                legend=dict(orientation="h", y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------
# Cash Flow (CFO vs CapEx vs FCF)
# ---------------------------------
st.subheader(f"Cash Flow ({period.title()})")
def make_cash_chart(df):
    fig = go.Figure()
    for t in df["ticker"].unique():
        dd = df[df["ticker"]==t][["fiscalDateEnding","operatingCashflow","capitalExpenditures","freeCashFlow"]].copy()
        if dd[["operatingCashflow","capitalExpenditures","freeCashFlow"]].isna().all(axis=None):
            continue
        fig.add_trace(go.Bar(x=dd["fiscalDateEnding"], y=dd["operatingCashflow"], name=f"{t} – CFO"))
        fig.add_trace(go.Bar(x=dd["fiscalDateEnding"], y=dd["capitalExpenditures"], name=f"{t} – CapEx"))
        fig.add_trace(go.Scatter(x=dd["fiscalDateEnding"], y=dd["freeCashFlow"],
                                 mode="lines+markers", name=f"{t} – FCF"))
    fig.update_layout(
        barmode="group", height=500, margin=dict(l=10,r=10,t=40,b=10),
        yaxis_title="USD", xaxis_title="Fiscal Date", legend=dict(orientation="h", y=1.02)
    )
    return fig

st.plotly_chart(make_cash_chart(df_all), use_container_width=True)

# ---------------------------------
# Data tables & exports
# ---------------------------------
st.subheader("Raw Tables")
for t in tickers:
    stmts = bundles[t]["stmts"]
    st.markdown(f"### {t}")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption("Income Statement")
        df = stmts["income"].copy()
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button("Download Income CSV", df.to_csv(index=False).encode(),
                               file_name=f"{t}_income_{period}_{snapshot}.csv", mime="text/csv")
        else:
            st.info("No income statement data.")
    with c2:
        st.caption("Balance Sheet")
        dfb = stmts["balance"].copy()
        if not dfb.empty:
            st.dataframe(dfb, use_container_width=True, hide_index=True)
            st.download_button("Download Balance CSV", dfb.to_csv(index=False).encode(),
                               file_name=f"{t}_balance_{period}_{snapshot}.csv", mime="text/csv")
        else:
            st.info("No balance sheet data.")
    with c3:
        st.caption("Cash Flow")
        dfc = stmts["cashflow"].copy()
        if not dfc.empty:
            st.dataframe(dfc, use_container_width=True, hide_index=True)
            st.download_button("Download Cashflow CSV", dfc.to_csv(index=False).encode(),
                               file_name=f"{t}_cashflow_{period}_{snapshot}.csv", mime="text/csv")
        else:
            st.info("No cashflow data.")

# ---------------------------------
# Model predictions & explainability (Model A)
# ---------------------------------
st.subheader("Model A Quantile Forecasts (12m return)")
model_snap = snapshot  # reuse selected snapshot
model_dir = Path("models/model_a") / model_snap
if not model_dir.exists():
    st.info(f"No Model A artifacts for snapshot {model_snap}. Train/predict first.")
else:
    try:
        preds = _load_model_a_predictions(data_root, model_snap, tuple(tickers), period)
    except Exception as e:
        st.warning(f"Could not load predictions: {e}")
        preds = None

    if preds is not None and not preds.empty:
        preds = preds.sort_values("ticker")

        def _uncertainty_flag(width: float, all_widths: pd.Series) -> str:
            if all_widths.empty or pd.isna(width):
                return "unknown"
            med = all_widths.median()
            if width <= 0.5 * med:
                return "narrow"
            if width >= 1.5 * med:
                return "wide"
            return "normal"

        widths = preds["p90_width"]
        preds["uncertainty"] = [_uncertainty_flag(w, widths) for w in widths]

        # Fan chart
        fig = go.Figure()
        for _, row in preds.iterrows():
            t = row["ticker"]
            x = [row["asof_date"]] * 5
            y = [row["q5"], row["q25"], row["q50"], row["q75"], row["q95"]]
            fig.add_trace(go.Scatter(
                x=[row["asof_date"], row["asof_date"]],
                y=[row["q25"], row["q75"]],
                mode="lines",
                line=dict(width=10, color="rgba(0,123,255,0.2)"),
                showlegend=False,
                hoverinfo="skip"
            ))
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="lines+markers",
                name=t, line=dict(shape="hv"), marker=dict(size=6)
            ))
        fig.update_layout(
            height=320, margin=dict(l=10,r=10,t=30,b=10),
            yaxis_title="Expected 12m return", xaxis_title="As-of date",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Predictions are quantile bands (non-Gaussian). Width indicates uncertainty; not a bell curve.")
        st.dataframe(preds[["ticker","asof_date","q5","q25","q50","q75","q95","p90_width","uncertainty"]]
                           .rename(columns={"p90_width":"p90_width (spread)"}),
                     use_container_width=True, hide_index=True)

        # Explainability
        sel = st.selectbox("Explain drivers for ticker", options=list(preds["ticker"]), key="xai_ticker")
        drivers = _load_model_a_drivers(data_root, model_snap, sel, period)
        if drivers is None or drivers.empty:
            st.info("No feature contributions available. Ensure model files exist for this snapshot.")
        else:
            st.caption("Top contributors to median (q50) prediction — positive pushes up, negative pushes down.")
            figc = px.bar(drivers.sort_values("contribution"), x="contribution", y="feature", orientation="h",
                          color="contribution", color_continuous_scale=["#d62728","#2ca02c"])
            figc.update_layout(height=320, margin=dict(l=10,r=10,t=30,b=10), coloraxis_showscale=False)
            st.plotly_chart(figc, use_container_width=True)
    else:
        st.info("No predictions available for selected tickers/snapshot.")

st.caption("V1 • Designed for unfiltered ingestion; downstream feature selection happens during training/FE.")
