# streamlit_app.py
# V1 finance fundamentals viewer for AAPL/MSFT/TSLA (or any tickers you’ve collected)
# Directory layout expected: data/raw/<SNAPSHOT>/<TICKER>/<endpoint>.json
# Endpoints: income_statement.json, balance_sheet.json, cashflow_statement.json,
#            enterprise_values.json, key_metrics.json, ratios.json

import json, os, re, glob
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

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
    if not df.empty:
        df["ticker"] = ticker
    return df.sort_values("fiscalDateEnding")

@st.cache_data(show_spinner=False)
def load_all(data_root: str, snapshot: str, tickers: Tuple[str, ...], period: str):
    bundles = {}
    for t in tickers:
        stmts = _load_statements(data_root, snapshot, t, period)
        overview = _load_overview_like(data_root, snapshot, t)
        bundles[t] = {"stmts": stmts, "overview": overview}
    return bundles

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.markdown("**Data Source**")
    data_root = st.text_input("Data root", value=DEFAULT_DATA_ROOT)
    snap_default = _latest_snapshot(data_root) or ""
    snapshot = st.text_input("Snapshot (YYYYMMDD or folder)", value=snap_default)
    period = st.radio("Period", options=("annual","quarterly"), index=1, horizontal=True)

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

st.caption("V1 • Designed for unfiltered ingestion; downstream feature selection happens during training/FE.")
