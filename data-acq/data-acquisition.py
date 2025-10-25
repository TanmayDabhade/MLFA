#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Layer-1 (Bronze): Lossless fundamentals ingestion with provider fallback.
- Tries FMP /stable first; on 402/403/401 auto-falls back to Alpha Vantage.
- Stores raw responses verbatim wrapped with metadata (including provider).
- Skips only if existing file is valid & non-empty.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import urllib.parse as up
import logging

import requests
from requests.adapters import HTTPAdapter, Retry

try:
    from dotenv import load_dotenv  # optional
except Exception:
    def load_dotenv(*_a, **_k): pass

# ----------------------------
# Providers & Config
# ----------------------------
FMP_BASE = "https://financialmodelingprep.com/stable"
AV_BASE  = "https://www.alphavantage.co/query"

DEFAULT_QPS_FMP = 3.0           # FMP: up to ~3 QPS on free/low tiers
MIN_INTERVAL_AV = 12.5          # Alpha Vantage: ~5 req/min -> 12s+ between calls

SUPPORTED_PERIODS = {"quarter", "annual"}  # Shape only. We do NO filtering.

BLOCK_STATUS = {401, 402, 403}  # auth/plan/forbidden -> fallback candidate

@dataclass(frozen=True)
class EndpointCfg:
    name: str
    fmp_route: str          # /stable route (e.g., "income-statement")
    has_period: bool = True
    default_limit: int = 400
    av_function: str | None = None  # Alpha Vantage function name

def build_endpoints(period: str) -> Dict[str, EndpointCfg]:
    """
    Map our six logical endpoints to FMP and Alpha Vantage equivalents.
    For AV: ratios/key_metrics/enterprise_values all map to OVERVIEW (duplicates allowed).
    """
    specs: Iterable[EndpointCfg] = [
        EndpointCfg("income_statement",   "income-statement",        True,  400, "INCOME_STATEMENT"),
        EndpointCfg("balance_sheet",      "balance-sheet-statement", True,  400, "BALANCE_SHEET"),
        EndpointCfg("cashflow_statement", "cash-flow-statement",     True,  400, "CASH_FLOW"),
        EndpointCfg("ratios",             "ratios",                  True,  400, "OVERVIEW"),
        EndpointCfg("key_metrics",        "key-metrics",             True,  400, "OVERVIEW"),
        EndpointCfg("enterprise_values",  "enterprise-values",       True,  400, "OVERVIEW"),
    ]
    return {s.name: s for s in specs}

# ----------------------------
# HTTP & Utils
# ----------------------------
def make_session() -> requests.Session:
    sess = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.7,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    sess.headers.update({"User-Agent": "fundamentals-pipeline/1.1"})
    return sess

def build_url(base: str, route: str, params: Dict[str, Any]) -> str:
    qp = up.urlencode(params, doseq=True)
    return f"{base.rstrip('/')}/{route}?{qp}"

def fetch_json(sess: requests.Session, url: str, timeout: int = 30) -> Tuple[int, Any]:
    r = sess.get(url, timeout=timeout)
    try:
        data = r.json()
    except Exception:
        data = {"_error": "non_json_response", "_text": r.text[:2000]}
    return r.status_code, data

def ensure_env(var: str) -> str | None:
    v = os.getenv(var)
    return v if (v and v.strip()) else None

def response_has_content(resp: Any) -> bool:
    if resp is None: return False
    if isinstance(resp, list): return len(resp) > 0
    if isinstance(resp, dict): return len(resp.keys()) > 0 and not {"error", "Error Message"} & {k.lower() for k in resp.keys()}
    return False

def existing_is_valid(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        meta = payload.get("_metadata") or {}
        status = meta.get("status_code")
        resp = payload.get("response")
        if not (isinstance(status, int) and 200 <= status < 300): return False
        return response_has_content(resp)
    except Exception:
        return False

def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def setup_logging(verbosity: int) -> None:
    lvl = logging.WARNING
    if verbosity == 1: lvl = logging.INFO
    elif verbosity >= 2: lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")

# ----------------------------
# Provider calls
# ----------------------------
def call_fmp(sess: requests.Session, cfg: EndpointCfg, ticker: str, apikey: str, period: str) -> Tuple[int, Any, str]:
    params = {"symbol": ticker, "limit": cfg.default_limit, "apikey": apikey}
    if cfg.has_period: params["period"] = period
    url = build_url(FMP_BASE, cfg.fmp_route, params)
    status, data = fetch_json(sess, url)
    return status, data, url

def call_alpha_vantage(sess: requests.Session, cfg: EndpointCfg, ticker: str, apikey: str) -> Tuple[int, Any, str]:
    if not cfg.av_function:
        # Unsupported on AV; return a 501-like status
        return 501, {"_error": "unsupported_on_alpha_vantage"}, f"{AV_BASE}?function=UNSUPPORTED&symbol={ticker}"
    params = {"function": cfg.av_function, "symbol": ticker, "apikey": apikey}
    url = build_url(AV_BASE, "", params)[:-1]  # remove trailing '?'
    status, data = fetch_json(sess, url)
    return status, data, url

# ----------------------------
# CLI & Main
# ----------------------------
def read_tickers(args) -> List[str]:
    tickers: List[str] = []
    if args.tickers: tickers.extend(args.tickers)
    if args.tickers_file:
        p = Path(args.tickers_file)
        if not p.exists(): raise FileNotFoundError(f"--tickers-file not found: {p}")
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s and not s.startswith("#"): tickers.append(s)
    tickers = sorted(set(t.upper() for t in tickers))
    if not tickers: raise ValueError("No tickers provided. Use --tickers and/or --tickers-file.")
    return tickers

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Lossless fundamentals ingestion with provider fallback (Layer-1).")
    parser.add_argument("--tickers", nargs="+", help="Tickers, e.g. AAPL MSFT TSLA")
    parser.add_argument("--tickers-file", help="Path to newline-delimited tickers")
    parser.add_argument("--snapshot", default=datetime.now(timezone.utc).strftime("%Y%m%d"), help="YYYYMMDD (default: today UTC)")
    parser.add_argument("--period", default="quarter", choices=sorted(SUPPORTED_PERIODS), help="FMP period param (default: quarter)")
    parser.add_argument("--data-dir", default="data/raw", help="Output base dir (default: data/raw)")
    parser.add_argument("--qps", type=float, default=DEFAULT_QPS_FMP, help=f"FMP queries per second (default: {DEFAULT_QPS_FMP})")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files regardless of validity")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv)")
    args = parser.parse_args()
    setup_logging(args.verbose)

    # Keys (FMP optional, AV recommended for fallback)
    fmp_key = ensure_env("FMP_API_KEY")
    av_key  = ensure_env("ALPHA_VANTAGE_API_KEY")

    try:
        tickers = read_tickers(args)
    except Exception as e:
        logging.error(str(e)); print(f"ERROR: {e}", file=sys.stderr); sys.exit(2)

    endpoints = build_endpoints(args.period)
    sess = make_session()
    base_out = Path(args.data_dir) / args.snapshot
    fetched_at = datetime.now(timezone.utc).isoformat()

    total_ok = total_skip = total_err = 0
    logging.info(f"Snapshot: {args.snapshot} | Period: {args.period} | FMP_QPS: {args.qps}")
    logging.info(f"Output base: {base_out}")

    # Rate control
    sleep_fmp = 1.0 / max(args.qps, 0.1)
    last_av_call_ts = 0.0

    for t in tickers:
        t_dir = base_out / t
        for name, cfg in endpoints.items():
            out_path = t_dir / f"{name}.json"

            if out_path.exists() and not args.overwrite and existing_is_valid(out_path):
                total_skip += 1
                logging.debug(f"Skip existing (valid): {out_path}")
                continue

            # 1) Try FMP first if key present
            used_provider = None
            status = None
            data = None
            url = None

            if fmp_key:
                status, data, url = call_fmp(sess, cfg, t, fmp_key, args.period)
                used_provider = "FMP"
                # If blocked by plan/auth, consider fallback
                if status in BLOCK_STATUS or not response_has_content(data):
                    logging.warning(f"FMP not usable (status={status} or empty) -> considering Alpha Vantage fallback for {t}/{name}")
                    used_provider = None  # will be replaced if AV works
                else:
                    time.sleep(sleep_fmp)

            # 2) Fallback to Alpha Vantage if needed & key available
            if used_provider is None and av_key:
                # enforce AV rate limit
                now = time.time()
                wait = MIN_INTERVAL_AV - (now - last_av_call_ts)
                if wait > 0:
                    time.sleep(wait)
                status, data, url = call_alpha_vantage(sess, cfg, t, av_key)
                last_av_call_ts = time.time()
                used_provider = "ALPHA_VANTAGE"

            # 3) If still nothing, mark error but save envelope for audit
            if used_provider is None:
                used_provider = "NONE"
                if status is None:
                    status, data, url = 520, {"_error": "no_provider_available"}, "about:none"

            envelope = {
                "_metadata": {
                    "ticker": t,
                    "endpoint": name,
                    "provider": used_provider,
                    "request_url": url,
                    "status_code": status,
                    "fetched_at": fetched_at,
                },
                "response": data,
            }

            try:
                write_json(out_path, envelope)
                if 200 <= status < 300 and response_has_content(data):
                    total_ok += 1
                    logging.info(f"OK [{status}] {used_provider} {t}/{name}")
                else:
                    total_err += 1
                    logging.warning(f"SAVED but not OK (status={status} or empty) {used_provider} {t}/{name}")
            except Exception as e:
                total_err += 1
                logging.error(f"WRITE FAIL {t}/{name}: {e}")

            # If previous call was FMP, we already slept; if AV, we throttled above.

    print(f"Done. Snapshot: {args.snapshot} | OK: {total_ok} | Skipped: {total_skip} | Errors: {total_err} | Output: {base_out}")

    # Non-zero exit if nothing worked
    if total_ok == 0 and total_err > 0 and total_skip == 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
