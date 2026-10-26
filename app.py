"""
AI-Powered Indian Stock Recommendation & Portfolio Intelligence System
=======================================================================

A single-file Streamlit application that analyzes 20 Indian (NSE-listed)
stocks using technical analysis, fundamental analysis, lexicon-based
sentiment analysis and a machine-learning model (Logistic Regression /
Random Forest / XGBoost, chronologically validated) to produce
explainable BUY / HOLD / SELL recommendations, plus portfolio tracking,
a watchlist, stock comparison, walk-forward backtesting and a simple
data-grounded Q&A assistant.

Run with:
    streamlit run app.py

DISCLAIMER: This application is for educational and research purposes
only. It does not constitute financial advice. Nothing in this app
guarantees profit or predicts future returns with certainty.

Modes
-----
APP_MODE=DEMO (default): all data (prices, fundamentals, news) is
    generated/bundled deterministically so the app works fully offline
    and is clearly labeled "Demo Data" in the UI.
APP_MODE=LIVE: attempts to fetch real data via `yfinance`. Any field
    that cannot be fetched falls back to DEMO data for that field only,
    and is labeled accordingly. Fundamentals that are simply unavailable
    are shown as "Data unavailable" rather than invented.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import secrets
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Optional heavy dependencies. The app must never hard-crash if one of these
# is missing -- it degrades gracefully and tells the user in the UI.
# ---------------------------------------------------------------------------
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:  # pragma: no cover
    HAS_STREAMLIT = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:  # pragma: no cover
    HAS_PLOTLY = False

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:  # pragma: no cover
    HAS_YFINANCE = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:  # pragma: no cover
    HAS_XGBOOST = False

try:
    import shap
    HAS_SHAP = True
except ImportError:  # pragma: no cover
    HAS_SHAP = False

APP_DIR = Path(__file__).resolve().parent
STORE_PATH = APP_DIR / "stockai_store.json"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

APP_MODE = os.environ.get("APP_MODE", "DEMO").upper()  # DEMO or LIVE
IS_DEMO_MODE = APP_MODE != "LIVE"

DISCLAIMER = (
    "This platform is for educational and research purposes only and does "
    "not constitute financial advice. AI-generated research signals are "
    "not guarantees of profit. Historical or backtested performance does "
    "not guarantee future results."
)

# =============================================================================
# SECTION 1 — STOCK UNIVERSE & CONFIGURATION
# =============================================================================
# Exactly 20 NSE-listed Indian stocks with correct ticker symbols (Yahoo
# Finance convention: NSE symbol + ".NS"). This is the single source of
# truth for the whole application.

STOCKS: list[dict[str, Any]] = [
    {"symbol": "WIPRO", "yf_symbol": "WIPRO.NS", "name": "Wipro Limited",
     "sector": "Information Technology", "industry": "IT Services & Consulting",
     "market_category": "Large Cap", "base_price": 480.0, "annual_vol": 0.28, "drift": 0.06},
    {"symbol": "TCS", "yf_symbol": "TCS.NS", "name": "Tata Consultancy Services Limited",
     "sector": "Information Technology", "industry": "IT Services & Consulting",
     "market_category": "Large Cap", "base_price": 3850.0, "annual_vol": 0.24, "drift": 0.09},
    {"symbol": "INFY", "yf_symbol": "INFY.NS", "name": "Infosys Limited",
     "sector": "Information Technology", "industry": "IT Services & Consulting",
     "market_category": "Large Cap", "base_price": 1550.0, "annual_vol": 0.27, "drift": 0.08},
    {"symbol": "HDFCBANK", "yf_symbol": "HDFCBANK.NS", "name": "HDFC Bank Limited",
     "sector": "Financial Services", "industry": "Private Sector Bank",
     "market_category": "Large Cap", "base_price": 1650.0, "annual_vol": 0.22, "drift": 0.10},
    {"symbol": "SBIN", "yf_symbol": "SBIN.NS", "name": "State Bank of India",
     "sector": "Financial Services", "industry": "Public Sector Bank",
     "market_category": "Large Cap", "base_price": 810.0, "annual_vol": 0.30, "drift": 0.11},
    {"symbol": "ICICIBANK", "yf_symbol": "ICICIBANK.NS", "name": "ICICI Bank Limited",
     "sector": "Financial Services", "industry": "Private Sector Bank",
     "market_category": "Large Cap", "base_price": 1220.0, "annual_vol": 0.24, "drift": 0.12},
    {"symbol": "SHRIRAMFIN", "yf_symbol": "SHRIRAMFIN.NS", "name": "Shriram Finance Limited",
     "sector": "Financial Services", "industry": "Non-Banking Financial Company",
     "market_category": "Large Cap", "base_price": 3050.0, "annual_vol": 0.33, "drift": 0.13},
    {"symbol": "PAYTM", "yf_symbol": "PAYTM.NS", "name": "One 97 Communications Limited (Paytm)",
     "sector": "Financial Services", "industry": "Fintech / Payments",
     "market_category": "Mid Cap", "base_price": 720.0, "annual_vol": 0.55, "drift": 0.04},
    {"symbol": "TATACOMM", "yf_symbol": "TATACOMM.NS", "name": "Tata Communications Limited",
     "sector": "Telecommunication", "industry": "Telecom Services",
     "market_category": "Mid Cap", "base_price": 1780.0, "annual_vol": 0.34, "drift": 0.05},
    {"symbol": "RELIANCE", "yf_symbol": "RELIANCE.NS", "name": "Reliance Industries Limited",
     "sector": "Energy / Conglomerate", "industry": "Refineries / Retail / Telecom",
     "market_category": "Large Cap", "base_price": 1330.0, "annual_vol": 0.26, "drift": 0.08},
    {"symbol": "HCLTECH", "yf_symbol": "HCLTECH.NS", "name": "HCL Technologies Limited",
     "sector": "Information Technology", "industry": "IT Services & Consulting",
     "market_category": "Large Cap", "base_price": 1780.0, "annual_vol": 0.26, "drift": 0.07},
    {"symbol": "BHARTIARTL", "yf_symbol": "BHARTIARTL.NS", "name": "Bharti Airtel Limited",
     "sector": "Telecommunication", "industry": "Telecom Services",
     "market_category": "Large Cap", "base_price": 1650.0, "annual_vol": 0.25, "drift": 0.14},
    {"symbol": "LT", "yf_symbol": "LT.NS", "name": "Larsen & Toubro Limited",
     "sector": "Industrials", "industry": "Construction & Engineering",
     "market_category": "Large Cap", "base_price": 3550.0, "annual_vol": 0.27, "drift": 0.10},
    {"symbol": "AXISBANK", "yf_symbol": "AXISBANK.NS", "name": "Axis Bank Limited",
     "sector": "Financial Services", "industry": "Private Sector Bank",
     "market_category": "Large Cap", "base_price": 1130.0, "annual_vol": 0.27, "drift": 0.09},
    {"symbol": "KOTAKBANK", "yf_symbol": "KOTAKBANK.NS", "name": "Kotak Mahindra Bank Limited",
     "sector": "Financial Services", "industry": "Private Sector Bank",
     "market_category": "Large Cap", "base_price": 1780.0, "annual_vol": 0.24, "drift": 0.07},
    {"symbol": "BAJFINANCE", "yf_symbol": "BAJFINANCE.NS", "name": "Bajaj Finance Limited",
     "sector": "Financial Services", "industry": "Non-Banking Financial Company",
     "market_category": "Large Cap", "base_price": 7100.0, "annual_vol": 0.30, "drift": 0.12},
    {"symbol": "ITC", "yf_symbol": "ITC.NS", "name": "ITC Limited",
     "sector": "Consumer Staples", "industry": "FMCG / Diversified",
     "market_category": "Large Cap", "base_price": 465.0, "annual_vol": 0.19, "drift": 0.06},
    {"symbol": "TATAMOTORS", "yf_symbol": "TATAMOTORS.NS", "name": "Tata Motors Limited",
     "sector": "Automobile", "industry": "Automobile Manufacturing",
     "market_category": "Large Cap", "base_price": 960.0, "annual_vol": 0.38, "drift": 0.09},
    {"symbol": "SUNPHARMA", "yf_symbol": "SUNPHARMA.NS", "name": "Sun Pharmaceutical Industries Limited",
     "sector": "Healthcare", "industry": "Pharmaceuticals",
     "market_category": "Large Cap", "base_price": 1780.0, "annual_vol": 0.23, "drift": 0.10},
    {"symbol": "MARUTI", "yf_symbol": "MARUTI.NS", "name": "Maruti Suzuki India Limited",
     "sector": "Automobile", "industry": "Automobile Manufacturing",
     "market_category": "Large Cap", "base_price": 12100.0, "annual_vol": 0.24, "drift": 0.08},
]
SYMBOLS: list[str] = [s["symbol"] for s in STOCKS]
STOCK_BY_SYMBOL: dict[str, dict[str, Any]] = {s["symbol"]: s for s in STOCKS}

BENCHMARK = {"symbol": "NIFTY50", "yf_symbol": "^NSEI", "name": "NIFTY 50 Index",
             "base_price": 24800.0, "annual_vol": 0.14, "drift": 0.10}

# Recommendation engine weights. Kept in one place -- change behavior here,
# not by hunting through the codebase. Must sum to 1.0.
RECO_WEIGHTS = {
    "ml_prediction": 0.40,
    "technical_score": 0.25,
    "fundamental_score": 0.20,
    "sentiment_score": 0.10,
    "risk_adjustment": 0.05,
}
assert abs(sum(RECO_WEIGHTS.values()) - 1.0) < 1e-9, "Recommendation weights must sum to 1.0"

RECO_THRESHOLDS = {"buy_min_score": 65, "sell_max_score": 40}
TARGET_HORIZON_DAYS = 20  # primary prediction horizon (trading days)
TRANSACTION_COST_BPS = 20  # 0.20% per side, used both in target labeling and backtest

HISTORY_YEARS = 6  # years of daily history generated/fetched
TRAIN_END_YEAR = 2023
VALIDATION_END_YEAR = 2024
# Everything after VALIDATION_END_YEAR is the held-out TEST period.

# =============================================================================
# SECTION 2 — CACHING HELPERS (graceful no-op outside a running Streamlit app)
# =============================================================================

def _cache_data(**kwargs):
    if HAS_STREAMLIT:
        return st.cache_data(**kwargs)
    return lambda fn: fn


def _cache_resource(**kwargs):
    if HAS_STREAMLIT:
        return st.cache_resource(**kwargs)
    return lambda fn: fn


# =============================================================================
# SECTION 3 — DATA LAYER (prices, fundamentals, news) with LIVE/DEMO modes
# =============================================================================
# Design: every function returns data plus a `source` tag ("LIVE" or
# "DEMO") so the UI can label exactly what the user is looking at. DEMO
# data is generated deterministically (seeded by ticker) so results are
# reproducible and the app works fully offline. LIVE mode never invents
# missing values -- unavailable fields stay None and the UI must show
# "Data unavailable".

def _seed_from_symbol(symbol: str) -> int:
    return int(hashlib.sha256(symbol.encode()).hexdigest(), 16) % (2**32)


def _synthetic_ohlcv(symbol: str, base_price: float, annual_vol: float, drift: float,
                      years: int = HISTORY_YEARS) -> pd.DataFrame:
    """Deterministic geometric-Brownian-motion price series with realistic
    OHLCV structure. Used only as DEMO data / LIVE-mode fallback -- always
    labeled as such by the caller, never presented as real market data."""
    rng = np.random.default_rng(_seed_from_symbol(symbol))
    end = pd.Timestamp.today().normalize()
    dates = pd.bdate_range(end=end, periods=years * 252)
    n = len(dates)
    daily_vol = annual_vol / math.sqrt(252)
    daily_drift = drift / 252
    # Fat-tailed-ish shocks via mixture: mostly normal, occasional larger move.
    shocks = rng.normal(daily_drift - 0.5 * daily_vol**2, daily_vol, n)
    jump_mask = rng.random(n) < 0.01
    shocks[jump_mask] += rng.normal(0, daily_vol * 4, jump_mask.sum())
    log_returns = shocks
    close = base_price * np.exp(np.cumsum(log_returns))
    # simple mean-reverting intraday range for OHLC
    intraday_range = np.abs(rng.normal(0.012, 0.006, n)) * close
    open_ = close * (1 + rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) + intraday_range * rng.uniform(0.2, 0.6, n)
    low = np.minimum(open_, close) - intraday_range * rng.uniform(0.2, 0.6, n)
    low = np.clip(low, 1.0, None)
    base_volume = rng.uniform(0.6, 1.4) * 3_000_000
    volume = np.abs(rng.normal(base_volume, base_volume * 0.35, n)) * (1 + 3 * np.abs(log_returns))
    df = pd.DataFrame({
        "date": dates, "open": open_, "high": high, "low": low,
        "close": close, "volume": volume.astype(np.int64),
    })
    df["source"] = "DEMO"
    return df.set_index("date")


@_cache_data(ttl=3600, show_spinner=False)
def get_price_history(symbol: str) -> pd.DataFrame:
    """Return a daily OHLCV DataFrame for `symbol` (index=date). Tries a
    live fetch when APP_MODE=LIVE, otherwise (or on any failure) falls
    back to deterministic demo data."""
    meta = STOCK_BY_SYMBOL[symbol]
    if not IS_DEMO_MODE and HAS_YFINANCE:
        try:
            raw = yf.download(meta["yf_symbol"], period=f"{HISTORY_YEARS}y",
                               interval="1d", progress=False, auto_adjust=True)
            if raw is not None and len(raw) > 60:
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = [c[0].lower() for c in raw.columns]
                else:
                    raw.columns = [c.lower() for c in raw.columns]
                raw = raw.rename_axis("date")[["open", "high", "low", "close", "volume"]]
                raw["source"] = "LIVE"
                return raw
        except Exception:
            pass
    return _synthetic_ohlcv(symbol, meta["base_price"], meta["annual_vol"], meta["drift"])


@_cache_data(ttl=3600, show_spinner=False)
def get_benchmark_history() -> pd.DataFrame:
    if not IS_DEMO_MODE and HAS_YFINANCE:
        try:
            raw = yf.download(BENCHMARK["yf_symbol"], period=f"{HISTORY_YEARS}y",
                               interval="1d", progress=False, auto_adjust=True)
            if raw is not None and len(raw) > 60:
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = [c[0].lower() for c in raw.columns]
                else:
                    raw.columns = [c.lower() for c in raw.columns]
                raw = raw.rename_axis("date")[["open", "high", "low", "close", "volume"]]
                raw["source"] = "LIVE"
                return raw
        except Exception:
            pass
    return _synthetic_ohlcv("NIFTY50", BENCHMARK["base_price"], BENCHMARK["annual_vol"], BENCHMARK["drift"])


# --- Fundamentals -----------------------------------------------------------
# DEMO fundamentals below are plausible, hand-set approximations for
# demonstration purposes only -- always labeled "Demo / sample data" in
# the UI and NEVER presented as live figures. In LIVE mode we attempt a
# real fetch via yfinance `.info`; any field yfinance doesn't return is
# left as None and rendered as "Data unavailable" (see section 51/50).
DEMO_FUNDAMENTALS: dict[str, dict[str, float | None]] = {
    "WIPRO": dict(pe=21.5, pb=3.1, roe=16.8, roce=19.2, debt_to_equity=0.15, dividend_yield=0.9, operating_margin=17.5, net_margin=13.8, revenue_growth=6.2, profit_growth=8.1, eps=22.3, market_cap_cr=250000),
    "TCS": dict(pe=28.4, pb=13.2, roe=48.0, roce=61.5, debt_to_equity=0.02, dividend_yield=1.6, operating_margin=25.1, net_margin=19.3, revenue_growth=7.8, profit_growth=9.0, eps=135.6, market_cap_cr=1400000),
    "INFY": dict(pe=25.1, pb=8.1, roe=32.4, roce=39.7, debt_to_equity=0.04, dividend_yield=2.1, operating_margin=21.3, net_margin=17.2, revenue_growth=6.9, profit_growth=7.5, eps=61.8, market_cap_cr=640000),
    "HDFCBANK": dict(pe=19.8, pb=2.9, roe=15.9, roce=None, debt_to_equity=None, dividend_yield=1.1, operating_margin=None, net_margin=27.5, revenue_growth=13.2, profit_growth=11.0, eps=83.3, market_cap_cr=1250000),
    "SBIN": dict(pe=10.2, pb=1.6, roe=17.1, roce=None, debt_to_equity=None, dividend_yield=1.8, operating_margin=None, net_margin=13.0, revenue_growth=11.5, profit_growth=15.3, eps=79.4, market_cap_cr=720000),
    "ICICIBANK": dict(pe=18.6, pb=3.2, roe=17.9, roce=None, debt_to_equity=None, dividend_yield=0.8, operating_margin=None, net_margin=27.9, revenue_growth=12.8, profit_growth=17.4, eps=65.6, market_cap_cr=850000),
    "SHRIRAMFIN": dict(pe=14.3, pb=2.4, roe=17.5, roce=15.9, debt_to_equity=3.8, dividend_yield=1.4, operating_margin=None, net_margin=24.6, revenue_growth=18.9, profit_growth=21.2, eps=213.0, market_cap_cr=115000),
    "PAYTM": dict(pe=None, pb=4.8, roe=-8.2, roce=-6.1, debt_to_equity=0.05, dividend_yield=0.0, operating_margin=-12.4, net_margin=-9.8, revenue_growth=3.1, profit_growth=None, eps=-9.7, market_cap_cr=45000),
    "TATACOMM": dict(pe=32.7, pb=13.9, roe=45.2, roce=18.7, debt_to_equity=1.9, dividend_yield=1.2, operating_margin=15.6, net_margin=8.9, revenue_growth=5.4, profit_growth=None, eps=54.2, market_cap_cr=51000),
    "RELIANCE": dict(pe=23.9, pb=2.2, roe=9.1, roce=9.8, debt_to_equity=0.4, dividend_yield=0.4, operating_margin=17.0, net_margin=7.8, revenue_growth=7.3, profit_growth=6.5, eps=55.6, market_cap_cr=1800000),
    "HCLTECH": dict(pe=24.6, pb=6.8, roe=27.6, roce=33.1, debt_to_equity=0.08, dividend_yield=3.4, operating_margin=19.8, net_margin=16.1, revenue_growth=6.1, profit_growth=7.2, eps=72.3, market_cap_cr=480000),
    "BHARTIARTL": dict(pe=41.2, pb=9.6, roe=24.3, roce=13.5, debt_to_equity=1.6, dividend_yield=0.5, operating_margin=44.0, net_margin=13.9, revenue_growth=14.6, profit_growth=None, eps=40.0, market_cap_cr=950000),
    "LT": dict(pe=31.5, pb=4.9, roe=15.8, roce=14.2, debt_to_equity=1.1, dividend_yield=0.8, operating_margin=11.2, net_margin=6.8, revenue_growth=16.3, profit_growth=12.1, eps=112.6, market_cap_cr=490000),
    "AXISBANK": dict(pe=13.9, pb=2.1, roe=16.2, roce=None, debt_to_equity=None, dividend_yield=0.1, operating_margin=None, net_margin=23.1, revenue_growth=13.9, profit_growth=None, eps=81.3, market_cap_cr=350000),
    "KOTAKBANK": dict(pe=19.4, pb=2.8, roe=14.6, roce=None, debt_to_equity=None, dividend_yield=0.1, operating_margin=None, net_margin=28.4, revenue_growth=12.1, profit_growth=13.5, eps=91.8, market_cap_cr=354000),
    "BAJFINANCE": dict(pe=32.1, pb=6.2, roe=20.4, roce=None, debt_to_equity=3.6, dividend_yield=0.4, operating_margin=None, net_margin=24.8, revenue_growth=27.3, profit_growth=18.6, eps=221.0, market_cap_cr=440000),
    "ITC": dict(pe=24.8, pb=6.4, roe=27.1, roce=35.4, debt_to_equity=0.02, dividend_yield=3.2, operating_margin=34.5, net_margin=27.8, revenue_growth=5.6, profit_growth=4.9, eps=18.7, market_cap_cr=580000),
    "TATAMOTORS": dict(pe=9.8, pb=3.4, roe=34.6, roce=17.9, debt_to_equity=1.2, dividend_yield=0.4, operating_margin=9.6, net_margin=6.1, revenue_growth=8.9, profit_growth=None, eps=98.1, market_cap_cr=355000),
    "SUNPHARMA": dict(pe=34.2, pb=6.7, roe=17.8, roce=19.3, debt_to_equity=0.06, dividend_yield=0.7, operating_margin=25.9, net_margin=19.4, revenue_growth=9.1, profit_growth=10.4, eps=41.6, market_cap_cr=428000),
    "MARUTI": dict(pe=27.6, pb=4.1, roe=15.2, roce=18.4, debt_to_equity=0.0, dividend_yield=1.0, operating_margin=10.8, net_margin=8.5, revenue_growth=10.7, profit_growth=13.9, eps=447.0, market_cap_cr=380000),
}


@_cache_data(ttl=3600, show_spinner=False)
def get_fundamentals(symbol: str) -> dict[str, Any]:
    demo = dict(DEMO_FUNDAMENTALS.get(symbol, {}))
    if not IS_DEMO_MODE and HAS_YFINANCE:
        try:
            info = yf.Ticker(STOCK_BY_SYMBOL[symbol]["yf_symbol"]).info
            live = {
                "pe": info.get("trailingPE"),
                "pb": info.get("priceToBook"),
                "roe": (info.get("returnOnEquity") or None) and info["returnOnEquity"] * 100,
                "roce": None,  # not exposed by yfinance; left unavailable in LIVE mode
                "debt_to_equity": (info.get("debtToEquity") or None) and info["debtToEquity"] / 100,
                "dividend_yield": (info.get("dividendYield") or None) and info["dividendYield"] * 100,
                "operating_margin": (info.get("operatingMargins") or None) and info["operatingMargins"] * 100,
                "net_margin": (info.get("profitMargins") or None) and info["profitMargins"] * 100,
                "revenue_growth": (info.get("revenueGrowth") or None) and info["revenueGrowth"] * 100,
                "profit_growth": (info.get("earningsGrowth") or None) and info["earningsGrowth"] * 100,
                "eps": info.get("trailingEps"),
                "market_cap_cr": (info.get("marketCap") or None) and info["marketCap"] / 1e7,
            }
            if any(v is not None for v in live.values()):
                live["source"] = "LIVE"
                return live
        except Exception:
            pass
    demo["source"] = "DEMO"
    return demo


# --- News / sentiment headlines ---------------------------------------------
# DEMO headlines are a small hand-curated bank of realistic, generic
# finance-news-style sentences per sector, deterministically assigned per
# stock + week so the sentiment feed looks alive without ever claiming to
# be real-time news. Clearly labeled "Demo Data" in the UI.
_POSITIVE_TEMPLATES = [
    "{name} posts better-than-expected quarterly results, beats analyst estimates",
    "{name} shares rally as brokerages raise target price",
    "{name} announces strong order book and expansion plans",
    "Analysts upgrade {name} citing improving margins",
    "{name} wins large new contract, stock gains on the news",
    "{name} declares higher dividend, investors cheer",
    "Foreign institutional investors increase stake in {name}",
]
_NEGATIVE_TEMPLATES = [
    "{name} misses revenue estimates, stock under pressure",
    "{name} faces regulatory scrutiny over compliance issue",
    "Brokerage downgrades {name} on margin concerns",
    "{name} flags rising input costs, margins under pressure",
    "{name} shares slide amid weak sector outlook",
    "{name} guidance disappoints investors for the coming quarter",
]
_NEUTRAL_TEMPLATES = [
    "{name} to announce quarterly results next week",
    "{name} holds annual general meeting, reviews strategy",
    "{name} in line with sector peers amid mixed market sentiment",
    "Analysts maintain hold rating on {name} pending more clarity",
    "{name} completes routine board meeting, no major announcements",
]
_NEWS_SOURCES = ["Economic Times", "Moneycontrol", "Business Standard", "LiveMint", "Reuters India"]


@_cache_data(ttl=3600, show_spinner=False)
def get_news_headlines(symbol: str, n: int = 8) -> list[dict[str, Any]]:
    """Return recent (demo) headlines for `symbol`. In LIVE mode this would
    call a configured news provider using NEWS_API_KEY; since no key is
    configured by default the app clearly falls back to demo headlines
    labeled as such rather than fabricating a live feed."""
    meta = STOCK_BY_SYMBOL[symbol]
    rng = np.random.default_rng(_seed_from_symbol(symbol + "_news"))
    items = []
    today = datetime.now()
    for i in range(n):
        bucket = rng.choice(["pos", "neg", "neu"], p=[0.42, 0.24, 0.34])
        template = rng.choice({"pos": _POSITIVE_TEMPLATES, "neg": _NEGATIVE_TEMPLATES,
                                "neu": _NEUTRAL_TEMPLATES}[bucket])
        headline = template.format(name=meta["name"].split(" Limited")[0])
        items.append({
            "headline": headline,
            "source": str(rng.choice(_NEWS_SOURCES)),
            "date": (today - timedelta(days=int(rng.integers(0, 14)))).strftime("%Y-%m-%d"),
            "label": bucket,
            "is_demo": True,
        })
    items.sort(key=lambda x: x["date"], reverse=True)
    return items


# =============================================================================
# SECTION 4 — TECHNICAL ANALYSIS (indicators + 0-100 technical score)
# =============================================================================

def compute_indicators(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the full technical-indicator set on a price DataFrame with
    columns open/high/low/close/volume. Returns a new DataFrame with all
    indicator columns appended. Every indicator uses only data up to and
    including the current row (no look-ahead)."""
    out = price_df.copy()
    close, high, low, volume = out["close"], out["high"], out["low"], out["volume"]

    out["sma_20"] = close.rolling(20).mean()
    out["sma_50"] = close.rolling(50).mean()
    out["sma_100"] = close.rolling(100).mean()
    out["sma_200"] = close.rolling(200).mean()
    out["ema_20"] = close.ewm(span=20, adjust=False).mean()
    out["ema_50"] = close.ewm(span=50, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi_14"] = (100 - (100 / (1 + rs))).fillna(50.0)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    out["bb_upper"] = bb_mid + 2 * bb_std
    out["bb_lower"] = bb_mid - 2 * bb_std
    out["bb_mid"] = bb_mid
    bb_width = (out["bb_upper"] - out["bb_lower"]).replace(0, np.nan)
    out["bollinger_position"] = (close - out["bb_lower"]) / bb_width

    prev_close = close.shift(1)
    true_range = pd.concat([
        (high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()
    ], axis=1).max(axis=1)
    out["atr_14"] = true_range.ewm(alpha=1 / 14, adjust=False).mean()

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=out.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=out.index)
    tr_ewm = true_range.ewm(alpha=1 / 14, adjust=False).mean().replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / tr_ewm
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / tr_ewm
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    out["plus_di"] = plus_di
    out["minus_di"] = minus_di
    out["adx_14"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    low14, high14 = low.rolling(14).min(), high.rolling(14).max()
    out["stoch_k"] = 100 * (close - low14) / (high14 - low14).replace(0, np.nan)
    out["stoch_d"] = out["stoch_k"].rolling(3).mean()

    direction = np.sign(close.diff().fillna(0))
    out["obv"] = (direction * volume).fillna(0).cumsum()

    out["momentum_10"] = close - close.shift(10)
    out["return_1d"] = close.pct_change(1)
    out["return_5d"] = close.pct_change(5)
    out["return_20d"] = close.pct_change(20)
    out["volatility_20d"] = out["return_1d"].rolling(20).std() * math.sqrt(252)
    out["volume_change"] = volume.pct_change(1)
    out["volume_sma_20"] = volume.rolling(20).mean()

    out["high_52w"] = high.rolling(252, min_periods=20).max()
    out["low_52w"] = low.rolling(252, min_periods=20).min()
    out["dist_from_52w_high"] = (close - out["high_52w"]) / out["high_52w"]
    out["dist_from_52w_low"] = (close - out["low_52w"]) / out["low_52w"]

    return out


def _clip(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return float(max(lo, min(hi, x)))


def _rsi_component_score(rsi: float) -> float:
    if pd.isna(rsi):
        return 50.0
    if rsi >= 80:
        return 30.0
    if rsi >= 70:
        return 55.0
    if rsi >= 55:
        return 85.0
    if rsi >= 45:
        return 70.0
    if rsi >= 30:
        return 40.0
    return 20.0


def technical_score(ind_df: pd.DataFrame) -> dict[str, Any]:
    """Compute a 0-100 technical score with a component breakdown from the
    most recent row of an indicator DataFrame (see compute_indicators)."""
    if len(ind_df) == 0:
        return {"overall": 50.0, "components": {}, "signals": [], "raw": {}}
    row = ind_df.iloc[-1]
    signals: list[str] = []

    rsi_score = _rsi_component_score(row["rsi_14"])
    if row["rsi_14"] >= 70:
        signals.append(f"RSI at {row['rsi_14']:.0f} — approaching overbought territory")
    elif row["rsi_14"] <= 30:
        signals.append(f"RSI at {row['rsi_14']:.0f} — oversold, potential reversal zone")
    else:
        signals.append(f"RSI at {row['rsi_14']:.0f} — neutral-to-healthy momentum")

    macd_hist_pct = 0.0 if row["close"] == 0 or pd.isna(row["macd_hist"]) else row["macd_hist"] / row["close"]
    macd_score = _clip(50 + macd_hist_pct * 4000)
    if row["macd"] > row["macd_signal"]:
        signals.append("MACD above signal line — bullish crossover in effect")
    else:
        signals.append("MACD below signal line — bearish crossover in effect")

    ma_checks = [
        row["close"] > row["sma_20"] if not pd.isna(row["sma_20"]) else None,
        row["sma_20"] > row["sma_50"] if not pd.isna(row["sma_50"]) else None,
        row["sma_50"] > row["sma_100"] if not pd.isna(row["sma_100"]) else None,
        row["sma_100"] > row["sma_200"] if not pd.isna(row["sma_200"]) else None,
        row["close"] > row["sma_200"] if not pd.isna(row["sma_200"]) else None,
    ]
    valid_checks = [c for c in ma_checks if c is not None]
    ma_score = (sum(valid_checks) / len(valid_checks) * 100) if valid_checks else 50.0
    if row["close"] > row.get("sma_50", np.nan):
        signals.append("Price trading above its 50-day moving average")
    if row["close"] < row.get("sma_50", np.nan):
        signals.append("Price trading below its 50-day moving average")

    ret20 = row["return_20d"] if not pd.isna(row["return_20d"]) else 0.0
    momentum_score = _clip(50 + ret20 * 300)
    if ret20 > 0.03:
        signals.append(f"Strong positive 20-day momentum ({ret20*100:.1f}%)")
    elif ret20 < -0.03:
        signals.append(f"Negative 20-day momentum ({ret20*100:.1f}%)")

    vol_ratio = (row["volume"] / row["volume_sma_20"]) if row.get("volume_sma_20", 0) else 1.0
    vol_ratio = 1.0 if pd.isna(vol_ratio) else vol_ratio
    volume_score = _clip(50 + (vol_ratio - 1) * 60)
    if vol_ratio > 1.3:
        signals.append("Volume significantly above its 20-day average — strong participation")

    components = {
        "rsi": round(rsi_score, 1),
        "macd": round(macd_score, 1),
        "moving_average": round(ma_score, 1),
        "momentum": round(momentum_score, 1),
        "volume": round(volume_score, 1),
    }
    overall = round(sum(components.values()) / len(components), 1)

    raw = {
        "rsi_14": row["rsi_14"], "macd": row["macd"], "macd_signal": row["macd_signal"],
        "sma_20": row["sma_20"], "sma_50": row["sma_50"], "sma_100": row["sma_100"], "sma_200": row["sma_200"],
        "ema_20": row["ema_20"], "ema_50": row["ema_50"], "atr_14": row["atr_14"], "adx_14": row["adx_14"],
        "stoch_k": row["stoch_k"], "stoch_d": row["stoch_d"], "obv": row["obv"],
        "bollinger_position": row["bollinger_position"], "volatility_20d": row["volatility_20d"],
        "volume_change": row["volume_change"], "high_52w": row["high_52w"], "low_52w": row["low_52w"],
        "dist_from_52w_high": row["dist_from_52w_high"], "dist_from_52w_low": row["dist_from_52w_low"],
        "momentum_10": row["momentum_10"], "return_1d": row["return_1d"], "return_5d": row["return_5d"],
        "return_20d": row["return_20d"], "close": row["close"],
    }
    return {"overall": overall, "components": components, "signals": signals, "raw": raw}


# =============================================================================
# SECTION 5 — FUNDAMENTAL ANALYSIS (0-100 score, never fabricates data)
# =============================================================================

def _band_score(value: float | None, bands: list[tuple[float, float]], higher_is_better: bool = True) -> float | None:
    """bands: list of (threshold, score) sorted best-to-worst threshold order
    matching higher_is_better. Returns None if value is None (never invents)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    for threshold, score in bands:
        if (higher_is_better and value >= threshold) or (not higher_is_better and value <= threshold):
            return score
    return bands[-1][1]


def fundamental_score(fund: dict[str, Any]) -> dict[str, Any]:
    """Score only the fundamentals that are actually available. Missing
    fields are never invented -- they are excluded from the score and
    surfaced to the UI as 'Data unavailable'."""
    specs = {
        "pe": ("P/E Ratio", [(15, 90), (25, 72), (35, 50)], False),
        "pb": ("P/B Ratio", [(2, 85), (4, 65), (7, 45)], False),
        "roe": ("Return on Equity", [(20, 90), (15, 72), (10, 50)], True),
        "roce": ("Return on Capital Employed", [(20, 90), (15, 72), (10, 50)], True),
        "debt_to_equity": ("Debt-to-Equity", [(0.3, 90), (1.0, 70), (2.0, 45)], False),
        "dividend_yield": ("Dividend Yield", [(2.5, 80), (1.0, 65), (0.0, 50)], True),
        "operating_margin": ("Operating Margin", [(25, 88), (15, 70), (8, 50)], True),
        "net_margin": ("Net Margin", [(18, 88), (10, 70), (5, 50)], True),
        "revenue_growth": ("Revenue Growth", [(15, 88), (8, 70), (0, 50)], True),
        "profit_growth": ("Profit Growth", [(15, 88), (8, 70), (0, 50)], True),
    }
    components: dict[str, float] = {}
    labels: dict[str, str] = {}
    for key, (label, bands, higher_better) in specs.items():
        score = _band_score(fund.get(key), bands, higher_better)
        labels[key] = label
        if score is not None:
            components[key] = round(score, 1)

    available_ratio = len(components) / len(specs)
    if components:
        overall = round(sum(components.values()) / len(components), 1)
    else:
        overall = None

    strengths = [f"{labels[k]} ({fund.get(k)})" for k, v in components.items() if v >= 72]
    weaknesses = [f"{labels[k]} ({fund.get(k)})" for k, v in components.items() if v <= 45]

    pe = fund.get("pe")
    if pe is None:
        valuation = "Data unavailable"
    elif pe < 15:
        valuation = "Appears undervalued relative to typical bands"
    elif pe <= 30:
        valuation = "Fairly valued"
    else:
        valuation = "Appears expensive / growth-priced"

    return {
        "overall": overall,
        "components": components,
        "labels": labels,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "valuation": valuation,
        "data_availability_pct": round(available_ratio * 100, 0),
        "low_confidence": available_ratio < 0.5,
    }


# =============================================================================
# SECTION 6 — SENTIMENT ANALYSIS (finance-lexicon NLP, FinBERT-style scoring)
# =============================================================================
# A lightweight, dependency-free, finance-oriented lexicon sentiment
# scorer. This stands in for a heavier transformer model (e.g. FinBERT)
# so the app has zero heavyweight ML-download requirements; the scoring
# approach (weighted keyword polarity + negation handling) is the same
# family of technique used by classic lexicon-based finance NLP (e.g.
# Loughran-McDonald word lists).

_POS_WORDS = {
    "beat", "beats", "beating", "rally", "rallies", "gain", "gains", "gained", "surge", "surged",
    "upgrade", "upgrades", "upgraded", "strong", "growth", "profit", "profits", "record", "expansion",
    "wins", "won", "bullish", "outperform", "higher", "rise", "rises", "rising", "positive", "boost",
    "cheer", "cheers", "improving", "improve", "improved", "robust", "exceed", "exceeds", "exceeded",
}
_NEG_WORDS = {
    "miss", "misses", "missed", "downgrade", "downgrades", "downgraded", "fall", "falls", "falling",
    "fell", "decline", "declines", "declined", "slide", "slides", "weak", "loss", "losses", "concern",
    "concerns", "scrutiny", "pressure", "disappoint", "disappoints", "disappointing", "bearish",
    "underperform", "lower", "cut", "cuts", "regulatory", "risk", "risks", "provisioning", "slump",
}
_NEGATIONS = {"not", "no", "never", "without"}


def score_headline_sentiment(text: str) -> tuple[str, float]:
    """Return (label, polarity) where polarity in [-1, 1]. Simple bag-of-
    words polarity with a one-token negation flip -- transparent and
    auditable, unlike a black-box model, which matters for an explainable
    finance application."""
    tokens = [t.strip(".,()'\"").lower() for t in text.split()]
    score = 0
    for i, tok in enumerate(tokens):
        polarity = 1 if tok in _POS_WORDS else (-1 if tok in _NEG_WORDS else 0)
        if polarity == 0:
            continue
        if i > 0 and tokens[i - 1] in _NEGATIONS:
            polarity *= -1
        score += polarity
    if score > 0:
        return "positive", min(1.0, score / 3)
    if score < 0:
        return "negative", max(-1.0, score / 3)
    return "neutral", 0.0


def analyze_sentiment(headlines: list[dict[str, Any]]) -> dict[str, Any]:
    """Run the lexicon scorer over each headline's actual text (ignoring
    any pre-assigned demo label) and aggregate into a 0-100 sentiment
    score plus positive/neutral/negative percentages."""
    if not headlines:
        return {"overall": 50.0, "positive_pct": 0.0, "neutral_pct": 0.0, "negative_pct": 0.0, "scored_headlines": []}
    scored = []
    counts = {"positive": 0, "neutral": 0, "negative": 0}
    polarity_sum = 0.0
    for h in headlines:
        label, polarity = score_headline_sentiment(h["headline"])
        counts[label] += 1
        polarity_sum += polarity
        scored.append({**h, "sentiment": label, "polarity": round(polarity, 2)})
    n = len(headlines)
    avg_polarity = polarity_sum / n
    overall = round(_clip(50 + avg_polarity * 50), 1)
    return {
        "overall": overall,
        "positive_pct": round(100 * counts["positive"] / n, 1),
        "neutral_pct": round(100 * counts["neutral"] / n, 1),
        "negative_pct": round(100 * counts["negative"] / n, 1),
        "scored_headlines": scored,
    }


# =============================================================================
# SECTION 7 — FEATURE ENGINEERING FOR MACHINE LEARNING
# =============================================================================
# Every feature below is computable using only information available up
# to and including day t (rolling windows, no forward shifts). This is
# the leakage-prevention contract for the whole ML pipeline -- see
# build_stock_dataset() for how the target is kept strictly separate.

FEATURE_COLUMNS = [
    "return_1d", "return_5d", "return_20d", "volatility_20d",
    "rsi_14", "macd_norm", "macd_hist_norm", "sma_ratio", "ema_ratio",
    "volume_change", "atr_norm", "adx_14", "bollinger_position",
    "momentum_norm", "dist_from_52w_high", "dist_from_52w_low",
    "stoch_k", "market_relative_return_20d",
]


def build_stock_dataset(symbol: str, horizon: int = TARGET_HORIZON_DAYS) -> pd.DataFrame:
    """Build a leakage-safe, per-day feature+target table for one stock.

    Target definition (matches docs/ml_methodology intent): a stock is
    labeled 1 ("outperform") if, over the next `horizon` trading days, its
    return exceeds the benchmark's return by more than the round-trip
    transaction cost -- i.e. a positive, cost-adjusted, *market-relative*
    (risk-adjusted) return. Because the label uses `close.shift(-horizon)`,
    it is computed once here and then never used as an input feature.
    """
    price = get_price_history(symbol)
    ind = compute_indicators(price)
    bench = compute_indicators(get_benchmark_history())

    df = ind.copy()
    df["macd_norm"] = df["macd"] / df["close"]
    df["macd_hist_norm"] = df["macd_hist"] / df["close"]
    df["sma_ratio"] = df["close"] / df["sma_50"]
    df["ema_ratio"] = df["ema_20"] / df["ema_50"]
    df["atr_norm"] = df["atr_14"] / df["close"]
    df["momentum_norm"] = df["momentum_10"] / df["close"]

    bench_ret20 = bench["return_20d"].reindex(df.index)
    df["market_relative_return_20d"] = df["return_20d"] - bench_ret20

    future_close = df["close"].shift(-horizon)
    future_return = future_close / df["close"] - 1
    bench_future_close = bench["close"].reindex(df.index).shift(-horizon)
    bench_future_return = bench_future_close / bench["close"].reindex(df.index) - 1
    excess_future_return = future_return - bench_future_return

    round_trip_cost = 2 * (TRANSACTION_COST_BPS / 10000)
    df["future_excess_return"] = excess_future_return
    # IMPORTANT (leakage/correctness guard): a plain `series > threshold`
    # comparison silently turns NaN into False in pandas/numpy, which would
    # mislabel the final `horizon` rows (whose future is genuinely unknown)
    # as a confident "underperform" instead of leaving them undefined. We
    # must preserve NaN explicitly so those rows get dropped downstream.
    raw_target = (excess_future_return > round_trip_cost).astype(float)
    raw_target[excess_future_return.isna()] = np.nan
    df["target"] = raw_target.astype("Int64")

    df["symbol"] = symbol
    df["date"] = df.index

    keep_cols = FEATURE_COLUMNS + ["target", "future_excess_return", "symbol", "date", "close"]
    result = df[keep_cols].copy()
    return result


@_cache_data(ttl=3600, show_spinner=False)
def build_training_dataset(horizon: int = TARGET_HORIZON_DAYS) -> pd.DataFrame:
    """Concatenate per-stock datasets across the full 20-stock universe,
    then drop rows where features aren't yet computable (early rows,
    e.g. before a 200-day SMA exists) or the target isn't yet knowable
    (the final `horizon` rows of history, where the future window falls
    beyond the last available price)."""
    frames = [build_stock_dataset(sym, horizon) for sym in SYMBOLS]
    full = pd.concat(frames, ignore_index=True)
    full = full.dropna(subset=FEATURE_COLUMNS + ["target"]).reset_index(drop=True)
    full["target"] = full["target"].astype(int)
    return full


def chronological_split(dataset: pd.DataFrame,
                         train_end_year: int = TRAIN_END_YEAR,
                         val_end_year: int = VALIDATION_END_YEAR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Strict chronological split -- NEVER shuffle financial time series.
    Train: up to and including train_end_year. Validation: train_end_year+1
    .. val_end_year. Test (fully held out): everything after val_end_year."""
    years = pd.to_datetime(dataset["date"]).dt.year
    train = dataset[years <= train_end_year]
    val = dataset[(years > train_end_year) & (years <= val_end_year)]
    test = dataset[years > val_end_year]
    return train, val, test


# =============================================================================
# SECTION 8 — MODEL TRAINING & EVALUATION
# =============================================================================

def _compute_metrics(y_true, y_pred, y_prob) -> dict[str, float]:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4) if len(set(y_true)) > 1 else float("nan"),
        "pr_auc": round(average_precision_score(y_true, y_prob), 4) if len(set(y_true)) > 1 else float("nan"),
    }


@_cache_resource(show_spinner="Training ML models (Logistic Regression, Random Forest, XGBoost)...")
def train_and_evaluate_models(horizon: int = TARGET_HORIZON_DAYS) -> dict[str, Any]:
    """Train all candidate models on the chronological TRAIN split, select
    the best by VALIDATION ROC-AUC (documented criterion — not
    train-set performance, and not the held-out test set, to avoid test
    leakage into model selection), then report final metrics on the fully
    held-out TEST split. Runs once per Streamlit session (cached)."""
    dataset = build_training_dataset(horizon)
    train, val, test = chronological_split(dataset)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[FEATURE_COLUMNS])  # fit ONLY on train
    X_val = scaler.transform(val[FEATURE_COLUMNS])
    X_test = scaler.transform(test[FEATURE_COLUMNS])
    y_train, y_val, y_test = train["target"].values, val["target"].values, test["target"].values

    candidates: dict[str, Any] = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced",
                                                    random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=25,
                                                  class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1),
    }
    if HAS_XGBOOST:
        candidates["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=RANDOM_SEED, eval_metric="logloss",
            reg_lambda=1.0, n_jobs=-1,
        )

    trained: dict[str, Any] = {}
    val_metrics: dict[str, dict] = {}
    test_metrics: dict[str, dict] = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        val_prob = model.predict_proba(X_val)[:, 1]
        test_prob = model.predict_proba(X_test)[:, 1]
        val_metrics[name] = _compute_metrics(y_val, (val_prob >= 0.5).astype(int), val_prob)
        test_metrics[name] = _compute_metrics(y_test, (test_prob >= 0.5).astype(int), test_prob)
        trained[name] = model

    best_name = max(val_metrics, key=lambda n: (val_metrics[n]["roc_auc"] if not math.isnan(val_metrics[n]["roc_auc"]) else -1))

    # Backtest buy-threshold is calibrated from the VALIDATION set's
    # probability distribution (never the test set) so the walk-forward
    # backtest has no look-ahead into its own held-out period.
    best_val_prob = trained[best_name].predict_proba(X_val)[:, 1]
    suggested_buy_threshold = float(np.clip(np.percentile(best_val_prob, 55), 0.50, 0.62))

    best_test_prob = trained[best_name].predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, best_test_prob) if len(set(y_test)) > 1 else ([0, 1], [0, 1], [0])
    cm = confusion_matrix(y_test, (best_test_prob >= 0.5).astype(int)).tolist()

    return {
        "models": trained,
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "best_model_name": best_name,
        "selection_criterion": "Highest ROC-AUC on the chronological VALIDATION split",
        "roc_curve": {"fpr": list(fpr), "tpr": list(tpr)},
        "confusion_matrix": cm,
        "train_rows": len(train), "val_rows": len(val), "test_rows": len(test),
        "train_years": sorted(pd.to_datetime(train["date"]).dt.year.unique().tolist()),
        "val_years": sorted(pd.to_datetime(val["date"]).dt.year.unique().tolist()),
        "test_years": sorted(pd.to_datetime(test["date"]).dt.year.unique().tolist()),
        "test_positive_rate": round(float(np.mean(y_test)), 4) if len(y_test) else float("nan"),
        "suggested_buy_threshold": round(suggested_buy_threshold, 4),
        "X_test": X_test, "y_test": y_test, "test_frame": test.reset_index(drop=True),
    }


def predict_latest(symbol: str, artifacts: dict[str, Any]) -> dict[str, Any]:
    """Score the most recent (current) feature row for `symbol` with the
    selected best model. This uses live/current indicator values only --
    the target horizon window for 'today' is in the future and therefore
    unknown, exactly as it would be in real deployment."""
    ds = build_stock_dataset(symbol)
    latest = ds.dropna(subset=FEATURE_COLUMNS).iloc[-1:]
    if latest.empty:
        return {"probability": 0.5, "row": None}
    X = artifacts["scaler"].transform(latest[FEATURE_COLUMNS])
    model = artifacts["models"][artifacts["best_model_name"]]
    prob = float(model.predict_proba(X)[:, 1][0])
    return {"probability": prob, "row": latest.iloc[0], "features": latest[FEATURE_COLUMNS].iloc[0].to_dict()}


# =============================================================================
# SECTION 9 — EXPLAINABLE AI (SHAP for tree models, coefficient-based for LR)
# =============================================================================

@_cache_resource(show_spinner=False)
def _get_shap_explainer(_model, model_name: str, background):
    if not HAS_SHAP:
        return None
    try:
        if model_name in ("Random Forest", "XGBoost"):
            return shap.TreeExplainer(_model)
        return shap.LinearExplainer(_model, background)
    except Exception:
        return None


def explain_prediction(symbol: str, artifacts: dict[str, Any], top_k: int = 5) -> dict[str, Any]:
    """Return the top positive and negative factors driving the current
    prediction for `symbol`, using SHAP when available, and a transparent
    coefficient/importance-based fallback otherwise (so the app degrades
    gracefully rather than losing explainability entirely)."""
    pred = predict_latest(symbol, artifacts)
    if pred["row"] is None:
        return {"positive": [], "negative": [], "method": "unavailable"}

    feat_row = pred["row"][FEATURE_COLUMNS].astype(float)
    X_row = artifacts["scaler"].transform(feat_row.values.reshape(1, -1))
    model_name = artifacts["best_model_name"]
    model = artifacts["models"][model_name]

    contributions: dict[str, float] = {}
    method = "SHAP"
    if HAS_SHAP:
        explainer = _get_shap_explainer(model, model_name, artifacts["X_test"][:200])
        if explainer is not None:
            try:
                sv = explainer.shap_values(X_row)
                if isinstance(sv, list):  # binary classifiers may return [class0, class1]
                    sv = sv[1] if len(sv) > 1 else sv[0]
                sv = np.array(sv).reshape(-1)
                contributions = dict(zip(FEATURE_COLUMNS, sv.tolist()))
            except Exception:
                contributions = {}
    if not contributions:
        method = "Model coefficients / feature importance (SHAP unavailable)"
        if hasattr(model, "coef_"):
            coefs = model.coef_.reshape(-1)
            contributions = dict(zip(FEATURE_COLUMNS, (coefs * X_row.reshape(-1)).tolist()))
        elif hasattr(model, "feature_importances_"):
            signs = np.sign(X_row.reshape(-1))
            contributions = dict(zip(FEATURE_COLUMNS,
                                      (model.feature_importances_ * signs).tolist()))

    ranked = sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)
    positive = [{"feature": _humanize_feature(k), "impact": round(v, 4)} for k, v in ranked if v > 0][:top_k]
    negative = [{"feature": _humanize_feature(k), "impact": round(v, 4)} for k, v in ranked if v < 0][-top_k:][::-1]
    return {"positive": positive, "negative": negative, "method": method,
            "probability": pred["probability"]}


_FEATURE_LABELS = {
    "return_1d": "1-day return", "return_5d": "5-day return", "return_20d": "20-day return",
    "volatility_20d": "20-day volatility", "rsi_14": "RSI (14)", "macd_norm": "MACD level",
    "macd_hist_norm": "MACD histogram", "sma_ratio": "Price vs 50-day SMA", "ema_ratio": "20/50-day EMA ratio",
    "volume_change": "Volume change", "atr_norm": "Average True Range", "adx_14": "ADX (trend strength)",
    "bollinger_position": "Bollinger Band position", "momentum_norm": "10-day momentum",
    "dist_from_52w_high": "Distance from 52-week high", "dist_from_52w_low": "Distance from 52-week low",
    "stoch_k": "Stochastic %K", "market_relative_return_20d": "Return vs NIFTY 50 (20d)",
}


def _humanize_feature(key: str) -> str:
    return _FEATURE_LABELS.get(key, key)


# =============================================================================
# SECTION 10 — RISK ANALYSIS
# =============================================================================

RISK_FREE_RATE = 0.07  # approximate long-run Indian risk-free rate (10y G-Sec), annualized


def risk_metrics(price_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> dict[str, Any]:
    returns = price_df["close"].pct_change().dropna()
    if len(returns) < 30:
        return {"risk_score": 50.0, "risk_level": "Moderate", "signals": ["Insufficient history for full risk analysis"]}

    ann_vol = float(returns.std() * math.sqrt(252))
    downside = returns[returns < 0]
    downside_std = float(downside.std() * math.sqrt(252)) if len(downside) > 5 else float("nan")
    ann_return = float(returns.mean() * 252)
    sharpe = (ann_return - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else float("nan")
    sortino = (ann_return - RISK_FREE_RATE) / downside_std if downside_std and downside_std > 0 else float("nan")

    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    max_drawdown = float(drawdown.min())

    var_95 = float(np.percentile(returns, 5))  # 1-day historical VaR at 95% confidence

    bench_returns = benchmark_df["close"].pct_change()
    aligned = pd.concat([returns, bench_returns], axis=1, join="inner").dropna()
    aligned.columns = ["stock", "bench"]
    if len(aligned) > 30 and aligned["bench"].var() > 0:
        beta = float(aligned["stock"].cov(aligned["bench"]) / aligned["bench"].var())
    else:
        beta = None

    vol_component = _clip(ann_vol / 0.55 * 100)
    dd_component = _clip(abs(max_drawdown) / 0.55 * 100)
    beta_component = _clip(50 + (beta - 1) * 50) if beta is not None else 50.0
    var_component = _clip(abs(var_95) / 0.06 * 100)
    risk_score = round((vol_component + dd_component + beta_component + var_component) / 4, 1)

    if risk_score < 35:
        risk_level = "Low"
    elif risk_score <= 65:
        risk_level = "Moderate"
    else:
        risk_level = "High"

    signals = [
        f"Annualized volatility: {ann_vol*100:.1f}%",
        f"Maximum drawdown (observed): {max_drawdown*100:.1f}%",
        f"1-day Value-at-Risk (95%): {var_95*100:.1f}%",
    ]
    if beta is not None:
        signals.append(f"Beta vs NIFTY 50: {beta:.2f}")

    return {
        "annual_volatility": round(ann_vol * 100, 2),
        "sharpe_ratio": round(sharpe, 2) if not math.isnan(sharpe) else None,
        "sortino_ratio": round(sortino, 2) if sortino == sortino else None,
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "var_95_pct": round(var_95 * 100, 2),
        "beta": round(beta, 2) if beta is not None else None,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "signals": signals,
    }


# =============================================================================
# SECTION 11 — RECOMMENDATION ENGINE (transparent, weighted, configurable)
# =============================================================================

def recommendation_engine(ml_probability: float, technical_score_val: float,
                           fundamental_score_val: float | None, sentiment_score_val: float,
                           risk_score_val: float, weights: dict[str, float] = RECO_WEIGHTS) -> dict[str, Any]:
    """Combine the ML model's probability with technical, fundamental,
    sentiment and risk signals into one 0-100 AI Score and a BUY/HOLD/SELL
    call. Weights live in RECO_WEIGHTS (config, not hard-coded logic) so
    the framework can be re-tuned without touching this function. If
    fundamentals are unavailable, that weight is redistributed
    proportionally across the remaining signals rather than inventing a
    fundamental score."""
    w = dict(weights)
    fund_score = fundamental_score_val
    fund_available = fund_score is not None
    if not fund_available:
        w_fund = w.pop("fundamental_score")
        remaining_total = sum(w.values())
        w = {k: v + (v / remaining_total) * w_fund for k, v in w.items()}
        w["fundamental_score"] = 0.0
        fund_score = 50.0  # neutral placeholder, weighted at 0 -- never influences the score

    ml_component = ml_probability * 100
    risk_adj_points = w["risk_adjustment"] * (50 - risk_score_val) / 50 * 100

    base_score = (
        w["ml_prediction"] * ml_component
        + w["technical_score"] * technical_score_val
        + w.get("fundamental_score", 0.0) * fund_score
        + w["sentiment_score"] * sentiment_score_val
    )
    ai_score = round(_clip(base_score + risk_adj_points), 1)

    if ai_score >= RECO_THRESHOLDS["buy_min_score"]:
        recommendation = "BUY"
    elif ai_score <= RECO_THRESHOLDS["sell_max_score"]:
        recommendation = "SELL"
    else:
        recommendation = "HOLD"

    ml_confidence = abs(ml_probability - 0.5) * 2 * 100
    signal_values = [ml_component, technical_score_val, sentiment_score_val]
    if fund_available:
        signal_values.append(fund_score)
    if recommendation == "BUY":
        agree = sum(1 for s in signal_values if s > 55)
    elif recommendation == "SELL":
        agree = sum(1 for s in signal_values if s < 45)
    else:
        agree = sum(1 for s in signal_values if 40 <= s <= 60)
    agreement_pct = agree / len(signal_values) * 100
    confidence = round(_clip(0.6 * ml_confidence + 0.4 * agreement_pct), 1)

    return {
        "ai_score": ai_score,
        "recommendation": recommendation,
        "confidence": confidence,
        "fundamental_available": fund_available,
        "inputs": {
            "ml_probability": round(ml_probability, 4), "ml_component": round(ml_component, 1),
            "technical_score": technical_score_val, "fundamental_score": fund_score if fund_available else None,
            "sentiment_score": sentiment_score_val, "risk_score": risk_score_val,
        },
        "effective_weights": {k: round(v, 3) for k, v in w.items()},
    }


def build_recommendation_explanation(symbol: str, reco: dict, tech: dict, fund: dict,
                                      sent: dict, risk: dict) -> dict[str, list[str]]:
    """Human-readable WHY / RISKS bullets grounded in the actual computed
    signals for this stock -- used by both the stock detail page and the
    'Explain Recommendation' button."""
    why: list[str] = []
    risks: list[str] = []
    r = reco["recommendation"]

    if r in ("BUY", "HOLD") and tech["overall"] >= 60:
        why.extend(tech["signals"][:2])
    if r == "SELL" and tech["overall"] <= 45:
        why.extend(tech["signals"][:2])
    if reco["fundamental_available"] and fund.get("strengths") and r != "SELL":
        why.append(f"Fundamental strengths: {', '.join(fund['strengths'][:2])}")
    if sent["overall"] >= 60:
        why.append(f"News sentiment skews positive ({sent['positive_pct']:.0f}% of recent headlines)")
    if reco["inputs"]["ml_probability"] >= 0.55:
        why.append(f"ML model estimates a {reco['inputs']['ml_probability']*100:.0f}% probability of "
                    f"outperforming the market over the next {TARGET_HORIZON_DAYS} trading days")
    if not why:
        why.append("Signals are mixed with no single dominant driver — see the full score breakdown below")

    if risk["risk_level"] == "High":
        risks.append(f"High risk profile — annualized volatility of {risk.get('annual_volatility', 'N/A')}%")
    if fund.get("weaknesses"):
        risks.append(f"Fundamental concerns: {', '.join(fund['weaknesses'][:2])}")
    if fund.get("valuation") == "Appears expensive / growth-priced":
        risks.append("Valuation looks stretched on trailing P/E")
    if sent["overall"] <= 40:
        risks.append(f"News sentiment skews negative ({sent['negative_pct']:.0f}% of recent headlines)")
    if risk.get("max_drawdown_pct", 0) < -30:
        risks.append(f"Large historical drawdown observed ({risk['max_drawdown_pct']}%)")
    if not risks:
        risks.append("No dominant red flags identified, but all equity investments carry market risk")

    return {"why": why, "risks": risks}


# =============================================================================
# SECTION 12 — BACKTESTING (walk-forward, held-out test period only)
# =============================================================================

def run_backtest(symbol: str, artifacts: dict[str, Any], prob_buy_threshold: float | None = None) -> dict[str, Any] | None:
    """Walk-forward simulation on the TEST period only (years strictly
    after VALIDATION_END_YEAR — never seen during training or model
    selection). At each `horizon`-day rebalance point: take the model's
    probability computed from features known at that point in time,
    go long if it clears `prob_buy_threshold` else stay in cash, apply
    the transaction cost on position changes, and mark the position to
    the stock's actual subsequent return. Compared against a naive
    Buy & Hold on the same stock and on the NIFTY 50 benchmark."""
    horizon = TARGET_HORIZON_DAYS
    threshold = prob_buy_threshold if prob_buy_threshold is not None else artifacts.get("suggested_buy_threshold", 0.55)
    ds = build_stock_dataset(symbol)
    years = pd.to_datetime(ds["date"]).dt.year
    test = ds[years > VALIDATION_END_YEAR].dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    if len(test) < horizon * 3:
        return None

    model = artifacts["models"][artifacts["best_model_name"]]
    X = artifacts["scaler"].transform(test[FEATURE_COLUMNS])
    test = test.assign(prob=model.predict_proba(X)[:, 1])

    bench = get_benchmark_history()
    cost = TRANSACTION_COST_BPS / 10000

    ai_equity, bh_equity, bench_equity, dates = [1.0], [1.0], [1.0], [test["date"].iloc[0]]
    position, trades, wins, periods_in_market = 0, 0, 0, 0

    idxs = list(range(0, len(test) - horizon, horizon))
    for i in idxs:
        entry_price = float(test["close"].iloc[i])
        exit_price = float(test["close"].iloc[i + horizon])
        period_return = exit_price / entry_price - 1

        entry_date, exit_date = test["date"].iloc[i], test["date"].iloc[i + horizon]
        bench_window = bench.loc[(bench.index >= entry_date) & (bench.index <= exit_date)]
        bench_period_return = (bench_window["close"].iloc[-1] / bench_window["close"].iloc[0] - 1) if len(bench_window) > 1 else 0.0

        new_position = 1 if test["prob"].iloc[i] >= threshold else 0
        txn_cost = cost if new_position != position else 0.0
        if txn_cost > 0:
            trades += 1
        strategy_return = period_return * new_position - txn_cost

        if new_position == 1:
            periods_in_market += 1
            if strategy_return > 0:
                wins += 1

        ai_equity.append(ai_equity[-1] * (1 + strategy_return))
        bh_equity.append(bh_equity[-1] * (1 + period_return))
        bench_equity.append(bench_equity[-1] * (1 + bench_period_return))
        dates.append(exit_date)
        position = new_position

    def _perf(equity: list[float]) -> dict[str, float]:
        eq = np.array(equity)
        period_returns = eq[1:] / eq[:-1] - 1
        periods_per_year = 252 / horizon
        total_return = eq[-1] - 1
        n_years = len(period_returns) / periods_per_year
        cagr = (eq[-1] ** (1 / n_years) - 1) if n_years > 0 and eq[-1] > 0 else float("nan")
        vol = period_returns.std() * math.sqrt(periods_per_year)
        sharpe = (period_returns.mean() * periods_per_year - RISK_FREE_RATE) / vol if vol > 0 else float("nan")
        downside = period_returns[period_returns < 0]
        d_vol = downside.std() * math.sqrt(periods_per_year) if len(downside) > 1 else float("nan")
        sortino = (period_returns.mean() * periods_per_year - RISK_FREE_RATE) / d_vol if d_vol and d_vol > 0 else float("nan")
        running_max = np.maximum.accumulate(eq)
        max_dd = float(((eq - running_max) / running_max).min())
        return {
            "total_return_pct": round(total_return * 100, 2),
            "cagr_pct": round(cagr * 100, 2) if cagr == cagr else None,
            "sharpe_ratio": round(sharpe, 2) if sharpe == sharpe else None,
            "sortino_ratio": round(sortino, 2) if sortino == sortino else None,
            "max_drawdown_pct": round(max_dd * 100, 2),
        }

    return {
        "symbol": symbol,
        "dates": [d.strftime("%Y-%m-%d") for d in dates],
        "ai_equity": ai_equity, "buy_hold_equity": bh_equity, "benchmark_equity": bench_equity,
        "ai_strategy": {**_perf(ai_equity), "num_trades": trades,
                        "win_rate_pct": round(100 * wins / periods_in_market, 1) if periods_in_market else None},
        "buy_hold": _perf(bh_equity),
        "benchmark": _perf(bench_equity),
        "rebalance_periods": len(idxs),
        "horizon_days": horizon,
    }


# =============================================================================
# SECTION 13 — PERSISTENCE (users, portfolio holdings, watchlist)
# =============================================================================
# A lightweight local JSON store stands in for the Postgres `users`,
# `portfolio_holdings` and `watchlists` tables from the full architecture.
# Passwords are salted + PBKDF2-HMAC hashed (stdlib only, no plaintext
# ever stored). This is appropriate for a single-file academic/demo app;
# docs/README notes how this would map onto real JWT + Postgres auth in
# a multi-service deployment.

def _default_store() -> dict[str, Any]:
    return {"users": {}, "watchlists": {}, "portfolios": {}}


def load_store() -> dict[str, Any]:
    if STORE_PATH.exists():
        try:
            with open(STORE_PATH, "r") as f:
                data = json.load(f)
            for key in ("users", "watchlists", "portfolios"):
                data.setdefault(key, {})
            return data
        except Exception:
            pass
    return _default_store()


def save_store(store: dict[str, Any]) -> None:
    tmp_path = STORE_PATH.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(store, f, indent=2, default=str)
    os.replace(tmp_path, STORE_PATH)


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000).hex()
    return digest, salt


def verify_password(password: str, digest: str, salt: str) -> bool:
    candidate, _ = hash_password(password, salt)
    return hmac.compare_digest(candidate, digest)


def register_user(store: dict[str, Any], username: str, name: str, email: str, password: str) -> tuple[bool, str]:
    username = username.strip().lower()
    if not username or not password:
        return False, "Username and password are required."
    if username in store["users"]:
        return False, "That username is already taken."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    digest, salt = hash_password(password)
    store["users"][username] = {
        "name": name.strip() or username, "email": email.strip(), "password_hash": digest,
        "salt": salt, "created_at": datetime.now().isoformat(),
        "preferences": {"theme": "dark", "risk_tolerance": "Moderate"},
    }
    store["watchlists"].setdefault(username, [])
    store["portfolios"].setdefault(username, [])
    save_store(store)
    return True, "Account created successfully."


def authenticate_user(store: dict[str, Any], username: str, password: str) -> tuple[bool, str]:
    username = username.strip().lower()
    user = store["users"].get(username)
    if not user or not verify_password(password, user["password_hash"], user["salt"]):
        return False, "Invalid username or password."
    return True, "Logged in."


# =============================================================================
# SECTION 14 — STREAMLIT UI
# =============================================================================
if HAS_STREAMLIT:

    THEME_CSS = """
    <style>
    .stApp { background: linear-gradient(180deg, #0b0f19 0%, #0e1424 100%); }
    #MainMenu, footer {visibility: hidden;}
    .block-container { padding-top: 1.5rem; max-width: 1400px; }
    h1, h2, h3, h4 { font-family: 'Segoe UI', system-ui, sans-serif; letter-spacing: -0.02em; }
    .glass-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px; padding: 1.1rem 1.3rem; backdrop-filter: blur(10px);
        box-shadow: 0 4px 24px rgba(0,0,0,0.25); margin-bottom: 0.9rem;
    }
    .glass-card:hover { border-color: rgba(99,179,237,0.4); transition: 0.2s ease; }
    .badge { display:inline-block; padding: 3px 12px; border-radius: 999px; font-weight:700; font-size:0.78rem; letter-spacing:0.03em;}
    .badge-buy { background: rgba(34,197,94,0.18); color:#4ade80; border:1px solid rgba(74,222,128,0.4);}
    .badge-sell { background: rgba(239,68,68,0.18); color:#f87171; border:1px solid rgba(248,113,113,0.4);}
    .badge-hold { background: rgba(234,179,8,0.18); color:#facc15; border:1px solid rgba(250,204,21,0.4);}
    .badge-demo { background: rgba(99,179,237,0.15); color:#63b3ed; border:1px solid rgba(99,179,237,0.35); font-size:0.68rem; padding:2px 9px;}
    .risk-low {color:#4ade80; font-weight:600;} .risk-moderate {color:#facc15; font-weight:600;} .risk-high {color:#f87171; font-weight:600;}
    .metric-big { font-size: 1.9rem; font-weight: 800; }
    .subtle { color: rgba(255,255,255,0.55); font-size:0.85rem; }
    .disclaimer-box { background: rgba(250,204,21,0.06); border:1px solid rgba(250,204,21,0.25);
        border-radius: 12px; padding: 0.8rem 1rem; font-size: 0.82rem; color: rgba(255,255,255,0.75); }
    .hero-title { font-size: 2.8rem; font-weight: 800; background: linear-gradient(90deg,#63b3ed,#a78bfa);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    div[data-testid="stMetricValue"] { font-weight: 700; }
    </style>
    """

    def inject_css() -> None:
        st.markdown(THEME_CSS, unsafe_allow_html=True)

    def mode_badge() -> str:
        return "LIVE" if not IS_DEMO_MODE else "DEMO DATA"

    def recommendation_badge_html(rec: str) -> str:
        cls = {"BUY": "badge-buy", "SELL": "badge-sell", "HOLD": "badge-hold"}[rec]
        return f'<span class="badge {cls}">{rec}</span>'

    def risk_class(level: str) -> str:
        return {"Low": "risk-low", "Moderate": "risk-moderate", "High": "risk-high"}[level]

    def disclaimer_footer() -> None:
        st.markdown(f'<div class="disclaimer-box">⚠️ {DISCLAIMER}</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------
    # Snapshot: bundles every computed signal for one stock (cheap enough
    # to compute for all 20 stocks on every dashboard render; SHAP
    # explanation is deliberately excluded here and computed on-demand
    # only on the stock detail page, since it's the more expensive step).
    # -------------------------------------------------------------------
    @st.cache_data(ttl=300, show_spinner=False)
    def get_stock_snapshot(symbol: str, _artifacts: dict[str, Any]) -> dict[str, Any]:
        meta = STOCK_BY_SYMBOL[symbol]
        price = get_price_history(symbol)
        bench = get_benchmark_history()
        ind = compute_indicators(price)
        tech = technical_score(ind)
        fund_raw = get_fundamentals(symbol)
        fund = fundamental_score(fund_raw)
        news = get_news_headlines(symbol)
        sent = analyze_sentiment(news)
        risk = risk_metrics(price, bench)
        pred = predict_latest(symbol, _artifacts)
        reco = recommendation_engine(pred["probability"], tech["overall"], fund["overall"],
                                      sent["overall"], risk["risk_score"])
        latest, prev = price["close"].iloc[-1], price["close"].iloc[-2]
        change_pct = (latest / prev - 1) * 100
        return {
            "symbol": symbol, "meta": meta, "current_price": float(latest), "change_pct": float(change_pct),
            "data_source": price["source"].iloc[-1], "tech": tech, "fund_raw": fund_raw, "fund": fund,
            "news": news, "sentiment": sent, "risk": risk, "prediction": pred, "reco": reco,
        }

    def get_all_snapshots(_artifacts: dict[str, Any]) -> dict[str, dict]:
        return {sym: get_stock_snapshot(sym, _artifacts) for sym in SYMBOLS}

    # -------------------------------------------------------------------
    # Sidebar: branding, mode badge, navigation, auth widget
    # -------------------------------------------------------------------
    NAV_PAGES = ["Landing", "Dashboard", "Stock Detail", "Compare Stocks", "Portfolio",
                 "Watchlist", "Model Performance", "AI Assistant"]

    def go_to(page: str, symbol: str | None = None) -> None:
        st.session_state.page = page
        if symbol:
            st.session_state.selected_symbol = symbol

    def render_sidebar(store: dict[str, Any]) -> None:
        with st.sidebar:
            st.markdown("### 📈 AI Stock Intelligence")
            st.caption("Indian Equity Research Platform")
            st.markdown(f'<span class="badge badge-demo">{mode_badge()}</span>', unsafe_allow_html=True)
            st.divider()

            page = st.radio("Navigate", NAV_PAGES, index=NAV_PAGES.index(st.session_state.get("page", "Landing")),
                             label_visibility="collapsed")
            if page != st.session_state.get("page"):
                st.session_state.page = page
                st.rerun()

            st.divider()
            if st.session_state.get("user"):
                user = store["users"].get(st.session_state.user, {})
                st.success(f"Signed in as **{user.get('name', st.session_state.user)}**")
                if st.button("Log out", use_container_width=True):
                    st.session_state.user = None
                    st.rerun()
            else:
                with st.expander("🔐 Log in / Register", expanded=False):
                    tab_login, tab_register = st.tabs(["Log in", "Register"])
                    with tab_login:
                        u = st.text_input("Username", key="login_user")
                        p = st.text_input("Password", type="password", key="login_pass")
                        if st.button("Log in", key="login_btn", use_container_width=True):
                            ok, msg = authenticate_user(store, u, p)
                            if ok:
                                st.session_state.user = u.strip().lower()
                                st.rerun()
                            else:
                                st.error(msg)
                    with tab_register:
                        name = st.text_input("Full name", key="reg_name")
                        email = st.text_input("Email", key="reg_email")
                        u2 = st.text_input("Choose a username", key="reg_user")
                        p2 = st.text_input("Choose a password", type="password", key="reg_pass")
                        if st.button("Create account", key="reg_btn", use_container_width=True):
                            ok, msg = register_user(store, u2, name, email, p2)
                            if ok:
                                st.session_state.user = u2.strip().lower()
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)
            st.divider()
            st.caption(DISCLAIMER)

    def require_login(store: dict[str, Any]) -> str | None:
        if not st.session_state.get("user"):
            st.info("👈 Please log in or register from the sidebar to use this feature.")
            return None
        return st.session_state.user

    # -------------------------------------------------------------------
    # LANDING PAGE
    # -------------------------------------------------------------------
    def render_landing() -> None:
        st.markdown('<div class="hero-title">AI-Powered Stock Intelligence</div>', unsafe_allow_html=True)
        st.markdown("##### Analyze Indian equities using Machine Learning, Technical Analysis, "
                     "Fundamentals and Market Sentiment.")
        c1, c2, _ = st.columns([1, 1, 3])
        with c1:
            if st.button("🔍 Explore Stocks", use_container_width=True, type="primary"):
                go_to("Dashboard"); st.rerun()
        with c2:
            if st.button("🤖 View AI Recommendations", use_container_width=True):
                go_to("Dashboard"); st.rerun()

        st.write("")
        sections = [
            ("⚙️ How It Works", "Live/demo price history feeds a technical + fundamental + sentiment "
             "scoring pipeline and a trained ML model. All four signals are combined by a transparent, "
             "configurable recommendation engine into one AI Score."),
            ("🧠 AI Analytics", "Logistic Regression, Random Forest and XGBoost are trained on a "
             "chronological (never shuffled) split and compared on Accuracy, Precision, Recall, F1 and ROC-AUC."),
            ("📊 Technical Analysis", "20+ indicators — SMA/EMA, RSI, MACD, Bollinger Bands, ATR, ADX, "
             "Stochastic Oscillator, OBV, momentum, volatility and 52-week range — rolled into a 0-100 score."),
            ("📑 Fundamental Analysis", "P/E, P/B, ROE, ROCE, margins, growth and leverage ratios scored "
             "0-100. Unavailable data is shown as 'Data unavailable' — never fabricated."),
            ("📰 Sentiment Analysis", "Recent headlines are scored with a transparent finance-lexicon NLP "
             "model into Positive / Neutral / Negative percentages."),
            ("💼 Portfolio Intelligence", "Track holdings, P/L, sector allocation and the recommendation "
             "mix (BUY/HOLD/SELL) across your whole portfolio."),
            ("⏪ Backtesting", "Walk-forward simulation on a held-out test period compares the AI strategy "
             "against Buy & Hold and the NIFTY 50 — with transaction costs applied."),
            ("🔬 Explainable AI", "SHAP-based (or coefficient-based fallback) explanations show exactly "
             "which signals pushed a recommendation up or down."),
        ]
        cols = st.columns(2)
        for i, (title, desc) in enumerate(sections):
            with cols[i % 2]:
                st.markdown(f'<div class="glass-card"><b>{title}</b><br><span class="subtle">{desc}</span></div>',
                             unsafe_allow_html=True)
        st.write("")
        disclaimer_footer()

    # -------------------------------------------------------------------
    # DASHBOARD (market overview + 20 stock cards + search)
    # -------------------------------------------------------------------
    def render_stock_card(snap: dict) -> None:
        meta = snap["meta"]
        change_color = "#4ade80" if snap["change_pct"] >= 0 else "#f87171"
        arrow = "▲" if snap["change_pct"] >= 0 else "▼"
        with st.container():
            st.markdown(f"""
            <div class="glass-card">
              <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div><b>{meta['name'].split(' Limited')[0]}</b><br><span class="subtle">{snap['symbol']} · {meta['sector']}</span></div>
                {recommendation_badge_html(snap['reco']['recommendation'])}
              </div>
              <div class="metric-big">₹{snap['current_price']:,.2f}</div>
              <div style="color:{change_color}; font-weight:600;">{arrow} {abs(snap['change_pct']):.2f}%</div>
              <div class="subtle">Confidence: {snap['reco']['confidence']:.0f}% &nbsp;·&nbsp;
                Risk: <span class="{risk_class(snap['risk']['risk_level'])}">{snap['risk']['risk_level']}</span> &nbsp;·&nbsp;
                AI Score: {snap['reco']['ai_score']:.0f}/100</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("View analysis →", key=f"card_{snap['symbol']}", use_container_width=True):
                go_to("Stock Detail", snap["symbol"]); st.rerun()

    def render_dashboard() -> None:
        st.header("Market Dashboard")
        artifacts = train_and_evaluate_models()
        bench = get_benchmark_history()
        bchange = (bench["close"].iloc[-1] / bench["close"].iloc[-2] - 1) * 100
        now = datetime.now()
        market_open = now.weekday() < 5 and 9 <= now.hour < 16

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("NIFTY 50 (demo proxy)" if IS_DEMO_MODE else "NIFTY 50", f"{bench['close'].iloc[-1]:,.0f}", f"{bchange:+.2f}%")
        m2.metric("Market Status", "Open" if market_open else "Closed")
        m3.metric("Universe Sentiment", f"{np.mean([get_stock_snapshot(s, artifacts)['sentiment']['overall'] for s in SYMBOLS[:5]]):.0f}/100 (sample)")
        m4.metric("Data Mode", mode_badge())

        st.write("")
        query = st.text_input("🔍 Search a stock (name or ticker)", placeholder="e.g. TCS, Infosys, HDFC Bank")
        symbols_to_show = SYMBOLS
        if query:
            q = query.strip().lower()
            symbols_to_show = [s for s in SYMBOLS if q in s.lower() or q in STOCK_BY_SYMBOL[s]["name"].lower()]
            if not symbols_to_show:
                st.warning("No stocks match your search.")

        sort_by = st.selectbox("Sort by", ["AI Score (high to low)", "Change % (high to low)", "Name"])
        snapshots = {s: get_stock_snapshot(s, artifacts) for s in symbols_to_show}
        if sort_by.startswith("AI"):
            symbols_to_show = sorted(symbols_to_show, key=lambda s: -snapshots[s]["reco"]["ai_score"])
        elif sort_by.startswith("Change"):
            symbols_to_show = sorted(symbols_to_show, key=lambda s: -snapshots[s]["change_pct"])
        else:
            symbols_to_show = sorted(symbols_to_show, key=lambda s: STOCK_BY_SYMBOL[s]["name"])

        buy_n = sum(1 for s in snapshots.values() if s["reco"]["recommendation"] == "BUY")
        hold_n = sum(1 for s in snapshots.values() if s["reco"]["recommendation"] == "HOLD")
        sell_n = sum(1 for s in snapshots.values() if s["reco"]["recommendation"] == "SELL")
        st.caption(f"Showing {len(symbols_to_show)} stocks — 🟢 {buy_n} BUY · 🟡 {hold_n} HOLD · 🔴 {sell_n} SELL")

        cols = st.columns(3)
        for i, sym in enumerate(symbols_to_show):
            with cols[i % 3]:
                render_stock_card(snapshots[sym])

        st.write("")
        disclaimer_footer()

    # -------------------------------------------------------------------
    # STOCK DETAIL PAGE
    # -------------------------------------------------------------------
    def render_price_chart(symbol: str, ind: pd.DataFrame) -> None:
        range_map = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252, "3Y": 756, "5Y": 1260, "Max": len(ind)}
        choice = st.radio("Range", list(range_map.keys()), index=3, horizontal=True, label_visibility="collapsed")
        window = ind.tail(range_map[choice])
        show_ma = st.multiselect("Overlay indicators", ["SMA 20", "SMA 50", "SMA 200", "EMA 20", "Bollinger Bands"],
                                  default=["SMA 50"])

        if not HAS_PLOTLY:
            st.line_chart(window["close"])
            return

        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=window.index, open=window["open"], high=window["high"],
                                      low=window["low"], close=window["close"], name=symbol,
                                      increasing_line_color="#4ade80", decreasing_line_color="#f87171"))
        overlay_map = {"SMA 20": ("sma_20", "#63b3ed"), "SMA 50": ("sma_50", "#f6ad55"),
                       "SMA 200": ("sma_200", "#a78bfa"), "EMA 20": ("ema_20", "#38bdf8")}
        for label in show_ma:
            if label in overlay_map:
                col, color = overlay_map[label]
                fig.add_trace(go.Scatter(x=window.index, y=window[col], name=label,
                                          line=dict(width=1.4, color=color)))
        if "Bollinger Bands" in show_ma:
            fig.add_trace(go.Scatter(x=window.index, y=window["bb_upper"], name="BB Upper",
                                      line=dict(width=1, color="rgba(160,160,255,0.4)")))
            fig.add_trace(go.Scatter(x=window.index, y=window["bb_lower"], name="BB Lower",
                                      line=dict(width=1, color="rgba(160,160,255,0.4)"), fill="tonexty",
                                      fillcolor="rgba(160,160,255,0.07)"))
        fig.update_layout(
            template="plotly_dark", height=440, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_rangeslider_visible=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

        vol_fig = go.Figure(go.Bar(x=window.index, y=window["volume"], marker_color="rgba(99,179,237,0.5)"))
        vol_fig.update_layout(template="plotly_dark", height=140, margin=dict(l=10, r=10, t=0, b=10),
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               yaxis_title="Volume")
        st.plotly_chart(vol_fig, use_container_width=True)

    def render_gauge(value: float, title: str) -> None:
        if not HAS_PLOTLY:
            st.metric(title, f"{value:.0f}/100")
            return
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=value, title={"text": title, "font": {"size": 14}},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#63b3ed"},
                   "steps": [{"range": [0, 40], "color": "rgba(248,113,113,0.25)"},
                             {"range": [40, 65], "color": "rgba(250,204,21,0.25)"},
                             {"range": [65, 100], "color": "rgba(74,222,128,0.25)"}]},
        ))
        fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=10),
                           paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    def render_stock_detail(store: dict[str, Any]) -> None:
        symbol = st.session_state.get("selected_symbol", "TCS")
        c0, c1 = st.columns([3, 1])
        with c1:
            symbol = st.selectbox("Switch stock", SYMBOLS, index=SYMBOLS.index(symbol),
                                   format_func=lambda s: f"{s} — {STOCK_BY_SYMBOL[s]['name']}")
            st.session_state.selected_symbol = symbol

        artifacts = train_and_evaluate_models()
        snap = get_stock_snapshot(symbol, artifacts)
        meta, reco, tech, fund, sent, risk = snap["meta"], snap["reco"], snap["tech"], snap["fund"], snap["sentiment"], snap["risk"]

        with c0:
            st.markdown(f"## {meta['name']}")
            st.caption(f"{symbol} · {meta['sector']} · {meta['industry']} · {meta['market_category']} · "
                       f'<span class="badge badge-demo">{snap["data_source"]}</span>', unsafe_allow_html=True)
            change_color = "#4ade80" if snap["change_pct"] >= 0 else "#f87171"
            st.markdown(f'<span class="metric-big">₹{snap["current_price"]:,.2f}</span> '
                        f'<span style="color:{change_color}; font-weight:700;">'
                        f'{"▲" if snap["change_pct"]>=0 else "▼"} {abs(snap["change_pct"]):.2f}%</span>',
                        unsafe_allow_html=True)

        st.write("")
        r1, r2, r3, r4 = st.columns(4)
        r1.markdown(f'<div class="glass-card">Recommendation<br>{recommendation_badge_html(reco["recommendation"])}</div>', unsafe_allow_html=True)
        r2.markdown(f'<div class="glass-card">Confidence<br><span class="metric-big">{reco["confidence"]:.0f}%</span></div>', unsafe_allow_html=True)
        r3.markdown(f'<div class="glass-card">AI Score<br><span class="metric-big">{reco["ai_score"]:.0f}/100</span></div>', unsafe_allow_html=True)
        r4.markdown(f'<div class="glass-card">Risk<br><span class="metric-big {risk_class(risk["risk_level"])}">{risk["risk_level"]}</span></div>', unsafe_allow_html=True)

        colA, colB = st.columns([1, 1])
        with colA:
            if st.button("💡 Explain this recommendation", use_container_width=True, type="primary"):
                st.session_state["show_explanation"] = symbol
        with colB:
            in_watchlist = st.session_state.get("user") and symbol in store["watchlists"].get(st.session_state["user"], [])
            if st.session_state.get("user"):
                label = "★ Remove from Watchlist" if in_watchlist else "☆ Add to Watchlist"
                if st.button(label, use_container_width=True):
                    wl = store["watchlists"].setdefault(st.session_state["user"], [])
                    if in_watchlist:
                        wl.remove(symbol)
                    else:
                        wl.append(symbol)
                    save_store(store)
                    st.rerun()

        if st.session_state.get("show_explanation") == symbol:
            expl = build_recommendation_explanation(symbol, reco, tech, fund, sent, risk)
            with st.container():
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown(f"#### Why {reco['recommendation']}?")
                for w in expl["why"]:
                    st.markdown(f"- ✅ {w}")
                st.markdown("#### Risks to consider")
                for w in expl["risks"]:
                    st.markdown(f"- ⚠️ {w}")

                shap_expl = explain_prediction(symbol, artifacts)
                st.markdown(f"###### ML model explanation method: {shap_expl['method']}")
                if shap_expl["positive"] or shap_expl["negative"]:
                    exp_df = pd.DataFrame(
                        [{"factor": p["feature"], "impact": p["impact"], "direction": "Positive"} for p in shap_expl["positive"]]
                        + [{"factor": n["feature"], "impact": n["impact"], "direction": "Negative"} for n in shap_expl["negative"]]
                    )
                    if HAS_PLOTLY and not exp_df.empty:
                        fig = go.Figure(go.Bar(
                            x=exp_df["impact"], y=exp_df["factor"], orientation="h",
                            marker_color=["#4ade80" if d == "Positive" else "#f87171" for d in exp_df["direction"]],
                        ))
                        fig.update_layout(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=20, b=10),
                                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                           title="Top factors driving the ML prediction (SHAP-style)")
                        st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Score weights used: {reco['effective_weights']}")
                st.markdown('</div>', unsafe_allow_html=True)

        st.write("")
        tabs = st.tabs(["📈 Price Chart", "📊 Technical", "📑 Fundamentals", "📰 Sentiment", "⚖️ Risk", "⏪ Backtest"])

        with tabs[0]:
            render_price_chart(symbol, compute_indicators(get_price_history(symbol)))

        with tabs[1]:
            cc = st.columns(5)
            for i, (k, v) in enumerate(tech["components"].items()):
                cc[i].metric(k.replace("_", " ").title(), f"{v:.0f}/100")
            st.markdown(f"**Overall Technical Score: {tech['overall']:.0f}/100**")
            for s in tech["signals"]:
                st.markdown(f"- {s}")
            raw = tech["raw"]
            st.dataframe(pd.DataFrame({
                "Indicator": ["SMA 20", "SMA 50", "SMA 100", "SMA 200", "EMA 20", "EMA 50", "RSI (14)",
                              "MACD", "MACD Signal", "ATR (14)", "ADX (14)", "Stochastic %K", "Stochastic %D",
                              "OBV", "Bollinger Position", "20d Volatility (ann.)", "52w High", "52w Low",
                              "Dist. from 52w High", "Dist. from 52w Low"],
                "Value": [raw["sma_20"], raw["sma_50"], raw["sma_100"], raw["sma_200"], raw["ema_20"], raw["ema_50"],
                          raw["rsi_14"], raw["macd"], raw["macd_signal"], raw["atr_14"], raw["adx_14"], raw["stoch_k"],
                          raw["stoch_d"], raw["obv"], raw["bollinger_position"], raw["volatility_20d"],
                          raw["high_52w"], raw["low_52w"], raw["dist_from_52w_high"], raw["dist_from_52w_low"]],
            }).round(4), use_container_width=True, hide_index=True)

        with tabs[2]:
            if fund["overall"] is None:
                st.warning("Fundamental data unavailable for this stock.")
            else:
                st.markdown(f"**Fundamental Score: {fund['overall']:.0f}/100** "
                            f"({fund['data_availability_pct']:.0f}% of metrics available) — {fund['valuation']}")
                if fund["low_confidence"]:
                    st.caption("⚠️ Fewer than half of the standard metrics are available — score has lower confidence.")
                fc1, fc2 = st.columns(2)
                with fc1:
                    st.markdown("**Strengths**")
                    for s in fund["strengths"] or ["None identified"]:
                        st.markdown(f"- ✅ {s}")
                with fc2:
                    st.markdown("**Weaknesses**")
                    for w in fund["weaknesses"] or ["None identified"]:
                        st.markdown(f"- ⚠️ {w}")
                rows = []
                for k, label in fund["labels"].items():
                    val = snap["fund_raw"].get(k)
                    rows.append({"Metric": label, "Value": val if val is not None else "Data unavailable",
                                 "Score": fund["components"].get(k, "—")})
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                st.caption(f"Source: {snap['fund_raw'].get('source', 'DEMO')} fundamentals data.")

        with tabs[3]:
            st.markdown(f"**Sentiment Score: {sent['overall']:.0f}/100**")
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Positive", f"{sent['positive_pct']:.0f}%")
            sc2.metric("Neutral", f"{sent['neutral_pct']:.0f}%")
            sc3.metric("Negative", f"{sent['negative_pct']:.0f}%")
            st.caption('<span class="badge badge-demo">Demo headlines — labeled, not live news</span>'
                       if IS_DEMO_MODE else "Live news headlines", unsafe_allow_html=True)
            for h in sent["scored_headlines"]:
                icon = {"positive": "🟢", "neutral": "🟡", "negative": "🔴"}[h["sentiment"]]
                st.markdown(f"{icon} **{h['headline']}** — *{h['source']}, {h['date']}* "
                            f"({h['sentiment']}, polarity {h['polarity']:+.2f})")

        with tabs[4]:
            gc1, gc2 = st.columns([1, 2])
            with gc1:
                render_gauge(risk["risk_score"], "Risk Score (higher = riskier)")
            with gc2:
                st.markdown(f"**Risk Level: <span class='{risk_class(risk['risk_level'])}'>{risk['risk_level']}</span>**",
                            unsafe_allow_html=True)
                for s in risk["signals"]:
                    st.markdown(f"- {s}")
                if risk.get("sharpe_ratio") is not None:
                    st.markdown(f"- Sharpe Ratio: {risk['sharpe_ratio']}")
                if risk.get("sortino_ratio") is not None:
                    st.markdown(f"- Sortino Ratio: {risk['sortino_ratio']}")

        with tabs[5]:
            bt = run_backtest(symbol, artifacts)
            if bt is None:
                st.info("Not enough held-out test-period history to backtest this stock yet.")
            else:
                st.caption(f"Walk-forward backtest on held-out TEST period only "
                          f"({bt['rebalance_periods']} rebalances of {bt['horizon_days']} trading days each). "
                          "Past/backtested performance does not guarantee future results.")
                if HAS_PLOTLY:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=bt["dates"], y=bt["ai_equity"], name="AI Strategy", line=dict(color="#63b3ed", width=2)))
                    fig.add_trace(go.Scatter(x=bt["dates"], y=bt["buy_hold_equity"], name="Buy & Hold", line=dict(color="#4ade80", width=2)))
                    fig.add_trace(go.Scatter(x=bt["dates"], y=bt["benchmark_equity"], name="NIFTY 50", line=dict(color="#a78bfa", width=2, dash="dot")))
                    fig.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=10, b=10),
                                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                       yaxis_title="Growth of ₹1")
                    st.plotly_chart(fig, use_container_width=True)
                bcols = st.columns(3)
                for col, (label, perf) in zip(bcols, [("AI Strategy", bt["ai_strategy"]), ("Buy & Hold", bt["buy_hold"]), ("NIFTY 50", bt["benchmark"])]):
                    with col:
                        st.markdown(f"**{label}**")
                        st.metric("Total Return", f"{perf['total_return_pct']:.1f}%")
                        st.metric("CAGR", f"{perf['cagr_pct']}%" if perf['cagr_pct'] is not None else "N/A")
                        st.metric("Max Drawdown", f"{perf['max_drawdown_pct']:.1f}%")
                        if "num_trades" in perf:
                            st.metric("Trades / Win Rate", f"{perf['num_trades']} / {perf['win_rate_pct']}%" if perf['win_rate_pct'] is not None else f"{perf['num_trades']} / N/A")

        st.write("")
        disclaimer_footer()

    # -------------------------------------------------------------------
    # STOCK COMPARISON (up to 4 stocks)
    # -------------------------------------------------------------------
    def render_compare() -> None:
        st.header("Compare Stocks")
        default = SYMBOLS[:2]
        chosen = st.multiselect("Choose up to 4 stocks", SYMBOLS, default=default, max_selections=4,
                                 format_func=lambda s: f"{s} — {STOCK_BY_SYMBOL[s]['name']}")
        if len(chosen) < 2:
            st.info("Select at least 2 stocks to compare.")
            return
        artifacts = train_and_evaluate_models()
        snaps = {s: get_stock_snapshot(s, artifacts) for s in chosen}

        rows = {
            "Price (₹)": lambda s: f"{snaps[s]['current_price']:,.2f}",
            "1-day Change": lambda s: f"{snaps[s]['change_pct']:+.2f}%",
            "Recommendation": lambda s: snaps[s]["reco"]["recommendation"],
            "Confidence": lambda s: f"{snaps[s]['reco']['confidence']:.0f}%",
            "AI Score": lambda s: f"{snaps[s]['reco']['ai_score']:.0f}/100",
            "Technical Score": lambda s: f"{snaps[s]['tech']['overall']:.0f}/100",
            "Fundamental Score": lambda s: (f"{snaps[s]['fund']['overall']:.0f}/100" if snaps[s]['fund']['overall'] is not None else "Data unavailable"),
            "Sentiment Score": lambda s: f"{snaps[s]['sentiment']['overall']:.0f}/100",
            "Risk Level": lambda s: snaps[s]["risk"]["risk_level"],
            "Volatility (ann.)": lambda s: f"{snaps[s]['risk'].get('annual_volatility', 'N/A')}%",
            "P/E": lambda s: snaps[s]["fund_raw"].get("pe", "Data unavailable"),
            "P/B": lambda s: snaps[s]["fund_raw"].get("pb", "Data unavailable"),
            "ROE": lambda s: snaps[s]["fund_raw"].get("roe", "Data unavailable"),
        }
        table = pd.DataFrame({s: {label: str(fn(s)) for label, fn in rows.items()} for s in chosen})
        st.dataframe(table, use_container_width=True)

        if HAS_PLOTLY:
            fig = go.Figure()
            for s in chosen:
                fig.add_trace(go.Scatterpolar(
                    r=[snaps[s]["tech"]["overall"], snaps[s]["fund"]["overall"] or 0,
                       snaps[s]["sentiment"]["overall"], 100 - snaps[s]["risk"]["risk_score"],
                       snaps[s]["reco"]["ai_score"]],
                    theta=["Technical", "Fundamental", "Sentiment", "Safety (100-Risk)", "AI Score"],
                    fill="toself", name=s,
                ))
            fig.update_layout(template="plotly_dark", height=420, paper_bgcolor="rgba(0,0,0,0)",
                               polar=dict(radialaxis=dict(visible=True, range=[0, 100])))
            st.plotly_chart(fig, use_container_width=True)
        disclaimer_footer()

    # -------------------------------------------------------------------
    # PORTFOLIO
    # -------------------------------------------------------------------
    def render_portfolio(store: dict[str, Any]) -> None:
        st.header("Portfolio")
        user = require_login(store)
        if not user:
            return
        artifacts = train_and_evaluate_models()
        holdings = store["portfolios"].setdefault(user, [])

        with st.expander("➕ Add a holding", expanded=len(holdings) == 0):
            c1, c2, c3, c4 = st.columns(4)
            symbol = c1.selectbox("Stock", SYMBOLS, format_func=lambda s: f"{s} — {STOCK_BY_SYMBOL[s]['name']}")
            qty = c2.number_input("Quantity", min_value=1, value=10, step=1)
            price = c3.number_input("Purchase price (₹)", min_value=0.01,
                                     value=float(get_stock_snapshot(symbol, artifacts)["current_price"]), step=0.5)
            pdate = c4.date_input("Purchase date", value=datetime.now().date())
            if st.button("Add holding", type="primary"):
                holdings.append({"id": secrets.token_hex(6), "symbol": symbol, "quantity": qty,
                                  "purchase_price": price, "purchase_date": str(pdate)})
                save_store(store)
                st.rerun()

        if not holdings:
            st.info("No holdings yet — add one above.")
            disclaimer_footer()
            return

        rows, total_invested, total_current = [], 0.0, 0.0
        sector_alloc: dict[str, float] = {}
        reco_dist = {"BUY": 0, "HOLD": 0, "SELL": 0}
        for h in holdings:
            snap = get_stock_snapshot(h["symbol"], artifacts)
            invested = h["quantity"] * h["purchase_price"]
            current = h["quantity"] * snap["current_price"]
            pl = current - invested
            total_invested += invested
            total_current += current
            sector_alloc[snap["meta"]["sector"]] = sector_alloc.get(snap["meta"]["sector"], 0) + current
            reco_dist[snap["reco"]["recommendation"]] += 1
            rows.append({
                "id": h["id"], "Symbol": h["symbol"], "Qty": h["quantity"], "Buy Price": h["purchase_price"],
                "Current Price": round(snap["current_price"], 2), "Invested": round(invested, 2),
                "Current Value": round(current, 2), "P/L": round(pl, 2),
                "Return %": round(pl / invested * 100, 2) if invested else 0.0,
                "Allocation %": 0.0, "AI Recommendation": snap["reco"]["recommendation"],
            })
        for r in rows:
            r["Allocation %"] = round(r["Current Value"] / total_current * 100, 1) if total_current else 0.0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Invested", f"₹{total_invested:,.0f}")
        m2.metric("Current Value", f"₹{total_current:,.0f}")
        m3.metric("P/L", f"₹{total_current-total_invested:,.0f}",
                   f"{(total_current/total_invested-1)*100:+.2f}%" if total_invested else None)
        m4.metric("Holdings", len(holdings))

        df = pd.DataFrame(rows)
        st.dataframe(df.drop(columns=["id"]), use_container_width=True, hide_index=True)

        del_id = st.selectbox("Remove a holding", ["—"] + [f'{r["id"]} · {r["Symbol"]} x{r["Qty"]}' for r in rows])
        if del_id != "—" and st.button("🗑️ Remove selected holding"):
            hid = del_id.split(" · ")[0]
            store["portfolios"][user] = [h for h in holdings if h["id"] != hid]
            save_store(store)
            st.rerun()

        pc1, pc2 = st.columns(2)
        if HAS_PLOTLY:
            with pc1:
                fig = go.Figure(go.Pie(labels=list(sector_alloc.keys()), values=list(sector_alloc.values()), hole=0.5))
                fig.update_layout(template="plotly_dark", height=320, title="Sector Allocation",
                                   paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
            with pc2:
                fig2 = go.Figure(go.Bar(x=list(reco_dist.keys()), y=list(reco_dist.values()),
                                         marker_color=["#4ade80", "#facc15", "#f87171"]))
                fig2.update_layout(template="plotly_dark", height=320, title="AI Recommendation Distribution",
                                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig2, use_container_width=True)

        port_returns_weighted = sum(get_stock_snapshot(h["symbol"], artifacts)["risk"]["risk_score"] * (h["quantity"] * get_stock_snapshot(h["symbol"], artifacts)["current_price"]) for h in holdings) / total_current if total_current else 50
        st.markdown(f"**Portfolio Risk Score (value-weighted): {port_returns_weighted:.0f}/100**")
        disclaimer_footer()

    # -------------------------------------------------------------------
    # WATCHLIST
    # -------------------------------------------------------------------
    def render_watchlist(store: dict[str, Any]) -> None:
        st.header("Watchlist")
        user = require_login(store)
        if not user:
            return
        artifacts = train_and_evaluate_models()
        wl = store["watchlists"].setdefault(user, [])

        add_sym = st.selectbox("Add a stock to your watchlist", ["—"] + [s for s in SYMBOLS if s not in wl],
                                format_func=lambda s: s if s == "—" else f"{s} — {STOCK_BY_SYMBOL[s]['name']}")
        if add_sym != "—" and st.button("➕ Add to watchlist"):
            wl.append(add_sym)
            save_store(store)
            st.rerun()

        if not wl:
            st.info("Your watchlist is empty — add stocks from here or from any stock's detail page.")
            disclaimer_footer()
            return

        sort_by = st.selectbox("Sort by", ["AI Score", "Change %", "Symbol"])
        snaps = {s: get_stock_snapshot(s, artifacts) for s in wl}
        order = {"AI Score": lambda s: -snaps[s]["reco"]["ai_score"], "Change %": lambda s: -snaps[s]["change_pct"],
                 "Symbol": lambda s: s}[sort_by]
        for s in sorted(wl, key=order):
            snap = snaps[s]
            c1, c2, c3, c4, c5, c6, c7 = st.columns([2, 1, 1, 1, 1, 1, 1])
            c1.markdown(f"**{s}** — {snap['meta']['name'].split(' Limited')[0]}")
            c2.markdown(f"₹{snap['current_price']:,.2f}")
            color = "#4ade80" if snap["change_pct"] >= 0 else "#f87171"
            c3.markdown(f"<span style='color:{color}'>{snap['change_pct']:+.2f}%</span>", unsafe_allow_html=True)
            c4.markdown(recommendation_badge_html(snap["reco"]["recommendation"]), unsafe_allow_html=True)
            c5.markdown(f"AI {snap['reco']['ai_score']:.0f}")
            c6.markdown(f"<span class='{risk_class(snap['risk']['risk_level'])}'>{snap['risk']['risk_level']}</span>", unsafe_allow_html=True)
            if c7.button("Remove", key=f"rm_{s}"):
                wl.remove(s)
                save_store(store)
                st.rerun()
        disclaimer_footer()

    # -------------------------------------------------------------------
    # MODEL PERFORMANCE PAGE
    # -------------------------------------------------------------------
    def render_model_performance() -> None:
        st.header("ML Model Performance")
        artifacts = train_and_evaluate_models()
        st.caption(f"Selection criterion: {artifacts['selection_criterion']}. "
                   f"Best model: **{artifacts['best_model_name']}**.")
        st.caption(f"Train years: {artifacts['train_years']} ({artifacts['train_rows']} rows) · "
                   f"Validation years: {artifacts['val_years']} ({artifacts['val_rows']} rows) · "
                   f"Test years: {artifacts['test_years']} ({artifacts['test_rows']} rows, held out — "
                   f"never used for training or model selection).")

        rows = []
        for name in artifacts["models"]:
            v, t = artifacts["val_metrics"][name], artifacts["test_metrics"][name]
            rows.append({"Model": name, "Val Accuracy": v["accuracy"], "Val ROC-AUC": v["roc_auc"],
                         "Test Accuracy": t["accuracy"], "Test Precision": t["precision"],
                         "Test Recall": t["recall"], "Test F1": t["f1"], "Test ROC-AUC": t["roc_auc"],
                         "Test PR-AUC": t["pr_auc"]})
        st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)
        st.caption(f"Test-set positive rate (class balance): {artifacts['test_positive_rate']*100:.1f}%. "
                   "ROC-AUC near 0.50 on this bundled DEMO data is expected — demo prices are a "
                   "randomized synthetic walk with no genuine predictable edge, which is also a "
                   "sanity check that the pipeline has no data leakage inflating results.")

        cc1, cc2 = st.columns(2)
        with cc1:
            if HAS_PLOTLY:
                cm = artifacts["confusion_matrix"]
                fig = go.Figure(go.Heatmap(z=cm, x=["Pred: Underperform", "Pred: Outperform"],
                                            y=["Actual: Underperform", "Actual: Outperform"],
                                            colorscale="Blues", showscale=False, text=cm, texttemplate="%{text}"))
                fig.update_layout(template="plotly_dark", height=320, title=f"Confusion Matrix — {artifacts['best_model_name']} (test set)",
                                   paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
        with cc2:
            if HAS_PLOTLY:
                roc = artifacts["roc_curve"]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], name="ROC curve", line=dict(color="#63b3ed")))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(color="gray", dash="dash")))
                fig.update_layout(template="plotly_dark", height=320, title="ROC Curve (test set)",
                                   xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                                   paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

        best_model = artifacts["models"][artifacts["best_model_name"]]
        if hasattr(best_model, "feature_importances_"):
            imp = pd.Series(best_model.feature_importances_, index=artifacts["feature_columns"]).sort_values()
            if HAS_PLOTLY:
                fig = go.Figure(go.Bar(x=imp.values, y=[_humanize_feature(f) for f in imp.index], orientation="h",
                                        marker_color="#a78bfa"))
                fig.update_layout(template="plotly_dark", height=420, title="Feature Importance",
                                   paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
        elif hasattr(best_model, "coef_"):
            imp = pd.Series(best_model.coef_.reshape(-1), index=artifacts["feature_columns"]).sort_values()
            if HAS_PLOTLY:
                fig = go.Figure(go.Bar(x=imp.values, y=[_humanize_feature(f) for f in imp.index], orientation="h",
                                        marker_color="#a78bfa"))
                fig.update_layout(template="plotly_dark", height=420, title="Model Coefficients",
                                   paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

        st.caption(f"SHAP available: {'Yes' if HAS_SHAP else 'No (fallback explanations in use)'}. "
                   f"XGBoost available: {'Yes' if HAS_XGBOOST else 'No (skipped)'}.")
        disclaimer_footer()

    # -------------------------------------------------------------------
    # AI FINANCIAL ASSISTANT (rule-based, grounded in the app's real data)
    # -------------------------------------------------------------------
    _GLOSSARY = {
        "rsi": "RSI (Relative Strength Index) measures the speed/magnitude of recent price changes on a 0-100 scale. "
               "Above 70 is traditionally considered overbought, below 30 oversold.",
        "macd": "MACD (Moving Average Convergence Divergence) is the difference between a 12-day and 26-day EMA. "
                "When it crosses above its own 9-day signal line, that's typically read as bullish.",
        "sharpe": "The Sharpe Ratio measures risk-adjusted return: (annualized return − risk-free rate) / volatility. "
                  "Higher is generally better.",
        "sortino": "The Sortino Ratio is like the Sharpe Ratio but only penalizes downside volatility, not upside swings.",
        "volatility": "Volatility is the annualized standard deviation of daily returns — a common measure of risk.",
        "var": "Value-at-Risk (VaR) estimates the maximum expected loss over a given period at a given confidence level (e.g. 95%).",
        "beta": "Beta measures a stock's sensitivity to the overall market (NIFTY 50 here). Beta > 1 means it tends to move more than the market.",
        "bollinger": "Bollinger Bands plot a moving average with bands 2 standard deviations above/below it, used to gauge relative highs/lows.",
        "backtest": "Backtesting simulates a strategy on historical data to see how it would have performed — it does not guarantee future results.",
        "shap": "SHAP (SHapley Additive exPlanations) attributes a model's prediction to each input feature's contribution, based on game theory.",
    }

    def _assistant_answer(query: str, artifacts: dict, store: dict) -> str:
        q = query.lower().strip()
        snaps = {s: get_stock_snapshot(s, artifacts) for s in SYMBOLS}

        for term, definition in _GLOSSARY.items():
            if f"what is {term}" in q or f"explain {term}" in q or q == term:
                return definition

        matched = [s for s in SYMBOLS if s.lower() in q or STOCK_BY_SYMBOL[s]["name"].lower().split(" limited")[0] in q]

        if "compare" in q and len(matched) >= 2:
            lines = [f"**Comparing {' vs '.join(matched)}:**"]
            for s in matched:
                r = snaps[s]["reco"]
                lines.append(f"- **{s}**: {r['recommendation']} (AI Score {r['ai_score']:.0f}/100, "
                              f"confidence {r['confidence']:.0f}%, risk {snaps[s]['risk']['risk_level']})")
            return "\n".join(lines)

        if "why" in q and matched:
            s = matched[0]
            snap = snaps[s]
            expl = build_recommendation_explanation(s, snap["reco"], snap["tech"], snap["fund"], snap["sentiment"], snap["risk"])
            return (f"**{s} is rated {snap['reco']['recommendation']}** (AI Score {snap['reco']['ai_score']:.0f}/100, "
                    f"confidence {snap['reco']['confidence']:.0f}%).\n\n"
                    + "\n".join(f"- {w}" for w in expl["why"]))

        if "highest" in q and "scor" in q:
            top = sorted(SYMBOLS, key=lambda s: -snaps[s]["reco"]["ai_score"])[:5]
            return "**Highest AI-scoring stocks right now:**\n" + "\n".join(
                f"- {s}: {snaps[s]['reco']['ai_score']:.0f}/100 ({snaps[s]['reco']['recommendation']})" for s in top)

        if ("lowest" in q or "least" in q) and "volat" in q:
            top = sorted(SYMBOLS, key=lambda s: snaps[s]["risk"].get("annual_volatility", 999))[:5]
            return "**Lowest-volatility stocks:**\n" + "\n".join(
                f"- {s}: {snaps[s]['risk'].get('annual_volatility', 'N/A')}% annualized volatility" for s in top)

        if "portfolio" in q and "risk" in q:
            user = st.session_state.get("user")
            if not user or not store["portfolios"].get(user):
                return "You don't have any portfolio holdings yet — add some on the Portfolio page and ask me again."
            holdings = store["portfolios"][user]
            total = sum(h["quantity"] * snaps[h["symbol"]]["current_price"] for h in holdings)
            weighted_risk = sum(h["quantity"] * snaps[h["symbol"]]["current_price"] * snaps[h["symbol"]]["risk"]["risk_score"] for h in holdings) / total if total else 0
            level = "Low" if weighted_risk < 35 else ("Moderate" if weighted_risk <= 65 else "High")
            return f"Your value-weighted portfolio risk score is **{weighted_risk:.0f}/100 ({level})**, across {len(holdings)} holding(s)."

        if matched:
            s = matched[0]
            snap = snaps[s]
            return (f"**{s} — {snap['meta']['name']}**: ₹{snap['current_price']:,.2f} "
                    f"({snap['change_pct']:+.2f}% today). Recommendation: {snap['reco']['recommendation']} "
                    f"(AI Score {snap['reco']['ai_score']:.0f}/100, confidence {snap['reco']['confidence']:.0f}%). "
                    f"Technical {snap['tech']['overall']:.0f}/100, Sentiment {snap['sentiment']['overall']:.0f}/100, "
                    f"Risk {snap['risk']['risk_level']}.")

        return ("I can answer questions grounded in this app's own data — try things like "
                "*'Why is TCS rated BUY?'*, *'Compare TCS and Infosys'*, *'What are the highest-scoring stocks?'*, "
                "*'Which stocks have the lowest volatility?'*, *'Explain RSI'*, or *'What are the risks in my portfolio?'*")

    def render_assistant(store: dict[str, Any]) -> None:
        st.header("🤖 AI Financial Assistant")
        st.caption("Answers are generated from this app's own computed data — not fabricated market claims.")
        artifacts = train_and_evaluate_models()

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        examples = ["Why is TCS rated BUY?", "Compare TCS and INFY", "What are the highest-scoring stocks?",
                    "Which stocks have the lowest volatility?", "Explain RSI", "What are the risks in my portfolio?"]
        cols = st.columns(3)
        for i, ex in enumerate(examples):
            if cols[i % 3].button(ex, key=f"ex_{i}"):
                st.session_state.chat_history.append(("user", ex))
                st.session_state.chat_history.append(("assistant", _assistant_answer(ex, artifacts, store)))

        for role, msg in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(msg)

        prompt = st.chat_input("Ask about any stock, comparison, or concept...")
        if prompt:
            st.session_state.chat_history.append(("user", prompt))
            st.session_state.chat_history.append(("assistant", _assistant_answer(prompt, artifacts, store)))
            st.rerun()
        disclaimer_footer()

    # -------------------------------------------------------------------
    # MAIN
    # -------------------------------------------------------------------
    def main() -> None:
        st.set_page_config(page_title="AI Stock Intelligence", page_icon="📈", layout="wide",
                            initial_sidebar_state="expanded")
        inject_css()
        if "page" not in st.session_state:
            st.session_state.page = "Landing"
        if "selected_symbol" not in st.session_state:
            st.session_state.selected_symbol = "TCS"
        if "user" not in st.session_state:
            st.session_state.user = None

        store = load_store()
        render_sidebar(store)

        page = st.session_state.page
        try:
            if page == "Landing":
                render_landing()
            elif page == "Dashboard":
                render_dashboard()
            elif page == "Stock Detail":
                render_stock_detail(store)
            elif page == "Compare Stocks":
                render_compare()
            elif page == "Portfolio":
                render_portfolio(store)
            elif page == "Watchlist":
                render_watchlist(store)
            elif page == "Model Performance":
                render_model_performance()
            elif page == "AI Assistant":
                render_assistant(store)
        except Exception as exc:  # never show a broken blank screen
            st.error("Something went wrong rendering this page.")
            st.exception(exc)


if __name__ == "__main__":
    if not HAS_STREAMLIT:
        raise SystemExit("Streamlit is not installed. Run: pip install -r requirements.txt, "
                          "then: streamlit run app.py")
    main()
