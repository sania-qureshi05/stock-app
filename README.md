# AI-Powered Indian Stock Recommendation & Portfolio Intelligence System

A single-file Streamlit application that analyzes **20 NSE-listed Indian stocks** using
Machine Learning, Technical Analysis, Fundamental Analysis, Sentiment Analysis and
Explainable AI (SHAP) to produce transparent **BUY / HOLD / SELL** recommendations —
plus portfolio tracking, a watchlist, stock comparison, walk-forward backtesting and a
data-grounded Q&A assistant.

> ⚠️ **Disclaimer**: This platform is for educational and research purposes only and
> does not constitute financial advice. AI-generated research signals are not
> guarantees of profit. Historical or backtested performance does not guarantee
> future results.

---

## 1. Architecture note — why one file

The original project brief specified a full microservice architecture (Next.js
frontend, FastAPI backend, PostgreSQL, Alembic, Docker Compose). Partway through
development the scope was **deliberately consolidated, at the requester's explicit
instruction, into a single Python file** (`app.py`) built as a Streamlit application,
to keep the whole system runnable with one command and reviewable in one file, which
better suits an academic/demo deliverable. This README documents what was actually
built. See [`FINAL_STATUS.md`](FINAL_STATUS.md) for the honest scope/limitations
summary.

Everything — data layer, technical/fundamental/sentiment scoring, ML training and
evaluation, the recommendation engine, risk analytics, backtesting, a local
JSON-backed auth/portfolio/watchlist store, and the entire Streamlit UI — lives in
`app.py`, organized into clearly numbered sections (see the file's table of contents
in its module docstring / section comments).

## 2. Features

- **20-stock universe** with correct NSE tickers (Yahoo Finance `.NS` convention),
  sector, industry and market-cap metadata (`STOCKS` in `app.py`).
- **LIVE / DEMO modes**: `APP_MODE=LIVE` attempts real data via `yfinance`; any field
  that can't be fetched degrades to clearly-labeled DEMO data rather than being
  invented. `APP_MODE=DEMO` (default) works fully offline with deterministic,
  seeded synthetic price data and hand-set sample fundamentals — always labeled
  "Demo Data" in the UI.
- **Technical analysis**: SMA(20/50/100/200), EMA(20/50), RSI(14), MACD + signal,
  Bollinger Bands, ATR(14), ADX(14), Stochastic %K/%D, OBV, momentum, daily/5d/20d
  returns, rolling volatility, volume change, 52-week high/low and distance from
  each — rolled into a 0-100 **Technical Score** with a 5-component breakdown.
- **Fundamental analysis**: P/E, P/B, ROE, ROCE, Debt/Equity, Dividend Yield,
  Operating Margin, Net Margin, Revenue/Profit Growth. Missing values are **never
  fabricated** — shown as "Data unavailable" and excluded from the score.
- **Sentiment analysis**: a transparent finance-lexicon NLP scorer (positive/negative
  keyword polarity with negation handling) run over recent headlines (demo bank of
  realistic, labeled sample headlines) → Positive/Neutral/Negative % + a 0-100 score.
- **Machine learning**: Logistic Regression, Random Forest and XGBoost trained on a
  **strict chronological split** (never shuffled) — train ≤2023, validate 2024, test
  2025+ (fully held out). Best model selected by **validation ROC-AUC**. Target =
  positive, transaction-cost-adjusted, market-relative (excess-over-NIFTY) return
  over the next 20 trading days — see `docs/ml_methodology.md`.
- **Explainable AI**: SHAP (`TreeExplainer`/`LinearExplainer`) top positive/negative
  factors per prediction, with a coefficient/importance-based fallback if SHAP is
  unavailable.
- **Recommendation engine**: configurable weighted combination
  (`RECO_WEIGHTS` in `app.py`) of ML probability (40%), technical score (25%),
  fundamental score (20%), sentiment score (10%) and a risk adjustment (5%) → a 0-100
  **AI Score** → BUY/HOLD/SELL with a calibrated confidence score and a plain-English
  WHY/RISKS explanation.
- **Risk analytics**: annualized volatility, Sharpe & Sortino ratios, max drawdown,
  95% historical VaR, beta vs NIFTY 50 → a 0-100 risk score (Low/Moderate/High).
- **Backtesting**: walk-forward simulation on the held-out TEST period only, AI
  strategy vs Buy & Hold vs NIFTY 50, with transaction costs applied on every
  position change.
- **Portfolio & watchlist**: add/remove holdings and watchlist stocks, P/L,
  allocation %, sector allocation chart, AI recommendation distribution.
- **Stock comparison**: up to 4 stocks side-by-side plus a radar chart.
- **AI assistant**: a rule-based Q&A layer grounded in the app's own computed data
  (never fabricates numbers) — "Why is TCS rated BUY?", "Compare TCS and INFY",
  "What are the highest-scoring stocks?", "Explain RSI", "What are the risks in my
  portfolio?", etc.
- **Simple local auth**: username/password with salted PBKDF2-HMAC hashing, stored in
  a local JSON file (`stockai_store.json`) alongside the watchlist/portfolio — see
  §5 for why this replaces JWT + Postgres in the single-file build.
- **Dark, glassmorphism-styled UI** with gauges, candlestick + volume charts, radar
  charts, and graceful error/empty states throughout.

## 3. Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (default `http://localhost:8501`). No database or
external API key is required — the app runs entirely on bundled DEMO data.

### Run the automated tests

```bash
pip install -r requirements.txt   # includes pytest
pytest test_app.py -v
```

33 tests cover indicator correctness, fundamental "never fabricate" behavior,
sentiment scoring, the recommendation engine, risk metrics, **data-leakage
safeguards** (chronological split integrity, target/feature separation, scaler
fit-on-train-only), model training/selection, and backtesting.

## 4. Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `APP_MODE` | `DEMO` | `DEMO` = fully offline, bundled/synthetic data. `LIVE` = attempt real fetches via `yfinance`, falling back to DEMO per-field on failure. |

No API keys are required for either mode in this build (yfinance needs none). The
code is structured so a `NEWS_API_KEY`/paid market-data key could be wired into
`get_news_headlines()` / `get_price_history()` for a richer LIVE mode without
touching anything else — this is intentionally left as documented future work (see
§7) rather than adding unused/placeholder key plumbing.

## 5. How this maps back to the original full-stack spec

| Original spec | This build |
|---|---|
| Next.js + React + TypeScript + shadcn/ui frontend | Streamlit UI (Python), styled with custom CSS (glassmorphism, dark theme) |
| FastAPI backend + Pydantic + SQLAlchemy | Plain Python functions in `app.py`; no HTTP API boundary |
| PostgreSQL + Alembic migrations | Local JSON file store (`stockai_store.json`) for users/portfolio/watchlist; all market/analysis data is computed in-memory per session with Streamlit caching (no persistent price/prediction tables) |
| JWT authentication | Local salted PBKDF2-HMAC password hashing + Streamlit `session_state` (no cross-device tokens — see Limitations) |
| Docker Compose / multi-service deployment | Single `streamlit run app.py` — deployable as-is to Streamlit Community Cloud or any container that runs that one command |

This trade-off was made deliberately to prioritize a **working, testable, reviewable
single artifact** over infrastructure breadth. All the *analytical* substance of the
brief — the ML pipeline, leakage prevention, recommendation engine, explainability,
risk analytics and backtesting — is implemented for real, not mocked.

## 6. Documentation set

- [`docs/architecture.md`](docs/architecture.md) — module layout inside `app.py`, data/control flow
- [`docs/ml_methodology.md`](docs/ml_methodology.md) — target definition, feature list, leakage prevention, model selection
- [`docs/data_pipeline.md`](docs/data_pipeline.md) — LIVE vs DEMO data sourcing
- [`docs/backtesting.md`](docs/backtesting.md) — backtest methodology and caveats
- [`docs/explainable_ai.md`](docs/explainable_ai.md) — SHAP usage and fallback
- [`docs/testing.md`](docs/testing.md) — test suite coverage
- [`PROJECT_REPORT.md`](PROJECT_REPORT.md) — full academic report
- [`PRESENTATION_CONTENT.md`](PRESENTATION_CONTENT.md) — ~18-slide deck content
- [`VIVA_QUESTIONS.md`](VIVA_QUESTIONS.md) — 50+ viva Q&A
- [`FINAL_STATUS.md`](FINAL_STATUS.md) — what's implemented, how to run/train/test, limitations, demo script

## 7. Known limitations & future scope

- **DEMO price data is synthetic** (seeded geometric Brownian motion), not real
  historical NSE prices — chosen because live data providers were unreliable/rate
  limited from the build environment. In `APP_MODE=LIVE`, `yfinance` is used instead;
  results and model performance will differ (see `docs/data_pipeline.md`).
- **No persistent database** — market/prediction data is recomputed per session
  (with in-memory caching); only users/portfolio/watchlist persist, to a local JSON
  file, not Postgres.
- **Auth is single-machine/local-file**, not JWT-based multi-device auth — acceptable
  for a demo, not production-grade.
- **Sentiment uses a hand-built finance lexicon**, not a transformer model like
  FinBERT, to keep the app dependency-light and fast to start; the module is
  isolated (`score_headline_sentiment`) so swapping in a HuggingFace pipeline is a
  contained change.
- **News headlines are demo/sample data**, not a live feed (no `NEWS_API_KEY`
  wired up) — clearly labeled as such in the UI.
- On synthetic DEMO data, ML ROC-AUC sits near 0.50 — this is *expected and
  correct*: a random walk has no genuine predictable edge, and it doubles as a
  sanity check that the pipeline isn't leaking information (a leaky pipeline would
  show unrealistically high accuracy even on random data). See
  `docs/ml_methodology.md`.
- Future scope: real NSE historical data ingestion, a persistent database, FinBERT
  sentiment, multi-device JWT auth, and splitting back out into a full frontend/
  backend/DB deployment for production use are all natural next steps and were the
  original target architecture before the single-file pivot.
