# FINAL_STATUS

## Scope note (read this first)

The original request specified a full microservice architecture (Next.js frontend,
FastAPI backend, PostgreSQL + Alembic, Docker Compose, JWT auth). Partway into
scaffolding that architecture, **the requester explicitly asked to consolidate
everything into a single Python file**, and chose "Streamlit app" when asked to
clarify the format. Everything below describes what was actually built under that
revised, explicit instruction — not the original multi-service brief. `README.md`
§5 has a full mapping table from the original spec to what replaced each piece.

## What was implemented

A single file, `app.py` (~2,250 lines, 14 numbered sections), that genuinely
implements:

- 20-stock NSE universe with correct tickers, sector/industry/market-cap metadata.
- LIVE (`yfinance`) / DEMO (deterministic synthetic + curated sample data) dual-mode
  data layer, every value source-tagged in the UI.
- Full technical-indicator suite (SMA/EMA/RSI/MACD/Bollinger/ATR/ADX/Stochastic/OBV/
  momentum/volatility/52-week range) → 0-100 Technical Score with component
  breakdown and plain-English signals.
- Fundamental scoring (P/E, P/B, ROE, ROCE, D/E, dividend yield, margins, growth)
  that **never fabricates** missing data — verified by a dedicated test.
- Lexicon-based sentiment analysis over headlines → 0-100 score + pos/neu/neg %.
- A real ML pipeline: leakage-safe feature engineering, an explicit and
  leakage-checked target definition, a strict chronological train/validation/test
  split, three trained and fairly compared models (Logistic Regression, Random
  Forest, XGBoost), validation-ROC-AUC-based model selection, and held-out test
  metrics (Accuracy/Precision/Recall/F1/ROC-AUC/PR-AUC, confusion matrix, ROC curve).
- SHAP-based explainability with a labeled, graceful fallback.
- Risk analytics (volatility, Sharpe, Sortino, max drawdown, VaR, beta) → 0-100
  risk score and Low/Moderate/High classification.
- A transparent, configurably-weighted recommendation engine → AI Score,
  BUY/HOLD/SELL, calibrated confidence, and grounded WHY/RISKS explanations.
- Walk-forward backtesting on the held-out test period vs Buy & Hold vs NIFTY 50,
  with transaction costs.
- Local-JSON-backed registration/login (salted PBKDF2-HMAC hashing), portfolio
  (add/remove holdings, P/L, allocation, sector/recommendation-distribution charts),
  watchlist, and up-to-4-stock comparison with a radar chart.
- A rule-based AI Assistant that answers only from the app's own computed data.
- A dark, glassmorphism-styled Streamlit UI across 8 pages (Landing, Dashboard,
  Stock Detail, Compare, Portfolio, Watchlist, Model Performance, AI Assistant),
  wrapped in a top-level exception handler so no page can show a blank crash screen.
- 33 automated `pytest` tests, 5 of them specifically targeting the exact
  leakage classes named in the brief (look-ahead bias, improper scaling, silently
  mislabeled undefined targets).

## How to run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Then open the printed local URL (default `http://localhost:8501`). Works fully
offline in the default `APP_MODE=DEMO`. Set `APP_MODE=LIVE` to attempt real
`yfinance` fetches (network permitting) — see `docs/data_pipeline.md`.

## How to "train the models"

There's no separate training script — models train automatically, once per server
process, the first time any page needs them (via `train_and_evaluate_models()`,
cached with `st.cache_resource`, ~6 seconds for all 20 stocks × 3 models on this
build's hardware). To force a retrain in a running session, restart the Streamlit
process (or clear its cache via the ⋮ menu → "Clear cache").

## How to "load demo data"

Nothing to load — DEMO data (prices, fundamentals, headlines) is generated
deterministically on first access and cached in-memory for the session; the app
is never empty. User accounts/portfolio/watchlist start empty until you register
and add holdings (by design — there's no seeded demo user/portfolio).

## Test results

```
$ pytest test_app.py -v
...
33 passed, 2 warnings in ~10s
```
Covers: stock universe config, price/indicator correctness, fundamental
"never-fabricate" behavior, sentiment scoring, the recommendation engine
(BUY/SELL cases, weight redistribution, score bounds), risk metrics, **5 dedicated
data-leakage tests**, model training/selection/metric-range checks, prediction and
explainability, backtest integrity, and auth. Additionally, the full Streamlit UI
(all 8 pages, registration/login, adding a portfolio holding, adding/removing a
watchlist entry, the "Explain this recommendation" flow, and AI Assistant example
questions) was driven end-to-end with Streamlit's headless `AppTest` harness during
development with zero unhandled exceptions — that exploration also caught and fixed
two real bugs, documented below.

### Bugs found and fixed during this build

1. **Data-leakage bug**: `NaN > threshold` in pandas evaluates to `False`, not
   `NaN`, so rows near the end of history (whose future outcome isn't knowable yet)
   were silently mislabeled "underperform" instead of being excluded. Fixed by
   explicitly preserving `NaN` before casting to the integer target. See
   `docs/ml_methodology.md` §1 for the full write-up — this is arguably the most
   important single fix in the project, given the brief's emphasis on leakage
   prevention.
2. **Arrow serialization warning** on the stock-comparison table from mixed
   `str`/`float` dtypes in one column — fixed by stringifying comparison values.

## Known limitations

- DEMO price history is a **seeded synthetic random walk**, not real historical NSE
  data (chosen for offline reliability and determinism — live Yahoo Finance access
  was rate-limited/unreliable from the build environment). Consequently, ML
  ROC-AUC on DEMO data sits near 0.50, which is the *expected and correct* result
  for data with no genuine predictable structure (see `docs/ml_methodology.md` §5) —
  it is not a claim about how the same pipeline would perform on real market data.
- No persistent relational database — only users/portfolio/watchlist persist
  (to a local JSON file); market/prediction data is recomputed per session with
  in-memory caching, not stored historically.
- Authentication is local-file-based (salted password hashing + Streamlit session
  state), not JWT/multi-device.
- Sentiment analysis uses a hand-built finance lexicon, not a transformer model
  (FinBERT) — a deliberate dependency/startup-time trade-off, isolated behind one
  swappable function.
- No live news provider is wired up (`NEWS_API_KEY` unused) — news is always
  clearly-labeled demo/sample data in this build.
- `APP_MODE=LIVE` was implemented and code-reviewed but not exercised end-to-end in
  this session (the build environment's outbound access to Yahoo Finance returned
  HTTP 429 during testing) — it falls back to DEMO data per-field on any fetch
  failure, so the app degrades safely rather than breaking.

## Recommended demo sequence

1. **Landing page** — read the hero, the disclaimer, and the 8 feature sections.
2. **Dashboard** — show the NIFTY overview strip, search for "HDFC", sort by AI
   Score, point out the BUY/HOLD/SELL badges and the DEMO DATA badge.
3. Click into **TCS** (or any stock) → walk through the 4 top metrics → click
   **"Explain this recommendation"** → show the WHY/RISKS bullets and the SHAP bar
   chart → switch through the Technical / Fundamentals / Sentiment / Risk / Backtest
   tabs, pointing out the "Data unavailable" handling on Fundamentals for a stock
   with gaps (e.g. Paytm's P/E) and the disclaimer above the backtest chart.
4. **Register a user** from the sidebar, then go to **Portfolio** → add a
   holding → show P/L, sector allocation, and recommendation distribution.
5. **Watchlist** → add 2-3 stocks, sort by AI Score.
6. **Compare Stocks** → pick 3-4 stocks → show the comparison table and radar chart.
7. **Model Performance** → walk through the model comparison table, confusion
   matrix, ROC curve, and feature importance — explain the near-0.50 AUC honestly
   as expected behavior on synthetic data, per `docs/ml_methodology.md`.
8. **AI Assistant** → click a couple of the example questions (e.g. "Why is TCS
   rated BUY?", "Explain RSI"), then type a custom one.
9. Close by pointing to `test_app.py` and running `pytest test_app.py -v` live to
   show the 33 passing tests, especially the 5 leakage-specific ones.
