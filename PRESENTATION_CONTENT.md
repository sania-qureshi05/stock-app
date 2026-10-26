# Presentation Content — ~18 Slides

Speaker notes are in *italics* under each slide's bullets. Use alongside a live demo
(see `FINAL_STATUS.md` for the recommended demo sequence).

---

**Slide 1 — Title**
AI-Powered Indian Stock Recommendation & Portfolio Intelligence System
*Machine Learning · Technical Analysis · Fundamental Analysis · Sentiment Analysis · Explainable AI*

**Slide 2 — Problem**
- Retail investors rely on scattered, single-indicator "tips" with no combined view.
- Existing tools are either single-signal, fundamentals-only, or black-box "AI scores."
- No transparency into *why* a call was made or *how confident* it really is.

**Slide 3 — Motivation**
- Financial ML is one of the highest-stakes domains for data leakage and false confidence.
- A good academic project should *demonstrate*, not just claim, leakage prevention and explainability.
- India's NSE large/mid-caps are a well-understood, data-available universe to build on.

**Slide 4 — Objectives**
- Ingest correct 20-stock NSE data (LIVE + offline DEMO mode).
- Engineer leakage-safe features; train & fairly compare 3 ML models.
- Fuse ML + technical + fundamental + sentiment + risk into one transparent score.
- Explain every recommendation (SHAP + plain English).
- Ship one tested, runnable artifact.

**Slide 5 — Existing System**
- Single-indicator screeners (RSI-only, MACD-only).
- Fundamentals-only screeners with no time-aware prediction.
- Black-box "AI Score" apps with undisclosed weighting/methodology.

**Slide 6 — Proposed System**
- Multi-signal, weighted, *configurable* recommendation engine.
- Chronologically validated ML with an explicit, leakage-checked target.
- Every score traceable to a formula; every gap in data shown, never invented.

**Slide 7 — System Architecture**
- Single Python/Streamlit app, 14 logical sections: config → data → scoring →
  ML → explainability → risk → recommendation → backtest → persistence → UI.
- *(show the architecture diagram from docs/architecture.md)*

**Slide 8 — Dataset**
- 20 NSE stocks across IT, Banking/NBFC, Telecom, Energy, Industrials, FMCG, Auto, Pharma.
- LIVE mode: `yfinance`. DEMO mode: seeded synthetic OHLCV + sample fundamentals/headlines — always labeled.

**Slide 9 — Feature Engineering**
- 18 features: returns, volatility, RSI, MACD, MA ratios, ATR, ADX, Bollinger position,
  momentum, 52-week distance, stochastic %K, market-relative return.
- Every feature uses only data up to day *t* — no forward shifts.

**Slide 10 — ML Models**
- Logistic Regression (baseline) · Random Forest · XGBoost.
- Chronological split: train ≤2023 / validate 2024 / test 2025+ — never shuffled.
- Best model chosen by **validation** ROC-AUC; final metrics reported on **held-out test**.

**Slide 11 — Target Definition & Leakage Prevention**
- Target = positive, cost-adjusted, market-relative return over next 20 trading days.
- `StandardScaler` fit on train split only.
- *(mention the NaN→False leakage bug found & fixed during development — great talking point)*

**Slide 12 — Recommendation Engine**
- `AI Score = 40%·ML + 25%·Technical + 20%·Fundamental + 10%·Sentiment + 5%·Risk`.
- BUY ≥ 65, SELL ≤ 40, else HOLD. Weights live in one config dict, not scattered logic.
- Missing fundamentals → weight redistributed, never guessed.

**Slide 13 — Explainable AI**
- SHAP (TreeExplainer / LinearExplainer) → top positive/negative factors per prediction.
- Graceful fallback to coefficients/importances if SHAP unavailable, clearly labeled.
- Recommendation-level WHY/RISKS bullets grounded in the same real signals.

**Slide 14 — UI/UX**
- Dark, glassmorphism-styled dashboard; 20 stock cards; candlestick + volume charts.
- Stock detail, comparison (radar chart), portfolio, watchlist, model performance, AI assistant.
- *(live demo moment)*

**Slide 15 — Backtesting**
- Walk-forward, 20-day rebalance, held-out test period only, transaction costs applied.
- AI Strategy vs Buy & Hold vs NIFTY 50, with Sharpe/Sortino/CAGR/Max Drawdown/Win Rate.
- Explicit "not a guarantee of future results" disclaimer on every backtest.

**Slide 16 — Results**
- 33/33 automated tests pass, including 5 specifically for data-leakage classes.
- All 8 pages verified exception-free via headless UI testing.
- ROC-AUC ≈0.50 on synthetic DEMO data — expected, and a leakage sanity check, not a shortfall.

**Slide 17 — Portfolio Intelligence**
- Add/remove holdings, live P/L, sector allocation, AI recommendation distribution.
- Value-weighted portfolio risk score.
- Watchlist + up-to-4-stock comparison with radar chart.

**Slide 18 — Limitations & Future Scope**
- DEMO prices are synthetic, not historical NSE data; no persistent DB; local-file auth;
  lexicon (not FinBERT) sentiment; no live news provider wired up.
- Future: real historical ingestion, Postgres-backed deployment, FinBERT, JWT auth,
  multi-asset backtesting with position sizing.

**Slide 19 — Conclusion**
- A genuinely leakage-checked, multi-signal, explainable recommendation pipeline —
  validated by tests, not just claimed — in one reviewable, runnable artifact.

**Slide 20 — Demo Flow**
1. Landing → Dashboard (20-stock grid, search, sort).
2. Open a stock (e.g. TCS) → Explain recommendation → SHAP chart → Backtest tab.
3. Register/login → add a portfolio holding → Watchlist.
4. Compare 2-3 stocks → Model Performance page → AI Assistant Q&A.
*(see FINAL_STATUS.md for the exact recommended sequence)*
