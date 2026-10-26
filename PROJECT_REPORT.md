# PROJECT REPORT

## AI-Powered Indian Stock Recommendation & Portfolio Intelligence System Using Machine Learning, Technical Analysis, Fundamental Analysis, Sentiment Analysis and Explainable AI

---

### 1. Abstract

This project builds an end-to-end system that generates explainable BUY/HOLD/SELL
recommendations for 20 major NSE-listed Indian equities. It combines a supervised
machine-learning model (trained on chronologically split, leakage-checked data),
classical technical analysis, fundamental-ratio analysis, lexicon-based sentiment
analysis of news headlines, and a risk model, into one transparent, weighted
**Recommendation Engine**. Every recommendation is explained via SHAP-based
attribution and plain-English reasoning. The system also provides portfolio
tracking, a watchlist, multi-stock comparison, and walk-forward backtesting against
Buy & Hold and the NIFTY 50 benchmark. The application is implemented as a single
Python/Streamlit program so it can be reviewed and run end-to-end with one command.

### 2. Introduction

Retail investors in India increasingly rely on scattered, opaque "tips" or
single-indicator signals (e.g. "RSI is oversold, buy") without a systematic way to
combine multiple analytical lenses or understand *why* a signal was generated. This
project addresses that gap academically: it demonstrates a full pipeline — data →
features → model → explainable decision — applied to a well-known domain
(Indian large/mid-cap equities) where the correctness pitfalls (data leakage,
survivorship bias, fabricated confidence) are well documented and can be concretely
guarded against and tested for.

### 3. Problem Statement

Design and implement a system that, for each of 20 specified Indian stocks,
produces a BUY/HOLD/SELL recommendation that is (a) derived from multiple
independent signal sources, not a single arbitrary threshold, (b) reproducible and
free of look-ahead bias, and (c) explainable — a user can see exactly which factors
drove the call and how confident the system is, without being told or shown a false
guarantee of profit.

### 4. Objectives

1. Ingest and process historical price data (and, where available, fundamentals and
   news) for a 20-stock universe, with correct NSE identifiers.
2. Engineer a leakage-safe technical/statistical feature set.
3. Train and fairly compare multiple ML classifiers using a strict chronological
   (train/validation/test) split.
4. Combine ML output with technical, fundamental, sentiment and risk scores via a
   transparent, configurable weighted recommendation engine.
5. Provide model-level (SHAP) and recommendation-level (plain-English) explanations
   for every call.
6. Provide portfolio, watchlist, comparison and backtesting tooling grounded in the
   same signals.
7. Prevent and actively test for data leakage throughout.
8. Ship a single, runnable, tested artifact with full documentation.

### 5. Literature Review (brief)

Technical indicators (RSI, MACD, Bollinger Bands, ADX, Stochastic Oscillator) are
long-standing tools in quantitative technical analysis, originally formalized by
Wilder (RSI, ADX, ATR) and Appel (MACD). Fundamental screening via P/E, P/B, ROE and
leverage ratios traces to classical value-investing methodology. Lexicon-based
financial sentiment analysis (e.g. the Loughran-McDonald word lists) predates
transformer models like FinBERT and remains a transparent, auditable alternative
where model weight downloads or GPU inference aren't practical. On the ML side,
gradient-boosted trees (XGBoost) and random forests are consistently competitive on
tabular financial features versus deep learning in published Kaggle- and
academic-style comparisons, motivating their inclusion alongside a linear baseline.
SHAP (Lundberg & Lee, 2017) is the standard modern approach to model-agnostic,
theoretically grounded feature attribution, superseding older ad hoc "variable
importance" methods.

### 6. Existing System

Most retail-facing "stock screener" tools present either (a) a single technical
signal (e.g. RSI crossovers) with no ML or fundamentals, or (b) fundamentals-only
screening with no time-series-aware prediction, or (c) black-box "AI" scores with no
disclosed methodology, weighting, or explanation. Data-leakage-free, chronologically
validated ML is rarely surfaced to end users at all.

### 7. Proposed System

A single application that: (1) computes technical, fundamental and sentiment scores
per stock from auditable formulas; (2) trains a properly validated ML classifier
whose target is explicitly defined and leakage-checked; (3) merges all of the above
via a documented, configurable weighted formula into one AI Score and BUY/HOLD/SELL
call; (4) explains that call at both the recommendation and model level; (5) never
fabricates data it doesn't have (fundamentals, live news) — it labels the gap
instead; and (6) is validated by an automated test suite that specifically checks
for the leakage classes named in the brief (look-ahead bias, improper scaling,
future-value comparisons).

### 8. System Architecture

See `docs/architecture.md` for the full diagram. In summary: a single Python module
(`app.py`) organized into a data layer (LIVE/DEMO), a scoring layer (technical,
fundamental, sentiment, risk), an ML layer (feature engineering → training →
explainability), a recommendation-engine layer that fuses all scores, a
backtesting layer, a lightweight local-JSON persistence layer for user
accounts/portfolio/watchlist, and a Streamlit UI layer.

### 9. Data Collection

20 NSE-listed stocks (Wipro, TCS, Infosys, HDFC Bank, SBI, ICICI Bank, Shriram
Finance, Paytm, Tata Communications, Reliance, HCL Technologies, Bharti Airtel,
Larsen & Toubro, Axis Bank, Kotak Mahindra Bank, Bajaj Finance, ITC, Tata Motors,
Sun Pharmaceutical, Maruti Suzuki), configured with correct `.NS` ticker symbols,
sector, industry and market-cap category. Two data-sourcing modes are implemented:
LIVE (via `yfinance`) and DEMO (deterministic synthetic price series + curated
sample fundamentals/headlines), with every value in the UI tagged by its actual
source — see `docs/data_pipeline.md`.

### 10. Data Preprocessing

Price series are indexed by trading date; indicators are computed with
`pandas`/`numpy` rolling-window operations that only ever reference rows up to and
including the current date. Feature scaling (`StandardScaler`) is fit exclusively
on the training split and applied (transform-only) to validation/test, preventing
information about the validation/test distribution from leaking into training.

### 11. Feature Engineering

18 leakage-safe features spanning returns (1d/5d/20d), volatility, RSI, MACD
(normalized), moving-average ratios, volume change, ATR, ADX, Bollinger position,
momentum, distance from 52-week high/low, stochastic %K, and market-relative
(excess-over-NIFTY) 20-day return. Full list and definitions in
`docs/ml_methodology.md` §2.

### 12. Technical Analysis

20+ indicators computed per stock (SMA 20/50/100/200, EMA 20/50, RSI, MACD +
signal, Bollinger Bands, ATR, ADX, Stochastic Oscillator, OBV, momentum, returns,
rolling volatility, volume change, 52-week high/low and distances) are rolled into
a 0-100 **Technical Score** via a 5-component breakdown (RSI, MACD, Moving Average
stack, Momentum, Volume), each independently interpretable and displayed to the
user with plain-English signal bullets (e.g. "MACD above signal line — bullish
crossover in effect").

### 13. Fundamental Analysis

P/E, P/B, ROE, ROCE, Debt/Equity, Dividend Yield, Operating Margin, Net Margin,
Revenue Growth and Profit Growth are each scored 0-100 against banded thresholds.
Critically, **any metric that is unavailable is excluded from the score, not
imputed or guessed** — the UI explicitly renders "Data unavailable," and the
overall score is flagged "low confidence" when fewer than half the metrics are
present (see `test_fundamental_score_never_invents_a_value_for_missing_pe` in the
test suite, which locks in this behavior for Paytm's genuinely-missing P/E).

### 14. Sentiment Analysis

A transparent finance-lexicon polarity scorer (positive/negative keyword sets with
one-token negation handling) is run over recent headlines per stock, producing
Positive/Neutral/Negative percentages and a 0-100 sentiment score. This was chosen
over a transformer model (e.g. FinBERT) to keep the single-file app dependency-light
and fast to start, while remaining fully auditable — every word's polarity
contribution can be inspected directly in `_POS_WORDS`/`_NEG_WORDS`. The module is
isolated (`score_headline_sentiment`) so a transformer-based scorer could be
substituted without touching any other part of the pipeline.

### 15. Machine Learning Methodology

Target: positive, transaction-cost-adjusted, market-relative return over the next
20 trading days (full derivation and the leakage bug found/fixed during development,
in `docs/ml_methodology.md` §1). Chronological split: train ≤2023 / validate 2024 /
test 2025+, never shuffled. See §16-17 below and the full write-up in
`docs/ml_methodology.md`.

### 16. Model Selection

Logistic Regression, Random Forest and XGBoost are trained identically and compared
on the **validation** split; the model with the highest validation ROC-AUC is
selected (`train_and_evaluate_models()`'s documented `selection_criterion`). XGBoost
was included (rather than assumed superior) because published tabular-data
benchmarks favor gradient-boosted trees, but the app deliberately evaluates rather
than assumes this, and degrades gracefully (skips XGBoost) if the package is
unavailable.

### 17. Evaluation Metrics

Accuracy, Precision, Recall, F1, ROC-AUC and PR-AUC are reported for every model on
both the validation and (once, after selection) the held-out test split, plus a
confusion matrix and ROC curve for the selected model on test — all visible on the
in-app **Model Performance** page. On the bundled synthetic DEMO data, ROC-AUC sits
near 0.50 across all models — the correct, expected result for data with no genuine
predictable structure, and a useful leakage sanity check (see
`docs/ml_methodology.md` §5).

### 18. Explainable AI

SHAP (`TreeExplainer` for RF/XGBoost, `LinearExplainer` for Logistic Regression)
attributes each live prediction to its top positive/negative contributing features,
rendered as a labeled bar chart behind an "Explain this recommendation" button. A
coefficient/importance-based fallback activates automatically (and is labeled as
such) if `shap` is unavailable. See `docs/explainable_ai.md`.

### 19. Recommendation Engine

```
AI Score = clip( wₘₗ·MLProb·100 + wₜₑ𝒸ₕ·TechScore + wf𝒖ₙd·FundScore
                 + wₛₑₙₜ·SentScore + w𝓇ᵢₛₖ·(50 − RiskScore)/50·100 , 0, 100 )
```
with default weights `{ml: 0.40, technical: 0.25, fundamental: 0.20, sentiment: 0.10,
risk: 0.05}` (must sum to 1.0, enforced by an assertion at import time), configurable
in one place (`RECO_WEIGHTS`). BUY if AI Score ≥ 65, SELL if ≤ 40, else HOLD
(`RECO_THRESHOLDS`). If fundamentals are unavailable for a stock, that weight is
**redistributed proportionally** across the remaining signals rather than
substituting a guessed value. Confidence blends the ML model's distance from 0.5
(calibrated confidence) with cross-signal agreement (do technical/fundamental/
sentiment point the same direction as the call). Full derivation:
`docs/ml_methodology.md`, §11 of `app.py`.

### 20. Backtesting

A walk-forward long/flat simulation on the held-out test period only, rebalancing
every 20 trading days, entering a position when the model's probability clears a
validation-calibrated threshold, charging a transaction cost on every position
change, and comparing against Buy & Hold and the NIFTY 50. Full methodology and
caveats: `docs/backtesting.md`.

### 21. Results

- 33/33 automated tests pass (`pytest test_app.py`), including 5 tests specifically
  targeting data-leakage classes (chronological split integrity, target/feature
  separation, train-only scaler fitting).
- All 8 UI pages verified exception-free via Streamlit's headless `AppTest` harness,
  including registration/login, adding/removing portfolio holdings and watchlist
  entries, the "Explain this recommendation" flow, and the AI Assistant's example
  questions.
- On bundled DEMO data, model ROC-AUC is ~0.50-0.53 across candidates — expected and
  discussed above; real predictive performance would need to be re-evaluated against
  genuine historical NSE data in `APP_MODE=LIVE`.

### 22. Limitations

See `README.md` §7 and `FINAL_STATUS.md` for the full, honest list — most notably:
DEMO price data is synthetic (not real historical NSE prices); there is no
persistent relational database (a local JSON file backs only users/portfolio/
watchlist); authentication is local-file-based, not JWT/multi-device; sentiment
uses a hand-built lexicon, not FinBERT; and no live news provider is wired up by
default.

### 23. Future Scope

Real historical NSE data ingestion for LIVE mode at scale; a persistent
PostgreSQL-backed deployment (the original architecture target before the
single-file pivot); FinBERT or another transformer-based sentiment model; JWT-based
multi-device authentication; sector-relative and peer-relative scoring; multi-asset
portfolio-level backtesting with position sizing; and short-horizon (5-day)
prediction alongside the current 20-day primary horizon.

### 24. Conclusion

This project demonstrates that a genuinely leakage-checked, multi-signal,
explainable stock-recommendation pipeline can be built and validated end-to-end in
a single reviewable artifact — not just as a UI mockup, but with real (if
appropriately modest, on synthetic data) ML evaluation, an auditable weighted
recommendation formula, model- and recommendation-level explanations, and an
automated test suite that actively checks for the exact leakage failure modes the
domain is known for.

### 25. References

- Wilder, J. W. (1978). *New Concepts in Technical Trading Systems* (RSI, ADX, ATR).
- Appel, G. (2005). *Technical Analysis: Power Tools for Active Investors* (MACD).
- Bollinger, J. (2001). *Bollinger on Bollinger Bands*.
- Loughran, T., & McDonald, B. (2011). "When Is a Liability Not a Liability? Textual
  Analysis, Dictionaries, and 10-Ks." *Journal of Finance* (finance-domain sentiment
  lexicons).
- Lundberg, S. M., & Lee, S.-I. (2017). "A Unified Approach to Interpreting Model
  Predictions." *NeurIPS* (SHAP).
- Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System."
  *KDD*.
- Breiman, L. (2001). "Random Forests." *Machine Learning*, 45(1).
- Sharpe, W. F. (1966). "Mutual Fund Performance." *Journal of Business* (Sharpe
  Ratio).
- Sortino, F. A., & Price, L. N. (1994). "Performance Measurement in a Downside Risk
  Framework." *Journal of Investing* (Sortino Ratio).
- `yfinance`, `scikit-learn`, `XGBoost`, `SHAP`, `Streamlit`, `Plotly` — open-source
  library documentation.
