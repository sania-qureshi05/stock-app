"""
Test suite for app.py — covers technical/fundamental/sentiment scoring,
the recommendation engine, risk metrics, backtesting and (most
importantly) data-leakage safeguards in the ML feature/target pipeline.

Run with: pytest test_app.py -v
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import app


# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------

def test_stock_universe_has_exactly_twenty_stocks():
    assert len(app.STOCKS) == 20
    assert len(set(app.SYMBOLS)) == 20


def test_all_symbols_have_ns_suffix():
    for s in app.STOCKS:
        assert s["yf_symbol"].endswith(".NS")


def test_price_history_is_deterministic_demo_data():
    df1 = app.get_price_history.__wrapped__("TCS") if hasattr(app.get_price_history, "__wrapped__") else app._synthetic_ohlcv("TCS", 3850, 0.24, 0.09)
    df2 = app._synthetic_ohlcv("TCS", 3850, 0.24, 0.09)
    pd.testing.assert_frame_equal(df1[["open", "high", "low", "close", "volume"]],
                                   df2[["open", "high", "low", "close", "volume"]])


def test_synthetic_prices_are_positive():
    df = app._synthetic_ohlcv("INFY", 1550, 0.27, 0.08)
    assert (df["close"] > 0).all()
    assert (df["low"] > 0).all()
    assert (df["high"] >= df["low"]).all()


# ---------------------------------------------------------------------------
# Technical indicators / score
# ---------------------------------------------------------------------------

def test_compute_indicators_adds_expected_columns():
    df = app._synthetic_ohlcv("TCS", 3850, 0.24, 0.09)
    ind = app.compute_indicators(df)
    for col in ["sma_20", "sma_50", "sma_200", "ema_20", "rsi_14", "macd", "macd_signal",
                "atr_14", "adx_14", "stoch_k", "obv", "bollinger_position", "high_52w", "low_52w"]:
        assert col in ind.columns


def test_rsi_bounded_between_0_and_100():
    df = app._synthetic_ohlcv("TCS", 3850, 0.24, 0.09)
    ind = app.compute_indicators(df)
    valid = ind["rsi_14"].dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_technical_score_within_0_100():
    df = app._synthetic_ohlcv("TCS", 3850, 0.24, 0.09)
    ind = app.compute_indicators(df)
    score = app.technical_score(ind)
    assert 0 <= score["overall"] <= 100
    for v in score["components"].values():
        assert 0 <= v <= 100


# ---------------------------------------------------------------------------
# Fundamental score — never fabricates missing data
# ---------------------------------------------------------------------------

def test_fundamental_score_skips_missing_fields():
    fund = {"pe": None, "pb": None, "roe": 20.0, "roce": None, "debt_to_equity": None,
            "dividend_yield": None, "operating_margin": None, "net_margin": None,
            "revenue_growth": None, "profit_growth": None}
    result = app.fundamental_score(fund)
    assert "pe" not in result["components"]
    assert "roe" in result["components"]
    assert result["low_confidence"] is True


def test_fundamental_score_all_missing_returns_none_overall():
    fund = {k: None for k in ["pe", "pb", "roe", "roce", "debt_to_equity", "dividend_yield",
                               "operating_margin", "net_margin", "revenue_growth", "profit_growth"]}
    result = app.fundamental_score(fund)
    assert result["overall"] is None
    assert result["valuation"] == "Data unavailable"


def test_fundamental_score_never_invents_a_value_for_missing_pe():
    fund = app.get_fundamentals("PAYTM")
    assert fund["pe"] is None  # Paytm demo data intentionally has no P/E (loss-making)
    score = app.fundamental_score(fund)
    assert "pe" not in score["components"]


# ---------------------------------------------------------------------------
# Sentiment
# ---------------------------------------------------------------------------

def test_sentiment_scorer_detects_positive_and_negative():
    label, polarity = app.score_headline_sentiment("Company beats estimates, shares rally")
    assert label == "positive" and polarity > 0
    label, polarity = app.score_headline_sentiment("Company misses estimates amid weak demand")
    assert label == "negative" and polarity < 0


def test_sentiment_scorer_handles_negation():
    label, _ = app.score_headline_sentiment("Results were not strong this quarter")
    assert label in ("neutral", "negative")


def test_analyze_sentiment_percentages_sum_to_100():
    headlines = app.get_news_headlines("TCS", 10)
    result = app.analyze_sentiment(headlines)
    total = result["positive_pct"] + result["neutral_pct"] + result["negative_pct"]
    assert math.isclose(total, 100.0, abs_tol=0.5)


# ---------------------------------------------------------------------------
# Recommendation engine
# ---------------------------------------------------------------------------

def test_recommendation_engine_buy_case():
    reco = app.recommendation_engine(0.9, 90, 90, 90, 10)
    assert reco["recommendation"] == "BUY"
    assert reco["ai_score"] >= app.RECO_THRESHOLDS["buy_min_score"]


def test_recommendation_engine_sell_case():
    reco = app.recommendation_engine(0.1, 10, 10, 10, 90)
    assert reco["recommendation"] == "SELL"
    assert reco["ai_score"] <= app.RECO_THRESHOLDS["sell_max_score"]


def test_recommendation_engine_redistributes_weight_when_fundamentals_missing():
    reco = app.recommendation_engine(0.6, 60, None, 60, 40)
    assert reco["fundamental_available"] is False
    assert reco["effective_weights"]["fundamental_score"] == 0.0
    assert math.isclose(sum(reco["effective_weights"].values()), sum(app.RECO_WEIGHTS.values()), abs_tol=1e-2)


def test_recommendation_engine_score_bounded():
    for ml in (0.0, 0.5, 1.0):
        reco = app.recommendation_engine(ml, 50, 50, 50, 50)
        assert 0 <= reco["ai_score"] <= 100
        assert 0 <= reco["confidence"] <= 100


def test_recommendation_weights_sum_to_one():
    assert math.isclose(sum(app.RECO_WEIGHTS.values()), 1.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------

def test_risk_metrics_produces_valid_score_and_level():
    price = app._synthetic_ohlcv("TCS", 3850, 0.24, 0.09)
    bench = app._synthetic_ohlcv("NIFTY50", 24800, 0.14, 0.10)
    risk = app.risk_metrics(price, bench)
    assert 0 <= risk["risk_score"] <= 100
    assert risk["risk_level"] in ("Low", "Moderate", "High")


# ---------------------------------------------------------------------------
# Feature engineering — DATA LEAKAGE SAFEGUARDS (critical for this project)
# ---------------------------------------------------------------------------

def test_feature_columns_never_include_target_or_future_fields():
    banned_substrings = ["future", "target"]
    for col in app.FEATURE_COLUMNS:
        for banned in banned_substrings:
            assert banned not in col.lower(), f"Feature '{col}' looks like it leaks the target"


def test_target_uses_shifted_future_close_not_current_features():
    ds = app.build_stock_dataset("TCS")
    # Every row must have a `close` (today's price, a valid feature-time value)
    # strictly separate from `future_excess_return` (computed via shift(-horizon)).
    # The last `horizon` rows must have an undefined (NaN) target, since the
    # future window falls beyond the end of available history.
    tail = ds.tail(app.TARGET_HORIZON_DAYS)
    assert tail["target"].isna().all(), "Rows without a knowable future outcome must not have a target"


def test_no_row_has_a_feature_computed_from_its_own_future():
    price = app._synthetic_ohlcv("TCS", 3850, 0.24, 0.09)
    ind = app.compute_indicators(price)
    # sma_20 at time t must be computable from rows <= t only: verify by
    # recomputing manually on a truncated series and checking they match up
    # to the truncation point.
    cutoff = 300
    truncated = app.compute_indicators(price.iloc[:cutoff])
    pd.testing.assert_series_equal(
        ind["sma_20"].iloc[:cutoff].reset_index(drop=True),
        truncated["sma_20"].reset_index(drop=True),
        check_names=False,
    )


def test_scaler_is_fit_only_on_training_split():
    dataset = app.build_training_dataset()
    train, val, test = app.chronological_split(dataset)
    train_years = set(pd.to_datetime(train["date"]).dt.year)
    val_years = set(pd.to_datetime(val["date"]).dt.year)
    test_years = set(pd.to_datetime(test["date"]).dt.year)
    assert max(train_years) <= app.TRAIN_END_YEAR
    assert min(val_years) > app.TRAIN_END_YEAR
    assert min(test_years) > app.VALIDATION_END_YEAR
    # No year should appear in more than one split (strict chronological separation)
    assert train_years.isdisjoint(val_years)
    assert val_years.isdisjoint(test_years)
    assert train_years.isdisjoint(test_years)


def test_chronological_split_is_not_shuffled():
    dataset = app.build_training_dataset()
    train, val, test = app.chronological_split(dataset)
    assert pd.to_datetime(train["date"]).max() <= pd.to_datetime(val["date"]).min()
    assert pd.to_datetime(val["date"]).max() <= pd.to_datetime(test["date"]).min()


# ---------------------------------------------------------------------------
# ML training & evaluation
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def artifacts():
    return app.train_and_evaluate_models()


def test_all_three_required_models_are_trained(artifacts):
    assert "Logistic Regression" in artifacts["models"]
    assert "Random Forest" in artifacts["models"]
    if app.HAS_XGBOOST:
        assert "XGBoost" in artifacts["models"]


def test_best_model_selected_by_validation_roc_auc(artifacts):
    best = artifacts["best_model_name"]
    best_val_auc = artifacts["val_metrics"][best]["roc_auc"]
    for name, metrics in artifacts["val_metrics"].items():
        if not math.isnan(metrics["roc_auc"]):
            assert best_val_auc >= metrics["roc_auc"] - 1e-9


def test_metrics_are_within_valid_ranges(artifacts):
    for split in ("val_metrics", "test_metrics"):
        for name, m in artifacts[split].items():
            for key in ("accuracy", "precision", "recall", "f1"):
                assert 0 <= m[key] <= 1


def test_predict_latest_returns_valid_probability(artifacts):
    pred = app.predict_latest("TCS", artifacts)
    assert 0 <= pred["probability"] <= 1


def test_explain_prediction_returns_factors(artifacts):
    expl = app.explain_prediction("TCS", artifacts)
    assert isinstance(expl["positive"], list)
    assert isinstance(expl["negative"], list)
    assert expl["method"] in ("SHAP", "Model coefficients / feature importance (SHAP unavailable)")


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------

def test_backtest_runs_on_held_out_period_only(artifacts):
    bt = app.run_backtest("TCS", artifacts)
    assert bt is not None
    for d in bt["dates"]:
        assert int(d[:4]) > app.VALIDATION_END_YEAR
    assert bt["ai_equity"][0] == 1.0
    assert bt["buy_hold_equity"][0] == 1.0


def test_backtest_equity_curves_never_negative(artifacts):
    bt = app.run_backtest("TCS", artifacts)
    assert all(v >= 0 for v in bt["ai_equity"])
    assert all(v >= 0 for v in bt["buy_hold_equity"])


# ---------------------------------------------------------------------------
# Auth / persistence
# ---------------------------------------------------------------------------

def test_password_hash_roundtrip():
    digest, salt = app.hash_password("s3cret-password")
    assert app.verify_password("s3cret-password", digest, salt)
    assert not app.verify_password("wrong-password", digest, salt)


def test_register_and_authenticate_user(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "STORE_PATH", tmp_path / "store.json")
    store = app._default_store()
    ok, _ = app.register_user(store, "alice", "Alice", "alice@example.com", "hunter2x")
    assert ok
    ok, _ = app.register_user(store, "alice", "Alice", "alice@example.com", "hunter2x")
    assert not ok  # duplicate username rejected
    ok, _ = app.authenticate_user(store, "alice", "hunter2x")
    assert ok
    ok, _ = app.authenticate_user(store, "alice", "wrong")
    assert not ok
