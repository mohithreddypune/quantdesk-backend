
import numpy as np
import pandas as pd
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from services.analytics import (
    sharpe_ratio, sortino_ratio, max_drawdown, drawdown_series,
    annualised_volatility, beta_alpha, var_cvar, rolling_sharpe,
    rsi, macd, ma_signal, equity_curve, daily_returns,
    monte_carlo_gbm, correlation_matrix, weighted_portfolio_returns,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def flat_returns():
    """Zero returns — edge case for ratio functions."""
    return pd.Series([0.0] * 252)


@pytest.fixture
def positive_returns():
    """Steady +0.05% daily — should produce positive Sharpe."""
    return pd.Series([0.0005] * 252)


@pytest.fixture
def noisy_returns():
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0.0003, 0.012, 252))


@pytest.fixture
def price_series():
    rng = np.random.default_rng(1)
    prices = 100 * np.cumprod(1 + rng.normal(0.0003, 0.012, 300))
    return pd.Series(prices, name="Close")


# ── Sharpe ratio ─────────────────────────────────────────────────────────────

class TestSharpeRatio:
    def test_zero_returns_gives_zero(self, flat_returns):
        assert sharpe_ratio(flat_returns) == 0.0

    def test_positive_returns_positive_sharpe(self, positive_returns):
        assert sharpe_ratio(positive_returns) > 0

    def test_negative_returns_negative_sharpe(self):
        returns = pd.Series([-0.001] * 252)
        assert sharpe_ratio(returns) < 0

    def test_annualisation_factor(self):
        # With daily std = 1% and mean = 0.05%, Sharpe ≈ 0.05/1 * sqrt(252)
        returns = pd.Series([0.0005] * 100 + [-0.0005] * 100 + [0.001] * 52)
        s = sharpe_ratio(returns)
        assert isinstance(s, float)
        assert not np.isnan(s)

    def test_single_value_no_crash(self):
        assert sharpe_ratio(pd.Series([0.01])) == 0.0


# ── Sortino ratio ─────────────────────────────────────────────────────────────

class TestSortinoRatio:
    def test_no_downside_returns_zero(self, positive_returns):
        # All positive returns → no downside std → returns 0
        assert sortino_ratio(positive_returns) == 0.0

    def test_mixed_returns(self, noisy_returns):
        result = sortino_ratio(noisy_returns)
        assert isinstance(result, float)
        assert not np.isnan(result)

    def test_sortino_geq_sharpe_for_positive_skew(self, noisy_returns):
        # Sortino ignores upside vol so should be >= Sharpe for typical returns
        s = sharpe_ratio(noisy_returns)
        so = sortino_ratio(noisy_returns)
        # Both can be negative, but Sortino >= Sharpe when mean > 0
        if noisy_returns.mean() > 0:
            assert so >= s


# ── Drawdown ──────────────────────────────────────────────────────────────────

class TestDrawdown:
    def test_flat_equity_zero_drawdown(self):
        eq = pd.Series([1.0] * 100)
        assert max_drawdown(eq) == 0.0

    def test_known_drawdown(self):
        # Rises to 2.0 then drops to 1.0 → 50% drawdown
        prices = [1.0, 1.5, 2.0, 1.5, 1.0]
        eq = pd.Series(prices)
        dd = max_drawdown(eq)
        assert abs(dd - (-0.5)) < 0.001

    def test_drawdown_series_always_lte_zero(self, noisy_returns):
        eq = equity_curve(noisy_returns)
        dd = drawdown_series(eq)
        assert (dd <= 0.001).all()

    def test_drawdown_series_length_matches_equity(self, noisy_returns):
        eq = equity_curve(noisy_returns)
        dd = drawdown_series(eq)
        assert len(dd) == len(eq)


# ── VaR / CVaR ────────────────────────────────────────────────────────────────

class TestVaRCVaR:
    def test_var_negative(self, noisy_returns):
        result = var_cvar(noisy_returns, portfolio_value=100_000)
        assert result["var_1d"] < 0

    def test_cvar_leq_var(self, noisy_returns):
        result = var_cvar(noisy_returns, portfolio_value=100_000)
        assert result["cvar_1d"] <= result["var_1d"]

    def test_10d_scaling(self, noisy_returns):
        result = var_cvar(noisy_returns, portfolio_value=100_000)
        expected_10d = result["var_1d"] * np.sqrt(10)
        assert abs(result["var_10d"] - expected_10d) < 1.0  # within $1

    def test_confidence_level_stored(self, noisy_returns):
        result = var_cvar(noisy_returns, 100_000, confidence=0.99)
        assert result["confidence"] == 0.99

    def test_insufficient_data_returns_zeros(self):
        result = var_cvar(pd.Series([0.01] * 5), 100_000)
        assert result["var_1d"] == 0


# ── Beta / Alpha ─────────────────────────────────────────────────────────────

class TestBetaAlpha:
    def test_identical_series_beta_one(self, noisy_returns):
        b, a = beta_alpha(noisy_returns, noisy_returns)
        assert abs(b - 1.0) < 0.001

    def test_double_returns_beta_two(self, noisy_returns):
        doubled = noisy_returns * 2
        b, a = beta_alpha(doubled, noisy_returns)
        assert abs(b - 2.0) < 0.01

    def test_returns_floats(self, noisy_returns):
        b, a = beta_alpha(noisy_returns, noisy_returns)
        assert isinstance(b, float)
        assert isinstance(a, float)


# ── RSI ───────────────────────────────────────────────────────────────────────

class TestRSI:
    def test_rsi_bounds(self, price_series):
        result = rsi(price_series)
        valid  = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_rising_prices_high_rsi(self):
        prices = pd.Series([100 * (1.01 ** i) for i in range(100)], dtype=float)
        result = rsi(prices).dropna()
        assert result.iloc[-1] > 70

    def test_falling_prices_low_rsi(self):
        prices = pd.Series(range(100, 0, -1), dtype=float)
        result = rsi(prices).dropna()
        assert result.iloc[-1] < 30

    def test_flat_prices_rsi_50(self):
        prices = pd.Series([100.0] * 50)
        result = rsi(prices).dropna()
        # All gains/losses zero → fallback to 50
        assert (result == 50).all()


# ── MACD ──────────────────────────────────────────────────────────────────────

class TestMACD:
    def test_returns_three_series(self, price_series):
        m, s, h = macd(price_series)
        assert isinstance(m, pd.Series)
        assert isinstance(s, pd.Series)
        assert isinstance(h, pd.Series)

    def test_histogram_is_macd_minus_signal(self, price_series):
        m, s, h = macd(price_series)
        diff = (m - s - h).dropna().abs()
        assert (diff < 1e-10).all()

    def test_length_preserved(self, price_series):
        m, s, h = macd(price_series)
        assert len(m) == len(price_series)


# ── MA signal ─────────────────────────────────────────────────────────────────

class TestMASignal:
    def test_signal_is_zero_or_one(self, price_series):
        sig = ma_signal(price_series).dropna()
        assert set(sig.unique()).issubset({0, 1})

    def test_signal_shifted_no_lookahead(self, price_series):
        # Signal at index i should depend only on data up to index i-1
        sig = ma_signal(price_series)
        assert pd.isna(sig.iloc[0])  # first value should be NaN after shift


# ── Monte Carlo ───────────────────────────────────────────────────────────────

class TestMonteCarlo:
    def test_returns_all_percentiles(self, noisy_returns):
        result = monte_carlo_gbm(noisy_returns)
        for key in ["p5", "p25", "p50", "p75", "p95"]:
            assert key in result
            assert len(result[key]) == 252

    def test_percentile_order(self, noisy_returns):
        result = monte_carlo_gbm(noisy_returns)
        f = result["final"]
        assert f["p5"] <= f["p25"] <= f["p50"] <= f["p75"] <= f["p95"]

    def test_deterministic_with_seed(self, noisy_returns):
        r1 = monte_carlo_gbm(noisy_returns, seed=42)
        r2 = monte_carlo_gbm(noisy_returns, seed=42)
        assert r1["final"]["p50"] == r2["final"]["p50"]

    def test_different_seeds_differ(self, noisy_returns):
        r1 = monte_carlo_gbm(noisy_returns, seed=1)
        r2 = monte_carlo_gbm(noisy_returns, seed=2)
        assert r1["final"]["p50"] != r2["final"]["p50"]


# ── Correlation ───────────────────────────────────────────────────────────────

class TestCorrelation:
    def test_self_correlation_is_one(self, noisy_returns):
        df = pd.DataFrame({"A": noisy_returns, "B": noisy_returns})
        corr = correlation_matrix(df)
        assert abs(corr["A"]["A"] - 1.0) < 0.001
        assert abs(corr["B"]["B"] - 1.0) < 0.001

    def test_symmetric(self, noisy_returns):
        rng = np.random.default_rng(5)
        df  = pd.DataFrame({
            "A": noisy_returns,
            "B": pd.Series(rng.normal(0, 0.01, len(noisy_returns))),
        })
        corr = correlation_matrix(df)
        assert abs(corr["A"]["B"] - corr["B"]["A"]) < 1e-9


# ── Weighted portfolio returns ─────────────────────────────────────────────────

class TestWeightedPortfolioReturns:
    def test_single_asset_equal_to_itself(self, noisy_returns):
        df = pd.DataFrame({"A": noisy_returns})
        result = weighted_portfolio_returns(df, np.array([1.0]))
        pd.testing.assert_series_equal(result, noisy_returns, check_names=False)

    def test_equal_weight_two_assets(self, noisy_returns):
        rng = np.random.default_rng(7)
        b   = pd.Series(rng.normal(0.0003, 0.012, len(noisy_returns)))
        df  = pd.DataFrame({"A": noisy_returns, "B": b})
        result = weighted_portfolio_returns(df, np.array([0.5, 0.5]))
        expected = (noisy_returns + b) / 2
        pd.testing.assert_series_equal(result, expected, check_names=False)
