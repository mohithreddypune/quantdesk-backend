"""
services/analytics.py — Pure financial math.

No I/O. No HTTP. No yfinance. Just numpy/pandas.
Every function here is independently unit-testable.
"""
import numpy as np
import pandas as pd
from typing import Tuple


# ── Returns & equity ─────────────────────────────────────────────────────────

def daily_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().dropna()


def equity_curve(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod()


def weighted_portfolio_returns(
    returns_df: pd.DataFrame,
    weights: np.ndarray,
) -> pd.Series:
    """Value-weighted portfolio daily returns."""
    return returns_df.dot(weights)


# ── Risk metrics ─────────────────────────────────────────────────────────────

def sharpe_ratio(returns: pd.Series, periods: int = 252) -> float:
    s = returns.std()
    if s == 0 or np.isnan(s):
        return 0.0
    return float(returns.mean() / s * np.sqrt(periods))


def sortino_ratio(returns: pd.Series, periods: int = 252) -> float:
    downside = returns[returns < 0].std()
    if downside == 0 or np.isnan(downside):
        return 0.0
    return float(returns.mean() / downside * np.sqrt(periods))


def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    return float(dd.min())


def drawdown_series(equity: pd.Series) -> pd.Series:
    roll_max = equity.cummax()
    return ((equity - roll_max) / roll_max).fillna(0)


def annualised_volatility(returns: pd.Series, periods: int = 252) -> float:
    return float(returns.std() * np.sqrt(periods))


def beta_alpha(
    port_returns: pd.Series,
    bench_returns: pd.Series,
) -> Tuple[float, float]:
    aligned = pd.concat([port_returns, bench_returns], axis=1).dropna()
    aligned.columns = ["p", "b"]
    if len(aligned) < 2:
        return 0.0, 0.0
    cov = np.cov(aligned["p"], aligned["b"])
    beta  = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 0.0
    alpha = float(aligned["p"].mean() * 252 - beta * aligned["b"].mean() * 252)
    return round(float(beta), 4), round(alpha, 4)


def var_cvar(
    returns: pd.Series,
    portfolio_value: float,
    confidence: float = 0.95,
) -> dict:
    if len(returns) < 20:
        return {"confidence": confidence, "var_1d": 0, "cvar_1d": 0, "var_10d": 0, "cvar_10d": 0}

    cutoff  = np.percentile(returns, (1 - confidence) * 100)
    var_1d  = cutoff * portfolio_value
    tail    = returns[returns <= cutoff]
    cvar_1d = tail.mean() * portfolio_value if len(tail) > 0 else var_1d

    return {
        "confidence": confidence,
        "var_1d":   round(float(var_1d),              2),
        "cvar_1d":  round(float(cvar_1d),             2),
        "var_10d":  round(float(var_1d  * np.sqrt(10)), 2),
        "cvar_10d": round(float(cvar_1d * np.sqrt(10)), 2),
    }


def rolling_sharpe(returns: pd.Series, window: int = 30) -> pd.Series:
    def _s(x: pd.Series) -> float:
        s = x.std()
        return float(x.mean() / s * np.sqrt(252)) if s and not np.isnan(s) else 0.0
    return returns.rolling(window).apply(_s, raw=False).fillna(0)


# ── Indicators ───────────────────────────────────────────────────────────────

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def macd(
    close: pd.Series,
    fast: int  = 12,
    slow: int  = 26,
    signal: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast    = close.ewm(span=fast,   adjust=False).mean()
    ema_slow    = close.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def ma_signal(close: pd.Series, window: int = 50) -> pd.Series:
    """1 when price > MA, 0 otherwise, shifted 1 day forward (no lookahead)."""
    ma = close.rolling(window).mean()
    signal = pd.Series((close.values > ma.values).astype(int), index=close.index)
    return signal.shift(1)


# ── Monte Carlo (GBM) ─────────────────────────────────────────────────────────

def monte_carlo_gbm(
    daily_returns: pd.Series,
    n_paths: int  = 500,
    n_days: int   = 252,
    seed: int     = 42,
) -> dict:
    mu    = daily_returns.mean()
    sigma = daily_returns.std()

    rng   = np.random.default_rng(seed)
    Z     = rng.standard_normal((n_paths, n_days))
    paths = np.exp((mu - 0.5 * sigma**2) + sigma * Z)
    paths = np.cumprod(paths, axis=1)

    pcts  = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)

    return {
        "p5":  pcts[0].tolist(),
        "p25": pcts[1].tolist(),
        "p50": pcts[2].tolist(),
        "p75": pcts[3].tolist(),
        "p95": pcts[4].tolist(),
        "final": {k: round(float(v), 4) for k, v in zip(
            ["p5", "p25", "p50", "p75", "p95"],
            [pcts[i, -1] for i in range(5)],
        )},
    }


# ── Correlation ───────────────────────────────────────────────────────────────

def correlation_matrix(returns_df: pd.DataFrame) -> dict:
    corr = returns_df.corr().round(3)
    tickers = list(corr.columns)
    return {
        t: {t2: float(corr.loc[t, t2]) for t2 in tickers}
        for t in tickers
    }
