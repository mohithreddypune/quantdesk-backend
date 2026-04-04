"""
routers/backtest.py

Thin router — validates input, checks rate limit, delegates to services.
No math here. No yfinance here.
"""
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, field_validator
from typing import List
import pandas as pd
import numpy as np

from cache import cache
from rate_limit import backtest_limiter, get_client_ip
from services.market_data import get_close_series, MarketDataError, get_multi_close
from services import analytics

router = APIRouter(prefix="/backtest", tags=["backtest"])

VALID_PERIODS = {"1mo","3mo","6mo","1y","2y","5y"}


class CompareRequest(BaseModel):
    tickers: List[str]
    period:  str = "1y"

    @field_validator("tickers")
    @classmethod
    def validate_tickers(cls, v):
        if len(v) < 2:
            raise ValueError("Provide at least 2 tickers")
        if len(v) > 8:
            raise ValueError("Maximum 8 tickers")
        return [t.upper().strip() for t in v]

    @field_validator("period")
    @classmethod
    def validate_period(cls, v):
        if v not in VALID_PERIODS:
            raise ValueError(f"period must be one of {VALID_PERIODS}")
        return v


@router.get("/{ticker}")
async def run_backtest(ticker: str, period: str = "1y", request: Request = None):
    # Rate limiting
    ip = get_client_ip(request) if request else "local"
    backtest_limiter.check(ip)

    ticker = ticker.upper().strip()
    if period not in VALID_PERIODS:
        raise HTTPException(400, f"period must be one of {VALID_PERIODS}")

    # Cache check
    cache_key = f"backtest:{ticker}:{period}"
    cached = cache.get(cache_key)
    if cached:
        return {**cached, "_cached": True}

    # Fetch data via service layer
    try:
        close = get_close_series(ticker, period)
    except MarketDataError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Data fetch failed: {e}")

    if len(close) < 60:
        raise HTTPException(422, f"Not enough data for {ticker} ({len(close)} days). Try a longer period.")

    # Build strategy
    rets            = analytics.daily_returns(close)
    ma50            = close.rolling(50).mean()
    signal          = (close.values > ma50.values).astype(int)
    signal_series   = pd.Series(signal, index=close.index).shift(1)
    ma_rets         = rets * signal_series
    bh_equity       = analytics.equity_curve(rets)
    ma_equity       = analytics.equity_curve(ma_rets.fillna(0))

    # Indicators
    rsi_series              = analytics.rsi(close)
    macd_line, sig, hist    = analytics.macd(close)
    roll_sharpe             = analytics.rolling_sharpe(rets)
    dd                      = analytics.drawdown_series(bh_equity)

    # Align everything to same index after dropna
    df = pd.DataFrame({
        "close":    close,
        "ma50":     ma50,
        "bh":       bh_equity,
        "ma":       ma_equity,
        "returns":  rets,
        "ma_rets":  ma_rets,
        "rsi":      rsi_series,
        "macd":     macd_line,
        "sig":      sig,
        "hist":     hist,
        "roll_sh":  roll_sharpe,
        "dd":       dd,
    }).dropna()

    dates = df.index.strftime("%Y-%m-%d").tolist()

    result = {
        "ticker":        ticker,
        "period":        period,
        "trading_days":  len(dates),
        "dates":         dates,
        "buyAndHold":    df["bh"].tolist(),
        "movingAverage": df["ma"].tolist(),
        "close":         df["close"].tolist(),
        "ma50":          df["ma50"].tolist(),
        "rsi":           df["rsi"].tolist(),
        "macd": {
            "macd":      df["macd"].tolist(),
            "signal":    df["sig"].tolist(),
            "histogram": df["hist"].tolist(),
        },
        "rollingSharpe": df["roll_sh"].tolist(),
        "drawdown":      df["dd"].tolist(),
        "metrics": {
            "bh_return":   round(float(df["bh"].iloc[-1] - 1), 4),
            "ma_return":   round(float(df["ma"].iloc[-1] - 1), 4),
            "bh_sharpe":   round(analytics.sharpe_ratio(df["returns"]), 3),
            "ma_sharpe":   round(analytics.sharpe_ratio(df["ma_rets"]), 3),
            "bh_drawdown": round(analytics.max_drawdown(df["bh"]), 4),
            "ma_drawdown": round(analytics.max_drawdown(df["ma"]), 4),
            "bh_sortino":  round(analytics.sortino_ratio(df["returns"]), 3),
            "volatility":  round(analytics.annualised_volatility(df["returns"]), 4),
        },
        "_cached": False,
    }

    cache.set(cache_key, result, ttl=300)
    return result


@router.post("/compare")
async def compare_tickers(req: CompareRequest, request: Request = None):
    ip = get_client_ip(request) if request else "local"
    backtest_limiter.check(ip)

    cache_key = f"compare:{','.join(sorted(req.tickers))}:{req.period}"
    cached = cache.get(cache_key)
    if cached:
        return {**cached, "_cached": True}

    try:
        prices = get_multi_close(req.tickers, req.period)
    except MarketDataError as e:
        raise HTTPException(404, str(e))

    equity = prices / prices.iloc[0]
    dates  = [d.strftime("%Y-%m-%d") for d in prices.index]

    metrics = {}
    for t in equity.columns:
        rets = analytics.daily_returns(prices[t])
        metrics[t] = {
            "total_return": round(float(equity[t].iloc[-1] - 1), 4),
            "sharpe":       round(analytics.sharpe_ratio(rets), 3),
            "sortino":      round(analytics.sortino_ratio(rets), 3),
            "max_drawdown": round(analytics.max_drawdown(equity[t]), 4),
            "volatility":   round(analytics.annualised_volatility(rets), 4),
        }

    result = {
        "dates":   dates,
        "curves":  {t: [round(v, 4) for v in equity[t].tolist()] for t in equity.columns},
        "metrics": metrics,
        "_cached": False,
    }
    cache.set(cache_key, result, ttl=300)
    return result
