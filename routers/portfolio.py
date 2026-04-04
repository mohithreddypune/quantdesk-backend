"""
routers/portfolio.py — Thin router, delegates to service layer.
"""
from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, field_validator
from typing import List
import numpy as np
import pandas as pd

from cache import cache
from rate_limit import portfolio_limiter, get_client_ip
from services.market_data import get_multi_close, get_sector, get_news, MarketDataError
from services import analytics

router = APIRouter(prefix="/portfolio", tags=["portfolio"])

VALID_PERIODS = {"1mo","3mo","6mo","1y","2y","5y"}


class Holding(BaseModel):
    ticker:   str
    shares:   float
    avg_cost: float

    @field_validator("ticker")
    @classmethod
    def upper_ticker(cls, v):
        return v.upper().strip()

    @field_validator("shares", "avg_cost")
    @classmethod
    def positive(cls, v):
        if v <= 0:
            raise ValueError("Must be positive")
        return round(v, 6)


class PortfolioRequest(BaseModel):
    holdings: List[Holding]
    period:   str = "1y"

    @field_validator("holdings")
    @classmethod
    def deduplicate(cls, v):
        seen, out = set(), []
        for h in v:
            if h.ticker not in seen:
                seen.add(h.ticker)
                out.append(h)
        return out

    @field_validator("period")
    @classmethod
    def validate_period(cls, v):
        if v not in VALID_PERIODS:
            raise ValueError(f"period must be one of {VALID_PERIODS}")
        return v


@router.post("/analyze")
async def analyze_portfolio(
    req: PortfolioRequest,
    background_tasks: BackgroundTasks,
    request: Request = None,
):
    ip = get_client_ip(request) if request else "local"
    portfolio_limiter.check(ip)

    if not req.holdings:
        raise HTTPException(400, "No holdings provided")

    tickers     = [h.ticker for h in req.holdings]
    all_tickers = list(set(tickers + ["SPY"]))

    cache_key = f"portfolio:{','.join(sorted(tickers))}:{req.period}"
    cached = cache.get(cache_key)
    if cached:
        return {**cached, "_cached": True}

    try:
        prices = get_multi_close(all_tickers, req.period)
    except MarketDataError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Data fetch failed: {e}")

    # Per-holding P&L
    holdings_out, total_cost, total_value = [], 0.0, 0.0
    for h in req.holdings:
        if h.ticker not in prices.columns:
            continue
        cp  = float(prices[h.ticker].iloc[-1])
        cb  = h.shares * h.avg_cost
        mv  = h.shares * cp
        pnl = mv - cb
        holdings_out.append({
            "ticker":        h.ticker,
            "shares":        h.shares,
            "avg_cost":      h.avg_cost,
            "current_price": round(cp, 2),
            "cost_basis":    round(cb, 2),
            "market_value":  round(mv, 2),
            "pnl":           round(pnl, 2),
            "pnl_pct":       round(pnl / cb * 100, 2) if cb else 0.0,
        })
        total_cost  += cb
        total_value += mv

    if not holdings_out:
        raise HTTPException(404, "None of the provided tickers returned price data")

    valid   = [h["ticker"] for h in holdings_out]
    weights = np.array([h["market_value"] for h in holdings_out])
    weights = weights / weights.sum()

    port_prices  = prices[valid].dropna()
    daily_rets   = port_prices.pct_change().dropna()
    port_ret     = analytics.weighted_portfolio_returns(daily_rets[valid], weights)
    bench_ret    = analytics.daily_returns(prices["SPY"]) if "SPY" in prices.columns else None

    port_eq  = analytics.equity_curve(port_ret)
    bench_eq = analytics.equity_curve(bench_ret) if bench_ret is not None else None

    beta, alpha = analytics.beta_alpha(port_ret, bench_ret) if bench_ret is not None else (0.0, 0.0)

    # Sector (blocking — could move to background, but fast with cache)
    sector_map: dict = {}
    for h in holdings_out:
        sec = get_sector(h["ticker"])
        sector_map[sec] = sector_map.get(sec, 0.0) + h["market_value"]

    # News
    all_news = []
    for t in valid:
        for item in get_news(t):
            all_news.append({"ticker": t, **item})

    dates = [d.strftime("%Y-%m-%d") for d in port_eq.index]

    result = {
        "holdings":      holdings_out,
        "total_cost":    round(total_cost, 2),
        "total_value":   round(total_value, 2),
        "total_pnl":     round(total_value - total_cost, 2),
        "total_pnl_pct": round((total_value - total_cost) / total_cost * 100, 2) if total_cost else 0,
        "allocation": [
            {"ticker": h["ticker"], "value": round(h["market_value"],2),
             "weight": round(h["market_value"]/total_value*100, 2)}
            for h in holdings_out
        ],
        "sectors": [
            {"sector": s, "value": round(v,2), "weight": round(v/total_value*100,2)}
            for s, v in sorted(sector_map.items(), key=lambda x: -x[1])
        ],
        "news": all_news,
        "var_cvar": analytics.var_cvar(port_ret, total_value),
        "metrics": {
            "portfolio_return":  round(float(port_eq.iloc[-1]-1), 4),
            "benchmark_return":  round(float(bench_eq.iloc[-1]-1),4) if bench_eq is not None else None,
            "alpha":             alpha,
            "beta":              beta,
            "sharpe":            round(analytics.sharpe_ratio(port_ret), 3),
            "sortino":           round(analytics.sortino_ratio(port_ret), 3),
            "max_drawdown":      round(analytics.max_drawdown(port_eq), 4),
            "volatility":        round(analytics.annualised_volatility(port_ret), 4),
        },
        "correlation": analytics.correlation_matrix(daily_rets[valid]),
        "montecarlo":  analytics.monte_carlo_gbm(port_ret),
        "equity_curve": {
            "dates":     dates,
            "portfolio": [round(v,4) for v in port_eq.tolist()],
            "benchmark": [round(v,4) for v in bench_eq.reindex(port_eq.index).tolist()]
                         if bench_eq is not None else [],
        },
        "_cached": False,
    }

    cache.set(cache_key, result, ttl=300)
    return result
