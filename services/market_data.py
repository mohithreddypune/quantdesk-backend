"""
services/market_data.py — All yfinance interactions live here.

Routers never touch yfinance directly. This layer:
  - Normalises the MultiIndex mess yfinance returns
  - Caches results with configurable TTL
  - Raises consistent domain exceptions (not HTTP errors — that's the router's job)
  - Is independently testable with mocked data
"""
import hashlib
import yfinance as yf
import pandas as pd
from typing import List

from cache import cache


class MarketDataError(Exception):
    """Raised when we can't get clean data for a ticker."""


# ── Cache TTLs ──────────────────────────────────────────────────────────────
PRICE_TTL   = 300   # 5 min  — historical OHLC
INFO_TTL    = 3600  # 1 hour — company metadata (sector, name)
NEWS_TTL    = 600   # 10 min — news feed
FAST_TTL    = 30    # 30 sec — live quote (watchlist)


def _cache_key(*parts) -> str:
    raw = ":".join(str(p) for p in parts)
    return hashlib.md5(raw.encode()).hexdigest()


# ── Price data ──────────────────────────────────────────────────────────────

def get_close_series(ticker: str, period: str) -> pd.Series:
    """Return a clean daily Close pd.Series for one ticker, cached."""
    key = _cache_key("close", ticker.upper(), period)
    cached = cache.get(key)
    if cached is not None:
        return cached

    raw = yf.download(
        ticker.upper(),
        period=period,
        auto_adjust=True,
        progress=False,
    )

    if raw is None or raw.empty:
        raise MarketDataError(f"No price data returned for {ticker}")

    # Normalise MultiIndex (yfinance changed this in 0.2.x)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = ["_".join(c).strip() for c in raw.columns]
        close_col = next((c for c in raw.columns if "Close" in c), None)
        if close_col is None:
            raise MarketDataError(f"No Close column found for {ticker}")
        series = raw[close_col].squeeze()
    else:
        if "Close" not in raw.columns:
            raise MarketDataError(f"No Close column found for {ticker}")
        series = raw["Close"].squeeze()

    series = series.dropna()
    if len(series) < 10:
        raise MarketDataError(f"Insufficient price history for {ticker} ({len(series)} rows)")

    cache.set(key, series, ttl=PRICE_TTL)
    return series


def get_multi_close(tickers: List[str], period: str) -> pd.DataFrame:
    """Return a DataFrame of Close prices for multiple tickers, cached."""
    key = _cache_key("multi_close", ",".join(sorted(tickers)), period)
    cached = cache.get(key)
    if cached is not None:
        return cached

    upper = [t.upper() for t in tickers]
    raw = yf.download(
        upper,
        period=period,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    closes: dict[str, pd.Series] = {}
    for t in upper:
        try:
            if len(upper) == 1:
                col = next((c for c in raw.columns if "Close" in str(c)), None)
                closes[t] = raw[col].squeeze() if col else pd.Series(dtype=float)
            else:
                closes[t] = raw[t]["Close"].squeeze()
        except Exception:
            pass  # ticker may not trade on all days — handled downstream

    df = pd.DataFrame(closes).dropna()
    if df.empty:
        raise MarketDataError("No overlapping price data for the given tickers and period")

    cache.set(key, df, ttl=PRICE_TTL)
    return df


def get_live_quote(ticker: str) -> dict:
    """Return latest price + change, cached for FAST_TTL seconds."""
    key = _cache_key("quote", ticker.upper())
    cached = cache.get(key)
    if cached is not None:
        return cached

    try:
        info  = yf.Ticker(ticker.upper()).fast_info
        price = float(info.last_price or 0)
        prev  = float(info.previous_close or price)
        chg   = price - prev
        pct   = (chg / prev * 100) if prev else 0.0
        result = {
            "ticker":     ticker.upper(),
            "price":      round(price, 2),
            "change":     round(chg, 2),
            "change_pct": round(pct, 2),
            "prev_close": round(prev, 2),
        }
    except Exception as e:
        result = {
            "ticker": ticker.upper(),
            "price": None, "change": None,
            "change_pct": None, "prev_close": None,
            "error": str(e),
        }

    cache.set(key, result, ttl=FAST_TTL)
    return result


# ── Company metadata ─────────────────────────────────────────────────────────

def get_sector(ticker: str) -> str:
    key = _cache_key("sector", ticker.upper())
    cached = cache.get(key)
    if cached is not None:
        return cached
    try:
        sector = yf.Ticker(ticker.upper()).info.get("sector", "Unknown")
    except Exception:
        sector = "Unknown"
    cache.set(key, sector, ttl=INFO_TTL)
    return sector


def get_news(ticker: str, n: int = 4) -> list:
    key = _cache_key("news", ticker.upper(), n)
    cached = cache.get(key)
    if cached is not None:
        return cached

    try:
        items = yf.Ticker(ticker.upper()).news or []
        out = []
        for item in items[:n]:
            c = item.get("content", {})
            out.append({
                "title":     c.get("title",    item.get("title", "")),
                "publisher": c.get("provider", {}).get("displayName", item.get("publisher", "")),
                "link":      c.get("canonicalUrl", {}).get("url", item.get("link", "")),
                "published": c.get("pubDate",  ""),
            })
    except Exception:
        out = []

    cache.set(key, out, ttl=NEWS_TTL)
    return out
