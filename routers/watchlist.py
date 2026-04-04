"""
routers/watchlist.py — REST + WebSocket for live price streaming.

WebSocket upgrades the watchlist from "30s polling" to real-time push,
which is what signals engineering maturity.
"""
import asyncio
import json
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect, Query
from typing import List

from cache import cache
from rate_limit import watchlist_limiter, get_client_ip
from services.market_data import get_live_quote

router = APIRouter(prefix="/watchlist", tags=["watchlist"])


# ── REST endpoint (initial load) ─────────────────────────────────────────────

@router.get("/prices")
async def get_prices(
    tickers: str = Query(..., description="Comma-separated e.g. AAPL,TSLA"),
    request: Request = None,
):
    ip = get_client_ip(request) if request else "local"
    watchlist_limiter.check(ip)

    symbols = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not symbols:
        raise HTTPException(400, "No tickers provided")
    if len(symbols) > 20:
        raise HTTPException(400, "Maximum 20 tickers per request")

    return [get_live_quote(sym) for sym in symbols]


# ── WebSocket endpoint (streaming updates) ───────────────────────────────────

@router.websocket("/ws")
async def watchlist_ws(websocket: WebSocket):
    """
    Protocol:
      Client sends:  {"tickers": ["AAPL", "TSLA", ...], "interval": 30}
      Server sends:  {"type": "prices", "data": [...], "timestamp": "..."}
      Server sends:  {"type": "error",  "message": "..."}

    Client can send updated tickers list at any time to change subscription.
    """
    await websocket.accept()
    tickers: List[str] = []
    interval: int = 30

    try:
        while True:
            # Check for new client message (non-blocking)
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=0.5)
                msg = json.loads(raw)
                tickers  = [t.upper().strip() for t in msg.get("tickers", []) if t.strip()][:20]
                interval = max(10, min(300, int(msg.get("interval", 30))))
            except asyncio.TimeoutError:
                pass  # no new message, continue with current subscription
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            if not tickers:
                await asyncio.sleep(1)
                continue

            # Fetch prices and broadcast
            prices = [get_live_quote(t) for t in tickers]
            from datetime import datetime, timezone
            await websocket.send_json({
                "type":      "prices",
                "data":      prices,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            await asyncio.sleep(interval)

    except WebSocketDisconnect:
        pass  # client closed connection — clean exit
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
