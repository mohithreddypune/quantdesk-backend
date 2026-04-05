import os
import time
import logging
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from cache import cache
from routers import backtest, portfolio, watchlist


# ── Structured logging ───────────────────────────────────────────────────────

class JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "time":    self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level":   record.levelname,
            "message": record.getMessage(),
            "module":  record.module,
        })

handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logging.basicConfig(handlers=[handler], level=logging.INFO)
logger = logging.getLogger("quantdesk")


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info({"event": "startup", "service": "quantdesk-api"})
    yield
    cache.clear()
    logger.info({"event": "shutdown"})


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "QuantDesk API",
    version     = "2.0.0",
    description = "Production-grade stock analytics API",
    lifespan    = lifespan,
)


# ── CORS ─────────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "https://quantdesk-frontend-hwn2.vercel.app,http://localhost:3000,http://localhost:3001",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

# ── Request timing middleware ─────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = round((time.perf_counter() - start) * 1000, 2)

    logger.info({
        "method":   request.method,
        "path":     request.url.path,
        "status":   response.status_code,
        "ms":       duration,
        "ip":       request.client.host if request.client else "unknown",
    })

    response.headers["X-Response-Time"] = f"{duration}ms"
    return response


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(backtest.router)
app.include_router(portfolio.router)
app.include_router(watchlist.router)


# ── Utility endpoints ────────────────────────────────────────────────────────

@app.get("/", tags=["meta"])
def root():
    return {"service": "QuantDesk API", "version": "2.0.0", "status": "ok"}


@app.get("/health", tags=["meta"])
def health():
    """Used by Railway/Vercel for health checks."""
    return {"status": "healthy", "cache": cache.stats()}


@app.get("/cache/stats", tags=["meta"])
def cache_stats():
    return cache.stats()


@app.delete("/cache", tags=["meta"])
def clear_cache():
    cache.clear()
    return {"cleared": True}
