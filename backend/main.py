"""FastAPI backend: REST API for the frontend + the MCP server over streamable HTTP.

Endpoints:
  POST /ask       -> run the agent (rate-limited)
  GET  /health    -> liveness + agent version
  GET  /metrics   -> monitor snapshot (cost, error rates, alerts)
  *    /mcp       -> MCP streamable-HTTP endpoint (API-key protected)
"""
import contextlib
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src import agent  # noqa: E402
from src.mcp_server import mcp  # noqa: E402
from src.monitor import MONITOR  # noqa: E402

MCP_API_KEY = os.getenv("MCP_API_KEY", "")

mcp_app = mcp.streamable_http_app()


@contextlib.asynccontextmanager
async def lifespan(app):
    async with mcp.session_manager.run():
        yield


app = FastAPI(title="WattWise (prodagent)", version=agent.AGENT_VERSION["version"],
              lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple per-IP rate limit: 10 requests / minute on /ask (demo-to-production gap fix)
_hits: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT, RATE_WINDOW = 10, 60.0


def _rate_limited(ip: str) -> bool:
    now = time.time()
    _hits[ip] = [t for t in _hits[ip] if now - t < RATE_WINDOW]
    if len(_hits[ip]) >= RATE_LIMIT:
        return True
    _hits[ip].append(now)
    return False


@app.middleware("http")
async def mcp_auth(request: Request, call_next):
    """API key on the MCP endpoint: Authorization: Bearer <key> or ?key=<key>."""
    if request.url.path.startswith("/mcp") and MCP_API_KEY:
        supplied = request.headers.get("authorization", "").removeprefix("Bearer ").strip() \
            or request.query_params.get("key", "")
        if supplied != MCP_API_KEY:
            from starlette.responses import JSONResponse
            return JSONResponse({"error": "invalid or missing API key"}, status_code=401)
    return await call_next(request)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


@app.post("/ask")
def ask(body: AskRequest, request: Request):
    ip = request.client.host if request.client else "unknown"
    if _rate_limited(ip):
        raise HTTPException(429, "Rate limit: 10 requests per minute.")
    try:
        result = agent.run(body.question)
    except RuntimeError as e:            # budget / quota exceeded
        raise HTTPException(402, str(e))
    except Exception as e:
        raise HTTPException(500, f"Agent error: {type(e).__name__}: {e}")
    result.pop("version", None)
    return {**result, "agent_version": agent.AGENT_VERSION}


@app.get("/health")
def health():
    return {"status": "ok", "agent_version": agent.AGENT_VERSION,
            "mcp_endpoint": "/mcp", "disclosure": agent.DISCLOSURE}


@app.get("/metrics")
def metrics():
    return MONITOR.snapshot()


app.mount("/", mcp_app)   # serves the MCP protocol at /mcp
