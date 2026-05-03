"""
Per-IP sliding-window rate limiter + request body size cap.

In-memory only — fine for a single-process deployment. If we scale to
multiple gunicorn workers, swap the backing dict for Redis (each worker
keeps its own counter today, so the effective limit is N * max_requests).
"""

import time
import threading
from collections import defaultdict, deque
from typing import Deque, Dict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


_MAX_BODY_BYTES = 8 * 1024  # 8 KB — chat queries are tiny; larger = abuse


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, path_prefix: str, max_requests: int, window_seconds: int):
        super().__init__(app)
        self.path_prefix = path_prefix
        self.max_requests = max_requests
        self.window = window_seconds
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _client_id(self, request: Request) -> str:
        # Honor X-Forwarded-For when behind a reverse proxy / load balancer
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            return fwd.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if not path.startswith(self.path_prefix):
            return await call_next(request)

        # Body size cap — Content-Length is cheap to check upfront
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > _MAX_BODY_BYTES:
            return JSONResponse(
                {"results": [{"type": "chat", "message": "Request too large."}]},
                status_code=413,
            )

        client = self._client_id(request)
        now = time.time()
        cutoff = now - self.window

        with self._lock:
            q = self._hits[client]
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) >= self.max_requests:
                retry = int(self.window - (now - q[0])) + 1
                return JSONResponse(
                    {"results": [{"type": "chat", "message": f"Bhai thoda dheere, {retry}s ruk jao."}]},
                    status_code=429,
                    headers={"Retry-After": str(retry)},
                )
            q.append(now)

        return await call_next(request)
