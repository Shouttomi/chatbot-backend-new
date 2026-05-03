# Production Deployment

## What changed for production readiness

| # | Change | File |
|---|---|---|
| 1 | Dropped Groq from LLM fallback chain (tokens exhausted). Chain is now **OpenRouter → Cerebras → Ollama**. | `app/services/v2_ollama_engine.py` |
| 2 | Thread-safe locks on the LLM cache and Cerebras client singleton (was racy under concurrent requests). | `app/services/v2_ollama_engine.py` |
| 3 | LLM cache is now bounded (max 500 entries, evicts oldest 20% when full) — was an unbounded memory leak. | `app/services/v2_ollama_engine.py` |
| 4 | Input length cap (500 chars) and reasoning cap (400 chars) to protect upstream APIs from token bombs. | `app/services/v2_ollama_engine.py` |
| 5 | DB pool: `pool_size=20`, `max_overflow=30`, `pool_timeout=30s`, `pool_recycle=1800s`. Old default of 5 would choke past ~20 concurrent users. | `app/db/database.py` |
| 6 | `/health` endpoint reports DB + LLM provider status — wire to your uptime monitor. | `app/main.py` |
| 7 | Per-IP rate limit: **30 requests / 60s** on `/v2-chatbot/*`. Returns 429 with Retry-After. | `app/middleware/rate_limit.py` |
| 8 | Request body size cap: **8 KB** on chatbot endpoints. | `app/middleware/rate_limit.py` |

## How to run in production

### Linux / macOS (recommended)

```bash
gunicorn app.main:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  -b 0.0.0.0:8000 \
  --timeout 60 \
  --graceful-timeout 30 \
  --access-logfile - \
  --error-logfile -
```

- `-w 4`: workers. Rule of thumb = `2 * CPU_cores + 1`. Each worker holds its own DB pool, so 4 workers × 20 = 80 steady DB connections — confirm your MySQL `max_connections` is at least 100.
- `--timeout 60`: kill workers stuck longer than this (LLM call timeout is 20s, so 60s is safe headroom).

### Windows (dev only — gunicorn doesn't run on Windows)

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Behind a reverse proxy (nginx/Cloudflare)

The rate limiter reads `X-Forwarded-For` for the real client IP. Make sure your proxy sets it. Without it, every request looks like it comes from the proxy IP and the limit is global.

## Required env vars

```
DATABASE_URL=mysql+pymysql://user:pass@host:3306/dbname
OPENROUTER_API_KEY=sk-or-...      # PRIMARY
CEREBRAS_API_KEY=csk-...          # FALLBACK
# GROQ_API_KEY_1, GROQ_API_KEY_2 — currently unused (out of tokens)
OLLAMA_BASE_URL=http://localhost:11434   # last-resort local fallback
LOCAL_MODEL=qwen2.5:7b
```

## Smoke test after deploy

```bash
curl https://your-host/health
# Expect: {"status": "ok", "db": {"ok": true, ...}, "llm": {...}}

curl -X POST https://your-host/v2-chatbot/ \
  -H "Content-Type: application/json" \
  -d '{"query": "hello", "history": []}'
```

## Known limits / next steps

- **Rate limiter is per-worker, in-memory.** With 4 workers, the effective limit is ~120 req/min/IP. If you scale beyond one machine or want exact limits, swap the in-memory dict in `rate_limit.py` for Redis.
- **No prometheus metrics.** Add `prometheus-fastapi-instrumentator` if you want latency/error histograms.
- **Groq is commented out, not deleted** in `ask_local_llm`. To re-enable after funding a new key, uncomment the `_call_groq(messages)` step.
