import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.db.database import engine
from app.middleware.rate_limit import RateLimitMiddleware
from app.routers.auth import router as auth_router
from app.routers.inventory_dropdown import router as inventory_router
from app.routers.inventory_smart import router as inventory_smart_router
from app.routers.v2_chatbot import router as v2_chatbot_router
from app.routers.chatbot3 import router as chatbot3_router
from app.services.v2_ollama_engine import health_check as llm_health
from app.services.entity_resolver import cache_stats as resolver_stats

app = FastAPI()

_cors_raw = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:8000,http://localhost:8000"
).strip()

# If CORS_ORIGINS is "*", allow all origins (public API)
if _cors_raw == "*":
    allowed_origins = ["*"]
    _allow_credentials = False   # browsers reject credentials + wildcard
else:
    allowed_origins = [o.strip() for o in _cors_raw.split(",") if o.strip()]
    _allow_credentials = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Per-IP rate limit on chatbot to protect upstream LLM tokens & DB pool.
app.add_middleware(
    RateLimitMiddleware,
    path_prefix="/v2-chatbot",
    max_requests=30,
    window_seconds=60,
)

app.include_router(v2_chatbot_router)
app.include_router(chatbot3_router)
app.include_router(auth_router)
app.include_router(inventory_router)
app.include_router(inventory_smart_router)


@app.get("/")
def root():
    return {"message": "Mewar ERP API running"}


@app.get("/health")
def health():
    """Liveness + dependency health for load balancers / uptime monitors."""
    db_ok = True
    db_err = None
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        db_ok = False
        db_err = str(e)[:200]
    llm = llm_health()
    status = "ok" if db_ok and (llm.get("openrouter") or llm.get("cerebras") or llm.get("ollama")) else "degraded"
    return {
        "status": status,
        "db": {"ok": db_ok, "error": db_err},
        "llm": llm,
        "entity_cache": resolver_stats(),
    }

# @app.get("/che     ck-db")
# def check_db(db: Session = Depends(get_db)):
#     result = db.execute(text("SHOW TABLES;"))
#     tables = result.fetchall()
#     return [row[0] for row in tables]

# @app.get("/inventory")
# def get_inventory(db: Session = Depends(get_db)):
#     result = db.execute(text("SELECT * FROM inventories;"))
#     rows = result.fetchall()
#     return {
#         "table": "inventories",
#         "count": len(rows),
#         "data": [dict(row._mapping) for row in rows]
#     }