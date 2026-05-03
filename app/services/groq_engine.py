"""
groq_engine.py — Groq-only LLM Engine
Uses openai SDK pointed at Groq's OpenAI-compatible endpoint.
Fallback chain: best model -> next model, across GROQ_API_KEY_1 / GROQ_API_KEY_2
"""

import json
import re
import datetime
import time
import os
import hashlib
import threading

from openai import OpenAI, RateLimitError, APIStatusError, APITimeoutError
from dotenv import load_dotenv

load_dotenv(override=True)

# ── Keys & Models ─────────────────────────────────────────────────────────────
GROQ_BASE = "https://api.groq.com/openai/v1"
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "qwen/qwen3-32b",
    "llama-3.1-8b-instant",
]
GROQ_KEYS = list(filter(None, [
    os.getenv("GROQ_API_KEY_1"),
    os.getenv("GROQ_API_KEY_2"),
]))

# One SDK client per key (reused across requests)
_clients: dict[str, OpenAI] = {}
_clients_lock = threading.Lock()

def _get_client(key: str) -> OpenAI:
    if key not in _clients:
        with _clients_lock:
            if key not in _clients:
                _clients[key] = OpenAI(
                    api_key=key,
                    base_url=GROQ_BASE,
                    timeout=12.0,
                    max_retries=0,   # we handle retries ourselves
                )
    return _clients[key]


# ── Cache ─────────────────────────────────────────────────────────────────────
_cache      = {}
_cache_lock = threading.Lock()
_CACHE_TTL  = 300
_CACHE_MAX  = 300

_MAX_USER_TEXT    = 500
_REQUEST_DEADLINE = 25


def _cache_key(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode()).hexdigest()

def _cache_get(text: str):
    with _cache_lock:
        hit = _cache.get(_cache_key(text))
        if hit and (time.time() - hit[1]) < _CACHE_TTL:
            return hit[0]
    return None

def _cache_put(text: str, result: dict):
    with _cache_lock:
        if len(_cache) >= _CACHE_MAX:
            for k in sorted(_cache, key=lambda k: _cache[k][1])[:_CACHE_MAX // 5]:
                _cache.pop(k, None)
        _cache[_cache_key(text)] = (result, time.time())


# ── JSON helpers ──────────────────────────────────────────────────────────────
def _clean_json(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?", "", raw).rstrip("`").strip()
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    return m.group(0) if m else raw

_DEFAULTS = {
    "status": None, "from_date": None, "to_date": None, "limit": 5,
    "priority": None, "city": None, "machine": None, "category": None,
}

def _fill_defaults(parsed: dict) -> dict:
    filters = parsed.setdefault("filters", {})
    for k, v in _DEFAULTS.items():
        filters.setdefault(k, v)
    parsed.setdefault("intents",       ["general_chat"])
    parsed.setdefault("search_target", "")
    parsed.setdefault("specific_items", [])
    parsed.setdefault("reasoning",      "")
    return parsed


# ── System prompt ─────────────────────────────────────────────────────────────
def _build_system_prompt() -> str:
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    return f"""You are mewar erp chatbot, a highly intelligent, friendly shop manager at Mewar.
Today's date is {today}.

--- LANGUAGE RULE ---
If query is in English reply in English. If Hinglish/Hindi reply in Hinglish.

--- NAME SAFETY RULE ---
NEVER shorten, crop, or guess any supplier, project, or company name. Keep it EXACTLY as typed.
Only fix spelling for generic inventory items (bearing, bolt, belt, oil seal, etc).

--- CLEANING RULE (search_target) ---
Remove these words from search_target:
(bhai, dikhao, batao, check, zara, list, latest, last, de, do, please, wale, wala,
 supplier, vendor, party, details, contact, profile, project, site, machine,
 ka, ki, ke, ko, se, aur, or, for, the, a, an)
Example: "Arawali supplier details" -> search_target = "Arawali"
Example: "bearing ka stock kitna hai" -> search_target = "bearing"

--- INTENT MAPPING ---
- "search"          = inventory/stock query
- "supplier_search" = supplier details, contact, GSTIN, mobile, email, city
- "project_search"  = project/site/machine query
- "po_search"       = purchase orders, balance, advance, pending, payments
- "general_chat"    = greetings, off-topic, unclear

--- SMART UNDERSTANDING ---
- "Maal/Stock" -> search | "Paisa/Rokra/Baaki" -> po_search | "Party/Vendor" -> supplier_search
- "phone/mobile/email/contact/gstin" mention + entity name -> ALWAYS supplier_search
- Extract status: pending/draft, completed | priority: urgent, high, low
- Extract dates: "last week" -> from_date/to_date | "this month" -> from_date/to_date
- "last N orders" -> filters.limit = N

--- OUTPUT FORMAT (STRICT JSON ONLY, no extra text) ---
{{
  "intents": ["search"],
  "search_target": "clean entity name",
  "specific_items": [],
  "filters": {{
    "status": null,
    "priority": null,
    "city": null,
    "machine": null,
    "category": null,
    "from_date": null,
    "to_date": null,
    "limit": 5
  }},
  "reasoning": "brief explanation"
}}"""


def _build_messages(user_text: str, history) -> list:
    msgs = [{"role": "system", "content": _build_system_prompt()}]
    for msg in (history or [])[-6:]:
        msgs.append({
            "role":    msg.get("role", "user"),
            "content": str(msg.get("content") or msg.get("raw_content") or ""),
        })
    msgs.append({"role": "user", "content": user_text})
    return msgs


# ── Main entry point ──────────────────────────────────────────────────────────
def ask_local_llm(user_text: str, history=None) -> dict:
    user_text = (user_text or "")[:_MAX_USER_TEXT].strip()
    if not user_text:
        raise RuntimeError("Empty query")

    cached = _cache_get(user_text)
    if cached:
        print("[GROQ-ENGINE] cache hit")
        return cached

    if not GROQ_KEYS:
        raise RuntimeError("No Groq API keys configured. Set GROQ_API_KEY_1 in .env")

    messages = _build_messages(user_text, history)
    deadline  = time.time() + _REQUEST_DEADLINE
    last_err  = None

    for model in GROQ_MODELS:
        for key in GROQ_KEYS:
            if time.time() >= deadline:
                raise RuntimeError("Groq request deadline exceeded (25s)")
            try:
                client   = _get_client(key)
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=400,
                    response_format={"type": "json_object"},
                )
                raw    = response.choices[0].message.content
                result = _fill_defaults(json.loads(_clean_json(raw)))
                _cache_put(user_text, result)
                print(f"[GROQ-ENGINE] answered by {model}")
                return result
            except RateLimitError:
                print(f"[GROQ-ENGINE] {model} key=...{key[-6:]} rate limited, trying next")
                continue
            except APITimeoutError:
                print(f"[GROQ-ENGINE] {model} key=...{key[-6:]} timed out")
                last_err = "timeout"
            except APIStatusError as e:
                print(f"[GROQ-ENGINE] {model} key=...{key[-6:]} status {e.status_code}: {str(e)[:80]}")
                last_err = e
            except Exception as e:
                last_err = e
                print(f"[GROQ-ENGINE] {model} key=...{key[-6:]} failed: {str(e)[:80]}")

    raise RuntimeError(f"All Groq models failed: {last_err}")


def health_check() -> dict:
    return {"groq": bool(GROQ_KEYS), "keys_configured": len(GROQ_KEYS)}
