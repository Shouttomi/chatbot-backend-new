"""
v4_llm_engine.py — Production LLM Engine

Fallback Chain:
Cerebras (qwen3-235b) → Gemini (2.5-pro) → Groq (llama-3.3-70b) → OpenRouter (free) → Ollama
"""

import json
import re
import datetime
import time
import os
import hashlib
import threading
import requests

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

# ✅ Cerebras SDK
try:
    from cerebras.cloud.sdk import Cerebras
except ImportError:
    Cerebras = None

load_dotenv(override=True)

# ─── Groq (best → worst, verified against /v1/models 2026-04-27) ────
GROQ_BASE   = "https://api.groq.com/openai/v1"
GROQ_MODELS = [
    "openai/gpt-oss-120b",                       # 1st — 120B GPT OSS, most capable on Groq
    "llama-3.3-70b-versatile",                   # 2nd — 70B dense, proven & reliable
    "meta-llama/llama-4-scout-17b-16e-instruct", # 3rd — Llama-4 Scout (newest gen)
    "qwen/qwen3-32b",                            # 4th — Qwen3 32B, strong reasoning
    "groq/compound",                             # 5th — Groq compound (tool-augmented)
    "groq/compound-mini",                        # 6th — lighter compound model
    "llama-3.1-8b-instant",                      # 7th — fastest, smallest last resort
]
GROQ_KEYS  = list(filter(None, [
    os.getenv("GROQ_API_KEY_1"),
    os.getenv("GROQ_API_KEY_2"),
]))

# ─── Cerebras (best → worst, verified against SDK 2026-04-27) ───────
CEREBRAS_MODELS = [
    "qwen-3-235b-a22b-instruct-2507",  # 1st — 235B MoE flagship, best quality
    "llama3.1-8b",                     # 2nd — lightweight fallback
]
CEREBRAS_KEY = os.getenv("CEREBRAS_API_KEY", "")

# ─── Gemini (best → worst, free API quota) ──────────────────────────
GEMINI_MODELS = [
    "gemini-2.5-pro",    # 1st — most intelligent Google model, free quota
    "gemini-2.5-flash",  # 2nd — faster, slightly less capable
    "gemini-2.0-flash",  # 3rd — stable older release, last resort
]
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")

# ─── OpenRouter (best → worst, free models verified 2026-04-27) ─────
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODELS = [
    "nousresearch/hermes-3-llama-3.1-405b:free",    # 1st — 405B, most powerful free
    "openai/gpt-oss-120b:free",                      # 2nd — 120B GPT OSS
    "nvidia/nemotron-3-super-120b-a12b:free",        # 3rd — 120B MoE, strong reasoning
    "qwen/qwen3-next-80b-a3b-instruct:free",         # 4th — Qwen3 80B MoE, very new
    "meta-llama/llama-3.3-70b-instruct:free",        # 5th — reliable open 70B
    "google/gemma-4-31b-it:free",                    # 6th — Gemma 4 31B
    "google/gemma-3-27b-it:free",                    # 7th — Gemma 3 27B, stable fallback
]

# ─── DeepSeek ────────────────────────────────────────────────────────
DEEPSEEK_KEY    = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE   = "https://api.deepseek.com"
DEEPSEEK_MODELS = [
    "deepseek-chat",      # 1st — DeepSeek-V3, fast + cheap, great for structured JSON
    "deepseek-reasoner",  # 2nd — R1 reasoning model, slower but stronger
]

# ─── Ollama ─────────────────────────────────────────────────────────
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "qwen2.5:7b")

# ─── HTTP Session ───────────────────────────────────────────────────
_session = requests.Session()
_retry   = Retry(
    total=2,
    backoff_factor=0.4,
    status_forcelist=[500, 502, 503, 504],
    allowed_methods=["POST", "GET"],
)
_session.mount("http://",  HTTPAdapter(max_retries=_retry))
_session.mount("https://", HTTPAdapter(max_retries=_retry))

# ─── Cache (thread-safe, bounded) ───────────────────────────────────
_cache = {}
_cache_lock = threading.Lock()
_CACHE_TTL = 300
_CACHE_MAX = 500
_MAX_USER_TEXT = 500
_MAX_REASONING = 400

def _cache_key(text):
    return hashlib.md5(text.strip().lower().encode()).hexdigest()

def _cache_get(text):
    with _cache_lock:
        hit = _cache.get(_cache_key(text))
        if hit and (time.time() - hit[1]) < _CACHE_TTL:
            return hit[0]
    return None

def _cache_put(text, result):
    with _cache_lock:
        if len(_cache) >= _CACHE_MAX:
            # evict oldest 20%
            for k in sorted(_cache, key=lambda k: _cache[k][1])[: _CACHE_MAX // 5]:
                _cache.pop(k, None)
        _cache[_cache_key(text)] = (result, time.time())

# ─── Defaults ───────────────────────────────────────────────────────
_DEFAULTS = {
    "status": None, "from_date": None, "to_date": None, "limit": 5,
    "priority": None, "city": None, "machine": None, "category": None,
}

# Top-level fields beyond filters that the LLM may emit for complex queries.
_TOP_DEFAULTS = {
    "secondary_target": "",   # for comparisons: "Arawali vs DCL" -> primary=Arawali, secondary=DCL
    "aggregation": None,      # one of: "top_n","sum","max","min","count","compare","none"
    "aggregation_field": None,# field to aggregate over: "total_amount","balance_amount","spend"
    "negate": False,          # invert the filter (e.g. "NOT in Rajsamand")
    "comparison": None,       # {"op": "gt"|"lt"|"gte"|"lte"|"eq", "value": <number>}
    "group_by": None,         # "supplier","city","status","month","project"
}

def _build_system_prompt() -> str:
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    # Live DB schema injected so the LLM doesn't guess column names.
    # Lazy-loaded on first call; cached for process lifetime.
    try:
        from app.services.schema_doc import get_schema_text
        schema_block = get_schema_text() or ""
    except Exception as e:
        print(f"[PROMPT] schema doc unavailable: {e}")
        schema_block = ""
    schema_section = f"\n\n{schema_block}\n" if schema_block else ""
    return f"""You are mewar erp chatbot, a highly intelligent, friendly, and human-like shop manager at Mewar.
Today's date is {today}.{schema_section}
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
- "phone/mobile/email/contact/gstin" mention + entity name -> ALWAYS supplier_search (party contact details)
- Extract status: pending/draft, completed | priority: urgent, high, low
- Extract dates: "last week" -> from_date/to_date | "this month" -> from_date/to_date
- "last N orders" -> filters.limit = N

--- OUTPUT FORMAT (STRICT JSON ONLY, no extra text) ---
{{
  "intents": ["search"],
  "search_target": "clean entity name",
  "secondary_target": "",
  "specific_items": [],
  "aggregation": null,
  "aggregation_field": null,
  "negate": false,
  "comparison": null,
  "group_by": null,
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
  "reasoning": "human-like casual reply in user language"
}}

--- COMPLEX QUERY FIELDS (use only when needed) ---
- secondary_target: second entity for comparison queries ("Arawali vs DCL" -> primary=Arawali, secondary=DCL)
- aggregation: one of "top_n", "sum", "max", "min", "count", "compare" (or null for normal queries)
- aggregation_field: what to aggregate — "total_amount", "balance_amount", "spend"
- negate: true if user said NOT/except/excluding ("projects NOT in Rajsamand")
- comparison: {{"op": "gt"|"lt"|"gte"|"lte", "value": <number>}} for "balance > 50000"
- group_by: "supplier","city","status","month" — for grouped aggregations

--- EXAMPLES ---
User: "bearing ka stock kitna hai"
AI: {{"intents": ["search"], "search_target": "bearing", "specific_items": [], "filters": {{"status": null, "priority": null, "city": null, "machine": null, "category": null, "from_date": null, "to_date": null, "limit": 5}}, "reasoning": "ek sec bearing ka stock dekh raha hoon"}}

User: "Arawali supplier details"
AI: {{"intents": ["supplier_search"], "search_target": "Arawali", "specific_items": [], "filters": {{"status": null, "priority": null, "city": null, "machine": null, "category": null, "from_date": null, "to_date": null, "limit": 5}}, "reasoning": "Arawali ki details nikaalte hain"}}

User: "pending orders dikhao"
AI: {{"intents": ["po_search"], "search_target": "", "specific_items": [], "filters": {{"status": "draft", "priority": null, "city": null, "machine": null, "category": null, "from_date": null, "to_date": null, "limit": 10}}, "reasoning": "pending orders check karta hoon"}}

User: "urgent projects dikhao"
AI: {{"intents": ["project_search"], "search_target": "", "specific_items": [], "filters": {{"status": null, "priority": "urgent", "city": null, "machine": null, "category": null, "from_date": null, "to_date": null, "limit": 5}}, "reasoning": "urgent projects dekh raha hoon"}}

User: "total gst kitna bana sabka"
AI: {{"intents": ["po_search"], "search_target": "", "specific_items": [], "filters": {{"status": null, "priority": null, "city": null, "machine": null, "category": null, "from_date": null, "to_date": null, "limit": 5}}, "reasoning": "total GST calculate karta hoon"}}

User: "DCL ki detail aur orders"
AI: {{"intents": ["supplier_search", "po_search"], "search_target": "DCL", "specific_items": [], "filters": {{"status": null, "priority": null, "city": null, "machine": null, "category": null, "from_date": null, "to_date": null, "limit": 5}}, "reasoning": "DCL ki details aur orders dono nikaalte hain"}}

User: "Rajsamand ke projects"
AI: {{"intents": ["project_search"], "search_target": "Rajsamand", "specific_items": [], "filters": {{"status": null, "priority": null, "city": null, "machine": null, "category": null, "from_date": null, "to_date": null, "limit": 5}}, "reasoning": "Rajsamand ke projects check karta hoon"}}

User: "last 5 orders"
AI: {{"intents": ["po_search"], "search_target": "", "specific_items": [], "filters": {{"status": null, "priority": null, "city": null, "machine": null, "category": null, "from_date": null, "to_date": null, "limit": 5}}, "reasoning": "last 5 orders dekh raha hoon"}}

User: "hello"
AI: {{"intents": ["general_chat"], "search_target": "", "specific_items": [], "filters": {{"status": null, "priority": null, "city": null, "machine": null, "category": null, "from_date": null, "to_date": null, "limit": 5}}, "reasoning": "Haan bhai! Kya haal hai? Inventory, suppliers ya orders mein kuch chahiye?"}}

User: "advance diya gaya hai kisi ko"
AI: {{"intents": ["po_search"], "search_target": "", "specific_items": [], "filters": {{"status": null, "priority": null, "city": null, "machine": null, "category": null, "from_date": null, "to_date": null, "limit": 10}}, "reasoning": "advance diye gaye orders check karta hoon"}}

--- COMPLEX QUERY EXAMPLES ---

User: "top 5 suppliers by spend"
AI: {{"intents": ["po_search"], "search_target": "", "aggregation": "top_n", "aggregation_field": "total_amount", "group_by": "supplier", "filters": {{"limit": 5, "status": null, "priority": null, "city": null, "machine": null, "category": null, "from_date": null, "to_date": null}}, "reasoning": "top 5 suppliers by spend nikaalta hoon"}}

User: "sabse bada PO kiska hai"
AI: {{"intents": ["po_search"], "search_target": "", "aggregation": "max", "aggregation_field": "total_amount", "filters": {{"limit": 1, "status": null, "priority": null, "city": null, "machine": null, "category": null, "from_date": null, "to_date": null}}, "reasoning": "sabse bada PO check karta hoon"}}

User: "Arawali vs DCL pending balance"
AI: {{"intents": ["po_search"], "search_target": "Arawali", "secondary_target": "DCL", "aggregation": "compare", "aggregation_field": "balance_amount", "filters": {{"limit": 5, "status": null, "priority": null, "city": null, "machine": null, "category": null, "from_date": null, "to_date": null}}, "reasoning": "Arawali aur DCL ka pending balance compare karta hoon"}}

User: "projects NOT in Rajsamand"
AI: {{"intents": ["project_search"], "search_target": "", "negate": true, "filters": {{"city": "Rajsamand", "limit": 10, "status": null, "priority": null, "machine": null, "category": null, "from_date": null, "to_date": null}}, "reasoning": "Rajsamand ke alawa baaki projects nikaalta hoon"}}

User: "POs with balance over 50000"
AI: {{"intents": ["po_search"], "search_target": "", "comparison": {{"op": "gt", "value": 50000}}, "aggregation_field": "balance_amount", "filters": {{"limit": 10, "status": null, "priority": null, "city": null, "machine": null, "category": null, "from_date": null, "to_date": null}}, "reasoning": "50000 se zyada balance wale POs check karta hoon"}}

User: "this month spend by supplier"
AI: {{"intents": ["po_search"], "search_target": "", "aggregation": "sum", "aggregation_field": "total_amount", "group_by": "supplier", "filters": {{"limit": 10, "from_date": "month_start", "to_date": "today", "status": null, "priority": null, "city": null, "machine": null, "category": null}}, "reasoning": "is mahine supplier-wise spend nikaalta hoon"}}
"""

# ─── Helpers ────────────────────────────────────────────────────────
def _clean_json(text):
    text = re.sub(r"^```json\s*", "", text.strip())
    text = re.sub(r"```$", "", text)
    s, e = text.find("{"), text.rfind("}") + 1
    return text[s:e] if s != -1 else text

def _fill_defaults(parsed):
    if "filters" not in parsed:
        parsed["filters"] = dict(_DEFAULTS)
    for k, v in _DEFAULTS.items():
        parsed["filters"].setdefault(k, v)
    parsed.setdefault("intents", ["search"])
    parsed.setdefault("search_target", "")
    parsed.setdefault("specific_items", [])
    parsed.setdefault("reasoning", "...")
    for k, v in _TOP_DEFAULTS.items():
        parsed.setdefault(k, v)
    return parsed

def _build_messages(user_text, history):
    msgs = [{"role": "system", "content": _build_system_prompt()}]
    for msg in (history or [])[-6:]:
        msgs.append({
            "role": msg.get("role", "user"),
            "content": str(msg.get("content") or msg.get("raw_content") or "")
        })
    msgs.append({"role": "user", "content": user_text})
    return msgs

# ─── Generic OpenAI-style (Groq only) ───────────────────────────────
def _call_cloud(base, model, key, messages):
    resp = _session.post(
        f"{base}/chat/completions",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 400,
            "response_format": {"type": "json_object"},
        },
        timeout=15,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]
    return _fill_defaults(json.loads(_clean_json(raw)))

# ─── Groq ───────────────────────────────────────────────────────────
def _call_groq(messages):
    last_err = None
    for model in GROQ_MODELS:
        for key in GROQ_KEYS:
            try:
                return _call_cloud(GROQ_BASE, model, key, messages)
            except Exception as e:
                last_err = e
                print(f"[GROQ] {model} key=...{key[-6:]} failed - {str(e)[:80]}")
    raise RuntimeError(f"Groq failed: {last_err}")

# ─── Cerebras (PRIMARY SDK) ─────────────────────────────────────────
_cerebras_client = None
_cerebras_lock = threading.Lock()

def _get_cerebras():
    global _cerebras_client
    if Cerebras is None:
        raise RuntimeError("cerebras-cloud-sdk is not installed")
    if _cerebras_client is None:
        with _cerebras_lock:
            if _cerebras_client is None:
                if not CEREBRAS_KEY:
                    raise RuntimeError("No CEREBRAS_API_KEY")
                _cerebras_client = Cerebras(api_key=CEREBRAS_KEY)
    return _cerebras_client

def _call_cerebras(messages):
    client = _get_cerebras()
    last_err = None
    for model in CEREBRAS_MODELS:
        try:
            resp = client.chat.completions.create(model=model, messages=messages)
            raw = resp.choices[0].message.content
            print(f"[CEREBRAS] {model}: {raw[:120]}")
            return _fill_defaults(json.loads(_clean_json(raw)))
        except Exception as e:
            last_err = e
            print(f"[CEREBRAS] {model} failed - {str(e)[:80]}")
    raise RuntimeError(f"Cerebras failed: {last_err}")

# ─── Gemini (Google) ────────────────────────────────────────────────
def _to_gemini_contents(messages):
    """Translate OpenAI-style [{role,content}] into Gemini's contents + systemInstruction."""
    sys_text = ""
    contents = []
    for m in messages:
        role, txt = m.get("role"), m.get("content", "") or ""
        if role == "system":
            sys_text = (sys_text + "\n\n" + txt).strip()
        else:
            contents.append({
                "role": "user" if role != "assistant" else "model",
                "parts": [{"text": txt}],
            })
    return sys_text, contents


def _call_gemini(messages):
    if not GEMINI_KEY:
        raise RuntimeError("No GEMINI_API_KEY")
    sys_text, contents = _to_gemini_contents(messages)
    body = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 400,
            "responseMimeType": "application/json",
        },
    }
    if sys_text:
        body["systemInstruction"] = {"parts": [{"text": sys_text}]}

    last_err = None
    for model in GEMINI_MODELS:
        try:
            resp = _session.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                params={"key": GEMINI_KEY},
                headers={"Content-Type": "application/json"},
                json=body,
                timeout=20,
            )
            if resp.status_code == 429:
                raise RuntimeError("Rate limited")
            resp.raise_for_status()
            data = resp.json()
            try:
                raw = data["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError) as e:
                raise RuntimeError(f"Gemini response shape unexpected: {str(data)[:200]}") from e
            print(f"[GEMINI] {model}: {raw[:120]}")
            return _fill_defaults(json.loads(_clean_json(raw)))
        except Exception as e:
            last_err = e
            print(f"[GEMINI] {model} failed - {str(e)[:80]}")
    raise RuntimeError(f"Gemini failed: {last_err}")


# ─── OpenRouter ─────────────────────────────────────────────────────
def _call_openrouter(messages):
    if not OPENROUTER_KEY:
        raise RuntimeError("No OPENROUTER_API_KEY")

    last_err = None

    for model in OPENROUTER_MODELS:
        try:
            resp = _session.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_KEY}",
                    "HTTP-Referer": "https://mewar-erp.local",
                    "X-OpenRouter-Title": "Mewar ERP",
                    "Content-Type": "application/json",
                },
                data=json.dumps({
                    "model": model,
                    "messages": messages,
                    "temperature": 0.0,
                    "max_tokens": 400,
                }),
                timeout=20,
            )

            if resp.status_code == 429:
                raise RuntimeError("Rate limited")

            resp.raise_for_status()

            print(resp.json()["choices"][0]["message"]["content"])

            raw = resp.json()["choices"][0]["message"]["content"]
            return _fill_defaults(json.loads(_clean_json(raw)))

        except Exception as e:
            last_err = e
            print(f"[ENGINE] OpenRouter {model} failed: {str(e)[:80]}")

    raise RuntimeError(f"OpenRouter failed: {last_err}")


def _call_openrouter_wrapper(messages):
    for model in OPENROUTER_MODELS:
        try:
            return _call_openrouter(messages, model)
        except:
            continue
    raise RuntimeError("OpenRouter failed")

# ─── DeepSeek ────────────────────────────────────────────────────────
def _call_deepseek(messages):
    if not DEEPSEEK_KEY:
        raise RuntimeError("No DEEPSEEK_API_KEY")
    last_err = None
    for model in DEEPSEEK_MODELS:
        try:
            resp = _session.post(
                url=f"{DEEPSEEK_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_KEY}",
                    "Content-Type": "application/json",
                },
                data=json.dumps({
                    "model":           model,
                    "messages":        messages,
                    "temperature":     0.0,
                    "max_tokens":      400,
                    "response_format": {"type": "json_object"},
                }),
                timeout=25,
            )
            if resp.status_code == 429:
                raise RuntimeError("Rate limited")
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
            return _fill_defaults(json.loads(_clean_json(raw)))
        except Exception as e:
            last_err = e
            print(f"[ENGINE] DeepSeek {model} failed: {str(e)[:80]}")
    raise RuntimeError(f"DeepSeek failed: {last_err}")


# ─── Ollama ─────────────────────────────────────────────────────────
def _call_ollama(messages):
    resp = _session.post(
        f"{OLLAMA_BASE}/api/chat",
        json={
            "model": LOCAL_MODEL,
            "messages": messages,
            "stream": False,
        },
        timeout=8,
    )
    resp.raise_for_status()
    raw = resp.json()["message"]["content"]
    return _fill_defaults(json.loads(_clean_json(raw)))

# ─── MAIN ───────────────────────────────────────────────────────────
# ─── Lazy liveness + smart routing ──────────────────────────────────────
# Each real request that fails marks its provider "down" for _FAIL_BACKOFF seconds,
# so subsequent requests skip it instead of paying its timeout. Cheap, self-healing —
# no proactive token-burn. Use probe_providers() (below) for an active check.
_FAIL_BACKOFF = 120  # keep failed providers skipped for 2 min
_provider_status: dict = {}      # name -> {"healthy": bool, "until": ts, "err": str}
_status_lock = threading.Lock()


def _mark_down(name: str, err) -> None:
    with _status_lock:
        _provider_status[name] = {
            "healthy": False,
            "until":   time.time() + _FAIL_BACKOFF,
            "err":     str(err)[:120],
        }


def _mark_up(name: str) -> None:
    with _status_lock:
        _provider_status[name] = {"healthy": True, "until": 0, "err": None}


def _is_skippable(name: str) -> bool:
    """True if provider is in fail-backoff window — skip it from this request's chain."""
    with _status_lock:
        s = _provider_status.get(name)
    if not s or s["healthy"]:
        return False
    return time.time() < s["until"]


# Quality ordering — smart-first for complex JSON, fast-first for simple lookups
_CHAIN_COMPLEX = ["cerebras", "openrouter", "gemini", "groq", "ollama"]
_CHAIN_SIMPLE  = ["gemini",   "groq",  "cerebras", "openrouter", "ollama"]

# Substrings that indicate the query needs aggregation/comparison/negation reasoning
_COMPLEX_KEYWORDS = (
    "top ", "biggest", "smallest", "highest", "lowest", "max ", "min ",
    "compare", " vs ", " versus ", "rank", "ranked",
    "more than", "less than", "greater than", "over ", "above ", "below ",
    ">", "<",
    "group by", "monthly", "month wise", "weekly", "category wise",
    " not in ", "except", "excluding",
    "kis-kis", "sabse", "sabse bada", "sabse chhota",
)


def is_complex_query(text: str) -> bool:
    t = (text or "").lower()
    return any(kw in t for kw in _COMPLEX_KEYWORDS)


def _provider_configured(name: str) -> bool:
    return {
        "cerebras":   bool(CEREBRAS_KEY and Cerebras is not None),
        "gemini":     bool(GEMINI_KEY),
        "openrouter": bool(OPENROUTER_KEY),
        "groq":       bool(GROQ_KEYS),
        "deepseek":   bool(DEEPSEEK_KEY),
        "ollama":     True,  # always attempt as last resort
    }.get(name, False)


def _pick_chain(text: str) -> list:
    return ["deepseek", "gemini", "cerebras", "groq", "openrouter", "ollama"]


def probe_providers(force: bool = False) -> dict:
    """Active liveness probe — sends a 5-token call to each provider and reports
    {ok, ms, err}. Use this for the /health and /llm-status endpoints. Not called
    on the hot request path (we use lazy backoff there)."""
    msgs = [{"role": "user", "content": "ok"}]
    out  = {}
    probes = [
        ("deepseek",   lambda: _call_deepseek(msgs)),
        ("cerebras",   lambda: _call_cerebras(msgs)),
        ("gemini",     lambda: _call_gemini(msgs)),
        ("openrouter", lambda: _call_openrouter_wrapper(msgs)),
        ("groq",       lambda: _call_groq(msgs)),
        ("ollama",     lambda: _call_ollama(msgs)),
    ]
    for name, fn in probes:
        if not _provider_configured(name):
            out[name] = {"ok": False, "ms": 0, "err": "not configured"}
            continue
        t0 = time.time()
        try:
            fn()
            out[name] = {"ok": True, "ms": int((time.time() - t0) * 1000), "err": None}
            _mark_up(name)
        except Exception as e:
            out[name] = {"ok": False, "ms": int((time.time() - t0) * 1000), "err": str(e)[:120]}
            _mark_down(name, e)
    return out


_PROVIDER_CALL = {
    "deepseek":   _call_deepseek,
    "cerebras":   _call_cerebras,
    "gemini":     _call_gemini,
    "openrouter": _call_openrouter_wrapper,
    "groq":       _call_groq,
    "ollama":     _call_ollama,
}


_REQUEST_DEADLINE = 28  # seconds — hard ceiling per chatbot request

def ask_local_llm(user_text, history=None):
    # Cap input length to protect upstream APIs and prevent token-bomb abuse
    user_text = (user_text or "")[:_MAX_USER_TEXT].strip()
    if not user_text:
        raise RuntimeError("Empty query")

    messages  = _build_messages(user_text, history)
    chain     = _pick_chain(user_text)
    complex_  = is_complex_query(user_text)
    deadline  = time.time() + _REQUEST_DEADLINE
    print(f"[ROUTE] complex={complex_} chain={chain}")

    if not chain:
        chain = ["ollama"]

    result = None
    errors = []
    for provider in chain:
        if time.time() >= deadline:
            errors.append("deadline_exceeded")
            break
        if not _provider_configured(provider):
            continue
        if _is_skippable(provider):
            print(f"[ROUTE] skipping {provider} (in fail-backoff)")
            continue
        try:
            result = _PROVIDER_CALL[provider](messages)
            _mark_up(provider)
            print(f"[ROUTE] answered by {provider}")
            break
        except Exception as e:
            errors.append(f"{provider}:{str(e)[:80]}")
            _mark_down(provider, e)
            print(f"[ENGINE] {provider} failed - {str(e)[:120]}")

    if result is None:
        raise RuntimeError(f"All LLM providers failed: {' | '.join(errors)}")

    # Cap reasoning length so a runaway model can't bloat responses
    if isinstance(result.get("reasoning"), str) and len(result["reasoning"]) > _MAX_REASONING:
        result["reasoning"] = result["reasoning"][:_MAX_REASONING].rstrip() + "..."

    return result

# ─── Text-to-SQL (raw text, no JSON) ────────────────────────────────

def _raw_text_from_messages(provider: str, messages: list) -> str:
    """Call a provider and return plain text (no JSON parsing)."""
    if provider == "cerebras":
        client = _get_cerebras()
        for model in CEREBRAS_MODELS:
            try:
                resp = client.chat.completions.create(model=model, messages=messages,
                                                      max_tokens=500, temperature=0.0)
                return resp.choices[0].message.content
            except Exception as e:
                print(f"[SQL-GEN] cerebras {model}: {str(e)[:80]}")
        raise RuntimeError("Cerebras raw failed")

    if provider == "gemini":
        sys_text, contents = _to_gemini_contents(messages)
        body = {"contents": contents,
                "generationConfig": {"temperature": 0.0, "maxOutputTokens": 500}}
        if sys_text:
            body["systemInstruction"] = {"parts": [{"text": sys_text}]}
        for model in GEMINI_MODELS:
            try:
                resp = _session.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                    params={"key": GEMINI_KEY}, json=body, timeout=20)
                if resp.status_code == 429:
                    raise RuntimeError("Rate limited")
                resp.raise_for_status()
                return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            except Exception as e:
                print(f"[SQL-GEN] gemini {model}: {str(e)[:80]}")
        raise RuntimeError("Gemini raw failed")

    # OpenAI-compatible (deepseek / groq / openrouter)
    if provider == "deepseek":
        url, key = f"{DEEPSEEK_BASE}/chat/completions", DEEPSEEK_KEY
        models = DEEPSEEK_MODELS
    elif provider == "groq":
        url, key = f"{GROQ_BASE}/chat/completions", (GROQ_KEYS[0] if GROQ_KEYS else "")
        models = GROQ_MODELS
    elif provider == "openrouter":
        url, key = "https://openrouter.ai/api/v1/chat/completions", OPENROUTER_KEY
        models = OPENROUTER_MODELS
    else:
        raise RuntimeError(f"Unknown provider for raw text: {provider}")

    for model in models:
        try:
            resp = _session.post(url, headers={
                "Authorization": f"Bearer {key}", "Content-Type": "application/json",
                **({"HTTP-Referer": "https://mewar-erp.local"} if provider == "openrouter" else {}),
            }, json={"model": model, "messages": messages,
                     "temperature": 0.0, "max_tokens": 500}, timeout=20)
            if resp.status_code == 429:
                raise RuntimeError("Rate limited")
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[SQL-GEN] {provider}/{model}: {str(e)[:80]}")
    raise RuntimeError(f"{provider} raw failed")


_SQL_SYSTEM = """\
You are a MySQL query generator for an ERP database.
{schema}

Rules:
1. Write ONLY a MySQL SELECT — never INSERT/UPDATE/DELETE/DROP/TRUNCATE/ALTER/EXEC.
2. Use ONLY tables and columns listed above.
3. Always add LIMIT 25 unless the query is a COUNT/SUM/AVG aggregation.
4. Return ONLY the SQL query — no markdown, no explanation, no code fences.
5. If the question cannot be answered with a SELECT on this schema, reply with exactly: CANNOT_ANSWER
"""


def ask_for_sql(user_query: str, schema_text: str) -> str:
    """
    Ask an LLM to write a MySQL SELECT for user_query.
    Returns a SQL string, or raises RuntimeError.
    Never returns CANNOT_ANSWER — raises instead.
    """
    system = _SQL_SYSTEM.format(schema=schema_text)
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_query},
    ]
    chain = _pick_chain(user_query)
    errors = []
    for provider in chain:
        if not _provider_configured(provider) or _is_skippable(provider):
            continue
        try:
            raw = _raw_text_from_messages(provider, messages).strip()
            # Strip markdown code fences if the model wrapped it anyway
            raw = re.sub(r"^```(?:sql)?\s*", "", raw, flags=re.I)
            raw = re.sub(r"\s*```$", "", raw)
            raw = raw.strip()
            if raw.upper().startswith("CANNOT_ANSWER"):
                raise RuntimeError("LLM said CANNOT_ANSWER")
            if not raw.upper().startswith("SELECT"):
                raise RuntimeError(f"Non-SELECT returned: {raw[:60]}")
            _mark_up(provider)
            print(f"[SQL-GEN] answered by {provider}: {raw[:80]}")
            return raw
        except Exception as e:
            errors.append(f"{provider}:{str(e)[:60]}")
            _mark_down(provider, e)
    raise RuntimeError(f"ask_for_sql failed: {' | '.join(errors)}")


# ─── Health Check ───────────────────────────────────────────────────
def health_check():
    """Lightweight — does NOT probe providers. Reports config + lazy-tracked state."""
    with _status_lock:
        status_snap = dict(_provider_status)
    return {
        "primary": f"cerebras/{CEREBRAS_MODELS[0]}",
        "cerebras":   bool(CEREBRAS_KEY and Cerebras is not None),
        "gemini":     bool(GEMINI_KEY),
        "openrouter": bool(OPENROUTER_KEY),
        "groq":       bool(GROQ_KEYS),
        "ollama":     True,
        "complex_chain": _CHAIN_COMPLEX,
        "simple_chain":  _CHAIN_SIMPLE,
        "lazy_status":   status_snap,  # populated as real requests succeed/fail
        "cache_disabled": True,
    }
