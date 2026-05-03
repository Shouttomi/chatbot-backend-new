"""
v2_chatbot.py
Improved ERP chatbot using local qwen2.5:7b (Ollama) instead of Groq API.

Key improvements over chatbot.py:
- 100% offline: no external API calls, no rate limits
- Cleaner FAISS state management via dataclass
- Confidence-based entity matching (dynamic thresholds per category)
- Smarter intent routing with fewer hard-coded overrides
- Better multi-intent result ordering
- Status/health endpoint (/v2-chatbot/status)
- Proper result pagination — no arbitrary `limit+2` cap
- Better Hinglish + English language detection
"""

import re
import difflib
import json
import time
import calendar
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field

# FAISS disabled for Python 3.13 compatibility
# import faiss
# import numpy as np
# FAISS + sentence-transformers disabled for Python 3.13 compatibility
# from sentence_transformers import SentenceTransformer
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.db.database import get_db, get_local_db
from app.schemas.chat import ChatRequest
from app.services.v2_ollama_engine import ask_local_llm, health_check, probe_providers, is_complex_query
from app.services.entity_resolver import resolve as resolve_entity, resolve_with_confidence, invalidate as invalidate_resolver
from app.services.complex_query import handle_complex


# ─── Confirm-before-act helpers ──────────────────────────────────────────────
# Map a user reply to one of the candidate names from a previous confirmation prompt.
# Returns the chosen canonical name, or None if the user clearly didn't pick one
# (in which case the caller should treat the input as a fresh query).
_AFFIRM = {"👍", "yes", "haan", "haa", "ji", "ji haan", "y", "yep", "yeah", "sahi", "correct", "right", "ok", "okay", "1"}
_DENY   = {"👎", "no", "nahi", "na", "n", "nope", "galat", "wrong", "incorrect"}


def _parse_choice(reply: str, candidates: list) -> Optional[str]:
    """
    candidates: list of canonical names (strings).
    Accepts:
      "👍" / "yes" / "haan"  → first candidate
      "👎" / "no"  / "nahi"  → None (user rejected; treat as new query)
      "1"/"2"/"3"            → that candidate (1-indexed)
      direct typing of one of the candidate names → that one
    """
    if not reply or not candidates:
        return None
    r = reply.strip().lower()
    if r in _AFFIRM:
        return candidates[0]
    if r in _DENY:
        return None
    # fuzzy deny — "nahi bhai", "nahi koi nahi", "galat hai bhai" etc.
    if any(dw in r.split() for dw in _DENY):
        return None
    # numeric pick
    if r.isdigit():
        idx = int(r) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx]
        return None
    # direct name retype (case-insensitive substring match)
    for c in candidates:
        if r == c.lower() or r in c.lower():
            return c
    return None


def _last_pending_resolution(history: list) -> Optional[Dict[str, Any]]:
    """Find pending_resolution attached to the most recent assistant turn."""
    for msg in reversed(history or []):
        if msg.get("role") == "assistant":
            pr = msg.get("pending_resolution")
            return pr if isinstance(pr, dict) else None
    return None


def _last_context_entity(history: list) -> Optional[Dict[str, Any]]:
    """Return context_entity from the most recent assistant turn (if any)."""
    for msg in reversed(history or []):
        if msg.get("role") == "assistant":
            ce = msg.get("context_entity")
            return ce if isinstance(ce, dict) else None
    return None


def _in_clause(ids: list) -> tuple[str, dict]:
    """Build a safe IN-clause placeholder string + params dict for a list of ints."""
    ph = ",".join(f":_id{i}" for i in range(len(ids)))
    pm = {f"_id{i}": int(v) for i, v in enumerate(ids)}
    return ph, pm


def _context_followup(ce: Dict, intents: list, low_q: str,
                      db, limit: int) -> Optional[list]:
    """
    Given a previous context_entity, answer follow-up intents via targeted
    WHERE id IN (...) queries instead of fuzzy LIKE matching.
    Returns a list of result dicts, or None to fall through to normal flow.
    """
    if not ce:
        return None
    ce_type = ce.get("type", "")
    results = []

    # ── Supplier context ───────────────────────────────────────────────────────
    if ce_type == "supplier":
        sup_id   = ce.get("id")
        sup_name = ce.get("name", "")

        wants_po = ("po_search" in intents or
                    any(w in low_q for w in ["order","po","purchase","balance","paisa","payment","baaki"]))
        wants_inv = ("search" in intents or
                     any(w in low_q for w in ["item","items","maal","inventory","stock","material",
                                               "hai kya","paas","available","rakha","rakhte","milega"]))
        wants_sup = "supplier_search" in intents
        # Generic product-name follow-up: "inke paas bearing hai kya" — no explicit inv word
        # but there's a meaningful non-stop keyword → treat as inventory check
        if not wants_po and not wants_inv and not wants_sup:
            _stop_words = {"hai","kya","hain","nahi","bhi","aur","kya","jo","ke","ki","ka","se",
                           "mein","par","ya","agar","to","toh","wo","woh","yeh","ye","inke","unke",
                           "paas","iske","uske","batao","dikhao"}
            _keywords = [w for w in low_q.split() if w not in _stop_words and len(w) > 2]
            if _keywords:
                wants_inv = True

        if wants_po:
            po_ids = ce.get("po_ids") or []
            if po_ids:
                ph, pm = _in_clause(po_ids)
                pm["l"] = limit
                rows = db.execute(text(
                    f"SELECT p.*,s.supplier_name FROM purchase_orders p "
                    f"LEFT JOIN suppliers s ON p.supplier_id=s.id "
                    f"WHERE p.id IN ({ph}) ORDER BY p.po_date DESC LIMIT :l"
                ), pm).fetchall()
            else:
                rows = db.execute(text(
                    "SELECT p.*,s.supplier_name FROM purchase_orders p "
                    "LEFT JOIN suppliers s ON p.supplier_id=s.id "
                    "WHERE p.supplier_id=:sid ORDER BY p.po_date DESC LIMIT :l"
                ), {"sid": sup_id, "l": limit}).fetchall()
            if rows:
                bal = sum(float(r.balance_amount or 0) for r in rows)
                results.append({"type":"chat","message":
                    f"**{sup_name}** ke **{len(rows)} orders** mile. "
                    f"Pending balance: **₹{bal:,.2f}**"})
                for r in rows:
                    results.append({
                        "type":"po","po_id":int(r.id),"po_no":str(r.po_number),
                        "supplier":str(r.supplier_name or ""),
                        "date":str(r.po_date),
                        "total":float(r.total_amount or 0),
                        "advance":float(r.advance_amount or 0),
                        "balance":float(r.balance_amount or 0),
                        "status":str(r.status or "").capitalize(),
                    })
            else:
                results.append({"type":"chat","message":
                    f"**{sup_name}** ke koi purchase orders nahi mile database mein. 🧐"})
            return results

        if wants_inv:
            inv_ids = ce.get("inventory_ids") or []
            if not inv_ids and sup_id:
                # Fallback: find inventories this supplier provided via POs
                po_inv_rows = db.execute(text(
                    "SELECT DISTINCT poi.inventory_id FROM purchase_order_items poi "
                    "JOIN purchase_orders p ON poi.purchase_order_id=p.id "
                    "WHERE p.supplier_id=:sid AND poi.inventory_id IS NOT NULL LIMIT 200"
                ), {"sid": sup_id}).fetchall()
                inv_ids = [r.inventory_id for r in po_inv_rows]

            # Extract keyword from query regardless of inv_ids (used for filtering + messages)
            _stop_kw = {"hai","kya","kya hai","inke","unke","paas","iske","uske","aur","bhi",
                        "stock","item","items","maal","inventory","material","yahan","yaha",
                        "batao","dikhao","show","list","available","kitna","kitne","pas"}
            kw = " ".join(w for w in low_q.split() if w not in _stop_kw and len(w) > 2)

            if inv_ids:
                ph, pm = _in_clause(inv_ids)
                base_sql = (
                    f"SELECT i.id,i.name,i.unit,"
                    f"COALESCE(SUM(CASE WHEN LOWER(t.txn_type)='in' THEN t.quantity ELSE -t.quantity END),0) AS stock "
                    f"FROM inventories i LEFT JOIN stock_transactions t ON i.id=t.inventory_id "
                    f"WHERE i.id IN ({ph})"
                )
                if kw:
                    kw_parts = [w for w in kw.split() if len(w) > 2]
                    for i, part in enumerate(kw_parts[:3]):
                        pm[f"kw{i}"] = f"%{part}%"
                        base_sql += f" AND i.name LIKE :kw{i}"
                base_sql += " GROUP BY i.id,i.name,i.unit ORDER BY stock DESC LIMIT :l"
                pm["l"] = limit + 10

                rows = db.execute(text(base_sql), pm).fetchall()
                if rows:
                    kw_label = f' "{kw}"' if kw else ""
                    results.append({"type":"chat","message":
                        f"**{sup_name}** ke paas{kw_label} ke **{len(rows)} items** hain:"})
                    results.append({"type":"dropdown","message":"Select item for details:",
                        "items":[{"id":r.id,"name":f"{r.name} (stock: {float(r.stock):.0f} {r.unit or 'units'})"} for r in rows]})
                    return results
                else:
                    kw_label = f' "{kw}"' if kw else ""
                    results.append({"type":"chat","message":
                        f"**{sup_name}** ke paas{kw_label} ka koi stock nahi mila database mein. 🧐"})
                    return results
            else:
                # No inventory data at all for this supplier
                results.append({"type":"chat","message":
                    f"**{sup_name}** ka koi inventory/stock record nahi mila. "
                    f"Is supplier se koi purchase order ya stock transaction nahi hua hai. 🧐"})
                return results

        if wants_sup:
            row = db.execute(text("SELECT * FROM suppliers WHERE id=:id"), {"id": sup_id}).fetchone()
            if row:
                results.append({"type":"result","supplier":{
                    "id":row.id,"name":row.supplier_name,
                    "code":str(getattr(row,"supplier_code","N/A") or "N/A"),
                    "mobile":getattr(row,"mobile","N/A") or "N/A",
                    "city":getattr(row,"city","N/A") or "N/A",
                    "email":getattr(row,"email","N/A") or "N/A",
                    "gstin":getattr(row,"gstin","N/A") or "N/A",
                }})
                return results

    # ── Purchase Order context ─────────────────────────────────────────────────
    elif ce_type == "purchase_order":
        po_id    = ce.get("id")
        po_no    = ce.get("po_no", "this PO")
        sup_id   = ce.get("supplier_id")
        sup_name = ce.get("supplier_name", "")

        wants_items = (
            "search" in intents or
            any(w in low_q for w in ["item","items","kya","maal","material","inventory","content","andar"]))
        wants_sup   = ("supplier_search" in intents or
                       any(w in low_q for w in ["supplier","party","vendor","contact","mobile","detail"]))

        if wants_items:
            rows = db.execute(text(
                "SELECT poi.ordered_qty, poi.unit_price, poi.line_total, i.name AS inv_name, i.unit "
                "FROM purchase_order_items poi "
                "LEFT JOIN inventories i ON poi.inventory_id=i.id "
                "WHERE poi.purchase_order_id=:pid"
            ), {"pid": po_id}).fetchall()
            if rows:
                lines = "\n".join(
                    f"• **{r.inv_name or 'Unknown'}** — "
                    f"qty: {float(r.ordered_qty or 0):.0f} {r.unit or ''}, "
                    f"rate: ₹{float(r.unit_price or 0):,.2f}, "
                    f"total: ₹{float(r.line_total or 0):,.2f}"
                    for r in rows)
                results.append({"type":"chat","message":
                    f"**{po_no}** mein **{len(rows)} items** hain:\n\n{lines}"})
                return results

        if wants_sup and sup_id:
            row = db.execute(text("SELECT * FROM suppliers WHERE id=:id"), {"id": sup_id}).fetchone()
            if row:
                results.append({"type":"chat","message":f"**{po_no}** ka supplier:"})
                results.append({"type":"result","supplier":{
                    "id":row.id,"name":row.supplier_name,
                    "code":str(getattr(row,"supplier_code","N/A") or "N/A"),
                    "mobile":getattr(row,"mobile","N/A") or "N/A",
                    "city":getattr(row,"city","N/A") or "N/A",
                    "email":getattr(row,"email","N/A") or "N/A",
                    "gstin":getattr(row,"gstin","N/A") or "N/A",
                }})
                return results

    # ── Inventory context ──────────────────────────────────────────────────────
    elif ce_type == "inventory":
        inv_id   = ce.get("id")
        inv_name = ce.get("name", "this item")

        wants_po  = ("po_search" in intents or
                     any(w in low_q for w in ["order","po","purchase","aaya","kahan se","laya"]))
        wants_sup = ("supplier_search" in intents or
                     any(w in low_q for w in ["supplier","vendor","party","kahan se","kiska"]))

        if wants_po or wants_sup:
            po_rows = db.execute(text(
                "SELECT DISTINCT p.id,p.po_number,p.po_date,p.total_amount,"
                "p.balance_amount,p.status,p.supplier_id,s.supplier_name "
                "FROM purchase_order_items poi "
                "JOIN purchase_orders p ON poi.purchase_order_id=p.id "
                "LEFT JOIN suppliers s ON p.supplier_id=s.id "
                "WHERE poi.inventory_id=:iid ORDER BY p.po_date DESC LIMIT 20"
            ), {"iid": inv_id}).fetchall()

            if po_rows and wants_sup:
                seen = set()
                sups = [r for r in po_rows if r.supplier_id not in seen and not seen.add(r.supplier_id)]
                if len(sups) == 1:
                    results.append({"type":"chat","message":
                        f"**{inv_name}** sirf **{sups[0].supplier_name}** se aaya hai."})
                    sup = db.execute(text("SELECT * FROM suppliers WHERE id=:id"),
                                     {"id": sups[0].supplier_id}).fetchone()
                    if sup:
                        results.append({"type":"result","supplier":{
                            "id":sup.id,"name":sup.supplier_name,
                            "code":str(getattr(sup,"supplier_code","N/A") or "N/A"),
                            "mobile":getattr(sup,"mobile","N/A") or "N/A",
                            "city":getattr(sup,"city","N/A") or "N/A",
                            "email":getattr(sup,"email","N/A") or "N/A",
                            "gstin":getattr(sup,"gstin","N/A") or "N/A",
                        }})
                else:
                    results.append({"type":"chat","message":
                        f"**{inv_name}** in **{len(sups)} suppliers** se aaya:"})
                    results.append({"type":"dropdown","message":"Select supplier:",
                        "items":[{"id":str(r.supplier_id),"name":r.supplier_name} for r in sups]})
                return results

            if po_rows and wants_po:
                results.append({"type":"chat","message":
                    f"**{inv_name}** in **{len(po_rows)} POs** mein aaya:"})
                for r in po_rows:
                    results.append({
                        "type":"po","po_id":int(r.id),"po_no":str(r.po_number),
                        "supplier":str(r.supplier_name or "N/A"),
                        "date":str(r.po_date),
                        "total":float(r.total_amount or 0),
                        "advance":0.0,
                        "balance":float(r.balance_amount or 0),
                        "status":str(r.status or "").capitalize(),
                    })
                return results

    # ── Project context ────────────────────────────────────────────────────────
    elif ce_type == "project":
        proj_id   = ce.get("id")
        proj_name = ce.get("name", "this project")

        wants_items = ("search" in intents or
                       any(w in low_q for w in ["item","maal","material","inventory","kya","bom","lagega","chahiye"]))
        wants_products = any(w in low_q for w in ["product","finished","machine","parts"])

        if wants_items:
            rows = db.execute(text(
                "SELECT i.id,i.name,i.unit,pi.quantity "
                "FROM project_item pi JOIN inventories i ON pi.inventory_id=i.id "
                "WHERE pi.project_id=:pid"
            ), {"pid": proj_id}).fetchall()
            if rows:
                lines = "\n".join(
                    f"• **{r.name}** — {float(r.quantity or 0):.0f} {r.unit or 'units'}"
                    for r in rows)
                results.append({"type":"chat","message":
                    f"**{proj_name}** mein **{len(rows)} raw materials** lage hain:\n\n{lines}"})
                return results

        if wants_products:
            rows = db.execute(text(
                "SELECT p.id,p.name,pp.quantity "
                "FROM project_products pp JOIN products p ON pp.product_id=p.id "
                "WHERE pp.project_id=:pid"
            ), {"pid": proj_id}).fetchall()
            if rows:
                lines = "\n".join(f"• **{r.name}** — qty: {float(r.quantity or 0):.0f}" for r in rows)
                results.append({"type":"chat","message":
                    f"**{proj_name}** mein **{len(rows)} finished products** use ho rahe hain:\n\n{lines}"})
                return results

    return None

router = APIRouter(prefix="/v2-chatbot", tags=["V2 Chatbot"])

# ─── FAISS State (Disabled for Python 3.13) ──────────────────────────────────
# FAISS semantic search disabled due to Python 3.13 / sentence-transformers incompatibility
# The LLM-based routing in ask_local_llm() provides sufficient intent extraction

def load_faiss_v2(db: Session):
    """Stub — FAISS disabled in this version."""
    print("[V2-CHATBOT] FAISS disabled (Python 3.13 compatibility). Using LLM-only routing.")
    pass

def faiss_match(query: str, category: str) -> Optional[str]:
    """Stub — returns None (FAISS disabled)."""
    return None


# ─── Noise Cleaning ───────────────────────────────────────────────────────────
_FILLERS = [
    "batao","bata","dikhao","dikha","mujhe","bhai","hai","ho","hoga","hogi","hota","hoti",
    "kon","kya","kab","kahan","kaun","kaise","kaisa","kaisi","check","chahiye","dedo","de",
    "do","mil","milega","milegi","mile","jayegi","jayega","info","information","jankari",
    "please","pls","plz","sakta","sakti","sakte","thodi","thoda",
    "show","tell","give","get","need","want","any","some","this","that",
    "what","is","are","was","were","does","do","has","have",
    "can","could","would","will","which","when","why","find","fetch",
    "all","saare","sabhi","list","sab","poora","pura",
    "mein","main","ka","ki","ke","ko","se","par","pe","aur","or",
    "for","of","the","a","an","about","me","my",
    "uska","uski","uske","iska","iski","iske","inka","inke","inki",
    "diya","gaya","gaye","gayi","liya","liye","hua","hue","hui",
    "wala","wali","wale",
]

_CTX_NOISE: Dict[str, List[str]] = {
    "po":       ["pending","order","orders","po","purchase","draft","completed","latest",
                 "last","paisa","baki","baaki","balance","rokra","transit","dispatch",
                 "delivery","hisab","udhar","payment","profile","details","info","show","their","and",
                 "advance","prepaid","diya","gaya","gaye","kisi","kuch",
                 "kitna","kitni","kitne","total","bhi","bhi","iska","uska","sabka"],
    "supplier": ["supplier","vendor","party","contact","mobile","phone","number","gst",
                 "gstin","email","address","city","details","profile","account","bank",
                 "ifsc","pan","po","order","orders","purchase",
                 "bhi","bola","chahiye","hai","hain","ka","ki","ke"],
    "project":  ["project","site","running","urgent","completed","refurbish","refurbished",
                 "machine","high","low","priority","status","stage"],
    "inventory":["stock","maal","item","inventory","quantity","kitna","qty","available","piece","pieces"],
}


def clean_noise(text: str, ctx: str = "") -> str:
    if not text:
        return ""
    noise = list(_FILLERS) + _CTX_NOISE.get(ctx, [])
    out   = text.lower()
    for w in noise:
        out = re.sub(rf"\b{re.escape(w)}\b", " ", out)
    out = re.sub(r"[^\w\s/\-]", " ", out)
    return re.sub(r"\s+", " ", out).strip()


# ─── Follow-up / Sticky Context ───────────────────────────────────────────────
_FOLLOWUP = [
    "uska","uski","uske","iska","iski","inka","inke","inki",
    "bhi","aur","yeh","ye","usi","same","same wala",
    "usme","usme","usmein","isme","ismein","inme","inmein",
    "unka","unki","unke","unhe","unhen",
    "yahi","wahi","usi","ussi","wohi",
]


def _extract_sticky(history: list, low_q: str) -> str:
    if not history:
        return ""
    is_fu = any(re.search(rf"\b{w}\b", low_q) for w in _FOLLOWUP) or len(low_q.split()) <= 2
    if not is_fu:
        return ""
    _skip = {"payment alert", "financial summary", "top supplier", "note"}
    # 1st pass: bold markers in assistant messages
    for msg in reversed(history[-6:]):
        if str(msg.get("role", "")).lower() not in ("assistant", "ai", "bot"):
            continue
        content = str(msg.get("content") or msg.get("message") or "")
        m = re.search(r"\*\*([^*]+?)\*\*", content)
        if m:
            candidate = m.group(1).strip()
            if candidate.lower() not in _skip:
                return candidate
    # 2nd pass: extract from last user message (for plain-text history)
    for msg in reversed(history[-6:]):
        if str(msg.get("role", "")).lower() not in ("user",):
            continue
        user_q = str(msg.get("content") or "").strip()
        cleaned = clean_noise(user_q, "")
        cleaned = re.sub(r"\b(dikhao|batao|details|profile|list|latest|last|stock|orders|kitna|kitni|kitne|hai|hain)\b", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if len(cleaned) > 2:
            return cleaned
    return ""


def _is_followup_query(low_q: str) -> bool:
    return any(re.search(rf"\b{w}\b", low_q) for w in _FOLLOWUP)


# ─── Smart Entity Resolver ────────────────────────────────────────────────────
def _resolve_entity(original_target: str, intents: list, low_q: str) -> list:
    """
    If LLM defaulted to 'search' (inventory) but the target looks like a supplier/project,
    reroute. Inventory always wins on conflict.
    (FAISS disabled — relies on difflib in inventory loop for typo correction.)
    """
    # Without FAISS, we skip smart entity resolution here.
    # Difflib in the inventory search loop handles typo correction.
    return intents


# ─── Logging ─────────────────────────────────────────────────────────────────
def _log(query: str, intent: Any, result: Any):
    entry = {
        "ts":     datetime.now().isoformat(),
        "query":  query,
        "intent": intent,
        "fail":   any(w in str(result).lower() for w in ["nahi mila","not found","error","samajh nahi"]) or not result,
    }
    try:
        with open("logs.json", "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def _truncate_for_log(value: Any, limit: int = 4000) -> Any:
    text_value = json.dumps(value, default=str, ensure_ascii=False)
    if len(text_value) <= limit:
        return value
    return {"truncated": True, "chars": len(text_value), "preview": text_value[:limit]}


def _is_zero_result(response_data: Dict[str, Any]) -> bool:
    """A response is 'zero-result' if it has no concrete data — only chat/clarification messages.
    Mining these queries weekly is the highest-signal correctness feedback loop."""
    results = (response_data or {}).get("results") or []
    if not results:
        return True
    # Any result with a real entity payload counts as a hit.
    data_types = {"result", "supplier", "project", "po", "purchase_order", "inventory", "po_summary", "count_answer", "supplier_list", "po_list", "project_list"}
    for r in results:
        if isinstance(r, dict) and r.get("type") in data_types:
            return False
        # Also accept chat results that explicitly set db_checked=True with a hit
        if isinstance(r, dict) and r.get("db_checked") and r.get("type") != "chat_no_data":
            return False
    return True


def _chatbot_reqres_log(request_data: Dict[str, Any], response_data: Dict[str, Any], elapsed_ms: int, error: str | None = None, request_id: str | None = None):
    entry = {
        "ts": datetime.now().isoformat(),
        "request_id": request_id,
        "route": "/v2-chatbot",
        "elapsed_ms": elapsed_ms,
        "zero_result": _is_zero_result(response_data) if not error else None,
        "request": _truncate_for_log(request_data),
        "response": _truncate_for_log(response_data),
    }
    if error:
        entry["error"] = error
    try:
        with open("chatbot_reqres.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


def _request_to_log_dict(request: ChatRequest) -> Dict[str, Any]:
    return {
        "query": request.query,
        "history": request.history or [],
        "role": request.role,
        "ui_filters": request.ui_filters or {},
    }


def _v2_chatbot_impl(request: ChatRequest, db: Session):
    load_faiss_v2(db)

    raw_q    = request.query.strip()
    low_q    = raw_q.lower()
    history  = getattr(request, "history", []) or []
    role     = str(getattr(request, "role", None) or "").lower().strip()

    # ── Empty guard ──────────────────────────────────────────────────────────
    if not raw_q:
        return {"results": [{"type": "chat", "message": "AI brain abhi busy hai, thodi der mein try karein. 🙏"}]}

    # ── Confirm-before-act resume ─────────────────────────────────────────────
    # Must run BEFORE any garbage/greeting guards so that "👍", "1", "haan" etc.
    # are handled as confirmation tokens, not rejected as non-ERP input.
    # If the previous assistant turn asked for confirmation (👍/👎/1-2-3) and the
    # user's current reply is a clear pick, replay the *original* query with the
    # chosen entity. Must run BEFORE the digit-as-inventory-id shortcut etc.
    _pending = _last_pending_resolution(history)
    if _pending and raw_q:
        _picked = _parse_choice(raw_q, _pending.get("candidates") or [])
        if _picked:
            print(f"[V2-CONFIRM] resuming with picked='{_picked}'")
            # Re-issue the original query as if the LLM had returned the chosen target
            replayed = ChatRequest(
                query     = _pending.get("original_query") or raw_q,
                history   = history,
                role      = role,
                ui_filters= getattr(request, "ui_filters", {}) or {},
            )
            # We attach the chosen target via a thread-local mechanism: stash it
            # in request.ui_filters so the resolver step picks it up unconditionally.
            replayed.ui_filters = dict(replayed.ui_filters or {})
            replayed.ui_filters["__resolved_entity__"] = _picked
            replayed.ui_filters["__resolved_category__"] = _pending.get("category")
            return _v2_chatbot_legacy_flow(
                replayed, db,
                replayed.query.strip(), replayed.query.strip().lower(),
                history, role,
            )

    # ── Garbage / injection guards (after confirm-resume so 👍/👎 pass through) ─
    _alpha_ratio = sum(c.isalpha() for c in raw_q) / max(len(raw_q), 1)
    _is_pure_special = _alpha_ratio < 0.2 and not any(c.isdigit() for c in raw_q)
    # Always allow confirm/deny tokens — they may arrive outside a pending_resolution context
    _is_confirm_token = (raw_q.strip().lower() in (_AFFIRM | _DENY)
                         or (raw_q.strip().isdigit() and len(raw_q.strip()) == 1))
    if _is_pure_special and len(raw_q) < 20 and not _is_confirm_token:
        return {"results": [{"type": "chat", "message": "Bhai, kuch samajh aane wala poochho. 😅"}]}

    # ── SQL injection guard ── catches SELECT/INSERT/UPDATE/DELETE/DROP/UNION etc.
    # regardless of spacing, casing, comment padding, or semicolon chaining.
    _SQL_INJECT_RE = re.compile(
        r"\b(select|insert|update|delete|drop|truncate|alter|create|replace"
        r"|exec|execute|grant|revoke|union|load_file|outfile|information_schema)\b"
        r"|--|/\*|;\s*(drop|delete|insert|update|alter)",
        re.I,
    )
    if _SQL_INJECT_RE.search(raw_q):
        return {"results": [{"type": "chat", "message": "Ye mujhse nahi hoga bhai. 😅 Koi ERP query poochho."}]}


    if raw_q.isdigit() and len(raw_q) >= 8:
        return {"results": [{"type": "chat", "message": "Koi item ID, supplier naam ya order number batao. 🙂"}]}

    # Fast-path greeting bypass — no LLM needed
    if _GREETING_RE.match(raw_q.strip()):
        return {"results": [{"type": "chat", "message": (
            "Haan bhai! Kya haal hai? 😊\n\nAap mujhse yeh pooch sakte ho:\n\n"
            "📦 **Inventory** — item ka stock (e.g. 'bearing stock kitna hai')\n"
            "🏭 **Supplier** — party ki details (e.g. 'Adinath details')\n"
            "🧾 **PO** — purchase orders (e.g. 'pending orders dikhao')\n"
            "📁 **Projects** — site status (e.g. 'chalu projects')\n\n"
            "Bas likhein aur main check karta hoon!"
        )}]}

    # If query looks like an aggregation/comparison/negation, skip the legacy
    # SQL-first shortcut — those need the new complex_query handler via the LLM.
    _COMPLEX_KW = (
        "top ", "biggest", "smallest", "highest", "lowest", "max ", "min ",
        "compare", " vs ", " versus ", "rank", "ranked",
        "more than", "less than", "greater than", "over ", "under ", "above ", "below ",
        ">", "<",
        "group by", "month wise", "monthly", "weekly", "category wise",
        " not in ", "except", "excluding",
        "kis-kis", "sabse", "sabse bada", "sabse chhota",
    )
    looks_complex = any(kw in low_q for kw in _COMPLEX_KW)

    try:
     #   if not looks_complex:
      #      sql_answer = _try_sql_first_answer(raw_q, low_q, db, history)
       #     if sql_answer:
        #        _log(raw_q, "sql_first", sql_answer)
         #       return sql_answer
       # else:
        #    sql_answer = None
        pass
    except Exception as e:
        print(f"[V2-SQL-FIRST] skipped -> {e}")

    return _v2_chatbot_legacy_flow(request, db, raw_q, low_q, history, role)


# ─── Aggregation keyword sets ─────────────────────────────────────────────────
_AGG_LOW_STOCK  = ["sabse kam stock","minimum stock","lowest stock","kam stock","least stock","stock khatam"]
_AGG_HIGH_STOCK = ["sabse zyada stock","maximum stock","highest stock","sabse jyada stock","most stock"]
_AGG_PEND_BAL   = ["total pending balance","total balance","sabka balance","sabka pending",
                    "total pending","sab ka balance","overall balance","total baaki"]
_AGG_COUNT      = ["kitne","how many","count","total suppliers","total items",
                    "total projects","total inventory","kitni",
                    "number of","no of","no. of"]

_ERP_WORDS = {
    "inventory", "stock", "maal", "item", "items", "quantity", "qty", "bearing", "belt",
    "oil", "seal", "bolt", "supplier", "vendor", "party", "gst", "gstin", "pan", "mobile",
    "phone", "contact", "email", "address", "bank", "ifsc", "po", "order", "orders",
    "purchase", "balance", "advance", "pending", "draft", "project", "projects", "site",
    "budget", "priority", "status", "request", "pr",
    # additional inventory item keywords
    "chain", "sprocket", "grease", "motor", "pump", "valve", "cylinder", "filter",
    "cable", "wire", "paint", "fastener", "washer", "gasket", "coupling", "pulley",
    "gear", "shaft", "bushing", "liner", "plate", "pipe", "hose", "fitting",
}

_GREETING_RE = re.compile(
    r'^(hi+|hello|hey|hii+|namaste|namaskar|kya haal|kaise ho|sab theek|'
    r'good morning|good evening|good afternoon|how are you|kya chal raha|'
    r'help|kya kar sakte|what can you do|assalamu|salam)\b',
    re.I,
)

_CHAT_ONLY_WORDS = {
    "hello", "hi", "hey", "weather", "mausam", "joke", "song", "news", "time",
    "asdfghjkl", "qwerty",
}


def _erp_help_response():
    return {
        "results": [{
            "type": "chat",
            "message": (
                "Main ERP database se inventory, suppliers, purchase orders, projects "
                "aur purchase requests check kar sakta hoon. Kripya inme se kisi ka "
                "naam, number, status ya detail poochhiye."
            ),
        }]
    }


def _sql_ids(ids):
    return ",".join(f":id{i}" for i, _ in enumerate(ids))


def _sql_id_params(ids):
    return {f"id{i}": int(v) for i, v in enumerate(ids)}


def _stock_for_ids(db: Session, ids: list[int]) -> float:
    if not ids:
        return 0.0
    row = db.execute(text(
        "SELECT COALESCE(SUM(CASE WHEN LOWER(txn_type)='in' THEN quantity ELSE -quantity END),0) "
        f"FROM stock_transactions WHERE inventory_id IN ({_sql_ids(ids)})"
    ), _sql_id_params(ids)).scalar()
    return float(row or 0)


def _stock_for_id(db: Session, inv_id: int) -> float:
    return _stock_for_ids(db, [inv_id])


def _clean_entity_query(raw_q: str, ctx: str) -> str:
    cleaned = clean_noise(raw_q, ctx)
    cleaned = re.sub(r"\b(total|latest|last|all|list|number|no|details?|profile|dikhao|batao)\b", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _inventory_targets(raw_q: str) -> list[str]:
    month_words = "january february march april may june july august september october november december jan feb mar apr jun jul aug sep sept oct nov dec".split()
    raw_parts = re.split(r"\s*(?:,|\baur\b|\band\b|\bor\b|\+)\s*", raw_q.lower())
    targets = []
    for raw_part in raw_parts:
        part = clean_noise(raw_part, "inventory")
        part = re.sub(r"\b(dono|sab|saare|sabhi|all|total|kitna|kitni|aaya|tha|month|week)\b", " ", part)
        for month in month_words:
            part = re.sub(rf"\b{month}\b", " ", part)
        part = re.sub(r"\s+", " ", part).strip()
        if len(part) > 1 and part not in targets:
            targets.append(part)
    if targets:
        return targets
    cleaned = clean_noise(raw_q, "inventory")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return [cleaned] if cleaned else []


def _find_inventory_items(db: Session, target: str, limit: int = 30):
    target = re.sub(r"\s+", " ", target.lower()).strip()
    if not target:
        return []

    if target.isdigit() and len(target) < 8:
        row = db.execute(text(
            "SELECT id,name,model,type,classification,placement,unit FROM inventories WHERE id=:id LIMIT 1"
        ), {"id": int(target)}).fetchall()
        if row:
            return row

    phrase = f"%{target}%"
    rows = db.execute(text(
        "SELECT id,name,model,type,classification,placement,unit FROM inventories "
        "WHERE LOWER(CONCAT(name,' ',COALESCE(model,''))) LIKE :q "
        "ORDER BY name LIMIT :l"
    ), {"q": phrase, "l": limit}).fetchall()
    if rows:
        return rows

    words = [w for w in target.split() if len(w) > 1]
    if not words:
        return []
    conds = " AND ".join(
        f"LOWER(CONCAT(name,' ',COALESCE(model,''))) LIKE :w{i}" for i in range(len(words))
    )
    params = {f"w{i}": f"%{w}%" for i, w in enumerate(words)}
    params["l"] = limit
    return db.execute(text(
        "SELECT id,name,model,type,classification,placement,unit FROM inventories "
        f"WHERE {conds} ORDER BY name LIMIT :l"
    ), params).fetchall()


_MONTH_MAP = {
    "january": 1, "jan": 1, "february": 2, "feb": 2,
    "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
    # Hindi month names
    "janvari": 1, "janvary": 1, "farvari": 2, "march": 3,
    "april": 4, "may": 5, "june": 6, "july": 7,
    "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}


def _date_range_from_query(low_q: str):
    today = date.today()

    # aaj / today
    if re.search(r"\baaj\b", low_q) or "today" in low_q:
        return str(today), str(today)

    # kal / yesterday
    if re.search(r"\bkal\b", low_q) or "yesterday" in low_q:
        yday = today - timedelta(days=1)
        return str(yday), str(yday)

    # last N months / pichle N mahine
    m = re.search(r"(?:last|pichle?|pichhle?)\s+(\d+)\s+(?:month|mahine?|months?)", low_q)
    if m:
        n = int(m.group(1))
        start = today - timedelta(days=30 * n)
        return str(start), str(today)

    # this year / is saal
    if "is saal" in low_q or "this year" in low_q or "current year" in low_q:
        return str(today.replace(month=1, day=1)), str(today.replace(month=12, day=31))

    # last year / pichle saal
    if "pichle saal" in low_q or "pichhle saal" in low_q or "last year" in low_q:
        ly = today.year - 1
        return str(date(ly, 1, 1)), str(date(ly, 12, 31))

    # last month / pichle mahine
    if "last month" in low_q or "pichle mahine" in low_q or "pichhle mahine" in low_q:
        first_this = today.replace(day=1)
        last_prev  = first_this - timedelta(days=1)
        return str(last_prev.replace(day=1)), str(last_prev)

    # this month / is mahine
    if "this month" in low_q or "is mahine" in low_q:
        first = today.replace(day=1)
        last  = today.replace(day=calendar.monthrange(today.year, today.month)[1])
        return str(first), str(last)

    # last week
    if "last week" in low_q or "pichle hafte" in low_q or "pichhle hafte" in low_q:
        start = today - timedelta(days=today.weekday() + 7)
        return str(start), str(start + timedelta(days=6))

    # this week
    if "this week" in low_q or "is hafte" in low_q:
        start = today - timedelta(days=today.weekday())
        return str(start), str(start + timedelta(days=6))

    # Named month (e.g. "april ke orders", "january mein")
    for mn, mv in _MONTH_MAP.items():
        if re.search(rf"\b{mn}\b", low_q):
            year     = today.year if today.month >= mv else today.year - 1
            last_day = calendar.monthrange(year, mv)[1]
            return str(date(year, mv, 1)), str(date(year, mv, last_day))

    return None, None


def _inventory_sql_answer(raw_q: str, low_q: str, db: Session):
    if any(w in low_q for w in _AGG_LOW_STOCK + _AGG_HIGH_STOCK):
        return None
    if not any(w in low_q for w in ["stock", "inventory", "item", "maal", "quantity", "qty", "bearing", "belt", "oil", "seal", "bolt"]):
        return None
    targets = _inventory_targets(raw_q)
    if not targets and not any(w in low_q for w in ["stock", "inventory", "item", "maal"]):
        return None

    results = [{"type": "chat", "message": "Database se stock cross-check kar raha hoon.", "db_checked": True}]
    found = False
    for target in targets[:4]:
        items = _find_inventory_items(db, target)
        if not items:
            continue
        found = True
        if len(items) == 1:
            inv = items[0]
            stock = _stock_for_id(db, inv.id)
            cls = str(inv.classification or "").upper()
            results.append({"type": "chat", "message": f"{inv.name} ka current SQL stock {stock:.2f} {inv.unit or 'units'} hai.", "db_checked": True})
            results.append({
                "type": "result",
                "db_checked": True,
                "inventory": {
                    "id": inv.id,
                    "name": f"{inv.name} {inv.model or ''}".strip(),
                    "category": inv.type or cls or "Raw Material",
                    "placement": inv.placement or "N/A",
                    "unit": inv.unit or "units",
                },
                "total_stock": stock,
                "finish_stock": stock if cls in ("", "FINISH") else 0,
                "semi_finish_stock": stock if "SEMI" in cls else 0,
                "machining_stock": stock if "MACH" in cls else 0,
            })
        else:
            ids = [int(i.id) for i in items]
            total = _stock_for_ids(db, ids)
            results.append({"type": "chat", "message": f"{target.title()} ke {len(items)} matching items mile. Total SQL stock {total:.2f} units hai.", "db_checked": True})
            results.append({
                "type": "dropdown",
                "db_checked": True,
                "message": "Select item for details:",
                "items": [{"id": i.id, "name": f"{i.name} {i.model or ''}".strip()} for i in items],
            })
    if found:
        return {"results": results}
    if any(w in low_q for w in ["stock", "inventory", "item", "maal", "quantity", "qty"]):
        return {"results": [{"type": "chat", "message": "SQL database mein ye inventory item nahi mila. Spelling ya model number check karke poochhiye.", "db_checked": True}]}
    return None


def _inventory_aggregation_sql_answer(raw_q: str, low_q: str, db: Session):
    if any(w in low_q for w in _AGG_LOW_STOCK):
        row = db.execute(text(
            "SELECT i.id,i.name,i.placement,COALESCE(SUM(CASE WHEN LOWER(t.txn_type)='in' THEN t.quantity ELSE -t.quantity END),0) AS stock "
            "FROM inventories i LEFT JOIN stock_transactions t ON i.id=t.inventory_id "
            "GROUP BY i.id,i.name,i.placement ORDER BY stock ASC LIMIT 1"
        )).fetchone()
        if row:
            return {"results": [{"type": "chat", "message": f"Sabse kam stock: {row.name}, {float(row.stock or 0):.2f} units, placement {row.placement or 'N/A'}.", "db_checked": True}]}
    if any(w in low_q for w in _AGG_HIGH_STOCK):
        row = db.execute(text(
            "SELECT i.id,i.name,i.placement,COALESCE(SUM(CASE WHEN LOWER(t.txn_type)='in' THEN t.quantity ELSE -t.quantity END),0) AS stock "
            "FROM inventories i LEFT JOIN stock_transactions t ON i.id=t.inventory_id "
            "GROUP BY i.id,i.name,i.placement ORDER BY stock DESC LIMIT 1"
        )).fetchone()
        if row:
            return {"results": [{"type": "chat", "message": f"Sabse zyada stock: {row.name}, {float(row.stock or 0):.2f} units, placement {row.placement or 'N/A'}.", "db_checked": True}]}
    return None


def _find_suppliers(db: Session, raw_q: str, limit: int = 10):
    target = _clean_entity_query(raw_q, "supplier")
    supplier_noise = {
        "mobile", "phone", "number", "contact", "gst", "gstin", "pan", "email",
        "bank", "ifsc", "details", "profile", "address", "po", "order", "orders",
        "purchase", "batao", "dikhao", "aur", "and", "last", "latest",
    }
    target = " ".join(w for w in target.split() if w not in supplier_noise)
    if not target:
        return []
    if re.match(r"^sup[-\s]?\d+$", target):
        code = re.sub(r"^sup[-\s]?", "", target)
        return db.execute(text("SELECT * FROM suppliers WHERE supplier_code=:c OR id=:c LIMIT 1"), {"c": code}).fetchall()
    words = [w for w in target.lower().split() if len(w) > 1]
    conds = " AND ".join(
        f"(LOWER(supplier_name) LIKE :w{i} OR mobile LIKE :w{i} OR LOWER(COALESCE(gstin,'')) LIKE :w{i})"
        for i in range(len(words))
    )
    if not conds:
        return []
    params = {f"w{i}": f"%{w}%" for i, w in enumerate(words)}
    params["l"] = limit
    return db.execute(text(f"SELECT * FROM suppliers WHERE {conds} ORDER BY supplier_name LIMIT :l"), params).fetchall()


def _supplier_sql_answer(raw_q: str, low_q: str, db: Session):
    if not any(w in low_q for w in ["supplier", "vendor", "party", "gst", "gstin", "pan", "mobile", "phone", "contact", "email", "address", "bank", "ifsc", "detail", "details", "profile"]):
        return None
    sups = _find_suppliers(db, raw_q)
    if not sups:
        if any(w in low_q for w in ["supplier", "vendor", "party"]):
            cnt = db.execute(text("SELECT COUNT(*) FROM suppliers")).scalar() or 0
            return {"results": [{"type": "chat", "message": f"SQL database mein total {cnt} suppliers hain. Kisi supplier ka naam ya city poochhiye.", "db_checked": True}]}
        return None
    if len(sups) > 1:
        return {"results": [
            {"type": "chat", "message": f"SQL database mein {len(sups)} matching suppliers mile.", "db_checked": True},
            {"type": "dropdown", "message": "Select supplier:", "db_checked": True, "items": [{"id": s.id, "name": s.supplier_name} for s in sups]},
        ]}
    s = sups[0]
    if any(w in low_q for w in ["gst", "gstin"]):
        msg = f"{s.supplier_name} ka GSTIN: {s.gstin or 'N/A'}"
        return {"results": [{"type": "chat", "message": msg, "db_checked": True}]}
    if any(w in low_q for w in ["mobile", "phone", "contact", "number"]):
        msg = f"{s.supplier_name} ka mobile/contact: {s.mobile or 'N/A'}"
        return {"results": [{"type": "chat", "message": msg, "db_checked": True}]}
    if "email" in low_q or "mail" in low_q:
        return {"results": [{"type": "chat", "message": f"{s.supplier_name} ka email: {s.email or 'N/A'}", "db_checked": True}]}
    if "pan" in low_q:
        return {"results": [{"type": "chat", "message": f"{s.supplier_name} ka PAN: {s.pan or 'N/A'}", "db_checked": True}]}
    if "bank" in low_q or "ifsc" in low_q:
        return {"results": [{"type": "chat", "message": f"{s.supplier_name} bank: {s.bank_name or 'N/A'}, IFSC: {s.ifsc or 'N/A'}, account: {s.account_number or 'N/A'}", "db_checked": True}]}
    return {"results": [{
        "type": "result",
        "db_checked": True,
        "supplier": {
            "id": s.id,
            "name": s.supplier_name,
            "code": str(s.supplier_code or "N/A"),
            "mobile": s.mobile or "N/A",
            "city": s.city or "N/A",
            "email": s.email or "N/A",
            "gstin": s.gstin or "N/A",
            "pan": s.pan or "N/A",
            "address": s.supplier_address or "N/A",
        },
    }]}


def _supplier_po_combo_sql_answer(raw_q: str, low_q: str, db: Session):
    wants_supplier = any(w in low_q for w in ["gst", "gstin", "phone", "mobile", "contact", "number", "email", "details", "profile"])
    wants_po = any(w in low_q for w in ["po", "order", "orders", "purchase", "balance"])
    if not (wants_supplier and wants_po):
        return None

    sups = _find_suppliers(db, raw_q, limit=1)
    if not sups:
        return None

    s = sups[0]
    results = [{
        "type": "chat",
        "message": (
            f"{s.supplier_name}: mobile {s.mobile or 'N/A'}, "
            f"GSTIN {s.gstin or 'N/A'}."
        ),
        "db_checked": True,
    }]

    rows = db.execute(text(
        "SELECT p.*,s.supplier_name FROM purchase_orders p "
        "LEFT JOIN suppliers s ON p.supplier_id=s.id "
        "WHERE p.supplier_id=:sid ORDER BY p.po_date DESC,p.id DESC LIMIT 5"
    ), {"sid": s.id}).fetchall()
    if rows:
        bal = sum(float(r.balance_amount or 0) for r in rows)
        results.append({"type": "chat", "message": f"Is supplier ke latest {len(rows)} POs mile. Balance total Rs {bal:,.2f}.", "db_checked": True})
        for r in rows:
            results.append({
                "type": "po",
                "db_checked": True,
                "po_id": int(r.id),
                "po_no": str(r.po_number),
                "supplier": str(r.supplier_name or "N/A"),
                "date": str(r.po_date),
                "total": float(r.total_amount or 0),
                "advance": float(r.advance_amount or 0),
                "balance": float(r.balance_amount or 0),
                "status": str(r.status or "N/A"),
            })
    else:
        results.append({"type": "chat", "message": "Is supplier ke purchase orders SQL database mein nahi mile.", "db_checked": True})
    return {"results": results}


def _po_sql_answer(raw_q: str, low_q: str, db: Session):
    if not any(w in low_q for w in ["po", "order", "orders", "purchase", "balance", "advance", "pending", "draft"]):
        return None
    if any(w in low_q for w in ["sabse bada", "biggest", "highest po", "highest order"]):
        row = db.execute(text(
            "SELECT p.*,s.supplier_name FROM purchase_orders p LEFT JOIN suppliers s ON p.supplier_id=s.id "
            "ORDER BY p.total_amount DESC LIMIT 1"
        )).fetchone()
        if row:
            return {"results": [{"type": "chat", "message": f"Sabse bada PO {row.po_number} hai ({row.supplier_name or 'N/A'}), total Rs {float(row.total_amount or 0):,.2f}.", "db_checked": True}, {
                "type": "po", "db_checked": True, "po_id": int(row.id), "po_no": str(row.po_number), "supplier": str(row.supplier_name or "N/A"),
                "date": str(row.po_date), "total": float(row.total_amount or 0), "advance": float(row.advance_amount or 0),
                "balance": float(row.balance_amount or 0), "status": str(row.status or "N/A"),
            }]}
    if any(w in low_q for w in ["sabse chota", "sabse chhota", "smallest", "lowest po", "lowest order"]):
        row = db.execute(text(
            "SELECT p.*,s.supplier_name FROM purchase_orders p LEFT JOIN suppliers s ON p.supplier_id=s.id "
            "ORDER BY p.total_amount ASC LIMIT 1"
        )).fetchone()
        if row:
            return {"results": [{"type": "chat", "message": f"Sabse chota PO {row.po_number} hai ({row.supplier_name or 'N/A'}), total Rs {float(row.total_amount or 0):,.2f}.", "db_checked": True}, {
                "type": "po", "db_checked": True, "po_id": int(row.id), "po_no": str(row.po_number), "supplier": str(row.supplier_name or "N/A"),
                "date": str(row.po_date), "total": float(row.total_amount or 0), "advance": float(row.advance_amount or 0),
                "balance": float(row.balance_amount or 0), "status": str(row.status or "N/A"),
            }]}
    if any(w in low_q for w in ["highest balance", "sabse jada balance", "sabse jyada balance", "maximum balance"]):
        row = db.execute(text(
            "SELECT s.supplier_name,COUNT(p.id) AS cnt,COALESCE(SUM(p.balance_amount),0) AS bal "
            "FROM purchase_orders p LEFT JOIN suppliers s ON p.supplier_id=s.id "
            "WHERE p.balance_amount>0 GROUP BY s.id,s.supplier_name ORDER BY bal DESC LIMIT 1"
        )).fetchone()
        if row:
            return {"results": [{"type": "chat", "message": f"Sabse zyada pending balance {row.supplier_name or 'N/A'} ka hai: Rs {float(row.bal or 0):,.2f} across {row.cnt} POs.", "db_checked": True}]}
    if any(w in low_q for w in ["lowest balance", "sabse kam balance", "minimum balance"]):
        row = db.execute(text(
            "SELECT s.supplier_name,COUNT(p.id) AS cnt,COALESCE(SUM(p.balance_amount),0) AS bal "
            "FROM purchase_orders p LEFT JOIN suppliers s ON p.supplier_id=s.id "
            "WHERE p.balance_amount>0 GROUP BY s.id,s.supplier_name ORDER BY bal ASC LIMIT 1"
        )).fetchone()
        if row:
            return {"results": [{"type": "chat", "message": f"Sabse kam pending balance {row.supplier_name or 'N/A'} ka hai: Rs {float(row.bal or 0):,.2f} across {row.cnt} POs.", "db_checked": True}]}
    if "total pending balance" in low_q or "total balance" in low_q:
        row = db.execute(text("SELECT COUNT(*) AS cnt, COALESCE(SUM(balance_amount),0) AS bal FROM purchase_orders WHERE balance_amount>0")).fetchone()
        return {"results": [{"type": "chat", "message": f"Total pending balance Rs {float(row.bal or 0):,.2f} across {row.cnt} POs.", "db_checked": True}]}
    if any(w in low_q for w in ["kitne", "count", "total", "how many"]) and any(w in low_q for w in ["order", "orders", "po", "purchase"]):
        q = (
            "SELECT COUNT(p.id) AS cnt, COALESCE(SUM(p.balance_amount),0) AS bal "
            "FROM purchase_orders p LEFT JOIN suppliers s ON p.supplier_id=s.id WHERE 1=1"
        )
        params = {}
        target = _clean_entity_query(raw_q, "po")
        words = [
            w for w in target.split()
            if len(w) > 2 and w not in ("total", "count", "kitne", "kitna", "kitni", "hain", "hai", "order", "orders", "purchase", "how", "many")
        ]
        if words:
            conds = " AND ".join(f"LOWER(COALESCE(s.supplier_name,'')) LIKE :s{i}" for i in range(len(words)))
            q += f" AND ({conds})"
            for i, w in enumerate(words):
                params[f"s{i}"] = f"%{w}%"
        row = db.execute(text(q), params).fetchone()
        scope = f" {' '.join(words).title()} ke" if words else ""
        return {"results": [{"type": "chat", "message": f"SQL database mein{scope} {row.cnt} purchase orders hain. Total balance Rs {float(row.bal or 0):,.2f}.", "db_checked": True}]}

    q = "SELECT p.*,s.supplier_name FROM purchase_orders p LEFT JOIN suppliers s ON p.supplier_id=s.id WHERE 1=1"
    params = {"l": 10}
    from_date, to_date = _date_range_from_query(low_q)
    if from_date and to_date:
        q += " AND p.po_date BETWEEN :from_date AND :to_date"
        params["from_date"] = from_date
        params["to_date"] = to_date
    if any(w in low_q for w in ["latest", "last", "recent"]):
        params["l"] = 5
    if any(w in low_q for w in ["pending", "draft"]):
        q += " AND LOWER(p.status)='draft'"
    if "completed" in low_q:
        q += " AND LOWER(p.status)='completed'"
    po_pat = re.search(r"[A-Za-z]{2,}[/\-][A-Za-z0-9]+[/\-][A-Za-z0-9\-/]+|\b\d{4,}\b", raw_q)
    if po_pat:
        q += " AND LOWER(p.po_number) LIKE :po"
        params["po"] = f"%{po_pat.group(0).lower()}%"
    else:
        target = _clean_entity_query(raw_q, "po")
        words = [w for w in target.split() if len(w) > 2 and w not in ("latest", "last", "pending", "draft", "month", "week", "this", "show", "purchase", "order", "orders", "advance", "prepaid", "diya", "gaya", "kisi", "kitna", "kitni", "kitne", "total", "bhi", "iska", "uska", "sabka")]
        if words:
            conds = " AND ".join(f"LOWER(COALESCE(s.supplier_name,'')) LIKE :s{i}" for i in range(len(words)))
            q += f" AND ({conds})"
            for i, w in enumerate(words):
                params[f"s{i}"] = f"%{w}%"
    rows = db.execute(text(q + " ORDER BY p.po_date DESC,p.id DESC LIMIT :l"), params).fetchall()
    if not rows:
        return {"results": [{"type": "chat", "message": "SQL database mein is filter par purchase order nahi mila.", "db_checked": True}]}
    bal = sum(float(r.balance_amount or 0) for r in rows)
    results = [{"type": "chat", "message": f"SQL database se {len(rows)} purchase orders mile. Balance total Rs {bal:,.2f}.", "db_checked": True}]
    for r in rows:
        results.append({
            "type": "po",
            "db_checked": True,
            "po_id": int(r.id),
            "po_no": str(r.po_number),
            "supplier": str(r.supplier_name or "N/A"),
            "date": str(r.po_date),
            "total": float(r.total_amount or 0),
            "advance": float(r.advance_amount or 0),
            "balance": float(r.balance_amount or 0),
            "status": str(r.status or "N/A"),
        })
    return {"results": results}


def _project_sql_answer(raw_q: str, low_q: str, db: Session):
    if not any(w in low_q for w in ["project", "projects", "site", "budget", "priority"]):
        return None
    if any(w in low_q for w in ["kitne", "count", "total", "how many"]):
        row = db.execute(text(
            "SELECT COUNT(*) AS cnt, SUM(CASE WHEN LOWER(status) IN ('in progress','in_progress','new') THEN 1 ELSE 0 END) AS active "
            "FROM projects WHERE is_deleted=0"
        )).fetchone()
        return {"results": [{"type": "chat", "message": f"SQL database mein {row.cnt} active/non-deleted projects hain. Open/new count {int(row.active or 0)}.", "db_checked": True}]}
    q = "SELECT * FROM projects WHERE is_deleted=0"
    params = {"l": 10}
    if "completed" in low_q:
        q += " AND LOWER(status)='completed'"
    elif any(w in low_q for w in ["running", "active", "progress"]):
        q += " AND LOWER(status) IN ('in progress','in_progress','new')"
    target = _clean_entity_query(raw_q, "project")
    words = [
        w for w in target.split()
        if len(w) > 2 and w not in ("running", "active", "completed", "project", "projects", "site", "dikhao", "batao", "budget", "status")
    ]
    if words:
        conds = " AND ".join(f"LOWER(name) LIKE :p{i}" for i in range(len(words)))
        q += f" AND ({conds})"
        for i, w in enumerate(words):
            params[f"p{i}"] = f"%{w}%"
    rows = db.execute(text(q + " ORDER BY id DESC LIMIT :l"), params).fetchall()
    if not rows:
        return {"results": [{"type": "chat", "message": "SQL database mein is filter par project nahi mila.", "db_checked": True}]}
    results = [{"type": "chat", "message": f"SQL database se {len(rows)} projects mile.", "db_checked": True}]
    for p in rows:
        results.append({
            "type": "project",
            "db_checked": True,
            "project_name": str(p.name),
            "category": str(p.status or "N/A"),
            "amount": float(p.budget or 0),
            "start_date": str(p.start_date or "N/A"),
            "end_date": str(p.end_date or p.deadline or "N/A"),
            "priority": str(p.priority or "N/A"),
            "comments": str(p.comment or ""),
        })
    return {"results": results}


def _purchase_request_sql_answer(raw_q: str, low_q: str, db: Session):
    if not (
        "purchase request" in low_q
        or "requisition" in low_q
        or re.search(r"\bpr[-\s]?\d*\b", low_q)
    ):
        return None
    if any(w in low_q for w in ["kitne", "count", "total", "how many"]):
        cnt = db.execute(text("SELECT COUNT(*) FROM purchase_requests")).scalar() or 0
        return {"results": [{"type": "chat", "message": f"SQL database mein total {cnt} purchase requests hain.", "db_checked": True}]}
    pr_match = re.search(r"\bpr[-\s]?(\d+)\b", low_q)
    if pr_match:
        pr_no = f"PR-{pr_match.group(1)}"
        rows = db.execute(text("SELECT * FROM purchase_requests WHERE LOWER(pr_no)=:pr LIMIT 10"), {"pr": pr_no.lower()}).fetchall()
    else:
        rows = db.execute(text(
        "SELECT * FROM purchase_requests ORDER BY request_date DESC,id DESC LIMIT 10"
        )).fetchall()
    return {"results": [{"type": "chat", "message": f"SQL database mein latest {len(rows)} purchase requests mile.", "db_checked": True}] + [
        {"type": "purchase_request", "db_checked": True, "pr_no": r.pr_no, "date": str(r.request_date), "status": r.status, "priority": r.priority, "total_qty": float(r.total_qty or 0)}
        for r in rows
    ]}


def _history_contextual_query(raw_q: str, low_q: str, history: list | None) -> str:
    sticky = _extract_sticky(history or [], low_q)
    if not sticky:
        return raw_q

    if low_q.strip() in ("yes", "haa", "ha", "ji", "details", "detail", "profile", "poori profile"):
        return f"{sticky} details"
    if low_q.strip() in ("orders", "po", "purchase", "orders dikhao", "po dikhao"):
        return f"{sticky} orders"
    if _is_followup_query(low_q) or any(w in low_q for w in ["gst", "mobile", "phone", "email", "balance", "orders", "stock", "details"]):
        return f"{sticky} {raw_q}"
    return raw_q


def _try_sql_first_answer(raw_q: str, low_q: str, db: Session, history: list | None = None):
    if not raw_q:
        return _erp_help_response()

    raw_q = _history_contextual_query(raw_q, low_q, history)
    low_q = raw_q.lower()

    count_answer = None
    _is_count_q = (
        any(w in low_q for w in ["kitne", "kitni", "count", "total", "how many"])
        or re.search(r"\b(number|no\.?)\s+of\b", low_q) is not None
    )
    if _is_count_q:
        if any(w in low_q for w in ["supplier", "vendor", "party"]):
            count_answer = ("suppliers", "SELECT COUNT(*) FROM suppliers")
        elif any(w in low_q for w in ["inventory", "item", "stock", "maal"]):
            count_answer = ("inventory items", "SELECT COUNT(*) FROM inventories")
        elif any(w in low_q for w in ["user", "employee"]):
            count_answer = ("users", "SELECT COUNT(*) FROM users WHERE is_delete=0")
    if count_answer:
        label, sql = count_answer
        cnt = db.execute(text(sql)).scalar() or 0
        return {"results": [{"type": "chat", "message": f"SQL database mein total {cnt} {label} hain.", "db_checked": True}]}

    for handler in (
        _purchase_request_sql_answer,
        _supplier_po_combo_sql_answer,
        _po_sql_answer,
        _project_sql_answer,
        _supplier_sql_answer,
        _inventory_aggregation_sql_answer,
        _inventory_sql_answer,
    ):
        answer = handler(raw_q, low_q, db)
        if answer:
            return answer

    tokens = set(re.findall(r"[a-z0-9]+", low_q))
    if tokens.intersection(_CHAT_ONLY_WORDS) or not tokens.intersection(_ERP_WORDS):
        return _erp_help_response()
    return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────
@router.post("/")
def v2_chatbot(request: ChatRequest, db: Session = Depends(get_db)):
    import uuid
    request_id = uuid.uuid4().hex[:16]
    start = time.time()
    try:
        response = _v2_chatbot_impl(request, db)
        # Surface request_id to client so they can submit feedback on this exact reply
        if isinstance(response, dict):
            response["request_id"] = request_id
        _chatbot_reqres_log(
            _request_to_log_dict(request),
            response,
            int((time.time() - start) * 1000),
            request_id=request_id,
        )
        return response
    except Exception as e:
        response = {"results": [{"type": "chat", "message": "Request process karte waqt error aa gaya."}], "request_id": request_id}
        _chatbot_reqres_log(
            _request_to_log_dict(request),
            response,
            int((time.time() - start) * 1000),
            error=str(e),
            request_id=request_id,
        )
        raise


def _v2_chatbot_legacy_flow(request: ChatRequest, db: Session, raw_q: str, low_q: str, history: list, role: str):
    # context_entity emitted by this response (set in each handler below)
    _ce_out: Optional[Dict] = None

    # ── Fast-track: pure numeric ID ───────────────────────────────────────────
    if low_q.isdigit() and len(low_q) < 8:
        try:
            inv = db.execute(
                text("SELECT id, name, classification, placement FROM inventories WHERE id=:id"),
                {"id": int(low_q)}
            ).fetchone()
            if inv:
                stock = float(db.execute(
                    text("SELECT SUM(CASE WHEN LOWER(txn_type)='in' THEN quantity ELSE -quantity END) "
                         "FROM stock_transactions WHERE inventory_id=:id"),
                    {"id": inv.id}
                ).scalar() or 0)
                cls = (inv.classification or "").lower()
                m = stock if "machining" in cls else 0
                f = 0 if "machining" in cls else (0 if "semi" in cls else stock)
                sf= stock if "semi" in cls else 0
                return {"results": [{"type":"result","inventory":{
                    "id": inv.id,"name": inv.name,
                    "category": cls.upper(),"placement": inv.placement or "N/A"},
                    "total_stock": stock,"finish_stock": f,
                    "semi_finish_stock": sf,"machining_stock": m}]}
        except Exception:
            pass

    # ── LLM Intent Extraction ─────────────────────────────────────────────────
    try:
        ai = ask_local_llm(raw_q, history)
        print(f"[V2-LLM] {ai}")
    except RuntimeError as e:
        # Don't leak internal stack traces / API keys / hostnames to the user.
        print(f"[V2-LLM-FAIL] {e}")
        return {"results": [{"type": "chat", "message": "AI brain abhi busy hai, thodi der mein try karein. 🙏"}]}
    except Exception as e:
        _log(raw_q, "unknown", str(e))
        return {"results": [{"type": "chat", "message": "Bhai, AI brain abhi connect nahi ho raha. Thodi der mein try karein. 🙏"}]}

    # ── Parse intents ─────────────────────────────────────────────────────────
    intents = ai.get("intents") or []
    if not intents and "intent" in ai:
        intents = [ai["intent"]]
    if isinstance(intents, str):
        intents = [intents]
    if not intents:
        intents = ["search"]

    # ── Parse target ──────────────────────────────────────────────────────────
    target = str(ai.get("search_target") or "").strip()
    if not target:
        sp = ai.get("specific_items") or []
        if isinstance(sp, list) and sp:
            target = str(sp[0] or "").strip()

    # ── Detect global / independent queries that should NOT inherit context ──────
    # "saare/sab po batao" = global list; "0030 wala po" = specific ID search.
    # In both cases: clear any context-leaked search_target and skip sticky.
    _PO_NUM_RE    = re.compile(r'\b\d{4,}\b|MHEL/PO|PO/\d', re.IGNORECASE)
    _GLOBAL_WORDS = {"saare","sab","sabhi","all","poore","global","har ek"}
    # "X ke saare Y" = all Y of X  →  entity-specific, NOT global
    # "saare Y" alone = global list
    _possessive_global = bool(re.search(r'\b(ke|ka|ki)\s+(saare|sab|sabhi|all|poore)\b', low_q))
    _is_global  = (any(w in low_q.split() for w in _GLOBAL_WORDS) and not _possessive_global)
    _has_po_num = bool(_PO_NUM_RE.search(raw_q))
    _ce_bypass  = _is_global or _has_po_num
    if _is_global:
        target = ""   # "saare po" → no target, return all
    elif _has_po_num:
        # Extract the PO number as target; don't use any context-leaked supplier name
        _po_match = _PO_NUM_RE.search(raw_q)
        target = _po_match.group(0) if _po_match else target
        # Force po_search intent so the PO handler runs (LLM may have guessed "search")
        if "po_search" not in intents:
            intents = list(intents) + ["po_search"]

    # ── Sticky context for follow-up queries ──────────────────────────────────
    is_fu = _is_followup_query(low_q)
    if not _ce_bypass and (not target or is_fu):
        sticky = _extract_sticky(history, low_q)
        if sticky and (not target or is_fu):
            print(f"[V2-STICKY] '{sticky}' (was: '{target}')")
            target = sticky

    # ── Context-entity follow-up (highest accuracy path for multi-turn) ─────────
    # Before touching target/intents at all, check if the user is following up on
    # a previous entity. If yes, do targeted IN-list queries — no LLM, no LIKE.
    _ce_in = _last_context_entity(history)
    if _ce_in and not _ce_bypass and (_is_followup_query(low_q) or len(raw_q.split()) <= 3):
        _ctx_results = _context_followup(_ce_in, intents, low_q, db,
                                         max(1, int((ai.get("filters") or {}).get("limit") or 5)))
        if _ctx_results:
            print(f"[V2-CTX] context_followup hit: type={_ce_in.get('type')} q='{raw_q}'")
            return {"results": [{"type":"chat","message": ai.get("reasoning") or "haan dekh raha hoon"}]
                    + _ctx_results,
                    "context_entity": _ce_in}   # preserve context for next turn

    # ── Human-like filler message ─────────────────────────────────────────────
    reasoning    = ai.get("reasoning") or "hmm ek sec... check karta hoon 👍"
    final_results = [{"type": "chat", "message": reasoning}]

    # ── Simple follow-up: "yes" / "details" / "orders" ───────────────────────
    _simple_fu   = low_q.strip() in ("yes","haa","ha","ji","details","detail","profile","poori profile")
    _orders_fu   = low_q.strip() in ("orders","po","purchase","orders dikhao","po dikhao")
    if (_simple_fu or _orders_fu) and history:
        sticky = _extract_sticky(history, low_q)
        if sticky:
            target  = sticky
            intents = ["po_search"] if _orders_fu else ["supplier_search"]
            print(f"[V2-FU] '{low_q}' → target='{sticky}' intents={intents}")

    # ── Filters (defined early so seatbelt and overrides can use them) ────────
    filters    = dict(ai.get("filters") or {})
    ui_filters = dict(getattr(request, "ui_filters", {}) or {})
    for k, v in ui_filters.items():
        if v:
            filters[k] = v
    limit = max(1, int(filters.get("limit") or 15))

    # ── Seatbelt: supplier/project without a name and no useful filters ──────
    _has_filter = any(filters.get(k) for k in ("city","status","priority","from_date","to_date"))
    _is_list_q  = any(w in low_q for w in ["all","saare","sabhi","list","kitne","how many","count","total"])
    # Status/condition keywords also bypass seatbelt for projects
    _is_status_q = any(w in low_q for w in [
        "overdue","hold","ruka","urgent","running","active","chalu","completed",
        "progress","refurbish","high priority","pending","latest","last",
    ])
    if not target and not _has_filter and not _is_list_q and not _is_status_q \
            and any(i in intents for i in ["supplier_search","project_search"]):
        return {"results": [{"type":"chat","message":"Bhai, kripya thoda clear batao — kis company ya project ki baat kar rahe ho? 🙂"}]}

    # ── Fuzzy-resolve target against real DB entities ────────────────────────
    # Three confidence zones:
    #   high       : silently accept (current behavior for clean matches)
    #   borderline : ask user to confirm via 👍/👎/1-2-3 — biggest accuracy win
    #   low        : pass through, let LIKE matching try
    pre_resolved = (getattr(request, "ui_filters", None) or {}).get("__resolved_entity__")
    if pre_resolved:
        # Confirm-before-act resume path injected this. Trust it, skip resolver.
        target = pre_resolved
        ai["search_target"] = target

    if not pre_resolved and target and not target.isdigit() and len(target) >= 2:
        category = None
        if "supplier_search" in intents:
            category = "supplier"
        elif "project_search" in intents:
            category = "project"
        elif "search" in intents:  # inventory
            category = "inventory"
        elif "po_search" in intents:
            category = "supplier"   # POs are queried via supplier_name

        if category:
            res = resolve_with_confidence(db, target, category)
            conf = res["confidence"]

            if conf == "borderline":
                # Don't act — ask user. Backend returns a structured response the
                # frontend renders as 👍/👎 buttons (or numbered list of candidates).
                cands = [c[0] for c in res["candidates"]] or [res["canonical"]]
                if len(cands) == 1:
                    msg = (f"Sure aap **{cands[0]}** ki baat kar rahe ho?\n\n"
                           f"👍 Yes  /  👎 No")
                else:
                    body = "\n".join(f"  {i+1}. **{n}**" for i, n in enumerate(cands))
                    msg = (f"Mujhe {len(cands)} possible matches mile. Konsa wala?\n\n{body}\n\n"
                           f"👍 (first one)  /  👎 (none of these)  /  ya number type karein")

                # Fetch DB ids for each candidate so the frontend can do a direct
                # SQL card fetch on click — no LLM roundtrip needed.
                _id_sql = {
                    "supplier":  "SELECT id FROM suppliers  WHERE supplier_name=:n LIMIT 1",
                    "inventory": "SELECT id FROM inventories WHERE name=:n          LIMIT 1",
                    "project":   "SELECT id FROM projects   WHERE name=:n           LIMIT 1",
                }
                cand_ids = []
                if category in _id_sql:
                    for cname in cands:
                        row = db.execute(text(_id_sql[category]), {"n": cname}).fetchone()
                        cand_ids.append(int(row[0]) if row else None)

                return {
                    "results": [{
                        "type":          "confirm_resolution",
                        "message":       msg,
                        "candidates":    cands,
                        "candidate_ids": cand_ids,
                        "category":      category,
                        "score":         res["score"],
                    }],
                    "pending_resolution": {
                        "candidates":     cands,
                        "category":       category,
                        "intents":        intents,
                        "original_query": raw_q,
                    },
                }

            if conf in ("high", "alias") and res["canonical"].lower() != target.lower():
                print(f"[V2-RESOLVE] {category}: '{target}' -> '{res['canonical']}' (score={res['score']}, {conf})")
                target = res["canonical"]

    ai["search_target"] = target

    # ── Resolve secondary_target too (for "A vs B" comparison queries) ───────
    sec = (ai.get("secondary_target") or "").strip()
    if sec and not sec.isdigit() and len(sec) >= 2:
        sec_resolved, sec_score = resolve_entity(db, sec, "supplier")
        if sec_score is not None:
            ai["secondary_target"] = sec_resolved

    # ── Complex query short-circuit (top_n, max/min, compare, threshold, negate)
    # Returns None if not a complex query → falls through to legacy handlers.
    try:
        complex_resp = handle_complex(ai, db, target)
        if complex_resp is not None:
            print(f"[V2-COMPLEX] handled: agg={ai.get('aggregation')} neg={ai.get('negate')} cmp={ai.get('comparison')}")
            return complex_resp
    except Exception as e:
        # Never let the new path break the legacy fallback
        print(f"[V2-COMPLEX] handler error, falling through: {e}")

    # ── Hinglish keyword overrides (only when LLM returned generic "search") ──
    _po_words  = ["paisa","baki","baaki","rokra","payment","balance","hisab","udhar",
                   "po","order","orders","purchase","pending","draft","transit","dispatch","delivery"]
    _sup_words = ["supplier","vendor","party","contact","mobile","gstin"]
    _proj_words= ["project","projects","site","crusher","refurbish"]

    if any(w in low_q for w in ["tax","gst","cgst","sgst"]) and \
       any(w in low_q for w in ["total","kitna","bana","calculate","how much","kitni","lagta"]):
        intents = ["po_search"]

    # Auto-promote: "profile + orders" → add supplier_search if missing
    if "po_search" in intents and target and \
       any(w in low_q for w in ["profile","details","detail"]) and \
       "supplier_search" not in intents:
        intents = ["supplier_search"] + [i for i in intents if i != "supplier_search"]

    elif intents == ["search"]:
        if any(re.search(rf"\b{w}\b", low_q) for w in _po_words):
            intents = ["po_search"]
        elif any(re.search(rf"\b{w}\b", low_q) for w in _sup_words):
            intents = ["supplier_search"]
        elif any(w in low_q for w in _proj_words):
            intents = ["project_search"]

    # ── Smart entity resolver ─────────────────────────────────────────────────
    intents = _resolve_entity(target, intents, low_q)
    print(f"[V2-ROUTER] intents={intents} | target='{target}'")

    # ── Role-based permission ─────────────────────────────────────────────────
    _blocked = {"supervisor": {"po_search"}}
    blocked  = _blocked.get(role, set())
    if blocked.intersection(set(intents)):
        intents = [i for i in intents if i not in blocked]
        if not intents:
            return {"results": [{"type":"chat","message":"🚫 Aapke paas Purchase Orders dekhne ki permission nahi hai."}]}
        final_results.append({"type":"chat","message":"🚫 Note: PO info restricted — baaki data dikha raha hoon."})

    # ── Aggregation fast-paths ────────────────────────────────────────────────
    if any(w in low_q for w in _AGG_LOW_STOCK):
        try:
            row = db.execute(text(
                "SELECT i.id,i.name,i.placement,"
                "SUM(CASE WHEN LOWER(t.txn_type)='in' THEN t.quantity ELSE -t.quantity END) AS stock "
                "FROM inventories i JOIN stock_transactions t ON i.id=t.inventory_id "
                "GROUP BY i.id,i.name,i.placement ORDER BY stock ASC LIMIT 1"
            )).fetchone()
            if row:
                final_results.append({"type":"chat","message":
                    f"📉 Sabse kam stock: **{row.name}** — sirf **{float(row.stock):.2f} units** (📍 {row.placement or 'N/A'})"})
            return {"results": final_results}
        except Exception as e:
            final_results.append({"type":"chat","message":f"Aggregation error: {e}"})
            return {"results": final_results}

    if any(w in low_q for w in _AGG_HIGH_STOCK):
        try:
            row = db.execute(text(
                "SELECT i.id,i.name,i.placement,"
                "SUM(CASE WHEN LOWER(t.txn_type)='in' THEN t.quantity ELSE -t.quantity END) AS stock "
                "FROM inventories i JOIN stock_transactions t ON i.id=t.inventory_id "
                "GROUP BY i.id,i.name,i.placement ORDER BY stock DESC LIMIT 1"
            )).fetchone()
            if row:
                final_results.append({"type":"chat","message":
                    f"📈 Sabse zyada stock: **{row.name}** — **{float(row.stock):.2f} units** (📍 {row.placement or 'N/A'})"})
            return {"results": final_results}
        except Exception as e:
            final_results.append({"type":"chat","message":f"Aggregation error: {e}"})
            return {"results": final_results}

    if any(w in low_q for w in _AGG_PEND_BAL):
        try:
            row = db.execute(text(
                "SELECT SUM(balance_amount) AS total_bal, COUNT(id) AS cnt,"
                "SUM(CASE WHEN LOWER(status)='draft' THEN balance_amount ELSE 0 END) AS draft_bal "
                "FROM purchase_orders WHERE balance_amount>0"
            )).fetchone()
            if row and row.total_bal:
                final_results.append({"type":"chat","message":(
                    f"💰 **Total Pending Balance:**\n\n"
                    f"📊 Outstanding: **₹{float(row.total_bal):,.2f}**\n"
                    f"📄 Orders with balance: **{row.cnt}**\n"
                    f"🟡 Draft/Pending: **₹{float(row.draft_bal or 0):,.2f}**"
                )})
            else:
                final_results.append({"type":"chat","message":"Koi pending balance nahi mila. Sab clear! 👍"})
            return {"results": final_results}
        except Exception as e:
            final_results.append({"type":"chat","message":f"Balance error: {e}"})
            return {"results": final_results}

    if any(w in low_q for w in _AGG_COUNT):
        try:
            if any(w in low_q for w in ["supplier","vendor","party"]):
                if any(w in low_q for w in ["city","shahar","shehr","citywise","city wise","city-wise"]):
                    city_rows = db.execute(text(
                        "SELECT city, COUNT(*) AS cnt FROM suppliers "
                        "WHERE city IS NOT NULL AND city != '' GROUP BY city ORDER BY cnt DESC"
                    )).fetchall()
                    if city_rows:
                        lines = "\n".join(f"• **{r.city}**: {r.cnt}" for r in city_rows)
                        final_results.append({"type":"chat","message":
                            f"🏙️ City-wise supplier count ({len(city_rows)} cities):\n\n{lines}"})
                    else:
                        final_results.append({"type":"chat","message":"City data available nahi hai. 🧐"})
                    return {"results": final_results}
                cnt = db.execute(text("SELECT COUNT(*) FROM suppliers")).scalar() or 0
                final_results.append({"type":"chat","message":f"👥 Total **{cnt} suppliers** registered hain."})
                return {"results": final_results}
            if any(w in low_q for w in ["project","site"]):
                cnt    = db.execute(text("SELECT COUNT(*) FROM projects WHERE is_deleted=0")).scalar() or 0
                active = db.execute(text("SELECT COUNT(*) FROM projects WHERE is_deleted=0 AND LOWER(status)='in progress'")).scalar() or 0
                final_results.append({"type":"chat","message":f"🏗️ **{cnt} projects** total, **{active} active** abhi."})
                return {"results": final_results}
            if any(w in low_q for w in ["item","inventory","maal","stock"]):
                cnt = db.execute(text("SELECT COUNT(*) FROM inventories")).scalar() or 0
                final_results.append({"type":"chat","message":f"📦 Inventory mein **{cnt} unique items** hain."})
                return {"results": final_results}
            if any(w in low_q for w in ["order","po","purchase"]):
                cnt  = db.execute(text("SELECT COUNT(*) FROM purchase_orders")).scalar() or 0
                pend = db.execute(text("SELECT COUNT(*) FROM purchase_orders WHERE LOWER(status)='draft'")).scalar() or 0
                final_results.append({"type":"chat","message":f"🧾 **{cnt} purchase orders** total, **{pend} pending/draft**."})
                return {"results": final_results}
        except Exception as e:
            final_results.append({"type":"chat","message":f"Count error: {e}"})
            return {"results": final_results}

    # ── Multi-intent processing loop ──────────────────────────────────────────
    for intent in intents:

        # ── PROJECT ──────────────────────────────────────────────────────────
        if intent == "project_search":
            try:
                t          = clean_noise(target, "project")
                status_f   = str(filters.get("status") or "").lower().strip()
                q          = "SELECT * FROM projects WHERE is_deleted=0"
                params: dict = {}

                # Keyword-based status overrides (higher priority than LLM filters)
                if "overdue" in low_q or "late project" in low_q or "deadline miss" in low_q:
                    q += (" AND COALESCE(end_date, deadline) IS NOT NULL"
                          " AND COALESCE(end_date, deadline) < :today"
                          " AND LOWER(COALESCE(status,'')) NOT IN ('completed')")
                    params["today"] = str(date.today())
                    status_f = ""  # already applied, don't double-apply
                elif any(w in low_q for w in ["chalu","running","active","in progress","jari"]) and not status_f:
                    status_f = "in progress"
                elif any(w in low_q for w in ["hold","ruka","ruke"]) and not status_f:
                    status_f = "hold"
                elif any(w in low_q for w in ["urgent","high priority"]) and not status_f:
                    q += " AND LOWER(COALESCE(priority,''))='high'"

                if status_f and status_f != "all":
                    if status_f == "refurbished":
                        q += " AND refurbish=1"
                    else:
                        # Map user-friendly status to database status
                        status_map = {
                            "chalu": "in_progress",
                            "running": "in_progress",
                            "active": "in_progress",
                            "in progress": "in_progress",
                            "jari": "in_progress",
                            "started": "in_progress",
                            "ongoing": "in_progress"
                        }
                        db_status = status_map.get(status_f.lower(), status_f)
                        q += " AND LOWER(status)=:st"; params["st"] = db_status

                if filters.get("from_date") and filters.get("to_date"):
                    q += " AND start_date BETWEEN :fd AND :td"
                    params.update({"fd": filters["from_date"], "td": filters["to_date"]})
                elif filters.get("from_date"):
                    q += " AND start_date >= :fd"; params["fd"] = filters["from_date"]

                if t and t not in ("all","list","projects","latest"):
                    words = [w for w in t.split() if len(w) > 2]
                    if words:
                        conds = " AND ".join(
                            f"(LOWER(name) LIKE :t{i} OR LOWER(comment) LIKE :t{i})"
                            for i in range(len(words))
                        )
                        q += f" AND ({conds})"
                        for i, w in enumerate(words):
                            params[f"t{i}"] = f"%{w}%"

                rows = db.execute(text(q + " ORDER BY id DESC LIMIT :l"), {**params,"l":limit}).fetchall()

                if not rows and t and len(t) > 3:
                    corrected = faiss_match(t, "project")
                    if corrected and corrected.lower() != t:
                        rows = db.execute(text(
                            "SELECT * FROM projects WHERE is_deleted=0 AND LOWER(name) LIKE :cn LIMIT :l"
                        ), {"cn": f"%{corrected.lower()}%", "l": limit}).fetchall()

                if not rows:
                    final_results.append({"type":"chat","message":f"'{t or 'is filter'}' wala koi project nahi mila. 🧐"})
                else:
                    final_results.append({"type":"chat","message":f"haan mil gaya 👍 **{len(rows)} projects** mile:"})
                    for p in rows:
                        s      = str(p.status or "").lower()
                        stage  = ("100%" if s=="completed" else "50%" if s=="in progress"
                                  else "Hold" if s=="hold" else "0%")
                        rtype  = "Refurbished" if getattr(p,"refurbish",0)==1 else "New Machine"
                        final_results.append({
                            "type": "project","project_name": str(p.name),
                            "category": f"{rtype} | {str(p.status or '').capitalize()}",
                            "amount":   float(p.budget or 0),
                            "start_date": str(p.start_date) if p.start_date else "N/A",
                            "end_date":   str(getattr(p,"end_date",None) or getattr(p,"deadline",None) or "N/A"),
                            "comments":   str(p.comment or ""),
                            "stage":      getattr(p,"stage",stage),
                            "priority":   str(p.priority or "").upper(),
                        })
                    # Emit context_entity for single project so follow-ups can ask items/products
                    if len(rows) == 1:
                        _p0 = rows[0]
                        _inv_ids_p = [r.inventory_id for r in db.execute(text(
                            "SELECT inventory_id FROM project_item WHERE project_id=:pid AND inventory_id IS NOT NULL"
                        ), {"pid": _p0.id}).fetchall()]
                        _prod_ids_p = [r.product_id for r in db.execute(text(
                            "SELECT product_id FROM project_products WHERE project_id=:pid AND product_id IS NOT NULL"
                        ), {"pid": _p0.id}).fetchall()]
                        _ce_out = {
                            "type":         "project",
                            "id":           _p0.id,
                            "name":         str(_p0.name),
                            "inventory_ids": _inv_ids_p,
                            "product_ids":  _prod_ids_p,
                        }
            except Exception as e:
                final_results.append({"type":"chat","message":f"Project error: {e}"})

        # ── SUPPLIER ─────────────────────────────────────────────────────────
        elif intent == "supplier_search":
            try:
                t_lower      = target.lower()
                t_clean      = clean_noise(target, "supplier")
                city_f       = str(filters.get("city") or "").lower().strip()
                is_all       = not t_clean and not city_f and any(w in low_q for w in ["all","saare","sabhi","list","kitne","total"])
                sups         = []

                if city_f and not t_clean:
                    sups = db.execute(text(
                        "SELECT * FROM suppliers WHERE LOWER(city) LIKE :c ORDER BY supplier_name LIMIT :l"
                    ), {"c": f"%{city_f}%", "l": max(limit, 50)}).fetchall()
                elif is_all:
                    if any(w in low_q for w in ["city","shahar","shehr","citywise","city wise"]):
                        # City-wise grouped summary instead of a raw list
                        city_rows = db.execute(text(
                            "SELECT city, COUNT(*) AS cnt FROM suppliers "
                            "WHERE city IS NOT NULL AND city != '' GROUP BY city ORDER BY cnt DESC"
                        )).fetchall()
                        if city_rows:
                            lines = "\n".join(f"• **{r.city}**: {r.cnt} suppliers" for r in city_rows)
                            final_results.append({"type":"chat","message":
                                f"🏙️ City-wise suppliers ({len(city_rows)} cities):\n\n{lines}"})
                        # sups stays [] — "not sups" with empty target won't add error message
                    else:
                        _all_lim = max(limit, 20)
                        sups = db.execute(text(
                            "SELECT * FROM suppliers ORDER BY id DESC LIMIT :l"
                        ), {"l": _all_lim}).fetchall()
                else:
                    # 1. SUP-code exact match
                    if re.match(r"^sup[-\s]?\d+$", t_lower):
                        code = re.sub(r"^sup[-\s]?", "", t_lower)
                        sups = db.execute(text(
                            "SELECT * FROM suppliers WHERE supplier_code=:c OR id=:c LIMIT 1"
                        ), {"c": code}).fetchall()

                    # 2. Exact name match
                    if not sups and t_clean:
                        sups = db.execute(text(
                            "SELECT * FROM suppliers WHERE LOWER(supplier_name)=:q LIMIT 1"
                        ), {"q": t_clean}).fetchall()

                    # 3. LIKE multi-word match
                    if not sups and t_clean:
                        words = [w for w in t_clean.split() if len(w) > 2]
                        if words:
                            conds  = " AND ".join(
                                f"(LOWER(supplier_name) LIKE :w{i} OR mobile LIKE :w{i})"
                                for i in range(len(words))
                            )
                            params = {f"w{i}": f"%{w}%" for i, w in enumerate(words)}
                            params["l"] = limit
                            sups = db.execute(text(f"SELECT * FROM suppliers WHERE {conds} LIMIT :l"), params).fetchall()

                    # 4. FAISS correction
                    if not sups and t_clean and len(t_clean) > 2:
                        corrected = faiss_match(t_clean, "supplier")
                        if corrected and corrected.lower() != t_clean:
                            sups = db.execute(text(
                                "SELECT * FROM suppliers WHERE LOWER(supplier_name)=:q LIMIT 1"
                            ), {"q": corrected.lower()}).fetchall()

                    # 5. difflib fallback against live DB names
                    if not sups and t_clean and len(t_clean) > 2:
                        all_sup_names = [r[0].lower() for r in
                                         db.execute(text("SELECT supplier_name FROM suppliers")).fetchall() if r[0]]
                        close = difflib.get_close_matches(t_clean, all_sup_names, n=1, cutoff=0.6)
                        if close:
                            sups = db.execute(text(
                                "SELECT * FROM suppliers WHERE LOWER(supplier_name)=:q LIMIT 1"
                            ), {"q": close[0]}).fetchall()

                    # 6. City fallback — treat target as city name when all name lookups fail
                    if not sups and t_clean and len(t_clean) > 2:
                        sups = db.execute(text(
                            "SELECT * FROM suppliers WHERE LOWER(COALESCE(city,'')) LIKE :c "
                            "ORDER BY supplier_name LIMIT :l"
                        ), {"c": f"%{t_clean}%", "l": max(limit, 50)}).fetchall()

                if not sups:
                    if target:
                        final_results.append({"type":"chat","message":f"'{target}' naam ka koi supplier nahi mila. 🧐"})
                elif len(sups) > 1:
                    final_results.append({"type":"chat","message":f"haan mil gaya 👍 **{len(sups)} suppliers** mile:"})
                    final_results.append({
                        "type":    "dropdown",
                        "message": "Select a supplier for details:",
                        "items":   [{"id": str(getattr(s,"supplier_name","?")), "name": str(getattr(s,"supplier_name","?"))} for s in sups],
                    })
                else:
                    s        = sups[0]
                    sup_name = str(getattr(s,"supplier_name","Unknown"))

                    # Surgical single-field responses (check both raw query and LLM-extracted specific_items)
                    _llm_items_str = " ".join([str(x).lower() for x in (ai.get("specific_items") or [])])
                    _want_gst     = any(w in low_q for w in ["gst","gstin","tax"]) or any(w in _llm_items_str for w in ["gst","gstin"])
                    _want_email   = any(w in low_q for w in ["email","mail"]) or "email" in _llm_items_str
                    _want_mobile  = any(w in low_q for w in ["mobile","phone","call","contact"]) or any(w in _llm_items_str for w in ["phone","mobile","number","contact"])
                    _want_city    = any(w in low_q for w in ["city","address","kaha","kahan","location"]) or any(w in _llm_items_str for w in ["city","address"])
                    _want_details = any(w in low_q for w in ["detail","details","profile","hisab","account","info"]) or any(w in _llm_items_str for w in ["details","profile"])

                    detail_msg = None
                    if _want_gst and _want_mobile:
                        detail_msg = (f"📞 **{sup_name}** contact: **{getattr(s,'mobile','N/A') or 'N/A'}**  \n"
                                      f"🏢 GST: **{getattr(s,'gstin','N/A') or 'N/A'}**")
                    elif _want_email and _want_mobile:
                        mobile = getattr(s,'mobile','N/A') or 'N/A'
                        email = getattr(s,'email','N/A') or 'N/A'
                        detail_msg = (f"📞 **{sup_name}** contact: **{mobile}**  \n"
                                      f"📧 Email: **{email}**")
                    elif _want_gst:
                        detail_msg = f"🏢 **{sup_name}** ka GST: **{getattr(s,'gstin','N/A') or 'N/A'}**"
                    elif _want_email:
                        detail_msg = f"📧 **{sup_name}** email: **{getattr(s,'email','N/A') or 'N/A'}**"
                    elif _want_mobile:
                        detail_msg = f"📞 **{sup_name}** ka number: **{getattr(s,'mobile','N/A') or 'N/A'}**"
                    elif _want_city and getattr(s,"city",""):
                        detail_msg = f"📍 **{sup_name}** — **{getattr(s,'city','')}** mein hain."

                    if detail_msg:
                        final_results.append({"type":"chat","message": detail_msg,"db_checked":True})
                    else:
                        if "po_search" not in intents:
                            final_results.append({"type":"chat","message":f"ye raha 👍 **{sup_name}** ka profile:"})
                        inv_items = db.execute(text(
                            "SELECT i.id, i.name, SUM(CASE WHEN LOWER(t.txn_type)='in' THEN t.quantity ELSE -t.quantity END) AS stock "
                            "FROM inventories i JOIN stock_transactions t ON i.id=t.inventory_id "
                            "WHERE t.supplier_id=:sid GROUP BY i.id,i.name HAVING stock!=0"
                        ), {"sid": s.id}).fetchall()
                        final_results.append({
                            "type": "result",
                            "supplier": {
                                "id":     s.id,
                                "name":   sup_name,
                                "code":   str(getattr(s,"supplier_code","N/A") or "N/A"),
                                "mobile": str(getattr(s,"mobile","N/A")        or "N/A"),
                                "city":   str(getattr(s,"city","N/A")          or "N/A"),
                                "email":  str(getattr(s,"email","N/A")         or "N/A"),
                                "gstin":  str(getattr(s,"gstin","N/A")         or "N/A"),
                            },
                            "items": [{"name": r.name, "stock": float(r.stock)} for r in inv_items],
                        })
                        # Emit context_entity so follow-ups can use targeted queries
                        po_id_rows = db.execute(text(
                            "SELECT id FROM purchase_orders WHERE supplier_id=:sid LIMIT 100"
                        ), {"sid": s.id}).fetchall()
                        _ce_out = {
                            "type": "supplier",
                            "id": s.id,
                            "name": sup_name,
                            "po_ids": [r.id for r in po_id_rows],
                            "inventory_ids": [r.id for r in inv_items],
                        }
            except Exception as e:
                final_results.append({"type":"chat","message":f"Supplier error: {e}"})

        # ── PURCHASE ORDERS ───────────────────────────────────────────────────
        elif intent == "po_search":
            try:
                # Boss Mode 1: Highest pending balance supplier
                if any(w in low_q for w in ["sabse jada balance","highest balance","paisa baaki",
                                             "sabse jyada balance","maximum balance"]):
                    r = db.execute(text(
                        "SELECT s.supplier_name,SUM(p.balance_amount) AS bal,COUNT(p.id) AS cnt,s.mobile "
                        "FROM purchase_orders p JOIN suppliers s ON p.supplier_id=s.id "
                        "WHERE p.balance_amount>0 AND LOWER(p.status)!='completed' "
                        "GROUP BY s.id,s.supplier_name ORDER BY bal DESC LIMIT 1"
                    )).fetchone()
                    if r:
                        final_results.append({"type":"chat","message":
                            f"💸 **Payment Alert:** Sabse zyada pending — **{r.supplier_name}**\n\n"
                            f"💰 Pending: **₹{float(r.bal):,.2f}** | 📄 Orders: **{r.cnt}** | 📞 {r.mobile}"})
                    continue

                # Boss Mode 2: Lowest pending balance supplier
                if any(w in low_q for w in ["sabse kam balance","lowest balance","minimum balance"]):
                    r = db.execute(text(
                        "SELECT s.supplier_name,SUM(p.balance_amount) AS bal,COUNT(p.id) AS cnt,s.mobile "
                        "FROM purchase_orders p JOIN suppliers s ON p.supplier_id=s.id "
                        "WHERE p.balance_amount>0 AND LOWER(p.status)!='completed' "
                        "GROUP BY s.id,s.supplier_name ORDER BY bal ASC LIMIT 1"
                    )).fetchone()
                    if r:
                        final_results.append({"type":"chat","message":
                            f"💸 **Payment Alert:** Sabse kam pending — **{r.supplier_name}**\n\n"
                            f"💰 Pending: **₹{float(r.bal):,.2f}** | 📄 Orders: **{r.cnt}** | 📞 {r.mobile}"})
                    continue

                # Boss Mode 3: Highest PO amount
                if any(w in low_q for w in ["highest po","sabse bada po","biggest order","sabse bada order"]):
                    r = db.execute(text(
                        "SELECT p.*,s.supplier_name FROM purchase_orders p "
                        "JOIN suppliers s ON p.supplier_id=s.id ORDER BY p.total_amount DESC LIMIT 1"
                    )).fetchone()
                    if r:
                        final_results.append({"type":"chat","message":
                            f"🏆 Sabse bada PO — **{r.supplier_name}**"})
                        final_results.append({
                            "type":"po","po_id":int(r.id),"po_no":str(r.po_number),"supplier":str(r.supplier_name),
                            "date":str(r.po_date),"total":float(r.total_amount or 0),
                            "advance":float(r.advance_amount or 0),"balance":float(r.balance_amount or 0),
                            "status":str(r.status).capitalize()})
                    continue

                # Boss Mode 4: Lowest PO amount
                if any(w in low_q for w in ["lowest po","sabse chota po","sabse chhota po","smallest order","smallest po"]):
                    r = db.execute(text(
                        "SELECT p.*,s.supplier_name FROM purchase_orders p "
                        "JOIN suppliers s ON p.supplier_id=s.id ORDER BY p.total_amount ASC LIMIT 1"
                    )).fetchone()
                    if r:
                        final_results.append({"type":"chat","message":
                            f"📉 Sabse chhota PO — **{r.supplier_name}**"})
                        final_results.append({
                            "type":"po","po_id":int(r.id),"po_no":str(r.po_number),"supplier":str(r.supplier_name),
                            "date":str(r.po_date),"total":float(r.total_amount or 0),
                            "advance":float(r.advance_amount or 0),"balance":float(r.balance_amount or 0),
                            "status":str(r.status).capitalize()})
                    continue

                # Boss Mode 5: GST / Tax analytics
                if any(w in low_q for w in ["tax","gst","cgst","sgst"]):
                    tax_q      = "SELECT SUM(tax_amount) AS tt, COUNT(id) AS cnt FROM purchase_orders p WHERE 1=1"
                    tax_params : dict = {}
                    _gst_noise = {"gst","tax","cgst","sgst","total","kitna","bana","hai","calculate",
                                  "how","much","kitni","lagta","ka","ki"}
                    sup_part   = " ".join(w for w in target.lower().split()
                                         if w not in _gst_noise and len(w) > 2)
                    if filters.get("from_date") and filters.get("to_date"):
                        tax_q += " AND p.po_date BETWEEN :s AND :e"
                        tax_params.update({"s": filters["from_date"], "e": filters["to_date"]})
                    if sup_part:
                        tax_q = ("SELECT SUM(p.tax_amount) AS tt, COUNT(p.id) AS cnt "
                                 "FROM purchase_orders p JOIN suppliers s ON p.supplier_id=s.id WHERE 1=1")
                        words = [w for w in sup_part.split() if len(w) > 1]
                        if words:
                            conds = " AND ".join(f"LOWER(s.supplier_name) LIKE :s{i}" for i in range(len(words)))
                            tax_q += f" AND ({conds})"
                            for i, w in enumerate(words):
                                tax_params[f"s{i}"] = f"%{w}%"
                    r = db.execute(text(tax_q), tax_params).fetchone()
                    if r and r.tt:
                        label = f"**{target.title()}** ke " if target else "In "
                        final_results.append({"type":"chat","message":
                            f"🧾 **Tax (GST) Report:**\n\n{label}**{r.cnt} orders** — total tax: **₹{float(r.tt):,.2f}**"})
                    else:
                        final_results.append({"type":"chat","message":"Is filter par koi tax data nahi mila. 🧐"})
                    continue

                # 📦 Items Ordered from a Supplier (button: "items ordered")
                if "items ordered" in low_q and target:
                    rows = db.execute(text("""
                        SELECT i.name, i.unit, SUM(poi.ordered_qty) AS total_ordered
                        FROM purchase_order_items poi
                        JOIN inventories i        ON poi.inventory_id       = i.id
                        JOIN purchase_orders po   ON poi.purchase_order_id  = po.id
                        JOIN suppliers s          ON po.supplier_id         = s.id
                        WHERE LOWER(s.supplier_name) LIKE :name
                        GROUP BY i.id, i.name, i.unit
                        ORDER BY total_ordered DESC
                        LIMIT 25
                    """), {"name": f"%{target.lower()}%"}).fetchall()
                    if rows:
                        final_results.append({"type": "chat", "message": f"📦 **{target}** se ye items order kiye gaye hain:"})
                        final_results.append({
                            "type": "supplier_items",
                            "rows": [{"name": r.name, "total_amount": float(r.total_ordered), "status": str(r.unit or "")} for r in rows]
                        })
                    else:
                        final_results.append({"type": "chat", "message": f"**{target}** se koi items order nahi mile. 🧐"})
                    continue

                # 🧾 Payment History for a Supplier (button: "payment history")
                if "payment history" in low_q and target:
                    rows = db.execute(text("""
                        SELECT pt.pay_amount, pt.transaction_date, po.po_number
                        FROM po_transactions pt
                        JOIN purchase_orders po ON pt.po_id       = po.id
                        JOIN suppliers s        ON po.supplier_id = s.id
                        WHERE LOWER(s.supplier_name) LIKE :name
                        ORDER BY pt.transaction_date DESC
                        LIMIT 25
                    """), {"name": f"%{target.lower()}%"}).fetchall()
                    if rows:
                        total_paid = sum(float(r.pay_amount) for r in rows)
                        final_results.append({"type": "chat", "message": f"🧾 **{target}** ko total **₹{total_paid:,.2f}** bhugtan kiya gaya:"})
                        final_results.append({
                            "type": "supplier_payments",
                            "rows": [{"po_number": r.po_number, "total_amount": float(r.pay_amount), "status": str(r.transaction_date)[:10]} for r in rows]
                        })
                    else:
                        final_results.append({"type": "chat", "message": f"**{target}** ka koi payment history nahi mila. 🧐"})
                    continue

                # ── Normal PO search ──────────────────────────────────────────
                valid_statuses = {"draft","completed","pending","in progress","cancelled","approved"}
                status_f = str(filters.get("status") or "").lower().strip()
                if status_f not in valid_statuses:
                    status_f = ""

                # Advance total aggregation — "advance diya hua kitna hai"
                if re.search(r"advance\s+(?:diya\s+)?(?:hua\s+)?kitna|total\s+advance|advance\s+total", low_q):
                    row = db.execute(text(
                        "SELECT COALESCE(SUM(advance_amount),0) AS total, COUNT(*) AS cnt "
                        "FROM purchase_orders WHERE advance_amount > 0"
                    )).fetchone()
                    final_results.append({"type":"chat","message":
                        f"💰 Total advance paid: **₹{float(row.total):,.2f}** across **{row.cnt} orders**."})
                    continue

                _pending_draft = False
                if any(w in low_q for w in ["pending","draft","kacha"]):
                    _pending_draft = True  # use IN ('draft','pending') to cover both DB conventions

                # "last/latest/advance/detail" queries ignore status filter
                if not any(w in low_q for w in ["pending","draft","kacha","completed"]):
                    if any(w in low_q for w in ["last","latest","detail","details","profile","advance","prepaid"]):
                        status_f = ""
                        _pending_draft = False

                # Widen limit for "all" / "saare" / "pending" queries
                if any(w in low_q for w in ["all","saare","sabhi","pure","poore","sab","pending","draft","batao"]):
                    limit = 50

                last_n = re.search(r"\b(?:last|latest)\s+(\d+)\b", low_q)
                if last_n:
                    limit = int(last_n.group(1))
                elif any(w in low_q for w in ["last","latest"]):
                    limit = 1

                q       = "SELECT p.*,s.supplier_name FROM purchase_orders p JOIN suppliers s ON p.supplier_id=s.id WHERE 1=1"
                params  = {"l": limit}

                if _pending_draft:
                    q += " AND LOWER(p.status) IN ('draft','pending')"
                elif status_f:
                    q += " AND LOWER(p.status)=:pst"; params["pst"] = status_f

                # Amount threshold filters ("50000 se zyada", "1 lakh se kam")
                _tab_m = re.search(
                    r"(\d[\d,]*)(?:\s*lakh)?\s*se\s+(?:zyada|jyada|upar|bada|above)"
                    r"|(?:above|more\s*than|over)\s+(?:rs\.?\s*)?(\d[\d,]*)(?:\s*lakh)?",
                    low_q)
                if _tab_m:
                    _tv = float((_tab_m.group(1) or _tab_m.group(2) or "0").replace(",",""))
                    if re.search(r"\blakh\b|\blac\b", low_q): _tv *= 100000
                    q += " AND p.total_amount > :tamin"; params["tamin"] = _tv

                _tab_m2 = re.search(
                    r"(\d[\d,]*)(?:\s*lakh)?\s*se\s+(?:kam|chhota|neeche|below)"
                    r"|(?:below|less\s*than|under)\s+(?:rs\.?\s*)?(\d[\d,]*)(?:\s*lakh)?",
                    low_q)
                if _tab_m2:
                    _tv2 = float((_tab_m2.group(1) or _tab_m2.group(2) or "0").replace(",",""))
                    if re.search(r"\blakh\b|\blac\b", low_q): _tv2 *= 100000
                    q += " AND p.total_amount < :tamax"; params["tamax"] = _tv2

                if any(w in low_q for w in ["advance","prepaid","advance diya"]):
                    q += " AND p.advance_amount>0"
                    if not any(w in low_q for w in ["last","latest"]):
                        limit = 50; params["l"] = 50

                # Date range — use LLM filters first, fall back to keyword parsing
                _fd = filters.get("from_date")
                _td = filters.get("to_date")
                if not _fd:
                    _fd, _td = _date_range_from_query(low_q)
                if _fd and _td:
                    q += " AND p.po_date BETWEEN :fd AND :td"
                    params["fd"] = _fd; params["td"] = _td
                elif _fd:
                    q += " AND p.po_date >= :fd"
                    params["fd"] = _fd

                # Explicit PO number pattern
                po_pat = re.search(r"[A-Za-z]{2,}[/\-][A-Za-z0-9]+[/\-][A-Za-z0-9\-/]+", raw_q)
                if po_pat:
                    q += " AND LOWER(p.po_number) LIKE :pn"; params["pn"] = f"%{po_pat.group(0).lower()}%"
                else:
                    _skip = (any(w in low_q for w in ["advance","prepaid"]) and
                             clean_noise(target, "po").strip().lower() in ("","advance","prepaid"))
                    t_clean = "" if _skip else clean_noise(target, "po")
                    if t_clean and len(t_clean) >= 3:
                        words = [w for w in t_clean.split() if len(w) > 2]
                        if words:
                            conds = " AND ".join(
                                f"(LOWER(s.supplier_name) LIKE :s{i} OR LOWER(p.po_number) LIKE :s{i})"
                                for i in range(len(words))
                            )
                            q += f" AND ({conds})"
                            for i, w in enumerate(words):
                                params[f"s{i}"] = f"%{w}%"

                pos = db.execute(text(q + " ORDER BY p.po_date DESC,p.id DESC LIMIT :l"), params).fetchall()

                if not pos:
                    final_results.append({"type":"chat","message":"In filters par koi orders nahi mile. 🧐"})
                else:
                    _bal_q       = any(w in low_q for w in ["balance","baaki","baki","rokra","paisa"])
                    total_bal    = sum(float(p.balance_amount or 0) for p in pos)
                    pend_bal     = sum(float(p.balance_amount or 0) for p in pos if str(p.status).lower() != "completed")
                    msg = f"📄 **{len(pos)} orders** mile."
                    if _bal_q and total_bal > 0:
                        msg += f" Total balance: **₹{total_bal:,.2f}**"
                        if pend_bal > 0 and pend_bal != total_bal:
                            msg += f" (pending: **₹{pend_bal:,.2f}**)"
                    elif pend_bal > 0:
                        msg += f" Pending balance: **₹{pend_bal:,.2f}**"
                    final_results.append({"type":"chat","message": msg})

                    _wants_payments = any(w in low_q for w in
                        ["payment","history","payment history","transactions","paid","kitna diya",
                         "paisa diya","advance kab","kab diya","kitne diye"])

                    for p in pos:
                        final_results.append({
                            "type":    "po",
                            "po_id":   int(p.id),
                            "po_no":   str(p.po_number),
                            "supplier":str(p.supplier_name),
                            "date":    str(p.po_date),
                            "total":   float(p.total_amount   or 0),
                            "advance": float(p.advance_amount or 0),
                            "balance": float(p.balance_amount or 0),
                            "status":  str(p.status).capitalize(),
                        })
                        # Inline payment history when explicitly asked
                        if _wants_payments:
                            txn_rows = db.execute(text(
                                "SELECT pay_amount, transaction_date FROM po_transactions "
                                "WHERE po_id=:pid ORDER BY transaction_date ASC"
                            ), {"pid": p.id}).fetchall()
                            if txn_rows:
                                lines = "\n".join(
                                    f"• ₹{float(r.pay_amount or 0):,.2f} — {r.transaction_date}"
                                    for r in txn_rows)
                                total_paid = sum(float(r.pay_amount or 0) for r in txn_rows)
                                final_results.append({"type":"chat","message":
                                    f"**{p.po_number}** ke **{len(txn_rows)} payments** hain "
                                    f"(total paid: **₹{total_paid:,.2f}**):\n\n{lines}"})
                            else:
                                final_results.append({"type":"chat","message":
                                    f"**{p.po_number}** ke liye abhi tak **koi payment nahi** hui hai. "
                                    f"Pending balance: **₹{float(p.balance_amount or 0):,.2f}**"})

                    # Emit context_entity for single PO so follow-ups can ask about items/supplier
                    if len(pos) == 1:
                        p0 = pos[0]
                        _inv_ids = [r.inventory_id for r in db.execute(text(
                            "SELECT DISTINCT inventory_id FROM purchase_order_items "
                            "WHERE purchase_order_id=:pid AND inventory_id IS NOT NULL LIMIT 50"
                        ), {"pid": p0.id}).fetchall()]
                        _ce_out = {
                            "type":          "purchase_order",
                            "id":            p0.id,
                            "po_no":         str(p0.po_number),
                            "supplier_id":   getattr(p0, "supplier_id", None),
                            "supplier_name": str(p0.supplier_name),
                            "inventory_ids": _inv_ids,
                        }
            except Exception as e:
                final_results.append({"type":"chat","message":f"PO error: {e}"})

        # ── INVENTORY SEARCH ──────────────────────────────────────────────────
        elif intent == "search":
            try:
                specific   = ai.get("specific_items") or []
                raw_target = str(ai.get("search_target") or "").lower().strip()

                # "saare items / list all" — no specific item asked
                _list_all_kw = ["saare", "sabhi", "all items", "list karo", "list all",
                                 "sab items", "poore items", "sab maal"]
                if not raw_target and any(w in low_q for w in _list_all_kw):
                    final_results.append({"type": "chat", "message":
                        "Bhai, ek specific item ka naam batao ya category likhein — "
                        "jaise 'bearing', 'oil seal', 'belt', etc. 🙂\n\n"
                        "Poori list bahut badi hai, isliye narrow down karna padega."})
                    continue

                raw_target = raw_target or low_q
                targets    = []

                # Multi-item: use specific_items list
                if isinstance(specific, list) and len(specific) > 1:
                    for item in specific:
                        cleaned = clean_noise(str(item or "").lower(), "inventory")
                        if cleaned and len(cleaned) > 1:
                            corrected = faiss_match(cleaned, "inventory") or cleaned
                            targets.append(corrected)

                if not targets:
                    cleaned  = clean_noise(raw_target, "inventory") or re.sub(r"[?'\"!.,]","",raw_target).strip()
                    if len(cleaned) > 1:
                        corrected = faiss_match(cleaned, "inventory") or cleaned
                        targets.append(corrected)

                # Emergency fallback for common words
                if not targets:
                    for keyword in ("bearing","belt","oil","seal","bolt"):
                        if keyword in low_q:
                            targets.append(keyword)
                            break

                all_inv_lower = [r.name.lower() for r in
                                 db.execute(text("SELECT name FROM inventories")).fetchall() if r.name]
                found_any = False
                q_str     = "SELECT id,name,model,type,classification,placement FROM inventories WHERE (LOWER(name) LIKE :q OR LOWER(model) LIKE :q)"

                for t in targets:
                    items = db.execute(text(q_str + " LIMIT 30"), {"q": f"%{t}%"}).fetchall()

                    if not items:
                        close = difflib.get_close_matches(t, all_inv_lower, n=1, cutoff=0.65)
                        if close:
                            items = db.execute(text(q_str + " LIMIT 30"), {"q": f"%{close[0]}%"}).fetchall()
                            t     = close[0]

                    if not items:
                        continue
                    found_any = True

                    ids        = tuple(i.id for i in items)
                    date_cond  = ""
                    date_params: dict = {}
                    if filters.get("from_date") and filters.get("to_date"):
                        date_cond   = " AND txn_date BETWEEN :fd AND :td"
                        date_params = {"fd": filters["from_date"], "td": filters["to_date"]}
                    elif filters.get("from_date"):
                        date_cond   = " AND txn_date >= :fd"
                        date_params = {"fd": filters["from_date"]}

                    if len(items) > 1:
                        total = db.execute(text(
                            f"SELECT SUM(CASE WHEN LOWER(txn_type)='in' THEN quantity ELSE -quantity END) "
                            f"FROM stock_transactions WHERE inventory_id IN :ids{date_cond}"
                        ), {"ids": ids, **date_params}).scalar() or 0
                        date_lbl = (f" ({filters['from_date']} → {filters['to_date']})"
                                    if date_cond else "")
                        final_results.append({"type":"chat","message":
                            f"haan mil gaya 👍 **Total {t.title()} stock{date_lbl}:** {float(total):.2f} units"})
                        final_results.append({
                            "type":"dropdown","message":"Select item for details:",
                            "items": [{"id": i.id, "name": f"{i.name} {i.model or ''}".strip()} for i in items],
                        })
                    else:
                        i     = items[0]
                        stock = float(db.execute(text(
                            f"SELECT SUM(CASE WHEN LOWER(txn_type)='in' THEN quantity ELSE -quantity END) "
                            f"FROM stock_transactions WHERE inventory_id=:id{date_cond}"
                        ), {"id": i.id, **date_params}).scalar() or 0)
                        cls   = str(i.classification or "").upper()
                        f_s   = stock if cls in ("FINISH","") else 0
                        sf_s  = stock if "SEMI" in cls       else 0
                        m_s   = stock if "MACH" in cls       else 0
                        final_results.append({"type":"chat","message":f"ye raha 👍 **{i.name}** ka data:"})
                        final_results.append({
                            "type": "result",
                            "inventory": {
                                "id":        i.id,
                                "name":      f"{i.name} {i.model or ''}".strip(),
                                "category":  i.type or "Raw Material",
                                "placement": i.placement or "Main Store",
                            },
                            "total_stock":       stock,
                            "finish_stock":      f_s,
                            "semi_finish_stock": sf_s,
                            "machining_stock":   m_s,
                        })
                        # Emit context_entity for follow-up PO/supplier queries on this item
                        _ce_out = {"type": "inventory", "id": i.id, "name": i.name}

                if not found_any:
                    final_results.append({"type":"chat","message":
                        "Ye item mere system mein nahi mila. 🧐 Spelling check karoge?"})
            except Exception as e:
                final_results.append({"type":"chat","message":f"Inventory error: {e}"})

        # ── GENERAL CHAT ──────────────────────────────────────────────────────
        elif intent == "general_chat":
            final_results.append({"type":"chat","message":
                "Main ERP assistant hoon — inventory, suppliers, purchase orders aur "
                "projects ke baare mein help kar sakta hoon. 😊\n\n"
                "Kya poochhna hai?"})

        # ── PURCHASE REQUESTS ─────────────────────────────────────────────────
        elif intent in ("purchase_request", "pr_search"):
            try:
                pr_match = re.search(r"\bpr[-\s]?(\d+)\b", low_q)
                if pr_match:
                    pr_no = f"PR-{pr_match.group(1)}"
                    rows = db.execute(text(
                        "SELECT * FROM purchase_requests WHERE LOWER(pr_no)=:pr LIMIT 10"
                    ), {"pr": pr_no.lower()}).fetchall()
                elif any(w in low_q for w in ["kitne","count","total","how many"]):
                    cnt = db.execute(text("SELECT COUNT(*) FROM purchase_requests")).scalar() or 0
                    final_results.append({"type":"chat","message":
                        f"📋 Total **{cnt} purchase requests** hain."})
                    continue
                else:
                    rows = db.execute(text(
                        "SELECT * FROM purchase_requests ORDER BY request_date DESC,id DESC LIMIT 10"
                    )).fetchall()
                if not rows:
                    final_results.append({"type":"chat","message":"Koi purchase request nahi mili. 🧐"})
                else:
                    final_results.append({"type":"chat","message":f"📋 **{len(rows)} purchase requests** mile:"})
                    for r in rows:
                        final_results.append({
                            "type":      "purchase_request",
                            "pr_no":     getattr(r, "pr_no", "N/A"),
                            "date":      str(getattr(r, "request_date", "N/A")),
                            "status":    getattr(r, "status", "N/A"),
                            "priority":  getattr(r, "priority", "N/A"),
                            "total_qty": float(getattr(r, "total_qty", 0) or 0),
                        })
            except Exception as e:
                final_results.append({"type":"chat","message":f"PR error: {e}"})

    # ── Final return ──────────────────────────────────────────────────────────
    _resp = {"results": final_results}
    if _ce_out:
        _resp["context_entity"] = _ce_out

    # If we got real data (POs, suppliers, projects, inventory cards) — done.
    if len(final_results) > 1 and not _is_zero_result(_resp):
        return _resp

    # ── Text-to-SQL fallback (FK traversal queries the handlers didn't cover) ─
    # Triggers when: handlers produced no real data (only "not found" chat msgs)
    # AND the query is substantial (not a greeting / general chat).
    _SKIP_SQL_INTENTS = {"general_chat", "purchase_request", "pr_search"}
    _sql_worthy = (
        bool(intents) and
        not all(i in _SKIP_SQL_INTENTS for i in intents) and
        len(raw_q.split()) >= 3 and
        not _GREETING_RE.match(raw_q.strip())
    )
    if _sql_worthy:
        try:
            from app.services.schema_doc import get_schema_text
            from app.services.complex_query import handle_fk_query
            _schema   = get_schema_text()
            _ce_in    = _last_context_entity(history)
            _fk_limit = max(1, int((filters or {}).get("limit") or 15))
            _fk_result = handle_fk_query(
                raw_q, _schema, db,
                context_entity=_ce_in,
                limit=_fk_limit,
            )
            if _fk_result:
                print(f"[FK-FALLBACK] answered by text-to-SQL: '{raw_q}'")
                return _fk_result
        except Exception as exc:
            print(f"[FK-FALLBACK] skipped: {exc}")

    # Return whatever the handlers produced (even if zero-result)
    if len(final_results) > 1:
        return _resp

    # Fallback suggestions
    is_eng = any(w in low_q for w in ["what","how","show","list","get","who","where","tell","check"])
    if is_eng:
        if "project" in low_q or "site" in low_q:
            msg = "Looking for **Project** info? Please provide the project name."
        elif any(w in low_q for w in ["money","balance","account","due"]):
            msg = "Want to check a **Supplier Balance**? Type the party name."
        else:
            msg = ("I couldn't understand that. 😅\n\n"
                   "You can ask about:\n"
                   "1. **Stock** — e.g. 'bearing stock'\n"
                   "2. **Orders** — e.g. 'latest PO'\n"
                   "3. **Suppliers** — e.g. 'DCL details'\n"
                   "4. **Projects** — e.g. 'running projects'")
    else:
        if "project" in low_q or "site" in low_q:
            msg = "Lagta hai aap kisi **Project** ki baat kar rahe hain. Project ka naam batao. 🏗️"
        elif any(w in low_q for w in ["paisa","balance","hisab","rokra"]):
            msg = "Kisi Supplier ka **Balance** check karna hai? Party ka naam likhein. 💰"
        else:
            msg = ("Maaf kijiye, samajh nahi paaya. 😅\n\n"
                   "Aap poochh sakte hain:\n"
                   "1. **Stock** — jaise 'bearing kitna hai'\n"
                   "2. **Orders** — jaise 'latest PO dikhao'\n"
                   "3. **Supplier** — jaise 'DCL ka profile'\n"
                   "4. **Projects** — jaise 'running projects'")

    _log(raw_q, intents, msg)
    return {"results": [{"type": "chat", "message": msg}]}


@router.post("/query")
def v2_chatbot_query(request: ChatRequest, db: Session = Depends(get_db)):
    return v2_chatbot(request, db)


# ─── Health / Status ──────────────────────────────────────────────────────────
@router.get("/status")
def v2_status():
    """Check if Ollama server + qwen2.5:7b model are ready."""
    return health_check()


# ─── Live LLM probe + classifier (tells you which providers can take traffic) ─
@router.get("/llm-status")
def v2_llm_status(probe: bool = False):
    """
    GET /v2-chatbot/llm-status         — fast: returns lazy state from real requests
    GET /v2-chatbot/llm-status?probe=1 — slow: actively pings each provider (~5–15s)

    Use ?probe=1 for an explicit "are my LLMs alive right now" check (e.g. after
    topping up OpenRouter or refilling Groq tokens). Use the no-arg form on a
    monitoring dashboard — it's instant and burns no quota.
    """
    if probe:
        return {"mode": "active", "providers": probe_providers()}
    return {"mode": "lazy", **health_check()}


@router.get("/route-preview")
def v2_route_preview(query: str):
    """Show which chain a given query would use without actually calling any LLM."""
    from app.services.v2_ollama_engine import _pick_chain
    return {
        "query":   query,
        "complex": is_complex_query(query),
        "chain":   _pick_chain(query),
    }


# ─── Reload Index ────────────────────────────────────────────────────────────
@router.post("/reload-index")
def v2_reload(db: Session = Depends(get_db)):
    """Stub — FAISS disabled. LLM routing handles all entity matching."""
    return {"status": "ok", "note": "FAISS disabled (Python 3.13). LLM routing active."}


# ─── Feedback ────────────────────────────────────────────────────────────────


class FeedbackPayload(BaseModel):
    request_id: str = Field(..., min_length=1, max_length=64)
    rating: Literal[-1, 1]  # 1 = thumbs up, -1 = thumbs down
    comment: Optional[str] = Field(None, max_length=500)
    query: Optional[str] = Field(None, max_length=500)
    response_summary: Optional[str] = Field(None, max_length=1000)


@router.post("/feedback")
def v2_feedback(payload: FeedbackPayload, db: Session = Depends(get_local_db)):
    """Capture 👍/👎 from the UI. The single highest-signal correctness loop you have."""
    try:
        db.execute(
            text(
                "INSERT INTO chatbot_feedback (request_id, rating, query, response_summary, comment) "
                "VALUES (:rid, :r, :q, :rs, :c)"
            ),
            {
                "rid": payload.request_id,
                "r": payload.rating,
                "q": (payload.query or "")[:500],
                "rs": (payload.response_summary or "")[:1000],
                "c": (payload.comment or "")[:500],
            },
        )
        db.commit()
        return {"status": "ok"}
    except Exception as e:
        db.rollback()
        # Don't 500 — feedback failure shouldn't break the UX. Log + return ack.
        print(f"[FEEDBACK] insert failed: {e}")
        return {"status": "logged_locally", "error": str(e)[:200]}


# ─── Alias admin (curated alias -> canonical name overrides) ─────────────────
class AliasPayload(BaseModel):
    alias: str = Field(..., min_length=1, max_length=255)
    canonical_name: str = Field(..., min_length=1, max_length=255)
    category: Literal["supplier", "project", "inventory"]


@router.post("/alias")
def v2_add_alias(payload: AliasPayload, db: Session = Depends(get_local_db)):
    """Add or update an alias. Beats fuzzy matching — exact, curated, instant fix."""
    try:
        db.execute(
            text(
                "INSERT INTO entity_aliases (alias, canonical_name, category) "
                "VALUES (:a, :c, :cat) "
                "ON DUPLICATE KEY UPDATE canonical_name=VALUES(canonical_name)"
            ),
            {"a": payload.alias.strip(), "c": payload.canonical_name.strip(), "cat": payload.category},
        )
        db.commit()
        invalidate_resolver(payload.category)
        return {"status": "ok"}
    except Exception as e:
        db.rollback()
        return {"status": "error", "error": str(e)[:200]}


@router.get("/aliases")
def v2_list_aliases(db: Session = Depends(get_local_db), category: Optional[str] = None):
    """List curated aliases — for an admin UI."""
    try:
        if category:
            rows = db.execute(
                text("SELECT id, alias, canonical_name, category, created_at FROM entity_aliases WHERE category=:c ORDER BY created_at DESC"),
                {"c": category},
            ).fetchall()
        else:
            rows = db.execute(
                text("SELECT id, alias, canonical_name, category, created_at FROM entity_aliases ORDER BY created_at DESC")
            ).fetchall()
        return {"aliases": [dict(r._mapping) for r in rows]}
    except Exception as e:
        return {"aliases": [], "error": str(e)[:200]}


# ─── Alias suggestions (mine zero-results, cluster, propose canonicals) ─────
class AliasBulkPayload(BaseModel):
    aliases:        List[str] = Field(..., min_length=1, max_length=100)
    canonical_name: str       = Field(..., min_length=1, max_length=255)
    category:       Literal["supplier", "project", "inventory"]


class AliasSkipPayload(BaseModel):
    aliases: List[str] = Field(..., min_length=1, max_length=100)
    reason:  str       = Field("other", max_length=32)


@router.get("/alias-suggestions")
def v2_alias_suggestions(db: Session = Depends(get_db), days: int = 7, force: bool = False):
    """
    Returns clusters of similar zero-result queries grouped by likely canonical entity.
    UI flow: admin reviews each cluster, picks a canonical (or edits), then POSTs
    to /alias-bulk to accept. Sort order is by hit_count then confidence.
    """
    from app.services.alias_suggester import get_suggestions
    return get_suggestions(db, days=max(1, min(days, 90)), force=force)


@router.post("/alias-bulk")
def v2_alias_bulk(payload: AliasBulkPayload, db: Session = Depends(get_local_db)):
    """One-click accept — INSERTs all aliases in the cluster mapping to one canonical."""
    inserted = 0
    failed   = []
    for a in payload.aliases:
        try:
            db.execute(
                text(
                    "INSERT INTO entity_aliases (alias, canonical_name, category) "
                    "VALUES (:a, :c, :cat) "
                    "ON DUPLICATE KEY UPDATE canonical_name=VALUES(canonical_name)"
                ),
                {"a": a.strip(), "c": payload.canonical_name.strip(), "cat": payload.category},
            )
            inserted += 1
        except Exception as e:
            failed.append({"alias": a, "error": str(e)[:120]})
    db.commit()
    invalidate_resolver(payload.category)
    from app.services.alias_suggester import invalidate_cache
    invalidate_cache()
    return {"status": "ok", "inserted": inserted, "failed": failed}


@router.post("/alias-skip")
def v2_alias_skip(payload: AliasSkipPayload, db: Session = Depends(get_local_db)):
    """Mark cluster as 'not an alias' so it stops appearing in suggestions."""
    skipped = 0
    for a in payload.aliases:
        try:
            db.execute(
                text(
                    "INSERT INTO alias_suggestions_skipped (alias, reason) "
                    "VALUES (:a, :r) "
                    "ON DUPLICATE KEY UPDATE reason=VALUES(reason), skipped_at=NOW()"
                ),
                {"a": a.strip(), "r": payload.reason},
            )
            skipped += 1
        except Exception as e:
            print(f"[ALIAS-SKIP] {a} → {e}")
    db.commit()
    from app.services.alias_suggester import invalidate_cache
    invalidate_cache()
    return {"status": "ok", "skipped": skipped}


# ─── Supplier detail endpoints (direct SQL, no LLM) ─────────────────────────

@router.get("/supplier/{supplier_id}/pos")
def supplier_pos(supplier_id: int, db: Session = Depends(get_db)):
    rows = db.execute(text("""
        SELECT id, po_number, po_date, total_amount, advance_amount, balance_amount,
               status, expected_delivery, delivery_status
        FROM purchase_orders
        WHERE supplier_id = :sid
        ORDER BY po_date DESC
        LIMIT 50
    """), {"sid": supplier_id}).fetchall()
    return {"rows": [
        {
            "id":              int(r.id),
            "po_number":       str(r.po_number),
            "date":            str(r.po_date or ""),
            "total":           float(r.total_amount   or 0),
            "advance":         float(r.advance_amount or 0),
            "balance":         float(r.balance_amount or 0),
            "status":          str(r.status            or ""),
            "expected":        str(r.expected_delivery or ""),
            "delivery_status": str(r.delivery_status   or ""),
        }
        for r in rows
    ]}


@router.get("/supplier/{supplier_id}/balance")
def supplier_balance(supplier_id: int, db: Session = Depends(get_db)):
    r = db.execute(text("""
        SELECT
            COUNT(*)                                                               AS total_pos,
            COALESCE(SUM(total_amount),   0)                                       AS total_ordered,
            COALESCE(SUM(advance_amount), 0)                                       AS total_advance,
            COALESCE(SUM(balance_amount), 0)                                       AS total_balance,
            COALESCE(SUM(CASE WHEN LOWER(status) != 'completed'
                              THEN balance_amount ELSE 0 END), 0)                  AS pending_balance,
            COUNT(CASE WHEN LOWER(status) = 'completed' THEN 1 END)               AS completed_pos,
            COUNT(CASE WHEN LOWER(status) != 'completed' THEN 1 END)              AS open_pos
        FROM purchase_orders
        WHERE supplier_id = :sid
    """), {"sid": supplier_id}).fetchone()
    return {
        "total_pos":       int(r.total_pos),
        "total_ordered":   float(r.total_ordered),
        "total_advance":   float(r.total_advance),
        "total_balance":   float(r.total_balance),
        "pending_balance": float(r.pending_balance),
        "completed_pos":   int(r.completed_pos),
        "open_pos":        int(r.open_pos),
    }


@router.get("/supplier/{supplier_id}/items")
def supplier_items(supplier_id: int, db: Session = Depends(get_db)):
    rows = db.execute(text("""
        SELECT i.name, i.unit,
               COALESCE(SUM(poi.ordered_qty),  0) AS total_ordered,
               COALESCE(SUM(poi.received_qty), 0) AS total_received
        FROM purchase_order_items poi
        JOIN inventories   i  ON poi.inventory_id       = i.id
        JOIN purchase_orders po ON poi.purchase_order_id = po.id
        WHERE po.supplier_id = :sid
        GROUP BY i.id, i.name, i.unit
        ORDER BY total_ordered DESC
        LIMIT 50
    """), {"sid": supplier_id}).fetchall()
    return {"rows": [
        {
            "name":     str(r.name),
            "unit":     str(r.unit     or ""),
            "ordered":  float(r.total_ordered),
            "received": float(r.total_received),
        }
        for r in rows
    ]}


@router.get("/supplier/{supplier_id}/payments")
def supplier_payments(supplier_id: int, db: Session = Depends(get_db)):
    rows = db.execute(text("""
        SELECT pt.pay_amount, pt.transaction_date, po.po_number
        FROM po_transactions pt
        JOIN purchase_orders po ON pt.po_id       = po.id
        WHERE po.supplier_id = :sid
        ORDER BY pt.transaction_date DESC
        LIMIT 50
    """), {"sid": supplier_id}).fetchall()
    total_paid = sum(float(r.pay_amount) for r in rows)
    return {
        "total_paid": total_paid,
        "rows": [
            {
                "po_number": str(r.po_number),
                "amount":    float(r.pay_amount),
                "date":      str(r.transaction_date)[:10],
            }
            for r in rows
        ],
    }


# ─── PO detail endpoints (direct SQL, no LLM) ───────────────────────────────

@router.get("/po/{po_id}/items")
def po_items(po_id: int, db: Session = Depends(get_db)):
    rows = db.execute(text("""
        SELECT i.name, i.unit, poi.hsn,
               poi.ordered_qty, poi.received_qty,
               poi.unit_price, poi.discount, poi.tax_percent,
               poi.tax_amount, poi.line_total
        FROM purchase_order_items poi
        JOIN inventories i ON poi.inventory_id = i.id
        WHERE poi.purchase_order_id = :pid
        ORDER BY poi.id
    """), {"pid": po_id}).fetchall()
    return {"rows": [
        {
            "name":        str(r.name),
            "unit":        str(r.unit        or ""),
            "hsn":         str(r.hsn         or ""),
            "ordered":     float(r.ordered_qty   or 0),
            "received":    float(r.received_qty  or 0),
            "unit_price":  float(r.unit_price    or 0),
            "discount":    float(r.discount      or 0),
            "tax_percent": float(r.tax_percent   or 0),
            "tax_amount":  float(r.tax_amount    or 0),
            "line_total":  float(r.line_total    or 0),
        }
        for r in rows
    ]}


@router.get("/po/{po_id}/payments")
def po_payments(po_id: int, db: Session = Depends(get_db)):
    rows = db.execute(text("""
        SELECT pay_amount, transaction_date
        FROM po_transactions
        WHERE po_id = :pid
        ORDER BY transaction_date DESC
    """), {"pid": po_id}).fetchall()
    total_paid = sum(float(r.pay_amount) for r in rows)
    return {
        "total_paid": total_paid,
        "rows": [
            {"amount": float(r.pay_amount), "date": str(r.transaction_date)[:10]}
            for r in rows
        ],
    }


@router.get("/po/{po_id}/status-log")
def po_status_log(po_id: int, db: Session = Depends(get_db)):
    rows = db.execute(text("""
        SELECT status, changed_at, remarks
        FROM po_status_logs
        WHERE purchase_order_id = :pid
        ORDER BY changed_at DESC
    """), {"pid": po_id}).fetchall()
    return {"rows": [
        {
            "status":  str(r.status     or ""),
            "date":    str(r.changed_at)[:16],
            "remarks": str(r.remarks    or ""),
        }
        for r in rows
    ]}


@router.get("/po/{po_id}/supplier")
def po_supplier(po_id: int, db: Session = Depends(get_db)):
    row = db.execute(text(
        "SELECT s.* FROM purchase_orders p "
        "JOIN suppliers s ON p.supplier_id = s.id "
        "WHERE p.id = :id LIMIT 1"
    ), {"id": po_id}).fetchone()
    if not row:
        return {"error": "Supplier not found for this PO"}
    return {
        "supplier": {
            "id":     int(row.id),
            "name":   str(row.supplier_name or ""),
            "code":   str(getattr(row, "supplier_code", "N/A") or "N/A"),
            "mobile": str(getattr(row, "mobile", "N/A") or "N/A"),
            "city":   str(getattr(row, "city", "N/A") or "N/A"),
            "email":  str(getattr(row, "email", "N/A") or "N/A"),
            "gstin":  str(getattr(row, "gstin", "N/A") or "N/A"),
        }
    }


# ─── Quick-search endpoints (used by the floating access buttons) ────────────

@router.get("/quick-search/supplier")
def quick_search_supplier(q: str = "", limit: int = 60, db: Session = Depends(get_db)):
    if not q.strip():
        rows = db.execute(text(
            "SELECT id, supplier_name, city, mobile FROM suppliers "
            "ORDER BY id DESC LIMIT :l"
        ), {"l": limit}).fetchall()
    else:
        words = q.strip().lower().split()
        cond = " AND ".join(
            f"(LOWER(supplier_name) LIKE :w{i} OR LOWER(COALESCE(city,'')) LIKE :w{i} "
            f"OR COALESCE(mobile,'') LIKE :w{i})"
            for i in range(len(words))
        )
        params = {f"w{i}": f"%{w}%" for i, w in enumerate(words)}
        params["l"] = limit
        rows = db.execute(text(
            f"SELECT id, supplier_name, city, mobile FROM suppliers "
            f"WHERE {cond} ORDER BY supplier_name LIMIT :l"
        ), params).fetchall()
    return {"rows": [
        {"id": int(r.id), "name": str(r.supplier_name or ""),
         "city": str(r.city or ""), "mobile": str(r.mobile or "")}
        for r in rows
    ]}


@router.get("/quick-search/po")
def quick_search_po(q: str = "", limit: int = 60, db: Session = Depends(get_db)):
    base = (
        "SELECT p.id, p.po_number, p.po_date, p.total_amount, p.status, s.supplier_name "
        "FROM purchase_orders p LEFT JOIN suppliers s ON p.supplier_id=s.id "
    )
    if not q.strip():
        rows = db.execute(text(base + "ORDER BY p.id DESC LIMIT :l"), {"l": limit}).fetchall()
    else:
        words = q.strip().lower().split()
        cond = " AND ".join(
            f"(LOWER(p.po_number) LIKE :w{i} OR LOWER(COALESCE(s.supplier_name,'')) LIKE :w{i} "
            f"OR LOWER(COALESCE(p.status,'')) LIKE :w{i})"
            for i in range(len(words))
        )
        params = {f"w{i}": f"%{w}%" for i, w in enumerate(words)}
        params["l"] = limit
        rows = db.execute(text(base + f"WHERE {cond} ORDER BY p.po_date DESC LIMIT :l"), params).fetchall()
    return {"rows": [
        {"id": int(r.id), "po_number": str(r.po_number or ""),
         "date": str(r.po_date or ""), "total": float(r.total_amount or 0),
         "status": str(r.status or ""), "supplier": str(r.supplier_name or "")}
        for r in rows
    ]}


@router.get("/quick-search/inventory")
def quick_search_inventory(q: str = "", limit: int = 60, db: Session = Depends(get_db)):
    base = (
        "SELECT i.id, i.name, i.unit, "
        "COALESCE(SUM(CASE WHEN LOWER(st.txn_type)='in' THEN st.quantity "
        "                  ELSE -st.quantity END), 0) AS stock "
        "FROM inventories i "
        "LEFT JOIN stock_transactions st ON st.inventory_id = i.id "
        "WHERE i.is_deleted = 0 "
    )
    if not q.strip():
        rows = db.execute(text(
            base + "GROUP BY i.id, i.name, i.unit ORDER BY i.id DESC LIMIT :l"
        ), {"l": limit}).fetchall()
    else:
        words = q.strip().lower().split()
        cond = " AND ".join(f"LOWER(i.name) LIKE :w{i}" for i in range(len(words)))
        params = {f"w{i}": f"%{w}%" for i, w in enumerate(words)}
        params["l"] = limit
        rows = db.execute(text(
            base + f"AND {cond} GROUP BY i.id, i.name, i.unit ORDER BY i.name LIMIT :l"
        ), params).fetchall()
    return {"rows": [
        {"id": int(r.id), "name": str(r.name or ""),
         "unit": str(r.unit or ""), "stock": float(r.stock or 0)}
        for r in rows
    ]}


# ─── PO card endpoint (direct DB, no LLM — used by quick-search and confirm picks)

@router.get("/po/{po_id}/card")
def po_card(po_id: int, db: Session = Depends(get_db)):
    row = db.execute(text(
        "SELECT p.*, s.supplier_name FROM purchase_orders p "
        "LEFT JOIN suppliers s ON p.supplier_id=s.id WHERE p.id=:id LIMIT 1"
    ), {"id": po_id}).fetchone()
    if not row:
        return {"results": [{"type": "chat", "message": "PO nahi mila."}]}
    return {"results": [{
        "type":     "chat",
        "message":  f"ye raha 👍 **{row.po_number}** ka detail:",
    }, {
        "type":     "po",
        "po_id":    int(row.id),
        "po_no":    str(row.po_number or ""),
        "supplier": str(row.supplier_name or ""),
        "date":     str(row.po_date or ""),
        "total":    float(row.total_amount   or 0),
        "advance":  float(row.advance_amount or 0),
        "balance":  float(row.balance_amount or 0),
        "status":   str(row.status           or ""),
    }]}


# ─── Supplier card endpoint (direct DB, no LLM — used by confirm-resolution picks)

@router.get("/supplier/{supplier_id}/card")
def supplier_card(supplier_id: int, db: Session = Depends(get_db)):
    s = db.execute(text("SELECT * FROM suppliers WHERE id=:id LIMIT 1"), {"id": supplier_id}).fetchone()
    if not s:
        return {"results": [{"type": "chat", "message": "Supplier nahi mila."}]}
    inv_items = db.execute(text(
        "SELECT i.id, i.name, "
        "SUM(CASE WHEN LOWER(t.txn_type)='in' THEN t.quantity ELSE -t.quantity END) AS stock "
        "FROM inventories i JOIN stock_transactions t ON i.id=t.inventory_id "
        "WHERE t.supplier_id=:sid GROUP BY i.id,i.name HAVING stock!=0"
    ), {"sid": supplier_id}).fetchall()
    return {"results": [
        {"type": "chat", "message": f"ye raha 👍 **{s.supplier_name}** ka profile:"},
        {
            "type": "result",
            "supplier": {
                "id":     int(s.id),
                "name":   str(s.supplier_name or ""),
                "code":   str(getattr(s, "supplier_code", "") or ""),
                "mobile": str(getattr(s, "mobile", "")        or ""),
                "city":   str(getattr(s, "city", "")          or ""),
                "email":  str(getattr(s, "email", "")         or ""),
                "gstin":  str(getattr(s, "gstin", "")         or ""),
            },
            "items": [{"name": r.name, "stock": float(r.stock)} for r in inv_items],
        },
    ]}


# ─── Inventory card endpoint (direct DB, no LLM — used by confirm-resolution picks)

@router.get("/inventory/{inventory_id}/card")
def inventory_card(inventory_id: int, db: Session = Depends(get_db)):
    inv = db.execute(text("SELECT * FROM inventories WHERE id=:id LIMIT 1"), {"id": inventory_id}).fetchone()
    if not inv:
        return {"results": [{"type": "chat", "message": "Item nahi mila."}]}
    stock = float(db.execute(text(
        "SELECT COALESCE(SUM(CASE WHEN LOWER(txn_type)='in' THEN quantity ELSE -quantity END),0) "
        "FROM stock_transactions WHERE inventory_id=:id"
    ), {"id": inventory_id}).scalar() or 0)
    cls = (getattr(inv, "classification", "") or "").lower()
    finish   = 0 if ("machining" in cls or "semi" in cls) else stock
    semi     = stock if "semi" in cls else 0
    machining= stock if "machining" in cls else 0
    return {"results": [
        {"type": "chat", "message": f"ye raha 👍 **{inv.name}** ka stock:"},
        {
            "type":              "result",
            "inventory": {
                "id":           int(inv.id),
                "name":         str(inv.name or ""),
                "category":     cls.upper(),
                "placement":    str(getattr(inv, "placement", "") or ""),
                "unit":         str(getattr(inv, "unit", "")      or ""),
                "model":        str(getattr(inv, "model", "")     or ""),
                "grade":        str(getattr(inv, "grade", "")     or ""),
            },
            "total_stock":        stock,
            "finish_stock":       finish,
            "semi_finish_stock":  semi,
            "machining_stock":    machining,
        },
    ]}


# ─── Inventory detail endpoints (direct SQL, no LLM) ────────────────────────

@router.get("/inventory/{inventory_id}/po-history")
def inventory_po_history(inventory_id: int, db: Session = Depends(get_db)):
    rows = db.execute(text("""
        SELECT po.id, po.po_number, po.po_date, po.status,
               s.supplier_name,
               poi.ordered_qty, poi.received_qty, poi.unit_price, poi.line_total
        FROM purchase_order_items poi
        JOIN purchase_orders po ON poi.purchase_order_id = po.id
        JOIN suppliers s        ON po.supplier_id        = s.id
        WHERE poi.inventory_id = :iid
        ORDER BY po.po_date DESC
        LIMIT 50
    """), {"iid": inventory_id}).fetchall()
    return {"rows": [
        {
            "po_id":      int(r.id),
            "po_number":  str(r.po_number),
            "date":       str(r.po_date or ""),
            "status":     str(r.status  or ""),
            "supplier":   str(r.supplier_name or ""),
            "ordered":    float(r.ordered_qty  or 0),
            "received":   float(r.received_qty or 0),
            "unit_price": float(r.unit_price   or 0),
            "line_total": float(r.line_total   or 0),
        }
        for r in rows
    ]}


@router.get("/inventory/{inventory_id}/suppliers")
def inventory_suppliers(inventory_id: int, db: Session = Depends(get_db)):
    rows = db.execute(text("""
        SELECT s.id, s.supplier_name, s.city,
               COUNT(DISTINCT po.id)        AS po_count,
               COALESCE(SUM(poi.ordered_qty), 0)  AS total_ordered,
               COALESCE(MIN(poi.unit_price),  0)  AS min_price,
               COALESCE(MAX(poi.unit_price),  0)  AS max_price
        FROM purchase_order_items poi
        JOIN purchase_orders po ON poi.purchase_order_id = po.id
        JOIN suppliers s        ON po.supplier_id        = s.id
        WHERE poi.inventory_id = :iid
        GROUP BY s.id, s.supplier_name, s.city
        ORDER BY total_ordered DESC
    """), {"iid": inventory_id}).fetchall()
    return {"rows": [
        {
            "id":            int(r.id),
            "name":          str(r.supplier_name or ""),
            "city":          str(r.city          or ""),
            "po_count":      int(r.po_count),
            "total_ordered": float(r.total_ordered),
            "min_price":     float(r.min_price),
            "max_price":     float(r.max_price),
        }
        for r in rows
    ]}


@router.get("/inventory/{inventory_id}/stock-log")
def inventory_stock_log(inventory_id: int, db: Session = Depends(get_db)):
    rows = db.execute(text("""
        SELECT txn_date, txn_type, quantity, ref_type, ref_no, remarks
        FROM stock_transactions
        WHERE inventory_id = :iid
        ORDER BY txn_date DESC, id DESC
        LIMIT 100
    """), {"iid": inventory_id}).fetchall()
    running = 0.0
    result_rows = []
    for r in reversed(rows):
        qty = float(r.quantity or 0)
        running += qty if str(r.txn_type).lower() == "in" else -qty
        result_rows.append({
            "date":     str(r.txn_date or ""),
            "type":     str(r.txn_type or ""),
            "qty":      qty,
            "balance":  round(running, 2),
            "ref_type": str(r.ref_type or ""),
            "ref_no":   str(r.ref_no   or ""),
            "remarks":  str(r.remarks  or ""),
        })
    result_rows.reverse()   # most recent first
    return {"rows": result_rows, "current_stock": round(running, 2)}


@router.get("/inventory/{inventory_id}/grns")
def inventory_grns(inventory_id: int, db: Session = Depends(get_db)):
    rows = db.execute(text("""
        SELECT g.grn_number, g.grn_date, g.invoice_no, g.remarks,
               gi.received_qty, gi.accepted_qty, gi.rejected_qty, gi.placement
        FROM grn_items gi
        JOIN grns g ON gi.grn_id = g.id
        WHERE gi.inventory_id = :iid
        ORDER BY g.grn_date DESC
        LIMIT 50
    """), {"iid": inventory_id}).fetchall()
    return {"rows": [
        {
            "grn_number":  str(r.grn_number  or ""),
            "date":        str(r.grn_date    or ""),
            "invoice_no":  str(r.invoice_no  or ""),
            "received":    float(r.received_qty  or 0),
            "accepted":    float(r.accepted_qty  or 0),
            "rejected":    float(r.rejected_qty  or 0),
            "placement":   str(r.placement   or ""),
            "remarks":     str(r.remarks     or ""),
        }
        for r in rows
    ]}


# ─── Zero-result mining (operator endpoint) ──────────────────────────────────
@router.get("/zero-results")
def v2_zero_results(limit: int = 50):
    """
    Scan recent chatbot_reqres.log entries for zero-result queries.
    Use this weekly: each entry is a query the bot couldn't answer — turn the
    common ones into aliases or improve handlers.
    """
    found = []
    try:
        with open("chatbot_reqres.log", "r", encoding="utf-8") as f:
            # Read from end is expensive on huge files; for now scan all.
            for line in f:
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                if entry.get("zero_result"):
                    found.append({
                        "ts": entry.get("ts"),
                        "request_id": entry.get("request_id"),
                        "query": (entry.get("request") or {}).get("query"),
                        "elapsed_ms": entry.get("elapsed_ms"),
                    })
        found = found[-limit:][::-1]  # most recent first
    except FileNotFoundError:
        pass
    except Exception as e:
        return {"queries": [], "error": str(e)[:200]}
    return {"count": len(found), "queries": found}
