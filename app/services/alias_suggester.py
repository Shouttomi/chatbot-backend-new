"""
alias_suggester — mine zero-result queries from chatbot_reqres.log, cluster
similar ones, and suggest a canonical entity name per cluster.

Pipeline:
  1. read recent log entries where zero_result=True
  2. drop queries already aliased or admin-skipped
  3. cluster similar queries (rapidfuzz token-set ratio >= 80)
  4. for each cluster, fuzzy-match the representative against suppliers /
     projects / inventories — return top 3 candidates with scores
  5. sort by hit count, return JSON for admin review

All work is in-memory + cached for 60s. Reading the log is sequential (cheap
for typical sizes); switch to a tail/index if logs grow past 100MB.
"""

import json
import os
import re
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

from rapidfuzz import process, fuzz
from sqlalchemy import text
from sqlalchemy.orm import Session


_LOG_PATH       = "chatbot_reqres.log"
_DEFAULT_DAYS   = 7
_CLUSTER_FUZZ   = 80     # WRatio threshold to merge two queries into one cluster
_CANDIDATE_TOP  = 3      # how many candidate canonicals to suggest per cluster
_MIN_HITS       = 1      # minimum cluster size to surface (1 = show everything)
_RESULT_CACHE: Dict[str, Tuple[float, list]] = {}
_CACHE_TTL      = 60     # seconds
_lock = threading.Lock()


# Strip noise so "MBE" and "MBE balance" cluster together
_FILLER = {
    "balance", "details", "detail", "info", "show", "tell", "give", "ka", "ki", "ke",
    "ko", "se", "aur", "or", "for", "the", "a", "an", "ka", "kitna", "kitni",
    "kitne", "do", "de", "dikhao", "dikha", "batao", "bata", "bhai", "please",
    "plz", "wala", "wale", "wali", "supplier", "vendor", "party",
    "uska", "uski", "iska", "iski", "po", "order", "orders",
}


def _norm(q: str) -> str:
    """Lowercase, strip filler, collapse whitespace — for clustering."""
    q = (q or "").lower()
    q = re.sub(r"[^\w\s\-]", " ", q)
    toks = [t for t in q.split() if t and t not in _FILLER and len(t) > 1]
    return " ".join(toks).strip()


def _read_zero_result_queries(days: int) -> List[Dict[str, Any]]:
    """Tail the log file and return zero_result entries from last `days` days."""
    if not os.path.exists(_LOG_PATH):
        return []
    cutoff = datetime.now() - timedelta(days=days)
    out = []
    try:
        with open(_LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                if not e.get("zero_result"):
                    continue
                ts = e.get("ts", "")
                try:
                    if datetime.fromisoformat(ts) < cutoff:
                        continue
                except Exception:
                    pass
                q = ((e.get("request") or {}).get("query") or "").strip()
                if q:
                    out.append({"q": q, "ts": ts, "rid": e.get("request_id")})
    except Exception as e:
        print(f"[SUGGESTER] log read failed: {e}")
    return out


def _existing_aliases_lower() -> set:
    """Aliases already curated — read from LOCAL chatbot-ops DB."""
    try:
        from app.db.database import LocalSessionLocal
        with LocalSessionLocal() as local_db:
            rows = local_db.execute(text("SELECT alias FROM entity_aliases")).fetchall()
        return {r[0].lower() for r in rows if r[0]}
    except Exception:
        return set()


def _skipped_aliases_lower() -> set:
    """Aliases the admin already dismissed — read from LOCAL chatbot-ops DB."""
    try:
        from app.db.database import LocalSessionLocal
        with LocalSessionLocal() as local_db:
            rows = local_db.execute(text("SELECT alias FROM alias_suggestions_skipped")).fetchall()
        return {r[0].lower() for r in rows if r[0]}
    except Exception:
        return set()


def _entity_names(db: Session) -> Dict[str, List[str]]:
    """All canonical names per category — for matching cluster reps to candidates."""
    out = {}
    try:
        rows = db.execute(text("SELECT supplier_name FROM suppliers WHERE supplier_name IS NOT NULL")).fetchall()
        out["supplier"] = [r[0].strip() for r in rows if r[0]]
    except Exception:
        out["supplier"] = []
    try:
        rows = db.execute(text("SELECT name FROM projects WHERE is_deleted=0 AND name IS NOT NULL")).fetchall()
        out["project"] = [r[0].strip() for r in rows if r[0]]
    except Exception:
        out["project"] = []
    try:
        rows = db.execute(text("SELECT name FROM inventories WHERE name IS NOT NULL")).fetchall()
        out["inventory"] = [r[0].strip() for r in rows if r[0]]
    except Exception:
        out["inventory"] = []
    return out


def _cluster(queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Greedy clustering — for each new query, attach to first existing cluster
    whose representative has WRatio >= _CLUSTER_FUZZ; else start a new cluster."""
    clusters: List[Dict[str, Any]] = []
    for entry in queries:
        norm_q = _norm(entry["q"])
        if not norm_q:
            continue
        attached = False
        for c in clusters:
            if fuzz.WRatio(norm_q, c["norm_rep"]) >= _CLUSTER_FUZZ:
                c["aliases"].append(entry["q"])
                attached = True
                break
        if not attached:
            clusters.append({
                "rep":      entry["q"],
                "norm_rep": norm_q,
                "aliases":  [entry["q"]],
            })
    # collapse exact duplicates within a cluster, count occurrences
    for c in clusters:
        from collections import Counter
        ct = Counter(c["aliases"])
        c["aliases"] = sorted(ct.keys(), key=lambda a: -ct[a])
        c["counts"]  = dict(ct)
        c["hit_count"] = sum(ct.values())
    return clusters


def _best_candidates(rep: str, names: List[str]) -> List[Tuple[str, int]]:
    """Top-3 fuzzy matches above score 30 (very lax — admin can reject)."""
    if not rep or not names:
        return []
    matches = process.extract(rep, names, scorer=fuzz.WRatio, limit=_CANDIDATE_TOP, score_cutoff=30)
    return [(n, int(s)) for n, s, _ in matches]


def get_suggestions(db: Session, days: int = _DEFAULT_DAYS, force: bool = False) -> Dict[str, Any]:
    """Main entry point — returns clusters + candidates, sorted by hit count desc."""
    cache_key = f"{days}"
    now = time.time()
    with _lock:
        if not force and cache_key in _RESULT_CACHE:
            ts, cached = _RESULT_CACHE[cache_key]
            if now - ts < _CACHE_TTL:
                return {"cached": True, "suggestions": cached, "days": days}

    raw = _read_zero_result_queries(days)
    if not raw:
        return {"cached": False, "suggestions": [], "days": days, "note": "no zero-result queries in window"}

    existing = _existing_aliases_lower()
    skipped  = _skipped_aliases_lower()
    raw      = [r for r in raw if r["q"].lower() not in existing and r["q"].lower() not in skipped]

    clusters = _cluster(raw)
    clusters = [c for c in clusters if c["hit_count"] >= _MIN_HITS]

    names_by_cat = _entity_names(db)

    suggestions = []
    for c in clusters:
        # try each category; pick the one whose top match scores highest
        best_cat:    Optional[str] = None
        best_cands:  List[Tuple[str,int]] = []
        best_score:  int = -1
        for cat, names in names_by_cat.items():
            cands = _best_candidates(c["rep"], names)
            if cands and cands[0][1] > best_score:
                best_score = cands[0][1]
                best_cat   = cat
                best_cands = cands
        suggestions.append({
            "aliases":      c["aliases"],
            "counts":       c["counts"],
            "hit_count":    c["hit_count"],
            "category":     best_cat,
            "candidates":   [{"name": n, "score": s} for n, s in best_cands],
            "suggested":    best_cands[0][0] if best_cands else None,
            "confidence":   best_score if best_score >= 0 else 0,
        })
    # high-confidence + frequent first — that's where admin should click first
    suggestions.sort(key=lambda s: (-s["hit_count"], -s["confidence"]))

    with _lock:
        _RESULT_CACHE[cache_key] = (now, suggestions)
    return {"cached": False, "suggestions": suggestions, "days": days, "total": len(suggestions)}


def invalidate_cache():
    with _lock:
        _RESULT_CACHE.clear()
