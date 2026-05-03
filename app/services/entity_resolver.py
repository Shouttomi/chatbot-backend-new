"""
Entity resolver — canonicalize LLM-extracted search_target against real DB entities.

The LLM often produces typos ("Arawal" instead of "Arawali") or partial names. SQL
LIKE matching misses these. This module:

  1. Caches the canonical name list per entity type for 5 min (one query, not per-request).
  2. Uses rapidfuzz to find the best match above a confidence threshold.
  3. Returns the canonical name OR the original target if no good match exists,
     so the caller can fall back to LIKE matching for genuinely new/partial queries.

Thread-safe. Safe to call from multiple FastAPI workers (each has its own cache).
"""

import time
import threading
from typing import Optional, Dict, List, Tuple

from rapidfuzz import process, fuzz
from sqlalchemy import text
from sqlalchemy.orm import Session


_TTL = 300  # 5 minutes
_cache: Dict[str, Tuple[List[str], float]] = {}
_alias_cache: Dict[str, Tuple[Dict[str, str], float]] = {}  # category -> {alias_lower: canonical}
_lock = threading.Lock()

# Static fallback — works even when the local alias DB is down
_STATIC_ALIASES: Dict[str, Dict[str, str]] = {
    "inventory": {
        "bering":       "Bearing",
        "beering":      "Bearing",
        "bearning":     "Bearing",
        "seel":         "Oil Seal",
        "seal":         "Oil Seal",
        "hydrualic oil":"Hydraulic",
        "hydrualic":    "Hydraulic",
        "hydralic":     "Hydraulic",
        "hydrlic":      "Hydraulic",
        "natt bolt":    "Bolt",
        "geer box":     "Gear Box",
        "sprocet":      "Sprocket",
        "silindar":     "Cylinder",
        "sylinder":     "Cylinder",
        "v belt":       "V Belt",
        "vbelt":        "V Belt",
        "grece":        "Grease",
        "grees":        "Grease",
        "chaen":        "Chain",
        "cheyn":        "Chain",
        "motter":       "Motor",
        "pummp":        "Pump",
        "filtar":       "Filter",
        "filtter":      "Filter",
    },
    "supplier": {
        "total suppliers":     "supplier count",
        "kitne suppliers":     "supplier count",
        "kitne suppliers hain":"supplier count",
        "number of suppliers": "supplier count",
        # Arawali disambiguation — two similar-named suppliers exist
        "arawali minerals":    "Arawali Minerals",
        "arawali mineral":     "Arawali Minerals",
        "arawali crushing":    "Arawali Crushing Plant-Rajsamand (Raj.)",
        "arawali crushing plant": "Arawali Crushing Plant-Rajsamand (Raj.)",
        "arawali rajsamand":   "Arawali Crushing Plant-Rajsamand (Raj.)",
    },
}

# Three-zone confidence model — see resolve_with_confidence() docstring.
HIGH_CONFIDENCE_FROM    = 92   # >= : silently accept
BORDERLINE_FROM         = 75   # 75-91: ask user to confirm via 👍 / 👎
# < 75: pass through untouched, let LIKE matching do its thing

# Per-category SQL — one column each, deduped, non-null
_QUERIES = {
    "supplier":  "SELECT DISTINCT supplier_name FROM suppliers WHERE supplier_name IS NOT NULL AND supplier_name != ''",
    "project":   "SELECT DISTINCT name FROM projects WHERE is_deleted=0 AND name IS NOT NULL AND name != ''",
    "inventory": "SELECT DISTINCT name FROM inventories WHERE name IS NOT NULL AND name != ''",
}

# Confidence thresholds — tuned per category.
# Suppliers/projects are proper nouns, demand high confidence (typos OK, but no wild guesses).
# Inventory items are generic English/Hindi nouns ("bearing", "bolt") — lower bar OK.
_THRESHOLDS = {
    "supplier":  85,
    "project":   85,
    "inventory": 75,
}


def _load(db: Session, category: str) -> List[str]:
    """Fetch + cache the canonical name list for a category."""
    now = time.time()
    with _lock:
        hit = _cache.get(category)
        if hit and (now - hit[1]) < _TTL:
            return hit[0]

    # Query outside the lock — DB call shouldn't block other resolves
    sql = _QUERIES.get(category)
    if not sql:
        return []
    try:
        rows = db.execute(text(sql)).fetchall()
        names = [r[0].strip() for r in rows if r[0]]
    except Exception as e:
        print(f"[RESOLVER] Failed to load {category}: {e}")
        return []

    with _lock:
        _cache[category] = (names, now)
    return names


def _load_aliases(_unused_db, category: str) -> Dict[str, str]:
    """Load alias -> canonical map for a category (cached). Reads from the LOCAL
    chatbot-ops DB (where entity_aliases lives), not the prod DB the caller passed in.
    Tolerates missing table."""
    now = time.time()
    with _lock:
        hit = _alias_cache.get(category)
        if hit and (now - hit[1]) < _TTL:
            return hit[0]
    try:
        from app.db.database import LocalSessionLocal
        with LocalSessionLocal() as local_db:
            rows = local_db.execute(
                text("SELECT alias, canonical_name FROM entity_aliases WHERE category=:c"),
                {"c": category},
            ).fetchall()
        mp = {r[0].strip().lower(): r[1].strip() for r in rows if r[0] and r[1]}
    except Exception as e:
        # Table may not exist yet (pre-migration). Treat as empty, don't spam logs.
        if "doesn't exist" not in str(e) and "no such table" not in str(e).lower():
            print(f"[RESOLVER] alias load failed for {category}: {e}")
        mp = {}
    with _lock:
        _alias_cache[category] = (mp, now)
    return mp


def resolve(db: Session, target: str, category: str) -> Tuple[str, Optional[int]]:
    """
    Try to canonicalize `target` against known entities of `category`.

    Order: explicit alias table (score=100) → fuzzy match against live entities.
    Returns (resolved_name, confidence). If no good match, returns (target, None)
    so the caller can fall back to LIKE matching.
    """
    target = (target or "").strip()
    if not target or len(target) < 2:
        return target, None

    # 0. Static typo aliases — always available, no DB required
    _static = _STATIC_ALIASES.get(category, {}).get(target.lower())
    if _static:
        print(f"[RESOLVER] static alias {category}: '{target}' -> '{_static}'")
        return _static, 100

    # 1. Alias table — exact, curated, beats fuzzy guess every time
    aliases = _load_aliases(db, category)
    canonical = aliases.get(target.lower())
    if canonical:
        print(f"[RESOLVER] alias hit {category}: '{target}' -> '{canonical}'")
        return canonical, 100

    # 2. Fuzzy match against live entity names
    names = _load(db, category)
    if not names:
        return target, None

    threshold = _THRESHOLDS.get(category, 80)
    match = process.extractOne(target, names, scorer=fuzz.WRatio, score_cutoff=threshold)
    if match:
        canonical, score, _ = match
        if canonical.lower() != target.lower():
            print(f"[RESOLVER] fuzzy {category}: '{target}' -> '{canonical}' (score={score:.0f})")
        return canonical, int(score)

    return target, None


def resolve_with_confidence(db: Session, target: str, category: str) -> Dict[str, object]:
    """
    Richer version of resolve() — returns the confidence zone plus top-3 candidates
    for borderline matches, so the caller can ask the user to confirm.

    Returns a dict with:
      - canonical:  str — best match (or original target if low/no match)
      - confidence: "high" | "borderline" | "low"  | "alias"
      - score:      int  — best fuzzy score, or 100 for alias hit
      - candidates: list[(name, score)]  — top 3 above 70 (only when borderline)
    """
    target = (target or "").strip()
    if not target or len(target) < 2:
        return {"canonical": target, "confidence": "low", "score": 0, "candidates": []}

    # 0. Static typo aliases — always available
    _static = _STATIC_ALIASES.get(category, {}).get(target.lower())
    if _static:
        print(f"[RESOLVER] static alias {category}: '{target}' -> '{_static}'")
        return {"canonical": _static, "confidence": "alias", "score": 100, "candidates": []}

    # 1. Alias table — exact, curated, beats fuzzy guess
    aliases = _load_aliases(db, category)
    canonical = aliases.get(target.lower())
    if canonical:
        print(f"[RESOLVER] alias hit {category}: '{target}' -> '{canonical}'")
        return {"canonical": canonical, "confidence": "alias", "score": 100, "candidates": []}

    # 2. Fuzzy match
    names = _load(db, category)
    if not names:
        return {"canonical": target, "confidence": "low", "score": 0, "candidates": []}

    best = process.extractOne(target, names, scorer=fuzz.WRatio)  # no cutoff — get the score
    if not best:
        return {"canonical": target, "confidence": "low", "score": 0, "candidates": []}
    top_name, top_score, _ = best
    top_score = int(top_score)

    if top_score >= HIGH_CONFIDENCE_FROM:
        if top_name.lower() != target.lower():
            print(f"[RESOLVER] high {category}: '{target}' -> '{top_name}' (score={top_score})")
        return {"canonical": top_name, "confidence": "high", "score": top_score, "candidates": []}

    if top_score >= BORDERLINE_FROM:
        # gather alternates so the user has a real choice
        alts = process.extract(target, names, scorer=fuzz.WRatio, limit=3, score_cutoff=70)
        cand_list = [(n, int(s)) for n, s, _ in alts]
        print(f"[RESOLVER] borderline {category}: '{target}' candidates={cand_list}")
        return {
            "canonical":  top_name,
            "confidence": "borderline",
            "score":      top_score,
            "candidates": cand_list,
        }

    # below threshold — let the caller fall back to LIKE matching
    return {"canonical": target, "confidence": "low", "score": top_score, "candidates": []}


def invalidate(category: Optional[str] = None) -> None:
    """Drop both name + alias cache after writes. Call category=None for all."""
    with _lock:
        if category is None:
            _cache.clear()
            _alias_cache.clear()
        else:
            _cache.pop(category, None)
            _alias_cache.pop(category, None)


def cache_stats() -> Dict[str, int]:
    with _lock:
        return {cat: len(names) for cat, (names, _) in _cache.items()}
