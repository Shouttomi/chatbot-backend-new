"""
complex_query — handlers for queries the single-shot intent extractor can't express.

Includes handle_fk_query: a Text-to-SQL fallback that asks the LLM to generate a
MySQL SELECT using the live schema doc, then executes it safely and formats the result.
top-N rankings, max/min, side-by-side comparison, negation, threshold comparison.

Returns a v2-chatbot response dict if it handled the query, or None to fall through
to the existing legacy handlers.

Defensive: every SQL is parameterized; aggregation_field is whitelisted; LIMIT is capped.
"""

import re
from typing import Any, Dict, List, Optional
from sqlalchemy import text
from sqlalchemy.orm import Session


# Whitelist what the LLM is allowed to aggregate over — never trust raw strings in SQL.
_AGG_FIELDS = {
    "total_amount":   "p.total_amount",
    "balance_amount": "p.balance_amount",
    "spend":          "p.total_amount",
    "tax_amount":     "p.tax_amount",
}

_GROUP_BY_SQL = {
    "supplier": ("s.supplier_name",                "supplier"),
    "city":     ("s.city",                         "city"),
    "status":   ("p.status",                       "status"),
    "month":    ("DATE_FORMAT(p.po_date, '%Y-%m')", "month"),
}

_MAX_LIMIT = 50


def _clamp_limit(filters: Dict[str, Any], default: int = 10) -> int:
    try:
        n = int(filters.get("limit") or default)
    except (TypeError, ValueError):
        n = default
    return max(1, min(n, _MAX_LIMIT))


def _agg_column(field: Optional[str]) -> Optional[str]:
    if not field:
        return None
    return _AGG_FIELDS.get(field.lower())


def handle_complex(ai: Dict[str, Any], db: Session, target: str) -> Optional[Dict[str, Any]]:
    """
    Try to answer with a complex-query handler. Returns None if not applicable.
    Caller should fall through to legacy handlers in that case.
    """
    aggregation = (ai.get("aggregation") or "").lower() or None
    secondary   = (ai.get("secondary_target") or "").strip()
    negate      = bool(ai.get("negate"))
    comparison  = ai.get("comparison") or None
    group_by    = (ai.get("group_by") or "").lower() or None
    intents     = ai.get("intents") or []
    filters     = ai.get("filters") or {}

    # Comparison between two entities (e.g. "Arawali vs DCL pending")
    if aggregation == "compare" and target and secondary:
        return _handle_compare(db, target, secondary, ai)

    # Top-N / max / min / sum / count over POs
    if aggregation in ("top_n", "max", "min", "sum", "count") and "po_search" in intents:
        return _handle_po_aggregate(db, ai, aggregation, group_by, target)

    # Threshold filter on POs ("balance > 50000")
    if comparison and "po_search" in intents:
        return _handle_po_threshold(db, ai, comparison, target)

    # Negation on projects ("NOT in Rajsamand")
    if negate and "project_search" in intents:
        return _handle_project_negate(db, ai, filters)

    return None


def _handle_compare(db: Session, a: str, b: str, ai: Dict[str, Any]) -> Dict[str, Any]:
    field = _agg_column(ai.get("aggregation_field")) or "p.balance_amount"
    rows = db.execute(
        text(
            f"SELECT s.supplier_name AS name, "
            f"       COUNT(p.id)     AS po_count, "
            f"       COALESCE(SUM({field}), 0) AS total "
            f"FROM suppliers s "
            f"LEFT JOIN purchase_orders p ON p.supplier_id = s.id "
            f"WHERE LOWER(s.supplier_name) LIKE :a OR LOWER(s.supplier_name) LIKE :b "
            f"GROUP BY s.id, s.supplier_name "
            f"ORDER BY total DESC "
            f"LIMIT 10"
        ),
        {"a": f"%{a.lower()}%", "b": f"%{b.lower()}%"},
    ).fetchall()

    if not rows:
        return {"results": [{"type": "chat", "message": f"{a} ya {b} ka data nahi mila. 🤔"}]}

    pretty = "\n".join(f"• **{r.name}** — {int(r.po_count)} POs, ₹{float(r.total):,.0f}" for r in rows)
    return {
        "results": [
            {"type": "chat", "message": f"📊 **{a} vs {b}**\n\n{pretty}", "db_checked": True},
            {"type": "po_summary", "rows": [dict(r._mapping) for r in rows]},
        ]
    }


def _handle_po_aggregate(db: Session, ai: Dict[str, Any], agg: str, group_by: Optional[str], target: str) -> Dict[str, Any]:
    field    = _agg_column(ai.get("aggregation_field")) or "p.total_amount"
    limit    = _clamp_limit(ai.get("filters") or {}, default=5)
    where    = ["1=1"]
    params: Dict[str, Any] = {"l": limit}

    if target:
        where.append("LOWER(s.supplier_name) LIKE :tgt")
        params["tgt"] = f"%{target.lower()}%"

    where_sql = " AND ".join(where)

    if agg == "count":
        row = db.execute(
            text(f"SELECT COUNT(*) AS cnt FROM purchase_orders p LEFT JOIN suppliers s ON p.supplier_id=s.id WHERE {where_sql}"),
            params,
        ).fetchone()
        return {"results": [{"type": "chat", "message": f"Total **{int(row.cnt)}** POs mile.", "db_checked": True}]}

    if group_by and group_by in _GROUP_BY_SQL:
        grp_sql, grp_label = _GROUP_BY_SQL[group_by]
        agg_sql = "SUM" if agg in ("top_n", "sum") else agg.upper()
        rows = db.execute(
            text(
                f"SELECT {grp_sql} AS grp, "
                f"       COUNT(p.id) AS po_count, "
                f"       COALESCE({agg_sql}({field}), 0) AS total "
                f"FROM purchase_orders p LEFT JOIN suppliers s ON p.supplier_id=s.id "
                f"WHERE {where_sql} "
                f"GROUP BY {grp_sql} "
                f"ORDER BY total DESC "
                f"LIMIT :l"
            ),
            params,
        ).fetchall()
        if not rows:
            return {"results": [{"type": "chat", "message": "Koi data nahi mila. 🤔"}]}
        pretty = "\n".join(f"{i+1}. **{r.grp or 'Unknown'}** — {int(r.po_count)} POs, ₹{float(r.total):,.0f}" for i, r in enumerate(rows))
        return {
            "results": [
                {"type": "chat", "message": f"📊 Top {len(rows)} by {grp_label}:\n\n{pretty}", "db_checked": True},
                {"type": "po_summary", "rows": [dict(r._mapping) for r in rows]},
            ]
        }

    # max / min single PO
    if agg in ("max", "min"):
        order = "DESC" if agg == "max" else "ASC"
        rows = db.execute(
            text(
                f"SELECT p.po_number, p.total_amount, p.balance_amount, p.status, p.po_date, s.supplier_name "
                f"FROM purchase_orders p LEFT JOIN suppliers s ON p.supplier_id=s.id "
                f"WHERE {where_sql} "
                f"ORDER BY {field} {order} "
                f"LIMIT :l"
            ),
            params,
        ).fetchall()
        if not rows:
            return {"results": [{"type": "chat", "message": "Koi PO nahi mila. 🤔"}]}
        top = rows[0]
        msg = f"{'Sabse bada' if agg=='max' else 'Sabse chhota'} PO **{top.po_number}** — {top.supplier_name or 'Unknown'} — ₹{float(top.total_amount or 0):,.0f}"
        return {"results": [
            {"type": "chat", "message": msg, "db_checked": True},
            {"type": "po_list", "rows": [dict(r._mapping) for r in rows]},
        ]}

    # sum (no grouping)
    row = db.execute(
        text(f"SELECT COALESCE(SUM({field}), 0) AS total, COUNT(*) AS cnt FROM purchase_orders p LEFT JOIN suppliers s ON p.supplier_id=s.id WHERE {where_sql}"),
        params,
    ).fetchone()
    return {"results": [{"type": "chat", "message": f"Total ₹{float(row.total):,.0f} ({int(row.cnt)} POs).", "db_checked": True}]}


def _handle_po_threshold(db: Session, ai: Dict[str, Any], comparison: Dict[str, Any], target: str) -> Optional[Dict[str, Any]]:
    op_map = {"gt": ">", "lt": "<", "gte": ">=", "lte": "<=", "eq": "="}
    op = op_map.get((comparison.get("op") or "").lower())
    try:
        value = float(comparison.get("value"))
    except (TypeError, ValueError):
        return None
    if not op:
        return None

    field    = _agg_column(ai.get("aggregation_field")) or "p.balance_amount"
    limit    = _clamp_limit(ai.get("filters") or {}, default=10)
    where    = [f"{field} {op} :v"]
    params: Dict[str, Any]   = {"v": value, "l": limit}

    if target:
        where.append("LOWER(s.supplier_name) LIKE :tgt")
        params["tgt"] = f"%{target.lower()}%"

    rows = db.execute(
        text(
            f"SELECT p.po_number, p.total_amount, p.balance_amount, p.status, p.po_date, s.supplier_name "
            f"FROM purchase_orders p LEFT JOIN suppliers s ON p.supplier_id=s.id "
            f"WHERE {' AND '.join(where)} "
            f"ORDER BY {field} DESC "
            f"LIMIT :l"
        ),
        params,
    ).fetchall()
    if not rows:
        return {"results": [{"type": "chat", "message": f"Is condition pe koi PO nahi mila."}]}
    pretty = "\n".join(f"• **{r.po_number}** — {r.supplier_name or 'Unknown'} — ₹{float(r.balance_amount or 0):,.0f} balance" for r in rows)
    return {"results": [
        {"type": "chat", "message": f"📊 {len(rows)} POs match karte hain:\n\n{pretty}", "db_checked": True},
        {"type": "po_list", "rows": [dict(r._mapping) for r in rows]},
    ]}


def _handle_project_negate(db: Session, ai: Dict[str, Any], filters: Dict[str, Any]) -> Dict[str, Any]:
    where:  List[str]    = ["is_deleted=0"]
    params: Dict[str, Any] = {}
    limit   = _clamp_limit(filters, default=10)
    params["l"] = limit

    # `projects` table has: name, status, priority, deadline (no city/machine column)
    if filters.get("status"):
        where.append("LOWER(COALESCE(status,'')) != :st")
        params["st"] = str(filters["status"]).lower()
    if filters.get("priority"):
        where.append("LOWER(COALESCE(priority,'')) != :pr")
        params["pr"] = str(filters["priority"]).lower()
    # If user asked "NOT in <name>", treat the city filter the LLM emitted as a name match
    if filters.get("city"):
        where.append("LOWER(COALESCE(name,'')) NOT LIKE :nm")
        params["nm"] = f"%{str(filters['city']).lower()}%"

    rows = db.execute(
        text(f"SELECT * FROM projects WHERE {' AND '.join(where)} ORDER BY id DESC LIMIT :l"),
        params,
    ).fetchall()
    if not rows:
        return {"results": [{"type": "chat", "message": "Koi matching project nahi mila."}]}
    return {"results": [
        {"type": "chat", "message": f"📋 {len(rows)} projects (excluded matched filter):", "db_checked": True},
        {"type": "project_list", "rows": [dict(r._mapping) for r in rows]},
    ]}


# ─── Text-to-SQL fallback ─────────────────────────────────────────────────────

# Absolute block list — these keywords must never appear in LLM-generated SQL.
_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|REPLACE|EXEC|EXECUTE|GRANT|REVOKE)\b",
    re.I,
)
_SQL_LIMIT_CAP = 30  # hard cap injected/enforced on every generated query


def _safe_sql(raw_sql: str) -> str:
    """
    Validate that `raw_sql` is a safe SELECT, enforce LIMIT cap, return cleaned SQL.
    Raises ValueError on anything that looks dangerous.
    """
    sql = raw_sql.strip().rstrip(";")
    if not sql.upper().startswith("SELECT"):
        raise ValueError(f"Non-SELECT: {sql[:60]}")
    if _FORBIDDEN.search(sql):
        raise ValueError(f"Forbidden keyword in SQL: {sql[:80]}")
    # Enforce LIMIT
    if not re.search(r"\bLIMIT\b", sql, re.I):
        sql += f" LIMIT {_SQL_LIMIT_CAP}"
    else:
        # Cap existing LIMIT value
        def _cap(m):
            n = min(int(m.group(1)), _SQL_LIMIT_CAP)
            return f"LIMIT {n}"
        sql = re.sub(r"\bLIMIT\s+(\d+)", _cap, sql, flags=re.I)
    return sql


def _format_rows(rows, max_rows: int = 25) -> List[Dict[str, Any]]:
    """
    Convert SQLAlchemy rows into typed result cards where shape is recognisable,
    falling back to a single formatted chat message for generic results.

    Returns a list of result dicts (same format as the legacy handlers).
    """
    if not rows:
        return [{"type": "chat", "message": "Koi data nahi mila. 🧐"}]

    cols = list(rows[0]._mapping.keys())
    col_set = set(c.lower() for c in cols)

    # ── Purchase Order shape ──────────────────────────────────────────
    if "po_number" in col_set and "total_amount" in col_set:
        cards = []
        total_bal = sum(float(getattr(r, "balance_amount", 0) or 0) for r in rows[:max_rows])
        cards.append({"type": "chat",
                      "message": f"📄 **{len(rows)} orders** mile. Total balance: **₹{total_bal:,.2f}**",
                      "db_checked": True})
        for r in rows[:max_rows]:
            m = dict(r._mapping)
            cards.append({
                "type":     "po",
                "po_no":    str(m.get("po_number") or ""),
                "supplier": str(m.get("supplier_name") or m.get("supplier_id") or ""),
                "date":     str(m.get("po_date") or ""),
                "total":    float(m.get("total_amount") or 0),
                "advance":  float(m.get("advance_amount") or 0),
                "balance":  float(m.get("balance_amount") or 0),
                "status":   str(m.get("status") or "").capitalize(),
            })
        return cards

    # ── Supplier shape ────────────────────────────────────────────────
    if "supplier_name" in col_set and "gstin" in col_set:
        cards = [{"type": "chat",
                  "message": f"🏭 **{len(rows)} suppliers** mile:",
                  "db_checked": True}]
        for r in rows[:max_rows]:
            m = dict(r._mapping)
            cards.append({"type": "result", "db_checked": True, "supplier": {
                "id":     m.get("id", ""),
                "name":   str(m.get("supplier_name") or ""),
                "code":   str(m.get("supplier_code") or "N/A"),
                "mobile": str(m.get("mobile") or "N/A"),
                "city":   str(m.get("city") or "N/A"),
                "email":  str(m.get("email") or "N/A"),
                "gstin":  str(m.get("gstin") or "N/A"),
            }})
        return cards

    # ── Inventory shape ───────────────────────────────────────────────
    if "inventory_id" in col_set or ("name" in col_set and "ordered_qty" in col_set):
        lines = []
        for r in rows[:max_rows]:
            m = dict(r._mapping)
            name = m.get("inv_name") or m.get("name") or m.get("inventory_id") or "?"
            qty  = m.get("ordered_qty") or m.get("quantity") or m.get("stock") or ""
            rate = m.get("unit_price") or m.get("rate") or ""
            parts = [f"**{name}**"]
            if qty: parts.append(f"qty: {float(qty):.0f}")
            if rate: parts.append(f"rate: ₹{float(rate):,.2f}")
            lines.append("• " + " | ".join(parts))
        return [{"type": "chat",
                 "message": f"📦 **{len(rows)} items** mile:\n\n" + "\n".join(lines),
                 "db_checked": True}]

    # ── Generic: build readable bullet list ───────────────────────────
    lines = []
    for r in rows[:max_rows]:
        parts = []
        for k, v in dict(r._mapping).items():
            if v is None or str(v).strip() == "":
                continue
            label = k.replace("_", " ").title()
            val   = str(v)
            if len(val) > 55:
                val = val[:52] + "..."
            parts.append(f"**{label}**: {val}")
        if parts:
            lines.append("• " + " | ".join(parts))
    text_body = "\n".join(lines) if lines else "Koi data nahi mila."
    return [{"type": "chat",
             "message": f"🔍 **{len(rows)} records** mile:\n\n{text_body}",
             "db_checked": True}]


def handle_fk_query(
    user_query: str,
    schema_text: str,
    db: Session,
    context_entity: Optional[Dict[str, Any]] = None,
    limit: int = 15,
) -> Optional[Dict[str, Any]]:
    """
    Text-to-SQL fallback for arbitrary FK-traversal queries.

    1. Enriches user_query with context_entity IDs when available.
    2. Asks the LLM to write a MySQL SELECT via ask_for_sql().
    3. Validates SQL (SELECT-only, forbidden keyword check, LIMIT cap).
    4. Executes and formats as typed JSON cards (PO / supplier / inventory / generic).
    Returns None on any failure — caller falls through gracefully.
    """
    from app.services.v2_ollama_engine import ask_for_sql

    # Enrich query with context so LLM writes targeted WHERE clauses
    enriched = user_query
    if context_entity:
        ct = context_entity.get("type", "")
        if ct == "supplier":
            enriched += (f" [context: supplier id={context_entity.get('id')}"
                         f" name='{context_entity.get('name')}']")
        elif ct == "purchase_order":
            enriched += (f" [context: purchase_order id={context_entity.get('id')}"
                         f" po_number='{context_entity.get('po_no')}"
                         f"' supplier_id={context_entity.get('supplier_id')}]")
        elif ct == "inventory":
            enriched += (f" [context: inventory id={context_entity.get('id')}"
                         f" name='{context_entity.get('name')}']")
        elif ct == "project":
            enriched += (f" [context: project id={context_entity.get('id')}"
                         f" name='{context_entity.get('name')}']")

    try:
        raw_sql = ask_for_sql(enriched, schema_text)
    except RuntimeError as e:
        print(f"[FK-QUERY] ask_for_sql failed: {e}")
        return None

    try:
        safe_sql = _safe_sql(raw_sql)
    except ValueError as e:
        print(f"[FK-QUERY] unsafe SQL rejected: {e}")
        return None

    print(f"[FK-QUERY] executing: {safe_sql[:120]}")
    try:
        rows = db.execute(text(safe_sql)).fetchall()
    except Exception as e:
        print(f"[FK-QUERY] SQL execution error: {e}")
        return None

    cards = _format_rows(rows, max_rows=limit)
    return {"results": cards}
