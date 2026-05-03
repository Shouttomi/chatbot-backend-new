"""
schema_doc — ground the LLM in the real DB schema (column names + FK relationships).

Why this exists: hand-typed column names in the prompt drift from reality and produce
silent wrong answers (e.g. we once had `quantity/rate/amount` for purchase_order_items
but the real columns are `ordered_qty/unit_price/line_total`).

This module:
  1. Queries INFORMATION_SCHEMA once, caches for process lifetime.
  2. Keeps only the columns the chatbot reasons about (tight ~350-token block).
  3. Appends a static FK section so the LLM knows how tables join.

READ-ONLY. Never touches row data.
"""

import threading
from typing import Dict, List, Tuple, Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine


_schema_text: str = ""
_initialized: bool = False
_lock = threading.Lock()


# Whitelist — only tables + columns the chatbot queries.
# Key = table name, Value = set of columns to expose.
_RELEVANT: Dict[str, set] = {
    "suppliers": {
        "id", "supplier_name", "supplier_code", "category",
        "city", "state", "mobile", "email",
        "gstin", "pan", "gst_registered",
        "supplier_address", "bank_name", "ifsc", "account_number",
        "contact_person", "registration_date",
    },
    "inventories": {
        "id", "name", "unit", "type", "classification",
        "placement", "model", "grade", "is_deleted",
        "opening_quantity", "min_quantity",
    },
    "stock_transactions": {
        "inventory_id", "quantity", "txn_type", "txn_date", "supplier_id",
    },
    "products": {
        "id", "name", "is_deleted", "estimation_budget", "start_date",
    },
    "product_items": {
        "id", "product_id", "inventory_id", "quantity",
    },
    "projects": {
        "id", "name", "status", "priority", "deadline",
        "start_date", "end_date", "budget", "comment",
        "refurbish", "is_deleted",
    },
    "project_item": {
        "id", "project_id", "inventory_id", "quantity",
    },
    "project_products": {
        "id", "project_id", "product_id", "quantity", "status",
    },
    "purchase_orders": {
        "id", "po_number", "supplier_id", "po_date", "expected_delivery",
        "total_qty", "subtotal", "tax_amount", "total_amount",
        "advance_amount", "balance_amount", "remaining_amount",
        "loading_cutting_charges", "freight_charges",
        "status", "delivery_status", "remarks", "created_at", "completed_at",
    },
    "purchase_order_items": {
        "id", "purchase_order_id", "inventory_id",
        "ordered_qty", "received_qty",
        "unit_price", "line_total",
        "tax_amount", "tax_percent", "tax_type",
        "discount", "discount_amount", "hsn",
    },
    "po_transactions": {
        "id", "po_id", "pay_amount", "transaction_date",
    },
    "po_status_logs": {
        "id", "purchase_order_id", "status", "changed_at", "remarks",
    },
    "purchase_requests": {
        "pr_no", "request_date", "status", "priority",
    },
}


# Foreign-key relationships — static, never changes without a migration.
# Format: child_table.child_col → parent_table.parent_col
_FK_LINES = [
    "-- Foreign keys (use these for JOINs):",
    "purchase_orders.supplier_id          → suppliers.id",
    "purchase_order_items.purchase_order_id → purchase_orders.id",
    "purchase_order_items.inventory_id    → inventories.id",
    "po_transactions.po_id                → purchase_orders.id",
    "po_status_logs.purchase_order_id     → purchase_orders.id",
    "stock_transactions.inventory_id      → inventories.id",
    "project_item.project_id              → projects.id",
    "project_item.inventory_id            → inventories.id",
    "project_products.project_id          → projects.id",
    "project_products.product_id          → products.id",
    "product_items.product_id             → products.id",
    "product_items.inventory_id           → inventories.id",
]


def _load(engine: Engine) -> str:
    table_names = tuple(_RELEVANT.keys())
    rows: List[Tuple[str, str, str]] = []
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT table_name, column_name, column_type "
                    "FROM information_schema.columns "
                    "WHERE table_schema = DATABASE() AND table_name IN :tables "
                    "ORDER BY table_name, ordinal_position"
                ),
                {"tables": table_names},
            ).fetchall()
            rows = [(r[0], r[1], r[2]) for r in result]
    except Exception as e:
        print(f"[SCHEMA] introspection failed: {e}")
        return ""

    by_table: Dict[str, List[Tuple[str, str]]] = {}
    for table, col, ctype in rows:
        if col in _RELEVANT.get(table, set()):
            by_table.setdefault(table, []).append((col, ctype))

    if not by_table:
        return ""

    out = ["DATABASE SCHEMA (live — use ONLY these column names, never guess):"]
    for table in _RELEVANT:
        cols = by_table.get(table)
        if not cols:
            continue
        out.append(f"\n{table}:")
        for name, ctype in cols:
            out.append(f"  {name:28} {ctype}")

    out.append("")
    out.extend(_FK_LINES)
    out.append("\nIMPORTANT: never reference columns or tables not listed above.")
    return "\n".join(out)


def get_schema_text(engine: Optional[Engine] = None) -> str:
    """Return cached schema doc; loads on first call. Thread-safe."""
    global _schema_text, _initialized
    if _initialized:
        return _schema_text
    with _lock:
        if _initialized:
            return _schema_text
        if engine is None:
            from app.db.database import engine as default_engine
            engine = default_engine
        _schema_text = _load(engine)
        _initialized = True
    return _schema_text


def refresh(engine: Optional[Engine] = None) -> str:
    """Force reload after migrations."""
    global _schema_text, _initialized
    with _lock:
        if engine is None:
            from app.db.database import engine as default_engine
            engine = default_engine
        _schema_text = _load(engine)
        _initialized = True
    return _schema_text
