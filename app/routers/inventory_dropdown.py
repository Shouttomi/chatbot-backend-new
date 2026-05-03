from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.db.database import get_db

router = APIRouter(prefix="/inventory", tags=["Supplier Smart"])


@router.post("/supplier-search")
def supplier_smart_search(query: str, db: Session = Depends(get_db)):

    q = str(query).lower().strip()

    # ================= FIND SUPPLIERS =================
    if q.isdigit():
        suppliers = db.execute(
            text("""
                SELECT id, supplier_name, supplier_code, email, gstin
                FROM suppliers
                WHERE id = :id
                LIMIT 10
            """),
            {"id": int(q)}
        ).fetchall()
    else:
        suppliers = db.execute(
            text("""
                SELECT id, supplier_name, supplier_code, email, gstin
                FROM suppliers
                WHERE LOWER(supplier_name) LIKE LOWER(:q)
                   OR LOWER(supplier_code) LIKE LOWER(:q)
                ORDER BY supplier_name
                LIMIT 10
            """),
            {"q": f"%{q}%"}
        ).fetchall()

    # -------- MULTIPLE SUPPLIERS → DROPDOWN --------
    if len(suppliers) > 1:
        return {
            "type": "dropdown",
            "items": [
                {
                    "id": s.id,
                    "name": s.supplier_name,
                    "code": s.supplier_code
                }
                for s in suppliers
            ]
        }

    # -------- NONE --------
    if not suppliers:
        return {"message": "Supplier not found"}

    # -------- SINGLE SUPPLIER --------
    supplier = suppliers[0]
    supplier_id = supplier.id

    # ================= GET ALL INVENTORIES =================
    inventories = db.execute(
        text("""
            SELECT id, name, classification
            FROM inventories
            ORDER BY name
        """)
    ).fetchall()

    finish_total = 0
    semi_finish_total = 0
    items = []

    # ================= LOOP INVENTORIES =================
    for inv in inventories:

        txns = db.execute(
            text("""
                SELECT txn_type, ref_type, quantity
                FROM stock_transactions
                WHERE inventory_id = :inv_id
                  AND supplier_id = :supplier_id
            """),
            {"inv_id": inv.id, "supplier_id": supplier_id}
        ).fetchall()

        in_qty = 0
        out_qty = 0
        finish_in = 0
        machining_out = 0

        for t in txns:
            txn_type = (t.txn_type or "").lower()
            ref_type = (t.ref_type or "").lower()
            qty = float(t.quantity or 0)

            if txn_type == "in" and ref_type != "finish":
                in_qty += qty

            if txn_type == "out" and ref_type != "machining":
                out_qty += qty

            if txn_type == "in" and ref_type == "finish":
                finish_in += qty

            if txn_type == "out" and ref_type == "machining":
                machining_out += qty

        classification = (inv.classification or "").upper().strip()

        machining_stock = 0
        semi_finish_stock = 0
        finish_stock = 0
        total = in_qty - out_qty

        # ================= CLASSIFICATION LOGIC =================
        if classification in ["", "FINISH", "NULL"]:
            finish_stock = in_qty - out_qty

        elif classification == "SEMI_FINISH":
            final_mc = machining_out - finish_in
            final_fnsh = finish_in - out_qty
            semifinish = in_qty - out_qty - final_mc - final_fnsh

            machining_stock = final_mc
            finish_stock = final_fnsh
            semi_finish_stock = semifinish

        else:
            finish_stock = in_qty - finish_in

        # ================= FILTER ZERO STOCK =================
        if total != 0:

            if classification == "FINISH":
                finish_total += total
            else:
                semi_finish_total += total

            items.append({
                "inventory_id": inv.id,
                "name": inv.name,
                "stock": total
            })

    # ================= FINAL RESPONSE =================
    return {
        "type": "result",
        "supplier": {
            "id": supplier.id,
            "name": supplier.supplier_name,
            "code": supplier.supplier_code,
            "email": supplier.email,
            "gstin": supplier.gstin
        },
        "finish_stock": finish_total,
        "semi_finish_stock": semi_finish_total,
        "items": items
    }