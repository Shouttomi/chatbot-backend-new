from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.db.database import get_db

router = APIRouter(prefix="/supplier", tags=["Supplier"])


# =====================================================
# 🔹 1. SUPPLIER SEARCH (DROPDOWN)
# =====================================================
@router.get("/search")
def search_suppliers(
    q: str = Query(..., description="Type supplier name or code"),
    db: Session = Depends(get_db)
):
    rows = db.execute(
        text("""
            SELECT id, supplier_name, supplier_code, city
            FROM suppliers
            WHERE
                LOWER(supplier_name) LIKE LOWER(:q)
                OR LOWER(supplier_code) LIKE LOWER(:q)
            ORDER BY supplier_name
            LIMIT 20
        """),
        {"q": f"%{q}%"}
    ).fetchall()

    if not rows:
        return {"type": "message", "message": "Supplier not found"}

    return {
        "type": "dropdown",
        "items": [
            {
                "id": r.id,
                "name": r.supplier_name,
                "code": r.supplier_code,
                "city": r.city
            }
            for r in rows
        ]
    }


# =====================================================
# 🔹 2. SUPPLIER DETAILS + INVENTORY
# =====================================================
@router.get("/details/{supplier_id}")
def supplier_details(
    supplier_id: int,
    db: Session = Depends(get_db)
):
    # 🔥 GET FULL SUPPLIER DATA
    supplier = db.execute(
        text("""
            SELECT *
            FROM suppliers
            WHERE id = :id
            LIMIT 1
        """),
        {"id": supplier_id}
    ).fetchone()

    print(supplier._mapping, "supplier details")

    if not supplier:
        raise HTTPException(status_code=404, detail="Supplier not found")

    # 🔥 INVENTORY FOR THIS SUPPLIER
    inventories = db.execute(
        text("""
            SELECT 
                i.id,
                i.name,
                SUM(
                    CASE 
                        WHEN LOWER(t.txn_type) = 'in' THEN t.quantity
                        ELSE -t.quantity
                    END
                ) as stock
            FROM inventories i
            JOIN stock_transactions t ON i.id = t.inventory_id
            WHERE t.supplier_id = :sid
            GROUP BY i.id, i.name
            HAVING stock != 0
            ORDER BY i.name
        """),
        {"sid": supplier_id}
    ).fetchall()

    print(inventories, "hello this side") #no inventory for the supplier in db

    total_stock = sum([float(inv.stock) for inv in inventories])

    return {
        "type": "dropdown",
        "supplier": {
            "id": supplier.id,
            "name": supplier.supplier_name,
            "code": supplier.supplier_code,
            "email": getattr(supplier, "email", None),
            "gstin": getattr(supplier, "gstin", None),
            "mobile": getattr(supplier, "mobile", None),
            "city": getattr(supplier, "city", None),
            "state": getattr(supplier, "state", None),
            "address": getattr(supplier, "address", None)
        },
        "total_stock": total_stock,
        "message": f"Select inventory for {supplier.supplier_name}",
        "items": [
            {
                "id": inv.id,
                "name": inv.name,
                "stock": float(inv.stock)
            }
            for inv in inventories
        ]
    }


# =====================================================
# 🔹 3. SUPPLIER DETAILS BY CODE
# =====================================================
@router.get("/by-code")
def supplier_by_code(
    supplier_code: str = Query(...),
    db: Session = Depends(get_db)
):
    supplier = db.execute(
        text("""
            SELECT *
            FROM suppliers
            WHERE LOWER(supplier_code) = LOWER(:code)
            LIMIT 1
        """),
        {"code": supplier_code}
    ).fetchone()

    if not supplier:
        raise HTTPException(status_code=404, detail="Supplier not found")

    # 🔥 INVENTORY FOR THIS SUPPLIER
    inventories = db.execute(
        text("""
            SELECT 
                i.id,
                i.name,
                SUM(
                    CASE 
                        WHEN LOWER(t.txn_type) = 'in' THEN t.quantity
                        ELSE -t.quantity
                    END
                ) as stock
            FROM inventories i
            JOIN stock_transactions t ON i.id = t.inventory_id
            WHERE t.supplier_id = :sid
            GROUP BY i.id, i.name
            HAVING stock != 0
            ORDER BY i.name
        """),
        {"sid": supplier.id}
    ).fetchall()

    return {
        "type": "dropdown",
        "supplier": {
            "id": supplier.id,
            "name": supplier.supplier_name,
            "code": supplier.supplier_code,
            "email": getattr(supplier, "email", None),
            "gstin": getattr(supplier, "gstin", None),
            "mobile": getattr(supplier, "mobile", None),
            "city": getattr(supplier, "city", None),
            "state": getattr(supplier, "state", None),
            "address": getattr(supplier, "address", None)
        },
        "message": f"Select inventory for {supplier.supplier_name}",
        "items": [
            {
                "id": inv.id,
                "name": inv.name,
                "stock": float(inv.stock)
            }
            for inv in inventories
        ]
    }