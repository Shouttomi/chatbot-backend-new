from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from pydantic import BaseModel
from app.db.database import get_db

router = APIRouter(prefix="/inventory", tags=["Inventory Only Mode"])


# these are the details available for inventory
# ['id', 'name', 'opening_quantity', 'min_quantity', 
# 'unit_id', 'unit', 'model', 'category_id',
#  'grade', 'height', 'width', 'thikness', '
# is_deleted', 'created_at', 'updated_at', 
# 'opening_stock', 'type', 'classification',
#  'placement', 'composition', 'diameter']#

class SearchRequest(BaseModel):
    query: str


@router.post("/search")
def inventory_search(request: SearchRequest, db: Session = Depends(get_db)):

    q = request.query.lower().strip()

    if not q:
        return {"type": "message", "message": "Please enter a search term"}

    # =====================================================
    # 🔹 SHOW ALL INVENTORY (BUTTON CLICK)
    # =====================================================
    if q == "inventory":
        inventories = db.execute(text("""
            SELECT id, name, opening_quantity,min_quantity,model
            FROM inventories
            ORDER BY name
            LIMIT 50
        """)).fetchall()

       



        return {
            "type": "dropdown",
            "message": "Select inventory:",
            "items": [
                {
                    "id": inv.id,
                    "name": inv.name,
                    "opening_quantity": inv.opening_quantity,
                    "min_quantity": inv.min_quantity,
                    "model": inv.model

                }
                for inv in inventories
            ]
        }

    # =====================================================
    # 🔹 ID CLICK → SHOW SUPPLIERS + STOCK
    # =====================================================
    if q.isdigit():

        inv = db.execute(text("""
            SELECT id, name, classification,
                   unit, placement, height, width, thikness
            FROM inventories
            WHERE id = :id
        """), {"id": int(q)}).fetchone()

        if not inv:
            return {"type": "message", "message": "Inventory not found"}

        # 🔥 SUPPLIER-WISE STOCK
        supplier_data = db.execute(text("""
            SELECT 
                s.id as supplier_id,
                s.supplier_name,
                SUM(
                    CASE 
                        WHEN LOWER(t.txn_type) = 'in' THEN t.quantity
                        ELSE -t.quantity
                    END
                ) as stock
            FROM stock_transactions t
            JOIN suppliers s ON t.supplier_id = s.id
            WHERE t.inventory_id = :inv_id
            GROUP BY s.id, s.supplier_name
            HAVING stock != 0
            ORDER BY s.supplier_name
        """), {"inv_id": inv.id}).fetchall()

        return {
            "type": "result",
            "inventory": {
                "id": inv.id,
                "name": inv.name,
                "classification": inv.classification,
                "unit": inv.unit
            },
            "suppliers": [
                {
                    "id": s.supplier_id,
                    "name": s.supplier_name,
                    "stock": float(s.stock)
                }
                for s in supplier_data
            ]
        }

    # =====================================================
    # 🔹 SEARCH INVENTORY ONLY (NO SUPPLIER)
    # =====================================================
    words = q.split()

    conditions = " OR ".join(
        [f"LOWER(name) LIKE :w{i}" for i in range(len(words))]
    )

    query_sql = f"""
        SELECT id, name, opening_quantity,min_quantity,model
        FROM inventories
        WHERE {conditions}
        ORDER BY name
        LIMIT 20
    """

    params = {f"w{i}": f"%{word}%" for i, word in enumerate(words)}

    inventories = db.execute(text(query_sql), params).fetchall()


    inventories_test = db.execute(text("""
            SELECT *
            FROM inventories
            ORDER BY name
            LIMIT 50
        """)).fetchall()

    for inv in inventories_test:
     print(inv._mapping.keys())




    if not inventories:
        return {"type": "message", "message": "Inventory not found"}

    return {
        "type": "dropdown",
        "message": "Select inventory:",
        "items": [
            {
                "id": inv.id,
                "name": inv.name,
                "opening_quantity": inv.opening_quantity,
                "min_quantity": inv.min_quantity,
                "model": inv.model

            }
            for inv in inventories
        ]
    }

# =====================================================
# 🔹 INVENTORY DETAILS BY ID (NEW ROUTE)
# =====================================================
@router.get("/details/{inventory_id}")
def inventory_details(
    inventory_id: int,
    db: Session = Depends(get_db)
):

    # 🔹 Get inventory details
    inv = db.execute(text("""
        SELECT id, name, classification,
               unit, placement, height, width, thikness, grade
        FROM inventories
        WHERE id = :id
    """), {"id": inventory_id}).fetchone()

    inventory = db.execute(text("""
    SELECT *
    FROM inventories
    WHERE id = :id
    """), {"id": inventory_id}).fetchone()

    print("FULL INVENTORY:", inventory._mapping)
    print("FIELDS:", list(inventory._mapping.keys()))

    

    

    if not inv:
        return {"type": "message", "message": "Inventory not found"}

    # 🔥 Supplier-wise stock
    supplier_data = db.execute(text("""
        SELECT 
            s.id as supplier_id,
            s.supplier_name,
            SUM(
                CASE 
                    WHEN LOWER(t.txn_type) = 'in' THEN t.quantity
                    ELSE -t.quantity
                END
            ) as stock
        FROM stock_transactions t
        JOIN suppliers s ON t.supplier_id = s.id
        WHERE t.inventory_id = :inv_id
        GROUP BY s.id, s.supplier_name
        HAVING stock != 0
        ORDER BY s.supplier_name
    """), {"inv_id": inventory_id}).fetchall()

    return {
        "type": "result",

        # 🔥 RETURN ALL FIELDS AUTOMATICALLY
        "inventory": dict(inv._mapping),

        "suppliers": [
            {
                "id": s.supplier_id,
                "name": s.supplier_name,
                "stock": float(s.stock)
            }
            for s in supplier_data
        ],

        "total_stock": sum([float(s.stock) for s in supplier_data])
    }