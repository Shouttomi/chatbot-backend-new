from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.db.database import get_db
from app.schemas.chat import ChatRequest

router = APIRouter(prefix="/supplier", tags=["Supplier"])

@router.post("/suggest")
def suggest_supplier(request: ChatRequest, db: Session = Depends(get_db)):

    q = request.query.strip().lower()

    if len(q) < 1:
        return {"items": []}

    rows = db.execute(
        text("""
            SELECT id, supplier_name, supplier_code
            FROM suppliers
            WHERE
                LOWER(supplier_name) LIKE LOWER(:q)
                OR LOWER(supplier_code) LIKE LOWER(:q)
            ORDER BY supplier_name
            LIMIT 20
        """),
        {"q": f"%{q}%"}
    ).fetchall()

    return {
        "items": [
            {
                "id": r[0],
                "name": r[1],
                "code": r[2]
            }
            for r in rows
        ]
    }
