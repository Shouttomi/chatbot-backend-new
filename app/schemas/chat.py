from pydantic import BaseModel
from typing import Any, Dict, List, Optional

class ChatRequest(BaseModel):
    query: str
    history: Optional[List[Dict[str, Any]]] = []
    role: Optional[str] = None  # e.g. "supervisor", "admin", "manager"
    ui_filters: Optional[Dict[str, Any]] = {}