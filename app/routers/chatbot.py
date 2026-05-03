from asyncio import threads
import time
from datetime import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
#from app.services.love_brain import check_license
from app.db.database import get_db
from app.schemas.chat import ChatRequest
import re
import numpy as np
import faiss
#from sentence_transformers import SentenceTransformer
from fastembed import TextEmbedding
import difflib
#from rapidfuzz import process, fuzz
#import jellyfish
import json
import os
from app.services.ollama_engine import ask_ollama

router = APIRouter(prefix="/chatbot", tags=["Chatbot"])

# ==========================================
# 🛡️ MEWAR ERP - ROLE PERMISSIONS
# ==========================================
ROLE_PERMISSIONS = {
    "supervisor": ["inventory", "project", "general_chat"],
    "sales": ["inventory", "general_chat"],
    "purchase": ["inventory", "supplier", "po", "general_chat"],
    "purchase admin": ["inventory", "supplier", "po", "financials", "general_chat"],
    "store admin": ["inventory", "po", "project", "general_chat"],
    "store department": ["inventory", "general_chat"],
    "hod": ["inventory", "project", "supplier", "po", "financials", "general_chat"],
    "hr": ["general_chat"]
}

# ==========================================
#        FAISS Setup & Model
# ==========================================
print("⏳ Loading Semantic Search Model... (10-15 seconds)")

# ==========================================
# 🧠 FAISS SEMANTIC SEARCH ENGINE SETUP
# ==========================================
semantic_model = None
inv_names_list = []
sup_names_list = []
inv_faiss_index = None
sup_faiss_index = None
is_faiss_loaded = False
proj_names_list = []
proj_faiss_index = None

# 1. MODEL & KEYS:
#generic_inv_words = set(["item", "items", "stock", "maal", "inventory", "nag", "quantity", "qty", "piece", "pieces"])
# generic_inv_words = set()
# generic_sup_words = set()   # 🆕
# generic_proj_words = set()  # 🆕 

import time  # 👈 Sabse upar ye zaroor add karna

def load_faiss_once(db: Session):
    global semantic_model, inv_names_list, sup_names_list, inv_faiss_index, sup_faiss_index, is_faiss_loaded, proj_names_list, proj_faiss_index
    
    if is_faiss_loaded: return
    
    print("⏳ Loading Semantic Search Model... (threads=1)")
    semantic_model = TextEmbedding('BAAI/bge-small-en-v1.5', threads=1)
    
    # 🔄 RETRY LOGIC: Yahan se loop shuru hota hai
    for attempt in range(3):
        try:
            print(f"🛠️ Building FAISS Memory (Attempt {attempt+1}/3)...")

            # 1. Inventory Indexing
            inv_data = db.execute(text("SELECT name FROM inventories WHERE name IS NOT NULL")).fetchall()
            inv_names_list = [row[0] for row in inv_data if row[0]]
            if inv_names_list:
                inv_embeddings = np.array(list(semantic_model.embed(inv_names_list, batch_size=50))).astype('float32')
                inv_faiss_index = faiss.IndexFlatL2(inv_embeddings.shape[1])
                inv_faiss_index.add(inv_embeddings)

            # 2. Supplier Indexing
            sup_data = db.execute(text("SELECT supplier_name FROM suppliers WHERE supplier_name IS NOT NULL")).fetchall()
            sup_names_list = [row[0] for row in sup_data if row[0]]
            if sup_names_list:
                sup_embeddings = np.array(list(semantic_model.embed(sup_names_list, batch_size=50))).astype('float32')
                sup_faiss_index = faiss.IndexFlatL2(sup_embeddings.shape[1])
                sup_faiss_index.add(sup_embeddings)

            # 3. Project Indexing
            proj_data = db.execute(text("SELECT name FROM projects WHERE name IS NOT NULL AND is_deleted = 0")).fetchall()
            proj_names_list = [row[0] for row in proj_data if row[0]]
            if proj_names_list:
                proj_embeddings = np.array(list(semantic_model.embed(proj_names_list, batch_size=50))).astype('float32')
                proj_faiss_index = faiss.IndexFlatL2(proj_embeddings.shape[1])
                proj_faiss_index.add(proj_embeddings)

            # ✅ Agar yahan tak code pahunch gaya, toh success!
            is_faiss_loaded = True
            print(f"✅ FAISS Ready! Indexed {len(inv_names_list)} Items, {len(sup_names_list)} Suppliers & {len(proj_names_list)} Projects.")
            return # Loop se bahar nikal jao

        except Exception as e:
            print(f"⚠️ Attempt {attempt+1} failed: {e}")
            
            # 👇 NAYI LINE: Kharaab connection ko reset karne ke liye
            try:
                db.rollback() 
            except:
                pass
                
            if attempt < 2: # Agar 3rd attempt nahi hai, toh ruko
                print("🔄 Database reset done. Retrying in 3 seconds...")
                time.sleep(3)
            else:
                print("❌ All attempts failed. FAISS load error.")

def smart_match(query_text, category="inventory"):
    if not query_text or len(query_text) < 2 or not is_faiss_loaded: return query_text
    try:
       # query_vector = semantic_model.encode([query_text]).astype('float32')
        query_vector = np.array(list(semantic_model.embed([query_text]))).astype('float32') # <-- Naya logic
        if category == "inventory" and inv_faiss_index:
            distances, indices = inv_faiss_index.search(query_vector, 3)
            if distances[0][0] < 0.7:  # Threshold for inventory matching (Stricter)   
                return inv_names_list[indices[0][0]]
                
        elif category == "supplier" and sup_faiss_index:
            distances, indices = sup_faiss_index.search(query_vector, 3)
            if distances[0][0] < 0.7:
                return sup_names_list[indices[0][0]]

        # 🟢 FIX 2: Project ke liye FAISS check add kiya
        elif category == "project" and proj_faiss_index:
            distances, indices = proj_faiss_index.search(query_vector, 3)
            if distances[0][0] < 1.0:
                return proj_names_list[indices[0][0]]
                
    except Exception as e: 
        pass
    
    return query_text
# ==========================================

# 🛠️ THE SLANG LIBRARY
def translate_slang(text: str):
    slang_map = {
        r'\bmaal\b': 'inventory',
        r'\bstock\b': 'inventory',
        r'\bkharcha\b': 'budget',
        r'\brokra\b': 'balance_amount',
        r'\bpaisa\b': 'amount',
        r'\bkitna\b': 'total_stock',
        r'\bitem\b': 'inventory',
    }
    for slang, official in slang_map.items():
        text = re.sub(slang, official, text, flags=re.IGNORECASE)
    return text

# 🌟 advanced_intent_detector
def advanced_intent_detector(query: str):
    q = query.lower()
    score = {"po_search": 0, "supplier_search": 0, "project_search": 0, "search": 0}

    # 1. Scoring Logic
    po_words = ["po", "order", "orders", "purchase", "transit", "raste", "pending", "dispatch", "delivery"]
    sup_words = ["supplier", "vendor", "party", "contact", "mobile", "number", "account", "details", "profile"]
    proj_words = ["project", "site", "crusher", "running", "urgent", "completed", "refurbish"]
    inv_words = ["stock", "maal", "item", "inventory", "quantity", "kitna", "qty", "nag", "available"]

    for w in po_words: 
        if w in q: score["po_search"] += 2
    for w in sup_words: 
        if w in q: score["supplier_search"] += 2
    for w in proj_words: 
        if w in q: score["project_search"] += 2
    for w in inv_words: 
        if w in q: score["search"] += 2

    if any(w in q for w in ["stock", "maal", "kitna"]) and any(w in q for w in ["supplier", "party"]):
        score["search"] += 3 

    best_intent = max(score, key=score.get)
    return best_intent if score[best_intent] > 0 else "search"

def clean_target_ultimate(target: str):
    noise = ["dikhao", "batao", "check", "ka", "ki", "ke", "mein", "inventory", "stock", "orders", "po", "list", "mujhe", "hai", "bhai", "details", "contact"]
    words = target.split()
    cleaned = [w for w in words if w.lower() not in noise]
    return " ".join(cleaned) if cleaned else target

# --- MASTER CHATBOT LOGGER 📊 ---
def log_query_pro(user_role, query, intents, final_results, process_time):
    bot_reply = "No Response"
    if isinstance(final_results, dict) and "results" in final_results:
        for res in final_results["results"]:
            if res.get("type") == "chat":
                bot_reply = res.get("message", "")
                break 
    
    is_fail = any(w in str(bot_reply).lower() for w in ["nahi mila", "error", "samajh nahi", "maaf kijiye", "kripya", "permission nahi"])
    
    log_entry = {
        "date_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "role": user_role,
        "user_query": query,
        "intent": str(intents),
        "bot_response": bot_reply,
        "time_taken_sec": round(process_time, 2),
        "status": "Fail ❌" if is_fail else "Success ✅"
    }
    try:
        with open("chat_history.json", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"❌ File Log Error: {e}")

@router.post("/")
def chatbot(request: ChatRequest, db: Session = Depends(get_db)):
    start_time = time.time()
    load_faiss_once(db)
    raw_q = request.query.strip()
    low_q = raw_q.lower()

    # ==========================================
    # 🛡️ STEP 0: USER KA ROLE NIKALO
    # ==========================================
    user_role = (request.role or "guest").lower().strip()
    is_super = user_role in ["superadmin", "super admin"]
    allowed_perms_global = ROLE_PERMISSIONS.get(user_role, [])
    import datetime as _dt

    # ==========================================
    # 🔄 STEP 0.3: HISTORY CONTEXT EXTRACTION
    # ==========================================
    _h_history = getattr(request, "history", []) or []
    _h_ctx_type = None    # "project", "supplier", "po", "inventory"
    _h_ctx_entity = None  # Last named entity from previous bot response

    if _h_history:
        _last_bot = next((m.get("content", "") for m in reversed(_h_history)
                          if m.get("role") == "assistant"), "")
        if _last_bot:
            _lb = _last_bot.lower()
            if re.search(r'\d+\s+projects', _lb):
                _h_ctx_type = "project"
            elif re.search(r'\d+\s+orders', _lb) or re.search(r'purchase order', _lb):
                _h_ctx_type = "po"
            elif (re.search(r'\d+\s+suppliers', _lb) or
                  (re.search(r'mil gaye\b', _lb) and 'project' not in _lb and 'order' not in _lb)):
                _h_ctx_type = "supplier"
            elif re.search(r'(stock|item).*(mil gaya|data)', _lb):
                _h_ctx_type = "inventory"

            _bold_all = re.findall(r'\*\*([A-Z][A-Za-z0-9 \-&.,/()]{1,50}?)\*\*', _last_bot)
            _skip_bold = {"total", "orders", "projects", "pending", "advance", "stock", "items",
                          "suppliers", "po", "units", "rs", "result", "purchase order", "payment",
                          "arawali", "dcl"}  # lowercase check happens below
            _ctx_names = [n for n in _bold_all
                          if n.lower() not in _skip_bold
                          and not re.match(r'^[\d\s.,₹Rs:\-]+$', n)
                          and len(n) > 2]
            _h_ctx_entity = _ctx_names[0] if _ctx_names else None

    # 🔗 PRONOUN RESOLUTION: "aur inka/uski/unka X" → "EntityName ka X"
    _pronoun_m = re.match(r'^(?:aur\s+)?(?:in|un|is|us)(?:ka|ki|ke)\s+(.+)', low_q)
    if _pronoun_m and _h_ctx_entity:
        _what = _pronoun_m.group(1).strip()
        raw_q = f"{_h_ctx_entity} ka {_what}"
        low_q = raw_q.lower()

    # ==========================================
    # 🚀 STEP 0.5: PRE-AI DETERMINISTIC FAST-TRACKS
    # ==========================================

    def _po_card(po):
        return {"type": "po", "po_no": str(po.po_number), "supplier": str(po.supplier_name),
                "date": str(po.po_date), "total": float(po.total_amount or 0),
                "advance": float(po.advance_amount or 0), "balance": float(po.balance_amount or 0),
                "status": str(po.status).capitalize()}

    def _proj_card(p):
        sn = str(p.status).lower()
        stage = getattr(p, 'stage', "100%" if sn == "completed" else "50%" if sn == "in_progress" else "0%")
        tag = "Refurbished" if getattr(p, 'refurbish', 0) == 1 else "New Machine"
        return {"type": "project", "project_name": str(p.name),
                "category": f"{tag} | {str(p.status).replace('_',' ').capitalize()}",
                "amount": float(p.budget or 0),
                "start_date": str(p.start_date) if p.start_date else "N/A",
                "end_date": str(p.end_date or p.deadline or "N/A"),
                "comments": str(p.comment or ""), "stage": stage,
                "priority": str(p.priority or "Normal").upper()}

    # 🏭 SUPPLIER COUNT
    if re.search(r'(kitne|kitni|total|count|how many).{0,20}(supplier|vendor|party|parties|compan)', low_q) or \
       re.search(r'(supplier|vendor|party|parties).{0,20}(kitne|kitni|total|count)', low_q):
        if is_super or "supplier" in allowed_perms_global:
            c = db.execute(text("SELECT COUNT(*) FROM suppliers")).scalar()
            return {"results": [{"type": "chat", "message": f"Hamare system mein total **{c} suppliers/parties** registered hain! 🏭"}]}
        return {"results": [{"type": "chat", "message": f"Aapka role '{user_role.title()}' hai. Supplier details dekhne ki permission nahi hai. 🛑"}]}

    # 📁 PROJECT COUNT
    if re.search(r'(kitne|total|count|how many).{0,15}project', low_q) or \
       re.search(r'project.{0,15}(kitne|total|count|hain|hai)', low_q):
        if is_super or "project" in allowed_perms_global:
            c = db.execute(text("SELECT COUNT(*) FROM projects WHERE is_deleted=0")).scalar()
            return {"results": [{"type": "chat", "message": f"System mein total **{c} projects** hain! 📁"}]}
        return {"results": [{"type": "chat", "message": f"Aapka role '{user_role.title()}' hai. Projects dekhne ki permission nahi hai. 🛑"}]}

    # 📦 INVENTORY COUNT
    if re.search(r'(kitni?|total|count|how many).{0,20}(item|inventory|maal|cheez|product)', low_q) or \
       re.search(r'(item|inventory|maal).{0,20}(kitni?|total|count|kitna)', low_q):
        if is_super or "inventory" in allowed_perms_global:
            c = db.execute(text("SELECT COUNT(*) FROM inventories")).scalar()
            return {"results": [{"type": "chat", "message": f"Hamare system mein total **{c} inventory items** hain! 📦"}]}
        return {"results": [{"type": "chat", "message": f"Aapka role '{user_role.title()}' hai. Inventory dekhne ki permission nahi hai. 🛑"}]}

    # 📉 LOWEST STOCK ITEM
    if any(p in low_q for p in ["sabse kam stock", "sabse kum stock", "minimum stock", "lowest stock",
                                  "least stock", "sabse chhota stock", "sabse kam maal", "sabse thoda stock",
                                  "sabse kam wala item", "kam stock wala"]):
        if is_super or "inventory" in allowed_perms_global:
            res = db.execute(text("""
                SELECT i.id, i.name, i.classification, i.placement,
                       COALESCE(SUM(CASE WHEN LOWER(t.txn_type)='in' THEN t.quantity ELSE -t.quantity END), 0) as total
                FROM inventories i
                JOIN stock_transactions t ON i.id=t.inventory_id
                GROUP BY i.id, i.name, i.classification, i.placement
                ORDER BY total ASC LIMIT 1
            """)).fetchone()
            if res:
                cls = str(res.classification).upper() if res.classification else "FINISH"
                f = float(res.total) if "FINISH" in cls else 0
                sf = float(res.total) if "SEMI" in cls else 0
                m = float(res.total) if "MACHINING" in cls else 0
                return {"results": [
                    {"type": "chat", "message": f"Sabse kam stock **{res.name}** ka hai — **{float(res.total):,.0f} units**! 📉"},
                    {"type": "result", "inventory": {"id": res.id, "name": res.name, "category": cls, "placement": res.placement or "N/A"},
                     "total_stock": float(res.total), "finish_stock": f, "semi_finish_stock": sf, "machining_stock": m}
                ]}
            return {"results": [{"type": "chat", "message": "Koi bhi item stock transactions mein nahi mila. 🤷"}]}
        return {"results": [{"type": "chat", "message": f"Aapka role '{user_role.title()}' hai. Inventory dekhne ki permission nahi hai. 🛑"}]}

    # 🏆 HIGHEST STOCK ITEM
    if any(p in low_q for p in ["sabse jyada stock", "sabse zyada stock", "highest stock",
                                  "max stock", "sabse jyada maal", "sabse bada stock", "most stock",
                                  "sabse adhik stock"]):
        if is_super or "inventory" in allowed_perms_global:
            res = db.execute(text("""
                SELECT i.id, i.name, i.classification, i.placement,
                       COALESCE(SUM(CASE WHEN LOWER(t.txn_type)='in' THEN t.quantity ELSE -t.quantity END), 0) as total
                FROM inventories i LEFT JOIN stock_transactions t ON i.id=t.inventory_id
                GROUP BY i.id, i.name, i.classification, i.placement
                HAVING total > 0 ORDER BY total DESC LIMIT 1
            """)).fetchone()
            if res:
                cls = str(res.classification).upper() if res.classification else "FINISH"
                f = float(res.total) if "FINISH" in cls else 0
                sf = float(res.total) if "SEMI" in cls else 0
                m = float(res.total) if "MACHINING" in cls else 0
                return {"results": [
                    {"type": "chat", "message": f"Sabse zyada stock **{res.name}** ka hai — **{float(res.total):,.0f} units**! 🏆"},
                    {"type": "result", "inventory": {"id": res.id, "name": res.name, "category": cls, "placement": res.placement or "N/A"},
                     "total_stock": float(res.total), "finish_stock": f, "semi_finish_stock": sf, "machining_stock": m}
                ]}
            return {"results": [{"type": "chat", "message": "Abhi kisi bhi item ka positive stock nahi hai system mein. 🤷"}]}
        return {"results": [{"type": "chat", "message": f"Aapka role '{user_role.title()}' hai. Inventory dekhne ki permission nahi hai. 🛑"}]}

    # 🧾 BIGGEST PO (with optional supplier filter: "X ka sabse bada order")
    if any(p in low_q for p in ["sabse bada po", "sabse bada order", "biggest po", "biggest order",
                                  "largest po", "highest po", "maximum po", "sabse bada purchase"]):
        if is_super or "po" in allowed_perms_global:
            _bf_sup = re.match(r'^(.+?)\s+ka\s+sabse\s+(?:bada?|bara?)\s+(?:po|order|purchase)', low_q)
            if _bf_sup:
                _bfsn = _bf_sup.group(1).strip()
                po = db.execute(text(
                    "SELECT p.*, s.supplier_name FROM purchase_orders p JOIN suppliers s ON p.supplier_id=s.id "
                    "WHERE LOWER(s.supplier_name) LIKE :n ORDER BY p.total_amount DESC LIMIT 1"
                ), {"n": f"%{_bfsn}%"}).fetchone()
                if po:
                    return {"results": [
                        {"type": "chat", "message": f"**{po.supplier_name}** ka sabse bada Purchase Order ye hai! 🏆"},
                        _po_card(po)
                    ]}
                return {"results": [{"type": "chat", "message": f"**{_bfsn.title()}** ke koi orders nahi mile. 🤷"}]}
            else:
                po = db.execute(text("SELECT p.*, s.supplier_name FROM purchase_orders p JOIN suppliers s ON p.supplier_id=s.id ORDER BY p.total_amount DESC LIMIT 1")).fetchone()
                if po:
                    return {"results": [
                        {"type": "chat", "message": f"Sabse bada Purchase Order **{po.supplier_name}** ka hai! 🏆"},
                        _po_card(po)
                    ]}
                return {"results": [{"type": "chat", "message": "Koi bhi PO nahi mila system mein. 🤷"}]}
        return {"results": [{"type": "chat", "message": f"Aapka role '{user_role.title()}' hai. PO dekhne ki permission nahi hai. 🛑"}]}

    # 🧾 SMALLEST PO
    if any(p in low_q for p in ["sabse chota po", "sabse chhota po", "smallest po", "lowest po",
                                  "sabse kam po", "minimum po", "sabse chhota order", "sabse chota order"]):
        if is_super or "po" in allowed_perms_global:
            po = db.execute(text("SELECT p.*, s.supplier_name FROM purchase_orders p JOIN suppliers s ON p.supplier_id=s.id WHERE p.total_amount > 0 ORDER BY p.total_amount ASC LIMIT 1")).fetchone()
            if po:
                return {"results": [
                    {"type": "chat", "message": f"Sabse chhota Purchase Order **{po.supplier_name}** ka hai! 📉"},
                    _po_card(po)
                ]}
            return {"results": [{"type": "chat", "message": "Koi bhi PO nahi mila system mein. 🤷"}]}
        return {"results": [{"type": "chat", "message": f"Aapka role '{user_role.title()}' hai. PO dekhne ki permission nahi hai. 🛑"}]}

    # 💰 POs WITH ADVANCE PAYMENT
    if any(p in low_q for p in ["advance diya", "advance payment", "advance wale", "jisme advance",
                                  "po with advance", "advance kiya", "advance paid", "advance hua",
                                  "jinka advance", "jinka advance diya"]):
        if is_super or "po" in allowed_perms_global:
            adv_pos = db.execute(text(
                "SELECT p.*, s.supplier_name FROM purchase_orders p JOIN suppliers s ON p.supplier_id=s.id "
                "WHERE p.advance_amount > 0 ORDER BY p.advance_amount DESC LIMIT 50"
            )).fetchall()
            if adv_pos:
                total_adv = sum(float(p.advance_amount or 0) for p in adv_pos)
                results = [{"type": "chat", "message": f"**{len(adv_pos)} orders** mein advance payment ki gayi hai. Total advance: **Rs.{total_adv:,.2f}** 💰"}]
                results.extend(_po_card(po) for po in adv_pos)
                return {"results": results}
            return {"results": [{"type": "chat", "message": "Koi bhi PO nahi mila jisme advance payment ki gayi ho. 🤷"}]}
        return {"results": [{"type": "chat", "message": f"Aapka role '{user_role.title()}' hai. PO dekhne ki permission nahi hai. 🛑"}]}

    # 💰 SUPPLIER-SPECIFIC PENDING BALANCE: "Arawali ka total pending balance kitna hai"
    _sup_bal_m = re.match(r'^(.+?)\s+ka\s+(?:total\s+)?(?:pending\s+)?balance', low_q)
    if _sup_bal_m:
        _sbn = _sup_bal_m.group(1).strip()
        _sbn_skip = {"sabse", "sabse jyada", "sabse zyada", "highest", "maximum", "balance", "total", "pending"}
        if len(_sbn) > 2 and _sbn not in _sbn_skip:
            if is_super or "po" in allowed_perms_global:
                _sb_pos = db.execute(text(
                    "SELECT p.*, s.supplier_name FROM purchase_orders p JOIN suppliers s ON p.supplier_id=s.id "
                    "WHERE LOWER(s.supplier_name) LIKE :n ORDER BY p.po_date DESC"
                ), {"n": f"%{_sbn}%"}).fetchall()
                if _sb_pos:
                    _sb_total = sum(float(p.balance_amount or 0) for p in _sb_pos)
                    results = [{"type": "chat", "message": f"**{_sbn.title()}** ka total pending balance **Rs.{_sb_total:,.2f}** hai ({len(_sb_pos)} orders) 💸"}]
                    results.extend(_po_card(po) for po in _sb_pos)
                    return {"results": results}
                return {"results": [{"type": "chat", "message": f"**{_sbn.title()}** ke koi orders nahi mile. 🤷"}]}
            return {"results": [{"type": "chat", "message": f"Aapka role '{user_role.title()}' hai. PO dekhne ki permission nahi hai. 🛑"}]}

    # 💸 BALANCE THRESHOLD FILTER
    _bal_match = (re.search(r'balance\s*[>>=]\s*(\d[\d,]+)', low_q) or
                  re.search(r'(\d[\d,]+)\s*(?:se zyada|se jyada|se adhik|above|more than)\s*(?:balance|baqi|pending)', low_q) or
                  re.search(r'balance\s+(\d[\d,]+)\s+se\s+(?:zyada|jyada|adhik|upar|bada)', low_q) or
                  re.search(r'(?:balance|baqi|pending)\D{0,10}(\d[\d,]+)', low_q))
    if _bal_match:
        _thresh = int(_bal_match.group(1).replace(",", ""))
        if is_super or "po" in allowed_perms_global:
            # Optional: last month date filter
            _date_cond = ""
            _date_params = {}
            if any(w in low_q for w in ["last month", "pichhle mahine", "pichle mahine"]):
                _today = _dt.date.today()
                _first = (_today.replace(day=1) - _dt.timedelta(days=1)).replace(day=1)
                _last = _today.replace(day=1) - _dt.timedelta(days=1)
                _date_cond = " AND p.po_date BETWEEN :sd AND :ed"
                _date_params = {"sd": str(_first), "ed": str(_last)}
            _bp = db.execute(text(
                "SELECT p.*, s.supplier_name FROM purchase_orders p JOIN suppliers s ON p.supplier_id=s.id "
                f"WHERE p.balance_amount > :thresh{_date_cond} ORDER BY p.balance_amount DESC LIMIT 50"
            ), {"thresh": _thresh, **_date_params}).fetchall()
            if _bp:
                total_bal = sum(float(p.balance_amount or 0) for p in _bp)
                results = [{"type": "chat", "message": f"**{len(_bp)} orders** hain jinका balance **Rs.{_thresh:,}** se zyada hai. Total pending: **Rs.{total_bal:,.2f}** 💸"}]
                results.extend(_po_card(po) for po in _bp)
                return {"results": results}
            return {"results": [{"type": "chat", "message": f"Koi bhi order nahi mila jiska balance Rs.{_thresh:,} se zyada ho. 🤷"}]}
        return {"results": [{"type": "chat", "message": f"Aapka role '{user_role.title()}' hai. PO dekhne ki permission nahi hai. 🛑"}]}

    # 📅 LAST N ORDERS
    _last_n = (re.search(r'last\s+(\d+)\s+(?:order|po)', low_q) or
               re.search(r'pichle?\s+(\d+)\s+(?:order|po)', low_q) or
               re.search(r'(\d+)\s+(?:last|recent|pichle?)\s+(?:order|po)', low_q))
    if _last_n or re.search(r'^last\s+\d+\s+orders?$', low_q.strip()):
        if not _last_n:
            _last_n = re.search(r'(\d+)', low_q)
        _n = int(_last_n.group(1)) if _last_n else 5
        if is_super or "po" in allowed_perms_global:
            _lpos = db.execute(text(
                "SELECT p.*, s.supplier_name FROM purchase_orders p JOIN suppliers s ON p.supplier_id=s.id "
                "ORDER BY p.po_date DESC, p.id DESC LIMIT :l"
            ), {"l": _n}).fetchall()
            if _lpos:
                results = [{"type": "chat", "message": f"Ye rahe last **{len(_lpos)} orders**: 📄"}]
                results.extend(_po_card(po) for po in _lpos)
                return {"results": results}
            return {"results": [{"type": "chat", "message": "Koi bhi order nahi mila. 🤷"}]}
        return {"results": [{"type": "chat", "message": f"Aapka role '{user_role.title()}' hai. PO dekhne ki permission nahi hai. 🛑"}]}

    # 📅 IS MONTH KE PROJECTS
    if any(p in low_q for p in ["is month", "iss month", "is mahine", "this month",
                                  "current month", "aaj ka mahina", "chalte mahine"]):
        if is_super or "project" in allowed_perms_global:
            _today = _dt.date.today()
            _ms = _today.replace(day=1).strftime('%Y-%m-%d')
            _me = _today.strftime('%Y-%m-%d')
            _mprojs = db.execute(text(
                "SELECT * FROM projects WHERE is_deleted=0 "
                "AND start_date<=:me AND (end_date>=:ms OR deadline>=:ms OR LOWER(status)='in_progress') "
                "ORDER BY id DESC"
            ), {"ms": _ms, "me": _me}).fetchall()
            if _mprojs:
                results = [{"type": "chat", "message": f"Is mahine **{len(_mprojs)} projects** active/running hain: 📅"}]
                results.extend(_proj_card(p) for p in _mprojs)
                return {"results": results}
            return {"results": [{"type": "chat", "message": "Is mahine koi project active nahi mila. 🤷"}]}
        return {"results": [{"type": "chat", "message": f"Aapka role '{user_role.title()}' hai. Projects dekhne ki permission nahi hai. 🛑"}]}

    # 📅 LAST MONTH PROJECTS
    if any(p in low_q for p in ["pichle month", "pichhle month", "pichle mahine", "pichhle mahine",
                                  "last month project", "last month ke project", "pichle mahine ke project"]):
        if is_super or "project" in allowed_perms_global:
            _today = _dt.date.today()
            _lme = _today.replace(day=1) - _dt.timedelta(days=1)
            _lms = _lme.replace(day=1)
            _lprojs = db.execute(text(
                "SELECT * FROM projects WHERE is_deleted=0 "
                "AND start_date<=:me AND (end_date>=:ms OR deadline>=:ms OR LOWER(status)='in_progress') "
                "ORDER BY id DESC"
            ), {"ms": str(_lms), "me": str(_lme)}).fetchall()
            if _lprojs:
                results = [{"type": "chat", "message": f"Pichle mahine **{len(_lprojs)} projects** active/running the: 📅"}]
                results.extend(_proj_card(p) for p in _lprojs)
                return {"results": results}
            return {"results": [{"type": "chat", "message": "Pichle mahine koi project active nahi tha. 🤷"}]}
        return {"results": [{"type": "chat", "message": f"Aapka role '{user_role.title()}' hai. Projects dekhne ki permission nahi hai. 🛑"}]}

    # 🏃 ACTIVE/RUNNING PROJECTS (keyword-based, not AI-routed)
    if re.search(r'active\s+project|running\s+project|chalu\s+project|chal\s+rahe\s+project', low_q):
        if is_super or "project" in allowed_perms_global:
            _aprojs = db.execute(text(
                "SELECT * FROM projects WHERE is_deleted=0 AND LOWER(status)='in_progress' ORDER BY id DESC"
            )).fetchall()
            if _aprojs:
                results = [{"type": "chat", "message": f"Abhi **{len(_aprojs)} projects** active/running hain: 🏃"}]
                results.extend(_proj_card(p) for p in _aprojs)
                return {"results": results}
            return {"results": [{"type": "chat", "message": "Abhi koi bhi project 'active' status mein nahi hai. 🤷"}]}
        return {"results": [{"type": "chat", "message": f"Aapka role '{user_role.title()}' hai. Projects dekhne ki permission nahi hai. 🛑"}]}

    # 🏢 SUPPLIER ORDERS: "X ke orders" / "X ke kitne orders" / "X ki po" pattern
    _sup_order = re.search(r'^(.+?)\s+k[ei]\s+(?:kitne?\s+)?(?:order|po|purchase)', low_q)
    if _sup_order:
        _sup_name = _sup_order.group(1).strip()
        if len(_sup_name) > 2 and _sup_name not in ["sab", "saare", "sabhi", "kisi"]:
            if is_super or "po" in allowed_perms_global:
                _sp = db.execute(text(
                    "SELECT p.*, s.supplier_name FROM purchase_orders p JOIN suppliers s ON p.supplier_id=s.id "
                    "WHERE LOWER(s.supplier_name) LIKE :n ORDER BY p.po_date DESC, p.id DESC LIMIT 20"
                ), {"n": f"%{_sup_name}%"}).fetchall()
                if _sp:
                    total_bal = sum(float(p.balance_amount or 0) for p in _sp)
                    results = [{"type": "chat", "message": f"**{_sup_name.title()}** ke **{len(_sp)} orders** mile hain. Pending balance: **Rs.{total_bal:,.2f}** 📄"}]
                    results.extend(_po_card(po) for po in _sp)
                    return {"results": results}
                return {"results": [{"type": "chat", "message": f"**{_sup_name.title()}** ke koi orders nahi mile. 🤷"}]}
            return {"results": [{"type": "chat", "message": f"Aapka role '{user_role.title()}' hai. PO dekhne ki permission nahi hai. 🛑"}]}

    # 📋 PENDING PO
    if re.search(r'pending\s+po|draft\s+po|kacha\s+po|po.{0,10}pending|po.{0,10}draft', low_q):
        if is_super or "po" in allowed_perms_global:
            _ppos = db.execute(text(
                "SELECT p.*, s.supplier_name FROM purchase_orders p JOIN suppliers s ON p.supplier_id=s.id "
                "WHERE LOWER(p.status) IN ('draft','pending') ORDER BY p.po_date DESC, p.id DESC LIMIT 50"
            )).fetchall()
            if _ppos:
                results = [{"type": "chat", "message": f"**{len(_ppos)} pending/draft POs** hain: 📋"}]
                results.extend(_po_card(po) for po in _ppos)
                return {"results": results}
            return {"results": [{"type": "chat", "message": "Koi bhi pending/draft PO nahi mila. 🤷"}]}
        return {"results": [{"type": "chat", "message": f"Aapka role '{user_role.title()}' hai. PO dekhne ki permission nahi hai. 🛑"}]}

    # 🔢 PO BY PARTIAL NUMBER: "MHEL/PO/00020 wala order" or "00025 number ka PO"
    _po_num_m = (re.search(r'(mhel/po/\d+)', low_q) or
                 re.search(r'\b(0\d{3,5})\b.*(?:po|order|purchase|number|wala)', low_q))
    if _po_num_m:
        if is_super or "po" in allowed_perms_global:
            _pnum = _po_num_m.group(1)
            _pnum_pos = db.execute(text(
                "SELECT p.*, s.supplier_name FROM purchase_orders p JOIN suppliers s ON p.supplier_id=s.id "
                "WHERE LOWER(p.po_number) LIKE :n ORDER BY p.po_date DESC LIMIT 5"
            ), {"n": f"%{_pnum}%"}).fetchall()
            if _pnum_pos:
                results = [{"type": "chat", "message": f"**{_pnum.upper()}** se match karte **{len(_pnum_pos)} orders** mile: 📄"}]
                results.extend(_po_card(po) for po in _pnum_pos)
                return {"results": results}
            return {"results": [{"type": "chat", "message": f"PO number **{_pnum.upper()}** se koi order nahi mila. 🤷"}]}
        return {"results": [{"type": "chat", "message": f"Aapka role '{user_role.title()}' hai. PO dekhne ki permission nahi hai. 🛑"}]}

    # 📋 GENERIC PO LIST: "po dikhao", "orders dikhao", "saare po" etc.
    if re.search(r'^(?:saare?\s+)?po\s+(?:dikhao|batao|list|check)|^orders?\s+(?:dikhao|batao|list)|^(?:purchase\s+)?orders?\s+(?:dikhao|batao)', low_q):
        if is_super or "po" in allowed_perms_global:
            _gpos = db.execute(text(
                "SELECT p.*, s.supplier_name FROM purchase_orders p JOIN suppliers s ON p.supplier_id=s.id "
                "ORDER BY p.po_date DESC, p.id DESC LIMIT 10"
            )).fetchall()
            if _gpos:
                results = [{"type": "chat", "message": f"Ye rahe recent **{len(_gpos)} orders**: 📄"}]
                results.extend(_po_card(po) for po in _gpos)
                return {"results": results}
        return {"results": [{"type": "chat", "message": f"Aapka role '{user_role.title()}' hai. PO dekhne ki permission nahi hai. 🛑"}]}

    # 🔄 HISTORY FOLLOW-UP: Entity from previous list → search in the right DB before calling AI
    # e.g. "balaji crusher ka batao" after a project list, "Adinath Enterprises wala" after supplier dropdown
    if _h_ctx_type and _h_ctx_entity and not any(
        w in low_q for w in ["stock", "maal", "inventory", "kitna stock", "quantity"]
    ):
        _fup_noise = r'\b(ka|ki|ke|ko|batao|dikhao|detail|details|info|data|bhi|aur|wala|wali|dikhana|kya|hai|hain|mein|se)\b'
        _fup_target = re.sub(_fup_noise, '', low_q).strip()
        _fup_target = ' '.join(_fup_target.split())

        if len(_fup_target) > 2:
            if _h_ctx_type == "project" and (is_super or "project" in allowed_perms_global):
                _fup_projs = db.execute(text(
                    "SELECT * FROM projects WHERE is_deleted=0 AND LOWER(name) LIKE :n ORDER BY id DESC LIMIT 5"
                ), {"n": f"%{_fup_target}%"}).fetchall()
                if _fup_projs:
                    if len(_fup_projs) == 1:
                        p = _fup_projs[0]
                        return {"results": [{"type": "chat", "message": f"Lijiye bhai, **{p.name}** ki details 📋:"}, _proj_card(p)]}
                    results = [{"type": "chat", "message": f"haan mil gaya 👍 **{len(_fup_projs)} projects** mile hain:"}]
                    results.extend(_proj_card(p) for p in _fup_projs)
                    return {"results": results}

            elif _h_ctx_type in ("supplier", "supplier_list") and (is_super or "supplier" in allowed_perms_global):
                _fup_sups = db.execute(text(
                    "SELECT * FROM suppliers WHERE LOWER(supplier_name) LIKE :n LIMIT 5"
                ), {"n": f"%{_fup_target}%"}).fetchall()
                if _fup_sups and len(_fup_sups) == 1:
                    s = _fup_sups[0]
                    sname = str(getattr(s, 'supplier_name', 'Unknown'))
                    _sinv = db.execute(text(
                        "SELECT i.name, SUM(CASE WHEN LOWER(t.txn_type)='in' THEN t.quantity ELSE -t.quantity END) as stock "
                        "FROM inventories i JOIN stock_transactions t ON i.id=t.inventory_id "
                        "WHERE t.supplier_id=:sid GROUP BY i.id, i.name HAVING stock != 0"), {"sid": s.id}).fetchall()
                    return {"results": [
                        {"type": "chat", "message": f"ye raha 👍 **{sname}** ka profile:"},
                        {"type": "result", "supplier": {
                            "id": s.id, "name": sname,
                            "code": str(getattr(s, 'supplier_code', 'N/A') or 'N/A'),
                            "mobile": str(getattr(s, 'mobile', 'N/A') or 'N/A'),
                            "city": str(getattr(s, 'city', 'N/A') or 'N/A'),
                            "email": str(getattr(s, 'email', 'N/A') or 'N/A'),
                            "gstin": str(getattr(s, 'gstin', 'N/A') or 'N/A')},
                         "items": [{"name": str(r.name), "stock": float(r.stock)} for r in _sinv]}
                    ]}
                elif _fup_sups:
                    return {"results": [
                        {"type": "chat", "message": f"haan mil gaya 👍 {len(_fup_sups)} suppliers mile:"},
                        {"type": "dropdown", "items": [{"id": str(getattr(s, 'supplier_name', '')), "name": str(getattr(s, 'supplier_name', ''))} for s in _fup_sups]}
                    ]}

    # 👍 YES/CONFIRMATION: "haa" / "haan" / "yes" after a supplier was shown
    if re.match(r'^(haa+n?|yes|ok|okay|han\b|ji\b|theek hai|sure)\s*[.!?]*$', low_q.strip()):
        if _h_ctx_entity and _h_ctx_type == "supplier":
            if is_super or "supplier" in allowed_perms_global:
                _cs = db.execute(text("SELECT * FROM suppliers WHERE LOWER(supplier_name) LIKE :q LIMIT 1"),
                                 {"q": f"%{_h_ctx_entity.lower()}%"}).fetchone()
                if _cs:
                    _csname = str(getattr(_cs, 'supplier_name', 'Unknown'))
                    _csinv = db.execute(text(
                        "SELECT i.name, SUM(CASE WHEN LOWER(t.txn_type)='in' THEN t.quantity ELSE -t.quantity END) as stock "
                        "FROM inventories i JOIN stock_transactions t ON i.id=t.inventory_id "
                        "WHERE t.supplier_id=:sid GROUP BY i.id, i.name HAVING stock != 0"), {"sid": _cs.id}).fetchall()
                    return {"results": [
                        {"type": "chat", "message": f"ye raha 👍 **{_csname}** ka profile:"},
                        {"type": "result", "supplier": {
                            "id": _cs.id, "name": _csname,
                            "code": str(getattr(_cs, 'supplier_code', 'N/A') or 'N/A'),
                            "mobile": str(getattr(_cs, 'mobile', 'N/A') or 'N/A'),
                            "city": str(getattr(_cs, 'city', 'N/A') or 'N/A'),
                            "email": str(getattr(_cs, 'email', 'N/A') or 'N/A'),
                            "gstin": str(getattr(_cs, 'gstin', 'N/A') or 'N/A')},
                         "items": [{"name": str(r.name), "stock": float(r.stock)} for r in _csinv]}
                    ]}

    # 👋 GREETINGS / GENERAL CHAT
    if re.match(r'^(hello|hi|hey|hii+|namaste|namaskar|kya haal|kaise ho|sab theek|good morning|good evening|how are you|kya chal raha|help|kya kar sakte)\b', low_q):
        return {"results": [{"type": "chat", "message": "Haan bhai! Kya haal hai? 😊 Aap mujhse yeh pooch sakte ho:\n\n📦 **Inventory** – kisi item ka stock\n🏭 **Supplier** – party ki details\n🧾 **PO** – purchase orders\n📁 **Projects** – site status\n\nBas likhein aur main check karta hoon!"}]}

    # 🎯 STEP 1: FAST-TRACK ID
    if low_q.isdigit() and len(low_q) < 8:
        # 🔒 NAYI SECURITY: Fast-Track bhi wahi use kar payega jiske paas Inventory ka access ho
        allowed_perms = ROLE_PERMISSIONS.get(user_role, [])
        if user_role in ["superadmin", "super admin"] or "inventory" in allowed_perms:
            try:
                inv = db.execute(text("SELECT id, name, classification, placement FROM inventories WHERE id = :id"), {"id": int(low_q)}).fetchone()
                if inv:
                    stock_res = db.execute(text("SELECT SUM(CASE WHEN LOWER(txn_type) = 'in' THEN quantity ELSE -quantity END) FROM stock_transactions WHERE inventory_id = :id"), {"id": inv.id}).scalar()
                    total_qty = float(stock_res or 0)
                    cls = str(inv.classification).lower() if inv.classification else ""
                    m, f, sf = (total_qty, 0, 0) if "machining" in cls else (0, 0, total_qty) if "semi" in cls else (0, total_qty, 0)
                    return {"results": [{"type": "result", "inventory": {"id": inv.id, "name": inv.name, "category": cls.upper(), "placement": inv.placement or "N/A"}, "total_stock": total_qty, "finish_stock": f, "semi_finish_stock": sf, "machining_stock": m}]}
            except: pass
        else:
            return {"results": [{"type": "chat", "message": f"Aapka role '{user_role.title()}' hai. Aapko Item Codes (Inventory) se search karne ki permission nahi hai. 🛑"}]}

    # 🚀 STEP 2: PURE AI ENGINE
    try:
        try:
            ai_data = ask_ollama(raw_q, getattr(request, "history", []))
        except:
            time.sleep(1) # Rate limit buffer
            ai_data = ask_ollama(raw_q, getattr(request, "history", []))
            
        print("🤖 PURE AI BRAIN DECISION:", ai_data)
    except Exception as e:
        print(f"❌ AI CRASHED: {str(e)}")
        log_query_pro(user_role if 'user_role' in dir() else "unknown", raw_q, ["error"], {}, 0)
        return {"results": [{"type": "chat", "message": "Bhai, mera AI brain abhi connect nahi ho pa raha. Kripya thodi der mein try karein. 🙏"}]}

    # 🧠 3. PARSE AI DATA (Multi-Intent)
    intents = ai_data.get("intents") or []
    if "intent" in ai_data and not intents:
        intents = [ai_data["intent"]]
    if isinstance(intents, str): 
        intents = [intents]
    if not intents:
        intents = ["search"]

    original_target = str(ai_data.get("search_target") or "").strip()
    
    # 🗣️ HUMAN-LIKE FILLER MESSAGE (WTS Style)
    reasoning = ai_data.get("reasoning") or "hmm ek sec... main check karta hoon 👍"
    final_results = [{"type": "chat", "message": reasoning}]

   # 🧹 NOISE CLEANER
    noise_words = ["supplier", "vendor", "party", "details", "contact", "profile", "ki", "ka", "ke", "project", "site", "machine"]
    for word in noise_words:
        original_target = re.sub(rf'\b{word}\b', '', original_target, flags=re.IGNORECASE).strip()

    ai_data["search_target"] = original_target

    if re.match(r'^sup[-\s]?\d+$', original_target.lower()):
        if "supplier_search" not in intents: intents = ["supplier_search"]

    # ✅ 1. SABSE PEHLE FILTERS DEFINE KARO (Taki Seatbelt use kar sake)
    filters = ai_data.get("filters", {})
    ui_filters = getattr(request, "ui_filters", {}) or {}
    for key, value in ui_filters.items():
        if value: filters[key] = value
    limit = filters.get("limit", 5) or 5

    print(f"✅ FINAL ROUTER DECISION: {intents} | TARGET: {original_target}")

    # ==========================================
    # 🛡️ SMART SEATBELT (UPGRADED FOR AI FILTERS)
    # ==========================================
    is_project_list_req = any(w in low_q for w in [
        "all", "saare", "sabhi", "list", "latest", "naya", "new", 
        "running", "chalu", "progress", "completed", "khatam", "done", 
        "hold", "ruka", "pending", "refurbished", "purana", "repair",
        "urgent", "normal", "high", "priority",
        "remaining", "baki", "bache", "kitne", "kitna",
        "sabse", "bada", "highest", "biggest", "mehenga", "lowest", "chhota", "kam",
        "late", "overdue", "delay", "dikhao", "batao"
    ])

    # 🧠 NAYA AI BYPASS LOGIC: Ab ye error nahi dega kyunki 'filters' upar ban chuka hai
    if filters.get("status") or filters.get("priority"):
        is_project_list_req = True

    if not original_target:
        if "supplier_search" in intents and not any(w in low_q for w in ["all", "saare", "list"]):
            return {"results": [{"type": "chat", "message": "Bhai, kripya thoda clear batao ki aap kis company ki baat kar rahe ho? 🙂"}]}
        
        elif "project_search" in intents and not is_project_list_req:
            return {"results": [{"type": "chat", "message": "Bhai, kripya project ka naam batao, ya fir 'chalu projects', 'urgent projects' likho. 🙂"}]}

    # 🚀 TAX & ADVANCE OVERRIDE
    if any(w in low_q for w in ["tax", "gst", "cgst", "sgst", "advance", "adv"]):
        intents = ["po_search"]

    # ==================================================
    # 🛑 SECURITY CHECK 2: Main Intent/Role Checking (graceful multi-intent)
    # ==================================================
    if user_role not in ["superadmin", "super admin"]:
        allowed_perms = ROLE_PERMISSIONS.get(user_role, [])
        _perm_map = {
            "po_search": ("po", "Purchase Orders (PO)"),
            "supplier_search": ("supplier", "Supplier details"),
            "project_search": ("project", "Project details"),
            "search": ("inventory", "Stock/Inventory"),
            "inventory_search": ("inventory", "Stock/Inventory"),
            "financial_search": ("financials", "Financial details"),
        }
        _blocked_intents = []
        _allowed_intents = []
        for intent in intents:
            perm_key, label = _perm_map.get(intent, (None, None))
            if perm_key and perm_key not in allowed_perms:
                _blocked_intents.append((intent, label))
            else:
                _allowed_intents.append(intent)

        if _blocked_intents and not _allowed_intents:
            _label = _blocked_intents[0][1]
            return {"results": [{"type": "chat", "message": f"Aapka role **'{user_role.title()}'** hai. **{_label}** dekhne ki permission nahi hai. 🛑"}]}
        elif _blocked_intents:
            intents = _allowed_intents  # strip blocked intents, proceed with rest

##### ==========================================
    # 🧠 AUTO-CORRECT FILTER (TYPO HANDLER 🚀)
    # ==========================================
    smart_keywords = [
        "all", "saare", "sabhi", "pure", "list", "batao", "dikhao", "kitne", "kitna",
        "running", "chalu", "progress", "chal",
        "completed", "poora", "khatam", "done",
        "hold", "ruka", "pending", "remaining", "baki", "bache",
        "new", "naya", "budget", "paisa", "rupay", "cost", "amount", "mehenga",
        "deadline", "target", "kab tak", "time", "din",
        "stage", "percent", "status",
        "sabse", "bada", "highest", "biggest", "lowest", "chhota", "kam",
        "late", "overdue", "delay"
    ]
    
    fixed_words = []
    for word in low_q.split():
        matches = difflib.get_close_matches(word, smart_keywords, n=1, cutoff=0.8)
        fixed_words.append(matches[0] if matches else word)
            
    low_q = " ".join(fixed_words)
    # ==========================================


    # =========================================================
    # 🔄 MULTI-INTENT PROCESSING LOOP ( main code )
    # =========================================================
    for intent in intents:
        
       # ---------------------------------------------------------
        # 📁 BRANCH 1: PROJECT LOGIC (FULLY UPGRADED 🚀)
        # ---------------------------------------------------------
        if intent == "project_search":
            try:
                target = original_target.strip()
                target_lower = target.lower()
                projs = []
                
                # 🧠 1. NLP OVERRIDES (Sentence-based limits)
                # ✅ FIX 1: 'kitne', 'kitna', 'dikhao' add kiya gaya hai
                if any(w in low_q for w in ["all", "saare", "sabhi", "pure", "list", "batao", "kitne", "kitna", "dikhao"]): 
                    limit = 50
                if any(w in low_q for w in ["last", "latest", "naya", "new"]): 
                    limit = 1
                    
                # 📊 2. BOSS MODE: Highest Budget Project
                if any(w in low_q for w in ["sabse bada project", "highest budget", "sabse mehenga", "biggest project", "sabse bada"]):
                    big_proj = db.execute(text("SELECT * FROM projects WHERE is_deleted = 0 ORDER BY budget DESC LIMIT 1")).fetchone()
                    if big_proj:
                        final_results.append({"type": "chat", "message": f"🏆 **Highest Budget Project:** System ke hisaab se sabse bada project **{big_proj.name}** hai."})
                        projs = [big_proj] 
                        target = "SKIP_SEARCH" 
                
               # 🔍 3. NORMAL SEARCH & FILTERS
                if target != "SKIP_SEARCH":
                    # ✅ 1. Sabse pehle Flags ko 'False' set karo (Taki koi purana data na rahe)
                    is_refurbished = False 
                    is_time_remaining = False
                    is_overdue = False
                    
                    raw_status = str(filters.get("status") or "").lower().strip()
                    active_priority = str(filters.get("priority") or "").lower().strip()

                    # 🛠️ 2. MAPPING DICTIONARY (Updated)
                    status_mapping = {
                        "in progress": "in_progress",
                        "in_progress": "in_progress",
                        "on_hold": "hold",
                        "pending": "hold",
                        "completed": "completed",
                        "new_project": "new",
                        "new": "new",
                        "refurbished": "refurbished", # 👈 Add this
                        "remaining": "remaining",     # 👈 Add this
                        "overdue": "overdue"         # 👈 Add this
                    }
                    active_status = status_mapping.get(raw_status, raw_status)

                    # 🧠 3. AI DRIVEN LOGIC (AI Brain ke decision ko Flags mein badlo)
                    if active_status == "refurbished":
                        is_refurbished = True
                        active_status = "" 
                    elif active_status == "remaining":
                        is_time_remaining = True
                        active_status = ""
                    elif active_status == "overdue":
                        is_overdue = True
                        active_status = ""

                    # 🔹 4. NLP CHECKS (Ab ye sirf backup ki tarah kaam karenge)
                    if any(w in low_q for w in ["running", "chalu", "progress", "chal", "active", "ongoing"]):
                        active_status = "in_progress"
                    elif any(w in low_q for w in ["completed", "poora", "khatam", "done"]): 
                        active_status = "completed"
                    elif any(w in low_q for w in ["hold", "ruka", "pending"]): 
                        active_status = "hold"
                    elif any(w in low_q for w in ["new", "naya"]): 
                        active_status = "new"
                    
                    # ✅ FIX: Yahan 'is_refurbished = True' sirf 'if' ke andar hai
                    if any(w in low_q for w in ["refurbished", "purana", "repair"]): 
                        is_refurbished = True

                    # ✅ 5. TIME CHECKS (Sirf tab jab AI ne upar set na kiya ho)
                    if not is_time_remaining and any(w in low_q for w in ["remaining", "baki", "bache"]):
                        is_time_remaining = True
                        active_status = ""
                        limit = 50
                    elif not is_overdue and any(w in low_q for w in ["late", "overdue", "delay"]):
                        is_overdue = True
                        active_status = ""
                        limit = 50

                    # 🧹 4. CLEANUP: Ignore words (Inko project ka naam mat samjho)
                    ignore_words = [
                        "all", "list", "projects", "latest", "project", "site", 
                        "refurbished", "purana", "repair", "running", "chalu", "progress", 
                        "completed", "poora", "khatam", "hold", "ruka", "new", "naya", 
                        "urgent", "emergency", "high", "normal", "priority", "batao", "dikhao",
                        "kitne", "kitna", "remaining", "baki", "bache", "late", "overdue", "delay"
                    ]
                    
                    # 🛡️ SUPER FOOLPROOF CHECK: Agar target mein refurbish/purana hai, toh usey clear karo
                    if "refurb" in target_lower or "purana" in target_lower:
                        target = ""
                        limit = 50
                        is_refurbished = True # Double safety check

                    if target_lower in ignore_words:
                        target = "" 
                        limit = 50

                    # 🏗️ 5. BUILD THE QUERY
                    query = "SELECT * FROM projects WHERE is_deleted = 0"
                    params = {}

                    # Status filter
                    if active_status and active_status != "all" and active_status != "refurbished":
                        query += " AND LOWER(status) = :st"
                        params["st"] = active_status
                        
                    # Priority filter
                    if active_priority and active_priority != "all":
                        query += " AND LOWER(priority) = :pr"
                        params["pr"] = active_priority
                        
                    # Refurbished filter
                    if is_refurbished:
                        query += " AND refurbish = 1"
                    
                    # Date filter
                    if filters.get("from_date") and filters.get("to_date"):
                        query += " AND start_date BETWEEN :sd AND :ed"
                        params["sd"] = filters["from_date"]
                        params["ed"] = filters["to_date"]

                    # ✅ Asli COUNTDOWN (Remaining) & LATE (Overdue) filter
                    if is_time_remaining or is_overdue:
                        import datetime
                        today_str = datetime.date.today().strftime('%Y-%m-%d')
                        if is_time_remaining:
                            # Wo projects dikhao jinki end_date aaj ya aaj ke baad ki hai (Time bacha hai)
                            query += " AND (end_date >= :today OR deadline >= :today)"
                            params["today"] = today_str
                        elif is_overdue:
                            # Wo projects jo late hain (date nikal gayi aur complete nahi hue)
                            query += " AND (end_date < :today OR deadline < :today) AND LOWER(status) != 'completed'"
                            params["today"] = today_str

                    # Name / Comment search
                    if target:
                        words = target_lower.split()
                        target_conds = " AND ".join([f"(LOWER(name) LIKE :t{i} OR LOWER(comment) LIKE :t{i})" for i in range(len(words))])
                        query += f" AND ({target_conds})"
                        for i, w in enumerate(words): params[f"t{i}"] = f"%{w}%"

                    # 🚀 6. EXECUTE SEARCH
                    projs = db.execute(text(query + f" ORDER BY id DESC LIMIT :limit"), {**params, "limit": limit}).fetchall()
                    
                    # 🧠 FAISS Fallback
                    if not projs and target and len(target) > 3:
                        corrected_name = smart_match(target, category="project")
                        if corrected_name and corrected_name.lower() != target_lower:
                            projs = db.execute(text(f"SELECT * FROM projects WHERE is_deleted = 0 AND LOWER(name) LIKE :cn LIMIT :limit"), {"cn": f"%{corrected_name.lower()}%", "limit": limit}).fetchall()

                # 💬 7. RENDER RESULTS (TALKATIVE VERSION 🗣️)
                if not projs:
                    status_text = f" '{active_status}' " if active_status else " "
                    final_results.append({"type": "chat", "message": f"Bhai, lagta hai{status_text}wala koi project abhi nahi mil raha. 🧐"})
                
                elif len(projs) == 1:
                    p = projs[0]
                    p_name = str(p.name)
                    
                    # 🛠️ 1. Refurbished Check
                    machine_type = "Refurbished (Purani/Repair) 🛠️" if getattr(p, 'refurbish', 0) == 1 else "New Machine 🆕"
                    
                    # ⏳ 2. Remaining Time Logic
                    remaining_str = ""
                    target_date = p.end_date or p.deadline
                    if target_date:
                        try:
                            import datetime
                            t_date = datetime.datetime.strptime(str(target_date), '%Y-%m-%d').date() if isinstance(target_date, str) else target_date
                            days_left = (t_date - datetime.date.today()).days
                            
                            if days_left > 0: remaining_str = f" ⏳ (Abhi **{days_left} din** bache hain)"
                            elif days_left == 0: remaining_str = f" 🚨 (Deadline **AAJ** hai!)"
                            else: remaining_str = f" ⚠️ (Deadline **{abs(days_left)} din** pehle nikal chuki hai!)"
                        except: pass
                    
                    # 🗣️ PROJECT SPECIFIC TALKATIVE LOGIC
                    detail_msg = None
                    show_full_card = False

                    # ✅ NAYA LOGIC: Agar user "sab", "han" ya "details" maange toh Card flag ON karo
                    # ✅ NAYA LOGIC: .split() lagaya taaki "sabse" ko "sab" na samjhe
                    if any(w in low_q.split() for w in ["sab", "sabhi", "puri", "poori", "detail", "details", "all", "pura", "han", "haan", "yes"]):
                        show_full_card = True
                    elif any(w in low_q for w in ["budget", "paisa", "rupay", "cost", "amount", "mehenga"]):
                        detail_msg = f"💰 **{p_name}** ka total budget **₹{float(p.budget or 0):,.2f}** set kiya gaya hai."
                    elif any(w in low_q for w in ["deadline", "khatam", "date", "target", "kab tak", "time", "bache", "din"]):
                        detail_msg = f"📅 **{p_name}** ki target deadline **{str(target_date or 'N/A')}** hai.{remaining_str}"
                    elif any(w in low_q for w in ["stage", "progress", "kaha pahuncha", "percent", "status"]):
                        status_now = str(p.status).lower()
                        auto_stage = "100%" if status_now == "completed" else "50%" if status_now == "in_progress" else "0%"
                        actual_stage = getattr(p, 'stage', auto_stage)
                        detail_msg = f"🏗️ **{p_name}** abhi **'{str(p.status).replace('_', ' ').capitalize()}'** status par hai aur lagbhag **{actual_stage}** complete ho chuka hai."
                    elif any(w in low_q for w in ["type", "machine", "refurbished", "purana", "naya", "new"]):
                        detail_msg = f"⚙️ Ye ek **{machine_type}** wala project hai."

                    # ✅ DECISION MAKING (Dikhana kya hai?)
                    if show_full_card:
                        # Pura Card Generate karo
                        type_tag = "Refurbished" if getattr(p, 'refurbish', 0) == 1 else "New Machine"
                        status_now = str(p.status).lower()
                        auto_stage = "100%" if status_now == "completed" else "50%" if status_now == "in_progress" else "Hold" if status_now == "hold" else "0%"
                        
                        card_data = {
                            "type": "project", "project_name": str(p.name),
                            "category": f"{type_tag} | {str(p.status).replace('_', ' ').capitalize()}", "amount": float(p.budget or 0),
                            "start_date": str(p.start_date) if p.start_date else "N/A", "end_date": str(p.end_date or p.deadline or "N/A"),
                            "comments": str(p.comment or ""), "stage": getattr(p, 'stage', auto_stage), "priority": str(p.priority).upper()
                        }
                        final_results.append({"type": "chat", "message": f"Lijiye bhai, **{p_name}** ki poori details 📋:"})
                        final_results.append(card_data) # Card UI render ho jayega
                        
                    elif detail_msg:
                        final_results.append({"type": "chat", "message": detail_msg + "\n\n💡 *Kya aap is project ki baaki details dekhna chahte hain? (Type: Sab)*"})
                    else:
                        msg = f"Bhai, mujhe **{p_name}** system mein mil gaya hai. Ye ek **{machine_type}** project hai.\nAap iska kya dekhna chahte hain?\n\n" \
                              f"💰 Type **'Budget'**\n" \
                              f"📅 Type **'Deadline'**\n" \
                              f"🏗️ Type **'Stage'**\n" \
                              f"📋 Type **'Sab'** *(Poori detail ka card dekhne ke liye)*"
                        final_results.append({"type": "chat", "message": msg})

                else:
                    final_results.append({"type": "chat", "message": f"haan mil gaya 👍 Mujhe **{len(projs)} projects** mile hain:"})
                    proj_results = []
                    for p in projs:
                        type_tag = "Refurbished" if getattr(p, 'refurbish', 0) == 1 else "New Machine"
                        status_now = str(p.status).lower()
                        auto_stage = "100%" if status_now == "completed" else "50%" if status_now == "in_progress" else "Hold" if status_now == "hold" else "0%"
                        
                        proj_results.append({
                            "type": "project", "project_name": str(p.name),
                            "category": f"{type_tag} | {str(p.status).replace('_', ' ').capitalize()}", "amount": float(p.budget or 0),
                            "start_date": str(p.start_date) if p.start_date else "N/A", "end_date": str(p.end_date or p.deadline or "N/A"),
                            "comments": str(p.comment or ""), "stage": getattr(p, 'stage', auto_stage), "priority": str(p.priority).upper()
                        })
                    final_results.extend(proj_results)
            except Exception as e: final_results.append({"type": "chat", "message": f"Project Error: {str(e)}"})
        # ---------------------------------------------------------
        # 🏭 BRANCH 2: SUPPLIER LOGIC (TALKATIVE)
        # ---------------------------------------------------------
        elif intent == "supplier_search":
            try:
                target_lower = original_target.lower()
                is_all_request = not original_target and any(w in low_q for w in ["all", "saare", "sabhi", "list"])
                sups = []

                if is_all_request:
                    # "saare suppliers ki city batao" — group by city instead of listing all
                    if any(w in low_q for w in ["city", "location", "kahan", "kaha", "address", "shahar"]):
                        city_rows = db.execute(text(
                            "SELECT city, COUNT(*) as cnt FROM suppliers WHERE city IS NOT NULL AND city != '' "
                            "GROUP BY city ORDER BY cnt DESC"
                        )).fetchall()
                        if city_rows:
                            lines = "\n".join(f"  • **{r.city}** — {r.cnt} supplier{'s' if r.cnt > 1 else ''}" for r in city_rows)
                            total = sum(r.cnt for r in city_rows)
                            final_results.append({"type": "chat", "message": f"📍 **{total} suppliers** {len(city_rows)} cities mein hain:\n\n{lines}"})
                        else:
                            final_results.append({"type": "chat", "message": "Koi city data nahi mila. 🧐"})
                        continue
                    # cap to 20 so the dropdown stays usable
                    capped_limit = min(limit, 20)
                    sups = db.execute(text("SELECT * FROM suppliers ORDER BY id DESC LIMIT :l"), {"l": capped_limit}).fetchall()
                else:
                    if re.match(r'^sup[-\s]?\d+$', target_lower):
                        code_search = re.sub(r'^sup[-\s]?', '', target_lower)
                        sups = db.execute(text("SELECT * FROM suppliers WHERE supplier_code = :c OR id = :c LIMIT 1"), {"c": code_search}).fetchall()
                    if not sups and original_target:
                        sups = db.execute(text("SELECT * FROM suppliers WHERE LOWER(supplier_name) = :q LIMIT 1"), {"q": target_lower}).fetchall()
                    if not sups and original_target:
                        words = target_lower.split()
                        if words:
                            like_conds = " AND ".join([f"(LOWER(supplier_name) LIKE :w{i} OR mobile LIKE :w{i})" for i in range(len(words))])
                            params = {f"w{i}": f"%{w}%" for i, w in enumerate(words)}
                            params["l"] = limit
                            sups = db.execute(text(f"SELECT * FROM suppliers WHERE {like_conds} LIMIT :l"), params).fetchall()
                    if not sups and len(original_target) > 2: 
                        corrected = smart_match(original_target, category="supplier")
                        if corrected and corrected != original_target:
                            sups = db.execute(text("SELECT * FROM suppliers WHERE LOWER(supplier_name) = :q LIMIT 1"), {"q": corrected.lower()}).fetchall()

                if not sups: 
                    if original_target: final_results.append({"type": "chat", "message": f"Bhai, '{original_target}' naam ka koi Supplier nahi mila mujhe. 🧐"})
                elif len(sups) > 1: 
                    final_results.append({"type": "chat", "message": f"haan mil gaya 👍 Mujhe {len(sups)} suppliers mile hain:"})
                    final_results.append({"type": "dropdown", "message": "Select a supplier for details:", "items": [{"id": str(getattr(s, 'supplier_name', 'Unknown')), "name": str(getattr(s, 'supplier_name', 'Unknown'))} for s in sups]})
                else:
                    s = sups[0]
                    sup_name = str(getattr(s, 'supplier_name', 'Unknown'))
                    
                    # 🗣️ SURGICAL TALKATIVE LOGIC
                    detail_msg = None
                    if any(w in low_q for w in ["mobile", "phone", "number", "call", "contact"]):
                        detail_msg = f"📞 **{sup_name}** ka contact number **{str(getattr(s, 'mobile', 'N/A') or 'N/A')}** hai."
                    elif any(w in low_q for w in ["email", "mail", "id"]):
                        detail_msg = f"📧 **{sup_name}** ki email ID **{str(getattr(s, 'email', 'N/A') or 'N/A')}** hai."
                    elif any(w in low_q for w in ["gst", "gstin", "tax"]):
                        detail_msg = f"🏢 **{sup_name}** ka GST number **{str(getattr(s, 'gstin', 'N/A') or 'N/A')}** hai."
                    elif any(w in low_q for w in ["city", "address", "kaha", "location"]):
                        detail_msg = f"📍 **{sup_name}** **{str(getattr(s, 'city', 'N/A') or 'N/A')}** mein based hain."

                    if detail_msg:
                        final_results.append({"type": "chat", "message": detail_msg + "\n\n💡 *Kya main inki poori profile ya orders load karun?*"})
                    
                    else:
                        is_asking_details = any(w in low_q for w in ["detail", "details", "contact", "number", "profile", "hisab", "account", "info"])
                        
                        # if the user hasn’t asked for something specific (like a point, list, or detailed explanation)
                        if not is_asking_details and not is_all_request and "sup-" not in target_lower and not any(w in low_q for w in ["po", "order", "bill"]):
                            msg = f"Bhai, mujhe **{sup_name}** system mein mil gaye hain. Aap inka kya dekhna chahte hain?\n\n" \
                                  f"📦 Type **'Orders'** (Inke pending aur complete orders dekhne ke liye)\n" \
                                  f"👤 Type **'Details'** (Inki profile, GST, contact info dekhne ke liye)"
                            final_results.append({"type": "chat", "message": msg})
                            # Ye 'continue' isliye lagaya taaki choice dene ke baad wo lamba card na khole
                            continue
                                       
                        if "po_search" not in intents:
                            final_results.append({"type": "chat", "message": f"haan ye raha 👍 **{sup_name}** ka profile mil gaya hai:"})

                        inv_items = db.execute(text("SELECT i.name, SUM(CASE WHEN LOWER(t.txn_type) = 'in' THEN t.quantity ELSE -t.quantity END) as stock FROM inventories i JOIN stock_transactions t ON i.id = t.inventory_id WHERE t.supplier_id = :sid GROUP BY i.id, i.name HAVING stock != 0"), {"sid": s.id}).fetchall()
                        final_results.append({
                            "type": "result", 
                            "supplier": {
                                "id": s.id, "name": sup_name, "code": str(getattr(s, 'supplier_code', 'N/A') or 'N/A'), 
                                "mobile": str(getattr(s, 'mobile', 'N/A') or 'N/A'), "city": str(getattr(s, 'city', 'N/A') or 'N/A'), 
                                "email": str(getattr(s, 'email', 'N/A') or 'N/A'), "gstin": str(getattr(s, 'gstin', 'N/A') or 'N/A')
                            }, 
                            "items": [{"name": str(row.name), "stock": float(row.stock)} for row in inv_items]
                        })
            except Exception as e: final_results.append({"type": "chat", "message": f"Supplier search error: {str(e)}"})

       # ---------------------------------------------------------
        # 🧾 BRANCH 3: PURCHASE ORDERS (ALL BOSS MODES INCLUDED)
        # ---------------------------------------------------------
        elif intent == "po_search":
            try:
                # 📊 BOSS MODE 1: Sabse Jada Balance
                if any(w in low_q for w in ["sabse jada balance", "highest balance", "paisa baaki", "sabse jyada balance", "maximum balance"]):
                    bal_res = db.execute(text("SELECT s.supplier_name, SUM(p.balance_amount) as total_bal, COUNT(p.id) as pending_orders, s.mobile FROM purchase_orders p JOIN suppliers s ON p.supplier_id = s.id WHERE p.balance_amount > 0 AND LOWER(p.status) != 'completed' GROUP BY s.id, s.supplier_name ORDER BY total_bal DESC LIMIT 1")).fetchone()
                    if bal_res:
                        final_results.append({"type": "chat", "message": f"💸 **Payment Alert:** Sabse zyada pending balance **{bal_res.supplier_name}** ka hai.\n\n💰 Total Pending: **₹{float(bal_res.total_bal):,.2f}**\n📄 Orders: **{bal_res.pending_orders} pending**\n📞 Contact: {bal_res.mobile}"})
                        continue

                # 📊 BOSS MODE 2: Sabse Kam Balance/lowest balance
                if any(w in low_q for w in ["sab se kam balance", "sabse kam balance", "lowest balance", "minimum balance"]):
                    bal_res = db.execute(text("SELECT s.supplier_name, SUM(p.balance_amount) as total_bal, COUNT(p.id) as pending_orders, s.mobile FROM purchase_orders p JOIN suppliers s ON p.supplier_id = s.id WHERE p.balance_amount > 0 AND LOWER(p.status) != 'completed' GROUP BY s.id, s.supplier_name ORDER BY total_bal ASC LIMIT 1")).fetchone()
                    if bal_res:
                        final_results.append({"type": "chat", "message": f"💸 **Payment Alert:** Sabse kam pending balance **{bal_res.supplier_name}** ka hai.\n\n💰 Total Pending: **₹{float(bal_res.total_bal):,.2f}**\n📄 Orders: **{bal_res.pending_orders} pending**\n📞 Contact: {bal_res.mobile}"})
                        continue

                # 📊 BOSS MODE 3: Highest / Sabse Bada PO
                if any(w in low_q for w in ["highest po", "sabse bada po", "biggest order", "sabse bada order"]):
                    big_po = db.execute(text("SELECT p.*, s.supplier_name FROM purchase_orders p JOIN suppliers s ON p.supplier_id = s.id ORDER BY p.total_amount DESC LIMIT 1")).fetchone()
                    if big_po:
                        final_results.append({"type": "chat", "message": f"🏆 **Highest Order:** Poore system mein sabse bada Purchase Order **{big_po.supplier_name}** ka hai."})
                        final_results.append({
                            "type": "po", "po_no": str(big_po.po_number), "supplier": str(big_po.supplier_name),
                            "date": str(big_po.po_date), "total": float(big_po.total_amount or 0), 
                            "advance": float(big_po.advance_amount or 0), "balance": float(big_po.balance_amount or 0),
                            "status": str(big_po.status).capitalize()
                        })
                        continue
                
                # 📊 BOSS MODE 4: Lowest / Sabse Chhota PO (New Fix ✅)
                if any(w in low_q for w in ["lowest po", "sabse chota po", "sabse kam po", "smallest order"]):
                    small_po = db.execute(text("SELECT p.*, s.supplier_name FROM purchase_orders p JOIN suppliers s ON p.supplier_id = s.id ORDER BY p.total_amount ASC LIMIT 1")).fetchone()
                    if small_po:
                        final_results.append({"type": "chat", "message": f"📉 **Lowest Order:** Poore system mein sabse chhota Purchase Order **{small_po.supplier_name}** ka hai."})
                        final_results.append({
                            "type": "po", "po_no": str(small_po.po_number), "supplier": str(small_po.supplier_name),
                            "date": str(small_po.po_date), "total": float(small_po.total_amount or 0), 
                            "advance": float(small_po.advance_amount or 0), "balance": float(small_po.balance_amount or 0),
                            "status": str(small_po.status).capitalize()
                        })
                        continue

                # 📊 BOSS MODE 5: TAX / GST Analytics (NEW FIX ✅)
                if any(w in low_q for w in ["tax", "gst", "cgst", "sgst"]):
                    tax_query = "SELECT SUM(tax_amount) as total_tax, COUNT(id) as po_count FROM purchase_orders p WHERE 1=1"
                    tax_params = {}
                    
                    if filters.get("from_date") and filters.get("to_date"):
                        tax_query += " AND p.po_date BETWEEN :start AND :end"
                        tax_params["start"] = filters['from_date']
                        tax_params["end"] = filters['to_date']
                        
                    if original_target:
                        tax_query = "SELECT SUM(p.tax_amount) as total_tax, COUNT(p.id) as po_count FROM purchase_orders p JOIN suppliers s ON p.supplier_id = s.id WHERE 1=1"
                        words = [w for w in original_target.split() if len(w) > 1]
                        if words:
                            search_conds = " AND ".join([f"LOWER(s.supplier_name) LIKE :s{i}" for i in range(len(words))])
                            tax_query += f" AND ({search_conds})"
                            for i, w in enumerate(words): tax_params[f"s{i}"] = f"%{w}%"

                    tax_res = db.execute(text(tax_query), tax_params).fetchone()
                    if tax_res and tax_res.total_tax:
                        party_name = f"**{original_target.title()}** ke " if original_target else "In "
                        msg = f"🧾 **Tax (GST) Report:**\n\n{party_name}**{tax_res.po_count} orders** par total **₹{float(tax_res.total_tax):,.2f}** ka Tax/GST bana hai."
                        final_results.append({"type": "chat", "message": msg})
                        continue
                    else:
                        final_results.append({"type": "chat", "message": "Bhai, in filters par mujhe koi tax ya GST ka data nahi mila. 🧐"})
                        continue

                # 🛑 HALUCCINATION FILTER & Normal Search
                valid_statuses = ["draft", "completed", "pending", "in progress", "cancelled", "approved"]
                active_status = str(filters.get("status") or "").lower().strip()
                if active_status not in valid_statuses:
                    active_status = "" 

                if any(w in low_q for w in ["pending", "draft", "kacha"]):
                    active_status = "draft"
                    
                # Limit Increase for Pending/All
                if any(w in low_q for w in ["all", "saare", "sabhi", "pure", "poore", "sab", "pending", "draft", "batao"]):
                    limit = 50
                if any(w in low_q for w in ["last", "latest", "nayan"]): limit = 1
                
                query = "SELECT p.*, s.supplier_name FROM purchase_orders p JOIN suppliers s ON p.supplier_id = s.id WHERE 1=1"
                params = {"l": limit}
                
                if active_status: query += " AND LOWER(p.status) = :pst"; params["pst"] = active_status
                
                if original_target:
                    words = [w for w in original_target.split() if len(w) > 1]
                    if words:
                        search_conds = " AND ".join([f"(LOWER(s.supplier_name) LIKE :s{i} OR LOWER(p.po_number) LIKE :s{i})" for i in range(len(words))])
                        query += f" AND ({search_conds})"
                        for i, w in enumerate(words): params[f"s{i}"] = f"%{w}%"
                
                pos = db.execute(text(query + " ORDER BY p.po_date DESC, p.id DESC LIMIT :l"), params).fetchall()
                if not pos:
                    final_results.append({"type": "chat", "message": "Bhai, in filters par mujhe koi orders nahi mile. 🧐"})
                else:
                    total_pend = sum(float(po.balance_amount or 0) for po in pos if str(po.status).lower() != 'completed')
                    msg = f"📄 Mujhe कुल **{len(pos)} orders** mile hain."
                    if total_pend > 0: msg += f" Inka total pending balance **₹{total_pend:,.2f}** hai."
                    final_results.append({"type": "chat", "message": msg})

                    for po in pos:
                        final_results.append({
                            "type": "po", "po_no": str(po.po_number), "supplier": str(po.supplier_name),
                            "date": str(po.po_date), "total": float(po.total_amount or 0), 
                            "advance": float(po.advance_amount or 0), "balance": float(po.balance_amount or 0),
                            "status": str(po.status).capitalize()
                        })
            except Exception as e: final_results.append({"type": "chat", "message": f"PO Error: {str(e)}"})

        # ---------------------------------------------------------
        # 📦 BRANCH 4: INVENTORY SEARCH
        # ---------------------------------------------------------
        elif intent == "search":
            try:
                raw_target = str(ai_data.get("search_target") or "").lower().strip()
                if not raw_target: raw_target = low_q

                clean_target = re.sub(r'[?\'"!.,]', '', raw_target).strip()
                if not clean_target or len(clean_target) < 2:
                    clean_target = low_q

                query_str = "SELECT id, name, model, type, classification, placement FROM inventories WHERE (LOWER(name) LIKE :q OR LOWER(model) LIKE :q)"
                all_inv_names = [row.name.lower() for row in db.execute(text("SELECT name FROM inventories")).fetchall() if row.name]
                found_any = False

                # Try 1: Direct search with original term
                items = db.execute(text(query_str + " LIMIT 100"), {"q": f"%{clean_target}%"}).fetchall()
                t = clean_target

                # Try 2: Word-split for noisy targets ("bearing ka stock" → "bearing")
                _stop = {"ka", "ki", "ke", "ko", "se", "mein", "hai", "hain", "batao", "dikhao",
                         "stock", "kitna", "check", "karo", "wala", "wali", "batana", "bata"}
                if not items and len(clean_target.split()) > 1:
                    _words = [w for w in clean_target.split() if len(w) > 2 and w not in _stop]
                    for _w in _words:
                        items = db.execute(text(query_str + " LIMIT 100"), {"q": f"%{_w}%"}).fetchall()
                        if items: t = _w; break

                # Try 3: FAISS semantic correction (for misspellings, single word only)
                if not items and len(clean_target) > 2:
                    corrected = smart_match(clean_target, category="inventory")
                    if corrected.lower() != clean_target:
                        items = db.execute(text(query_str + " LIMIT 100"), {"q": f"%{corrected}%"}).fetchall()
                        if items: t = corrected

                # Try 4: Difflib fuzzy match (last resort)
                if not items:
                    closest = difflib.get_close_matches(clean_target, all_inv_names, n=1, cutoff=0.65)
                    if closest:
                        items = db.execute(text(query_str + " LIMIT 100"), {"q": f"%{closest[0]}%"}).fetchall()
                        if items: t = closest[0]

                clean_targets = [(items, t)]

                for items, t in clean_targets:

                    if not items: continue
                    found_any = True

                    ids = [i.id for i in items]
                    if len(items) > 1:
                        total_sum = db.execute(text(f"SELECT SUM(CASE WHEN LOWER(txn_type) = 'in' THEN quantity ELSE -quantity END) FROM stock_transactions WHERE inventory_id IN ({','.join(str(x) for x in ids)})")).scalar() or 0
                        final_results.append({"type": "chat", "message": f"haan mil gaya 👍 **Total {t.title()} Stock:** {total_sum:.2f} units available hain."})
                        final_results.append({"type": "dropdown", "message": f"Mujhe {len(items)} items mile hain. Kiski details dekhni hai?", "items": [{"id": i.id, "name": f"{i.name} {i.model or ''}"} for i in items]})
                    elif len(items) == 1:
                        i = items[0]
                        stock = float(db.execute(text("SELECT SUM(CASE WHEN LOWER(txn_type) = 'in' THEN quantity ELSE -quantity END) FROM stock_transactions WHERE inventory_id = :id"), {"id": i.id}).scalar() or 0)
                        disp_cat = i.type if i.type else "Raw Material"
                        disp_loc = i.placement if i.placement else "Main Store"
                        disp_class = str(i.classification).upper() if i.classification else "FINISH"
                        f, sf, m = (stock, 0, 0) if disp_class == "FINISH" else (0, stock, 0) if "SEMI" in disp_class else (0, 0, stock)
                        
                        final_results.append({"type": "chat", "message": f"haan ye raha 👍 **{i.name}** ka data mil gaya:"})
                        final_results.append({
                            "type": "result", 
                            "inventory": {"id": i.id, "name": f"{i.name} {i.model or ''}", "category": disp_cat, "placement": disp_loc}, 
                            "total_stock": stock, "finish_stock": f, "semi_finish_stock": sf, "machining_stock": m
                        })
                
                if not found_any:
                    # History context fallback: if previous response listed projects, try project search
                    if _h_ctx_type == "project":
                        _ctx_projs = db.execute(text(
                            "SELECT * FROM projects WHERE is_deleted=0 AND LOWER(name) LIKE :n ORDER BY id DESC LIMIT 5"
                        ), {"n": f"%{clean_target}%"}).fetchall()
                        if _ctx_projs:
                            if len(_ctx_projs) == 1:
                                p = _ctx_projs[0]
                                final_results.append({"type": "chat", "message": f"Lijiye bhai, **{p.name}** ki details 📋:"})
                                final_results.append(_proj_card(p))
                            else:
                                final_results.append({"type": "chat", "message": f"haan mil gaya 👍 Mujhe **{len(_ctx_projs)} projects** mile hain:"})
                                final_results.extend(_proj_card(p) for p in _ctx_projs)
                            found_any = True
                    # History context fallback: if previous response listed suppliers, try supplier search
                    elif _h_ctx_type in ("supplier", "supplier_list"):
                        _ctx_sups = db.execute(text(
                            "SELECT * FROM suppliers WHERE LOWER(supplier_name) LIKE :n LIMIT 5"
                        ), {"n": f"%{clean_target}%"}).fetchall()
                        if _ctx_sups:
                            s = _ctx_sups[0]
                            sup_name = str(getattr(s, 'supplier_name', 'Unknown'))
                            _sinv = db.execute(text(
                                "SELECT i.name, SUM(CASE WHEN LOWER(t.txn_type)='in' THEN t.quantity ELSE -t.quantity END) as stock "
                                "FROM inventories i JOIN stock_transactions t ON i.id=t.inventory_id "
                                "WHERE t.supplier_id=:sid GROUP BY i.id, i.name HAVING stock != 0"), {"sid": s.id}).fetchall()
                            final_results.append({"type": "chat", "message": f"ye raha 👍 **{sup_name}** ka profile:"})
                            final_results.append({"type": "result", "supplier": {
                                "id": s.id, "name": sup_name,
                                "code": str(getattr(s, 'supplier_code', 'N/A') or 'N/A'),
                                "mobile": str(getattr(s, 'mobile', 'N/A') or 'N/A'),
                                "city": str(getattr(s, 'city', 'N/A') or 'N/A'),
                                "email": str(getattr(s, 'email', 'N/A') or 'N/A'),
                                "gstin": str(getattr(s, 'gstin', 'N/A') or 'N/A')},
                                "items": [{"name": str(r.name), "stock": float(r.stock)} for r in _sinv]})
                            found_any = True

                    if not found_any:
                        final_results.append({"type": "chat", "message": "Bhai, ye item mere system mein nahi mila. 🧐 Thoda spelling check karoge?"})
            except Exception as e: final_results.append({"type": "chat", "message": f"Inventory Error: {str(e)}"})

        # ---------------------------------------------------------
        # 💬 BRANCH 5: GENERAL CHAT / CLARIFY
        # ---------------------------------------------------------
        elif intent in ["general_chat", "clarify"]:
            reasoning = ai_data.get("reasoning") or "Haan bhai, poochho kya jaanna chahte ho? Main yahan hoon! 😊"
            if len(final_results) == 1 and final_results[0].get("type") == "chat":
                final_results[0]["message"] = reasoning
            else:
                final_results.append({"type": "chat", "message": reasoning})

   # =========================================================
    # 🏁 FINAL RETURN & FALLBACK (Language Adaptive)
    # =========================================================

    # 1. Multiple Results (return all — DB queries already limit counts)
    if len(final_results) > 1:
        final_response = {"results": final_results}
    else:
        # 2. Language Detection
        is_english = any(word in low_q for word in ["what", "how", "show", "list", "get", "who", "where", "tell", "check"])
        
        # 3. Smart Suggestion Logic
        if is_english:
            if "project" in low_q or "site" in low_q:
                suggestion_text = "It seems you are looking for **Project** info. Please provide the project name."
            elif any(w in low_q for w in ["money", "balance", "account", "due"]):
                suggestion_text = "Do you want to check a supplier's **Balance**? Please type the party name."
            else:
                suggestion_text = "I'm sorry, I couldn't quite understand that. 😅\n\nYou can ask about:\n1. **Purchase Orders**\n2. **Inventory**\n3. **Suppliers**"
        else:
            if "project" in low_q or "site" in low_q:
                suggestion_text = "lagta hai aap **Projects** ki jankari chahte hain. Kripya us project ka naam batayein."
            elif any(w in low_q for w in ["paisa", "balance", "hisab", "rokra"]):
                suggestion_text = "Kya aap kisi Supplier ka **Balance** check karna chahte hain? Kripya party ka naam likhein."
            else:
                suggestion_text = "Maaf kijiye, main abhi theek se samajh nahi paaya. 😅\n\nAap inme se kuch poochna chahte hain?\n1. **Purchase Orders**\n2. **Inventory**\n3. **Suppliers**"

        final_response = {"results": [{"type": "chat", "message": suggestion_text}]}
    
    # 4. Logging (Naya Pro Logger 📊)
    process_time = time.time() - start_time
    try:
        log_query_pro(user_role, raw_q, intents, final_response, process_time)
    except Exception as e:
        print(f"Logger Crash Prevented: {e}")
        
    return final_response

# ---------------------------------------------------------
# 🕵️‍♂️ SECRET LOGS ENDPOINT (Live browser me dekhne ke liye)
# ---------------------------------------------------------
@router.get("/logs")
def view_live_logs():
    log_file = "chat_history.json"
    if not os.path.exists(log_file):
        return {"message": "Bhai, abhi tak koi chat nahi hui hai. File empty hai! 🤷‍♂️"}
    
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            logs = [json.loads(line) for line in f.readlines() if line.strip()]
            
        return {
            "total_chats": len(logs),
            "latest_logs": logs[::-1]  # Sabse naye messages sabse upar dikhenge
        }
    except Exception as e:
        return {"error": f"Logs padhne me dikkat aayi: {str(e)}"}



#hello hugging face! This is your friendly neighborhood chatbot router. If you have any questions or need help, just ask!