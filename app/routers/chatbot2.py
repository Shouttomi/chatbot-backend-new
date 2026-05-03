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
from sentence_transformers import SentenceTransformer
import difflib
#from rapidfuzz import process, fuzz
#import jellyfish
import json
from app.services.ollama_engine import ask_ollama

router = APIRouter(prefix="/chatbot2", tags=["Chatbot2"])

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

def load_faiss_once(db: Session):
    global semantic_model, inv_names_list, sup_names_list, inv_faiss_index, sup_faiss_index, is_faiss_loaded, proj_names_list, proj_faiss_index
    
    if is_faiss_loaded: return
    
    print("⏳ Loading Semantic Search Model... (10-15 seconds) - Ye sirf Server Start par 1 baar hoga!")
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("🛠️ Building FAISS Memory from Database...")
    try:
        # 1. Inventory Indexing
        inv_data = db.execute(text("SELECT name FROM inventories WHERE name IS NOT NULL")).fetchall()
        inv_names_list = [row[0] for row in inv_data if row[0]]
        if inv_names_list:
            inv_embeddings = semantic_model.encode(inv_names_list).astype('float32')
            inv_faiss_index = faiss.IndexFlatL2(inv_embeddings.shape[1])
            inv_faiss_index.add(inv_embeddings)
            
            # # 🧠 THE MAGIC: DYNAMIC ITEM DICTIONARY GENERATOR
            # # Bot khud database ke items padh kar list banayega (e.g., 'round bar', 'oil seal')
            # for name in inv_names_list:
            #     clean_name = re.sub(r'[^a-zA-Z\s]', '', str(name)).lower()
            #     words = clean_name.split()
            #     if words:
            #         if len(words[0]) > 2: # Single words (e.g., 'round', 'oil', 'tape')
            #             generic_inv_words.add(words[0]) 
            #         if len(words) > 1 and len(words[0]) > 1: # Double words (e.g., 'round bar', 'oil seal')
            #             generic_inv_words.add(f"{words[0]} {words[1]}")

        # 2. Supplier Indexing
        sup_data = db.execute(text("SELECT supplier_name FROM suppliers WHERE supplier_name IS NOT NULL")).fetchall()
        sup_names_list = [row[0] for row in sup_data if row[0]]
        if sup_names_list:
            sup_embeddings = semantic_model.encode(sup_names_list).astype('float32')
            sup_faiss_index = faiss.IndexFlatL2(sup_embeddings.shape[1])
            sup_faiss_index.add(sup_embeddings)

            # for name in sup_names_list:
            #     words = re.findall(r'\w+', str(name).lower())
            #     if words: 
            #         generic_sup_words.update([w for w in words if len(w) > 2])

        # 3. 🏗️ PROJECT INDEXING
        proj_data = db.execute(text("SELECT name FROM projects WHERE name IS NOT NULL AND is_deleted = 0")).fetchall()
        proj_names_list = [row[0] for row in proj_data if row[0]]
        if proj_names_list:
            proj_embeddings = semantic_model.encode(proj_names_list).astype('float32')
            proj_faiss_index = faiss.IndexFlatL2(proj_embeddings.shape[1])
            proj_faiss_index.add(proj_embeddings)

            # for name in proj_names_list:
            #     words = re.findall(r'\w+', str(name).lower())
            #     if words: 
            #         generic_proj_words.update([w for w in words if len(w) > 2])   
            
        is_faiss_loaded = True
        print(f"✅ FAISS Ready! Indexed {len(inv_names_list)} Items, {len(sup_names_list)} Suppliers & {len(proj_names_list)} Projects.")
    except Exception as e:
        print(f"⚠️ FAISS Load Error: {e}")

def smart_match(query_text, category="inventory"):
    if not query_text or len(query_text) < 2 or not is_faiss_loaded: return query_text
    try:
        query_vector = semantic_model.encode([query_text]).astype('float32')
        
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

# --- Existing helpers ---
def log_query(query, intent, result):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "intent": intent,
        "is_fail": any(w in str(result).lower() for w in ["nahi mila", "not found", "error", "samajh nahi"]) or not result
    }
    try:
        with open("logs.json", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"❌ Logging Error: {e}")

@router.post("/")
def chatbot(request: ChatRequest, db: Session = Depends(get_db)):
    raw_q = request.query.strip()
    low_q = raw_q.lower()
    
    # 🎯 STEP 1: FAST-TRACK ID
    if low_q.isdigit() and len(low_q) < 8:
        try:
            inv = db.execute(text("SELECT id, name, classification, placement FROM inventories WHERE id = :id"), {"id": int(low_q)}).fetchone()
            if inv:
                stock_res = db.execute(text("SELECT SUM(CASE WHEN LOWER(txn_type) = 'in' THEN quantity ELSE -quantity END) FROM stock_transactions WHERE inventory_id = :id"), {"id": inv.id}).scalar()
                total_qty = float(stock_res or 0)
                cls = str(inv.classification).lower() if inv.classification else ""
                m, f, sf = (total_qty, 0, 0) if "machining" in cls else (0, 0, total_qty) if "semi" in cls else (0, total_qty, 0)
                return {"results": [{"type": "result", "inventory": {"id": inv.id, "name": inv.name, "category": cls.upper(), "placement": inv.placement or "N/A"}, "total_stock": total_qty, "finish_stock": f, "semi_finish_stock": sf, "machining_stock": m}]}
        except: pass

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
        log_query(raw_q, "unknown", {"error": str(e)})
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

    # 🛡️ SEATBELT
    if not original_target and any(i in intents for i in ["supplier_search", "project_search"]):
        return {"results": [{"type": "chat", "message": "Bhai, kripya thoda clear batao ki aap kis company ya project ki baat kar rahe ho? 🙂"}]}

    # 🧹 NOISE CLEANER
    noise_words = ["supplier", "vendor", "party", "details", "contact", "profile", "ki", "ka", "ke", "project", "site", "machine"]
    for word in noise_words:
        original_target = re.sub(rf'\b{word}\b', '', original_target, flags=re.IGNORECASE).strip()

    ai_data["search_target"] = original_target

    if re.match(r'^sup[-\s]?\d+$', original_target.lower()):
        if "supplier_search" not in intents: intents = ["supplier_search"]

    print(f"✅ FINAL ROUTER DECISION: {intents} | TARGET: {original_target}")

    filters = ai_data.get("filters", {})
    ui_filters = getattr(request, "ui_filters", {}) or {}
    for key, value in ui_filters.items():
        if value: filters[key] = value
    limit = filters.get("limit", 5) or 5

    # 🚀 TAX & ADVANCE OVERRIDE
    if any(w in low_q for w in ["tax", "gst", "cgst", "sgst", "advance", "adv"]):
        intents = ["po_search"]


####


    # =========================================================
    # 🔄 MULTI-INTENT PROCESSING LOOP ( main code )
    # =========================================================
    for intent in intents:
        
        # ---------------------------------------------------------
        # 📁 BRANCH 1: PROJECT LOGIC
        # ---------------------------------------------------------
        if intent == "project_search":
            try:
                target = original_target.strip()
                active_status = str(filters.get("status") or "").lower().strip()
                query = "SELECT * FROM projects WHERE is_deleted = 0"
                params = {}

                if active_status and active_status != "all":
                    if active_status == "refurbished": query += " AND refurbish = 1"
                    else: query += " AND LOWER(status) = :st"; params["st"] = active_status
                
                if target and target.lower() not in ["all", "list", "projects", "latest"]:
                    words = target.lower().split()
                    target_conds = " AND ".join([f"(LOWER(name) LIKE :t{i} OR LOWER(comment) LIKE :t{i})" for i in range(len(words))])
                    query += f" AND ({target_conds})"
                    for i, w in enumerate(words): params[f"t{i}"] = f"%{w}%"

                projs = db.execute(text(query + f" ORDER BY id DESC LIMIT :limit"), {**params, "limit": limit}).fetchall()

                if not projs and target and len(target) > 3:
                    corrected_name = smart_match(target, category="project")
                    if corrected_name and corrected_name.lower() != target.lower():
                        projs = db.execute(text(f"SELECT * FROM projects WHERE is_deleted = 0 AND LOWER(name) LIKE :cn LIMIT :limit"), {"cn": f"%{corrected_name.lower()}%", "limit": limit}).fetchall()

                if not projs:
                    final_results.append({"type": "chat", "message": f"lagta hai '{target or 'is filter'}' wala koi project abhi nahi chal raha. 🧐"})
                else:
                    final_results.append({"type": "chat", "message": f"haan mil gaya 👍 Mujhe **{len(projs)} projects** mile hain:"})
                    proj_results = []
                    for p in projs:
                        type_tag = "Refurbished" if getattr(p, 'refurbish', 0) == 1 else "New Machine"
                        status_now = str(p.status).lower()
                        auto_stage = "100%" if status_now == "completed" else "50%" if status_now == "in progress" else "Hold" if status_now == "hold" else "0%"
                        
                        proj_results.append({
                            "type": "project", "project_name": str(p.name),
                            "category": f"{type_tag} | {str(p.status).capitalize()}", "amount": float(p.budget or 0),
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
                    sups = db.execute(text("SELECT * FROM suppliers ORDER BY id DESC LIMIT :l"), {"l": limit}).fetchall()
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
                clean_targets = []
                if len(clean_target) > 1: 
                    corrected = smart_match(clean_target, category="inventory")
                    clean_targets.append(corrected)
                if not clean_targets and ("bearing" in low_q or "belt" in low_q): 
                    clean_targets = ["bearing" if "bearing" in low_q else "belt"]

                all_inv_names = [row.name.lower() for row in db.execute(text("SELECT name FROM inventories")).fetchall() if row.name]
                found_any = False

                for t in clean_targets:
                    query_str = "SELECT id, name, model, type, classification, placement FROM inventories WHERE (LOWER(name) LIKE :q OR LOWER(model) LIKE :q)"
                    items = db.execute(text(query_str + " LIMIT 30"), {"q": f"%{t}%"}).fetchall()
                    
                    if not items:
                        closest = difflib.get_close_matches(t, all_inv_names, n=1, cutoff=0.65)
                        if closest: 
                            items = db.execute(text(query_str + " LIMIT 30"), {"q": f"%{closest[0]}%"}).fetchall()
                            t = closest[0]

                    if not items: continue
                    found_any = True

                    ids = tuple([i.id for i in items])
                    if len(items) > 1:
                        total_sum = db.execute(text("SELECT SUM(CASE WHEN LOWER(txn_type) = 'in' THEN quantity ELSE -quantity END) FROM stock_transactions WHERE inventory_id IN :ids"), {"ids": ids}).scalar() or 0
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
                    final_results.append({"type": "chat", "message": "Bhai, ye item mere system mein nahi mila. 🧐 Thoda spelling check karoge?"})
            except Exception as e: final_results.append({"type": "chat", "message": f"Inventory Error: {str(e)}"})

   # =========================================================
    # 🏁 FINAL RETURN & FALLBACK (Language Adaptive)
    # =========================================================
    
    # 1. Multiple Results Fallback (Without AI Confusion)
    if len(final_results) > 1:
        return {"results": final_results[:limit + 2]}
    
    # 2. Language Detection
    is_english = any(word in low_q for word in ["what", "how", "show", "list", "get", "who", "where", "tell", "check"])
    
    # 3. Smart Suggestion Logic (Merging your Project/Paisa logic)
    if is_english:
        if "project" in low_q or "site" in low_q:
            suggestion_text = "It seems you are looking for **Project** info. Please provide the project name."
        elif any(w in low_q for w in ["money", "balance", "account", "due"]):
            suggestion_text = "Do you want to check a supplier's **Balance**? Please type the party name."
        else:
            suggestion_text = "I'm sorry, I couldn't quite understand that. 😅\n\nYou can ask about:\n1. **Purchase Orders** (e.g., 'Show latest PO')\n2. **Inventory** (e.g., 'Check bearing stock')\n3. **Suppliers** (e.g., 'Supplier details')"
    else:
        # older Logic
        if "project" in low_q or "site" in low_q:
            suggestion_text = "lagta hai aap **Projects** ki jankari chahte hain. Kripya us project ka naam batayein."
        elif any(w in low_q for w in ["paisa", "balance", "hisab", "rokra"]):
            suggestion_text = "Kya aap kisi Supplier ka **Balance** check karna chahte hain? Kripya party ka naam likhein."
        else:
            suggestion_text = "Maaf kijiye, main abhi theek se samajh nahi paaya. 😅\n\nAap inme se kuch poochna chahte hain?\n1. **Purchase Orders** (e.g., 'Latest PO dikhao')\n2. **Inventory** (e.g., 'Stock check karo')\n3. **Suppliers**"

    fallback_res = {"results": [{"type": "chat", "message": suggestion_text}]}
    
    # 4. Logging (Aapka purana function)
    if 'log_query' in globals() or 'log_query' in locals():
        log_query(raw_q, intents, fallback_res)
        
    return fallback_res