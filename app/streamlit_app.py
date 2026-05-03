import streamlit as st
import requests
import pandas as pd
import datetime

# ==========================================
# 🚀 API CONFIGURATION
# ==========================================
#API_BASE = "https://mewar-erp.vercel.app"
API_BASE = "http://127.0.0.1:8000"
CHAT_URL = f"{API_BASE}/chatbot/"

st.set_page_config(page_title="Mewar ERP AI", page_icon="🧠", layout="centered")

# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state: st.session_state.messages = []
if "next_query" not in st.session_state: st.session_state.next_query = None

def set_next_query(query_text):
    st.session_state.next_query = query_text

# ==========================================
# CENTRALIZED UI RENDERER
# ==========================================
def render_bot_response(data, msg_idx):
    if "error" in data:
        st.error(f"🔌 {data['error']}")
        return
    if "detail" in data:
        st.error(f"🔴 Access Error: {data['detail']}")
        return

    results_list = data.get("results", [data])

    for res in results_list:
        res_type = res.get("type")

        # 🟢 CASE 1: EXACT STOCK MATCH
        if res_type == "result" and "inventory" in res:
            inv = res["inventory"]
            st.success(f"📦 **{inv['name']}**")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Stock", res.get("total_stock", 0))
            c2.metric("Finish", res.get("finish_stock", 0))
            c3.metric("Semi-Finish", res.get("semi_finish_stock", 0))
            c4.metric("Machining", res.get("machining_stock", 0))
            
            st.caption(f"Item ID: #{inv['id']} | Category: {inv.get('classification', 'N/A')} | 📍 Location: {inv.get('placement', 'Not Assigned')}")
        
        # 🔵 CASE 2: SUPPLIER MATCH 
        elif res_type == "result" and "supplier" in res:
            sup = res["supplier"]
            st.info(f"🏭 **{sup['name']}**")
            
            code = sup.get('code', 'N/A')
            email = sup.get('email', 'N/A')
            gstin = sup.get('gstin', 'N/A')
            mobile = sup.get('mobile', 'N/A')
            city = sup.get('city', 'N/A')
            state = sup.get('state', 'N/A')
            
            st.markdown(f"**Code:** {code}  \n**Mobile:** {mobile}  \n**Location:** {city}, {state}  \n**GSTIN:** {gstin}")
            
            if "items" in res:
                st.write("---")
                items = res.get("items", [])
                if items:
                    st.write("**📦 Inventory from this Supplier:**")
                    for item in items:
                        st.write(f"- {item.get('name')}: **{item.get('stock')}** in stock")
                else:
                    st.info("📦 No active inventory currently in stock from this supplier.")

        # 🧾 CASE 3: PURCHASE ORDER (PO) RESULT
        elif res_type == "po":
            st.info(f"🧾 **Purchase Order: {res.get('po_no')}**")
            st.write(f"**Supplier:** {res.get('supplier')}")
            st.caption(f"📅 **Date:** {res.get('date')}")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Amount", f"₹{res.get('total', 0):,.2f}")
            c2.metric("Advance", f"₹{res.get('advance', 0):,.2f}")
            
            balance = res.get('balance', 0)
            if balance > 0:
                c3.error(f"Balance: ₹{balance:,.2f}")
            else:
                c3.success(f"Balance: ₹0.00")
            st.write("---")

        # 📁 CASE 4: PROJECT RESULT (Upgraded with Stage & Comments)
        elif res_type == "project":
            st.info(f"📁 **Project: {res.get('project_name')}**")
            
            p1, p2 = st.columns(2)
            p1.write(f"**Status/Priority:** {res.get('category', 'N/A')}")
            
            # ✅ STAGE (Progress Bar logic)
            stage_val = res.get('stage', '0%').replace('%', '')
            try:
                progress = int(stage_val) / 100
            except:
                progress = 0.0
            
            st.write(f"**🏗️ Project Stage: {stage_val}%**")
            st.progress(progress)
            
            # ✅ COMMENTS (Expander mein taaki UI ganda na ho)
            with st.expander("💬 View Project Comments"):
                st.write(res.get('comments', 'No comments available.'))

            # Timeline Logic
            end_date_str = str(res.get('end_date', 'N/A'))
            st.caption(f"📅 **Timeline:** {res.get('start_date')}  ➔  {end_date_str}")
            
            # ✅ Live Countdown Timer
            if end_date_str not in ['N/A', 'None', '']:
                try:
                    clean_date = end_date_str.split()[0] 
                    end_dt = datetime.datetime.strptime(clean_date, '%Y-%m-%d').date()
                    days_left = (end_dt - datetime.date.today()).days
                    
                    if days_left > 0:
                        st.success(f"⏳ **{days_left} Days Remaining**")
                    elif days_left == 0:
                        st.warning("⏳ **Project Deadline Today!**")
                    else:
                        st.error(f"⚠️ **Overdue by {abs(days_left)} Days**")
                except: pass
            
            # ✅ AMOUNT (Budget)
            st.metric("Estimated Budget", f"₹{res.get('amount', 0):,.2f}")
            st.divider()
            
        # 🟡 CASE 5: DROPDOWN MENU
        elif res_type == "dropdown":
            st.warning(res.get("message", "Select an item:"))
            cols = st.columns(2)
            for i, item in enumerate(res.get("items", [])):
                button_label = f"🔎 {item['name']} (#{item['id']})"
                cols[i % 2].button(
                    button_label, 
                    key=f"btn_{item['id']}_{msg_idx}_{i}", 
                    on_click=set_next_query, 
                    args=(str(item['id']),) 
                )
                
        # 🟡 CASE 6: SUPPLIER LIST MENU
        elif res_type == "supplier_list":
            st.warning(res.get("message", "Select a supplier:"))
            for i, s in enumerate(res.get("suppliers", [])):
                st.button(
                    f"🏭 {s['name']}", 
                    key=f"sup_{s['id']}_{msg_idx}_{i}", 
                    on_click=set_next_query, 
                    args=(s['name'],)
                )

        # 📊 CASE 7: MANAGER ANALYTICS CHARTS
        elif res_type == "analytics_chart":
            st.subheader(res.get("title", "📊 Analytics Report"))
            df = pd.DataFrame(res.get("data", []))
            if not df.empty:
                st.dataframe(df, use_container_width=True, hide_index=True)
                if res.get("chart_type") == "bar":
                    st.write("---")
                    st.bar_chart(df.set_index("Name")["Stock"])
            else:
                st.info("No data available for this report.")
        
        # 💬 CASE 8: Simple Text
        elif "message" in res and not res_type:
            st.write(res["message"])
        elif res_type == "chat":
            st.write(res["message"])


# ==========================================
# PAGE: CHATBOT INTERFACE
# ==========================================
with st.sidebar:
    st.header("Admin Panel")
    
    # 🛡️ THE NEW TESTING DROPDOWN
    selected_role = st.selectbox(
        "🎭 Select Role (For Testing)",
        ["Super Admin", "HOD", "Purchase Admin", "Purchase", "Store Admin", "Store Department", "Supervisor", "Sales", "HR"]
    )
    
    st.write(f"Logged in as: **{selected_role}**")
    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    st.caption("Mewar ERP AI - Testing Mode")

st.title("ERP Intelligence 🧠")

# 🚀 THE NEW HYBRID UI
if not st.session_state.messages:
    with st.container(border=True):
        st.markdown("### Hi, Welcome!")
        st.markdown(
            "<span style='color: #666;'>Search for supplier, inventory, and stock details, or speak using the microphone.</span>", 
            unsafe_allow_html=True
        )
        st.write("") # Spacing
        
        c1, c2, c3, c4 = st.columns(4)
        
        if c1.button("👥 Supplier", use_container_width=True):
            set_next_query("show all suppliers")
        if c2.button("📦 Inventory", use_container_width=True):
            set_next_query("show latest inventory")
        if c3.button("🛒 Purchase Order", use_container_width=True):
            set_next_query("latest po")
        if c4.button("📁 Project", use_container_width=True):
            set_next_query("project kon kon se chal rahe hai")
            
    st.write("") # Spacing

# 🟢 UPDATE: ask_erp ab role bhi backend ko bhejega
def ask_erp(query, role):
    headers = {"Content-Type": "application/json"}
    history = [{"role": m["role"], "content": m.get("raw_content", "")} for m in st.session_state.messages]
    try:
        # NAYA: json payload mein "role" daal diya
        r = requests.post(CHAT_URL, json={"query": query, "history": history, "role": role}, headers=headers)
        return r.json()
    except Exception as e: 
        return {"error": f"FastAPI Connection Failed. {str(e)}"}

# Render Chat History
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and "data" in msg:
            render_bot_response(msg["data"], idx)
        else:
            st.markdown(msg.get("raw_content", ""))

u_input = st.chat_input("Ask about inventory, POs, Projects, or Suppliers...")
final_query = u_input or st.session_state.next_query

if final_query:
    st.session_state.next_query = None 
    
    with st.chat_message("user"):
        st.markdown(final_query)
    st.session_state.messages.append({"role": "user", "raw_content": final_query})
    
    # 🟢 UPDATE: Yahan ask_erp ko selected role bhej rahe hain
    data = ask_erp(final_query, selected_role)
    
    with st.chat_message("assistant"):
        render_bot_response(data, len(st.session_state.messages))
        
        # 🧠 FIX: Backend ke "results" array me se asali message nikalna!
        bot_text_list = []
        for res in data.get("results", [{"message": data.get("message", "Processed.")}]):
            if "message" in res:
                bot_text_list.append(str(res["message"]))
        
        actual_bot_message = " ".join(bot_text_list) if bot_text_list else "Data rendered."

        st.session_state.messages.append({
            "role": "assistant", 
            "data": data, 
            "raw_content": actual_bot_message # Ab history me "Processed." nahi, asli baat jayegi!
        })