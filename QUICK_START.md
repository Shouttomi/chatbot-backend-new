# Quick Start — v2 Chatbot

## 1. Download Ollama + Model (First Time Only)
```powershell
powershell -ExecutionPolicy Bypass -File "D:\mewar_erp\setup_ollama_d.ps1"
```
⏱️ Takes ~20-40 min. Downloads ~4.7 GB to `D:\ollama_models\`

---

## 2. Start Server
```powershell
powershell -ExecutionPolicy Bypass -File "D:\mewar_erp\start_server.ps1"
```
✅ Server runs at `http://localhost:8000`

---

## 3. Test It
```bash
# Health check (verify Ollama is working)
curl http://localhost:8000/v2-chatbot/status

# Ask a question
curl -X POST http://localhost:8000/v2-chatbot/ \
  -H "Content-Type: application/json" \
  -d '{"query": "bearing stock kitna hai"}'
```

---

## After Restart

Each time you restart your PC:

1. **Start Ollama:**
```
D:\Ollama\ollama.exe serve
```

2. **Start Server:**
```powershell
powershell -ExecutionPolicy Bypass -File "D:\mewar_erp\start_server.ps1"
```

---

## Files to Know

| File | Use |
|------|-----|
| `SETUP_COMPLETE.md` | Full setup guide |
| `chatbot_edge_cases.md` | 30 test questions |
| `app/services/v2_ollama_engine.py` | LLM client (Ollama API) |
| `app/routers/v2_chatbot.py` | Chatbot logic |
| `setup_ollama_d.ps1` | Download Ollama + model |
| `start_server.ps1` | Launch FastAPI server |

---

## Endpoints

```
POST /v2-chatbot/                 Main chatbot
GET  /v2-chatbot/status           Health check
POST /v2-chatbot/reload-index     Rebuild indexes
```

---

## If Something Breaks

**Module errors?**
```bash
cd d:\mewar_erp
python -m pip install -r requirements.txt --upgrade
```

**Ollama not responding?**
```powershell
D:\Ollama\ollama.exe serve  # in a new terminal
```

**Python encoding errors?**
```powershell
$env:PYTHONIOENCODING = "utf-8"
```

---

That's it! 🚀
