"use client";

import { useState, useRef, useCallback, useEffect } from "react";

// ─────────────────────────────────────────────────────────────────────────────
// CONFIG
// Override via Vite env vars in .env.local:
//   VITE_CHATBOT_API="http://your-backend.com/v2-chatbot/"
//   VITE_CHATBOT_FEEDBACK_API="http://your-backend.com/v2-chatbot/feedback"
// ─────────────────────────────────────────────────────────────────────────────
const API_BASE =
  import.meta.env?.VITE_CHATBOT_API || "http://localhost:8000/v2-chatbot/";
const FEEDBACK_URL =
  import.meta.env?.VITE_CHATBOT_FEEDBACK_API ||
  API_BASE.replace(/\/$/, "") + "/feedback";

const CFG = {
  title:       "ERP Assistant",
  subtitle:    "Suppliers · Inventory · Orders",
  apiUrl:      API_BASE,
  feedbackUrl: FEEDBACK_URL,
  defaultRole: "superadmin",
  roles:       ["superadmin", "supervisor", "admin", "user", "guest"],
  G:           "linear-gradient(135deg,#454545 0%,#e65e38 50%)",
  P:           "#666666",
  A:           "#f6591b",
  CPS:         120,   // characters per second for streaming
};

// ─────────────────────────────────────────────────────────────────────────────
// SCOPED CSS  — obfuscated class names, injected once, zero global leakage
// ─────────────────────────────────────────────────────────────────────────────
const Q = {
  fab:    "_ew4k2",  overlay: "_ew9x7",  panel:   "_ew2r8",
  hdr:    "_ew5t3",  hrow:    "_ew7p1",  hico:    "_ew3n4",
  htit:   "_ew8m6",  hsub:    "_ew1q9",  hbtn:    "_ew6j2",
  rpill:  "_ew4v5",  rmenu:   "_ew9k3",  ropt:    "_ew2w6",
  roptA:  "_ew7t8",  chat:    "_ew5m1",  bwrap:   "_ew3q7",
  bwrapU: "_ew6n4",  bav:     "_ew8p2",  ubub:    "_ew1k5",
  bbub:   "_ew4w9",  dots:    "_ew9v3",  dot:     "_ew2t6",
  cur:    "_ew7m8",  ts:      "_ew5k1",  errtxt:  "_ew3w4",
  icard:  "_ew8q7",  ihdr:    "_ew1p3",  iico:    "_ew6t9",
  iname:  "_ew4n2",  imeta:   "_ew9w5",  iid:     "_ew2k8",
  stgrid: "_ew7v1",  sbadge:  "_ew5q4",  sbadgeH: "_ew3t7",
  dcard:  "_ew8n2",  dmsg:    "_ew1w5",  dlist:   "_ew6q8",
  ditem:  "_ew4p1",  did:     "_ew9m3",  dname:   "_ew2v6",
  ibar:   "_ew7k9",  iinn:    "_ew5n2",  iinput:  "_ew3m5",
  clr:    "_ew8w1",  send:    "_ew1v4",  spin:    "_ew6k7",
  chatWrap:"_ew0c1",
  tabbar: "_ew_tb",  tab:     "_ew_t1",  tabA:    "_ew_t2",
  tabX:   "_ew_t3",  tabP:    "_ew_t4",
};

(function injectCSS() {
  
  const ID = "_ew_s";
  if (typeof document === "undefined" || document.getElementById(ID)) return;
  const el = document.createElement("style");
  el.id = ID;
  el.textContent = `
/* ── ERP Chat Widget ── scoped, zero global leak ── */
.${Q.fab}{
  position:fixed;bottom:24px;right:24px;z-index:2147483640;
  width:58px;height:58px;border-radius:50%;border:none;cursor:pointer;
  background:${CFG.G};display:flex;align-items:center;justify-content:center;
  box-shadow:0 8px 28px rgba(230,94,56,0.3),0 4px 14px rgba(0,0,0,.15);
  transition:all .18s cubic-bezier(.22,.68,0,1.18);
  font-weight:600;
}
.${Q.fab}:hover{transform:translateY(-4px);box-shadow:0 12px 40px rgba(230,94,56,0.4),0 6px 20px rgba(0,0,0,.18);}
.${Q.fab}:active{transform:translateY(-2px);}

.${Q.overlay}{
  position:fixed;inset:0;z-index:2147483638;
  background:rgba(0,0,0,.42);backdrop-filter:blur(4px);
  animation:_ewFade .2s ease;
}

.${Q.panel}{
  position:fixed;bottom:22px;right:22px;z-index:2147483639;
  width:430px;height:min(88vh,720px);
  background:#fff;border-radius:24px;overflow:hidden;
  display:flex;flex-direction:column;
  font-family:'DM Sans','Segoe UI',system-ui,sans-serif;
  box-shadow:0 30px 90px rgba(0,0,0,.15),0 8px 32px rgba(0,0,0,.1),inset 0 1px 0 rgba(255,255,255,.5);
  border:1px solid rgba(0,0,0,.05);
  animation:_ewSlide .28s cubic-bezier(.22,.68,0,1.18);
}

@media(max-width:520px){
  .${Q.overlay}{display:block;}
  .${Q.panel}{
    bottom:0;right:0;left:0;width:100%;
    border-radius:20px 20px 0 0;height:92dvh;
    animation:_ewSlideM .26s cubic-bezier(.22,.68,0,1.1);
  }
}

/* Header */
.${Q.hdr}{
  background:${CFG.G};flex-shrink:0;
  padding:0 16px;background-image:linear-gradient(135deg,${CFG.G},#d14827);
  box-shadow:0 4px 16px rgba(0,0,0,.08);
}
.${Q.hrow}{
  display:flex;align-items:center;justify-content:space-between;
  padding:15px 0;gap:8px;
}
.${Q.hico}{
  width:38px;height:38px;border-radius:12px;font-size:18px;
  background:rgba(255,255,255,.18);flex-shrink:0;
  display:flex;align-items:center;justify-content:center;
  box-shadow:0 2px 8px rgba(0,0,0,.1);
}
.${Q.htit}{color:#fff;font-size:16px;font-weight:800;letter-spacing:-.3px;white-space:nowrap;text-shadow:0 1px 2px rgba(0,0,0,.1);}
.${Q.hsub}{color:rgba(255,255,255,.65);font-size:11.5px;margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-weight:500;}
.${Q.hbtn}{
  background:rgba(255,255,255,.12);border:none;border-radius:10px;color:rgba(255,255,255,.85);
  width:32px;height:32px;display:flex;align-items:center;justify-content:center;
  cursor:pointer;padding:0;flex-shrink:0;transition:all .15s;
  box-shadow:0 1px 3px rgba(0,0,0,.1);
}
.${Q.hbtn}:hover{background:rgba(255,255,255,.25);transform:translateY(-1px);box-shadow:0 2px 6px rgba(0,0,0,.12);}

/* Role pill */
.${Q.rpill}{
  display:flex;align-items:center;gap:5px;padding:5px 10px;
  background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.25);
  border-radius:11px;cursor:pointer;color:#fff;font-family:inherit;
  font-size:0;transition:all .14s;max-width:140px;
  box-shadow:0 2px 6px rgba(0,0,0,.1);
}
.${Q.rpill}:hover{background:rgba(255,255,255,.22);border-color:rgba(255,255,255,.35);transform:translateY(-1px);box-shadow:0 3px 10px rgba(0,0,0,.12);}
.${Q.rmenu}{
  position:absolute;right:0;top:calc(100% + 10px);background:#fff;
  border-radius:14px;box-shadow:0 14px 40px rgba(0,0,0,.14),0 4px 12px rgba(0,0,0,.08);
  padding:8px;min-width:156px;z-index:2147483641;
  border:1px solid rgba(0,0,0,.06);animation:_ewFade .15s ease;
}
.${Q.ropt}{
  display:block;width:100%;text-align:left;padding:9px 13px;
  border:none;background:none;border-radius:9px;cursor:pointer;
  font-size:13px;color:#444;font-family:inherit;transition:all .11s;font-weight:500;
}
.${Q.ropt}:hover{background:#f3f3f9;transform:translateX(2px);}
.${Q.roptA}{background:${CFG.P} !important;color:#fff !important;font-weight:600 !important;}

/* Chat area */
.${Q.chatWrap}{position:relative;flex:1;min-height:0;display:flex;flex-direction:column;}
.${Q.chat}{
  flex:1;overflow-y:auto;padding:18px 16px 14px;
  scrollbar-width:thin;scrollbar-color:#d8d8e0 transparent;
  background:linear-gradient(180deg,#f8f8fc 0%,#fafafa 50%,#fff 100%);
}
.${Q.chat}::-webkit-scrollbar{width:6px;}
.${Q.chat}::-webkit-scrollbar-thumb{background:#d8d8e0;border-radius:6px;transition:background .2s;}
.${Q.chat}::-webkit-scrollbar-thumb:hover{background:#c0c0ce;}

/* Bubbles */
.${Q.bwrap}{display:flex;align-items:flex-end;gap:10px;margin-bottom:16px;}
.${Q.bwrapU}{justify-content:flex-end;}

.${Q.bav}{
  width:32px;height:32px;border-radius:50%;background:${CFG.P};flex-shrink:0;
  display:flex;align-items:center;justify-content:center;font-size:16px;
  box-shadow:0 4px 12px rgba(0,0,0,.15),0 1px 3px rgba(0,0,0,.2);
}

.${Q.ubub}{
  max-width:78%;padding:12px 16px;background:${CFG.P};color:#fff;
  border-radius:20px 20px 6px 20px;font-size:14px;line-height:1.6;font-weight:500;
  word-break:break-word;box-shadow:0 6px 20px rgba(39,22,39,.2),0 2px 6px rgba(0,0,0,.1);
}

.${Q.bbub}{
  max-width:88%;padding:12px 16px;background:#f9f9fb;color:#222;
  border-radius:6px 20px 20px 20px;border:1px solid #e8e8f0;
  box-shadow:0 2px 8px rgba(0,0,0,.04),0 1px 3px rgba(0,0,0,.08);font-size:13.5px;
  line-height:1.6;word-break:break-word;font-weight:500;
}

/* Thinking dots */
.${Q.dots}{display:flex;align-items:center;gap:5px;padding:5px 2px;}
.${Q.dot}{
  width:7px;height:7px;border-radius:50%;background:#c5c5d0;
  animation:_ewPulse 1.3s ease-in-out infinite;
}
.${Q.dot}:nth-child(2){animation-delay:.18s;}
.${Q.dot}:nth-child(3){animation-delay:.36s;}

/* Streaming cursor */
.${Q.cur}{
  display:inline-block;width:2px;height:.95em;background:#555;
  vertical-align:text-bottom;margin-left:1px;border-radius:1px;
  animation:_ewBlink .55s step-end infinite;
}

/* Timestamp */
.${Q.ts}{font-size:9.5px;color:#c8c8d0;margin-top:9px;}
.${Q.errtxt}{color:#dc2626;font-size:13px;}

/* Inventory card */
.${Q.icard}{
  margin-top:14px;border-radius:16px;border:1px solid #e5e5f0;
  background:#f9f9fd;overflow:hidden;
  box-shadow:0 4px 14px rgba(0,0,0,.05),0 1px 4px rgba(0,0,0,.08);
}
.${Q.ihdr}{
  display:flex;align-items:center;gap:12px;padding:13px 14px;
  border-bottom:1px solid #ececf3;background:linear-gradient(135deg,#fafbfc 0%,#f8f9fd 100%);
}
.${Q.iico}{
  width:40px;height:40px;border-radius:12px;background:${CFG.P};flex-shrink:0;
  display:flex;align-items:center;justify-content:center;font-size:20px;
  box-shadow:0 3px 10px rgba(102,102,102,0.2);
}
.${Q.iname}{font-size:14px;font-weight:800;color:${CFG.P};line-height:1.35;}
.${Q.imeta}{font-size:11px;color:#aaa;margin-top:4px;font-weight:500;}
.${Q.iid}{font-size:10px;color:#bbb;font-family:monospace;flex-shrink:0;font-weight:600;}
.${Q.stgrid}{
  display:grid;grid-template-columns:repeat(4,1fr);gap:9px;padding:14px 14px;
}
@media(max-width:400px){.${Q.stgrid}{grid-template-columns:repeat(2,1fr);}}
.${Q.sbadge}{
  text-align:center;padding:11px 6px;border-radius:12px;
  background:#fff;border:1px solid #e5e5ed;
  box-shadow:0 2px 6px rgba(0,0,0,.04);
  transition:all .12s;
}
.${Q.sbadge}:hover{transform:translateY(-2px);box-shadow:0 4px 12px rgba(0,0,0,.08);}
.${Q.sbadgeH}{background:${CFG.P};border-color:${CFG.P};box-shadow:0 3px 12px rgba(102,102,102,0.2);}

/* Dropdown card */
.${Q.dcard}{
  margin-top:14px;border-radius:16px;border:1px solid #e5e5f0;
  background:#f9f9fd;overflow:hidden;
  box-shadow:0 4px 14px rgba(0,0,0,.05),0 1px 4px rgba(0,0,0,.08);
}
.${Q.dmsg}{
  font-size:12.5px;color:#555;padding:12px 14px;
  border-bottom:1px solid #ececf3;line-height:1.55;font-weight:500;
}
.${Q.dlist}{padding:8px;}
.${Q.ditem}{
  display:flex;align-items:center;gap:10px;width:100%;text-align:left;
  padding:11px 12px;border:1px solid #e8e8f0;border-radius:11px;
  cursor:pointer;font-family:inherit;background:#fff;margin-bottom:6px;
  transition:all .13s;font-weight:500;
  box-shadow:0 1px 3px rgba(0,0,0,.04);
}
.${Q.ditem}:hover{background:#f5f5fa;border-color:#d5d5e5;transform:translateY(-1px);box-shadow:0 3px 8px rgba(0,0,0,.08);}
.${Q.ditem}:last-child{margin-bottom:0;}
.${Q.did}{font-size:10px;color:#b8b8c8;font-family:monospace;flex-shrink:0;font-weight:600;}
.${Q.dname}{flex:1;font-size:13px;font-weight:600;color:${CFG.P};}

/* Input bar */
.${Q.ibar}{
  display:flex;gap:10px;padding:13px 14px;border-top:1px solid #ececf3;
  background:#fff;flex-shrink:0;align-items:center;
}
.${Q.iinn}{
  flex:1;display:flex;align-items:center;gap:10px;background:#f5f5fa;
  border-radius:16px;padding:0 14px;border:1.5px solid #e8e8f0;
  transition:all .15s;
  box-shadow:0 2px 6px rgba(0,0,0,.03);
}
.${Q.iinn}:focus-within{border-color:${CFG.A};background:#fff;box-shadow:0 4px 12px rgba(${CFG.A},0.1);}
.${Q.iinput}{
  flex:1;border:none;background:transparent;font-size:14px;
  color:#222;outline:none;padding:12px 0;font-family:inherit;font-weight:500;
}
@media(max-width:520px){.${Q.iinput}{font-size:16px;}}
.${Q.clr}{
  background:none;border:none;cursor:pointer;color:#d0d0d8;
  font-size:20px;line-height:1;padding:0;flex-shrink:0;transition:all .12s;
}
.${Q.clr}:hover{color:#999;transform:scale(1.1);}
.${Q.send}{
  width:44px;height:44px;border-radius:14px;border:none;
  background:${CFG.G};display:flex;align-items:center;justify-content:center;
  cursor:pointer;flex-shrink:0;transition:all .15s;
  box-shadow:0 4px 14px rgba(230,94,56,0.3),0 2px 6px rgba(0,0,0,.1);
  font-weight:600;
}
.${Q.send}:hover:not(:disabled){transform:translateY(-2px);box-shadow:0 6px 20px rgba(230,94,56,0.4),0 3px 8px rgba(0,0,0,.12);}
.${Q.send}:active:not(:disabled){transform:translateY(0);}
.${Q.send}:disabled{opacity:.5;cursor:not-allowed;}
.${Q.spin}{
  display:inline-block;width:16px;height:16px;
  border:2.5px solid rgba(255,255,255,.3);border-top-color:#fff;
  border-radius:50%;animation:_ewSpin .7s linear infinite;
}

/* Tab bar — Chrome style */
.${Q.tabbar}{
  display:flex;align-items:flex-end;overflow-x:auto;scrollbar-width:none;
  padding:8px 10px 0;gap:2px;flex-shrink:0;
  background:linear-gradient(180deg,#404043 0%,#3a3a3d 100%);min-height:38px;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.1);
}
.${Q.tabbar}::-webkit-scrollbar{display:none;}
.${Q.tab}{
  display:flex;align-items:center;gap:6px;
  padding:0 10px 0 12px;height:28px;
  flex:1 1 0;min-width:76px;max-width:160px;
  border:1px solid rgba(255,255,255,.08);border-bottom:none;
  cursor:pointer;font-family:inherit;font-size:11.5px;font-weight:600;
  color:rgba(255,255,255,.55);background:rgba(255,255,255,.07);
  white-space:nowrap;transition:all .12s;
  border-radius:8px 8px 0 0;overflow:hidden;outline:none;
  box-shadow:0 1px 3px rgba(0,0,0,.1);
}
.${Q.tab}:hover{background:rgba(255,255,255,.13);color:rgba(255,255,255,.9);transform:translateY(-1px);}
.${Q.tabA}{
  background:#fff !important;color:#222 !important;
  font-weight:700 !important;border-color:rgba(0,0,0,.08) !important;
  border-bottom:1px solid #fff !important;
  box-shadow:0 -2px 8px rgba(0,0,0,.1) !important;
}
.${Q.tabX}{
  background:none;border:none;color:rgba(255,255,255,.35);
  cursor:pointer;font-size:13px;line-height:1;
  padding:0;flex-shrink:0;transition:all .1s;
  width:18px;height:18px;border-radius:50%;
  display:flex;align-items:center;justify-content:center;
}
.${Q.tabX}:hover{color:rgba(255,255,255,.95);background:rgba(255,255,255,.2);}
.${Q.tabA} .${Q.tabX}{color:rgba(0,0,0,.35);}
.${Q.tabA} .${Q.tabX}:hover{background:rgba(0,0,0,.12);color:rgba(0,0,0,.8);}
.${Q.tabP}{
  background:none;border:none;color:rgba(255,255,255,.45);
  cursor:pointer;font-size:16px;line-height:1;padding:4px 10px;
  flex-shrink:0;transition:all .12s;border-radius:7px;
  align-self:center;margin-bottom:4px;
}
.${Q.tabP}:hover{color:#fff;background:rgba(255,255,255,.15);transform:translateY(-1px);}

/* Keyframes */
@keyframes _ewSpin  {to{transform:rotate(360deg)}}
@keyframes _ewBlink {0%,100%{opacity:1}50%{opacity:0}}
@keyframes _ewPulse {0%,80%,100%{transform:scale(.65);opacity:.35}40%{transform:scale(1);opacity:1}}
@keyframes _ewFade  {from{opacity:0}to{opacity:1}}
@keyframes _ewSlide {from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:translateY(0)}}
@keyframes _ewSlideM{from{opacity:0;transform:translateY(100%)}to{opacity:1;transform:translateY(0)}}
@keyframes _ewSlideL{from{opacity:0;transform:translateX(-18px)}to{opacity:1;transform:translateX(0)}}
`;
  document.head.appendChild(el);
})();

// ─────────────────────────────────────────────────────────────────────────────
// API
// ─────────────────────────────────────────────────────────────────────────────
async function askBot(query, history, role) {
  const r = await fetch(CFG.apiUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, history, ui_filters: {}, role }),
  });
  if (!r.ok) throw new Error(`Server error: ${r.status}`);
  return r.json();
}

// ─────────────────────────────────────────────────────────────────────────────
// HELPERS
// ─────────────────────────────────────────────────────────────────────────────
function ts() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function renderMarkdown(text) {
  if (!text) return null;
  return text.split("\n").map((line, li) => (
    <span key={li}>
      {li > 0 && <br />}
      {line.split(/(\*\*[^*]+\*\*)/g).map((part, pi) =>
        part.startsWith("**") && part.endsWith("**")
          ? <strong key={pi}>{part.slice(2, -2)}</strong>
          : part
      )}
    </span>
  ));
}

function UnknownPart({ part }) {
  return (
    <div style={{
      marginTop: 10,
      padding: 10,
      borderRadius: 10,
      background: "#fff7ed",
      border: "1px solid #fed7aa",
      fontSize: 12,
      fontFamily: "monospace",
      whiteSpace: "pre-wrap"
    }}>
      <strong>Unknown type: {part.type}</strong>
      {"\n"}
      {JSON.stringify(part, null, 2)}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// TYPEWRITER — ChatGPT-style character streaming
// ─────────────────────────────────────────────────────────────────────────────
function TypeWriter({ text, onDone }) {
  const [shown, setShown] = useState("");
  const [done, setDone]   = useState(!text);
  const rafRef = useRef(null);
  const onDoneRef = useRef(onDone);
  useEffect(() => {
    onDoneRef.current = onDone;
  }, [onDone]);

  useEffect(() => {
    if (!text) {
      onDoneRef.current?.();
      return;
    }

    let idx = 0;
    let last = null;

    function frame(t) {
      if (last === null) last = t;
      const dt = t - last;
      last = t;
      const add = Math.max(1, Math.round((CFG.CPS * dt) / 1000));
      idx = Math.min(idx + add, text.length);
      setShown(text.slice(0, idx));
      if (idx < text.length) {
        rafRef.current = requestAnimationFrame(frame);
      } else {
        setDone(true);
        onDoneRef.current?.();
      }
    }

    rafRef.current = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(rafRef.current);
  }, [text]);

  return (
    <>
      {renderMarkdown(shown)}
      {!done && <span className={Q.cur} />}
    </>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// STOCK BADGE
// ─────────────────────────────────────────────────────────────────────────────
function StockBadge({ label, value, hl }) {
  return (
    <div className={`${Q.sbadge}${hl ? " " + Q.sbadgeH : ""}`}>
      <div style={{ fontSize: 15, fontWeight: 700, color: hl ? "#fff" : CFG.P }}>
        {value ?? 0}
      </div>
      <div style={{ fontSize: 9.5, marginTop: 3, color: hl ? "rgba(255,255,255,.65)" : "#aaa" }}>
        {label}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// INVENTORY CARD
// ─────────────────────────────────────────────────────────────────────────────
function InventoryCard({ result }) {
  const { inventory: inv, total_stock, finish_stock, semi_finish_stock, machining_stock } = result;
  const name  = inv?.name?.trim() || "";
  const invId = inv?.id || null;

  const [activeSection, setActiveSection] = useState(null);
  const [sectionData,   setSectionData]   = useState({});
  const [secLoading,    setSecLoading]    = useState(false);
  const [secSearch,     setSecSearch]     = useState({});
  const [secPage,       setSecPage]       = useState({});

  const sq    = (k) => secSearch[k] || "";
  const sp    = (k) => secPage[k]   || 1;
  const setSQ = (k,v) => { setSecSearch(p=>({...p,[k]:v})); setSecPage(p=>({...p,[k]:1})); };
  const setSP = (k,v) => setSecPage(p=>({...p,[k]:v}));

  const BTNS = [
    { key: "po-history", icon: "📋", label: "PO History"  },
    { key: "suppliers",  icon: "🏭", label: "Suppliers"   },
    { key: "stock-log",  icon: "📈", label: "Stock Log"   },
    { key: "grns",       icon: "🚚", label: "GRNs"        },
  ];

  const handleBtn = async (key) => {
    if (activeSection === key) { setActiveSection(null); return; }
    setActiveSection(key);
    if (sectionData[key]) return;
    if (!invId) { setSectionData((p) => ({ ...p, [key]: { error: "Inventory ID not available" } })); return; }
    setSecLoading(true);
    try {
      const base = API_BASE.replace(/\/$/, "");
      const res  = await fetch(`${base}/inventory/${invId}/${key}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setSectionData((p) => ({ ...p, [key]: data }));
    } catch (e) {
      setSectionData((p) => ({ ...p, [key]: { error: e.message } }));
    } finally {
      setSecLoading(false);
    }
  };

  const renderSection = () => {
    const data = sectionData[activeSection];
    if (secLoading && !data)
      return <div style={{ padding: "10px 13px", fontSize: 12, color: "#999" }}>Loading…</div>;
    if (!data) return null;
    if (data.error)
      return <div style={{ padding: "10px 13px", fontSize: 12, color: "#dc2626" }}>{data.error}</div>;

    const rows = data.rows || [];

    const mkPaged = (key, toStr) => {
      const q = sq(key), pg = sp(key);
      const filtered = q.trim() ? rows.filter(r => fuzzyMatch(q, toStr(r))) : rows;
      const totalPgs = Math.max(1, Math.ceil(filtered.length / PGSZ));
      const safePg   = Math.min(pg, totalPgs);
      const paged    = filtered.slice((safePg-1)*PGSZ, safePg*PGSZ);
      return { q, pg: safePg, filtered, totalPgs, paged, setQ: v=>setSQ(key,v), setPage: v=>setSP(key,v), total: rows.length };
    };

    /* ── PO History ── */
    if (activeSection === "po-history") {
      if (!rows.length) return <div style={{ padding: "10px 13px", fontSize: 12, color: "#999" }}>No POs found.</div>;
      const { q, pg, filtered, totalPgs, paged, setQ, setPage, total } = mkPaged("po-history",
        r => `${r.po_number} ${r.status} ${r.supplier} ${r.ordered} ${r.unit_price} ${dateSearchStr(r.date)}`);
      return (
        <div>
          <SectionSearchBar q={q} setQ={setQ} page={pg} setPage={setPage} totalPgs={totalPgs} total={total} filtered={filtered.length} />
          <div style={{ overflowY: "auto" }}>
            {paged.length === 0
              ? <div style={{ padding:"10px 13px", fontSize:12, color:"#999" }}>No results for "{q}"</div>
              : paged.map((r, i) => {
                const sc = r.status === "Completed" ? "#16a34a" : r.status === "Cancelled" ? "#dc2626" : "#d97706";
                return (
                  <div key={i} className={Q.ditem} style={{ cursor:"default", alignItems:"flex-start" }}>
                    <span className={Q.did} style={{ marginTop:2 }}>#{(pg-1)*PGSZ+i+1}</span>
                    <span className={Q.dname} style={{ minWidth:0 }}>
                      <span style={{ display:"block", fontFamily:"monospace", fontSize:11 }}>{r.po_number}</span>
                      <span style={{ fontSize:10, color:"#999", fontWeight:400 }}>{r.supplier} · {r.date}</span>
                    </span>
                    <span style={{ display:"flex", flexDirection:"column", alignItems:"flex-end", flexShrink:0, gap:2 }}>
                      <span style={{ fontSize:11, fontWeight:700, color:CFG.P }}>{fmtINR(r.line_total)}</span>
                      <span style={{ fontSize:9, fontWeight:700, color:sc, background:sc+"18", padding:"1px 5px", borderRadius:4 }}>{r.status}</span>
                    </span>
                  </div>
                );
              })
            }
          </div>
        </div>
      );
    }

    /* ── Suppliers ── */
    if (activeSection === "suppliers") {
      if (!rows.length) return <div style={{ padding: "10px 13px", fontSize: 12, color: "#999" }}>No suppliers found.</div>;
      const { q, pg, filtered, totalPgs, paged, setQ, setPage, total } = mkPaged("suppliers",
        r => `${r.name} ${r.city} ${r.po_count} ${r.total_ordered}`);
      return (
        <div>
          <SectionSearchBar q={q} setQ={setQ} page={pg} setPage={setPage} totalPgs={totalPgs} total={total} filtered={filtered.length} />
          <div style={{ overflowY: "auto" }}>
            {paged.length === 0
              ? <div style={{ padding:"10px 13px", fontSize:12, color:"#999" }}>No results for "{q}"</div>
              : paged.map((r, i) => (
                <div key={i} className={Q.ditem} style={{ cursor:"default" }}>
                  <span className={Q.did}>#{(pg-1)*PGSZ+i+1}</span>
                  <span className={Q.dname} style={{ minWidth:0 }}>
                    <span style={{ display:"block" }}>{r.name}</span>
                    <span style={{ fontSize:10, color:"#999", fontWeight:400 }}>{r.city||"—"} · {r.po_count} PO{r.po_count!==1?"s":""}</span>
                  </span>
                  <span style={{ display:"flex", flexDirection:"column", alignItems:"flex-end", flexShrink:0, gap:2 }}>
                    <span style={{ fontSize:11, fontWeight:700, color:CFG.P }}>{fmtINR(r.total_ordered)}</span>
                    {r.min_price>0 && <span style={{ fontSize:9, color:"#aaa" }}>₹{r.min_price}–₹{r.max_price}/unit</span>}
                  </span>
                </div>
              ))
            }
          </div>
        </div>
      );
    }

    /* ── Stock Log ── */
    if (activeSection === "stock-log") {
      if (!rows.length) return <div style={{ padding: "10px 13px", fontSize: 12, color: "#999" }}>No transactions found.</div>;
      const { q, pg, filtered, totalPgs, paged, setQ, setPage, total } = mkPaged("stock-log",
        r => `${r.type} ${r.ref_type} ${r.ref_no} ${r.qty} ${r.remarks} ${dateSearchStr(r.date)}`);
      return (
        <div>
          <SectionSearchBar q={q} setQ={setQ} page={pg} setPage={setPage} totalPgs={totalPgs} total={total} filtered={filtered.length} />
          {data.current_stock !== undefined && (
            <div style={{ padding:"4px 12px 6px", fontSize:11.5, fontWeight:700, color: data.current_stock>=0?"#16a34a":"#dc2626" }}>
              Current Stock: {data.current_stock}
            </div>
          )}
          <div style={{ overflowY: "auto" }}>
            {paged.length === 0
              ? <div style={{ padding:"10px 13px", fontSize:12, color:"#999" }}>No results for "{q}"</div>
              : paged.map((r, i) => {
                const isIn = (r.type||"").toLowerCase()==="in";
                return (
                  <div key={i} className={Q.ditem} style={{ cursor:"default" }}>
                    <span className={Q.did} style={{ color:isIn?"#16a34a":"#dc2626", fontWeight:700 }}>{isIn?"▲":"▼"}</span>
                    <span className={Q.dname} style={{ minWidth:0 }}>
                      <span style={{ display:"block" }}>{r.ref_type||r.type}{r.ref_no?` · ${r.ref_no}`:""}</span>
                      <span style={{ fontSize:10, color:"#999", fontWeight:400 }}>{r.date}{r.remarks?` · ${r.remarks}`:""}</span>
                    </span>
                    <span style={{ display:"flex", flexDirection:"column", alignItems:"flex-end", flexShrink:0, gap:2 }}>
                      <span style={{ fontSize:11, fontWeight:700, color:isIn?"#16a34a":"#dc2626" }}>{isIn?"+":"-"}{r.qty}</span>
                      <span style={{ fontSize:9, color:"#aaa" }}>bal: {r.balance}</span>
                    </span>
                  </div>
                );
              })
            }
          </div>
        </div>
      );
    }

    /* ── GRNs ── */
    if (activeSection === "grns") {
      if (!rows.length) return <div style={{ padding: "10px 13px", fontSize: 12, color: "#999" }}>No GRNs found.</div>;
      const { q, pg, filtered, totalPgs, paged, setQ, setPage, total } = mkPaged("grns",
        r => `${r.grn_number} ${r.invoice_no} ${r.placement} ${r.received} ${r.accepted} ${r.rejected} ${dateSearchStr(r.date)}`);
      return (
        <div>
          <SectionSearchBar q={q} setQ={setQ} page={pg} setPage={setPage} totalPgs={totalPgs} total={total} filtered={filtered.length} />
          <div style={{ overflowY: "auto" }}>
            {paged.length === 0
              ? <div style={{ padding:"10px 13px", fontSize:12, color:"#999" }}>No results for "{q}"</div>
              : paged.map((r, i) => (
                <div key={i} className={Q.ditem} style={{ cursor:"default", alignItems:"flex-start" }}>
                  <span className={Q.did} style={{ marginTop:2 }}>#{(pg-1)*PGSZ+i+1}</span>
                  <span className={Q.dname} style={{ minWidth:0 }}>
                    <span style={{ display:"block", fontFamily:"monospace", fontSize:11 }}>{r.grn_number}</span>
                    <span style={{ fontSize:10, color:"#999", fontWeight:400 }}>
                      {r.date}{r.invoice_no?` · Inv: ${r.invoice_no}`:""}{r.placement?` · ${r.placement}`:""}
                    </span>
                  </span>
                  <span style={{ display:"flex", flexDirection:"column", alignItems:"flex-end", flexShrink:0, gap:2 }}>
                    <span style={{ fontSize:11, fontWeight:700, color:"#16a34a" }}>✓ {r.accepted}</span>
                    {r.rejected>0 && <span style={{ fontSize:9, color:"#dc2626" }}>✗ {r.rejected}</span>}
                  </span>
                </div>
              ))
            }
          </div>
        </div>
      );
    }

    return null;
  };

  const infoRows = [
    { label: "Unit",      value: inv?.unit      },
    { label: "Grade",     value: inv?.grade     },
    { label: "Model",     value: inv?.model     },
    { label: "Placement", value: inv?.placement },
  ].filter((r) => r.value && r.value !== "N/A");

  return (
    <div className={Q.icard} style={{ marginTop: 10 }}>
      <div className={Q.ihdr}>
        <div className={Q.iico}>📦</div>
        <div style={{ minWidth: 0, flex: 1 }}>
          <div className={Q.iname}>{name}</div>
          <div className={Q.imeta}>
            {[inv?.category, inv?.placement].filter(Boolean).join(" · ")}
          </div>
        </div>
        <div className={Q.iid}>#{inv?.id}</div>
      </div>

      <div className={Q.stgrid}>
        <StockBadge label="Total"       value={total_stock}       hl />
        <StockBadge label="Finish"      value={finish_stock}         />
        <StockBadge label="Semi-Finish" value={semi_finish_stock}    />
        <StockBadge label="Machining"   value={machining_stock}      />
      </div>

      {infoRows.length > 0 && (
        <div style={{ padding: "6px 13px 0", display: "flex", flexWrap: "wrap", gap: "4px 16px" }}>
          {infoRows.map((r) => (
            <div key={r.label} style={{ display: "flex", gap: 5, fontSize: 11.5 }}>
              <span style={{ color: "#aaa" }}>{r.label}:</span>
              <span style={{ fontWeight: 600, color: "#555" }}>{r.value}</span>
            </div>
          ))}
        </div>
      )}

      <div style={{ display: "flex", flexWrap: "wrap", gap: 6, padding: "8px 13px 10px" }}>
        {BTNS.map((b) => {
          const active = activeSection === b.key;
          return (
            <button
              key={b.key}
              onClick={() => handleBtn(b.key)}
              style={{
                display: "flex", alignItems: "center", gap: 4,
                padding: "5px 10px", borderRadius: 20,
                border: `1px solid ${active ? CFG.A : CFG.A + "44"}`,
                background: active ? CFG.A : `${CFG.A}0f`,
                color: active ? "#fff" : CFG.A,
                fontSize: 11.5, fontWeight: 600,
                cursor: "pointer", fontFamily: "inherit",
                transition: "all .12s",
              }}
            >
              {b.icon} {b.label}
            </button>
          );
        })}
      </div>

      {activeSection && (
        <div style={{ borderTop: "1px solid #e8e8f2" }}>
          {renderSection()}
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// DROPDOWN CARD
// ─────────────────────────────────────────────────────────────────────────────
function DropdownCard({ dropdown, onSelect }) {
  const items = dropdown.items || [];
  const [q, setQ] = useState("");
  const [page, setPage] = useState(1);
  const filtered = q.trim() ? items.filter(it => fuzzyMatch(q, it.name || "")) : items;
  const totalPgs = Math.max(1, Math.ceil(filtered.length / PGSZ));
  const safePg   = Math.min(page, totalPgs);
  const paged    = filtered.slice((safePg - 1) * PGSZ, safePg * PGSZ);
  const setQS    = (v) => { setQ(v); setPage(1); };

  return (
    <div className={Q.dcard}>
      <div className={Q.dmsg}>{renderMarkdown(dropdown.message)}</div>
      {items.length >= 5 && (
        <SectionSearchBar q={q} setQ={setQS} page={safePg} setPage={setPage}
          totalPgs={totalPgs} total={items.length} filtered={filtered.length} />
      )}
      <div className={Q.dlist}>
        {paged.length === 0
          ? <div style={{ padding:"8px 13px", fontSize:12, color:"#999" }}>No results for "{q}"</div>
          : paged.map((item) => (
            <button
              key={item.id}
              className={Q.ditem}
              onClick={() => onSelect(item.name?.trim())}
            >
              <span className={Q.did}>#{item.id}</span>
              <span className={Q.dname}>{item.name?.trim()}</span>
              <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#aaa" strokeWidth="2.5" strokeLinecap="round">
                <polyline points="9 18 15 12 9 6" />
              </svg>
            </button>
          ))
        }
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// CONFIRM RESOLUTION CARD  — backend asked "did you mean X?"
// User can click 👍 / 👎 / numbered candidate. Selection is sent as a regular
// chat message so the backend's pending_resolution check picks it up.
// ─────────────────────────────────────────────────────────────────────────────
function ConfirmCard({ part, onPick, onDirectPick }) {
  const cands   = part.candidates    || [];
  const ids     = part.candidate_ids || [];
  const category= part.category      || "";
  const single  = cands.length <= 1;
  const [picked, setPicked] = useState(null);

  const pick = (i) => {
    setPicked(i);
    const id = ids[i];
    if (id && onDirectPick) {
      onDirectPick(id, category, cands[i]);
    } else {
      onPick(String(i + 1));
    }
  };

  const thumbsUp = () => {
    setPicked(0);
    const id = ids[0];
    if (id && onDirectPick) {
      onDirectPick(id, category, cands[0]);
    } else {
      onPick("👍");
    }
  };

  return (
    <div className={Q.dcard} style={{ marginTop: 10 }}>
      <div className={Q.dmsg}>{renderMarkdown(part.message)}</div>

      {!single && (
        <div className={Q.dlist}>
          {cands.map((name, i) => (
            <button
              key={i}
              className={Q.ditem}
              onClick={() => pick(i)}
              style={{ background: picked === i ? `${CFG.A}15` : "#fff", borderColor: picked === i ? CFG.A+"55" : "#eaeaf2" }}
            >
              <span className={Q.did}>{i + 1}</span>
              <span className={Q.dname}>{name}</span>
            </button>
          ))}
        </div>
      )}

      <div style={{ display: "flex", gap: 8, padding: "8px 10px 10px" }}>
        <button
          className={Q.ditem}
          style={{ flex: 1, justifyContent: "center", margin: 0 }}
          onClick={thumbsUp}
        >
          <span className={Q.dname} style={{ textAlign: "center" }}>👍 Yes</span>
        </button>
        <button
          className={Q.ditem}
          style={{ flex: 1, justifyContent: "center", margin: 0 }}
          onClick={() => onPick("👎")}
        >
          <span className={Q.dname} style={{ textAlign: "center" }}>👎 No</span>
        </button>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// LIST CARD  — generic renderer for po_list / po_summary / project_list /
// supplier_list. Picks reasonable display fields from each row.
// ─────────────────────────────────────────────────────────────────────────────
function fmtINR(n) {
  if (n === null || n === undefined || n === "") return null;
  const num = Number(n);
  if (Number.isNaN(num)) return null;
  return "₹" + num.toLocaleString("en-IN", { maximumFractionDigits: 0 });
}

// ─── Fuzzy search utilities ───────────────────────────────────────────────────
const PGSZ = 8;

const _MO  = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"];
const _MOF = ["january","february","march","april","may","june","july","august",
              "september","october","november","december"];
function dateSearchStr(s) {
  if (!s) return "";
  const raw = String(s).toLowerCase();
  try {
    const d = new Date(s);
    if (isNaN(d.getTime())) return raw;
    return `${raw} ${d.getDate()} ${String(d.getDate()).padStart(2,"0")} `
         + `${_MO[d.getMonth()]} ${_MOF[d.getMonth()]} `
         + `${d.getMonth()+1} ${d.getFullYear()}`;
  } catch { return raw; }
}

function fuzzyMatch(query, text) {
  if (!query || !query.trim()) return true;
  const q = query.toLowerCase().trim();
  const t = String(text ?? "").toLowerCase();
  if (!t) return false;
  if (t.includes(q)) return true;
  const tokens = q.split(/\s+/).filter(Boolean);
  if (tokens.length > 1 && tokens.every(tok => t.includes(tok))) return true;
  if (q.length >= 3) {
    let qi = 0;
    for (let i = 0; i < t.length && qi < q.length; i++) {
      if (t[i] === q[qi]) qi++;
    }
    if (qi === q.length) return true;
  }
  return false;
}

function SectionSearchBar({ q, setQ, page, setPage, totalPgs, total, filtered }) {
  const showSearch = total >= 5;
  const showPager  = totalPgs > 1;
  if (!showSearch && !showPager) return null;
  const pgBtn = (dis) => ({
    border: "1px solid #e0e0ec", borderRadius: 5,
    background: dis ? "#f7f7fc" : "#fff", color: dis ? "#ccc" : "#555",
    cursor: dis ? "default" : "pointer",
    width: 22, height: 22, padding: 0,
    fontFamily: "inherit", fontSize: 13, lineHeight: "20px",
  });
  return (
    <div style={{ padding: "5px 8px 4px", borderBottom: "1px solid #f0f0f8", display: "flex", gap: 5, alignItems: "center" }}>
      {showSearch && (
        <div style={{ display: "flex", alignItems: "center", flex: 1, minWidth: 110, gap: 4,
                      background: "#f7f7fc", borderRadius: 8, padding: "3px 7px", border: "1px solid #e8e8f2" }}>
          <span style={{ color: "#ccc", fontSize: 10 }}>🔍</span>
          <input type="text" value={q} onChange={e => setQ(e.target.value)} placeholder="Search…"
            style={{ flex:1, border:"none", background:"none", fontSize:11.5,
                     outline:"none", fontFamily:"inherit", minWidth:0 }} />
          {q && <button onClick={() => setQ("")}
            style={{ border:"none", background:"none", cursor:"pointer", color:"#bbb", fontSize:13, padding:0, lineHeight:1 }}>✕</button>}
        </div>
      )}
      <div style={{ display:"flex", alignItems:"center", gap:4, fontSize:10.5, color:"#999", flexShrink:0 }}>
        {q.trim() && <span style={{ color:"#aaa" }}>{filtered}/{total}</span>}
        {showPager && (
          <>
            <button onClick={() => setPage(p => Math.max(1,p-1))} disabled={page<=1} style={pgBtn(page<=1)}>‹</button>
            <span style={{ minWidth:28, textAlign:"center" }}>{page}/{totalPgs}</span>
            <button onClick={() => setPage(p => Math.min(totalPgs,p+1))} disabled={page>=totalPgs} style={pgBtn(page>=totalPgs)}>›</button>
          </>
        )}
      </div>
    </div>
  );
}

function ListCard({ part }) {
  const rows = part.rows || [];
  if (!rows.length) return null;

  const [q, setQ] = useState("");
  const [page, setPage] = useState(1);

  const titleByType = {
    po_list:           "Purchase Orders",
    po_summary:        "PO Summary",
    project_list:      "Projects",
    supplier_list:     "Suppliers",
    supplier_items:    "Items Ordered",
    supplier_payments: "Payment History",
  };
  const title = titleByType[part.type] || "Results";

  const toStr = (r) =>
    `${r.po_number||""} ${r.supplier_name||""} ${r.name||""} ${r.grp||""} ${r.status||""} ${r.priority||""} ${dateSearchStr(r.date||r.po_date||"")}`;

  const filtered = q.trim() ? rows.filter(r => fuzzyMatch(q, toStr(r))) : rows;
  const totalPgs = Math.max(1, Math.ceil(filtered.length / PGSZ));
  const safePg   = Math.min(page, totalPgs);
  const paged    = filtered.slice((safePg - 1) * PGSZ, safePg * PGSZ);
  const setQS    = (v) => { setQ(v); setPage(1); };

  return (
    <div className={Q.dcard} style={{ marginTop: 10 }}>
      <div className={Q.dmsg}>
        {title} <span style={{ color: "#aaa" }}>· {rows.length}</span>
      </div>
      <SectionSearchBar q={q} setQ={setQS} page={safePg} setPage={setPage}
        totalPgs={totalPgs} total={rows.length} filtered={filtered.length} />
      <div className={Q.dlist}>
        {paged.length === 0
          ? <div style={{ padding:"8px 13px", fontSize:12, color:"#999" }}>No results for "{q}"</div>
          : paged.map((r, i) => {
            const label =
              r.po_number || r.supplier_name || r.name || r.grp || r.po || `Row ${(safePg-1)*PGSZ+i+1}`;
            const amount = fmtINR(r.total_amount ?? r.total ?? r.balance_amount ?? r.budget);
            const meta = [r.status, r.priority, r.po_count && `${r.po_count} POs`].filter(Boolean).join(" · ") || null;
            return (
              <div key={i} className={Q.ditem} style={{ cursor: "default" }}>
                <span className={Q.did}>#{(safePg-1)*PGSZ+i+1}</span>
                <span className={Q.dname} style={{ minWidth: 0 }}>
                  <span style={{ display: "block" }}>{label}</span>
                  {meta && <span style={{ fontSize: 10.5, color: "#999", fontWeight: 400 }}>{meta}</span>}
                </span>
                {amount && <span style={{ fontSize: 11, color: "#444", fontWeight: 600, flexShrink: 0 }}>{amount}</span>}
              </div>
            );
          })
        }
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// FEEDBACK BUTTONS  — small 👍 / 👎 below each bot reply, sends to /feedback
// ─────────────────────────────────────────────────────────────────────────────
function FeedbackButtons({ requestId, query, summary }) {
  const [voted, setVoted] = useState(null);
  if (!requestId) return null;

  const vote = async (rating) => {
    if (voted !== null) return;
    setVoted(rating);
    try {
      await fetch(CFG.feedbackUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          request_id: requestId,
          rating,
          query: query || "",
          response_summary: summary || "",
        }),
      });
    } catch {
      /* feedback failure is silent — never break UX */
    }
  };

  const btn = (active) => ({
    padding: "3px 9px",
    fontSize: 12,
    border: "1px solid #e0e0ec",
    borderRadius: 8,
    background: active ? "#f1f1f8" : "#fff",
    cursor: voted === null ? "pointer" : "default",
    opacity: voted === null || voted === active ? 1 : 0.4,
    transition: "all .12s",
  });

  return (
    <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
      <button style={btn(voted === 1)}  onClick={() => vote(1)}  title="This was helpful">
        {voted === 1 ? "✅ 👍" : "👍"}
      </button>
      <button style={btn(voted === -1)} onClick={() => vote(-1)} title="This was wrong">
        {voted === -1 ? "✅ 👎" : "👎"}
      </button>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// USER BUBBLE
// ─────────────────────────────────────────────────────────────────────────────
function UserBubble({ msg }) {
  return (
    <div className={`${Q.bwrap} ${Q.bwrapU}`}>
      <div className={Q.ubub}>
        <div style={{ fontWeight: 600 }}>{msg.content}</div>
        <div className={Q.ts} style={{ textAlign: "right" }}>{msg.time}</div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// SUPPLIER CARD
// ─────────────────────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────
// ACTION BUTTONS  — quick-action chips below a card
// ─────────────────────────────────────────────────────────────────────────────
function ActionChips({ actions, onAction }) {
  if (!actions?.length) return null;
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 6, padding: "8px 13px 10px" }}>
      {actions.map((a) => (
        <button
          key={a.label}
          onClick={() => onAction(a.query)}
          style={{
            display: "flex", alignItems: "center", gap: 4,
            padding: "5px 10px", borderRadius: 20,
            border: `1px solid ${CFG.A}44`, background: `${CFG.A}0f`,
            color: CFG.A, fontSize: 11.5, fontWeight: 600,
            cursor: "pointer", fontFamily: "inherit",
            transition: "background .12s, border-color .12s",
          }}
          onMouseEnter={(e) => { e.currentTarget.style.background = `${CFG.A}22`; }}
          onMouseLeave={(e) => { e.currentTarget.style.background = `${CFG.A}0f`; }}
        >
          {a.icon} {a.label}
        </button>
      ))}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// PO LIST INLINE  — clickable PO rows that expand a full POCard below
// ─────────────────────────────────────────────────────────────────────────────
function POListInline({ rows, supplierName }) {
  const [selectedId, setSelectedId] = useState(null);
  const [q,    setQ]    = useState("");
  const [page, setPage] = useState(1);

  const toStr = (r) =>
    `${r.po_number} ${r.status} ${r.total} ${r.balance} ${r.advance} ${r.expected} ${dateSearchStr(r.date)}`;

  const handleSetQ = (v) => { setQ(v); setPage(1); };

  const filtered  = q.trim() ? rows.filter(r => fuzzyMatch(q, toStr(r))) : rows;
  const totalPgs  = Math.max(1, Math.ceil(filtered.length / PGSZ));
  const safePg    = Math.min(page, totalPgs);
  const paged     = filtered.slice((safePg - 1) * PGSZ, safePg * PGSZ);
  const selectedRow = rows.find(r => r.id === selectedId) || null;

  return (
    <div style={{ padding: "6px 0 0" }}>
      <SectionSearchBar q={q} setQ={handleSetQ} page={safePg} setPage={setPage}
        totalPgs={totalPgs} total={rows.length} filtered={filtered.length} />
      <div>
        {paged.length === 0
          ? <div style={{ padding: "10px 13px", fontSize: 12, color: "#999" }}>No results for "{q}"</div>
          : paged.map((r) => {
            const sc     = r.status === "Completed" ? "#16a34a" : r.status === "Cancelled" ? "#dc2626" : "#d97706";
            const active = selectedId === r.id;
            return (
              <div key={r.id ?? r.po_number}
                className={Q.ditem}
                onClick={() => setSelectedId(prev => (prev === r.id ? null : r.id))}
                style={{ alignItems:"flex-start", background: active ? `${CFG.A}0f` : "#fff",
                         borderColor: active ? CFG.A+"55" : "#eaeaf2" }}>
                <span className={Q.dname} style={{ minWidth: 0 }}>
                  <span style={{ display:"block", fontFamily:"monospace", fontSize:11 }}>{r.po_number}</span>
                  <span style={{ fontSize:10, color:"#999", fontWeight:400 }}>{r.date}</span>
                </span>
                <span style={{ display:"flex", flexDirection:"column", alignItems:"flex-end", flexShrink:0, gap:2 }}>
                  <span style={{ fontSize:11, fontWeight:700, color:CFG.P }}>{fmtINR(r.total)}</span>
                  <span style={{ fontSize:9, fontWeight:700, color:sc, background:sc+"18", padding:"1px 5px", borderRadius:4 }}>{r.status}</span>
                </span>
              </div>
            );
          })
        }
      </div>

      {selectedRow && (
        <div style={{ margin: "8px 8px 4px" }}>
          <POCard part={{
            po_id:    selectedRow.id,
            po_no:    selectedRow.po_number,
            supplier: supplierName || "",
            date:     selectedRow.date,
            total:    selectedRow.total,
            advance:  selectedRow.advance,
            balance:  selectedRow.balance,
            status:   selectedRow.status,
          }} />
        </div>
      )}
    </div>
  );
}

function SupplierCard({ result }) {
  const s = result.supplier;
  if (!s) return null;

  const [activeSection, setActiveSection] = useState(null);
  const [sectionData,   setSectionData]   = useState({});
  const [secLoading,    setSecLoading]    = useState(false);
  const [secSearch,     setSecSearch]     = useState({});
  const [secPage,       setSecPage]       = useState({});

  const sq  = (k) => secSearch[k] || "";
  const sp  = (k) => secPage[k]   || 1;
  const setSQ = (k, v) => { setSecSearch(p => ({...p, [k]: v})); setSecPage(p => ({...p, [k]: 1})); };
  const setSP = (k, v) => setSecPage(p => ({...p, [k]: v}));

  const infoRows = [
    { label: "Code",   value: s.code   },
    { label: "Mobile", value: s.mobile },
    { label: "City",   value: s.city   },
    { label: "Email",  value: s.email  },
    { label: "GSTIN",  value: s.gstin  },
  ].filter((r) => r.value && r.value !== "N/A");

  const BTNS = [
    { key: "pos",      icon: "📋", label: "View POs"      },
    { key: "balance",  icon: "💰", label: "Balance"        },
    { key: "items",    icon: "📦", label: "Items Ordered"  },
    { key: "payments", icon: "🧾", label: "Payments"       },
  ];

  const handleBtn = async (key) => {
    if (activeSection === key) { setActiveSection(null); return; }
    setActiveSection(key);
    if (sectionData[key]) return;
    setSecLoading(true);
    try {
      const base = API_BASE.replace(/\/$/, "");
      const res  = await fetch(`${base}/supplier/${s.id}/${key}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setSectionData((p) => ({ ...p, [key]: data }));
    } catch (e) {
      setSectionData((p) => ({ ...p, [key]: { error: e.message } }));
    } finally {
      setSecLoading(false);
    }
  };

  const renderSection = () => {
    const data = sectionData[activeSection];
    if (secLoading && !data)
      return <div style={{ padding: "10px 13px", fontSize: 12, color: "#999" }}>Loading…</div>;
    if (!data) return null;
    if (data.error)
      return <div style={{ padding: "10px 13px", fontSize: 12, color: "#dc2626" }}>{data.error}</div>;

    /* ── Balance ── */
    if (activeSection === "balance") {
      const stats = [
        { label: "Total Ordered",   value: fmtINR(data.total_ordered)   },
        { label: "Total Advance",   value: fmtINR(data.total_advance)   },
        { label: "Pending Balance", value: fmtINR(data.pending_balance) },
        { label: "Open POs",        value: data.open_pos                 },
        { label: "Completed POs",   value: data.completed_pos            },
        { label: "Total POs",       value: data.total_pos                },
      ];
      return (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 7, padding: "10px 12px" }}>
          {stats.map((st) => (
            <div key={st.label} style={{ textAlign: "center", padding: "7px 4px", borderRadius: 9, background: "#fff", border: "1px solid #e2e2ef" }}>
              <div style={{ fontSize: 12.5, fontWeight: 700, color: CFG.P }}>{st.value ?? "—"}</div>
              <div style={{ fontSize: 9.5, marginTop: 2, color: "#aaa" }}>{st.label}</div>
            </div>
          ))}
        </div>
      );
    }

    /* ── View POs ── */
    if (activeSection === "pos") {
      const rows = data.rows || [];
      if (!rows.length) return <div style={{ padding: "10px 13px", fontSize: 12, color: "#999" }}>No POs found.</div>;
      return <POListInline rows={rows} supplierName={s.name} />;
    }

    /* ── Items Ordered ── */
    if (activeSection === "items") {
      const rows = data.rows || [];
      if (!rows.length) return <div style={{ padding: "10px 13px", fontSize: 12, color: "#999" }}>No items found.</div>;
      const toStr = (r) => `${r.name} ${r.unit} ${r.ordered} ${r.received}`;
      const q = sq("items"), pg = sp("items");
      const filtered = q.trim() ? rows.filter(r => fuzzyMatch(q, toStr(r))) : rows;
      const totalPgs = Math.max(1, Math.ceil(filtered.length / PGSZ));
      const safePg   = Math.min(pg, totalPgs);
      const paged    = filtered.slice((safePg-1)*PGSZ, safePg*PGSZ);
      return (
        <div>
          <SectionSearchBar q={q} setQ={v => setSQ("items",v)} page={safePg} setPage={v => setSP("items",v)}
            totalPgs={totalPgs} total={rows.length} filtered={filtered.length} />
          <div style={{ overflowY: "auto" }}>
            {paged.length === 0
              ? <div style={{ padding:"10px 13px", fontSize:12, color:"#999" }}>No results for "{q}"</div>
              : paged.map((r, i) => (
                <div key={i} className={Q.ditem} style={{ cursor: "default" }}>
                  <span className={Q.did}>#{(safePg-1)*PGSZ+i+1}</span>
                  <span className={Q.dname} style={{ minWidth: 0 }}>
                    <span style={{ display: "block" }}>{r.name}</span>
                    <span style={{ fontSize: 10, color: "#999", fontWeight: 400 }}>Received: {r.received} {r.unit}</span>
                  </span>
                  <span style={{ fontSize: 11, fontWeight: 700, color: CFG.P, flexShrink: 0 }}>{r.ordered} {r.unit}</span>
                </div>
              ))
            }
          </div>
        </div>
      );
    }

    /* ── Payments ── */
    if (activeSection === "payments") {
      const rows = data.rows || [];
      if (!rows.length) return <div style={{ padding: "10px 13px", fontSize: 12, color: "#999" }}>No payments found.</div>;
      const toStr = (r) => `${r.po_number} ${r.amount} ${dateSearchStr(r.date)}`;
      const q = sq("payments"), pg = sp("payments");
      const filtered = q.trim() ? rows.filter(r => fuzzyMatch(q, toStr(r))) : rows;
      const totalPgs = Math.max(1, Math.ceil(filtered.length / PGSZ));
      const safePg   = Math.min(pg, totalPgs);
      const paged    = filtered.slice((safePg-1)*PGSZ, safePg*PGSZ);
      return (
        <div>
          <SectionSearchBar q={q} setQ={v => setSQ("payments",v)} page={safePg} setPage={v => setSP("payments",v)}
            totalPgs={totalPgs} total={rows.length} filtered={filtered.length} />
          {data.total_paid > 0 && (
            <div style={{ padding: "4px 12px 6px", fontSize: 11.5, fontWeight: 700, color: "#16a34a" }}>
              Total Paid: {fmtINR(data.total_paid)}
            </div>
          )}
          <div style={{ overflowY: "auto" }}>
            {paged.length === 0
              ? <div style={{ padding:"10px 13px", fontSize:12, color:"#999" }}>No results for "{q}"</div>
              : paged.map((r, i) => (
                <div key={i} className={Q.ditem} style={{ cursor: "default" }}>
                  <span className={Q.did}>#{(safePg-1)*PGSZ+i+1}</span>
                  <span className={Q.dname} style={{ minWidth: 0 }}>
                    <span style={{ display: "block", fontFamily: "monospace", fontSize: 11 }}>{r.po_number}</span>
                    <span style={{ fontSize: 10, color: "#999", fontWeight: 400 }}>{r.date}</span>
                  </span>
                  <span style={{ fontSize: 11, fontWeight: 700, color: "#16a34a", flexShrink: 0 }}>{fmtINR(r.amount)}</span>
                </div>
              ))
            }
          </div>
        </div>
      );
    }

    return null;
  };

  return (
    <div className={Q.icard} style={{ marginTop: 10 }}>
      <div className={Q.ihdr}>
        <div className={Q.iico}>🏭</div>
        <div style={{ minWidth: 0, flex: 1 }}>
          <div className={Q.iname}>{s.name || ""}</div>
          <div className={Q.imeta}>Supplier #{s.id} · Code: {s.code || "—"}</div>
        </div>
      </div>

      {infoRows.length > 0 && (
        <div style={{ padding: "10px 13px", display: "flex", flexDirection: "column", gap: 5 }}>
          {infoRows.map((r) => (
            <div key={r.label} style={{ display: "flex", gap: 8, fontSize: 12.5 }}>
              <span style={{ color: "#999", minWidth: 52 }}>{r.label}</span>
              <span style={{ fontWeight: 600, color: "#333", wordBreak: "break-all" }}>{r.value}</span>
            </div>
          ))}
        </div>
      )}

      <div style={{ display: "flex", flexWrap: "wrap", gap: 6, padding: "8px 13px 10px" }}>
        {BTNS.map((b) => {
          const active = activeSection === b.key;
          return (
            <button
              key={b.key}
              onClick={() => handleBtn(b.key)}
              style={{
                display: "flex", alignItems: "center", gap: 4,
                padding: "5px 10px", borderRadius: 20,
                border: `1px solid ${active ? CFG.A : CFG.A + "44"}`,
                background: active ? CFG.A : `${CFG.A}0f`,
                color: active ? "#fff" : CFG.A,
                fontSize: 11.5, fontWeight: 600,
                cursor: "pointer", fontFamily: "inherit",
                transition: "all .12s",
              }}
            >
              {b.icon} {b.label}
            </button>
          );
        })}
      </div>

      {activeSection && (
        <div style={{ borderTop: "1px solid #e8e8f2" }}>
          {renderSection()}
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// PO CARD
// ─────────────────────────────────────────────────────────────────────────────
function POCard({ part }) {
  const statusColor = part.status === "Completed" ? "#16a34a" : part.status === "Cancelled" ? "#dc2626" : "#d97706";
  const poId = part.po_id || null;

  const [activeSection, setActiveSection] = useState(null);
  const [sectionData,   setSectionData]   = useState({});
  const [secLoading,    setSecLoading]    = useState(false);
  const [secSearch,     setSecSearch]     = useState({});
  const [secPage,       setSecPage]       = useState({});

  const sq    = (k) => secSearch[k] || "";
  const sp    = (k) => secPage[k]   || 1;
  const setSQ = (k,v) => { setSecSearch(p=>({...p,[k]:v})); setSecPage(p=>({...p,[k]:1})); };
  const setSP = (k,v) => setSecPage(p=>({...p,[k]:v}));

  const BTNS = [
    { key: "items",      icon: "📦", label: "Line Items"  },
    { key: "payments",   icon: "💳", label: "Payments"    },
    { key: "status-log", icon: "📋", label: "Status Log"  },
    { key: "supplier",   icon: "🏭", label: "Supplier"    },
  ];

  const handleBtn = async (key) => {
    if (activeSection === key) { setActiveSection(null); return; }
    setActiveSection(key);
    if (sectionData[key]) return;
    if (!poId) { setSectionData((p) => ({ ...p, [key]: { error: "PO ID not available" } })); return; }
    setSecLoading(true);
    try {
      const base = API_BASE.replace(/\/$/, "");
      const res  = await fetch(`${base}/po/${poId}/${key}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setSectionData((p) => ({ ...p, [key]: data }));
    } catch (e) {
      setSectionData((p) => ({ ...p, [key]: { error: e.message } }));
    } finally {
      setSecLoading(false);
    }
  };

  const renderSection = () => {
    const data = sectionData[activeSection];
    if (secLoading && !data)
      return <div style={{ padding: "10px 13px", fontSize: 12, color: "#999" }}>Loading…</div>;
    if (!data) return null;
    if (data.error)
      return <div style={{ padding: "10px 13px", fontSize: 12, color: "#dc2626" }}>{data.error}</div>;

    const mkPaged = (key, toStr) => {
      const q = sq(key), pg = sp(key);
      const rows2 = data.rows || [];
      const filtered = q.trim() ? rows2.filter(r => fuzzyMatch(q, toStr(r))) : rows2;
      const totalPgs = Math.max(1, Math.ceil(filtered.length / PGSZ));
      const safePg   = Math.min(pg, totalPgs);
      const paged    = filtered.slice((safePg-1)*PGSZ, safePg*PGSZ);
      return { q, pg:safePg, filtered, totalPgs, paged, setQ:v=>setSQ(key,v), setPage:v=>setSP(key,v), total:rows2.length };
    };

    /* ── Line Items ── */
    if (activeSection === "items") {
      const rows = data.rows || [];
      if (!rows.length) return <div style={{ padding: "10px 13px", fontSize: 12, color: "#999" }}>No items found.</div>;
      const { q, pg, filtered, totalPgs, paged, setQ, setPage, total } = mkPaged("items",
        r => `${r.name} ${r.unit} ${r.hsn} ${r.ordered} ${r.received} ${r.unit_price}`);
      return (
        <div>
          <SectionSearchBar q={q} setQ={setQ} page={pg} setPage={setPage} totalPgs={totalPgs} total={total} filtered={filtered.length} />
          <div style={{ overflowY:"auto" }}>
            {paged.length===0
              ? <div style={{ padding:"10px 13px", fontSize:12, color:"#999" }}>No results for "{q}"</div>
              : paged.map((r,i) => (
                <div key={i} className={Q.ditem} style={{ cursor:"default", alignItems:"flex-start" }}>
                  <span className={Q.did} style={{ marginTop:2 }}>#{(pg-1)*PGSZ+i+1}</span>
                  <span className={Q.dname} style={{ minWidth:0 }}>
                    <span style={{ display:"block" }}>{r.name}</span>
                    <span style={{ fontSize:10, color:"#999", fontWeight:400 }}>
                      Rcvd: {r.received}/{r.ordered} {r.unit} · Tax: {r.tax_percent}%
                    </span>
                  </span>
                  <span style={{ display:"flex", flexDirection:"column", alignItems:"flex-end", flexShrink:0, gap:1 }}>
                    <span style={{ fontSize:11, fontWeight:700, color:CFG.P }}>{fmtINR(r.line_total)}</span>
                    <span style={{ fontSize:9.5, color:"#aaa" }}>{fmtINR(r.unit_price)}/unit</span>
                  </span>
                </div>
              ))
            }
          </div>
        </div>
      );
    }

    /* ── Payments ── */
    if (activeSection === "payments") {
      const rows = data.rows || [];
      if (!rows.length) return <div style={{ padding: "10px 13px", fontSize: 12, color: "#999" }}>No payments found.</div>;
      const { q, pg, filtered, totalPgs, paged, setQ, setPage, total } = mkPaged("payments",
        r => `${r.amount} ${dateSearchStr(r.date)}`);
      return (
        <div>
          <SectionSearchBar q={q} setQ={setQ} page={pg} setPage={setPage} totalPgs={totalPgs} total={total} filtered={filtered.length} />
          {data.total_paid>0 && (
            <div style={{ padding:"4px 12px 6px", fontSize:11.5, fontWeight:700, color:"#16a34a" }}>
              Total Paid: {fmtINR(data.total_paid)}
            </div>
          )}
          <div style={{ overflowY:"auto" }}>
            {paged.length===0
              ? <div style={{ padding:"10px 13px", fontSize:12, color:"#999" }}>No results for "{q}"</div>
              : paged.map((r,i) => (
                <div key={i} className={Q.ditem} style={{ cursor:"default" }}>
                  <span className={Q.did}>#{(pg-1)*PGSZ+i+1}</span>
                  <span className={Q.dname}><span style={{ fontSize:10, color:"#999", fontWeight:400 }}>{r.date}</span></span>
                  <span style={{ fontSize:11, fontWeight:700, color:"#16a34a", flexShrink:0 }}>{fmtINR(r.amount)}</span>
                </div>
              ))
            }
          </div>
        </div>
      );
    }

    /* ── Status Log ── */
    if (activeSection === "status-log") {
      const rows = data.rows || [];
      if (!rows.length) return <div style={{ padding: "10px 13px", fontSize: 12, color: "#999" }}>No status history.</div>;
      const { q, pg, filtered, totalPgs, paged, setQ, setPage, total } = mkPaged("status-log",
        r => `${r.status} ${r.remarks} ${dateSearchStr(r.date)}`);
      return (
        <div>
          <SectionSearchBar q={q} setQ={setQ} page={pg} setPage={setPage} totalPgs={totalPgs} total={total} filtered={filtered.length} />
          <div style={{ overflowY:"auto" }}>
            {paged.length===0
              ? <div style={{ padding:"10px 13px", fontSize:12, color:"#999" }}>No results for "{q}"</div>
              : paged.map((r,i) => {
                const sc = r.status==="Completed"?"#16a34a":r.status==="Cancelled"?"#dc2626":"#d97706";
                return (
                  <div key={i} className={Q.ditem} style={{ cursor:"default" }}>
                    <span className={Q.did}>#{(pg-1)*PGSZ+i+1}</span>
                    <span className={Q.dname} style={{ minWidth:0 }}>
                      <span style={{ fontSize:10.5, fontWeight:700, color:sc }}>{r.status}</span>
                      {r.remarks && <span style={{ display:"block", fontSize:10, color:"#999", fontWeight:400 }}>{r.remarks}</span>}
                    </span>
                    <span style={{ fontSize:10, color:"#bbb", flexShrink:0 }}>{r.date}</span>
                  </div>
                );
              })
            }
          </div>
        </div>
      );
    }

    /* ── Supplier ── */
    if (activeSection === "supplier") {
      if (data.error) return <div style={{ padding:"10px 13px", fontSize:12, color:"#dc2626" }}>{data.error}</div>;
      if (!data.supplier) return <div style={{ padding:"10px 13px", fontSize:12, color:"#999" }}>Supplier not found.</div>;
      return <div style={{ padding:"0 0 4px" }}><SupplierCard result={{ supplier: data.supplier }} /></div>;
    }

    return null;
  };

  return (
    <div className={Q.icard} style={{ marginTop: 8 }}>
      <div className={Q.ihdr}>
        <div className={Q.iico} style={{ fontSize: 15 }}>📋</div>
        <div style={{ minWidth: 0, flex: 1 }}>
          <div className={Q.iname} style={{ fontFamily: "monospace" }}>{part.po_no || ""}</div>
          <div className={Q.imeta}>{part.supplier || ""} · {part.date}</div>
        </div>
        <span style={{ fontSize: 10, fontWeight: 700, color: statusColor, background: statusColor + "18", padding: "2px 7px", borderRadius: 6, flexShrink: 0 }}>
          {part.status}
        </span>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 8, padding: "10px 13px" }}>
        {[
          { label: "Total",   value: part.total   },
          { label: "Advance", value: part.advance  },
          { label: "Balance", value: part.balance  },
        ].map((r) => (
          <div key={r.label} style={{ textAlign: "center", padding: "7px 4px", borderRadius: 9, background: "#fff", border: "1px solid #e2e2ef" }}>
            <div style={{ fontSize: 13, fontWeight: 700, color: CFG.P }}>₹{Number(r.value || 0).toLocaleString("en-IN", { maximumFractionDigits: 0 })}</div>
            <div style={{ fontSize: 9.5, marginTop: 2, color: "#aaa" }}>{r.label}</div>
          </div>
        ))}
      </div>

      <div style={{ display: "flex", flexWrap: "wrap", gap: 6, padding: "0 13px 10px" }}>
        {BTNS.map((b) => {
          const active = activeSection === b.key;
          return (
            <button
              key={b.key}
              onClick={() => handleBtn(b.key)}
              style={{
                display: "flex", alignItems: "center", gap: 4,
                padding: "5px 10px", borderRadius: 20,
                border: `1px solid ${active ? CFG.A : CFG.A + "44"}`,
                background: active ? CFG.A : `${CFG.A}0f`,
                color: active ? "#fff" : CFG.A,
                fontSize: 11.5, fontWeight: 600,
                cursor: "pointer", fontFamily: "inherit",
                transition: "all .12s",
              }}
            >
              {b.icon} {b.label}
            </button>
          );
        })}
      </div>

      {activeSection && (
        <div style={{ borderTop: "1px solid #e8e8f2" }}>
          {renderSection()}
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// BOT BUBBLE  — streams text then reveals cards
// ─────────────────────────────────────────────────────────────────────────────
function BotBubble({ msg, onDropdownSelect, onConfirmPick, onDirectPick, onAction }) {
  // Chat-text parts are streamed first; structured cards reveal after.
  // Confirm cards are NOT considered chat — their `message` is shown inside
  // the ConfirmCard itself so users see the buttons.
  const parts       = msg.parts ?? [];
  const chatParts   = parts.filter((p) => p.type === "chat");
  const resultPart  = parts.find((p)   => p.type === "result");
  const poParts     = parts.filter((p) => p.type === "po");
  const dropPart    = parts.find((p)   => p.type === "dropdown");
  const confirmPart = parts.find((p)   => p.type === "confirm_resolution");
  const listParts   = parts.filter((p) =>
    ["po_list", "po_summary", "project_list", "supplier_list",
     "supplier_items", "supplier_payments"].includes(p.type)
  );
  const fullText    = chatParts.map((p) => p.message).join("\n\n");

  const [typedFor, setTypedFor] = useState("");
  const typingDone = fullText === "" || typedFor === fullText;
  const showCards  = !msg.loading && !msg.error && typingDone;


  return (
    <div className={Q.bwrap}>
      <div className={Q.bav}>🤖</div>
      <div className={Q.bbub}>

        {msg.loading && (
          <div className={Q.dots}>
            <div className={Q.dot} />
            <div className={Q.dot} />
            <div className={Q.dot} />
          </div>
        )}

        {!msg.loading && msg.error && (
          <div className={Q.errtxt}>⚠️ {msg.error}</div>
        )}

        {!msg.loading && !msg.error && (
          <>
            {fullText ? (
              <TypeWriter text={fullText} onDone={() => setTypedFor(fullText)} />
            ) : null}

            {showCards && resultPart && resultPart.supplier && <SupplierCard result={resultPart} onAction={onAction} />}
            {showCards && resultPart && resultPart.inventory && <InventoryCard result={resultPart} />}
            {showCards && poParts.map((p, i) => <POCard key={i} part={p} onAction={onAction} />)}
            {showCards && dropPart && (
              <DropdownCard dropdown={dropPart} onSelect={onDropdownSelect} />
            )}
            {showCards && confirmPart && (
              <ConfirmCard
                part={confirmPart}
                onPick={(reply) => onConfirmPick(msg.id, reply)}
                onDirectPick={(id, category, name) => onDirectPick(msg.id, id, category, name)}
              />
            )}
            {showCards && listParts.map((p, i) => <ListCard key={i} part={p} />)}

            {/* feedback shown only on substantive responses, not on confirms */}
            {showCards && !confirmPart && (
              <FeedbackButtons
                requestId={msg.requestId}
                query={msg.userQuery}
                summary={fullText.slice(0, 240)}
              />
            )}
          </>
        )}

        {!msg.loading && <div className={Q.ts}>{msg.time}</div>}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// ROLE SELECTOR
// ─────────────────────────────────────────────────────────────────────────────
function RoleSelector({ role, onChange }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);

  useEffect(() => {
    const handler = (e) => {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  return (
    <div ref={ref} style={{ position: "relative" }}>
      <button className={Q.rpill} onClick={() => setOpen((o) => !o)} title="Switch role">
        <span style={{ fontSize: 9, opacity: 0.65, letterSpacing: 0.5 }}>ROLE</span>
        <span style={{ fontSize: 11, fontWeight: 700 }}>{role}</span>
        <svg width="8" height="8" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </button>

      {open && (
        <div className={Q.rmenu}>
          {CFG.roles.map((r) => (
            <button
              key={r}
              className={`${Q.ropt}${r === role ? " " + Q.roptA : ""}`}
              onClick={() => { onChange(r); setOpen(false); }}
            >
              {r}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// QUICK SEARCH OVERLAY  — full-cover panel inside the chat area
// ─────────────────────────────────────────────────────────────────────────────
const QB = [
  { key: "supplier",  icon: "🏭", label: "Suppliers",      color: "#7c3aed" },
  { key: "po",        icon: "📋", label: "Purchase Orders", color: "#0369a1" },
  { key: "inventory", icon: "📦", label: "Inventory",       color: "#15803d" },
];

function QuickSearch({ type, onClose, onSelect }) {
  const [q,       setQ]       = useState("");
  const [rows,    setRows]    = useState([]);
  const [loading, setLoading] = useState(false);
  const [page,    setPage]    = useState(1);
  const inputRef = useRef(null);

  useEffect(() => { setQ(""); setRows([]); setLoading(true); fetch_rows(""); }, [type]);

  useEffect(() => {
    const t = setTimeout(() => fetch_rows(q), 260);
    return () => clearTimeout(t);
  }, [q]);

  useEffect(() => { setPage(1); }, [rows]);

  const fetch_rows = async (query) => {
    setLoading(true);
    try {
      const base = API_BASE.replace(/\/$/, "");
      const res  = await fetch(`${base}/quick-search/${type}?q=${encodeURIComponent(query)}&limit=80`);
      if (res.ok) setRows((await res.json()).rows || []);
    } catch {}
    setLoading(false);
    if (inputRef.current) inputRef.current.focus();
  };

  const cfg      = QB.find(b => b.key === type) || QB[0];
  const totalPgs = Math.max(1, Math.ceil(rows.length / PGSZ));
  const safePg   = Math.min(page, totalPgs);
  const paged    = rows.slice((safePg - 1) * PGSZ, safePg * PGSZ);

  const pgBtn = (dis) => ({
    border: "1px solid #e0e0ec", borderRadius: 5,
    background: dis ? "#f7f7fc" : "#fff", color: dis ? "#ccc" : "#555",
    cursor: dis ? "default" : "pointer",
    width: 22, height: 22, padding: 0,
    fontFamily: "inherit", fontSize: 13, lineHeight: "20px",
  });

  return (
    <div style={{
      position: "absolute", inset: 0, zIndex: 8,
      background: "#fff", display: "flex", flexDirection: "column",
      animation: "_ewSlideL .15s ease",
    }}>
      {/* Header — single search bar */}
      <div style={{
        display: "flex", alignItems: "center", gap: 8,
        padding: "10px 12px 8px",
        borderBottom: `2px solid ${cfg.color}22`,
        background: `${cfg.color}08`,
      }}>
        <span style={{ fontSize: 16 }}>{cfg.icon}</span>
        <input
          ref={inputRef}
          autoFocus
          type="text" value={q}
          onChange={e => { setQ(e.target.value); setPage(1); }}
          placeholder={`Search ${cfg.label.toLowerCase()}…`}
          style={{
            flex: 1, border: "none", background: "transparent",
            fontSize: 13, outline: "none", fontFamily: "inherit", fontWeight: 500,
            color: "#222",
          }}
        />
        {q && (
          <button onClick={() => { setQ(""); setPage(1); }}
            style={{ border:"none", background:"none", cursor:"pointer", color:"#bbb", fontSize:14, padding:0 }}>✕</button>
        )}
        <button onClick={onClose}
          style={{
            border: "none", background: cfg.color, color: "#fff",
            borderRadius: 8, cursor: "pointer", padding: "3px 8px",
            fontSize: 11, fontWeight: 600, fontFamily: "inherit",
          }}>Close</button>
      </div>

      {/* Pagination bar */}
      {rows.length > PGSZ && (
        <div style={{ padding:"3px 8px 3px", borderBottom:"1px solid #f0f0f8", display:"flex", alignItems:"center", justifyContent:"flex-end", gap:4, fontSize:10.5, color:"#999", flexShrink:0 }}>
          <span>{rows.length} results</span>
          <button onClick={() => setPage(p => Math.max(1,p-1))} disabled={safePg<=1} style={pgBtn(safePg<=1)}>‹</button>
          <span style={{ minWidth:28, textAlign:"center" }}>{safePg}/{totalPgs}</span>
          <button onClick={() => setPage(p => Math.min(totalPgs,p+1))} disabled={safePg>=totalPgs} style={pgBtn(safePg>=totalPgs)}>›</button>
        </div>
      )}

      {/* Results */}
      <div style={{ flex: 1, overflowY: "auto" }}>
        {loading && (
          <div style={{ padding: "16px 14px", fontSize: 12, color: "#aaa" }}>Searching…</div>
        )}
        {!loading && paged.length === 0 && (
          <div style={{ padding: "16px 14px", fontSize: 12, color: "#aaa" }}>
            {q ? `No results for "${q}"` : "Nothing found."}
          </div>
        )}
        {!loading && paged.map((r, i) => {
          if (type === "supplier") return (
            <button key={r.id} onClick={() => onSelect("supplier", r.id, r.name)}
              style={{ width:"100%", display:"flex", alignItems:"center", gap:10,
                       padding:"9px 13px", border:"none", borderBottom:"1px solid #f4f4f8",
                       background:"#fff", cursor:"pointer", textAlign:"left", fontFamily:"inherit" }}
              onMouseEnter={e => e.currentTarget.style.background="#f9f9fd"}
              onMouseLeave={e => e.currentTarget.style.background="#fff"}>
              <span style={{ width:26, height:26, borderRadius:8, background:cfg.color+"18",
                             display:"flex", alignItems:"center", justifyContent:"center",
                             fontSize:13, flexShrink:0 }}>🏭</span>
              <span style={{ flex:1, minWidth:0 }}>
                <span style={{ display:"block", fontSize:12.5, fontWeight:600, color:"#222",
                               whiteSpace:"nowrap", overflow:"hidden", textOverflow:"ellipsis" }}>{r.name}</span>
                <span style={{ fontSize:10.5, color:"#999" }}>{r.city||"—"}{r.mobile?` · ${r.mobile}`:""}</span>
              </span>
              <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="#ccc" strokeWidth="2.5" strokeLinecap="round">
                <polyline points="9 18 15 12 9 6" />
              </svg>
            </button>
          );
          if (type === "po") {
            const sc = r.status==="Completed"?"#16a34a":r.status==="Cancelled"?"#dc2626":"#d97706";
            return (
              <button key={r.id} onClick={() => onSelect("po", r.id, r.po_number)}
                style={{ width:"100%", display:"flex", alignItems:"center", gap:10,
                         padding:"9px 13px", border:"none", borderBottom:"1px solid #f4f4f8",
                         background:"#fff", cursor:"pointer", textAlign:"left", fontFamily:"inherit" }}
                onMouseEnter={e => e.currentTarget.style.background="#f9f9fd"}
                onMouseLeave={e => e.currentTarget.style.background="#fff"}>
                <span style={{ flex:1, minWidth:0 }}>
                  <span style={{ display:"block", fontSize:11.5, fontFamily:"monospace", fontWeight:600, color:"#333" }}>{r.po_number}</span>
                  <span style={{ fontSize:10.5, color:"#999" }}>{r.supplier} · {r.date}</span>
                </span>
                <span style={{ flexShrink:0, textAlign:"right" }}>
                  <span style={{ display:"block", fontSize:11, fontWeight:700, color:"#444" }}>{fmtINR(r.total)}</span>
                  <span style={{ fontSize:9, fontWeight:700, color:sc, background:sc+"18",
                                 padding:"1px 5px", borderRadius:4 }}>{r.status}</span>
                </span>
              </button>
            );
          }
          // inventory
          const inStock = r.stock > 0;
          return (
            <button key={r.id} onClick={() => onSelect("inventory", r.id, r.name)}
              style={{ width:"100%", display:"flex", alignItems:"center", gap:10,
                       padding:"9px 13px", border:"none", borderBottom:"1px solid #f4f4f8",
                       background:"#fff", cursor:"pointer", textAlign:"left", fontFamily:"inherit" }}
              onMouseEnter={e => e.currentTarget.style.background="#f9f9fd"}
              onMouseLeave={e => e.currentTarget.style.background="#fff"}>
              <span style={{ width:26, height:26, borderRadius:8, background:cfg.color+"18",
                             display:"flex", alignItems:"center", justifyContent:"center",
                             fontSize:13, flexShrink:0 }}>📦</span>
              <span style={{ flex:1, minWidth:0 }}>
                <span style={{ display:"block", fontSize:12, fontWeight:600, color:"#222",
                               whiteSpace:"nowrap", overflow:"hidden", textOverflow:"ellipsis" }}>{r.name}</span>
                <span style={{ fontSize:10.5, color:"#999" }}>{r.unit||"—"}</span>
              </span>
              <span style={{ fontSize:11, fontWeight:700, color:inStock?"#16a34a":"#dc2626",
                             flexShrink:0 }}>{r.stock} {r.unit}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN WIDGET
// ─────────────────────────────────────────────────────────────────────────────
const _mkTab = (id, label) => ({ id, label: label || `Chat ${id}`, messages: [], history: [], quickSearch: null });

export default function ChatWidget() {
  const [isOpen,   setIsOpen]  = useState(false);
  const [query,    setQuery]   = useState("");
  const [loading,  setLoading] = useState(false);
  const [role,     setRole]    = useState(CFG.defaultRole);

  const tabIdCounter   = useRef(1);
  const [tabs, setTabs]          = useState(() => [_mkTab(1, "Chat 1")]);
  const [activeTabId, setActiveTabId] = useState(1);
  const activeTabIdRef = useRef(1);
  const tabsRef        = useRef(null);

  const roleRef  = useRef(CFG.defaultRole);
  const chatEnd  = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => { tabsRef.current = tabs; }, [tabs]);
  useEffect(() => { activeTabIdRef.current = activeTabId; }, [activeTabId]);

  const activeTab  = tabs.find(t => t.id === activeTabId) || tabs[0];
  const messages   = activeTab?.messages   || [];
  const quickSearch= activeTab?.quickSearch || null;

  const setQuickSearch = useCallback((qs) => {
    const tid = activeTabIdRef.current;
    setTabs(prev => prev.map(t => t.id === tid ? { ...t, quickSearch: qs } : t));
  }, []);

  const addTab = useCallback(() => {
    const id = ++tabIdCounter.current;
    setTabs(prev => [...prev, _mkTab(id)]);
    setActiveTabId(id);
    activeTabIdRef.current = id;
    setQuery("");
  }, []);

  const closeTab = useCallback((id) => {
    setTabs(prev => {
      const idx  = prev.findIndex(t => t.id === id);
      const next = prev.filter(t => t.id !== id);
      if (!next.length) {
        const newId = ++tabIdCounter.current;
        activeTabIdRef.current = newId;
        setActiveTabId(newId);
        return [_mkTab(newId, "Chat 1")];
      }
      if (activeTabIdRef.current === id) {
        const newActive = (next[idx] || next[idx - 1] || next[0]).id;
        activeTabIdRef.current = newActive;
        setActiveTabId(newActive);
      }
      return next;
    });
  }, []);

  const switchTab = useCallback((id) => {
    setActiveTabId(id);
    activeTabIdRef.current = id;
    setQuery("");
    setTimeout(() => inputRef.current?.focus(), 100);
  }, []);

  useEffect(() => {
    chatEnd.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const pushWelcome = useCallback(() => {
    const tid = activeTabIdRef.current;
    setTabs(prev => prev.map(t => {
      if (t.id !== tid || t.messages.length > 0) return t;
      return { ...t, messages: [{
        id: "welcome", role: "bot", loading: false, error: null, time: ts(),
        parts: [{ type: "chat", message:
          "👋 Namaste! Main aapka ERP assistant hoon.\n\n" +
          "Aap mujhse Hindi ya English mein kuch bhi pooch sakte hain:\n\n" +
          "• **Suppliers** — DCL ka mobile batao\n" +
          "• **Inventory** — bearing 6205 ka stock\n" +
          "• **Orders** — MHEL/PO/00020 dikhao\n\n" +
          "Role switch karne ke liye upar wala button dabao. 🚀",
        }],
      }]};
    }));
  }, []);

  useEffect(() => {
    if (isOpen && messages.length === 0) pushWelcome();
    if (isOpen) setTimeout(() => inputRef.current?.focus(), 120);
  }, [isOpen, activeTabId]);

  useEffect(() => {
    chatEnd.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendQuery = useCallback(async (q) => {
    const trimmed = (typeof q === "string" ? q : query).trim();
    if (!trimmed || loading) return;

    const tabId = activeTabIdRef.current;
    const curTab = tabsRef.current?.find(t => t.id === tabId);
    const curHistory = curTab?.history || [];

    const uid = `u_${Date.now()}`;
    const bid = `b_${Date.now() + 1}`;

    setTabs(prev => prev.map(t => {
      if (t.id !== tabId) return t;
      const isFirst = !t.messages.some(m => m.role === "user");
      return {
        ...t,
        label: isFirst ? trimmed.slice(0, 20) + (trimmed.length > 20 ? "…" : "") : t.label,
        messages: [
          ...t.messages,
          { id: uid, role: "user",  content: trimmed, time: ts() },
          { id: bid, role: "bot",   loading: true, parts: [], error: null, time: ts() },
        ],
      };
    }));
    setQuery("");
    setLoading(true);

    try {
      const data          = await askBot(trimmed, curHistory, roleRef.current);
      const results       = data.results ?? [];
      const reqId         = data.request_id;
      const pending       = data.pending_resolution || null;
      const contextEntity = data.context_entity || null;

      const botText = results.map((r) => {
        if (r.type === "chat" || r.type === "confirm_resolution" || r.type === "result") return r.message;
        if (r.type === "po") return `PO: ${r.po_no}, Supplier: ${r.supplier}, Total: ${r.total}, Balance: ${r.balance}, Status: ${r.status}`;
        if (r.type === "po_list" || r.type === "po_summary")
          return r.rows?.map(row => `PO: ${row.po_number||row.po||""}, Supplier: ${row.supplier_name||""}, Amount: ${row.total_amount??row.total??""}, Status: ${row.status||""}`).join("\n");
        if (r.type === "supplier_list") return r.rows?.map(row => `Supplier: ${row.name||row.supplier_name||""}`).join("\n");
        if (r.type === "project_list")  return r.rows?.map(row => `Project: ${row.name||row.grp||""}`).join("\n");
        if (r.type === "dropdown")      return r.items?.map(item => `Option: ${item.name}`).join(", ");
        return null;
      }).filter(Boolean).join("\n");

      const assistantTurn = { role: "assistant", content: botText, pending_resolution: pending };
      if (contextEntity) assistantTurn.context_entity = contextEntity;
      const newHistory = [...curHistory, { role: "user", content: trimmed }, assistantTurn];

      setTabs(prev => prev.map(t => {
        if (t.id !== tabId) return t;
        return {
          ...t,
          history: newHistory,
          messages: t.messages.map(m =>
            m.id === bid ? { ...m, loading: false, parts: results, requestId: reqId, userQuery: trimmed } : m
          ),
        };
      }));
    } catch (e) {
      setTabs(prev => prev.map(t => {
        if (t.id !== tabId) return t;
        return { ...t, messages: t.messages.map(m => m.id === bid ? { ...m, loading: false, error: e.message || "Something went wrong." } : m) };
      }));
    } finally {
      setLoading(false);
    }
  }, [loading, query]);

  const handleSubmit = useCallback(() => sendQuery(query), [query, sendQuery]);

  const handleConfirmPick = useCallback((bubbleId, reply) => {
    sendQuery(reply);
  }, [sendQuery]);

  const handleDirectPick = useCallback(async (bubbleId, entityId, category, name) => {
    const tabId = activeTabIdRef.current;
    const curTab = tabsRef.current?.find(t => t.id === tabId);
    const curHistory = curTab?.history || [];
    const userTurn = { role: "user", content: name };
    const bid = `bot_${Date.now()}`;

    setTabs(prev => prev.map(t => {
      if (t.id !== tabId) return t;
      return { ...t, messages: [...t.messages,
        { id: `user_${Date.now()}`, role: "user", content: name, time: ts() },
        { id: bid, role: "bot", loading: true, parts: [], time: ts() },
      ]};
    }));
    setLoading(true);
    try {
      const base = API_BASE.replace(/\/$/, "");
      const ep = category === "inventory" ? `${base}/inventory/${entityId}/card` : `${base}/supplier/${entityId}/card`;
      const res = await fetch(ep);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const results = data.results || [];
      const botText = results.filter(r => r.type === "chat").map(r => r.message).join("\n");
      setTabs(prev => prev.map(t => {
        if (t.id !== tabId) return t;
        return {
          ...t,
          history: [...curHistory, userTurn, { role: "assistant", content: botText }],
          messages: t.messages.map(m => m.id === bid ? { ...m, loading: false, parts: results } : m),
        };
      }));
    } catch (e) {
      setTabs(prev => prev.map(t => {
        if (t.id !== tabId) return t;
        return { ...t, messages: t.messages.map(m => m.id === bid ? { ...m, loading: false, error: e.message } : m) };
      }));
    } finally {
      setLoading(false);
    }
  }, []);

  const handleQuickSelect = useCallback(async (category, entityId, label) => {
    const tabId = activeTabIdRef.current;
    const curTab = tabsRef.current?.find(t => t.id === tabId);
    const curHistory = curTab?.history || [];
    setTabs(prev => prev.map(t => t.id === tabId ? { ...t, quickSearch: null } : t));
    const bid = `bot_${Date.now()}`;
    setTabs(prev => prev.map(t => {
      if (t.id !== tabId) return t;
      return { ...t, messages: [...t.messages,
        { id: `user_${Date.now()}`, role: "user", content: label, time: ts() },
        { id: bid, role: "bot", loading: true, parts: [], time: ts() },
      ]};
    }));
    setLoading(true);
    try {
      const base = API_BASE.replace(/\/$/, "");
      const ep = category === "inventory" ? `${base}/inventory/${entityId}/card`
        : category === "po" ? `${base}/po/${entityId}/card`
        : `${base}/supplier/${entityId}/card`;
      const res = await fetch(ep);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const results = data.results || [];
      const botText = results.filter(r => r.type === "chat" || r.type === "po")
        .map(r => r.message || r.po_no || "").filter(Boolean).join("\n");
      setTabs(prev => prev.map(t => {
        if (t.id !== tabId) return t;
        return {
          ...t,
          history: [...curHistory, { role: "user", content: label }, { role: "assistant", content: botText }],
          messages: t.messages.map(m => m.id === bid ? { ...m, loading: false, parts: results } : m),
        };
      }));
    } catch (e) {
      setTabs(prev => prev.map(t => {
        if (t.id !== tabId) return t;
        return { ...t, messages: t.messages.map(m => m.id === bid ? { ...m, loading: false, error: e.message } : m) };
      }));
    } finally {
      setLoading(false);
    }
  }, []);

  const handleRoleChange = useCallback((r) => {
    setRole(r);
    roleRef.current = r;
    const tid = activeTabIdRef.current;
    setTabs(prev => prev.map(t => t.id === tid ? { ...t, history: [] } : t));
  }, []);

  const handleClear = useCallback(() => {
    const tid = activeTabIdRef.current;
    setTabs(prev => prev.map(t => {
      if (t.id !== tid) return t;
      return { ...t, history: [], messages: [{
        id: `reset_${Date.now()}`, role: "bot", loading: false, error: null, time: ts(),
        parts: [{ type: "chat", message: "🧹 Chat clear ho gaya! Ab kya dhundhna hai?" }],
      }]};
    }));
  }, []);


  return (
    <>
      {/* FAB */}
      {!isOpen && (
        <button className={Q.fab} onClick={() => setIsOpen(true)} title={CFG.title}>
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" />
          </svg>
        </button>
      )}

      {/* Mobile backdrop */}
      {isOpen && <div className={Q.overlay} onClick={() => setIsOpen(false)} />}

      {/* Panel */}
      {isOpen && (
        <div className={Q.panel}>

          {/* Header */}
          <div className={Q.hdr}>
            <div className={Q.hrow}>
              <div style={{ display: "flex", alignItems: "center", gap: 10, minWidth: 0, flex: 1, overflow: "hidden" }}>
                <div className={Q.hico}>🤖</div>
                <div style={{ minWidth: 0, overflow: "hidden" }}>
                  <div className={Q.htit}>{CFG.title}</div>
                  <div className={Q.hsub}>{CFG.subtitle}</div>
                </div>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 5, flexShrink: 0 }}>
                <RoleSelector role={role} onChange={handleRoleChange} />
                <button className={Q.hbtn} onClick={handleClear} title="Clear chat">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                    <path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6" />
                  </svg>
                </button>
                <button className={Q.hbtn} onClick={() => setIsOpen(false)} title="Close">
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                    <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
                  </svg>
                </button>
              </div>
            </div>

          </div>

          {/* Tab bar — Chrome style */}
          <div className={Q.tabbar}>
            {tabs.map(tab => (
              <div
                key={tab.id}
                className={`${Q.tab}${tab.id === activeTabId ? " "+Q.tabA : ""}`}
                onClick={() => switchTab(tab.id)}
                title={tab.label}
                role="tab"
                tabIndex={0}
                onKeyDown={e => e.key === "Enter" && switchTab(tab.id)}
              >
                <span style={{ flex:1, overflow:"hidden", textOverflow:"ellipsis", minWidth:0, display:"block" }}>
                  {tab.label}
                </span>
                {tabs.length > 1 && (
                  <button
                    className={Q.tabX}
                    onClick={e => { e.stopPropagation(); closeTab(tab.id); }}
                    title="Close tab"
                  >×</button>
                )}
              </div>
            ))}
            <button className={Q.tabP} onClick={addTab} title="New tab">+</button>
          </div>

          {/* Chat area + quick search overlay */}
          <div className={Q.chatWrap}>

            {/* ── Floating quick-access buttons ── */}
            <div style={{
              position: "absolute", left: 0, top: 14, zIndex: 9,
              display: "flex", flexDirection: "column", gap: 3,
            }}>
              {QB.map((b) => {
                const active = quickSearch === b.key;
                return (
                  <button
                    key={b.key}
                    title={b.label}
                    onClick={() => setQuickSearch(quickSearch === b.key ? null : b.key)}
                    style={{
                      width: 30, height: 30,
                      border: `1px solid ${active ? b.color : "#e0e0ec"}`,
                      borderLeft: "none",
                      borderRadius: "0 8px 8px 0",
                      background: active ? b.color : "#fff",
                      color: active ? "#fff" : b.color,
                      cursor: "pointer", fontSize: 14,
                      display: "flex", alignItems: "center", justifyContent: "center",
                      boxShadow: active ? `2px 2px 8px ${b.color}44` : "1px 1px 4px rgba(0,0,0,.07)",
                      transition: "all .13s",
                    }}
                  >{b.icon}</button>
                );
              })}
            </div>

            {/* Messages */}
            <div className={Q.chat}>
              {messages.map((msg) =>
                msg.role === "user"
                  ? <UserBubble key={msg.id} msg={msg} />
                  : <BotBubble
                      key={msg.id}
                      msg={msg}
                      onDropdownSelect={sendQuery}
                      onConfirmPick={handleConfirmPick}
                      onDirectPick={handleDirectPick}
                      onAction={sendQuery}
                    />
              )}
              <div ref={chatEnd} />
            </div>

            {/* Quick search overlay */}
            {quickSearch && (
              <QuickSearch
                type={quickSearch}
                onClose={() => setQuickSearch(null)}
                onSelect={handleQuickSelect}
              />
            )}

          </div>

          {/* Input */}
          <div className={Q.ibar}>
            <div className={Q.iinn}>
              <svg style={{ flexShrink: 0 }} width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#bbb" strokeWidth="2" strokeLinecap="round">
                <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" />
              </svg>
              <input
                ref={inputRef}
                className={Q.iinput}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleSubmit()}
                placeholder="Kuch bhi poochein…"
              />
              {query && (
                <button className={Q.clr} onClick={() => setQuery("")}>×</button>
              )}
            </div>
            <button
              className={Q.send}
              onClick={handleSubmit}
              disabled={loading || !query.trim()}
            >
              {loading
                ? <span className={Q.spin} />
                : (
                  <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="22" y1="2" x2="11" y2="13" />
                    <polygon points="22 2 15 22 11 13 2 9 22 2" />
                  </svg>
                )}
            </button>
          </div>

        </div>
      )}
    </>
  );
}
