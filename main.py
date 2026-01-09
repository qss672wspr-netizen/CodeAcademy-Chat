import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

# ------------------------------------------------------------
# STEP 2: Add nickname gate + /check_nick (no DB, no rooms yet)
# Goal: prove that "nick taken" + query param WS works reliably.
# ------------------------------------------------------------

app = FastAPI()
log = logging.getLogger("chat")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

NICK_RE = re.compile(r"^[A-Za-z0-9ĄČĘĖĮŠŲŪŽąčęėįšųūž_\-\. ]{2,24}$")

state_lock = asyncio.Lock()
online_nicks_cf: Set[str] = set()

def ts() -> str:
    return datetime.utcnow().strftime("%H:%M:%S")

def valid_nick(n: str) -> bool:
    return bool(NICK_RE.match((n or "").strip()))

HTML = r"""<!doctype html>
<html lang="lt">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Step2 Nick Gate</title>
  <style>
    body { font-family: ui-monospace, Menlo, Consolas, monospace; margin: 0; background:#0b0f14; color:#d7e3f4; }
    .wrap{ max-width: 980px; margin: 18px auto; padding: 0 14px; }
    .card{ border: 1px solid rgba(255,255,255,.12); border-radius: 14px; background: rgba(255,255,255,.04); overflow:hidden;}
    .top{ padding: 12px 14px; border-bottom:1px solid rgba(255,255,255,.10); display:flex; gap:12px; align-items:center; justify-content:space-between;}
    .pill{ display:inline-flex; gap:8px; align-items:center; padding:6px 10px; border-radius:999px; border:1px solid rgba(255,255,255,.12); background:rgba(0,0,0,.2);}
    .dot{ width:10px; height:10px; border-radius:50%; background:#ff5c5c;}
    .dot.ok{ background:#21d07a;}
    #log{ height: 52vh; overflow:auto; padding: 12px 14px; white-space: pre-wrap; line-height: 1.4; }
    .bar{ display:flex; gap:10px; padding: 12px 14px; border-top:1px solid rgba(255,255,255,.10);}
    input{ flex:1; padding:12px 12px; border-radius: 12px; border:1px solid rgba(255,255,255,.14);
           background: rgba(0,0,0,.25); color:#d7e3f4; outline:none;}
    button{ padding: 12px 14px; border-radius: 12px; border:1px solid rgba(255,255,255,.14);
            background:#7cff6b; color:#061015; font-weight:900; cursor:pointer;}
    button:disabled{ opacity:.6; cursor:not-allowed;}
    .small{ color: rgba(215,227,244,.6); font-size: 12px;}
    .err{ color:#ff7a7a; }
    .ok{ color:#21d07a; }
    .row{ display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
    .nickbox{ display:flex; gap:10px; align-items:center; }
    #nick{ width: 280px; flex: unset; }
  </style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <div class="top">
      <div class="row">
        <b>Step 2</b>
        <span class="small">– nick gate + /check_nick</span>
      </div>
      <div class="pill"><span id="dot" class="dot"></span><span id="st">disconnected</span></div>
    </div>

    <div class="top" style="gap:14px;">
      <div class="nickbox">
        <span class="small">Nick:</span>
        <input id="nick" placeholder="pvz. Tomas" maxlength="24"/>
        <button id="join" disabled>Join</button>
      </div>
      <div id="nickStatus" class="small"></div>
    </div>

    <div id="log"></div>

    <div class="bar">
      <input id="msg" placeholder="type message and press Enter" maxlength="200" disabled/>
      <button id="btn" disabled>Send</button>
    </div>

    <div class="top" style="border-top:1px solid rgba(255,255,255,.10); border-bottom:0;">
      <div class="small">
        Health: <a href="/health" target="_blank">/health</a>
      </div>
      <div class="small" id="diag"></div>
    </div>
  </div>
</div>

<script>
  const logEl = document.getElementById("log");
  const msgEl = document.getElementById("msg");
  const btn = document.getElementById("btn");
  const dot = document.getElementById("dot");
  const st = document.getElementById("st");
  const diag = document.getElementById("diag");

  const nickEl = document.getElementById("nick");
  const joinBtn = document.getElementById("join");
  const nickStatus = document.getElementById("nickStatus");

  let ws = null;
  let reconnectTimer = null;
  let reconnectDelay = 800;
  const maxDelay = 8000;
  let connecting = false;

  let nick = "";
  let joinedOnce = false;

  function line(s){
    logEl.textContent += s + "\\n";
    logEl.scrollTop = logEl.scrollHeight;
  }

  function setConn(ok){
    if(ok){ dot.classList.add("ok"); st.textContent="connected"; }
    else { dot.classList.remove("ok"); st.textContent="disconnected"; }
  }

  function validateNick(n){
    return /^[A-Za-z0-9ĄČĘĖĮŠŲŪŽąčęėįšųūž_\\-\\. ]{2,24}$/.test((n||"").trim());
  }

  async function checkNickNow(){
    const n = (nickEl.value||"").trim();
    joinBtn.disabled = true;

    if(!validateNick(n)){
      nickStatus.innerHTML = '<span class="err">Netinkamas nick (2–24, raidės/skaičiai/tarpas/_-.)</span>';
      return;
    }

    nickStatus.textContent = "Tikrinama...";
    try{
      const qs = new URLSearchParams({nick:n}).toString();
      const r = await fetch(`/check_nick?${qs}`, {cache:"no-store"});
      const j = await r.json();
      if(j.ok){
        nickStatus.innerHTML = '<span class="ok">Nick laisvas</span>';
        joinBtn.disabled = false;
      }else{
        nickStatus.innerHTML = '<span class="err">' + (j.reason || 'Nick užimtas') + '</span>';
      }
    }catch{
      nickStatus.innerHTML = '<span class="err">Nepavyko patikrinti nick</span>';
    }
  }

  let checkTimer=null;
  function scheduleNickCheck(){
    if(checkTimer) clearTimeout(checkTimer);
    checkTimer = setTimeout(checkNickNow, 250);
  }

  nickEl.addEventListener("input", scheduleNickCheck);
  nickEl.addEventListener("keydown", (e)=>{ if(e.key==="Enter" && !joinBtn.disabled) joinBtn.click(); });

  function wsUrl(){
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const qs = new URLSearchParams({nick}).toString();
    return `${proto}://${location.host}/ws?${qs}`;
  }

  function scheduleReconnect(){
    if(reconnectTimer) return;
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      connect(true);
      reconnectDelay = Math.min(maxDelay, Math.floor(reconnectDelay * 1.5));
    }, reconnectDelay);
  }

  function stopReconnect(){
    if(reconnectTimer){ clearTimeout(reconnectTimer); reconnectTimer = null; }
    reconnectDelay = 800;
  }

  function connect(isReconnect=false){
    if(connecting) return;
    connecting = true;

    if(!isReconnect){
      line("=== joining as " + nick + " ===");
    }else{
      line(`=== reconnect in ${reconnectDelay}ms... ===`);
    }

    ws = new WebSocket(wsUrl());

    ws.onopen = () => {
      connecting = false;
      stopReconnect();
      setConn(true);
      msgEl.disabled = false;
      btn.disabled = false;
      line("[client] ws open");
      diag.textContent = "keepalive active";
      joinedOnce = true;
    };

    ws.onmessage = (ev) => {
      let o=null;
      try{ o = JSON.parse(ev.data); }catch{ return; }
      if(o.type === "ping"){
        ws.send(JSON.stringify({type:"pong", t: Date.now()}));
        return;
      }
      if(o.type === "error"){
        line("[server error] " + (o.text || "join failed"));
        // fatal: do not reconnect automatically; user must pick new nick
        try{ ws.close(); }catch{}
        msgEl.disabled = true;
        btn.disabled = true;
        return;
      }
      line(`[server ${o.t || ""}] ${o.text || ""}`);
    };

    ws.onclose = (ev) => {
      connecting = false;
      setConn(false);
      msgEl.disabled = true;
      btn.disabled = true;
      line(`[client] ws closed code=${ev.code} reason=${ev.reason || ""}`);

      // If we never managed to join properly, do not loop reconnect forever.
      if(!joinedOnce){
        line("[client] join not established; please re-check nick");
        return;
      }
      scheduleReconnect();
    };

    ws.onerror = () => {
      line("[client] ws error");
    };
  }

  function send(){
    const text = (msgEl.value||"").trim();
    if(!text) return;
    if(!ws || ws.readyState !== WebSocket.OPEN){
      line("[client] not connected");
      return;
    }
    ws.send(JSON.stringify({type:"say", text}));
    msgEl.value="";
  }

  btn.onclick = send;
  msgEl.addEventListener("keydown", (e)=>{ if(e.key==="Enter") send(); });

  joinBtn.onclick = async () => {
    await checkNickNow();
    if(joinBtn.disabled) return;
    nick = (nickEl.value||"").trim();
    localStorage.setItem("nick", nick);
    connect(false);
  };

  (function init(){
    const saved = (localStorage.getItem("nick")||"").trim();
    if(saved){ nickEl.value = saved; }
    scheduleNickCheck();
    msgEl.disabled = true;
    btn.disabled = true;
  })();
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML

@app.get("/health", response_class=PlainTextResponse)
async def health():
    return f"OK {ts()}"

@app.get("/check_nick")
async def check_nick(nick: str = ""):
    n = (nick or "").strip()
    if not valid_nick(n):
        return JSONResponse({"ok": False, "code": "BAD_NICK", "reason": "Netinkamas nick (2–24)."})
    async with state_lock:
        if n.casefold() in online_nicks_cf:
            return JSONResponse({"ok": False, "code": "NICK_TAKEN", "reason": "Nick užimtas."})
    return JSONResponse({"ok": True})

async def heartbeat(ws: WebSocket):
    try:
        while True:
            await asyncio.sleep(25)
            await ws.send_text(json.dumps({"type":"ping","t":ts()}))
    except Exception:
        return

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    nick = (ws.query_params.get("nick") or "").strip()

    await ws.accept()
    peer = f"{ws.client.host if ws.client else 'unknown'}"
    log.info("WS accepted from %s nick=%r", peer, nick)

    if not valid_nick(nick):
        await ws.send_text(json.dumps({"type":"error","code":"BAD_NICK","text":"Bad nick"}))
        await ws.close()
        return

    # Reserve nick
    async with state_lock:
        cf = nick.casefold()
        if cf in online_nicks_cf:
            await ws.send_text(json.dumps({"type":"error","code":"NICK_TAKEN","text":"Nick taken"}))
            await ws.close()
            return
        online_nicks_cf.add(cf)

    hb_task: Optional[asyncio.Task] = asyncio.create_task(heartbeat(ws))

    try:
        await ws.send_text(json.dumps({"type":"msg","t":ts(),"text":f"welcome {nick}"}))

        while True:
            msg = await ws.receive()
            mtype = msg.get("type")

            if mtype == "websocket.disconnect":
                raise WebSocketDisconnect

            if mtype != "websocket.receive":
                continue

            raw = msg.get("text") or msg.get("bytes")
            if raw is None:
                continue

            if isinstance(raw, (bytes, bytearray)):
                try:
                    raw = raw.decode("utf-8", errors="replace")
                except Exception:
                    raw = ""

            try:
                data = json.loads(raw)
            except Exception:
                await ws.send_text(json.dumps({"type":"msg","t":ts(),"text":"[server] bad json"}))
                continue

            if not isinstance(data, dict):
                continue

            if data.get("type") == "pong":
                continue

            if data.get("type") == "say":
                text = str(data.get("text",""))[:200]
                await ws.send_text(json.dumps({"type":"msg","t":ts(),"text":f"{nick}: {text}"}))
                continue

    except WebSocketDisconnect:
        log.info("WS disconnect %s nick=%r", peer, nick)
    except Exception:
        log.exception("WS error from %s nick=%r", peer, nick)
    finally:
        if hb_task:
            hb_task.cancel()
        async with state_lock:
            online_nicks_cf.discard(nick.casefold())
        try:
            await ws.close()
        except Exception:
            pass
        log.info("WS closed %s nick=%r", peer, nick)
