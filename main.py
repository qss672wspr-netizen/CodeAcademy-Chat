import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Dict, Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

# ------------------------------------------------------------
# STEP 3: One shared room (#main) with broadcast + online list
# Keeps Step 2 nick gate + /check_nick
# Goal: when two browsers join, they SEE each other's messages.
# ------------------------------------------------------------

app = FastAPI()
log = logging.getLogger("chat")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

NICK_RE = re.compile(r"^[A-Za-z0-9ĄČĘĖĮŠŲŪŽąčęėįšųūž_\-\. ]{2,24}$")

state_lock = asyncio.Lock()
clients: Set[WebSocket] = set()
nick_by_ws: Dict[WebSocket, str] = {}
# For /check_nick + uniqueness
online_nicks_cf: Set[str] = set()

def ts() -> str:
    return datetime.utcnow().strftime("%H:%M:%S")

def valid_nick(n: str) -> bool:
    return bool(NICK_RE.match((n or "").strip()))

def safe_send_text(ws: WebSocket, obj: dict) -> None:
    # helper used only inside async funcs, but we sometimes build tasks; keep as sync wrapper for readability
    pass

async def ws_send(ws: WebSocket, obj: dict) -> None:
    try:
        await ws.send_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
    except Exception:
        # ignore send failures; cleanup will happen elsewhere
        pass

async def broadcast(obj: dict) -> None:
    dead = []
    for w in list(clients):
        try:
            await w.send_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
        except Exception:
            dead.append(w)
    if dead:
        for w in dead:
            await disconnect(w)

async def broadcast_users() -> None:
    # We don't broadcast join/leave messages to main log; only update online list
    async with state_lock:
        items = sorted(set(nick_by_ws.values()), key=lambda x: x.casefold())
    await broadcast({"type": "users", "t": ts(), "items": items})

async def disconnect(ws: WebSocket) -> None:
    async with state_lock:
        nick = nick_by_ws.pop(ws, None)
        clients.discard(ws)
        if nick:
            online_nicks_cf.discard(nick.casefold())
    try:
        await ws.close()
    except Exception:
        pass
    # Update user list
    await broadcast_users()

async def heartbeat(ws: WebSocket) -> None:
    try:
        while True:
            await asyncio.sleep(25)
            await ws_send(ws, {"type": "ping", "t": ts()})
    except Exception:
        return

HTML = r"""<!doctype html>
<html lang="lt">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Step3 Broadcast</title>
  <style>
    :root{
      --bg:#0b0f14; --panel:rgba(255,255,255,.05); --border:rgba(255,255,255,.12);
      --text:#d7e3f4; --muted:rgba(215,227,244,.55); --accent:#7cff6b; --danger:#ff7a7a;
      --mono:ui-monospace, Menlo, Consolas, monospace;
      --radius:14px;
    }
    body{ margin:0; background:var(--bg); color:var(--text); font-family:var(--mono); height:100vh; overflow:hidden; }
    .wrap{ height:100vh; display:grid; grid-template-rows:auto 1fr auto; gap:12px; padding:14px; box-sizing:border-box; }
    .top{
      border:1px solid var(--border); border-radius:var(--radius); background:var(--panel);
      padding:12px 14px; display:flex; justify-content:space-between; align-items:center; gap:12px;
    }
    .pill{ display:inline-flex; align-items:center; gap:8px; padding:6px 10px; border-radius:999px; border:1px solid var(--border); background:rgba(0,0,0,.18);}
    .dot{ width:10px; height:10px; border-radius:50%; background:#ff5c5c; }
    .dot.ok{ background:#21d07a; }
    .main{ min-height:0; display:grid; grid-template-columns: 260px 1fr; gap:12px; }
    .panel{ min-height:0; border:1px solid var(--border); border-radius:var(--radius); background:var(--panel); overflow:hidden; }
    .head{ padding:10px 12px; border-bottom:1px solid var(--border); color:var(--muted); display:flex; justify-content:space-between; }
    #users{ padding:10px 12px; overflow:auto; min-height:0; }
    .u{ padding:8px 10px; border:1px solid rgba(255,255,255,.06); border-radius:12px; background:rgba(0,0,0,.14); margin-bottom:8px; }
    #log{ padding:12px 14px; overflow:auto; min-height:0; white-space:pre-wrap; line-height:1.45; }
    .bar{
      border:1px solid var(--border); border-radius:var(--radius); background:var(--panel);
      padding:12px; display:grid; grid-template-columns: 1fr 120px; gap:12px;
    }
    input{ width:100%; padding:12px 12px; border-radius:12px; border:1px solid var(--border);
           background:rgba(0,0,0,.22); color:var(--text); font-family:var(--mono); outline:none; box-sizing:border-box; }
    button{ padding:12px 12px; border-radius:12px; border:1px solid rgba(0,0,0,.2); background:var(--accent); color:#061015; font-weight:900; cursor:pointer; }
    button:disabled{ opacity:.6; cursor:not-allowed; }
    .small{ color:var(--muted); font-size:12px; }
    .err{ color:var(--danger); font-size:12px; }
    #lobby{
      position:fixed; inset:0; display:flex; align-items:center; justify-content:center;
      background:rgba(0,0,0,.55);
      padding:18px; box-sizing:border-box;
    }
    .card{
      width:min(820px, 96vw);
      border:1px solid var(--border); border-radius:16px; background:rgba(255,255,255,.06);
      overflow:hidden;
    }
    .card .top{ border:0; border-bottom:1px solid var(--border); border-radius:0; background:transparent; }
    .body{ padding:14px; display:grid; grid-template-columns: 1fr 1fr; gap:12px; }
    @media (max-width: 900px){
      .main{ grid-template-columns: 1fr; }
      .body{ grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div id="lobby">
    <div class="card">
      <div class="top">
        <div><b>Step 3</b> – bendras kanalas (#main)</div>
        <div class="small">broadcast + online</div>
      </div>
      <div class="body">
        <div class="panel" style="border-radius:14px;">
          <div class="head"><span>Nick</span><span class="small">2–24</span></div>
          <div style="padding:12px;">
            <input id="nick" placeholder="pvz. Tomas" maxlength="24"/>
            <div id="nickStatus" class="small" style="margin-top:10px;"></div>
            <div id="nickErr" class="err" style="margin-top:8px; display:none;"></div>
          </div>
        </div>
        <div class="panel" style="border-radius:14px;">
          <div class="head"><span>Start</span><span class="small">/check_nick</span></div>
          <div style="padding:12px;">
            <button id="join" disabled style="width:100%;">Join</button>
            <div class="small" style="margin-top:10px;">
              Patikrink 2 naršykles: turi matyti vienas kito žinutes.
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div class="wrap" id="app" style="display:none;">
    <div class="top">
      <div><b>#main</b> <span class="small">bendras kanalas</span></div>
      <div class="pill"><span id="dot" class="dot"></span><span id="st">disconnected</span></div>
    </div>

    <div class="main">
      <div class="panel" style="display:grid; grid-template-rows:auto 1fr; min-height:0;">
        <div class="head"><span>Online</span><span class="small" id="cnt">0</span></div>
        <div id="users"></div>
      </div>
      <div class="panel">
        <div id="log"></div>
      </div>
    </div>

    <div class="bar">
      <input id="msg" placeholder="rašyk žinutę..." maxlength="200" disabled/>
      <button id="btn" disabled>Send</button>
    </div>
  </div>

<script>
  const lobby = document.getElementById("lobby");
  const appEl = document.getElementById("app");

  const nickEl = document.getElementById("nick");
  const joinBtn = document.getElementById("join");
  const nickStatus = document.getElementById("nickStatus");
  const nickErr = document.getElementById("nickErr");

  const logEl = document.getElementById("log");
  const usersEl = document.getElementById("users");
  const cntEl = document.getElementById("cnt");

  const msgEl = document.getElementById("msg");
  const btn = document.getElementById("btn");
  const dot = document.getElementById("dot");
  const st = document.getElementById("st");

  let ws = null;
  let connecting = false;
  let reconnectTimer = null;
  let reconnectDelay = 900;
  const maxDelay = 8000;
  let joinedOnce = false;

  let nick = "";

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
  function showNickErr(text){
    if(!text){ nickErr.style.display="none"; nickErr.textContent=""; }
    else { nickErr.style.display="block"; nickErr.textContent=text; }
  }

  async function checkNickNow(){
    const n = (nickEl.value||"").trim();
    joinBtn.disabled = true;
    showNickErr("");

    if(!validateNick(n)){
      nickStatus.textContent = "";
      showNickErr("Netinkamas nick (2–24, raidės/skaičiai/tarpas/_-.)");
      return;
    }
    nickStatus.textContent = "Tikrinama...";
    try{
      const qs = new URLSearchParams({nick:n}).toString();
      const r = await fetch(`/check_nick?${qs}`, {cache:"no-store"});
      const j = await r.json();
      if(j.ok){
        nickStatus.textContent = "Nick laisvas";
        joinBtn.disabled = false;
      }else{
        nickStatus.textContent = "";
        showNickErr(j.reason || "Nick užimtas");
      }
    }catch{
      nickStatus.textContent = "";
      showNickErr("Nepavyko patikrinti nick");
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

  function stopReconnect(){
    if(reconnectTimer){ clearTimeout(reconnectTimer); reconnectTimer=null; }
    reconnectDelay = 900;
  }
  function scheduleReconnect(){
    if(reconnectTimer) return;
    reconnectTimer = setTimeout(() => {
      reconnectTimer=null;
      connect(true);
      reconnectDelay = Math.min(maxDelay, Math.floor(reconnectDelay*1.4));
    }, reconnectDelay);
  }

  function connect(isReconnect=false){
    if(connecting) return;
    connecting = true;

    ws = new WebSocket(wsUrl());

    ws.onopen = () => {
      connecting = false;
      stopReconnect();
      setConn(true);
      joinedOnce = true;
      msgEl.disabled = false;
      btn.disabled = false;
      line("[client] connected as " + nick);
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
        showNickErr(o.text || "Join failed");
        try{ ws.close(); }catch{}
        return;
      }
      if(o.type === "users"){
        const items = o.items || [];
        cntEl.textContent = String(items.length);
        usersEl.innerHTML = "";
        for(const name of items){
          const d = document.createElement("div");
          d.className = "u";
          d.textContent = name;
          usersEl.appendChild(d);
        }
        return;
      }
      if(o.type === "msg"){
        line("[" + (o.t||"") + "] " + (o.text||""));
        return;
      }
    };

    ws.onclose = (ev) => {
      connecting = false;
      setConn(false);
      msgEl.disabled = true;
      btn.disabled = true;
      line(`[client] closed code=${ev.code}`);

      // Only auto-reconnect after we had a real connection
      if(joinedOnce){
        scheduleReconnect();
      }
    };

    ws.onerror = () => {
      // most browsers will also call onclose
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

    lobby.style.display="none";
    appEl.style.display="grid";
    connect(false);
    setTimeout(()=>msgEl.focus(), 60);
  };

  (function init(){
    const saved = (localStorage.getItem("nick")||"").trim();
    if(saved) nickEl.value = saved;
    scheduleNickCheck();
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

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    nick = (ws.query_params.get("nick") or "").strip()

    await ws.accept()
    peer = f"{ws.client.host if ws.client else 'unknown'}"
    log.info("WS accepted %s nick=%r", peer, nick)

    if not valid_nick(nick):
        await ws_send(ws, {"type": "error", "code": "BAD_NICK", "text": "Bad nick"})
        await ws.close()
        return

    async with state_lock:
        cf = nick.casefold()
        if cf in online_nicks_cf:
            await ws_send(ws, {"type": "error", "code": "NICK_TAKEN", "text": "Nick taken"})
            await ws.close()
            return
        online_nicks_cf.add(cf)
        clients.add(ws)
        nick_by_ws[ws] = nick

    hb_task: Optional[asyncio.Task] = asyncio.create_task(heartbeat(ws))

    # Send initial state to this client, then broadcast user list to everyone
    await ws_send(ws, {"type": "msg", "t": ts(), "text": f"welcome {nick} (shared room)"})
    await broadcast_users()

    try:
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
                continue

            if not isinstance(data, dict):
                continue

            if data.get("type") == "pong":
                continue

            if data.get("type") == "say":
                text = str(data.get("text", "")).strip()[:200]
                if not text:
                    continue
                await broadcast({"type": "msg", "t": ts(), "text": f"{nick}: {text}"})

    except WebSocketDisconnect:
        log.info("WS disconnect %s nick=%r", peer, nick)
    except Exception:
        log.exception("WS error %s nick=%r", peer, nick)
    finally:
        if hb_task:
            hb_task.cancel()
        await disconnect(ws)
        log.info("WS closed %s nick=%r", peer, nick)
