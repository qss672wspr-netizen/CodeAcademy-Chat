import asyncio
import json
import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, PlainTextResponse

# ------------------------------------------------------------
# STEP 1: Minimal WebSocket "echo chat" with strong logging
# Goal: prove WS stability on Render before adding features.
# ------------------------------------------------------------

app = FastAPI()
log = logging.getLogger("chat")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

def ts() -> str:
    # Server-side timestamp in ISO-ish form (Render is UTC by default)
    return datetime.utcnow().strftime("%H:%M:%S")

HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Step1 WS Debug</title>
  <style>
    body { font-family: ui-monospace, Menlo, Consolas, monospace; margin: 0; background:#0b0f14; color:#d7e3f4; }
    .wrap{ max-width: 980px; margin: 18px auto; padding: 0 14px; }
    .card{ border: 1px solid rgba(255,255,255,.12); border-radius: 14px; background: rgba(255,255,255,.04); overflow:hidden;}
    .top{ padding: 12px 14px; border-bottom:1px solid rgba(255,255,255,.10); display:flex; gap:12px; align-items:center; justify-content:space-between;}
    .pill{ display:inline-flex; gap:8px; align-items:center; padding:6px 10px; border-radius:999px; border:1px solid rgba(255,255,255,.12); background:rgba(0,0,0,.2);}
    .dot{ width:10px; height:10px; border-radius:50%; background:#ff5c5c;}
    .dot.ok{ background:#21d07a;}
    #log{ height: 56vh; overflow:auto; padding: 12px 14px; white-space: pre-wrap; line-height: 1.4; }
    .bar{ display:flex; gap:10px; padding: 12px 14px; border-top:1px solid rgba(255,255,255,.10);}
    input{ flex:1; padding:12px 12px; border-radius: 12px; border:1px solid rgba(255,255,255,.14);
           background: rgba(0,0,0,.25); color:#d7e3f4; outline:none;}
    button{ padding: 12px 14px; border-radius: 12px; border:1px solid rgba(255,255,255,.14);
            background:#7cff6b; color:#061015; font-weight:900; cursor:pointer;}
    button:disabled{ opacity:.6; cursor:not-allowed;}
    .small{ color: rgba(215,227,244,.6); font-size: 12px;}
    a{ color:#9fd2ff; text-decoration:none;}
  </style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <div class="top">
      <div><b>Step 1</b> – Minimal WS</div>
      <div class="pill"><span id="dot" class="dot"></span><span id="st">disconnected</span></div>
    </div>
    <div id="log"></div>
    <div class="bar">
      <input id="msg" placeholder="type message and press Enter" maxlength="200"/>
      <button id="btn">Send</button>
    </div>
    <div class="top" style="border-top:1px solid rgba(255,255,255,.10); border-bottom:0;">
      <div class="small">
        Health: <a href="/health" target="_blank">/health</a> |
        This step only echoes messages back.
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

  let ws = null;
  let reconnectTimer = null;
  let reconnectDelay = 800;
  const maxDelay = 8000;
  let connecting = false;

  function line(s){
    logEl.textContent += s + "\n";
    logEl.scrollTop = logEl.scrollHeight;
  }

  function setConn(ok){
    if(ok){ dot.classList.add("ok"); st.textContent="connected"; }
    else { dot.classList.remove("ok"); st.textContent="disconnected"; }
  }

  function wsUrl(){
    const proto = location.protocol === "https:" ? "wss" : "ws";
    return `${proto}://${location.host}/ws`;
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
      line("=== page loaded, connecting... ===");
    }else{
      line(`=== reconnect in ${reconnectDelay}ms... ===`);
    }

    ws = new WebSocket(wsUrl());

    ws.onopen = () => {
      connecting = false;
      stopReconnect();
      setConn(true);
      line("[client] ws open");

      // client->server ping every 25s (some proxies like activity)
      // server will also ping; this is harmless.
      diag.textContent = "keepalive active";
    };

    ws.onmessage = (ev) => {
      let o=null;
      try{ o = JSON.parse(ev.data); }catch{ return; }
      if(o.type === "ping"){
        ws.send(JSON.stringify({type:"pong", t: Date.now()}));
        return;
      }
      line(`[server ${o.t || ""}] ${o.text || ""}`);
    };

    ws.onclose = (ev) => {
      connecting = false;
      setConn(false);
      line(`[client] ws closed code=${ev.code} reason=${ev.reason || ""}`);
      scheduleReconnect();
    };

    ws.onerror = () => {
      // will be followed by close in most browsers
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

  connect(false);
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

async def heartbeat(ws: WebSocket):
    """Server->client ping to keep the connection warm and detect dead peers."""
    try:
        while True:
            await asyncio.sleep(25)
            await ws.send_text(json.dumps({"type":"ping","t":ts()}))
    except Exception:
        # Connection is likely closed.
        return

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    # Accept immediately so we can send structured errors if needed later.
    await ws.accept()
    peer = f"{ws.client.host if ws.client else 'unknown'}"
    log.info("WS accepted from %s", peer)

    hb_task: Optional[asyncio.Task] = asyncio.create_task(heartbeat(ws))

    try:
        while True:
            # Use receive() instead of receive_json() to avoid silent JSON framing edge cases.
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
                await ws.send_text(json.dumps({"type":"msg","t":ts(),"text":f"echo: {text}"}))
                continue

    except WebSocketDisconnect:
        log.info("WS disconnect %s", peer)
    except Exception:
        log.exception("WS error from %s", peer)
    finally:
        if hb_task:
            hb_task.cancel()
        try:
            await ws.close()
        except Exception:
            pass
        log.info("WS closed %s", peer)
