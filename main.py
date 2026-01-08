import time
from typing import Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, PlainTextResponse

app = FastAPI()
clients: Set[WebSocket] = set()

def ts() -> str:
    return time.strftime("%H:%M:%S")

HTML = """<!doctype html>
<html lang="lt">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>iLoad.lt Chat</title>
  <style>
    body{margin:0;background:#0b0f0c;color:#7cff6b;
      font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Courier New",monospace;}
    .top{padding:12px 16px;border-bottom:1px solid #163016;color:#3bbf3b;font-size:14px}
    .wrap{display:grid;grid-template-rows:1fr auto;height:calc(100vh - 45px)}
    #log{padding:14px 16px;overflow:auto;white-space:pre-wrap;line-height:1.35}
    .bar{padding:10px 12px;border-top:1px solid #163016;display:grid;grid-template-columns:160px 1fr 120px;
      gap:10px;align-items:center;background:#070a08}
    input{width:100%;padding:10px;background:#050705;border:1px solid #163016;color:#a9ff9f;
      outline:none;border-radius:8px;font-size:14px}
    button{padding:10px 12px;background:#0e1a10;border:1px solid #1f3a22;color:#a9ff9f;border-radius:8px;
      cursor:pointer;font-weight:600}
    button:hover{filter:brightness(1.15)}
  </style>
</head>
<body>
  <div class="top">iLoad.lt — klasės chat. Įrašyk nick ir rašyk. Komandos: /who</div>
  <div class="wrap">
    <div id="log"></div>
    <div class="bar">
      <input id="nick" placeholder="nick (pvz. Tomas)" maxlength="24"/>
      <input id="msg" placeholder="rašyk žinutę ir Enter..." maxlength="300"/>
      <button id="btn">Siųsti</button>
    </div>
  </div>

<script>
  const log = document.getElementById("log");
  const nickEl = document.getElementById("nick");
  const msgEl  = document.getElementById("msg");
  const btn    = document.getElementById("btn");

  function addLine(s){
    log.textContent += s + "\\n";
    log.scrollTop = log.scrollHeight;
  }

  function wsUrl(){
    const proto = location.protocol === "https:" ? "wss" : "ws";
    return `${proto}://${location.host}/ws`;
  }

  let ws = null;
  let timer = null;

  function connect(){
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
    addLine("[sys] jungiamasi...");
    ws = new WebSocket(wsUrl());

    ws.onopen = () => addLine("[sys] prisijungta.");
    ws.onmessage = (ev) => {
      try{
        const o = JSON.parse(ev.data);
        if(o.type === "msg") addLine(`[${o.t}] ${o.nick}: ${o.text}`);
        else addLine(`[${o.t}] ${o.text}`);
      }catch{
        addLine(ev.data);
      }
    };
    ws.onclose = () => {
      addLine("[sys] ryšys nutrūko, reconnect...");
      if (!timer) timer = setInterval(connect, 1500);
    };
  }

  function send(){
    const nick = (nickEl.value || "guest").trim();
    const text = msgEl.value.trim();
    if(!text) return;
    if(!ws || ws.readyState !== WebSocket.OPEN){
      addLine("[sys] dar neprisijungta.");
      return;
    }
    ws.send(JSON.stringify({nick, text}));
    msgEl.value = "";
  }

  btn.onclick = send;
  msgEl.addEventListener("keydown", (e) => { if(e.key === "Enter") send(); });

  connect();
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML

@app.get("/health", response_class=PlainTextResponse)
def health():
    return f"OK {ts()}"

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    try:
        await ws.send_json({"type":"sys","t":ts(),"text":"Prisijungei. Rašyk žinutę."})
        while True:
            data = await ws.receive_json()
            nick = str(data.get("nick","guest"))[:24]
            text = str(data.get("text",""))[:300].strip()
            if not text:
                continue
            if text == "/who":
                await ws.send_json({"type":"sys","t":ts(),"text":f"Online: {len(clients)}"})
                continue

            msg = {"type":"msg","t":ts(),"nick":nick,"text":text}
            dead = []
            for c in list(clients):
                try:
                    await c.send_json(msg)
                except Exception:
                    dead.append(c)
            for d in dead:
                clients.discard(d)
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(ws)
