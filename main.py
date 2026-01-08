import asyncio
import random
import re
import time
from collections import deque
from dataclasses import dataclass
from html import escape
from typing import Deque, Dict, Optional, Set, Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, PlainTextResponse

app = FastAPI()

# =========================
# KONFIGŪRACIJA
# =========================
HISTORY_LIMIT = 250  # kiek žinučių laikom istorijoje

COLOR_PALETTE = [
    "#7CFF6B", "#6BE4FF", "#FF6BE8", "#FFD66B", "#6B9BFF",
    "#FF6B6B", "#6BFFB8", "#B56BFF", "#FF9F6B", "#6BFF6B",
    "#9CFF6B", "#6BFFD9", "#6B7CFF", "#FF6B9C", "#C9FF6B",
]

NICK_RE = re.compile(r"^[A-Za-z0-9ĄČĘĖĮŠŲŪŽąčęėįšųūž_\-\. ]{1,24}$")
ROLL_RE = re.compile(r"^/roll(?:\s+(\d{1,2})d(\d{1,3}))?$", re.IGNORECASE)

# =========================
# GLOBAL STATE (RAM)
# =========================
clients: Set[WebSocket] = set()
user_state: Dict[WebSocket, "User"] = {}
history: Deque[dict] = deque(maxlen=HISTORY_LIMIT)
topic: str = "Bendras kanalas"

state_lock = asyncio.Lock()

# =========================
# HTML (terminalinis UI)
# =========================
HTML = r"""<!doctype html>
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
    .dim{color:#3bbf3b}
    .sys{color:#3bbf3b}
    .t{color:#2f7f2f}
    .nick{font-weight:700}
    .msg{color:#a9ff9f}
    .topic{color:#a9ff9f;font-weight:700}
    .hint{color:#3bbf3b}
  </style>
</head>
<body>
  <div class="top">
    <span class="topic">iLoad.lt</span> — <span id="topic"></span>
    <span class="hint"> | Komandos: /help, /nick, /who, /roll, /me, /topic, /history, /clear</span>
  </div>
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
  const topicEl = document.getElementById("topic");

  function esc(s){
    return (s ?? "").toString()
      .replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;")
      .replaceAll('"',"&quot;").replaceAll("'","&#39;");
  }

  function addHtml(html){
    const div = document.createElement("div");
    div.innerHTML = html;
    log.appendChild(div);
    log.scrollTop = log.scrollHeight;
  }

  function clearLog(){
    log.innerHTML = "";
  }

  function wsUrl(){
    const proto = location.protocol === "https:" ? "wss" : "ws";
    return `${proto}://${location.host}/ws`;
  }

  let ws = null;
  let timer = null;

  function connect(){
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
    addHtml(`<span class="sys">[sys]</span> jungiamasi...`);
    ws = new WebSocket(wsUrl());

    ws.onopen = () => addHtml(`<span class="sys">[sys]</span> prisijungta.`);
    ws.onmessage = (ev) => {
      let o = null;
      try { o = JSON.parse(ev.data); } catch { addHtml(esc(ev.data)); return; }

      if(o.type === "topic"){
        topicEl.textContent = o.text || "";
        return;
      }

      if(o.type === "ctrl" && o.action === "clear"){
        clearLog();
        addHtml(`<span class="sys">[sys]</span> ekranas išvalytas.`);
        return;
      }

      if(o.type === "history"){
        clearLog();
        if(o.topic) topicEl.textContent = o.topic;
        (o.items || []).forEach(renderItem);
        addHtml(`<span class="sys">[sys]</span> istorija įkelta (${(o.items||[]).length} žinučių).`);
        return;
      }

      renderItem(o);
    };

    ws.onclose = () => {
      addHtml(`<span class="sys">[sys]</span> ryšys nutrūko, reconnect...`);
      if (!timer) timer = setInterval(connect, 1500);
    };
  }

  function renderItem(o){
    const t = esc(o.t || "");
    if(o.type === "msg"){
      const nick = esc(o.nick || "guest");
      const text = esc(o.text || "");
      const color = esc(o.color || "#a9ff9f");
      addHtml(`<span class="t">[${t}]</span> <span class="nick" style="color:${color}">${nick}</span>: <span class="msg">${text}</span>`);
      return;
    }
    if(o.type === "me"){
      const nick = esc(o.nick || "guest");
      const text = esc(o.text || "");
      const color = esc(o.color || "#a9ff9f");
      addHtml(`<span class="t">[${t}]</span> <span class="nick" style="color:${color}">*</span> <span class="nick" style="color:${color}">${nick}</span> <span class="msg">${text}</span>`);
      return;
    }
    // sys
    addHtml(`<span class="t">[${t}]</span> <span class="sys">${esc(o.text || "")}</span>`);
  }

  function send(){
    const nick = (nickEl.value || "").trim();    // serveris validuos / koreguos
    const text = msgEl.value.trim();
    if(!text) return;
    if(!ws || ws.readyState !== WebSocket.OPEN){
      addHtml(`<span class="sys">[sys]</span> dar neprisijungta.`);
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

# =========================
# DUOMENŲ MODELIS
# =========================
@dataclass
class User:
    nick: str
    color: str
    joined_at: float

def ts() -> str:
    return time.strftime("%H:%M:%S")

def make_default_nick() -> str:
    return f"guest{random.randint(1000, 9999)}"

def alloc_color(used: Set[str]) -> str:
    for c in COLOR_PALETTE:
        if c not in used:
            return c
    # jei spalvų neužtenka – fallback į pseudo-unikalią spalvą
    hue = random.randint(0, 359)
    return f"hsl({hue}, 90%, 65%)"

def valid_nick(n: str) -> bool:
    return bool(NICK_RE.match(n))

async def push_history(item: dict):
    # istorijoje laikome jau paruoštą struktūrą klientui
    history.append(item)

async def send(ws: WebSocket, obj: dict):
    await ws.send_json(obj)

async def broadcast(obj: dict, exclude: Optional[WebSocket] = None):
    dead = []
    for c in list(clients):
        if c is exclude:
            continue
        try:
            await c.send_json(obj)
        except Exception:
            dead.append(c)
    for d in dead:
        await remove_client(d)

async def remove_client(ws: WebSocket):
    async with state_lock:
        clients.discard(ws)
        u = user_state.pop(ws, None)
        # spalvą atlaisvinsim automatiškai, nes used skaičiuojam dinamiškai
    try:
        await ws.close()
    except Exception:
        pass
    return u

async def sysmsg(text: str, also_history: bool = True):
    item = {"type": "sys", "t": ts(), "text": text}
    if also_history:
        await push_history(item)
    await broadcast(item)

def used_colors() -> Set[str]:
    return {u.color for u in user_state.values()}

# =========================
# ROUTES
# =========================
@app.get("/", response_class=HTMLResponse)
def home():
    return HTML

@app.get("/health", response_class=PlainTextResponse)
def health():
    return f"OK {ts()}"

# =========================
# KOMANDOS
# =========================
HELP_TEXT = (
    "Komandos:\n"
    "  /help               - pagalba\n"
    "  /nick VARDAS        - pasikeisti slapyvardį\n"
    "  /who                - kas online (skaičius + vardai)\n"
    "  /topic TEKSTAS      - pakeisti temą (visiems)\n"
    "  /topic              - parodyti temą\n"
    "  /history [N]        - atsiųsti paskutines N žinučių (default 50)\n"
    "  /clear              - išvalyti savo ekraną\n"
    "  /me veiksmas        - action žinutė\n"
    "  /roll [NdM]         - kauliukas (pvz. /roll 2d6)\n"
    "  /flip               - monetos metimas\n"
    "  /time               - serverio laikas\n"
    "  /shrug              - prideda ¯\\_(ツ)_/¯ prie tavo žinutės\n"
    "  /color              - parodo tavo spalvą\n"
    "  /color new          - priskiria kitą laisvą spalvą (jei yra)\n"
)

async def handle_command(ws: WebSocket, nick_from_client: str, text: str) -> bool:
    text = text.strip()
    u = user_state.get(ws)
    if not u:
        return True

    if text in ("/help", "/?"):
        await send(ws, {"type": "sys", "t": ts(), "text": HELP_TEXT})
        return True

    if text.startswith("/nick "):
        new = text.split(" ", 1)[1].strip()
        if not valid_nick(new):
            await send(ws, {"type": "sys", "t": ts(), "text": "Netinkamas nick. Leista: raidės/skaičiai/tarpas/_-., iki 24."})
            return True
        old = u.nick
        u.nick = new
        await sysmsg(f"{old} dabar yra {u.nick}.", also_history=True)
        return True

    if text == "/who":
        async with state_lock:
            names = [user_state[c].nick for c in clients if c in user_state]
        await send(ws, {"type": "sys", "t": ts(), "text": f"Online ({len(names)}): " + ", ".join(sorted(set(names)))})
        return True

    if text.startswith("/topic"):
        parts = text.split(" ", 1)
        global topic
        if len(parts) == 1:
            await send(ws, {"type": "sys", "t": ts(), "text": f"Tema: {topic}"})
        else:
            topic = parts[1].strip()[:120] or "Bendras kanalas"
            await push_history({"type": "sys", "t": ts(), "text": f"Tema pakeista į: {topic}"})
            await broadcast({"type": "topic", "text": topic})
            await broadcast({"type": "sys", "t": ts(), "text": f"Tema pakeista į: {topic}"})
        return True

    if text.startswith("/history"):
        parts = text.split(" ", 1)
        n = 50
        if len(parts) == 2:
            try:
                n = int(parts[1].strip())
            except Exception:
                n = 50
        n = max(1, min(n, HISTORY_LIMIT))
        items = list(history)[-n:]
        await send(ws, {"type": "history", "topic": topic, "items": items})
        return True

    if text == "/clear":
        await send(ws, {"type": "ctrl", "action": "clear"})
        return True

    if text.startswith("/me "):
        action = text.split(" ", 1)[1].strip()
        if not action:
            return True
        item = {"type": "me", "t": ts(), "nick": u.nick, "color": u.color, "text": action[:240]}
        await push_history(item)
        await broadcast(item)
        return True

    m = ROLL_RE.match(text)
    if m:
        n = int(m.group(1)) if m.group(1) else 1
        sides = int(m.group(2)) if m.group(2) else 6
        n = max(1, min(n, 20))
        sides = max(2, min(sides, 1000))
        rolls = [random.randint(1, sides) for _ in range(n)]
        total = sum(rolls)
        item = {
            "type": "sys",
            "t": ts(),
            "text": f"{u.nick} meta {n}d{sides}: {rolls} (viso {total})"
        }
        await push_history(item)
        await broadcast(item)
        return True

    if text == "/flip":
        res = random.choice(["HERBAS", "SKAIČIUS"])
        item = {"type": "sys", "t": ts(), "text": f"{u.nick} meta monetą: {res}"}
        await push_history(item)
        await broadcast(item)
        return True

    if text == "/time":
        await send(ws, {"type": "sys", "t": ts(), "text": f"Serverio laikas: {ts()}"})
        return True

    if text.startswith("/shrug"):
        # Leidžiam naudoti kaip: "/shrug tekstas"
        parts = text.split(" ", 1)
        extra = (" " + parts[1].strip()) if len(parts) == 2 else ""
        item = {"type": "msg", "t": ts(), "nick": u.nick, "color": u.color, "text": f"{extra} ¯\\_(ツ)_/¯".strip()}
        await push_history(item)
        await broadcast(item)
        return True

    if text.startswith("/color"):
        parts = text.split(" ", 1)
        if len(parts) == 1:
            await send(ws, {"type": "sys", "t": ts(), "text": f"Tavo spalva: {u.color}"})
            return True
        if parts[1].strip().lower() == "new":
            # priskiriam kitą laisvą spalvą (jei yra)
            used = used_colors() - {u.color}
            u.color = alloc_color(used)
            item = {"type": "sys", "t": ts(), "text": f"{u.nick} pasikeitė spalvą."}
            await push_history(item)
            await broadcast(item)
            return True

    return False

# =========================
# WEBSOCKET
# =========================
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()

    async with state_lock:
        clients.add(ws)
        u = User(
            nick=make_default_nick(),
            color=alloc_color(used_colors()),
            joined_at=time.time(),
        )
        user_state[ws] = u

    # Išsiunčiam temą + istoriją
    await send(ws, {"type": "topic", "text": topic})
    await send(ws, {"type": "history", "topic": topic, "items": list(history)})

    # Prisijungimo pranešimas (į istoriją ir visiems)
    await sysmsg(f"{u.nick} prisijungė.", also_history=True)

    try:
        while True:
            data = await ws.receive_json()
            nick_in = str(data.get("nick", "")).strip()[:24]
            text = str(data.get("text", "")).strip()[:300]
            if not text:
                continue

            # jeigu klientas įvedė normalų nick (ne komanda), galim jį pritaikyti kaip „pirmą kartą“
            # bet tik jei dar guest*
            if nick_in and valid_nick(nick_in):
                if u.nick.startswith("guest"):
                    old = u.nick
                    u.nick = nick_in
                    await sysmsg(f"{old} dabar yra {u.nick}.", also_history=True)

            # komandos
            if text.startswith("/"):
                handled = await handle_command(ws, nick_in, text)
                if handled:
                    continue

            # paprasta žinutė
            item = {"type": "msg", "t": ts(), "nick": u.nick, "color": u.color, "text": text}
            await push_history(item)
            await broadcast(item)

    except WebSocketDisconnect:
        pass
    finally:
        left_user = await remove_client(ws)
        if left_user:
            await sysmsg(f"{left_user.nick} atsijungė.", also_history=True)
