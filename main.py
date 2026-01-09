import asyncio
import json
import logging
import random
import re
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Dict, Optional, Set, Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

# ------------------------------------------------------------
# STEP 4: Rooms + per-room user list + in-memory history
# ------------------------------------------------------------

app = FastAPI()
log = logging.getLogger("chat")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# --------- Validation ---------
NICK_RE = re.compile(r"^[A-Za-z0-9ĄČĘĖĮŠŲŪŽąčęėįšųūž_\-\. ]{2,24}$")
ROOM_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]{0,23}$")

def ts() -> str:
    return datetime.utcnow().strftime("%H:%M:%S")

def valid_nick(n: str) -> bool:
    return bool(NICK_RE.match((n or "").strip()))

def norm_room(room: str) -> str:
    return (room or "").strip().lstrip("#").lower()

def valid_room_key(r: str) -> bool:
    return bool(ROOM_RE.match(r))

# --------- Colors (very different) ---------
COLOR_PALETTE = [
    "#E6194B", "#3CB44B", "#FFE119", "#0082C8", "#F58231",
    "#911EB4", "#46F0F0", "#F032E6", "#D2F53C", "#FABEBE",
    "#008080", "#E6BEFF", "#AA6E28", "#FFD8B1", "#800000",
    "#AFFFc3", "#808000", "#000080", "#808080", "#111111",
]

def alloc_color(used: Set[str]) -> str:
    for c in COLOR_PALETTE:
        if c not in used:
            return c
    hue = random.randint(0, 359)
    return f"hsl({hue}, 90%, 45%)"

# --------- Models ---------
@dataclass
class User:
    nick: str
    color: str
    rooms: Set[str] = field(default_factory=set)
    active_room: str = "main"
    joined_at: float = field(default_factory=time.time)

@dataclass
class Room:
    key: str
    title: str
    topic: str
    clients: Set[WebSocket] = field(default_factory=set)
    users: Dict[WebSocket, User] = field(default_factory=dict)
    history: Deque[dict] = field(default_factory=lambda: deque(maxlen=240))

DEFAULT_ROOMS = {
    "main": {"title": "#main", "topic": "Bendras kanalas"},
    "games": {"title": "#games", "topic": "Žaidimai ir pramogos"},
    "help": {"title": "#help", "topic": "Pagalba / klausimai"},
}

# --------- In-memory state ---------
state_lock = asyncio.Lock()
rooms: Dict[str, Room] = {}
all_users_by_ws: Dict[WebSocket, User] = {}
all_ws_by_nick_cf: Dict[str, WebSocket] = {}

def ensure_room(room_key: str) -> Room:
    room_key = norm_room(room_key)
    if room_key in rooms:
        return rooms[room_key]
    meta = DEFAULT_ROOMS.get(room_key, None)
    title = meta["title"] if meta else f"#{room_key}"
    topic = meta["topic"] if meta else "Naujas kanalas"
    r = Room(key=room_key, title=title, topic=topic)
    rooms[room_key] = r
    return r

async def ws_send(ws: WebSocket, obj: dict) -> None:
    try:
        await ws.send_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
    except Exception:
        pass

async def room_broadcast(room_key: str, obj: dict, exclude: Optional[WebSocket] = None) -> None:
    r = ensure_room(room_key)
    payload = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    dead = []
    for w in list(r.clients):
        if exclude is not None and w is exclude:
            continue
        try:
            await w.send_text(payload)
        except Exception:
            dead.append(w)
    for w in dead:
        await disconnect_ws(w)

def room_userlist(room_key: str) -> list[dict]:
    r = ensure_room(room_key)
    items = [{"nick": u.nick, "color": u.color} for u in r.users.values()]
    items.sort(key=lambda x: x["nick"].casefold())
    return items

async def broadcast_userlist(room_key: str) -> None:
    await room_broadcast(room_key, {"type": "users", "room": room_key, "t": ts(), "items": room_userlist(room_key)})

async def send_rooms_list(ws: WebSocket) -> None:
    u = all_users_by_ws.get(ws)
    if not u:
        return
    keys = sorted(set(u.rooms) | set(DEFAULT_ROOMS.keys()))
    items = []
    for k in keys:
        r = ensure_room(k)
        items.append({"room": r.key, "title": r.title, "topic": r.topic})
    await ws_send(ws, {"type": "rooms", "t": ts(), "items": items})

async def join_room(ws: WebSocket, room_key: str) -> Tuple[bool, str]:
    u = all_users_by_ws.get(ws)
    if not u:
        return False, "no_user"

    key = norm_room(room_key)
    if not valid_room_key(key):
        return False, "bad_room"

    async with state_lock:
        r = ensure_room(key)
        if key in u.rooms:
            return True, "already"
        u.rooms.add(key)
        r.clients.add(ws)
        r.users[ws] = u

    await ws_send(ws, {"type": "topic", "room": key, "t": ts(), "text": f"{r.title} — {r.topic}"})
    await ws_send(ws, {"type": "history", "room": key, "t": ts(), "items": list(r.history)})
    await send_rooms_list(ws)
    await broadcast_userlist(key)
    return True, "ok"

async def leave_room(ws: WebSocket, room_key: str) -> Tuple[bool, str]:
    u = all_users_by_ws.get(ws)
    if not u:
        return False, "no_user"

    key = norm_room(room_key)
    if key == "main":
        return False, "deny_main"

    async with state_lock:
        if key not in u.rooms:
            return False, "not_member"
        u.rooms.discard(key)
        r = ensure_room(key)
        r.clients.discard(ws)
        r.users.pop(ws, None)

        if u.active_room == key:
            u.active_room = "main"

    await send_rooms_list(ws)
    await broadcast_userlist(key)
    return True, "ok"

async def focus_room(ws: WebSocket, room_key: str) -> None:
    u = all_users_by_ws.get(ws)
    if not u:
        return
    key = norm_room(room_key) or "main"
    if key not in u.rooms:
        ok, code = await join_room(ws, key)
        if not ok:
            await ws_send(ws, {"type": "sys", "t": ts(), "text": "Netinkamas kanalas."})
            return
    u.active_room = key
    r = ensure_room(key)
    await ws_send(ws, {"type": "topic", "room": key, "t": ts(), "text": f"{r.title} — {r.topic}"})
    await ws_send(ws, {"type": "users", "room": key, "t": ts(), "items": room_userlist(key)})
    await ws_send(ws, {"type": "history", "room": key, "t": ts(), "items": list(r.history)})

async def disconnect_ws(ws: WebSocket) -> None:
    async with state_lock:
        u = all_users_by_ws.pop(ws, None)
        if u:
            all_ws_by_nick_cf.pop(u.nick.casefold(), None)
            for rk in list(u.rooms):
                r = ensure_room(rk)
                r.clients.discard(ws)
                r.users.pop(ws, None)

    try:
        await ws.close()
    except Exception:
        pass

    if u:
        for rk in list(u.rooms):
            await broadcast_userlist(rk)

async def heartbeat(ws: WebSocket) -> None:
    try:
        while True:
            await asyncio.sleep(25)
            await ws_send(ws, {"type": "ping", "t": ts()})
    except Exception:
        return

# --------- Commands ---------
HELP_TEXT = (
    "Komandos:\\n"
    "  /help              - šis sąrašas\\n"
    "  /rooms             - kanalų sąrašas\\n"
    "  /join #room        - prisijungti/sukurti kanalą\\n"
    "  /leave #room       - palikti kanalą (iš #main negalima)\\n"
    "  /topic [TEKSTAS]   - rodyti / keisti temą aktyviame kanale\\n"
    "  /who               - online aktyviame kanale\\n"
    "  /history [N]       - istorija (default 120)\\n"
    "  /time              - serverio laikas (UTC)\\n"
)

async def handle_command(ws: WebSocket, u: User, active_room: str, text: str) -> bool:
    t0 = text.strip()
    low = t0.lower()

    if low in ("/help", "/?"):
        await ws_send(ws, {"type": "sys", "t": ts(), "text": HELP_TEXT})
        return True

    if low == "/rooms":
        await send_rooms_list(ws)
        return True

    if low.startswith("/join "):
        arg = t0.split(" ", 1)[1].strip()
        ok, code = await join_room(ws, arg)
        if not ok and code == "bad_room":
            await ws_send(ws, {"type": "sys", "t": ts(), "text": "Netinkamas kanalo pavadinimas. Pvz: #main, #games"})
        elif ok:
            await ws_send(ws, {"type": "sys", "t": ts(), "text": f"Prisijungei prie #{norm_room(arg)}"})
        return True

    if low.startswith("/leave "):
        arg = t0.split(" ", 1)[1].strip()
        ok, code = await leave_room(ws, arg)
        if not ok and code == "deny_main":
            await ws_send(ws, {"type": "sys", "t": ts(), "text": "Iš #main išeiti negalima."})
        elif not ok and code == "not_member":
            await ws_send(ws, {"type": "sys", "t": ts(), "text": "Tu nesi šiame kanale."})
        elif ok:
            await ws_send(ws, {"type": "sys", "t": ts(), "text": f"Palikai #{norm_room(arg)}"})
        return True

    if low == "/who":
        items = room_userlist(active_room)
        names = ", ".join([x["nick"] for x in items]) if items else "-"
        await ws_send(ws, {"type": "sys", "t": ts(), "text": f"Online {ensure_room(active_room).title}: {names}"})
        return True

    if low.startswith("/topic"):
        parts = t0.split(" ", 1)
        r = ensure_room(active_room)
        if len(parts) == 1:
            await ws_send(ws, {"type": "sys", "t": ts(), "text": f"Tema {r.title}: {r.topic}"})
            return True
        new_topic = parts[1].strip()[:120]
        if new_topic:
            r.topic = new_topic
            await room_broadcast(active_room, {"type": "topic", "room": active_room, "t": ts(), "text": f"{r.title} — {r.topic}"})
            await send_rooms_list(ws)
        return True

    if low.startswith("/history"):
        n = 120
        parts = t0.split(" ", 1)
        if len(parts) == 2:
            try:
                n = int(parts[1].strip())
            except Exception:
                n = 120
        r = ensure_room(active_room)
        items = list(r.history)[-max(1, min(n, 240)):]
        await ws_send(ws, {"type": "history", "room": active_room, "t": ts(), "items": items})
        return True

    if low == "/time":
        await ws_send(ws, {"type": "sys", "t": ts(), "text": f"Serverio laikas (UTC): {ts()}"})
        return True

    return False

# --------- HTTP routes ---------
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
        if n.casefold() in all_ws_by_nick_cf:
            return JSONResponse({"ok": False, "code": "NICK_TAKEN", "reason": "Nick užimtas."})
    return JSONResponse({"ok": True})

# --------- WebSocket endpoint ---------
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    nick = (ws.query_params.get("nick") or "").strip()

    await ws.accept()
    peer = f"{ws.client.host if ws.client else 'unknown'}"
    log.info("WS accepted %s nick=%r", peer, nick)

    if not valid_nick(nick):
        await ws_send(ws, {"type": "error", "code": "BAD_NICK", "text": "Netinkamas nick."})
        await ws.close()
        return

    for rk in DEFAULT_ROOMS.keys():
        ensure_room(rk)

    async with state_lock:
        cf = nick.casefold()
        if cf in all_ws_by_nick_cf:
            await ws_send(ws, {"type": "error", "code": "NICK_TAKEN", "text": "Nick užimtas."})
            await ws.close()
            return

        used = {u.color for u in all_users_by_ws.values()}
        color = alloc_color(used)

        u = User(nick=nick, color=color)
        all_users_by_ws[ws] = u
        all_ws_by_nick_cf[cf] = ws

    hb_task: Optional[asyncio.Task] = asyncio.create_task(heartbeat(ws))

    await join_room(ws, "main")
    await focus_room(ws, "main")
    await ws_send(ws, {"type": "me", "t": ts(), "nick": u.nick, "color": u.color})

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

            if data.get("type") == "focus":
                room_key = norm_room(data.get("room", "main"))
                await focus_room(ws, room_key)
                continue

            if data.get("type") == "say":
                room_key = norm_room(data.get("room", u.active_room or "main"))
                text = str(data.get("text", "")).strip()
                if not text:
                    continue

                if text.startswith("/"):
                    handled = await handle_command(ws, u, room_key, text)
                    if handled:
                        continue

                r = ensure_room(room_key)
                payload = {
                    "type": "msg",
                    "room": room_key,
                    "t": ts(),
                    "nick": u.nick,
                    "color": u.color,
                    "text": text[:300],
                }
                r.history.append(payload)
                await room_broadcast(room_key, payload)
                continue

    except WebSocketDisconnect:
        log.info("WS disconnect %s nick=%r", peer, nick)
    except Exception:
        log.exception("WS error %s nick=%r", peer, nick)
    finally:
        if hb_task:
            hb_task.cancel()
        await disconnect_ws(ws)
        log.info("WS closed %s nick=%r", peer, nick)

# --------- UI ---------
HTML = r"""<!doctype html>
<html lang="lt">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>HestioRooms – Step 4</title>
  <style>
    :root{
      --bg:#0b0f14; --panel:rgba(255,255,255,.05); --panel2:rgba(255,255,255,.04);
      --border:rgba(255,255,255,.12); --text:#d7e3f4; --muted:rgba(215,227,244,.55);
      --accent:#7cff6b; --accent2:#6be4ff; --danger:#ff7a7a;
      --mono:ui-monospace, Menlo, Consolas, monospace; --radius:14px; --shadow:0 16px 40px rgba(0,0,0,.35);
    }
    html,body{height:100%;}
    body{
      margin:0; color:var(--text); font-family:var(--mono); overflow:hidden;
      background:
        radial-gradient(900px 500px at 15% 10%, rgba(124,255,107,.12), transparent 60%),
        radial-gradient(900px 500px at 80% 35%, rgba(107,228,255,.10), transparent 60%),
        linear-gradient(180deg, rgba(0,0,0,.45), rgba(0,0,0,.55)), var(--bg);
    }
    .wrap{ height:100%; display:grid; grid-template-rows:auto 1fr auto; gap:12px; padding:14px; box-sizing:border-box; }
    .top{
      border:1px solid var(--border); border-radius:var(--radius); background:var(--panel);
      padding:12px 14px; display:flex; justify-content:space-between; align-items:center; gap:12px;
      backdrop-filter: blur(10px); box-shadow: var(--shadow);
    }
    .brand{ display:flex; gap:12px; align-items:baseline; }
    .brand b{ color:var(--accent); }
    .topic{ color:var(--text); font-weight:900; }
    .pill{ display:inline-flex; align-items:center; gap:8px; padding:6px 10px; border-radius:999px; border:1px solid var(--border); background:rgba(0,0,0,.18);}
    .dot{ width:10px; height:10px; border-radius:50%; background:#ff5c5c; }
    .dot.ok{ background:#21d07a; }
    .main{ min-height:0; display:grid; grid-template-columns: 260px 1fr 260px; gap:12px; }
    .panel{ min-height:0; border:1px solid var(--border); border-radius:var(--radius); background:var(--panel2); overflow:hidden; backdrop-filter: blur(10px); box-shadow: var(--shadow); }
    .head{ padding:10px 12px; border-bottom:1px solid var(--border); color:var(--muted); display:flex; justify-content:space-between; }
    .list{ padding:10px 10px 12px 10px; overflow:auto; min-height:0; }
    .item{
      display:flex; align-items:center; justify-content:space-between; gap:10px;
      padding:9px 10px; border-radius:14px; border:1px solid rgba(255,255,255,0.03);
      background:rgba(0,0,0,0.10); margin-bottom:8px; cursor:pointer; user-select:none;
    }
    .item:hover{ filter:brightness(1.05); }
    .item.active{ border-color: rgba(124,255,107,.28); background: rgba(124,255,107,.06); }
    .iname{ font-weight:900; font-size:13px; }
    .idesc{ color:var(--muted); font-size:11px; margin-top:2px; }
    .badge{
      min-width:22px; padding:2px 8px; border-radius:999px; font-size:12px; font-weight:900;
      color:var(--bg); background:var(--accent2); display:none; align-items:center; justify-content:center;
    }
    #log{ padding:12px 14px; overflow:auto; min-height:0; white-space:pre-wrap; line-height:1.45; }
    .line{ margin:2px 0; }
    .t{ color: rgba(124,255,107,.40); }
    .sys{ color: var(--muted); }
    .nick{ font-weight:900; }
    .bar{
      border:1px solid var(--border); border-radius:var(--radius); background:var(--panel);
      padding:12px; display:grid; grid-template-columns: 1fr 120px; gap:12px;
      backdrop-filter: blur(10px); box-shadow: var(--shadow);
    }
    input{
      width:100%; padding:12px 12px; border-radius:12px; border:1px solid var(--border);
      background:rgba(0,0,0,.22); color:var(--text); font-family:var(--mono); outline:none; box-sizing:border-box;
    }
    button{
      padding:12px 12px; border-radius:12px; border:1px solid rgba(0,0,0,.2);
      background:var(--accent); color:#061015; font-weight:900; cursor:pointer;
    }
    button:disabled{ opacity:.6; cursor:not-allowed; }
    #lobby{ position:fixed; inset:0; display:flex; align-items:center; justify-content:center; background:rgba(0,0,0,.55); padding:18px; box-sizing:border-box; }
    .card{
      width:min(920px, 96vw); border:1px solid var(--border); border-radius:16px; background:rgba(255,255,255,.06);
      overflow:hidden; backdrop-filter: blur(12px); box-shadow: var(--shadow);
    }
    .cardHead{ padding:14px 16px; border-bottom:1px solid var(--border); display:flex; justify-content:space-between; gap:12px; }
    .cardBody{ padding:14px 16px; display:grid; grid-template-columns: 1fr 1fr; gap:12px; }
    .err{ color:var(--danger); font-size:12px; display:none; margin-top:8px; }
    .small{ color:var(--muted); font-size:12px; }
    @media (max-width: 980px){ .main{ grid-template-columns: 1fr; } .cardBody{ grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <div id="lobby">
    <div class="card">
      <div class="cardHead">
        <div><b style="color:var(--accent)">HestioRooms</b> <span class="small">Step 4 – rooms</span></div>
        <div class="small">nick gate + rooms + history</div>
      </div>
      <div class="cardBody">
        <div class="panel" style="border-radius:14px;">
          <div class="head"><span>Nick</span><span class="small">2–24</span></div>
          <div style="padding:12px;">
            <input id="nick" placeholder="pvz. Tomas" maxlength="24"/>
            <div id="nickState" class="small" style="margin-top:10px;"></div>
            <div id="nickErr" class="err"></div>
          </div>
        </div>
        <div class="panel" style="border-radius:14px;">
          <div class="head"><span>Start</span><span class="small">#main</span></div>
          <div style="padding:12px;">
            <button id="join" disabled style="width:100%;">Join</button>
            <div class="small" style="margin-top:10px;">
              Testas: 2 naršyklės turi matyti vienas kito žinutes tame pačiame kanale.
            </div>
            <div class="small" style="margin-top:6px;">
              Komandos: /help, /join #games, /leave #games, /topic, /history
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div class="wrap" id="app" style="display:none;">
    <div class="top">
      <div class="brand">
        <b>HestioRooms</b>
        <span class="topic" id="topic">#main</span>
      </div>
      <div class="pill">
        <span id="dot" class="dot"></span>
        <span id="st">disconnected</span>
      </div>
    </div>

    <div class="main">
      <div class="panel" style="display:grid; grid-template-rows:auto 1fr auto; min-height:0;">
        <div class="head"><span>Kanalai</span><span class="small">click</span></div>
        <div class="list" id="rooms"></div>
        <div style="padding:10px 10px 12px 10px; border-top:1px solid var(--border);">
          <input id="roomJoin" placeholder="įrašyk #room ir Enter (pvz #games)" />
        </div>
      </div>

      <div class="panel">
        <div id="log"></div>
      </div>

      <div class="panel" style="display:grid; grid-template-rows:auto 1fr; min-height:0;">
        <div class="head"><span>Online</span><span class="small" id="cnt">0</span></div>
        <div class="list" id="users"></div>
      </div>
    </div>

    <div class="bar">
      <input id="msg" placeholder="rašyk žinutę..." maxlength="300" disabled/>
      <button id="btn" disabled>Send</button>
    </div>
  </div>

<script>
  const lobby = document.getElementById("lobby");
  const appEl = document.getElementById("app");

  const nickEl = document.getElementById("nick");
  const joinBtn = document.getElementById("join");
  const nickState = document.getElementById("nickState");
  const nickErr = document.getElementById("nickErr");

  const roomsEl = document.getElementById("rooms");
  const usersEl = document.getElementById("users");
  const cntEl = document.getElementById("cnt");
  const topicEl = document.getElementById("topic");
  const logEl = document.getElementById("log");

  const msgEl = document.getElementById("msg");
  const btn = document.getElementById("btn");
  const roomJoinEl = document.getElementById("roomJoin");

  const dot = document.getElementById("dot");
  const st = document.getElementById("st");

  let ws = null;
  let connecting = false;
  let reconnectTimer = null;
  let reconnectDelay = 900;
  const maxDelay = 8000;
  let joinedOnce = false;

  let nick = "";

  const roomState = new Map(); // roomKey -> {title, topic, unread, items:[]}
  let activeRoom = "main";

  function validateNick(n){
    return /^[A-Za-z0-9ĄČĘĖĮŠŲŪŽąčęėįšųūž_\\-\\. ]{2,24}$/.test((n||"").trim());
  }
  function esc(s){
    return (s ?? "").toString()
      .replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;")
      .replaceAll('"',"&quot;").replaceAll("'","&#39;");
  }
  function setConn(ok){
    if(ok){ dot.classList.add("ok"); st.textContent="connected"; }
    else { dot.classList.remove("ok"); st.textContent="disconnected"; }
  }
  function showNickErr(text){
    if(!text){ nickErr.style.display="none"; nickErr.textContent=""; }
    else { nickErr.style.display="block"; nickErr.textContent=text; }
  }
  function lineHtml(o){
    const t = esc(o.t || "");
    if(o.type === "msg"){
      return `<div class="line"><span class="t">[${t}]</span> <span class="nick" style="color:${esc(o.color||'#d7e3f4')}">${esc(o.nick||'???')}</span>: ${esc(o.text||'')}</div>`;
    }
    return `<div class="line"><span class="t">[${t}]</span> <span class="sys">${esc(o.text||'')}</span></div>`;
  }
  function ensureRoom(room, title, topic){
    if(!roomState.has(room)){
      roomState.set(room, {room, title:title||("#"+room), topic:topic||"", unread:0, items:[]});
    }else{
      const r = roomState.get(room);
      if(title) r.title = title;
      if(topic!==undefined) r.topic = topic;
    }
    return roomState.get(room);
  }
  function renderRooms(){
    const rooms = Array.from(roomState.values());
    rooms.sort((a,b)=>a.title.localeCompare(b.title, "lt"));
    roomsEl.innerHTML = "";
    for(const r of rooms){
      const row = document.createElement("div");
      row.className = "item" + (r.room === activeRoom ? " active" : "");
      const unread = r.unread || 0;
      row.innerHTML = `
        <div>
          <div class="iname">${esc(r.title)}</div>
          <div class="idesc">${esc(r.topic || "")}</div>
        </div>
        <div class="badge" style="${unread>0 ? 'display:inline-flex;' : ''}">${unread}</div>
      `;
      row.addEventListener("click", () => switchRoom(r.room));
      roomsEl.appendChild(row);
    }
  }
  function renderUsers(items){
    cntEl.textContent = String(items.length);
    usersEl.innerHTML = "";
    for(const u of items){
      const row = document.createElement("div");
      row.className = "item";
      row.style.cursor = "default";
      row.innerHTML = `
        <div>
          <div class="iname" style="color:${esc(u.color||'#d7e3f4')}">${esc(u.nick||'???')}</div>
          <div class="idesc">online</div>
        </div>
        <div class="badge" style="display:none;"></div>
      `;
      usersEl.appendChild(row);
    }
  }
  function renderActiveLog(){
    const r = ensureRoom(activeRoom);
    topicEl.textContent = "#" + activeRoom;
    logEl.innerHTML = r.items.map(lineHtml).join("");
    logEl.scrollTop = logEl.scrollHeight;
  }
  function appendToRoom(room, obj, noUnread=false){
    const r = ensureRoom(room);
    r.items.push(obj);
    if(r.items.length > 260) r.items.splice(0, r.items.length - 260);
    if(room !== activeRoom && !noUnread){
      r.unread = (r.unread||0) + 1;
    }
    if(room === activeRoom){
      const div = document.createElement("div");
      div.innerHTML = lineHtml(obj);
      logEl.appendChild(div.firstChild);
      logEl.scrollTop = logEl.scrollHeight;
    }
  }

  async function checkNickNow(){
    const n = (nickEl.value||"").trim();
    joinBtn.disabled = true;
    showNickErr("");
    nickState.textContent = "";

    if(!validateNick(n)){
      showNickErr("Netinkamas nick (2–24, raidės/skaičiai/tarpas/_-.)");
      return;
    }

    nickState.textContent = "Tikrinama...";
    try{
      const qs = new URLSearchParams({nick:n}).toString();
      const r = await fetch(`/check_nick?${qs}`, {cache:"no-store"});
      const j = await r.json();
      if(j.ok){
        nickState.textContent = "Nick laisvas";
        joinBtn.disabled = false;
      }else{
        showNickErr(j.reason || "Nick užimtas");
      }
    }catch{
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
  function wsSend(obj){
    if(!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify(obj));
  }
  function connect(){
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
      appendToRoom(activeRoom, {type:"sys", t:new Date().toLocaleTimeString(), text:"connected"}, true);
      wsSend({type:"focus", room:activeRoom});
    };

    ws.onmessage = (ev) => {
      let o=null;
      try{ o = JSON.parse(ev.data); }catch{ return; }

      if(o.type === "ping"){
        ws.send(JSON.stringify({type:"pong", t: Date.now()}));
        return;
      }
      if(o.type === "error"){
        appendToRoom(activeRoom, {type:"sys", t:o.t||"", text:o.text||"error"}, true);
        showNickErr(o.text || "Join failed");
        try{ ws.close(); }catch{}
        return;
      }
      if(o.type === "rooms"){
        for(const it of (o.items || [])){
          ensureRoom(it.room, it.title, it.topic);
        }
        renderRooms();
        return;
      }
      if(o.type === "topic"){
        const room = o.room || "main";
        const r = ensureRoom(room);
        const parts = (o.text || "").split("—");
        if(parts.length >= 2){
          r.topic = parts.slice(1).join("—").trim();
        }
        renderRooms();
        return;
      }
      if(o.type === "users"){
        const room = o.room || "main";
        if(room === activeRoom){
          renderUsers(o.items || []);
        }
        return;
      }
      if(o.type === "history"){
        const room = o.room || "main";
        const r = ensureRoom(room);
        r.items = [];
        for(const it of (o.items || [])){
          r.items.push(it);
        }
        if(room === activeRoom){
          renderActiveLog();
        }
        renderRooms();
        return;
      }
      if(o.type === "msg"){
        const room = o.room || "main";
        ensureRoom(room);
        appendToRoom(room, o, false);
        renderRooms();
        return;
      }
      if(o.type === "sys"){
        appendToRoom(activeRoom, {type:"sys", t:o.t||"", text:o.text||""}, false);
        return;
      }
    };

    ws.onclose = () => {
      connecting = false;
      setConn(false);
      msgEl.disabled = true;
      btn.disabled = true;
      appendToRoom(activeRoom, {type:"sys", t:new Date().toLocaleTimeString(), text:"disconnected, reconnecting..."}, false);
      if(joinedOnce) scheduleReconnect();
    };
  }

  function switchRoom(room){
    room = (room||"main").toString().replace(/^#/, "").toLowerCase();
    if(!room) room = "main";
    activeRoom = room;
    const r = ensureRoom(room);
    r.unread = 0;
    renderRooms();
    renderActiveLog();
    wsSend({type:"focus", room});
  }

  function send(){
    const text = (msgEl.value||"").trim();
    if(!text) return;
    if(!ws || ws.readyState !== WebSocket.OPEN){
      appendToRoom(activeRoom, {type:"sys", t:new Date().toLocaleTimeString(), text:"no connection"}, false);
      return;
    }
    wsSend({type:"say", room: activeRoom, text});
    msgEl.value="";
  }
  btn.onclick = send;
  msgEl.addEventListener("keydown", (e)=>{ if(e.key==="Enter") send(); });

  roomJoinEl.addEventListener("keydown", (e) => {
    if(e.key !== "Enter") return;
    const val = (roomJoinEl.value || "").trim();
    if(!val) return;
    wsSend({type:"say", room: activeRoom, text: `/join ${val}`});
    roomJoinEl.value = "";
  });

  joinBtn.onclick = async () => {
    await checkNickNow();
    if(joinBtn.disabled) return;

    nick = (nickEl.value||"").trim();
    localStorage.setItem("nick", nick);

    lobby.style.display="none";
    appEl.style.display="grid";

    ensureRoom("main", "#main", "Bendras kanalas");
    ensureRoom("games", "#games", "Žaidimai ir pramogos");
    ensureRoom("help", "#help", "Pagalba / klausimai");
    renderRooms();
    switchRoom("main");

    connect();
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
