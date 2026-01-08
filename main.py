import asyncio
import random
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, PlainTextResponse

app = FastAPI()

# =========================
# KONFIGŪRACIJA
# =========================
HISTORY_LIMIT = 250

COLOR_PALETTE = [
    "#E6194B",  # red
    "#3CB44B",  # green
    "#FFE119",  # yellow
    "#0082C8",  # blue
    "#F58231",  # orange
    "#911EB4",  # purple
    "#46F0F0",  # cyan
    "#F032E6",  # magenta
    "#D2F53C",  # lime
    "#FABEBE",  # pink
    "#008080",  # teal
    "#E6BEFF",  # lavender
    "#AA6E28",  # brown
    "#FFFAC8",  # light yellow
    "#800000",  # maroon
    "#AFFFc3",  # mint
    "#808000",  # olive
    "#FFD8B1",  # apricot
    "#000080",  # navy
    "#808080",  # gray
    "#000000",  # black
    "#FFFFFF",  # white (bus ryški, bet matysis ant juodo fono)
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
# HTML (terminalinis UI + ONLINE panelis)
# =========================
HTML = r"""<!doctype html>
<html lang="lt">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>iLoad.lt Chat</title>
  <style>
    :root{
      --bg: #07090b;
      --panel: rgba(10, 14, 12, 0.72);
      --panel2: rgba(8, 10, 9, 0.65);
      --border: rgba(70, 255, 150, 0.16);
      --text: #b8ffcf;
      --muted: rgba(184, 255, 207, 0.55);
      --accent: #7cff6b;
      --accent2: #6be4ff;
      --warn: #ffd66b;
      --danger: #ff6b6b;
      --shadow: 0 10px 30px rgba(0,0,0,.45);
      --radius: 14px;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Courier New", monospace;
    }

    /* THEMES */
    body.theme-cyber{
      --bg:#06080a; --panel:rgba(6,10,10,.72); --panel2:rgba(5,7,8,.65);
      --border:rgba(124,255,107,.18); --text:#caffd9; --muted:rgba(202,255,217,.55);
      --accent:#7cff6b; --accent2:#6be4ff;
    }
    body.theme-glass{
      --bg:#0a0c12; --panel:rgba(18, 22, 35, .62); --panel2:rgba(14, 18, 28, .55);
      --border:rgba(130,170,255,.18); --text:#e7eeff; --muted:rgba(231,238,255,.55);
      --accent:#82aaff; --accent2:#ff6be8;
    }
    body.theme-matrix{
      --bg:#020403; --panel:rgba(2, 8, 4, .70); --panel2:rgba(1, 6, 3, .60);
      --border:rgba(124,255,107,.16); --text:#bfffd0; --muted:rgba(191,255,208,.55);
      --accent:#00ff84; --accent2:#7cff6b;
    }
    body.theme-crt{
      --bg:#050607; --panel:rgba(7, 9, 10, .78); --panel2:rgba(6, 7, 8, .68);
      --border:rgba(255,214,107,.16); --text:#ffe9b8; --muted:rgba(255,233,184,.55);
      --accent:#ffd66b; --accent2:#7cff6b;
    }

    html, body { height:100%; }
    body{
      margin:0;
      background: var(--bg);
      color: var(--text);
      font-family: var(--mono);
      overflow:hidden;
    }

    /* Background “PRO” layers */
    .bg{
      position: fixed;
      inset: 0;
      z-index: 0;
      pointer-events:none;
    }
    /* subtle grid */
    .bg::before{
      content:"";
      position:absolute;
      inset:-2px;
      opacity: .20;
      background:
        linear-gradient(to right, rgba(255,255,255,.04) 1px, transparent 1px),
        linear-gradient(to bottom, rgba(255,255,255,.03) 1px, transparent 1px);
      background-size: 46px 46px;
      transform: translateZ(0);
      mask-image: radial-gradient(circle at 40% 10%, rgba(0,0,0,1) 0%, rgba(0,0,0,.7) 40%, rgba(0,0,0,0) 75%);
    }
    /* scanlines */
    .bg::after{
      content:"";
      position:absolute;
      inset:0;
      opacity:.10;
      background: repeating-linear-gradient(
        to bottom,
        rgba(0,0,0,0) 0px,
        rgba(0,0,0,0) 2px,
        rgba(0,0,0,.55) 3px
      );
      animation: scan 8s linear infinite;
    }
    @keyframes scan{
      0%{ transform: translateY(0); }
      100%{ transform: translateY(10px); }
    }

    /* layout */
    .app{
      position: relative;
      z-index: 1;
      height: 100%;
      display: grid;
      grid-template-rows: auto 1fr auto;
      gap: 12px;
      padding: 14px;
      box-sizing: border-box;
    }

    .topbar{
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:12px;
      padding: 12px 14px;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      background: linear-gradient(180deg, var(--panel), rgba(0,0,0,0));
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }
    .brand{
      display:flex;
      gap:10px;
      align-items:baseline;
      min-width: 240px;
    }
    .brand .title{
      font-weight: 800;
      letter-spacing: .2px;
      color: var(--accent);
    }
    .brand .topic{
      color: var(--text);
      font-weight: 700;
    }
    .status{
      display:flex;
      align-items:center;
      gap:10px;
      color: var(--muted);
      font-size: 13px;
      flex: 1;
      justify-content:center;
      text-align:center;
    }
    .pill{
      border: 1px solid var(--border);
      background: rgba(0,0,0,.18);
      padding: 6px 10px;
      border-radius: 999px;
      display:inline-flex;
      align-items:center;
      gap:8px;
      white-space:nowrap;
    }
    .dot{
      width:10px;height:10px;border-radius:999px;
      background: var(--danger);
      box-shadow: 0 0 14px rgba(255,107,107,.18);
    }
    .dot.ok{
      background: var(--accent);
      box-shadow: 0 0 14px rgba(124,255,107,.18);
    }

    .controls{
      display:flex;
      gap:10px;
      align-items:center;
      justify-content:flex-end;
      min-width: 320px;
    }
    select, .chk{
      font-family: var(--mono);
      color: var(--text);
      background: rgba(0,0,0,.25);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 8px 10px;
      outline:none;
    }
    .chk{
      display:flex;
      gap:8px;
      align-items:center;
      user-select:none;
      cursor:pointer;
    }
    .chk input{ accent-color: var(--accent); }

    .main{
      display:grid;
      grid-template-columns: 1fr 290px;
      gap: 12px;
      min-height: 0; /* allow scroll areas */
    }

    .panel{
      border: 1px solid var(--border);
      border-radius: var(--radius);
      background: var(--panel2);
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
      overflow: hidden;
      min-height: 0;
    }

    #log{
      padding: 14px 16px;
      overflow:auto;
      white-space: pre-wrap;
      line-height: 1.45;
      min-height: 0;
    }

    /* message styles */
    .line{ margin: 2px 0; }
    .t{ color: rgba(124,255,107,.40); }
    .sys{ color: var(--muted); }
    .msg{ color: var(--text); }
    .nick{ font-weight: 800; }
    .me{ color: var(--accent2); }

    /* sidebar */
    .sidehead{
      padding: 12px 12px;
      border-bottom: 1px solid var(--border);
      display:flex;
      justify-content:space-between;
      align-items:center;
      color: var(--muted);
      font-size: 13px;
    }
    .sidehead b{ color: var(--text); }
    .search{
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
    }
    .search input{
      width:100%;
      padding:10px 10px;
      font-family: var(--mono);
      background: rgba(0,0,0,.22);
      border: 1px solid var(--border);
      border-radius: 10px;
      color: var(--text);
      outline:none;
      box-sizing:border-box;
    }

    #users{
      padding: 10px 10px 12px 10px;
      overflow:auto;
      min-height:0;
      max-height:100%;
    }
    .user{
      display:flex;
      align-items:center;
      gap:10px;
      padding: 8px 10px;
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.03);
      background: rgba(0,0,0,0.12);
      margin-bottom: 8px;
    }
    .user .udot{
      width:10px;height:10px;border-radius:999px;
      box-shadow: 0 0 14px rgba(0,0,0,.25);
      flex:0 0 auto;
    }
    .user .uname{
      font-weight: 800;
      font-size: 13px;
      word-break: break-word;
    }

    .bottombar{
      display:grid;
      grid-template-columns: 180px 1fr 130px;
      gap: 12px;
      padding: 12px 12px;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      background: var(--panel);
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }
    .bottombar input{
      width:100%;
      padding: 12px 12px;
      font-family: var(--mono);
      background: rgba(0,0,0,.22);
      border: 1px solid var(--border);
      border-radius: 12px;
      color: var(--text);
      outline:none;
      box-sizing:border-box;
    }
    .bottombar button{
      padding: 12px 12px;
      font-family: var(--mono);
      font-weight: 800;
      color: var(--bg);
      background: var(--accent);
      border: 1px solid rgba(0,0,0,.2);
      border-radius: 12px;
      cursor: pointer;
    }
    .bottombar button:hover{ filter: brightness(1.08); }

    /* CRT theme extra (subtle) */
    body.theme-crt .bg::after{ opacity: .16; }
    body.theme-crt .topbar{ box-shadow: 0 10px 28px rgba(255,214,107,.07), var(--shadow); }

    /* Responsive */
    @media (max-width: 980px){
      .main{ grid-template-columns: 1fr; }
      .controls{ min-width: 0; }
      .brand{ min-width: 0; }
      .status{ display:none; }
    }
  </style>
</head>
<body class="theme-cyber">
  <div class="bg"></div>

  <div class="app">
    <div class="topbar">
      <div class="brand">
        <div class="title">iLoad.lt</div>
        <div class="topic" id="topic">Bendras kanalas</div>
      </div>

      <div class="status">
        <span class="pill">
          <span id="connDot" class="dot"></span>
          <span id="connText">Disconnected</span>
        </span>
        <span class="pill">Komandos: <span style="color:var(--text)">/help</span>, <span style="color:var(--text)">/nick</span>, <span style="color:var(--text)">/roll</span>, <span style="color:var(--text)">/topic</span>, <span style="color:var(--text)">/history</span></span>
      </div>

      <div class="controls">
        <label class="chk" title="Slėpti sistemines žinutes pagrindiniame lange (prisijungė/atsijungė ir pan.)">
          <input id="hideSys" type="checkbox"/>
          <span>Hide sys</span>
        </label>
        <select id="theme">
          <option value="theme-cyber">Cyber</option>
          <option value="theme-glass">Glass</option>
          <option value="theme-matrix">Matrix</option>
          <option value="theme-crt">CRT</option>
        </select>
      </div>
    </div>

    <div class="main">
      <div class="panel">
        <div id="log"></div>
      </div>

      <div class="panel">
        <div class="sidehead">
          <span>Online: <b id="onlineCount">0</b></span>
          <span id="onlineHint">live</span>
        </div>
        <div class="search">
          <input id="userSearch" placeholder="ieškoti vartotojo..." />
        </div>
        <div id="users"></div>
      </div>
    </div>

    <div class="bottombar">
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

  const usersEl = document.getElementById("users");
  const onlineCountEl = document.getElementById("onlineCount");
  const userSearchEl = document.getElementById("userSearch");

  const connDot = document.getElementById("connDot");
  const connText = document.getElementById("connText");

  const hideSysEl = document.getElementById("hideSys");
  const themeEl = document.getElementById("theme");

  let hideSys = false;
  let allUsers = [];

  function esc(s){
    return (s ?? "").toString()
      .replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;")
      .replaceAll('"',"&quot;").replaceAll("'","&#39;");
  }

  function setConn(ok){
    if(ok){
      connDot.classList.add("ok");
      connText.textContent = "Connected";
    }else{
      connDot.classList.remove("ok");
      connText.textContent = "Disconnected";
    }
  }

  function addLineHtml(html){
    const div = document.createElement("div");
    div.className = "line";
    div.innerHTML = html;
    log.appendChild(div);
    log.scrollTop = log.scrollHeight;
  }

  function clearLog(){ log.innerHTML = ""; }

  function applyUserFilter(){
    const q = (userSearchEl.value || "").trim().toLowerCase();
    const items = allUsers.filter(u => (u.nick || "").toLowerCase().includes(q));
    renderUsers(items);
  }

  function renderUsers(items){
    const arr = Array.isArray(items) ? items.slice() : [];
    arr.sort((a,b) => (a.nick||"").localeCompare(b.nick||"", "lt"));

    onlineCountEl.textContent = String(arr.length);
    usersEl.innerHTML = "";

    for(const u of arr){
      const nick = esc(u.nick || "guest");
      const color = esc(u.color || "#a9ff9f");
      const row = document.createElement("div");
      row.className = "user";
      row.innerHTML = `
        <span class="udot" style="background:${color}"></span>
        <span class="uname" style="color:${color}">${nick}</span>
      `;
      usersEl.appendChild(row);
    }
  }

  function wsUrl(){
    const proto = location.protocol === "https:" ? "wss" : "ws";
    return `${proto}://${location.host}/ws`;
  }

  let ws = null;
  let timer = null;

  function connect(){
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;

    addLineHtml(`<span class="sys">[sys]</span> jungiamasi...`);
    setConn(false);

    ws = new WebSocket(wsUrl());

    ws.onopen = () => { setConn(true); addLineHtml(`<span class="sys">[sys]</span> prisijungta.`); };
    ws.onmessage = (ev) => {
      let o = null;
      try { o = JSON.parse(ev.data); } catch { addLineHtml(esc(ev.data)); return; }

      if(o.type === "topic"){
        topicEl.textContent = o.text || "";
        return;
      }

      if(o.type === "users"){
        allUsers = o.items || [];
        applyUserFilter();
        return;
      }

      if(o.type === "ctrl" && o.action === "clear"){
        clearLog();
        addLineHtml(`<span class="sys">[sys]</span> ekranas išvalytas.`);
        return;
      }

      if(o.type === "history"){
        clearLog();
        if(o.topic) topicEl.textContent = o.topic;
        (o.items || []).forEach(renderItem);
        if(!hideSys) addLineHtml(`<span class="sys">[sys]</span> istorija įkelta (${(o.items||[]).length} žinučių).`);
        return;
      }

      renderItem(o);
    };

    ws.onclose = () => {
      setConn(false);
      if(!hideSys) addLineHtml(`<span class="sys">[sys]</span> ryšys nutrūko, reconnect...`);
      if (!timer) timer = setInterval(connect, 1500);
    };
  }

  function renderItem(o){
    const t = esc(o.t || "");
    if(o.type === "sys"){
      if(hideSys) return;
      addLineHtml(`<span class="t">[${t}]</span> <span class="sys">${esc(o.text || "")}</span>`);
      return;
    }
    if(o.type === "msg"){
      const nick = esc(o.nick || "guest");
      const text = esc(o.text || "");
      const color = esc(o.color || "#a9ff9f");
      addLineHtml(`<span class="t">[${t}]</span> <span class="nick" style="color:${color}">${nick}</span>: <span class="msg">${text}</span>`);
      return;
    }
    if(o.type === "me"){
      const nick = esc(o.nick || "guest");
      const text = esc(o.text || "");
      const color = esc(o.color || "#a9ff9f");
      addLineHtml(`<span class="t">[${t}]</span> <span class="me" style="color:${color}">* ${nick} ${text}</span>`);
      return;
    }
    // fallback
    if(!hideSys) addLineHtml(`<span class="t">[${t}]</span> <span class="sys">${esc(o.text || "")}</span>`);
  }

  function send(){
    const nick = (nickEl.value || "").trim();
    const text = msgEl.value.trim();
    if(!text) return;
    if(!ws || ws.readyState !== WebSocket.OPEN){
      if(!hideSys) addLineHtml(`<span class="sys">[sys]</span> dar neprisijungta.`);
      return;
    }
    ws.send(JSON.stringify({nick, text}));
    msgEl.value = "";
  }

  // UI bindings
  btn.onclick = send;
  msgEl.addEventListener("keydown", (e) => { if(e.key === "Enter") send(); });

  userSearchEl.addEventListener("input", applyUserFilter);

  hideSysEl.addEventListener("change", () => {
    hideSys = !!hideSysEl.checked;
    localStorage.setItem("hideSys", hideSys ? "1" : "0");
  });

  themeEl.addEventListener("change", () => {
    document.body.className = themeEl.value;
    localStorage.setItem("theme", themeEl.value);
  });

  // restore preferences
  (function initPrefs(){
    const t = localStorage.getItem("theme");
    if(t){
      document.body.className = t;
      themeEl.value = t;
    }
    const hs = localStorage.getItem("hideSys");
    hideSys = hs === "1";
    hideSysEl.checked = hideSys;
  })();

  connect();
</script>
</body>
</html>
"""


# =========================
# MODELIS
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

def valid_nick(n: str) -> bool:
    return bool(NICK_RE.match(n))

def used_colors() -> Set[str]:
    return {u.color for u in user_state.values()}

def alloc_color(used: Set[str]) -> str:
    # Pirmiausia – paimam iš aiškiai skirtingos paletės
    for c in COLOR_PALETTE:
        if c not in used:
            return c

    # Jei paletė baigėsi – generuojam naują spalvą (HSV per HSL string)
    # Naudojam "golden angle", kad naujos spalvos būtų tolygiai pasiskirsčiusios per ratą.
    golden_angle = 137.50776405

    # Iš used ištraukiam jau naudotų hsl(...) atspalvius, jei tokių buvo
    # (jei nenaudojat hsl, vis tiek veiks – tiesiog bus mažiau apribojimų)
    def parse_hue(s: str):
        if s.startswith("hsl(") and s.endswith(")"):
            try:
                inside = s[4:-1]
                h = float(inside.split(",")[0].strip())
                return h % 360
            except Exception:
                return None
        return None

    used_hues = []
    for c in used:
        h = parse_hue(c)
        if h is not None:
            used_hues.append(h)

    # Generuojam kelis kandidatus ir pasirenkam, kurio hue toliausiai nuo esamų
    best = None
    best_min_dist = -1.0

    base = random.random() * 360.0
    for i in range(1, 40):
        h = (base + i * golden_angle) % 360.0

        # minimalus atstumas iki jau naudotų hue (jei jų turim)
        if used_hues:
            min_dist = min(min(abs(h - uh), 360 - abs(h - uh)) for uh in used_hues)
        else:
            min_dist = 180.0

        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best = h

        # jeigu jau radom tikrai gerą atstumą – užtenka
        if best_min_dist >= 25:
            break

    # S ir L parinkti taip, kad ant juodo fono būtų ryšku ir aišku
    return f"hsl({best:.0f}, 90%, 60%)"

    hue = random.randint(0, 359)
    return f"hsl({hue}, 90%, 65%)"

async def push_history(item: dict):
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

# =========================
# REALTIME ONLINE LIST
# =========================
def build_userlist_items():
    # sortinsim klientuose, bet čia irgi tvarkingai
    items = [{"nick": u.nick, "color": u.color} for u in user_state.values()]
    return items

async def broadcast_userlist():
    await broadcast({"type": "users", "items": build_userlist_items()})

async def send_userlist(ws: WebSocket):
    await send(ws, {"type": "users", "items": build_userlist_items()})

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
    "  /who                - kas online (vardai)\n"
    "  /topic TEKSTAS      - pakeisti temą (visiems)\n"
    "  /topic              - parodyti temą\n"
    "  /history [N]        - atsiųsti paskutines N žinučių (default 50)\n"
    "  /clear              - išvalyti savo ekraną\n"
    "  /me veiksmas        - action žinutė\n"
    "  /roll [NdM]         - kauliukas (pvz. /roll 2d6)\n"
    "  /flip               - monetos metimas\n"
    "  /time               - serverio laikas\n"
    "  /shrug              - prideda ¯\\_(ツ)_/¯\n"
    "  /color              - parodo tavo spalvą\n"
    "  /color new          - priskiria kitą spalvą\n"
)

async def handle_command(ws: WebSocket, text: str) -> bool:
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
            await send(ws, {"type": "sys", "t": ts(),
                            "text": "Netinkamas nick. Leista: raidės/skaičiai/tarpas/_-., iki 24."})
            return True
        old = u.nick
        u.nick = new
        await sysmsg(f"{old} dabar yra {u.nick}.", also_history=True)
        await broadcast_userlist()
        return True

    if text == "/who":
        async with state_lock:
            names = [user_state[c].nick for c in clients if c in user_state]
        await send(ws, {"type": "sys", "t": ts(), "text": "Online: " + ", ".join(sorted(set(names)))})
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
        item = {"type": "sys", "t": ts(), "text": f"{u.nick} meta {n}d{sides}: {rolls} (viso {total})"}
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
            used = used_colors() - {u.color}
            u.color = alloc_color(used)
            item = {"type": "sys", "t": ts(), "text": f"{u.nick} pasikeitė spalvą."}
            await push_history(item)
            await broadcast(item)
            await broadcast_userlist()
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

    # Tema + istorija + online sąrašas
    await send(ws, {"type": "topic", "text": topic})
    await send(ws, {"type": "history", "topic": topic, "items": list(history)})
    await send_userlist(ws)
    await broadcast_userlist()  # kad visi pamatytų naują žmogų

# (nebebrandinam join žinutės, nes yra Online sąrašas)
# await sysmsg(f"{u.nick} prisijungė.", also_history=True)

    try:
        while True:
            data = await ws.receive_json()
            nick_in = str(data.get("nick", "")).strip()[:24]
            text = str(data.get("text", "")).strip()[:300]
            if not text:
                continue

            # jei žmogus įvedė nick ir dar guest* — pritaikom
            if nick_in and valid_nick(nick_in) and u.nick.startswith("guest"):
                old = u.nick
                u.nick = nick_in
                await sysmsg(f"{old} dabar yra {u.nick}.", also_history=True)
                await broadcast_userlist()

            # komandos
            if text.startswith("/"):
                handled = await handle_command(ws, text)
                if handled:
                    continue

            # žinutė
            item = {"type": "msg", "t": ts(), "nick": u.nick, "color": u.color, "text": text}
            await push_history(item)
            await broadcast(item)

    except WebSocketDisconnect:
        pass
    finally:
        left_user = await remove_client(ws)
        if left_user:
    # nebebrandinam leave žinutės, nes yra Online sąrašas
           await broadcast_userlist()
