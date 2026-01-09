import asyncio
import random
import re
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Set, Tuple, Dict, Deque, Any

from zoneinfo import ZoneInfo
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse

app = FastAPI()

# =========================
# TIMEZONE (Vilnius)
# =========================
TZ = ZoneInfo("Europe/Vilnius")


def ts() -> str:
    return datetime.now(TZ).strftime("%H:%M:%S")


# =========================
# KONFIGŪRACIJA
# =========================
HISTORY_LIMIT = 300

COLOR_PALETTE = [
    "#E6194B", "#3CB44B", "#FFE119", "#0082C8", "#F58231",
    "#911EB4", "#46F0F0", "#F032E6", "#D2F53C", "#FABEBE",
    "#008080", "#E6BEFF", "#AA6E28", "#FFD8B1", "#800000",
    "#AFFFc3", "#808000", "#000080", "#808080", "#FFFFFF",
]

NICK_RE = re.compile(r"^[A-Za-z0-9ĄČĘĖĮŠŲŪŽąčęėįšųūž_\-\. ]{2,24}$")
ROLL_RE = re.compile(r"^/roll(?:\s+(\d{1,2})d(\d{1,3}))?$", re.IGNORECASE)

ROOMS = {
    "main": {
        "title": "#main",
        "topic": "Bendras kanalas",
        "history": deque(maxlen=HISTORY_LIMIT),
        "clients": set(),
        "users": {},  # ws -> User
    }
}

# DM istorijos pagal porą (casefold), saugom atskirai nuo #main
DM_HISTORY: Dict[Tuple[str, str], Deque[dict]] = {}

state_lock = asyncio.Lock()

# =========================
# HTML (DM tabs + unread)
# =========================
HTML = r"""<!doctype html>
<html lang="lt">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>iLoad.lt Chat</title>
  <style>
    :root{
      --bg:#06080a;
      --panel:rgba(6,10,10,.78);
      --panel2:rgba(5,7,8,.70);
      --border:rgba(124,255,107,.18);
      --text:#caffd9;
      --muted:rgba(202,255,217,.55);
      --accent:#7cff6b;
      --accent2:#6be4ff;
      --danger:#ff6b6b;
      --shadow:0 10px 30px rgba(0,0,0,.45);
      --radius:16px;
      --mono:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Courier New",monospace;
    }

    /* THEMES */
    body.theme-cyber{
      --bg:#06080a; --panel:rgba(6,10,10,.78); --panel2:rgba(5,7,8,.70);
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

    html,body{height:100%;}
    body{
      margin:0; background:var(--bg); color:var(--text);
      font-family:var(--mono); overflow:hidden;
    }

    .bg{ position:fixed; inset:0; z-index:0; pointer-events:none; }
    .bg::before{
      content:""; position:absolute; inset:-2px; opacity:.22;
      background:
        linear-gradient(to right, rgba(255,255,255,.04) 1px, transparent 1px),
        linear-gradient(to bottom, rgba(255,255,255,.03) 1px, transparent 1px);
      background-size: 46px 46px;
      mask-image: radial-gradient(circle at 40% 10%, rgba(0,0,0,1) 0%, rgba(0,0,0,.7) 40%, rgba(0,0,0,0) 75%);
    }
    .bg::after{
      content:""; position:absolute; inset:0; opacity:.12;
      background: repeating-linear-gradient(
        to bottom,
        rgba(0,0,0,0) 0px,
        rgba(0,0,0,0) 2px,
        rgba(0,0,0,.55) 3px
      );
      animation: scan 8s linear infinite;
    }
    @keyframes scan{ 0%{transform:translateY(0);} 100%{transform:translateY(10px);} }

    .app{
      position:relative; z-index:1;
      height:100%;
      display:grid;
      grid-template-rows:auto 1fr auto;
      gap:12px;
      padding:14px;
      box-sizing:border-box;
    }

    .topbar{
      display:flex; align-items:center; justify-content:space-between; gap:12px;
      padding:12px 14px;
      border:1px solid var(--border);
      border-radius:var(--radius);
      background:linear-gradient(180deg, var(--panel), rgba(0,0,0,0));
      box-shadow:var(--shadow);
      backdrop-filter: blur(10px);
    }
    .brand{ display:flex; gap:10px; align-items:baseline; }
    .brand .title{ font-weight:900; color:var(--accent); }
    .brand .topic{ font-weight:800; color:var(--text); }

    .status{
      color:var(--muted); font-size:13px;
      display:flex; align-items:center; gap:10px;
      justify-content:center; flex:1;
      text-align:center;
    }
    .pill{
      border:1px solid var(--border);
      background:rgba(0,0,0,.18);
      padding:6px 10px;
      border-radius:999px;
      display:inline-flex; align-items:center; gap:8px;
      white-space:nowrap;
    }
    .dot{ width:10px; height:10px; border-radius:999px; background:var(--danger);
      box-shadow:0 0 14px rgba(255,107,107,.18); }
    .dot.ok{ background:var(--accent); box-shadow:0 0 14px rgba(124,255,107,.18); }

    .main{
      display:grid;
      grid-template-columns: 1fr 320px;
      gap:12px;
      min-height:0;
    }
    .panel{
      border:1px solid var(--border);
      border-radius:var(--radius);
      background:var(--panel2);
      box-shadow:var(--shadow);
      backdrop-filter: blur(10px);
      overflow:hidden;
      min-height:0;
    }
    #log{
      padding:14px 16px;
      overflow:auto;
      white-space:pre-wrap;
      line-height:1.45;
      min-height:0;
    }
    .line{ margin:2px 0; }
    .t{ color: rgba(124,255,107,.40); }
    .sys{ color: var(--muted); }
    .msg{ color: var(--text); }
    .nick{ font-weight:900; }
    .me{ color: var(--accent2); }
    .dmTag{ color: var(--accent2); font-weight:900; }

    .rightWrap{
      display:grid;
      grid-template-rows:auto auto 1fr auto auto 1fr;
      min-height:0;
    }
    .sidehead{
      padding:12px 12px;
      border-bottom:1px solid var(--border);
      display:flex; justify-content:space-between; align-items:center;
      color:var(--muted); font-size:13px;
    }
    .sidehead b{ color:var(--text); }
    .search{
      padding:10px 12px;
      border-bottom:1px solid var(--border);
    }
    .search input{
      width:100%; padding:10px 10px;
      font-family:var(--mono);
      background:rgba(0,0,0,.22);
      border:1px solid var(--border);
      border-radius:12px;
      color:var(--text);
      outline:none;
      box-sizing:border-box;
    }

    .sectionTitle{
      padding:10px 12px;
      border-bottom:1px solid var(--border);
      color:var(--muted);
      font-size:12px;
      letter-spacing:.2px;
    }

    #convos{
      padding:10px 10px 12px 10px;
      overflow:auto;
      min-height:0;
      border-bottom:1px solid var(--border);
    }
    .convo{
      display:flex; align-items:center; justify-content:space-between; gap:10px;
      padding:9px 10px;
      border-radius:14px;
      border:1px solid rgba(255,255,255,0.03);
      background:rgba(0,0,0,0.12);
      margin-bottom:8px;
      cursor:pointer;
    }
    .convo:hover{ filter:brightness(1.08); }
    .convo.active{
      border-color: rgba(124,255,107,.28);
      background: rgba(124,255,107,.06);
    }
    .cname{ font-weight:900; font-size:13px; }
    .badge{
      min-width:22px;
      padding:2px 8px;
      border-radius:999px;
      font-size:12px;
      font-weight:900;
      color:var(--bg);
      background:var(--accent2);
      display:none;
      align-items:center;
      justify-content:center;
    }

    #users{
      padding:10px 10px 12px 10px;
      overflow:auto;
      min-height:0;
    }
    .user{
      display:flex; align-items:center; gap:10px;
      padding:8px 10px;
      border-radius:14px;
      border:1px solid rgba(255,255,255,0.03);
      background:rgba(0,0,0,0.12);
      margin-bottom:8px;
      cursor:pointer;
    }
    .user:hover{ filter:brightness(1.08); }
    .udot{ width:10px; height:10px; border-radius:999px; box-shadow:0 0 14px rgba(0,0,0,.25); }
    .uname{ font-weight:900; font-size:13px; word-break:break-word; }

    .bottombar{
      display:grid;
      grid-template-columns: 1fr 130px;
      gap:12px;
      padding:12px 12px;
      border:1px solid var(--border);
      border-radius:var(--radius);
      background:var(--panel);
      box-shadow:var(--shadow);
      backdrop-filter: blur(10px);
    }
    .bottombar input{
      width:100%; padding:12px 12px;
      font-family:var(--mono);
      background:rgba(0,0,0,.22);
      border:1px solid var(--border);
      border-radius:12px;
      color:var(--text);
      outline:none;
      box-sizing:border-box;
    }
    .bottombar button{
      padding:12px 12px;
      font-family:var(--mono);
      font-weight:900;
      color:var(--bg);
      background:var(--accent);
      border:1px solid rgba(0,0,0,.2);
      border-radius:12px;
      cursor:pointer;
    }
    .bottombar button:hover{ filter:brightness(1.08); }
    .bottombar button:disabled{ opacity:.55; cursor:not-allowed; }

    /* Lobby */
    #lobby{
      position:fixed; inset:0; z-index:10;
      display:flex; align-items:center; justify-content:center;
      background:rgba(0,0,0,.62);
      padding:18px;
      box-sizing:border-box;
    }
    .card{
      width:min(860px, 96vw);
      border:1px solid var(--border);
      border-radius:18px;
      background:rgba(6,10,10,.92);
      box-shadow:0 24px 70px rgba(0,0,0,.55);
      backdrop-filter: blur(12px);
      overflow:hidden;
    }
    .card-head{
      padding:16px 18px;
      border-bottom:1px solid var(--border);
      display:flex; align-items:baseline; justify-content:space-between; gap:12px;
    }
    .card-head .h1{ font-weight:1000; color:var(--accent); letter-spacing:.2px; }
    .card-head .sub{ color:var(--muted); font-size:13px; }
    .card-body{
      display:grid;
      grid-template-columns: 320px 1fr;
      gap:14px;
      padding:16px 18px;
    }
    .box{
      border:1px solid var(--border);
      border-radius:16px;
      background:rgba(0,0,0,.22);
      padding:12px 12px;
    }
    .label{ color:var(--muted); font-size:12px; margin-bottom:8px; }
    .nickrow{ display:grid; grid-template-columns: 1fr; gap:10px; }
    .nickrow input{
      width:100%; padding:12px 12px;
      font-family:var(--mono);
      border-radius:12px;
      border:1px solid var(--border);
      background:rgba(0,0,0,.22);
      color:var(--text);
      outline:none;
      box-sizing:border-box;
    }
    .help{
      color:var(--muted);
      font-size:12px;
      line-height:1.35;
    }
    .channels{
      display:grid; gap:10px;
    }
    .chan{
      border:1px solid rgba(255,255,255,.06);
      background:rgba(0,0,0,.18);
      border-radius:16px;
      padding:12px 12px;
      display:flex; align-items:center; justify-content:space-between; gap:12px;
    }
    .chan .name{ font-weight:900; color:var(--text); }
    .chan .desc{ color:var(--muted); font-size:12px; margin-top:2px; }
    .joinbtn{
      padding:10px 12px;
      border-radius:12px;
      border:1px solid rgba(0,0,0,.2);
      background:var(--accent);
      color:var(--bg);
      font-weight:900;
      font-family:var(--mono);
      cursor:pointer;
      white-space:nowrap;
    }
    .joinbtn:disabled{ opacity:.55; cursor:not-allowed; }

    #nickErr{
      margin-top:8px;
      color:var(--danger);
      font-size:12px;
      display:none;
    }
    #nickState{
      margin-top:8px;
      color:var(--muted);
      font-size:12px;
      display:none;
    }

    #nickSuggestBox{
      margin-top:10px;
      display:none;
      border:1px solid var(--border);
      background:rgba(0,0,0,.18);
      padding:10px;
      border-radius:12px;
    }
    #nickSuggestBox b{ color:var(--text); }
    #applySuggest{
      margin-left:10px;
      padding:8px 10px;
      border-radius:10px;
      border:1px solid rgba(0,0,0,.2);
      background:var(--accent2);
      color:var(--bg);
      font-weight:900;
      font-family:var(--mono);
      cursor:pointer;
    }

    select.themeSel{
      padding:10px 10px;
      border-radius:12px;
      border:1px solid var(--border);
      background:rgba(0,0,0,.22);
      color:var(--text);
      font-family:var(--mono);
      outline:none;
      box-sizing:border-box;
    }
    .themeSel.wide{ width:100%; margin-top:8px; padding:12px; }

    @media (max-width: 900px){
      .main{ grid-template-columns: 1fr; }
      .status{ display:none; }
      .card-body{ grid-template-columns: 1fr; }
      .rightWrap{ grid-template-rows:auto auto 1fr auto auto 1fr; }
    }
  </style>
</head>
<body class="theme-cyber">
  <div class="bg"></div>

  <!-- Lobby -->
  <div id="lobby">
    <div class="card">
      <div class="card-head">
        <div>
          <div class="h1">iLoad.lt Chat</div>
          <div class="sub">Pirmiausia įvesk slapyvardį, tada prisijunk prie kanalo.</div>
        </div>
        <div class="sub">Klasės projektas</div>
      </div>

      <div class="card-body">
        <div class="box">
          <div class="label">Slapyvardis (2–24 simboliai)</div>
          <div class="nickrow">
            <input id="nickPick" placeholder="pvz. Tomas" maxlength="24"/>
          </div>
          <div id="nickErr"></div>
          <div id="nickState"></div>

          <div id="nickSuggestBox">
            Siūlomas nick: <b id="nickSuggestVal"></b>
            <button id="applySuggest">Pritaikyti</button>
          </div>

          <div class="help" style="margin-top:10px;">
            Leidžiami simboliai: raidės, skaičiai, tarpas, _ - .<br/>
            Nick privalo būti unikalus.
          </div>

          <div class="help" style="margin-top:12px;">
            Patarimas: nick ir tema išsaugomi naršyklėje.
          </div>

          <div class="help" style="margin-top:12px;">
            Tema:
          </div>
          <select id="themePick" class="themeSel wide">
            <option value="theme-cyber">Cyber</option>
            <option value="theme-glass">Glass</option>
            <option value="theme-matrix">Matrix</option>
            <option value="theme-crt">CRT</option>
          </select>
        </div>

        <div class="box">
          <div class="label">Kanalai</div>
          <div class="channels">
            <div class="chan">
              <div>
                <div class="name">#main</div>
                <div class="desc">Bendras kanalas (kol kas vienintelis).</div>
              </div>
              <button id="joinMain" class="joinbtn" disabled>Prisijungti</button>
            </div>
          </div>

          <div class="help" style="margin-top:10px;">
            Komandos: <span style="color:var(--text)">/help</span>, <span style="color:var(--text)">/dm</span>, <span style="color:var(--text)">/roll</span>, <span style="color:var(--text)">/me</span>, <span style="color:var(--text)">/topic</span>, <span style="color:var(--text)">/history</span>.
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- App -->
  <div class="app" id="app" style="display:none;">
    <div class="topbar">
      <div class="brand">
        <div class="title">iLoad.lt</div>
        <div class="topic" id="topic">#main</div>
      </div>

      <div class="status">
        <span class="pill">
          <span id="connDot" class="dot"></span>
          <span id="connText">Disconnected</span>
        </span>
        <span class="pill">Komandos: <span style="color:var(--text)">/help</span> <span style="color:var(--text)">/dm</span> <span style="color:var(--text)">/roll</span> <span style="color:var(--text)">/me</span> <span style="color:var(--text)">/topic</span> <span style="color:var(--text)">/history</span></span>
      </div>

      <div style="display:flex; gap:10px; align-items:center;">
        <select id="themeTop" class="themeSel" title="Keisti temą">
          <option value="theme-cyber">Cyber</option>
          <option value="theme-glass">Glass</option>
          <option value="theme-matrix">Matrix</option>
          <option value="theme-crt">CRT</option>
        </select>
        <div class="pill" id="meNickPill" title="Tavo nick" style="min-width:160px; justify-content:center;"></div>
      </div>
    </div>

    <div class="main">
      <div class="panel">
        <div id="log"></div>
      </div>

      <div class="panel rightWrap">
        <div class="sidehead">
          <span>Pokalbiai</span>
          <span>tabs</span>
        </div>
        <div id="convos"></div>

        <div class="sidehead">
          <span>Online: <b id="onlineCount">0</b></span>
          <span>live</span>
        </div>
        <div class="search">
          <input id="userSearch" placeholder="ieškoti vartotojo..." />
        </div>
        <div id="users"></div>
      </div>
    </div>

    <div class="bottombar">
      <input id="msg" placeholder="rašyk žinutę ir Enter..." maxlength="300"/>
      <button id="btn">Siųsti</button>
    </div>
  </div>

<script>
  const lobby = document.getElementById("lobby");
  const appEl = document.getElementById("app");

  const nickPick = document.getElementById("nickPick");
  const nickErr  = document.getElementById("nickErr");
  const nickState = document.getElementById("nickState");
  const joinMain = document.getElementById("joinMain");

  const nickSuggestBox = document.getElementById("nickSuggestBox");
  const nickSuggestVal = document.getElementById("nickSuggestVal");
  const applySuggest   = document.getElementById("applySuggest");

  const themePick = document.getElementById("themePick");
  const themeTop  = document.getElementById("themeTop");

  const log = document.getElementById("log");
  const msgEl = document.getElementById("msg");
  const btn = document.getElementById("btn");
  const topicEl = document.getElementById("topic");
  const meNickPill = document.getElementById("meNickPill");

  const convosEl = document.getElementById("convos");

  const usersEl = document.getElementById("users");
  const onlineCountEl = document.getElementById("onlineCount");
  const userSearchEl = document.getElementById("userSearch");

  const connDot = document.getElementById("connDot");
  const connText = document.getElementById("connText");

  let ws = null;
  let reconnectTimer = null;

  let room = "main";
  let nick = "";
  let myNick = ""; // iš serverio (canonical)

  // serverio #main topic
  let mainTopicText = "#main";

  // Pokalbių modelis (tabs)
  // convKey: "main" arba "dm:<Nick>"
  const convs = new Map(); // key -> { key, title, kind, peerNick, items:[], unread:int, loaded:bool }
  let activeKey = "main";

  // Online users
  let allUsers = [];

  let nickAvailable = false;
  let checkTimer = null;

  let joinEstablished = false;
  let fatalJoinError = false;

  function esc(s){
    return (s ?? "").toString()
      .replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;")
      .replaceAll('"',"&quot;").replaceAll("'","&#39;");
  }

  function setConn(ok){
    if(ok){ connDot.classList.add("ok"); connText.textContent = "Connected"; }
    else { connDot.classList.remove("ok"); connText.textContent = "Disconnected"; }
  }

  function addLineHtml(html){
    const div = document.createElement("div");
    div.className = "line";
    div.innerHTML = html;
    log.appendChild(div);
    log.scrollTop = log.scrollHeight;
  }

  function clearLog(){ log.innerHTML = ""; }

  function validateNick(n){
    return /^[A-Za-z0-9ĄČĘĖĮŠŲŪŽąčęėįšųūž_\-\. ]{2,24}$/.test(n);
  }

  function setNickError(text){
    if(!text){
      nickErr.style.display = "none";
      nickErr.textContent = "";
    }else{
      nickErr.style.display = "block";
      nickErr.textContent = text;
    }
  }
  function setNickState(text){
    if(!text){
      nickState.style.display = "none";
      nickState.textContent = "";
    }else{
      nickState.style.display = "block";
      nickState.textContent = text;
    }
  }
  function hideSuggest(){
    nickSuggestBox.style.display = "none";
    nickSuggestVal.textContent = "";
  }
  function showSuggest(s){
    nickSuggestVal.textContent = s;
    nickSuggestBox.style.display = "block";
  }

  // THEME
  function setTheme(cls){
    document.body.className = cls;
    localStorage.setItem("theme", cls);
    if(themePick) themePick.value = cls;
    if(themeTop) themeTop.value = cls;
  }
  if(themePick) themePick.addEventListener("change", () => setTheme(themePick.value));
  if(themeTop) themeTop.addEventListener("change", () => setTheme(themeTop.value));

  function wsUrl(){
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const qs = new URLSearchParams({ room, nick }).toString();
    return `${proto}://${location.host}/ws?${qs}`;
  }

  function ensureMainConvo(){
    if(!convs.has("main")){
      convs.set("main", { key:"main", title:"#main", kind:"main", peerNick:null, items:[], unread:0, loaded:true });
    }
  }

  function ensureDmConvo(peerNick){
    const key = `dm:${peerNick}`;
    if(!convs.has(key)){
      convs.set(key, { key, title:`DM: ${peerNick}`, kind:"dm", peerNick, items:[], unread:0, loaded:false });
    }
    return key;
  }

  function renderConvos(){
    convosEl.innerHTML = "";
    const arr = Array.from(convs.values());
    // main pirma, tada DM pagal title
    arr.sort((a,b) => {
      if(a.key === "main") return -1;
      if(b.key === "main") return 1;
      return (a.title||"").localeCompare(b.title||"", "lt");
    });

    for(const c of arr){
      const row = document.createElement("div");
      row.className = "convo" + (c.key === activeKey ? " active" : "");
      const unread = c.unread || 0;

      row.innerHTML = `
        <div class="cname">${esc(c.title)}</div>
        <div class="badge" style="${unread>0 ? 'display:inline-flex;' : ''}">${unread}</div>
      `;

      row.addEventListener("click", () => switchConvo(c.key));
      convosEl.appendChild(row);
    }
  }

  function setTopicForActive(){
    const c = convs.get(activeKey);
    if(!c) return;

    if(c.kind === "main"){
      topicEl.textContent = mainTopicText || "#main";
      msgEl.placeholder = "rašyk žinutę į #main ir Enter...";
    }else{
      topicEl.textContent = `DM su ${c.peerNick}`;
      msgEl.placeholder = `rašyk žinutę vartotojui ${c.peerNick} ir Enter...`;
    }
  }

  function renderActiveLog(){
    clearLog();
    const c = convs.get(activeKey);
    if(!c) return;
    for(const item of c.items){
      addLineHtml(item.__html);
    }
  }

  function switchConvo(key){
    if(!convs.has(key)) return;
    activeKey = key;
    const c = convs.get(key);
    if(c){
      c.unread = 0;
    }

    renderConvos();
    setTopicForActive();

    // jei DM dar neprašėm istorijos – prašom dabar
    if(c && c.kind === "dm" && !c.loaded){
      c.loaded = true; // užfiksuojam, kad jau užklausėm
      // įdedam "loading" žinutę lokaliai (tik jei dar nieko nėra)
      if(c.items.length === 0){
        pushToConvo(key, {type:"sys", t:"", text:"Kraunama DM istorija..."}, true);
      }
      requestDmHistory(c.peerNick);
    }

    renderActiveLog();
    msgEl.focus();
  }

  function stopReconnect(){
    if(reconnectTimer){
      clearInterval(reconnectTimer);
      reconnectTimer = null;
    }
  }

  function formatHtmlFromServerObj(o){
    const t = esc(o.t || "");
    if(o.type === "msg"){
      const nn = esc(o.nick || "???");
      const cc = esc(o.color || "#caffd9");
      const tx = esc(o.text || "");
      return `<span class="t">[${t}]</span> <span class="nick" style="color:${cc}">${nn}</span>: <span class="msg">${tx}</span>`;
    }
    if(o.type === "dm"){
      const f = esc(o.from || "???");
      const to = esc(o.to || "???");
      const cc = esc(o.color || "#caffd9");
      const tx = esc(o.text || "");
      return `<span class="t">[${t}]</span> <span class="dmTag">[DM]</span> <span class="nick" style="color:${cc}">${f}</span> → <span class="nick">${to}</span>: <span class="msg">${tx}</span>`;
    }
    if(o.type === "me_action"){
      const nn = esc(o.nick || "???");
      const cc = esc(o.color || "#caffd9");
      const tx = esc(o.text || "");
      return `<span class="t">[${t}]</span> <span class="me" style="color:${cc}">* ${nn} ${tx}</span>`;
    }
    const tx = esc(o.text || "");
    return `<span class="t">[${t}]</span> <span class="sys">${tx}</span>`;
  }

  function pushToConvo(key, o, doNotUnread=false){
    ensureMainConvo();

    const c = convs.get(key);
    if(!c) return;

    const html = formatHtmlFromServerObj(o);
    const item = { ...o, __html: html };
    c.items.push(item);

    // limit client-side memory
    if(c.items.length > 600){
      c.items.splice(0, c.items.length - 600);
    }

    // unread
    if(!doNotUnread && key !== activeKey){
      c.unread = (c.unread || 0) + 1;
      renderConvos();
    }

    // jei aktyvus - rodom iškart
    if(key === activeKey){
      addLineHtml(html);
    }
  }

  async function suggestNick(base){
    try{
      const qs = new URLSearchParams({ room: "main", base }).toString();
      const r = await fetch(`/suggest_nick?${qs}`, { cache: "no-store" });
      const j = await r.json();
      if(j && j.ok && j.suggestion){
        return String(j.suggestion);
      }
    }catch{}
    return "";
  }

  async function checkNickAvailabilityNow(){
    hideSuggest();
    const n = (nickPick.value || "").trim();
    nickAvailable = false;
    joinMain.disabled = true;

    if(!validateNick(n)){
      setNickState("");
      setNickError("Netinkamas nick. Reikia 2–24 simbolių (raidės/skaičiai/tarpas/_-.)");
      return;
    }

    setNickError("");
    setNickState("Tikrinama ar nick laisvas...");

    try{
      const qs = new URLSearchParams({ room: "main", nick: n }).toString();
      const r = await fetch(`/check_nick?${qs}`, { cache: "no-store" });
      const j = await r.json();

      if(j && j.ok){
        nickAvailable = true;
        setNickState("Nick laisvas. Galite prisijungti.");
        setNickError("");
        joinMain.disabled = false;
      }else{
        nickAvailable = false;
        setNickState("");
        setNickError((j && j.reason) ? j.reason : "Nick užimtas arba neteisingas.");
        joinMain.disabled = true;

        const sug = await suggestNick(n);
        if(sug && sug.toLowerCase() !== n.toLowerCase()){
          showSuggest(sug);
        }
      }
    }catch{
      nickAvailable = false;
      setNickState("");
      setNickError("Nepavyko patikrinti nick (serveris nepasiekiamas).");
      joinMain.disabled = true;
    }
  }

  function scheduleNickCheck(){
    if(checkTimer) clearTimeout(checkTimer);
    checkTimer = setTimeout(checkNickAvailabilityNow, 250);
  }

  nickPick.addEventListener("input", scheduleNickCheck);
  nickPick.addEventListener("keydown", (e) => {
    if(e.key === "Enter" && !joinMain.disabled) joinMain.click();
  });

  applySuggest.addEventListener("click", () => {
    const s = (nickSuggestVal.textContent || "").trim();
    if(!s) return;
    nickPick.value = s;
    hideSuggest();
    scheduleNickCheck();
    nickPick.focus();
  });

  function showLobby(show){
    lobby.style.display = show ? "flex" : "none";
    appEl.style.display = show ? "none" : "grid";
    if(show) setTimeout(() => nickPick.focus(), 40);
    else setTimeout(() => msgEl.focus(), 40);
  }

  function renderUsers(items){
    const arr = Array.isArray(items) ? items.slice() : [];
    arr.sort((a,b) => (a.nick||"").localeCompare(b.nick||"", "lt"));

    onlineCountEl.textContent = String(arr.length);
    usersEl.innerHTML = "";

    for(const u of arr){
      const nn = esc(u.nick || "???");
      const cc = esc(u.color || "#caffd9");
      const row = document.createElement("div");
      row.className = "user";
      row.innerHTML = `<span class="udot" style="background:${cc}"></span><span class="uname" style="color:${cc}">${nn}</span>`;
      row.addEventListener("click", () => {
        // neatidarom DM su savimi
        if(myNick && (u.nick || "").toLowerCase() === myNick.toLowerCase()){
          return;
        }
        const key = ensureDmConvo(u.nick);
        renderConvos();
        switchConvo(key);
      });
      usersEl.appendChild(row);
    }
  }

  function applyUserFilter(){
    const q = (userSearchEl.value || "").trim().toLowerCase();
    const items = allUsers.filter(u => (u.nick || "").toLowerCase().includes(q));
    renderUsers(items);
  }
  userSearchEl.addEventListener("input", applyUserFilter);

  function requestDmHistory(peerNick){
    if(!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ type: "dm_history_req", with: peerNick, limit: 120 }));
  }

  function connect(){
    joinEstablished = false;
    fatalJoinError = false;

    if(ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;

    setConn(false);
    ensureMainConvo();
    renderConvos();
    switchConvo("main");

    pushToConvo("main", {type:"sys", t: "", text:"jungiamasi..."}, true);

    ws = new WebSocket(wsUrl());

    ws.onopen = () => {
      setConn(true);
      pushToConvo("main", {type:"sys", t: tsLocalFallback(), text:"prisijungta."}, true);
      stopReconnect();
    };

    ws.onmessage = (ev) => {
      let o = null;
      try { o = JSON.parse(ev.data); } catch { return; }

      if(o.type === "error"){
        fatalJoinError = true;
        stopReconnect();
        try{ ws.close(); }catch{}

        showLobby(true);
        setNickState("");
        setNickError(o.text || "Prisijungti nepavyko.");
        joinMain.disabled = true;

        scheduleNickCheck();
        return;
      }

      if(o.type === "topic" || o.type === "history" || o.type === "users" || o.type === "me"){
        joinEstablished = true;
      }

      if(o.type === "topic"){
        mainTopicText = o.text || "#main";
        if(activeKey === "main") setTopicForActive();
        return;
      }

      if(o.type === "users"){
        allUsers = o.items || [];
        applyUserFilter();
        return;
      }

      if(o.type === "history"){
        // tai #main history
        const c = convs.get("main");
        if(c){
          c.items = [];
        }
        for(const it of (o.items || [])){
          pushToConvo("main", it, true);
        }
        // jei aktyvus main — per-render iš atminties (kad tvarkingai)
        if(activeKey === "main"){
          renderActiveLog();
        }
        return;
      }

      if(o.type === "me"){
        myNick = o.nick || "";
        const cc = esc(o.color || "#caffd9");
        const nn = esc(o.nick || "");
        meNickPill.innerHTML = `<b style="color:${cc}">${nn}</b>`;
        return;
      }

      if(o.type === "dm_history"){
        const peer = o.with || "";
        const key = ensureDmConvo(peer);
        const c = convs.get(key);
        if(c){
          // išvalom "loading" ir supildom istoriją
          c.items = [];
          for(const it of (o.items || [])){
            pushToConvo(key, it, true);
          }
          if(activeKey === key){
            renderActiveLog();
          }
        }
        renderConvos();
        return;
      }

      // DM routing
      if(o.type === "dm"){
        // peer = kitas žmogus (ne aš)
        const peer = (myNick && (o.from || "").toLowerCase() === myNick.toLowerCase()) ? (o.to || "") : (o.from || "");
        const key = ensureDmConvo(peer);
        renderConvos();
        pushToConvo(key, o, false);
        return;
      }

      // normal msg -> main
      pushToConvo("main", o, false);
    };

    ws.onclose = () => {
      setConn(false);

      if(!joinEstablished){
        stopReconnect();
        showLobby(true);
        if(!fatalJoinError){
          setNickError("Prisijungti nepavyko. Patikrink ar nick laisvas ir bandyk dar kartą.");
          scheduleNickCheck();
        }
        return;
      }

      if(fatalJoinError) return;

      pushToConvo("main", {type:"sys", t: tsLocalFallback(), text:"ryšys nutrūko, reconnect..."}, false);
      if(!reconnectTimer) reconnectTimer = setInterval(connect, 1500);
    };
  }

  function tsLocalFallback(){
    // tik tam, kad "prisijungta" žinutė turėtų laiką, jei serveris dar neatsiuntė nieko
    const d = new Date();
    const hh = String(d.getHours()).padStart(2,'0');
    const mm = String(d.getMinutes()).padStart(2,'0');
    const ss = String(d.getSeconds()).padStart(2,'0');
    return `${hh}:${mm}:${ss}`;
  }

  function send(){
    const text = (msgEl.value || "").trim();
    if(!text) return;

    if(!ws || ws.readyState !== WebSocket.OPEN){
      pushToConvo(activeKey, {type:"sys", t: tsLocalFallback(), text:"nėra ryšio su serveriu."}, false);
      return;
    }

    const c = convs.get(activeKey);
    if(!c){
      return;
    }

    // jei vartotojas rašo komandą - siunčiam kaip yra (komandos veikia visur)
    if(text.startsWith("/")){
      ws.send(JSON.stringify({ text }));
      msgEl.value = "";
      return;
    }

    // main
    if(c.kind === "main"){
      ws.send(JSON.stringify({ text }));
      msgEl.value = "";
      return;
    }

    // dm tab: siunčiam kaip /dm <nick> <tekstas>
    if(c.kind === "dm"){
      const peer = c.peerNick;
      ws.send(JSON.stringify({ text: `/dm ${peer} ${text}` }));
      msgEl.value = "";
      return;
    }
  }

  btn.onclick = send;
  msgEl.addEventListener("keydown", (e) => { if(e.key === "Enter") send(); });

  joinMain.onclick = async () => {
    await checkNickAvailabilityNow();
    if(!nickAvailable) return;

    const n = (nickPick.value || "").trim();
    nick = n;
    localStorage.setItem("nick", nick);
    room = "main";

    // init conversations
    convs.clear();
    ensureMainConvo();
    activeKey = "main";
    renderConvos();
    setTopicForActive();
    renderActiveLog();

    showLobby(false);
    connect();
  };

  (function init(){
    const savedNick = (localStorage.getItem("nick") || "").trim();
    if(savedNick) nickPick.value = savedNick;

    const savedTheme = (localStorage.getItem("theme") || "theme-cyber").trim();
    setTheme(savedTheme);

    scheduleNickCheck();
    showLobby(true);
  })();
</script>
</body>
</html>
"""

# =========================
# MODELIAI / PAGALBINĖS
# =========================
@dataclass
class User:
    nick: str
    color: str
    joined_at: float


def valid_nick(n: str) -> bool:
    return bool(NICK_RE.match(n))


def alloc_color(used: Set[str]) -> str:
    for c in COLOR_PALETTE:
        if c not in used:
            return c
    hue = random.randint(0, 359)
    return f"hsl({hue}, 90%, 60%)"


def room_used_colors(room_key: str) -> Set[str]:
    return {u.color for u in ROOMS[room_key]["users"].values()}


def is_nick_taken(room_key: str, new_nick: str) -> bool:
    k = new_nick.casefold()
    for u in ROOMS[room_key]["users"].values():
        if u.nick.casefold() == k:
            return True
    return False


def find_user_ws_by_nick(room_key: str, nick: str) -> Optional[WebSocket]:
    key = nick.casefold()
    for w, u in ROOMS[room_key]["users"].items():
        if u.nick.casefold() == key:
            return w
    return None


def dm_key(a: str, b: str) -> Tuple[str, str]:
    x, y = a.casefold(), b.casefold()
    return (x, y) if x <= y else (y, x)


def dm_history_for(a: str, b: str) -> Deque[dict]:
    k = dm_key(a, b)
    if k not in DM_HISTORY:
        DM_HISTORY[k] = deque(maxlen=HISTORY_LIMIT)
    return DM_HISTORY[k]


def dm_history_items(a: str, b: str, limit: int = 120) -> list[dict]:
    hist = dm_history_for(a, b)
    if limit <= 0:
        limit = 120
    limit = min(limit, HISTORY_LIMIT)
    return list(hist)[-limit:]


async def room_broadcast(room_key: str, obj: dict, exclude: Optional[WebSocket] = None):
    dead = []
    for ws in list(ROOMS[room_key]["clients"]):
        if ws is exclude:
            continue
        try:
            await ws.send_json(obj)
        except Exception:
            dead.append(ws)
    for ws in dead:
        await room_remove_client(room_key, ws)


async def room_send(ws: WebSocket, obj: dict):
    await ws.send_json(obj)


async def room_push_history(room_key: str, obj: dict):
    ROOMS[room_key]["history"].append(obj)


def room_userlist(room_key: str):
    return [{"nick": u.nick, "color": u.color} for u in ROOMS[room_key]["users"].values()]


async def room_broadcast_userlist(room_key: str):
    await room_broadcast(room_key, {"type": "users", "items": room_userlist(room_key)})


async def room_remove_client(room_key: str, ws: WebSocket):
    async with state_lock:
        ROOMS[room_key]["clients"].discard(ws)
        ROOMS[room_key]["users"].pop(ws, None)
    try:
        await ws.close()
    except Exception:
        pass


# =========================
# ROUTES
# =========================
@app.get("/", response_class=HTMLResponse)
def home():
    return HTML


@app.get("/health", response_class=PlainTextResponse)
def health():
    return f"OK {ts()}"


@app.get("/check_nick")
async def check_nick(room: str = "main", nick: str = ""):
    room_key = (room or "").strip()
    n = (nick or "").strip()

    if room_key not in ROOMS:
        return JSONResponse({"ok": False, "reason": "Neteisingas kanalas."})

    if not valid_nick(n):
        return JSONResponse({"ok": False, "reason": "Netinkamas nick (2–24, raidės/skaičiai/tarpas/_-.)."})

    async with state_lock:
        if is_nick_taken(room_key, n):
            return JSONResponse({"ok": False, "reason": "Nick užimtas. Pasirink kitą."})

    return JSONResponse({"ok": True})


def _fit_candidate(stem: str, suffix: str) -> str:
    max_len = 24
    stem = stem.strip()
    if len(stem) < 2:
        stem = "User"
    allow = max_len - len(suffix)
    if allow < 2:
        suffix = suffix[: max_len - 2]
        allow = max_len - len(suffix)
    stem = stem[:allow]
    return f"{stem}{suffix}"


@app.get("/suggest_nick")
async def suggest_nick(room: str = "main", base: str = ""):
    room_key = (room or "").strip()
    b = (base or "").strip()

    if room_key not in ROOMS:
        return JSONResponse({"ok": False, "reason": "Neteisingas kanalas."})

    if not b:
        return JSONResponse({"ok": True, "suggestion": "User" + str(random.randint(100, 999))})

    if not valid_nick(b):
        b = "User"

    stem = re.sub(r"\s*\d+$", "", b).rstrip()
    if len(stem) < 2:
        stem = b[:24]

    async with state_lock:
        if not is_nick_taken(room_key, b):
            return JSONResponse({"ok": True, "suggestion": b})

        for i in range(2, 10000):
            cand = _fit_candidate(stem, str(i))
            if valid_nick(cand) and not is_nick_taken(room_key, cand):
                return JSONResponse({"ok": True, "suggestion": cand})

    return JSONResponse({"ok": True, "suggestion": _fit_candidate(stem, str(random.randint(100, 999)))})


# =========================
# KOMANDOS
# =========================
HELP_TEXT = (
    "Komandos:\n"
    "  /help                     - pagalba\n"
    "  /who                      - kas online (vardai)\n"
    "  /dm VARDAS ŽINUTĖ         - private žinutė vartotojui\n"
    "  /topic                    - parodyti temą\n"
    "  /topic TEKSTAS            - pakeisti temą (visiems)\n"
    "  /history [N]              - atsiųsti paskutines N žinučių (default 120)\n"
    "  /me veiksmas              - action žinutė\n"
    "  /roll [NdM]               - kauliukas (pvz. /roll 2d6)\n"
    "  /flip                     - monetos metimas\n"
    "  /time                     - serverio laikas (Vilnius)\n"
)


async def handle_command(room_key: str, ws: WebSocket, text: str) -> bool:
    text = text.strip()
    u: User = ROOMS[room_key]["users"].get(ws)
    if not u:
        return True

    low = text.lower()

    if low in ("/help", "/?"):
        await room_send(ws, {"type": "sys", "t": ts(), "text": HELP_TEXT})
        return True

    if low == "/who":
        names = sorted({uu.nick for uu in ROOMS[room_key]["users"].values()})
        await room_send(ws, {"type": "sys", "t": ts(), "text": "Online: " + ", ".join(names)})
        return True

    if low.startswith("/dm "):
        parts = text.split(" ", 2)
        if len(parts) < 3:
            await room_send(ws, {"type": "sys", "t": ts(), "text": "Naudojimas: /dm VARDAS ŽINUTĖ"})
            return True
        target_nick = parts[1].strip()
        msg = parts[2].strip()[:300]
        if not msg:
            return True

        async with state_lock:
            target_ws = find_user_ws_by_nick(room_key, target_nick)
            if not target_ws:
                await room_send(ws, {"type": "sys", "t": ts(), "text": f"Vartotojas nerastas online: {target_nick}"})
                return True
            target_user = ROOMS[room_key]["users"].get(target_ws)
            if not target_user:
                await room_send(ws, {"type": "sys", "t": ts(), "text": f"Vartotojas nerastas online: {target_nick}"})
                return True

        item = {"type": "dm", "t": ts(), "from": u.nick, "to": target_user.nick, "color": u.color, "text": msg}
        dm_history_for(u.nick, target_user.nick).append(item)

        await room_send(ws, item)
        if target_ws is not ws:
            await room_send(target_ws, item)
        return True

    if low.startswith("/topic"):
        parts = text.split(" ", 1)
        if len(parts) == 1:
            await room_send(ws, {"type": "sys", "t": ts(), "text": f"Tema: {ROOMS[room_key]['topic']}"})
        else:
            new_topic = parts[1].strip()[:120] or "Bendras kanalas"
            ROOMS[room_key]["topic"] = new_topic
            await room_broadcast(room_key, {"type": "topic", "text": f"{ROOMS[room_key]['title']} — {new_topic}"})
            await room_push_history(room_key, {"type": "sys", "t": ts(), "text": f"Tema pakeista į: {new_topic}"})
        return True

    if low.startswith("/history"):
        parts = text.split(" ", 1)
        n = 120
        if len(parts) == 2:
            try:
                n = int(parts[1].strip())
            except Exception:
                n = 120
        n = max(1, min(n, HISTORY_LIMIT))
        items = list(ROOMS[room_key]["history"])[-n:]
        await room_send(ws, {"type": "history", "items": items})
        return True

    if low.startswith("/me "):
        action = text.split(" ", 1)[1].strip()[:240]
        if not action:
            return True
        item = {"type": "me_action", "t": ts(), "nick": u.nick, "color": u.color, "text": action}
        await room_push_history(room_key, item)
        await room_broadcast(room_key, item)
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
        await room_push_history(room_key, item)
        await room_broadcast(room_key, item)
        return True

    if low == "/flip":
        res = random.choice(["HERBAS", "SKAIČIUS"])
        item = {"type": "sys", "t": ts(), "text": f"{u.nick} meta monetą: {res}"}
        await room_push_history(room_key, item)
        await room_broadcast(room_key, item)
        return True

    if low == "/time":
        await room_send(ws, {"type": "sys", "t": ts(), "text": f"Serverio laikas (Vilnius): {ts()}"})
        return True

    return False


async def handle_client_message(room_key: str, ws: WebSocket, data: dict) -> None:
    """
    Priimam:
    - {"text": "..."}  (žinutė arba komanda)
    - {"type":"dm_history_req","with":"Nick","limit":120}
    """
    u: User = ROOMS[room_key]["users"].get(ws)
    if not u:
        return

    msg_type = str(data.get("type", "")).strip()

    if msg_type == "dm_history_req":
        peer = str(data.get("with", "")).strip()
        try:
            limit = int(data.get("limit", 120))
        except Exception:
            limit = 120
        limit = max(1, min(limit, HISTORY_LIMIT))

        if not peer:
            await room_send(ws, {"type": "sys", "t": ts(), "text": "DM istorijai reikia nurodyti vartotoją."})
            return

        items = dm_history_items(u.nick, peer, limit=limit)
        await room_send(ws, {"type": "dm_history", "with": peer, "items": items})
        return

    text = str(data.get("text", "")).strip()[:300]
    if not text:
        return

    if text.startswith("/"):
        handled = await handle_command(room_key, ws, text)
        if handled:
            return

    item = {"type": "msg", "t": ts(), "nick": u.nick, "color": u.color, "text": text}
    await room_push_history(room_key, item)
    await room_broadcast(room_key, item)


# =========================
# WEBSOCKET
# =========================
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    room_key = (ws.query_params.get("room") or "").strip()
    nick = (ws.query_params.get("nick") or "").strip()

    await ws.accept()

    if room_key not in ROOMS:
        await ws.send_json({"type": "error", "code": "BAD_ROOM", "text": "Neteisingas kanalas."})
        await ws.close()
        return

    if not valid_nick(nick):
        await ws.send_json({"type": "error", "code": "BAD_NICK", "text": "Netinkamas nick. Grįžk ir įvesk teisingą."})
        await ws.close()
        return

    async with state_lock:
        if is_nick_taken(room_key, nick):
            await ws.send_json({"type": "error", "code": "NICK_TAKEN", "text": "Nick užimtas. Pasirink kitą."})
            await ws.close()
            return

        used = room_used_colors(room_key)
        color = alloc_color(used)
        u = User(nick=nick, color=color, joined_at=time.time())

        ROOMS[room_key]["clients"].add(ws)
        ROOMS[room_key]["users"][ws] = u

    # Handshake
    await room_send(ws, {"type": "topic", "text": f"{ROOMS[room_key]['title']} — {ROOMS[room_key]['topic']}"})
    await room_send(ws, {"type": "history", "items": list(ROOMS[room_key]["history"])})
    await room_send(ws, {"type": "users", "items": room_userlist(room_key)})
    await room_send(ws, {"type": "me", "nick": u.nick, "color": u.color})
    await room_broadcast_userlist(room_key)

    try:
        while True:
            data = await ws.receive_json()
            if not isinstance(data, dict):
                continue
            await handle_client_message(room_key, ws, data)

    except WebSocketDisconnect:
        pass
    finally:
        await room_remove_client(room_key, ws)
        await room_broadcast_userlist(room_key)
