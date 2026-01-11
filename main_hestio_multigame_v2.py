"""
Hestio Rooms (chat) + Hestio Play (training lobby) — MULTI-GAME MVP

What this provides (single Render service):
- /            -> Chat page (served from ./static/chat/index.html if present; otherwise fallback)
- /play        -> Lobby + multiplayer training games
- /ws          -> WebSocket matchmaking + realtime state

Games included:
1) rebuild_6x6  - "Rebuild the Pattern" (6×6) drag & drop polyomino pieces onto the board
2) tile_3x3     - "Tile Puzzle" (3×3) drag tiles to recreate a reference image

Nick persistence (chat -> play):
- /play will auto-fill nickname from:
    1) URL param ?nick=...
    2) localStorage key "hestio_nick"
    3) cookie "hestio_nick"
- In your chat frontend, when user sets nickname, store:
    localStorage.setItem("hestio_nick", nick);
  (or set cookie if you prefer)

Images:
- If you put images under ./static/puzzles/ (jpg/png/webp), Tile Puzzle will use them.
- If no images exist, it uses built-in, generated "map-like" SVG backgrounds (royalty-free).
- Do NOT ship copyrighted game screenshots unless you have rights.

requirements.txt:
fastapi==0.115.0
uvicorn[standard]==0.30.6
"""

from __future__ import annotations

import asyncio
import json
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from starlette.responses import Response
from fastapi.staticfiles import StaticFiles


# -----------------------------
# Paths / static
# -----------------------------

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
CHAT_INDEX = STATIC_DIR / "chat" / "index.html"
PUZZLE_DIR = STATIC_DIR / "puzzles"

# -----------------------------
# Helpers
# -----------------------------

def now_ms() -> int:
    return int(time.time() * 1000)


def clamp_int(x: Any, lo: int, hi: int, default: int) -> int:
    try:
        v = int(x)
        return max(lo, min(hi, v))
    except Exception:
        return default


def safe_json_loads(s: str) -> Dict[str, Any]:
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def make_player_id() -> str:
    return secrets.token_urlsafe(8)


def make_room_id() -> str:
    return secrets.token_urlsafe(6)


# -----------------------------
# Game registry
# -----------------------------

GAMES: List[Dict[str, str]] = [
    {"id": "rebuild_6x6", "name": "Rebuild the Pattern (6×6)", "tag": "Drag pieces onto the grid"},
    {"id": "tile_3x3", "name": "Tile Puzzle (3×3)", "tag": "Recreate the image from tiles"},
]
GAME_IDS = {g["id"] for g in GAMES}


# -----------------------------
# Deterministic RNG (seeded)
# -----------------------------

def _seed_to_u32(seed: str) -> int:
    s = 0
    for ch in seed[:64]:
        s = (s * 131 + ord(ch)) & 0xFFFFFFFF
    return s or 2463534242


def _xorshift32(x: int) -> int:
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= (x >> 17) & 0xFFFFFFFF
    x ^= (x << 5) & 0xFFFFFFFF
    return x & 0xFFFFFFFF


def _shuffle_det(items: List[int], seed: str) -> List[int]:
    x = _seed_to_u32(seed)
    arr = list(items)
    for i in range(len(arr) - 1, 0, -1):
        x = _xorshift32(x)
        j = x % (i + 1)
        arr[i], arr[j] = arr[j], arr[i]
    return arr


# -----------------------------
# Game 1: Rebuild 6x6 (binary pattern + polyomino pieces)
# -----------------------------

def generate_reference_grid(seed: str, size: int = 6) -> List[int]:
    total = size * size
    grid = [0] * total
    x = _seed_to_u32(seed)
    filled_target = 16 + (x % 7)  # 16..22
    chosen = set()
    while len(chosen) < filled_target:
        x = _xorshift32(x)
        chosen.add(x % total)
    for idx in chosen:
        grid[idx] = 1
    return grid


def _neighbors(idx: int, size: int) -> List[int]:
    r, c = divmod(idx, size)
    out = []
    if r > 0:
        out.append((r - 1) * size + c)
    if r < size - 1:
        out.append((r + 1) * size + c)
    if c > 0:
        out.append(r * size + (c - 1))
    if c < size - 1:
        out.append(r * size + (c + 1))
    return out


def split_into_pieces(reference: List[int], size: int, seed: str) -> List[Dict[str, Any]]:
    filled = [i for i, v in enumerate(reference) if v == 1]
    filled_set = set(filled)
    visited = set()

    comps: List[List[int]] = []
    for start in filled:
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        comp: List[int] = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in _neighbors(cur, size):
                if nb in filled_set and nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        comps.append(comp)

    rng = _seed_to_u32(seed + "pieces")

    def pick_chunk(cells: List[int], chunk_size: int) -> List[int]:
        nonlocal rng
        cells_sorted = sorted(set(cells))
        rng = _xorshift32(rng)
        rot = rng % len(cells_sorted)
        rot_cells = cells_sorted[rot:] + cells_sorted[:rot]
        return rot_cells[:chunk_size]

    pieces_cells: List[List[int]] = []
    for comp in comps:
        comp = list(set(comp))
        if len(comp) <= 5:
            pieces_cells.append(comp)
        else:
            remaining = set(comp)
            while remaining:
                rem = len(remaining)
                if rem <= 5:
                    pieces_cells.append(list(remaining))
                    remaining.clear()
                    break
                chunk = pick_chunk(list(remaining), 4)
                for c in chunk:
                    remaining.discard(c)
                pieces_cells.append(chunk)

    pieces: List[Dict[str, Any]] = []
    for i, cells in enumerate(pieces_cells):
        coords = [(idx % size, idx // size) for idx in cells]  # (x,y)
        minx = min(x for x, y in coords)
        miny = min(y for x, y in coords)
        norm = [{"dx": x - minx, "dy": y - miny} for x, y in coords]
        pieces.append({"id": f"p{i+1}", "cells": norm, "cellCount": len(norm)})
    return pieces


def validate_binary_grid(grid: Any, size: int) -> Optional[List[int]]:
    total = size * size
    if not isinstance(grid, list) or len(grid) != total:
        return None
    out: List[int] = []
    for v in grid:
        if isinstance(v, bool):
            out.append(1 if v else 0)
        elif isinstance(v, (int, float)):
            out.append(1 if int(v) != 0 else 0)
        else:
            return None
    return out


def compute_accuracy_binary(reference: List[int], submitted: List[int]) -> float:
    matches = sum(1 for a, b in zip(reference, submitted) if a == b)
    return round(100.0 * matches / len(reference), 2) if reference else 0.0


# -----------------------------
# Game 2: Tile puzzle (3x3)
# -----------------------------

def list_puzzle_images() -> List[str]:
    if not PUZZLE_DIR.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = [p.name for p in PUZZLE_DIR.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    return files


def make_svg_data_url(seed: str, size: int = 720) -> str:
    """
    Returns a data: URL SVG image (royalty-free, generated).
    This is intentionally "map-like" (rooms/alleys/blocks) but NOT based on copyrighted game art.
    """
    x = _seed_to_u32(seed)

    def rnd(a: int, b: int) -> int:
        nonlocal x
        x = _xorshift32(x)
        return a + (x % (b - a + 1))

    themes = [
        {"bg1": "#d7b98e", "bg2": "#8a6d4f", "accent": "#ff5252", "label": "A"},  # desert
        {"bg1": "#d9c8b2", "bg2": "#4b5a6a", "accent": "#ff4fd8", "label": "B"},  # warm alley
        {"bg1": "#0b1330", "bg2": "#070a14", "accent": "#50f5ff", "label": "C"},  # neon night
        {"bg1": "#1f3a2a", "bg2": "#0b1f15", "accent": "#ffd54f", "label": "D"},  # forest
    ]
    theme = themes[_seed_to_u32(seed + "t") % len(themes)]

    rects = []
    for _ in range(20):
        rx = rnd(0, size - 140)
        ry = rnd(0, size - 140)
        rw = rnd(70, 260)
        rh = rnd(70, 260)
        o = rnd(14, 34) / 100
        rects.append(
            f"<rect x='{rx}' y='{ry}' width='{rw}' height='{rh}' rx='18' "
            f"fill='rgba(255,255,255,{o})'/>"
        )

    grid_lines = []
    step = size // 12
    for i in range(1, 12):
        p = i * step
        grid_lines.append(
            f"<line x1='{p}' y1='0' x2='{p}' y2='{size}' stroke='rgba(255,255,255,0.10)' stroke-width='3' />"
        )
        grid_lines.append(
            f"<line x1='0' y1='{p}' x2='{size}' y2='{p}' stroke='rgba(255,255,255,0.10)' stroke-width='3' />"
        )

    cx = rnd(size // 4, 3 * size // 4)
    cy = rnd(size // 4, 3 * size // 4)
    obj = f"""
      <g opacity="0.9">
        <circle cx="{cx}" cy="{cy}" r="{rnd(44, 92)}" fill="rgba(0,0,0,0.18)"/>
        <circle cx="{cx}" cy="{cy}" r="{rnd(22, 44)}" fill="{theme['accent']}" opacity="0.35"/>
        <text x="{cx}" y="{cy+10}" font-size="92" text-anchor="middle"
              font-family="system-ui,Segoe UI,Roboto" fill="{theme['accent']}" opacity="0.55">{theme['label']}</text>
      </g>
    """

    svg = f"""
<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="{theme['bg1']}"/>
      <stop offset="100%" stop-color="{theme['bg2']}"/>
    </linearGradient>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="10" stdDeviation="12" flood-color="#000" flood-opacity="0.25"/>
    </filter>
  </defs>
  <rect x="0" y="0" width="{size}" height="{size}" fill="url(#g)"/>
  <g filter="url(#shadow)">
    {''.join(rects)}
  </g>
  <g opacity="0.55">
    {''.join(grid_lines)}
  </g>
  {obj}
</svg>
""".strip()

    return "data:image/svg+xml;utf8," + quote(svg)


def pick_image_url(seed: str) -> str:
    imgs = list_puzzle_images()
    if imgs:
        idx = _seed_to_u32(seed) % len(imgs)
        return f"/static/puzzles/{imgs[idx]}"
    return make_svg_data_url(seed)


def validate_positions(payload: Any, size: int) -> Optional[List[int]]:
    n = size * size
    if not isinstance(payload, list) or len(payload) != n:
        return None
    out: List[int] = []
    for v in payload:
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            iv = int(v)
            if iv == -1 or (0 <= iv < n):
                out.append(iv)
            else:
                return None
        else:
            return None
    return out


def tile_accuracy(size: int, positions: List[int]) -> float:
    n = size * size
    correct = 0
    for cell in range(n):
        if positions[cell] == cell:
            correct += 1
    return round(100.0 * correct / n, 2)


# -----------------------------
# Room / session state
# -----------------------------

@dataclass
class PlayerConn:
    id: str
    name: str
    ws: WebSocket
    joined_at_ms: int = field(default_factory=now_ms)
    submitted_at_ms: Optional[int] = None
    submission_grid: Optional[List[int]] = None
    submission_positions: Optional[List[int]] = None


@dataclass
class GameSession:
    game_id: str
    game_type: str
    seed: str
    started_at_ms: int = 0
    ends_at_ms: int = 0
    finished: bool = False
    size: int = 6
    reference_grid: List[int] = field(default_factory=list)
    pieces: List[Dict[str, Any]] = field(default_factory=list)
    tile_size: int = 3
    image_url: str = ""
    tile_order: List[int] = field(default_factory=list)


@dataclass
class Room:
    id: str
    target_players: int
    game_type: str
    status: str = "waiting"  # waiting|countdown|in_game|results
    created_at_ms: int = field(default_factory=now_ms)
    players: Dict[str, PlayerConn] = field(default_factory=dict)
    game: Optional[GameSession] = None
    countdown_task: Optional[asyncio.Task] = None
    game_task: Optional[asyncio.Task] = None


class RoomManager:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.rooms_by_id: Dict[str, Room] = {}
        self.pool: Dict[Tuple[int, str], List[str]] = {}

    async def assign_room(self, target_players: int, game_type: str) -> Room:
        async with self.lock:
            target_players = max(1, min(10, target_players))
            if game_type not in GAME_IDS:
                game_type = "rebuild_6x6"
            key = (target_players, game_type)
            ids = self.pool.setdefault(key, [])

            for rid in list(ids):
                room = self.rooms_by_id.get(rid)
                if room and room.status == "waiting" and len(room.players) < room.target_players:
                    return room

            rid = make_room_id()
            room = Room(id=rid, target_players=target_players, game_type=game_type)
            self.rooms_by_id[rid] = room
            ids.append(rid)
            return room

    async def remove_room_from_pool(self, room: Room) -> None:
        key = (room.target_players, room.game_type)
        ids = self.pool.get(key, [])
        if room.id in ids:
            ids.remove(room.id)

    async def mark_room_not_joinable(self, room: Room) -> None:
        async with self.lock:
            await self.remove_room_from_pool(room)

    async def mark_room_joinable(self, room: Room) -> None:
        async with self.lock:
            if room.status == "waiting" and len(room.players) < room.target_players:
                key = (room.target_players, room.game_type)
                ids = self.pool.setdefault(key, [])
                if room.id not in ids:
                    ids.append(room.id)

    async def maybe_cleanup_room(self, room: Room) -> None:
        async with self.lock:
            if len(room.players) == 0:
                await self.remove_room_from_pool(room)
                self.rooms_by_id.pop(room.id, None)

    async def list_public_rooms(self) -> List[Dict[str, Any]]:
        async with self.lock:
            out = []
            for room in self.rooms_by_id.values():
                if room.status in ("waiting", "countdown"):
                    out.append(
                        {
                            "roomId": room.id,
                            "targetPlayers": room.target_players,
                            "count": len(room.players),
                            "status": room.status,
                            "gameType": room.game_type,
                        }
                    )
            out.sort(key=lambda r: (r["gameType"], r["targetPlayers"] - r["count"], -r["count"]))
            return out


manager = RoomManager()


# -----------------------------
# FastAPI app
# -----------------------------

app = FastAPI(title="HestioRooms + Play (Multi-game MVP)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # MVP; restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# -----------------------------
# Chat page (served from /static/chat/index.html if present)
# -----------------------------

CHAT_FALLBACK_HTML = """
<!doctype html>
<html>
<head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" />
<title>HestioRooms</title>
<style>
  body{font-family:system-ui; margin:0; background:#070a14; color:#eaf0ff;}
  header{padding:16px 18px; display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid rgba(255,255,255,0.12);}
  a{color:#50f5ff; text-decoration:none; font-weight:800;}
  main{padding:18px; max-width:1000px; margin:0 auto;}
  .card{padding:14px; border:1px solid rgba(255,255,255,0.12); border-radius:16px; background:rgba(255,255,255,0.06);}
  code{background:rgba(255,255,255,0.08); padding:2px 6px; border-radius:8px;}
  input,button{border-radius:12px;border:1px solid rgba(255,255,255,0.12);background:rgba(255,255,255,0.06);color:#eaf0ff;padding:10px 12px;font-size:14px;}
  button{cursor:pointer;}
</style></head>
<body>
<header>
  <div style="font-weight:900; letter-spacing:.8px;">HESTIOROOMS</div>
  <a href="/play">Open Hestio Play</a>
</header>
<main>
  <div class="card">
    <div style="opacity:.8">Chat placeholder</div>
    <p>Įdėk savo chat front-end į <code>./static/chat/index.html</code> (ir assets į <code>./static/chat/</code>).</p>

    <div style="margin-top:12px; display:flex; gap:10px; flex-wrap:wrap;">
      <input id="nick" placeholder="Set nickname (saved)" maxlength="24" />
      <button onclick="saveNick()">Save</button>
      <a href="/play" style="align-self:center;">Go to /play</a>
    </div>
    <div id="saved" style="margin-top:10px; opacity:.8;"></div>

    <script>
      function saveNick(){
        const v=(document.getElementById("nick").value||"").trim();
        if(!v) return;
        localStorage.setItem("hestio_nick", v);
        document.cookie = "hestio_nick="+encodeURIComponent(v)+"; path=/; max-age=31536000";
        document.getElementById("saved").textContent="Saved: "+v;
      }
      const existing = localStorage.getItem("hestio_nick") || "";
      if(existing){ document.getElementById("nick").value = existing; document.getElementById("saved").textContent="Saved: "+existing; }
    </script>
  </div>
</main>
</body>
</html>
""".strip()


@app.get("/")
async def chat_index() -> Response:
    if CHAT_INDEX.exists():
        return FileResponse(str(CHAT_INDEX))
    return HTMLResponse(CHAT_FALLBACK_HTML)


@app.get("/chat")
async def chat_alias() -> Response:
    return await chat_index()


# -----------------------------
# Play UI (Lobby + game)
# -----------------------------

PLAY_PAGE = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Hestio Play (Training)</title>
  <style>
    :root { --bg:#070a14; --panel:rgba(255,255,255,0.06); --text:#eaf0ff; --muted:rgba(234,240,255,0.7);
            --stroke:rgba(255,255,255,0.12); --good:rgba(80,245,255,0.9); --bad:rgba(255,110,220,0.9); }
    *{ box-sizing:border-box; }
    body{ margin:0; background:radial-gradient(1200px 800px at 50% 20%, #0b1330 0%, var(--bg) 60%);
          color:var(--text); font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
    header{ position:sticky; top:0; z-index:10; backdrop-filter:blur(14px); background:rgba(7,10,20,0.7);
            border-bottom:1px solid var(--stroke); padding:14px 18px; display:flex; align-items:center; justify-content:space-between; }
    .brand{ display:flex; align-items:center; gap:10px; font-weight:800; letter-spacing:0.8px; }
    .pill{ padding:6px 10px; border:1px solid var(--stroke); border-radius:999px; background:rgba(255,255,255,0.04); color:var(--muted); font-size:12px; }
    main{ max-width:1280px; margin:0 auto; padding:18px; }
    .grid{ display:grid; grid-template-columns:380px 1fr; gap:14px; }
    @media (max-width:1020px){ .grid{ grid-template-columns:1fr; } }
    .card{ border:1px solid var(--stroke); background:var(--panel); border-radius:18px; padding:14px; box-shadow:0 10px 30px rgba(0,0,0,0.2); }
    .card h3{ margin:0 0 10px 0; font-size:15px; color:var(--muted); font-weight:650; }
    .row{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
    select,input,button{ border-radius:12px; border:1px solid var(--stroke); background:rgba(255,255,255,0.06); color:var(--text);
                         padding:10px 12px; font-size:14px; outline:none; }
    button{ cursor:pointer; background:rgba(255,255,255,0.08); }
    button.primary{ border-color:rgba(80,245,255,0.35); box-shadow:0 0 0 3px rgba(80,245,255,0.08); }
    button:disabled{ cursor:not-allowed; opacity:0.6; }
    .muted{ color:var(--muted); font-size:13px; }
    .status{ font-size:14px; } .status strong{ color:var(--good); }
    a{ color:var(--good); text-decoration:none; font-weight:700; }

    .boardArea{ display:flex; gap:14px; flex-wrap:wrap; align-items:flex-start; }
    .miniGrid,.mainGrid{ border:1px solid var(--stroke); background:rgba(255,255,255,0.04); border-radius:16px; padding:12px; }
    .gridTitle{ display:flex; align-items:center; justify-content:space-between; margin-bottom:8px; }
    .timer{ font-weight:750; letter-spacing:0.6px; } .timer.bad{ color:var(--bad); }

    /* Game 1 */
    .gameWrap{ display:grid; grid-template-columns:340px 1fr; gap:14px; }
    @media (max-width:1020px){ .gameWrap{ grid-template-columns:1fr; } }
    .cells{ display:grid; gap:6px; justify-content:start; align-content:start; }
    .cell{ width:34px; height:34px; border-radius:10px; border:1px solid rgba(255,255,255,0.12); background:rgba(255,255,255,0.03); }
    .cell.filled{ background:rgba(80,245,255,0.22); border-color:rgba(80,245,255,0.42); }
    .cell.occupied{ background:rgba(255,255,255,0.11); border-color:rgba(255,255,255,0.22); }
    .cell.ghostOk{ background:rgba(80,245,255,0.18); outline:3px solid rgba(80,245,255,0.22); }
    .cell.ghostBad{ background:rgba(255,110,220,0.18); outline:3px solid rgba(255,110,220,0.22); }

    .tray{ position:relative; min-height:360px; border-radius:16px; border:1px solid var(--stroke); background:rgba(255,255,255,0.04);
           overflow:hidden; padding:10px; }
    .piece{ position:absolute; touch-action:none; user-select:none; border-radius:14px; padding:8px;
            border:1px solid rgba(255,255,255,0.16); background:rgba(255,255,255,0.06); box-shadow:0 12px 22px rgba(0,0,0,0.25); cursor:grab; }
    .piece.dragging{ cursor:grabbing; transform:scale(1.02); }
    .piece .pgrid{ display:grid; gap:6px; align-content:start; justify-content:start; }
    .pcell{ width:30px; height:30px; border-radius:9px; background:rgba(255,255,255,0.12); border:1px solid rgba(255,255,255,0.20); }

    /* Game 2 */
    .tileLayout{ display:grid; grid-template-columns:320px 1fr; gap:14px; }
    @media (max-width:1020px){ .tileLayout{ grid-template-columns:1fr; } }
    .tileTray{ display:grid; grid-template-columns:repeat(3, 96px); gap:10px; align-content:start; justify-content:start; min-height:340px; }
    .tile{ width:96px; height:96px; border-radius:14px; border:1px solid rgba(255,255,255,0.16); background-size:288px 288px; background-repeat:no-repeat;
           box-shadow:0 12px 22px rgba(0,0,0,0.25); cursor:grab; user-select:none; touch-action:none; }
    .tile.dragging{ cursor:grabbing; transform:scale(1.02); }
    .tileBoard{ display:grid; grid-template-columns:repeat(3, 106px); gap:10px; }
    .slot{ width:106px; height:106px; border-radius:16px; border:1px dashed rgba(255,255,255,0.18); background:rgba(255,255,255,0.03); display:flex; align-items:center; justify-content:center; }
    .slot.filled{ border-style:solid; background:rgba(255,255,255,0.06); }
    .slot.ghostOk{ outline:3px solid rgba(80,245,255,0.22); } .slot.ghostBad{ outline:3px solid rgba(255,110,220,0.22); }
    .refImg{ width:300px; height:300px; border-radius:18px; border:1px solid rgba(255,255,255,0.14); background-size:cover; background-position:center; }

    /* Results */
    .resultsList{ display:flex; flex-direction:column; gap:8px; }
    .resItem{ display:flex; align-items:center; justify-content:space-between; border:1px solid var(--stroke); background:rgba(255,255,255,0.05);
              border-radius:14px; padding:10px 12px; }
    .tag{ font-size:12px; padding:4px 8px; border-radius:999px; border:1px solid var(--stroke); color:var(--muted); }
    .rank{ font-weight:800; width:38px; text-align:center; color:var(--good); } .right{ display:flex; gap:8px; align-items:center; }
  </style>
</head>
<body>
<header>
  <div class="brand">HESTIO <span class="pill">PLAY · Training</span></div>
  <div class="row">
    <a class="pill" href="/">Back to Chat</a>
    <span class="pill" id="pillConn">Disconnected</span>
    <span class="pill" id="pillRoom">Room: —</span>
  </div>
</header>

<main>
  <div class="grid">
    <div class="card">
      <h3>Lobby</h3>
      <div class="row">
        <label class="muted">Game</label>
        <select id="gameType"></select>
      </div>
      <div class="row" style="margin-top:8px;">
        <label class="muted">Players</label>
        <select id="targetPlayers"><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option>
          <option>6</option><option>7</option><option>8</option><option>9</option><option selected>10</option></select>
      </div>
      <div class="row" style="margin-top:8px;">
        <input id="name" placeholder="Nickname (optional)" maxlength="24" style="flex:1; min-width:160px;" />
        <button class="primary" id="btnJoin" onclick="join()">Join</button>
        <button id="btnLeave" onclick="leave()" disabled>Leave</button>
      </div>
      <div style="margin-top:12px;" class="status"><div class="muted">Status</div><div id="statusLine">Not connected.</div></div>
      <div style="margin-top:12px;" class="muted" id="gameHint">—</div>
      <div style="margin-top:14px;" class="muted">Tip: open this page in two tabs to test multiplayer quickly.</div>
    </div>

    <div class="card">
      <h3>Match</h3>

      <div id="viewWaiting" style="display:block;">
        <div class="muted">Waiting room</div>
        <div id="waitingDetails" style="margin-top:8px;">—</div>
      </div>

      <div id="viewCountdown" style="display:none;">
        <div class="muted">Starting in</div>
        <div style="font-size:46px; font-weight:900; margin-top:8px;" id="countdownNum">3</div>
      </div>

      <!-- Game 1 -->
      <div id="viewGame1" style="display:none;">
        <div class="gameWrap">
          <div>
            <div class="muted">Pieces</div>
            <div class="tray" id="tray"></div>
            <div class="row" style="margin-top:10px;">
              <button class="primary" id="btnSubmit1" onclick="submitGame1()">Submit</button>
              <span class="muted" id="submitState1">—</span>
            </div>
          </div>
          <div>
            <div class="boardArea">
              <div class="miniGrid">
                <div class="gridTitle"><div class="muted">Reference</div></div>
                <div class="cells" id="refGrid"></div>
              </div>
              <div class="mainGrid">
                <div class="gridTitle"><div class="muted">Your board</div><div class="timer" id="timer">60.0</div></div>
                <div class="cells" id="mainGrid"></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Game 2 -->
      <div id="viewGame2" style="display:none;">
        <div class="tileLayout">
          <div>
            <div class="muted">Reference</div>
            <div class="refImg" id="refImg"></div>
            <div class="row" style="margin-top:10px;">
              <button class="primary" id="btnSubmit2" onclick="submitGame2()">Submit</button>
              <span class="muted" id="submitState2">—</span>
            </div>
          </div>
          <div>
            <div class="boardArea">
              <div class="mainGrid" style="min-width:unset;">
                <div class="gridTitle"><div class="muted">Recreate</div><div class="timer" id="timer2">60.0</div></div>
                <div class="tileBoard" id="tileBoard"></div>
              </div>
              <div class="miniGrid" style="width:340px;">
                <div class="gridTitle"><div class="muted">Tiles</div></div>
                <div class="tileTray" id="tileTray"></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Results -->
      <div id="viewResults" style="display:none;">
        <div class="muted">Results</div>
        <div class="resultsList" id="resultsList" style="margin-top:10px;"></div>
        <div class="row" style="margin-top:12px;">
          <button class="primary" onclick="playAgain()">Play again</button>
          <button onclick="backToLobby()">Back to lobby</button>
        </div>
      </div>
    </div>
  </div>
</main>

<script>
  const GAME_LIST = [
    {id:"rebuild_6x6", name:"Rebuild the Pattern (6×6)", hint:"Drag pieces onto the grid. Submit = your 6×6 pattern."},
    {id:"tile_3x3", name:"Tile Puzzle (3×3)", hint:"Drag tiles into the correct positions to recreate the image."}
  ];

  let ws=null, playerId=null, roomId=null, roomState=null;
  let game=null, countdownInterval=null, timerInterval=null;

  // Game1
  let pieceState={}, boardOcc=[];
  const SIZE=6;

  // Game2
  const T=3;
  let tilePositions=[], tileImageUrl="";

  function byId(id){ return document.getElementById(id); }
  function setPill(id,t){ byId(id).textContent=t; }
  function status(msg,good=false){ byId("statusLine").innerHTML = good ? `<strong>${msg}</strong>` : msg; }

  function showView(viewId){
    const views=["viewWaiting","viewCountdown","viewGame1","viewGame2","viewResults"];
    for(const id of views){ byId(id).style.display = (id===viewId) ? "block" : "none"; }
  }

  // ---- Nick persistence ----
  function getCookie(name){
    const m = document.cookie.match(new RegExp('(?:^|; )'+name.replace(/([$?*|{}\(\)\[\]\\\/\+^])/g,'\\$1')+'=([^;]*)'));
    return m ? decodeURIComponent(m[1]) : null;
  }
  function preferredNick(){
    const params = new URLSearchParams(location.search);
    const q = (params.get("nick")||"").trim();
    if(q) return q;
    const ls = (localStorage.getItem("hestio_nick")||"").trim();
    if(ls) return ls;
    const ck = (getCookie("hestio_nick")||"").trim();
    if(ck) return ck;
    return "";
  }
  function persistNick(v){
    const nick = (v||"").trim();
    if(!nick) return;
    localStorage.setItem("hestio_nick", nick);
    document.cookie = "hestio_nick="+encodeURIComponent(nick)+"; path=/; max-age=31536000";
  }

  function initLobby(){
    const sel=byId("gameType");
    sel.innerHTML="";
    for(const g of GAME_LIST){
      const opt=document.createElement("option");
      opt.value=g.id; opt.textContent=g.name;
      sel.appendChild(opt);
    }
    sel.value="rebuild_6x6";
    byId("gameHint").textContent = GAME_LIST.find(x=>x.id===sel.value).hint;
    sel.addEventListener("change", ()=>{
      byId("gameHint").textContent = GAME_LIST.find(x=>x.id===sel.value).hint;
    });

    // auto-fill nickname from chat
    const pn = preferredNick();
    if(pn && !byId("name").value) byId("name").value = pn;
    byId("name").addEventListener("change", ()=>persistNick(byId("name").value));
  }
  initLobby();

  function connect(){
    if(ws) return;
    const proto=(location.protocol==="https:")?"wss":"ws";
    ws=new WebSocket(`${proto}://${location.host}/ws`);
    ws.onopen=()=>{ setPill("pillConn","Connected"); status("Connected. Pick a game and join.", true); };
    ws.onmessage=(ev)=>{ let m=null; try{ m=JSON.parse(ev.data);}catch{return;} handleMsg(m.type, m.payload||{}); };
    ws.onclose=()=>{
      setPill("pillConn","Disconnected");
      ws=null; playerId=null; roomId=null; roomState=null; game=null;
      stopCountdown(); stopTimer();
      showView("viewWaiting");
      byId("btnLeave").disabled=true;
      status("Disconnected. Refresh to reconnect.");
    };
  }
  connect();

  function send(type,payload){ if(!ws) return; ws.send(JSON.stringify({type,payload})); }

  function join(){
    connect();
    const targetPlayers=parseInt(byId("targetPlayers").value,10);
    let name=(byId("name").value||"").trim();
    if(!name) name = preferredNick();
    if(name) { byId("name").value = name; persistNick(name); }

    const gameType=byId("gameType").value||"rebuild_6x6";
    send("training:join", {targetPlayers, name, gameType});
    byId("btnLeave").disabled=false;
    status("Joining…");
  }

  function leave(){
    send("room:leave", {});
    byId("btnLeave").disabled=true;
    setPill("pillRoom","Room: —");
    status("Left room.");
    showView("viewWaiting");
    byId("waitingDetails").textContent="—";
  }

  function playAgain(){ join(); }
  function backToLobby(){ showView("viewWaiting"); status("Pick a game and join."); }

  function handleMsg(type,payload){
    if(type==="session:hello"){ playerId=payload.playerId; status(`Session ready. Your ID: ${playerId.slice(0,6)}…`, true); return; }

    if(type==="room:state"){
      roomState=payload; roomId=payload.roomId;
      setPill("pillRoom",`Room: ${roomId} · ${payload.gameType}`);
      const count=payload.count, target=payload.targetPlayers, st=payload.status;
      const names=payload.players.map(p=>`${p.name}${p.submitted?" (✓)":""}`);
      byId("waitingDetails").innerHTML =
        `<div><strong>${st.toUpperCase()}</strong></div>
         <div style="margin-top:6px;">Filling <strong>${count}/${target}</strong></div>
         <div class="muted" style="margin-top:8px;">Players:</div>
         <div style="margin-top:6px;">${names.join(", ")||"—"}</div>`;

      if(st==="waiting"){ showView("viewWaiting"); status(`Waiting: ${count}/${target} players.`); }
      return;
    }

    if(type==="room:countdown"){ showView("viewCountdown"); startCountdown(payload.startsAtMs); status("Room full. Starting…", true); return; }

    if(type==="game:start"){
      stopCountdown();
      game=payload;
      if(payload.gameType==="rebuild_6x6"){
        showView("viewGame1");
        byId("submitState1").textContent="—";
        byId("btnSubmit1").disabled=false;
        initBoards(payload.referenceGrid);
        initPieces(payload.pieces);
        startTimer(payload.endsAtMs, "timer");
        status("Game started. Drag pieces onto the board.", true);
      } else if(payload.gameType==="tile_3x3"){
        showView("viewGame2");
        byId("submitState2").textContent="—";
        byId("btnSubmit2").disabled=false;
        tileImageUrl=payload.imageUrl;
        initTilePuzzle(payload.tileSize, payload.tileOrder, payload.imageUrl);
        startTimer(payload.endsAtMs, "timer2");
        status("Game started. Recreate the image from tiles.", true);
      }
      return;
    }

    if(type==="game:results"){
      stopTimer();
      showView("viewResults");
      renderResults(payload.results);
      status("Round finished. Results are in.", true);
      return;
    }

    if(type==="error"){ status(`Error: ${payload.message||"unknown"}`); return; }
  }

  // Timer + countdown
  function stopCountdown(){ if(countdownInterval) clearInterval(countdownInterval); countdownInterval=null; }
  function startCountdown(startsAtMs){
    stopCountdown();
    const el=byId("countdownNum");
    countdownInterval=setInterval(()=>{
      const ms=startsAtMs-Date.now();
      const s=Math.max(0, Math.ceil(ms/1000));
      el.textContent=String(s);
      if(ms<=0) stopCountdown();
    }, 50);
  }

  function stopTimer(){ if(timerInterval) clearInterval(timerInterval); timerInterval=null; }
  function startTimer(endsAtMs, elId){
    stopTimer();
    const el=byId(elId);
    timerInterval=setInterval(()=>{
      const ms=endsAtMs-Date.now();
      const sec=Math.max(0, ms/1000);
      el.textContent=sec.toFixed(1);
      el.classList.toggle("bad", sec<=10);
      if(ms<=0) stopTimer();
    }, 60);
  }

  // ---------------- Game 1 (rebuild_6x6) ----------------
  function initBoards(referenceGrid){
    const refEl=byId("refGrid");
    refEl.innerHTML=""; refEl.style.gridTemplateColumns=`repeat(${SIZE}, 34px)`;
    for(let i=0;i<SIZE*SIZE;i++){
      const d=document.createElement("div");
      d.className="cell"+(referenceGrid[i]?" filled":"");
      refEl.appendChild(d);
    }
    const mainEl=byId("mainGrid");
    mainEl.innerHTML=""; mainEl.style.gridTemplateColumns=`repeat(${SIZE}, 34px)`;
    boardOcc=Array.from({length:SIZE*SIZE},()=>0);
    for(let i=0;i<SIZE*SIZE;i++){
      const d=document.createElement("div");
      d.className="cell"; d.dataset.idx=String(i);
      mainEl.appendChild(d);
    }
  }

  function recomputeBoardOcc(){
    boardOcc=Array.from({length:SIZE*SIZE},()=>0);
    for(const pid of Object.keys(pieceState)){
      const ps=pieceState[pid];
      if(!ps.placed) continue;
      for(const c of ps.cells){
        const x=ps.x+c.dx, y=ps.y+c.dy;
        boardOcc[y*SIZE+x]=1;
      }
    }
    const mainEl=byId("mainGrid");
    const cells=mainEl.querySelectorAll(".cell");
    cells.forEach((cell,idx)=>cell.classList.toggle("occupied", boardOcc[idx]===1));
  }

  function hslColorFromString(s){
    let h=0;
    for(let i=0;i<s.length;i++) h=(h*31 + s.charCodeAt(i))%360;
    return `hsl(${h} 90% 60%)`;
  }

  function initPieces(pieces){
    pieceState={};
    const tray=byId("tray");
    tray.innerHTML="";
    let cursorX=10, cursorY=10, rowH=0;
    const trayW=tray.clientWidth||320;

    pieces.forEach((p)=>{
      const el=document.createElement("div");
      el.className="piece"; el.dataset.pid=p.id;
      const col = hslColorFromString(p.id);
      el.style.borderColor = "rgba(255,255,255,0.22)";
      el.style.boxShadow = "0 12px 22px rgba(0,0,0,0.25), 0 0 0 3px rgba(80,245,255,0.06)";

      let maxDx=0,maxDy=0;
      p.cells.forEach(c=>{ maxDx=Math.max(maxDx,c.dx); maxDy=Math.max(maxDy,c.dy); });

      const grid=document.createElement("div");
      grid.className="pgrid";
      grid.style.gridTemplateColumns=`repeat(${maxDx+1}, 30px)`;

      const cellSet=new Set(p.cells.map(c=>`${c.dx},${c.dy}`));
      for(let y=0;y<=maxDy;y++){
        for(let x=0;x<=maxDx;x++){
          const pc=document.createElement("div");
          if(cellSet.has(`${x},${y}`)) {
            pc.className="pcell";
            pc.style.background = col;
            pc.style.borderColor = "rgba(255,255,255,0.18)";
            pc.style.filter = "saturate(1.1)";
          } else {
            pc.style.width="30px"; pc.style.height="30px"; pc.style.opacity="0";
          }
          grid.appendChild(pc);
        }
      }
      el.appendChild(grid);
      tray.appendChild(el);

      // layout in tray
      const rect=el.getBoundingClientRect();
      const w=rect.width, h=rect.height;
      if(cursorX+w+10>trayW){ cursorX=10; cursorY+=rowH+10; rowH=0; }
      rowH=Math.max(rowH,h);

      el.style.left=`${cursorX}px`; el.style.top=`${cursorY}px`;

      pieceState[p.id]={ placed:false, x:0,y:0, trayX:cursorX, trayY:cursorY, el, cells:p.cells, hover:null };

      cursorX+=w+10;
      enablePieceDrag(el);
    });

    recomputeBoardOcc();
  }

  function clearGhost(){
    byId("mainGrid").querySelectorAll(".cell").forEach(c=>c.classList.remove("ghostOk","ghostBad"));
  }

  function gridCellFromPieceTopLeft(pieceEl){
    const mainEl=byId("mainGrid");
    const gridRect=mainEl.getBoundingClientRect();
    const pieceRect=pieceEl.getBoundingClientRect();

    const cellSize=34, gap=6, pitch=cellSize+gap;
    const pad=12; // mainGrid padding
    const localX = (pieceRect.left - gridRect.left - pad);
    const localY = (pieceRect.top  - gridRect.top  - pad);

    // round -> much clearer snapping than floor
    const col = Math.round(localX / pitch);
    const row = Math.round(localY / pitch);

    if(col<0 || col>=SIZE || row<0 || row>=SIZE) return null;
    return {x:col,y:row};
  }

  function canPlace(pid, cellX, cellY){
    const ps=pieceState[pid]; if(!ps) return false;
    for(const c of ps.cells){
      const x=cellX+c.dx, y=cellY+c.dy;
      if(x<0||x>=SIZE||y<0||y>=SIZE) return false;
      const idx=y*SIZE+x;
      if(boardOcc[idx]===1) return false;
    }
    return true;
  }

  function showGhost(pid, cellX, cellY, ok){
    clearGhost();
    const ps=pieceState[pid]; if(!ps) return;
    const main=byId("mainGrid");
    for(const c of ps.cells){
      const x=cellX+c.dx, y=cellY+c.dy;
      if(x<0||x>=SIZE||y<0||y>=SIZE) continue;
      const idx=y*SIZE+x;
      const el=main.querySelector(`.cell[data-idx="${idx}"]`);
      if(el) el.classList.add(ok?"ghostOk":"ghostBad");
    }
  }

  function enablePieceDrag(el){
    let dragging=false, startX=0, startY=0, origLeft=0, origTop=0;
    el.addEventListener("pointerdown",(e)=>{
      e.preventDefault();
      const pid=el.dataset.pid; if(!pid) return;
      dragging=true; el.classList.add("dragging"); el.setPointerCapture(e.pointerId);
      startX=e.clientX; startY=e.clientY;
      origLeft=parseFloat(el.style.left||"0"); origTop=parseFloat(el.style.top||"0");
      const ps=pieceState[pid];
      if(ps && ps.placed){ ps.placed=false; recomputeBoardOcc(); }
    });
    el.addEventListener("pointermove",(e)=>{
      if(!dragging) return;
      const dx=e.clientX-startX, dy=e.clientY-startY;
      el.style.left=`${origLeft+dx}px`; el.style.top=`${origTop+dy}px`;

      const pid=el.dataset.pid;
      const cell = gridCellFromPieceTopLeft(el);
      if(cell){
        const ok = canPlace(pid, cell.x, cell.y);
        pieceState[pid].hover = cell;
        showGhost(pid, cell.x, cell.y, ok);
      } else {
        pieceState[pid].hover = null;
        clearGhost();
      }
    });
    const up=()=>{
      if(!dragging) return;
      dragging=false; el.classList.remove("dragging");
      clearGhost();
      const pid=el.dataset.pid;
      const ok=trySnapToBoard(pid);
      if(!ok){
        const ps=pieceState[pid];
        el.style.left=`${ps.trayX}px`; el.style.top=`${ps.trayY}px`;
        ps.placed=false; ps.hover=null; recomputeBoardOcc();
      }
    };
    el.addEventListener("pointerup", up);
    el.addEventListener("pointercancel", up);
  }

  function trySnapToBoard(pid){
    const ps=pieceState[pid]; if(!ps) return false;
    const el=ps.el;

    const cell = ps.hover || gridCellFromPieceTopLeft(el);
    if(!cell) return false;
    if(!canPlace(pid, cell.x, cell.y)) return false;

    // Snap the piece so its top-left aligns to the anchor cell
    const mainEl=byId("mainGrid");
    const gridRect=mainEl.getBoundingClientRect();
    const cellSize=34, gap=6, pitch=cellSize+gap;
    const pad=12;

    const targetLeftPage = gridRect.left + pad + cell.x*pitch;
    const targetTopPage  = gridRect.top  + pad + cell.y*pitch;

    const tray=byId("tray");
    const trayRect=tray.getBoundingClientRect();
    const leftInTray = targetLeftPage - trayRect.left;
    const topInTray  = targetTopPage  - trayRect.top;

    el.style.left=`${leftInTray}px`;
    el.style.top =`${topInTray}px`;

    ps.placed=true; ps.x=cell.x; ps.y=cell.y; ps.hover=null;
    recomputeBoardOcc();
    return true;
  }

  function submitGame1(){
    if(!game) return;
    byId("btnSubmit1").disabled=true;
    byId("submitState1").textContent="Submitted. Waiting for others…";
    send("game:submit", {gameType:"rebuild_6x6", grid: boardOcc});
  }

  // ---------------- Game 2 (tile_3x3) ----------------
  function initTilePuzzle(tileSize, tileOrder, imageUrl){
    const ref=byId("refImg");
    ref.style.backgroundImage = `url("${imageUrl}")`;

    tilePositions = Array.from({length:tileSize*tileSize}, ()=>-1);

    const board=byId("tileBoard");
    board.innerHTML="";
    for(let i=0;i<tileSize*tileSize;i++){
      const slot=document.createElement("div");
      slot.className="slot";
      slot.dataset.slot=String(i);
      board.appendChild(slot);
    }

    const tray=byId("tileTray");
    tray.innerHTML="";
    for(const tileIdx of tileOrder){
      const tile=document.createElement("div");
      tile.className="tile";
      tile.dataset.tile=String(tileIdx);
      tile.style.backgroundImage = `url("${imageUrl}")`;
      const x = tileIdx % tileSize;
      const y = Math.floor(tileIdx / tileSize);
      tile.style.backgroundPosition = `${-x*96}px ${-y*96}px`;
      tile.style.backgroundSize = `${tileSize*96}px ${tileSize*96}px`;
      tray.appendChild(tile);
      enableTileDrag(tile, tileSize);
    }
  }

  function slotFromPoint(x,y, tileSize){
    const board=byId("tileBoard");
    const rect=board.getBoundingClientRect();
    if(x<rect.left||x>rect.right||y<rect.top||y>rect.bottom) return null;
    const pitch=106+10;
    const localX=x-rect.left, localY=y-rect.top;
    const col=Math.floor(localX/pitch);
    const row=Math.floor(localY/pitch);
    if(col<0||col>=tileSize||row<0||row>=tileSize) return null;
    return row*tileSize+col;
  }

  function clearSlotGhost(){
    byId("tileBoard").querySelectorAll(".slot").forEach(s=>s.classList.remove("ghostOk","ghostBad"));
  }
  function showSlotGhost(slot){
    clearSlotGhost();
    if(slot==null) return;
    const el=byId("tileBoard").querySelector(`.slot[data-slot="${slot}"]`);
    if(el) el.classList.add("ghostOk");
  }

  function enableTileDrag(tileEl, tileSize){
    let dragging=false, startX=0, startY=0;
    tileEl.style.position="relative";

    tileEl.addEventListener("pointerdown",(e)=>{
      e.preventDefault();
      dragging=true;
      tileEl.classList.add("dragging");
      tileEl.setPointerCapture(e.pointerId);
      startX=e.clientX; startY=e.clientY;
    });
    tileEl.addEventListener("pointermove",(e)=>{
      if(!dragging) return;
      const dx=e.clientX-startX, dy=e.clientY-startY;
      tileEl.style.transform = `translate(${dx}px, ${dy}px)`;
      showSlotGhost(slotFromPoint(e.clientX, e.clientY, tileSize));
    });
    const up=(e)=>{
      if(!dragging) return;
      dragging=false;
      tileEl.classList.remove("dragging");
      tileEl.style.transform="translate(0px,0px)";
      clearSlotGhost();
      const slot = slotFromPoint(e.clientX, e.clientY, tileSize);
      if(slot!=null){
        placeTileIntoSlot(tileEl, slot, tileSize);
      } else {
        // return to tray: just append back to tray
        byId("tileTray").appendChild(tileEl);
      }
    };
    tileEl.addEventListener("pointerup", up);
    tileEl.addEventListener("pointercancel", up);
  }

  function placeTileIntoSlot(tileEl, slot, tileSize){
    const board=byId("tileBoard");
    const slotEl=board.querySelector(`.slot[data-slot="${slot}"]`);
    if(!slotEl) return;

    const existing = slotEl.querySelector(".tile");
    if(existing){
      byId("tileTray").appendChild(existing);
    }

    slotEl.innerHTML="";
    slotEl.appendChild(tileEl);

    // recompute positions
    tilePositions = Array.from({length:tileSize*tileSize}, ()=>-1);
    board.querySelectorAll(".slot").forEach(s=>{
      const idx=parseInt(s.dataset.slot,10);
      const t=s.querySelector(".tile");
      tilePositions[idx] = t ? parseInt(t.dataset.tile,10) : -1;
      s.classList.toggle("filled", !!t);
    });
  }

  function submitGame2(){
    if(!game) return;
    byId("btnSubmit2").disabled=true;
    byId("submitState2").textContent="Submitted. Waiting for others…";
    send("game:submit", {gameType:"tile_3x3", positions: tilePositions});
  }

  // ---------------- Results ----------------
  function renderResults(results){
    const list=byId("resultsList");
    list.innerHTML="";
    results.forEach(r=>{
      const item=document.createElement("div");
      item.className="resItem";
      const left=document.createElement("div");
      left.className="row";
      const rank=document.createElement("div");
      rank.className="rank"; rank.textContent=`#${r.rank}`;
      const name=document.createElement("div");
      name.innerHTML=`<div style="font-weight:750;">${escapeHtml(r.name)}</div>
                      <div class="muted">${r.accuracy}% · ${(r.timeMs/1000).toFixed(2)}s</div>`;
      left.appendChild(rank); left.appendChild(name);
      const right=document.createElement("div"); right.className="right";
      const tag=document.createElement("div"); tag.className="tag";
      tag.textContent = (r.playerId && playerId && r.playerId===playerId) ? "You" : "";
      right.appendChild(tag);
      item.appendChild(left); item.appendChild(right);
      list.appendChild(item);
    });
  }
  function escapeHtml(s){ return String(s).replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[m])); }
</script>
</body>
</html>
""".strip()


@app.get("/play")
async def play() -> HTMLResponse:
    return HTMLResponse(PLAY_PAGE)


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"ok": True, "ts": now_ms(), "images": len(list_puzzle_images())})


@app.get("/games")
async def games() -> JSONResponse:
    return JSONResponse({"games": GAMES})


@app.get("/public-rooms")
async def public_rooms() -> JSONResponse:
    rooms = await manager.list_public_rooms()
    return JSONResponse({"rooms": rooms, "ts": now_ms()})


# -----------------------------
# WebSocket helpers
# -----------------------------

async def ws_send(ws: WebSocket, msg_type: str, payload: Dict[str, Any]) -> None:
    await ws.send_text(json.dumps({"type": msg_type, "payload": payload}))


async def room_broadcast(room: Room, msg_type: str, payload: Dict[str, Any]) -> None:
    dead: List[str] = []
    for pid, p in room.players.items():
        try:
            await ws_send(p.ws, msg_type, payload)
        except Exception:
            dead.append(pid)
    for pid in dead:
        room.players.pop(pid, None)


def room_state_payload(room: Room) -> Dict[str, Any]:
    return {
        "roomId": room.id,
        "targetPlayers": room.target_players,
        "gameType": room.game_type,
        "count": len(room.players),
        "status": room.status,
        "players": [
            {"id": p.id, "name": p.name, "submitted": p.submitted_at_ms is not None}
            for p in room.players.values()
        ],
    }


# -----------------------------
# Game lifecycle
# -----------------------------

async def start_countdown_if_ready(room: Room) -> None:
    if room.status != "waiting":
        return
    ready = (room.target_players == 1 and len(room.players) >= 1) or (len(room.players) >= room.target_players)
    if not ready:
        return

    await manager.mark_room_not_joinable(room)
    room.status = "countdown"
    starts_at = now_ms() + 3000
    await room_broadcast(room, "room:countdown", {"roomId": room.id, "startsAtMs": starts_at})
    await room_broadcast(room, "room:state", room_state_payload(room))

    async def _countdown():
        try:
            await asyncio.sleep(3.0)
            await start_game(room)
        except asyncio.CancelledError:
            return

    room.countdown_task = asyncio.create_task(_countdown())


async def start_game(room: Room) -> None:
    if room.status not in ("countdown", "waiting"):
        return

    game_id = secrets.token_urlsafe(10)
    seed = secrets.token_hex(8)
    started = now_ms()
    ends = started + 60_000

    gt = room.game_type
    session = GameSession(game_id=game_id, game_type=gt, seed=seed, started_at_ms=started, ends_at_ms=ends, finished=False)

    if gt == "rebuild_6x6":
        ref = generate_reference_grid(seed, size=6)
        session.size = 6
        session.reference_grid = ref
        session.pieces = split_into_pieces(ref, size=6, seed=seed)

        payload = {
            "roomId": room.id,
            "gameId": game_id,
            "gameType": gt,
            "seed": seed,
            "endsAtMs": ends,
            "size": 6,
            "referenceGrid": session.reference_grid,
            "pieces": session.pieces,
        }

    else:
        session.tile_size = 3
        session.image_url = pick_image_url(seed)
        session.tile_order = _shuffle_det(list(range(9)), seed)
        payload = {
            "roomId": room.id,
            "gameId": game_id,
            "gameType": gt,
            "seed": seed,
            "endsAtMs": ends,
            "tileSize": session.tile_size,
            "imageUrl": session.image_url,
            "tileOrder": session.tile_order,
        }

    room.game = session
    room.status = "in_game"

    for p in room.players.values():
        p.submitted_at_ms = None
        p.submission_grid = None
        p.submission_positions = None

    await room_broadcast(room, "game:start", payload)
    await room_broadcast(room, "room:state", room_state_payload(room))

    async def _timer():
        try:
            remaining = max(0, (ends - now_ms()) / 1000.0)
            await asyncio.sleep(remaining)
            await finish_game(room, reason="timer")
        except asyncio.CancelledError:
            return

    room.game_task = asyncio.create_task(_timer())


async def finish_game(room: Room, reason: str) -> None:
    game = room.game
    if not game or game.finished:
        return
    game.finished = True
    room.status = "results"

    if room.game_task and not room.game_task.done():
        room.game_task.cancel()

    # Fill missing submissions
    for p in room.players.values():
        if p.submitted_at_ms is None:
            p.submitted_at_ms = game.ends_at_ms
            if game.game_type == "rebuild_6x6":
                p.submission_grid = [0] * (game.size * game.size)
            else:
                p.submission_positions = [-1] * (game.tile_size * game.tile_size)

    results: List[Dict[str, Any]] = []
    for p in room.players.values():
        t_ms = max(0, (p.submitted_at_ms or game.ends_at_ms) - game.started_at_ms)

        if game.game_type == "rebuild_6x6":
            sub = p.submission_grid or [0] * (game.size * game.size)
            acc = compute_accuracy_binary(game.reference_grid, sub)
        else:
            pos = p.submission_positions or [-1] * (game.tile_size * game.tile_size)
            acc = tile_accuracy(game.tile_size, pos)

        results.append({"playerId": p.id, "name": p.name, "accuracy": acc, "timeMs": t_ms})

    results.sort(key=lambda r: (-r["accuracy"], r["timeMs"]))
    for i, r in enumerate(results, start=1):
        r["rank"] = i

    await room_broadcast(
        room,
        "game:results",
        {"roomId": room.id, "gameId": game.game_id, "gameType": game.game_type, "reason": reason, "results": results},
    )
    await room_broadcast(room, "room:state", room_state_payload(room))

    # Reopen room after a short pause
    await asyncio.sleep(1.0)
    room.status = "waiting"
    room.game = None
    await manager.mark_room_joinable(room)
    await room_broadcast(room, "room:state", room_state_payload(room))
    await start_countdown_if_ready(room)


async def handle_submit(room: Room, player_id: str, payload: Dict[str, Any]) -> None:
    game = room.game
    if not game or room.status != "in_game":
        return

    player = room.players.get(player_id)
    if not player or player.submitted_at_ms is not None:
        return

    submit_time = min(now_ms(), game.ends_at_ms)

    if game.game_type == "rebuild_6x6":
        grid = validate_binary_grid(payload.get("grid"), size=6)
        if grid is None:
            return
        player.submission_grid = grid
        player.submission_positions = None

    else:
        positions = validate_positions(payload.get("positions"), size=3)
        if positions is None:
            return
        player.submission_positions = positions
        player.submission_grid = None

    player.submitted_at_ms = submit_time
    await room_broadcast(room, "room:state", room_state_payload(room))

    if all(p.submitted_at_ms is not None for p in room.players.values()):
        await finish_game(room, reason="all_submitted")


# -----------------------------
# WebSocket endpoint
# -----------------------------

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()

    player_id = make_player_id()
    player_name = f"Player-{player_id[:4]}"
    current_room: Optional[Room] = None

    try:
        await ws_send(ws, "session:hello", {"playerId": player_id, "name": player_name})

        while True:
            raw = await ws.receive_text()
            msg = safe_json_loads(raw)
            msg_type = msg.get("type")
            payload = msg.get("payload") or {}

            if msg_type == "training:join":
                target = clamp_int(payload.get("targetPlayers"), 1, 10, 10)
                game_type = payload.get("gameType")
                if not isinstance(game_type, str) or game_type not in GAME_IDS:
                    game_type = "rebuild_6x6"

                name = payload.get("name")
                if isinstance(name, str) and name.strip():
                    player_name = name.strip()[:24]

                # Leave old room if needed
                if current_room is not None:
                    old = current_room
                    old.players.pop(player_id, None)
                    await room_broadcast(old, "room:state", room_state_payload(old))
                    await manager.mark_room_joinable(old)
                    await manager.maybe_cleanup_room(old)
                    current_room = None

                room = await manager.assign_room(target, game_type)
                current_room = room
                room.players[player_id] = PlayerConn(id=player_id, name=player_name, ws=ws)

                await room_broadcast(room, "room:state", room_state_payload(room))
                await start_countdown_if_ready(room)

            elif msg_type == "room:leave":
                if current_room is not None:
                    room = current_room
                    room.players.pop(player_id, None)
                    await room_broadcast(room, "room:state", room_state_payload(room))
                    await manager.mark_room_joinable(room)
                    await manager.maybe_cleanup_room(room)
                    current_room = None
                await ws_send(ws, "room:left", {})

            elif msg_type == "game:submit":
                if current_room is None:
                    continue
                await handle_submit(current_room, player_id, payload)

            elif msg_type == "ping":
                await ws_send(ws, "pong", {"ts": now_ms()})

            else:
                await ws_send(ws, "error", {"message": f"Unknown message type: {msg_type}"})

    except WebSocketDisconnect:
        if current_room is not None:
            room = current_room
            room.players.pop(player_id, None)
            await room_broadcast(room, "room:state", room_state_payload(room))
            await manager.mark_room_joinable(room)
            await manager.maybe_cleanup_room(room)
    except Exception:
        if current_room is not None:
            room = current_room
            room.players.pop(player_id, None)
            try:
                await room_broadcast(room, "room:state", room_state_payload(room))
            except Exception:
                pass
            await manager.mark_room_joinable(room)
            await manager.maybe_cleanup_room(room)
        try:
            await ws.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
