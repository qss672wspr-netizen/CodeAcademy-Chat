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
<div id="wm"></div>
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
async def chat_index():
    if CHAT_INDEX.exists():
        return FileResponse(str(CHAT_INDEX))
    return HTMLResponse(CHAT_FALLBACK_HTML)


@app.get("/chat")
async def chat_alias():
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
            --stroke:rgba(255,255,255,0.12); --good:rgba(80,245,255,0.9); --bad:rgba(255,110,220,0.9); 
            --wm:url('data:image/png;base64,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AAAgQwgAAAAAQIYQAAAAAIAMIQAAAAAAGUIAAAAAADKEAAAAAABkCAEAAAAAyBACAAAAAJAhBAAAAAAgQwgAAAAAQIYQAAAAAIAMIQAAAAAAGUIAAAAAADKEAAAAAABkCAEAAAAAyBACAAAAAJAhBAAAAAAgQwgAAAAAQIYQAAAAAIAMIQAAAAAAGUIAAAAAADKEAAAAAABkCAEAAAAAyBACAAAAAJAhBAAAAAAgQwgAAAAAQIYQAAAAAIAMIQAAAAAAGUIAAAAAADKEAAAAAABkCAEAAAAAyBACAAAAAJAhBAAAAAAgQwgAAAAAQIYQAAAAAIAMIQAAAAAAGUIAAAAAADKEAAAAAABkCAEAAAAAyBACAAAAAJAhBAAAAAAgQwgAAAAAQIYQAAAAAIAMIQAAAAAAGUIAAAAAADKEAAAAAABkCAEAAAAAyBACAAAAAJAhBAAAAAAgQwgAAAAAQIYQAAAAAIAMIQAAAAAAGUIAAAAAADKEAAAAAABkCAEAAAAAyBACAAAAAJAhBAAAAAAgQwgAAAAAQIYQAAAAAIAMIQAAAAAAGUIAAAAAADKEAAAAAABkCAEAAAAAyBACAAAAAJAhBAAAAAAgQwgAAAAAQIYQAAAAAIAMIQAAAAAAGUIAAAAAADKEAAAAAABkCAEAAAAAyBACAAAAAJAhBAAAAAAgQwgAAAAAQIYQAAAAAIAMIQAAAAAAGUIAAAAAADKEAAAAAABkCAEAAAAAyBACAAAAAJAhBAAAAAAgQwgAAAAAQIYQAAAAAIAMIQAAAAAAGUIAAAAAADKEAAAAAABkCAEAAAAAyBACAAAAAJAhBAAAAAAgQwgAAAAAQIYQAAAAAIAMIQAAAAAAGUIAAAAAADKEAAAAAABkCAEAAAAAyBACAAAAAJAhBAAAAAAgQwgAAAAAQIYQAAAAAIAMIQAAAAAAGUIAAAAAADKEAAAAAABkCAEAAAAAyBACAAAAAJAhBAAAAAAgQwgAAAAAQIYQAAAAAIAMIQAAAAAAGUIAAAAAADKEAAAAAABkCAEAAAAAyBACAAAAAJAhBAAAAAAgQwgAAAAAQIYQAAAAAIAMIQAAAAAAGUIAAAAAADKEAAAAAABkCAEAAAAAyBACAAAAAJAhBAAAAAAgQwgAAAAAQIYQAAAAAIAMIQAAAAAAGUIAAAAAADKEAAAAAABkCAEAAAAAyBACAAAAAJAhBAAAAAAgQwgAAAAAQIYQAAAAAIAMIQAAAAAAGUIAAAAAADKEAAAAAABkCAEAAAAAyBACAAAAAJAhBAAAAAAgQwgAAAAAQIYQAAAAAIAMIQAAAAAAGUIAAAAAADKEAAAAAABkCAEAAAAAyBACAAAAAJAhBAAAAAAgQwgAAAAAQIYQAAAAAIAMIQAAAAAAGUIAAAAAADKEAAAAAABkCAEAAAAAyBACAAAAAJAhBAAAAAAgQwgAAAAAQIYQAAAAAIAMIQAAAAAAGUIAAAAAADKEAAAAAABkCAEAAAAAyBACAAAAAJAhBAAAAAAgQwgAAAAAQIYQAAAAAIAMIQAAAAAAGUIAAAAAADKEAAAAAABkCAEAAAAAyBACAAAAAJAhBAAAAAAgQwgAAAAAQIYQAAAAAIAMIQAAAAAAGUIAAAAAADKEAAAAAABkCAEAAAAAyBACAAAAAJAhBAAAAAAgQwgAAAAAQIYQAAAAAIAMIQAAAAAAGUIAAAAAADKEAAAAAABkCAEAAAAAyBACAAAAAJAhBAAAAAAgQwgAAAAAQIYQAAAAAIAMIQAAAAAAGUIAAAAAADKEAAAAAABkCAEAAAAAyBACAAAAAJAhBAAAAAAgQwgAAAAAQIYQAAAAAIAMIQAAAAAAGUIAAAAAADKEAAAAAABkCAEAAAAAyBACAAAAAJAhBAAAAAAgQwgAAAAAQIYQAAAAAIAMIQAAAAAAGUIAAAAAADKEAAAAAABkCAEAAAAAyBACAAAAAJAhBAAAAAAgQwgAAAAAQIYQAAAAAIAMIQAAAAAAGUIAAAAAADKEAAAAAABkCAEAAAAAyBACAAAAAJAhBAAAAAAgQwgAAAAAQIYQAAAAAIAMIQAAAAAAGUIAAAAAADKEAAAAAABkCAEAAAAAyBACAAAAAJAhBAAAAAAgQwgAAAAAQIYQAAAAAIAMIQAAAAAAGUIAAAAAADKEAAAAAABkCAEAAAAAyBACAAAAAJAhBAAAAAAgQwgAAAAAQIYQAAAAAIAMIQAAAAAAGUIAAAAAADKEAAAAAABkCAEAAAAAyBACAAAAAJAhBAAAAAAgQwgAAAAAQIYQAAAAAIAMIQAAAAAAGUIAAAAAADKEAAAAAABkCAEAAAAAyBACAAAAAJAhBAAAAAAgQwgAAAAAQIYQAAAAAIAMIQAAAAAAGUIAAAAAADKEAAAAAABkCAEAAAAAyBACAAAAAJAhBAAAAAAgQwgAAAAAQIYQAAAAAIAMIQAAAAAAGUIAAAAAADKEAAAAAABkCAEAAAAAyBACAAAAAJAhBAAAAAAgQwgAAAAAQIYQAAAAAIAMIQAAAAAAGUIAAAAAADKEAAAAAABkCAEAAAAAyBACAAAAAJAhBAAAAAAgQwgAAAAAQIYQAAAAAIAMIQAAAAAAGUIAAAAAADKEAAAAAABkCAEAAAAAyBACAAAAAJAhBAAAAAAgQwgAAAAAQIYQAAAAAIAMIQAAAAAAGUIAAAAAADKEAAAAAABkCAEAAAAAyBACAAAAAJAhBAAAAAAgQwgAAAAAQIYQAAAAAIAMIQAAAAAAGUIAAAAAADKEAAAAAABkyP8PXjWcfcLPcTwAAAAASUVORK5CYII='); --wmOpacity:0.12; }
    *{ box-sizing:border-box; }
    body{ margin:0; background:radial-gradient(1200px 800px at 50% 20%, #0b1330 0%, var(--bg) 60%);
          color:var(--text); font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }

    /* Watermark */
    #wm{
      position:fixed;
      inset:0;
      background-image:var(--wm);
      background-repeat:no-repeat;
      background-position:50% 54%;
      background-size:min(980px, 92vw);
      opacity:var(--wmOpacity);
      pointer-events:none;
      z-index:0;
      filter:none;
    }
    header, main{ position:relative; z-index:1; }

    /* Brand */
    .brandIcon{
      width:34px; height:34px;
      border-radius:12px;
      border:1px solid var(--stroke);
      background:rgba(255,255,255,0.04);
      padding:6px;
      display:grid; place-items:center;
      box-shadow:0 10px 22px rgba(0,0,0,0.25);
    }
    .brandIcon img{ width:100%; height:100%; object-fit:contain; display:block; }
    .brandText{
      font-weight:900;
      font-size:18px;
      letter-spacing:0.6px;
      background:linear-gradient(90deg, rgba(80,245,255,0.95), rgba(255,110,220,0.95), rgba(255,209,102,0.95));
      -webkit-background-clip:text;
      background-clip:text;
      color:transparent;
      text-transform:none;
    }

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
  <div class="brand"><span class="brandIcon"><img alt="HestioPlay" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFAAAABQCAYAAACOEfKtAAAbQElEQVR4nO2ceZRdR3ngf1V1l7f0ou6W2lJLtmVbsmzLGzbyhgE7YIMdwGwJgSGBAMkZBmLgsAWGhDNAwpZAyEIIHIbxJDAESEjAhrHBYNkQxwuyMV5lWZZk7d1q9fL6Lffeqm/+uG+5b+lWt92C+YPvnPuWu1XV735bLe/Br+UZiTrWCYWBMVneIrO3O2bxx+H6pUl5Zv+ChXjHvQZtMn/jF6pl+xNUmb1yjCuPv/wSATYwtBrcq+mCQitQKj3uRNoItj4q/n+A+EsCuBA8lR6t79AKYgu1moAIQajxDSDSvK43xPa7/rJEH/8iMo1TzU+k4BSimjswGioJjA0YPvfWYb79kTVcuF5RjR1Kp+eDamH61VovcNw1MGN7dWVpwaMFQKXw5mrC5tU+X3rHSi7eHEDO57Y7D3PnowkqF4IIIilCRfoZperF/GrM+ThqYFbzVDu8jNal8BTlCDavDvinD4xy8RkhlXGHnbOMDHQ8YyU0jFk1ilFkbrjMScMx5DgB7DDbTs1rHK/Dq8XCaSMeX3n3Ss490aM24/ADhVGweV2enKdT822AakLkVw7xOGpgvTFdZltvrQKjFJFVrCxqvvTOlVy4MSAuOcKihxOgYtl8asBArkZiQbdBpO4Tl5IQLb8cB4AZP6QyOtE029SkNQqnQCcJn3nLEM87zyeZdYin+Pg/HWL3tAMRxtYEbFhjiGKL0qn3y5p/E6Jk0XVG5+MnywywPWgoaexV7fsB7WnmZmOuv9rwmufnqE4keEXFR2/Yywe/upfHdpUg0PQbw4suOgFQqOZzab+f9NS6X445LyPAdr/XBq+tvQqjFTOlhCs2wZ++eQ1JScgN+3zlOwf5xE1lTHEl3/7pFE4JiOJlWwqs7tPEFlSdYuO9UWQ2xWnXxOOrjcsEcAF4GVEq1aLYKVaFZT751hHynsHLG+69b4r3/K+jhP2DhDnD9+6vsGN/FXGKczcEXHl2jmok6EY3hSxEaYvOrddmyZl6Li/IZQDY3gtQzfqp9rd6Y7XWVGZLvPsVg1x09gBxJWFyKuLdXzrMlCriK8E3ikNzId/8wRTK1+hYeOML+hgIwEo7nPS+9a2REmYgdmtjts7PXJ4hwA54zf0diXKjMK0o1yznr3O8+dqV2ElL0Kf5zFcPcPsun4FigHUgTgjyAf/7jpi9BxPEKq48v8irLy5QrjqMaWkhtDQ7/SJ1PWsBy8av5TbrZwDwWPAyPdam91e4KOFdr1zFypEAHSjueaDE329NKA7ksNYhSnAIoa/ZOW345u0lVKjQVnj7ywZZO2CIrKCVAqUzJTcg1n1uG8j6GW0dleUx66cJsMPnNfe353pKZU1XUakJF54ML33uAK5kUYHiCzdNMRmH+BoQhapvOPB9xedvmWX7nhooOP+UAn/y8iHiyKF0XQlVVhtVRhtVM7cRFF3pTlvinW3T0uRpAOwdMHrByzhAlDbYyhy/d1XI0EiA8jQPPFri23fN0VfwcVbayhAnBFrx5JTwmW/PICjsnOUNL+zjuvOKzFYcpg5RNbqLXSB1s6q9QLabdLZti5clAlwsvEyllEYrRWwda0Y8XrhlBW4qQYWKf7ljhqNxHqMFJyAi9S0tyTohHyi+dmeFH95TxuQVPoo/f90I557gUY3T4S+U1IGlZt38THe605l8N/c/TYhLADh/qtIKGKrN36WNAa2hWrNceLJi42geFBwZr/KdeyKCXIi4XpVOfZNWQqQUH/7WUfYeTsAJp6/1+dTrVlEwQuJUGySlwKIQpTHGNGEulDc+k770IgEeI0nuSFUaGqjqtRNSv/aiLYMYA9rXbL2vxEMHhFygaefX7tSdQM5T/OeumA999QhWa6Ky5eoLC3zy1atQ4tL7103YidAnc3hxmZlyBacUxpi03lmQqt69fIYQFwFwkT2MBjytSKTljhQKK4rRfsPFZxQgARHH1ofLxAQN/e0ClxXrhIGC4f/cXebTX58kyIVEs463XLWCD7x4mDhOE2yjNeVawm+du5IfXP9s3rVlJXlbY7YWY7Tuada9IXa2fX55WlF4PnjaGGrViN89X1iVj4idRmtNYoWV+TnGhtLoWklg9yGD8XVzqL690t2bOEcQGD72/SN85aYDBP0+SSXh/det4rWbYbpUxjMa43lMVzVbThrmM6++gJt+/wKu3ThEzdp6MGtYxwIQOxPYBWSRADuDRjc8zzfMztR40/NCPvb7J6BdWmGt0zmOtSMeo30eaMXRCcujhzS+p3FNaJl3Ue1b/YhGEN/n3V/fx+13T+AHPjp2fPL16zl7GGYjR+h7PHw04eBEjaTsuHjdKr7xui2886KTSdogdmhiq6U9UpxnBHB+023C8wzTsxEvOtPxF28bo5jTFEKdDr9rjYsTTl5t8Io+KM3eg5bDJYenJGMl0gaszaTr+5wIoacoeYO88Yt72bmvilKa0aE8f/VfTqUQz4HWHJyL2TNVwwtCooqQj4RPXLmZj1x+OkmcpJaqdTvETHReSiBenAYuAE9rTSWynDIwx9+8c4w+rTBoRsIIaxO08cDGjORJw7ESnpqSNAXRvepaN1lpWHcdZB2idY5iaHiynOP6r+yk5iCpWF5wzkr+8LIByuUqTmByLgY0nqdQRhOXE9576SY++rwzcc62tSELsWnKi9TCYwPMursOeI0cxVXKfPota9m4tkhtOiFX0JzQb9LcLqXMyv4QRINTTJSEWJpVzQDKgqsfyYKsQ0ysY7AY8L0d8Ff/tg8vCLEz8L5rN3P5yQOUIktUs+AU1bJi/5TFD0PiiuU9l53De7ZsIrZxqoWqHVLWIBZjxccGOM/QFErhGc3sbJU/uKLIK58zRHI0wWgNnmLViiCtQ32G3K/7HRxMzrpM6jKfvWSDSPe5zgmFYo6/vHWc+x6bwRifkULIH1x8AkYcRd8Dq/B1nj+/41FufPIgfn+OuBTx3y+9gNecsZ5yFGE6/WFXQFlYjglQNV6z2qcUWimqsbBpOOGDrx2FqqTOOc1k2bi+gK9aT7hSc80KJT3hSUbzuiNz81hdGwXB1zBhfT51026s00g14aVnruL5Jw4wks9DovBzIWI83vTdn3HXngl8P8SPhT97/iWcu2oF5diite4oLdv2hWVRPrDXg1BakVRrfOi3h1k3liOxDovCiQKBDetyBDqpR1mYKsWQAMowlNfoeWsnrbc2BeyECNY6+vI5btlR5ue7SiiVZ1CHfPS557BhcDUu8RDryBvFeE14180PcKDsEBWwLhjkQ5dcSsHLAOvQwsXIkvLARgFGK+YqCVtOjnn584ZwsxYVaL55y2HmKhZEcc6Yz9gAxImgwxwTFQ9iAQenDnrkjGqZsXRUNhs86IjUrRYiCJ5STCaab961F4yHjRXnrVlF0WisKJTAzpkqoR+wbarEx356P+IFRLWYl63fyEtOOplSrYrRprsKywGwM2lG1fuZtsY7rxuhL++hPc2Dj5f44Nf28Nj+CpII69YWueacPJVyhDGGydkEagKJcOKIRyEA6zIomoGiV/VbEKXjuDiHF4bcsmOS6QmHpwpEiaB0gD84wH8enOLeQ1P4RhP6Pl/bvpNbn3yKIOhDx4a3nXcRw0YRi0NlwtpiZck9Ea2gElkuOLXINRcM4+YsBPClH86wp9zH97dNokKDspprLwjISxm0ZnImplZ14GBopc9gPu23tkHq+NYeQlpwswydCDnPY8fRMr/YV0FJDqU8bnnyMO//3t3815t/xmSUaqoGyqL57LafMx0JiYXL1m7gJes3UKlVUl/YKy98RgB7pS4WXn/pIAOBQRnNjifnuOmBhGBomO/f7zhyuIrUHJdtWsHmUUUcJxw+GjFZtYCiWNCMDRicpFVst1DppX8dO1poBcEAJV3gvn1HAYMvIXum5vjUXY/w+EyErzUighMh7/n89OAhfrBnN0FYQGLhtzaeRyCu6a+XIkv2gZEV1g9arr0kBAEVKL71kzn2TCv6+zzuO2S4ZdscKtD0Fz3++BUrKRBTSjxmSilA34NAxVgrHRabTQBpU8EGrt5ZjSBK8+D4DCQaRHPe0DCr8n6aPmWAa6DqFP/44P1UncU5x3PWn8Wzx9ZTSWrpVMESZEkAtYIoEjaeYDlx0EOcolS2bH08nQSSJEEFIf++LSGuOWzF8rJLhnjTpX1M1PKUamlXTYDK3BwijWeeZsvNZs6jgm3sMieKpKMx+8oRlQgQw1CQoyCQOJfet37zhhbesX8vPz9yGK19Vvh9XHnKWThnUz/YFo2fMcBMtZVG4hqXb8jj5X1UoNi+K+a+PZbAKKyDQqC48eGIf75tFlMwuIrwvuuG+Z2zAvLaQKJxsSJuDuFLz+JUZms/rSPQ1D9rpZipxVRiDRZG8v2sKQ6SONcWGkQETxumahV+9MTjKL8PVxMuGjmF/lyAlcV4vpYsbjCh/mpF6A89nvusFeAAX3P7oxHjVcHUR6aUCM4YPvLdGR7ZHWMCn7FcwBffvJbTh3JIVVOZEyYqaU9GpKV97SX2/i69XiU9Z6bmiKqAGDwdkPdDxLmMO6hf4RzaD7ll13Zm5wSRkHNXrmdV0E8idgn4FmnCjfmaKBHWDAqnnpiHWBALP3sqTk9KJzVwTsgZYdes8LYvj3P4iMMEOk2yrUblPB7cUeXAnE9gVHP+I0Mk86WjK9dl2i2z10AptpQTC2LwlcHgcOLa7iMCThyhCXh48gA7psZRymM0P8wFq9cT2aQ1GbVcAAFQCmuF0SHDUOBBrKiUYd9EglEqMyEkJFYo+oo7dke8/vP7efywww8DgtBwdMry2ZsPUpXGKtN2B6e6+r/pwS4t7DkQC2I9cAFiA8Rm7i0tvymApw0zNuGp0hxK9RFKjlMGxhCxSzLhRS/xVUrhkoixAUNee4AwMZ1weNpiMovAGxW2AsVA8+PdVa773B5efk4/2Co3PzTNg+NCIfSxrtXA1rDpfCLUk7Num86kWp6fA/FIEsEqv54mSXtyJILWmsg59pZLiM6hnGXQ66fxCBcrS1sjLY41/TmUMqAtsxVhrurqK3hVpreQfrAiFHzFzqMxH791ApzF8wy5UOFcQy3UIuClonqelRJ0IvT7PjlTAPGo2pjZWiU1sZ4JuyDa43BUQZRBoRjND+CZFKBSPZS8hyx5kXne98F6oIRqTZNELo3O0jmaUm+Yg8AoQqOB9LyW5vWA12k/8zSiUxGtOIaCgJzOgwo4Uj3Coco0pl63XjcQEWYrs6AMojTD+UECM8/588iSu3J+YxhZKWqRkDT6s81ktQVPJDUd51Jo1rmFzTYzot4cSO01S9YRdVQ9MJzUv4KCDoEc+6ZnmKjMpQB7tKNh0rUkwtanXT38JfeGF6WB2QoYbcAaQHBO4ZDMZFNLs5o+R1rv2dN6weupKNLOMGvGIpkBZefYMDSGb4pgYc/sDDWXECiVpjKd7VHp9ILCQ0mAFcvTmaRc8hUiKgWYaFBefeis0SS1dHj1bHkhq+l9qLXXkbqJ04bWgKRTCQ/NTCDap1GbHkkRiJDzCmgVIuIRWYdbUghZCsD6fcvlBCyAomg0vs6O67F0eCwM71iigEQc6/r7OXNwDJfAnIN7ntqOrqdX8zVIoSiGA1gUKMORqEKU2Pp1iyt/0V05EYf2PUoqh3Maqj5FZdCummolWXiNS3slwceC1ysP7LWnNZdRi8psyo9wet8Y4jTbJw6xvTJF6Id1gNJ1qRMhFxhOG1qHJBqrNPtKk1jrlrsrRysFEzg4KySiIVIMBB79BQ9rMz6mo8LZqncF2J49j15Xdp6X/abQzvHSUy8g7xdAhfzgwHbGo0o6jNV8YNLc0gTeERiflYURYic4NIdmxhcot7cs2oRTh60Yn42oRBqxmv4gx0h/Husa3aVjtbmlfc/UbBvuIHaWkbDA5WvPxcYwK/Cj/Y/QPkGfqUMaQUicZbSwghNyJ5CIppwodk4+hVJqnsffW5YAMF38vf9IjfHJBCQgb3KMFQzOJu3Lx+qFL6R9nec+HdFKUYsrbBndwPq+NShtuO3AI/zH/sfJm1zHiHerGI0ishGnr1jPycX1kHhMlso8Mb0H3/jLnQe23L6nFROVhD3jNZTSaDyeNbYC5boStuMuijT6htbxprOuoRgUmbPCPz66lYpNOgZGe5i9cpw/eg45v4gYwwNHHmN/ZQLfBLTGppdjSF/qL5IOqJZiy127ZoEcVH1ecOZJDAZSH3f75YlSikoScc7Iei4a3Yjgc/OTD/H9J+6jYPL1UZge16FwYukP+rh87DlUkgSjQ+49eD81W8Uo3dbmY8niV2fVxQncu3+WWhVsBBtWDrFpOKQWd5pxhw+a947d3459fipaKSpJDWM8Sk74i3u/QU1chx/rKEkpajbi1MF1nDF4OrETyknC3YcfQAf5JZkvLKInks38nQiBr3n40AxPlRJOKeQY1AG/ecZ67vrxw6hcQPfMjAIliDQmDaXVkZWO8yCzs3MpSfv5ToRQe2wvHeL6rV8gND4/m3yCQm6grn29Hma9BIn5zZOvZjAYIrKWByYeZNvEQ+S8PC4zoLpsKxOygSzQiicmK/z0sSmMyuGqmtddcDYnDeSJEtsjmMx/z96ntg3kp3t6D8MgCH5Q4F/33MPXnryTXG4gM8/SfXYatSM2DJ3Iq057BeVahK8D/u+TtzKTzOJp07WQ6ViytDmR+scExTce2s1cDWqR47SBFbx043qq1WrbOpMWTNUcymuDI63R7vnkmM9DhKJfoC8oLACvcS9NbGv8zmmvZnV+HQma/bOH+fcdNxL4hTa/uVh/vrgh/bb6CnnfY+vOfWzdMU4uN0ASaf7oki2syxlqNkl/ENisdPuNekHMguzcujsm3f7NNYfuF2io0lSTCmcOb+S3N72GclKj6BX41qP/yhOzu8l5OZzrjL7Llge2IpIIGJX2Nz/7k/spx4Y40WxatZa3bbmcWrmUzik0GakFIHaDnL/nn17T3NVrvfE8olA4oOD7fPCi9zGcXw3asH9uH/+y5986tK/R1u4J/l6yiLUxjUq0xDpHMZdn64HD3PjoLvJBH7MzNf7wghdw3elnU4oq6TrBTBM6WpTh0+3zekvmnMyP3lTXscwUZv1da0OpfIQ3n/4Grhy7iqlolrxX4PP3/QMPH9lO2KF9y94XlsZrpjukRMAP+fRd29h1JCHQ/fSrgE9f/XrOXTXGXBRhTLs/bNdEmQdk7615Xie8BZykAjzjMz17kBev/Q3e/qx3MFmZpd8UuH3Pj/jqI98gNMVM5M1Y2mLAsNSlHZk7p2mE4ecT43z8zp/gpEglgvXhKj5zxRtZXSxSTZIuTWwLLI23dBhxXitO/29B2uBlb9H63r7DaI+pucM8d81z+OzVf0dNNIGXY9/sXv749g9SMw5Pe60RmyXMzzTkaf5WLpXG/OoNj9zH/3zwPvK5lRydi7l49Az+9oVvZyQXUk6idBS7zcpUb8VpUzrJQJPMgdY95jNdVf910nTpMFeOXs7fXfNllC6QIDgV894fv5ed5b3kvXx9JLpVfHcrF5bFjwc2C2k35bRtHv/jru9yy76d9BXHmC7HvPjES/jCCz/AiX1DlG0ZrUxXT6UBcuF8sBtcV66Z+W60h3WWmblJXr3hVfzNS76M5/cTEZMPPf70tvez9cAdDORWkLgk023rhW85ozBZiK0vIoKvNdPVmHf88IvcOf4EfYVRxksVLl9zETdc80mev+48ysksDjDG6zC9FFD2R4HtqUz7sXZp0df1H1/PVI4SWvjwpR/mE1f/LTEBThwDYYEP/+gDfP2xf6a/MEzi4owitEx3KUNZTRYLSWFgTWsQL3NR58pVrTXVJOLkwSH++qr3cdno5YzPztCf66Mijn+4/wa+sO0GpuMKxdwgRmmc2PramEXVtaPmdahonDiqtkzB93neCVdw/ZZ3cfqqczlSnmYg6KMSH+VDt72X7+64kf7CUGq2mTJVz7Ql/VaeObAgo0UAHJMO3cv4inaIRmsqScTqviJ/cvn1XHfqy6lVIxIUhXCQe/Zt4+9/9kVu23sHFRsRBHlCL5c+EHHNadBe1WyU3PB9IkLsImJXI+/luHjtxbzh7Ddy2dhvkKCZiyusCofYPv5zPnDrO7l3Yhv9hZXYrOYdAx6oY/6D5SIBNm7aroXp3m5NjJ0lMI7fPftV/NH5b2UwGGayWsL3B9E4th26h5se+w5bd9/BntIBRGt8L8A3AVqli71VZr0NUIfriG2ElYTA81jbP8alay/jqlOvZcvq5+BTZLo2SzHM46P45i9u4HP3/iXjtQmKuUFs0+e1OKkufO3KsswA2y+bD6LS6c8dIjvHhWOb+W8XvoUrTrqKyHpUoho5r49C4PPk5E7uPvAf/OLQL7h7z93sm9tPzUVYF+PEobUmzIUgCl+nyzZOGzqFc0efxcbhTZw9+mxOHTqdaiJUogq+9ikEAY8cvJMv3P3XfH/XzeRyffgmyMBLFSE7l93b7y07wGwBx9bENAAYqkmVnO/xyjNfwmvP/D02rzqPcuSYjdJF3cUgj2dS579v6in2TO2h4kpU4zLaGAYKK9AqYDAYYrSwmpH8CfSF/dQiIRJLIo7Q+BSNZsfRh/na/V/mO9u/zbQt0ZcbTF1DYx1O48FLqy3SaIR0wzsOAJcGEZVGRydCzZYZ1H1cs+EaXnLWKzhv7Nl4KqSakK7JEwE0RnkI4Jn0FzDWplPdVixS1xylhMCEBB6UazNsO3gPt+66kdt2/4j9U0+RDwfwjN+ldQ2TXSw8OC4AWxXoBbHR7eqljVYSKtUSOe1zzug5XL3pWs4fu5ixgVMYCofJ6Tw150gSm67eUgpPGbRWaAPOOcq1OSbLh9g9u4NHJx/kjt0/ZvvkQxytzJHz8vjGxzmbGVnuhifZSi8AD44bwEbFum/R8I6dECENDFobrLNUojmUguHiClb3reWM0bPYMLKBFbkhBrx+ApMnTixRtUI5mWW8fJiDpcNsP7KD/XP7GC/txSqF0elyXk8bXD2Sd1azS+u62jA/iuMIcP4KtJl0r1IUzWhrnSWRmMTGoITQ8zH1f9yw1lKtVbHOQpKAMRg/xDMhnvaor3GFTLTurFpvk52/7p1ynP+Iu1kbsk661XNNP0mnsgrpCIhKJ4YCHRKaXHqovs4wiQWUIR+GoNI/bBQay4hdfetRpfnAtR3vbT1PR5bhX3yzfch2iOnRzB91Cu2mLbQ0KLsf6qte0z/kQepzVQv1WCR7eQe4JWrdUmQZ/wa5Ucv2p9s2CEG9TyEdjWtIj4RpQenqZM4DrktVlwfer+XX8quX/wcpkwGXFE2F2wAAAABJRU5ErkJggg=="/></span><span class="brandText">HestioPlay</span><span class="pill">Training</span></div>
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

    // shared with HestioRooms
    const lsChat = (localStorage.getItem("nick")||"").trim();
    if(lsChat) return lsChat;

    // legacy / explicit Play key
    const ls = (localStorage.getItem("hestio_nick")||"").trim();
    if(ls) return ls;

    const ck = (getCookie("hestio_nick")||"").trim();
    if(ck) return ck;

    return "";
  }
  function persistNick(v){
    const nick = (v||"").trim();
    if(!nick) return;

    // store in both keys so Play and Rooms always match
    localStorage.setItem("nick", nick);
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

  
  function wsBasePath(){
    const p = location.pathname || "/";
    if(p === "/play2" || p.startsWith("/play2/")) return "/play2";
    if(p === "/play" || p.startsWith("/play/")) return "/play";
    return "";
  }

function connect(){
    if(ws) return;
    const proto=(location.protocol==="https:")?"wss":"ws";
    ws=new WebSocket(`${proto}://${location.host}${wsBasePath()}/ws`);
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
