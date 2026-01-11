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
import random
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
    {"id": "snowmines_30x20", "name": "SnowMines (30×20)", "tag": "Competitive mines — open & flag"},
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




# -----------------------------
# SnowMines (mounted under /play/snowmines)
# -----------------------------

# =============================================================================
# Core configuration
# =============================================================================

COLS = 30
ROWS = 20
MAX_PLAYERS = 10

# Competition-oriented: all levels capped to 3 minutes (180s)
LEVELS: Dict[str, Dict[str, Any]] = {
    "easy": {"label": "Lengvas", "mines": 80, "timeS": 180},
    "medium": {"label": "Vidutinis", "mines": 110, "timeS": 180},
    "hard": {"label": "Sunkus", "mines": 140, "timeS": 180},
}

SAFE_START_RC = (ROWS // 2, COLS // 2)  # shared safe start for all players (fair board)


def now_s() -> float:
    return time.time()


def gen_room_code() -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(secrets.choice(alphabet) for _ in range(4))


def inb(r: int, c: int) -> bool:
    return 0 <= r < ROWS and 0 <= c < COLS


def idx(r: int, c: int) -> int:
    return r * COLS + c


def neighbors(r: int, c: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if inb(rr, cc):
                out.append((rr, cc))
    return out


# =============================================================================
# Room model (in-memory)
# =============================================================================

@dataclass
class Player:
    session_id: str
    ws: Optional[WebSocket] = None
    name: str = "Žaidėjas"
    joined_at: float = field(default_factory=now_s)

    # Per-player view/progress
    open: List[bool] = field(default_factory=list)
    flag: List[bool] = field(default_factory=list)
    revealed: int = 0
    flags: int = 0
    dead: bool = False
    finished_at: Optional[float] = None  # seconds since room.start_at

    last_seen: float = field(default_factory=now_s)

    @property
    def connected(self) -> bool:
        return self.ws is not None


@dataclass
class Room:
    code: str
    created_at: float = field(default_factory=now_s)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # lobby / running / ended
    status: str = "lobby"
    host_session: Optional[str] = None

    # settings
    level_key: str = "medium"

    # game field (shared)
    seed: int = field(default_factory=lambda: secrets.randbits(31))
    mines: List[bool] = field(default_factory=list)
    adj: List[int] = field(default_factory=list)

    # timing
    start_at: Optional[float] = None
    deadline_at: Optional[float] = None

    # competition outcome
    winner_session: Optional[str] = None
    end_reason: Optional[str] = None  # "solved" | "timeout" | "host_end"

    # players
    players: Dict[str, Player] = field(default_factory=dict)

    def capacity_left(self) -> int:
        return MAX_PLAYERS - len(self.players)

    def level(self) -> Dict[str, Any]:
        return LEVELS.get(self.level_key, LEVELS["medium"])

    def is_full(self) -> bool:
        return len(self.players) >= MAX_PLAYERS

    def both_started(self) -> bool:
        return self.status in ("running", "ended")

    def ensure_player_arrays(self, p: Player) -> None:
        n = ROWS * COLS
        if not p.open or len(p.open) != n:
            p.open = [False] * n
        if not p.flag or len(p.flag) != n:
            p.flag = [False] * n


ROOMS: Dict[str, Room] = {}
ROOMS_GUARD = asyncio.Lock()
RECONNECT_TTL_S = 120.0


# =============================================================================
# Minefield generation (shared for the room)
# =============================================================================

def build_field(level_key: str, seed: int) -> Tuple[List[bool], List[int]]:
    cfg = LEVELS.get(level_key, LEVELS["medium"])
    mine_count = int(cfg["mines"])
    n = ROWS * COLS

    mines = [False] * n
    adj = [0] * n

    rng = random.Random(seed)

    sr, sc = SAFE_START_RC
    safe = {idx(sr, sc)}
    for rr, cc in neighbors(sr, sc):
        safe.add(idx(rr, cc))

    candidates = [i for i in range(n) if i not in safe]
    rng.shuffle(candidates)
    mine_count = min(mine_count, len(candidates))
    for i in candidates[:mine_count]:
        mines[i] = True

    # adjacency
    for r in range(ROWS):
        for c in range(COLS):
            i = idx(r, c)
            if mines[i]:
                adj[i] = 0
                continue
            cnt = 0
            for rr, cc in neighbors(r, c):
                if mines[idx(rr, cc)]:
                    cnt += 1
            adj[i] = cnt

    return mines, adj


def flood_open(room: Room, p: Player, start_r: int, start_c: int) -> None:
    """Open a cell; if 0, flood fill. Does not allow opening mines."""
    if p.dead or p.finished_at is not None or room.status != "running":
        return

    i0 = idx(start_r, start_c)
    if p.open[i0] or p.flag[i0]:
        return
    if room.mines[i0]:
        # should not happen if starting at safe cell; but keep safe.
        p.open[i0] = True
        p.dead = True
        return

    stack = [(start_r, start_c)]
    while stack:
        r, c = stack.pop()
        i = idx(r, c)
        if p.open[i] or p.flag[i]:
            continue
        if room.mines[i]:
            continue
        p.open[i] = True
        p.revealed += 1
        if room.adj[i] == 0:
            for rr, cc in neighbors(r, c):
                ii = idx(rr, cc)
                if not p.open[ii] and not p.flag[ii] and not room.mines[ii]:
                    stack.append((rr, cc))


def check_finish(room: Room, p: Player) -> None:
    if p.dead or p.finished_at is not None or room.start_at is None:
        return
    total_non_mines = ROWS * COLS - room.level()["mines"]
    if p.revealed >= total_non_mines:
        p.finished_at = max(0.0, now_s() - room.start_at)


def compute_winner_on_timeout(room: Room) -> Optional[str]:
    """Winner by best progress, tiebreak by earliest finish (if any), then join time."""
    best_sid = None
    best_revealed = -1
    best_finished = None
    best_joined = None

    for sid, p in room.players.items():
        # dead players are allowed to "compete" but are heavily penalized
        revealed = p.revealed if not p.dead else -1
        finished = p.finished_at
        joined = p.joined_at

        key = (revealed, -(finished or 10**9), -joined)  # custom ordering applied below
        # We'll just compare explicitly:
        if revealed > best_revealed:
            best_sid = sid
            best_revealed = revealed
            best_finished = finished
            best_joined = joined
        elif revealed == best_revealed:
            # if someone finished, prefer smaller finished time
            if finished is not None and (best_finished is None or finished < best_finished):
                best_sid = sid
                best_finished = finished
                best_joined = joined
            elif finished == best_finished:
                # earliest join as deterministic
                if best_joined is None or joined < best_joined:
                    best_sid = sid
                    best_joined = joined

    return best_sid


# =============================================================================
# Networking helpers
# =============================================================================

async def ws_send(ws: WebSocket, msg: Dict[str, Any]) -> None:
    await ws.send_text(json.dumps(msg, ensure_ascii=False))


def room_summary(room: Room) -> Dict[str, Any]:
    players = []
    for sid, p in room.players.items():
        players.append(
            {
                "session": sid,
                "name": p.name,
                "connected": p.connected,
                "revealed": p.revealed,
                "flags": p.flags,
                "dead": p.dead,
                "finishedAt": p.finished_at,
                "isHost": (room.host_session == sid),
            }
        )
    # stable order: host first, then join order
    players.sort(key=lambda x: (0 if x["isHost"] else 1, room.players[x["session"]].joined_at))

    cfg = room.level()
    return {
        "room": room.code,
        "status": room.status,
        "host": room.host_session,
        "levelKey": room.level_key,
        "levelLabel": cfg["label"],
        "mines": cfg["mines"],
        "timeS": cfg["timeS"],
        "players": players,
        "startAt": room.start_at,
        "deadlineAt": room.deadline_at,
        "serverNow": now_s(),
        "winnerSession": room.winner_session,
        "endReason": room.end_reason,
    }


def player_view_grid(room: Room, p: Player) -> List[int]:
    """
    Returns flattened grid for the player:
      -3 covered
      -2 flag
      -1 mine (only if opened OR game ended OR player dead)
       0..8 number for opened cell
    """
    n = ROWS * COLS
    out = [-3] * n
    reveal_mines = (p.dead or room.status == "ended")

    for i in range(n):
        if p.flag[i] and not p.open[i]:
            out[i] = -2
            continue
        if p.open[i]:
            if room.mines[i]:
                out[i] = -1
            else:
                out[i] = int(room.adj[i])
            continue
        if reveal_mines and room.mines[i]:
            out[i] = -1

    return out


async def broadcast_room(room: Room) -> None:
    """Send per-player state + shared summary."""
    summary = room_summary(room)
    for sid, p in list(room.players.items()):
        if not p.ws:
            continue
        try:
            await ws_send(
                p.ws,
                {
                    "type": "state",
                    "summary": summary,
                    "you": {"session": sid},
                    "grid": player_view_grid(room, p) if room.status in ("running", "ended") else None,
                },
            )
        except Exception:
            # ignore send errors; disconnect handler will clean up
            pass


def purge_stale(room: Room) -> None:
    cutoff = now_s() - RECONNECT_TTL_S
    stale = []
    for sid, p in room.players.items():
        if (not p.connected) and p.last_seen < cutoff:
            stale.append(sid)
    for sid in stale:
        room.players.pop(sid, None)
        if room.host_session == sid:
            room.host_session = None
    if room.host_session is None and room.players:
        # promote earliest join
        room.host_session = min(room.players.values(), key=lambda pp: pp.joined_at).session_id


async def ensure_room(code: str) -> Room:
    async with ROOMS_GUARD:
        if code in ROOMS:
            return ROOMS[code]
        room = Room(code=code)
        ROOMS[code] = room
        return room


async def create_room() -> Room:
    async with ROOMS_GUARD:
        for _ in range(1000):
            code = gen_room_code()
            if code not in ROOMS:
                room = Room(code=code)
                ROOMS[code] = room
                return room
    # fallback
    code = secrets.token_hex(2).upper()
    room = Room(code=code)
    async with ROOMS_GUARD:
        ROOMS[code] = room
    return room


async def delete_room_if_empty(code: str) -> None:
    async with ROOMS_GUARD:
        room = ROOMS.get(code)
        if not room:
            return
        if not room.players:
            ROOMS.pop(code, None)


# =============================================================================
# Room gameplay lifecycle
# =============================================================================

def start_match(room: Room) -> None:
    cfg = room.level()
    room.seed = secrets.randbits(31)
    room.mines, room.adj = build_field(room.level_key, room.seed)
    room.status = "running"
    room.start_at = now_s()
    room.deadline_at = room.start_at + float(cfg["timeS"])
    room.winner_session = None
    room.end_reason = None

    # reset all players and apply shared initial opening
    sr, sc = SAFE_START_RC
    for p in room.players.values():
        room.ensure_player_arrays(p)
        p.open = [False] * (ROWS * COLS)
        p.flag = [False] * (ROWS * COLS)
        p.revealed = 0
        p.flags = 0
        p.dead = False
        p.finished_at = None

        flood_open(room, p, sr, sc)
        check_finish(room, p)


def end_match(room: Room, winner_session: Optional[str], reason: str) -> None:
    room.status = "ended"
    room.winner_session = winner_session
    room.end_reason = reason
    room.deadline_at = None


def winner_name(room: Room) -> str:
    if not room.winner_session:
        return "—"
    p = room.players.get(room.winner_session)
    return p.name if p else "—"


# =============================================================================
# Server timer task: closes matches on timeout
# =============================================================================

TIMER_TASK: Optional[asyncio.Task] = None


async def timer_loop() -> None:
    while True:
        await asyncio.sleep(0.5)
        async with ROOMS_GUARD:
            rooms = list(ROOMS.values())
        for room in rooms:
            if room.status != "running" or room.deadline_at is None:
                continue
            if now_s() < room.deadline_at:
                continue
            async with room.lock:
                if room.status != "running" or room.deadline_at is None:
                    continue
                if now_s() < room.deadline_at:
                    continue
                # if someone already solved, end should have happened via move; still guard
                if room.winner_session:
                    end_match(room, room.winner_session, "solved")
                else:
                    wsid = compute_winner_on_timeout(room)
                    end_match(room, wsid, "timeout")
                await broadcast_room(room)


@app.on_event("startup")
async def _startup() -> None:
    global TIMER_TASK
    if TIMER_TASK is None:
        TIMER_TASK = asyncio.create_task(timer_loop())


@app.on_event("shutdown")
async def _shutdown() -> None:
    global TIMER_TASK
    if TIMER_TASK is not None:
        TIMER_TASK.cancel()
        TIMER_TASK = None


# =============================================================================
# HTML UI (Lux themes: 3 winter + 3 summer)
# =============================================================================

SNOWMINES_HTML = r"""<!doctype html>
<html lang="lt">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover" />
  <title>Hestio SnowMines — Competition 30×20</title>
  <style>
    :root{
      --bg0:#07131a;
      --bg1:#0b1b24;
      --panel: rgba(18, 26, 34, .78);
      --panel2: rgba(12, 18, 25, .82);
      --text: #eef6ff;
      --muted: rgba(238,246,255,.72);
      --stroke: rgba(238,246,255,.14);

      --accent: #cfe6ff;
      --accent2:#ffffff;

      --tile1: rgba(223,239,255,.92);
      --tile2: rgba(168,198,232,.84);

      --moundA: rgba(255,255,255,.92);
      --moundB: rgba(243,251,255,.78);
      --moundC: rgba(170,200,230,.55);

      --frameTop:#2b3d4b;
      --frameBot:#15222c;
      --insetTop:#0e202b;
      --insetBot:#0a141c;

      --flag1: rgba(255,90,105,.95);
      --flag2: rgba(180,25,40,.95);
    }

    html, body { height: 100%; }
    body{
      margin:0;
      background:
        radial-gradient(1100px 700px at 18% 10%, rgba(207,230,255,.14), transparent 60%),
        radial-gradient(900px 600px at 100% 20%, rgba(255,255,255,.08), transparent 60%),
        radial-gradient(800px 700px at 40% 100%, rgba(140,220,255,.07), transparent 62%),
        linear-gradient(180deg, var(--bg0), var(--bg1));
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
      overflow: hidden;
    }

    .app{
      display:grid;
      grid-template-columns: 460px 1fr;
      height:100%;
      gap: 18px;
      padding: 18px;
      box-sizing: border-box;
    }

    @media (max-width: 1120px){
      body{ overflow:auto; }
      .app{ grid-template-columns: 1fr; height:auto; }
      #boardWrap{ min-height: 620px; }
    }

    .card{
      background: var(--panel);
      border: 1px solid var(--stroke);
      border-radius: 22px;
      box-shadow:
        0 30px 90px rgba(0,0,0,.42),
        inset 0 1px 0 rgba(255,255,255,.06);
      backdrop-filter: blur(14px);
      -webkit-backdrop-filter: blur(14px);
    }

    .side{
      padding: 16px;
      display:flex;
      flex-direction: column;
      gap: 12px;
      min-height: 620px;
    }

    .title{
      padding: 16px 16px 10px 16px;
      border-radius: 22px;
      background:
        radial-gradient(780px 260px at 0% 0%, rgba(207,230,255,.20), transparent 62%),
        linear-gradient(180deg, rgba(255,255,255,.06), transparent);
      border: 1px solid rgba(255,255,255,.08);
    }
    .title h1{
      margin:0;
      font-size: 18px;
      letter-spacing: .6px;
      font-weight: 720;
    }
    .title p{
      margin:6px 0 0 0;
      font-size: 12px;
      color: var(--muted);
      line-height: 1.45;
    }

    .row{ display:flex; gap:10px; flex-wrap: wrap; align-items: center; }
    .kpi{
      flex: 1 1 210px;
      padding: 12px 12px;
      border-radius: 18px;
      background: var(--panel2);
      border: 1px solid var(--stroke);
    }
    .kpi .label{ font-size: 11px; color: var(--muted); }
    .kpi .value{ font-size: 14px; margin-top: 6px; font-weight: 650; line-height: 1.25; }

    .controls{
      padding: 12px;
      border-radius: 18px;
      background: var(--panel2);
      border: 1px solid var(--stroke);
      display:flex;
      flex-direction: column;
      gap: 10px;
    }

    .grid2{ display:grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .grid3{ display:grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; }
    .grid4{ display:grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 10px; }

    button, select, input{
      appearance: none;
      background: rgba(255,255,255,.06);
      border: 1px solid rgba(255,255,255,.12);
      color: var(--text);
      padding: 10px 12px;
      border-radius: 14px;
      font-size: 13px;
      outline: none;
      transition: transform .08s ease, border-color .18s ease, background .18s ease;
      box-sizing: border-box;
    }
    button{ cursor: pointer; }
    input{ width: 100%; }
    button:hover, select:hover, input:hover{ border-color: rgba(255,255,255,.22); background: rgba(255,255,255,.08); }
    button:active{ transform: translateY(1px); }
    button.primary{
      background: linear-gradient(180deg, rgba(207,230,255,.22), rgba(207,230,255,.08));
      border-color: rgba(207,230,255,.42);
    }
    button.danger{
      background: linear-gradient(180deg, rgba(255,90,105,.22), rgba(255,90,105,.08));
      border-color: rgba(255,90,105,.38);
    }
    button:disabled{ opacity:.55; cursor:not-allowed; }

    .hr{ height:1px; background: rgba(255,255,255,.08); margin: 6px 0; }

    .status{
      padding: 12px;
      border-radius: 18px;
      background: linear-gradient(180deg, rgba(255,255,255,.05), transparent);
      border: 1px solid rgba(255,255,255,.10);
      font-size: 12px;
      color: var(--muted);
      line-height: 1.55;
      min-height: 120px;
    }
    .status strong{ color: var(--text); font-weight: 650; }

    .pill{
      display:inline-flex;
      align-items:center;
      gap:8px;
      padding: 7px 10px;
      border-radius: 999px;
      border: 1px solid rgba(255,255,255,.12);
      background: rgba(255,255,255,.05);
      font-size: 12px;
      color: var(--muted);
      user-select:none;
      white-space: nowrap;
    }
    a.pill{ text-decoration:none; }
    .dot{
      width: 10px; height: 10px;
      border-radius: 999px;
      box-shadow: 0 0 0 2px rgba(255,255,255,.08), 0 0 18px rgba(207,230,255,.20);
      background: rgba(207,230,255,.9);
    }
    .mono{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; letter-spacing: .6px; }

    #boardWrap{
      position: relative;
      padding: 16px;
      border-radius: 22px;
      overflow: hidden;
      min-height: 720px;
    }
    #canvas{
      width: 100%;
      height: 100%;
      display:block;
      border-radius: 22px;
      touch-action: manipulation;
    }

    .watermark{
      position:absolute;
      right: 18px;
      bottom: 14px;
      font-size: 12px;
      color: rgba(238,246,255,.35);
      letter-spacing: .6px;
      user-select: none;
    }

    .leader{
      display:flex;
      flex-direction: column;
      gap: 6px;
    }
    .leaderItem{
      display:flex;
      align-items:center;
      justify-content: space-between;
      padding: 10px 10px;
      border-radius: 14px;
      border: 1px solid rgba(255,255,255,.10);
      background: rgba(255,255,255,.04);
      font-size: 12px;
      color: rgba(238,246,255,.78);
    }
    .leaderItem strong{
      color: var(--text);
      font-weight: 650;
    }
    .badge{
      display:inline-flex;
      align-items:center;
      gap: 8px;
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid rgba(255,255,255,.12);
      background: rgba(255,255,255,.06);
      font-size: 12px;
      color: rgba(238,246,255,.78);
    }
  </style>
</head>

<body>
  <div class="app">
    <div class="side card">
      <div class="title">
        <h1>Hestio SnowMines — Competition (1–10)</h1>
        <p>30×20. Visi žaidėjai sprendžia <strong>tą pačią lentą</strong>. Pradžioje atveriamas bendras saugus centras. Laimi tas, kas <strong>greičiausiai išsprendžia</strong> (arba pagal progresą, jei baigiasi laikas).</p>
        <div class="row" style="margin-top:10px;"><a class="pill" id="backToPlay" href="#">Back to HestioPlay</a></div>
      </div>

      <div class="row">
        <div class="kpi">
          <div class="label">Jūs / Kambarys</div>
          <div class="value" id="youLabel">—</div>
        </div>
        <div class="kpi">
          <div class="label">Laikas / Statusas</div>
          <div class="value" id="timeLabel">—</div>
        </div>
      </div>

      <div class="controls">
        <div class="grid2">
          <button class="primary" id="createBtn">Sukurti lobby</button>
          <button id="copyBtn">Kopijuoti nuorodą</button>
        </div>

        <div class="grid2">
          <input id="roomInput" class="mono" placeholder="Kambarys (pvz. A7KD)" maxlength="8" />
          <button id="joinBtn">Prisijungti</button>
        </div>

        <div class="grid3">
          <input id="nameInput" placeholder="Vardas (pvz. Mantas)" />
          <select id="levelSel" title="Lygis">
            <option value="easy">Lengvas</option>
            <option value="medium" selected>Vidutinis</option>
            <option value="hard">Sunkus</option>
          </select>
          <button id="startBtn" class="primary">Start</button>
        </div>

        <div class="grid2">
          <select id="themeSel" title="Tema (tik vizualiai)"></select>
          <button id="newLobbyBtn" class="danger">Naujas match</button>
        </div>

        <div class="hr"></div>

        <div class="row">
          <span class="pill"><span class="dot"></span><span id="connPill">Neprisijungta</span></span>
          <span class="pill"><span class="dot"></span><span id="roomPill">Kambarys: —</span></span>
          <span class="pill"><span class="dot"></span><span id="lvlPill">Lygis: —</span></span>
          <span class="pill"><span class="dot"></span><span id="clockPill">Laikas: —</span></span>
        </div>
      </div>

      <div class="status" id="statusBox">
        <strong>Statusas:</strong> sukurkite arba įveskite kambario kodą.
      </div>

      <div class="controls">
        <div class="row" style="justify-content: space-between;">
          <span class="badge" id="winnerBadge">Nugalėtojas: —</span>
          <span class="badge" id="youProgress">Progresas: —</span>
        </div>
        <div class="leader" id="leader"></div>
      </div>
    </div>

    <div id="boardWrap" class="card">
      <canvas id="canvas"></canvas>
      <div class="watermark">Hestio • SnowMines</div>
    </div>
  </div>

<script>
(() => {
  "use strict";

  const COLS = 30;
  const ROWS = 20;

  // Luxury themes: 3 winter + 3 summer
  // Note: theme is client-side only; gameplay is server authoritative.
  const THEMES = {
    // Winter
    winter_glacier: {
      label: "Winter — Glacier Platinum",
      bg0:"#07131a", bg1:"#0b1b24",
      accent:"#cfe6ff", accent2:"#ffffff",
      tile1:"rgba(223,239,255,.92)", tile2:"rgba(168,198,232,.84)",
      moundA:"rgba(255,255,255,.94)", moundB:"rgba(244,252,255,.80)", moundC:"rgba(170,200,230,.56)",
      frameTop:"#2b3d4b", frameBot:"#15222c", insetTop:"#0e202b", insetBot:"#0a141c",
      flag1:"rgba(255,90,105,.95)", flag2:"rgba(180,25,40,.95)"
    },
    winter_aurora: {
      label: "Winter — Aurora Silk",
      bg0:"#061014", bg1:"#071f1a",
      accent:"#baf5e7", accent2:"#eafcff",
      tile1:"rgba(219,248,242,.90)", tile2:"rgba(140,213,206,.82)",
      moundA:"rgba(255,255,255,.92)", moundB:"rgba(236,255,250,.78)", moundC:"rgba(150,220,210,.52)",
      frameTop:"#214a42", frameBot:"#0f2a24", insetTop:"#0b2621", insetBot:"#071b18",
      flag1:"rgba(255,115,125,.95)", flag2:"rgba(190,35,55,.95)"
    },
    winter_royal: {
      label: "Winter — Royal Sapphire",
      bg0:"#070b16", bg1:"#0b122a",
      accent:"#cdd9ff", accent2:"#ffffff",
      tile1:"rgba(226,234,255,.90)", tile2:"rgba(154,176,230,.84)",
      moundA:"rgba(255,255,255,.92)", moundB:"rgba(243,246,255,.78)", moundC:"rgba(150,175,235,.56)",
      frameTop:"#283a6a", frameBot:"#141e3d", insetTop:"#0e1a33", insetBot:"#0a1327",
      flag1:"rgba(255,95,110,.95)", flag2:"rgba(180,25,40,.95)"
    },

    // Summer
    summer_sand: {
      label: "Summer — Sand Dune Gold",
      bg0:"#0e0d0a", bg1:"#19150f",
      accent:"#ffd9a0", accent2:"#fff3dd",
      tile1:"rgba(255,244,225,.92)", tile2:"rgba(232,205,165,.84)",
      moundA:"rgba(255,248,236,.92)", moundB:"rgba(247,232,206,.78)", moundC:"rgba(200,165,120,.58)",
      frameTop:"#4a3623", frameBot:"#2b1f15", insetTop:"#241a12", insetBot:"#17100b",
      flag1:"rgba(255,90,105,.95)", flag2:"rgba(180,25,40,.95)"
    },
    summer_lagoon: {
      label: "Summer — Lagoon Pearl",
      bg0:"#061014", bg1:"#0b1717",
      accent:"#bff7ff", accent2:"#ffffff",
      tile1:"rgba(226,252,255,.90)", tile2:"rgba(158,222,230,.84)",
      moundA:"rgba(250,255,255,.92)", moundB:"rgba(232,252,255,.78)", moundC:"rgba(140,205,215,.56)",
      frameTop:"#245059", frameBot:"#102d33", insetTop:"#0c262a", insetBot:"#081a1d",
      flag1:"rgba(255,100,115,.95)", flag2:"rgba(185,25,45,.95)"
    },
    summer_olive: {
      label: "Summer — Olive Marble",
      bg0:"#0b0f0a", bg1:"#121a10",
      accent:"#d7f0c6", accent2:"#ffffff",
      tile1:"rgba(238,252,230,.90)", tile2:"rgba(175,212,160,.84)",
      moundA:"rgba(252,255,246,.92)", moundB:"rgba(238,252,235,.78)", moundC:"rgba(160,200,145,.56)",
      frameTop:"#31422a", frameBot:"#1b2417", insetTop:"#141f12", insetBot:"#0e160d",
      flag1:"rgba(255,95,110,.95)", flag2:"rgba(180,25,40,.95)"
    }
  };

  const THEME_ORDER = [
    "winter_glacier","winter_aurora","winter_royal",
    "summer_sand","summer_lagoon","summer_olive"
  ];

  const NUM_COLORS = {
    1: "#1c6fb8",
    2: "#2a8e4b",
    3: "#c53a37",
    4: "#6a49d6",
    5: "#a06a2a",
    6: "#1e8b8d",
    7: "#1c1f2a",
    8: "#596075"
  };

  const state = {
    // connection
    ws: null,
    connected: false,
    sessionId: null,
    room: null,
    summary: null,

    // player view
    grid: null, // flattened ints (len ROWS*COLS), or null in lobby

    // UI
    themeKey: "winter_glacier",
    dpr: 1,
    layout: null,
    noise: null,
    decoSeeds: [],
    lastPointerDownAt: 0,
    _clientNowAtState: null
  };

  // DOM
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d", { alpha: false });

  const youLabel = document.getElementById("youLabel");
  const timeLabel = document.getElementById("timeLabel");
  const connPill = document.getElementById("connPill");
  const roomPill = document.getElementById("roomPill");
  const lvlPill = document.getElementById("lvlPill");
  const clockPill = document.getElementById("clockPill");
  const statusBox = document.getElementById("statusBox");

    function basePrefix(){
    const p = location.pathname;
    const i = p.indexOf("/snowmines");
    return (i >= 0) ? p.slice(0, i) : "";
  }

  // Set back link (works under mounts like /play2)
  const backToPlay = document.getElementById("backToPlay");
  if (backToPlay){ backToPlay.href = (basePrefix() || "") + "/play"; }

const createBtn = document.getElementById("createBtn");
  const copyBtn = document.getElementById("copyBtn");
  const joinBtn = document.getElementById("joinBtn");
  const roomInput = document.getElementById("roomInput");

  const nameInput = document.getElementById("nameInput");
function preferredNick(){ try{ return localStorage.getItem("hestio_nick")||""; }catch(e){ return ""; } }
function persistNick(n){ try{ localStorage.setItem("hestio_nick", n); }catch(e){} }

// Nick prefill: ?nick=... > localStorage("hestio_nick")
try{
  const qp = new URLSearchParams(location.search);
  const qpNick = (qp.get("nick")||"").trim();
  if(qpNick){
    nameInput.value = qpNick.slice(0,24);
    persistNick(nameInput.value);
  }else{
    const pn = preferredNick();
    if(pn && !nameInput.value) nameInput.value = pn.slice(0,24);
  }
}catch(e){}
  const levelSel = document.getElementById("levelSel");
  const startBtn = document.getElementById("startBtn");
  const newLobbyBtn = document.getElementById("newLobbyBtn");
  const themeSel = document.getElementById("themeSel");

  const leader = document.getElementById("leader");
  const winnerBadge = document.getElementById("winnerBadge");
  const youProgress = document.getElementById("youProgress");

  // utils
  const clamp = (v,a,b) => Math.max(a, Math.min(b, v));
  const inb = (r,c) => r>=0 && r<ROWS && c>=0 && c<COLS;
  const idx = (r,c) => r*COLS + c;

  function setStatus(html){
    statusBox.innerHTML = `<strong>Statusas:</strong> ${html}`;
  }

  function ensureSession(){
    let sid = localStorage.getItem("hestio_snowmines_session");
    if (!sid){
      sid = (crypto.randomUUID ? crypto.randomUUID() : String(Math.random()).slice(2) + String(Date.now()));
      localStorage.setItem("hestio_snowmines_session", sid);
    }
    state.sessionId = sid;
  }

  function wsUrl(){
    const proto = (location.protocol === "https:") ? "wss" : "ws";
    const p = location.pathname;
    const i = p.indexOf("/snowmines");
    const prefix = (i >= 0) ? p.slice(0, i) : "";
    return `${proto}://${location.host}${prefix}/snowmines/ws`;
  }

  function send(msg){
    if (!state.ws || state.ws.readyState !== 1) return;
    state.ws.send(JSON.stringify(msg));
  }

  function connect(mode, code){
    if (state.ws){
      try { state.ws.close(); } catch(e) {}
      state.ws = null;
    }
    const ws = new WebSocket(wsUrl());
    state.ws = ws;

    ws.onopen = () => {
      state.connected = true;
      updateHUD();
      if (mode === "create"){
        send({type:"create", sessionId: state.sessionId});
      } else {
        send({type:"join", sessionId: state.sessionId, room: code});
      }
      setStatus("Jungiama…");
    };

    ws.onmessage = (ev) => {
      let msg = null;
      try { msg = JSON.parse(ev.data); } catch(e){ return; }

      if (msg.type === "joined"){
        state.room = msg.room;
        if (state.room){
          const url = new URL(location.href);
          url.hash = `room=${state.room}`;
          history.replaceState(null, "", url.toString());
        }
        setStatus(`Prisijungta prie lobby: <span class="mono">${state.room}</span>.`);
        if (msg.summary) applyState(msg.summary, msg.grid);
        // send name if present
        const nm = (nameInput.value || "").trim();
        if (nm) send({type:"setName", room: state.room, name: nm});
        return;
      }

      if (msg.type === "state"){
        applyState(msg.summary, msg.grid);
        return;
      }

      if (msg.type === "error"){
        setStatus(`<strong>Klaida:</strong> ${msg.message}`);
        return;
      }
    };

    ws.onclose = () => {
      state.connected = false;
      updateHUD();
      setStatus("Ryšys nutrūko.");
    };

    ws.onerror = () => {
      state.connected = false;
      updateHUD();
      setStatus("Ryšio klaida.");
    };
  }

  function applyState(summary, grid){
    state.summary = summary;
    state.grid = grid || null;
    state._clientNowAtState = Date.now()/1000;
    updateHUD();
  }

  function fmtTime(sec){
    sec = Math.max(0, Math.floor(sec));
    const m = Math.floor(sec/60);
    const s = sec%60;
    return `${m}:${String(s).padStart(2,"0")}`;
  }

  function approxServerNow(){
    if (!state.summary || !state.summary.serverNow) return Date.now()/1000;
    const clientNow = Date.now()/1000;
    const offset = state.summary.serverNow - (state._clientNowAtState || clientNow);
    return clientNow + offset;
  }

  function updateHUD(){
    connPill.textContent = state.connected ? "Prisijungta" : "Neprisijungta";
    roomPill.textContent = `Kambarys: ${state.room ? state.room : "—"}`;

    const you = state.sessionId ? state.sessionId.slice(0,8) : "—";
    if (!state.summary){
      youLabel.innerHTML = `${(nameInput.value||"Jūs").trim() || "Jūs"}<br/><span class="mono">—</span>`;
      lvlPill.textContent = "Lygis: —";
      timeLabel.textContent = "—";
      clockPill.textContent = "Laikas: —";
      winnerBadge.textContent = "Nugalėtojas: —";
      youProgress.textContent = "Progresas: —";
      leader.innerHTML = "";
      startBtn.disabled = true;
      levelSel.disabled = true;
      newLobbyBtn.disabled = true;
      return;
    }

    const s = state.summary;
    lvlPill.textContent = `Lygis: ${s.levelLabel} • minos ${s.mines} • 30×20`;

    // host controls
    const isHost = (s.host === state.sessionId);
    levelSel.disabled = !isHost || s.status !== "lobby";
    startBtn.disabled = !isHost || s.status !== "lobby" || (s.players.length < 1);
    newLobbyBtn.disabled = !isHost;

    levelSel.value = s.levelKey;

    // time
    if (s.status === "lobby"){
      timeLabel.textContent = "Lobby";
      clockPill.textContent = `Laikas: ${fmtTime(s.timeS)}`;
    } else if (s.status === "running"){
      const left = s.deadlineAt ? (s.deadlineAt - approxServerNow()) : 0;
      timeLabel.textContent = `Vyksta • likę ${fmtTime(left)}`;
      clockPill.textContent = `Laikas: ${fmtTime(left)}`;
    } else {
      timeLabel.textContent = `Baigta • ${s.endReason === "solved" ? "išspręsta" : (s.endReason === "all_dead" ? "visi užlipo ant minos" : "laikas baigėsi")}`;
      clockPill.textContent = "Laikas: 0:00";
    }

    // winner badge
    let wn = "—";
    if (s.winnerSession){
      const p = s.players.find(pp => pp.session === s.winnerSession);
      wn = p ? p.name : "—";
    }
    winnerBadge.textContent = `Nugalėtojas: ${wn}`;

    // you label & progress
    const me = s.players.find(pp => pp.session === state.sessionId);
    const meName = (me && me.name) ? me.name : ((nameInput.value||"Jūs").trim() || "Jūs");
    youLabel.innerHTML = `${meName}<br/><span class="mono">${state.room || "—"}</span>`;

    if (me){
      const totalNonMines = (30*20) - s.mines;
      const pct = totalNonMines > 0 ? Math.floor((me.revealed / totalNonMines) * 100) : 0;
      const fin = me.finishedAt != null ? ` • ${fmtTime(me.finishedAt)}` : "";
      const dead = me.dead ? " • BOOM" : "";
      youProgress.textContent = `Progresas: ${me.revealed}/${totalNonMines} (${pct}%)${fin}${dead}`;
    } else {
      youProgress.textContent = "Progresas: —";
    }

    // leaderboard
    const sorted = [...s.players].sort((a,b) => {
      // Alive players first. Then: finished (smaller time), else higher revealed.
      const ad = a.dead ? 1 : 0;
      const bd = b.dead ? 1 : 0;
      if (ad !== bd) return ad - bd;
      const af = (a.finishedAt==null) ? 1e18 : a.finishedAt;
      const bf = (b.finishedAt==null) ? 1e18 : b.finishedAt;
      if (af !== bf) return af - bf;
      if (a.revealed !== b.revealed) return b.revealed - a.revealed;
      return 0;
    });

    leader.innerHTML = "";
    for (let i=0;i<sorted.length;i++){
      const p = sorted[i];
      const el = document.createElement("div");
      el.className = "leaderItem";
      const tag = p.isHost ? " • host" : "";
      const fin = p.finishedAt != null ? ` • ${fmtTime(p.finishedAt)}` : "";
      const dead = p.dead ? " • BOOM" : "";
      el.innerHTML = `<div><strong>${i+1}. ${escapeHtml(p.name)}</strong>${tag}${dead}</div><div>${p.revealed}${fin}</div>`;
      leader.appendChild(el);
    }

    // status box text
    if (s.status === "lobby"){
      setStatus(`Lobby atidarytas. Dalyviai: <strong>${s.players.length}</strong>/10. Host paspaudžia <strong>Start</strong>.`);
    } else if (s.status === "running"){
      setStatus(`Vyksta match. Visi sprendžia tą pačią lentą. Kairys click — atidengti, dešinys click (arba ilgai paliesti) — vėliava, dvigubas click — chord.`);
    } else {
      const reason = s.endReason === "solved" ? "Išspręsta" : (s.endReason === "all_dead" ? "Visi užlipo ant minos" : "Baigėsi laikas");
      setStatus(`${reason}. Nugalėtojas: <strong>${wn}</strong>. Host gali paspausti <strong>Naujas match</strong>.`);
    }
  }

  function escapeHtml(s){
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"
    })[c]);
  }

  // ---------------------------------------------------------------------------
  // Theme application
  // ---------------------------------------------------------------------------
  function populateThemes(){
    themeSel.innerHTML = "";
    for (const key of THEME_ORDER){
      const opt = document.createElement("option");
      opt.value = key;
      opt.textContent = THEMES[key].label;
      themeSel.appendChild(opt);
    }
    themeSel.value = state.themeKey;
  }

  function applyTheme(key){
    state.themeKey = key;
    const t = THEMES[key];
    const r = document.documentElement.style;
    r.setProperty("--bg0", t.bg0);
    r.setProperty("--bg1", t.bg1);
    r.setProperty("--accent", t.accent);
    r.setProperty("--accent2", t.accent2);
    r.setProperty("--tile1", t.tile1);
    r.setProperty("--tile2", t.tile2);
    r.setProperty("--moundA", t.moundA);
    r.setProperty("--moundB", t.moundB);
    r.setProperty("--moundC", t.moundC);
    r.setProperty("--frameTop", t.frameTop);
    r.setProperty("--frameBot", t.frameBot);
    r.setProperty("--insetTop", t.insetTop);
    r.setProperty("--insetBot", t.insetBot);
    r.setProperty("--flag1", t.flag1);
    r.setProperty("--flag2", t.flag2);
  }

  // ---------------------------------------------------------------------------
  // Rendering: "mounds" + ambiguous decor
  // ---------------------------------------------------------------------------
  function ensureNoise(){
    if (state.noise) return;
    const nc = document.createElement("canvas");
    nc.width = 320; nc.height = 320;
    const nctx = nc.getContext("2d");
    const img = nctx.createImageData(nc.width, nc.height);
    for (let i=0;i<img.data.length;i+=4){
      const v = (Math.random()*255)|0;
      img.data[i]=v; img.data[i+1]=v; img.data[i+2]=v; img.data[i+3]=20;
    }
    nctx.putImageData(img,0,0);
    state.noise = nc;
  }

  function resize(){
    const rect = canvas.getBoundingClientRect();
    state.dpr = clamp(window.devicePixelRatio || 1, 1, 2.5);

    canvas.width = Math.floor(rect.width * state.dpr);
    canvas.height = Math.floor(rect.height * state.dpr);

    ctx.setTransform(1,0,0,1,0,0);
    ctx.scale(state.dpr, state.dpr);

    const pad = 26;
    const w = rect.width - pad*2;
    const h = rect.height - pad*2;

    const cell = Math.floor(Math.min(w / COLS, h / ROWS));
    const gridW = cell * COLS;
    const gridH = cell * ROWS;
    const x = Math.floor((rect.width - gridW)/2);
    const y = Math.floor((rect.height - gridH)/2);
    state.layout = { x, y, cell, gridW, gridH, w: rect.width, h: rect.height };

    // decor seeds: blur edges and create mound-vs-decor ambiguity
    state.decoSeeds = [];
    for (let i=0;i<200;i++){
      state.decoSeeds.push({
        x: Math.random(), y: Math.random(),
        r: 0.008 + Math.random()*0.02,
        a: 0.05 + Math.random()*0.12
      });
    }

    ensureNoise();
  }

  function rr(x,y,w,h,r){
    ctx.beginPath();
    ctx.moveTo(x+r,y);
    ctx.arcTo(x+w,y,x+w,y+h,r);
    ctx.arcTo(x+w,y+h,x,y+h,r);
    ctx.arcTo(x,y+h,x,y,r);
    ctx.arcTo(x,y,x+w,y,r);
    ctx.closePath();
  }

  function css(name, fallback){
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  }

  function drawBackground(){
    const {w,h} = state.layout;
    const g = ctx.createLinearGradient(0,0,0,h);
    g.addColorStop(0, css("--bg0","#07131a"));
    g.addColorStop(1, css("--bg1","#0b1b24"));
    ctx.fillStyle = g;
    ctx.fillRect(0,0,w,h);

    const s1 = ctx.createRadialGradient(w*0.20, h*0.12, 0, w*0.20, h*0.12, Math.min(w,h)*0.75);
    s1.addColorStop(0, "rgba(255,255,255,.08)");
    s1.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = s1; ctx.fillRect(0,0,w,h);
  }

  function drawDecorativeMounds(x,y,w,h,cell){
    ctx.save();
    for (const s of state.decoSeeds){
      const px = x + w*s.x;
      const py = y + h*s.y;
      const edge = Math.min(s.x, 1-s.x, s.y, 1-s.y);
      if (edge > 0.18) continue;

      const r = (cell*2.4) * s.r;
      const g = ctx.createRadialGradient(px-r*0.3, py-r*0.4, r*0.2, px, py, r*1.2);
      g.addColorStop(0, css("--moundA","rgba(255,255,255,.72)"));
      g.addColorStop(0.55, css("--moundB","rgba(240,250,255,.38)"));
      g.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.ellipse(px, py, r*1.2, r*0.95, (Math.random()*0.6)-0.3, 0, Math.PI*2);
      ctx.globalAlpha = 0.9;
      ctx.fill();
    }
    ctx.restore();
  }

  function drawFrame(){
    const {x,y,gridW,gridH,cell} = state.layout;
    const pad = Math.max(18, cell*0.45);

    ctx.save();
    ctx.shadowColor = "rgba(0,0,0,.45)";
    ctx.shadowBlur = 40;
    ctx.shadowOffsetY = 24;
    rr(x-pad, y-pad, gridW+pad*2, gridH+pad*2, 24);
    const frame = ctx.createLinearGradient(x-pad, y-pad, x-pad, y+gridH+pad);
    frame.addColorStop(0, css("--frameTop","#2b3d4b"));
    frame.addColorStop(1, css("--frameBot","#15222c"));
    ctx.fillStyle = frame;
    ctx.fill();
    ctx.restore();

    rr(x-10, y-10, gridW+20, gridH+20, 20);
    const inset = ctx.createLinearGradient(x, y-10, x, y+gridH+10);
    inset.addColorStop(0, css("--insetTop","#0e202b"));
    inset.addColorStop(1, css("--insetBot","#0a141c"));
    ctx.fillStyle = inset;
    ctx.fill();

    drawDecorativeMounds(x-pad, y-pad, gridW+pad*2, gridH+pad*2, cell);
  }

  function cellRect(r,c){
    const {x,y,cell} = state.layout;
    return { x0: x + c*cell, y0: y + r*cell, w: cell };
  }

  function drawCoveredCell(r,c){
    const {x0,y0,w} = cellRect(r,c);
    const pad = w*0.10;

    // subtle tile base
    ctx.save();
    ctx.globalAlpha = 0.10;
    ctx.fillStyle = "rgba(255,255,255,.35)";
    ctx.fillRect(x0+1, y0+1, w-2, w-2);
    ctx.restore();

    const cx = x0 + w*0.52 + (Math.sin((r*7+c*11)*0.2)*w*0.03);
    const cy = y0 + w*0.58 + (Math.cos((r*5+c*9)*0.18)*w*0.03);
    const rx = w*0.54;
    const ry = w*0.44;

    ctx.save();

    // shadow
    ctx.beginPath();
    ctx.ellipse(cx, cy + ry*0.35, rx*0.80, ry*0.38, 0, 0, Math.PI*2);
    ctx.fillStyle = "rgba(0,0,0,.28)";
    ctx.filter = `blur(${Math.max(2, w*0.10)}px)`;
    ctx.fill();
    ctx.filter = "none";

    // mound
    const g = ctx.createRadialGradient(cx-rx*0.25, cy-ry*0.35, rx*0.18, cx, cy, rx*1.2);
    g.addColorStop(0, css("--moundA","rgba(255,255,255,.92)"));
    g.addColorStop(0.55, css("--moundB","rgba(243,251,255,.78)"));
    g.addColorStop(1, css("--moundC","rgba(170,200,230,.55)"));
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI*2);
    ctx.fill();

    // rim highlight
    ctx.strokeStyle = "rgba(255,255,255,.30)";
    ctx.lineWidth = Math.max(1, w*0.06);
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx*0.96, ry*0.92, 0, 0, Math.PI*2);
    ctx.stroke();

    // texture (adds ambiguity)
    ctx.save();
    ctx.globalAlpha = 0.10;
    ctx.drawImage(state.noise, x0-pad, y0-pad, w+pad*2, w+pad*2);
    ctx.restore();

    ctx.restore();
  }

  function drawOpenCell(r,c,val){
    const {x0,y0,w} = cellRect(r,c);

    ctx.save();
    const g = ctx.createLinearGradient(x0, y0, x0+w, y0+w);
    g.addColorStop(0, css("--tile1","rgba(223,239,255,.90)"));
    g.addColorStop(1, css("--tile2","rgba(168,198,232,.82)"));
    ctx.fillStyle = g;
    ctx.fillRect(x0, y0, w, w);

    ctx.strokeStyle = "rgba(0,0,0,.10)";
    ctx.lineWidth = 1;
    ctx.strokeRect(x0+0.5, y0+0.5, w-1, w-1);

    ctx.globalAlpha = 0.12;
    ctx.drawImage(state.noise, x0, y0, w, w);
    ctx.globalAlpha = 1.0;
    ctx.restore();

    if (val === -1){
      drawMine(x0+w*0.5, y0+w*0.53, w);
    } else if (val > 0){
      drawNumber(x0+w*0.5, y0+w*0.56, w, val);
    }
  }

  function drawFlag(x,y,size){
    ctx.save();
    ctx.shadowColor = "rgba(0,0,0,.25)";
    ctx.shadowBlur = size*0.22;
    ctx.shadowOffsetY = size*0.10;

    // pole
    ctx.strokeStyle = "rgba(20,25,30,.55)";
    ctx.lineWidth = Math.max(2, size*0.08);
    ctx.beginPath();
    ctx.moveTo(x, y-size*0.28);
    ctx.lineTo(x, y+size*0.26);
    ctx.stroke();

    const g = ctx.createLinearGradient(x, y-size*0.26, x+size*0.35, y+size*0.05);
    g.addColorStop(0, css("--flag1","rgba(255,90,105,.95)"));
    g.addColorStop(1, css("--flag2","rgba(180,25,40,.95)"));
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.moveTo(x, y-size*0.24);
    ctx.quadraticCurveTo(x+size*0.18, y-size*0.20, x+size*0.32, y-size*0.10);
    ctx.quadraticCurveTo(x+size*0.20, y-size*0.04, x, y-size*0.02);
    ctx.closePath();
    ctx.fill();

    // base
    ctx.fillStyle = "rgba(15,20,25,.35)";
    ctx.beginPath();
    ctx.ellipse(x, y+size*0.30, size*0.18, size*0.08, 0, 0, Math.PI*2);
    ctx.fill();

    ctx.restore();
  }

  function drawMine(x,y,size){
    ctx.save();
    ctx.beginPath();
    ctx.arc(x, y, size*0.22, 0, Math.PI*2);
    const g = ctx.createRadialGradient(x-size*0.08, y-size*0.10, size*0.04, x, y, size*0.26);
    g.addColorStop(0, "rgba(90,95,110,.95)");
    g.addColorStop(0.55, "rgba(20,25,32,.95)");
    g.addColorStop(1, "rgba(0,0,0,.95)");
    ctx.fillStyle = g;
    ctx.fill();

    ctx.beginPath();
    ctx.arc(x+size*0.12, y-size*0.14, size*0.04, 0, Math.PI*2);
    ctx.fillStyle = "rgba(255,170,70,.95)";
    ctx.shadowColor = "rgba(255,170,70,.45)";
    ctx.shadowBlur = size*0.18;
    ctx.fill();

    ctx.shadowBlur = 0;
    ctx.strokeStyle = "rgba(0,0,0,.35)";
    ctx.lineWidth = Math.max(1, size*0.05);
    const sp = size*0.30;
    const lines = [
      [x-sp, y, x+sp, y],
      [x, y-sp, x, y+sp],
      [x-sp*0.72, y-sp*0.72, x+sp*0.72, y+sp*0.72],
      [x-sp*0.72, y+sp*0.72, x+sp*0.72, y-sp*0.72]
    ];
    for (const L of lines){
      ctx.beginPath();
      ctx.moveTo(L[0], L[1]);
      ctx.lineTo(L[2], L[3]);
      ctx.stroke();
    }

    ctx.restore();
  }

  function drawNumber(x,y,size,n){
    ctx.save();
    ctx.font = `700 ${Math.max(12, size*0.42)}px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = NUM_COLORS[n] || "#1c6fb8";

    ctx.shadowColor = "rgba(0,0,0,.25)";
    ctx.shadowBlur = size*0.10;
    ctx.shadowOffsetY = size*0.06;
    ctx.fillText(String(n), x, y);

    ctx.shadowBlur = 0;
    ctx.strokeStyle = "rgba(255,255,255,.28)";
    ctx.lineWidth = Math.max(1, size*0.05);
    ctx.strokeText(String(n), x, y);
    ctx.restore();
  }

  function draw(){
    if (!state.layout){ requestAnimationFrame(draw); return; }

    drawBackground();
    drawFrame();

    const {x,y,cell,gridW,gridH} = state.layout;

    // soft mist overlay (adds ambiguity)
    ctx.save();
    rr(x-12, y-12, gridW+24, gridH+24, 18);
    ctx.clip();
    const mist = ctx.createLinearGradient(x, y, x, y+gridH);
    mist.addColorStop(0, "rgba(255,255,255,.05)");
    mist.addColorStop(1, "rgba(255,255,255,.01)");
    ctx.fillStyle = mist;
    ctx.fillRect(x-12, y-12, gridW+24, gridH+24);
    ctx.restore();

    // cells
    for (let r=0;r<ROWS;r++){
      for (let c=0;c<COLS;c++){
        const i = idx(r,c);
        const v = state.grid ? state.grid[i] : -3;
        if (v >= 0 || v === -1){
          drawOpenCell(r,c,v);
        } else {
          drawCoveredCell(r,c);
          if (v === -2){
            const rc = cellRect(r,c);
            drawFlag(rc.x0 + rc.w*0.52, rc.y0 + rc.w*0.52, rc.w);
          }
        }
      }
    }

    // inner border
    ctx.save();
    rr(x-1, y-1, gridW+2, gridH+2, 18);
    ctx.strokeStyle = "rgba(255,255,255,.10)";
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.restore();

    requestAnimationFrame(draw);
  }

  // ---------------------------------------------------------------------------
  // Input handling (only when running)
  // ---------------------------------------------------------------------------
  function canPlay(){
    return state.summary && state.summary.status === "running" && !!state.grid;
  }

  function posToCell(px,py){
    const {x,y,cell,gridW,gridH} = state.layout;
    if (px < x || py < y || px > x+gridW || py > y+gridH) return null;
    const c = Math.floor((px-x)/cell);
    const r = Math.floor((py-y)/cell);
    if (!inb(r,c)) return null;
    return {r,c};
  }

  function openAction(r,c){
    if (!canPlay()) return;
    send({type:"open", room: state.room, r, c});
  }
  function flagAction(r,c){
    if (!canPlay()) return;
    send({type:"flag", room: state.room, r, c});
  }
  function chordAction(r,c){
    if (!canPlay()) return;
    send({type:"chord", room: state.room, r, c});
  }

  canvas.addEventListener("contextmenu", (e) => e.preventDefault());

  canvas.addEventListener("pointerdown", (e) => {
    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const p = posToCell(px,py);
    state.lastPointerDownAt = Date.now();
    if (!p) return;

    canvas.setPointerCapture(e.pointerId);

    if (e.button === 2){
      flagAction(p.r,p.c);
      return;
    }
    if (e.button === 0){
      openAction(p.r,p.c);
      return;
    }
  }, {passive:true});

  canvas.addEventListener("pointerup", (e) => {
    if (e.pointerType === "touch"){
      const dt = Date.now() - state.lastPointerDownAt;
      if (dt >= 420){
        const rect = canvas.getBoundingClientRect();
        const px = e.clientX - rect.left;
        const py = e.clientY - rect.top;
        const p = posToCell(px,py);
        if (p) flagAction(p.r,p.c);
      }
    }
  }, {passive:true});

  canvas.addEventListener("dblclick", (e) => {
    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const p = posToCell(px,py);
    if (!p) return;
    chordAction(p.r,p.c);
  }, {passive:true});

  // ---------------------------------------------------------------------------
  // Buttons
  // ---------------------------------------------------------------------------
  createBtn.addEventListener("click", () => {
    ensureSession();
    connect("create");
  });

  joinBtn.addEventListener("click", () => {
    ensureSession();
    const code = (roomInput.value||"").trim().toUpperCase();
    if (!code){ setStatus("Įveskite kambario kodą."); return; }
    connect("join", code);
  });

  copyBtn.addEventListener("click", async () => {
    if (!state.room){ setStatus("Nėra kambario kodo."); return; }
    const url = new URL(location.href);
    url.hash = `room=${state.room}`;
    try{
      await navigator.clipboard.writeText(url.toString());
      setStatus("Nuoroda nukopijuota.");
    } catch(e){
      setStatus(`Kopijavimas nepavyko. Nuoroda: <span class="mono">${url.toString()}</span>`);
    }
  });

  nameInput.addEventListener("change", () => {
    const nm = (nameInput.value||"").trim();
    if (!nm) return;
    if (state.room) send({type:"setName", room: state.room, name: nm});
  });

  levelSel.addEventListener("change", () => {
    if (!state.room) return;
    send({type:"setLevel", room: state.room, levelKey: levelSel.value});
  });

  startBtn.addEventListener("click", () => {
    if (!state.room) return;
    send({type:"start", room: state.room});
  });

  newLobbyBtn.addEventListener("click", () => {
    if (!state.room) return;
    send({type:"resetLobby", room: state.room});
  });

  themeSel.addEventListener("change", () => {
    applyTheme(themeSel.value);
    setStatus(`Tema: <strong>${THEMES[themeSel.value].label}</strong>.`);
  });

  // clock tick for HUD
  setInterval(() => {
    if (state.summary && state.summary.status === "running"){
      const left = state.summary.deadlineAt ? (state.summary.deadlineAt - approxServerNow()) : 0;
      clockPill.textContent = `Laikas: ${fmtTime(left)}`;
      timeLabel.textContent = `Vyksta • likę ${fmtTime(left)}`;
    }
  }, 250);

  // Boot
  ensureSession();
  populateThemes();
  applyTheme(state.themeKey);
  resize();
  requestAnimationFrame(draw);
  window.addEventListener("resize", () => resize());

  // Auto-join from URL hash
  const m = (location.hash || "").match(/room=([A-Z0-9]+)/i);
  if (m && m[1]){
    const code = m[1].toUpperCase();
    roomInput.value = code;
    connect("join", code);
  }

})();
</script>
</body>
</html>
"""



# -----------------------------
# SnowMines sub-app (mounted)
# -----------------------------

snow_app = FastAPI(title="Hestio SnowMines — Competitive 30×20")
app.mount("/snowmines", snow_app)

@snow_app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
    return HTMLResponse(SNOWMINES_HTML)


@snow_app.get("/health")
def health() -> Response:
    return Response("ok", media_type="text/plain")


@snow_app.get("/version")
def version() -> Response:
    return Response(f"SnowMines competitive mines_main.py {time.strftime('%Y-%m-%d %H:%M:%S')}", media_type="text/plain")


# =============================================================================
# WebSocket endpoint
# =============================================================================

@snow_app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()

    room: Optional[Room] = None
    session_id: Optional[str] = None

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                await ws_send(ws, {"type": "error", "message": "Blogas JSON."})
                continue

            mtype = msg.get("type")

            # CREATE
            if mtype == "create":
                session_id = str(msg.get("sessionId") or "")
                if not session_id:
                    await ws_send(ws, {"type": "error", "message": "Trūksta sessionId."})
                    continue
                room = await create_room()
                async with room.lock:
                    # join as player
                    if room.is_full():
                        await ws_send(ws, {"type": "error", "message": "Lobby pilnas."})
                        continue

                    # reconnect?
                    p = room.players.get(session_id)
                    if p is None:
                        p = Player(session_id=session_id)
                        room.players[session_id] = p
                    p.ws = ws
                    p.last_seen = now_s()
                    room.ensure_player_arrays(p)

                    if room.host_session is None:
                        room.host_session = session_id

                    await ws_send(ws, {"type": "joined", "room": room.code, "summary": room_summary(room), "grid": None})
                    await broadcast_room(room)
                continue

            # JOIN
            if mtype == "join":
                session_id = str(msg.get("sessionId") or "")
                code = str(msg.get("room") or "").strip().upper()
                if not session_id:
                    await ws_send(ws, {"type": "error", "message": "Trūksta sessionId."})
                    continue
                if not code or len(code) > 8:
                    await ws_send(ws, {"type": "error", "message": "Neteisingas kambario kodas."})
                    continue

                room = await ensure_room(code)
                async with room.lock:
                    purge_stale(room)
                    if session_id not in room.players and room.is_full():
                        await ws_send(ws, {"type": "error", "message": "Lobby pilnas (max 10)." })
                        continue

                    p = room.players.get(session_id)
                    if p is None:
                        p = Player(session_id=session_id)
                        room.players[session_id] = p
                        room.ensure_player_arrays(p)

                    p.ws = ws
                    p.last_seen = now_s()

                    if room.host_session is None:
                        room.host_session = session_id

                    grid = player_view_grid(room, p) if room.status in ("running", "ended") else None
                    await ws_send(ws, {"type": "joined", "room": room.code, "summary": room_summary(room), "grid": grid})
                    await broadcast_room(room)
                continue

            # require room + player
            if room is None or session_id is None:
                await ws_send(ws, {"type": "error", "message": "Pirma prisijunkite prie lobby."})
                continue
            if session_id not in room.players:
                await ws_send(ws, {"type": "error", "message": "Nebėra vietos lobby arba sesija nebegalioja."})
                continue

            p = room.players[session_id]
            p.last_seen = now_s()

            # SET NAME
            if mtype == "setName":
                name = str(msg.get("name") or "").strip()
                if not name:
                    continue
                # keep short and clean
                name = name[:18]
                async with room.lock:
                    p.name = name
                    await broadcast_room(room)
                continue

            # HOST: SET LEVEL (only in lobby)
            if mtype == "setLevel":
                lvl = str(msg.get("levelKey") or "medium")
                async with room.lock:
                    if room.host_session != session_id:
                        await ws_send(ws, {"type": "error", "message": "Tik host gali keisti lygį."})
                        continue
                    if room.status != "lobby":
                        await ws_send(ws, {"type": "error", "message": "Lygį galima keisti tik lobby režime."})
                        continue
                    if lvl not in LEVELS:
                        lvl = "medium"
                    room.level_key = lvl
                    await broadcast_room(room)
                continue

            # HOST: START
            if mtype == "start":
                async with room.lock:
                    if room.host_session != session_id:
                        await ws_send(ws, {"type": "error", "message": "Tik host gali startuoti."})
                        continue
                    if room.status != "lobby":
                        await ws_send(ws, {"type": "error", "message": "Match jau vyksta arba baigtas."})
                        continue
                    if len(room.players) < 1:
                        await ws_send(ws, {"type": "error", "message": "Reikia bent 1 žaidėjo."})
                        continue

                    start_match(room)
                    await broadcast_room(room)
                continue

            # HOST: reset to lobby (new match)
            if mtype == "resetLobby":
                async with room.lock:
                    if room.host_session != session_id:
                        await ws_send(ws, {"type": "error", "message": "Tik host gali perstatyti match."})
                        continue
                    room.status = "lobby"
                    room.start_at = None
                    room.deadline_at = None
                    room.winner_session = None
                    room.end_reason = None
                    # clear shared field for safety
                    room.mines = []
                    room.adj = []
                    for pp in room.players.values():
                        room.ensure_player_arrays(pp)
                        pp.open = [False] * (ROWS * COLS)
                        pp.flag = [False] * (ROWS * COLS)
                        pp.revealed = 0
                        pp.flags = 0
                        pp.dead = False
                        pp.finished_at = None
                    await broadcast_room(room)
                continue

            # Game actions require running
            if mtype in ("open", "flag", "chord"):
                async with room.lock:
                    if room.status != "running":
                        await ws_send(ws, {"type": "error", "message": "Match nevyksta (lobby arba baigtas)." })
                        await ws_send(ws, {"type": "state", "summary": room_summary(room), "grid": None})
                        continue
                    if room.winner_session is not None:
                        end_match(room, room.winner_session, "solved")
                        await broadcast_room(room)
                        continue
                    if room.deadline_at is not None and now_s() >= room.deadline_at:
                        wsid = compute_winner_on_timeout(room)
                        end_match(room, wsid, "timeout")
                        await broadcast_room(room)
                        continue
                    if p.dead or p.finished_at is not None:
                        await ws_send(ws, {"type": "state", "summary": room_summary(room), "grid": player_view_grid(room, p)})
                        continue

                    try:
                        r = int(msg.get("r"))
                        c = int(msg.get("c"))
                    except Exception:
                        await ws_send(ws, {"type": "error", "message": "Blogos koordinatės."})
                        continue
                    if not inb(r, c):
                        await ws_send(ws, {"type": "error", "message": "Už lentos ribų."})
                        continue

                    i = idx(r, c)

                    if mtype == "flag":
                        if p.open[i]:
                            await ws_send(ws, {"type": "state", "summary": room_summary(room), "grid": player_view_grid(room, p)})
                            continue
                        p.flag[i] = not p.flag[i]
                        p.flags += 1 if p.flag[i] else -1
                        if p.flags < 0:
                            p.flags = 0

                    elif mtype == "open":
                        if p.flag[i] or p.open[i]:
                            await ws_send(ws, {"type": "state", "summary": room_summary(room), "grid": player_view_grid(room, p)})
                            continue
                        if room.mines[i]:
                            p.open[i] = True
                            p.dead = True
                            # end match immediately only if you want "first alive wins"; here: player eliminated
                        else:
                            flood_open(room, p, r, c)

                    elif mtype == "chord":
                        # open neighbors if open number and flags match
                        if not p.open[i]:
                            await ws_send(ws, {"type": "state", "summary": room_summary(room), "grid": player_view_grid(room, p)})
                            continue
                        if room.mines[i]:
                            await ws_send(ws, {"type": "state", "summary": room_summary(room), "grid": player_view_grid(room, p)})
                            continue
                        need = int(room.adj[i])
                        if need <= 0:
                            await ws_send(ws, {"type": "state", "summary": room_summary(room), "grid": player_view_grid(room, p)})
                            continue
                        fcnt = 0
                        for rr, cc in neighbors(r, c):
                            if p.flag[idx(rr, cc)]:
                                fcnt += 1
                        if fcnt != need:
                            await ws_send(ws, {"type": "state", "summary": room_summary(room), "grid": player_view_grid(room, p)})
                            continue
                        # open surrounding
                        for rr, cc in neighbors(r, c):
                            ii = idx(rr, cc)
                            if p.flag[ii] or p.open[ii]:
                                continue
                            if room.mines[ii]:
                                p.open[ii] = True
                                p.dead = True
                                break
                            else:
                                flood_open(room, p, rr, cc)

                    # after action
                    check_finish(room, p)
                    # If everyone is dead, end early.
                    if all(pp.dead for pp in room.players.values()):
                        end_match(room, None, "all_dead")
                        await broadcast_room(room)
                        continue
                    if p.finished_at is not None and room.winner_session is None:
                        room.winner_session = session_id
                        end_match(room, room.winner_session, "solved")

                    await broadcast_room(room)
                continue

            await ws_send(ws, {"type": "error", "message": "Nežinomas veiksmų tipas."})

    except WebSocketDisconnect:
        if room is not None and session_id is not None:
            async with room.lock:
                pp = room.players.get(session_id)
                if pp:
                    pp.ws = None
                    pp.last_seen = now_s()
                purge_stale(room)
                await broadcast_room(room)
        if room is not None:
            await delete_room_if_empty(room.code)
    except Exception:
        # best-effort close
        try:
            await ws.close()
        except Exception:
            pass


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

  /* Watermark (HestioPlay logo) */
  body{position:relative;}
  body::before{
    content:"";
    position:fixed;
    inset:0;
    background-image:url('data:image/png;base64,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');
    background-repeat:no-repeat;
    background-position:50% 50%;
    background-size:min(88vmin, 860px);
    opacity:0.12;
    pointer-events:none;
    z-index:0;
  }
  header, main{position:relative; z-index:1;}

  .gameGrid{
    display:flex;
    flex-direction:column;
    gap:10px;
    margin-top:6px;
  }
  .gameCard{
    width:100%;
    border:1px solid rgba(255,255,255,0.14);
    background:rgba(0,0,0,0.28);
    border-radius:12px;
    padding:10px;
    display:flex;
    gap:10px;
    align-items:center;
    cursor:pointer;
    transition:transform .08s ease, border-color .08s ease, background .08s ease;
    text-align:left;
    color:inherit;
  }
  .gameCard:hover{transform:translateY(-1px); border-color:rgba(80,245,255,0.55); background:rgba(0,0,0,0.36);}
  .gameCard.selected{border-color:rgba(80,245,255,0.95); box-shadow:0 0 0 2px rgba(80,245,255,0.15) inset;}
  .thumb{
    width:48px; height:48px; border-radius:12px;
    flex:0 0 48px;
    border:1px solid rgba(255,255,255,0.12);
    position:relative;
    overflow:hidden;
  }
  .thumb.rebuild{background:linear-gradient(135deg, rgba(80,245,255,0.25), rgba(120,80,255,0.25));}
  .thumb.tile{background:linear-gradient(135deg, rgba(255,180,80,0.22), rgba(255,80,200,0.22));}
  .thumb.snowmines{background:linear-gradient(135deg, rgba(130,210,255,0.26), rgba(255,255,255,0.10));}
  .thumb::after{
    content:"";
    position:absolute; inset:0;
    background:radial-gradient(circle at 30% 25%, rgba(255,255,255,0.22), rgba(255,255,255,0) 55%);
    mix-blend-mode:screen;
  }
  .gmeta{display:flex; flex-direction:column; gap:2px; min-width:0;}
  .gname{font-weight:900; letter-spacing:.2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
  .gsub{font-size:12px; opacity:.75; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}

  .playersCtl{display:flex; align-items:center; gap:10px; justify-content:flex-start; margin-top:6px;}
  .miniBtn{
    width:34px; height:34px;
    border-radius:10px;
    border:1px solid rgba(255,255,255,0.16);
    background:rgba(0,0,0,0.28);
    color:#eaf0ff;
    font-size:18px;
    cursor:pointer;
  }
  .miniBtn:hover{border-color:rgba(80,245,255,0.55);}
  .playersBadge{
    min-width:64px; height:34px;
    border-radius:12px;
    display:flex; align-items:center; justify-content:center;
    font-weight:900;
    border:1px solid rgba(255,255,255,0.16);
    background:linear-gradient(90deg, rgba(80,245,255,0.22), rgba(255,80,200,0.18));
  }

  </style>
</head>
<body>
<header>
  <div class="brand"><span class="brandIcon"><img alt="HestioPlay" src="data:image/png;base64,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"/></span><span class="brandText">HestioPlay</span><span class="pill">Training</span></div>
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
      
      <div style="margin-top:2px;">
        <label class="muted">Game</label>
        <div class="gameGrid" id="gameCards"></div>
        <input type="hidden" id="gameType" value="rebuild_6x6" />
      </div>

      <div class="row" style="margin-top:10px;">
        <label class="muted">Players</label>
        <div class="playersCtl">
          <button class="miniBtn" id="plMinus" type="button">−</button>
          <div class="playersBadge"><span id="plVal">2</span></div>
          <button class="miniBtn" id="plPlus" type="button">+</button>
          <input type="hidden" id="targetPlayers" value="2" />
        </div>
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
    {id:"tile_3x3", name:"Tile Puzzle (3×3)", hint:"Drag tiles into the correct positions to recreate the image."},
    {id:"snowmines_30x20", name:"SnowMines (30×20)", hint:"Competitive Mines. Left click = open, right click = flag. Same board for everyone, 3:00 cap. Opens in a dedicated game page."}
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

    function selectGame(id){
    byId("gameType").value = id;
    const g = GAME_LIST.find(x=>x.id===id);
    byId("gameHint").textContent = (g && g.hint) ? g.hint : "";
    const cards = byId("gameCards");
    if(cards){
      for(const el of cards.querySelectorAll(".gameCard")){
        el.classList.toggle("selected", el.dataset.game===id);
      }
    }
  }

  function initLobby(){
    // Build game cards (image tiles instead of dropdown)
    const cards = byId("gameCards");
    if(cards){
      cards.innerHTML = "";
      for(const g of GAME_LIST){
        const btn=document.createElement("button");
        btn.type="button";
        btn.className="gameCard";
        btn.dataset.game=g.id;

        const thumb=document.createElement("div");
        thumb.className="thumb " + (g.id==="rebuild_6x6" ? "rebuild" : (g.id==="snowmines_30x20" ? "snowmines" : "tile"));

        const meta=document.createElement("div");
        meta.className="gmeta";

        const name=document.createElement("div");
        name.className="gname";
        name.textContent=g.name;

        const sub=document.createElement("div");
        sub.className="gsub";
        sub.textContent=g.hint;

        meta.appendChild(name);
        meta.appendChild(sub);

        btn.appendChild(thumb);
        btn.appendChild(meta);

        btn.addEventListener("click", ()=>selectGame(g.id));
        cards.appendChild(btn);
      }
    }

    // Players (pretty badge + +/-)
    const hidden = byId("targetPlayers");
    const badge = byId("plVal");
    function clamp(n){ return Math.max(1, Math.min(10, n)); }
    function setPlayers(n){
      n = clamp(n);
      if(hidden) hidden.value = String(n);
      if(badge) badge.textContent = String(n);
    }
    const initial = parseInt((hidden && hidden.value) ? hidden.value : "2", 10);
    setPlayers(initial);

    const minus = byId("plMinus");
    const plus  = byId("plPlus");
    if(minus) minus.onclick = ()=> setPlayers(parseInt(hidden.value||"2",10)-1);
    if(plus)  plus.onclick  = ()=> setPlayers(parseInt(hidden.value||"2",10)+1);

    // Default game
    // Prefill nick from HestioRooms (shared localStorage key)
    const pref = preferredNick();
    if(pref && byId("name")) { byId("name").value = pref; persistNick(pref); }

    selectGame(byId("gameType").value || "rebuild_6x6");
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

    // SnowMines runs on a dedicated page (with its own lobby + websocket).
    if(gameType==="snowmines_30x20"){
      const nick = (name||preferredNick()||"").trim();
      if(nick){ byId("name").value = nick; persistNick(nick); }
      const prefix = location.pathname.replace(/\/play\/?$/, "");
      window.location.href = (prefix || "") + "/snowmines/?nick=" + encodeURIComponent(nick);
      return;
    }

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

    # Keep the room in "results" until players explicitly decide what to do next (play again, switch games, or leave).
    # This prevents the server from auto-starting a new round immediately, which would cut the results screen short.
    room.game = None
    await manager.mark_room_not_joinable(room)
    return


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
