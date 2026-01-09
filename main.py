import asyncio
import json
import random
import re
import sqlite3
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, Optional, Set, Tuple

from zoneinfo import ZoneInfo
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

app = FastAPI()

# =====================================
# TIMEZONE (Vilnius)
# =====================================
TZ = ZoneInfo("Europe/Vilnius")


def ts() -> str:
    return datetime.now(TZ).strftime("%H:%M:%S")


# =====================================
# CONFIG
# =====================================
DB_PATH = "chat.db"
HISTORY_LIMIT = 300
CLIENT_CACHE_LIMIT = 600

# message edit window (seconds)
EDIT_WINDOW_SEC = 5 * 60

# rate limiting (per-user)
MAX_MSG_PER_10S = 20
MAX_CHARS_PER_10S = 2500

# Allowed nick: 2..24
NICK_RE = re.compile(r"^[A-Za-z0-9ĄČĘĖĮŠŲŪŽąčęėįšųūž_\-\. ]{2,24}$")
ROOM_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]{0,23}$")  # "main", "games", "help" etc.

ROLL_RE = re.compile(r"^/roll(?:\s+(\d{1,2})d(\d{1,3}))?$", re.IGNORECASE)

COLOR_PALETTE = [
    "#E6194B", "#3CB44B", "#FFE119", "#0082C8", "#F58231",
    "#911EB4", "#46F0F0", "#F032E6", "#D2F53C", "#FABEBE",
    "#008080", "#E6BEFF", "#AA6E28", "#FFD8B1", "#800000",
    "#AFFFc3", "#808000", "#000080", "#808080", "#FFFFFF",
]

DEFAULT_ROOMS = {
    "main": {"title": "#main", "topic": "Bendras kanalas"},
    "games": {"title": "#games", "topic": "Žaidimai ir pramogos"},
    "help": {"title": "#help", "topic": "Pagalba / klausimai"},
}

HELP_TEXT = (
    "Komandos:\n"
    "  /help                          - pagalba\n"
    "  /rooms                         - kanalų sąrašas\n"
    "  /join #room                    - prisijungti/sukurti kanalą\n"
    "  /leave #room                   - palikti kanalą\n"
    "  /topic                         - parodyti temą (aktyviame kanale)\n"
    "  /topic TEKSTAS                 - pakeisti temą (aktyviame kanale)\n"
    "  /who                           - kas online (aktyviame kanale)\n"
    "  /dm VARDAS ŽINUTĖ              - private žinutė (DM)\n"
    "  /dmhistory VARDAS [N]          - DM istorija su vartotoju\n"
    "  /history [N]                   - kanalo istorija (default 120)\n"
    "  /me veiksmas                   - action žinutė\n"
    "  /roll [NdM]                    - kauliukas (pvz. /roll 2d6)\n"
    "  /flip                          - monetos metimas\n"
    "  /time                          - serverio laikas (Vilnius)\n"
    "  /pin ID                        - prisegti žinutę (pin) aktyviame kanale\n"
    "  /pins                          - parodyti pin’us\n"
    "  /quote ID                      - pacituoti žinutę\n"
    "  /edit ID NAUJAS_TEKSTAS         - redaguoti savo žinutę (iki 5 min)\n"
    "  /del ID                        - ištrinti savo žinutę (iki 5 min)\n"
    "  /react ID 😀                    - reakcija į žinutę\n"
    "  /game start                    - pradėti 'Guess 1..100' žaidimą kanale\n"
    "  /guess SKAIČIUS                - spėti skaičių (aktyviame kanale)\n"
)

# =====================================
# DB (SQLite) – persistencija
# =====================================
db_lock = asyncio.Lock()


def db() -> sqlite3.Connection:
    # check_same_thread=False dėl async aplinkos; lock užtikrina saugumą
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def db_init() -> None:
    conn = db()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          scope TEXT NOT NULL,               -- 'room' arba 'dm'
          room TEXT,                         -- jei scope='room'
          a TEXT, b TEXT,                    -- jei scope='dm' (pair)
          msg_type TEXT NOT NULL,            -- 'msg','sys','me_action','deleted'
          ts TEXT NOT NULL,
          nick TEXT,
          color TEXT,
          text TEXT,
          extra TEXT                          -- JSON
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS pins (
          room TEXT NOT NULL,
          msg_id INTEGER NOT NULL,
          PRIMARY KEY(room, msg_id)
        )
        """
    )

    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_messages_room ON messages(scope, room, id)
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_messages_dm ON messages(scope, a, b, id)
        """
    )

    conn.commit()
    conn.close()


db_init()

# =====================================
# MODELS
# =====================================
@dataclass
class User:
    nick: str
    color: str
    status: str = "online"     # online/away
    joined_at: float = field(default_factory=time.time)
    rooms: Set[str] = field(default_factory=set)

    # rate limiting
    msg_events: Deque[Tuple[float, int]] = field(default_factory=deque)  # (time, chars)


@dataclass
class Room:
    key: str
    title: str
    topic: str
    clients: Set[WebSocket] = field(default_factory=set)
    users: Dict[WebSocket, User] = field(default_factory=dict)
    typing: Set[str] = field(default_factory=set)  # nick set
    game_target: Optional[int] = None  # for /game start
    game_active: bool = False


# =====================================
# IN-MEMORY STATE
# =====================================
rooms: Dict[str, Room] = {}
all_users_by_ws: Dict[WebSocket, User] = {}
all_ws_by_nick_cf: Dict[str, WebSocket] = {}

state_lock = asyncio.Lock()


def valid_nick(n: str) -> bool:
    return bool(NICK_RE.match(n))


def norm_room(room: str) -> str:
    r = (room or "").strip().lstrip("#").lower()
    return r


def valid_room_key(r: str) -> bool:
    return bool(ROOM_RE.match(r))


def alloc_color(used: Set[str]) -> str:
    for c in COLOR_PALETTE:
        if c not in used:
            return c
    hue = random.randint(0, 359)
    return f"hsl({hue}, 90%, 60%)"


def room_used_colors(r: Room) -> Set[str]:
    return {u.color for u in r.users.values()}


def dm_pair(a: str, b: str) -> Tuple[str, str]:
    x, y = a.casefold(), b.casefold()
    return (x, y) if x <= y else (y, x)


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def json_loads(s: Optional[str]) -> dict:
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        return {}


# =====================================
# DB helpers (async-safe via lock)
# =====================================
async def db_insert_message(msg: dict) -> int:
    """
    msg fields:
    - scope: 'room'/'dm'
    - room OR (a,b)
    - msg_type, ts, nick, color, text, extra(dict)
    """
    async with db_lock:
        conn = db()
        cur = conn.cursor()
        extra = json_dumps(msg.get("extra", {}))

        cur.execute(
            """
            INSERT INTO messages(scope, room, a, b, msg_type, ts, nick, color, text, extra)
            VALUES(?,?,?,?,?,?,?,?,?,?)
            """,
            (
                msg["scope"],
                msg.get("room"),
                msg.get("a"),
                msg.get("b"),
                msg.get("msg_type"),
                msg.get("ts"),
                msg.get("nick"),
                msg.get("color"),
                msg.get("text"),
                extra,
            ),
        )
        conn.commit()
        msg_id = int(cur.lastrowid)
        conn.close()
        return msg_id


async def db_update_message_text(msg_id: int, new_text: str, new_extra: dict) -> None:
    async with db_lock:
        conn = db()
        cur = conn.cursor()
        cur.execute(
            "UPDATE messages SET text=?, extra=? WHERE id=?",
            (new_text, json_dumps(new_extra), msg_id),
        )
        conn.commit()
        conn.close()


async def db_mark_deleted(msg_id: int) -> None:
    async with db_lock:
        conn = db()
        cur = conn.cursor()
        cur.execute(
            "UPDATE messages SET msg_type=?, text=?, extra=? WHERE id=?",
            ("deleted", "", json_dumps({"deleted": True}), msg_id),
        )
        conn.commit()
        conn.close()


async def db_get_message(msg_id: int) -> Optional[sqlite3.Row]:
    async with db_lock:
        conn = db()
        cur = conn.cursor()
        cur.execute("SELECT * FROM messages WHERE id=?", (msg_id,))
        row = cur.fetchone()
        conn.close()
        return row


async def db_load_room_history(room_key: str, limit: int) -> list[dict]:
    limit = max(1, min(limit, HISTORY_LIMIT))
    async with db_lock:
        conn = db()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT * FROM messages
            WHERE scope='room' AND room=?
            ORDER BY id DESC
            LIMIT ?
            """,
            (room_key, limit),
        )
        rows = cur.fetchall()
        conn.close()

    items = []
    for r in reversed(rows):
        items.append(
            {
                "id": int(r["id"]),
                "type": r["msg_type"],
                "t": r["ts"],
                "nick": r["nick"],
                "color": r["color"],
                "text": r["text"],
                "extra": json_loads(r["extra"]),
            }
        )
    return items


async def db_load_dm_history(a_nick: str, b_nick: str, limit: int) -> list[dict]:
    limit = max(1, min(limit, HISTORY_LIMIT))
    a, b = dm_pair(a_nick, b_nick)
    async with db_lock:
        conn = db()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT * FROM messages
            WHERE scope='dm' AND a=? AND b=?
            ORDER BY id DESC
            LIMIT ?
            """,
            (a, b, limit),
        )
        rows = cur.fetchall()
        conn.close()

    items = []
    for r in reversed(rows):
        extra = json_loads(r["extra"])
        items.append(
            {
                "id": int(r["id"]),
                "type": "dm",
                "t": r["ts"],
                "from": extra.get("from"),
                "to": extra.get("to"),
                "color": r["color"],
                "text": r["text"],
                "extra": extra,
            }
        )
    return items


async def db_pin(room_key: str, msg_id: int) -> None:
    async with db_lock:
        conn = db()
        cur = conn.cursor()
        cur.execute("INSERT OR IGNORE INTO pins(room, msg_id) VALUES(?,?)", (room_key, msg_id))
        conn.commit()
        conn.close()


async def db_unpin(room_key: str, msg_id: int) -> None:
    async with db_lock:
        conn = db()
        cur = conn.cursor()
        cur.execute("DELETE FROM pins WHERE room=? AND msg_id=?", (room_key, msg_id))
        conn.commit()
        conn.close()


async def db_list_pins(room_key: str, limit: int = 50) -> list[int]:
    async with db_lock:
        conn = db()
        cur = conn.cursor()
        cur.execute("SELECT msg_id FROM pins WHERE room=? ORDER BY msg_id DESC LIMIT ?", (room_key, limit))
        rows = cur.fetchall()
        conn.close()
    return [int(r["msg_id"]) for r in rows]


# =====================================
# SERVER OPERATIONS
# =====================================
def ensure_room(room_key: str) -> Room:
    if room_key in rooms:
        return rooms[room_key]
    # create new room (default values)
    title = f"#{room_key}"
    topic = "Naujas kanalas"
    if room_key in DEFAULT_ROOMS:
        title = DEFAULT_ROOMS[room_key]["title"]
        topic = DEFAULT_ROOMS[room_key]["topic"]
    r = Room(key=room_key, title=title, topic=topic)
    rooms[room_key] = r
    return r


def room_userlist(room_key: str) -> list[dict]:
    r = ensure_room(room_key)
    out = []
    for u in r.users.values():
        out.append({"nick": u.nick, "color": u.color, "status": u.status})
    out.sort(key=lambda x: x["nick"].casefold())
    return out


async def ws_send(ws: WebSocket, obj: dict) -> None:
    try:
        await ws.send_json(obj)
    except Exception:
        # ignore send errors
        pass


async def room_broadcast(room_key: str, obj: dict, exclude: Optional[WebSocket] = None) -> None:
    r = ensure_room(room_key)
    dead = []
    for w in list(r.clients):
        if exclude is not None and w is exclude:
            continue
        try:
            await w.send_json(obj)
        except Exception:
            dead.append(w)
    for w in dead:
        await disconnect_ws(w)


async def broadcast_userlist(room_key: str) -> None:
    await room_broadcast(room_key, {"type": "users", "room": room_key, "items": room_userlist(room_key)})


async def broadcast_rooms_list(ws: WebSocket) -> None:
    # send only rooms where user is a member + discoverable defaults
    u = all_users_by_ws.get(ws)
    if not u:
        return
    # include user's rooms + defaults
    keys = set(u.rooms) | set(DEFAULT_ROOMS.keys())
    items = []
    for k in sorted(keys):
        r = ensure_room(k)
        items.append({"room": k, "title": r.title, "topic": r.topic})
    await ws_send(ws, {"type": "rooms", "items": items})


async def disconnect_ws(ws: WebSocket) -> None:
    async with state_lock:
        u = all_users_by_ws.pop(ws, None)
        if u:
            all_ws_by_nick_cf.pop(u.nick.casefold(), None)
            # remove from rooms
            for rk in list(u.rooms):
                r = ensure_room(rk)
                r.clients.discard(ws)
                r.users.pop(ws, None)
                # typing cleanup
                r.typing.discard(u.nick)

    try:
        await ws.close()
    except Exception:
        pass

    # broadcast user lists for affected rooms
    if u:
        for rk in list(u.rooms):
            await broadcast_userlist(rk)
            await broadcast_typing(rk)


async def join_room(ws: WebSocket, room_key: str) -> Tuple[bool, str]:
    room_key = norm_room(room_key)
    if not valid_room_key(room_key):
        return False, "Netinkamas kanalo pavadinimas. Pvz: #main, #games, #help (tik a-z, 0-9, _ -)."

    u = all_users_by_ws.get(ws)
    if not u:
        return False, "Neidentifikuotas vartotojas."

    async with state_lock:
        r = ensure_room(room_key)
        if room_key in u.rooms:
            return True, "Jau esate šiame kanale."
        u.rooms.add(room_key)
        r.clients.add(ws)
        r.users[ws] = u

    await ws_send(ws, {"type": "sys", "room": room_key, "t": ts(), "text": f"Prisijungėte prie {ensure_room(room_key).title}."})
    await broadcast_userlist(room_key)
    await broadcast_rooms_list(ws)
    # send room topic + history snapshot
    await ws_send(ws, {"type": "topic", "room": room_key, "text": f"{ensure_room(room_key).title} — {ensure_room(room_key).topic}"})
    hist = await db_load_room_history(room_key, 120)
    await ws_send(ws, {"type": "history", "room": room_key, "items": hist})
    return True, "OK"


async def leave_room(ws: WebSocket, room_key: str) -> Tuple[bool, str]:
    room_key = norm_room(room_key)
    u = all_users_by_ws.get(ws)
    if not u:
        return False, "Neidentifikuotas vartotojas."

    if room_key == "main":
        return False, "Iš #main išeiti negalima (bazinis kanalas)."

    async with state_lock:
        if room_key not in u.rooms:
            return False, "Jūs nesate šiame kanale."
        u.rooms.discard(room_key)
        r = ensure_room(room_key)
        r.clients.discard(ws)
        r.users.pop(ws, None)
        r.typing.discard(u.nick)

    await ws_send(ws, {"type": "sys", "t": ts(), "room": room_key, "text": f"Palikote {ensure_room(room_key).title}."})
    await broadcast_userlist(room_key)
    await broadcast_typing(room_key)
    await broadcast_rooms_list(ws)
    return True, "OK"


async def broadcast_typing(room_key: str) -> None:
    r = ensure_room(room_key)
    await room_broadcast(room_key, {"type": "typing", "room": room_key, "items": sorted(r.typing)})


def check_rate_limit(u: User, msg_len: int) -> Tuple[bool, str]:
    now = time.time()
    # drop old
    while u.msg_events and now - u.msg_events[0][0] > 10:
        u.msg_events.popleft()
    # compute
    count = len(u.msg_events)
    chars = sum(c for _, c in u.msg_events)

    if count >= MAX_MSG_PER_10S:
        return False, "Per daug žinučių per trumpą laiką. Palaukite kelias sekundes."
    if chars + msg_len > MAX_CHARS_PER_10S:
        return False, "Per daug teksto per trumpą laiką. Palaukite kelias sekundes."

    u.msg_events.append((now, msg_len))
    return True, "OK"


# =====================================
# COMMANDS
# =====================================
async def cmd_rooms(ws: WebSocket) -> None:
    u = all_users_by_ws.get(ws)
    if not u:
        return
    keys = sorted(set(u.rooms) | set(DEFAULT_ROOMS.keys()))
    lines = ["Kanalai:"]
    for k in keys:
        r = ensure_room(k)
        lines.append(f"  {r.title} — {r.topic}")
    await ws_send(ws, {"type": "sys", "t": ts(), "text": "\n".join(lines)})


async def cmd_who(ws: WebSocket, room_key: str) -> None:
    room_key = norm_room(room_key)
    if room_key not in rooms:
        ensure_room(room_key)
    items = room_userlist(room_key)
    names = [x["nick"] + (" (away)" if x["status"] == "away" else "") for x in items]
    await ws_send(ws, {"type": "sys", "t": ts(), "text": f"Online {ensure_room(room_key).title}: " + ", ".join(names)})


async def cmd_topic(ws: WebSocket, room_key: str, arg: Optional[str]) -> None:
    room_key = norm_room(room_key)
    r = ensure_room(room_key)
    if not arg:
        await ws_send(ws, {"type": "sys", "t": ts(), "text": f"{r.title} tema: {r.topic}"})
        return
    new_topic = arg.strip()[:120] or r.topic
    r.topic = new_topic
    # store as sys message (optional)
    sys_msg = {
        "scope": "room",
        "room": room_key,
        "msg_type": "sys",
        "ts": ts(),
        "nick": None,
        "color": None,
        "text": f"Tema pakeista į: {new_topic}",
        "extra": {"kind": "topic_change"},
    }
    msg_id = await db_insert_message(sys_msg)
    await room_broadcast(room_key, {"type": "topic", "room": room_key, "text": f"{r.title} — {new_topic}"})
    await room_broadcast(room_key, {"type": "sys", "room": room_key, "t": sys_msg["ts"], "text": sys_msg["text"], "id": msg_id})
    # update room list display for clients
    for w in list(r.clients):
        await broadcast_rooms_list(w)


async def cmd_history(ws: WebSocket, room_key: str, n: int) -> None:
    room_key = norm_room(room_key)
    n = max(1, min(n, HISTORY_LIMIT))
    items = await db_load_room_history(room_key, n)
    await ws_send(ws, {"type": "history", "room": room_key, "items": items})


async def cmd_dm(ws: WebSocket, target_nick: str, text: str) -> None:
    u = all_users_by_ws.get(ws)
    if not u:
        return
    target = (target_nick or "").strip()
    if not target or not text:
        await ws_send(ws, {"type": "sys", "t": ts(), "text": "Naudojimas: /dm VARDAS ŽINUTĖ"})
        return

    async with state_lock:
        target_ws = all_ws_by_nick_cf.get(target.casefold())
        if not target_ws:
            await ws_send(ws, {"type": "sys", "t": ts(), "text": f"Vartotojas nerastas online: {target}"})
            return
        target_user = all_users_by_ws.get(target_ws)
        if not target_user:
            await ws_send(ws, {"type": "sys", "t": ts(), "text": f"Vartotojas nerastas online: {target}"})
            return

    # persist DM
    a, b = dm_pair(u.nick, target_user.nick)
    extra = {"from": u.nick, "to": target_user.nick}
    msg = {
        "scope": "dm",
        "a": a,
        "b": b,
        "msg_type": "dm",
        "ts": ts(),
        "nick": u.nick,
        "color": u.color,
        "text": text[:300],
        "extra": extra,
    }
    msg_id = await db_insert_message(msg)

    payload = {
        "type": "dm",
        "id": msg_id,
        "t": msg["ts"],
        "from": u.nick,
        "to": target_user.nick,
        "color": u.color,
        "text": msg["text"],
        "extra": extra,
    }

    await ws_send(ws, payload)
    if target_ws is not ws:
        await ws_send(target_ws, payload)


async def cmd_dmhistory(ws: WebSocket, with_nick: str, n: int) -> None:
    u = all_users_by_ws.get(ws)
    if not u:
        return
    n = max(1, min(n, HISTORY_LIMIT))
    items = await db_load_dm_history(u.nick, with_nick, n)
    await ws_send(ws, {"type": "dm_history", "with": with_nick, "items": items})


async def cmd_roll(ws: WebSocket, room_key: str, n: int, sides: int) -> None:
    u = all_users_by_ws.get(ws)
    if not u:
        return
    room_key = norm_room(room_key)
    n = max(1, min(n, 20))
    sides = max(2, min(sides, 1000))
    rolls = [random.randint(1, sides) for _ in range(n)]
    total = sum(rolls)
    text = f"{u.nick} meta {n}d{sides}: {rolls} (viso {total})"
    msg = {
        "scope": "room",
        "room": room_key,
        "msg_type": "sys",
        "ts": ts(),
        "nick": None,
        "color": None,
        "text": text,
        "extra": {"kind": "roll", "by": u.nick},
    }
    msg_id = await db_insert_message(msg)
    await room_broadcast(room_key, {"type": "sys", "room": room_key, "t": msg["ts"], "text": text, "id": msg_id})


async def cmd_flip(ws: WebSocket, room_key: str) -> None:
    u = all_users_by_ws.get(ws)
    if not u:
        return
    room_key = norm_room(room_key)
    res = random.choice(["HERBAS", "SKAIČIUS"])
    text = f"{u.nick} meta monetą: {res}"
    msg = {
        "scope": "room",
        "room": room_key,
        "msg_type": "sys",
        "ts": ts(),
        "nick": None,
        "color": None,
        "text": text,
        "extra": {"kind": "flip", "by": u.nick},
    }
    msg_id = await db_insert_message(msg)
    await room_broadcast(room_key, {"type": "sys", "room": room_key, "t": msg["ts"], "text": text, "id": msg_id})


async def cmd_me(ws: WebSocket, room_key: str, action: str) -> None:
    u = all_users_by_ws.get(ws)
    if not u:
        return
    room_key = norm_room(room_key)
    action = (action or "").strip()[:240]
    if not action:
        return
    msg = {
        "scope": "room",
        "room": room_key,
        "msg_type": "me_action",
        "ts": ts(),
        "nick": u.nick,
        "color": u.color,
        "text": action,
        "extra": {"kind": "me"},
    }
    msg_id = await db_insert_message(msg)
    await room_broadcast(
        room_key,
        {"type": "me_action", "room": room_key, "t": msg["ts"], "nick": u.nick, "color": u.color, "text": action, "id": msg_id},
    )


async def cmd_pin(ws: WebSocket, room_key: str, msg_id: int) -> None:
    room_key = norm_room(room_key)
    row = await db_get_message(msg_id)
    if not row or row["scope"] != "room" or row["room"] != room_key:
        await ws_send(ws, {"type": "sys", "t": ts(), "text": "Pin nepavyko: žinutė nerasta šiame kanale."})
        return
    await db_pin(room_key, msg_id)
    await room_broadcast(room_key, {"type": "pin_update", "room": room_key, "action": "pin", "id": msg_id})
    await ws_send(ws, {"type": "sys", "t": ts(), "text": f"Prisegta žinutė ID={msg_id}."})


async def cmd_pins(ws: WebSocket, room_key: str) -> None:
    room_key = norm_room(room_key)
    ids = await db_list_pins(room_key, 50)
    if not ids:
        await ws_send(ws, {"type": "sys", "t": ts(), "text": "Pin’ų nėra."})
        return
    await ws_send(ws, {"type": "pins", "room": room_key, "items": ids})


async def cmd_quote(ws: WebSocket, room_key: str, msg_id: int) -> None:
    room_key = norm_room(room_key)
    row = await db_get_message(msg_id)
    if not row or row["scope"] != "room" or row["room"] != room_key:
        await ws_send(ws, {"type": "sys", "t": ts(), "text": "Quote nepavyko: žinutė nerasta šiame kanale."})
        return
    nick = row["nick"] or "sys"
    text = row["text"] or ""
    await ws_send(ws, {"type": "quote", "room": room_key, "id": msg_id, "nick": nick, "text": text})


async def cmd_edit(ws: WebSocket, room_key: str, msg_id: int, new_text: str) -> None:
    u = all_users_by_ws.get(ws)
    if not u:
        return
    room_key = norm_room(room_key)
    row = await db_get_message(msg_id)
    if not row or row["scope"] != "room" or row["room"] != room_key:
        await ws_send(ws, {"type": "sys", "t": ts(), "text": "Edit nepavyko: žinutė nerasta šiame kanale."})
        return
    if (row["nick"] or "") != u.nick:
        await ws_send(ws, {"type": "sys", "t": ts(), "text": "Edit nepavyko: galima redaguoti tik savo žinutes."})
        return
    # time window
    try:
        # we store ts string only; use row id as proxy? We'll use insertion time (not exact), but keep window check by storing extra 'created_at' in extra
        extra = json_loads(row["extra"])
        created_at = extra.get("created_at")
        if not created_at:
            # fallback: allow edit within window based on current time and joined_at (rough) – better than nothing
            created_at = time.time()
    except Exception:
        extra = {}
        created_at = time.time()

    # Better: store created_at for new messages; for old ones, allow once
    now = time.time()
    if extra.get("created_at") and (now - float(extra["created_at"]) > EDIT_WINDOW_SEC):
        await ws_send(ws, {"type": "sys", "t": ts(), "text": "Edit nepavyko: praėjo laiko limitas (5 min)."})
        return

    new_text = (new_text or "").strip()[:300]
    if not new_text:
        await ws_send(ws, {"type": "sys", "t": ts(), "text": "Edit nepavyko: tuščias tekstas."})
        return

    extra["edited"] = True
    extra.setdefault("created_at", now)  # if missing
    await db_update_message_text(msg_id, new_text, extra)
    await room_broadcast(room_key, {"type": "edit", "room": room_key, "id": msg_id, "text": new_text, "edited": True})


async def cmd_delete(ws: WebSocket, room_key: str, msg_id: int) -> None:
    u = all_users_by_ws.get(ws)
    if not u:
        return
    room_key = norm_room(room_key)
    row = await db_get_message(msg_id)
    if not row or row["scope"] != "room" or row["room"] != room_key:
        await ws_send(ws, {"type": "sys", "t": ts(), "text": "Del nepavyko: žinutė nerasta šiame kanale."})
        return
    if (row["nick"] or "") != u.nick:
        await ws_send(ws, {"type": "sys", "t": ts(), "text": "Del nepavyko: galima ištrinti tik savo žinutes."})
        return

    extra = json_loads(row["extra"])
    now = time.time()
    if extra.get("created_at") and (now - float(extra["created_at"]) > EDIT_WINDOW_SEC):
        await ws_send(ws, {"type": "sys", "t": ts(), "text": "Del nepavyko: praėjo laiko limitas (5 min)."})
        return

    await db_mark_deleted(msg_id)
    await room_broadcast(room_key, {"type": "delete", "room": room_key, "id": msg_id})


async def cmd_react(ws: WebSocket, room_key: str, msg_id: int, emoji: str) -> None:
    u = all_users_by_ws.get(ws)
    if not u:
        return
    room_key = norm_room(room_key)
    row = await db_get_message(msg_id)
    if not row:
        await ws_send(ws, {"type": "sys", "t": ts(), "text": "React nepavyko: žinutė nerasta."})
        return
    if row["scope"] != "room" or row["room"] != room_key:
        await ws_send(ws, {"type": "sys", "t": ts(), "text": "React nepavyko: žinutė ne iš šio kanalo."})
        return

    emoji = (emoji or "").strip()
    if not emoji:
        await ws_send(ws, {"type": "sys", "t": ts(), "text": "Naudojimas: /react ID 😀"})
        return
    # minimal sanity: keep small
    emoji = emoji[:8]

    extra = json_loads(row["extra"])
    reactions = extra.get("reactions", {})  # emoji -> [nicks]
    lst = reactions.get(emoji, [])
    if u.nick in lst:
        lst.remove(u.nick)
    else:
        lst.append(u.nick)
    if lst:
        reactions[emoji] = lst
    else:
        reactions.pop(emoji, None)
    extra["reactions"] = reactions

    await db_update_message_text(msg_id, row["text"] or "", extra)
    await room_broadcast(room_key, {"type": "react_update", "room": room_key, "id": msg_id, "reactions": reactions})


async def cmd_game_start(ws: WebSocket, room_key: str) -> None:
    room_key = norm_room(room_key)
    r = ensure_room(room_key)
    r.game_target = random.randint(1, 100)
    r.game_active = True

    text = "Žaidimas pradėtas: atspėk skaičių 1..100. Komanda: /guess 42"
    msg = {
        "scope": "room",
        "room": room_key,
        "msg_type": "sys",
        "ts": ts(),
        "nick": None,
        "color": None,
        "text": text,
        "extra": {"kind": "game_start"},
    }
    msg_id = await db_insert_message(msg)
    await room_broadcast(room_key, {"type": "sys", "room": room_key, "t": msg["ts"], "text": text, "id": msg_id})


async def cmd_guess(ws: WebSocket, room_key: str, guess: int) -> None:
    u = all_users_by_ws.get(ws)
    if not u:
        return
    room_key = norm_room(room_key)
    r = ensure_room(room_key)
    if not r.game_active or not r.game_target:
        await ws_send(ws, {"type": "sys", "t": ts(), "text": "Žaidimas nepradėtas. Paleiskite /game start"})
        return
    guess = int(guess)
    if guess < 1 or guess > 100:
        await ws_send(ws, {"type": "sys", "t": ts(), "text": "Spėjimas turi būti 1..100."})
        return

    if guess == r.game_target:
        r.game_active = False
        target = r.game_target
        r.game_target = None
        text = f"{u.nick} atspėjo skaičių! Buvo {target}. Jei norit dar kartą: /game start"
        msg_type = "sys"
        extra = {"kind": "game_win", "by": u.nick, "target": target}
    elif guess < r.game_target:
        text = f"{u.nick} spėja {guess}: per mažai."
        msg_type = "sys"
        extra = {"kind": "game_hint", "by": u.nick, "guess": guess, "hint": "low"}
    else:
        text = f"{u.nick} spėja {guess}: per daug."
        msg_type = "sys"
        extra = {"kind": "game_hint", "by": u.nick, "guess": guess, "hint": "high"}

    msg = {
        "scope": "room",
        "room": room_key,
        "msg_type": msg_type,
        "ts": ts(),
        "nick": None,
        "color": None,
        "text": text,
        "extra": extra,
    }
    msg_id = await db_insert_message(msg)
    await room_broadcast(room_key, {"type": "sys", "room": room_key, "t": msg["ts"], "text": text, "id": msg_id})


async def handle_command(ws: WebSocket, active_room: str, text: str) -> bool:
    """
    Commands are executed in context of active_room.
    """
    t = text.strip()
    low = t.lower()

    if low in ("/help", "/?"):
        await ws_send(ws, {"type": "sys", "t": ts(), "text": HELP_TEXT})
        return True

    if low == "/rooms":
        await cmd_rooms(ws)
        return True

    if low.startswith("/join "):
        arg = t.split(" ", 1)[1].strip()
        if not arg:
            return True
        ok, msg = await join_room(ws, arg)
        if not ok:
            await ws_send(ws, {"type": "sys", "t": ts(), "text": msg})
        return True

    if low.startswith("/leave "):
        arg = t.split(" ", 1)[1].strip()
        ok, msg = await leave_room(ws, arg)
        if not ok:
            await ws_send(ws, {"type": "sys", "t": ts(), "text": msg})
        return True

    if low == "/who":
        await cmd_who(ws, active_room)
        return True

    if low.startswith("/topic"):
        parts = t.split(" ", 1)
        arg = parts[1] if len(parts) == 2 else None
        await cmd_topic(ws, active_room, arg)
        return True

    if low.startswith("/history"):
        parts = t.split(" ", 1)
        n = 120
        if len(parts) == 2:
            try:
                n = int(parts[1].strip())
            except Exception:
                n = 120
        await cmd_history(ws, active_room, n)
        return True

    if low.startswith("/dmhistory "):
        parts = t.split(" ")
        # /dmhistory Nick [N]
        if len(parts) < 2:
            await ws_send(ws, {"type": "sys", "t": ts(), "text": "Naudojimas: /dmhistory VARDAS [N]"})
            return True
        who = parts[1]
        n = 120
        if len(parts) >= 3:
            try:
                n = int(parts[2])
            except Exception:
                n = 120
        await cmd_dmhistory(ws, who, n)
        return True

    if low.startswith("/dm "):
        parts = t.split(" ", 2)
        if len(parts) < 3:
            await ws_send(ws, {"type": "sys", "t": ts(), "text": "Naudojimas: /dm VARDAS ŽINUTĖ"})
            return True
        await cmd_dm(ws, parts[1], parts[2])
        return True

    if low.startswith("/me "):
        await cmd_me(ws, active_room, t.split(" ", 1)[1])
        return True

    m = ROLL_RE.match(t)
    if m:
        n = int(m.group(1)) if m.group(1) else 1
        sides = int(m.group(2)) if m.group(2) else 6
        await cmd_roll(ws, active_room, n, sides)
        return True

    if low == "/flip":
        await cmd_flip(ws, active_room)
        return True

    if low == "/time":
        await ws_send(ws, {"type": "sys", "t": ts(), "text": f"Serverio laikas (Vilnius): {ts()}"})
        return True

    if low.startswith("/pin "):
        try:
            msg_id = int(t.split(" ", 1)[1].strip())
        except Exception:
            await ws_send(ws, {"type": "sys", "t": ts(), "text": "Naudojimas: /pin ID"})
            return True
        await cmd_pin(ws, active_room, msg_id)
        return True

    if low == "/pins":
        await cmd_pins(ws, active_room)
        return True

    if low.startswith("/quote "):
        try:
            msg_id = int(t.split(" ", 1)[1].strip())
        except Exception:
            await ws_send(ws, {"type": "sys", "t": ts(), "text": "Naudojimas: /quote ID"})
            return True
        await cmd_quote(ws, active_room, msg_id)
        return True

    if low.startswith("/edit "):
        parts = t.split(" ", 2)
        if len(parts) < 3:
            await ws_send(ws, {"type": "sys", "t": ts(), "text": "Naudojimas: /edit ID NAUJAS_TEKSTAS"})
            return True
        try:
            msg_id = int(parts[1])
        except Exception:
            await ws_send(ws, {"type": "sys", "t": ts(), "text": "Naudojimas: /edit ID NAUJAS_TEKSTAS"})
            return True
        await cmd_edit(ws, active_room, msg_id, parts[2])
        return True

    if low.startswith("/del "):
        try:
            msg_id = int(t.split(" ", 1)[1].strip())
        except Exception:
            await ws_send(ws, {"type": "sys", "t": ts(), "text": "Naudojimas: /del ID"})
            return True
        await cmd_delete(ws, active_room, msg_id)
        return True

    if low.startswith("/react "):
        parts = t.split(" ", 2)
        if len(parts) < 3:
            await ws_send(ws, {"type": "sys", "t": ts(), "text": "Naudojimas: /react ID 😀"})
            return True
        try:
            msg_id = int(parts[1])
        except Exception:
            await ws_send(ws, {"type": "sys", "t": ts(), "text": "Naudojimas: /react ID 😀"})
            return True
        await cmd_react(ws, active_room, msg_id, parts[2])
        return True

    if low == "/game start":
        await cmd_game_start(ws, active_room)
        return True

    if low.startswith("/guess "):
        parts = t.split(" ", 1)
        try:
            guess = int(parts[1].strip())
        except Exception:
            await ws_send(ws, {"type": "sys", "t": ts(), "text": "Naudojimas: /guess 42"})
            return True
        await cmd_guess(ws, active_room, guess)
        return True

    return False


# =====================================
# HTTP ROUTES
# =====================================
@app.get("/", response_class=HTMLResponse)
def home():
    return HTML


@app.get("/health", response_class=PlainTextResponse)
def health():
    return f"OK {ts()}"


@app.get("/check_nick")
async def check_nick(nick: str = ""):
    n = (nick or "").strip()
    if not valid_nick(n):
        return JSONResponse({"ok": False, "reason": "Netinkamas nick (2–24, raidės/skaičiai/tarpas/_-.)."})
    async with state_lock:
        if n.casefold() in all_ws_by_nick_cf:
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
async def suggest_nick(base: str = ""):
    b = (base or "").strip()
    if not b:
        return JSONResponse({"ok": True, "suggestion": f"User{random.randint(100,999)}"})
    if not valid_nick(b):
        b = "User"
    stem = re.sub(r"\s*\d+$", "", b).rstrip()
    if len(stem) < 2:
        stem = b[:24]

    async with state_lock:
        if b.casefold() not in all_ws_by_nick_cf:
            return JSONResponse({"ok": True, "suggestion": b})
        for i in range(2, 10000):
            cand = _fit_candidate(stem, str(i))
            if valid_nick(cand) and cand.casefold() not in all_ws_by_nick_cf:
                return JSONResponse({"ok": True, "suggestion": cand})

    return JSONResponse({"ok": True, "suggestion": _fit_candidate(stem, str(random.randint(100,999)))})


# =====================================
# CLIENT HTML (single page app)
# =====================================
HTML = r"""<!doctype html>
<html lang="lt">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>HestioRooms Chat</title>
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
    .brand{ display:flex; gap:10px; align-items:baseline; min-width: 220px; }
    .brand .title{ font-weight:900; color:var(--accent); }
    .brand .topic{ font-weight:800; color:var(--text); }

    .status{
      color:var(--muted); font-size:13px;
      display:flex; align-items:center; gap:10px;
      justify-content:center; flex:1;
      text-align:center;
      overflow:hidden;
    }
    .pill{
      border:1px solid var(--border);
      background:rgba(0,0,0,.18);
      padding:6px 10px;
      border-radius:999px;
      display:inline-flex; align-items:center; gap:8px;
      white-space:nowrap;
      max-width: 100%;
    }
    .pill span.ellip{
      overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
      max-width: 520px;
      display:inline-block;
      vertical-align:bottom;
    }
    .dot{ width:10px; height:10px; border-radius:999px; background:var(--danger);
      box-shadow:0 0 14px rgba(255,107,107,.18); }
    .dot.ok{ background:var(--accent); box-shadow:0 0 14px rgba(124,255,107,.18); }

    .main{
      display:grid;
      grid-template-columns: 260px 1fr 320px;
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

    /* LEFT: channels */
    .sidehead{
      padding:12px 12px;
      border-bottom:1px solid var(--border);
      display:flex; justify-content:space-between; align-items:center;
      color:var(--muted); font-size:13px;
    }
    .sidehead b{ color:var(--text); }
    .sideSub{
      padding:10px 12px;
      border-bottom:1px solid var(--border);
      color:var(--muted);
      font-size:12px;
    }
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
    .list{
      padding:10px 10px 12px 10px;
      overflow:auto;
      min-height:0;
    }
    .item{
      display:flex; align-items:center; justify-content:space-between; gap:10px;
      padding:9px 10px;
      border-radius:14px;
      border:1px solid rgba(255,255,255,0.03);
      background:rgba(0,0,0,0.12);
      margin-bottom:8px;
      cursor:pointer;
      user-select:none;
    }
    .item:hover{ filter:brightness(1.08); }
    .item.active{
      border-color: rgba(124,255,107,.28);
      background: rgba(124,255,107,.06);
    }
    .iname{ font-weight:900; font-size:13px; }
    .idesc{ color:var(--muted); font-size:11px; margin-top:2px; }
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

    /* CENTER: chat */
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
    .idTag{
      color: rgba(202,255,217,.38);
      font-size: 12px;
      margin-right: 8px;
    }
    .meta{
      color: rgba(202,255,217,.32);
      font-size: 12px;
      margin-left: 8px;
    }
    .reactions{
      display:inline-flex;
      gap:6px;
      margin-left:10px;
      flex-wrap:wrap;
    }
    .react{
      border:1px solid rgba(255,255,255,.06);
      background:rgba(0,0,0,.18);
      padding:2px 8px;
      border-radius:999px;
      font-size:12px;
      color:var(--text);
      cursor:pointer;
      user-select:none;
    }

    /* RIGHT: DMs + online */
    .rightGrid{
      display:grid;
      grid-template-rows:auto 1fr auto 1fr;
      min-height:0;
    }

    /* bottom */
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
      width:min(920px, 96vw);
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
      grid-template-columns: 340px 1fr;
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

    .typingLine{
      color: var(--muted);
      font-size: 12px;
      margin-left: 10px;
    }

    @media (max-width: 1000px){
      .main{ grid-template-columns: 1fr; }
      .status{ display:none; }
      .brand{ min-width: unset; }
      .card-body{ grid-template-columns: 1fr; }
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
          <div class="h1">HestioRooms Chat</div>
          <div class="sub">Įvesk slapyvardį. Kol nick neužimtas – prisijungti negalima.</div>
        </div>
        <div class="sub">for fun</div>
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

          <div class="label" style="margin-top:12px;">Tema</div>
          <select id="themePick" class="themeSel wide">
            <option value="theme-cyber">Cyber</option>
            <option value="theme-glass">Glass</option>
            <option value="theme-matrix">Matrix</option>
            <option value="theme-crt">CRT</option>
          </select>

          <div class="sub" style="margin-top:12px; color:var(--muted); font-size:12px; line-height:1.35;">
            Patarimas: /help parodo visas komandas. /join #games – žaidimų kanalas.
          </div>
        </div>

        <div class="box">
          <div class="label">Start</div>
          <button id="joinBtn" class="joinbtn" disabled>Prisijungti</button>

          <div class="sub" style="margin-top:12px; color:var(--muted); font-size:12px; line-height:1.35;">
            Nick ir tema išsaugomi naršyklėje.
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- App -->
  <div class="app" id="app" style="display:none;">
    <div class="topbar">
      <div class="brand">
        <div class="title">HestioRooms</div>
        <div class="topic" id="topic">#main</div>
      </div>

      <div class="status">
        <span class="pill">
          <span id="connDot" class="dot"></span>
          <span id="connText">Disconnected</span>
        </span>
        <span class="pill"><span class="ellip" id="typingText"></span></span>
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
      <!-- Left: Channels -->
      <div class="panel" style="display:grid; grid-template-rows:auto auto 1fr; min-height:0;">
        <div class="sidehead"><span>Kanalai</span><span>rooms</span></div>
        <div class="search">
          <input id="roomJoin" placeholder="įrašyk #room ir Enter (pvz #games)" />
        </div>
        <div class="list" id="rooms"></div>
      </div>

      <!-- Center: Chat -->
      <div class="panel">
        <div id="log"></div>
      </div>

      <!-- Right: DM + Online -->
      <div class="panel rightGrid">
        <div class="sidehead"><span>DM</span><span>tabs</span></div>
        <div class="list" id="dms"></div>

        <div class="sidehead"><span>Online: <b id="onlineCount">0</b></span><span>live</span></div>
        <div class="list" id="users"></div>
      </div>
    </div>

    <div class="bottombar">
      <input id="msg" placeholder="rašyk žinutę ir Enter..." maxlength="300"/>
      <button id="btn">Siųsti</button>
    </div>
  </div>

<script>
  // Elements
  const lobby = document.getElementById("lobby");
  const appEl = document.getElementById("app");

  const nickPick = document.getElementById("nickPick");
  const nickErr  = document.getElementById("nickErr");
  const nickState = document.getElementById("nickState");
  const joinBtn = document.getElementById("joinBtn");

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

  const connDot = document.getElementById("connDot");
  const connText = document.getElementById("connText");
  const typingText = document.getElementById("typingText");

  const roomsEl = document.getElementById("rooms");
  const dmsEl = document.getElementById("dms");
  const usersEl = document.getElementById("users");
  const onlineCountEl = document.getElementById("onlineCount");
  const roomJoinEl = document.getElementById("roomJoin");

  // WS
  let ws = null;
  let reconnectTimer = null;
  let joinEstablished = false;
  let fatalJoinError = false;

  // Identity
  let nick = "";
  let myNick = "";
  let myColor = "#caffd9";

  // Convos: key => {kind:'room'|'dm', room?, peer?, title, unread, items[], loaded}
  const convs = new Map();
  let activeKey = "room:main";

  // quick DOM indexing for edits/reactions
  const msgDom = new Map(); // msgId -> element

  // Typing
  let typing = false;
  let typingTimer = null;
  let lastTypingSend = 0;

  // Utility
  function esc(s){
    return (s ?? "").toString()
      .replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;")
      .replaceAll('"',"&quot;").replaceAll("'","&#39;");
  }

  function setConn(ok){
    if(ok){ connDot.classList.add("ok"); connText.textContent = "Connected"; }
    else { connDot.classList.remove("ok"); connText.textContent = "Disconnected"; }
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

  function validateNick(n){
    return /^[A-Za-z0-9ĄČĘĖĮŠŲŪŽąčęėįšųūž_\-\. ]{2,24}$/.test(n);
  }

  // THEME
  function setTheme(cls){
    document.body.className = cls;
    localStorage.setItem("theme", cls);
    if(themePick) themePick.value = cls;
    if(themeTop) themeTop.value = cls;
  }
  themePick.addEventListener("change", () => setTheme(themePick.value));
  themeTop.addEventListener("change", () => setTheme(themeTop.value));

  function showLobby(show){
    lobby.style.display = show ? "flex" : "none";
    appEl.style.display = show ? "none" : "grid";
    if(show) setTimeout(() => nickPick.focus(), 40);
    else setTimeout(() => msgEl.focus(), 40);
  }

  function stopReconnect(){
    if(reconnectTimer){
      clearInterval(reconnectTimer);
      reconnectTimer = null;
    }
  }

  function wsUrl(){
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const qs = new URLSearchParams({ nick }).toString();
    return `${proto}://${location.host}/ws?${qs}`;
  }

  function tsLocal(){
    const d = new Date();
    const hh = String(d.getHours()).padStart(2,'0');
    const mm = String(d.getMinutes()).padStart(2,'0');
    const ss = String(d.getSeconds()).padStart(2,'0');
    return `${hh}:${mm}:${ss}`;
  }

  // Conversations
  function ensureRoomConvo(roomKey, title, topic){
    const key = `room:${roomKey}`;
    if(!convs.has(key)){
      convs.set(key, {key, kind:"room", room:roomKey, title: title || `#${roomKey}`, topic: topic||"", unread:0, items:[], loaded:true});
    }else{
      const c = convs.get(key);
      c.title = title || c.title;
      c.topic = topic || c.topic;
    }
    return key;
  }

  function ensureDmConvo(peer){
    const key = `dm:${peer}`;
    if(!convs.has(key)){
      convs.set(key, {key, kind:"dm", peer, title:`DM: ${peer}`, unread:0, items:[], loaded:false});
    }
    return key;
  }

  function setActive(key){
    if(!convs.has(key)) return;
    activeKey = key;
    const c = convs.get(key);
    if(c) c.unread = 0;
    renderSidebars();
    renderActiveLog();
    setHeader();
  }

  function setHeader(){
    const c = convs.get(activeKey);
    if(!c) return;
    if(c.kind === "room"){
      topicEl.textContent = `#${c.room}`;
      msgEl.placeholder = `rašyk į #${c.room} ir Enter... (komandos: /help)`;
    }else{
      topicEl.textContent = `DM su ${c.peer}`;
      msgEl.placeholder = `rašyk DM ${c.peer} ir Enter...`;
    }
  }

  function renderSidebars(){
    // Rooms
    const rooms = Array.from(convs.values()).filter(x => x.kind === "room");
    rooms.sort((a,b) => a.title.localeCompare(b.title, "lt"));
    roomsEl.innerHTML = "";
    for(const c of rooms){
      const row = document.createElement("div");
      row.className = "item" + (c.key === activeKey ? " active" : "");
      const unread = c.unread || 0;
      row.innerHTML = `
        <div>
          <div class="iname">${esc(c.title)}</div>
          <div class="idesc">${esc(c.topic || "")}</div>
        </div>
        <div class="badge" style="${unread>0 ? 'display:inline-flex;' : ''}">${unread}</div>
      `;
      row.addEventListener("click", () => setActive(c.key));
      roomsEl.appendChild(row);
    }

    // DMs
    const dms = Array.from(convs.values()).filter(x => x.kind === "dm");
    dms.sort((a,b) => a.title.localeCompare(b.title, "lt"));
    dmsEl.innerHTML = "";
    for(const c of dms){
      const row = document.createElement("div");
      row.className = "item" + (c.key === activeKey ? " active" : "");
      const unread = c.unread || 0;
      row.innerHTML = `
        <div>
          <div class="iname">${esc(c.title)}</div>
          <div class="idesc">privatus pokalbis</div>
        </div>
        <div class="badge" style="${unread>0 ? 'display:inline-flex;' : ''}">${unread}</div>
      `;
      row.addEventListener("click", () => {
        setActive(c.key);
        // lazy load dm history
        if(!c.loaded){
          c.loaded = true;
          wsSend({type:"dm_history_req", with:c.peer, limit:120});
        }
      });
      dmsEl.appendChild(row);
    }
  }

  function clearLog(){
    log.innerHTML = "";
    msgDom.clear();
  }

  function addLineElement(el){
    log.appendChild(el);
    log.scrollTop = log.scrollHeight;
  }

  function renderReactions(reactions){
    if(!reactions) return "";
    const keys = Object.keys(reactions);
    if(keys.length === 0) return "";
    let html = `<span class="reactions">`;
    for(const k of keys){
      const n = (reactions[k] || []).length;
      html += `<span class="react" data-emoji="${esc(k)}">${esc(k)} ${n}</span>`;
    }
    html += `</span>`;
    return html;
  }

  function renderMessageObj(o){
    const el = document.createElement("div");
    el.className = "line";
    const t = esc(o.t || "");
    const id = o.id != null ? Number(o.id) : null;

    // message header id
    const idHtml = (id != null) ? `<span class="idTag">#${id}</span>` : `<span class="idTag"></span>`;

    if(o.type === "msg"){
      const nn = esc(o.nick || "???");
      const cc = esc(o.color || "#caffd9");
      const tx = esc(o.text || "");
      const extra = o.extra || {};
      const edited = extra.edited ? `<span class="meta">(edited)</span>` : "";
      const reacts = renderReactions(extra.reactions || {});
      el.innerHTML = `${idHtml}<span class="t">[${t}]</span> <span class="nick" style="color:${cc}">${nn}</span>: <span class="msg">${tx}</span>${edited}${reacts}`;
    } else if(o.type === "me_action"){
      const nn = esc(o.nick || "???");
      const cc = esc(o.color || "#caffd9");
      const tx = esc(o.text || "");
      const extra = o.extra || {};
      const reacts = renderReactions(extra.reactions || {});
      el.innerHTML = `${idHtml}<span class="t">[${t}]</span> <span class="me" style="color:${cc}">* ${nn} ${tx}</span>${reacts}`;
    } else if(o.type === "dm"){
      const f = esc(o.from || "???");
      const to = esc(o.to || "???");
      const cc = esc(o.color || "#caffd9");
      const tx = esc(o.text || "");
      el.innerHTML = `${idHtml}<span class="t">[${t}]</span> <span class="dmTag">[DM]</span> <span class="nick" style="color:${cc}">${f}</span> → <span class="nick">${to}</span>: <span class="msg">${tx}</span>`;
    } else if(o.type === "deleted"){
      el.innerHTML = `${idHtml}<span class="t">[${t}]</span> <span class="sys">[deleted]</span>`;
    } else {
      const tx = esc(o.text || "");
      el.innerHTML = `${idHtml}<span class="t">[${t}]</span> <span class="sys">${tx}</span>`;
    }

    if(id != null){
      el.dataset.id = String(id);
      msgDom.set(id, el);

      // click reaction shortcut: Alt+Click message -> prompt emoji
      el.addEventListener("click", (ev) => {
        if(!ev.altKey) return;
        const emoji = prompt("Reakcija (pvz 😀):");
        if(!emoji) return;
        // only for room messages
        const c = convs.get(activeKey);
        if(c && c.kind === "room"){
          wsSend({ text: `/react ${id} ${emoji}`, room: c.room });
        }
      });
    }

    return el;
  }

  function pushToConvo(key, obj, noUnread=false){
    if(!convs.has(key)) return;
    const c = convs.get(key);

    c.items.push(obj);
    if(c.items.length > 600){
      c.items.splice(0, c.items.length - 600);
    }

    if(!noUnread && key !== activeKey){
      c.unread = (c.unread||0) + 1;
    }

    if(key === activeKey){
      const el = renderMessageObj(obj);
      addLineElement(el);
    }
  }

  function renderActiveLog(){
    clearLog();
    const c = convs.get(activeKey);
    if(!c) return;
    for(const obj of c.items){
      addLineElement(renderMessageObj(obj));
    }
  }

  function routeIncoming(o){
    // topic/rooms/users are special
    if(o.type === "rooms"){
      // ensure room convos exist
      for(const it of (o.items || [])){
        ensureRoomConvo(it.room, it.title, it.topic);
      }
      renderSidebars();
      // keep active if missing
      if(!convs.has(activeKey)){
        setActive("room:main");
      }
      return;
    }

    if(o.type === "topic"){
      const room = o.room || "main";
      const key = ensureRoomConvo(room, `#${room}`, (o.text || "").split("—").slice(1).join("—").trim());
      // store full for display
      const c = convs.get(key);
      if(c){
        // try parse "title — topic"
        c.topic = (o.text || "").split("—").slice(1).join("—").trim() || c.topic;
      }
      if(activeKey === key){
        setHeader();
      }
      renderSidebars();
      return;
    }

    if(o.type === "users"){
      const items = o.items || [];
      onlineCountEl.textContent = String(items.length);
      usersEl.innerHTML = "";
      for(const u of items){
        const row = document.createElement("div");
        row.className = "item";
        const st = (u.status === "away") ? "away" : "online";
        row.innerHTML = `
          <div>
            <div class="iname" style="color:${esc(u.color||'#caffd9')}">${esc(u.nick||'???')} ${st==='away' ? '(away)' : ''}</div>
            <div class="idesc">spausk DM</div>
          </div>
          <div class="badge" style="display:none;"></div>
        `;
        row.addEventListener("click", () => {
          if(myNick && (u.nick||"").toLowerCase() === myNick.toLowerCase()) return;
          const key = ensureDmConvo(u.nick);
          renderSidebars();
          setActive(key);
          const c = convs.get(key);
          if(c && !c.loaded){
            c.loaded = true;
            wsSend({type:"dm_history_req", with:c.peer, limit:120});
          }
        });
        usersEl.appendChild(row);
      }
      return;
    }

    if(o.type === "typing"){
      const items = o.items || [];
      if(items.length === 0){
        typingText.textContent = "";
      } else if(items.length === 1){
        typingText.textContent = `${items[0]} rašo...`;
      } else {
        typingText.textContent = `${items.slice(0,2).join(", ")} ir dar ${items.length-2} rašo...`;
      }
      return;
    }

    if(o.type === "pins"){
      const room = o.room || "main";
      const ids = o.items || [];
      const txt = ids.length ? `Pin’ai: ${ids.map(x => "#"+x).join(", ")}` : "Pin’ų nėra.";
      pushToConvo(`room:${room}`, {type:"sys", t: tsLocal(), text: txt}, true);
      return;
    }

    if(o.type === "quote"){
      const room = o.room || "main";
      const id = o.id;
      const nick = o.nick || "sys";
      const text = o.text || "";
      // insert quote into input
      msgEl.value = `> #${id} ${nick}: ${text}\n`;
      msgEl.focus();
      return;
    }

    // edits
    if(o.type === "edit"){
      const room = o.room || "main";
      const id = Number(o.id);
      const el = msgDom.get(id);
      if(el){
        // Update in conv cache too
        const key = `room:${room}`;
        const c = convs.get(key);
        if(c){
          const it = c.items.find(x => Number(x.id) === id);
          if(it){
            it.text = o.text;
            it.extra = it.extra || {};
            it.extra.edited = true;
          }
        }
        // rerender by replacing innerHTML
        const c2 = convs.get(`room:${room}`);
        if(c2){
          const it2 = c2.items.find(x => Number(x.id) === id);
          if(it2){
            const newEl = renderMessageObj(it2);
            el.replaceWith(newEl);
            msgDom.set(id, newEl);
          }
        }
      }
      return;
    }

    if(o.type === "delete"){
      const room = o.room || "main";
      const id = Number(o.id);
      const key = `room:${room}`;
      const c = convs.get(key);
      if(c){
        const it = c.items.find(x => Number(x.id) === id);
        if(it){
          it.type = "deleted";
          it.text = "";
          it.extra = {deleted:true};
        }
      }
      const el = msgDom.get(id);
      if(el){
        const it2 = c ? c.items.find(x => Number(x.id) === id) : null;
        if(it2){
          const newEl = renderMessageObj(it2);
          el.replaceWith(newEl);
          msgDom.set(id, newEl);
        }
      }
      return;
    }

    if(o.type === "react_update"){
      const room = o.room || "main";
      const id = Number(o.id);
      const key = `room:${room}`;
      const c = convs.get(key);
      if(c){
        const it = c.items.find(x => Number(x.id) === id);
        if(it){
          it.extra = it.extra || {};
          it.extra.reactions = o.reactions || {};
        }
      }
      const el = msgDom.get(id);
      if(el){
        const it2 = c ? c.items.find(x => Number(x.id) === id) : null;
        if(it2){
          const newEl = renderMessageObj(it2);
          el.replaceWith(newEl);
          msgDom.set(id, newEl);
        }
      }
      return;
    }

    // history
    if(o.type === "history"){
      const room = o.room || "main";
      const key = ensureRoomConvo(room, `#${room}`, "");
      const c = convs.get(key);
      c.items = [];
      for(const it of (o.items || [])){
        // normalize
        const item = {
          id: it.id,
          type: it.type,
          t: it.t,
          nick: it.nick,
          color: it.color,
          text: it.text,
          extra: it.extra || {}
        };
        c.items.push(item);
      }
      if(activeKey === key){
        renderActiveLog();
      }
      renderSidebars();
      return;
    }

    if(o.type === "dm_history"){
      const peer = o.with || "";
      const key = ensureDmConvo(peer);
      const c = convs.get(key);
      c.items = [];
      for(const it of (o.items || [])){
        c.items.push({
          id: it.id,
          type: "dm",
          t: it.t,
          from: it.from,
          to: it.to,
          color: it.color,
          text: it.text,
          extra: it.extra || {}
        });
      }
      if(activeKey === key){
        renderActiveLog();
      }
      renderSidebars();
      return;
    }

    // DM
    if(o.type === "dm"){
      const peer = (myNick && (o.from||"").toLowerCase() === myNick.toLowerCase()) ? (o.to||"") : (o.from||"");
      const key = ensureDmConvo(peer);
      renderSidebars();
      pushToConvo(key, {
        id:o.id,
        type:"dm",
        t:o.t,
        from:o.from,
        to:o.to,
        color:o.color,
        text:o.text,
        extra:o.extra || {}
      }, false);
      return;
    }

    // Room messages
    const room = o.room || "main";
    const key = ensureRoomConvo(room, `#${room}`, "");
    const obj = {
      id: o.id,
      type: o.type,
      t: o.t || tsLocal(),
      nick: o.nick,
      color: o.color,
      text: o.text,
      extra: o.extra || {}
    };
    pushToConvo(key, obj, false);
    renderSidebars();
  }

  function wsSend(obj){
    if(!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify(obj));
  }

  function connect(){
    joinEstablished = false;
    fatalJoinError = false;

    setConn(false);

    // init default convs
    convs.clear();
    ensureRoomConvo("main", "#main", "Bendras kanalas");
    activeKey = "room:main";
    renderSidebars();
    setHeader();
    renderActiveLog();

    pushToConvo("room:main", {type:"sys", t: tsLocal(), text:"jungiamasi..."}, true);

    ws = new WebSocket(wsUrl());

    ws.onopen = () => {
      setConn(true);
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
        joinBtn.disabled = true;
        scheduleNickCheck();
        return;
      }

      if(o.type === "me"){
        joinEstablished = true;
        myNick = o.nick || "";
        myColor = o.color || "#caffd9";
        meNickPill.innerHTML = `<b style="color:${esc(myColor)}">${esc(myNick)}</b>`;
        // ensure main active
        setActive("room:main");
        return;
      }

      // first server packets mark established
      if(["rooms","topic","history","users"].includes(o.type)) joinEstablished = true;

      routeIncoming(o);
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

      pushToConvo("room:main", {type:"sys", t: tsLocal(), text:"ryšys nutrūko, reconnect..."}, false);
      if(!reconnectTimer) reconnectTimer = setInterval(connect, 1500);
    };
  }

  // Typing
  function setTyping(on){
    const now = Date.now();
    if(on){
      if(now - lastTypingSend < 700) return;
      lastTypingSend = now;
      const c = convs.get(activeKey);
      if(c && c.kind === "room"){
        wsSend({type:"typing", room:c.room, on:true});
      }
    }else{
      const c = convs.get(activeKey);
      if(c && c.kind === "room"){
        wsSend({type:"typing", room:c.room, on:false});
      }
    }
  }

  msgEl.addEventListener("input", () => {
    if(!typing){
      typing = true;
      setTyping(true);
    }
    if(typingTimer) clearTimeout(typingTimer);
    typingTimer = setTimeout(() => {
      typing = false;
      setTyping(false);
    }, 1200);
  });

  // Sending
  function send(){
    const text = (msgEl.value || "").trim();
    if(!text) return;

    const c = convs.get(activeKey);
    if(!c) return;

    if(!ws || ws.readyState !== WebSocket.OPEN){
      pushToConvo(activeKey, {type:"sys", t: tsLocal(), text:"nėra ryšio su serveriu."}, false);
      return;
    }

    // stop typing broadcast
    typing = false;
    setTyping(false);

    // commands: always send with active room context
    if(text.startsWith("/")){
      if(c.kind === "room"){
        wsSend({type:"say", room:c.room, text});
      }else{
        // command in DM tab executes in #main context (reasonable default)
        wsSend({type:"say", room:"main", text});
      }
      msgEl.value = "";
      return;
    }

    // normal message
    if(c.kind === "room"){
      wsSend({type:"say", room:c.room, text});
    }else{
      // DM tab: convert to /dm
      wsSend({type:"say", room:"main", text:`/dm ${c.peer} ${text}`});
    }

    msgEl.value = "";
  }

  btn.onclick = send;
  msgEl.addEventListener("keydown", (e) => { if(e.key === "Enter") send(); });

  // join room input
  roomJoinEl.addEventListener("keydown", (e) => {
    if(e.key !== "Enter") return;
    const val = (roomJoinEl.value || "").trim();
    if(!val) return;
    wsSend({type:"say", room:"main", text:`/join ${val}`});
    roomJoinEl.value = "";
  });

  // Nick availability
  let checkTimer = null;
  let nickAvailable = false;

  async function suggestNick(base){
    try{
      const qs = new URLSearchParams({ base }).toString();
      const r = await fetch(`/suggest_nick?${qs}`, { cache: "no-store" });
      const j = await r.json();
      if(j && j.ok && j.suggestion) return String(j.suggestion);
    }catch{}
    return "";
  }

  async function checkNickAvailabilityNow(){
    hideSuggest();
    const n = (nickPick.value || "").trim();
    nickAvailable = false;
    joinBtn.disabled = true;

    if(!validateNick(n)){
      setNickState("");
      setNickError("Netinkamas nick. Reikia 2–24 simbolių (raidės/skaičiai/tarpas/_-.)");
      return;
    }

    setNickError("");
    setNickState("Tikrinama ar nick laisvas...");

    try{
      const qs = new URLSearchParams({ nick: n }).toString();
      const r = await fetch(`/check_nick?${qs}`, { cache: "no-store" });
      const j = await r.json();

      if(j && j.ok){
        nickAvailable = true;
        setNickState("Nick laisvas. Galite prisijungti.");
        setNickError("");
        joinBtn.disabled = false;
      }else{
        nickAvailable = false;
        setNickState("");
        setNickError((j && j.reason) ? j.reason : "Nick užimtas arba neteisingas.");
        joinBtn.disabled = true;

        const sug = await suggestNick(n);
        if(sug && sug.toLowerCase() !== n.toLowerCase()){
          showSuggest(sug);
        }
      }
    }catch{
      nickAvailable = false;
      setNickState("");
      setNickError("Nepavyko patikrinti nick (serveris nepasiekiamas).");
      joinBtn.disabled = true;
    }
  }

  function scheduleNickCheck(){
    if(checkTimer) clearTimeout(checkTimer);
    checkTimer = setTimeout(checkNickAvailabilityNow, 250);
  }

  nickPick.addEventListener("input", scheduleNickCheck);
  nickPick.addEventListener("keydown", (e) => {
    if(e.key === "Enter" && !joinBtn.disabled) joinBtn.click();
  });

  applySuggest.addEventListener("click", () => {
    const s = (nickSuggestVal.textContent || "").trim();
    if(!s) return;
    nickPick.value = s;
    hideSuggest();
    scheduleNickCheck();
    nickPick.focus();
  });

  joinBtn.onclick = async () => {
    await checkNickAvailabilityNow();
    if(!nickAvailable) return;

    nick = (nickPick.value || "").trim();
    localStorage.setItem("nick", nick);

    showLobby(false);
    connect();
  };

  // Init
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

# =====================================
# WEBSOCKET ENDPOINT
# =====================================
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    nick = (ws.query_params.get("nick") or "").strip()

    await ws.accept()

    if not valid_nick(nick):
        await ws.send_json({"type": "error", "code": "BAD_NICK", "text": "Netinkamas nick. Grįžk ir įvesk teisingą."})
        await ws.close()
        return

    async with state_lock:
        if nick.casefold() in all_ws_by_nick_cf:
            await ws.send_json({"type": "error", "code": "NICK_TAKEN", "text": "Nick užimtas. Pasirink kitą."})
            await ws.close()
            return

        # allocate color across all online users (global distinctness)
        used = {u.color for u in all_users_by_ws.values()}
        color = alloc_color(used)

        u = User(nick=nick, color=color)
        all_users_by_ws[ws] = u
        all_ws_by_nick_cf[nick.casefold()] = ws

    # ensure default rooms exist
    for rk in DEFAULT_ROOMS.keys():
        ensure_room(rk)

    # auto join #main
    await join_room(ws, "#main")

    # send full rooms list
    await broadcast_rooms_list(ws)

    # initial topic + users list for #main
    await ws_send(ws, {"type": "topic", "room": "main", "text": f"{ensure_room('main').title} — {ensure_room('main').topic}"})
    await ws_send(ws, {"type": "users", "room": "main", "items": room_userlist("main")})

    # identify
    await ws_send(ws, {"type": "me", "nick": u.nick, "color": u.color})

    try:
        while True:
            data = await ws.receive_json()
            if not isinstance(data, dict):
                continue

            # typing event
            if data.get("type") == "typing":
                room_key = norm_room(data.get("room", "main"))
                on = bool(data.get("on", False))
                async with state_lock:
                    r = ensure_room(room_key)
                    if on:
                        r.typing.add(u.nick)
                    else:
                        r.typing.discard(u.nick)
                await broadcast_typing(room_key)
                continue

            # dm history request
            if data.get("type") == "dm_history_req":
                peer = str(data.get("with", "")).strip()
                try:
                    limit = int(data.get("limit", 120))
                except Exception:
                    limit = 120
                items = await db_load_dm_history(u.nick, peer, limit)
                await ws_send(ws, {"type": "dm_history", "with": peer, "items": items})
                continue

            # main say payload
            if data.get("type") == "say":
                room_key = norm_room(data.get("room", "main"))
                text = str(data.get("text", "")).strip()
                if not text:
                    continue

                # must be member to talk
                async with state_lock:
                    if room_key not in u.rooms:
                        # auto-join if room exists and user tries to speak
                        # (optional), but keep it friendly:
                        pass

                # rate limiting
                ok, reason = check_rate_limit(u, len(text))
                if not ok:
                    await ws_send(ws, {"type": "sys", "t": ts(), "text": reason})
                    continue

                # commands
                if text.startswith("/"):
                    handled = await handle_command(ws, room_key, text)
                    if handled:
                        continue

                # normal message
                extra = {
                    "created_at": time.time(),
                    "edited": False,
                    "reactions": {},
                }
                msg = {
                    "scope": "room",
                    "room": room_key,
                    "msg_type": "msg",
                    "ts": ts(),
                    "nick": u.nick,
                    "color": u.color,
                    "text": text[:300],
                    "extra": extra,
                }
                msg_id = await db_insert_message(msg)
                payload = {
                    "type": "msg",
                    "room": room_key,
                    "id": msg_id,
                    "t": msg["ts"],
                    "nick": u.nick,
                    "color": u.color,
                    "text": msg["text"],
                    "extra": extra,
                }
                await room_broadcast(room_key, payload)
                continue

    except WebSocketDisconnect:
        pass
    finally:
        await disconnect_ws(ws)
