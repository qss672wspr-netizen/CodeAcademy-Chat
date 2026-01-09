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

EDIT_WINDOW_SEC = 5 * 60  # 5 min

MAX_MSG_PER_10S = 20
MAX_CHARS_PER_10S = 2500

NICK_RE = re.compile(r"^[A-Za-z0-9ĄČĘĖĮŠŲŪŽąčęėįšųūž_\-\. ]{2,24}$")
ROOM_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]{0,23}$")

ROLL_RE = re.compile(r"^/roll(?:\s+(\d{1,2})d(\d{1,3}))?$", re.IGNORECASE)

# Ryškiai skirtingos spalvos (didesnis kontrastas, mažiau panašių atspalvių)
COLOR_PALETTE = [
    "#e6194b",  # red
    "#3cb44b",  # green
    "#ffe119",  # yellow
    "#4363d8",  # blue
    "#f58231",  # orange
    "#911eb4",  # purple
    "#46f0f0",  # cyan
    "#f032e6",  # magenta
    "#bcf60c",  # lime
    "#fabebe",  # pink
    "#008080",  # teal
    "#e6beff",  # lavender
    "#9a6324",  # brown
    "#fffac8",  # light yellow
    "#800000",  # maroon
    "#aaffc3",  # mint
    "#808000",  # olive
    "#000075",  # navy
    "#808080",  # gray
    "#000000",  # black
    "#ffd700",  # gold
    "#00bfff",  # deep sky blue
    "#ff1493",  # deep pink
    "#7fffd4",  # aquamarine
]

DEFAULT_ROOMS = {
    "main": {"title": "#main", "topic": "Bendras kanalas"},
    "games": {"title": "#games", "topic": "Žaidimai ir pramogos"},
    "help": {"title": "#help", "topic": "Pagalba / klausimai"},
}

# =====================================
# I18N (LT/EN)
# =====================================
LANGS = {"lt", "en"}

TR = {"lt": {}, "en": {}}

TR["lt"] = {
    "BAD_NICK": "Netinkamas nick. Grįžk ir įvesk teisingą.",
    "NICK_TAKEN": "Nick užimtas. Pasirink kitą.",
    "CHECK_FAIL": "Nepavyko patikrinti nick (serveris nepasiekiamas).",
    "BAD_ROOM": "Neteisingas kanalas.",
    "JOIN_OK": "Prisijungėte prie {room}.",
    "JOIN_ALREADY": "Jau esate šiame kanale.",
    "JOIN_BAD_ROOM": "Netinkamas kanalo pavadinimas. Pvz: #main, #games, #help (tik a-z, 0-9, _ -).",
    "LEAVE_NOT_MEMBER": "Jūs nesate šiame kanale.",
    "LEAVE_MAIN_DENY": "Iš #main išeiti negalima (bazinis kanalas).",
    "LEAVE_OK": "Palikote {room}.",
    "DM_USAGE": "Naudojimas: /dm VARDAS ŽINUTĖ",
    "DM_NOT_FOUND": "Vartotojas nerastas online: {nick}",
    "TIME": "Serverio laikas (Vilnius): {t}",
    "RATE_LIMIT": "Per daug žinučių per trumpą laiką. Palaukite kelias sekundes.",
    "RATE_LIMIT_CHARS": "Per daug teksto per trumpą laiką. Palaukite kelias sekundes.",
    "PINS_NONE": "Pin’ų nėra.",
    "PIN_FAIL": "Pin nepavyko: žinutė nerasta šiame kanale.",
    "PIN_OK": "Prisegta žinutė ID={id}.",
    "QUOTE_FAIL": "Quote nepavyko: žinutė nerasta šiame kanale.",
    "EDIT_FAIL_NOTFOUND": "Edit nepavyko: žinutė nerasta šiame kanale.",
    "EDIT_FAIL_OWN": "Edit nepavyko: galima redaguoti tik savo žinutes.",
    "EDIT_FAIL_TIME": "Edit nepavyko: praėjo laiko limitas (5 min).",
    "EDIT_FAIL_EMPTY": "Edit nepavyko: tuščias tekstas.",
    "DEL_FAIL_NOTFOUND": "Del nepavyko: žinutė nerasta šiame kanale.",
    "DEL_FAIL_OWN": "Del nepavyko: galima ištrinti tik savo žinutes.",
    "DEL_FAIL_TIME": "Del nepavyko: praėjo laiko limitas (5 min).",
    "REACT_FAIL": "React nepavyko: žinutė nerasta.",
    "REACT_FAIL_ROOM": "React nepavyko: žinutė ne iš šio kanalo.",
    "REACT_USAGE": "Naudojimas: /react ID 😀",
    "GAME_NOT_STARTED": "Žaidimas nepradėtas. Paleiskite /game start",
    "GUESS_RANGE": "Spėjimas turi būti 1..100.",
    "LANG_SET": "Kalba pakeista į: {lang}",
    "LANG_USAGE": "Naudojimas: /lang lt arba /lang en",
    "ROOMS_LIST": "Kanalai:",
    "ONLINE_ROOM": "Online {room}: {names}",
    "TOPIC_SHOW": "{room} tema: {topic}",
    "TOPIC_CHANGED": "Tema pakeista į: {topic}",
    "GAME_START": "Žaidimas pradėtas: atspėk skaičių 1..100. Komanda: /guess 42",
}

TR["en"] = {
    "BAD_NICK": "Invalid nickname. Go back and enter a valid one.",
    "NICK_TAKEN": "Nickname is taken. Choose another.",
    "CHECK_FAIL": "Could not check nickname (server unreachable).",
    "BAD_ROOM": "Invalid channel.",
    "JOIN_OK": "You joined {room}.",
    "JOIN_ALREADY": "You are already in this channel.",
    "JOIN_BAD_ROOM": "Invalid channel name. Example: #main, #games, #help (only a-z, 0-9, _ -).",
    "LEAVE_NOT_MEMBER": "You are not in this channel.",
    "LEAVE_MAIN_DENY": "You cannot leave #main (base channel).",
    "LEAVE_OK": "You left {room}.",
    "DM_USAGE": "Usage: /dm NAME MESSAGE",
    "DM_NOT_FOUND": "User not found online: {nick}",
    "TIME": "Server time (Vilnius): {t}",
    "RATE_LIMIT": "Too many messages too fast. Wait a few seconds.",
    "RATE_LIMIT_CHARS": "Too much text too fast. Wait a few seconds.",
    "PINS_NONE": "No pinned messages.",
    "PIN_FAIL": "Pin failed: message not found in this channel.",
    "PIN_OK": "Pinned message ID={id}.",
    "QUOTE_FAIL": "Quote failed: message not found in this channel.",
    "EDIT_FAIL_NOTFOUND": "Edit failed: message not found in this channel.",
    "EDIT_FAIL_OWN": "Edit failed: you can only edit your own messages.",
    "EDIT_FAIL_TIME": "Edit failed: time limit exceeded (5 min).",
    "EDIT_FAIL_EMPTY": "Edit failed: empty text.",
    "DEL_FAIL_NOTFOUND": "Delete failed: message not found in this channel.",
    "DEL_FAIL_OWN": "Delete failed: you can only delete your own messages.",
    "DEL_FAIL_TIME": "Delete failed: time limit exceeded (5 min).",
    "REACT_FAIL": "React failed: message not found.",
    "REACT_FAIL_ROOM": "React failed: message is not from this channel.",
    "REACT_USAGE": "Usage: /react ID 😀",
    "GAME_NOT_STARTED": "Game not started. Run /game start",
    "GUESS_RANGE": "Guess must be between 1 and 100.",
    "LANG_SET": "Language set to: {lang}",
    "LANG_USAGE": "Usage: /lang lt or /lang en",
    "ROOMS_LIST": "Channels:",
    "ONLINE_ROOM": "Online {room}: {names}",
    "TOPIC_SHOW": "{room} topic: {topic}",
    "TOPIC_CHANGED": "Topic changed to: {topic}",
    "GAME_START": "Game started: guess a number 1..100. Command: /guess 42",
}


def t(lang: str, key: str, **kwargs) -> str:
    lang = (lang or "lt").lower()
    if lang not in LANGS:
        lang = "lt"
    s = TR[lang].get(key, key)
    try:
        return s.format(**kwargs)
    except Exception:
        return s


HELP_TEXT_LT = (
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
    "  /lang lt|en                    - pakeisti kalbą\n"
)

HELP_TEXT_EN = (
    "Commands:\n"
    "  /help                          - help\n"
    "  /rooms                         - list channels\n"
    "  /join #room                    - join/create a channel\n"
    "  /leave #room                   - leave a channel\n"
    "  /topic                         - show topic (active channel)\n"
    "  /topic TEXT                    - set topic (active channel)\n"
    "  /who                           - who is online (active channel)\n"
    "  /dm NAME MESSAGE               - private message (DM)\n"
    "  /dmhistory NAME [N]            - DM history with a user\n"
    "  /history [N]                   - channel history (default 120)\n"
    "  /me action                     - action message\n"
    "  /roll [NdM]                    - dice (e.g. /roll 2d6)\n"
    "  /flip                          - coin flip\n"
    "  /time                          - server time (Vilnius)\n"
    "  /pin ID                        - pin a message\n"
    "  /pins                          - list pinned messages\n"
    "  /quote ID                      - quote a message\n"
    "  /edit ID NEW_TEXT              - edit own message (within 5 min)\n"
    "  /del ID                        - delete own message (within 5 min)\n"
    "  /react ID 😀                    - react to a message\n"
    "  /game start                    - start Guess 1..100 in channel\n"
    "  /guess NUMBER                  - guess number (active channel)\n"
    "  /lang lt|en                    - set language\n"
)


def help_text(lang: str) -> str:
    return HELP_TEXT_EN if (lang or "lt").lower() == "en" else HELP_TEXT_LT


# =====================================
# DB (SQLite)
# =====================================
db_lock = asyncio.Lock()


def db() -> sqlite3.Connection:
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
          a TEXT, b TEXT,                    -- jei scope='dm'
          msg_type TEXT NOT NULL,            -- 'msg','sys','me_action','deleted','dm'
          ts TEXT NOT NULL,
          nick TEXT,
          color TEXT,
          text TEXT,
          extra TEXT
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

    cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_room ON messages(scope, room, id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_dm ON messages(scope, a, b, id)")

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
    lang: str = "lt"
    status: str = "online"
    joined_at: float = field(default_factory=time.time)
    rooms: Set[str] = field(default_factory=set)
    msg_events: Deque[Tuple[float, int]] = field(default_factory=deque)


@dataclass
class Room:
    key: str
    title: str
    topic: str
    clients: Set[WebSocket] = field(default_factory=set)
    users: Dict[WebSocket, User] = field(default_factory=dict)
    typing: Set[str] = field(default_factory=set)
    game_target: Optional[int] = None
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
    return (room or "").strip().lstrip("#").lower()


def valid_room_key(r: str) -> bool:
    return bool(ROOM_RE.match(r))


def alloc_color(used: Set[str]) -> str:
    for c in COLOR_PALETTE:
        if c not in used:
            return c
    # jei pritrūko paletės – generuojam ryškų HSL
    hue = random.randint(0, 359)
    return f"hsl({hue}, 90%, 45%)"


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
# DB helpers
# =====================================
async def db_insert_message(msg: dict) -> int:
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
        cur.execute("UPDATE messages SET text=?, extra=? WHERE id=?", (new_text, json_dumps(new_extra), msg_id))
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
    out = [{"nick": u.nick, "color": u.color, "status": u.status} for u in r.users.values()]
    out.sort(key=lambda x: x["nick"].casefold())
    return out


async def ws_send(ws: WebSocket, obj: dict) -> None:
    try:
        await ws.send_json(obj)
    except Exception:
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
    u = all_users_by_ws.get(ws)
    if not u:
        return
    keys = set(u.rooms) | set(DEFAULT_ROOMS.keys())
    items = []
    for k in sorted(keys):
        r = ensure_room(k)
        items.append({"room": k, "title": r.title, "topic": r.topic})
    await ws_send(ws, {"type": "rooms", "items": items})


async def broadcast_typing(room_key: str) -> None:
    r = ensure_room(room_key)
    await room_broadcast(room_key, {"type": "typing", "room": room_key, "items": sorted(r.typing)})


async def disconnect_ws(ws: WebSocket) -> None:
    u: Optional[User] = None
    rooms_to_update: Set[str] = set()

    async with state_lock:
        u = all_users_by_ws.pop(ws, None)
        if u:
            all_ws_by_nick_cf.pop(u.nick.casefold(), None)
            for rk in list(u.rooms):
                r = ensure_room(rk)
                r.clients.discard(ws)
                r.users.pop(ws, None)
                r.typing.discard(u.nick)
                rooms_to_update.add(rk)

    try:
        await ws.close()
    except Exception:
        pass

    if u:
        for rk in rooms_to_update:
            await broadcast_userlist(rk)
            await broadcast_typing(rk)


async def join_room(ws: WebSocket, room_key: str) -> Tuple[bool, str]:
    u = all_users_by_ws.get(ws)
    if not u:
        return False, "no_user"

    room_key = norm_room(room_key)
    if not valid_room_key(room_key):
        return False, "bad_room"

    async with state_lock:
        r = ensure_room(room_key)
        if room_key in u.rooms:
            return True, "already"
        u.rooms.add(room_key)
        r.clients.add(ws)
        r.users[ws] = u

    # Svarbu: nesiunčiam "X joined" visiems; tik pačiam useriui + online sąrašas
    await ws_send(ws, {"type": "sys", "room": room_key, "t": ts(), "text": t(u.lang, "JOIN_OK", room=ensure_room(room_key).title)})
    await broadcast_userlist(room_key)
    await broadcast_rooms_list(ws)
    await ws_send(ws, {"type": "topic", "room": room_key, "text": f"{ensure_room(room_key).title} — {ensure_room(room_key).topic}"})
    hist = await db_load_room_history(room_key, 120)
    await ws_send(ws, {"type": "history", "room": room_key, "items": hist})
    return True, "ok"


async def leave_room(ws: WebSocket, room_key: str) -> Tuple[bool, str]:
    u = all_users_by_ws.get(ws)
    if not u:
        return False, "no_user"

    room_key = norm_room(room_key)
    if room_key == "main":
        return False, "deny_main"

    async with state_lock:
        if room_key not in u.rooms:
            return False, "not_member"
        u.rooms.discard(room_key)
        r = ensure_room(room_key)
        r.clients.discard(ws)
        r.users.pop(ws, None)
        r.typing.discard(u.nick)

    await ws_send(ws, {"type": "sys", "t": ts(), "room": room_key, "text": t(u.lang, "LEAVE_OK", room=ensure_room(room_key).title)})
    await broadcast_userlist(room_key)
    await broadcast_typing(room_key)
    await broadcast_rooms_list(ws)
    return True, "ok"


def check_rate_limit(u: User, msg_len: int) -> Tuple[bool, str]:
    now = time.time()
    while u.msg_events and now - u.msg_events[0][0] > 10:
        u.msg_events.popleft()

    count = len(u.msg_events)
    chars = sum(c for _, c in u.msg_events)

    if count >= MAX_MSG_PER_10S:
        return False, "rate"
    if chars + msg_len > MAX_CHARS_PER_10S:
        return False, "rate_chars"

    u.msg_events.append((now, msg_len))
    return True, "ok"


# =====================================
# COMMANDS
# =====================================
async def cmd_rooms(ws: WebSocket) -> None:
    u = all_users_by_ws.get(ws)
    if not u:
        return
    keys = sorted(set(u.rooms) | set(DEFAULT_ROOMS.keys()))
    lines = [t(u.lang, "ROOMS_LIST")]
    for k in keys:
        r = ensure_room(k)
        lines.append(f"  {r.title} — {r.topic}")
    await ws_send(ws, {"type": "sys", "t": ts(), "text": "\n".join(lines)})


async def cmd_who(ws: WebSocket, room_key: str) -> None:
    u = all_users_by_ws.get(ws)
    if not u:
        return
    room_key = norm_room(room_key)
    items = room_userlist(room_key)
    names = [x["nick"] + (" (away)" if x["status"] == "away" else "") for x in items]
    await ws_send(ws, {"type": "sys", "t": ts(), "text": t(u.lang, "ONLINE_ROOM", room=ensure_room(room_key).title, names=", ".join(names))})


async def cmd_topic(ws: WebSocket, room_key: str, arg: Optional[str]) -> None:
    u = all_users_by_ws.get(ws)
    if not u:
        return
    room_key = norm_room(room_key)
    r = ensure_room(room_key)
    if not arg:
        await ws_send(ws, {"type": "sys", "t": ts(), "text": t(u.lang, "TOPIC_SHOW", room=r.title, topic=r.topic)})
        return

    new_topic = arg.strip()[:120] or r.topic
    r.topic = new_topic

    sys_msg = {
        "scope": "room",
        "room": room_key,
        "msg_type": "sys",
        "ts": ts(),
        "nick": None,
        "color": None,
        "text": t(u.lang, "TOPIC_CHANGED", topic=new_topic),
        "extra": {"kind": "topic_change"},
    }
    msg_id = await db_insert_message(sys_msg)

    await room_broadcast(room_key, {"type": "topic", "room": room_key, "text": f"{r.title} — {new_topic}"})
    await room_broadcast(room_key, {"type": "sys", "room": room_key, "t": sys_msg["ts"], "text": sys_msg["text"], "id": msg_id})

    # atnaujinti rooms list (topic matosi ten)
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
        await ws_send(ws, {"type": "sys", "t": ts(), "text": t(u.lang, "DM_USAGE")})
        return

    async with state_lock:
        target_ws = all_ws_by_nick_cf.get(target.casefold())
        if not target_ws:
            await ws_send(ws, {"type": "sys", "t": ts(), "text": t(u.lang, "DM_NOT_FOUND", nick=target)})
            return
        target_user = all_users_by_ws.get(target_ws)
        if not target_user:
            await ws_send(ws, {"type": "sys", "t": ts(), "text": t(u.lang, "DM_NOT_FOUND", nick=target)})
            return

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

    text = f"{u.nick} {('meta' if u.lang!='en' else 'rolls')} {n}d{sides}: {rolls} (total {total})"
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
    res_lt = random.choice(["HERBAS", "SKAIČIUS"])
    res_en = "HEADS" if res_lt == "HERBAS" else "TAILS"
    res = res_en if u.lang == "en" else res_lt

    text = f"{u.nick} {('flips a coin' if u.lang=='en' else 'meta monetą')}: {res}"
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
        "extra": {"kind": "me", "created_at": time.time(), "edited": False, "reactions": {}},
    }
    msg_id = await db_insert_message(msg)
    await room_broadcast(
        room_key,
        {"type": "me_action", "room": room_key, "t": msg["ts"], "nick": u.nick, "color": u.color, "text": action, "id": msg_id, "extra": msg["extra"]},
    )


async def cmd_pin(ws: WebSocket, room_key: str, msg_id: int) -> None:
    u = all_users_by_ws.get(ws)
    if not u:
        return
    room_key = norm_room(room_key)
    row = await db_get_message(msg_id)
    if not row or row["scope"] != "room" or row["room"] != room_key:
        await ws_send(ws, {"type": "sys", "t": ts(), "text": t(u.lang, "PIN_FAIL")})
        return
    await db_pin(room_key, msg_id)
    await room_broadcast(room_key, {"type": "pin_update", "room": room_key, "action": "pin", "id": msg_id})
    await ws_send(ws, {"type": "sys", "t": ts(), "text": t(u.lang, "PIN_OK", id=msg_id)})


async def cmd_pins(ws: WebSocket, room_key: str) -> None:
    u = all_users_by_ws.get(ws)
    if not u:
        return
    room_key = norm_room(room_key)
    ids = await db_list_pins(room_key, 50)
    if not ids:
        await ws_send(ws, {"type": "sys", "t": ts(), "text": t(u.lang, "PINS_NONE")})
        return
    await ws_send(ws, {"type": "pins", "room": room_key, "items": ids})


async def cmd_quote(ws: WebSocket, room_key: str, msg_id: int) -> None:
    u = all_users_by_ws.get(ws)
    if not u:
        return
    room_key = norm_room(room_key)
    row = await db_get_message(msg_id)
    if not row or row["scope"] != "room" or row["room"] != room_key:
        await ws_send(ws, {"type": "sys", "t": ts(), "text": t(u.lang, "QUOTE_FAIL")})
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
        await ws_send(ws, {"type": "sys", "t": ts(), "text": t(u.lang, "EDIT_FAIL_NOTFOUND")})
        return
    if (row["nick"] or "") != u.nick:
        await ws_send(ws, {"type": "sys", "t": ts(), "text": t(u.lang, "EDIT_FAIL_OWN")})
        return

    extra = json_loads(row["extra"])
    created_at = extra.get("created_at", None)
    if created_at is not None:
        try:
            if time.time() - float(created_at) > EDIT_WINDOW_SEC:
                await ws_send(ws, {"type": "sys", "t": ts(), "text": t(u.lang, "EDIT_FAIL_TIME")})
                return
        except Exception:
            pass

    new_text = (new_text or "").strip()[:300]
    if not new_text:
        await ws_send(ws, {"type": "sys", "t": ts(), "text": t(u.lang, "EDIT_FAIL_EMPTY")})
        return

    extra["edited"] = True
    extra.setdefault("created_at", time.time())
    await db_update_message_text(msg_id, new_text, extra)
    await room_broadcast(room_key, {"type": "edit", "room": room_key, "id": msg_id, "text": new_text, "edited": True, "extra": extra})


async def cmd_delete(ws: WebSocket, room_key: str, msg_id: int) -> None:
    u = all_users_by_ws.get(ws)
    if not u:
        return
    room_key = norm_room(room_key)
    row = await db_get_message(msg_id)
    if not row or row["scope"] != "room" or row["room"] != room_key:
        await ws_send(ws, {"type": "sys", "t": ts(), "text": t(u.lang, "DEL_FAIL_NOTFOUND")})
        return
    if (row["nick"] or "") != u.nick:
        await ws_send(ws, {"type": "sys", "t": ts(), "text": t(u.lang, "DEL_FAIL_OWN")})
        return

    extra = json_loads(row["extra"])
    created_at = extra.get("created_at", None)
    if created_at is not None:
        try:
            if time.time() - float(created_at) > EDIT_WINDOW_SEC:
                await ws_send(ws, {"type": "sys", "t": ts(), "text": t(u.lang, "DEL_FAIL_TIME")})
                return
        except Exception:
            pass

    await db_mark_deleted(msg_id)
    await room_broadcast(room_key, {"type": "delete", "room": room_key, "id": msg_id})


async def cmd_react(ws: WebSocket, room_key: str, msg_id: int, emoji: str) -> None:
    u = all_users_by_ws.get(ws)
    if not u:
        return
    room_key = norm_room(room_key)
    row = await db_get_message(msg_id)
    if not row:
        await ws_send(ws, {"type": "sys", "t": ts(), "text": t(u.lang, "REACT_FAIL")})
        return
    if row["scope"] != "room" or row["room"] != room_key:
        await ws_send(ws, {"type": "sys", "t": ts(), "text": t(u.lang, "REACT_FAIL_ROOM")})
        return

    emoji = (emoji or "").strip()
    if not emoji:
        await ws_send(ws, {"type": "sys", "t": ts(), "text": t(u.lang, "REACT_USAGE")})
        return
    emoji = emoji[:8]

    extra = json_loads(row["extra"])
    reactions = extra.get("reactions", {})
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
    u = all_users_by_ws.get(ws)
    if not u:
        return
    room_key = norm_room(room_key)
    r = ensure_room(room_key)
    r.game_target = random.randint(1, 100)
    r.game_active = True

    text = t(u.lang, "GAME_START")
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
        await ws_send(ws, {"type": "sys", "t": ts(), "text": t(u.lang, "GAME_NOT_STARTED")})
        return

    guess = int(guess)
    if guess < 1 or guess > 100:
        await ws_send(ws, {"type": "sys", "t": ts(), "text": t(u.lang, "GUESS_RANGE")})
        return

    if guess == r.game_target:
        r.game_active = False
        target = r.game_target
        r.game_target = None
        text = f"{u.nick} {'atspėjo' if u.lang!='en' else 'guessed'}! {('Buvo' if u.lang!='en' else 'It was')} {target}. /game start"
        extra = {"kind": "game_win", "by": u.nick, "target": target}
    elif guess < r.game_target:
        text = f"{u.nick} {('spėja' if u.lang!='en' else 'guesses')} {guess}: {('per mažai' if u.lang!='en' else 'too low')}."
        extra = {"kind": "game_hint", "by": u.nick, "guess": guess, "hint": "low"}
    else:
        text = f"{u.nick} {('spėja' if u.lang!='en' else 'guesses')} {guess}: {('per daug' if u.lang!='en' else 'too high')}."
        extra = {"kind": "game_hint", "by": u.nick, "guess": guess, "hint": "high"}

    msg = {
        "scope": "room",
        "room": room_key,
        "msg_type": "sys",
        "ts": ts(),
        "nick": None,
        "color": None,
        "text": text,
        "extra": extra,
    }
    msg_id = await db_insert_message(msg)
    await room_broadcast(room_key, {"type": "sys", "room": room_key, "t": msg["ts"], "text": text, "id": msg_id})


async def cmd_lang(ws: WebSocket, lang_arg: str) -> None:
    u = all_users_by_ws.get(ws)
    if not u:
        return
    arg = (lang_arg or "").strip().lower()
    if arg not in LANGS:
        await ws_send(ws, {"type": "sys", "t": ts(), "text": t(u.lang, "LANG_USAGE")})
        return
    u.lang = arg
    await ws_send(ws, {"type": "lang", "lang": u.lang})
    await ws_send(ws, {"type": "sys", "t": ts(), "text": t(u.lang, "LANG_SET", lang=u.lang)})


async def handle_command(ws: WebSocket, active_room: str, text: str) -> bool:
    u = all_users_by_ws.get(ws)
    if not u:
        return True

    t0 = text.strip()
    low = t0.lower()

    if low in ("/help", "/?"):
        await ws_send(ws, {"type": "sys", "t": ts(), "text": help_text(u.lang)})
        return True

    if low == "/rooms":
        await cmd_rooms(ws)
        return True

    if low.startswith("/join "):
        arg = t0.split(" ", 1)[1].strip()
        ok, code = await join_room(ws, arg)
        if not ok and code == "bad_room":
            await ws_send(ws, {"type": "sys", "t": ts(), "text": t(u.lang, "JOIN_BAD_ROOM")})
        else:
            if code == "already":
                await ws_send(ws, {"type": "sys", "t": ts(), "text": t(u.lang, "JOIN_ALREADY")})
        return True

    if low.startswith("/leave "):
        arg = t0.split(" ", 1)[1].strip()
        ok, code = await leave_room(ws, arg)
        if not ok:
            if code == "deny_main":
                await ws_send(ws, {"type": "sys", "t": ts(), "text": t(u.lang, "LEAVE_MAIN_DENY")})
            elif code == "not_member":
                await ws_send(ws, {"type": "sys", "t": ts(), "text": t(u.lang, "LEAVE_NOT_MEMBER")})
        return True

    if low == "/who":
        await cmd_who(ws, active_room)
        return True

    if low.startswith("/topic"):
        parts = t0.split(" ", 1)
        arg = parts[1] if len(parts) == 2 else None
        await cmd_topic(ws, active_room, arg)
        return True

    if low.startswith("/history"):
        parts = t0.split(" ", 1)
        n = 120
        if len(parts) == 2:
            try:
                n = int(parts[1].strip())
            except Exception:
                n = 120
        await cmd_history(ws, active_room, n)
        return True

    if low.startswith("/dmhistory "):
        parts = t0.split(" ")
        who = parts[1] if len(parts) >= 2 else ""
        n = 120
        if len(parts) >= 3:
            try:
                n = int(parts[2])
            except Exception:
                n = 120
        await cmd_dmhistory(ws, who, n)
        return True

    if low.startswith("/dm "):
        parts = t0.split(" ", 2)
        if len(parts) < 3:
            await ws_send(ws, {"type": "sys", "t": ts(), "text": t(u.lang, "DM_USAGE")})
            return True
        await cmd_dm(ws, parts[1], parts[2])
        return True

    if low.startswith("/me "):
        await cmd_me(ws, active_room, t0.split(" ", 1)[1])
        return True

    m = ROLL_RE.match(t0)
    if m:
        n = int(m.group(1)) if m.group(1) else 1
        sides = int(m.group(2)) if m.group(2) else 6
        await cmd_roll(ws, active_room, n, sides)
        return True

    if low == "/flip":
        await cmd_flip(ws, active_room)
        return True

    if low == "/time":
        await ws_send(ws, {"type": "sys", "t": ts(), "text": t(u.lang, "TIME", t=ts())})
        return True

    if low.startswith("/pin "):
        try:
            msg_id = int(t0.split(" ", 1)[1].strip())
        except Exception:
            await ws_send(ws, {"type": "sys", "t": ts(), "text": "Usage: /pin ID"})
            return True
        await cmd_pin(ws, active_room, msg_id)
        return True

    if low == "/pins":
        await cmd_pins(ws, active_room)
        return True

    if low.startswith("/quote "):
        try:
            msg_id = int(t0.split(" ", 1)[1].strip())
        except Exception:
            await ws_send(ws, {"type": "sys", "t": ts(), "text": "Usage: /quote ID"})
            return True
        await cmd_quote(ws, active_room, msg_id)
        return True

    if low.startswith("/edit "):
        parts = t0.split(" ", 2)
        if len(parts) < 3:
            await ws_send(ws, {"type": "sys", "t": ts(), "text": "Usage: /edit ID NEW_TEXT"})
            return True
        try:
            msg_id = int(parts[1])
        except Exception:
            await ws_send(ws, {"type": "sys", "t": ts(), "text": "Usage: /edit ID NEW_TEXT"})
            return True
        await cmd_edit(ws, active_room, msg_id, parts[2])
        return True

    if low.startswith("/del "):
        try:
            msg_id = int(t0.split(" ", 1)[1].strip())
        except Exception:
            await ws_send(ws, {"type": "sys", "t": ts(), "text": "Usage: /del ID"})
            return True
        await cmd_delete(ws, active_room, msg_id)
        return True

    if low.startswith("/react "):
        parts = t0.split(" ", 2)
        if len(parts) < 3:
            await ws_send(ws, {"type": "sys", "t": ts(), "text": t(u.lang, "REACT_USAGE")})
            return True
        try:
            msg_id = int(parts[1])
        except Exception:
            await ws_send(ws, {"type": "sys", "t": ts(), "text": t(u.lang, "REACT_USAGE")})
            return True
        await cmd_react(ws, active_room, msg_id, parts[2])
        return True

    if low == "/game start":
        await cmd_game_start(ws, active_room)
        return True

    if low.startswith("/guess "):
        parts = t0.split(" ", 1)
        try:
            guess = int(parts[1].strip())
        except Exception:
            await ws_send(ws, {"type": "sys", "t": ts(), "text": "Usage: /guess 42"})
            return True
        await cmd_guess(ws, active_room, guess)
        return True

    if low.startswith("/lang "):
        parts = t0.split(" ", 1)
        await cmd_lang(ws, parts[1] if len(parts) == 2 else "")
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
async def check_nick(nick: str = "", lang: str = "lt"):
    lang = (lang or "lt").lower()
    if lang not in LANGS:
        lang = "lt"

    n = (nick or "").strip()
    if not valid_nick(n):
        return JSONResponse({"ok": False, "code": "BAD_NICK", "reason": t(lang, "BAD_NICK")})
    async with state_lock:
        if n.casefold() in all_ws_by_nick_cf:
            return JSONResponse({"ok": False, "code": "NICK_TAKEN", "reason": t(lang, "NICK_TAKEN")})
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
# CLIENT HTML (Light/Dark + LT/EN)
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

    /* Dark themes */
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

    /* Light theme */
    body.theme-light{
      --bg:#f6f8fb;
      --panel:rgba(255,255,255,.88);
      --panel2:rgba(255,255,255,.80);
      --border:rgba(8,20,40,.14);
      --text:#0b1220;
      --muted:rgba(11,18,32,.55);
      --accent:#246BFD;
      --accent2:#12B981;
      --danger:#e23d3d;
      --shadow:0 12px 28px rgba(15, 25, 35, .14);
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
    body.theme-light .bg::before{ opacity:.10; }
    .bg::after{
      content:""; position:absolute; inset:0; opacity:.10;
      background: repeating-linear-gradient(
        to bottom,
        rgba(0,0,0,0) 0px,
        rgba(0,0,0,0) 2px,
        rgba(0,0,0,.40) 3px
      );
      animation: scan 8s linear infinite;
    }
    body.theme-light .bg::after{ opacity:.04; }
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
      background:rgba(0,0,0,.10);
      padding:6px 10px;
      border-radius:999px;
      display:inline-flex; align-items:center; gap:8px;
      white-space:nowrap;
      max-width: 100%;
    }
    body.theme-light .pill{ background:rgba(11,18,32,.04); }
    .pill span.ellip{
      overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
      max-width: 520px;
      display:inline-block;
      vertical-align:bottom;
    }
    .dot{ width:10px; height:10px; border-radius:999px; background:var(--danger);
      box-shadow:0 0 14px rgba(255,107,107,.18); }
    .dot.ok{ background:var(--accent2); box-shadow:0 0 14px rgba(18,185,129,.18); }

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
      background:rgba(0,0,0,.12);
      border:1px solid var(--border);
      border-radius:12px;
      color:var(--text);
      outline:none;
      box-sizing:border-box;
    }
    body.theme-light .search input{ background:rgba(11,18,32,.04); }

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
      background:rgba(0,0,0,0.08);
      margin-bottom:8px;
      cursor:pointer;
      user-select:none;
    }
    body.theme-light .item{ background:rgba(11,18,32,.03); border-color:rgba(11,18,32,.06); }
    .item:hover{ filter:brightness(1.04); }
    .item.active{
      border-color: rgba(124,255,107,.28);
      background: rgba(124,255,107,.06);
    }
    body.theme-light .item.active{
      border-color: rgba(36,107,253,.28);
      background: rgba(36,107,253,.06);
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

    #log{
      padding:14px 16px;
      overflow:auto;
      white-space:pre-wrap;
      line-height:1.45;
      min-height:0;
    }
    .line{ margin:2px 0; }
    .t{ color: rgba(124,255,107,.40); }
    body.theme-light .t{ color: rgba(11,18,32,.38); }
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
    body.theme-light .idTag{ color: rgba(11,18,32,.35); }
    .meta{
      color: rgba(202,255,217,.32);
      font-size: 12px;
      margin-left: 8px;
    }
    body.theme-light .meta{ color: rgba(11,18,32,.42); }

    .reactions{
      display:inline-flex;
      gap:6px;
      margin-left:10px;
      flex-wrap:wrap;
    }
    .react{
      border:1px solid rgba(255,255,255,.06);
      background:rgba(0,0,0,.12);
      padding:2px 8px;
      border-radius:999px;
      font-size:12px;
      color:var(--text);
      cursor:pointer;
      user-select:none;
    }
    body.theme-light .react{ background:rgba(11,18,32,.04); border-color:rgba(11,18,32,.10); }

    .rightGrid{
      display:grid;
      grid-template-rows:auto 1fr auto 1fr;
      min-height:0;
    }

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
      background:rgba(0,0,0,.12);
      border:1px solid var(--border);
      border-radius:12px;
      color:var(--text);
      outline:none;
      box-sizing:border-box;
    }
    body.theme-light .bottombar input{ background:rgba(11,18,32,.04); }
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
    .bottombar button:hover{ filter:brightness(1.06); }
    .bottombar button:disabled{ opacity:.55; cursor:not-allowed; }

    /* Lobby */
    #lobby{
      position:fixed; inset:0; z-index:10;
      display:flex; align-items:center; justify-content:center;
      background:rgba(0,0,0,.62);
      padding:18px;
      box-sizing:border-box;
    }
    body.theme-light #lobby{ background:rgba(15, 25, 35, .45); }
    .card{
      width:min(980px, 96vw);
      border:1px solid var(--border);
      border-radius:18px;
      background:var(--panel);
      box-shadow:0 24px 70px rgba(0,0,0,.35);
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
      grid-template-columns: 360px 1fr;
      gap:14px;
      padding:16px 18px;
    }
    .box{
      border:1px solid var(--border);
      border-radius:16px;
      background:rgba(0,0,0,.08);
      padding:12px 12px;
    }
    body.theme-light .box{ background:rgba(11,18,32,.03); }
    .label{ color:var(--muted); font-size:12px; margin-bottom:8px; }
    .row2{ display:grid; grid-template-columns: 1fr 1fr; gap:10px; }
    .nickrow input{
      width:100%; padding:12px 12px;
      font-family:var(--mono);
      border-radius:12px;
      border:1px solid var(--border);
      background:rgba(0,0,0,.12);
      color:var(--text);
      outline:none;
      box-sizing:border-box;
    }
    body.theme-light .nickrow input{ background:rgba(11,18,32,.04); }

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
      background:rgba(0,0,0,.08);
      padding:10px;
      border-radius:12px;
    }
    body.theme-light #nickSuggestBox{ background:rgba(11,18,32,.03); }
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

    select.sel{
      padding:10px 10px;
      border-radius:12px;
      border:1px solid var(--border);
      background:rgba(0,0,0,.12);
      color:var(--text);
      font-family:var(--mono);
      outline:none;
      box-sizing:border-box;
      width: 100%;
    }
    body.theme-light select.sel{ background:rgba(11,18,32,.04); }

    @media (max-width: 1000px){
      .main{ grid-template-columns: 1fr; }
      .status{ display:none; }
      .brand{ min-width: unset; }
      .card-body{ grid-template-columns: 1fr; }
      .row2{ grid-template-columns: 1fr; }
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
          <div class="h1" id="lobbyTitle">HestioRooms Chat</div>
          <div class="sub" id="lobbySub">Įvesk slapyvardį. Kol nick neužimtas – prisijungti negalima.</div>
        </div>
        <div class="sub" id="lobbyTag">for fun</div>
      </div>

      <div class="card-body">
        <div class="box">
          <div class="label" id="lblNick">Slapyvardis (2–24 simboliai)</div>
          <div class="nickrow">
            <input id="nickPick" placeholder="pvz. Tomas" maxlength="24"/>
          </div>
          <div id="nickErr"></div>
          <div id="nickState"></div>

          <div id="nickSuggestBox">
            <span id="suggestLabel">Siūlomas nick:</span> <b id="nickSuggestVal"></b>
            <button id="applySuggest">Pritaikyti</button>
          </div>

          <div class="label" style="margin-top:12px;" id="lblTheme">Tema</div>
          <select id="themePick" class="sel">
            <option value="theme-cyber">Cyber</option>
            <option value="theme-glass">Glass</option>
            <option value="theme-matrix">Matrix</option>
            <option value="theme-crt">CRT</option>
            <option value="theme-light">Light</option>
          </select>

          <div class="label" style="margin-top:12px;" id="lblLang">Kalba</div>
          <select id="langPick" class="sel">
            <option value="lt">LT</option>
            <option value="en">EN</option>
          </select>

          <div class="sub" style="margin-top:12px;" id="lobbyHint">
            Patarimas: /help parodo visas komandas. /join #games – žaidimų kanalas.
          </div>
        </div>

        <div class="box">
          <div class="label" id="lblStart">Start</div>
          <button id="joinBtn" class="joinbtn" disabled>Prisijungti</button>
          <div class="sub" style="margin-top:12px;" id="lobbySaveHint">
            Nick, tema ir kalba išsaugomi naršyklėje.
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

      <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
        <select id="langTop" class="sel" style="width:auto; min-width:86px;" title="Language">
          <option value="lt">LT</option>
          <option value="en">EN</option>
        </select>
        <select id="themeTop" class="sel" style="width:auto; min-width:140px;" title="Theme">
          <option value="theme-cyber">Cyber</option>
          <option value="theme-glass">Glass</option>
          <option value="theme-matrix">Matrix</option>
          <option value="theme-crt">CRT</option>
          <option value="theme-light">Light</option>
        </select>
        <div class="pill" id="meNickPill" title="Me" style="min-width:160px; justify-content:center;"></div>
      </div>
    </div>

    <div class="main">
      <!-- Left: Channels -->
      <div class="panel" style="display:grid; grid-template-rows:auto auto 1fr; min-height:0;">
        <div class="sidehead"><span id="roomsTitle">Kanalai</span><span>rooms</span></div>
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
        <div class="sidehead"><span id="dmTitle">DM</span><span>tabs</span></div>
        <div class="list" id="dms"></div>

        <div class="sidehead"><span><span id="onlineTitle">Online</span>: <b id="onlineCount">0</b></span><span>live</span></div>
        <div class="list" id="users"></div>
      </div>
    </div>

    <div class="bottombar">
      <input id="msg" placeholder="rašyk žinutę ir Enter..." maxlength="300"/>
      <button id="btn">Siųsti</button>
    </div>
  </div>

<script>
  // ========= I18N client =========
  const I18N = {
    lt: {
      lobbySub: "Įvesk slapyvardį. Kol nick neužimtas – prisijungti negalima.",
      lblNick: "Slapyvardis (2–24 simboliai)",
      suggestLabel: "Siūlomas nick:",
      applySuggest: "Pritaikyti",
      lblTheme: "Tema",
      lblLang: "Kalba",
      lobbyHint: "Patarimas: /help parodo visas komandas. /join #games – žaidimų kanalas.",
      lblStart: "Start",
      joinBtn: "Prisijungti",
      lobbySaveHint: "Nick, tema ir kalba išsaugomi naršyklėje.",
      roomsTitle: "Kanalai",
      roomJoinPh: "įrašyk #room ir Enter (pvz #games)",
      dmTitle: "DM",
      onlineTitle: "Online",
      btnSend: "Siųsti",
      msgPhRoom: (room) => `rašyk į #${room} ir Enter... (komandos: /help)`,
      msgPhDM: (peer) => `rašyk DM ${peer} ir Enter...`,
      connConnected: "Connected",
      connDisconnected: "Disconnected",
      nickInvalid: "Netinkamas nick. Reikia 2–24 simbolių (raidės/skaičiai/tarpas/_-.)",
      nickChecking: "Tikrinama ar nick laisvas...",
      nickFree: "Nick laisvas. Galite prisijungti.",
      nickTaken: "Nick užimtas. Pasirink kitą.",
      noConn: "nėra ryšio su serveriu.",
      typingOne: (u) => `${u} rašo...`,
      typingMany: (a,b,n) => `${a}, ${b} ir dar ${n} rašo...`,
      reactPrompt: "Reakcija (pvz 😀):"
    },
    en: {
      lobbySub: "Enter a nickname. You cannot join until it is available.",
      lblNick: "Nickname (2–24 chars)",
      suggestLabel: "Suggested nick:",
      applySuggest: "Apply",
      lblTheme: "Theme",
      lblLang: "Language",
      lobbyHint: "Tip: /help shows commands. /join #games opens the games channel.",
      lblStart: "Start",
      joinBtn: "Join",
      lobbySaveHint: "Nick, theme and language are saved in your browser.",
      roomsTitle: "Channels",
      roomJoinPh: "type #room and Enter (e.g. #games)",
      dmTitle: "DM",
      onlineTitle: "Online",
      btnSend: "Send",
      msgPhRoom: (room) => `type in #${room} and Enter... (commands: /help)`,
      msgPhDM: (peer) => `type DM to ${peer} and Enter...`,
      connConnected: "Connected",
      connDisconnected: "Disconnected",
      nickInvalid: "Invalid nick. Use 2–24 chars (letters/numbers/space/_-.)",
      nickChecking: "Checking nickname availability...",
      nickFree: "Nick is available. You can join.",
      nickTaken: "Nick is taken. Choose another.",
      noConn: "no connection to server.",
      typingOne: (u) => `${u} is typing...`,
      typingMany: (a,b,n) => `${a}, ${b} and ${n} more are typing...`,
      reactPrompt: "Reaction (e.g. 😀):"
    }
  };

  let UI_LANG = (localStorage.getItem("lang") || "lt").toLowerCase();
  if(!I18N[UI_LANG]) UI_LANG = "lt";

  function T(key){ return I18N[UI_LANG][key]; }

  function applyLangToUI(){
    document.documentElement.lang = UI_LANG;
    document.getElementById("lobbySub").textContent = T("lobbySub");
    document.getElementById("lblNick").textContent = T("lblNick");
    document.getElementById("suggestLabel").textContent = T("suggestLabel");
    document.getElementById("applySuggest").textContent = T("applySuggest");
    document.getElementById("lblTheme").textContent = T("lblTheme");
    document.getElementById("lblLang").textContent = T("lblLang");
    document.getElementById("lobbyHint").textContent = T("lobbyHint");
    document.getElementById("lblStart").textContent = T("lblStart");
    document.getElementById("joinBtn").textContent = T("joinBtn");
    document.getElementById("lobbySaveHint").textContent = T("lobbySaveHint");
    document.getElementById("roomsTitle").textContent = T("roomsTitle");
    document.getElementById("dmTitle").textContent = T("dmTitle");
    document.getElementById("onlineTitle").textContent = T("onlineTitle");
    document.getElementById("roomJoin").placeholder = T("roomJoinPh");
    document.getElementById("btn").textContent = T("btnSend");
  }

  // ========= Elements =========
  const lobby = document.getElementById("lobby");
  const appEl = document.getElementById("app");

  const nickPick = document.getElementById("nickPick");
  const nickErr  = document.getElementById("nickErr");
  const nickState = document.getElementById("nickState");
  const joinBtn = document.getElementById("joinBtn");

  const nickSuggestBox = document.getElementById("nickSuggestBox");
  const nickSuggestVal = document.getElementById("nickSuggestVal");
  const applySuggestBtn = document.getElementById("applySuggest");

  const themePick = document.getElementById("themePick");
  const themeTop  = document.getElementById("themeTop");
  const langPick  = document.getElementById("langPick");
  const langTop   = document.getElementById("langTop");

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
  let reconnectTimeout = null;
  let reconnecting = false;
  let joinEstablished = false;
  let fatalJoinError = false;

  // Identity
  let nick = "";
  let myNick = "";
  let myColor = "#caffd9";

  // Convos
  const convs = new Map();
  let activeKey = "room:main";
  const msgDom = new Map();

  // Typing
  let typing = false;
  let typingTimer = null;
  let lastTypingSend = 0;

  // ========= Utilities =========
  function esc(s){
    return (s ?? "").toString()
      .replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;")
      .replaceAll('"',"&quot;").replaceAll("'","&#39;");
  }

  function setConn(ok){
    if(ok){
      connDot.classList.add("ok");
      connText.textContent = T("connConnected");
    }else{
      connDot.classList.remove("ok");
      connText.textContent = T("connDisconnected");
    }
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

  function setTheme(cls){
    document.body.className = cls;
    localStorage.setItem("theme", cls);
    themePick.value = cls;
    themeTop.value = cls;
  }

  function setLang(lang){
    UI_LANG = (lang || "lt").toLowerCase();
    if(!I18N[UI_LANG]) UI_LANG = "lt";
    localStorage.setItem("lang", UI_LANG);
    langPick.value = UI_LANG;
    langTop.value = UI_LANG;
    applyLangToUI();
    if(ws && ws.readyState === WebSocket.OPEN){
      wsSend({type:"say", room:"main", text:`/lang ${UI_LANG}`});
    }
    setHeader();
    setConn(ws && ws.readyState === WebSocket.OPEN);
  }

  themePick.addEventListener("change", () => setTheme(themePick.value));
  themeTop.addEventListener("change", () => setTheme(themeTop.value));
  langPick.addEventListener("change", () => setLang(langPick.value));
  langTop.addEventListener("change", () => setLang(langTop.value));

  function showLobby(show){
    lobby.style.display = show ? "flex" : "none";
    appEl.style.display = show ? "none" : "grid";
    if(show) setTimeout(() => nickPick.focus(), 40);
    else setTimeout(() => msgEl.focus(), 40);
  }

  function clearReconnect(){
    if(reconnectTimeout){
      clearTimeout(reconnectTimeout);
      reconnectTimeout = null;
    }
    reconnecting = false;
  }

  function scheduleReconnect(){
    if(reconnecting) return;
    reconnecting = true;
    reconnectTimeout = setTimeout(() => {
      reconnecting = false;
      connect();
    }, 1500);
  }

  function wsUrl(){
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const qs = new URLSearchParams({ nick, lang: UI_LANG }).toString();
    return `${proto}://${location.host}/ws?${qs}`;
  }

  function tsLocal(){
    const d = new Date();
    const hh = String(d.getHours()).padStart(2,'0');
    const mm = String(d.getMinutes()).padStart(2,'0');
    const ss = String(d.getSeconds()).padStart(2,'0');
    return `${hh}:${mm}:${ss}`;
  }

  // ========= Conversations =========
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

  function setHeader(){
    const c = convs.get(activeKey);
    if(!c) return;
    if(c.kind === "room"){
      topicEl.textContent = `#${c.room}`;
      msgEl.placeholder = T("msgPhRoom")(c.room);
    }else{
      topicEl.textContent = `DM ${c.peer}`;
      msgEl.placeholder = T("msgPhDM")(c.peer);
    }
  }

  function renderSidebars(){
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
          <div class="idesc">${UI_LANG==='en' ? 'private chat' : 'privatus pokalbis'}</div>
        </div>
        <div class="badge" style="${unread>0 ? 'display:inline-flex;' : ''}">${unread}</div>
      `;
      row.addEventListener("click", () => {
        setActive(c.key);
        if(!c.loaded){
          c.loaded = true;
          wsSend({type:"dm_history_req", with:c.peer, limit:120});
        }
      });
      dmsEl.appendChild(row);
    }
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
      el.addEventListener("click", (ev) => {
        if(!ev.altKey) return;
        const emoji = prompt(T("reactPrompt"));
        if(!emoji) return;
        const c = convs.get(activeKey);
        if(c && c.kind === "room"){
          wsSend({ type:"say", room: c.room, text: `/react ${id} ${emoji}` });
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
      addLineElement(renderMessageObj(obj));
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

  // ========= Routing =========
  function routeIncoming(o){
    if(o.type === "lang"){
      if(o.lang && (o.lang === "lt" || o.lang === "en")){
        UI_LANG = o.lang;
        localStorage.setItem("lang", UI_LANG);
        langPick.value = UI_LANG;
        langTop.value = UI_LANG;
        applyLangToUI();
        setHeader();
        setConn(true);
      }
      return;
    }

    if(o.type === "rooms"){
      for(const it of (o.items || [])){
        ensureRoomConvo(it.room, it.title, it.topic);
      }
      renderSidebars();
      if(!convs.has(activeKey)){
        ensureRoomConvo("main", "#main", "");
        setActive("room:main");
      }
      return;
    }

    if(o.type === "topic"){
      const room = o.room || "main";
      const key = ensureRoomConvo(room, `#${room}`, "");
      const c = convs.get(key);
      if(c){
        c.topic = (o.text || "").split("—").slice(1).join("—").trim() || c.topic;
      }
      if(activeKey === key) setHeader();
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
            <div class="idesc">${UI_LANG==='en' ? 'click for DM' : 'spausk DM'}</div>
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
        typingText.textContent = T("typingOne")(items[0]);
      } else {
        typingText.textContent = T("typingMany")(items[0], items[1], items.length-2);
      }
      return;
    }

    if(o.type === "pins"){
      const room = o.room || "main";
      const ids = o.items || [];
      const txt = ids.length ? `Pins: ${ids.map(x => "#"+x).join(", ")}` : (UI_LANG==='en' ? "No pins." : "Pin’ų nėra.");
      pushToConvo(`room:${room}`, {type:"sys", t: tsLocal(), text: txt}, true);
      return;
    }

    if(o.type === "quote"){
      const id = o.id;
      const nick = o.nick || "sys";
      const text = o.text || "";
      msgEl.value = `> #${id} ${nick}: ${text}\n`;
      msgEl.focus();
      return;
    }

    if(o.type === "edit"){
      const room = o.room || "main";
      const id = Number(o.id);
      const key = `room:${room}`;
      const c = convs.get(key);
      if(c){
        const it = c.items.find(x => Number(x.id) === id);
        if(it){
          it.text = o.text;
          it.extra = o.extra || it.extra || {};
          it.extra.edited = true;
        }
      }
      const el = msgDom.get(id);
      if(el && c){
        const it2 = c.items.find(x => Number(x.id) === id);
        if(it2){
          const newEl = renderMessageObj(it2);
          el.replaceWith(newEl);
          msgDom.set(id, newEl);
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
      if(el && c){
        const it2 = c.items.find(x => Number(x.id) === id);
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
      if(el && c){
        const it2 = c.items.find(x => Number(x.id) === id);
        if(it2){
          const newEl = renderMessageObj(it2);
          el.replaceWith(newEl);
          msgDom.set(id, newEl);
        }
      }
      return;
    }

    if(o.type === "history"){
      const room = o.room || "main";
      const key = ensureRoomConvo(room, `#${room}`, "");
      const c = convs.get(key);
      c.items = [];
      for(const it of (o.items || [])){
        c.items.push({
          id: it.id,
          type: it.type,
          t: it.t,
          nick: it.nick,
          color: it.color,
          text: it.text,
          extra: it.extra || {}
        });
      }
      if(activeKey === key) renderActiveLog();
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
      if(activeKey === key) renderActiveLog();
      renderSidebars();
      return;
    }

    if(o.type === "dm"){
      const peer = (myNick && (o.from||"").toLowerCase() === myNick.toLowerCase()) ? (o.to||"") : (o.from||"");
      const key = ensureDmConvo(peer);
      renderSidebars();
      pushToConvo(key, { id:o.id, type:"dm", t:o.t, from:o.from, to:o.to, color:o.color, text:o.text, extra:o.extra||{} }, false);
      return;
    }

    const room = o.room || "main";
    const key = ensureRoomConvo(room, `#${room}`, "");
    pushToConvo(key, { id:o.id, type:o.type, t:o.t||tsLocal(), nick:o.nick, color:o.color, text:o.text, extra:o.extra||{} }, false);
    renderSidebars();
  }

  function wsSend(obj){
    if(!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify(obj));
  }

  // ========= WS connect =========
  function connect(){
    joinEstablished = false;
    fatalJoinError = false;
    setConn(false);
    clearReconnect();

    convs.clear();
    ensureRoomConvo("main", "#main", "Bendras kanalas");
    activeKey = "room:main";
    renderSidebars();
    setHeader();
    renderActiveLog();

    pushToConvo("room:main", {type:"sys", t: tsLocal(), text: UI_LANG==='en' ? "connecting..." : "jungiamasi..."}, true);

    ws = new WebSocket(wsUrl());

    ws.onopen = () => {
      setConn(true);
      clearReconnect();
      // server-side language sync
      wsSend({type:"say", room:"main", text:`/lang ${UI_LANG}`});
    };

    ws.onmessage = (ev) => {
      let o = null;
      try { o = JSON.parse(ev.data); } catch { return; }

      // KEEPALIVE: server ping -> client pong
      if(o && o.type === "ping"){
        wsSend({ type: "pong", t: Date.now() });
        return;
      }

      if(o.type === "error"){
        fatalJoinError = true;
        clearReconnect();
        try{ ws.close(); }catch{}
        showLobby(true);
        setNickState("");
        setNickError(o.text || (UI_LANG==='en' ? "Join failed." : "Prisijungti nepavyko."));
        joinBtn.disabled = true;
        scheduleNickCheck();
        return;
      }

      if(o.type === "me"){
        joinEstablished = true;
        myNick = o.nick || "";
        myColor = o.color || "#caffd9";
        meNickPill.innerHTML = `<b style="color:${esc(myColor)}">${esc(myNick)}</b>`;
        setActive("room:main");
        return;
      }

      if(["rooms","topic","history","users"].includes(o.type)) joinEstablished = true;
      routeIncoming(o);
    };

    ws.onclose = () => {
      setConn(false);

      if(!joinEstablished){
        clearReconnect();
        showLobby(true);
        if(!fatalJoinError){
          setNickError(UI_LANG==='en'
            ? "Could not connect. Check nickname and try again."
            : "Prisijungti nepavyko. Patikrink nick ir bandyk dar kartą.");
          scheduleNickCheck();
        }
        return;
      }

      if(fatalJoinError) return;

      pushToConvo("room:main", {type:"sys", t: tsLocal(), text: UI_LANG==='en' ? "connection lost, reconnect..." : "ryšys nutrūko, reconnect..."}, false);
      scheduleReconnect();
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
      pushToConvo(activeKey, {type:"sys", t: tsLocal(), text: T("noConn")}, false);
      return;
    }

    typing = false;
    setTyping(false);

    if(text.startsWith("/")){
      wsSend({type:"say", room:"main", text}); // komandas leidžiam iš bet kur
      msgEl.value = "";
      return;
    }

    if(c.kind === "room"){
      wsSend({type:"say", room:c.room, text});
    }else{
      wsSend({type:"say", room:"main", text:`/dm ${c.peer} ${text}`});
    }
    msgEl.value = "";
  }

  btn.onclick = send;
  msgEl.addEventListener("keydown", (e) => { if(e.key === "Enter") send(); });

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
      setNickError(T("nickInvalid"));
      return;
    }

    setNickError("");
    setNickState(T("nickChecking"));

    try{
      const qs = new URLSearchParams({ nick: n, lang: UI_LANG }).toString();
      const r = await fetch(`/check_nick?${qs}`, { cache: "no-store" });
      const j = await r.json();

      if(j && j.ok){
        nickAvailable = true;
        setNickState(T("nickFree"));
        setNickError("");
        joinBtn.disabled = false;
      }else{
        nickAvailable = false;
        setNickState("");
        setNickError((j && j.reason) ? j.reason : T("nickTaken"));
        joinBtn.disabled = true;

        const sug = await suggestNick(n);
        if(sug && sug.toLowerCase() !== n.toLowerCase()){
          showSuggest(sug);
        }
      }
    }catch{
      nickAvailable = false;
      setNickState("");
      setNickError(T("CHECK_FAIL"));
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

  applySuggestBtn.addEventListener("click", () => {
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
    const savedTheme = (localStorage.getItem("theme") || "theme-cyber").trim();
    setTheme(savedTheme);

    const savedNick = (localStorage.getItem("nick") || "").trim();
    if(savedNick) nickPick.value = savedNick;

    const savedLang = (localStorage.getItem("lang") || "lt").trim().toLowerCase();
    if(I18N[savedLang]) UI_LANG = savedLang;

    langPick.value = UI_LANG;
    langTop.value = UI_LANG;

    applyLangToUI();
    scheduleNickCheck();
    showLobby(true);
  })();
</script>
</body>
</html>
"""


# =====================================
# KEEPALIVE (Server ping)
# =====================================
async def heartbeat_sender(ws: WebSocket):
    """
    Periodiškai siunčia ping, kad proxy/naršyklė nenutrauktų WS ryšio dėl 'idle timeout'.
    """
    try:
        while True:
            await asyncio.sleep(25)  # kas 25 s
            await ws.send_json({"type": "ping", "t": ts()})
    except Exception:
        pass


# =====================================
# WEBSOCKET ENDPOINT
# =====================================
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    hb_task: Optional[asyncio.Task] = None

    nick = (ws.query_params.get("nick") or "").strip()
    lang = (ws.query_params.get("lang") or "lt").strip().lower()
    if lang not in LANGS:
        lang = "lt"

    await ws.accept()

    if not valid_nick(nick):
        await ws.send_json({"type": "error", "code": "BAD_NICK", "text": t(lang, "BAD_NICK")})
        await ws.close()
        return

    async with state_lock:
        if nick.casefold() in all_ws_by_nick_cf:
            await ws.send_json({"type": "error", "code": "NICK_TAKEN", "text": t(lang, "NICK_TAKEN")})
            await ws.close()
            return

        used = {u.color for u in all_users_by_ws.values()}
        color = alloc_color(used)

        u = User(nick=nick, color=color, lang=lang)
        all_users_by_ws[ws] = u
        all_ws_by_nick_cf[nick.casefold()] = ws

    for rk in DEFAULT_ROOMS.keys():
        ensure_room(rk)

    await join_room(ws, "#main")
    await broadcast_rooms_list(ws)
    await ws_send(ws, {"type": "topic", "room": "main", "text": f"{ensure_room('main').title} — {ensure_room('main').topic}"})
    await ws_send(ws, {"type": "users", "room": "main", "items": room_userlist("main")})
    await ws_send(ws, {"type": "me", "nick": u.nick, "color": u.color})
    await ws_send(ws, {"type": "lang", "lang": u.lang})

    # start keepalive after successful handshake/init
    hb_task = asyncio.create_task(heartbeat_sender(ws))

    try:
        while True:
            data = await ws.receive_json()

            # client pong (keepalive)
            if isinstance(data, dict) and data.get("type") == "pong":
                continue

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

            if data.get("type") == "say":
                room_key = norm_room(data.get("room", "main"))
                text = str(data.get("text", "")).strip()
                if not text:
                    continue

                ok, code = check_rate_limit(u, len(text))
                if not ok:
                    await ws_send(ws, {"type": "sys", "t": ts(), "text": t(u.lang, "RATE_LIMIT" if code == "rate" else "RATE_LIMIT_CHARS")})
                    continue

                if text.startswith("/"):
                    handled = await handle_command(ws, room_key, text)
                    if handled:
                        continue

                extra = {"created_at": time.time(), "edited": False, "reactions": {}}
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
                payload = {"type": "msg", "room": room_key, "id": msg_id, "t": msg["ts"], "nick": u.nick, "color": u.color, "text": msg["text"], "extra": extra}
                await room_broadcast(room_key, payload)
                continue

    except WebSocketDisconnect:
        pass
    finally:
        if hb_task is not None:
            try:
                hb_task.cancel()
            except Exception:
                pass
        await disconnect_ws(ws)
