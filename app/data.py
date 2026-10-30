"""Data layer: pulls results and fixtures from ESPN's public soccer feed and caches them in SQLite."""
import os
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone

import requests

ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard"
DB_PATH = os.environ.get("DB_PATH", "data/football.db")
HISTORY_DAYS = int(os.environ.get("HISTORY_DAYS", 3 * 365))   # how far back the first sync goes
FUTURE_DAYS = 45                                                # how far ahead to pull fixtures
CHUNK_DAYS = 30
STALE_HOURS = 6

EUROPE_TOP5 = ["eng.1", "esp.1", "ita.1", "ger.1", "fra.1"]
AFRICA = ["caf.champions", "caf.confed", "rsa.1", "egy.1", "mar.1"]

# key: (display name, ESPN slug of the competition, slugs used to train the model)
COMPETITIONS = {
    "psl":        ("South Africa - Premiership", "rsa.1", ["rsa.1"]),
    "caf_cl":     ("CAF Champions League", "caf.champions", AFRICA),
    "caf_cc":     ("CAF Confederation Cup", "caf.confed", AFRICA),
    "epl":        ("England - Premier League", "eng.1", ["eng.1"]),
    "laliga":     ("Spain - La Liga", "esp.1", ["esp.1"]),
    "seriea":     ("Italy - Serie A", "ita.1", ["ita.1"]),
    "bundesliga": ("Germany - Bundesliga", "ger.1", ["ger.1"]),
    "ligue1":     ("France - Ligue 1", "fra.1", ["fra.1"]),
    "ucl":        ("UEFA Champions League", "uefa.champions", ["uefa.champions"] + EUROPE_TOP5),
    "mls":        ("USA - MLS", "usa.1", ["usa.1"]),
}

_lock = threading.Lock()
_session = requests.Session()
_session.headers["User-Agent"] = "Mozilla/5.0 (football-model)"


# ---------------------------------------------------------------- database
def db():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE IF NOT EXISTS matches (
        id TEXT, slug TEXT, date TEXT, home_id TEXT, home TEXT, away_id TEXT, away TEXT,
        hg INTEGER, ag INTEGER, state TEXT, completed INTEGER, neutral INTEGER,
        PRIMARY KEY (id, slug))""")
    con.execute("""CREATE TABLE IF NOT EXISTS sync (
        slug TEXT PRIMARY KEY, synced_at TEXT, covered_from TEXT, error TEXT)""")
    return con


# ---------------------------------------------------------------- ESPN parsing
def _score(c):
    s = c.get("score")
    if isinstance(s, dict):
        s = s.get("value", s.get("displayValue"))
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


def parse_event(ev, slug):
    """Turn one ESPN event into a row. Returns None if the shape is unexpected."""
    try:
        comp = ev["competitions"][0]
        cs = comp["competitors"]
        home = next(c for c in cs if c.get("homeAway") == "home")
        away = next(c for c in cs if c.get("homeAway") == "away")
        st = (ev.get("status") or comp.get("status") or {}).get("type", {})
        completed = bool(st.get("completed"))
        hg, ag = (_score(home), _score(away)) if completed else (None, None)
        if completed and (hg is None or ag is None):
            completed = False
        return {
            "id": str(ev["id"]), "slug": slug, "date": ev["date"],
            "home_id": str(home["team"]["id"]), "home": home["team"].get("displayName", "?"),
            "away_id": str(away["team"]["id"]), "away": away["team"].get("displayName", "?"),
            "hg": hg, "ag": ag, "state": st.get("state", ""), "completed": int(completed),
            "neutral": int(bool(comp.get("neutralSite"))),
        }
    except (KeyError, IndexError, StopIteration, TypeError):
        return None


class Rejected(Exception):
    """ESPN answered 400 Bad Request: the date window or parameters were not accepted."""


_span = {}          # slug -> largest date window (days) ESPN has accepted
_use_limit = {}     # slug -> whether the 'limit' parameter is accepted


def _get(slug, a, b):
    dates = f"{a:%Y%m%d}" if a.date() == b.date() else f"{a:%Y%m%d}-{b:%Y%m%d}"
    options = [True, False] if _use_limit.get(slug, True) else [False]
    for with_limit in options:
        params = {"dates": dates, "limit": 300} if with_limit else {"dates": dates}
        r = _session.get(ESPN_URL.format(slug=slug), params=params, timeout=20)
        if r.status_code == 400:
            continue
        r.raise_for_status()
        _use_limit[slug] = with_limit
        events = r.json().get("events", [])
        return [row for row in (parse_event(e, slug) for e in events) if row]
    raise Rejected(f"ESPN rejected dates={dates} for {slug}")


def fetch_range(slug, start, end):
    """Fetch [start, end], shrinking the window (30 -> 7 -> 1 day) whenever ESPN rejects it."""
    span = _span.get(slug, CHUNK_DAYS)
    rows, cur = [], start
    while cur <= end:
        nxt = min(cur + timedelta(days=span - 1), end)
        try:
            rows += _get(slug, cur, nxt)
            cur = nxt + timedelta(days=1)
            if span < CHUNK_DAYS:
                time.sleep(0.05)
        except Rejected:
            if span == 1:
                raise
            span = max(1, span // 4)
            _span[slug] = span
    return rows


# ---------------------------------------------------------------- sync
def sync_slug(con, slug, force=False):
    now = datetime.now(timezone.utc)
    row = con.execute("SELECT * FROM sync WHERE slug=?", (slug,)).fetchone()
    if row and row["synced_at"] and not force:
        age = now - datetime.fromisoformat(row["synced_at"])
        if age < timedelta(hours=STALE_HOURS):
            return {"slug": slug, "status": "fresh"}
    if row and row["covered_from"] and not row["error"]:
        start = now - timedelta(days=21)                 # refresh recent results only
    else:
        start = now - timedelta(days=HISTORY_DAYS)       # first sync: full history
    end = now + timedelta(days=FUTURE_DAYS)
    n, cur = 0, start
    try:
        while cur < end:
            nxt = min(cur + timedelta(days=CHUNK_DAYS - 1), end)
            rows = fetch_range(slug, cur, nxt)
            con.executemany("""INSERT OR REPLACE INTO matches VALUES
                (:id,:slug,:date,:home_id,:home,:away_id,:away,:hg,:ag,:state,:completed,:neutral)""", rows)
            n += len(rows)
            cur = nxt + timedelta(days=1)
            time.sleep(0.15)
        covered = row["covered_from"] if row and row["covered_from"] and not row["error"] else start.isoformat()
        con.execute("INSERT OR REPLACE INTO sync VALUES (?,?,?,NULL)", (slug, now.isoformat(), covered))
        con.commit()
        return {"slug": slug, "status": "ok", "events": n}
    except Exception as e:  # network error, unknown league slug, changed format...
        con.execute("INSERT OR REPLACE INTO sync VALUES (?,?,?,?)", (slug, None, None, str(e)[:300]))
        con.commit()
        return {"slug": slug, "status": "error", "error": str(e)[:300]}


def sync_competition(key, force=False):
    _, _, train = COMPETITIONS[key]
    with _lock:
        con = db()
        try:
            return [sync_slug(con, s, force) for s in train]
        finally:
            con.close()


# ---------------------------------------------------------------- queries
def _to_dt(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def training_matches(key, before=None):
    _, _, train = COMPETITIONS[key]
    con = db()
    q = f"SELECT * FROM matches WHERE completed=1 AND slug IN ({','.join('?' * len(train))})"
    rows = [dict(r) for r in con.execute(q, train)]
    con.close()
    seen, out = set(), []
    for r in rows:
        if r["id"] in seen:               # same match can appear under two slugs
            continue
        seen.add(r["id"])
        r["dt"] = _to_dt(r["date"])
        if before is None or r["dt"] < before:
            out.append(r)
    out.sort(key=lambda r: r["dt"])
    return out


def upcoming(key, days=14):
    _, slug, _ = COMPETITIONS[key]
    now = datetime.now(timezone.utc)
    con = db()
    rows = [dict(r) for r in con.execute(
        "SELECT * FROM matches WHERE slug=? AND completed=0 AND state IN ('pre','')", (slug,))]
    con.close()
    out = []
    for r in rows:
        r["dt"] = _to_dt(r["date"])
        if now - timedelta(hours=3) <= r["dt"] <= now + timedelta(days=days):
            out.append(r)
    out.sort(key=lambda r: r["dt"])
    return out


def recent_results(key, days=180):
    _, slug, _ = COMPETITIONS[key]
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    return [m for m in training_matches(key) if m["slug"] == slug and m["dt"] >= cutoff]


def sync_status():
    con = db()
    rows = {r["slug"]: dict(r) for r in con.execute("SELECT * FROM sync")}
    counts = {r["slug"]: r["n"] for r in con.execute(
        "SELECT slug, COUNT(*) n FROM matches WHERE completed=1 GROUP BY slug")}
    con.close()
    return rows, counts
