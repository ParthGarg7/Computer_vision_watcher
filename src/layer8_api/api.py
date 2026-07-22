"""
src/layer8_api/api.py
─────────────────────────────────────────────────────────────────────────────
Layer 8: REST API (FastAPI) — basic structure

The interface between the pipeline's stored results and the frontend
dashboard (Layer 9). This is the READ side of the system: everything the
pipeline has persisted becomes queryable JSON, plus interactive docs at
/docs (FastAPI generates OpenAPI automatically).

── Scope of this scaffold ───────────────────────────────────────────────────

IN (working now):
    GET  /                                        dashboard (Layer 9)
    GET  /api/health                              liveness + storage status
    GET  /api/stats                               per-person totals
    GET  /api/sessions?limit=N                    recent sessions + appearances
    GET  /api/sessions/{session_id}               one session in full
    GET  /api/sessions/{session_id}/expression_trend?bucket=S
                                                  time-bucketed mood/expression
    GET  /api/identities                          registered people (FAISS meta)
    GET  /api/identities/{label}/history          one person across sessions
    GET  /api/events/recent?limit=N               latest expression readings

OUT (integration phase, once the frontend design lands):
    WS   /live            — pushes the live_expression_update / presence_alert
                            / threshold_alert events Layer 6 already emits.
                            Needs an event bus between the pipeline process
                            and this server (Arch Doc §2: Redis pub/sub at
                            MVP), so it arrives with the integration work.
    POST /identities/register — runs InsightFace on an uploaded image; needs
                            upload handling + the embedder in this process.

── Design decisions ─────────────────────────────────────────────────────────

READ-ONLY database access, one connection per request, opened with SQLite's
`mode=ro`. Two reasons:
  1. StorageLayer's constructor deletes unregistered people's data (by
     design, at pipeline startup) — an API server must NEVER trigger that.
  2. The pipeline writes in WAL mode, so read-only connections can query
     live while a run is in progress without blocking it.

Paths are configurable via app.state (tests point them at temp files):
    app.state.db_path     — Layer 7 SQLite database
    app.state.faces_meta  — Layer 4 identity registry sidecar (names only —
                            embeddings are never exposed over the API)

Ref: Layer 8 Architecture Doc — Sections 1, 2, 3, 4.1
"""

import json
import os
import sqlite3

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from src.core.logger import get_logger
from src.layer4_identity.identity_store import DEFAULT_STORE_PATH
from src.layer7_storage.store import DEFAULT_DB_PATH

log = get_logger("watcher.layer8")

_DASHBOARD = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "layer9_dashboard", "index.html")

app = FastAPI(
    title="The Watcher — API",
    description="Read API over the pipeline's stored sessions, appearances, "
                "expression events and registered identities. Live WebSocket "
                "and registration endpoints arrive with dashboard integration.",
    version="0.6.0",
)
app.state.db_path = DEFAULT_DB_PATH
app.state.faces_meta = DEFAULT_STORE_PATH + ".meta.json"


# ─── Read-only database access ────────────────────────────────────────────────

def _connect_ro() -> sqlite3.Connection:
    """
    Open the Layer 7 database READ-ONLY. Raises 503 if it doesn't exist —
    the pipeline simply hasn't produced data yet.
    """
    path = app.state.db_path
    if not os.path.exists(path):
        raise HTTPException(
            status_code=503,
            detail=f"No database at {path} — run the pipeline first "
                   f"(python main.py).")
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _rows(conn, sql, args=()) -> list:
    return [dict(r) for r in conn.execute(sql, args).fetchall()]


def _load_json_fields(row: dict, *fields) -> dict:
    for f in fields:
        if row.get(f):
            row[f] = json.loads(row[f])
        else:
            row[f] = {}
    return row


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def dashboard():
    """Serve the Layer 9 dashboard."""
    if not os.path.exists(_DASHBOARD):
        raise HTTPException(status_code=404, detail="dashboard not found")
    return FileResponse(_DASHBOARD, media_type="text/html")


@app.get("/api/health")
def health():
    """Liveness + storage status — the first thing a dashboard should call."""
    db_exists = os.path.exists(app.state.db_path)
    counts = {}
    if db_exists:
        conn = _connect_ro()
        try:
            for t in ("sessions", "appearances", "expression_events",
                      "presence_events"):
                counts[t] = conn.execute(
                    f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        finally:
            conn.close()
    return {
        "status": "ok",
        "database": {"path": app.state.db_path, "exists": db_exists,
                     "rows": counts},
        "registry": {"path": app.state.faces_meta,
                     "exists": os.path.exists(app.state.faces_meta)},
    }


@app.get("/api/stats")
def stats():
    """Per-person totals across all recorded sessions."""
    conn = _connect_ro()
    try:
        people = _rows(conn, """
            SELECT COALESCE(identity_label, '(unidentified)') AS person,
                   COUNT(*) AS sessions,
                   SUM(appearance_count)      AS appearances,
                   SUM(total_present_seconds) AS present_seconds,
                   SUM(frames_observed)       AS frames
            FROM sessions GROUP BY person
            ORDER BY present_seconds DESC""")
        span = conn.execute("SELECT MIN(session_start), MAX(session_end) "
                            "FROM sessions").fetchone()
        return {"people": people,
                "data_range": {"first": span[0], "last": span[1]}}
    finally:
        conn.close()


@app.get("/api/sessions")
def sessions(limit: int = 20):
    """Most recent sessions, newest first, each with its appearances."""
    conn = _connect_ro()
    try:
        out = _rows(conn, "SELECT * FROM sessions "
                          "ORDER BY session_start DESC LIMIT ?", (limit,))
        for s in out:
            _load_json_fields(s, "expression_trend",
                              "dominant_expression_distribution")
            s["appearances"] = _rows(
                conn, "SELECT started, ended, duration_seconds, "
                      "frames_observed FROM appearances "
                      "WHERE session_id = ? ORDER BY started",
                (s["session_id"],))
        return {"sessions": out}
    finally:
        conn.close()


@app.get("/api/sessions/{session_id}")
def session_detail(session_id: str):
    """One session in full — summary, appearances, presence log."""
    conn = _connect_ro()
    try:
        rows = _rows(conn, "SELECT * FROM sessions WHERE session_id = ?",
                     (session_id,))
        if not rows:
            raise HTTPException(status_code=404,
                                detail=f"no session '{session_id}'")
        s = _load_json_fields(rows[0], "expression_trend",
                              "dominant_expression_distribution")
        s["appearances"] = _rows(
            conn, "SELECT started, ended, duration_seconds, frames_observed "
                  "FROM appearances WHERE session_id = ? ORDER BY started",
            (session_id,))
        s["presence_events"] = _rows(
            conn, "SELECT timestamp, event_type FROM presence_events "
                  "WHERE session_id = ? ORDER BY timestamp", (session_id,))
        return s
    finally:
        conn.close()


@app.get("/api/sessions/{session_id}/expression_trend")
def session_expression_trend(session_id: str, bucket: float = 30.0):
    """
    Time-bucketed expression summary for one session — the SQLite stand-in
    for TimescaleDB's time_bucket() (Arch Doc §4.2). One row per
    (bucket, dominant_expression) with measurement count, mean confidence,
    and mean valence/arousal.
    """
    conn = _connect_ro()
    try:
        rows = _rows(conn, """
            SELECT CAST(timestamp / ? AS INTEGER) * ? AS bucket_start,
                   dominant_expression,
                   COUNT(*)        AS measurements,
                   AVG(confidence) AS avg_confidence,
                   AVG(valence)    AS avg_valence,
                   AVG(arousal)    AS avg_arousal
            FROM expression_events WHERE session_id = ?
            GROUP BY bucket_start, dominant_expression
            ORDER BY bucket_start""", (bucket, bucket, session_id))
        return {"session_id": session_id, "bucket_seconds": bucket,
                "trend": rows}
    finally:
        conn.close()


@app.get("/api/identities")
def identities():
    """
    Registered people, from the Layer 4 registry's metadata sidecar.
    Names and sample counts ONLY — embeddings are biometric data and are
    never exposed over the API.
    """
    path = app.state.faces_meta
    if not os.path.exists(path):
        return {"identities": []}
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return {"identities": [
        {"person_id": pid, "name": info.get("name"),
         "samples": info.get("count", 0)}
        for pid, info in payload.get("meta", {}).items()
    ]}


@app.get("/api/identities/{label}/history")
def identity_history(label: str):
    """Every recorded session for one person, by display name."""
    conn = _connect_ro()
    try:
        out = _rows(conn, "SELECT * FROM sessions WHERE identity_label = ? "
                          "ORDER BY session_start DESC", (label,))
        if not out:
            raise HTTPException(status_code=404,
                                detail=f"no sessions for '{label}'")
        total = sum(s["total_present_seconds"] or 0 for s in out)
        for s in out:
            _load_json_fields(s, "expression_trend",
                              "dominant_expression_distribution")
        return {"identity": label, "sessions": len(out),
                "total_present_seconds": round(total, 1), "history": out}
    finally:
        conn.close()


@app.get("/api/events/recent")
def recent_events(limit: int = 50):
    """
    Latest expression readings, newest first. The dashboard polls this
    until the /live WebSocket lands with the integration work.
    """
    conn = _connect_ro()
    try:
        rows = _rows(conn, "SELECT timestamp, track_id, identity_label, "
                           "dominant_expression, confidence, valence, "
                           "arousal, mood FROM expression_events "
                           "ORDER BY id DESC LIMIT ?", (limit,))
        return {"events": rows}
    finally:
        conn.close()
