"""
src/layer8_api/api.py
─────────────────────────────────────────────────────────────────────────────
Layer 8: REST API (FastAPI) — basic structure

The interface between the pipeline's stored results and the frontend
dashboard (Layer 9). This is the READ side of the system: everything the
pipeline has persisted becomes queryable JSON, plus interactive docs at
/docs (FastAPI generates OpenAPI automatically).

── Endpoints ────────────────────────────────────────────────────────────────

    GET  /                                        dashboard (Layer 9 bundle)
    GET  /api/health                              liveness + storage status
    GET  /api/stats                               per-person totals
    GET  /api/sessions?limit=N                    recent sessions + appearances
    GET  /api/sessions/{session_id}               one session in full
    GET  /api/sessions/{session_id}/expression_trend?bucket=S
                                                  time-bucketed mood/expression
    GET  /api/identities                          registered people (FAISS meta)
    GET  /api/identities/{label}/history          one person across sessions
    GET  /api/events/recent?limit=N               latest expression readings
    POST /api/identities/register                 enrol a face from an upload
    WS   /ws/live                                 real-time Layer 6 events

The WebSocket forwards the events Layer 6 already emits
(live_expression_update / presence_alert / threshold_alert). Because the
pipeline and this server are separate processes, those events travel through
the `live_events` table in the shared WAL database rather than an in-process
queue — see StorageLayer.publish_live_event for why, and what it trades off.

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

import asyncio
import json
import os
import sqlite3

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect

from src.core.logger import get_logger
from src.layer4_identity.identity_store import DEFAULT_STORE_PATH
from src.layer7_storage.store import DEFAULT_DB_PATH

log = get_logger("watcher.layer8")

_L9 = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "layer9_dashboard")
# The built React bundle. Falls back to a message if the dashboard hasn't
# been built yet (npm run build inside src/layer9_dashboard).
_DIST = os.path.join(_L9, "dist")
_DASHBOARD = os.path.join(_DIST, "index.html")

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
    """Serve the built Layer 9 dashboard."""
    if not os.path.exists(_DASHBOARD):
        raise HTTPException(
            status_code=503,
            detail="Dashboard not built. Run: cd src/layer9_dashboard && "
                   "npm install && npm run build")
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
    Latest expression readings, newest first. Also the dashboard's fallback
    when the live WebSocket cannot connect (pipeline not running).
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


# ─── Live channel ─────────────────────────────────────────────────────────────

# How often the server checks the live bus for new events. The pipeline
# writes them from another process, so this is a poll rather than a push;
# 250ms is well under the eye's tolerance for a status dashboard.
LIVE_POLL_SEC = 0.25


@app.websocket("/ws/live")
async def live_channel(ws: WebSocket):
    """
    Stream Layer 6's real-time events to a dashboard.

    Starts from "now" (the newest event id at connect time) rather than
    replaying the buffer, so a client that connects mid-run sees current
    activity instead of a burst of history.
    """
    await ws.accept()
    log.debug("[Layer8] live client connected")

    last_id = 0
    try:
        conn = _connect_ro()
        try:
            row = conn.execute("SELECT MAX(id) FROM live_events").fetchone()
            last_id = row[0] or 0
        finally:
            conn.close()
    except HTTPException:
        # No database yet — sit with the client and pick up when it appears.
        pass
    except sqlite3.Error:
        pass

    try:
        while True:
            try:
                conn = _connect_ro()
                try:
                    rows = conn.execute(
                        "SELECT id, payload FROM live_events WHERE id > ? "
                        "ORDER BY id LIMIT 200", (last_id,)).fetchall()
                finally:
                    conn.close()
                for row_id, payload in rows:
                    last_id = row_id
                    await ws.send_text(payload)
            except HTTPException:
                pass          # database not created yet; keep waiting
            except sqlite3.Error as e:
                log.debug(f"[Layer8] live poll skipped: {e}")
            await asyncio.sleep(LIVE_POLL_SEC)
    except WebSocketDisconnect:
        log.debug("[Layer8] live client disconnected")
    except Exception:
        log.exception("[Layer8] live channel failed")
        try:
            await ws.close()
        except Exception:
            pass


# ─── Registration ─────────────────────────────────────────────────────────────

@app.post("/api/identities/register")
async def register_identity(
    name: str = Form(...),
    image: UploadFile = File(...),
    person_id: str | None = Form(None),
):
    """
    Enrol a face into the Layer 4 registry from an uploaded photo.

    Runs the same InsightFace embedder the pipeline uses, at full-image
    detection size (the upload is a whole photo, not a crop). Pass an
    existing person_id to add another sample to that person instead of
    creating a new one — more samples means better recognition at angles.

    The heavy imports happen HERE, not at module import: the API server
    should start in a second and serve history without loading ~340 MB of
    face models that most requests never need.
    """
    import cv2
    import numpy as np

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")

    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400,
                            detail="could not decode image (jpg/png expected)")

    try:
        from src.layer4_identity.embedder import FaceEmbedder, FULL_IMAGE_DET_SIZE
        from src.layer4_identity.identity_store import IdentityStore
    except Exception as e:
        raise HTTPException(status_code=503,
                            detail=f"face models unavailable: {e}")

    global _EMBEDDER
    if _EMBEDDER is None:
        log.info("  [Layer8] loading InsightFace for registration…")
        _EMBEDDER = FaceEmbedder(det_size=FULL_IMAGE_DET_SIZE)

    embedding, _ = _EMBEDDER.get_embedding(frame)
    if embedding is None:
        raise HTTPException(
            status_code=422,
            detail="no face found in that image — try a clearer, "
                   "front-facing photo with good lighting")

    # Registration WRITES, so it needs its own read-write store rather than
    # the read-only connections the query endpoints use. clear_unregistered
    # is not involved: this is the FAISS registry, not the session database.
    store = IdentityStore(store_path=app.state.faces_meta.replace(".meta.json", ""))
    pid = store.register(embedding=embedding, person_id=person_id,
                         name=name if not person_id else None)
    store.save()
    log.info(f"  [Layer8] registered '{name}' -> {pid}")
    return {"person_id": pid, "name": name,
            "samples": store.get_person(pid)["embedding_count"]}


_EMBEDDER = None


# ─── Static assets (mounted last so /api and /ws win) ────────────────────────

if os.path.isdir(_DIST):
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="dashboard")
