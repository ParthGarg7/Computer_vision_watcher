"""
src/layer7_storage/store.py
─────────────────────────────────────────────────────────────────────────────
Layer 7: Storage — Persistence Layer

Stores every durable artefact the pipeline produces: session records,
expression event time-series, and presence events. Nothing upstream
persists data; everything downstream (Layer 8 API, dashboards) reads
from this layer.

Backend: SQLite (development substitute for PostgreSQL + TimescaleDB)
    The Layer 7 Architecture Doc (Section 6, MVP phase) explicitly allows
    SQLite to substitute for PostgreSQL in development, to be replaced
    before any concurrent write workload. The schema below mirrors the
    planned production layout so migration is a connection-string change:

        sessions           →  PostgreSQL `sessions` table
        expression_events  →  TimescaleDB `expression_events` hypertable
        presence_events    →  TimescaleDB `presence_events` hypertable

    FAISS embedding persistence (the fourth Layer 7 store) already lives in
    Layer 4's IdentityStore (faces/db/identity_store.*) — write_index /
    read_index per the Architecture Doc Section 4.3. Redis (active session
    cache) is optional at MVP and not used; Layer 6 keeps active-session
    state in-process instead.

Write path performance:
    Expression events arrive once per detected face per frame (~30-60/s).
    Writes are buffered and flushed with executemany() every BATCH_SIZE
    events (or on flush()/close()) so the frame loop never waits on a
    per-row transaction. WAL journal mode keeps readers non-blocking.

Ref: Layer 7 Architecture Doc — Sections 1, 2, 3, 4.1, 4.2, 6
"""

import json
import os
import sqlite3
from typing import Optional

# ─── Constants ────────────────────────────────────────────────────────────────

DEFAULT_DB_PATH = "data/watcher.db"

# Expression events buffered before an executemany() flush.
# 64 events ≈ 1-2 seconds of a single-face stream — bounded loss window
# on hard crash, negligible per-frame overhead.
BATCH_SIZE = 64

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id                       TEXT PRIMARY KEY,
    camera_id                        TEXT NOT NULL,
    track_id                         INTEGER NOT NULL,
    identity_label                   TEXT,
    session_start                    REAL NOT NULL,
    session_end                      REAL NOT NULL,
    presence_duration_seconds        REAL NOT NULL,
    frames_observed                  INTEGER NOT NULL,
    expression_trend                 TEXT,   -- JSON: class -> rolling mean
    dominant_expression_distribution TEXT    -- JSON: class -> frame count
);

CREATE TABLE IF NOT EXISTS expression_events (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp           REAL NOT NULL,
    camera_id           TEXT NOT NULL,
    frame_seq           INTEGER NOT NULL,
    track_id            INTEGER NOT NULL,
    identity_label      TEXT,
    dominant_expression TEXT NOT NULL,
    confidence          REAL NOT NULL,
    expression_scores   TEXT NOT NULL    -- JSON: class -> probability
);
CREATE INDEX IF NOT EXISTS idx_expr_ts    ON expression_events (timestamp);
CREATE INDEX IF NOT EXISTS idx_expr_track ON expression_events (track_id);

CREATE TABLE IF NOT EXISTS presence_events (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp      REAL NOT NULL,
    camera_id      TEXT NOT NULL,
    track_id       INTEGER NOT NULL,
    identity_label TEXT,
    event_type     TEXT NOT NULL    -- 'appeared' | 'departed'
);
CREATE INDEX IF NOT EXISTS idx_presence_ts ON presence_events (timestamp);
"""


class StorageLayer:
    """
    Layer 7 storage backed by SQLite.

    Thread safety: NOT thread-safe — call from the pipeline thread only
    (same contract as the rest of the pipeline).

    Usage
    -----
        store = StorageLayer()
        store.write_expression_event({...})   # buffered
        store.write_presence_event({...})     # immediate
        store.write_session({...})            # immediate
        store.close()                         # flush + close
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        """
        Parameters
        ----------
        db_path : str
            SQLite database file. Parent directory is created if missing.
        """
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        # WAL: writers don't block readers (Layer 8 API can query live).
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

        # Buffered expression-event rows awaiting executemany() flush
        self._event_buffer: list = []

        # Session-level write counters for the shutdown summary
        self.n_expression_events = 0
        self.n_presence_events = 0
        self.n_sessions = 0

        print(f"  [Layer7] Storage ready: {db_path} (SQLite WAL — dev "
              f"substitute for PostgreSQL/TimescaleDB per Arch Doc §6)")

    # ─── Write API (called by Layer 6) ────────────────────────────────────────

    def write_expression_event(self, record: dict):
        """
        Buffer one expression event record (one per detected face per frame).

        Expected keys: timestamp, camera_id, frame_seq, track_id,
        identity_label, dominant_expression, confidence, expression_scores.
        """
        self._event_buffer.append((
            record["timestamp"],
            record["camera_id"],
            record["frame_seq"],
            record["track_id"],
            record.get("identity_label"),
            record["dominant_expression"],
            record["confidence"],
            json.dumps(record["expression_scores"]),
        ))
        if len(self._event_buffer) >= BATCH_SIZE:
            self.flush()

    def write_presence_event(self, record: dict):
        """
        Write one presence event ('appeared' / 'departed') immediately.
        Presence events are rare (track lifecycle), so no buffering.
        """
        self._conn.execute(
            "INSERT INTO presence_events "
            "(timestamp, camera_id, track_id, identity_label, event_type) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                record["timestamp"],
                record["camera_id"],
                record["track_id"],
                record.get("identity_label"),
                record["event_type"],
            ),
        )
        self._conn.commit()
        self.n_presence_events += 1

    def write_session(self, record: dict):
        """
        Write one finalized session record immediately.

        Expected keys match the Layer 6 aggregated session metrics output:
        session_id, camera_id, track_id, identity_label, session_start,
        session_end, presence_duration_seconds, frames_observed,
        expression_trend, dominant_expression_distribution.
        """
        self._conn.execute(
            "INSERT OR REPLACE INTO sessions VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                record["session_id"],
                record["camera_id"],
                record["track_id"],
                record.get("identity_label"),
                record["session_start"],
                record["session_end"],
                record["presence_duration_seconds"],
                record["frames_observed"],
                json.dumps(record.get("expression_trend") or {}),
                json.dumps(record.get("dominant_expression_distribution") or {}),
            ),
        )
        self._conn.commit()
        self.n_sessions += 1

    def flush(self):
        """Flush buffered expression events with a single executemany()."""
        if not self._event_buffer:
            return
        self._conn.executemany(
            "INSERT INTO expression_events "
            "(timestamp, camera_id, frame_seq, track_id, identity_label, "
            " dominant_expression, confidence, expression_scores) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            self._event_buffer,
        )
        self._conn.commit()
        self.n_expression_events += len(self._event_buffer)
        self._event_buffer = []

    # ─── Read API (for Layer 8 / ad-hoc queries) ──────────────────────────────

    def recent_sessions(self, limit: int = 20) -> list:
        """Most recent finalized sessions, newest first, as dicts."""
        cur = self._conn.execute(
            "SELECT session_id, camera_id, track_id, identity_label, "
            "session_start, session_end, presence_duration_seconds, "
            "frames_observed, expression_trend, "
            "dominant_expression_distribution "
            "FROM sessions ORDER BY session_end DESC LIMIT ?",
            (limit,),
        )
        cols = [d[0] for d in cur.description]
        rows = []
        for r in cur.fetchall():
            row = dict(zip(cols, r))
            row["expression_trend"] = json.loads(row["expression_trend"] or "{}")
            row["dominant_expression_distribution"] = json.loads(
                row["dominant_expression_distribution"] or "{}")
            rows.append(row)
        return rows

    def expression_trend(
        self,
        track_id: int,
        bucket_seconds: float = 30.0,
        since: Optional[float] = None,
    ) -> list:
        """
        Time-bucketed dominant-expression summary for one track — the SQLite
        equivalent of TimescaleDB's time_bucket() aggregation (Arch Doc §4.2).

        Returns one row per (bucket, dominant_expression):
        bucket_start (epoch), dominant_expression, frames, avg_confidence.
        """
        since = since if since is not None else 0.0
        cur = self._conn.execute(
            "SELECT CAST(timestamp / ? AS INTEGER) * ? AS bucket_start, "
            "dominant_expression, COUNT(*) AS frames, "
            "AVG(confidence) AS avg_confidence "
            "FROM expression_events "
            "WHERE track_id = ? AND timestamp >= ? "
            "GROUP BY bucket_start, dominant_expression "
            "ORDER BY bucket_start",
            (bucket_seconds, bucket_seconds, track_id, since),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    # ─── Lifecycle ────────────────────────────────────────────────────────────

    def close(self):
        """Flush pending events and close the connection. Safe to call twice."""
        if self._conn is None:
            return
        try:
            self.flush()
        finally:
            self._conn.close()
            self._conn = None
