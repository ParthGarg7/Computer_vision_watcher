"""
src/layer7_storage/store.py
─────────────────────────────────────────────────────────────────────────────
Layer 7: Storage — Persistence Layer

Stores every durable artefact the pipeline produces. Nothing upstream
persists data; everything downstream (Layer 8 API, dashboards) reads here.

── Data model: sessions and appearances ─────────────────────────────────────

A SESSION is one person, for one run of the program. It opens the first time
that person is seen and closes when the pipeline shuts down.

An APPEARANCE is one continuous stretch of that person being on screen. A
session contains many appearances — step away for tea and come back, and you
get a second appearance inside the SAME session.

    Session: Parth, program ran 15:00 -> 16:00, present 25 min, 3 appearances
        ├── appearance  15:00 – 15:05
        ├── appearance  15:10 – 15:30
        └── appearance  15:45 – 15:50

This separates two numbers that are easy to confuse:
    session_start..session_end  — the span from first to last sighting
    total_present_seconds       — time actually on screen (sum of appearances)

A session row can always be rebuilt by summing its appearances, so a crash
before the final update loses nothing that matters.

── Unregistered people ──────────────────────────────────────────────────────

Sessions are recorded for EVERY confirmed person, registered or not.
Unregistered people have person_id = NULL and are identified only by
track_id, which is meaningful solely within one run (track numbering restarts
each time the program does).

Because of that, data for unregistered people is DELETED at startup
(clear_unregistered). Their records live exactly as long as the run they
belong to. Registered people — anyone in the FAISS registry — keep their
full history across runs.

── Backend ──────────────────────────────────────────────────────────────────

SQLite, the development substitute for PostgreSQL + TimescaleDB that the
Layer 7 Architecture Doc (§6, MVP phase) explicitly permits. The schema
mirrors the planned production layout so migration is a connection-string
change:

    sessions / appearances  →  PostgreSQL tables
    expression_events       →  TimescaleDB hypertable
    presence_events         →  TimescaleDB hypertable

FAISS embedding persistence (the other Layer 7 store) lives in Layer 4's
IdentityStore (faces/db/identity_store.*).

Write path: expression events arrive once per measurement per face and are
buffered, then flushed with executemany() so the frame loop never waits on a
transaction. WAL journal mode keeps readers non-blocking.

Ref: Layer 7 Architecture Doc — Sections 1, 2, 3, 4.1, 4.2, 6
"""

import json
import os
import sqlite3
import time
from typing import Optional

# ─── Constants ────────────────────────────────────────────────────────────────

DEFAULT_DB_PATH = "data/watcher.db"

# Expression events buffered before an executemany() flush.
BATCH_SIZE = 64

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id                       TEXT PRIMARY KEY,
    run_id                           TEXT NOT NULL,
    camera_id                        TEXT NOT NULL,
    track_id                         INTEGER NOT NULL,
    person_id                        TEXT,    -- FAISS UUID; NULL = unregistered
    identity_label                   TEXT,    -- display name; NULL if unknown
    session_start                    REAL NOT NULL,
    session_end                      REAL,    -- NULL while the session is open
    total_present_seconds            REAL DEFAULT 0,
    appearance_count                 INTEGER DEFAULT 0,
    frames_observed                  INTEGER DEFAULT 0,
    expression_trend                 TEXT,    -- JSON: class -> rolling mean
    dominant_expression_distribution TEXT     -- JSON: class -> measurement count
);
CREATE INDEX IF NOT EXISTS idx_sess_person ON sessions (person_id);
CREATE INDEX IF NOT EXISTS idx_sess_run    ON sessions (run_id);

CREATE TABLE IF NOT EXISTS appearances (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       TEXT NOT NULL,
    track_id         INTEGER NOT NULL,
    started          REAL NOT NULL,
    ended            REAL NOT NULL,
    duration_seconds REAL NOT NULL,
    frames_observed  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_app_session ON appearances (session_id);

CREATE TABLE IF NOT EXISTS expression_events (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id          TEXT,
    timestamp           REAL NOT NULL,
    camera_id           TEXT NOT NULL,
    frame_seq           INTEGER NOT NULL,
    track_id            INTEGER NOT NULL,
    identity_label      TEXT,
    dominant_expression TEXT NOT NULL,
    confidence          REAL NOT NULL,
    expression_scores   TEXT NOT NULL    -- JSON: class -> probability
);
CREATE INDEX IF NOT EXISTS idx_expr_ts      ON expression_events (timestamp);
CREATE INDEX IF NOT EXISTS idx_expr_track   ON expression_events (track_id);
CREATE INDEX IF NOT EXISTS idx_expr_session ON expression_events (session_id);

CREATE TABLE IF NOT EXISTS presence_events (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id     TEXT,
    timestamp      REAL NOT NULL,
    camera_id      TEXT NOT NULL,
    track_id       INTEGER NOT NULL,
    identity_label TEXT,
    event_type     TEXT NOT NULL    -- 'appeared' | 'departed'
);
CREATE INDEX IF NOT EXISTS idx_presence_ts      ON presence_events (timestamp);
CREATE INDEX IF NOT EXISTS idx_presence_session ON presence_events (session_id);
"""


class StorageLayer:
    """
    Layer 7 storage backed by SQLite.

    Thread safety: NOT thread-safe — call from the pipeline thread only.

    Usage
    -----
        store = StorageLayer()               # clears last run's stranger data
        store.open_session({...})            # when a person is first seen
        store.write_appearance({...})        # each time they leave the frame
        store.write_expression_event({...})  # buffered
        store.write_presence_event({...})
        store.close_session({...})           # on shutdown
        store.close()
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH,
                 clear_unregistered: bool = True):
        """
        Parameters
        ----------
        db_path : str
            SQLite database file. Parent directory is created if missing.
        clear_unregistered : bool
            Delete all data belonging to unregistered people (person_id IS
            NULL) left over from previous runs. Their track numbers are
            meaningless across runs, so the records cannot be interpreted
            later. Set False for tests or forensic inspection.
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

        # Identifies this run of the program. Every session belongs to one.
        self.run_id = time.strftime("%Y%m%d_%H%M%S")

        self._event_buffer: list = []

        # Write counters for the shutdown summary
        self.n_sessions = 0
        self.n_appearances = 0
        self.n_expression_events = 0
        self.n_presence_events = 0

        removed = self.clear_unregistered() if clear_unregistered else 0

        print(f"  [Layer7] Storage ready: {db_path} (SQLite WAL — dev "
              f"substitute for PostgreSQL/TimescaleDB per Arch Doc §6)")
        print(f"  [Layer7] Run id: {self.run_id}"
              + (f" | cleared {removed} rows from unregistered people "
                 f"in previous runs" if removed else ""))

    # ─── Housekeeping ─────────────────────────────────────────────────────────

    def clear_unregistered(self) -> int:
        """
        Delete every record belonging to a person who was never recognised.

        Track numbers restart with each run, so an unregistered person's rows
        cannot be attributed to anyone once their run has ended — keeping them
        stores behavioural data about strangers that can never be interpreted.
        Rows are removed via their session link, so a registered person's
        early events (recorded before FAISS identified them, hence with a NULL
        identity_label) are correctly retained.

        Returns the number of rows deleted.
        """
        c = self._conn
        sub = "SELECT session_id FROM sessions WHERE person_id IS NULL"
        total = 0
        for table in ("expression_events", "presence_events", "appearances"):
            cur = c.execute(f"DELETE FROM {table} WHERE session_id IN ({sub})")
            total += cur.rowcount if cur.rowcount > 0 else 0
        cur = c.execute("DELETE FROM sessions WHERE person_id IS NULL")
        total += cur.rowcount if cur.rowcount > 0 else 0
        c.commit()
        return total

    # ─── Write API (called by Layer 6) ────────────────────────────────────────

    def open_session(self, record: dict):
        """
        Insert a session row the moment a person is first seen this run.

        Written up front (rather than at shutdown) so appearances always have
        a parent row and a crash cannot orphan them. close_session() fills in
        the totals later.

        Keys: session_id, camera_id, track_id, person_id, identity_label,
              session_start.
        """
        self._conn.execute(
            "INSERT OR REPLACE INTO sessions "
            "(session_id, run_id, camera_id, track_id, person_id, "
            " identity_label, session_start) VALUES (?,?,?,?,?,?,?)",
            (record["session_id"], self.run_id, record["camera_id"],
             record["track_id"], record.get("person_id"),
             record.get("identity_label"), record["session_start"]),
        )
        self._conn.commit()
        self.n_sessions += 1

    def close_session(self, record: dict):
        """
        Fill in a session's totals when the pipeline shuts down.

        Keys: session_id, session_end, total_present_seconds,
              appearance_count, frames_observed, expression_trend,
              dominant_expression_distribution, person_id, identity_label.
        """
        self._conn.execute(
            "UPDATE sessions SET session_end = ?, total_present_seconds = ?, "
            "appearance_count = ?, frames_observed = ?, expression_trend = ?, "
            "dominant_expression_distribution = ?, person_id = ?, "
            "identity_label = ? WHERE session_id = ?",
            (record["session_end"], record["total_present_seconds"],
             record["appearance_count"], record["frames_observed"],
             json.dumps(record.get("expression_trend") or {}),
             json.dumps(record.get("dominant_expression_distribution") or {}),
             record.get("person_id"), record.get("identity_label"),
             record["session_id"]),
        )
        self._conn.commit()

    def write_appearance(self, record: dict):
        """
        Write one appearance — a continuous stretch of screen time — when the
        person leaves the frame. Rare enough that no buffering is needed.

        Keys: session_id, track_id, started, ended, duration_seconds,
              frames_observed.
        """
        self._conn.execute(
            "INSERT INTO appearances "
            "(session_id, track_id, started, ended, duration_seconds, "
            " frames_observed) VALUES (?,?,?,?,?,?)",
            (record["session_id"], record["track_id"], record["started"],
             record["ended"], record["duration_seconds"],
             record["frames_observed"]),
        )
        self._conn.commit()
        self.n_appearances += 1

    def write_expression_event(self, record: dict):
        """
        Buffer one expression measurement. Only FRESH readings should reach
        here — Layer 5 carries labels forward between measurements and those
        must not be recorded as new data points.
        """
        self._event_buffer.append((
            record.get("session_id"),
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
        """Write one 'appeared' / 'departed' event immediately."""
        self._conn.execute(
            "INSERT INTO presence_events "
            "(session_id, timestamp, camera_id, track_id, identity_label, "
            " event_type) VALUES (?,?,?,?,?,?)",
            (record.get("session_id"), record["timestamp"],
             record["camera_id"], record["track_id"],
             record.get("identity_label"), record["event_type"]),
        )
        self._conn.commit()
        self.n_presence_events += 1

    def flush(self):
        """Flush buffered expression events with a single executemany()."""
        if not self._event_buffer:
            return
        self._conn.executemany(
            "INSERT INTO expression_events "
            "(session_id, timestamp, camera_id, frame_seq, track_id, "
            " identity_label, dominant_expression, confidence, "
            " expression_scores) VALUES (?,?,?,?,?,?,?,?,?)",
            self._event_buffer,
        )
        self._conn.commit()
        self.n_expression_events += len(self._event_buffer)
        self._event_buffer = []

    # ─── Read API (for Layer 8 / ad-hoc queries) ──────────────────────────────

    def recent_sessions(self, limit: int = 20) -> list:
        """Most recent sessions, newest first, each with its appearances."""
        cur = self._conn.execute(
            "SELECT session_id, run_id, camera_id, track_id, person_id, "
            "identity_label, session_start, session_end, "
            "total_present_seconds, appearance_count, frames_observed, "
            "expression_trend, dominant_expression_distribution "
            "FROM sessions ORDER BY session_start DESC LIMIT ?", (limit,))
        cols = [d[0] for d in cur.description]
        rows = []
        for r in cur.fetchall():
            row = dict(zip(cols, r))
            row["expression_trend"] = json.loads(row["expression_trend"] or "{}")
            row["dominant_expression_distribution"] = json.loads(
                row["dominant_expression_distribution"] or "{}")
            row["appearances"] = self.appearances_for(row["session_id"])
            rows.append(row)
        return rows

    def appearances_for(self, session_id: str) -> list:
        """Every appearance belonging to one session, oldest first."""
        cur = self._conn.execute(
            "SELECT id, session_id, track_id, started, ended, "
            "duration_seconds, frames_observed FROM appearances "
            "WHERE session_id = ? ORDER BY started", (session_id,))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def expression_trend(
        self,
        track_id: int,
        bucket_seconds: float = 30.0,
        since: Optional[float] = None,
    ) -> list:
        """
        Time-bucketed dominant-expression summary for one track — the SQLite
        equivalent of TimescaleDB's time_bucket() aggregation (Arch Doc §4.2).
        """
        since = since if since is not None else 0.0
        cur = self._conn.execute(
            "SELECT CAST(timestamp / ? AS INTEGER) * ? AS bucket_start, "
            "dominant_expression, COUNT(*) AS measurements, "
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
