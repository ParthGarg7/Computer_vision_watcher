"""
src/layer6_analytics/aggregator.py
─────────────────────────────────────────────────────────────────────────────
Layer 6: Analytics & Business Logic

Receives raw per-frame event records from Layer 5 and turns them into
business-meaningful metrics. Raw AI outputs — track IDs, expression
probability dicts, timestamps — are too low-level for any user or business
decision; this layer aggregates and contextualises them.

── Sessions and appearances ─────────────────────────────────────────────────

A SESSION is one person, for one run of the program. It opens the first time
that person is seen and closes when the pipeline shuts down.

An APPEARANCE is one continuous stretch of being on screen. Stepping away and
coming back produces a second appearance inside the SAME session — the person
is not counted as a new visitor.

    Session: Parth, program ran 15:00 -> 16:00, present 25 min, 3 appearances
        ├── appearance  15:00 – 15:05
        ├── appearance  15:10 – 15:30
        └── appearance  15:45 – 15:50

An appearance ends after APPEARANCE_TIMEOUT_SEC without a sighting. That grace
period absorbs the ordinary gaps — a turned head, someone walking past — that
would otherwise shatter one visit into dozens of fragments.

Sessions are keyed by track_id, which Layer 4 guarantees is stable for a whole
run: a registered person's display ID is pinned to their FAISS identity, so
leaving and returning restores the same number. person_id and identity_label
are recorded on the session as soon as recognition supplies them.

── Outputs (Arch Doc §3) ────────────────────────────────────────────────────

1. Session + appearance records, written to Layer 7 Storage.
   Expression aggregation consumes only FRESH readings (see
   Detection.expression_is_fresh): Layer 5 throttles inference and carries
   labels forward in between, and counting those carry-forwards would
   multiply every real measurement by the throttle factor.
   dominant_expression_distribution therefore counts measurements, not
   frames — ratios identical, magnitudes ~N x smaller.

2. Real-time events returned to the caller each frame (main.py prints them;
   Layer 8 will push them over WebSocket):
       presence_alert         — a person appeared or left the frame
       threshold_alert        — negative-expression ratio over threshold
       live_expression_update — latest scores for every active person

Implementation note — stdlib accumulators instead of Pandas:
    The Arch Doc (§4.1) recommends Pandas for MVP aggregation. The per-frame
    hot path uses plain dicts/deques instead: the streaming aggregations
    needed (running sums, rolling means, counters) are O(1) per event without
    materialising a DataFrame 30x per second. Pandas remains the right tool
    for OFFLINE analysis of the Layer 7 tables (pd.read_sql on watcher.db).

Ref: Layer 6 Architecture Doc — Sections 1, 2, 3, 4.1, 6, 7
"""

from collections import deque

from src.core.frame_context import FrameContext

from src.core.logger import get_logger

log = get_logger("watcher.layer6")

# ─── Constants ────────────────────────────────────────────────────────────────

# Seconds without a sighting before the current appearance is closed.
# Absorbs brief occlusions; does NOT end the session.
APPEARANCE_TIMEOUT_SEC = 5.0

# Rolling window for the expression trend (Arch Doc §1: 30-second window).
TREND_WINDOW_SEC = 30.0

# Expression classes counted as "negative" for the threshold alert.
NEGATIVE_CLASSES = frozenset({"anger", "disgust", "fear", "sadness", "contempt"})

# threshold_alert fires when the rolling mean of negative-class probability
# mass exceeds this ratio. Arch Doc §7: alert rules must be documented and
# reviewed — this default is deliberately conservative.
NEGATIVE_RATIO_THRESHOLD = 0.60

# Minimum seconds between repeated threshold alerts for the same person.
ALERT_COOLDOWN_SEC = 30.0

# live_expression_update events are emitted every N processed frames.
LIVE_UPDATE_EVERY_N_FRAMES = 15


class _PersonSession:
    """One person's session for this run, plus their in-progress appearance."""

    __slots__ = (
        "session_id", "camera_id", "track_id", "person_id", "identity_label",
        "session_start", "last_seen", "frames_observed",
        "score_window", "score_sums", "dominant_counts", "last_scores",
        "last_threshold_alert",
        "appearance_start", "appearance_frames", "appearance_count",
        "total_present_seconds",
    )

    def __init__(self, camera_id: str, track_id: int, now: float, run_id: str):
        self.session_id = f"{camera_id}:{run_id}:{track_id}"
        self.camera_id = camera_id
        self.track_id = track_id
        self.person_id = None
        self.identity_label = None
        self.session_start = now
        self.last_seen = now
        self.frames_observed = 0

        # (timestamp, scores_dict) pairs inside the trend window
        self.score_window: deque = deque()
        # Running per-class sums over score_window, maintained incrementally
        # on append/evict — trend() is O(classes) instead of O(window).
        self.score_sums: dict = {}
        # Counts of MEASUREMENTS (fresh inferences), not frames.
        self.dominant_counts: dict = {}
        self.last_scores: dict = {}
        self.last_threshold_alert = 0.0

        # Current appearance (None when the person is off screen)
        self.appearance_start = None
        self.appearance_frames = 0
        self.appearance_count = 0
        self.total_present_seconds = 0.0

    # ── appearance lifecycle ──────────────────────────────────────────────
    @property
    def is_present(self) -> bool:
        return self.appearance_start is not None

    def begin_appearance(self, now: float):
        self.appearance_start = now
        self.appearance_frames = 0

    def end_appearance(self, now: float) -> dict:
        """Close the current appearance and return its record."""
        started = self.appearance_start
        duration = max(0.0, now - started)
        self.appearance_start = None
        self.appearance_count += 1
        self.total_present_seconds += duration
        record = {
            "session_id": self.session_id,
            "track_id": self.track_id,
            "started": started,
            "ended": now,
            "duration_seconds": round(duration, 3),
            "frames_observed": self.appearance_frames,
        }
        self.appearance_frames = 0
        return record

    # ── expression accumulation ───────────────────────────────────────────
    def add_scores(self, now: float, scores: dict, window_sec: float):
        """Append one scores sample and evict entries older than window_sec."""
        self.score_window.append((now, scores))
        for cls, p in scores.items():
            self.score_sums[cls] = self.score_sums.get(cls, 0.0) + p
        cutoff = now - window_sec
        while self.score_window and self.score_window[0][0] < cutoff:
            _, old = self.score_window.popleft()
            for cls, p in old.items():
                self.score_sums[cls] -= p

    def trend(self) -> dict:
        """Rolling mean of each expression class over the current window."""
        n = len(self.score_window)
        if n == 0:
            return {}
        return {cls: max(s, 0.0) / n for cls, s in self.score_sums.items()}


class SessionAggregator:
    """
    Layer 6 orchestrator — streaming session analytics over Layer 5 events.

    Usage
    -----
        aggregator = SessionAggregator(storage=StorageLayer())
        # Per frame, after Layer 5:
        events = aggregator.process(ctx)   # real-time alert/update events
        # On shutdown:
        aggregator.close()                 # closes appearances and sessions
    """

    def __init__(self, storage=None):
        """
        Parameters
        ----------
        storage : StorageLayer or None
            Layer 7 store. None disables persistence (metrics and alerts are
            still computed and returned — useful for tests).
        """
        self._storage = storage
        self._sessions: dict = {}          # track_id → _PersonSession
        self._frames_processed = 0
        self.n_sessions_opened = 0
        self.n_appearances_closed = 0

        self._run_id = getattr(storage, "run_id", "norun")

        log.info(f"  [Layer6] Analytics ready. Appearance timeout: "
              f"{APPEARANCE_TIMEOUT_SEC:.0f}s, trend window: "
              f"{TREND_WINDOW_SEC:.0f}s, storage: "
              f"{'ON' if storage is not None else 'OFF'}")

    # ─── Public API ───────────────────────────────────────────────────────────

    def process(self, ctx: FrameContext) -> list:
        """
        Consume one frame's detections; return real-time events.

        Parameters
        ----------
        ctx : FrameContext
            Must have passed Layers 3-5 (track_id and expression fields
            populated where available).

        Returns
        -------
        list of dict — real-time events for this frame, each with an
        "event_type" of 'presence_alert', 'threshold_alert', or
        'live_expression_update'.
        """
        now = ctx.timestamp
        events = []
        self._frames_processed += 1

        for det in ctx.detections:
            if det.track_id is None:
                continue  # tentative track — no session until confirmed

            state = self._sessions.get(det.track_id)
            if state is None:
                state = _PersonSession(ctx.camera_id, det.track_id, now,
                                       self._run_id)
                self._sessions[det.track_id] = state
                self.n_sessions_opened += 1
                if self._storage is not None:
                    self._storage.open_session({
                        "session_id": state.session_id,
                        "camera_id": state.camera_id,
                        "track_id": state.track_id,
                        "person_id": state.person_id,
                        "identity_label": state.identity_label,
                        "session_start": state.session_start,
                    })

            # Identity sticks once known — later frames where recognition
            # momentarily fails must not erase it.
            if getattr(det, "person_id", None):
                state.person_id = det.person_id
            if det.identity_label and det.identity_label != "unknown":
                state.identity_label = det.identity_label

            # ── Appearance opens when the person (re)enters the frame ─────
            if not state.is_present:
                state.begin_appearance(now)
                events.append(self._presence_event(state, now, "appeared"))

            state.last_seen = now
            state.frames_observed += 1
            state.appearance_frames += 1

            # ── Expression aggregation (FRESH readings only) ──────────────
            if det.expression_scores and det.expression_is_fresh:
                state.last_scores = det.expression_scores
                state.add_scores(now, det.expression_scores, TREND_WINDOW_SEC)

                if det.dominant_expression:
                    state.dominant_counts[det.dominant_expression] = \
                        state.dominant_counts.get(det.dominant_expression, 0) + 1

                if self._storage is not None:
                    self._storage.write_expression_event({
                        "session_id": state.session_id,
                        "timestamp": now,
                        "camera_id": ctx.camera_id,
                        "frame_seq": ctx.frame_seq,
                        "track_id": det.track_id,
                        "identity_label": state.identity_label,
                        "dominant_expression": det.dominant_expression,
                        "confidence": det.expression_confidence or 0.0,
                        "expression_scores": det.expression_scores,
                        "valence": det.valence,
                        "arousal": det.arousal,
                        "mood": det.mood,
                    })

                # ── Threshold alert ───────────────────────────────────────
                trend = state.trend()
                neg_ratio = sum(
                    p for cls, p in trend.items() if cls in NEGATIVE_CLASSES
                )
                if (neg_ratio > NEGATIVE_RATIO_THRESHOLD
                        and now - state.last_threshold_alert > ALERT_COOLDOWN_SEC):
                    state.last_threshold_alert = now
                    events.append({
                        "event_type": "threshold_alert",
                        "timestamp": now,
                        "camera_id": ctx.camera_id,
                        "track_id": det.track_id,
                        "identity_label": state.identity_label,
                        "metric": "negative_expression_ratio",
                        "value": round(neg_ratio, 3),
                        "threshold": NEGATIVE_RATIO_THRESHOLD,
                    })

        # ── Close appearances for people who have left the frame ─────────────
        # The SESSION stays open — they may come back, and either way it is
        # finalised only when the pipeline shuts down.
        for state in self._sessions.values():
            if state.is_present and now - state.last_seen > APPEARANCE_TIMEOUT_SEC:
                events.append(self._presence_event(state, state.last_seen,
                                                   "departed"))
                self._close_appearance(state, state.last_seen)

        # ── Periodic live update (Layer 8 will push these via WebSocket) ─────
        present = {tid: s for tid, s in self._sessions.items() if s.is_present}
        if present and self._frames_processed % LIVE_UPDATE_EVERY_N_FRAMES == 0:
            events.append({
                "event_type": "live_expression_update",
                "timestamp": now,
                "camera_id": ctx.camera_id,
                "tracks": {
                    tid: {
                        "identity_label": s.identity_label,
                        "expression_scores": s.last_scores,
                        "present_seconds": round(
                            s.total_present_seconds
                            + (now - s.appearance_start), 1),
                    }
                    for tid, s in present.items()
                },
            })

        return events

    def close(self):
        """
        Close every open appearance and finalise every session.
        Call on pipeline shutdown.
        """
        for state in list(self._sessions.values()):
            if state.is_present:
                self._close_appearance(state, state.last_seen)
            self._finalise(state)
        self._sessions.clear()

    @property
    def active_session_count(self) -> int:
        """Sessions opened this run (people seen at least once)."""
        return len(self._sessions)

    @property
    def present_count(self) -> int:
        """People currently on screen."""
        return sum(1 for s in self._sessions.values() if s.is_present)

    # ─── Internal ─────────────────────────────────────────────────────────────

    def _close_appearance(self, state: _PersonSession, ended_at: float):
        record = state.end_appearance(ended_at)
        self.n_appearances_closed += 1
        if self._storage is not None:
            self._storage.write_appearance(record)
        return record

    def _presence_event(self, state: _PersonSession, now: float,
                        event_type: str) -> dict:
        event = {
            "event_type": "presence_alert",
            "presence": event_type,          # 'appeared' | 'departed'
            "timestamp": now,
            "camera_id": state.camera_id,
            "track_id": state.track_id,
            "identity_label": state.identity_label,
        }
        if self._storage is not None:
            self._storage.write_presence_event({
                "session_id": state.session_id,
                "timestamp": now,
                "camera_id": state.camera_id,
                "track_id": state.track_id,
                "identity_label": state.identity_label,
                "event_type": event_type,
            })
        return event

    def _finalise(self, state: _PersonSession) -> dict:
        """Build the completed session record and persist it (Arch Doc §3)."""
        record = {
            "session_id": state.session_id,
            "camera_id": state.camera_id,
            "track_id": state.track_id,
            "person_id": state.person_id,
            "identity_label": state.identity_label,
            "session_start": state.session_start,
            "session_end": state.last_seen,
            "total_present_seconds": round(state.total_present_seconds, 3),
            "appearance_count": state.appearance_count,
            "frames_observed": state.frames_observed,
            "expression_trend": {
                cls: round(v, 4) for cls, v in state.trend().items()
            },
            "dominant_expression_distribution": dict(state.dominant_counts),
        }
        if self._storage is not None:
            self._storage.close_session(record)
        return record
