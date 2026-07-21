"""
src/layer6_analytics/aggregator.py
─────────────────────────────────────────────────────────────────────────────
Layer 6: Analytics & Business Logic

Receives raw per-frame event records from Layer 5 and transforms them into
business-meaningful metrics. Raw AI outputs — track IDs, expression
probability dicts, timestamps — are too low-level for any user or business
decision; this layer aggregates, summarises, and contextualises them.

Transformations performed (Arch Doc §1):
    Per-frame expression scores → rolling expression trend (30 s window)
    Track events across frames  → presence duration per identity per session
    Raw event stream            → alert when a metric crosses a threshold

Outputs (Arch Doc §3):
    1. Aggregated session metrics — written to Layer 7 Storage when a track
       departs (and for still-open sessions on shutdown):
           session_id, identity_label, presence_duration_seconds,
           expression_trend, dominant_expression_distribution,
           session_start, session_end

       Expression aggregation consumes only FRESH readings (see
       Detection.expression_is_fresh). Layer 5 throttles inference and
       carries labels forward between measurements; counting those
       carry-forwards would multiply every real measurement by the throttle
       factor. dominant_expression_distribution therefore counts
       measurements, not frames — ratios are identical, magnitudes are not.
    2. Real-time events — returned to the caller each frame (main.py prints
       them; Layer 8 will push them over WebSocket):
           presence_alert         (track appeared / departed)
           threshold_alert        (negative-expression ratio over threshold)
           live_expression_update (latest scores, every N frames)

Implementation note — stdlib accumulators instead of Pandas:
    The Arch Doc (§4.1) recommends Pandas for MVP aggregation. This module
    uses plain dicts/deques in the per-frame hot path instead: the streaming
    aggregations needed (running sums, rolling window means, counters) are
    O(1) per event without materialising a DataFrame 30x per second, and it
    avoids adding a heavyweight dependency. Pandas remains the right tool
    for OFFLINE analysis of the Layer 7 tables (pd.read_sql on watcher.db).

Session definition:
    One session = one track's continuous presence on one camera. A track
    that is not seen for TRACK_TIMEOUT_SEC is considered departed and its
    session is finalized. DeepSORT track IDs are not reused within a run,
    so session_id = "{camera_id}:{track_id}:{start_epoch}" is unique.

Ref: Layer 6 Architecture Doc — Sections 1, 2, 3, 4.1, 6, 7
"""

from collections import deque

from src.core.frame_context import FrameContext

# ─── Constants ────────────────────────────────────────────────────────────────

# Track not updated for this long → departed, session finalized.
# Matches the Redis TTL example in the Arch Doc (§4.2, "e.g. 5 seconds").
TRACK_TIMEOUT_SEC = 5.0

# Rolling window for the expression trend (Arch Doc §1: 30-second window).
TREND_WINDOW_SEC = 30.0

# Expression classes counted as "negative" for the threshold alert.
NEGATIVE_CLASSES = frozenset({"anger", "disgust", "fear", "sadness", "contempt"})

# threshold_alert fires when the rolling mean of negative-class probability
# mass exceeds this ratio. Arch Doc §7: alert rules must be documented and
# reviewed — this default is deliberately conservative.
NEGATIVE_RATIO_THRESHOLD = 0.60

# Minimum seconds between repeated threshold alerts for the same track.
ALERT_COOLDOWN_SEC = 30.0

# live_expression_update events are emitted every N processed frames.
LIVE_UPDATE_EVERY_N_FRAMES = 15


class _TrackSession:
    """Mutable per-track session state (in-process; Redis at scale-up)."""

    __slots__ = (
        "session_id", "camera_id", "track_id", "identity_label",
        "session_start", "last_seen", "frames_observed",
        "score_window", "score_sums", "dominant_counts", "last_scores",
        "last_threshold_alert",
    )

    def __init__(self, camera_id: str, track_id: int, now: float):
        self.session_id = f"{camera_id}:{track_id}:{int(now)}"
        self.camera_id = camera_id
        self.track_id = track_id
        self.identity_label = None
        self.session_start = now
        self.last_seen = now
        self.frames_observed = 0
        # (timestamp, scores_dict) pairs inside the trend window
        self.score_window: deque = deque()
        # Running per-class sums over score_window, maintained incrementally
        # on append/evict — trend() is O(classes) instead of O(window) so the
        # per-frame threshold check stays cheap (window is ~900 entries at
        # 30 fps x 30 s).
        self.score_sums: dict = {}
        # Counts of MEASUREMENTS (fresh inferences), not frames — Layer 5
        # samples every EXPRESSION_EVERY_N_FRAMES frames per track. Ratios
        # are unaffected; absolute counts are ~N x smaller than frame counts.
        self.dominant_counts: dict = {}
        self.last_scores: dict = {}
        self.last_threshold_alert = 0.0

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
        aggregator.close()                 # finalizes open sessions
    """

    def __init__(self, storage=None):
        """
        Parameters
        ----------
        storage : StorageLayer or None
            Layer 7 store. None disables persistence (metrics and alerts
            still computed and returned — useful for tests).
        """
        self._storage = storage
        self._sessions: dict = {}          # track_id → _TrackSession
        self._frames_processed = 0
        self.n_sessions_finalized = 0

        print(f"  [Layer6] Analytics ready. Track timeout: "
              f"{TRACK_TIMEOUT_SEC:.0f}s, trend window: "
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
        list of dict — real-time events for this frame. Each has an
        "event_type" key: 'presence_alert', 'threshold_alert', or
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
                state = _TrackSession(ctx.camera_id, det.track_id, now)
                self._sessions[det.track_id] = state
                events.append(self._presence_event(state, now, "appeared"))

            state.last_seen = now
            state.frames_observed += 1
            if det.identity_label and det.identity_label != "unknown":
                state.identity_label = det.identity_label

            # ── Expression aggregation ────────────────────────────────────
            # Only FRESH readings count. Layer 5 measures every N frames and
            # carries the label forward in between (so the display doesn't
            # flicker); aggregating those carry-forwards would record one
            # real measurement N times — inflating dominant_counts, bloating
            # the events table N-fold, and filling the trend window with
            # duplicates.
            if det.expression_scores and det.expression_is_fresh:
                state.last_scores = det.expression_scores
                state.add_scores(now, det.expression_scores, TREND_WINDOW_SEC)

                if det.dominant_expression:
                    state.dominant_counts[det.dominant_expression] = \
                        state.dominant_counts.get(det.dominant_expression, 0) + 1

                if self._storage is not None:
                    self._storage.write_expression_event({
                        "timestamp": now,
                        "camera_id": ctx.camera_id,
                        "frame_seq": ctx.frame_seq,
                        "track_id": det.track_id,
                        "identity_label": state.identity_label,
                        "dominant_expression": det.dominant_expression,
                        "confidence": det.expression_confidence or 0.0,
                        "expression_scores": det.expression_scores,
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

        # ── Departed tracks → finalize sessions ──────────────────────────────
        departed = [
            tid for tid, s in self._sessions.items()
            if now - s.last_seen > TRACK_TIMEOUT_SEC
        ]
        for tid in departed:
            state = self._sessions.pop(tid)
            events.append(self._presence_event(state, now, "departed"))
            self._finalize(state, end_ts=state.last_seen)

        # ── Periodic live update (Layer 8 will push these via WebSocket) ─────
        if self._sessions and self._frames_processed % LIVE_UPDATE_EVERY_N_FRAMES == 0:
            events.append({
                "event_type": "live_expression_update",
                "timestamp": now,
                "camera_id": ctx.camera_id,
                "tracks": {
                    tid: {
                        "identity_label": s.identity_label,
                        "expression_scores": s.last_scores,
                        "presence_seconds": round(now - s.session_start, 1),
                    }
                    for tid, s in self._sessions.items()
                },
            })

        return events

    def close(self):
        """Finalize all still-open sessions (call on pipeline shutdown)."""
        for state in list(self._sessions.values()):
            self._finalize(state, end_ts=state.last_seen)
        self._sessions.clear()

    @property
    def active_session_count(self) -> int:
        return len(self._sessions)

    # ─── Internal ─────────────────────────────────────────────────────────────

    def _presence_event(self, state: _TrackSession, now: float,
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
                "timestamp": now,
                "camera_id": state.camera_id,
                "track_id": state.track_id,
                "identity_label": state.identity_label,
                "event_type": event_type,
            })
        return event

    def _finalize(self, state: _TrackSession, end_ts: float):
        """Build the aggregated session record and persist it (Arch Doc §3)."""
        record = {
            "session_id": state.session_id,
            "camera_id": state.camera_id,
            "track_id": state.track_id,
            "identity_label": state.identity_label,
            "session_start": state.session_start,
            "session_end": end_ts,
            "presence_duration_seconds": round(end_ts - state.session_start, 3),
            "frames_observed": state.frames_observed,
            "expression_trend": {
                cls: round(v, 4) for cls, v in state.trend().items()
            },
            "dominant_expression_distribution": dict(state.dominant_counts),
        }
        self.n_sessions_finalized += 1
        if self._storage is not None:
            self._storage.write_session(record)
        return record
