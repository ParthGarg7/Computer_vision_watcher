"""
tests/test_analytics_storage.py
─────────────────────────────────────────────────────────────────────────────
Unit + integration tests for Layer 6 (SessionAggregator) and Layer 7
(StorageLayer), including the session/appearance model:

    session    = one person, one run of the program (closes at shutdown)
    appearance = one continuous stretch on screen (closes after 5s absence)

No models required.

Run:  python -m unittest discover tests
"""

import os
import sys
import tempfile
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.layer6_analytics.aggregator import (
    SessionAggregator, APPEARANCE_TIMEOUT_SEC, NEGATIVE_RATIO_THRESHOLD,
)
from src.layer7_storage.store import StorageLayer


class FakeDet:
    def __init__(self, tid, label=None, scores=None, fresh=True, person_id=None,
                 valence=None, arousal=None, mood=None):
        self.track_id = tid
        self.identity_label = label
        self.person_id = person_id
        self.expression_scores = scores
        self.dominant_expression = (
            max(scores, key=scores.get) if scores else None)
        self.expression_confidence = (
            max(scores.values()) if scores else None)
        # Layer 6 only aggregates fresh readings; default True so existing
        # tests read as "a real measurement arrived this frame".
        self.expression_is_fresh = fresh
        # Dimensional affect — None for the plain 8-class models
        self.valence = valence
        self.arousal = arousal
        self.mood = mood


class FakeCtx:
    def __init__(self, ts, seq, dets, camera_id="cam_test"):
        self.timestamp = ts
        self.frame_seq = seq
        self.camera_id = camera_id
        self.detections = dets


HAPPY = {"happiness": 0.9, "neutral": 0.1}
ANGRY = {"anger": 0.7, "disgust": 0.2, "neutral": 0.1}
GONE = APPEARANCE_TIMEOUT_SEC + 1


class SessionAppearanceTests(unittest.TestCase):
    """The core model: one session, many appearances."""

    def _events_of(self, events, event_type):
        return [e for e in events if e["event_type"] == event_type]

    def test_leaving_and_returning_keeps_one_session(self):
        # THE headline behaviour: step away, come back, still one session.
        agg = SessionAggregator(storage=None)
        t = 100.0
        for i in range(5):                      # present
            agg.process(FakeCtx(t + i / 30, i, [FakeDet(1, "alice", HAPPY)]))
        agg.process(FakeCtx(t + GONE, 10, []))  # leaves -> appearance closes
        agg.process(FakeCtx(t + 30, 20, [FakeDet(1, "alice", HAPPY)]))  # back

        self.assertEqual(agg.active_session_count, 1, "must not start a 2nd session")
        state = agg._sessions[1]
        self.assertEqual(state.appearance_count, 1)   # the closed one
        self.assertTrue(state.is_present)             # a new one is open

    def test_appearance_closes_after_timeout_session_does_not(self):
        agg = SessionAggregator(storage=None)
        agg.process(FakeCtx(100.0, 0, [FakeDet(1, "alice", HAPPY)]))
        self.assertTrue(agg._sessions[1].is_present)
        agg.process(FakeCtx(100.0 + GONE, 1, []))
        self.assertFalse(agg._sessions[1].is_present)   # appearance closed
        self.assertEqual(agg.active_session_count, 1)   # session still open

    def test_brief_gap_does_not_split_the_appearance(self):
        agg = SessionAggregator(storage=None)
        agg.process(FakeCtx(100.0, 0, [FakeDet(1, "alice", HAPPY)]))
        agg.process(FakeCtx(102.0, 1, []))              # 2s gap, under timeout
        agg.process(FakeCtx(103.0, 2, [FakeDet(1, "alice", HAPPY)]))
        self.assertEqual(agg._sessions[1].appearance_count, 0)  # never closed
        self.assertTrue(agg._sessions[1].is_present)

    def test_three_appearances_accumulate_present_time(self):
        agg = SessionAggregator(storage=None)
        t = 100.0
        for _ in range(3):
            agg.process(FakeCtx(t, 0, [FakeDet(1, "alice", HAPPY)]))
            agg.process(FakeCtx(t + 10, 1, [FakeDet(1, "alice", HAPPY)]))
            agg.process(FakeCtx(t + 10 + GONE, 2, []))   # leaves
            t += 60
        state = agg._sessions[1]
        self.assertEqual(state.appearance_count, 3)
        self.assertAlmostEqual(state.total_present_seconds, 30.0, places=3)

    def test_presence_events_fire_per_appearance(self):
        agg = SessionAggregator(storage=None)
        evs = agg.process(FakeCtx(100.0, 0, [FakeDet(1, "alice", HAPPY)]))
        self.assertEqual(self._events_of(evs, "presence_alert")[0]["presence"],
                         "appeared")
        evs = agg.process(FakeCtx(100.0 + GONE, 1, []))
        self.assertEqual(self._events_of(evs, "presence_alert")[0]["presence"],
                         "departed")

    def test_close_finalises_open_appearance_and_session(self):
        agg = SessionAggregator(storage=None)
        agg.process(FakeCtx(100.0, 0, [FakeDet(1, "alice", HAPPY)]))
        agg.process(FakeCtx(105.0, 1, [FakeDet(1, "alice", HAPPY)]))
        agg.close()
        self.assertEqual(agg.active_session_count, 0)
        self.assertEqual(agg.n_appearances_closed, 1)

    def test_untracked_detections_ignored(self):
        agg = SessionAggregator(storage=None)
        agg.process(FakeCtx(100.0, 0, [FakeDet(None, None, HAPPY)]))
        self.assertEqual(agg.active_session_count, 0)

    def test_identity_and_person_id_stick_once_known(self):
        agg = SessionAggregator(storage=None)
        agg.process(FakeCtx(100.0, 0, [FakeDet(1, "alice", HAPPY, person_id="u1")]))
        agg.process(FakeCtx(100.1, 1, [FakeDet(1, "unknown", HAPPY)]))
        self.assertEqual(agg._sessions[1].identity_label, "alice")
        self.assertEqual(agg._sessions[1].person_id, "u1")

    def test_present_count_vs_session_count(self):
        agg = SessionAggregator(storage=None)
        agg.process(FakeCtx(100.0, 0, [FakeDet(1, "a", HAPPY),
                                       FakeDet(2, None, HAPPY)]))
        self.assertEqual(agg.present_count, 2)
        agg.process(FakeCtx(100.0 + GONE, 1, [FakeDet(1, "a", HAPPY)]))
        self.assertEqual(agg.present_count, 1)          # track 2 left
        self.assertEqual(agg.active_session_count, 2)   # both sessions live


class ExpressionAggregationTests(unittest.TestCase):

    def _events_of(self, events, event_type):
        return [e for e in events if e["event_type"] == event_type]

    def test_trend_running_sums_match_window(self):
        agg = SessionAggregator(storage=None)
        for i in range(10):
            agg.process(FakeCtx(100.0 + i / 30, i, [FakeDet(1, None, HAPPY)]))
        trend = agg._sessions[1].trend()
        self.assertAlmostEqual(trend["happiness"], 0.9, places=6)
        self.assertAlmostEqual(trend["neutral"], 0.1, places=6)

    def test_carried_forward_readings_are_not_aggregated(self):
        agg = SessionAggregator(storage=None)
        for i in range(10):
            agg.process(FakeCtx(100.0 + i / 30, i,
                                [FakeDet(1, "alice", HAPPY, fresh=(i % 5 == 0))]))
        state = agg._sessions[1]
        self.assertEqual(state.frames_observed, 10)              # every frame
        self.assertEqual(sum(state.dominant_counts.values()), 2)  # 2 measurements
        self.assertEqual(len(state.score_window), 2)

    def test_threshold_alert_fires_once_with_cooldown(self):
        agg = SessionAggregator(storage=None)
        alerts = []
        for i in range(60):
            evs = agg.process(FakeCtx(100.0 + i / 30, i, [FakeDet(2, None, ANGRY)]))
            alerts += self._events_of(evs, "threshold_alert")
        self.assertEqual(len(alerts), 1)
        self.assertGreater(alerts[0]["value"], NEGATIVE_RATIO_THRESHOLD)


class StorageLayerTests(unittest.TestCase):

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self._dir.name, "test.db")
        self.store = StorageLayer(db_path=self.db_path)

    def tearDown(self):
        self.store.close()
        self._dir.cleanup()

    def _open(self, sid="s1", person_id=None, label=None):
        self.store.open_session({
            "session_id": sid, "camera_id": "cam", "track_id": 1,
            "person_id": person_id, "identity_label": label,
            "session_start": 100.0,
        })

    def test_open_then_close_session(self):
        self._open(person_id="u1", label="alice")
        self.store.close_session({
            "session_id": "s1", "session_end": 200.0,
            "total_present_seconds": 80.0, "appearance_count": 2,
            "frames_observed": 300, "expression_trend": {"happiness": 0.9},
            "dominant_expression_distribution": {"happiness": 60},
            "person_id": "u1", "identity_label": "alice",
        })
        rows = self.store.recent_sessions()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["identity_label"], "alice")
        self.assertEqual(rows[0]["appearance_count"], 2)
        self.assertEqual(rows[0]["total_present_seconds"], 80.0)

    def test_appearances_linked_to_session(self):
        self._open(person_id="u1", label="alice")
        for i in range(3):
            self.store.write_appearance({
                "session_id": "s1", "track_id": 1,
                "started": 100.0 + i * 50, "ended": 110.0 + i * 50,
                "duration_seconds": 10.0, "frames_observed": 300,
            })
        apps = self.store.appearances_for("s1")
        self.assertEqual(len(apps), 3)
        self.assertEqual(sum(a["duration_seconds"] for a in apps), 30.0)

    def test_expression_events_batched(self):
        self._open()
        for i in range(10):
            self.store.write_expression_event({
                "session_id": "s1", "timestamp": 100 + i, "camera_id": "cam",
                "frame_seq": i, "track_id": 1, "identity_label": "alice",
                "dominant_expression": "happiness", "confidence": 0.9,
                "expression_scores": HAPPY,
            })
        self.store.flush()
        self.assertEqual(self.store.n_expression_events, 10)

    def test_clear_unregistered_removes_only_strangers(self):
        # A registered person and a stranger, each with child rows
        self._open("known", person_id="u1", label="alice")
        self._open("stranger", person_id=None, label=None)
        for sid in ("known", "stranger"):
            self.store.write_appearance({
                "session_id": sid, "track_id": 1, "started": 100.0,
                "ended": 110.0, "duration_seconds": 10.0, "frames_observed": 30,
            })
            # NULL identity_label on both — a registered person's early events
            # look exactly like a stranger's, so deletion must go by session.
            self.store.write_presence_event({
                "session_id": sid, "timestamp": 100.0, "camera_id": "cam",
                "track_id": 1, "identity_label": None, "event_type": "appeared",
            })
        removed = self.store.clear_unregistered()
        self.assertGreater(removed, 0)
        ids = [r["session_id"] for r in self.store.recent_sessions()]
        self.assertEqual(ids, ["known"])
        self.assertEqual(len(self.store.appearances_for("stranger")), 0)
        self.assertEqual(len(self.store.appearances_for("known")), 1)
        # The registered person's NULL-label presence row must survive
        n = self.store._conn.execute(
            "SELECT COUNT(*) FROM presence_events").fetchone()[0]
        self.assertEqual(n, 1)

    def test_close_is_idempotent(self):
        self._open()
        self.store.close()
        self.store.close()   # must not raise


class EndToEndTests(unittest.TestCase):

    def test_full_pipeline_writes_session_and_appearances(self):
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "e2e.db")
            store = StorageLayer(db_path=db)
            agg = SessionAggregator(storage=store)

            t = 100.0
            # Two separate stretches on screen, one session
            for i in range(10):
                agg.process(FakeCtx(t + i / 30, i,
                                    [FakeDet(1, "alice", HAPPY, person_id="u1")]))
            agg.process(FakeCtx(t + GONE, 50, []))
            for i in range(10):
                agg.process(FakeCtx(t + 30 + i / 30, 60 + i,
                                    [FakeDet(1, "alice", HAPPY, person_id="u1")]))
            agg.close()
            store.close()

            check = StorageLayer(db_path=db, clear_unregistered=False)
            sessions = check.recent_sessions()
            self.assertEqual(len(sessions), 1, "one session, not two")
            s = sessions[0]
            self.assertEqual(s["identity_label"], "alice")
            self.assertEqual(s["person_id"], "u1")
            self.assertEqual(s["appearance_count"], 2)
            self.assertEqual(len(s["appearances"]), 2)
            self.assertEqual(s["frames_observed"], 20)
            check.close()

    def test_stranger_data_cleared_on_next_run(self):
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "runs.db")
            # Run 1: one registered, one stranger
            store = StorageLayer(db_path=db)
            agg = SessionAggregator(storage=store)
            agg.process(FakeCtx(100.0, 0, [
                FakeDet(1, "alice", HAPPY, person_id="u1"),
                FakeDet(2, None, HAPPY),
            ]))
            agg.close()
            store.close()

            # Run 2: startup clears the stranger, keeps alice
            store2 = StorageLayer(db_path=db)
            labels = [r["identity_label"] for r in store2.recent_sessions()]
            self.assertEqual(labels, ["alice"])
            store2.close()


if __name__ == "__main__":
    unittest.main()
