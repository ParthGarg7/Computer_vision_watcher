"""
tests/test_analytics_storage.py
─────────────────────────────────────────────────────────────────────────────
Unit + integration tests for Layer 6 (SessionAggregator) and Layer 7
(StorageLayer). No models required.

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
    SessionAggregator, TRACK_TIMEOUT_SEC, NEGATIVE_RATIO_THRESHOLD,
)
from src.layer7_storage.store import StorageLayer


class FakeDet:
    def __init__(self, tid, label=None, scores=None, fresh=True):
        self.track_id = tid
        self.identity_label = label
        self.expression_scores = scores
        self.dominant_expression = (
            max(scores, key=scores.get) if scores else None)
        self.expression_confidence = (
            max(scores.values()) if scores else None)
        # Layer 6 only aggregates fresh readings; default True so existing
        # tests read as "a real measurement arrived this frame".
        self.expression_is_fresh = fresh


class FakeCtx:
    def __init__(self, ts, seq, dets, camera_id="cam_test"):
        self.timestamp = ts
        self.frame_seq = seq
        self.camera_id = camera_id
        self.detections = dets


HAPPY = {"happiness": 0.9, "neutral": 0.1}
ANGRY = {"anger": 0.7, "disgust": 0.2, "neutral": 0.1}


class SessionAggregatorTests(unittest.TestCase):

    def _events_of(self, events, event_type):
        return [e for e in events if e["event_type"] == event_type]

    def test_appear_and_depart_lifecycle(self):
        agg = SessionAggregator(storage=None)
        events = agg.process(FakeCtx(100.0, 0, [FakeDet(1, "alice", HAPPY)]))
        appeared = self._events_of(events, "presence_alert")
        self.assertEqual(len(appeared), 1)
        self.assertEqual(appeared[0]["presence"], "appeared")
        self.assertEqual(agg.active_session_count, 1)

        # Empty frame past the timeout → departed + finalized
        events = agg.process(FakeCtx(100.0 + TRACK_TIMEOUT_SEC + 1, 1, []))
        departed = self._events_of(events, "presence_alert")
        self.assertEqual(len(departed), 1)
        self.assertEqual(departed[0]["presence"], "departed")
        self.assertEqual(agg.active_session_count, 0)
        self.assertEqual(agg.n_sessions_finalized, 1)

    def test_untracked_detections_ignored(self):
        agg = SessionAggregator(storage=None)
        agg.process(FakeCtx(100.0, 0, [FakeDet(None, None, HAPPY)]))
        self.assertEqual(agg.active_session_count, 0)

    def test_trend_running_sums_match_window(self):
        agg = SessionAggregator(storage=None)
        for i in range(10):
            agg.process(FakeCtx(100.0 + i / 30, i, [FakeDet(1, None, HAPPY)]))
        state = agg._sessions[1]
        trend = state.trend()
        self.assertAlmostEqual(trend["happiness"], 0.9, places=6)
        self.assertAlmostEqual(trend["neutral"], 0.1, places=6)
        # Sums stay consistent after eviction (samples older than the window)
        agg.process(FakeCtx(100.0 + 31.0, 10, [FakeDet(1, None, ANGRY)]))
        trend = state.trend()
        # Window now holds only the ANGRY sample
        self.assertAlmostEqual(trend["anger"], 0.7, places=6)
        self.assertEqual(len(state.score_window), 1)

    def test_threshold_alert_fires_once_with_cooldown(self):
        agg = SessionAggregator(storage=None)
        alerts = []
        for i in range(60):  # 2 seconds of angry frames
            events = agg.process(
                FakeCtx(100.0 + i / 30, i, [FakeDet(2, None, ANGRY)]))
            alerts += self._events_of(events, "threshold_alert")
        self.assertEqual(len(alerts), 1)  # cooldown suppresses repeats
        self.assertGreater(alerts[0]["value"], NEGATIVE_RATIO_THRESHOLD)

    def test_carried_forward_readings_are_not_aggregated(self):
        # Regression: Layer 5 measures every 5th frame and carries the label
        # forward in between. Aggregating those carry-forwards recorded one
        # real measurement 5 times — inflating counts and the events table.
        agg = SessionAggregator(storage=None)
        # 1 fresh reading followed by 4 carry-forwards, twice over
        for i in range(10):
            fresh = (i % 5 == 0)
            agg.process(FakeCtx(100.0 + i / 30, i,
                                [FakeDet(1, "alice", HAPPY, fresh=fresh)]))
        state = agg._sessions[1]
        self.assertEqual(state.frames_observed, 10)       # every frame counts
        self.assertEqual(sum(state.dominant_counts.values()), 2)  # 2 measurements
        self.assertEqual(len(state.score_window), 2)      # window holds measurements

    def test_only_fresh_readings_reach_storage(self):
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "fresh.db")
            store = StorageLayer(db_path=db)
            agg = SessionAggregator(storage=store)
            for i in range(20):
                agg.process(FakeCtx(100.0 + i / 30, i,
                                    [FakeDet(1, "alice", HAPPY,
                                             fresh=(i % 5 == 0))]))
            agg.close()
            store.close()
            # 20 frames, 4 measurements -> 4 rows, not 20
            self.assertEqual(store.n_expression_events, 4)

    def test_close_finalizes_open_sessions(self):
        agg = SessionAggregator(storage=None)
        agg.process(FakeCtx(100.0, 0, [FakeDet(1, "alice", HAPPY)]))
        agg.close()
        self.assertEqual(agg.active_session_count, 0)
        self.assertEqual(agg.n_sessions_finalized, 1)

    def test_identity_label_sticks_once_known(self):
        agg = SessionAggregator(storage=None)
        agg.process(FakeCtx(100.0, 0, [FakeDet(1, "alice", HAPPY)]))
        # Later frames say 'unknown' — the session keeps the known label
        agg.process(FakeCtx(100.1, 1, [FakeDet(1, "unknown", HAPPY)]))
        self.assertEqual(agg._sessions[1].identity_label, "alice")


class StorageLayerTests(unittest.TestCase):

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self._dir.name, "test.db")
        self.store = StorageLayer(db_path=self.db_path)

    def tearDown(self):
        self.store.close()
        self._dir.cleanup()

    def _expr_record(self, ts, seq, tid=1):
        return {
            "timestamp": ts, "camera_id": "cam", "frame_seq": seq,
            "track_id": tid, "identity_label": "alice",
            "dominant_expression": "happiness", "confidence": 0.9,
            "expression_scores": HAPPY,
        }

    def test_expression_events_batched_flush(self):
        for i in range(10):
            self.store.write_expression_event(self._expr_record(100 + i, i))
        self.store.flush()
        self.assertEqual(self.store.n_expression_events, 10)

    def test_session_round_trip(self):
        self.store.write_session({
            "session_id": "cam:1:100", "camera_id": "cam", "track_id": 1,
            "identity_label": "alice", "session_start": 100.0,
            "session_end": 110.0, "presence_duration_seconds": 10.0,
            "frames_observed": 300,
            "expression_trend": {"happiness": 0.9},
            "dominant_expression_distribution": {"happiness": 300},
        })
        rows = self.store.recent_sessions()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["identity_label"], "alice")
        self.assertEqual(rows[0]["expression_trend"], {"happiness": 0.9})
        self.assertEqual(
            rows[0]["dominant_expression_distribution"], {"happiness": 300})

    def test_expression_trend_bucketing(self):
        for i in range(60):  # 2s of events at 30 fps
            self.store.write_expression_event(
                self._expr_record(100 + i / 30, i))
        self.store.flush()
        buckets = self.store.expression_trend(track_id=1, bucket_seconds=1.0)
        self.assertEqual(len(buckets), 2)
        self.assertEqual(sum(b["frames"] for b in buckets), 60)
        self.assertAlmostEqual(buckets[0]["avg_confidence"], 0.9, places=6)

    def test_presence_events(self):
        self.store.write_presence_event({
            "timestamp": 100.0, "camera_id": "cam", "track_id": 1,
            "identity_label": None, "event_type": "appeared",
        })
        self.assertEqual(self.store.n_presence_events, 1)

    def test_close_flushes_and_is_idempotent(self):
        self.store.write_expression_event(self._expr_record(100.0, 0))
        self.store.close()
        self.store.close()  # must not raise
        reopened = StorageLayer(db_path=self.db_path)
        buckets = reopened.expression_trend(track_id=1)
        self.assertEqual(sum(b["frames"] for b in buckets), 1)
        reopened.close()


class EndToEndL6L7Tests(unittest.TestCase):

    def test_aggregator_persists_through_storage(self):
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "e2e.db")
            store = StorageLayer(db_path=db)
            agg = SessionAggregator(storage=store)
            for i in range(30):
                agg.process(FakeCtx(100.0 + i / 30, i,
                                    [FakeDet(1, "alice", HAPPY)]))
            agg.close()
            store.close()

            check = StorageLayer(db_path=db)
            sessions = check.recent_sessions()
            self.assertEqual(len(sessions), 1)
            self.assertEqual(sessions[0]["frames_observed"], 30)
            self.assertEqual(sessions[0]["identity_label"], "alice")
            check.close()


if __name__ == "__main__":
    unittest.main()
