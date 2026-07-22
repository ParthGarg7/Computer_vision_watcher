"""
tests/test_api.py
─────────────────────────────────────────────────────────────────────────────
Tests for the Layer 8 REST API scaffold.

A temp database is seeded through the REAL StorageLayer (so the API is
tested against exactly what the pipeline writes), then queried through
FastAPI's TestClient. Also guards the two safety properties:

  - the API must be READ-ONLY (it must never delete stranger data the way
    the pipeline's own StorageLayer startup does)
  - /api/identities must expose names and counts ONLY, never embeddings

Run:  python -m unittest discover tests
"""

import json
import os
import sys
import tempfile
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fastapi.testclient import TestClient

from src.layer7_storage.store import StorageLayer
from src.layer8_api.api import app


def seed_db(db_path):
    """Write one registered session (2 appearances) + one stranger session."""
    store = StorageLayer(db_path=db_path, clear_unregistered=False)
    store.open_session({"session_id": "s-alice", "camera_id": "cam",
                        "track_id": 1, "person_id": "uuid-alice",
                        "identity_label": "alice", "session_start": 100.0})
    for started, ended in ((100.0, 110.0), (130.0, 145.0)):
        store.write_appearance({"session_id": "s-alice", "track_id": 1,
                                "started": started, "ended": ended,
                                "duration_seconds": ended - started,
                                "frames_observed": 100})
    for i in range(6):
        store.write_expression_event({
            "session_id": "s-alice", "timestamp": 100.0 + i, "camera_id": "cam",
            "frame_seq": i, "track_id": 1, "identity_label": "alice",
            "dominant_expression": "happiness", "confidence": 0.9,
            "expression_scores": {"happiness": 0.9, "neutral": 0.1},
            "valence": 0.5, "arousal": 0.1, "mood": "pleased"})
    store.close_session({"session_id": "s-alice", "session_end": 145.0,
                         "total_present_seconds": 25.0, "appearance_count": 2,
                         "frames_observed": 200,
                         "expression_trend": {"happiness": 0.9},
                         "dominant_expression_distribution": {"happiness": 6},
                         "person_id": "uuid-alice", "identity_label": "alice"})
    store.open_session({"session_id": "s-stranger", "camera_id": "cam",
                        "track_id": 2, "person_id": None,
                        "identity_label": None, "session_start": 200.0})
    store.close()


class ApiTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._dir = tempfile.TemporaryDirectory()
        cls.db = os.path.join(cls._dir.name, "api.db")
        seed_db(cls.db)
        cls.meta = os.path.join(cls._dir.name, "store.meta.json")
        with open(cls.meta, "w", encoding="utf-8") as f:
            json.dump({"meta": {"uuid-alice": {"name": "alice", "count": 3}},
                       "id_map": ["uuid-alice"] * 3, "threshold": 0.45}, f)
        cls._saved = (app.state.db_path, app.state.faces_meta)
        app.state.db_path = cls.db
        app.state.faces_meta = cls.meta
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        app.state.db_path, app.state.faces_meta = cls._saved
        cls._dir.cleanup()

    # ── endpoints ─────────────────────────────────────────────────────────

    def test_health(self):
        r = self.client.get("/api/health").json()
        self.assertEqual(r["status"], "ok")
        self.assertTrue(r["database"]["exists"])
        self.assertEqual(r["database"]["rows"]["sessions"], 2)

    def test_stats_groups_people(self):
        r = self.client.get("/api/stats").json()
        people = {p["person"]: p for p in r["people"]}
        self.assertIn("alice", people)
        self.assertIn("(unidentified)", people)
        self.assertEqual(people["alice"]["appearances"], 2)

    def test_sessions_include_appearances(self):
        r = self.client.get("/api/sessions").json()
        alice = next(s for s in r["sessions"]
                     if s["session_id"] == "s-alice")
        self.assertEqual(len(alice["appearances"]), 2)
        self.assertEqual(alice["dominant_expression_distribution"],
                         {"happiness": 6})

    def test_session_detail_and_404(self):
        r = self.client.get("/api/sessions/s-alice")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.json()["appearances"]), 2)
        self.assertEqual(self.client.get("/api/sessions/nope").status_code, 404)

    def test_expression_trend_buckets(self):
        r = self.client.get("/api/sessions/s-alice/expression_trend"
                            "?bucket=2").json()
        self.assertEqual(sum(b["measurements"] for b in r["trend"]), 6)
        self.assertAlmostEqual(r["trend"][0]["avg_valence"], 0.5, places=4)

    def test_identities_expose_no_embeddings(self):
        r = self.client.get("/api/identities").json()
        self.assertEqual(r["identities"][0]["name"], "alice")
        self.assertEqual(r["identities"][0]["samples"], 3)
        # Nothing vector-like may leak through this endpoint
        self.assertNotIn("embedding", json.dumps(r).lower())
        self.assertNotIn("id_map", json.dumps(r))

    def test_identity_history(self):
        r = self.client.get("/api/identities/alice/history").json()
        self.assertEqual(r["sessions"], 1)
        self.assertEqual(r["total_present_seconds"], 25.0)
        self.assertEqual(
            self.client.get("/api/identities/nobody/history").status_code, 404)

    def test_recent_events_include_mood(self):
        r = self.client.get("/api/events/recent?limit=3").json()
        self.assertEqual(len(r["events"]), 3)
        self.assertEqual(r["events"][0]["mood"], "pleased")

    def test_dashboard_served(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("WATCHER", r.text)

    # ── safety properties ─────────────────────────────────────────────────

    def test_api_is_read_only(self):
        # The stranger session must SURVIVE any number of API calls — the
        # pipeline's own startup clears strangers; the API must never.
        for _ in range(3):
            self.client.get("/api/health")
            self.client.get("/api/sessions")
        r = self.client.get("/api/sessions").json()
        ids = [s["session_id"] for s in r["sessions"]]
        self.assertIn("s-stranger", ids)

    def test_missing_db_gives_503_not_crash(self):
        app.state.db_path = os.path.join(self._dir.name, "absent.db")
        try:
            r = self.client.get("/api/stats")
            self.assertEqual(r.status_code, 503)
        finally:
            app.state.db_path = self.db


if __name__ == "__main__":
    unittest.main()
