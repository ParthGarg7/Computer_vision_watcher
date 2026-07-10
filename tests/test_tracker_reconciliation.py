"""
tests/test_tracker_reconciliation.py
─────────────────────────────────────────────────────────────────────────────
Tests for FaceTracker's stable display-ID reconciliation — the fix for two
observed field problems:
  1. A person who left the frame for longer than DeepSORT's max_age came
     back with a NEW track ID (DeepSORT has no memory of deleted tracks).
  2. Raw DeepSORT IDs inflate fast (3 people → "ID:98") because its internal
     counter increments for every initiated track, including 1-frame
     detection flickers.

Uses the real deep_sort_realtime tracker (embedder=None → no model
downloads, pure CPU logic), so this exercises the genuine track lifecycle.

Run:  python -m unittest discover tests
"""

import os
import sys
import unittest

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.layer4_identity.tracker import FaceTracker

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
FPS = 30.0


def unit_emb(seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(512).astype(np.float32)
    return v / np.linalg.norm(v)


EMB_A = unit_emb(101)
EMB_B = unit_emb(202)
EMB_C = unit_emb(303)


class TrackerReconciliationTests(unittest.TestCase):

    def setUp(self):
        # Short max_age so departure/deletion happens quickly in tests
        self.tracker = FaceTracker(max_age=10, n_init=2)
        self.t = 1000.0

    def _step(self, detections):
        """Advance one frame; detections = list of (bbox_ltwh, conf, emb)."""
        result = self.tracker.update(detections, FRAME, timestamp=self.t)
        self.t += 1.0 / FPS
        return result

    def _confirm_person(self, bbox, emb, frames=4):
        """Feed a person for several frames until confirmed; return their ID."""
        last = {}
        for _ in range(frames):
            last = self._step([(bbox, 0.9, emb)])
        self.assertIn(0, last, "track should be confirmed by now")
        return last[0]

    def _step_known(self, detections, identity_keys):
        """Advance one frame with per-detection FAISS identity keys."""
        result = self.tracker.update(detections, FRAME, timestamp=self.t,
                                     identity_keys=identity_keys)
        self.t += 1.0 / FPS
        return result

    def _confirm_known(self, bbox, emb, pid, frames=4):
        last = {}
        for _ in range(frames):
            last = self._step_known([(bbox, 0.9, emb)], [pid])
        self.assertIn(0, last)
        return last[0]

    def test_person_reacquires_id_after_leaving(self):
        # Person A confirmed → display ID assigned
        id_a = self._confirm_person([100, 100, 80, 80], EMB_A)

        # A leaves for well past max_age → DeepSORT deletes the track
        for _ in range(20):
            self._step([])

        # A returns at a DIFFERENT position — must get the SAME display ID
        id_a_back = self._confirm_person([400, 250, 80, 80], EMB_A)
        self.assertEqual(id_a_back, id_a,
                         "returning person must re-acquire their previous ID")

    def test_different_person_gets_new_id(self):
        id_a = self._confirm_person([100, 100, 80, 80], EMB_A)
        for _ in range(20):
            self._step([])
        # A different face (near-orthogonal embedding) appears where A was
        id_b = self._confirm_person([100, 100, 80, 80], EMB_B)
        self.assertNotEqual(id_b, id_a,
                            "a stranger must never inherit a departed ID")

    def test_display_ids_are_small_and_sequential(self):
        # Three people confirmed together → IDs from {1, 2, 3}, never 98.
        boxes = ([50, 50, 80, 80], [250, 50, 80, 80], [450, 50, 80, 80])
        embs = (EMB_A, EMB_B, EMB_C)
        last = {}
        for _ in range(5):
            last = self._step([(b, 0.9, e) for b, e in zip(boxes, embs)])
        ids = set(last.values())
        self.assertEqual(len(ids), 3)
        self.assertEqual(ids, {1, 2, 3})

    def test_detection_flicker_does_not_inflate_ids(self):
        # 15 isolated 1-frame flickers: tentative tracks never confirm
        # (n_init=2), so they must not consume display IDs.
        for i in range(15):
            self._step([([50 + i * 5, 50, 80, 80], 0.6, unit_emb(1000 + i))])
            for _ in range(12):  # gap long enough to delete each tentative
                self._step([])
        # Now a real person arrives — they must still get ID 1.
        id_real = self._confirm_person([300, 300, 80, 80], EMB_A)
        self.assertEqual(id_real, 1)

    def test_reacquire_window_expires(self):
        id_a = self._confirm_person([100, 100, 80, 80], EMB_A)
        for _ in range(20):
            self._step([])
        # Jump time past the re-acquisition window
        self.t += 200.0
        id_a_late = self._confirm_person([100, 100, 80, 80], EMB_A)
        self.assertNotEqual(id_a_late, id_a,
                            "IDs must not be re-acquired after the window")

    def test_registered_person_keeps_id_despite_pose_change(self):
        # The reported bug: registered person leaves ~25s, returns with a
        # DIFFERENT embedding (side profile), FAISS still recognises them.
        # Their display ID must NOT jump.
        pid = "uuid-parth"
        frontal = EMB_A
        profile = unit_emb(555)  # deliberately dissimilar to frontal
        self.assertLess(float(np.dot(frontal, profile)), 0.5)  # would fail centroid match

        id_first = self._confirm_known([100, 100, 80, 80], frontal, pid)

        for _ in range(20):  # gone long enough for DeepSORT to delete the track
            self._step([])

        # Returns as a profile view — centroid would miss, but FAISS pins it
        id_back = self._confirm_known([420, 260, 80, 80], profile, pid)
        self.assertEqual(id_back, id_first,
                         "registered person must keep their ID across pose change")

    def test_registered_id_survives_beyond_reacquire_window(self):
        pid = "uuid-parth"
        id_first = self._confirm_known([100, 100, 80, 80], EMB_A, pid)
        for _ in range(20):
            self._step([])
        self.t += 300.0  # far beyond REACQUIRE_WINDOW_SEC
        id_back = self._confirm_known([100, 100, 80, 80], EMB_A, pid)
        self.assertEqual(id_back, id_first,
                         "identity pinning has no time limit, unlike the gallery")

    def test_two_registered_people_keep_distinct_ids(self):
        ids = {}
        for _ in range(5):
            ids = self._step_known(
                [([50, 50, 80, 80], 0.9, EMB_A),
                 ([300, 50, 80, 80], 0.9, EMB_B)],
                ["uuid-a", "uuid-b"])
        self.assertEqual(len(set(ids.values())), 2)

    def test_unknown_then_recognised_heals_to_pinned_id(self):
        pid = "uuid-late"
        # First frames: FAISS hasn't matched yet (None) — track gets some ID
        last = {}
        for _ in range(4):
            last = self._step_known([([100, 100, 80, 80], 0.9, EMB_A)], [None])
        unknown_id = last[0]
        # FAISS now recognises the same continuous track
        for _ in range(3):
            last = self._step_known([([100, 100, 80, 80], 0.9, EMB_A)], [pid])
        # The pinned ID equals the one already on screen (healed in place)
        self.assertEqual(last[0], unknown_id)

    def test_missing_embedding_never_matches_gallery(self):
        id_a = self._confirm_person([100, 100, 80, 80], EMB_A)
        for _ in range(20):
            self._step([])
        # New track with NO embedding (fallback path) → must get a new ID,
        # not steal A's from the departed gallery.
        last = {}
        for _ in range(4):
            last = self._step([([100, 100, 80, 80], 0.9, None)])
        self.assertIn(0, last)
        self.assertNotEqual(last[0], id_a)


if __name__ == "__main__":
    unittest.main()
