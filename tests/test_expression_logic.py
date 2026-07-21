"""
tests/test_expression_logic.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for Layer 5's ExpressionAnalyser logic — label mapping, RGB
conversion, throttling, carry-forward, and smoothing — WITHOUT loading the
ONNX model. The analyser is built via __new__ with a fake model injected,
which keeps the tests fast and offline while exercising the exact code
paths that produced the historical wrong-label bugs.

Run:  python -m unittest discover tests
"""

import os
import sys
import unittest

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.layer5_expression.analyser import ExpressionAnalyser, EMOTION_CLASSES


class FakeModel:
    """Mimics HSEmotionRecognizer.predict_emotions for a fixed output."""

    def __init__(self, probs):
        self.probs = np.asarray(probs, dtype=np.float32)
        self.last_input = None
        self.calls = 0

    def predict_emotions(self, img, logits=False):
        self.last_input = img
        self.calls += 1
        idx = int(np.argmax(self.probs))
        return EMOTION_CLASSES[idx].capitalize(), self.probs.copy()


def make_analyser(probs, every_n_frames=5):
    """Build an ExpressionAnalyser without running __init__ (no ONNX load)."""
    a = ExpressionAnalyser.__new__(ExpressionAnalyser)
    a.every_n_frames = every_n_frames
    a._frame_counter = {}
    a._last_known = {}
    a._score_buffer = {}
    a._class_names = list(EMOTION_CLASSES)
    a._model = FakeModel(probs)
    return a


class FakeDet:
    def __init__(self, tid, crop):
        self.track_id = tid
        self.face_crop = crop
        self.expression_scores = None
        self.dominant_expression = None
        self.expression_confidence = None
        self.expression_is_fresh = False


class FakeCtx:
    def __init__(self, dets):
        self.detections = dets


def crop(h=128, w=128):
    return np.zeros((h, w, 3), dtype=np.uint8)


# Model output order is alphabetical: anger, contempt, disgust, fear,
# happiness, neutral, sadness, surprise (index 5 = neutral).
NEUTRAL_PROBS = [0.02, 0.02, 0.02, 0.02, 0.05, 0.80, 0.05, 0.02]


class ExpressionLogicTests(unittest.TestCase):

    def test_label_mapping_uses_model_order(self):
        # The historical bug: index 5 (neutral) was labelled 'disgust'.
        a = make_analyser(NEUTRAL_PROBS, every_n_frames=1)
        ctx = FakeCtx([FakeDet(1, crop())])
        a.analyse(ctx)
        self.assertEqual(ctx.detections[0].dominant_expression, "neutral")
        self.assertAlmostEqual(
            ctx.detections[0].expression_scores["neutral"], 0.80, places=4)

    def test_scores_sum_to_one(self):
        a = make_analyser(NEUTRAL_PROBS, every_n_frames=1)
        ctx = FakeCtx([FakeDet(1, crop())])
        a.analyse(ctx)
        self.assertAlmostEqual(
            sum(ctx.detections[0].expression_scores.values()), 1.0, places=5)

    def test_bgr_converted_to_rgb_before_model(self):
        # Pure-blue BGR crop (255,0,0 per pixel) must reach the model as
        # pure-blue RGB (0,0,255 per pixel).
        a = make_analyser(NEUTRAL_PROBS, every_n_frames=1)
        blue_bgr = np.zeros((128, 128, 3), dtype=np.uint8)
        blue_bgr[:, :, 0] = 255
        a.analyse(FakeCtx([FakeDet(1, blue_bgr)]))
        seen = a._model.last_input
        self.assertEqual(int(seen[0, 0, 0]), 0)
        self.assertEqual(int(seen[0, 0, 2]), 255)

    def test_throttle_runs_every_n_frames(self):
        a = make_analyser(NEUTRAL_PROBS, every_n_frames=5)
        for _ in range(10):
            a.analyse(FakeCtx([FakeDet(1, crop())]))
        self.assertEqual(a._model.calls, 2)  # frames 0 and 5

    def test_carry_forward_on_throttled_frames(self):
        a = make_analyser(NEUTRAL_PROBS, every_n_frames=5)
        a.analyse(FakeCtx([FakeDet(1, crop())]))       # frame 0: inference
        det = FakeDet(1, crop())
        a.analyse(FakeCtx([det]))                      # frame 1: throttled
        self.assertEqual(det.dominant_expression, "neutral")  # carried

    def test_freshness_flag_marks_only_real_inferences(self):
        # Without this flag, Layer 6 recorded one measurement N times:
        # inflated dominant_counts, an N-fold larger events table, and a
        # trend window full of duplicates.
        a = make_analyser(NEUTRAL_PROBS, every_n_frames=5)
        fresh_flags = []
        for _ in range(10):
            det = FakeDet(1, crop())
            a.analyse(FakeCtx([det]))
            fresh_flags.append(det.expression_is_fresh)
        # Frames 0 and 5 ran inference; the rest carried forward
        self.assertEqual(fresh_flags,
                         [True, False, False, False, False,
                          True, False, False, False, False])
        self.assertEqual(a._model.calls, 2)

    def test_carried_forward_detection_still_has_label_for_display(self):
        # Freshness must NOT blank the label — the UI draws every frame.
        a = make_analyser(NEUTRAL_PROBS, every_n_frames=5)
        a.analyse(FakeCtx([FakeDet(1, crop())]))
        det = FakeDet(1, crop())
        a.analyse(FakeCtx([det]))
        self.assertFalse(det.expression_is_fresh)
        self.assertIsNotNone(det.expression_scores)
        self.assertEqual(det.dominant_expression, "neutral")

    def test_untracked_detection_skipped(self):
        a = make_analyser(NEUTRAL_PROBS, every_n_frames=1)
        det = FakeDet(None, crop())
        a.analyse(FakeCtx([det]))
        self.assertIsNone(det.dominant_expression)
        self.assertEqual(a._model.calls, 0)

    def test_small_crop_skipped(self):
        a = make_analyser(NEUTRAL_PROBS, every_n_frames=1)
        det = FakeDet(1, crop(h=32, w=32))  # below MIN_CROP_SIZE
        a.analyse(FakeCtx([det]))
        self.assertIsNone(det.dominant_expression)
        self.assertEqual(a._model.calls, 0)

    def test_mtl_extra_outputs_sliced(self):
        # MTL variants append valence/arousal after the 8 class outputs —
        # they must not appear in the score dict or corrupt normalisation.
        probs_with_va = NEUTRAL_PROBS + [0.31, -0.12]
        a = make_analyser(probs_with_va, every_n_frames=1)
        ctx = FakeCtx([FakeDet(1, crop())])
        a.analyse(ctx)
        scores = ctx.detections[0].expression_scores
        self.assertEqual(set(scores.keys()), set(EMOTION_CLASSES))
        self.assertAlmostEqual(sum(scores.values()), 1.0, places=5)

    def test_smoothing_averages_across_analyses(self):
        a = make_analyser(NEUTRAL_PROBS, every_n_frames=1)
        a.analyse(FakeCtx([FakeDet(1, crop())]))
        # Switch the fake model to a happy output; smoothed result should
        # sit between the two, not jump instantly.
        happy = [0.02, 0.02, 0.02, 0.02, 0.80, 0.05, 0.05, 0.02]
        a._model = FakeModel(happy)
        det = FakeDet(1, crop())
        a.analyse(FakeCtx([det]))
        s = det.expression_scores
        self.assertGreater(s["happiness"], 0.3)
        self.assertGreater(s["neutral"], 0.3)

    def test_clear_stale_tracks(self):
        a = make_analyser(NEUTRAL_PROBS, every_n_frames=1)
        a.analyse(FakeCtx([FakeDet(1, crop()), FakeDet(2, crop())]))
        a.clear_stale_tracks(active_track_ids={2})
        self.assertNotIn(1, a._frame_counter)
        self.assertNotIn(1, a._last_known)
        self.assertIn(2, a._frame_counter)


if __name__ == "__main__":
    unittest.main()
