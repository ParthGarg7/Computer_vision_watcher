"""
tests/test_embedder_context.py
─────────────────────────────────────────────────────────────────────────────
Regression tests for the SCRFD-context bug in Layer 4's FaceEmbedder.

The bug: Layer 3 produces a face crop padded by only 15%, so the face fills
almost the whole crop. SCRFD is trained on scenes where a face occupies a
FRACTION of the frame, so it detected ZERO faces in a perfectly clear crop —
get_embedding returned None on every frame, identity never ran, and because
identity pinning needs a person_id, track IDs drifted too. It went unnoticed
for weeks because the failure rendered as silence.

Measured on a real 229x256 crop at YOLO confidence 0.90:
    tight crop        -> 0 faces  -> no identity
    crop + 50% border -> 1 face   -> matches the registered face at 0.77

These tests use a fake InsightFace app (no 500 MB model download, no GPU) and
assert on the FRAMING of what the detector receives.

Run:  python -m unittest discover tests
"""

import os
import sys
import unittest

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.layer4_identity.embedder import FaceEmbedder, MIN_CROP_SIZE


class _FakeFace:
    def __init__(self, dim=512):
        self.det_score = 0.9
        rng = np.random.RandomState(7)
        v = rng.randn(dim).astype(np.float32)
        self.normed_embedding = v / np.linalg.norm(v)
        self.kps = np.array([[30, 40], [70, 40], [50, 60], [35, 80], [65, 80]],
                            dtype=np.float32)


class _FakeApp:
    """Records every image handed to get(), and can simulate detection failure."""

    def __init__(self, find_face=True):
        self.find_face = find_face
        self.seen_shapes = []
        self.calls = 0

    def get(self, img):
        self.calls += 1
        self.seen_shapes.append(img.shape[:2])
        return [_FakeFace()] if self.find_face else []


def make_embedder(find_face=True):
    """Build a FaceEmbedder without running __init__ (no InsightFace load)."""
    e = FaceEmbedder.__new__(FaceEmbedder)
    e._app = _FakeApp(find_face=find_face)
    e._embed_warned = False
    e._device_id = -1
    return e


def crop(h=120, w=100):
    return np.zeros((h, w, 3), dtype=np.uint8)


class EmbedderContextTests(unittest.TestCase):

    def test_detector_receives_padded_not_tight_crop(self):
        # THE regression guard: whatever reaches SCRFD must be strictly larger
        # than the crop, or the face fills the frame and detection fails.
        e = make_embedder()
        e.get_embedding(crop(h=120, w=100))
        self.assertEqual(len(e._app.seen_shapes), 1)
        seen_h, seen_w = e._app.seen_shapes[0]
        self.assertGreater(seen_h, 120, "crop must be padded before detection")
        self.assertGreater(seen_w, 100, "crop must be padded before detection")

    def test_padding_is_50_percent_each_side(self):
        e = make_embedder()
        e.get_embedding(crop(h=120, w=100))
        seen_h, seen_w = e._app.seen_shapes[0]
        # h//2 top + h//2 bottom => 2x height; same for width
        self.assertEqual((seen_h, seen_w), (120 + 120, 100 + 100))

    def test_single_detector_call_no_wasted_retry(self):
        # The old code called SCRFD on the tight crop, then AGAIN on a useless
        # upscale — two failing calls per face per frame. One call now.
        e = make_embedder()
        e.get_embedding(crop())
        self.assertEqual(e._app.calls, 1)

    def test_successful_embedding_shape_and_norm(self):
        e = make_embedder()
        emb, aligned = e.get_embedding(crop())
        self.assertIsNotNone(emb)
        self.assertEqual(emb.shape, (512,))
        self.assertEqual(emb.dtype, np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(emb)), 1.0, places=4)

    def test_detection_failure_returns_none_and_warns_once(self):
        e = make_embedder(find_face=False)
        emb, aligned = e.get_embedding(crop())
        self.assertIsNone(emb)
        self.assertIsNone(aligned)
        self.assertTrue(e._embed_warned, "failure must warn, never be silent")

    def test_none_crop_is_safe(self):
        e = make_embedder()
        self.assertEqual(e.get_embedding(None), (None, None))
        self.assertEqual(e._app.calls, 0)

    def test_tiny_crop_skipped_before_detection(self):
        e = make_embedder()
        emb, _ = e.get_embedding(crop(h=MIN_CROP_SIZE - 1, w=MIN_CROP_SIZE - 1))
        self.assertIsNone(emb)
        self.assertEqual(e._app.calls, 0)

    def test_exception_is_caught_and_warned(self):
        e = make_embedder()

        class Boom:
            def get(self, img):
                raise RuntimeError("cuda exploded")

        e._app = Boom()
        emb, _ = e.get_embedding(crop())
        self.assertIsNone(emb)
        self.assertTrue(e._embed_warned)


if __name__ == "__main__":
    unittest.main()
