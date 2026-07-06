"""
tests/test_pipeline_units.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for Layers 1-3 pieces that need no models:
  - Layer 1: source resolution + media-time timestamps for video files
  - Layer 2: preprocessing contract (RGB, 640x640, no normalisation)
  - Layer 3: padded face-crop extraction (padding, clamping, degenerates)

Run:  python -m unittest discover tests
"""

import os
import sys
import tempfile
import unittest

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.layer1_ingestion.capture import VideoCapture
from src.layer2_preprocessing.preprocessor import Preprocessor
from src.layer3_detection.detector import FaceDetector, CROP_PADDING_RATIO


class CaptureTests(unittest.TestCase):

    def _write_test_video(self, path, n_frames=10, fps=10.0):
        w, h = 64, 48
        writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for i in range(n_frames):
            frame = np.full((h, w, 3), i * 10, dtype=np.uint8)
            writer.write(frame)
        writer.release()

    def test_resolve_source_digit_string(self):
        cap = VideoCapture.__new__(VideoCapture)  # skip device open
        cap.source = "0"
        self.assertEqual(cap._resolve_source(), 0)
        cap.source = "video.mp4"
        self.assertEqual(cap._resolve_source(), "video.mp4")

    def test_video_file_uses_media_timestamps(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "clip.mp4")
            self._write_test_video(path, n_frames=10, fps=10.0)
            with VideoCapture(path, camera_id="test") as cap:
                self.assertFalse(cap.is_live)
                stamps = [ts for _, ts, _ in cap.frames()]
            self.assertEqual(len(stamps), 10)
            # Media time: consecutive frames exactly 1/fps apart, regardless
            # of decode speed.
            for i in range(1, len(stamps)):
                self.assertAlmostEqual(stamps[i] - stamps[i - 1], 0.1,
                                       places=6)

    def test_missing_file_raises(self):
        with self.assertRaises(RuntimeError):
            VideoCapture("definitely_not_a_file.mp4")


class PreprocessorTests(unittest.TestCase):

    def test_contract_rgb_640_uint8(self):
        pre = Preprocessor()
        # Pure-blue BGR frame → pure-blue RGB after conversion
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 0] = 255
        ctx = pre.process(frame, camera_id="cam", timestamp=1.0, frame_seq=0)

        self.assertEqual(ctx.preprocessed_frame.shape, (640, 640, 3))
        self.assertEqual(ctx.preprocessed_frame.dtype, np.uint8)  # no float cast
        self.assertEqual(ctx.original_shape, (480, 640))
        self.assertEqual(ctx.resized_shape, (640, 640))
        # Channel order swapped: blue must now be in channel 2
        self.assertEqual(int(ctx.preprocessed_frame[0, 0, 0]), 0)
        self.assertEqual(int(ctx.preprocessed_frame[0, 0, 2]), 255)
        # Original frame preserved untouched (BGR)
        self.assertEqual(int(ctx.original_frame[0, 0, 0]), 255)
        self.assertEqual(ctx.detections, [])


class PaddedCropTests(unittest.TestCase):
    """_extract_padded_crop uses no instance state — call it unbound."""

    def _crop(self, frame, bbox):
        h, w = frame.shape[:2]
        return FaceDetector._extract_padded_crop(None, frame, bbox, h, w)

    def test_padding_applied(self):
        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        bbox = [100, 100, 200, 200]  # 100x100 box
        crop, shape = self._crop(frame, bbox)
        pad = int(100 * CROP_PADDING_RATIO)
        self.assertEqual(shape, (100 + 2 * pad, 100 + 2 * pad))

    def test_clamped_at_frame_edge(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        crop, shape = self._crop(frame, [0, 0, 100, 100])
        self.assertIsNotNone(crop)
        # Padding cannot extend past (0,0) — crop starts at the frame edge
        self.assertLessEqual(shape[0], 100 + int(100 * CROP_PADDING_RATIO))

    def test_degenerate_bbox_returns_none(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        crop, shape = self._crop(frame, [150, 150, 150, 150])  # zero area
        self.assertIsNone(crop)
        self.assertIsNone(shape)

    def test_bbox_outside_frame_returns_none(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        crop, shape = self._crop(frame, [500, 500, 600, 600])
        self.assertIsNone(crop)


if __name__ == "__main__":
    unittest.main()
