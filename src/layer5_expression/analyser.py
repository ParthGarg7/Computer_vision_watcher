"""
src/layer5_expression/analyser.py
─────────────────────────────────────────────────────────────────────────────
Layer 5: Expression Analysis

Estimates visible facial expression (emotion class probabilities) from the
face crop produced by Layer 3 and identity-enriched by Layer 4.

Technology: hsemotion-onnx
    - ONNX-exported MobileNet/EfficientNet trained on AffectNet (~460K images,
      8 emotion classes: neutral, happy, sad, surprise, fear, disgust, anger,
      contempt).
    - Pure ONNX Runtime inference — no TensorFlow, no Keras, no PyTorch.
    - Compatible with Python 3.14+.
    - Runs fully offline. ONNX model weights auto-downloaded on first use to
      ~/.hsemotion directory (one-time, ~50 MB). Flagged as offline-first.
    - 8-class output vs. DeepFace's 7-class (adds 'contempt').

Why hsemotion-onnx over DeepFace (original plan):
    DeepFace requires TensorFlow as a backend. TensorFlow does not yet support
    Python 3.14 (available wheels max out at 3.12). The project venv uses
    Python 3.14.3, making DeepFace installation impossible without downgrading
    Python. hsemotion-onnx is the highest-quality Python 3.14-compatible
    alternative — it uses ONNX Runtime which has Python 3.14 wheels.

Why hsemotion-onnx over FER library:
    FER also has Python 3.14 build failures (Pillow wheel build error).
    hsemotion-onnx has pre-built ONNX models that bypass any source compilation.

Why hsemotion-onnx over custom PyTorch model:
    Custom model requires labelled training data and training infrastructure.
    hsemotion-onnx provides out-of-the-box AffectNet-trained performance
    suitable for MVP validation before custom training is invested in.

Throttling:
    Expression inference is NOT run on every frame. Running it per-frame
    at 30 FPS would create a bottleneck (~50-150ms per crop on CPU).
    Instead, each track_id is analysed every EXPRESSION_EVERY_N_FRAMES frames.
    Between analyses, the last known expression is carried forward on the
    Detection object (sticky labels). Configurable via constructor.

Output:
    det.expression_scores       — dict of 8 class→probability (sums to 1.0)
    det.dominant_expression     — argmax class name
    det.expression_confidence   — probability of dominant class

Ref: Layer 5 Architecture Doc — Sections 1, 2, 3, 4.1, 6
"""

import cv2
import numpy as np
from typing import Optional

from src.core.frame_context import FrameContext

# ─── Constants ────────────────────────────────────────────────────────────────

# Analyse expression every N frames per track_id (throttle).
# Lower = more responsive, higher CPU load.
# Higher = smoother CPU usage, expression updates less frequent.
EXPRESSION_EVERY_N_FRAMES = 5

# Minimum face crop dimension for reliable expression inference.
MIN_CROP_SIZE = 48

# hsemotion-onnx model variant. Options:
#   'enet_b0_8_best_afew'      — EfficientNet-B0, 8 classes, best on AFEW (video)
#   'enet_b0_8_best_vgaf'      — EfficientNet-B0, 8 classes, best on VGAF
#   'enet_b0_8_va_mtl'         — Multi-task learning variant
#   'enet_b2_8'                — EfficientNet-B2, 8 classes (slower, more accurate)
DEFAULT_MODEL = "enet_b0_8_best_afew"

# 8 AffectNet emotion classes in hsemotion-onnx output order
EMOTION_CLASSES = [
    "neutral", "happy", "sad", "surprise",
    "fear", "disgust", "anger", "contempt"
]


class ExpressionAnalyser:
    """
    Layer 5 Expression Analyser using hsemotion-onnx.

    Runs ONNX-based emotion classification on face crops from Layer 3/4,
    throttled per track_id to control CPU usage.

    Sticky labels: if a track is not analysed this frame (throttle gate),
    the detection's expression fields are left as None. The pipeline's
    drawing code should display the last known expression for that track_id
    (carry-forward logic is in main.py's display layer).

    Usage
    -----
        analyser = ExpressionAnalyser()
        # Per frame:
        ctx = analyser.analyse(ctx)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        every_n_frames: int = EXPRESSION_EVERY_N_FRAMES
    ):
        """
        Parameters
        ----------
        model_name : str
            hsemotion-onnx model variant name.
        every_n_frames : int
            Analyse expression every this many frames per track_id.
            Default 5 — tune based on target latency and CPU budget.
        """
        self.every_n_frames = every_n_frames

        # Per-track_id frame counter for throttling
        # Maps track_id (int) → frame count since last analysis
        self._frame_counter: dict = {}

        # Sticky carry-forward storage: track_id → last known result dict
        # {expression_scores, dominant_expression, expression_confidence}
        self._last_known: dict = {}

        print(f"  [Layer5] Loading expression model: {model_name}")
        from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
        self._model = HSEmotionRecognizer(model_name=model_name)
        print(f"  [Layer5] Expression analyser ready. "
              f"Throttle: every {every_n_frames} frames per track.\\n")

    # ─── Public API ───────────────────────────────────────────────────────────

    def analyse(self, ctx: FrameContext) -> FrameContext:
        """
        Run expression analysis on all detections in a FrameContext.

        For each detection:
            - If track_id is None or throttle gate blocks → expression fields
              remain None (caller can fall back to sticky from last_known).
            - If throttle gate passes → run inference, update expression fields
              and store result in _last_known[track_id].

        Parameters
        ----------
        ctx : FrameContext
            Must have ctx.detections populated by Layers 3 and 4.
            det.face_crop (BGR crop from L3) is the primary input.
            det.track_id (from L4 DeepSORT) is used for throttle gating.

        Returns
        -------
        FrameContext
            Same object with Layer 5 expression fields populated where analysis ran.
        """
        for det in ctx.detections:
            track_id = det.track_id
            face_crop = det.face_crop

            if face_crop is None:
                continue

            h, w = face_crop.shape[:2]
            if h < MIN_CROP_SIZE or w < MIN_CROP_SIZE:
                continue

            # ── Throttle gate ─────────────────────────────────────────────────
            # Use track_id as throttle key. If no track_id (tentative track),
            # use a fallback key based on bbox to avoid running every frame.
            gate_key = track_id if track_id is not None else id(det)

            count = self._frame_counter.get(gate_key, 0)
            self._frame_counter[gate_key] = count + 1

            if count % self.every_n_frames != 0:
                # Throttled — carry forward last known result if available
                if track_id is not None and track_id in self._last_known:
                    last = self._last_known[track_id]
                    det.expression_scores = last["expression_scores"]
                    det.dominant_expression = last["dominant_expression"]
                    det.expression_confidence = last["expression_confidence"]
                continue

            # ── Run expression inference ──────────────────────────────────────
            scores, dominant, confidence = self._run_inference(face_crop)

            if scores is None:
                continue

            det.expression_scores = scores
            det.dominant_expression = dominant
            det.expression_confidence = confidence

            # Store for carry-forward on throttled frames
            if track_id is not None:
                self._last_known[track_id] = {
                    "expression_scores": scores,
                    "dominant_expression": dominant,
                    "expression_confidence": confidence,
                }

        return ctx

    def clear_stale_tracks(self, active_track_ids: set):
        """
        Remove throttle state for tracks that are no longer active.

        Call periodically (e.g. every 100 frames) to prevent memory growth
        from accumulating state for ended tracks.

        Parameters
        ----------
        active_track_ids : set of int
            Set of currently confirmed track IDs from DeepSORT.
        """
        stale = [k for k in self._frame_counter if k not in active_track_ids]
        for k in stale:
            self._frame_counter.pop(k, None)
            self._last_known.pop(k, None)

    # ─── Internal ─────────────────────────────────────────────────────────────

    def _run_inference(
        self,
        face_crop: np.ndarray
    ) -> tuple:
        """
        Run hsemotion-onnx inference on a BGR face crop.

        hsemotion-onnx accepts BGR input (same as InsightFace).
        Returns (expression_scores_dict, dominant_expression, confidence).
        Returns (None, None, None) on any failure.

        Parameters
        ----------
        face_crop : np.ndarray  BGR uint8 (H, W, 3)

        Returns
        -------
        (dict, str, float) or (None, None, None)
        """
        try:
            # predict_emotions returns (dominant_label, probabilities_array)
            dominant_label, probs = self._model.predict_emotions(
                face_crop, logits=False
            )

            if probs is None or len(probs) == 0:
                return None, None, None

            probs = np.asarray(probs, dtype=np.float32)

            # Build probability dict — class names to probability values
            # Normalise to sum=1.0 (model output should already be softmax,
            # but we normalise defensively).
            probs_sum = float(probs.sum())
            if probs_sum < 1e-6:
                return None, None, None
            probs_norm = probs / probs_sum

            # Map to class names (use available classes from model output length)
            n_classes = min(len(probs_norm), len(EMOTION_CLASSES))
            scores = {
                EMOTION_CLASSES[i]: float(probs_norm[i])
                for i in range(n_classes)
            }

            dominant = max(scores, key=scores.get)
            confidence = scores[dominant]

            return scores, dominant, confidence

        except Exception as e:
            # Graceful fallback — expression failure should never crash the pipeline
            return None, None, None
