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
    - GPU acceleration: When onnxruntime-gpu is installed and CUDA is available,
      the ONNX session is patched to use CUDAExecutionProvider automatically.
      This drops per-frame inference from ~50-80ms (CPU) to ~2-5ms (GPU).

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
import onnxruntime as ort
from typing import Optional

from src.core.frame_context import FrameContext
from src.core.gpu_setup import register_nvidia_dlls, cuda_is_usable

# ─── Constants ────────────────────────────────────────────────────────────────

# Analyse expression every N frames per track_id (throttle).
# Lower = more responsive, higher CPU load.
# Higher = smoother CPU usage, expression updates less frequent.
EXPRESSION_EVERY_N_FRAMES = 5

# Minimum face crop dimension for reliable expression inference.
# 64 ensures the face is large enough for the model to read fine details.
MIN_CROP_SIZE = 64

# hsemotion-onnx model variants. Options:
#   'enet_b0_8_best_afew'      — EfficientNet-B0, trained on AFEW (acted video clips)
#                                Strong disgust/fear bias on natural resting faces.
#   'enet_b0_8_best_vgaf'      — EfficientNet-B0, trained on VGAF (real-world video)
#                                Better on natural/non-acted faces. Default.
#   'enet_b0_8_va_mtl'         — Multi-task learning variant (valence+arousal)
#   'enet_b2_8'                — EfficientNet-B2. DO NOT USE: despite its higher
#                                published AffectNet score, the ONNX export is
#                                miscalibrated with this package's preprocessing.
#                                Measured on real smiling faces: b0_vgaf gives
#                                Happiness at 0.98-0.999, b2_8 gives a flat
#                                ~0.33-0.37 max (reads as "uncertain") and
#                                sometimes the wrong class outright.
DEFAULT_MODEL = "enet_b0_8_best_vgaf"

# Number of past analyses to smooth over per track_id.
# Averaging probabilities across frames reduces single-frame wrong labels.
# Set to 1 to disable smoothing.
SMOOTHING_WINDOW = 5

# 8 AffectNet emotion classes in hsemotion-onnx output order (alphabetical —
# matches HSEmotionRecognizer.idx_to_class exactly). The actual mapping is
# read from the model at init; this list is only the documented default.
# WARNING: this order is model-defined. Do NOT reorder — a mismatch here
# makes neutral faces read as "disgust" and contempt read as "happy".
EMOTION_CLASSES = [
    "anger", "contempt", "disgust", "fear",
    "happiness", "neutral", "sadness", "surprise"
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
        model_name: Optional[str] = None,
        every_n_frames: int = EXPRESSION_EVERY_N_FRAMES
    ):
        """
        Parameters
        ----------
        model_name : str or None
            hsemotion-onnx model variant name. None (default) uses
            enet_b0_8_best_vgaf — empirically the most reliable variant
            (see model options comment above; enet_b2_8 is miscalibrated).
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

        # Temporal smoothing buffer: track_id → list of recent score dicts
        # Averaging probabilities across SMOOTHING_WINDOW frames reduces
        # single-frame wrong predictions (e.g. one frame of fear mid-smile).
        self._score_buffer: dict = {}

        # Register CUDA DLLs before any session is created, then check
        # whether CUDA is genuinely usable (get_available_providers() lists
        # CUDA whenever onnxruntime-gpu is installed, even when session
        # creation would silently fall back to CPU).
        register_nvidia_dlls()
        _cuda_available = cuda_is_usable()

        if model_name is None:
            model_name = DEFAULT_MODEL
        self.model_name = model_name

        print(f"  [Layer5] Loading expression model: {model_name}")
        print(f"  [Layer5] ONNX execution provider  : "
              f"{'CUDA (GPU)' if _cuda_available else 'CPU'}")
        from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
        self._model = HSEmotionRecognizer(model_name=model_name)

        # Read the class-index mapping straight from the model so labels can
        # never drift out of sync with the ONNX output order (7- and 8-class
        # variants differ). Falls back to the documented 8-class default.
        idx_to_class = getattr(self._model, "idx_to_class", None)
        if idx_to_class:
            self._class_names = [
                idx_to_class[i].lower() for i in sorted(idx_to_class)
            ]
        else:
            self._class_names = list(EMOTION_CLASSES)

        # ── GPU patch ────────────────────────────────────────────────────────
        # hsemotion-onnx does not expose a `providers` argument, so we patch
        # the internal ONNX InferenceSession to use CUDAExecutionProvider.
        # This moves EfficientNet-B0 inference onto the RTX GPU (~10x speedup).
        if _cuda_available:
            self._patch_model_to_gpu()

        print(f"  [Layer5] Expression analyser ready. "
              f"Throttle: every {every_n_frames} frames per track.\n")

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
            # Keyed by track_id. Detections without a track_id (tentative
            # tracks, first few frames) are skipped: id(det) is a fresh object
            # every frame, so keying on it defeats the throttle (inference ran
            # every frame) and leaks counter/buffer entries that
            # clear_stale_tracks can never match. The track confirms within
            # a few frames and analysis starts then.
            if track_id is None:
                continue
            gate_key = track_id

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

            # ── Temporal smoothing ────────────────────────────────────────────
            # Average probability scores over the last SMOOTHING_WINDOW analyses
            # to suppress single-frame wrong labels.
            if gate_key not in self._score_buffer:
                self._score_buffer[gate_key] = []
            self._score_buffer[gate_key].append(scores)
            # Keep only the last SMOOTHING_WINDOW entries
            self._score_buffer[gate_key] = \
                self._score_buffer[gate_key][-SMOOTHING_WINDOW:]

            # Compute averaged scores
            buf = self._score_buffer[gate_key]
            smoothed_scores = {
                cls: float(np.mean([s[cls] for s in buf if cls in s]))
                for cls in scores
            }
            # Re-normalise after averaging
            total = sum(smoothed_scores.values())
            if total > 1e-6:
                smoothed_scores = {k: v / total for k, v in smoothed_scores.items()}
            dominant = max(smoothed_scores, key=smoothed_scores.get)
            confidence = smoothed_scores[dominant]

            det.expression_scores = smoothed_scores
            det.dominant_expression = dominant
            det.expression_confidence = confidence

            # Store for carry-forward on throttled frames
            if track_id is not None:
                self._last_known[track_id] = {
                    "expression_scores": smoothed_scores,
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
            self._score_buffer.pop(k, None)

    # ─── Internal ─────────────────────────────────────────────────────────────

    def _patch_model_to_gpu(self):
        """
        Patch the hsemotion-onnx model's internal ONNX InferenceSession to use
        CUDAExecutionProvider instead of the CPU-only default.

        hsemotion-onnx creates its InferenceSession in its __init__ without
        exposing a providers argument. We locate the session attribute and
        recreate it with GPU providers. This is safe as long as hsemotion-onnx
        stores the session in a consistent attribute name.
        """
        gpu_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        # hsemotion-onnx stores the session as self._model.ort_session
        # or self._model.session depending on the version. Try both.
        patched = False
        for attr_name in ("ort_session", "session", "_session"):
            session = getattr(self._model, attr_name, None)
            if session is not None and hasattr(session, "get_inputs"):
                try:
                    # Re-create the session with GPU providers using the same model path
                    model_path = session._model_path if hasattr(session, "_model_path") else None
                    if model_path is None:
                        # Fallback: read path from session metadata
                        model_path = getattr(session, "model_path", None)
                    if model_path:
                        new_session = ort.InferenceSession(
                            model_path,
                            providers=gpu_providers
                        )
                        actual = new_session.get_providers()
                        # ORT silently drops CUDA and falls back to CPU when
                        # provider DLLs fail to load — verify, don't assume.
                        if "CUDAExecutionProvider" in actual:
                            setattr(self._model, attr_name, new_session)
                            print(f"  [Layer5] GPU patch applied ({attr_name}): {actual}")
                        else:
                            print(f"  [Layer5] GPU patch fell back to CPU "
                                  f"({attr_name}): {actual} — CUDA provider "
                                  f"failed to initialise. Keeping original session.")
                        patched = True
                        break
                except Exception as e:
                    print(f"  [Layer5] GPU patch failed on {attr_name}: {e}")
                    break

        if not patched:
            # Try finding any ONNX session attribute by inspecting the model object
            for attr_name, val in vars(self._model).items():
                if hasattr(val, "get_inputs") and hasattr(val, "run"):
                    try:
                        # Try to get model path from session
                        inner = val
                        mp = getattr(inner, "_model_path", None) or getattr(inner, "model_path", None)
                        if mp:
                            new_session = ort.InferenceSession(mp, providers=gpu_providers)
                            actual = new_session.get_providers()
                            if "CUDAExecutionProvider" in actual:
                                setattr(self._model, attr_name, new_session)
                                print(f"  [Layer5] GPU patch applied ({attr_name}): {actual}")
                            else:
                                print(f"  [Layer5] GPU patch fell back to CPU "
                                      f"({attr_name}): {actual} — keeping original session.")
                            patched = True
                            break
                    except Exception:
                        pass

        if not patched:
            print("  [Layer5] GPU patch: could not locate ONNX session — running on CPU.")
            print("  [Layer5]   (This is harmless; expression model will use CPU fallback.)")

    def _run_inference(
        self,
        face_crop: np.ndarray
    ) -> tuple:
        """
        Run hsemotion-onnx inference on a BGR face crop.

        The crop is converted to RGB before inference: hsemotion's preprocess
        normalises channels with ImageNet RGB statistics, so feeding OpenCV's
        native BGR order swaps the red/blue channels and degrades accuracy.
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
            # No CLAHE / histogram equalisation here: the model was trained on
            # natural (unequalised) AffectNet images, so contrast manipulation
            # shifts the input away from the training distribution and hurts
            # accuracy. It was originally added to compensate for the BGR/label
            # bugs, both now fixed.

            # hsemotion expects RGB (ImageNet normalisation is per-RGB-channel)
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

            # predict_emotions returns (dominant_label, probabilities_array)
            dominant_label, probs = self._model.predict_emotions(
                face_rgb, logits=False
            )

            if probs is None or len(probs) == 0:
                return None, None, None

            probs = np.asarray(probs, dtype=np.float32)

            # MTL variants append 2 extra outputs (valence, arousal) after the
            # emotion logits — keep only the emotion class probabilities.
            probs = probs[:len(self._class_names)]

            # Build probability dict — class names to probability values
            # Normalise to sum=1.0 (model output should already be softmax,
            # but we normalise defensively).
            probs_sum = float(probs.sum())
            if probs_sum < 1e-6:
                return None, None, None
            probs_norm = probs / probs_sum

            # Map to class names using the model's own index→class order
            scores = {
                self._class_names[i]: float(probs_norm[i])
                for i in range(len(probs_norm))
            }

            dominant = max(scores, key=scores.get)
            confidence = scores[dominant]

            return scores, dominant, confidence

        except Exception as e:
            # Graceful fallback — expression failure should never crash the
            # pipeline. Warn once so failures aren't silently invisible.
            if not getattr(self, "_inference_error_reported", False):
                self._inference_error_reported = True
                print(f"  [Layer5] WARNING: expression inference failed "
                      f"({type(e).__name__}: {e}). Further errors suppressed.")
            return None, None, None
