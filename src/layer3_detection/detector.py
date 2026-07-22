"""
src/layer3_detection/detector.py
─────────────────────────────────────────────────────────────────────────────
Layer 3: Face Detection

Receives the FrameContext from Layer 2, runs YOLOv8n-face inference on the
preprocessed frame, scales bounding box coordinates back to original frame
space, extracts padded face crops, and enriches the FrameContext with a
populated detections list.

Model: yolov8n-face.pt (arnabdhar/YOLOv8-Face-Detection on Hugging Face)
    - Task: detect (detection-only, no landmark head in this checkpoint)
    - Landmarks: None for this checkpoint — will be available when a
      pose-capable checkpoint (e.g. deepcam-ru/yolov8-face) is used.
      The code handles both cases gracefully.
    - Normalization: handled internally by Ultralytics (0-255 → 0.0-1.0)
    - NMS: handled internally
    - Device: auto-detects CUDA (RTX 5060) or falls back to CPU

COORDINATE SCALING (most critical step):
    YOLOv8 returns bounding boxes in the resized frame's coordinate space
    (640x640 input). These must be mapped back to original frame coordinates
    before drawing, cropping, or passing downstream.

    x_original = x_resized × (original_W / resized_W)
    y_original = y_resized × (original_H / resized_H)

    If this step is skipped, boxes appear shifted/scaled incorrectly on
    the full-resolution output frame. This is the most common Layer 3 bug.

Ref: Layer 3 Architecture Doc — Section 7: The Coordinate Scaling Problem
Ref: Layer 3 Architecture Doc — Section 3.5: The Face Crop
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.core.frame_context import FrameContext, Detection
from src.core.logger import get_logger

log = get_logger("watcher.layer3")

# ─── Constants ────────────────────────────────────────────────────────────────

DEFAULT_MODEL_PATH = "models/yolov8n-face.pt"
DEFAULT_CONFIDENCE = 0.5

# Padding applied around each face crop (fraction of box dimension per side).
# 15% is the midpoint of the 10-20% range recommended in the Layer 3 doc.
CROP_PADDING_RATIO = 0.15


class FaceDetector:
    """
    Layer 3 Face Detector using YOLOv8n-face.

    Loads the face detection model once at initialization, then runs
    inference on each FrameContext's preprocessed_frame, scales coordinates
    back to original frame space, extracts padded face crops, and returns
    the enriched FrameContext with detections populated.

    GPU is auto-detected and used if available (RTX 5060 will be used).
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        confidence_threshold: float = DEFAULT_CONFIDENCE,
        device: str = None
    ):
        """
        Parameters
        ----------
        model_path : str
            Path to yolov8n-face.pt checkpoint.
        confidence_threshold : float
            Minimum confidence score (0.0-1.0) for a detection to be kept.
            Below this threshold detections are discarded before passing
            downstream. Default 0.5 — tune lower (0.3) for harder detection
            scenarios, higher (0.7) to reduce false positives.
        device : str or None
            'cuda', 'cpu', or None (auto-detect). Auto-detect is preferred.
        """
        # Device selection
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.confidence_threshold = confidence_threshold

        log.info(f"  [Layer3] Loading: {model_path}")
        device_line = f"  [Layer3] Device : {self.device.upper()}"
        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            device_line += f" ({gpu_name}, {gpu_mem:.1f}GB)"
        log.info(device_line)

        self.model = YOLO(model_path)
        self.model.to(self.device)

        # Detect whether this checkpoint has a landmark/keypoint head
        # DetectionModel = detection only (no landmarks)
        # PoseModel or KeypointModel = landmarks available
        model_type = type(self.model.model).__name__
        self._has_landmark_head = "Pose" in model_type or "Keypoint" in model_type

        if not self._has_landmark_head:
            log.info(
                f"  [Layer3] Checkpoint type: {model_type} — "
                f"landmark output NOT available for this checkpoint.\n"
                f"  [Layer3] landmarks_original will be None. "
                f"Swap to a pose-capable checkpoint for 5-point landmarks (Layer 4 prep)."
            )
        else:
            log.info(f"  [Layer3] Checkpoint type: {model_type} — 5-point landmarks AVAILABLE.")

        log.info(f"  [Layer3] Confidence threshold: {self.confidence_threshold}")
        log.info(f"  [Layer3] Ready.\n")

    # ─── Public API ───────────────────────────────────────────────────────────

    def detect(self, ctx: FrameContext) -> FrameContext:
        """
        Run face detection on the preprocessed frame in the FrameContext.

        Enriches ctx.detections with one Detection object per face found.
        Zero faces is a valid result — ctx.detections will be an empty list.

        Parameters
        ----------
        ctx : FrameContext
            Must have preprocessed_frame (640x640 RGB uint8),
            original_shape (H, W), and resized_shape (640, 640) set.

        Returns
        -------
        FrameContext
            Same object with detections list populated.
        """
        orig_h, orig_w = ctx.original_shape
        resized_h, resized_w = ctx.resized_shape

        # Scale factors for mapping resized coords → original frame coords
        scale_x = orig_w / resized_w   # e.g. 1920/640 = 3.0
        scale_y = orig_h / resized_h   # e.g. 1080/640 = 1.6875

        # Run YOLOv8 inference on the preprocessed (RGB 640x640) frame.
        # conf= applies the threshold internally (no post-filter needed).
        # verbose=False suppresses per-frame console spam.
        results = self.model(
            ctx.preprocessed_frame,
            conf=self.confidence_threshold,
            verbose=False,
            device=self.device
        )

        detections = []

        if results and len(results) > 0:
            result = results[0]

            # Check keypoint availability for this result
            has_kp = (
                self._has_landmark_head
                and result.keypoints is not None
                and result.keypoints.data is not None
                and len(result.keypoints.data) > 0
            )

            if result.boxes is not None and len(result.boxes) > 0:
                for i, box in enumerate(result.boxes):
                    # ── Confidence ──────────────────────────────────────────
                    conf = float(box.conf[0].cpu().numpy())

                    # ── Bounding box in resized (640x640) space ─────────────
                    x1_r, y1_r, x2_r, y2_r = box.xyxy[0].cpu().numpy().tolist()
                    bbox_resized = [x1_r, y1_r, x2_r, y2_r]

                    # ── Scale back to original frame space ──────────────────
                    # This is the coordinate scaling step described in Layer 3
                    # doc Section 7. Skipping this is the #1 Layer 3 bug.
                    x1_o = x1_r * scale_x
                    y1_o = y1_r * scale_y
                    x2_o = x2_r * scale_x
                    y2_o = y2_r * scale_y
                    bbox_original = [x1_o, y1_o, x2_o, y2_o]

                    # ── Landmarks (5-point) ──────────────────────────────────
                    # Only available if the checkpoint has a landmark head.
                    # Landmarks also need scaling from resized to original space.
                    landmarks_original = None
                    if has_kp and i < len(result.keypoints.data):
                        kp_tensor = result.keypoints.data[i].cpu().numpy()  # (5, 2 or 3)
                        landmarks_original = []
                        for kp in kp_tensor:
                            kx = float(kp[0]) * scale_x
                            ky = float(kp[1]) * scale_y
                            kconf = float(kp[2]) if kp.shape[0] > 2 else 1.0
                            landmarks_original.append((kx, ky, kconf))

                    # ── Face crop with padding ───────────────────────────────
                    # Crop taken from original_frame (full resolution),
                    # NOT from the resized model input.
                    face_crop, face_crop_shape = self._extract_padded_crop(
                        ctx.original_frame,
                        bbox_original,
                        orig_h, orig_w
                    )

                    detections.append(Detection(
                        bbox_original=bbox_original,
                        bbox_resized=bbox_resized,
                        confidence=conf,
                        landmarks_original=landmarks_original,
                        face_crop=face_crop,
                        face_crop_shape=face_crop_shape
                    ))

        ctx.detections = detections
        return ctx

    # ─── Internal ─────────────────────────────────────────────────────────────

    def _extract_padded_crop(
        self,
        original_frame: np.ndarray,
        bbox: list,
        frame_h: int,
        frame_w: int
    ) -> tuple:
        """
        Crop the face region from the original frame with padding.

        Padding is 15% of the bounding box dimensions on each side.
        Coordinates are clamped to frame boundaries to prevent out-of-bounds
        array access (which would cause a silent crop failure).

        Parameters
        ----------
        original_frame : np.ndarray  BGR (H, W, 3) uint8
        bbox : list  [x1, y1, x2, y2] in original frame pixel coordinates
        frame_h, frame_w : int  dimensions of original_frame

        Returns
        -------
        (face_crop, face_crop_shape)
            face_crop: np.ndarray BGR uint8 or None if crop is empty
            face_crop_shape: tuple (H, W) or None
        """
        x1, y1, x2, y2 = bbox
        box_w = x2 - x1
        box_h = y2 - y1

        # Add padding on each side
        pad_x = box_w * CROP_PADDING_RATIO
        pad_y = box_h * CROP_PADDING_RATIO

        # Clamp to valid frame indices
        cx1 = max(0, int(x1 - pad_x))
        cy1 = max(0, int(y1 - pad_y))
        cx2 = min(frame_w - 1, int(x2 + pad_x))
        cy2 = min(frame_h - 1, int(y2 + pad_y))

        # Safety check: crop must have positive dimensions
        if cx2 <= cx1 or cy2 <= cy1:
            return None, None

        face_crop = original_frame[cy1:cy2, cx1:cx2]
        return face_crop, face_crop.shape[:2]  # (H, W)
