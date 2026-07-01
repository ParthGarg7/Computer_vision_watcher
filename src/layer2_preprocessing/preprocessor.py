"""
src/layer2_preprocessing/preprocessor.py
─────────────────────────────────────────────────────────────────────────────
Layer 2: Preprocessing

Transforms a raw BGR frame from Layer 1 into a FrameContext object ready for
the detection model in Layer 3.

Contract for YOLOv8 (per Layer 2 Architecture Doc, Table 2):
    Color order : RGB (must convert from OpenCV's default BGR)
    Input size  : 640x640 pixels (model's fixed input dimension)
    dtype       : uint8 (not float32 — YOLO normalizes 0-255 → 0.0-1.0 internally)
    Normalization: NOT applied here — YOLOv8 handles it internally.
                  Pre-normalizing would cause double-normalization and degrade accuracy.
    Array format: HWC (height x width x channels) — YOLO accepts this directly.
    Batch dim   : Not added — YOLO handles batching internally.

CRITICAL — original_shape must be preserved.
    YOLOv8 returns bounding box coordinates in the resized frame's space (640x640).
    The only way to map those back to the original frame is via the ratio:
        x_orig = x_640 × (original_W / 640)
        y_orig = y_640 × (original_H / 640)
    This is why both original_shape and resized_shape travel in the FrameContext.

Ref: Layer 2 Architecture Doc — Sections 3.1, 3.2, 5 (Per-Model Preprocessing Contract)
"""

import cv2
import numpy as np
from src.core.frame_context import FrameContext

# YOLOv8 default input size (width, height) for cv2.resize
YOLOV8_INPUT_SIZE_WH = (640, 640)


class Preprocessor:
    """
    Layer 2 Preprocessor for YOLOv8 face detection.

    Applies the minimal transformations required by the YOLOv8 model:
        1. BGR → RGB color conversion
        2. Resize to 640x640 using INTER_AREA (best for downscaling)

    Wraps the result in a FrameContext with all metadata fields
    needed by downstream layers for coordinate scaling and visualization.
    """

    def __init__(self, model_input_size_wh: tuple = YOLOV8_INPUT_SIZE_WH):
        """
        Parameters
        ----------
        model_input_size_wh : tuple (W, H)
            Target size for cv2.resize. Default (640, 640) for YOLOv8.
            Note: cv2.resize uses (W, H) order, not (H, W).
        """
        self.model_input_size_wh = model_input_size_wh  # (W, H) for cv2.resize
        # Store as (H, W) for consistent use in shape tuples
        self.resized_shape_hw = (model_input_size_wh[1], model_input_size_wh[0])

    def process(
        self,
        original_frame: np.ndarray,
        camera_id: str,
        timestamp: float,
        frame_seq: int
    ) -> FrameContext:
        """
        Preprocess a single raw BGR frame into a FrameContext.

        Parameters
        ----------
        original_frame : np.ndarray
            Raw BGR uint8 array from Layer 1 (H, W, 3).
        camera_id : str
            Source identifier carried from Layer 1.
        timestamp : float
            Capture timestamp carried from Layer 1.
        frame_seq : int
            Frame sequence number carried from Layer 1.

        Returns
        -------
        FrameContext
            Populated with original_frame, preprocessed_frame,
            original_shape, resized_shape, and all metadata.
            detections list is empty — Layer 3 will populate it.
        """
        # Record original dimensions BEFORE any transformation.
        # This is required for correct coordinate scaling in Layer 3.
        orig_h, orig_w = original_frame.shape[:2]
        original_shape = (orig_h, orig_w)

        # Step 1 — BGR → RGB
        # OpenCV reads frames in BGR by default.
        # YOLOv8 expects RGB (trained on RGB data).
        # Swapping channels here prevents degraded detection accuracy.
        rgb_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)

        # Step 2 — Resize to 640x640
        # INTER_AREA is the correct interpolation for downscaling:
        # it averages pixel values in the source region, reducing aliasing
        # artifacts that can hurt detection at low resolution.
        # INTER_LINEAR is used for upscaling (not needed here for typical cameras).
        preprocessed = cv2.resize(
            rgb_frame,
            self.model_input_size_wh,   # (W=640, H=640) — cv2.resize order
            interpolation=cv2.INTER_AREA
        )
        # preprocessed is now: (640, 640, 3) RGB uint8
        # Ready for direct YOLO input. No float cast, no normalization.

        return FrameContext(
            original_frame=original_frame,        # Raw BGR — for drawing & cropping
            preprocessed_frame=preprocessed,      # RGB 640x640 — for model input
            original_shape=original_shape,        # (H, W) — for coord scaling
            resized_shape=self.resized_shape_hw,  # (640, 640) as (H, W)
            camera_id=camera_id,
            timestamp=timestamp,
            frame_seq=frame_seq,
            # detections starts empty — Layer 3 fills it
        )
