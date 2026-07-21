"""
src/core/frame_context.py
─────────────────────────────────────────────────────────────────────────────
Shared data structures that travel through every layer of the pipeline.

The FrameContext object is the single container passed from Layer 1 → Layer 2
→ Layer 3 (and beyond). Each layer reads the fields it needs and adds its own
output fields. No layer replaces the object — it always accumulates.

Ref: Layer 2 Architecture Doc — Section 6: Output: The Frame Context Object
Ref: Layer 3 Architecture Doc — Section 9: Exact Output
Ref: Layer 4 Architecture Doc — Section 10: Exact Output
Ref: Layer 5 Architecture Doc — Section 3: Exact Output
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Detection:
    """
    A single face detection result produced by Layer 3, enriched by Layers 4 and 5.

    Layer 3 (Detection) populates:
        bbox_original, bbox_resized, confidence, landmarks_original,
        face_crop, face_crop_shape

    Layer 4 (Identity) adds:
        track_id, identity_label, embedding, similarity_score,
        is_known, aligned_face

    Layer 5 (Expression) adds:
        expression_scores, dominant_expression, expression_confidence

    Fields
    ------
    bbox_original : list[float]
        Bounding box [x1, y1, x2, y2] in ORIGINAL frame pixel coordinates,
        after scaling back from the model's resized input space.
        Use these for drawing on original_frame and for downstream layers.

    bbox_resized : list[float]
        Bounding box [x1, y1, x2, y2] in the RESIZED model input space
        (e.g. 640x640). Preserved for reference and debugging.

    confidence : float
        Model confidence score 0.0-1.0. Already filtered by threshold.

    landmarks_original : list[tuple] or None
        5-point facial landmarks in original frame space:
            [(lx, ly, lconf), ...]
        Order: left_eye, right_eye, nose, mouth_left, mouth_right
        None if the loaded checkpoint does not include a landmark head.

    face_crop : np.ndarray or None
        Cropped face region from original_frame (BGR, uint8) with a 15%
        padding margin applied on each side and clamped to frame bounds.
        Used by Layer 4 (InsightFace alignment + ArcFace embedding).

    face_crop_shape : tuple or None
        (H, W) of the face_crop array after padding.

    ── Layer 4 fields (populated by FaceIdentifier) ──────────────────────

    track_id : int or None
        Persistent identity ID assigned by DeepSORT across frames.
        Stable while the track is alive. Resets on application restart.
        None if the tracker has not yet confirmed this detection.

    identity_label : str or None
        Human-readable name matched from the FAISS identity store.
        None if the face is unregistered (below similarity threshold).
        'unknown' string if explicitly labelled as unrecognised.

    embedding : np.ndarray or None
        L2-normalised ArcFace embedding vector, shape (512,) float32.
        Already normalised by InsightFace — use np.dot() for cosine sim.
        None if embedding extraction failed (e.g. crop too small).

    similarity_score : float or None
        Cosine similarity (0.0-1.0) between this embedding and the
        nearest known identity in FAISS. None if no identity store entry.

    is_known : bool or None
        True if similarity_score exceeds the recognition threshold.
        False if the face is unregistered. None before Layer 4 runs.

    aligned_face : np.ndarray or None
        112x112 BGR affine-warped face used as ArcFace input.
        Stored for debugging and audit purposes.

    ── Layer 5 fields (populated by ExpressionAnalyser) ──────────────────

    expression_scores : dict or None
        Probability distribution across emotion classes summing to 1.0.
        Example: {'happy': 0.72, 'neutral': 0.18, 'sad': 0.05, ...}
        None if Layer 5 has not run on this detection yet (throttled).

    dominant_expression : str or None
        Argmax of expression_scores. e.g. 'happy', 'neutral', 'sad'.
        None if expression_scores is None.

    expression_confidence : float or None
        Probability of dominant_expression (max of expression_scores).
        None if expression_scores is None.
    """
    # ── Layer 3 fields ────────────────────────────────────────────────────
    bbox_original: list          # [x1, y1, x2, y2] original frame space
    bbox_resized: list           # [x1, y1, x2, y2] resized model space
    confidence: float
    landmarks_original: Optional[list]    # [(lx, ly, lconf), ...] or None
    face_crop: Optional[np.ndarray]       # BGR crop from original_frame
    face_crop_shape: Optional[tuple]      # (H, W) of the crop

    # ── Layer 4 fields (default None — populated by FaceIdentifier) ───────
    track_id: Optional[int] = None
    identity_label: Optional[str] = None
    person_id: Optional[str] = None       # FAISS UUID — stable identity key
    embedding: Optional[np.ndarray] = None
    similarity_score: Optional[float] = None
    is_known: Optional[bool] = None
    aligned_face: Optional[np.ndarray] = None

    # ── Layer 5 fields (default None — populated by ExpressionAnalyser) ───
    expression_scores: Optional[dict] = None
    dominant_expression: Optional[str] = None
    expression_confidence: Optional[float] = None

    # True only on frames where inference ACTUALLY ran; False when the value
    # was carried forward from a previous analysis (sticky label).
    # Display code should ignore this and draw every frame — but anything
    # that RECORDS or AGGREGATES must honour it, or one real measurement is
    # counted EXPRESSION_EVERY_N_FRAMES times.
    expression_is_fresh: bool = False


@dataclass
class FrameContext:
    """
    The single travelling object passed between all pipeline layers.

    Layer 1 (Ingestion)      creates: original_frame, camera_id, timestamp, frame_seq
    Layer 2 (Preprocessing)   adds:  preprocessed_frame, original_shape, resized_shape
    Layer 3 (Detection)        adds:  detections list
    Layer 4 (Identity)         adds:  track_id, embedding, identity_label per detection
    Layer 5 (Expression)       adds:  expression_scores per detection

    Fields
    ------
    original_frame : np.ndarray
        Raw BGR uint8 frame directly from cv2.VideoCapture (H, W, 3).
        Preserved untouched throughout all layers.
        Used for final drawing and face crop extraction in Layer 3.

    preprocessed_frame : np.ndarray
        RGB uint8 frame resized to model input size (640x640, 3).
        Produced by Layer 2. Passed to YOLOv8 in Layer 3.
        Not normalized — YOLOv8 handles normalization internally.

    original_shape : tuple[int, int]
        (H, W) of original_frame before any resize.
        Critical for scaling bounding box coordinates back from the
        model's resized space to the original frame space.
        Coordinate scaling formula:
            x_orig = x_resized x (original_W / resized_W)
            y_orig = y_resized x (original_H / resized_H)

    resized_shape : tuple[int, int]
        (H, W) of preprocessed_frame fed to the model. Typically (640, 640).

    camera_id : str
        String identifier for the source (e.g. 'webcam_0', 'rtsp_192.168.1.100').

    timestamp : float
        Unix timestamp (time.time()) when the frame was captured in Layer 1.

    frame_seq : int
        Monotonically increasing frame index starting at 0.
        Used for ordering, gap detection, and DeepSORT frame counting (Layer 4).

    detections : list[Detection]
        Populated by Layer 3. Empty list is a valid result (no faces in frame).
        Zero detections must be handled explicitly downstream.
    """
    original_frame: np.ndarray            # Raw BGR from Layer 1
    preprocessed_frame: np.ndarray        # RGB 640x640 from Layer 2
    original_shape: tuple                 # (H, W) before resize
    resized_shape: tuple                  # (H, W) fed to model
    camera_id: str
    timestamp: float
    frame_seq: int
    detections: list = field(default_factory=list)  # List[Detection]
