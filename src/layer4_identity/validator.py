"""
src/layer4_identity/validator.py
─────────────────────────────────────────────────────────────────────────────
Layer 4 Companion: Identity Validation Pipeline

This is a verification tool, NOT a pipeline layer.

Purpose: visually confirm that Layer 4 (Identity) output is correct by
drawing track IDs, identity labels, ArcFace embedding visualisations, and
similarity scores back onto video frames, then saving an annotated video.

The most common Layer 4 errors that only become visible once drawn:
    1. Track ID switching — the same person's box changes ID mid-sequence.
       Visible as flickering ID numbers on a stationary face.
    2. Identity label flicker — known identity briefly labelled 'unknown'
       due to below-threshold frames.
    3. Tracker dropout — DeepSORT failed to confirm a track (needs 3
       consecutive detections). Visible as faces with no track_id.
    4. FAISS match on wrong person — label shows wrong name.
       Visible as wrong text on a face you know.

This follows the same pattern as Layer 3's ValidationPipeline class:
    Step 1  Open source + prepare VideoWriter
    Step 2  Read frames one at a time (raw BGR from Layer 1)
    Step 3  Run Layers 2 → 3 preprocessing and detection
    Step 4  Run Layer 4 identity pipeline
    Step 5  Draw track_id, identity_label, similarity_score on original_frame
    Step 6  Write annotated frame to VideoWriter
    Step 7  Release VideoCapture and VideoWriter
    Step 8  Print visual review checklist

Ref: Layer 4 Architecture Doc — Section 10 (Exact Output)
"""

import cv2
import numpy as np
import os
import time

from src.layer1_ingestion.capture import VideoCapture
from src.layer2_preprocessing.preprocessor import Preprocessor
from src.layer3_detection.detector import FaceDetector, DEFAULT_MODEL_PATH
from src.layer4_identity.identifier import FaceIdentifier
from src.layer4_identity.identity_store import DEFAULT_STORE_PATH

# ─── Drawing constants ────────────────────────────────────────────────────────

BOX_COLOR           = (0, 220, 80)       # Green bounding boxes
TRACK_ID_BG_COLOR   = (200, 100, 0)      # Dark orange — track ID background
TRACK_ID_TEXT_COLOR = (255, 255, 255)    # White
IDENTITY_BG_COLOR   = (120, 0, 180)      # Purple — identity label background
IDENTITY_TEXT_COLOR = (255, 255, 255)    # White
UNKNOWN_BG_COLOR    = (60, 60, 60)       # Dark grey — unknown person
SCORE_COLOR         = (0, 220, 220)      # Cyan — similarity score
HUD_COLOR           = (0, 220, 220)      # Cyan HUD overlay
HINT_COLOR          = (160, 160, 160)    # Grey hint text

FONT              = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_LABEL  = 0.55
FONT_SCALE_HUD    = 0.60
FONT_SCALE_HINT   = 0.45
BOX_THICKNESS     = 2
TEXT_THICKNESS    = 1


class Layer4ValidationPipeline:
    """
    Runs Layers 1 → 2 → 3 → 4 on a video source, draws all identity
    outputs on the original frames, and saves the annotated video.

    Mirrors the structure of Layer 3's ValidationPipeline class.
    Adds per-detection track_id and identity_label overlays.

    Usage
    -----
        vp = Layer4ValidationPipeline(source="test_clip.mp4")
        stats = vp.run(show_preview=True)
    """

    def __init__(
        self,
        source,
        model_path: str = DEFAULT_MODEL_PATH,
        confidence_threshold: float = 0.5,
        store_path: str = DEFAULT_STORE_PATH,
        output_dir: str = "output",
        camera_id: str = "validation_l4"
    ):
        """
        Parameters
        ----------
        source : str or int
            Video file path, webcam index (0), or RTSP URL.
        model_path : str
            Path to yolov8n-face.pt weights.
        confidence_threshold : float
            Passed to FaceDetector.
        store_path : str
            Base path for the FAISS identity store.
        output_dir : str
            Directory for annotated output video.
        camera_id : str
            Label for FrameContext objects.
        """
        self.source = source
        self.output_dir = output_dir
        self.camera_id = camera_id

        os.makedirs(output_dir, exist_ok=True)

        # Instantiate Layers 2, 3, and 4
        self.preprocessor = Preprocessor()
        self.detector = FaceDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold
        )
        self.identifier = FaceIdentifier(store_path=store_path)

    # ─── Drawing ──────────────────────────────────────────────────────────────

    def _draw_detections(
        self,
        frame: np.ndarray,
        detections: list,
        frame_seq: int,
        n_faces: int,
        n_tracked: int
    ) -> np.ndarray:
        """
        Draw Layer 4 identity output on a copy of the original frame.

        Each detected face receives:
            - Green bounding box (from Layer 3)
            - Orange track_id label (from DeepSORT)
            - Purple identity label + similarity score (from FAISS)
            - Dark grey 'unknown' label for unregistered faces
        """
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox_original]
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

            label_y = y1  # current y cursor for stacking labels above the box

            # ── Track ID label ────────────────────────────────────────────────
            if det.track_id is not None:
                tid_text = f"ID:{det.track_id}"
            else:
                tid_text = "ID:?"  # tentative / unconfirmed track

            (tw, th), base = cv2.getTextSize(tid_text, FONT, FONT_SCALE_LABEL, TEXT_THICKNESS)
            ty_top = max(0, label_y - th - base - 6)
            cv2.rectangle(annotated, (x1, ty_top), (x1 + tw + 6, label_y), TRACK_ID_BG_COLOR, -1)
            cv2.putText(annotated, tid_text, (x1 + 3, label_y - base - 2),
                        FONT, FONT_SCALE_LABEL, TRACK_ID_TEXT_COLOR, TEXT_THICKNESS)
            label_y = ty_top

            # ── Identity label ─────────────────────────────────────────────────
            if det.is_known and det.identity_label:
                id_text = det.identity_label
                score_text = f"{det.similarity_score:.2f}" if det.similarity_score else ""
                bg_color = IDENTITY_BG_COLOR
            else:
                id_text = "unknown"
                score_text = f"{det.similarity_score:.2f}" if det.similarity_score else "0.00"
                bg_color = UNKNOWN_BG_COLOR

            full_label = f"{id_text} ({score_text})" if score_text else id_text
            (iw, ih), ibase = cv2.getTextSize(full_label, FONT, FONT_SCALE_LABEL, TEXT_THICKNESS)
            iy_top = max(0, label_y - ih - ibase - 6)
            cv2.rectangle(annotated, (x1, iy_top), (x1 + iw + 6, label_y), bg_color, -1)
            cv2.putText(annotated, full_label, (x1 + 3, label_y - ibase - 2),
                        FONT, FONT_SCALE_LABEL, IDENTITY_TEXT_COLOR, TEXT_THICKNESS)

        # HUD overlay
        hud = (f"Frame: {frame_seq:05d}  |  Faces: {n_faces}  |  "
               f"Tracked: {n_tracked}")
        cv2.putText(annotated, hud, (10, 28), FONT, FONT_SCALE_HUD, HUD_COLOR, 2)

        # Hint
        cv2.putText(
            annotated, "Q: Quit  |  F: Fullscreen",
            (10, annotated.shape[0] - 10),
            FONT, FONT_SCALE_HINT, HINT_COLOR, TEXT_THICKNESS
        )
        return annotated

    # ─── Main Run ─────────────────────────────────────────────────────────────

    def run(
        self,
        show_preview: bool = True,
        max_frames: int = None,
        fps_override: float = None
    ) -> dict:
        """
        Execute the full Layer 1 → 2 → 3 → 4 validation pipeline.

        Returns a stats dict with frame and track counts.
        """
        ts_str = time.strftime("%Y%m%d_%H%M%S")
        cam_label = str(self.camera_id).replace("/", "_").replace(":", "_")
        output_path = os.path.join(self.output_dir, f"validation_l4_{cam_label}_{ts_str}.mp4")

        print(f"\n{'─'*58}")
        print(f"  [L4-Validator] Layer 4 Identity Validation Pipeline")
        print(f"  [L4-Validator] Source  : {self.source}")
        print(f"  [L4-Validator] Output  : {output_path}")
        print(f"  [L4-Validator] Preview : {'ON -- Q to quit, F fullscreen' if show_preview else 'OFF'}")
        print(f"{'─'*58}\n")

        stats = {
            "frames_processed": 0,
            "frames_with_detections": 0,
            "frames_no_detections": 0,
            "total_faces_detected": 0,
            "total_confirmed_tracks": 0,
            "unique_track_ids": set(),
            "errors": 0,
        }

        writer = None
        cap = None
        writer_initialized = False
        _val_fullscreen = False

        VAL_WIN = "The Watcher -- Layer 4 Validation"
        _val_gui_window = False
        if show_preview:
            try:
                cv2.namedWindow(VAL_WIN, cv2.WINDOW_NORMAL)
                _val_gui_window = True
            except cv2.error:
                print("  [Warn] WINDOW_NORMAL not available.")

        try:
            # Step 1 — Open source
            cap = VideoCapture(str(self.source), camera_id=self.camera_id)
            print(f"  [L4-Validator] Capture : {cap}\n")

            for frame_seq, timestamp, original_frame in cap.frames():

                if max_frames is not None and frame_seq >= max_frames:
                    print(f"\n  [L4-Validator] Reached max_frames={max_frames}. Stopping.")
                    break

                try:
                    # Step 2 — Layer 2: Preprocess
                    ctx = self.preprocessor.process(
                        original_frame,
                        camera_id=self.camera_id,
                        timestamp=timestamp,
                        frame_seq=frame_seq
                    )

                    # Step 3 — Layer 3: Detect (includes coordinate scaling)
                    ctx = self.detector.detect(ctx)

                    # Step 4 — Layer 4: Identify
                    ctx = self.identifier.identify(ctx)

                    n_faces = len(ctx.detections)
                    n_tracked = sum(
                        1 for d in ctx.detections if d.track_id is not None
                    )

                    stats["frames_processed"] += 1
                    stats["total_faces_detected"] += n_faces
                    if n_faces > 0:
                        stats["frames_with_detections"] += 1
                    else:
                        stats["frames_no_detections"] += 1
                    stats["total_confirmed_tracks"] += n_tracked
                    for d in ctx.detections:
                        if d.track_id is not None:
                            stats["unique_track_ids"].add(d.track_id)

                    # Step 5 — Draw
                    annotated = self._draw_detections(
                        ctx.original_frame, ctx.detections,
                        frame_seq, n_faces, n_tracked
                    )

                    # Step 1 continued — init VideoWriter
                    if not writer_initialized:
                        fh, fw = annotated.shape[:2]
                        fps = fps_override if fps_override else cap.fps
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(output_path, fourcc, fps, (fw, fh))
                        print(f"  [L4-Validator] VideoWriter: {fw}x{fh} "
                              f"@ {fps:.1f}fps → {output_path}\n")
                        writer_initialized = True

                    # Step 6 — Write annotated frame
                    if writer is not None:
                        writer.write(annotated)

                    if show_preview:
                        cv2.imshow(VAL_WIN, annotated)
                        if _val_gui_window and cv2.getWindowProperty(VAL_WIN, cv2.WND_PROP_VISIBLE) < 1:
                            print("\n  [L4-Validator] Window closed by user.")
                            break
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            print("\n  [L4-Validator] User pressed Q. Stopping early.")
                            break
                        elif key == ord("f") and _val_gui_window:
                            _val_fullscreen = not _val_fullscreen
                            prop = cv2.WINDOW_FULLSCREEN if _val_fullscreen else cv2.WINDOW_NORMAL
                            cv2.setWindowProperty(VAL_WIN, cv2.WND_PROP_FULLSCREEN, prop)

                    if frame_seq % 30 == 0:
                        print(f"  Frame {frame_seq:5d} | Faces: {n_faces} | "
                              f"Tracked: {n_tracked} | "
                              f"Unique IDs: {len(stats['unique_track_ids'])}")

                except Exception as e:
                    stats["errors"] += 1
                    print(f"  [L4-Validator] ERROR frame {frame_seq}: {e}")
                    continue

        finally:
            # Step 7 — Release
            if writer is not None:
                writer.release()
            if cap is not None:
                cap.release()
            if show_preview:
                cv2.destroyAllWindows()

        # Step 8 — Checklist
        self._print_review_checklist(stats, output_path)
        return stats

    def _print_review_checklist(self, stats: dict, output_path: str):
        """Print the Layer 4 visual review checklist."""
        unique_ids = len(stats["unique_track_ids"])
        print(f"\n{'='*58}")
        print(f"  L4 VALIDATION COMPLETE")
        print(f"{'='*58}")
        print(f"  Output video           : {output_path}")
        print(f"  Frames processed       : {stats['frames_processed']}")
        print(f"  Frames w/ detections   : {stats['frames_with_detections']}")
        print(f"  Frames w/o detection   : {stats['frames_no_detections']}")
        print(f"  Total faces            : {stats['total_faces_detected']}")
        print(f"  Unique track IDs seen  : {unique_ids}")
        print(f"  Processing errors      : {stats['errors']}")
        print(f"\n  Visual Review Checklist (watch the output video):")
        print(f"  [ ] Orange 'ID:N' labels appear above every confirmed face")
        print(f"  [ ] Track ID stays STABLE on the same person across frames")
        print(f"  [ ] ID numbers do NOT flicker/switch on a stationary face")
        print(f"  [ ] Known faces show correct purple identity label")
        print(f"  [ ] Unknown faces show dark grey 'unknown (0.XX)' label")
        print(f"  [ ] Similarity score is visible and > 0.45 for known faces")
        print(f"  [ ] Empty frames produce zero detections (no crash)")
        print(f"  [ ] Brief face disappearance does NOT create a new ID (max_age)")
        print(f"{'='*58}\n")
