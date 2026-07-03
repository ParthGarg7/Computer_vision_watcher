"""
src/layer5_expression/validator.py
─────────────────────────────────────────────────────────────────────────────
Layer 5 Companion: Expression Analysis Validation Pipeline

This is a verification tool, NOT a pipeline layer.

Purpose: visually confirm that Layer 5 (Expression Analysis) output is
correct by drawing expression class probabilities and dominant emotion labels
back onto video frames and saving an annotated output video.

The most common Layer 5 errors that are visible in annotated output:
    1. Expression label stuck — throttle is working but display code is not
       carrying forward the last known label (all faces show 'no expression').
    2. Wrong dominant emotion — visible as 'angry' on a clearly smiling face.
    3. Inference never running — all crops below MIN_CROP_SIZE (tiny detections).
    4. Crash on missing face_crop — Layer 3 didn't produce a crop (handled
       gracefully by analyser, but no expression data downstream).

This follows the same pattern as Layers 3 and 4 ValidationPipeline classes:
    Step 1  Open source + prepare VideoWriter
    Step 2  Read frames one at a time (raw BGR from Layer 1)
    Step 3  Run Layers 2 → 3 → 4 preprocessing, detection, identity
    Step 4  Run Layer 5 expression analysis
    Step 5  Draw expression label, probability bar, and dominant class
    Step 6  Write annotated frame to VideoWriter
    Step 7  Release VideoCapture and VideoWriter
    Step 8  Print visual review checklist

Ref: Layer 5 Architecture Doc — Sections 2, 3, 4.1 (hsemotion-onnx)
"""

import cv2
import numpy as np
import os
import time

from src.layer1_ingestion.capture import VideoCapture
from src.layer2_preprocessing.preprocessor import Preprocessor
from src.layer3_detection.detector import FaceDetector, DEFAULT_MODEL_PATH
from src.layer4_identity.identifier import FaceIdentifier
from src.layer5_expression.analyser import ExpressionAnalyser

# ─── Drawing constants ────────────────────────────────────────────────────────

BOX_COLOR           = (0, 220, 80)       # Green bounding boxes
EXPR_BG_COLOR       = (0, 130, 180)      # Teal — expression label background
EXPR_TEXT_COLOR     = (255, 255, 255)    # White
BAR_BG_COLOR        = (40, 40, 40)       # Dark grey — probability bar background
BAR_FG_COLOR        = (0, 200, 160)      # Teal green — probability fill
TRACK_COLOR         = (200, 100, 0)      # Orange — track ID
HUD_COLOR           = (0, 220, 220)      # Cyan
HINT_COLOR          = (160, 160, 160)    # Grey

FONT              = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_LABEL  = 0.52
FONT_SCALE_SMALL  = 0.38
FONT_SCALE_HUD    = 0.60
FONT_SCALE_HINT   = 0.45
BOX_THICKNESS     = 2
TEXT_THICKNESS    = 1

# Show top-N emotion classes as probability bars below each face
TOP_N_CLASSES = 3


class Layer5ValidationPipeline:
    """
    Runs Layers 1 → 2 → 3 → 4 → 5, draws expression overlays, saves video.

    Mirrors the structure of Layers 3 and 4 ValidationPipeline classes.
    Adds per-detection dominant expression label and top-3 probability bars.

    Usage
    -----
        vp = Layer5ValidationPipeline(source="test_clip.mp4")
        stats = vp.run(show_preview=True)
    """

    def __init__(
        self,
        source,
        model_path: str = DEFAULT_MODEL_PATH,
        confidence_threshold: float = 0.5,
        store_path: str = "models/identity_store",
        expression_every_n: int = 5,
        output_dir: str = "output",
        camera_id: str = "validation_l5"
    ):
        """
        Parameters
        ----------
        source : str or int
            Video file path, webcam index (0), or RTSP URL.
        model_path : str
            Path to yolov8n-face.pt.
        confidence_threshold : float
            Passed to FaceDetector.
        store_path : str
            Base path for the FAISS identity store.
        expression_every_n : int
            Expression throttle — analyse every N frames per track.
        output_dir : str
            Directory for annotated output video.
        camera_id : str
            Label for FrameContext objects.
        """
        self.source = source
        self.output_dir = output_dir
        self.camera_id = camera_id

        os.makedirs(output_dir, exist_ok=True)

        # Instantiate Layers 2, 3, 4, and 5
        self.preprocessor = Preprocessor()
        self.detector = FaceDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold
        )
        self.identifier = FaceIdentifier(store_path=store_path)
        self.analyser = ExpressionAnalyser(every_n_frames=expression_every_n)

    # ─── Drawing ──────────────────────────────────────────────────────────────

    def _draw_detections(
        self,
        frame: np.ndarray,
        detections: list,
        frame_seq: int,
        n_faces: int
    ) -> np.ndarray:
        """
        Draw Layer 5 expression output on a copy of the original frame.

        Each face receives:
            - Green bounding box
            - Orange track_id
            - Teal dominant_expression + confidence label
            - Top-3 probability bars below/beside the face box
        """
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox_original]
            fh, fw = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(fw - 1, x2), min(fh - 1, y2)

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

            label_y = y1  # cursor for stacking labels above box

            # ── Track ID ──────────────────────────────────────────────────────
            if det.track_id is not None:
                tid = f"ID:{det.track_id}"
                (tw, th), tb = cv2.getTextSize(tid, FONT, FONT_SCALE_LABEL, TEXT_THICKNESS)
                ty = max(0, label_y - th - tb - 6)
                cv2.rectangle(annotated, (x1, ty), (x1 + tw + 6, label_y), (200, 100, 0), -1)
                cv2.putText(annotated, tid, (x1 + 3, label_y - tb - 2),
                            FONT, FONT_SCALE_LABEL, (255, 255, 255), TEXT_THICKNESS)
                label_y = ty

            # ── Dominant expression label ─────────────────────────────────────
            if det.dominant_expression:
                expr_label = (
                    f"{det.dominant_expression} "
                    f"{det.expression_confidence:.0%}"
                    if det.expression_confidence else det.dominant_expression
                )
            else:
                expr_label = "analysing..."

            (ew, eh), eb = cv2.getTextSize(expr_label, FONT, FONT_SCALE_LABEL, TEXT_THICKNESS)
            ey = max(0, label_y - eh - eb - 6)
            cv2.rectangle(annotated, (x1, ey), (x1 + ew + 6, label_y), EXPR_BG_COLOR, -1)
            cv2.putText(annotated, expr_label, (x1 + 3, label_y - eb - 2),
                        FONT, FONT_SCALE_LABEL, EXPR_TEXT_COLOR, TEXT_THICKNESS)

            # ── Probability bars (top-N classes) ──────────────────────────────
            if det.expression_scores:
                top_classes = sorted(
                    det.expression_scores.items(),
                    key=lambda kv: kv[1],
                    reverse=True
                )[:TOP_N_CLASSES]

                bar_x = x2 + 6
                bar_y = y1
                bar_w = 90
                bar_h = 14
                gap = 3

                for cls_name, prob in top_classes:
                    # Clip bar within frame
                    if bar_x + bar_w + 40 > fw:
                        break

                    # Background bar
                    cv2.rectangle(annotated,
                                  (bar_x, bar_y),
                                  (bar_x + bar_w, bar_y + bar_h),
                                  BAR_BG_COLOR, -1)
                    # Filled bar (proportional to probability)
                    fill_w = int(bar_w * prob)
                    cv2.rectangle(annotated,
                                  (bar_x, bar_y),
                                  (bar_x + fill_w, bar_y + bar_h),
                                  BAR_FG_COLOR, -1)
                    # Label
                    bar_label = f"{cls_name[:6]} {prob:.0%}"
                    cv2.putText(annotated, bar_label,
                                (bar_x + 3, bar_y + bar_h - 3),
                                FONT, FONT_SCALE_SMALL,
                                (255, 255, 255), TEXT_THICKNESS)
                    bar_y += bar_h + gap

        # HUD overlay
        hud = f"Frame: {frame_seq:05d}  |  Faces: {n_faces}"
        cv2.putText(annotated, hud, (10, 28), FONT, FONT_SCALE_HUD, HUD_COLOR, 2)

        cv2.putText(annotated, "Q: Quit  |  F: Fullscreen",
                    (10, annotated.shape[0] - 10),
                    FONT, FONT_SCALE_HINT, HINT_COLOR, TEXT_THICKNESS)
        return annotated

    # ─── Main Run ─────────────────────────────────────────────────────────────

    def run(
        self,
        show_preview: bool = True,
        max_frames: int = None,
        fps_override: float = None
    ) -> dict:
        """Run the full Layer 1 → 2 → 3 → 4 → 5 validation pipeline."""
        ts_str = time.strftime("%Y%m%d_%H%M%S")
        cam_label = str(self.camera_id).replace("/", "_").replace(":", "_")
        output_path = os.path.join(
            self.output_dir, f"validation_l5_{cam_label}_{ts_str}.mp4"
        )

        print(f"\n{'─'*60}")
        print(f"  [L5-Validator] Layer 5 Expression Validation Pipeline")
        print(f"  [L5-Validator] Source  : {self.source}")
        print(f"  [L5-Validator] Output  : {output_path}")
        print(f"  [L5-Validator] Preview : {'ON -- Q to quit, F fullscreen' if show_preview else 'OFF'}")
        print(f"{'─'*60}\n")

        stats = {
            "frames_processed": 0,
            "frames_with_detections": 0,
            "frames_no_detections": 0,
            "total_faces": 0,
            "frames_with_expression": 0,
            "expression_counts": {},
            "errors": 0,
        }

        writer = None
        cap = None
        writer_initialized = False
        _fullscreen = False

        VAL_WIN = "The Watcher -- Layer 5 Validation"
        _gui_window = False
        if show_preview:
            try:
                cv2.namedWindow(VAL_WIN, cv2.WINDOW_NORMAL)
                _gui_window = True
            except cv2.error:
                print("  [Warn] WINDOW_NORMAL not available.")

        try:
            cap = VideoCapture(str(self.source), camera_id=self.camera_id)
            print(f"  [L5-Validator] Capture : {cap}\n")

            for frame_seq, timestamp, original_frame in cap.frames():

                if max_frames is not None and frame_seq >= max_frames:
                    print(f"\n  [L5-Validator] Reached max_frames={max_frames}. Stopping.")
                    break

                try:
                    # Layer 2
                    ctx = self.preprocessor.process(
                        original_frame,
                        camera_id=self.camera_id,
                        timestamp=timestamp,
                        frame_seq=frame_seq
                    )
                    # Layer 3
                    ctx = self.detector.detect(ctx)
                    # Layer 4
                    ctx = self.identifier.identify(ctx)
                    # Layer 5
                    ctx = self.analyser.analyse(ctx)

                    n_faces = len(ctx.detections)
                    n_expr = sum(
                        1 for d in ctx.detections
                        if d.dominant_expression is not None
                    )

                    stats["frames_processed"] += 1
                    stats["total_faces"] += n_faces
                    if n_faces > 0:
                        stats["frames_with_detections"] += 1
                    else:
                        stats["frames_no_detections"] += 1
                    if n_expr > 0:
                        stats["frames_with_expression"] += 1
                    for d in ctx.detections:
                        if d.dominant_expression:
                            c = d.dominant_expression
                            stats["expression_counts"][c] = \
                                stats["expression_counts"].get(c, 0) + 1

                    annotated = self._draw_detections(
                        ctx.original_frame, ctx.detections, frame_seq, n_faces
                    )

                    if not writer_initialized:
                        fh, fw = annotated.shape[:2]
                        fps = fps_override if fps_override else cap.fps
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(output_path, fourcc, fps, (fw, fh))
                        print(f"  [L5-Validator] VideoWriter: {fw}x{fh} "
                              f"@ {fps:.1f}fps → {output_path}\n")
                        writer_initialized = True

                    if writer is not None:
                        writer.write(annotated)

                    if show_preview:
                        cv2.imshow(VAL_WIN, annotated)
                        if _gui_window and cv2.getWindowProperty(VAL_WIN, cv2.WND_PROP_VISIBLE) < 1:
                            print("\n  [L5-Validator] Window closed by user.")
                            break
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            print("\n  [L5-Validator] User pressed Q. Stopping.")
                            break
                        elif key == ord("f") and _gui_window:
                            _fullscreen = not _fullscreen
                            prop = cv2.WINDOW_FULLSCREEN if _fullscreen else cv2.WINDOW_NORMAL
                            cv2.setWindowProperty(VAL_WIN, cv2.WND_PROP_FULLSCREEN, prop)

                    if frame_seq % 30 == 0:
                        print(f"  Frame {frame_seq:5d} | Faces: {n_faces} | "
                              f"With expression: {n_expr}")

                except Exception as e:
                    stats["errors"] += 1
                    print(f"  [L5-Validator] ERROR frame {frame_seq}: {e}")
                    continue

        finally:
            if writer is not None:
                writer.release()
            if cap is not None:
                cap.release()
            if show_preview:
                cv2.destroyAllWindows()

        self._print_review_checklist(stats, output_path)
        return stats

    def _print_review_checklist(self, stats: dict, output_path: str):
        print(f"\n{'='*60}")
        print(f"  L5 VALIDATION COMPLETE")
        print(f"{'='*60}")
        print(f"  Output video           : {output_path}")
        print(f"  Frames processed       : {stats['frames_processed']}")
        print(f"  Frames w/ detections   : {stats['frames_with_detections']}")
        print(f"  Frames w/o detection   : {stats['frames_no_detections']}")
        print(f"  Total faces            : {stats['total_faces']}")
        print(f"  Frames w/ expression   : {stats['frames_with_expression']}")
        print(f"  Expression distribution:")
        for cls, cnt in sorted(stats["expression_counts"].items(),
                               key=lambda x: x[1], reverse=True):
            print(f"    {cls:12} : {cnt:5} frames")
        print(f"  Processing errors      : {stats['errors']}")
        print(f"\n  Visual Review Checklist (watch the output video):")
        print(f"  [ ] Teal expression label appears above confirmed face boxes")
        print(f"  [ ] Dominant emotion name matches what you see on screen")
        print(f"  [ ] Probability bars visible and update ~every 5 frames")
        print(f"  [ ] Label carries forward between throttled frames (no flicker)")
        print(f"  [ ] 'analysing...' only appears on very first frames of a track")
        print(f"  [ ] Pipeline does NOT crash on empty detection frames")
        print(f"  [ ] Neutral faces show 'neutral' or 'happy', not 'angry'")
        print(f"{'='*60}\n")
