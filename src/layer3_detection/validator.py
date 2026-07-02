"""
src/layer3_detection/validator.py
─────────────────────────────────────────────────────────────────────────────
Layer 3 Companion: Detection Validation Pipeline

This is a verification tool, NOT a pipeline layer.

Purpose: visually confirm that detection output is correct by drawing
bounding boxes, landmark points, and confidence scores back onto the original
video frames, then saving the result as a playable annotated video file.

A coordinate scaling bug (the most common Layer 3 error) is invisible in raw
numeric output but immediately obvious once boxes are drawn — they appear
floating in empty space or offset from actual faces.

The 9-step process follows the Layer 3 Validation Pipeline Architecture Doc:
    Step 1  Open source + prepare VideoWriter
    Step 2  Read frames one at a time (raw BGR from Layer 1)
    Step 3  Run Layer 2 preprocessing (BGR→RGB, resize, build FrameContext)
    Step 4  Run Layer 3 detection (YOLOv8n-face inference)
    Step 5  Scale coordinates back (handled inside detector.py)
    Step 6  Draw bounding boxes, landmarks, confidence on original_frame
    Step 7  Write annotated frame to VideoWriter
    Step 8  Release VideoCapture and VideoWriter
    Step 9  Print visual review checklist

This tool is reusable for Layer 4 and Layer 5 validation — only the drawing
step changes (add track_id labels for Layer 4, expression labels for Layer 5).

Ref: Layer 3 Companion Doc — Detection Validation Pipeline Architecture
"""

import cv2
import numpy as np
import os
import time

from src.layer1_ingestion.capture import VideoCapture
from src.layer2_preprocessing.preprocessor import Preprocessor
from src.layer3_detection.detector import FaceDetector, DEFAULT_MODEL_PATH

# ─── Drawing constants ────────────────────────────────────────────────────────

BOX_COLOR         = (0, 220, 80)      # Green bounding boxes (BGR)
LANDMARK_COLOR    = (0, 120, 255)     # Orange landmark dots (BGR)
CONF_BG_COLOR     = (0, 180, 60)      # Dark green confidence label background
CONF_TEXT_COLOR   = (255, 255, 255)   # White confidence text
HUD_COLOR         = (0, 220, 220)     # Cyan HUD overlay
HINT_COLOR        = (160, 160, 160)   # Grey hint text

FONT              = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_CONF   = 0.52
FONT_SCALE_HUD    = 0.6
FONT_SCALE_HINT   = 0.45
BOX_THICKNESS     = 2
LANDMARK_RADIUS   = 5
TEXT_THICKNESS    = 1


class ValidationPipeline:
    """
    Runs the full Layer 1 → 2 → 3 pipeline on a video source, draws
    all detections on the original frames, and saves the annotated video.

    Also supports live preview via cv2.imshow() for fast iteration
    during development (press Q to stop early).

    Usage
    -----
        # Validate a recorded video file
        vp = ValidationPipeline(source="test_clip.mp4")
        stats = vp.run(show_preview=True)

        # Validate from webcam (saves annotated live output)
        vp = ValidationPipeline(source="0")
        stats = vp.run(show_preview=True)
    """

    def __init__(
        self,
        source,
        model_path: str = DEFAULT_MODEL_PATH,
        confidence_threshold: float = 0.5,
        output_dir: str = "output",
        camera_id: str = "validation"
    ):
        """
        Parameters
        ----------
        source : str or int
            Video file path, webcam index (0), or RTSP URL.
        model_path : str
            Path to yolov8n-face.pt weights.
        confidence_threshold : float
            Passed directly to FaceDetector.
        output_dir : str
            Directory where annotated video is saved. Created if missing.
        camera_id : str
            Label carried in FrameContext objects.
        """
        self.source = source
        self.output_dir = output_dir
        self.camera_id = camera_id

        os.makedirs(output_dir, exist_ok=True)

        # Instantiate Layer 2 and Layer 3
        self.preprocessor = Preprocessor()
        self.detector = FaceDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold
        )

    # ─── Drawing ──────────────────────────────────────────────────────────────

    def _draw_detections(
        self,
        frame: np.ndarray,
        detections: list,
        frame_seq: int,
        n_faces: int
    ) -> np.ndarray:
        """
        Draw all detections onto a copy of the original frame (BGR).

        Drawing always happens on original_frame, never on the resized/
        preprocessed tensor (which exists only for model input).

        Returns the annotated copy.
        """
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox_original]

            # Clamp to frame bounds (defensive — crop clamping should prevent this)
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

            # Confidence label with filled background rectangle
            label = f"{det.confidence:.2f}"
            (lw, lh), baseline = cv2.getTextSize(label, FONT, FONT_SCALE_CONF, TEXT_THICKNESS)
            label_y_top = max(0, y1 - lh - baseline - 6)
            cv2.rectangle(
                annotated,
                (x1, label_y_top),
                (x1 + lw + 6, y1),
                CONF_BG_COLOR, -1
            )
            cv2.putText(
                annotated, label,
                (x1 + 3, y1 - baseline - 2),
                FONT, FONT_SCALE_CONF, CONF_TEXT_COLOR, TEXT_THICKNESS
            )

            # Landmark dots (if available — checkpoint-dependent)
            if det.landmarks_original:
                for lx, ly, _lconf in det.landmarks_original:
                    cv2.circle(
                        annotated,
                        (int(lx), int(ly)),
                        LANDMARK_RADIUS, LANDMARK_COLOR, -1
                    )

        # HUD overlay: frame number + face count
        hud_text = f"Frame: {frame_seq:05d}  |  Faces: {n_faces}"
        cv2.putText(
            annotated, hud_text,
            (10, 28), FONT, FONT_SCALE_HUD, HUD_COLOR, 2
        )

        # Corner hint
        hint_text = "Q: Quit  |  F: Fullscreen"
        fh = annotated.shape[0]
        cv2.putText(
            annotated, hint_text,
            (10, fh - 10), FONT, FONT_SCALE_HINT, HINT_COLOR, TEXT_THICKNESS
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
        Execute the full validation pipeline.

        Parameters
        ----------
        show_preview : bool
            Show cv2.imshow() preview window. Press Q to stop early.
            Disable on headless machines.
        max_frames : int or None
            Limit total frames processed. Useful for fast iteration on short clips.
        fps_override : float or None
            Override FPS for VideoWriter (useful if source FPS is 0 or unreliable).

        Returns
        -------
        dict
            Stats: frames_processed, frames_with_detections, total_faces, errors, etc.
        """
        # Generate timestamped output filename
        ts_str = time.strftime("%Y%m%d_%H%M%S")
        cam_label = str(self.camera_id).replace("/", "_").replace(":", "_")
        output_path = os.path.join(self.output_dir, f"validation_{cam_label}_{ts_str}.mp4")

        print(f"\n{'─'*55}")
        print(f"  [Validator] Detection Validation Pipeline")
        print(f"  [Validator] Source  : {self.source}")
        print(f"  [Validator] Output  : {output_path}")
        print(f"  [Validator] Preview : {'ON -- press Q to quit, F for fullscreen' if show_preview else 'OFF'}")
        print(f"{'─'*55}\n")

        stats = {
            "frames_processed": 0,
            "frames_with_detections": 0,
            "frames_no_detections": 0,
            "total_faces_detected": 0,
            "max_faces_in_frame": 0,
            "errors": 0,
        }

        writer = None
        cap = None
        writer_initialized = False
        _val_fullscreen = False

        # Create resizable window once before the loop (show_preview path)
        VAL_WIN = "The Watcher -- Layer 3 Validation"
        _val_gui_window = False
        if show_preview:
            try:
                cv2.namedWindow(VAL_WIN, cv2.WINDOW_NORMAL)
                _val_gui_window = True
            except cv2.error:
                print("  [Warn] WINDOW_NORMAL not available -- headless opencv detected.")
                print("  [Fix]  pip uninstall opencv-python-headless -y && pip install --force-reinstall opencv-python")

        try:
            # Step 1 — Open source (Layer 1)
            cap = VideoCapture(str(self.source), camera_id=self.camera_id)
            print(f"  [Validator] Capture : {cap}\n")

            for frame_seq, timestamp, original_frame in cap.frames():

                # Respect max_frames limit
                if max_frames is not None and frame_seq >= max_frames:
                    print(f"\n  [Validator] Reached max_frames={max_frames}. Stopping.")
                    break

                try:
                    # Step 2 — Layer 2: Preprocess
                    ctx = self.preprocessor.process(
                        original_frame,
                        camera_id=self.camera_id,
                        timestamp=timestamp,
                        frame_seq=frame_seq
                    )

                    # Step 3/4 — Layer 3: Detect (includes coordinate scaling step 5)
                    ctx = self.detector.detect(ctx)

                    n_faces = len(ctx.detections)
                    stats["frames_processed"] += 1
                    stats["total_faces_detected"] += n_faces

                    if n_faces > 0:
                        stats["frames_with_detections"] += 1
                        stats["max_faces_in_frame"] = max(stats["max_faces_in_frame"], n_faces)
                    else:
                        stats["frames_no_detections"] += 1

                    # Step 6 — Draw on original_frame (not preprocessed tensor)
                    annotated = self._draw_detections(
                        ctx.original_frame, ctx.detections, frame_seq, n_faces
                    )

                    # Step 1 (continued) — Initialize VideoWriter on first frame
                    # Done here so we get actual frame dimensions
                    if not writer_initialized:
                        fh, fw = annotated.shape[:2]
                        fps = fps_override if fps_override else cap.fps
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(output_path, fourcc, fps, (fw, fh))
                        print(
                            f"  [Validator] VideoWriter: {fw}x{fh} "
                            f"@ {fps:.1f}fps → {output_path}\n"
                        )
                        writer_initialized = True

                    # Step 7 — Write annotated frame
                    if writer is not None:
                        writer.write(annotated)

                    # Live preview
                    if show_preview:
                        cv2.imshow(VAL_WIN, annotated)

                        # X button close detection (only when WINDOW_NORMAL is supported)
                        if _val_gui_window and cv2.getWindowProperty(VAL_WIN, cv2.WND_PROP_VISIBLE) < 1:
                            print("\n  [Validator] Window closed by user (X button).")
                            break

                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            print("\n  [Validator] User pressed Q. Stopping early.")
                            break
                        elif key == ord("f") and _val_gui_window:
                            _val_fullscreen = not _val_fullscreen
                            prop = cv2.WINDOW_FULLSCREEN if _val_fullscreen else cv2.WINDOW_NORMAL
                            cv2.setWindowProperty(VAL_WIN, cv2.WND_PROP_FULLSCREEN, prop)

                    # Progress log (every 30 frames)
                    if frame_seq % 30 == 0:
                        print(
                            f"  Frame {frame_seq:5d} | "
                            f"Faces: {n_faces} | "
                            f"Total: {stats['total_faces_detected']}"
                        )

                except Exception as e:
                    stats["errors"] += 1
                    print(f"  [Validator] ERROR frame {frame_seq}: {e}")
                    continue

        finally:
            # Step 8 — Release resources
            if writer is not None:
                writer.release()
            if cap is not None:
                cap.release()
            if show_preview:
                cv2.destroyAllWindows()

        # Step 9 — Visual review checklist
        self._print_review_checklist(stats, output_path)

        return stats

    def _print_review_checklist(self, stats: dict, output_path: str):
        """
        Print the Layer 3 Validation doc's visual review checklist.
        These items must be confirmed by watching the output video.
        """
        print(f"\n{'='*55}")
        print(f"  VALIDATION COMPLETE")
        print(f"{'='*55}")
        print(f"  Output video         : {output_path}")
        print(f"  Frames processed     : {stats['frames_processed']}")
        print(f"  Frames w/ detections : {stats['frames_with_detections']}")
        print(f"  Frames w/o detection : {stats['frames_no_detections']}")
        print(f"  Total faces found    : {stats['total_faces_detected']}")
        print(f"  Max faces in 1 frame : {stats['max_faces_in_frame']}")
        print(f"  Processing errors    : {stats['errors']}")
        print(f"\n  Visual Review Checklist (watch the output video):")
        print(f"  [ ] Boxes sit TIGHTLY around real faces")
        print(f"  [ ] No boxes floating in empty background space")
        print(f"  [ ] Boxes do not drift as the person moves")
        print(f"  [ ] Confidence scores on real faces are mostly > 0.5")
        print(f"  [ ] Empty frames produce zero detections (no error)")
        print(f"  [ ] Multiple faces produce one box per face correctly")
        print(f"  [ ] If landmarks shown: dots land on eyes/nose/mouth")
        print(f"{'='*55}\n")
