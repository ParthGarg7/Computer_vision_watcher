#!/usr/bin/env python3
"""
main.py — The Watcher MVP Entry Point
─────────────────────────────────────────────────────────────────────────────
Computer Vision Pipeline: Layers 1 → 2 → 3 → 4 → 5

Terminal menu to choose the input source. Optional layers can be disabled
with flags for faster startup when dependencies haven't downloaded yet.

Flags:
    --validate          Save annotated output video to the output/ folder
    --source STR        Skip the menu and use source directly
    --no-identity       Skip Layer 4 (InsightFace not downloaded yet)
    --no-expression     Skip Layer 5 (expression model not downloaded yet)

Examples:
    python main.py                         # Full pipeline, interactive menu
    python main.py --source 0             # Webcam, all layers
    python main.py --source 0 --no-identity --no-expression  # Layers 1-3 only
    python main.py --source video.mp4 --validate
"""

import sys
import os
import time
import argparse

# ── Fix Windows console encoding ─────────────────────────────────────────────
# Windows terminals default to cp1252 which cannot print Unicode box-drawing
# characters (╔═╝ etc.). Reconfigure stdout/stderr to UTF-8 at startup.
# This is a no-op on Linux/macOS where UTF-8 is already the default.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import torch

os.environ["YOLO_VERBOSE"] = "False"


# ─── Banner ───────────────────────────────────────────────────────────────────

def print_banner(enable_identity: bool = True, enable_expression: bool = True):
    gpu_info = "CPU only"
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_info = f"{name}  ({mem_gb:.1f} GB VRAM)"

    layers_str = "1 → 2 → 3"
    if enable_identity:
        layers_str += " → 4"
    if enable_expression:
        layers_str += " → 5"

    print()
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║         THE WATCHER — Computer Vision MVP            ║")
    print(f"  ║     Layers {layers_str:<43}║")
    print("  ╚══════════════════════════════════════════════════════╝")
    print(f"  GPU  : {gpu_info}")
    print(f"  Time : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()


# ─── Input source selection ───────────────────────────────────────────────────

def select_source() -> tuple:
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │              Select Input Source                     │")
    print("  ├─────────────────────────────────────────────────────┤")
    print("  │  [1]  Webcam          (built-in camera, index 0)    │")
    print("  │  [2]  Recorded Video  (opens file browser)          │")
    print("  │  [3]  RTSP Stream     (enter URL manually)          │")
    print("  │  [Q]  Quit                                          │")
    print("  └─────────────────────────────────────────────────────┘")
    print()

    while True:
        try:
            choice = input("  Your choice: ").strip().upper()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Interrupted. Exiting.")
            sys.exit(0)

        if choice == "1":
            return _select_webcam()
        elif choice == "2":
            return _select_video_file()
        elif choice == "3":
            return _select_rtsp()
        elif choice == "Q":
            print("\n  Goodbye.\n")
            sys.exit(0)
        else:
            print(f"  '{choice}' is not a valid option. Enter 1, 2, 3, or Q.\n")


def _select_webcam() -> tuple:
    index = 0
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"  ERROR: Could not open webcam at index {index}.")
        cap.release()
        sys.exit(1)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"  Webcam ready: {w}x{h} at index {index}")
    return str(index), f"webcam_{index}"


def _select_video_file() -> tuple:
    print("\n  Opening file browser...")
    file_path = None
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        file_path = filedialog.askopenfilename(
            title="The Watcher — Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm *.m4v *.ts"),
                ("All files", "*.*"),
            ]
        )
        root.destroy()
        if not file_path:
            print("  No file selected. Returning to menu.\n")
            return select_source()
    except (ImportError, Exception) as e:
        print(f"  File browser unavailable ({e}). Enter path manually.")
        try:
            file_path = input("  Video file path: ").strip().strip('"')
        except (KeyboardInterrupt, EOFError):
            sys.exit(0)

    if not os.path.exists(file_path):
        print(f"  ERROR: File not found: {file_path}")
        sys.exit(1)

    size_mb = os.path.getsize(file_path) / 1e6
    filename = os.path.basename(file_path)
    cam_id = f"video_{os.path.splitext(filename)[0]}"
    print(f"  Selected: {filename} ({size_mb:.1f} MB)")
    return file_path, cam_id


def _select_rtsp() -> tuple:
    print()
    print("  RTSP URL format: rtsp://username:password@ip_address:port/stream_path")
    print()
    try:
        url = input("  Enter RTSP URL: ").strip()
    except (KeyboardInterrupt, EOFError):
        sys.exit(0)

    if not url:
        print("  No URL entered. Returning to menu.\n")
        return select_source()

    try:
        host_part = url.split("@")[-1].split("/")[0]
        cam_id = f"rtsp_{host_part}".replace(":", "_").replace(".", "_")
    except Exception:
        cam_id = "rtsp_stream"

    cap = cv2.VideoCapture(url)
    if cap.isOpened():
        ret, _ = cap.read()
        cap.release()
        if ret:
            print("  RTSP connection: OK")
        else:
            print("  RTSP opened but no frame received.")
    else:
        print("  WARNING: Could not verify RTSP stream.")
    return url, cam_id


# ─── Drawing helpers ──────────────────────────────────────────────────────────

# Color palette (BGR)
_C = {
    "box":        (0, 220, 80),     # Green — bounding box
    "track_bg":   (200, 100, 0),    # Orange — track ID background
    "track_txt":  (255, 255, 255),  # White
    "known_bg":   (120, 0, 180),    # Purple — known identity
    "unknown_bg": (60, 60, 60),     # Dark grey — unknown
    "id_txt":     (255, 255, 255),  # White
    "expr_bg":    (0, 130, 180),    # Teal — expression
    "expr_txt":   (255, 255, 255),  # White
    "hud":        (0, 220, 220),    # Cyan
    "hint":       (160, 160, 160),  # Grey
    "bar_bg":     (40, 40, 40),
    "bar_fg":     (0, 200, 160),
}
_F = cv2.FONT_HERSHEY_SIMPLEX


def _put_label(img, text, x, y_bottom, bg_color, text_color,
               font_scale=0.52, thickness=1):
    """Draw a filled-background text label. Returns y_top of the label."""
    (tw, th), base = cv2.getTextSize(text, _F, font_scale, thickness)
    y_top = max(0, y_bottom - th - base - 6)
    cv2.rectangle(img, (x, y_top), (x + tw + 6, y_bottom), bg_color, -1)
    cv2.putText(img, text, (x + 3, y_bottom - base - 2),
                _F, font_scale, text_color, thickness)
    return y_top


def draw_all_detections(
    display: np.ndarray,
    detections: list,
    frame_seq: int,
    display_fps: float,
    enable_identity: bool,
    enable_expression: bool
) -> np.ndarray:
    """Draw all layer outputs onto the display frame."""
    n_faces = len(detections)

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox_original]
        fh, fw = display.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(fw - 1, x2), min(fh - 1, y2)

        cv2.rectangle(display, (x1, y1), (x2, y2), _C["box"], 2)

        label_y = y1

        # ── Confidence (Layer 3) ──────────────────────────────────────────
        conf_txt = f"{det.confidence:.2f}"
        label_y = _put_label(display, conf_txt, x1, label_y,
                              (0, 180, 60), (255, 255, 255))

        # ── Track ID (Layer 4) ────────────────────────────────────────────
        if enable_identity and det.track_id is not None:
            tid_txt = f"ID:{det.track_id}"
            label_y = _put_label(display, tid_txt, x1, label_y,
                                  _C["track_bg"], _C["track_txt"])

        # ── Identity label (Layer 4) ──────────────────────────────────────
        if enable_identity:
            if det.is_known and det.identity_label:
                score_txt = f"{det.similarity_score:.2f}" if det.similarity_score else ""
                id_txt = f"{det.identity_label} ({score_txt})"
                bg = _C["known_bg"]
            elif det.embedding is not None:
                score_txt = f"{det.similarity_score:.2f}" if det.similarity_score else "0.00"
                id_txt = f"unknown ({score_txt})"
                bg = _C["unknown_bg"]
            else:
                id_txt = None

            if id_txt:
                label_y = _put_label(display, id_txt, x1, label_y,
                                      bg, _C["id_txt"])

        # ── Expression (Layer 5) ──────────────────────────────────────────
        if enable_expression and det.dominant_expression:
            expr_txt = (f"{det.dominant_expression} "
                        f"{det.expression_confidence:.0%}"
                        if det.expression_confidence
                        else det.dominant_expression)
            _put_label(display, expr_txt, x1, label_y,
                       _C["expr_bg"], _C["expr_txt"])

            # Probability bars (top-3)
            if det.expression_scores:
                top3 = sorted(det.expression_scores.items(),
                               key=lambda kv: kv[1], reverse=True)[:3]
                bx, by = x2 + 6, y1
                bw, bh, gap = 80, 12, 2
                for cls_name, prob in top3:
                    if bx + bw + 35 > fw:
                        break
                    cv2.rectangle(display, (bx, by), (bx + bw, by + bh),
                                  _C["bar_bg"], -1)
                    cv2.rectangle(display, (bx, by),
                                  (bx + int(bw * prob), by + bh),
                                  _C["bar_fg"], -1)
                    cv2.putText(display, f"{cls_name[:5]} {prob:.0%}",
                                (bx + 2, by + bh - 2),
                                _F, 0.34, (255, 255, 255), 1)
                    by += bh + gap

        # Landmarks (Layer 3, if available)
        if det.landmarks_original:
            for lx, ly, _ in det.landmarks_original:
                cv2.circle(display, (int(lx), int(ly)), 4, (0, 120, 255), -1)

    # HUD
    hud = f"FPS: {display_fps:5.1f}  |  Faces: {n_faces}  |  Frame: {frame_seq}"
    cv2.putText(display, hud, (10, 28), _F, 0.6, _C["hud"], 2)
    cv2.putText(display, "Q: Quit  |  F: Fullscreen",
                (10, display.shape[0] - 10), _F, 0.45, _C["hint"], 1)
    return display


# ─── Live Pipeline ────────────────────────────────────────────────────────────

def run_pipeline(
    source: str,
    camera_id: str,
    validate: bool = False,
    enable_identity: bool = True,
    enable_expression: bool = True
):
    """
    Run the full Layer 1 → 2 → 3 → (4) → (5) pipeline.

    Layers 4 and 5 can be disabled via flags for partial-pipeline runs.
    """
    from src.layer1_ingestion.capture import VideoCapture
    from src.layer2_preprocessing.preprocessor import Preprocessor
    from src.layer3_detection.detector import FaceDetector

    print(f"\n  Initializing pipeline components...")
    preprocessor = Preprocessor()
    detector = FaceDetector(model_path="models/yolov8n-face.pt",
                            confidence_threshold=0.5)

    identifier = None
    if enable_identity:
        from src.layer4_identity.identifier import FaceIdentifier
        identifier = FaceIdentifier()

    analyser = None
    if enable_expression and enable_identity:
        from src.layer5_expression.analyser import ExpressionAnalyser
        analyser = ExpressionAnalyser()
    elif enable_expression and not enable_identity:
        print("  [Warn] --no-identity was set; Layer 5 requires Layer 4. "
              "Skipping expression analysis.")

    # VideoWriter
    writer = None
    output_path = None
    writer_ready = False

    print(f"\n  Starting capture. Press [Q] to quit, [F] to toggle fullscreen.\n")

    WIN_NAME = "The Watcher -- Pipeline"
    _gui_window = False
    try:
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        _gui_window = True
        cv2.resizeWindow(WIN_NAME, 960, 540)
    except cv2.error:
        print("  [ERROR] cv2.namedWindow failed — opencv-python-headless may be installed.")
        print("  [FIX]   pip uninstall opencv-python-headless -y && "
              "pip install --force-reinstall opencv-python")
    _is_fullscreen = False
    _stop = False  # shared stop flag for X button / Q key

    fps_start = time.time()
    fps_count = 0
    display_fps = 0.0
    frame_seq = 0  # Initialize to prevent NameError if loop never executes

    try:
        with VideoCapture(source, camera_id=camera_id) as cap:
            print(f"  Source: {cap}\n")

            for frame_seq, timestamp, original_frame in cap.frames():

                # Layer 2: Preprocess
                ctx = preprocessor.process(
                    original_frame,
                    camera_id=camera_id,
                    timestamp=timestamp,
                    frame_seq=frame_seq
                )

                # Layer 3: Detect
                ctx = detector.detect(ctx)

                # Layer 4: Identity (optional)
                if identifier is not None:
                    ctx = identifier.identify(ctx)

                # Layer 5: Expression (optional)
                if analyser is not None:
                    ctx = analyser.analyse(ctx)

                # FPS
                fps_count += 1
                if fps_count >= 30:
                    elapsed = time.time() - fps_start
                    display_fps = fps_count / elapsed if elapsed > 0 else 0.0
                    fps_count = 0
                    fps_start = time.time()

                # Draw
                display = original_frame.copy()
                display = draw_all_detections(
                    display, ctx.detections, frame_seq, display_fps,
                    enable_identity=(identifier is not None),
                    enable_expression=(analyser is not None)
                )

                # VideoWriter
                if validate:
                    if not writer_ready:
                        fh_w, fw_w = display.shape[:2]
                        ts_str = time.strftime("%Y%m%d_%H%M%S")
                        output_path = f"output/{camera_id}_{ts_str}.mp4"
                        os.makedirs("output", exist_ok=True)
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(output_path, fourcc, cap.fps, (fw_w, fh_w))
                        print(f"  [Validate] Saving → {output_path}")
                        writer_ready = True
                    if writer is not None:
                        writer.write(display)

                cv2.imshow(WIN_NAME, display)

                # ── X button / Q key exit detection ──────────────────────────
                # cv2.getWindowProperty returns -1.0 when the window has been
                # closed by the OS (X button). The < 1 check catches both
                # -1.0 (destroyed) and 0.0 (hidden/minimised then closed).
                if _gui_window:
                    try:
                        vis = cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE)
                        if vis < 1:
                            print("\n  Window closed by user (X button). Stopping.")
                            _stop = True
                    except cv2.error:
                        _stop = True

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # Q or ESC
                    print("\n  Stopped by user (Q / ESC key).")
                    _stop = True
                elif key == ord("f") and _gui_window:
                    _is_fullscreen = not _is_fullscreen
                    prop = cv2.WINDOW_FULLSCREEN if _is_fullscreen else cv2.WINDOW_NORMAL
                    cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, prop)

                if _stop:
                    break

    finally:
        if writer is not None:
            writer.release()
            if output_path:
                print(f"  Annotated video saved → {output_path}")
        cv2.destroyAllWindows()

    print(f"\n  Session complete. Processed {frame_seq + 1} frames.\n")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="The Watcher — Face Detection + Identity + Expression Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py                         # Interactive menu, all layers\n"
            "  python main.py --source 0             # Webcam, all layers\n"
            "  python main.py --source 0 --no-identity --no-expression  # Layers 1-3 only\n"
            "  python main.py --source video.mp4 --validate\n"
        )
    )
    parser.add_argument("--validate", action="store_true",
                        help="Save annotated output video to output/")
    parser.add_argument("--source", type=str, default=None,
                        help="Source: '0' for webcam, file path, or rtsp:// URL")
    parser.add_argument("--no-identity", action="store_true",
                        help="Skip Layer 4 Identity (InsightFace). Useful before model download.")
    parser.add_argument("--no-expression", action="store_true",
                        help="Skip Layer 5 Expression Analysis.")

    args = parser.parse_args()

    enable_identity = not args.no_identity
    enable_expression = not args.no_expression

    print_banner(enable_identity, enable_expression)

    if args.source is not None:
        source = args.source
        if source.isdigit():
            camera_id = f"webcam_{source}"
        elif source.startswith("rtsp://"):
            camera_id = "rtsp_stream"
        else:
            camera_id = f"video_{os.path.splitext(os.path.basename(source))[0]}"
    else:
        source, camera_id = select_source()

    print(f"\n  Source confirmed : {source}")
    print(f"  Camera ID        : {camera_id}")
    print(f"  Validate mode    : {'ON' if args.validate else 'OFF'}")
    print(f"  Layer 4 Identity : {'ON' if enable_identity else 'OFF (--no-identity)'}")
    print(f"  Layer 5 Expression: {'ON' if enable_expression else 'OFF (--no-expression)'}")

    run_pipeline(
        source, camera_id,
        validate=args.validate,
        enable_identity=enable_identity,
        enable_expression=enable_expression
    )


if __name__ == "__main__":
    main()
