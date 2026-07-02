#!/usr/bin/env python3
"""
main.py — The Watcher MVP Entry Point
─────────────────────────────────────────────────────────────────────────────
Computer Vision Pipeline: Layers 1 → 2 → 3 (Face Detection)

Terminal menu lets you choose the input source:
    [1] Webcam        — built-in or USB camera at index 0
    [2] Recorded Video — opens a native Windows file picker to select a file
    [3] RTSP Stream   — enter the stream URL manually

Flags:
    --validate    Also saves an annotated output video to the output/ folder
    --source STR  Skip the menu and use this source directly
                  (e.g. --source 0  or  --source video.mp4  or  --source rtsp://...)

Examples:
    python main.py                         # Interactive menu, live preview only
    python main.py --validate              # Interactive menu, save annotated video
    python main.py --source 0             # Webcam directly
    python main.py --source video.mp4 --validate  # Video file + save output
"""

import sys
import os
import time
import argparse

import cv2
import torch

# Suppress Ultralytics per-frame console output before any YOLO import
os.environ["YOLO_VERBOSE"] = "False"


# ─── Banner ───────────────────────────────────────────────────────────────────

def print_banner():
    gpu_info = "CPU only"
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_info = f"{name}  ({mem_gb:.1f} GB VRAM)"

    print()
    print("  ╔═══════════════════════════════════════════════════╗")
    print("  ║        THE WATCHER — Computer Vision MVP          ║")
    print("  ║     Layers 1 → 2 → 3  |  YOLOv8n Face Detection  ║")
    print("  ╚═══════════════════════════════════════════════════╝")
    print(f"  GPU  : {gpu_info}")
    print(f"  Time : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()


# ─── Input source selection ───────────────────────────────────────────────────

def select_source() -> tuple:
    """
    Interactive terminal menu.
    Returns (source: str, camera_id: str).
    """
    print("  ┌─────────────────────────────────────────────────┐")
    print("  │              Select Input Source                 │")
    print("  ├─────────────────────────────────────────────────┤")
    print("  │  [1]  Webcam          (built-in camera, index 0) │")
    print("  │  [2]  Recorded Video  (opens file browser)       │")
    print("  │  [3]  RTSP Stream     (enter URL manually)       │")
    print("  │  [Q]  Quit                                       │")
    print("  └─────────────────────────────────────────────────┘")
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
    """Configure webcam source at device index 0."""
    index = 0
    print(f"\n  Verifying webcam at device index {index}...")

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"  ERROR: Could not open webcam at index {index}.")
        print("  Check that your camera is connected and not in use by another app.")
        cap.release()
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"  Webcam ready: {w}x{h} at index {index}")
    return str(index), f"webcam_{index}"


def _select_video_file() -> tuple:
    """
    Open a native Windows file picker to select a video file.
    Falls back to manual path entry if tkinter is unavailable.
    """
    print("\n  Opening file browser...")
    file_path = None

    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()                  # Hide the blank root window
        root.attributes("-topmost", True)  # Bring dialog to front

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
    """Prompt for an RTSP stream URL."""
    print()
    print("  RTSP URL format: rtsp://username:password@ip_address:port/stream_path")
    print("  Example        : rtsp://admin:admin123@192.168.1.100:554/stream1")
    print("  No-auth example: rtsp://192.168.1.100:554/live")
    print()

    try:
        url = input("  Enter RTSP URL: ").strip()
    except (KeyboardInterrupt, EOFError):
        sys.exit(0)

    if not url:
        print("  No URL entered. Returning to menu.\n")
        return select_source()

    if not url.startswith("rtsp://"):
        print("  WARNING: URL does not start with 'rtsp://'. Proceeding anyway.")

    # Derive a safe camera_id from the URL (remove credentials)
    try:
        host_part = url.split("@")[-1].split("/")[0]
        cam_id = f"rtsp_{host_part}".replace(":", "_").replace(".", "_")
    except Exception:
        cam_id = "rtsp_stream"

    print(f"  Testing RTSP connection (this may take a few seconds)...")
    cap = cv2.VideoCapture(url)
    if cap.isOpened():
        ret, _ = cap.read()
        cap.release()
        if ret:
            print(f"  RTSP connection: OK")
        else:
            print(f"  RTSP opened but no frame received. Stream may be slow to start.")
    else:
        print(f"  WARNING: Could not verify RTSP stream. Will attempt at pipeline start.")

    return url, cam_id


# ─── Live Pipeline ────────────────────────────────────────────────────────────

def run_pipeline(source: str, camera_id: str, validate: bool = False):
    """
    Run the full Layer 1 → 2 → 3 pipeline.

    Displays a live preview window with bounding boxes drawn on each frame.
    If validate=True, also saves an annotated video to the output/ directory.

    Press Q in the preview window to stop.
    """
    from src.layer1_ingestion.capture import VideoCapture
    from src.layer2_preprocessing.preprocessor import Preprocessor
    from src.layer3_detection.detector import FaceDetector

    # Drawing constants (BGR color tuples)
    BOX_COLOR       = (0, 220, 80)    # Green
    LANDMARK_COLOR  = (0, 120, 255)   # Orange
    CONF_BG_COLOR   = (0, 180, 60)    # Dark green
    CONF_TEXT_COLOR = (255, 255, 255) # White
    HUD_COLOR       = (0, 220, 220)   # Cyan
    HINT_COLOR      = (160, 160, 160) # Grey
    FONT            = cv2.FONT_HERSHEY_SIMPLEX

    print(f"\n  Initializing pipeline components...")

    preprocessor = Preprocessor()
    detector = FaceDetector(
        model_path="models/yolov8n-face.pt",
        confidence_threshold=0.5
    )

    # VideoWriter (only in validate mode)
    writer = None
    output_path = None
    writer_ready = False

    print(f"\n  Starting capture. Press [Q] to quit, [F] to toggle fullscreen.\n")

    # Create resizable window BEFORE the frame loop.
    # WINDOW_NORMAL: frame stretches to fill the window when resized or maximized.
    # Without this flag, the window is fixed-size and shows a black border.
    WIN_NAME = "The Watcher -- Layer 3 Detection"
    _gui_window = False
    try:
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        _gui_window = True
    except cv2.error:
        # Happens when opencv-python-headless is installed instead of opencv-python.
        # Both packages cannot coexist -- headless wins and disables all GUI functions.
        # Fix: pip uninstall opencv-python-headless; pip install --force-reinstall opencv-python
        print()
        print("  [ERROR] cv2.namedWindow failed -- opencv-python-headless is likely installed.")
        print("  [FIX]   Run: pip uninstall opencv-python-headless -y")
        print("  [FIX]   Then: pip install --force-reinstall opencv-python")
        print("  [INFO]  Falling back to fixed-size window (fullscreen/resize disabled).")
        print()
    _is_fullscreen = False

    # FPS tracking
    fps_start   = time.time()
    fps_count   = 0
    display_fps = 0.0

    try:
        with VideoCapture(source, camera_id=camera_id) as cap:
            print(f"  Source: {cap}\n")

            for frame_seq, timestamp, original_frame in cap.frames():

                # ── Layer 2: Preprocess ──────────────────────────────────────
                ctx = preprocessor.process(
                    original_frame,
                    camera_id=camera_id,
                    timestamp=timestamp,
                    frame_seq=frame_seq
                )

                # ── Layer 3: Detect ──────────────────────────────────────────
                ctx = detector.detect(ctx)

                # ── FPS calculation ──────────────────────────────────────────
                fps_count += 1
                if fps_count >= 30:
                    elapsed = time.time() - fps_start
                    display_fps = fps_count / elapsed if elapsed > 0 else 0.0
                    fps_count = 0
                    fps_start = time.time()

                # ── Draw detections ──────────────────────────────────────────
                display = original_frame.copy()
                n_faces = len(ctx.detections)

                for det in ctx.detections:
                    x1, y1, x2, y2 = [int(v) for v in det.bbox_original]
                    fh, fw = display.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(fw - 1, x2), min(fh - 1, y2)

                    # Bounding box
                    cv2.rectangle(display, (x1, y1), (x2, y2), BOX_COLOR, 2)

                    # Confidence label
                    label = f"{det.confidence:.2f}"
                    (lw, lh), base = cv2.getTextSize(label, FONT, 0.52, 1)
                    ly_top = max(0, y1 - lh - base - 6)
                    cv2.rectangle(display, (x1, ly_top), (x1 + lw + 6, y1), CONF_BG_COLOR, -1)
                    cv2.putText(display, label, (x1 + 3, y1 - base - 2), FONT, 0.52, CONF_TEXT_COLOR, 1)

                    # Landmark dots
                    if det.landmarks_original:
                        for lx, ly, _ in det.landmarks_original:
                            cv2.circle(display, (int(lx), int(ly)), 5, LANDMARK_COLOR, -1)

                # HUD overlay
                hud = f"FPS: {display_fps:5.1f}  |  Faces: {n_faces}  |  Frame: {frame_seq}"
                cv2.putText(display, hud, (10, 28), FONT, 0.6, HUD_COLOR, 2)

                # Bottom hint (F = fullscreen, Q = quit)
                fh_d = display.shape[0]
                cv2.putText(display, "Q: Quit  |  F: Fullscreen", (10, fh_d - 10), FONT, 0.45, HINT_COLOR, 1)

                # ── VideoWriter (validate mode) ──────────────────────────────
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

                # ── Show preview ─────────────────────────────────────────────
                cv2.imshow(WIN_NAME, display)

                # ── X button close detection ──────────────────────────────────
                # cv2.waitKey() only catches keyboard, not the window close button.
                # WND_PROP_VISIBLE returns < 1 when user clicks the X button.
                # Only available when WINDOW_NORMAL is supported (not headless).
                if _gui_window and cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    print("\n  Window closed by user (X button).")
                    break

                # ── Key handling ──────────────────────────────────────────────
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    print("\n  Stopped by user (Q key).")
                    break

                elif key == ord("f") and _gui_window:
                    # Toggle between fullscreen and normal window
                    _is_fullscreen = not _is_fullscreen
                    if _is_fullscreen:
                        cv2.setWindowProperty(
                            WIN_NAME,
                            cv2.WND_PROP_FULLSCREEN,
                            cv2.WINDOW_FULLSCREEN
                        )
                    else:
                        cv2.setWindowProperty(
                            WIN_NAME,
                            cv2.WND_PROP_FULLSCREEN,
                            cv2.WINDOW_NORMAL
                        )

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
        description="The Watcher — Face Detection Pipeline (Layers 1-3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py                         # Interactive menu\n"
            "  python main.py --validate              # Menu + save annotated video\n"
            "  python main.py --source 0             # Webcam directly\n"
            "  python main.py --source video.mp4     # Video file directly\n"
            "  python main.py --source rtsp://... --validate\n"
        )
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Save an annotated output video to the output/ folder."
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help=(
            "Skip the menu and use this source directly. "
            "Pass '0' for webcam, a file path, or an rtsp:// URL."
        )
    )
    args = parser.parse_args()

    print_banner()

    # Determine source
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
    print(f"  Validate mode    : {'ON (saving annotated video)' if args.validate else 'OFF'}")

    run_pipeline(source, camera_id, validate=args.validate)


if __name__ == "__main__":
    main()
