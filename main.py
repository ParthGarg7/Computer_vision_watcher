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

# ── NVIDIA DLL registration (MUST happen before any ONNX/GPU imports) ─────────
# Shared with the layer validators — see src/core/gpu_setup.py.
# ONNX Runtime's CUDAExecutionProvider needs cublas64_12.dll, cudnn64_9.dll etc.
# on the Windows DLL search path; without this it silently falls back to CPU.
from src.core.gpu_setup import register_nvidia_dlls
register_nvidia_dlls()

import cv2
import numpy as np
import torch

from src.core import drawing

os.environ["YOLO_VERBOSE"] = "False"


# ─── Banner ───────────────────────────────────────────────────────────────────

def print_banner(enable_identity: bool = True, enable_expression: bool = True,
                 enable_analytics: bool = True):
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
    if enable_analytics and enable_identity:
        layers_str += " → 6 → 7"

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
    except Exception as e:
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


# ─── Drawing ──────────────────────────────────────────────────────────────────
# All overlay rendering lives in src/core/drawing.py, shared with the three
# layer validators — see that module for the colour convention.


def draw_all_detections(
    display: np.ndarray,
    detections: list,
    frame_seq: int,
    display_fps: float,
    enable_identity: bool,
    enable_expression: bool
) -> np.ndarray:
    """Draw all layer outputs onto the display frame."""
    drawing.draw_detections(
        display, detections,
        show_confidence=True,
        show_track=enable_identity,
        show_identity=enable_identity,
        show_expression=enable_expression,
        show_mood=enable_expression,
        show_bars=enable_expression,
        show_landmarks=True,
    )
    drawing.draw_hud(
        display,
        f"FPS: {display_fps:5.1f}  |  Faces: {len(detections)}  |  "
        f"Frame: {frame_seq}")
    drawing.draw_hint(display)
    return display


# ─── Live Pipeline ────────────────────────────────────────────────────────────

def run_pipeline(
    source: str,
    camera_id: str,
    validate: bool = False,
    enable_identity: bool = True,
    enable_expression: bool = True,
    enable_analytics: bool = True
):
    """
    Run the full Layer 1 → 2 → 3 → (4) → (5) → (6 → 7) pipeline.

    Layers 4-7 can be disabled via flags for partial-pipeline runs.
    Layers 6 (Analytics) and 7 (Storage) require Layer 4 (track IDs).
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

    # Layers 6 (Analytics) + 7 (Storage) — session metrics need track IDs
    aggregator = None
    storage = None
    if enable_analytics and enable_identity:
        from src.layer6_analytics.aggregator import SessionAggregator
        from src.layer7_storage.store import StorageLayer
        storage = StorageLayer()
        aggregator = SessionAggregator(storage=storage)
    elif enable_analytics and not enable_identity:
        print("  [Warn] --no-identity was set; Layers 6/7 require Layer 4 "
              "track IDs. Skipping analytics and storage.")

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
                    # Drop throttle/smoothing state for ended tracks so
                    # per-track buffers don't grow forever in long sessions.
                    if frame_seq % 100 == 0:
                        active_ids = {d.track_id for d in ctx.detections
                                      if d.track_id is not None}
                        analyser.clear_stale_tracks(active_ids)

                # Layers 6 + 7: Analytics → Storage (optional)
                if aggregator is not None:
                    for event in aggregator.process(ctx):
                        et = event["event_type"]
                        if et == "presence_alert":
                            who = event["identity_label"] or "unknown"
                            print(f"  [Layer6] Track {event['track_id']} "
                                  f"({who}) {event['presence']}")
                        elif et == "threshold_alert":
                            who = event["identity_label"] or "unknown"
                            print(f"  [Layer6] ALERT track {event['track_id']} "
                                  f"({who}): {event['metric']} = "
                                  f"{event['value']} > {event['threshold']}")
                        # live_expression_update events are for Layer 8
                        # (WebSocket push) — not printed to keep console quiet.

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
                # destroyed by the OS (user clicked X). We check both
                # WND_PROP_VISIBLE (== 0 when closed) AND WND_PROP_AUTOSIZE
                # (== -1 when window no longer exists) for cross-platform
                # reliability on Windows + OpenCV builds.
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # Q or ESC
                    print("\n  Stopped by user (Q / ESC key).")
                    _stop = True
                elif key == ord("f") and _gui_window:
                    _is_fullscreen = not _is_fullscreen
                    prop = cv2.WINDOW_FULLSCREEN if _is_fullscreen else cv2.WINDOW_NORMAL
                    cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, prop)

                if _gui_window and not _stop:
                    try:
                        # WND_PROP_VISIBLE: -1.0 = destroyed, 0.0 = hidden
                        vis = cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE)
                        # WND_PROP_AUTOSIZE: -1.0 = window no longer exists
                        auto = cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_AUTOSIZE)
                        if vis < 0 or auto < 0:
                            print("\n  Window closed by user (X button). Stopping.")
                            _stop = True
                    except cv2.error:
                        _stop = True

                if _stop:
                    break

    finally:
        if writer is not None:
            writer.release()
            if output_path:
                print(f"  Annotated video saved → {output_path}")
        if aggregator is not None:
            aggregator.close()  # close appearances + finalise sessions → L7
        if storage is not None:
            storage.close()
            print(f"\n  [Layer7] Persisted: {storage.n_sessions} sessions, "
                  f"{storage.n_appearances} appearances, "
                  f"{storage.n_expression_events} expression events, "
                  f"{storage.n_presence_events} presence events "
                  f"→ {storage.db_path}")
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
    parser.add_argument("--no-analytics", action="store_true",
                        help="Skip Layers 6 (Analytics) and 7 (Storage).")

    args = parser.parse_args()

    enable_identity = not args.no_identity
    enable_expression = not args.no_expression
    enable_analytics = not args.no_analytics

    print_banner(enable_identity, enable_expression, enable_analytics)

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
    print(f"  Layers 6+7 Analytics/Storage: "
          f"{'ON' if enable_analytics else 'OFF (--no-analytics)'}")

    run_pipeline(
        source, camera_id,
        validate=args.validate,
        enable_identity=enable_identity,
        enable_expression=enable_expression,
        enable_analytics=enable_analytics
    )


if __name__ == "__main__":
    main()
