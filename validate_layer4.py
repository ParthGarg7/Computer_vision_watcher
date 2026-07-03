#!/usr/bin/env python3
"""
validate_layer4.py — Layer 4 Identity Validation Entry Point

Runs the Layer 1 → 2 → 3 → 4 pipeline on a source and saves an annotated
video showing track IDs and identity labels. Mirrors validate_layer3.py.

Usage:
    python validate_layer4.py                          # webcam
    python validate_layer4.py --source video.mp4
    python validate_layer4.py --source 0 --max-frames 300
    python validate_layer4.py --source video.mp4 --no-preview

Run from the project root (Computer_vision_watcher/).
"""

import argparse
import sys
import os

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.layer4_identity.validator import Layer4ValidationPipeline


def main():
    parser = argparse.ArgumentParser(
        description="The Watcher — Layer 4 Identity Validation"
    )
    parser.add_argument("--source", type=str, default="0",
                        help="Source: '0' for webcam, path to video, or rtsp:// URL")
    parser.add_argument("--store", type=str, default="models/identity_store",
                        help="Base path for FAISS identity store")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum frames to process (useful for quick test)")
    parser.add_argument("--no-preview", action="store_true",
                        help="Disable live preview window (headless mode)")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Face detection confidence threshold")
    args = parser.parse_args()

    source = args.source
    camera_id = "webcam_0" if source == "0" else f"source_{os.path.basename(str(source))}"

    vp = Layer4ValidationPipeline(
        source=source,
        confidence_threshold=args.confidence,
        store_path=args.store,
        camera_id=camera_id
    )
    stats = vp.run(
        show_preview=not args.no_preview,
        max_frames=args.max_frames
    )
    sys.exit(0 if stats["errors"] == 0 else 1)


if __name__ == "__main__":
    main()
