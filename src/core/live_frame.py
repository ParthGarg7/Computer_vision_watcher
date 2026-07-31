"""
src/core/live_frame.py
─────────────────────────────────────────────────────────────────────────────
Live-frame relay between the capture pipeline (writer) and the Layer 8 API
(reader).

main.py (the pipeline) and run_api.py (the API server) are separate processes,
and on Windows only one of them can hold the webcam. So the pipeline drops its
already-annotated frame here as a small JPEG and the API streams that file to
the dashboard — no second camera handle is ever opened.

Footprint / privacy (the frame IS the user's webcam):
  • throttled + downscaled JPEG → tiny CPU/RAM, no GPU work
  • written to ONE file in the OS temp dir (user-scoped, never the repo)
  • removed when the pipeline stops (LiveFrameWriter.close)
  • the API serves it localhost-only and only while it is FRESH
"""

import os
import tempfile
import time

# One path both processes agree on. Same machine + same user share the OS temp
# dir, so this needs no configuration and leaves nothing inside the project.
LIVE_FRAME_PATH = os.path.join(tempfile.gettempdir(), "watcher_live_frame.jpg")

# A frame older than this means the pipeline isn't writing (stopped / hung),
# so the API treats the feed as offline rather than serving a frozen image.
FRESH_WINDOW_SEC = 2.0


def frame_is_fresh(path: str = LIVE_FRAME_PATH,
                   window: float = FRESH_WINDOW_SEC) -> bool:
    """True if the relay file exists and was written within `window` seconds."""
    try:
        return (time.time() - os.path.getmtime(path)) <= window
    except OSError:
        return False


class LiveFrameWriter:
    """
    Throttled, downscaled JPEG writer for the pipeline's annotated frame.

    Call ``write(frame)`` every loop iteration; it rate-limits internally.
    Writes via an atomic temp-file swap so the API never reads a half-written
    frame.
    """

    def __init__(self, max_width: int = 640, max_fps: float = 15.0,
                 quality: int = 70, path: str = LIVE_FRAME_PATH):
        self.path = path
        self.tmp = path + ".tmp"
        self.max_width = max_width
        self.min_interval = 1.0 / max_fps
        self.quality = quality
        self._last = 0.0

    def write(self, frame) -> None:
        now = time.time()
        if now - self._last < self.min_interval:
            return
        import cv2  # lazy: keeps the read-only API import path cv2-free

        h, w = frame.shape[:2]
        if w > self.max_width:
            scale = self.max_width / float(w)
            frame = cv2.resize(frame, (self.max_width, int(h * scale)))
        ok, buf = cv2.imencode(".jpg", frame,
                               [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            return
        try:
            with open(self.tmp, "wb") as f:
                f.write(buf.tobytes())
            os.replace(self.tmp, self.path)   # atomic on the same filesystem
        except OSError:
            # Windows can briefly lock the target while the reader has it open;
            # drop this frame — the next one (~66 ms later) will land.
            return
        self._last = now

    def close(self) -> None:
        """Remove the relay file so no webcam frame lingers after a run."""
        for p in (self.tmp, self.path):
            try:
                os.remove(p)
            except OSError:
                pass
