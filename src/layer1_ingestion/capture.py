"""
src/layer1_ingestion/capture.py
─────────────────────────────────────────────────────────────────────────────
Layer 1: Ingestion

The sole responsibility of this layer is to capture raw video frames from a
source and yield them as BGR NumPy arrays. No processing, detection, or
analysis happens here.

Supported sources (all via a single cv2.VideoCapture interface):
    - Webcam      : pass integer index 0 (or "0" as string)
    - Video file  : pass a file path string e.g. "C:/Videos/test.mp4"
    - RTSP stream : pass an RTSP URL string e.g. "rtsp://user:pass@ip/stream"

Ref: Layer 1 Architecture Doc — OpenCV (cv2) section
"""

import cv2
import time
from typing import Generator

from src.core.logger import get_logger

log = get_logger("watcher.layer1")


class VideoCapture:
    """
    Layer 1 VideoCapture wrapper.

    Provides a unified interface for webcam, video files, and RTSP streams.
    Yields raw BGR NumPy arrays (H, W, 3) uint8 — the exact format the rest
    of the pipeline expects as input.

    For RTSP streams, auto-reconnect is attempted up to `reconnect_attempts`
    times if the connection drops.

    Usage
    -----
        with VideoCapture("0") as cap:               # webcam
            for seq, ts, frame in cap.frames():
                # frame is BGR (H, W, 3) uint8

        with VideoCapture("video.mp4") as cap:       # video file
            ...

        with VideoCapture("rtsp://...") as cap:      # RTSP
            ...
    """

    def __init__(
        self,
        source,
        camera_id: str = "cam0",
        reconnect_attempts: int = 3,
        reconnect_delay_sec: float = 2.0
    ):
        """
        Parameters
        ----------
        source : int or str
            Integer webcam index, video file path, or RTSP URL string.
            A string containing only digits (e.g. "0") is automatically
            converted to int so the webcam is correctly selected.
        camera_id : str
            Human-readable identifier carried into FrameContext objects.
        reconnect_attempts : int
            How many times to retry an RTSP connection on failure.
        reconnect_delay_sec : float
            Seconds to wait between reconnect attempts.
        """
        self.source = source
        self.camera_id = camera_id
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay_sec = reconnect_delay_sec
        self._cap = None
        self._connect()

    # ─── Internal ────────────────────────────────────────────────────────────

    def _resolve_source(self):
        """Convert string digit sources to int for webcam."""
        if isinstance(self.source, str) and self.source.isdigit():
            return int(self.source)
        return self.source

    def _connect(self):
        """Open or re-open the capture device."""
        if self._cap is not None:
            self._cap.release()

        src = self._resolve_source()
        self._cap = cv2.VideoCapture(src)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"[Layer1] Failed to open source: {self.source!r}\n"
                f"  Check that the webcam/file/RTSP URL is correct and accessible."
            )
        log.debug(f"[Layer1] connected: source={self.source!r} "
                  f"{self.width}x{self.height} @ {self.fps:.1f}fps "
                  f"live={self.is_live}")

    def _attempt_reconnect(self, kind: str):
        """
        Try to recover a LIVE source (webcam or RTSP) after a read failure.

        Webcams need this as much as RTSP: a USB or driver hiccup makes
        cv2 return one failed read, and treating that as end-of-stream ends
        the whole run — which, mid-demo, is indistinguishable from a crash.
        The window closes, no explanation. (Observed live: runs that ended
        abruptly after ~30-45s with the camera image healthy to the final
        frame.) A live source gets reconnect_attempts chances to come back
        before we give up, and every attempt is logged.

        Returns (success, frame) tuple.
        """
        for attempt in range(1, self.reconnect_attempts + 1):
            log.warning(
                f"  [Layer1] {kind} stopped delivering frames. "
                f"Reconnect attempt {attempt}/{self.reconnect_attempts} in "
                f"{self.reconnect_delay_sec}s..."
            )
            time.sleep(self.reconnect_delay_sec)
            try:
                self._connect()
                ret, frame = self._cap.read()
                if ret and frame is not None:
                    log.info(f"  [Layer1] Reconnected successfully.")
                    return True, frame
            except RuntimeError:
                pass
        log.error(f"  [Layer1] All {self.reconnect_attempts} reconnect "
                  f"attempts failed — the {kind} is gone. Stopping stream.")
        return False, None

    # ─── Properties ──────────────────────────────────────────────────────────

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def fps(self) -> float:
        """Source FPS. Falls back to 30.0 if not available (e.g. webcam)."""
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        return fps if fps > 0 else 30.0

    @property
    def total_frames(self) -> int:
        """
        Total frame count for video files.
        Returns -1 for live sources (webcam / RTSP) where it is undefined.
        """
        total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return total if total > 0 else -1

    @property
    def is_live(self) -> bool:
        """True for webcam and RTSP sources (no finite end point)."""
        return self.total_frames == -1

    # ─── Frame reading ────────────────────────────────────────────────────────

    def read_frame(self) -> tuple:
        """
        Read a single frame.

        Returns
        -------
        (success: bool, frame: np.ndarray or None)
            success=False and frame=None signals that the source is exhausted
            or cannot be recovered (end of video file, RTSP failure).
        """
        ret, frame = self._cap.read()

        if not ret or frame is None:
            # LIVE sources (webcam, RTSP) get reconnect attempts — one failed
            # read must not end a live run. Video files: EOF is normal.
            src = self._resolve_source()
            if isinstance(src, str) and src.startswith("rtsp://"):
                return self._attempt_reconnect("RTSP stream")
            if isinstance(src, int):
                return self._attempt_reconnect(f"webcam {src}")
            log.info(f"  [Layer1] End of video file reached.")
            return False, None

        return True, frame

    def frames(self) -> Generator:
        """
        Generator that yields (frame_seq, timestamp, frame) tuples.

        frame_seq : int   — monotonically increasing index starting at 0
        timestamp : float — capture time as a Unix timestamp.
            Live sources (webcam/RTSP): wall clock (time.time()).
            Video files: start_time + frame_seq / fps — media time, so
            downstream analytics (presence durations, trend windows) reflect
            the video's own timeline rather than how fast this machine
            happens to decode it.
        frame     : np.ndarray — BGR uint8 array (H, W, 3)

        Stops when the source is exhausted or cannot reconnect.

        Example
        -------
            for seq, ts, frame in cap.frames():
                # process frame
        """
        seq = 0
        # Cache source properties once — is_live/fps query the backend and
        # must not run per frame.
        live = self.is_live
        fps = self.fps
        start_time = time.time()
        while True:
            ret, frame = self.read_frame()
            if not ret:
                # The reason the run is ending — the log line that was
                # missing when demo runs terminated with no explanation.
                log.warning(f"  [Layer1] Frame stream ended at frame {seq} "
                            f"({'live source lost' if live else 'end of file'}).")
                break
            ts = time.time() if live else start_time + seq / fps
            yield seq, ts, frame
            seq += 1

    # ─── Lifecycle ────────────────────────────────────────────────────────────

    def release(self):
        """Release the capture device. Safe to call multiple times."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()

    def __repr__(self) -> str:
        return (
            f"VideoCapture("
            f"source={self.source!r}, "
            f"{self.width}x{self.height} @ {self.fps:.1f}fps, "
            f"frames={'live' if self.is_live else self.total_frames}"
            f")"
        )
