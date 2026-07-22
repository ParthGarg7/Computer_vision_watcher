"""
tests/test_logging_and_offline.py
─────────────────────────────────────────────────────────────────────────────
Tests for the crash-hardening added after the pipeline died twice during a
live demo with WiFi off:

  1. src/core/logger.py — messages and uncaught-exception tracebacks must
     reach the log file (there was previously NO record of any crash).
  2. ExpressionAnalyser offline behaviour — a model that cannot be loaded
     (cold cache + no internet) must degrade Layer 5 to a no-op, never
     take the whole pipeline down.

Run:  python -m unittest discover tests
"""

import logging
import os
import sys
import tempfile
import unittest
from unittest import mock

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.core import logger as watcher_logger
from src.layer5_expression import analyser as analyser_mod
from src.layer5_expression.analyser import ExpressionAnalyser


class LoggerTests(unittest.TestCase):
    """Configure logging into a temp dir and prove messages hit the file."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        # Reset module + root logger state so each test configures cleanly
        watcher_logger._configured = False
        root = logging.getLogger()
        self._saved_handlers = root.handlers[:]
        for h in root.handlers[:]:
            root.removeHandler(h)
        self._saved_hook = sys.excepthook

    def tearDown(self):
        root = logging.getLogger()
        for h in root.handlers[:]:
            h.close()
            root.removeHandler(h)
        for h in self._saved_handlers:
            root.addHandler(h)
        sys.excepthook = self._saved_hook
        watcher_logger._configured = False
        self._dir.cleanup()

    def _log_path(self):
        return os.path.join(self._dir.name, watcher_logger.LOG_FILE)

    def _read_log(self):
        with open(self._log_path(), encoding="utf-8") as f:
            return f.read()

    def test_messages_reach_the_file_with_metadata(self):
        watcher_logger.setup_logging(log_dir=self._dir.name)
        watcher_logger.get_logger("watcher.test").warning("belt AND braces")
        content = self._read_log()
        self.assertIn("belt AND braces", content)
        self.assertIn("WARNING", content)          # level recorded
        self.assertIn("watcher.test", content)     # component recorded

    def test_debug_goes_to_file_only(self):
        watcher_logger.setup_logging(log_dir=self._dir.name)
        watcher_logger.get_logger("watcher.test").debug("file-only detail")
        self.assertIn("file-only detail", self._read_log())

    def test_uncaught_exception_traceback_is_captured(self):
        # THE reason this module exists: the demo crashes left no trace.
        watcher_logger.setup_logging(log_dir=self._dir.name)
        try:
            raise RuntimeError("simulated demo crash")
        except RuntimeError:
            watcher_logger._log_uncaught(*sys.exc_info())
        content = self._read_log()
        self.assertIn("UNCAUGHT EXCEPTION", content)
        self.assertIn("simulated demo crash", content)
        self.assertIn("Traceback", content)        # full traceback, not just msg

    def test_excepthook_installed(self):
        watcher_logger.setup_logging(log_dir=self._dir.name)
        self.assertIs(sys.excepthook, watcher_logger._log_uncaught)

    def test_setup_is_idempotent(self):
        watcher_logger.setup_logging(log_dir=self._dir.name)
        n = len(logging.getLogger().handlers)
        watcher_logger.setup_logging(log_dir=self._dir.name)
        self.assertEqual(len(logging.getLogger().handlers), n,
                         "second setup must not duplicate handlers")


def make_offline_analyser(cached=(), loadable=()):
    """
    Build an ExpressionAnalyser in a simulated offline world.

    cached   — model names that exist on disk (so fallbacks may try them)
    loadable — model names whose construction succeeds; anything else
               raises URLError, as a real offline download does.
    """
    from urllib.error import URLError

    def fake_is_cached(name):
        return name in cached

    def fake_load(name):
        if name in loadable:
            m = mock.Mock()
            m.idx_to_class = {i: c.capitalize() for i, c in
                              enumerate(analyser_mod.EMOTION_CLASSES)}
            return m
        raise URLError("no internet (simulated)")

    with mock.patch.object(analyser_mod, "_is_cached", fake_is_cached), \
         mock.patch.object(ExpressionAnalyser, "_load_model",
                           staticmethod(fake_load)), \
         mock.patch.object(ExpressionAnalyser, "_patch_model_to_gpu",
                           lambda self: None):
        return ExpressionAnalyser()


class OfflineFallbackTests(unittest.TestCase):
    """The crash scenario: cold model cache, no internet."""

    def test_total_offline_disables_layer5_without_raising(self):
        # Nothing cached, nothing downloadable — the old code died HERE.
        a = make_offline_analyser(cached=(), loadable=())
        self.assertIsNone(a._model)

    def test_disabled_analyser_is_a_clean_noop(self):
        a = make_offline_analyser(cached=(), loadable=())

        class Det:
            track_id = 1
            face_crop = None
            expression_scores = None
            dominant_expression = None

        class Ctx:
            detections = [Det()]

        ctx = Ctx()
        result = a.analyse(ctx)              # must not raise
        self.assertIs(result, ctx)
        self.assertIsNone(ctx.detections[0].dominant_expression)

    def test_falls_back_to_cached_model_when_download_fails(self):
        # Default model not cached and not downloadable; vgaf IS cached.
        a = make_offline_analyser(cached=("enet_b0_8_best_vgaf",),
                                  loadable=("enet_b0_8_best_vgaf",))
        self.assertIsNotNone(a._model)
        self.assertEqual(a.model_name, "enet_b0_8_best_vgaf")

    def test_uncached_fallbacks_are_not_attempted(self):
        # Offline with an empty cache: trying every fallback would mean
        # three failed downloads. Only the requested model may be attempted.
        attempts = []
        from urllib.error import URLError

        def fake_load(name):
            attempts.append(name)
            raise URLError("offline")

        with mock.patch.object(analyser_mod, "_is_cached",
                               lambda n: False), \
             mock.patch.object(ExpressionAnalyser, "_load_model",
                               staticmethod(fake_load)), \
             mock.patch.object(ExpressionAnalyser, "_patch_model_to_gpu",
                               lambda self: None):
            ExpressionAnalyser()
        self.assertEqual(attempts, [analyser_mod.DEFAULT_MODEL])

    def test_online_first_run_still_attempts_download(self):
        # The requested model is not cached but IS downloadable (online) —
        # the legitimate first-run path must still work.
        a = make_offline_analyser(cached=(),
                                  loadable=(analyser_mod.DEFAULT_MODEL,))
        self.assertIsNotNone(a._model)
        self.assertEqual(a.model_name, analyser_mod.DEFAULT_MODEL)


if __name__ == "__main__":
    unittest.main()
