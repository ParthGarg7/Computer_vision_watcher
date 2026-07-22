"""
src/core/logger.py
─────────────────────────────────────────────────────────────────────────────
Central logging for every pipeline component.

Why this exists: the pipeline crashed twice during a live demo and left no
record of where or why — every diagnostic was a print() that died with the
console window. This module gives every component a logger that writes to
BOTH:

    console   — bare messages, so the terminal output looks exactly as
                before (the banner, the [LayerN] init lines, the alerts)
    logs/watcher.log
              — every message, timestamped, with level and component name,
                rotated at 5 MB (5 backups kept). Survives the crash.

plus a global exception hook: any UNCAUGHT exception is written to the log
file with its full traceback before the process dies. The next unexplained
crash will be in logs/watcher.log.

Usage
-----
    # Entry points (main.py, validators, registration CLI), once, first:
    from src.core.logger import setup_logging
    setup_logging()

    # Every module:
    from src.core.logger import get_logger
    log = get_logger("watcher.layer4.tracker")
    log.info("...")        # console + file
    log.warning("...")     # console + file, flagged WARNING in file
    log.exception("...")   # inside an except block: message + traceback

Levels: DEBUG goes to the file only; INFO and above go to both. So chatty
diagnostics can use log.debug() without spamming the console.

logs/ is gitignored — log lines contain identity labels and timestamps,
which is behavioural data that must not reach the repository.
"""

import logging
import logging.handlers
import os
import sys

LOG_DIR = "logs"
LOG_FILE = "watcher.log"

# Rotate at 5 MB, keep 5 backups — bounded disk use, weeks of history.
_MAX_BYTES = 5 * 1024 * 1024
_BACKUPS = 5

_configured = False


def setup_logging(log_dir: str = LOG_DIR, console_level: int = logging.INFO):
    """
    Configure root logging: rotating file (DEBUG+) + console (INFO+).

    Idempotent — safe to call from every entry point; only the first call
    does anything. Returns the "watcher" logger.
    """
    global _configured
    root = logging.getLogger()
    if _configured:
        return logging.getLogger("watcher")

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, LOG_FILE)

    root.setLevel(logging.DEBUG)

    # File: everything, timestamped, rotated
    fh = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=_MAX_BYTES, backupCount=_BACKUPS, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s"))
    root.addHandler(fh)

    # Console: bare message so existing output formatting is preserved
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    ch.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(ch)

    # Route warnings.warn() (deprecations etc.) into the log file too
    logging.captureWarnings(True)

    # Any uncaught exception gets its full traceback into the file BEFORE
    # the process dies — the reason this module exists.
    sys.excepthook = _log_uncaught

    _configured = True
    log = logging.getLogger("watcher")
    log.info("")
    log.debug("=" * 70)
    log.debug("logging initialised → %s", os.path.abspath(log_path))
    return log


def get_logger(name: str) -> logging.Logger:
    """Component logger. Name convention: 'watcher.layerN.component'."""
    return logging.getLogger(name)


def _log_uncaught(exc_type, exc, tb):
    """Log fatal tracebacks to the file, then die as Python normally would."""
    if issubclass(exc_type, KeyboardInterrupt):
        # Ctrl+C is a user action, not a crash
        sys.__excepthook__(exc_type, exc, tb)
        return
    logging.getLogger("watcher").critical(
        "UNCAUGHT EXCEPTION — process is terminating",
        exc_info=(exc_type, exc, tb))
    sys.__excepthook__(exc_type, exc, tb)
