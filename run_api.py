#!/usr/bin/env python3
"""
run_api.py — Layer 8 API + Layer 9 Dashboard server

Serves the read API over the pipeline's stored data, plus the dashboard.
Runs alongside the pipeline: the database is opened read-only per request
and the pipeline writes in WAL mode, so both can run at the same time.

Usage:
    python run_api.py                     # http://127.0.0.1:8000
    python run_api.py --port 9000
    python run_api.py --host 0.0.0.0     # reachable from other devices

Then open:
    http://127.0.0.1:8000/        the dashboard
    http://127.0.0.1:8000/docs    interactive API documentation

Run from the project root (Computer_vision_watcher/).
"""

import argparse
import sys
import os

# Fix Windows console encoding (cp1252 → UTF-8)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.core.logger import setup_logging, get_logger


def main():
    setup_logging()
    log = get_logger("watcher.layer8")

    parser = argparse.ArgumentParser(
        description="The Watcher — Layer 8 API + Layer 9 dashboard server")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address (0.0.0.0 for LAN access)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--db", default=None,
                        help="Override database path (default: data/watcher.db)")
    args = parser.parse_args()

    import uvicorn
    from src.layer8_api.api import app

    if args.db:
        app.state.db_path = args.db

    log.info(f"  [Layer8] API starting on http://{args.host}:{args.port}")
    log.info(f"  [Layer8] Dashboard: http://{args.host}:{args.port}/   "
             f"API docs: http://{args.host}:{args.port}/docs")
    log.info(f"  [Layer8] Database : {app.state.db_path} (read-only)")

    # uvicorn's own logs go through logging too, so they reach watcher.log
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
