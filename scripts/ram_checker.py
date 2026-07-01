#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ram_checker.py — Standalone RAM Usage Inspector
════════════════════════════════════════════════════════════════════════════════

Shows overall system RAM status and the top 10 most memory-hungry processes
currently running on your machine.

STANDALONE — copy this file anywhere and run it independently.
Does not depend on any other project file.

Requirements:
    pip install psutil

Usage:
    python ram_checker.py              # Single snapshot
    python ram_checker.py --watch      # Refresh every 3 seconds (Ctrl+C to stop)
    python ram_checker.py --watch 5    # Refresh every 5 seconds
    python ram_checker.py --top 15     # Show top 15 processes instead of 10

════════════════════════════════════════════════════════════════════════════════
"""

import sys
import time
import argparse
import os

# ─── Dependency check ─────────────────────────────────────────────────────────
try:
    import psutil
except ImportError:
    print("\n  [ERROR] psutil is not installed.")
    print("  Install it with:  pip install psutil")
    print("  Then re-run this script.\n")
    sys.exit(1)


# ─── ANSI colour codes (Windows 10+ supports these natively) ──────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    GREEN  = "\033[92m"
    CYAN   = "\033[96m"
    WHITE  = "\033[97m"
    GREY   = "\033[90m"
    BG_RED = "\033[41m"


def _enable_ansi_windows():
    """Enable ANSI escape codes in Windows terminal."""
    # Force UTF-8 output on Windows (avoids cp1252 encoding errors)
    if sys.platform == "win32" and hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass


def _ram_colour(pct: float) -> str:
    """Return colour code based on usage percentage."""
    if pct >= 90:
        return C.RED
    elif pct >= 75:
        return C.YELLOW
    else:
        return C.GREEN


def _bar(pct: float, width: int = 30) -> str:
    """Render a coloured ASCII progress bar."""
    filled = int(width * pct / 100)
    empty  = width - filled
    colour = _ram_colour(pct)
    bar    = f"[{colour}{'#' * filled}{C.GREY}{'-' * empty}{C.RESET}]"
    return bar


def _gb(bytes_val: int) -> str:
    """Format bytes as GB string."""
    return f"{bytes_val / 1e9:.2f} GB"


def _mb(bytes_val: int) -> str:
    """Format bytes as MB string."""
    return f"{bytes_val / 1e6:.1f} MB"


# ─── Core snapshot logic ──────────────────────────────────────────────────────

def get_top_processes(n: int = 10) -> list:
    """
    Collect top N processes by RSS memory (resident set size).
    Gracefully skips processes we cannot access (e.g. system processes).

    Returns list of dicts: {pid, name, rss_bytes, pct_of_total}
    """
    total_ram = psutil.virtual_memory().total
    procs = []

    for proc in psutil.process_iter(["pid", "name", "memory_info"]):
        try:
            mi = proc.info["memory_info"]
            if mi is None:
                continue
            rss = mi.rss
            if rss == 0:
                continue
            procs.append({
                "pid":       proc.info["pid"],
                "name":      proc.info["name"] or "unknown",
                "rss_bytes": rss,
                "pct":       rss / total_ram * 100,
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    # Sort by RSS descending, return top N
    procs.sort(key=lambda x: x["rss_bytes"], reverse=True)
    return procs[:n]


def print_snapshot(top_n: int = 10):
    """Print a full RAM snapshot: system overview + top N processes."""
    now  = time.strftime("%Y-%m-%d  %H:%M:%S")
    vm   = psutil.virtual_memory()
    swap = psutil.swap_memory()

    total   = vm.total
    used    = vm.used
    avail   = vm.available
    pct     = vm.percent
    colour  = _ram_colour(pct)

    # ── Header ────────────────────────────────────────────────────────────────
    print()
    print(f"  {C.BOLD}{C.WHITE}+======================================================+{C.RESET}")
    print(f"  {C.BOLD}{C.WHITE}|         RAM CHECKER  --  System Memory Status        |{C.RESET}")
    print(f"  {C.BOLD}{C.WHITE}+======================================================+{C.RESET}")
    print(f"  {C.GREY}Snapshot taken: {now}{C.RESET}")
    print()

    # ── Overall RAM ───────────────────────────────────────────────────────────
    print(f"  {C.BOLD}System RAM{C.RESET}")
    print(f"  {_bar(pct)}  {colour}{C.BOLD}{pct:.1f}%{C.RESET}")
    print(f"  Used      : {colour}{C.BOLD}{_gb(used)}{C.RESET}  /  {_gb(total)} total")
    print(f"  Available : {C.GREEN}{_gb(avail)}{C.RESET}")
    print()

    # Pressure assessment
    if pct >= 90:
        print(f"  {C.BG_RED}{C.WHITE} CRITICAL {C.RESET} RAM pressure is very high. Close unused apps before running the pipeline.")
    elif pct >= 80:
        print(f"  {C.YELLOW}⚠ WARNING{C.RESET}  High RAM usage. Consider closing a browser or heavy apps first.")
    elif pct >= 65:
        print(f"  {C.YELLOW}  CAUTION{C.RESET}  Moderate usage. Should be fine for single-camera pipeline.")
    else:
        print(f"  {C.GREEN}     OK   {C.RESET}  Plenty of RAM available.")
    print()

    # ── Swap ──────────────────────────────────────────────────────────────────
    if swap.total > 0:
        swap_colour = _ram_colour(swap.percent)
        print(f"  {C.BOLD}Swap / Page File{C.RESET}")
        print(f"  {_bar(swap.percent, 20)}  {swap_colour}{swap.percent:.1f}%  ({_gb(swap.used)} / {_gb(swap.total)}){C.RESET}")
        if swap.percent > 30:
            print(f"  {C.YELLOW}  High swap usage = system is memory-starved. Close more apps.{C.RESET}")
        print()

    # ── Top N processes ───────────────────────────────────────────────────────
    procs = get_top_processes(top_n)

    print(f"  {C.BOLD}Top {top_n} Processes by RAM Usage{C.RESET}")
    print(f"  {'─'*52}")
    print(f"  {'#':>2}  {'PID':>6}  {'RAM':>9}  {'% Total':>7}  {'Process Name'}")
    print(f"  {'─'*52}")

    for rank, p in enumerate(procs, start=1):
        name    = p["name"][:30]
        rss_mb  = _mb(p["rss_bytes"])
        pct_p   = p["pct"]
        p_colour = C.RED if pct_p > 5 else (C.YELLOW if pct_p > 2 else C.RESET)
        print(
            f"  {rank:>2}  {p['pid']:>6}  "
            f"{p_colour}{rss_mb:>9}{C.RESET}  "
            f"{p_colour}{pct_p:>6.2f}%{C.RESET}  "
            f"{C.WHITE}{name}{C.RESET}"
        )

    print(f"  {'─'*52}")
    print()

    # Friendly tip
    headroom_gb = avail / 1e9
    print(f"  {C.CYAN}Pipeline headroom estimate:{C.RESET}")
    print(f"    Available now              : {C.BOLD}{_gb(avail)}{C.RESET}")
    print(f"    Layer 1-3 typical usage   : ~700 MB – 1.0 GB")
    print(f"    Layer 4 additional (InsightFace): ~350 MB")
    print(f"    Layer 5 additional (DeepFace)  : ~300 MB")
    if headroom_gb > 2.5:
        print(f"  {C.GREEN}  Current headroom is sufficient for the full MVP pipeline.{C.RESET}")
    elif headroom_gb > 1.5:
        print(f"  {C.YELLOW}  Tight. Close Chrome/Edge and Teams before running Layers 4-5.{C.RESET}")
    else:
        print(f"  {C.RED}  VERY TIGHT. Close heavy applications before running the pipeline.{C.RESET}")
    print()


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    _enable_ansi_windows()

    parser = argparse.ArgumentParser(
        description="RAM Checker — System memory usage + top RAM-consuming processes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python ram_checker.py              # Single snapshot\n"
            "  python ram_checker.py --watch      # Refresh every 3s\n"
            "  python ram_checker.py --watch 5    # Refresh every 5s\n"
            "  python ram_checker.py --top 15     # Show top 15 processes\n"
        )
    )
    parser.add_argument(
        "--watch",
        nargs="?",
        const=3,
        type=float,
        metavar="SECONDS",
        help="Continuously refresh every N seconds (default 3). Ctrl+C to stop."
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        metavar="N",
        help="Number of top processes to show (default 10)."
    )
    args = parser.parse_args()

    if args.watch is not None:
        interval = max(1.0, args.watch)
        print(f"\n  Watching RAM every {interval:.0f}s — press Ctrl+C to stop.")
        try:
            while True:
                # Clear screen
                os.system("cls" if sys.platform == "win32" else "clear")
                print_snapshot(top_n=args.top)
                print(f"  {C.GREY}Refreshing in {interval:.0f}s... (Ctrl+C to stop){C.RESET}\n")
                time.sleep(interval)
        except KeyboardInterrupt:
            print(f"\n  {C.GREY}Stopped.{C.RESET}\n")
    else:
        print_snapshot(top_n=args.top)


if __name__ == "__main__":
    main()
