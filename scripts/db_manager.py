"""
scripts/db_manager.py
─────────────────────────────────────────────────────────────────────────────
Layer 7 Storage — Database Management CLI

Inspect, query, add to, and prune the pipeline's SQLite database
(data/watcher.db). This is the human-facing counterpart to StorageLayer,
which is the pipeline-facing writer.

The database holds three tables:
    sessions           — one row per person-visit (the summary)
    expression_events  — one row per fresh expression measurement (the detail)
    presence_events    — one row per arrival/departure (the log)

VIEW
    python scripts/db_manager.py --stats
    python scripts/db_manager.py --sessions 20
    python scripts/db_manager.py --events 20
    python scripts/db_manager.py --presence 20
    python scripts/db_manager.py --person "Parth"
    python scripts/db_manager.py --query "SELECT identity_label, COUNT(*) FROM sessions GROUP BY 1"

ADD
    python scripts/db_manager.py --add-session --identity "Parth" --duration 45
    python scripts/db_manager.py --add-presence --track-id 1 --event appeared

DELETE  (all destructive commands prompt for confirmation)
    python scripts/db_manager.py --delete-session <session_id>
    python scripts/db_manager.py --delete-before 2026-07-01
    python scripts/db_manager.py --clear-table expression_events
    python scripts/db_manager.py --clear-all
    python scripts/db_manager.py --vacuum          # reclaim disk after deletes

EXPORT
    python scripts/db_manager.py --export-csv out/

Run from the project root (Computer_vision_watcher/).

PRIVACY NOTE: rows here are identity-linked behavioural records. The Layer 7
architecture doc requires a defined retention window before production —
--delete-before is the tool for enforcing it.
"""

import argparse
import csv
import datetime
import json
import os
import sqlite3
import sys

# Fix Windows console encoding (cp1252 → UTF-8 for box-drawing chars)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Project root must be on sys.path for src.* imports
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.layer7_storage.store import DEFAULT_DB_PATH

TABLES = ("sessions", "expression_events", "presence_events")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def print_banner():
    print()
    print("  ╔══════════════════════════════════════════════════╗")
    print("  ║     THE WATCHER — Database Manager               ║")
    print("  ║     Layer 7 Storage Inspector                    ║")
    print("  ╚══════════════════════════════════════════════════╝")
    print()


def connect(db_path: str, must_exist: bool = True) -> sqlite3.Connection:
    if must_exist and not os.path.exists(db_path):
        print(f"  [ERROR] Database not found: {db_path}")
        print("  Run the pipeline first (python main.py) to create it.")
        sys.exit(1)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def fmt_ts(epoch) -> str:
    """Epoch seconds → readable local time."""
    if epoch is None:
        return "-"
    try:
        return datetime.datetime.fromtimestamp(float(epoch)).strftime(
            "%Y-%m-%d %H:%M:%S")
    except (ValueError, OSError, OverflowError):
        return str(epoch)


def parse_date(s: str) -> float:
    """'YYYY-MM-DD' (or with time) → epoch seconds."""
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.datetime.strptime(s, fmt).timestamp()
        except ValueError:
            continue
    print(f"  [ERROR] Bad date '{s}'. Use YYYY-MM-DD or 'YYYY-MM-DD HH:MM:SS'.")
    sys.exit(1)


def confirm(prompt: str, required: str = "yes") -> bool:
    try:
        got = input(f"  {prompt} (type '{required}'): ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n  Cancelled.")
        return False
    if got != required:
        print("  Cancelled.\n")
        return False
    return True


def top_n(dist_json: str, n: int = 3) -> str:
    d = json.loads(dist_json or "{}")
    if not d:
        return "-"
    top = sorted(d.items(), key=lambda kv: -kv[1])[:n]
    return ", ".join(f"{k}:{v}" for k, v in top)


# ─── View commands ────────────────────────────────────────────────────────────

def cmd_stats(conn, db_path):
    print("  ── DATABASE OVERVIEW ──\n")
    size_mb = os.path.getsize(db_path) / 1e6
    print(f"  File : {db_path}")
    print(f"  Size : {size_mb:.2f} MB\n")

    print(f"  {'Table':22} {'Rows':>10}")
    print(f"  {'─'*22} {'─'*10}")
    for t in TABLES:
        try:
            n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        except sqlite3.OperationalError:
            n = "missing"
        print(f"  {t:22} {n:>10}")

    row = conn.execute(
        "SELECT MIN(session_start), MAX(session_end) FROM sessions").fetchone()
    if row and row[0]:
        print(f"\n  Data range : {fmt_ts(row[0])}  →  {fmt_ts(row[1])}")

    people = conn.execute(
        "SELECT COALESCE(identity_label,'(unidentified)') AS who, COUNT(*) c, "
        "SUM(presence_duration_seconds) secs FROM sessions "
        "GROUP BY who ORDER BY secs DESC").fetchall()
    if people:
        print(f"\n  {'Person':24} {'Sessions':>9} {'Total time':>12}")
        print(f"  {'─'*24} {'─'*9} {'─'*12}")
        for p in people:
            mins = (p["secs"] or 0) / 60.0
            print(f"  {p['who'][:24]:24} {p['c']:>9} {mins:>10.1f} m")
    print()


def cmd_sessions(conn, limit):
    rows = conn.execute(
        "SELECT * FROM sessions ORDER BY session_end DESC LIMIT ?",
        (limit,)).fetchall()
    print(f"  ── SESSIONS (latest {len(rows)}) ──\n")
    if not rows:
        print("  (empty)\n")
        return
    print(f"  {'Started':20} {'Person':12} {'Dur':>8} {'Frames':>7}  Top expressions")
    print(f"  {'─'*20} {'─'*12} {'─'*8} {'─'*7}  {'─'*30}")
    for r in rows:
        print(f"  {fmt_ts(r['session_start']):20} "
              f"{str(r['identity_label'] or '-')[:12]:12} "
              f"{r['presence_duration_seconds']:>7.1f}s "
              f"{r['frames_observed']:>7}  "
              f"{top_n(r['dominant_expression_distribution'])}")
    print(f"\n  (session_id of newest: {rows[0]['session_id']})\n")


def cmd_events(conn, limit):
    rows = conn.execute(
        "SELECT * FROM expression_events ORDER BY id DESC LIMIT ?",
        (limit,)).fetchall()
    print(f"  ── EXPRESSION EVENTS (latest {len(rows)}) ──\n")
    if not rows:
        print("  (empty)\n")
        return
    print(f"  {'Time':20} {'Trk':>4} {'Person':10} {'Expression':12} {'Conf':>6}")
    print(f"  {'─'*20} {'─'*4} {'─'*10} {'─'*12} {'─'*6}")
    for r in rows:
        print(f"  {fmt_ts(r['timestamp']):20} {r['track_id']:>4} "
              f"{str(r['identity_label'] or '-')[:10]:10} "
              f"{r['dominant_expression']:12} {r['confidence']:>6.2f}")
    print()


def cmd_presence(conn, limit):
    rows = conn.execute(
        "SELECT * FROM presence_events ORDER BY id DESC LIMIT ?",
        (limit,)).fetchall()
    print(f"  ── PRESENCE EVENTS (latest {len(rows)}) ──\n")
    if not rows:
        print("  (empty)\n")
        return
    print(f"  {'Time':20} {'Trk':>4} {'Person':12} {'Event':10}")
    print(f"  {'─'*20} {'─'*4} {'─'*12} {'─'*10}")
    for r in rows:
        print(f"  {fmt_ts(r['timestamp']):20} {r['track_id']:>4} "
              f"{str(r['identity_label'] or '-')[:12]:12} {r['event_type']:10}")
    print()


def cmd_person(conn, name):
    rows = conn.execute(
        "SELECT * FROM sessions WHERE identity_label = ? "
        "ORDER BY session_end DESC", (name,)).fetchall()
    print(f"  ── PERSON: {name} ──\n")
    if not rows:
        print(f"  No sessions found for '{name}'.")
        print("  (Names are case-sensitive; try --stats to see who is recorded.)\n")
        return
    total = sum(r["presence_duration_seconds"] for r in rows)
    frames = sum(r["frames_observed"] for r in rows)
    print(f"  Sessions   : {len(rows)}")
    print(f"  Total time : {total/60:.1f} minutes ({total:.0f}s)")
    print(f"  Frames seen: {frames}")

    combined = {}
    for r in rows:
        for k, v in json.loads(r["dominant_expression_distribution"] or "{}").items():
            combined[k] = combined.get(k, 0) + v
    s = sum(combined.values()) or 1
    if combined:
        print(f"\n  Expression breakdown (measurements):")
        for k, v in sorted(combined.items(), key=lambda kv: -kv[1]):
            bar = "█" * int(30 * v / s)
            print(f"    {k:12} {v:6}  {v/s:5.1%}  {bar}")
    print()


def cmd_query(conn, sql):
    print(f"  ── QUERY ──\n  {sql}\n")
    try:
        cur = conn.execute(sql)
    except sqlite3.Error as e:
        print(f"  [SQL ERROR] {e}\n")
        return
    rows = cur.fetchall()
    if cur.description is None:
        conn.commit()
        print(f"  OK. Rows affected: {cur.rowcount}\n")
        return
    if not rows:
        print("  (no rows)\n")
        return
    cols = [d[0] for d in cur.description]
    widths = [max(len(c), *(len(str(r[i])) for r in rows)) for i, c in enumerate(cols)]
    widths = [min(w, 30) for w in widths]
    print("  " + " | ".join(c[:w].ljust(w) for c, w in zip(cols, widths)))
    print("  " + "-+-".join("-" * w for w in widths))
    for r in rows:
        print("  " + " | ".join(str(r[i])[:w].ljust(w) for i, w in enumerate(widths)))
    print(f"\n  {len(rows)} row(s)\n")


# ─── Add commands ─────────────────────────────────────────────────────────────

def cmd_add_session(conn, args):
    now = datetime.datetime.now().timestamp()
    dur = float(args.duration)
    start = now - dur
    sid = args.session_id or f"{args.camera}:{args.track_id}:{int(start)}"
    conn.execute(
        "INSERT OR REPLACE INTO sessions VALUES (?,?,?,?,?,?,?,?,?,?)",
        (sid, args.camera, args.track_id, args.identity, start, now, dur,
         args.frames, json.dumps({}), json.dumps({})))
    conn.commit()
    print(f"  ✅ Session added\n     session_id : {sid}\n"
          f"     identity   : {args.identity or '(none)'}\n"
          f"     duration   : {dur}s\n")


def cmd_add_presence(conn, args):
    if args.event not in ("appeared", "departed"):
        print("  [ERROR] --event must be 'appeared' or 'departed'.")
        return
    conn.execute(
        "INSERT INTO presence_events "
        "(timestamp, camera_id, track_id, identity_label, event_type) "
        "VALUES (?,?,?,?,?)",
        (datetime.datetime.now().timestamp(), args.camera, args.track_id,
         args.identity, args.event))
    conn.commit()
    print(f"  ✅ Presence event added: track {args.track_id} {args.event}\n")


# ─── Delete commands ──────────────────────────────────────────────────────────

def cmd_delete_session(conn, session_id):
    row = conn.execute("SELECT * FROM sessions WHERE session_id = ?",
                       (session_id,)).fetchone()
    if row is None:
        print(f"  [ERROR] No session with id '{session_id}'.\n")
        return
    print(f"  About to delete:")
    print(f"    session_id : {row['session_id']}")
    print(f"    person     : {row['identity_label'] or '(unidentified)'}")
    print(f"    duration   : {row['presence_duration_seconds']}s")
    print(f"  NOTE: the linked expression/presence events are NOT removed by")
    print(f"        this command (use --delete-before for a full time purge).")
    if not confirm("Confirm deletion"):
        return
    conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    conn.commit()
    print(f"  ✅ Session deleted.\n")


def cmd_delete_before(conn, date_str):
    cutoff = parse_date(date_str)
    counts = {
        "sessions": conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE session_end < ?", (cutoff,)
        ).fetchone()[0],
        "expression_events": conn.execute(
            "SELECT COUNT(*) FROM expression_events WHERE timestamp < ?", (cutoff,)
        ).fetchone()[0],
        "presence_events": conn.execute(
            "SELECT COUNT(*) FROM presence_events WHERE timestamp < ?", (cutoff,)
        ).fetchone()[0],
    }
    total = sum(counts.values())
    print(f"  ── RETENTION PURGE: everything before {fmt_ts(cutoff)} ──\n")
    for t, c in counts.items():
        print(f"    {t:22} {c:>8} rows")
    print(f"    {'TOTAL':22} {total:>8} rows")
    if total == 0:
        print("\n  Nothing to delete.\n")
        return
    if not confirm("\n  This is PERMANENT. Confirm", "DELETE"):
        return
    conn.execute("DELETE FROM sessions WHERE session_end < ?", (cutoff,))
    conn.execute("DELETE FROM expression_events WHERE timestamp < ?", (cutoff,))
    conn.execute("DELETE FROM presence_events WHERE timestamp < ?", (cutoff,))
    conn.commit()
    print(f"  ✅ {total} rows deleted. Run --vacuum to reclaim disk space.\n")


def cmd_clear_table(conn, table):
    if table not in TABLES:
        print(f"  [ERROR] Unknown table '{table}'. Choose from: {', '.join(TABLES)}\n")
        return
    n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"  ── CLEAR TABLE: {table} ({n} rows) ──\n")
    if n == 0:
        print("  Already empty.\n")
        return
    if not confirm("This is PERMANENT. Confirm", "DELETE"):
        return
    conn.execute(f"DELETE FROM {table}")
    conn.commit()
    print(f"  ✅ {n} rows deleted from {table}. Run --vacuum to reclaim space.\n")


def cmd_clear_all(conn):
    counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in TABLES}
    total = sum(counts.values())
    print(f"  ── CLEAR ENTIRE DATABASE ──\n")
    for t, c in counts.items():
        print(f"    {t:22} {c:>8} rows")
    if total == 0:
        print("\n  Already empty.\n")
        return
    print(f"\n  This deletes ALL {total} rows of recorded activity.")
    print(f"  (Registered faces in faces/db/ are NOT affected.)")
    if not confirm("Confirm", "DELETE ALL"):
        return
    for t in TABLES:
        conn.execute(f"DELETE FROM {t}")
    conn.commit()
    print(f"  ✅ Database cleared. Run --vacuum to reclaim disk space.\n")


def cmd_vacuum(conn, db_path):
    before = os.path.getsize(db_path) / 1e6
    print(f"  Size before : {before:.2f} MB")
    conn.execute("VACUUM")
    conn.commit()
    after = os.path.getsize(db_path) / 1e6
    print(f"  Size after  : {after:.2f} MB   (reclaimed {before-after:.2f} MB)\n")


# ─── Export ───────────────────────────────────────────────────────────────────

def cmd_export_csv(conn, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print(f"  ── EXPORT CSV → {out_dir} ──\n")
    for t in TABLES:
        rows = conn.execute(f"SELECT * FROM {t}").fetchall()
        path = os.path.join(out_dir, f"{t}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if rows:
                w.writerow(rows[0].keys())
                w.writerows([tuple(r) for r in rows])
        print(f"    {t:22} {len(rows):>7} rows → {path}")
    print()


# ─── Parser ───────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="db_manager.py",
        description="The Watcher — Layer 7 database inspector and manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/db_manager.py --stats\n"
            "  python scripts/db_manager.py --sessions 20\n"
            "  python scripts/db_manager.py --person Parth\n"
            "  python scripts/db_manager.py --query \"SELECT * FROM sessions LIMIT 3\"\n"
            "  python scripts/db_manager.py --delete-before 2026-07-01\n"
            "  python scripts/db_manager.py --vacuum\n"
        ))

    m = p.add_mutually_exclusive_group(required=True)
    # view
    m.add_argument("--stats", action="store_true", help="Overview: tables, sizes, people")
    m.add_argument("--sessions", nargs="?", type=int, const=15, metavar="N",
                   help="List the N most recent sessions (default 15)")
    m.add_argument("--events", nargs="?", type=int, const=15, metavar="N",
                   help="List the N most recent expression events")
    m.add_argument("--presence", nargs="?", type=int, const=15, metavar="N",
                   help="List the N most recent presence events")
    m.add_argument("--person", type=str, metavar="NAME",
                   help="Full report for one person by name")
    m.add_argument("--query", type=str, metavar="SQL", help="Run raw SQL")
    # add
    m.add_argument("--add-session", action="store_true", help="Insert a session row")
    m.add_argument("--add-presence", action="store_true", help="Insert a presence row")
    # delete
    m.add_argument("--delete-session", type=str, metavar="ID",
                   help="Delete one session by session_id")
    m.add_argument("--delete-before", type=str, metavar="YYYY-MM-DD",
                   help="Retention purge: delete all data older than this date")
    m.add_argument("--clear-table", type=str, metavar="TABLE",
                   help=f"Empty one table ({', '.join(TABLES)})")
    m.add_argument("--clear-all", action="store_true", help="Empty every table")
    m.add_argument("--vacuum", action="store_true",
                   help="Reclaim disk space after deletions")
    # export
    m.add_argument("--export-csv", type=str, metavar="DIR",
                   help="Export every table to CSV files in DIR")

    # options
    p.add_argument("--db", type=str, default=DEFAULT_DB_PATH,
                   help=f"Database path (default: {DEFAULT_DB_PATH})")
    p.add_argument("--identity", type=str, default=None, help="Person name (for --add-*)")
    p.add_argument("--track-id", type=int, default=1, help="Track id (for --add-*)")
    p.add_argument("--camera", type=str, default="manual", help="Camera id (for --add-*)")
    p.add_argument("--duration", type=float, default=10.0, help="Seconds (for --add-session)")
    p.add_argument("--frames", type=int, default=0, help="Frame count (for --add-session)")
    p.add_argument("--session-id", type=str, default=None, help="Explicit session id")
    p.add_argument("--event", type=str, default="appeared",
                   help="appeared|departed (for --add-presence)")
    return p


def main():
    args = build_parser().parse_args()
    print_banner()
    conn = connect(args.db)
    try:
        if args.stats:                 cmd_stats(conn, args.db)
        elif args.sessions is not None: cmd_sessions(conn, args.sessions)
        elif args.events is not None:   cmd_events(conn, args.events)
        elif args.presence is not None: cmd_presence(conn, args.presence)
        elif args.person:              cmd_person(conn, args.person)
        elif args.query:               cmd_query(conn, args.query)
        elif args.add_session:         cmd_add_session(conn, args)
        elif args.add_presence:        cmd_add_presence(conn, args)
        elif args.delete_session:      cmd_delete_session(conn, args.delete_session)
        elif args.delete_before:       cmd_delete_before(conn, args.delete_before)
        elif args.clear_table:         cmd_clear_table(conn, args.clear_table)
        elif args.clear_all:           cmd_clear_all(conn)
        elif args.vacuum:              cmd_vacuum(conn, args.db)
        elif args.export_csv:          cmd_export_csv(conn, args.export_csv)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
