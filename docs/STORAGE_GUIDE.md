# The Watcher — Storage Guide & Project Status

Everything about what Layer 7 records, how long it keeps it, how to look at it,
and what is left to build.

---

## 1. Where data lives

The project writes to **two separate stores**, because vectors and records have
completely different access patterns.

| Store | Path | Holds | Format |
|---|---|---|---|
| **Session database** | `data/watcher.db` | Sessions, expression readings, presence log | SQLite |
| **Identity registry** | `faces/db/identity_store.faiss` + `.meta.json` | Face embeddings + UUID↔name map | FAISS index + JSON |

Both are **gitignored** — they contain biometric and behavioural data and must
never be committed.

> **Why SQLite?** The architecture doc specifies PostgreSQL + TimescaleDB, and
> explicitly permits SQLite as the development substitute. The schema below
> deliberately mirrors the production layout, so migrating is largely a
> connection-string change. SQLite is correct while there is **one writer**
> (one pipeline process). It becomes wrong the moment multiple cameras write
> concurrently — SQLite locks the whole database on write.
>
> Two implementation details: **WAL mode** lets you read the database while the
> pipeline is running (that's why the VS Code viewer works live), and
> expression events are **batched 64 at a time** so the frame loop never stalls
> on disk I/O.

---

## 2. What a session stores

### 2.1 `sessions` — one row per person-visit

A **session** = one person's continuous presence on one camera. It opens when a
track is confirmed and closes when that person is unseen for 5 seconds
(`TRACK_TIMEOUT_SEC`), or when the pipeline shuts down.

| Column | Type | Meaning |
|---|---|---|
| `session_id` | TEXT (PK) | `{camera}:{track}:{start_epoch}` — e.g. `webcam_0:1:1784108437` |
| `camera_id` | TEXT | Which source — `webcam_0`, `rtsp_...`, `video_clip` |
| `track_id` | INTEGER | The on-screen ID during this visit |
| `identity_label` | TEXT | `"Parth"` if recognised, `NULL` if never identified |
| `session_start` | REAL | Epoch seconds when first seen |
| `session_end` | REAL | Epoch seconds when last seen |
| `presence_duration_seconds` | REAL | How long they were present |
| `frames_observed` | INTEGER | Frames they appeared in (**every** frame) |
| `expression_trend` | TEXT (JSON) | Mean probability per class over the final 30s window |
| `dominant_expression_distribution` | TEXT (JSON) | Count per winning class (**measurements**, not frames) |

**Example row:**

```
session_id                 webcam_0:1:1784108437
identity_label             Parth
presence_duration_seconds  78.907
frames_observed            1559
expression_trend           {"neutral": 0.68, "contempt": 0.11, "anger": 0.05, ...}
dominant_expression_distribution   {"neutral": 1072, "contempt": 95}
```

**Important distinction:** `frames_observed` counts every frame the person
appeared in. `dominant_expression_distribution` counts **measurements** —
Layer 5 only runs inference every 5th frame per person and carries the label
forward in between. So these two numbers are deliberately different, roughly by
the throttle factor. Ratios in the distribution are accurate; absolute counts
are measurement counts.

### 2.2 `expression_events` — one row per measurement

The raw time-series. One row each time the emotion model **actually runs** on a
face (not once per frame).

| Column | Type | Meaning |
|---|---|---|
| `id` | INTEGER (PK) | Auto-increment |
| `timestamp` | REAL | Epoch seconds (indexed) |
| `camera_id` | TEXT | Source |
| `frame_seq` | INTEGER | Frame number within the run |
| `track_id` | INTEGER | Who (indexed) |
| `identity_label` | TEXT | Name if known at that moment |
| `dominant_expression` | TEXT | Winning class |
| `confidence` | REAL | Probability of the winner |
| `expression_scores` | TEXT (JSON) | All 8 class probabilities, summing to 1.0 |

This is the table that would become a **TimescaleDB hypertable** in production.

### 2.3 `presence_events` — the arrival/departure log

| Column | Type | Meaning |
|---|---|---|
| `id` | INTEGER (PK) | Auto-increment |
| `timestamp` | REAL | Epoch seconds (indexed) |
| `camera_id` | TEXT | Source |
| `track_id` | INTEGER | Who |
| `identity_label` | TEXT | Usually `NULL` on `appeared` — recognition needs a frame or two |
| `event_type` | TEXT | `appeared` or `departed` |

### 2.4 What is **not** stored

- **No images.** No frames, no face crops, no video. Only numbers and labels.
- **No embeddings** in this database — those live in the FAISS index.
- **Nothing for unconfirmed tracks.** A face must survive 3 frames before it exists here.

---

## 3. How much data, and for how long

### Current retention policy: **none — data is kept forever.**

Nothing deletes automatically. The database grows until you prune it.

### Growth rate

Measured from real usage: **~350 bytes per expression event.**

With the throttle at every 5th frame, one person on camera produces roughly
**4–6 measurements per second**:

| Scenario | Rows/hour | Disk/hour |
|---|---|---|
| 1 person, continuous | ~20,000 | **~7 MB** |
| 3 people, continuous | ~60,000 | **~21 MB** |
| 1 person, 8-hour day | ~160,000 | **~56 MB** |
| 3 people, 8-hour day | ~480,000 | **~170 MB** |

Sessions and presence events are negligible by comparison (hundreds of bytes
each, a handful per visit).

> Before the freshness fix this was **~5× higher** — carry-forward labels were
> being written as if each were a new measurement.

### Enforcing a retention window

The Layer 7 architecture doc requires a defined retention window before
production. The tool exists; the **policy is yours to choose**:

```bash
# Keep 30 days, delete everything older
python scripts/db_manager.py --delete-before 2026-06-21
python scripts/db_manager.py --vacuum
```

`--vacuum` matters: SQLite marks deleted space reusable but does **not** shrink
the file until you vacuum.

To automate it, schedule that pair weekly (Windows Task Scheduler / cron).
**This is currently a manual step — there is no automatic purge.**

---

## 4. Viewing the data

### 4.1 Visual (recommended)

```bash
code data/watcher.db
```

Opens in VS Code with the **SQLite Viewer** extension (already installed) —
all three tables in a sortable, scrollable grid.

For writing SQL inside VS Code, add:
```bash
code --install-extension cweijan.vscode-database-client2
```

### 4.2 The CLI — `scripts/db_manager.py`

Complete command reference.

#### Viewing

| Command | What it does |
|---|---|
| `--dump` | **Every table, every row.** The whole database |
| `--dump 50` | Same, capped at 50 rows per table |
| `--stats` | Overview: table sizes, date range, who's been recorded and for how long |
| `--sessions` | 15 most recent sessions (`--sessions 50` for more) |
| `--events` | 15 most recent expression measurements |
| `--presence` | 15 most recent arrivals/departures |
| `--person NAME` | Full report for one person: total time, session count, expression bar chart |
| `--query "SQL"` | Run any SQL statement |

```bash
python scripts/db_manager.py --stats
python scripts/db_manager.py --person Parth
python scripts/db_manager.py --sessions 30
python scripts/db_manager.py --query "SELECT identity_label, COUNT(*) FROM sessions GROUP BY 1"
```

#### Adding rows

| Command | Use |
|---|---|
| `--add-session` | Insert a session manually (testing, demo data) |
| `--add-presence` | Insert an arrival/departure event |

```bash
python scripts/db_manager.py --add-session --identity "Test" --duration 45 --frames 900
python scripts/db_manager.py --add-presence --track-id 1 --event appeared
```

Supporting flags: `--identity`, `--track-id`, `--camera`, `--duration`,
`--frames`, `--session-id`, `--event`.

#### Deleting — all require typed confirmation

| Command | Deletes | Confirm with |
|---|---|---|
| `--delete-session ID` | One session row | `yes` |
| `--delete-before DATE` | **All data** older than a date (retention purge) | `DELETE` |
| `--clear-table NAME` | One entire table | `DELETE` |
| `--clear-all` | Every row in every table | `DELETE ALL` |

```bash
python scripts/db_manager.py --delete-session webcam_0:1:1784108437
python scripts/db_manager.py --delete-before 2026-07-01
python scripts/db_manager.py --clear-table expression_events
python scripts/db_manager.py --clear-all
```

> `--delete-session` removes only the session summary — its expression and
> presence events remain. Use `--delete-before` for a complete time-based purge.
> Registered faces in `faces/db/` are **never** touched by any of these.

#### Maintenance

| Command | Use |
|---|---|
| `--vacuum` | Reclaim disk space after deletions |
| `--export-csv DIR` | Write all three tables to CSV files |

```bash
python scripts/db_manager.py --vacuum
python scripts/db_manager.py --export-csv out/
```

#### Global options

| Flag | Default |
|---|---|
| `--db PATH` | `data/watcher.db` |

### 4.3 Programmatic access

```python
from src.layer7_storage.store import StorageLayer
store = StorageLayer()
store.recent_sessions(limit=20)
store.expression_trend(track_id=1, bucket_seconds=30)   # time-bucketed
store.close()
```

These two read methods exist so **Layer 8's API** can call them directly.

---

## 5. What remains to be built

### Status: 7 of 9 layers complete

| Layer | Status |
|---|---|
| 1 Ingestion | Done |
| 2 Preprocessing | Done |
| 3 Detection | Done |
| 4 Identity | Done |
| 5 Expression | Done |
| 6 Analytics | Done |
| 7 Storage | Done (SQLite dev form) |
| **8 REST API** | **Not started** |
| **9 Frontend Dashboard** | **Not started** |

### 5.1 Layer 8 — REST API (FastAPI)

The natural next build. All the data exists but is only reachable by opening
the database by hand. Planned endpoints:

- `GET /sessions` — recent sessions
- `GET /people/{name}` — one person's history
- `GET /trends?track_id=&bucket=` — time-bucketed expression data
- `WebSocket /live` — push `live_expression_update`, `presence_alert`,
  `threshold_alert` (Layer 6 **already generates** these; nothing consumes them yet)

### 5.2 Layer 9 — Dashboard

Web UI over Layer 8: live view, session history, per-person charts.

### 5.3 Open technical items

**Performance — unmeasured.** Observed 17–28 FPS, but never profiled per layer,
and the last measurement predates two significant changes. Candidates:

- **Layer 4 runs on every face, every frame** — unlike Layer 5 it is not
  throttled. Likely the most expensive layer.
- **`models/yolov8n-face-landmark.pt` is already downloaded** but unused. If
  Layer 3 provided the 5 landmarks, Layer 4 could skip SCRFD entirely and go
  straight to alignment + ArcFace. Potentially a large saving, no accuracy risk.

**Calibration.** Two thresholds are defaults, not tuned to your deployment:
recognition at **0.45** (must rise as the registry grows — more enrolled people
means more chances of a false match) and the negative-expression alert at
**0.60**.

**Robustness.** Only **one** face sample is registered per person. Adding 4–6
per person (different angles, lighting, with/without glasses) would improve
recognition at extreme poses — search takes the best match across all samples.

**Security & compliance.**
- Biometric data is **not encrypted at rest** — needs an OS-level decision
  (BitLocker / encrypted volume).
- Retention policy is **not automated** — the tool exists, the schedule doesn't.
- Alert rules on expression data should be documented and reviewed before any
  real deployment; the architecture doc is explicit about the compliance risk.

**Scale migration.** SQLite → PostgreSQL + TimescaleDB when multiple cameras
write concurrently. FAISS `IndexFlatIP` → `IndexIVFFlat` beyond a few thousand
identities. `delete_person()` currently rebuilds the whole index.

**Known minor issues.**
- The red `no-embed` label can clip off the right edge of the frame.
- Drawing code is duplicated between `main.py` and the three layer validators —
  should be extracted to `src/core/drawing.py`.

**Parked ideas.**
- **Custom emotions** — four routes were mapped out: rule-based combinations of
  the existing 8; the valence/arousal model (`enet_b0_8_va_mtl`, which the code
  already supports); landmark-based behavioural signals (drowsiness, attention);
  or training a new class properly.

---

## 6. Quick reference

```bash
# Look at everything
code data/watcher.db                                   # visual
python scripts/db_manager.py --dump                    # terminal
python scripts/db_manager.py --stats                   # summary

# Reports
python scripts/db_manager.py --person Parth
python scripts/db_manager.py --sessions 30

# Housekeeping
python scripts/db_manager.py --delete-before 2026-06-21
python scripts/db_manager.py --vacuum
python scripts/db_manager.py --export-csv out/

# Faces (separate store)
python scripts/register_face.py --list
python scripts/register_face.py --add-new --name "Alice"

# Run & test
python main.py --source 0
python -m unittest discover tests
```
