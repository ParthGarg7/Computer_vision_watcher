"""
scripts/register_face.py
─────────────────────────────────────────────────────────────────────────────
Face Registration CLI — Loop 2 Implementation

A command-line tool for managing the FAISS identity registry used by Layer 4.

Identity Rules (enforced in code):
    - Every unique person gets ONE unique UUID, generated once at first
      registration. The UUID is the identity key — never the name.
    - Multiple face images/samples of the SAME person are registered under
      their existing UUID (via --person-id or interactive selection).
    - The 'name' field is auxiliary display metadata. Two people with the
      same name get two different UUIDs — they are never merged by name.
    - All downstream layers reference people by UUID; name is display-only.

Modes:
    --add-new        Register a new person (generates new UUID)
    --add-sample     Add another face sample to an existing person (by UUID)
    --set-name       Attach/update a display name for a person (by UUID)
    --list           List all registered people
    --search NAME    Search for people by display name
    --info UUID      Show info for a specific person UUID
    --delete UUID    Delete a person and all their embeddings (cascade)
    --clear          Delete ALL registered people (confirmation required)

Examples:
    python scripts/register_face.py --add-new --image path/to/face.jpg
    python scripts/register_face.py --add-new --image path/to/face.jpg --name "Alice"
    python scripts/register_face.py --add-sample --person-id <UUID> --image path/to/face2.jpg
    python scripts/register_face.py --set-name --person-id <UUID> --name "Bob"
    python scripts/register_face.py --list
    python scripts/register_face.py --search "Alice"
    python scripts/register_face.py --delete <UUID>
    python scripts/register_face.py --clear

Usage without --image (interactive webcam capture):
    python scripts/register_face.py --add-new
    (opens webcam, press SPACE to capture, Q to cancel)

Run from the project root (Computer_vision_watcher/):
    python scripts/register_face.py --list
"""

import argparse
import os
import sys
import cv2
import numpy as np

# Fix Windows console encoding (cp1252 → UTF-8 for box-drawing chars)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Project root must be on sys.path for src.* imports
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.core.logger import setup_logging
from src.layer4_identity.embedder import FaceEmbedder, FULL_IMAGE_DET_SIZE
from src.layer4_identity.identity_store import IdentityStore, DEFAULT_STORE_PATH

# ─── Constants ────────────────────────────────────────────────────────────────

WEBCAM_INDEX = 0


# ─── CLI Banner ───────────────────────────────────────────────────────────────

def print_banner():
    print()
    print("  ╔══════════════════════════════════════════════════╗")
    print("  ║     THE WATCHER — Face Registration CLI          ║")
    print("  ║     Layer 4 Identity Store Manager               ║")
    print("  ╚══════════════════════════════════════════════════╝")
    print()


# ─── Webcam Capture ───────────────────────────────────────────────────────────

def capture_from_webcam() -> np.ndarray | None:
    """
    Open webcam and let user press SPACE to capture a frame.
    Returns the captured BGR frame, or None if cancelled.
    """
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print("  [ERROR] Could not open webcam. Use --image to provide a file instead.")
        return None

    print("  Webcam open. Press SPACE to capture | Q to cancel.")
    frame_captured = None

    win_name = "The Watcher -- Face Registration"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        cv2.putText(
            display, "SPACE: capture  |  Q: cancel",
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (0, 220, 220), 2
        )
        cv2.imshow(win_name, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            frame_captured = frame.copy()
            print("  Frame captured.")
            break
        elif key == ord("q"):
            print("  Cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return frame_captured


# ─── Load image ───────────────────────────────────────────────────────────────

def load_image(image_path: str) -> np.ndarray | None:
    """Load an image file as a BGR numpy array."""
    if not os.path.exists(image_path):
        print(f"  [ERROR] File not found: {image_path}")
        return None
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"  [ERROR] Could not read image: {image_path}")
        return None
    return frame


# ─── Embedding helper ─────────────────────────────────────────────────────────

def get_embedding_from_frame(
    embedder: FaceEmbedder,
    frame: np.ndarray,
    source_desc: str = "image"
) -> tuple:
    """
    Extract a face embedding from a full frame.
    Returns (embedding, aligned_face) or (None, None) on failure.
    """
    embedding, aligned_face = embedder.get_embedding(frame)
    if embedding is None:
        print(f"  [ERROR] No face detected in {source_desc}.")
        print("  Try a clearer, front-facing image with good lighting.")
        return None, None
    print(f"  Face detected and embedded. Embedding shape: {embedding.shape}")
    return embedding, aligned_face


# ─── Commands ─────────────────────────────────────────────────────────────────

def cmd_add_new(args, embedder: FaceEmbedder, store: IdentityStore):
    """Register a new person — generates a new UUID."""
    print(f"\
  ── ADD NEW PERSON ──")

    # Get image
    if args.image:
        frame = load_image(args.image)
    else:
        frame = capture_from_webcam()

    if frame is None:
        return

    embedding, aligned_face = get_embedding_from_frame(embedder, frame, args.image or "webcam")
    if embedding is None:
        return

    name = args.name if args.name else None

    # Register — person_id=None → new UUID generated
    person_id = store.register(embedding=embedding, name=name)
    store.save()

    print(f"\
  ✅ New person registered!")
    print(f"     person_id : {person_id}")
    print(f"     name      : {name or '(none — use --set-name to add later)'}")
    print(f"     samples   : 1")
    print(f"  Store saved → {args.store}\
")

    # Show aligned face if available
    if aligned_face is not None:
        try:
            win = "Registered Face (112x112 aligned)"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.imshow(win, aligned_face)
            print("  Showing aligned face — press any key to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception:
            pass


def cmd_add_sample(args, embedder: FaceEmbedder, store: IdentityStore):
    """Add another face sample to an existing person (by UUID)."""
    print(f"\
  ── ADD SAMPLE FOR EXISTING PERSON ──")

    if not args.person_id:
        print("  [ERROR] --person-id is required for --add-sample.")
        print("  Use --list to find existing UUIDs.")
        return

    person = store.get_person(args.person_id)
    if person is None:
        print(f"  [ERROR] No person found with person_id '{args.person_id}'.")
        return

    print(f"  Adding sample for: {person['name'] or '(unnamed)'} [{person['person_id']}]")
    print(f"  Current samples  : {person['embedding_count']}")

    # Get image
    if args.image:
        frame = load_image(args.image)
    else:
        frame = capture_from_webcam()

    if frame is None:
        return

    embedding, _ = get_embedding_from_frame(embedder, frame, args.image or "webcam")
    if embedding is None:
        return

    # Register under existing UUID
    store.register(embedding=embedding, person_id=args.person_id)
    store.save()

    updated = store.get_person(args.person_id)
    print(f"\
  ✅ Sample added!")
    print(f"     person_id : {args.person_id}")
    print(f"     name      : {updated['name'] or '(none)'}")
    print(f"     samples   : {updated['embedding_count']} (was {person['embedding_count']})")
    print(f"  Store saved → {args.store}\
")


def cmd_set_name(args, store: IdentityStore):
    """Attach or update a display name for an existing person."""
    print(f"\
  ── SET NAME ──")

    if not args.person_id:
        print("  [ERROR] --person-id is required for --set-name.")
        return
    if not args.name:
        print("  [ERROR] --name is required for --set-name.")
        return

    try:
        store.update_name(args.person_id, args.name)
        store.save()
        print(f"  ✅ Name updated for {args.person_id}: '{args.name}'\
")
    except ValueError as e:
        print(f"  [ERROR] {e}")


def cmd_list(store: IdentityStore):
    """List all registered people."""
    people = store.list_people()
    print(f"\
  ── REGISTERED PEOPLE ({len(people)}) ──")
    if not people:
        print("  (empty — no faces registered yet)")
        print("  Use --add-new to register the first person.\
")
        return

    print(f"  {'UUID':38}  {'Name':20}  {'Samples':7}")
    print(f"  {'─'*38}  {'─'*20}  {'─'*7}")
    for p in people:
        name_disp = p['name'] or '(unnamed)'
        print(
            f"  {p['person_id']:38}  "
            f"{name_disp:20}  "
            f"{p['embedding_count']:7}"
        )
    print()


def cmd_search(args, store: IdentityStore):
    """Search registered people by display name."""
    query = args.search.lower()
    print(f"\
  ── SEARCH: '{args.search}' ──")
    matches = [
        p for p in store.list_people()
        if p['name'] and query in p['name'].lower()
    ]
    if not matches:
        print(f"  No people found with name matching '{args.search}'.\
")
        return
    print(f"  Found {len(matches)} match(es):")
    for p in matches:
        print(f"    {p['person_id']}  |  {p['name']}  |  {p['embedding_count']} samples")
    print()


def cmd_info(args, store: IdentityStore):
    """Show detailed info for a specific person by UUID."""
    if not args.person_id:
        print("  [ERROR] --person-id is required for --info.")
        return
    person = store.get_person(args.person_id)
    if person is None:
        print(f"  [ERROR] No person found with UUID '{args.person_id}'.\
")
        return
    print(f"\
  ── PERSON INFO ──")
    print(f"  person_id      : {person['person_id']}")
    print(f"  name           : {person['name'] or '(unnamed)'}")
    print(f"  face samples   : {person['embedding_count']}")
    print()


def cmd_delete(args, store: IdentityStore):
    """Delete a person and cascade delete all their embeddings."""
    if not args.person_id:
        print("  [ERROR] --person-id is required for --delete.")
        return

    person = store.get_person(args.person_id)
    if person is None:
        print(f"  [ERROR] No person found with UUID '{args.person_id}'.\
")
        return

    print(f"\
  ── DELETE PERSON ──")
    print(f"  About to delete:")
    print(f"    person_id : {person['person_id']}")
    print(f"    name      : {person['name'] or '(unnamed)'}")
    print(f"    samples   : {person['embedding_count']} (will be permanently removed)")

    confirm = input("\
  Type 'yes' to confirm deletion: ").strip().lower()
    if confirm != "yes":
        print("  Cancelled.\
")
        return

    store.delete_person(args.person_id)
    store.save()
    print(f"  ✅ Person {args.person_id} and all {person['embedding_count']} "
          f"embedding(s) deleted.\
")


def cmd_clear(store: IdentityStore):
    """Delete ALL registered people."""
    people = store.list_people()
    print(f"\
  ── CLEAR ALL PEOPLE ({len(people)}) ──")
    if not people:
        print("  Registry already empty.\
")
        return
    print(f"  This will PERMANENTLY delete {len(people)} people "
          f"and all their embeddings.")
    confirm = input("  Type 'DELETE ALL' to confirm: ").strip()
    if confirm != "DELETE ALL":
        print("  Cancelled.\
")
        return

    for p in people:
        store.delete_person(p['person_id'])
    store.save()
    print(f"  ✅ All {len(people)} people deleted. Registry is now empty.\
")


# ─── Argument Parser ──────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="register_face.py",
        description="The Watcher — Face Registration CLI (Layer 4 Identity Store)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\
"
            "  python scripts/register_face.py --add-new --image alice.jpg --name Alice\
"
            "  python scripts/register_face.py --add-sample --person-id <UUID> --image alice2.jpg\
"
            "  python scripts/register_face.py --set-name --person-id <UUID> --name Bob\
"
            "  python scripts/register_face.py --list\
"
            "  python scripts/register_face.py --search Alice\
"
            "  python scripts/register_face.py --delete <UUID>\
"
        )
    )

    # Mode flags (mutually exclusive)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--add-new", action="store_true",
                      help="Register a new person (generates new UUID)")
    mode.add_argument("--add-sample", action="store_true",
                      help="Add face sample to existing person (requires --person-id)")
    mode.add_argument("--set-name", action="store_true",
                      help="Set/update name for existing person (requires --person-id, --name)")
    mode.add_argument("--list", action="store_true",
                      help="List all registered people")
    mode.add_argument("--search", type=str, metavar="QUERY",
                      help="Search people by display name")
    mode.add_argument("--info", action="store_true",
                      help="Show info for a person (requires --person-id)")
    mode.add_argument("--delete", action="store_true",
                      help="Delete a person by UUID (requires --person-id, prompts confirmation)")
    mode.add_argument("--clear", action="store_true",
                      help="Delete ALL registered people (prompts confirmation)")

    # Options
    parser.add_argument("--image", type=str, default=None,
                        help="Path to face image (jpg/png). If omitted, webcam is used.")
    parser.add_argument("--name", type=str, default=None,
                        help="Display name (metadata only, never the identity key)")
    parser.add_argument("--person-id", type=str, default=None,
                        help="UUID of existing person (for --add-sample, --set-name, "
                             "--info, --delete)")
    parser.add_argument("--store", type=str, default=DEFAULT_STORE_PATH,
                        help=f"Base path for identity store files (default: {DEFAULT_STORE_PATH})")
    parser.add_argument("--no-model", action="store_true",
                        help="Skip loading InsightFace (only for --list, --search, --info, "
                             "--delete, --clear which don't need embeddings)")

    return parser


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    setup_logging()
    parser = build_parser()
    args = parser.parse_args()

    print_banner()

    # Load identity store
    store = IdentityStore(store_path=args.store)

    # Commands that don't need the embedder model
    if args.list:
        cmd_list(store)
        return
    if args.search:
        cmd_search(args, store)
        return
    if args.info:
        cmd_info(args, store)
        return
    if args.delete:
        cmd_delete(args, store)
        return
    if args.clear:
        cmd_clear(store)
        return
    if args.set_name:
        # Pure metadata update — never load the ~500 MB embedder for a rename
        cmd_set_name(args, store)
        return

    # Commands that need the embedder (--add-new, --add-sample)
    if args.no_model:
        print("  [ERROR] --no-model cannot be used with embedding commands.")
        sys.exit(1)

    print("  Loading InsightFace embedding model (first run downloads weights)...")
    # Registration passes WHOLE photos, not the small crops Layer 3 produces,
    # so it needs the full detection resolution — otherwise a small face in a
    # large image can be missed entirely.
    embedder = FaceEmbedder(det_size=FULL_IMAGE_DET_SIZE)

    if args.add_new:
        cmd_add_new(args, embedder, store)
    elif args.add_sample:
        cmd_add_sample(args, embedder, store)


if __name__ == "__main__":
    main()
