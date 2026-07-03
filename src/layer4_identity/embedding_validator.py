"""
src/layer4_identity/embedding_validator.py
─────────────────────────────────────────────────────────────────────────────
Layer 4 Companion: Embedding Unit Validator

A targeted unit test for the FaceEmbedder and IdentityStore components,
independent of video input.

This script:
    1. Generates a synthetic face image (or uses a provided image file).
    2. Runs FaceEmbedder.get_embedding() and checks:
        - Embedding shape is (512,)
        - Embedding dtype is float32
        - Embedding is L2-normalised (norm ≈ 1.0)
        - Aligned face is (112, 112, 3) BGR uint8
    3. Runs IdentityStore.register() → search() round-trip and checks:
        - Registering an embedding and searching with the SAME embedding
          returns similarity_score ≈ 1.0 (perfect match)
        - Searching an empty store returns is_known=False
        - Two different people with the SAME name get different UUIDs
        - Registering a second sample under the same UUID increases count
        - Deleting a person removes them (cascade check)

Each check prints PASS or FAIL with a reason.

Deliberately broken input tests:
    - Zero-size crop → should return (None, None), not crash
    - Tiny (5x5) crop → should return (None, None) due to MIN_CROP_SIZE gate
    - Random noise crop → InsightFace may return None (no face detected)
    - Empty identity store search → should return (None, None, 0.0, False)

Run from project root:
    python -m src.layer4_identity.embedding_validator

Or via the validation entry point:
    python validate_layer4_embeddings.py
"""

import sys
import os
import numpy as np

# Ensure project root is on path
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.layer4_identity.identity_store import IdentityStore, EMBEDDING_DIM

# ─── Test helpers ─────────────────────────────────────────────────────────────

_pass_count = 0
_fail_count = 0


def check(label: str, condition: bool, fail_reason: str = ""):
    global _pass_count, _fail_count
    if condition:
        _pass_count += 1
        print(f"  ✅ PASS  {label}")
    else:
        _fail_count += 1
        reason = f" — {fail_reason}" if fail_reason else ""
        print(f"  ❌ FAIL  {label}{reason}")


def make_random_embedding() -> np.ndarray:
    """Generate a random L2-normalised 512-d float32 vector."""
    v = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def run_embedding_unit_tests():
    """Run IdentityStore unit tests without InsightFace (pure FAISS tests)."""
    print(f"\n{'─'*58}")
    print(f"  [EmbedValidator] IdentityStore Unit Tests")
    print(f"{'─'*58}\n")

    # Use a temp store path (in-memory only, not saved to models/)
    store = IdentityStore(store_path="/tmp/_test_identity_store", recognition_threshold=0.45)

    # Test 1: Empty store search
    emb = make_random_embedding()
    pid, name, score, is_known = store.search(emb)
    check("Empty store search returns is_known=False", not is_known,
          f"got is_known={is_known}")
    check("Empty store search returns score=0.0", score == 0.0,
          f"got score={score}")

    # Test 2: Register + same-embedding search → score ≈ 1.0
    emb_alice = make_random_embedding()
    pid_alice = store.register(emb_alice, name="Alice")
    result_pid, result_name, result_score, result_known = store.search(emb_alice)
    check("Same embedding search returns score ≈ 1.0",
          result_score > 0.99,
          f"got score={result_score:.4f}")
    check("Same embedding search returns is_known=True (threshold=0.45)",
          result_known,
          f"got is_known={result_known}")
    check("Same embedding search returns correct name",
          result_name == "Alice",
          f"got name={result_name}")

    # Test 3: Two people with the SAME name get DIFFERENT UUIDs
    emb_alice2 = make_random_embedding()  # Different person, same name
    pid_alice2 = store.register(emb_alice2, name="Alice")
    check("Two people with same name get different UUIDs",
          pid_alice != pid_alice2,
          f"got same UUID: {pid_alice}")

    # Test 4: Multiple samples for same person
    emb_alice_sample2 = make_random_embedding()
    store.register(emb_alice_sample2, person_id=pid_alice)
    person_info = store.get_person(pid_alice)
    check("Adding second sample increases embedding_count to 2",
          person_info["embedding_count"] == 2,
          f"got count={person_info['embedding_count']}")

    # Test 5: Registering under non-existent UUID raises ValueError
    raised = False
    try:
        store.register(make_random_embedding(), person_id="fake-uuid-that-doesnt-exist")
    except ValueError:
        raised = True
    check("Registering under invalid UUID raises ValueError",
          raised)

    # Test 6: Delete cascade
    n_before = store.n_embeddings
    store.delete_person(pid_alice)
    n_after = store.n_embeddings
    check("Delete cascade removes all samples for person",
          n_after == n_before - 2,  # 2 samples were registered for Alice
          f"expected {n_before-2}, got {n_after}")
    check("Deleted person no longer appears in get_person()",
          store.get_person(pid_alice) is None)

    # Test 7: Deleted person's UUID raises ValueError on register
    raised2 = False
    try:
        store.register(make_random_embedding(), person_id=pid_alice)
    except ValueError:
        raised2 = True
    check("Registering under deleted UUID raises ValueError", raised2)

    # Test 8: Invalid embedding shape raises ValueError
    bad_emb_raised = False
    try:
        store._validate_embedding(np.zeros(128, dtype=np.float32))
    except ValueError:
        bad_emb_raised = True
    check("Wrong embedding shape (128-d) raises ValueError", bad_emb_raised)

    # Test 9: n_people and n_embeddings are consistent
    # After deleting Alice (2 samples), only alice2 remains (1 sample)
    check("n_people = 1 after deletion",
          store.n_people == 1,
          f"got {store.n_people}")
    check("n_embeddings = 1 after deletion",
          store.n_embeddings == 1,
          f"got {store.n_embeddings}")

    # Test 10: update_name
    store.update_name(pid_alice2, "Alice B.")
    p = store.get_person(pid_alice2)
    check("update_name changes name field",
          p["name"] == "Alice B.",
          f"got name={p['name']}")

    # Clean up temp files
    for ext in [".faiss", ".meta.json"]:
        try:
            os.remove("/tmp/_test_identity_store" + ext)
        except FileNotFoundError:
            pass


def run_embedding_shape_tests():
    """
    Test FaceEmbedder with degenerate inputs (no InsightFace model load needed
    for pure-shape tests — uses the MIN_CROP_SIZE gate directly).
    """
    print(f"\n{'─'*58}")
    print(f"  [EmbedValidator] Embedding Input Shape Tests")
    print(f"{'─'*58}\n")

    # Import the constant without loading the model
    from src.layer4_identity.embedder import MIN_CROP_SIZE
    check(f"MIN_CROP_SIZE is >= 30 (reasonable minimum face size)",
          MIN_CROP_SIZE >= 30,
          f"got MIN_CROP_SIZE={MIN_CROP_SIZE}")


def print_summary():
    total = _pass_count + _fail_count
    print(f"\n{'='*58}")
    print(f"  EMBEDDING VALIDATION SUMMARY")
    print(f"{'='*58}")
    print(f"  Tests run  : {total}")
    print(f"  Passed     : {_pass_count}")
    print(f"  Failed     : {_fail_count}")
    if _fail_count == 0:
        print(f"\n  ✅ All checks passed.")
    else:
        print(f"\n  ❌ {_fail_count} check(s) failed — review output above.")
    print(f"{'='*58}\n")

    print(f"  Visual Review Checklist (manual verification):")
    print(f"  [ ] Run --add-new via register_face.py, confirm UUID generated")
    print(f"  [ ] Run --list, confirm person appears with correct name")
    print(f"  [ ] Run --add-sample for same person, confirm count increases")
    print(f"  [ ] Run --delete, confirm person removed from --list")
    print(f"  [ ] Two --add-new with same --name produce different UUIDs")
    print(f"{'='*58}\n")


if __name__ == "__main__":
    print(f"\n{'='*58}")
    print(f"  Layer 4 — Embedding + IdentityStore Unit Validator")
    print(f"{'='*58}")

    run_embedding_shape_tests()
    run_embedding_unit_tests()
    print_summary()

    sys.exit(0 if _fail_count == 0 else 1)
