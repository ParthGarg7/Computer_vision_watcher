"""
tests/test_identity_store.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for Layer 4's FAISS IdentityStore — pure FAISS + JSON logic,
no InsightFace model required.

Run:  python -m unittest discover tests
"""

import os
import sys
import tempfile
import unittest

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.layer4_identity.identity_store import IdentityStore, EMBEDDING_DIM


def _unit_vec(seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(EMBEDDING_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


class IdentityStoreTests(unittest.TestCase):

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.store_path = os.path.join(self._dir.name, "test_store")
        self.store = IdentityStore(store_path=self.store_path,
                                   recognition_threshold=0.45)

    def tearDown(self):
        self._dir.cleanup()

    def test_empty_store_search(self):
        pid, name, score, is_known = self.store.search(_unit_vec(1))
        self.assertIsNone(pid)
        self.assertEqual(score, 0.0)
        self.assertFalse(is_known)

    def test_register_and_exact_match(self):
        emb = _unit_vec(2)
        pid = self.store.register(emb, name="Alice")
        rpid, rname, score, is_known = self.store.search(emb)
        self.assertEqual(rpid, pid)
        self.assertEqual(rname, "Alice")
        self.assertAlmostEqual(score, 1.0, places=4)
        self.assertTrue(is_known)

    def test_unrelated_embedding_is_unknown(self):
        self.store.register(_unit_vec(3), name="Alice")
        # Random 512-d unit vectors are near-orthogonal → similarity ≈ 0
        _, _, score, is_known = self.store.search(_unit_vec(999))
        self.assertLess(score, 0.45)
        self.assertFalse(is_known)

    def test_multiple_samples_share_person_id(self):
        pid = self.store.register(_unit_vec(4), name="Bob")
        pid2 = self.store.register(_unit_vec(5), person_id=pid)
        self.assertEqual(pid, pid2)
        self.assertEqual(self.store.n_people, 1)
        self.assertEqual(self.store.n_embeddings, 2)

    def test_register_unknown_person_id_raises(self):
        with self.assertRaises(ValueError):
            self.store.register(_unit_vec(6), person_id="no-such-uuid")

    def test_bad_embedding_shape_raises(self):
        with self.assertRaises(ValueError):
            self.store.register(np.zeros(10, dtype=np.float32))

    def test_delete_person_rebuilds_index(self):
        pid_a = self.store.register(_unit_vec(7), name="A")
        pid_b = self.store.register(_unit_vec(8), name="B")
        self.store.delete_person(pid_a)
        self.assertEqual(self.store.n_people, 1)
        self.assertEqual(self.store.n_embeddings, 1)
        # Remaining person must still be searchable
        rpid, _, score, _ = self.store.search(_unit_vec(8))
        self.assertEqual(rpid, pid_b)
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_save_and_load_round_trip(self):
        emb = _unit_vec(9)
        pid = self.store.register(emb, name="Carol")
        self.store.save()
        # No stray .tmp files left behind by the atomic write
        self.assertFalse(os.path.exists(self.store_path + ".faiss.tmp"))
        self.assertFalse(os.path.exists(self.store_path + ".meta.json.tmp"))

        reloaded = IdentityStore(store_path=self.store_path)
        self.assertEqual(reloaded.n_people, 1)
        rpid, rname, score, is_known = reloaded.search(emb)
        self.assertEqual(rpid, pid)
        self.assertEqual(rname, "Carol")
        self.assertTrue(is_known)

    def test_update_name(self):
        pid = self.store.register(_unit_vec(10))
        self.assertIsNone(self.store.get_person(pid)["name"])
        self.store.update_name(pid, "Dave")
        self.assertEqual(self.store.get_person(pid)["name"], "Dave")
        with self.assertRaises(ValueError):
            self.store.update_name("no-such-uuid", "X")


if __name__ == "__main__":
    unittest.main()
