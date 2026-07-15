"""
src/layer4_identity/identity_store.py
─────────────────────────────────────────────────────────────────────────────
Layer 4 Sub-module: FAISS Identity Registry

Manages the face identity database — stores known-person embeddings, performs
fast nearest-neighbour cosine similarity search, and persists the index to
disk so identities survive application restarts.

Architecture decisions:
    - FAISS IndexFlatIP: exact brute-force inner product search.
      Chosen for MVP (< 10,000 identities). No training step required.
      100% accurate. Correct for cosine similarity because all embeddings
      are L2-normalised by InsightFace (inner product = cosine similarity
      on L2-normalised vectors).
    - person_id is a UUID string — primary key, never changes.
    - name is optional metadata stored separately in a JSON sidecar file.
    - Multiple embeddings per person_id are all stored in the FAISS index;
      similarity search returns the best match across all samples.

FAISS normalisation note:
    IndexFlatIP computes raw inner product. For cosine similarity to equal
    inner product, BOTH stored vectors AND query vectors must be L2-normalised.
    InsightFace's normed_embedding is already L2-normalised — do not re-normalise.
    External embeddings must be normalised before indexing.

Files persisted:
    <store_path>.faiss       — FAISS index binary
    <store_path>.meta.json   — UUID→name mapping + per-UUID embedding count

Ref: Layer 4 Architecture Doc — Sections 3 (Step 3), 7, 8
"""

import json
import os
import uuid
from typing import Optional

import faiss
import numpy as np

# ─── Constants ────────────────────────────────────────────────────────────────

EMBEDDING_DIM = 512          # ArcFace embedding dimensionality
DEFAULT_THRESHOLD = 0.45     # Cosine similarity threshold: recognised vs unknown
                             # Tune on real deployment data — see Arch Doc §8.

# Base path for the two store files. Lives under faces/ — NOT models/ —
# because this is irreplaceable biometric data, not a re-downloadable model
# weight. faces/ is gitignored; never commit it.
DEFAULT_STORE_PATH = "faces/db/identity_store"


class IdentityStore:
    """
    FAISS-backed face identity registry.

    Stores L2-normalised 512-d ArcFace embeddings indexed by person_id (UUID).
    Multiple face samples per person are all stored; search returns the highest
    similarity score across all samples for each person.

    Identity key is always person_id (UUID). Name is optional display metadata.
    Two different people with the same name get two different UUIDs — the name
    field is never used as a uniqueness criterion.
    """

    def __init__(
        self,
        store_path: str = DEFAULT_STORE_PATH,
        recognition_threshold: float = DEFAULT_THRESHOLD
    ):
        """
        Parameters
        ----------
        store_path : str
            Base path for the two sidecar files:
            <store_path>.faiss and <store_path>.meta.json
        recognition_threshold : float
            Cosine similarity threshold (0.0-1.0). Embeddings with
            similarity_score >= threshold → is_known=True.
            Default 0.45 — calibrate on deployment data.
        """
        self.store_path = store_path
        self.recognition_threshold = recognition_threshold
        self._faiss_path = store_path + ".faiss"
        self._meta_path = store_path + ".meta.json"

        # _meta: maps person_id (UUID str) → {"name": str|None, "count": int}
        self._meta: dict = {}

        # _id_map: ordered list of person_id strings, one entry per FAISS row.
        # FAISS rows are positional (0-indexed); _id_map[row] = person_id.
        # Multiple rows can share the same person_id (multiple face samples).
        self._id_map: list = []

        # FAISS index — IndexFlatIP for exact inner product search.
        self._index = faiss.IndexFlatIP(EMBEDDING_DIM)

        # Load existing store if files exist
        if os.path.exists(self._faiss_path) and os.path.exists(self._meta_path):
            self._load()
        else:
            print(f"  [IdentityStore] No existing store at '{store_path}'. "
                  f"Starting empty.")

    # ─── Public API ───────────────────────────────────────────────────────────

    @property
    def n_people(self) -> int:
        """Number of distinct registered persons."""
        return len(self._meta)

    @property
    def n_embeddings(self) -> int:
        """Total number of face embeddings stored (>= n_people)."""
        return self._index.ntotal

    def register(
        self,
        embedding: np.ndarray,
        person_id: Optional[str] = None,
        name: Optional[str] = None
    ) -> str:
        """
        Add a face embedding to the registry.

        Parameters
        ----------
        embedding : np.ndarray
            L2-normalised 512-d float32 vector from FaceEmbedder.
        person_id : str or None
            UUID string of an existing person to add a new sample for.
            If None, a new UUID is generated (new person registration).
        name : str or None
            Optional display name. Only used when creating a new person
            (person_id=None). Ignored when adding to existing person.

        Returns
        -------
        str
            The person_id used (newly generated or the one passed in).

        Raises
        ------
        ValueError
            If person_id is provided but not found in the registry.
        """
        if person_id is not None and person_id not in self._meta:
            raise ValueError(
                f"person_id '{person_id}' not found in registry. "
                f"Pass person_id=None to create a new person."
            )

        emb = self._validate_embedding(embedding)

        if person_id is None:
            # New person — generate UUID and register metadata
            person_id = str(uuid.uuid4())
            self._meta[person_id] = {"name": name, "count": 0}

        # Add embedding vector to FAISS index
        self._index.add(emb.reshape(1, -1))
        self._id_map.append(person_id)
        self._meta[person_id]["count"] += 1

        return person_id

    def search(
        self,
        embedding: np.ndarray,
        top_k: int = 1
    ) -> tuple:
        """
        Find the closest registered identity to the given embedding.

        Uses FAISS IndexFlatIP (exact inner product = cosine similarity
        for L2-normalised vectors). Returns the best match across ALL
        stored embeddings for each person — i.e., multiple samples per
        person are all candidates.

        Parameters
        ----------
        embedding : np.ndarray
            L2-normalised 512-d float32 query vector.
        top_k : int
            Number of nearest neighbours to retrieve. Default 1.

        Returns
        -------
        (person_id, name, similarity_score, is_known)
            person_id       : str  — UUID of best match, or None if empty.
            name            : str or None — display name of best match.
            similarity_score: float — cosine similarity (0.0-1.0).
            is_known        : bool — True if score >= recognition_threshold.

        Returns (None, None, 0.0, False) if the registry is empty.
        """
        if self._index.ntotal == 0:
            return None, None, 0.0, False

        emb = self._validate_embedding(embedding)
        scores, indices = self._index.search(emb.reshape(1, -1), top_k)

        # Aggregate: for each unique person, take their highest score
        best_score = -1.0
        best_pid = None

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            pid = self._id_map[idx]
            if score > best_score:
                best_score = float(score)
                best_pid = pid

        if best_pid is None:
            return None, None, 0.0, False

        name = self._meta[best_pid].get("name")
        is_known = best_score >= self.recognition_threshold
        return best_pid, name, best_score, is_known

    def get_person(self, person_id: str) -> Optional[dict]:
        """
        Look up a person by UUID.

        Returns
        -------
        dict with keys: person_id, name, embedding_count
        None if not found.
        """
        if person_id not in self._meta:
            return None
        info = self._meta[person_id]
        return {
            "person_id": person_id,
            "name": info.get("name"),
            "embedding_count": info.get("count", 0),
        }

    def list_people(self) -> list:
        """
        Return a list of all registered persons as dicts.
        Each dict: {person_id, name, embedding_count}
        """
        return [
            {
                "person_id": pid,
                "name": info.get("name"),
                "embedding_count": info.get("count", 0),
            }
            for pid, info in self._meta.items()
        ]

    def update_name(self, person_id: str, name: Optional[str]):
        """
        Set or update the display name for an existing person.

        The name is metadata only — it never serves as an identity key.
        Two different people can share a name; they remain distinct by UUID.

        Parameters
        ----------
        person_id : str — UUID of the person to update.
        name : str or None — new display name.

        Raises
        ------
        ValueError if person_id not found.
        """
        if person_id not in self._meta:
            raise ValueError(f"person_id '{person_id}' not found.")
        self._meta[person_id]["name"] = name

    def delete_person(self, person_id: str):
        """
        Remove a person and all their embeddings from the registry.

        FAISS IndexFlatIP does not support individual row deletion.
        This rebuilds the entire index from scratch minus the deleted person.
        For MVP scale this is acceptable; production should use IndexHNSWFlat
        or HNSWlib for true CRUD support.

        Parameters
        ----------
        person_id : str — UUID to delete.

        Raises
        ------
        ValueError if person_id not found.
        """
        if person_id not in self._meta:
            raise ValueError(f"person_id '{person_id}' not found.")

        # Collect all embeddings that are NOT from this person
        new_embeddings = []
        new_id_map = []

        if self._index.ntotal > 0:
            # Reconstruct all stored vectors
            all_vectors = np.zeros(
                (self._index.ntotal, EMBEDDING_DIM), dtype=np.float32
            )
            for i in range(self._index.ntotal):
                all_vectors[i] = self._index.reconstruct(i)

            for i, pid in enumerate(self._id_map):
                if pid != person_id:
                    new_embeddings.append(all_vectors[i])
                    new_id_map.append(pid)

        # Rebuild FAISS index
        self._index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self._id_map = new_id_map
        del self._meta[person_id]

        if new_embeddings:
            batch = np.stack(new_embeddings, axis=0)
            self._index.add(batch)

    def save(self):
        """
        Persist the FAISS index and metadata sidecar to disk.

        Creates parent directories if they don't exist.
        Safe to call after every registration for durability.

        Writes are atomic: both files are written to a .tmp sibling first
        and swapped in with os.replace(), so a crash mid-save can never
        leave a truncated index or metadata file (the previous complete
        version survives). The registry is biometric data — a corrupt
        store would silently break recognition for every registered person.
        """
        os.makedirs(os.path.dirname(self._faiss_path) or ".", exist_ok=True)

        faiss.write_index(self._index, self._faiss_path + ".tmp")
        os.replace(self._faiss_path + ".tmp", self._faiss_path)

        meta_payload = {
            "meta": self._meta,
            "id_map": self._id_map,
            "threshold": self.recognition_threshold,
        }
        with open(self._meta_path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, indent=2)
        os.replace(self._meta_path + ".tmp", self._meta_path)

    # ─── Internal ─────────────────────────────────────────────────────────────

    def _load(self):
        """Load a persisted index and metadata from disk."""
        print(f"  [IdentityStore] Loading store from '{self.store_path}'...")
        self._index = faiss.read_index(self._faiss_path)
        with open(self._meta_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self._meta = payload["meta"]
        self._id_map = payload["id_map"]
        if "threshold" in payload:
            self.recognition_threshold = payload["threshold"]
        print(
            f"  [IdentityStore] Loaded: {self.n_people} people, "
            f"{self.n_embeddings} embeddings."
        )

    @staticmethod
    def _validate_embedding(embedding: np.ndarray) -> np.ndarray:
        """
        Validate and normalise an embedding vector.

        InsightFace normed_embedding is already L2-normalised, so this
        is mostly a safety check. External embeddings must be normalised.
        """
        emb = np.asarray(embedding, dtype=np.float32)
        if emb.shape != (EMBEDDING_DIM,):
            raise ValueError(
                f"Expected embedding shape ({EMBEDDING_DIM},), "
                f"got {emb.shape}."
            )
        # Re-normalise for safety (noop if already normalised)
        norm = np.linalg.norm(emb)
        if norm > 1e-6:
            emb = emb / norm
        return emb
