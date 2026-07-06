"""
src/layer4_identity/identifier.py
─────────────────────────────────────────────────────────────────────────────
Layer 4: Identity — Main Orchestrator

The public-facing class for Layer 4. Orchestrates the full identity pipeline:
    1. For each detection: extract ArcFace embedding from face_crop (Embedder)
    2. Search FAISS identity store for closest known identity (IdentityStore)
    3. Update DeepSORT tracker with bboxes + ArcFace appearance features (Tracker)
    4. Write track_id, identity_label, embedding, similarity_score, is_known,
       and aligned_face back to each Detection object in the FrameContext.

Input:  FrameContext with ctx.detections populated by Layer 3.
Output: Same FrameContext with Layer 4 fields added to each Detection.

Contract:
    - Empty detection list (no faces): tracker.update() is still called with
      an empty list so DeepSORT can correctly age and expire existing tracks.
    - No face found by InsightFace in a crop: embedding=None, identity_label=None,
      is_known=None, similarity_score=None, track_id still assigned if possible.
    - Thread safety: not guaranteed — run in a single thread per FrameContext.

Ref: Layer 4 Architecture Doc — Sections 2, 3, 10
"""

from src.core.frame_context import FrameContext
from src.layer4_identity.embedder import FaceEmbedder
from src.layer4_identity.identity_store import IdentityStore
from src.layer4_identity.tracker import FaceTracker

# ─── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_STORE_PATH = "models/identity_store"
DEFAULT_CONFIDENCE = 0.5   # Only pass detections with >= this score to tracker


class FaceIdentifier:
    """
    Layer 4 orchestrator — connects Embedder, IdentityStore, and Tracker.

    Designed to be instantiated once and called per frame in the pipeline loop.
    Models are loaded at construction time; frame processing is stateless
    (state is held in the Tracker's Kalman filter and track gallery).

    Usage
    -----
        identifier = FaceIdentifier()
        # Per frame:
        ctx = identifier.identify(ctx)
        # ctx.detections[i].track_id, .identity_label, .embedding etc. are now set
    """

    def __init__(
        self,
        store_path: str = DEFAULT_STORE_PATH,
        recognition_threshold: float = 0.45,
        device: str = "auto"
    ):
        """
        Parameters
        ----------
        store_path : str
            Base path for FAISS index files (no extension).
            Passed to IdentityStore.
        recognition_threshold : float
            Cosine similarity cutoff for known vs. unknown (default 0.45).
            Calibrate on deployment data — see Arch Doc §8.
        device : str
            'cuda', 'cpu', or 'auto'. Passed to FaceEmbedder (InsightFace).
        """
        print(f"\n  Initializing Layer 4: Identity...")

        self.embedder = FaceEmbedder(device=device)
        self.store = IdentityStore(
            store_path=store_path,
            recognition_threshold=recognition_threshold
        )
        self.tracker = FaceTracker()

        print(
            f"  [Layer4] Ready. "
            f"Registry: {self.store.n_people} people, "
            f"{self.store.n_embeddings} embeddings.\n"
        )

    # ─── Public API ───────────────────────────────────────────────────────────

    def identify(self, ctx: FrameContext) -> FrameContext:
        """
        Run the full Layer 4 identity pipeline on a FrameContext.

        Enriches each Detection in ctx.detections with:
            track_id, identity_label, embedding, similarity_score,
            is_known, aligned_face

        Must be called every frame (even with empty ctx.detections) so
        DeepSORT can correctly manage track lifecycles.

        Parameters
        ----------
        ctx : FrameContext
            Must have ctx.detections populated by Layer 3 (FaceDetector).

        Returns
        -------
        FrameContext
            Same object with Layer 4 fields populated on each Detection.
        """
        raw_detections_for_tracker = []

        # ── Step 1 + 2: Embed + Identify each detection ──────────────────────
        for det in ctx.detections:
            embedding, aligned_face = None, None
            pid, label, score, is_known = None, None, None, None

            # Step 1 — Embed the face crop
            if det.face_crop is not None:
                embedding, aligned_face = self.embedder.get_embedding(det.face_crop)

            # Step 2 — Search identity store
            if embedding is not None and self.store.n_embeddings > 0:
                pid, label, score, is_known = self.store.search(embedding)
            elif embedding is not None:
                # Store is empty — face is unknown but embedding succeeded
                pid, label, score, is_known = None, None, 0.0, False

            # Write identity fields to Detection.
            # Contract (module docstring): when no embedding could be
            # extracted, identity fields stay None — "unknown" is reserved
            # for faces that WERE embedded but matched nothing.
            det.embedding = embedding
            det.aligned_face = aligned_face
            if embedding is None:
                det.identity_label = None
            else:
                det.identity_label = label if is_known else "unknown"
            det.similarity_score = score
            det.is_known = is_known

            # Prepare tracker input: bbox [l, t, w, h]
            x1, y1, x2, y2 = det.bbox_original
            bbox_ltwh = [x1, y1, x2 - x1, y2 - y1]
            raw_detections_for_tracker.append(
                (bbox_ltwh, det.confidence, embedding)
            )

        # ── Step 3: Update tracker (must run even with empty list) ────────────
        # ctx.timestamp drives the departed-track re-acquisition window, so
        # video files (media time) and live sources (wall clock) both work.
        track_id_map = self.tracker.update(
            raw_detections_for_tracker,
            frame=ctx.original_frame,
            timestamp=ctx.timestamp
        )

        # ── Step 4: Write track_ids back to detections ────────────────────────
        for det_idx, det in enumerate(ctx.detections):
            det.track_id = track_id_map.get(det_idx)

        return ctx
