"""
src/layer4_identity/embedder.py
─────────────────────────────────────────────────────────────────────────────
Layer 4 Sub-module: Face Embedding via InsightFace (ArcFace buffalo_l)

Responsibilities:
    1. Load the InsightFace buffalo_l model pack (ResNet-50 + ArcFace backbone).
    2. Accept a face_crop (BGR uint8 numpy array) from Layer 3.
    3. Run InsightFace's internal SCRFD detection + ArcFace pipeline on the crop.
    4. Return the L2-normalised 512-d embedding and the 112x112 aligned face.

Why InsightFace over alternatives:
    - Bundles SCRFD detector + ArcFace recognition in one package.
    - Returns normed_embedding (already L2-normalised) — use np.dot() directly.
    - BGR input throughout — no conversion needed after cv2.
    - buffalo_l achieves 99.83% on LFW (ResNet-50 backbone).

Why buffalo_l over antelopev2:
    - Community default, best documentation, sufficient accuracy for MVP.
    - antelopev2 (ResNet-100) is the production upgrade when higher accuracy
      is needed at the cost of increased inference time.

IMPORTANT — BGR input:
    InsightFace expects BGR throughout (detection, alignment, embedding).
    Do NOT convert face_crop to RGB before passing it here.

IMPORTANT — model download:
    buffalo_l weights (~500 MB) are downloaded from GitHub on first
    FaceAnalysis.prepare() call. Subsequent runs are fully offline.
    For air-gapped use: pre-download and set root= to a local directory.

Ref: Layer 4 Architecture Doc — Sections 3 (Steps 1–2), 5 (Model Packs)
"""

import numpy as np
import cv2
import os

# InsightFace import — requires: pip install insightface
from insightface.app import FaceAnalysis

# ─── Constants ────────────────────────────────────────────────────────────────

# Directory where InsightFace stores downloaded model packs.
# Set via environment variable for air-gapped deployments.
_INSIGHTFACE_ROOT = os.environ.get(
    "INSIGHTFACE_ROOT",
    os.path.join(os.path.expanduser("~"), ".insightface")
)

# Model pack — buffalo_l is the MVP default (ResNet-50, 99.83% LFW).
DEFAULT_MODEL_PACK = "buffalo_l"

# Input image size expected by InsightFace's detector.
# det_size is the resolution at which SCRFD runs — 640x640 is the standard.
DEFAULT_DET_SIZE = (640, 640)

# Minimum face crop dimension (H or W) below which embedding is skipped.
# Crops smaller than this produce unreliable embeddings.
MIN_CROP_SIZE = 40


class FaceEmbedder:
    """
    Layer 4 face embedding component using InsightFace (ArcFace buffalo_l).

    Accepts raw BGR face crops from Layer 3, runs InsightFace's full
    detection + alignment + ArcFace pipeline, and returns L2-normalised
    512-d embeddings and 112x112 aligned face crops.

    InsightFace's get() method:
        1. Runs SCRFD detector on the crop to locate faces within it.
        2. Computes 5-pt landmarks from the detected face.
        3. Applies affine warp to 112x112 (norm_crop / aligned_face).
        4. Runs ArcFace ResNet-50 backbone → 512-d embedding.
        5. L2-normalises the embedding → normed_embedding.
    """

    def __init__(
        self,
        model_pack: str = DEFAULT_MODEL_PACK,
        det_size: tuple = DEFAULT_DET_SIZE,
        device: str = "auto"
    ):
        """
        Parameters
        ----------
        model_pack : str
            InsightFace model pack name. Default 'buffalo_l'.
            Use 'buffalo_sc' for CPU-constrained deployments.
            Use 'antelopev2' for production ResNet-100 accuracy upgrade.
        det_size : tuple (W, H)
            Detection resolution for SCRFD. (640, 640) is standard.
            Reduce to (320, 320) if throughput is constrained.
        device : str
            'cuda', 'cpu', or 'auto' (auto-detects CUDA).
        """
        import torch
        if device == "auto":
            self._device_id = 0 if torch.cuda.is_available() else -1
        elif device == "cuda":
            self._device_id = 0
        else:
            self._device_id = -1  # -1 = CPU in InsightFace

        self.model_pack = model_pack
        self.det_size = det_size

        print(f"  [Layer4-Embedder] Loading InsightFace model pack: {model_pack}")
        print(f"  [Layer4-Embedder] Root       : {_INSIGHTFACE_ROOT}")
        print(f"  [Layer4-Embedder] Device ID  : {self._device_id} "
              f"({'CUDA' if self._device_id >= 0 else 'CPU'})")

        self._app = FaceAnalysis(
            name=model_pack,
            root=_INSIGHTFACE_ROOT,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self._device_id >= 0
            else ["CPUExecutionProvider"]
        )
        self._app.prepare(ctx_id=self._device_id, det_size=det_size)

        print(f"  [Layer4-Embedder] Ready.\\n")

    # ─── Public API ───────────────────────────────────────────────────────────

    def get_embedding(
        self,
        face_crop: np.ndarray
    ) -> tuple:
        """
        Extract ArcFace embedding from a face crop.

        InsightFace runs its own SCRFD detector on the crop to find the face
        within it, then aligns and embeds it. This means the crop does NOT need
        to be pre-aligned — InsightFace handles alignment internally.

        Parameters
        ----------
        face_crop : np.ndarray
            BGR uint8 array (H, W, 3) from Layer 3's Detection.face_crop.
            Do NOT convert to RGB — InsightFace expects BGR throughout.

        Returns
        -------
        (embedding, aligned_face) or (None, None)
            embedding    : np.ndarray shape (512,) float32, L2-normalised.
                           None if the crop is too small or no face found.
            aligned_face : np.ndarray shape (112, 112, 3) BGR uint8.
                           None if embedding failed.
        """
        if face_crop is None:
            return None, None

        h, w = face_crop.shape[:2]
        if h < MIN_CROP_SIZE or w < MIN_CROP_SIZE:
            return None, None

        try:
            faces = self._app.get(face_crop)
        except Exception:
            return None, None

        if not faces:
            # InsightFace found no face in the crop.
            # Fall back: try on a slightly larger version of the crop.
            try:
                upscaled = cv2.resize(face_crop, (max(w, 160), max(h, 160)))
                faces = self._app.get(upscaled)
            except Exception:
                return None, None

        if not faces:
            return None, None

        # Use the highest-confidence detection from InsightFace
        face = max(faces, key=lambda f: f.det_score)

        embedding = face.normed_embedding  # shape (512,), already L2-normalised
        aligned_face = face.normed_embedding  # placeholder

        # Extract aligned 112x112 crop via warp (stored in face object)
        # InsightFace stores this internally; we recompute via get_feat if needed.
        # The aligned BGR crop can be accessed via the face's aligned attribute
        # when using the app pipeline — fall back to a resized crop if absent.
        try:
            aligned_face = self._get_aligned_face(face_crop, face)
        except Exception:
            aligned_face = cv2.resize(face_crop, (112, 112))

        return embedding.astype(np.float32), aligned_face

    # ─── Internal ─────────────────────────────────────────────────────────────

    def _get_aligned_face(self, frame: np.ndarray, face) -> np.ndarray:
        """
        Produce the 112x112 aligned face crop using InsightFace's warp logic.

        InsightFace uses an affine similarity transform from 5 detected
        landmarks to 5 canonical reference points, then applies
        cv2.warpAffine() — this is the 'norm_crop' step.

        For display and audit purposes only — the embedding already
        comes from this aligned face internally.
        """
        from insightface.utils import face_align

        kps = face.kps  # 5-point keypoints in the crop's coordinate space
        aligned = face_align.norm_crop(frame, landmark=kps, image_size=112)
        return aligned
