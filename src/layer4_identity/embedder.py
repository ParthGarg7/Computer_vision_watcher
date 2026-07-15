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

from src.core.gpu_setup import register_nvidia_dlls, cuda_is_usable

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
        # Register CUDA DLLs before any ONNX session is created — without
        # this, InsightFace's CUDA provider silently falls back to CPU.
        register_nvidia_dlls()

        if device == "auto":
            self._device_id = 0 if cuda_is_usable() else -1
        elif device == "cuda":
            self._device_id = 0
        else:
            self._device_id = -1  # -1 = CPU in InsightFace

        self.model_pack = model_pack
        self.det_size = det_size

        # Embedding failures are silent by design (identity must never crash
        # the pipeline) — but silence is how the SCRFD-context bug survived a
        # demo and a full audit. Warn once so it can never hide again.
        self._embed_warned = False

        print(f"  [Layer4-Embedder] Loading InsightFace model pack: {model_pack}")
        print(f"  [Layer4-Embedder] Root       : {_INSIGHTFACE_ROOT}")
        print(f"  [Layer4-Embedder] Device ID  : {self._device_id} "
              f"({'CUDA' if self._device_id >= 0 else 'CPU'})")

        # allowed_modules: buffalo_l ships 5 models (SCRFD detection, ArcFace
        # recognition, gender/age, 2D-106 and 3D-68 landmarks). The pipeline
        # only uses detection + recognition — loading just those saves
        # ~200 MB RAM/VRAM and skips 3 extra inferences per crop.
        self._app = FaceAnalysis(
            name=model_pack,
            root=_INSIGHTFACE_ROOT,
            allowed_modules=["detection", "recognition"],
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self._device_id >= 0
            else ["CPUExecutionProvider"]
        )
        self._app.prepare(ctx_id=self._device_id, det_size=det_size)

        # Report the provider the recognition model actually got — ORT drops
        # CUDA silently if its DLLs fail to load.
        try:
            rec_model = self._app.models.get("recognition")
            actual = rec_model.session.get_providers()
            print(f"  [Layer4-Embedder] ONNX providers: {actual}")
        except Exception:
            pass

        print(f"  [Layer4-Embedder] Ready.\n")

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

        # ── Restore scene context before detection ───────────────────────────
        # SCRFD is trained on scenes where a face occupies a FRACTION of the
        # frame. Layer 3's crop is padded only CROP_PADDING_RATIO (15%), so the
        # face fills it almost entirely — which is out of distribution, and
        # detection returns zero faces on a perfectly clear face. Padding the
        # crop back out with a replicated border restores the framing SCRFD
        # expects. Measured on a real 229x256 crop (YOLO conf 0.90):
        #     tight crop        -> 0 faces          -> no identity at all
        #     crop + 50% border -> 1 face, 0.76     -> matches at 0.77
        # The bordered embedding agrees with a full-frame embedding of the same
        # face at cosine 0.93-0.99, so identity quality is unaffected.
        # NOTE: face.kps land in `padded` coordinates, so alignment below must
        # warp from `padded` — not from the original crop.
        padded = cv2.copyMakeBorder(
            face_crop, h // 2, h // 2, w // 2, w // 2, cv2.BORDER_REPLICATE
        )

        try:
            faces = self._app.get(padded)
        except Exception as e:
            self._warn_once(f"InsightFace raised {type(e).__name__}: {e}")
            return None, None

        if not faces:
            self._warn_once(
                "InsightFace found no face in a crop that Layer 3 detected. "
                "Check crop framing/quality."
            )
            return None, None

        # Use the highest-confidence detection from InsightFace
        face = max(faces, key=lambda f: f.det_score)

        embedding = face.normed_embedding  # shape (512,), already L2-normalised

        # Aligned 112x112 warp, for debugging/audit display only — the
        # embedding above already comes from this alignment internally.
        try:
            aligned_face = self._get_aligned_face(padded, face)
        except Exception:
            aligned_face = cv2.resize(face_crop, (112, 112))

        return embedding.astype(np.float32), aligned_face

    # ─── Internal ─────────────────────────────────────────────────────────────

    def _warn_once(self, msg: str):
        """Print an embedding-failure warning the first time only."""
        if not self._embed_warned:
            self._embed_warned = True
            print(f"  [Layer4-Embedder] WARNING: {msg} "
                  f"Faces will show 'no-embed'. Further warnings suppressed.")

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
