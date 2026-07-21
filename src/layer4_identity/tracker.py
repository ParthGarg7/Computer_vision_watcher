"""
src/layer4_identity/tracker.py
─────────────────────────────────────────────────────────────────────────────
Layer 4 Sub-module: Multi-face Temporal Tracker (DeepSORT)

Maintains consistent identity across video frames by assigning each detected
face a persistent track_id (integer) that survives brief occlusions, brief
exits from frame, and noisy detections.

── LOOP 4 EXPLANATION: DeepSORT Default vs. ArcFace Appearance Swap ─────────

WHAT DEEPSORT'S DEFAULT APPEARANCE MODEL DOES:
    DeepSORT extends SORT (Kalman filter + IoU matching) with a second
    association step using a deep appearance descriptor — a CNN that produces
    a 128-d "looks-like" vector per detection.
    The default CNN is a ResNet50 pretrained on the Market-1501 pedestrian
    re-identification dataset (people walking on a university campus).
    It was trained to distinguish BODIES: clothing colour, torso shape,
    walking gait pattern, etc.

WHY THE PEDESTRIAN MODEL IS SUBOPTIMAL FOR FACE TRACKING:
    1. Wrong feature space: Market-1501 images show full bodies at ~128x64px.
       ArcFace faces are 112x112 crops of facial regions. The feature hierarchy
       the pedestrian CNN learned (legs, torso, backpack patterns) has zero
       overlap with the features that discriminate between faces (jawline, eye
       spacing, skin texture, nose shape).
    2. Wrong input size: DeepSORT's ReID model expects the full bounding box
       region — for face tracking, this is just a face crop. The pedestrian
       model produces poor embeddings for face-sized inputs because it was
       trained on body-sized regions.
    3. Less discriminative: Two people with similar clothing would confuse the
       pedestrian model. Two people with similar faces (siblings, twins) would
       confuse ANY appearance model, but ArcFace is specifically optimised to
       separate faces in the angular embedding space via the ArcFace margin loss.
    4. Wasted compute: Running a separate ReID CNN just to get a 128-d vector
       that is worse than the 512-d ArcFace embedding being computed anyway
       is strictly worse in both accuracy and efficiency.

WHY ARCFACE EMBEDDINGS ARE A STRONG SUBSTITUTE (THE SWAP):
    1. Already computed: FaceEmbedder.get_embedding() runs InsightFace's
       ArcFace backbone for recognition. The 512-d normed_embedding comes
       for free — passing it to DeepSORT's appearance matcher costs zero
       additional inference.
    2. Face-discriminative: ArcFace was specifically trained with angular
       margin loss on 5.8M faces (WebFace260M / MS1M datasets) to maximally
       separate different-person faces in 512-d space. Same-person embeddings
       cluster tightly; different-person embeddings are well separated.
    3. Higher dimensionality: 512-d vs 128-d — more discriminative for
       the face re-association step when a person briefly leaves frame.

TRADE-OFFS AND RISKS OF THE SWAP:
    1. Dimensionality mismatch: DeepSORT's NearestNeighborDistanceMetric
       expects whatever dimension you give it — there is no hard-coded 128-d
       requirement. Passing 512-d vectors works with no code change beyond
       not loading the external ReID model. ✅ No issue.
    2. Distance metric: DeepSORT uses cosine distance internally for the
       appearance matching step. ArcFace embeddings are L2-normalised, so
       cosine distance = 1 - cosine_similarity = 1 - inner_product, which
       is exactly what DeepSORT expects. ✅ Consistent.
    3. Threshold re-tuning: The default max_cosine_distance=0.2 was tuned for
       128-d pedestrian embeddings. With 512-d ArcFace embeddings, same-person
       cosine distances tend to be LOWER (more similar) than with the
       pedestrian model — meaning the default 0.2 threshold may be too strict
       and cause missed re-associations. Recommended starting point: 0.4-0.5.
       MUST be calibrated on deployment data. ⚠ Needs tuning.
    4. Missing embedding frames: When a face is briefly undetected (fast
       movement, occlusion, low confidence), no ArcFace embedding is available
       for that frame. DeepSORT's Kalman filter handles position prediction
       through these gaps using just IoU matching (falling back gracefully to
       the motion model). This is handled by the update() call with an empty
       detection list — DeepSORT ages tracks without crashing. ✅ Handled.

CONFIRMATION: This swap is confirmed and implemented below.

── Parameters ────────────────────────────────────────────────────────────────
    max_age             : 70 frames — track survives 70 frames without detection
    n_init              : 3 frames  — 3 consecutive detections to confirm a track
    max_cosine_distance : 0.45      — tuned for ArcFace 512-d (was 0.2 for 128-d)
    nn_budget           : 100       — appearance gallery size per track

Ref: Layer 4 Architecture Doc — Sections 3 (Step 4), 6 (DeepSORT Parameters)
Ref: Wojke et al. (2017). DeepSORT. ICIP 2017. arXiv:1703.07402.
"""

import time
import numpy as np
from typing import List, Optional, Tuple

from deep_sort_realtime.deepsort_tracker import DeepSort

# ─── Constants ────────────────────────────────────────────────────────────────

# Track lifecycle parameters — see docstring above and Arch Doc §6.
#
# MAX_AGE is how long DeepSORT keeps predicting a track's position after the
# detector stops seeing it. It is the ONLY thing that preserves the ID of a
# person whose face cannot be embedded — a distant or blurry face that
# InsightFace cannot fingerprint (12% of detections in real webcam footage).
# Those tracks have nothing to store in the departed gallery, so once
# DeepSORT gives up, a returning person is indistinguishable from a new one.
#
# Measured over a 1628-frame clip, counting how many display IDs were minted:
#     max_age  seconds @21fps   IDs created
#         70        3.3s             4      <- previous value
#        150        7.1s             3      <- chosen
#        250       11.9s             3      (no further gain)
#
# 70 frames was ~3.3 seconds, so stepping out of frame for 5-10 seconds
# always produced a new ID. 150 covers that gap.
#
# TRADE-OFF: a track surviving longer without detections is matched on
# position alone (no appearance signal when there is no embedding), so a
# DIFFERENT person entering the same spot within the window can inherit the
# ID. Registered people are unaffected — their display ID is pinned to their
# FAISS identity and cannot be stolen this way. Lower this if strangers
# start merging; raise it if IDs still churn.
MAX_AGE = 150             # Frames before a lost track is deleted
N_INIT = 3                # Consecutive detections to confirm a new track
MAX_COSINE_DISTANCE = 0.45  # Tuned for ArcFace 512-d (default was 0.2 for 128-d)
NN_BUDGET = 100           # Max embeddings stored per track in appearance gallery

# Constant unit-vector substitute for detections with no ArcFace embedding.
# Computed once at import — regenerating it per frame wasted allocations.
# See the WHY NOT a zero-vector note in FaceTracker.update().
_rng = np.random.RandomState(42)
_FALLBACK_EMB = _rng.randn(512).astype(np.float32)
_FALLBACK_EMB /= np.linalg.norm(_FALLBACK_EMB)  # norm = 1.0
del _rng

# ── Display-ID reconciliation (stable IDs across departures) ─────────────────
# DeepSORT deletes a track after max_age frames without detection and has NO
# memory of deleted tracks — a person leaving frame for a few seconds always
# comes back as a brand-new internal ID. Its internal counter also increments
# for every initiated track, including 1-frame detection flickers, so raw IDs
# inflate fast (observed: 3 people → "ID:98").
#
# FaceTracker therefore never exposes DeepSORT IDs. It maps them to small
# sequential DISPLAY IDs, and re-acquires a returning person's display ID by
# two signals, strongest first:
#   1. FAISS identity (person_id) — for REGISTERED people. This is the same
#      recognition that shows their name, so it is robust to head pose and
#      brief absences and persists for the whole run. A recognised person
#      keeps ONE display ID no matter how they turn or how long they step
#      away. This is the primary fix for "returned as ID 7 / name still shows
#      from the side": the embedding centroid below was too pose-sensitive to
#      hold the ID even though FAISS still recognised the face.
#   2. ArcFace embedding centroid gallery — the fallback for UNKNOWN
#      (unregistered) people, who have no stable identity key. A new track's
#      embedding is compared against recently departed unknown tracks.

# Cosine similarity required to re-acquire a departed UNKNOWN track's display
# ID by embedding centroid. Same-person ArcFace similarity is typically
# 0.55-0.85 frontal, but drops sharply at profile angles — which is exactly
# why registered people are re-acquired by FAISS identity instead. Kept
# slightly stricter than the FAISS recognition threshold (0.45) so two
# strangers are never merged into one ID.
REACQUIRE_SIM_THRESHOLD = 0.50

# How long a departed UNKNOWN track stays eligible for centroid re-acquisition.
# (Registered people re-acquire by identity with no time limit.)
REACQUIRE_WINDOW_SEC = 120.0

# Max departed unknown tracks remembered (oldest evicted beyond this).
DEPARTED_GALLERY_MAX = 50

# EMA weight of the newest embedding in a track's identity centroid.
EMB_EMA_ALPHA = 0.30


class FaceTracker:
    """
    Layer 4 temporal tracker using DeepSORT with ArcFace appearance embeddings.

    Wraps deep_sort_realtime.DeepSort. Accepts per-frame detection lists
    (bboxes + ArcFace embeddings) and returns confirmed track IDs.

    The ArcFace embedding (512-d) replaces DeepSORT's default pedestrian
    ReID model (128-d) as the appearance descriptor — see module docstring
    for full explanation of this swap.

    Usage
    -----
        tracker = FaceTracker()
        # Per frame:
        track_ids = tracker.update(detections_with_embeddings)
        # detections_with_embeddings: list of (bbox_ltwh, confidence, embedding)
    """

    def __init__(
        self,
        max_age: int = MAX_AGE,
        n_init: int = N_INIT,
        max_cosine_distance: float = MAX_COSINE_DISTANCE,
        nn_budget: int = NN_BUDGET
    ):
        """
        Parameters
        ----------
        max_age : int
            Frames a track survives without a matching detection.
            Increase for high-occlusion scenes.
        n_init : int
            Consecutive detections required before a track is confirmed.
        max_cosine_distance : float
            Maximum cosine distance for appearance matching.
            0.45 is the tuned starting point for 512-d ArcFace embeddings.
            Calibrate on deployment data.
        nn_budget : int
            Maximum appearance embeddings stored per track in the gallery.
        """
        self._tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
            # embedder=None disables DeepSORT's internal ReID model —
            # we supply ArcFace embeddings directly via the feature param.
            embedder=None,
        )

        # ── Display-ID reconciliation state (see module constants) ───────────
        self._ds_to_display: dict = {}      # DeepSORT id → display id
        self._active_info: dict = {}        # display id → {"emb", "last_seen"}
        self._departed: dict = {}           # display id → {"emb", "last_seen"}
        self._identity_to_display: dict = {}  # FAISS person_id → display id
        self._next_display_id = 1
        print(
            f"  [Layer4-Tracker] DeepSORT initialized "
            f"(max_age={max_age}, n_init={n_init}, "
            f"max_cosine_dist={max_cosine_distance})"
        )
        print(f"  [Layer4-Tracker] Appearance model: ArcFace 512-d (swapped "
              f"from default pedestrian ReID 128-d)")

    # ─── Public API ───────────────────────────────────────────────────────────

    def update(
        self,
        raw_detections: List[Tuple],
        frame: np.ndarray,
        timestamp: Optional[float] = None,
        identity_keys: Optional[List] = None
    ) -> dict:
        """
        Update tracker with detections from the current frame.

        This method MUST be called every frame, even when no faces are detected
        (pass an empty list). DeepSORT's Kalman filter needs the update call to
        correctly age and expire tracks when faces disappear.
        Skipping the update on empty frames breaks track expiry timing.

        Parameters
        ----------
        raw_detections : list of (bbox_ltwh, confidence, embedding)
            bbox_ltwh   : [left, top, width, height] in original pixel coords
            confidence  : float 0.0-1.0
            embedding   : np.ndarray (512,) float32 ArcFace embedding, or None
                          If None (no InsightFace result), DeepSORT falls back
                          to motion-only IoU matching for this detection.

        frame : np.ndarray
            The original BGR frame (H, W, 3). Passed to DeepSort.update_tracks()
            as required by the API, but not re-processed internally since
            embedder=None.

        timestamp : float or None
            Capture time of this frame (ctx.timestamp). Used for the
            re-acquisition window on departed tracks. None → time.time().

        identity_keys : list or None
            Parallel to raw_detections: the FAISS person_id (str) for each
            recognised face, or None for unknown/unregistered faces. When
            present, a recognised person's display ID is pinned to their
            identity so it survives pose changes and long absences.

        Returns
        -------
        dict mapping detection index (int) → track_id (int).
            IDs are stable DISPLAY IDs (small, sequential): the same person
            re-entering the frame within REACQUIRE_WINDOW_SEC gets their
            previous ID back (matched by ArcFace embedding), and DeepSORT's
            internal ID churn is never exposed.
            Only confirmed tracks are included (n_init detections reached).
            Tentative tracks are excluded — they have no stable track_id yet.
        """
        now = timestamp if timestamp is not None else time.time()

        if not raw_detections:
            # Must still call update to age out existing tracks via Kalman.
            # MUST pass embeds=[] explicitly — DeepSORT raises if embeds=None
            # when embedder=None, even on an empty detection list.
            tracks = self._tracker.update_tracks([], embeds=[], frame=frame)
            self._retire_stale(tracks, now)
            return {}

        # Build the input format DeepSort.update_tracks expects.
        #
        # CORRECT API (deep_sort_realtime >= 1.3):
        #   update_tracks(raw_detections, embeds=<list>, frame=frame)
        #
        #   raw_detections : list of (bbox_ltwh, confidence, class_id)
        #                    ← ONLY 3 elements per tuple, NO feature inside
        #   embeds         : list of np.ndarray (512,) — ONE per detection,
        #                    in the SAME ORDER as raw_detections
        #
        # When embedder=None, embeds MUST be provided and MUST be non-None
        # for every entry. If InsightFace returned None (tiny/no face crop),
        # substitute a stable unit-vector fallback.
        #
        # WHY NOT a zero-vector?
        #   deep_sort_realtime's nn_matching normalises every embedding before
        #   cosine distance: v / ||v||. A zero-vector produces 0/0 = NaN,
        #   which corrupts the cost matrix and causes scipy to raise:
        #     "ValueError: matrix contains invalid numeric entries"
        #
        # FIX: use a tiny constant unit vector (seeded random, norm=1.0).
        #   Cosine distance to any real ArcFace embedding ≈ 1.0 (orthogonal),
        #   so these crops get no appearance match and DeepSORT falls back to
        #   pure Kalman/IoU tracking — exactly the intended behaviour.
        deepsort_input = []   # list of (bbox_ltwh, confidence, class_id)
        embeds_list    = []   # parallel list of 512-d embeddings
        det_indices    = []   # track which raw_detection index each entry maps to

        for det_idx, (bbox_ltwh, confidence, embedding) in enumerate(raw_detections):
            deepsort_input.append((bbox_ltwh, confidence, "face"))
            # Use ArcFace embedding if available; fallback unit vector = IoU-only tracking
            emb = embedding if embedding is not None else _FALLBACK_EMB
            embeds_list.append(emb)
            det_indices.append(det_idx)

        tracks = self._tracker.update_tracks(
            deepsort_input,
            embeds=embeds_list,
            frame=frame
        )


        # Map confirmed track IDs back to detection indices.
        # DeepSORT does not return a 1-to-1 mapping — we recover it via bbox overlap.
        confirmed = [t for t in tracks if t.is_confirmed()]

        # Translate DeepSORT internal IDs → stable display IDs, feeding each
        # detection's identity + embedding into the reconciler so a returning
        # person re-acquires their old ID.
        track_id_map = {}  # det_idx → display track_id
        for det_idx, (bbox_ltwh, confidence, embedding) in enumerate(raw_detections):
            best_track = self._find_best_track(bbox_ltwh, confirmed)
            if best_track is not None:
                ikey = (identity_keys[det_idx]
                        if identity_keys is not None else None)
                track_id_map[det_idx] = self._resolve_display_id(
                    best_track.track_id, embedding, now, ikey
                )

        self._retire_stale(tracks, now)
        return track_id_map

    # ─── Internal: display-ID reconciliation ──────────────────────────────────

    def _resolve_display_id(self, ds_id, embedding, now: float,
                            identity_key=None) -> int:
        """
        Translate a DeepSORT internal track ID to a stable display ID.

        Recognised person (identity_key set) → pin the display ID to their
        FAISS identity, healing any ID this track was mis-assigned before
        recognition kicked in. Robust to pose and absence duration.

        Unknown person → existing per-track mapping, else re-acquire a
        recently departed display ID by ArcFace centroid, else a new ID.
        """
        if identity_key is not None:
            return self._resolve_known(ds_id, embedding, now, identity_key)
        return self._resolve_unknown(ds_id, embedding, now)

    def _resolve_known(self, ds_id, embedding, now, identity_key) -> int:
        pinned = self._identity_to_display.get(identity_key)
        current = self._ds_to_display.get(ds_id)

        if pinned is None:
            # First sighting of this identity this run — adopt the display ID
            # this track already carries (if any), else mint a fresh one.
            pinned = current if current is not None else self._new_display_id()
            self._identity_to_display[identity_key] = pinned

        if current is not None and current != pinned:
            # Track was numbered before FAISS recognised it — retire the
            # stray ID so it doesn't linger in the active/departed maps.
            self._active_info.pop(current, None)
            self._departed.pop(current, None)

        self._ds_to_display[ds_id] = pinned
        info = self._active_info.setdefault(pinned, {"emb": None, "last_seen": now})
        info["last_seen"] = now
        self._update_centroid(info, embedding)
        return pinned

    def _resolve_unknown(self, ds_id, embedding, now) -> int:
        display_id = self._ds_to_display.get(ds_id)
        if display_id is None:
            display_id = self._match_departed(embedding, now)
            if display_id is None:
                display_id = self._new_display_id()
            self._ds_to_display[ds_id] = display_id
            self._active_info[display_id] = {"emb": None, "last_seen": now}

        info = self._active_info[display_id]
        info["last_seen"] = now
        self._update_centroid(info, embedding)
        return display_id

    def _new_display_id(self) -> int:
        did = self._next_display_id
        self._next_display_id += 1
        return did

    @staticmethod
    def _update_centroid(info: dict, embedding):
        """EMA-update a display ID's identity centroid. Ignores the fallback
        vector, which carries no identity information."""
        if embedding is None or embedding is _FALLBACK_EMB:
            return
        emb = np.asarray(embedding, dtype=np.float32)
        if info["emb"] is None:
            info["emb"] = emb.copy()
        else:
            blended = (1.0 - EMB_EMA_ALPHA) * info["emb"] + EMB_EMA_ALPHA * emb
            norm = np.linalg.norm(blended)
            if norm > 1e-6:
                info["emb"] = blended / norm

    def _match_departed(self, embedding, now: float):
        """
        Find a recently departed display ID whose embedding centroid matches
        this new track's embedding (cosine >= REACQUIRE_SIM_THRESHOLD within
        REACQUIRE_WINDOW_SEC). Pops and returns the match, or None.
        """
        if embedding is None or embedding is _FALLBACK_EMB:
            return None

        # Drop entries past the re-acquisition window first
        expired = [did for did, rec in self._departed.items()
                   if now - rec["last_seen"] > REACQUIRE_WINDOW_SEC]
        for did in expired:
            self._departed.pop(did)

        emb = np.asarray(embedding, dtype=np.float32)
        best_id, best_sim = None, REACQUIRE_SIM_THRESHOLD
        for did, rec in self._departed.items():
            sim = float(np.dot(emb, rec["emb"]))  # both L2-normalised
            if sim >= best_sim:
                best_sim, best_id = sim, did

        if best_id is not None:
            self._departed.pop(best_id)
        return best_id

    def _retire_stale(self, tracks: list, now: float):
        """
        Handle display IDs whose DeepSORT track no longer exists (deleted
        after max_age). Identity-pinned displays keep their persistent
        identity→display mapping (re-acquired by FAISS on return). Unknown
        displays move to the embedding gallery for centroid re-acquisition.
        """
        alive_ds_ids = {t.track_id for t in tracks}
        pinned_displays = set(self._identity_to_display.values())
        gone = [ds for ds in self._ds_to_display if ds not in alive_ds_ids]
        for ds_id in gone:
            display_id = self._ds_to_display.pop(ds_id)
            info = self._active_info.pop(display_id, None)
            if display_id in pinned_displays:
                # Re-acquired by identity, not by the embedding gallery.
                continue
            if info is not None and info["emb"] is not None:
                self._departed[display_id] = {
                    "emb": info["emb"],
                    "last_seen": info["last_seen"],
                }

        # Cap the gallery — evict oldest departures beyond the limit
        while len(self._departed) > DEPARTED_GALLERY_MAX:
            oldest = min(self._departed, key=lambda d: self._departed[d]["last_seen"])
            self._departed.pop(oldest)

    # ─── Internal ─────────────────────────────────────────────────────────────

    @staticmethod
    def _find_best_track(bbox_ltwh: list, confirmed_tracks: list):
        """
        Find the confirmed track whose predicted bbox has the highest IoU
        with the given detection bbox.

        Parameters
        ----------
        bbox_ltwh : list [left, top, width, height]
        confirmed_tracks : list of confirmed DeepSORT Track objects

        Returns
        -------
        Best matching Track object, or None if no tracks or low IoU.
        """
        if not confirmed_tracks:
            return None

        dl, dt, dw, dh = [float(v) for v in bbox_ltwh]
        dx1, dy1, dx2, dy2 = dl, dt, dl + dw, dt + dh

        best_iou = 0.0
        best_track = None

        for track in confirmed_tracks:
            try:
                tb = track.to_ltwh()  # predicted bbox from Kalman filter
                tl, tt, tw, th = float(tb[0]), float(tb[1]), float(tb[2]), float(tb[3])
                tx1, ty1, tx2, ty2 = tl, tt, tl + tw, tt + th

                # Intersection over Union
                ix1 = max(dx1, tx1)
                iy1 = max(dy1, ty1)
                ix2 = min(dx2, tx2)
                iy2 = min(dy2, ty2)

                inter_w = max(0.0, ix2 - ix1)
                inter_h = max(0.0, iy2 - iy1)
                inter_area = inter_w * inter_h

                det_area = dw * dh
                trk_area = tw * th
                union_area = det_area + trk_area - inter_area

                iou = inter_area / union_area if union_area > 0 else 0.0

                if iou > best_iou:
                    best_iou = iou
                    best_track = track

            except Exception:
                continue

        # Require at least 20% IoU to match — avoids spurious assignments
        return best_track if best_iou >= 0.20 else None
