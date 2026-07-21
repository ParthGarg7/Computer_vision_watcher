"""
src/core/drawing.py
─────────────────────────────────────────────────────────────────────────────
Shared overlay rendering for every pipeline entry point.

main.py and the three layer validators all draw the same things — boxes,
stacked labels, probability bars, a HUD — and each used to carry its own copy
of the maths. Four copies meant four places to fix any drawing bug, and they
had already drifted: the validators still rendered a failed embedding as
*nothing at all*, the silent-failure bug that hid a dead identity layer for
weeks in main.py.

One implementation, composed with flags, so each caller shows only the layers
it cares about:

    L3 validator : boxes + confidence + landmarks
    L4 validator : boxes + track id + identity
    L5 validator : boxes + track id + expression + bars
    main.py      : everything (per --no-identity / --no-expression)

Colour convention (BGR), consistent across all callers:
    green   bounding box, detection confidence
    orange  track id
    purple  recognised identity
    grey    unknown identity, or low-confidence expression
    red     no-embed — InsightFace found no face in the crop
    teal    expression
    cyan    HUD
"""

import cv2
import numpy as np

# ─── Palette (BGR) ────────────────────────────────────────────────────────────

COLORS = {
    "box":        (0, 220, 80),     # Green — bounding box
    "conf_bg":    (0, 180, 60),     # Green — detection confidence
    "track_bg":   (200, 100, 0),    # Orange — track ID
    "known_bg":   (120, 0, 180),    # Purple — recognised identity
    "unknown_bg": (60, 60, 60),     # Dark grey — embedded but unrecognised
    "fail_bg":    (0, 0, 200),      # Red — embedding failed (loud on purpose)
    "expr_bg":    (0, 130, 180),    # Teal — expression
    "expr_low_bg": (60, 60, 60),    # Dark grey — low-confidence expression
    "mood_bg":    (110, 70, 20),    # Deep blue — mood (valence/arousal state)
    "text":       (255, 255, 255),  # White
    "hud":        (0, 220, 220),    # Cyan
    "hint":       (160, 160, 160),  # Grey
    "bar_bg":     (40, 40, 40),
    "bar_fg":     (0, 200, 160),
    "landmark":   (0, 120, 255),    # Orange dots
}

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_LABEL = 0.52
FONT_SCALE_HUD = 0.60
FONT_SCALE_HINT = 0.45
FONT_SCALE_BAR = 0.34
BOX_THICKNESS = 2
TEXT_THICKNESS = 1

# Expression below this confidence is shown as "uncertain" rather than named.
# With 8 classes, random guessing is 12.5% — under 35% the model is
# essentially guessing, and a confident-looking wrong label is worse than
# admitting uncertainty.
EXPR_MIN_CONFIDENCE = 0.35


# ─── Primitives ───────────────────────────────────────────────────────────────

def put_label(img, text, x, y_bottom, bg_color,
              text_color=COLORS["text"],
              font_scale=FONT_SCALE_LABEL, thickness=TEXT_THICKNESS):
    """
    Draw a filled-background text label whose bottom edge sits at y_bottom.

    Returns the label's top y, so callers can stack labels upward by feeding
    the return value back in as the next y_bottom.

    The x position is clamped so a label never runs off the right edge — a
    face near the frame border used to push its label out of view entirely,
    which mattered most for the red 'no-embed' warning.
    """
    (tw, th), base = cv2.getTextSize(text, FONT, font_scale, thickness)
    box_w = tw + 6
    frame_w = img.shape[1]
    x = max(0, min(x, frame_w - box_w - 1))
    y_top = max(0, y_bottom - th - base - 6)
    cv2.rectangle(img, (x, y_top), (x + box_w, y_bottom), bg_color, -1)
    cv2.putText(img, text, (x + 3, y_bottom - base - 2),
                FONT, font_scale, text_color, thickness)
    return y_top


def clamp_bbox(bbox, frame_shape):
    """Integer-ise a bbox and clamp it inside the frame. Returns x1,y1,x2,y2."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    fh, fw = frame_shape[:2]
    return max(0, x1), max(0, y1), min(fw - 1, x2), min(fh - 1, y2)


def draw_hud(img, text):
    """Top-left status line."""
    cv2.putText(img, text, (10, 28), FONT, FONT_SCALE_HUD, COLORS["hud"], 2)


def draw_hint(img, text="Q: Quit  |  F: Fullscreen"):
    """Bottom-left key hint."""
    cv2.putText(img, text, (10, img.shape[0] - 10),
                FONT, FONT_SCALE_HINT, COLORS["hint"], TEXT_THICKNESS)


def draw_probability_bars(img, scores: dict, x, y, top_n=3,
                          bar_w=80, bar_h=12, gap=2):
    """Horizontal probability bars for the top-N expression classes."""
    if not scores:
        return
    frame_w = img.shape[1]
    for cls_name, prob in sorted(scores.items(),
                                 key=lambda kv: kv[1], reverse=True)[:top_n]:
        if x + bar_w + 35 > frame_w:
            return                       # would run off the frame
        cv2.rectangle(img, (x, y), (x + bar_w, y + bar_h), COLORS["bar_bg"], -1)
        cv2.rectangle(img, (x, y), (x + int(bar_w * prob), y + bar_h),
                      COLORS["bar_fg"], -1)
        cv2.putText(img, f"{cls_name[:5]} {prob:.0%}", (x + 2, y + bar_h - 2),
                    FONT, FONT_SCALE_BAR, COLORS["text"], TEXT_THICKNESS)
        y += bar_h + gap


# ─── Label builders ───────────────────────────────────────────────────────────

def identity_label(det):
    """
    (text, bg_colour) for a detection's identity, in three ALWAYS-VISIBLE
    states. The label is never blank: rendering a failure as silence is
    indistinguishable from "working fine", which is exactly how a completely
    dead identity layer survived a live demo and a full audit.

        purple  Parth (0.77)    recognised
        grey    unknown (0.32)  embedded, but no match above threshold
        red     no-embed        InsightFace found no face in the crop
    """
    if det.is_known and det.identity_label:
        # `is not None` — a real score of 0.0 is falsy and must not be
        # mistaken for "missing".
        score = (f"{det.similarity_score:.2f}"
                 if det.similarity_score is not None else "")
        text = f"{det.identity_label} ({score})" if score else det.identity_label
        return text, COLORS["known_bg"]
    if det.embedding is not None:
        score = (f"{det.similarity_score:.2f}"
                 if det.similarity_score is not None else "0.00")
        return f"unknown ({score})", COLORS["unknown_bg"]
    return "no-embed", COLORS["fail_bg"]


def expression_label(det):
    """(text, bg_colour) for a detection's expression, or None if unavailable."""
    if not det.dominant_expression:
        return None
    conf = det.expression_confidence or 0.0
    if conf >= EXPR_MIN_CONFIDENCE:
        return f"{det.dominant_expression} {conf:.0%}", COLORS["expr_bg"]
    return f"uncertain {conf:.0%}", COLORS["expr_low_bg"]


# ─── Composite ────────────────────────────────────────────────────────────────

def draw_detections(
    img: np.ndarray,
    detections: list,
    show_confidence: bool = False,
    show_track: bool = False,
    show_identity: bool = False,
    show_expression: bool = False,
    show_mood: bool = False,
    show_bars: bool = False,
    show_landmarks: bool = False,
    track_placeholder: bool = False,
    expression_placeholder: bool = False,
) -> np.ndarray:
    """
    Draw boxes and stacked labels for every detection, in place.

    Labels stack upward from the top of each box in a fixed order, so the
    same information always appears in the same position:

        expression        (top)
        identity
        track id
        confidence
        [ bounding box ]

    Parameters
    ----------
    show_confidence / show_track / show_identity / show_expression
        Which layers' labels to render.
    show_bars
        Top-3 expression probability bars to the right of the box.
    show_landmarks
        5-point facial landmarks, when the checkpoint provides them.
    track_placeholder
        Show "ID:?" for tentative (unconfirmed) tracks instead of omitting
        the label. Used by the Layer 4 validator, where seeing which
        detections have not yet been confirmed is the point.
    expression_placeholder
        Show "analysing..." when a face has no expression result yet.
        Used by the Layer 5 validator to distinguish "throttled, no reading
        yet" from "expression disabled" — its review checklist depends on
        this appearing only on a track's first few frames.
    """
    for det in detections:
        x1, y1, x2, y2 = clamp_bbox(det.bbox_original, img.shape)
        cv2.rectangle(img, (x1, y1), (x2, y2), COLORS["box"], BOX_THICKNESS)

        label_y = y1   # cursor, moves upward as labels are stacked

        if show_confidence:
            label_y = put_label(img, f"{det.confidence:.2f}", x1, label_y,
                                COLORS["conf_bg"])

        if show_track:
            if det.track_id is not None:
                label_y = put_label(img, f"ID:{det.track_id}", x1, label_y,
                                    COLORS["track_bg"])
            elif track_placeholder:
                label_y = put_label(img, "ID:?", x1, label_y,
                                    COLORS["track_bg"])

        if show_identity:
            text, bg = identity_label(det)
            label_y = put_label(img, text, x1, label_y, bg)

        if show_expression:
            expr = expression_label(det)
            if expr is not None:
                label_y = put_label(img, expr[0], x1, label_y, expr[1])
            elif expression_placeholder:
                label_y = put_label(img, "analysing...", x1, label_y,
                                    COLORS["expr_bg"])

        # Mood sits above the expression: it is the broader state, derived
        # from valence/arousal rather than the 8-way classification.
        if show_mood and getattr(det, "mood", None):
            text = det.mood
            if det.valence is not None:
                text = f"{det.mood}  {det.valence:+.2f}/{det.arousal:+.2f}"
            label_y = put_label(img, text, x1, label_y, COLORS["mood_bg"])

        if show_bars and det.expression_scores:
            draw_probability_bars(img, det.expression_scores, x2 + 6, y1)

        if show_landmarks and det.landmarks_original:
            for lx, ly, _ in det.landmarks_original:
                cv2.circle(img, (int(lx), int(ly)), 4, COLORS["landmark"], -1)

    return img
