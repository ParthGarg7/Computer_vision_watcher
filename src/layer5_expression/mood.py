"""
src/layer5_expression/mood.py
─────────────────────────────────────────────────────────────────────────────
Layer 5 Sub-module: Mood states from valence and arousal (+ custom rules)

── Why this exists ──────────────────────────────────────────────────────────

The 8-class emotion model puts every face into exactly one box: anger,
contempt, disgust, fear, happiness, neutral, sadness, surprise. That has a
blind spot — it cannot tell CALMLY happy from EXCITEDLY happy. Both are just
"happiness". And it has no way at all to express states like bored or tense,
because there is no such button.

The multi-task model (enet_b0_8_va_mtl) outputs two extra numbers alongside
the 8 classes:

    valence  — unpleasant (-1) .......... pleasant (+1)
    arousal  — calm       (-1) .......... excited  (+1)

Every mood is a point on that plane. This is Russell's circumplex model, a
long-established framework in affect research — not a recipe we invented:

                          AROUSAL +
                              │
                 tense        │        excited
            (unpleasant,      │      (pleasant,
              worked up)      │       worked up)
                              │
    VALENCE - ────────────────┼──────────────── VALENCE +
                              │
                subdued       │        content
            (unpleasant,      │      (pleasant,
                 flat)        │         calm)
                              │
                          AROUSAL -

That gives four states the 8-class model cannot express, plus a neutral zone
in the middle where neither dimension is strong enough to call.

── Honesty note ─────────────────────────────────────────────────────────────

The Layer 5 Architecture Doc is explicit: this layer estimates VISIBLE
FACIAL PATTERNS, not internal states. "subdued" here means the face shows
low arousal and slightly negative valence — it does NOT mean the person is
bored, unhappy, or disengaged. Any such claim needs separate validation
before it drives a business decision.

── Custom rules (extension point) ───────────────────────────────────────────

CUSTOM_RULES lets you define your own named states on top of what the model
already produces — combinations of the 8 class probabilities and/or the two
dimensions. Rules are checked in order, highest priority first; the first
match wins, and if none match the mood falls back to the valence/arousal
quadrant. Add a rule and it appears on screen and in the database with no
other code change.

Nothing is trained by adding a rule: you are defining the meaning yourself,
so keep names descriptive of what is observable rather than asserting what
someone feels.
"""

from typing import Callable, NamedTuple, Optional

# ─── Calibration ──────────────────────────────────────────────────────────────
#
# The model's neutral point is NOT (0, 0). Measured over 299 readings from
# real webcam footage, grouped by the model's own 8-class output:
#
#     expression   n     valence   arousal
#     neutral     212     -0.203    -0.177   <- the resting baseline
#     happiness    31     +0.557    -0.077
#     sadness       6     -0.418    -0.143
#     surprise     50     -0.076    +0.390
#
# The dimensions clearly work — happy is pleasant, sad is unpleasant,
# surprised is activated — but everything is shifted down and left. Without
# correcting for that, an ordinary resting face reads as "subdued", which is
# both wrong and unflattering.
#
# So readings are re-centred on the measured neutral point before being
# classified. RE-MEASURE THESE for a new camera, lighting setup, or model:
# run the pipeline, take the mean valence/arousal of frames the 8-class model
# calls "neutral", and put those numbers here.
VALENCE_BASELINE = -0.203
AROUSAL_BASELINE = -0.177

# How far from the baseline a reading must sit before it counts as clearly
# pleasant/unpleasant or calm/activated. Inside this band the dimension is
# reported as "mid". 0.20 keeps ordinary neutral faces in the middle cell
# while still separating the happiness/sadness/surprise clusters above.
VALENCE_DEADZONE = 0.20
AROUSAL_DEADZONE = 0.20

# ─── Mood states ──────────────────────────────────────────────────────────────
#
# Three bands per dimension give nine states rather than four quadrants.
# Quadrants alone are brittle: a reading sitting almost exactly on an axis
# gets forced into "pleasant" or "unpleasant" on the strength of a rounding
# error. The middle band absorbs that.
#
#                              AROUSAL
#                    calm        mid        activated
#                 ┌───────────┬───────────┬───────────┐
#   VALENCE  +ve  │  content  │  pleased  │  excited  │
#                 ├───────────┼───────────┼───────────┤
#            mid  │  relaxed  │  neutral  │   alert   │
#                 ├───────────┼───────────┼───────────┤
#            -ve  │  subdued  │displeased │   tense   │
#                 └───────────┴───────────┴───────────┘
#
# Names describe the observable configuration, not an inferred feeling.
MOOD_CONTENT    = "content"      # pleasant + calm
MOOD_PLEASED    = "pleased"      # pleasant
MOOD_EXCITED    = "excited"      # pleasant + activated
MOOD_RELAXED    = "relaxed"      # calm
MOOD_NEUTRAL    = "neutral"      # middle of both
MOOD_ALERT      = "alert"        # activated
MOOD_SUBDUED    = "subdued"      # unpleasant + calm
MOOD_DISPLEASED = "displeased"   # unpleasant
MOOD_TENSE      = "tense"        # unpleasant + activated

# [valence band][arousal band] -> name, bands ordered low, mid, high
_MOOD_GRID = (
    (MOOD_SUBDUED, MOOD_DISPLEASED, MOOD_TENSE),    # valence low
    (MOOD_RELAXED, MOOD_NEUTRAL,    MOOD_ALERT),    # valence mid
    (MOOD_CONTENT, MOOD_PLEASED,    MOOD_EXCITED),  # valence high
)

ALL_MOODS = tuple(name for row in _MOOD_GRID for name in row)


class MoodRule(NamedTuple):
    """
    A custom named state.

    name      : what to display, e.g. "confused"
    predicate : fn(scores: dict, valence: float, arousal: float) -> bool
                scores maps class name -> probability (sums to 1.0);
                valence/arousal are floats in roughly -1..+1, or None when
                the model in use does not provide them.
    priority  : higher wins when several rules match. Default 0.
    """
    name: str
    predicate: Callable
    priority: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM RULES — add your own here.
#
# Each entry is checked before the valence/arousal quadrants, so a rule can
# override the default naming. Examples are commented out; uncomment or write
# your own and it takes effect immediately — no other file needs changing.
#
# CUSTOM_RULES = [
#     # Surprise and fear together, without much happiness, often reads as
#     # confusion. NOTE: this is a definition YOU are choosing, not something
#     # the model was trained to detect.
#     MoodRule(
#         name="confused",
#         predicate=lambda s, v, a: (s.get("surprise", 0) > 0.25
#                                    and s.get("fear", 0) > 0.15
#                                    and s.get("happiness", 0) < 0.20),
#         priority=10,
#     ),
#
#     # Strongly unpleasant and highly activated.
#     MoodRule(
#         name="agitated",
#         predicate=lambda s, v, a: (v is not None and a is not None
#                                    and v < -0.35 and a > 0.30),
#         priority=5,
#     ),
# ]
# ─────────────────────────────────────────────────────────────────────────────

CUSTOM_RULES: list = []


# ─── Public API ───────────────────────────────────────────────────────────────

def _band(value: float, deadzone: float) -> int:
    """0 = below the dead zone, 1 = inside it, 2 = above."""
    if value < -deadzone:
        return 0
    if value > deadzone:
        return 2
    return 1


def quadrant_mood(valence: Optional[float],
                  arousal: Optional[float]) -> Optional[str]:
    """
    Map a valence/arousal pair to one of the nine mood names.

    Both values are re-centred on the measured neutral baseline first, so an
    ordinary resting face lands in the middle cell rather than being reported
    as "subdued".

    Returns None when the model provides no valence/arousal (the plain
    8-class models), so callers can fall back to the dominant expression.
    """
    if valence is None or arousal is None:
        return None
    v = _band(valence - VALENCE_BASELINE, VALENCE_DEADZONE)
    a = _band(arousal - AROUSAL_BASELINE, AROUSAL_DEADZONE)
    return _MOOD_GRID[v][a]


def resolve_mood(scores: dict,
                 valence: Optional[float],
                 arousal: Optional[float]) -> Optional[str]:
    """
    Determine the mood state for one reading.

    Custom rules are evaluated first, highest priority first; the first match
    wins. If none match (or none are defined), the valence/arousal quadrant
    is used. Returns None when neither applies — i.e. a plain 8-class model
    with no custom rules — leaving dominant_expression as the only label.

    A rule that raises is ignored rather than crashing the pipeline: a typo in
    a user-supplied lambda must not take the whole run down.
    """
    for rule in sorted(CUSTOM_RULES, key=lambda r: -r.priority):
        try:
            if rule.predicate(scores or {}, valence, arousal):
                return rule.name
        except Exception:
            continue
    return quadrant_mood(valence, arousal)


def describe(valence: Optional[float], arousal: Optional[float]) -> str:
    """Short human-readable form for logs/overlays, e.g. 'v+0.62 a-0.05'."""
    if valence is None or arousal is None:
        return ""
    return f"v{valence:+.2f} a{arousal:+.2f}"
