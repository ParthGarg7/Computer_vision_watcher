"""
tests/test_mood.py
─────────────────────────────────────────────────────────────────────────────
Tests for Layer 5's valence/arousal mood states and the custom-rule hook.

The mood grid is calibrated against real measurements — 299 readings from
webcam footage, grouped by the model's own 8-class output:

    expression   n     valence   arousal    expected mood
    neutral     212     -0.203    -0.177    neutral
    happiness    31     +0.557    -0.077    pleased
    sadness       6     -0.418    -0.143    displeased
    surprise     50     -0.076    +0.390    alert

Those four cases are asserted directly: if a future change to the baseline
or dead zones makes an ordinary resting face read as "subdued" again, these
fail.

Run:  python -m unittest discover tests
"""

import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.layer5_expression import mood as M


class QuadrantMoodTests(unittest.TestCase):

    def test_measured_clusters_map_correctly(self):
        cases = [
            ("neutral",   -0.203, -0.177, M.MOOD_NEUTRAL),
            ("happiness",  0.557, -0.077, M.MOOD_PLEASED),
            ("sadness",   -0.418, -0.143, M.MOOD_DISPLEASED),
            ("surprise",  -0.076,  0.390, M.MOOD_ALERT),
        ]
        for label, v, a, expected in cases:
            with self.subTest(label):
                self.assertEqual(M.quadrant_mood(v, a), expected)

    def test_resting_face_is_never_subdued(self):
        # The bug this calibration exists to prevent: an ordinary neutral
        # face reported as "subdued" because the model's baseline is not 0.
        self.assertEqual(M.quadrant_mood(M.VALENCE_BASELINE,
                                         M.AROUSAL_BASELINE), M.MOOD_NEUTRAL)

    def test_all_nine_states_reachable(self):
        vb, ab = M.VALENCE_BASELINE, M.AROUSAL_BASELINE
        d = M.VALENCE_DEADZONE + 0.1
        seen = {M.quadrant_mood(vb + dv, ab + da)
                for dv in (-d, 0, d) for da in (-d, 0, d)}
        self.assertEqual(len(seen), 9)
        self.assertEqual(seen, set(M.ALL_MOODS))

    def test_extremes_map_to_corners(self):
        vb, ab = M.VALENCE_BASELINE, M.AROUSAL_BASELINE
        self.assertEqual(M.quadrant_mood(vb + 0.8, ab + 0.8), M.MOOD_EXCITED)
        self.assertEqual(M.quadrant_mood(vb + 0.8, ab - 0.8), M.MOOD_CONTENT)
        self.assertEqual(M.quadrant_mood(vb - 0.8, ab + 0.8), M.MOOD_TENSE)
        self.assertEqual(M.quadrant_mood(vb - 0.8, ab - 0.8), M.MOOD_SUBDUED)

    def test_none_when_model_has_no_dimensions(self):
        # Plain 8-class models: mood must be None so callers fall back to
        # dominant_expression rather than inventing a state.
        self.assertIsNone(M.quadrant_mood(None, None))
        self.assertIsNone(M.quadrant_mood(0.5, None))
        self.assertIsNone(M.quadrant_mood(None, 0.5))


class CustomRuleTests(unittest.TestCase):
    """The Option-A extension point: named states layered on the 8 classes."""

    def setUp(self):
        self._saved = list(M.CUSTOM_RULES)

    def tearDown(self):
        M.CUSTOM_RULES[:] = self._saved

    def test_no_rules_falls_back_to_quadrant(self):
        M.CUSTOM_RULES[:] = []
        self.assertEqual(M.resolve_mood({}, M.VALENCE_BASELINE,
                                        M.AROUSAL_BASELINE), M.MOOD_NEUTRAL)

    def test_custom_rule_overrides_quadrant(self):
        M.CUSTOM_RULES[:] = [
            M.MoodRule(name="confused",
                       predicate=lambda s, v, a: s.get("surprise", 0) > 0.25
                                                 and s.get("fear", 0) > 0.15,
                       priority=10),
        ]
        scores = {"surprise": 0.40, "fear": 0.25, "neutral": 0.35}
        self.assertEqual(
            M.resolve_mood(scores, M.VALENCE_BASELINE, M.AROUSAL_BASELINE),
            "confused")

    def test_non_matching_rule_falls_through(self):
        M.CUSTOM_RULES[:] = [
            M.MoodRule(name="confused",
                       predicate=lambda s, v, a: s.get("surprise", 0) > 0.9),
        ]
        self.assertEqual(
            M.resolve_mood({"surprise": 0.1}, M.VALENCE_BASELINE,
                           M.AROUSAL_BASELINE), M.MOOD_NEUTRAL)

    def test_highest_priority_wins(self):
        M.CUSTOM_RULES[:] = [
            M.MoodRule(name="low",  predicate=lambda s, v, a: True, priority=1),
            M.MoodRule(name="high", predicate=lambda s, v, a: True, priority=99),
        ]
        self.assertEqual(M.resolve_mood({}, 0.0, 0.0), "high")

    def test_broken_rule_does_not_crash_the_pipeline(self):
        # A typo in a user-supplied lambda must not take down a live run.
        M.CUSTOM_RULES[:] = [
            M.MoodRule(name="boom",
                       predicate=lambda s, v, a: 1 / 0, priority=50),
            M.MoodRule(name="ok",
                       predicate=lambda s, v, a: True, priority=10),
        ]
        self.assertEqual(M.resolve_mood({}, 0.0, 0.0), "ok")

    def test_rule_can_use_dimensions_when_available(self):
        M.CUSTOM_RULES[:] = [
            M.MoodRule(name="agitated",
                       predicate=lambda s, v, a: v is not None and v < -0.5,
                       priority=5),
        ]
        self.assertEqual(M.resolve_mood({}, -0.9, 0.0), "agitated")
        # Must not crash for an 8-class model where v/a are None
        self.assertIsNone(M.resolve_mood({}, None, None))


if __name__ == "__main__":
    unittest.main()
