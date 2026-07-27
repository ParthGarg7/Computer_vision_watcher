/**
 * mood.ts — presentation metadata for the 9 mood states and 8 emotions.
 *
 * The states themselves are defined by the pipeline (Layer 5's mood.py); this
 * file only decides how they LOOK. Kept in one place so the head, the chips,
 * the bars and the valence/arousal pad always agree on a colour.
 */

import type { Emotion, Mood } from "./types";

/** CSS colour per mood, for chips and labels. */
export const MOOD_COLOR: Record<Mood, string> = {
  content:    "var(--good)",
  pleased:    "var(--acc-teal, var(--good))",
  excited:    "var(--acc-amber)",
  relaxed:    "var(--acc-cyan)",
  neutral:    "var(--acc-blue)",
  alert:      "var(--acc-violet)",
  subdued:    "var(--ink-faint)",
  displeased: "var(--warn)",
  tense:      "var(--bad)",
};

/** RGB 0..1 per mood, for the WebGL head. */
export const MOOD_RGB: Record<Mood, [number, number, number]> = {
  content:    [0.35, 0.92, 0.68],
  pleased:    [0.36, 0.90, 0.82],
  excited:    [1.00, 0.74, 0.32],
  relaxed:    [0.34, 0.84, 1.00],
  neutral:    [0.45, 0.65, 1.00],
  alert:      [0.68, 0.50, 1.00],
  subdued:    [0.58, 0.62, 0.72],
  displeased: [1.00, 0.62, 0.28],
  tense:      [1.00, 0.36, 0.34],
};

/** One-line explanation of each mood, shown as a tooltip. */
export const MOOD_HINT: Record<Mood, string> = {
  content:    "pleasant · calm",
  pleased:    "pleasant",
  excited:    "pleasant · activated",
  relaxed:    "calm",
  neutral:    "middle of both dimensions",
  alert:      "activated",
  subdued:    "unpleasant · calm",
  displeased: "unpleasant",
  tense:      "unpleasant · activated",
};

export const EMOTION_COLOR: Record<Emotion, string> = {
  happiness: "var(--good)",
  neutral:   "var(--acc-blue)",
  surprise:  "var(--acc-cyan)",
  contempt:  "var(--acc-violet)",
  sadness:   "var(--ink-faint)",
  fear:      "var(--acc-amber)",
  disgust:   "var(--warn)",
  anger:     "var(--bad)",
};

export const moodColor = (m: Mood | null | undefined): string =>
  m ? MOOD_COLOR[m] ?? "var(--ink-dim)" : "var(--ink-dim)";

export const emotionColor = (e: Emotion | null | undefined): string =>
  e ? EMOTION_COLOR[e] ?? "var(--ink-dim)" : "var(--ink-dim)";
