/**
 * types.ts — the Layer 8 API contract, in TypeScript.
 *
 * These mirror the JSON the FastAPI server returns (src/layer8_api/api.py).
 * If an endpoint changes shape, change it here first — every component reads
 * its data through these types, so the compiler will point at everything
 * that needs updating.
 */

/** The 8 AffectNet classes, in the model's own output order. */
export const EMOTIONS = [
  "anger", "contempt", "disgust", "fear",
  "happiness", "neutral", "sadness", "surprise",
] as const;
export type Emotion = (typeof EMOTIONS)[number];

/** The 9 mood states from the valence/arousal grid (Layer 5 mood.py). */
export const MOODS = [
  "content", "pleased", "excited",
  "relaxed", "neutral", "alert",
  "subdued", "displeased", "tense",
] as const;
export type Mood = (typeof MOODS)[number];

export type ExpressionScores = Partial<Record<Emotion, number>>;

/** One continuous stretch of a person being on screen. */
export interface Appearance {
  started: number;
  ended: number;
  duration_seconds: number;
  frames_observed: number;
}

/** One person, for one run of the pipeline. Contains many appearances. */
export interface Session {
  session_id: string;
  run_id?: string;
  camera_id: string;
  track_id: number;
  person_id: string | null;
  identity_label: string | null;
  session_start: number;
  session_end: number | null;
  total_present_seconds: number | null;
  appearance_count: number | null;
  frames_observed: number | null;
  expression_trend: ExpressionScores;
  dominant_expression_distribution: Partial<Record<Emotion, number>>;
  appearances?: Appearance[];
  presence_events?: { timestamp: number; event_type: string }[];
}

/** One expression measurement (only fresh inferences are recorded). */
export interface ExpressionEvent {
  timestamp: number;
  track_id: number;
  identity_label: string | null;
  dominant_expression: Emotion;
  confidence: number;
  valence: number | null;
  arousal: number | null;
  mood: Mood | null;
}

export interface PersonStat {
  person: string;
  sessions: number;
  appearances: number | null;
  present_seconds: number | null;
  frames: number | null;
}

export interface Stats {
  people: PersonStat[];
  data_range: { first: number | null; last: number | null };
}

export interface Health {
  status: string;
  database: {
    path: string;
    exists: boolean;
    rows: Partial<Record<
      "sessions" | "appearances" | "expression_events" | "presence_events",
      number
    >>;
  };
  registry: { path: string; exists: boolean };
}

export interface Identity {
  person_id: string;
  name: string | null;
  samples: number;
}

export interface TrendBucket {
  bucket_start: number;
  dominant_expression: Emotion;
  measurements: number;
  avg_confidence: number;
  avg_valence: number | null;
  avg_arousal: number | null;
}

/* ── Live WebSocket messages (pushed by Layer 6 through Layer 8) ─────────── */

export interface LiveTrack {
  identity_label: string | null;
  expression_scores: ExpressionScores;
  present_seconds: number;
  dominant_expression?: Emotion | null;
  mood?: Mood | null;
  valence?: number | null;
  arousal?: number | null;
}

export interface LiveExpressionUpdate {
  event_type: "live_expression_update";
  timestamp: number;
  camera_id: string;
  tracks: Record<string, LiveTrack>;
}

export interface PresenceAlert {
  event_type: "presence_alert";
  presence: "appeared" | "departed";
  timestamp: number;
  camera_id: string;
  track_id: number;
  identity_label: string | null;
}

export interface ThresholdAlert {
  event_type: "threshold_alert";
  timestamp: number;
  camera_id: string;
  track_id: number;
  identity_label: string | null;
  metric: string;
  value: number;
  threshold: number;
}

export type LiveEvent =
  | LiveExpressionUpdate
  | PresenceAlert
  | ThresholdAlert;

/** A person currently on screen, assembled from live events. */
export interface LivePerson {
  trackId: number;
  name: string;
  stranger: boolean;
  mood: Mood | null;
  dominant: Emotion | null;
  scores: ExpressionScores;
  valence: number | null;
  arousal: number | null;
  presentSeconds: number;
  lastSeen: number;
}
