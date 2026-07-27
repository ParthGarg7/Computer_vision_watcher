/**
 * api/client.ts — the ONLY place that talks to the Layer 8 server.
 *
 * Everything the dashboard knows about the backend lives here: URLs, fetch
 * mechanics, and error shape. Components consume the hooks in api/hooks.ts
 * and never touch fetch directly, so swapping transport (or pointing at a
 * different host) is a change to this file alone.
 */

import type {
  Health, Stats, Session, ExpressionEvent, Identity, TrendBucket,
} from "../types";

/** Same-origin in production (FastAPI serves the bundle); proxied in dev. */
const BASE = "";

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
    this.name = "ApiError";
  }
}

async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { signal });
  if (!res.ok) {
    // 503 is the expected "pipeline hasn't produced data yet" case; the UI
    // shows an empty state for it rather than an error.
    let detail = res.statusText;
    try {
      detail = (await res.json())?.detail ?? detail;
    } catch {
      /* non-JSON error body — keep statusText */
    }
    throw new ApiError(detail, res.status);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: (s?: AbortSignal) => get<Health>("/api/health", s),

  stats: (s?: AbortSignal) => get<Stats>("/api/stats", s),

  sessions: (limit = 20, s?: AbortSignal) =>
    get<{ sessions: Session[] }>(`/api/sessions?limit=${limit}`, s)
      .then((r) => r.sessions),

  session: (id: string, s?: AbortSignal) =>
    get<Session>(`/api/sessions/${encodeURIComponent(id)}`, s),

  expressionTrend: (id: string, bucket = 30, s?: AbortSignal) =>
    get<{ trend: TrendBucket[] }>(
      `/api/sessions/${encodeURIComponent(id)}/expression_trend?bucket=${bucket}`, s,
    ).then((r) => r.trend),

  identities: (s?: AbortSignal) =>
    get<{ identities: Identity[] }>("/api/identities", s)
      .then((r) => r.identities),

  identityHistory: (label: string, s?: AbortSignal) =>
    get<{ identity: string; sessions: number; total_present_seconds: number; history: Session[] }>(
      `/api/identities/${encodeURIComponent(label)}/history`, s,
    ),

  recentEvents: (limit = 50, s?: AbortSignal) =>
    get<{ events: ExpressionEvent[] }>(`/api/events/recent?limit=${limit}`, s)
      .then((r) => r.events),
};

/** WebSocket URL for the live channel, derived from the current origin. */
export function liveSocketUrl(): string {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}/ws/live`;
}
