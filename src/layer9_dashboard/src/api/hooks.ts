/**
 * api/hooks.ts — React bindings for the Layer 8 API.
 *
 * Two access patterns, matching the two the backend serves:
 *   usePoll  — HTTP request/response for stored history, on an interval
 *   useLive  — a WebSocket carrying Layer 6's real-time events
 *
 * useLive also assembles the raw event stream into "who is on screen right
 * now", which is the question the dashboard actually asks. It falls back to
 * polling /api/events/recent when the socket is unavailable, so the UI still
 * works if the pipeline isn't running.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { api, liveSocketUrl } from "./client";
import type {
  Emotion, LiveEvent, LivePerson, Mood, PresenceAlert, ThresholdAlert,
} from "../types";

export interface PollState<T> {
  data: T | undefined;
  error: Error | undefined;
  loading: boolean;
  refresh: () => void;
}

/**
 * Fetch on mount and every `intervalMs`. In-flight requests are aborted when
 * the interval fires again or the component unmounts, so a slow response can
 * never overwrite a newer one.
 */
export function usePoll<T>(
  fetcher: (signal: AbortSignal) => Promise<T>,
  intervalMs = 5000,
  deps: unknown[] = [],
): PollState<T> {
  const [data, setData] = useState<T>();
  const [error, setError] = useState<Error>();
  const [loading, setLoading] = useState(true);
  const [tick, setTick] = useState(0);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  useEffect(() => {
    const ctrl = new AbortController();
    let cancelled = false;

    const run = async () => {
      try {
        const result = await fetcherRef.current(ctrl.signal);
        if (!cancelled) { setData(result); setError(undefined); }
      } catch (err) {
        if (!cancelled && (err as Error).name !== "AbortError") {
          setError(err as Error);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    run();
    const id = intervalMs > 0 ? setInterval(run, intervalMs) : undefined;
    return () => {
      cancelled = true;
      ctrl.abort();
      if (id) clearInterval(id);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [intervalMs, tick, ...deps]);

  const refresh = useCallback(() => setTick((t) => t + 1), []);
  return { data, error, loading, refresh };
}

export interface LiveState {
  connected: boolean;
  /** People currently on screen, most recently seen first. */
  people: LivePerson[];
  /** Newest-first alert feed (appeared / departed / threshold). */
  alerts: AlertItem[];
  cameraId: string | null;
}

export interface AlertItem {
  id: string;
  timestamp: number;
  kind: "appeared" | "departed" | "threshold";
  who: string;
  note: string;
  level: "ok" | "warn" | "bad";
}

/** A track is dropped from "present now" after this long without an update. */
const STALE_MS = 12_000;
const MAX_ALERTS = 40;

export function useLive(): LiveState {
  const [connected, setConnected] = useState(false);
  const [people, setPeople] = useState<LivePerson[]>([]);
  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const [cameraId, setCameraId] = useState<string | null>(null);
  const tracks = useRef(new Map<number, LivePerson>());

  const commit = useCallback(() => {
    const now = Date.now();
    for (const [id, p] of tracks.current) {
      if (now - p.lastSeen > STALE_MS) tracks.current.delete(id);
    }
    setPeople([...tracks.current.values()].sort((a, b) => b.lastSeen - a.lastSeen));
  }, []);

  const pushAlert = useCallback((a: Omit<AlertItem, "id">) => {
    setAlerts((prev) => [
      { ...a, id: `${a.timestamp}-${a.kind}-${a.who}-${Math.random()}` },
      ...prev,
    ].slice(0, MAX_ALERTS));
  }, []);

  const onEvent = useCallback((ev: LiveEvent) => {
    if (ev.event_type === "live_expression_update") {
      setCameraId(ev.camera_id);
      for (const [tid, t] of Object.entries(ev.tracks)) {
        const trackId = Number(tid);
        const name = t.identity_label ?? "Unidentified";
        tracks.current.set(trackId, {
          trackId,
          name,
          stranger: !t.identity_label,
          mood: (t.mood ?? null) as Mood | null,
          dominant: (t.dominant_expression ?? dominantOf(t.expression_scores)) as Emotion | null,
          scores: t.expression_scores ?? {},
          valence: t.valence ?? null,
          arousal: t.arousal ?? null,
          presentSeconds: t.present_seconds,
          lastSeen: Date.now(),
        });
      }
      commit();
    } else if (ev.event_type === "presence_alert") {
      const e = ev as PresenceAlert;
      const who = e.identity_label ?? `Track ${e.track_id}`;
      if (e.presence === "departed") {
        tracks.current.delete(e.track_id);
        commit();
      }
      pushAlert({
        timestamp: e.timestamp,
        kind: e.presence,
        who,
        note: e.presence === "appeared" ? "entered frame" : "left frame",
        level: e.identity_label ? "ok" : "warn",
      });
    } else if (ev.event_type === "threshold_alert") {
      const e = ev as ThresholdAlert;
      pushAlert({
        timestamp: e.timestamp,
        kind: "threshold",
        who: e.identity_label ?? `Track ${e.track_id}`,
        note: `${e.metric} ${e.value} > ${e.threshold}`,
        level: "bad",
      });
    }
  }, [commit, pushAlert]);

  // ── socket lifecycle, with backoff reconnect ─────────────────────────────
  useEffect(() => {
    let ws: WebSocket | null = null;
    let retry: ReturnType<typeof setTimeout> | undefined;
    let attempt = 0;
    let closed = false;

    const connect = () => {
      if (closed) return;
      try {
        ws = new WebSocket(liveSocketUrl());
      } catch {
        schedule();
        return;
      }
      ws.onopen = () => { attempt = 0; setConnected(true); };
      ws.onmessage = (msg) => {
        try { onEvent(JSON.parse(msg.data) as LiveEvent); } catch { /* ignore malformed */ }
      };
      ws.onclose = () => { setConnected(false); schedule(); };
      ws.onerror = () => ws?.close();
    };

    const schedule = () => {
      if (closed) return;
      // 1s, 2s, 4s … capped at 15s — the pipeline may simply not be running.
      const delay = Math.min(1000 * 2 ** attempt++, 15_000);
      retry = setTimeout(connect, delay);
    };

    connect();
    const sweep = setInterval(commit, 3000);   // expire stale tracks
    return () => {
      closed = true;
      clearInterval(sweep);
      if (retry) clearTimeout(retry);
      ws?.close();
    };
  }, [onEvent, commit]);

  // ── fallback: if the socket never connects, show the last stored readings
  const { data: recent } = usePoll(
    (s) => api.recentEvents(12, s),
    connected ? 0 : 6000,
    [connected],
  );

  useEffect(() => {
    if (connected || !recent?.length) return;
    const seen = new Map<number, LivePerson>();
    for (const e of recent) {
      if (seen.has(e.track_id)) continue;
      seen.set(e.track_id, {
        trackId: e.track_id,
        name: e.identity_label ?? "Unidentified",
        stranger: !e.identity_label,
        mood: e.mood,
        dominant: e.dominant_expression,
        scores: {},
        valence: e.valence,
        arousal: e.arousal,
        presentSeconds: 0,
        lastSeen: Date.now(),
      });
    }
    setPeople([...seen.values()]);
  }, [recent, connected]);

  return { connected, people, alerts, cameraId };
}

function dominantOf(scores: Record<string, number> | undefined): Emotion | null {
  if (!scores) return null;
  let best: string | null = null;
  let bestV = -Infinity;
  for (const [k, v] of Object.entries(scores)) {
    if (v > bestV) { bestV = v; best = k; }
  }
  return best as Emotion | null;
}
