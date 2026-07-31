/**
 * panels.tsx — the dashboard's data widgets.
 *
 * Each one is a small, self-contained component that takes already-fetched
 * data as props (fetching happens in App.tsx, so a panel can appear in more
 * than one layout without duplicating requests).
 *
 * Contents: PresentNow, StatsPanel, SessionsPanel, EmotionBars, TrendChart,
 * ValenceArousalPad, AlertFeed, CameraFeed.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { Panel, Loading, Empty, ErrorState } from "./Panel";
import { emotionColor, moodColor, MOOD_HINT } from "../mood";
import { EMOTIONS } from "../types";
import type {
  Emotion, ExpressionEvent, LivePerson, Session, Stats,
} from "../types";
import type { AlertItem } from "../api/hooks";

/* ── helpers ─────────────────────────────────────────────────────────────── */

export const initials = (n: string) =>
  n === "Unidentified" ? "??" : n.split(" ").map((w) => w[0]).slice(0, 2).join("").toUpperCase();

export const fmtDur = (s: number | null | undefined) => {
  if (s == null) return "–";
  if (s < 60) return `${s.toFixed(0)}s`;
  const m = Math.floor(s / 60);
  return `${m}m ${String(Math.round(s % 60)).padStart(2, "0")}s`;
};

export const fmtTime = (epoch: number | null | undefined) =>
  epoch ? new Date(epoch * 1000).toLocaleTimeString() : "–";

/* ── Present now ─────────────────────────────────────────────────────────── */

export function PresentNow({
  people, selected, onSelect, className,
}: {
  people: LivePerson[];
  selected: number | null;
  onSelect: (trackId: number) => void;
  className?: string;
}) {
  return (
    <Panel title="Present now" count={`${people.length} tracked`} className={className}>
      {people.length === 0 ? (
        <Empty label="nobody on camera" hint="start the pipeline to see live tracks" />
      ) : (
        <div className="now-list">
          {people.map((p) => (
            <div
              key={p.trackId}
              className={`person${p.trackId === selected ? " sel" : ""}`}
              onClick={() => onSelect(p.trackId)}
            >
              <div className="avatar">{initials(p.name)}</div>
              <div className="meta">
                <div className="nm">
                  {p.name}
                  <span className="chip">ID {p.trackId}</span>
                </div>
                <div className="sub">
                  {p.dominant ?? "—"} · {fmtDur(p.presentSeconds)}
                </div>
              </div>
              {p.mood && (
                <span
                  className="chip mood"
                  style={{ background: moodColor(p.mood) }}
                  title={MOOD_HINT[p.mood]}
                >
                  {p.mood}
                </span>
              )}
            </div>
          ))}
        </div>
      )}
    </Panel>
  );
}

/* ── Stats tiles ─────────────────────────────────────────────────────────── */

export function StatsPanel({
  stats, error, loading, className,
}: {
  stats?: Stats; error?: Error; loading: boolean; className?: string;
}) {
  const tiles = useMemo(() => {
    if (!stats) return [];
    const people = stats.people;
    const known = people.filter((p) => p.person !== "(unidentified)");
    const totalPresent = people.reduce((s, p) => s + (p.present_seconds ?? 0), 0);
    const totalSessions = people.reduce((s, p) => s + p.sessions, 0);
    const totalAppearances = people.reduce((s, p) => s + (p.appearances ?? 0), 0);
    return [
      ["Sessions", totalSessions],
      ["People", known.length],
      ["Appearances", totalAppearances],
      ["Total present", fmtDur(totalPresent)],
    ] as [string, string | number][];
  }, [stats]);

  return (
    <Panel title="Overview" className={className}>
      {error ? <ErrorState error={error} />
        : loading ? <Loading />
        : tiles.length === 0 ? <Empty label="no sessions recorded" />
        : (
          <div className="stats">
            {tiles.map(([k, v]) => (
              <div className="stat" key={k}>
                <div className="v">{v}</div>
                <div className="k">{k}</div>
              </div>
            ))}
          </div>
        )}
    </Panel>
  );
}

/* ── Sessions table ──────────────────────────────────────────────────────── */

export function SessionsPanel({
  sessions, error, loading, className,
}: {
  sessions?: Session[]; error?: Error; loading: boolean; className?: string;
}) {
  return (
    <Panel
      title="Sessions"
      count={sessions ? sessions.length : ""}
      className={className}
      tight
    >
      {error ? <ErrorState error={error} />
        : loading ? <Loading />
        : !sessions?.length ? <Empty label="no sessions recorded" />
        : (
          <table>
            <thead>
              <tr>
                <th className="mono">Started</th>
                <th>Person</th>
                <th className="mono">Present</th>
                <th className="mono">Visits</th>
                <th>Top expression</th>
              </tr>
            </thead>
            <tbody>
              {sessions.map((s) => {
                const dist = s.dominant_expression_distribution ?? {};
                const top = Object.entries(dist).sort((a, b) => b[1] - a[1])[0];
                return (
                  <tr key={s.session_id}>
                    <td className="mono">{fmtTime(s.session_start)}</td>
                    <td>{s.identity_label ?? <span style={{ color: "var(--ink-faint)" }}>unidentified</span>}</td>
                    <td className="mono">{fmtDur(s.total_present_seconds)}</td>
                    <td className="mono">{s.appearance_count ?? 0}</td>
                    <td>
                      {top ? (
                        <span className="chip" style={{ borderColor: emotionColor(top[0] as Emotion) }}>
                          {top[0]} ×{top[1]}
                        </span>
                      ) : "–"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
    </Panel>
  );
}

/* ── Emotion bars ────────────────────────────────────────────────────────── */

export function EmotionBars({ person, bare }: { person: LivePerson | null; bare?: boolean }) {
  const scores = person?.scores ?? {};

  // Always render every emotion row. With no subject / no reading the bars sit
  // at 0% (null values) rather than showing an empty state, so the Expression
  // Mix stays present even when the pipeline is off — matches the reference.
  const body = (
    <div className="ebars">
      {EMOTIONS.map((e) => {
        const v = (scores[e] ?? 0) * 100;
        return (
          <div className="ebar" key={e}>
            <div className="lab">{e}</div>
            <div className="track">
              <div className="fill" style={{ width: `${v}%`, background: emotionColor(e) }} />
            </div>
            <div className="pct">{v.toFixed(0)}%</div>
          </div>
        );
      })}
    </div>
  );

  if (bare) return body;
  return <Panel title="Expression mix" count={person?.name ?? ""}>{body}</Panel>;
}

/* ── Trend chart (canvas, no chart library) ──────────────────────────────── */

export function TrendChart({
  events, bare, className,
}: {
  events?: ExpressionEvent[]; bare?: boolean; className?: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const cv = ref.current;
    if (!cv || !events?.length) return;
    const dpr = Math.min(devicePixelRatio, 2);
    const w = cv.clientWidth || 300;
    const h = 170;
    cv.width = w * dpr; cv.height = h * dpr;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    const css = getComputedStyle(document.documentElement);
    const line = css.getPropertyValue("--line").trim() || "#334";
    const dim = css.getPropertyValue("--ink-faint").trim() || "#889";

    // oldest -> newest along x
    const data = [...events].reverse();
    const pad = { l: 30, r: 8, t: 10, b: 18 };
    const cw = w - pad.l - pad.r;
    const ch = h - pad.t - pad.b;

    // axes: valence/arousal live in -1..1, confidence in 0..1
    ctx.strokeStyle = line; ctx.lineWidth = 1;
    ctx.beginPath();
    [0, 0.5, 1].forEach((f) => {
      const y = pad.t + ch * f;
      ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + cw, y);
    });
    ctx.stroke();
    ctx.fillStyle = dim; ctx.font = "10px ui-monospace, monospace";
    ctx.fillText("+1", 6, pad.t + 4);
    ctx.fillText(" 0", 6, pad.t + ch / 2 + 3);
    ctx.fillText("-1", 6, pad.t + ch + 3);

    const plot = (key: "valence" | "arousal", colour: string) => {
      ctx.strokeStyle = colour; ctx.lineWidth = 1.8;
      ctx.beginPath();
      let started = false;
      data.forEach((e, i) => {
        const v = e[key];
        if (v == null) return;
        const x = pad.l + (cw * i) / Math.max(data.length - 1, 1);
        const y = pad.t + ch * (1 - (v + 1) / 2);
        if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
      });
      ctx.stroke();
    };
    plot("valence", "#4ade80");
    plot("arousal", "#f0a030");

    // legend
    ctx.font = "10px ui-monospace, monospace";
    ctx.fillStyle = "#4ade80"; ctx.fillText("valence", pad.l + 2, h - 6);
    ctx.fillStyle = "#f0a030"; ctx.fillText("arousal", pad.l + 60, h - 6);
  }, [events]);

  const body = !events?.length
    ? <Empty label="no readings yet" />
    : <canvas ref={ref} style={{ width: "100%", height: 170 }} />;

  if (bare) return body;
  return <Panel title="Valence / arousal · recent" className={className}>{body}</Panel>;
}

/* ── Valence / arousal pad ───────────────────────────────────────────────── */

export function ValenceArousalPad({
  person, className,
}: { person: LivePerson | null; className?: string }) {
  const v = person?.valence ?? null;
  const a = person?.arousal ?? null;
  return (
    <Panel title="Mood space" count={person?.mood ?? ""} className={className}>
      {v == null || a == null ? (
        <Empty label="no dimensional reading" hint="needs the valence/arousal model" />
      ) : (
        <div className="pad">
          <div className="cross h" />
          <div className="cross v" />
          <div className="lab t">excited</div>
          <div className="lab b">calm</div>
          <div className="lab l">unpleasant</div>
          <div className="lab r">pleasant</div>
          <div
            className="pt"
            style={{
              left: `${((v + 1) / 2) * 100}%`,
              top: `${(1 - (a + 1) / 2) * 100}%`,
              background: moodColor(person?.mood),
            }}
          />
        </div>
      )}
    </Panel>
  );
}

/* ── Alert feed ──────────────────────────────────────────────────────────── */

export function AlertFeed({ alerts, className }: { alerts: AlertItem[]; className?: string }) {
  return (
    <Panel title="Events" count={alerts.length || ""} className={className}>
      {alerts.length === 0 ? (
        <Empty label="no events yet" hint="appearances and alerts show up here" />
      ) : (
        <div className="alerts">
          {alerts.map((a) => (
            <div className={`alert ${a.level}`} key={a.id}>
              <span className="t">{fmtTime(a.timestamp)}</span>
              <span className="who">{a.who}</span>
              <span className="note">{a.note}</span>
            </div>
          ))}
        </div>
      )}
    </Panel>
  );
}

/* ── Camera feed placeholder ─────────────────────────────────────────────── */

export function CameraFeed({
  people, mini, onExpand, streaming,
}: {
  people: LivePerson[]; mini?: boolean; onExpand?: () => void;
  /** Pipeline is running — show the real MJPEG feed instead of the placeholder. */
  streaming?: boolean;
}) {
  // When streaming, mirror the pipeline's annotated frames. The stream 503s
  // until the first frame is written, so on error we retry every 1.5s (with a
  // fresh <img> key to force a reconnect) rather than latching to the empty
  // state forever.
  const [imgError, setImgError] = useState(false);
  const [reconnect, setReconnect] = useState(0);
  const showVideo = !!streaming && !imgError;

  useEffect(() => {
    if (!streaming || !imgError) return;
    const t = setTimeout(() => { setImgError(false); setReconnect((k) => k + 1); }, 1500);
    return () => clearTimeout(t);
  }, [streaming, imgError]);

  useEffect(() => { if (streaming) setImgError(false); }, [streaming]);

  return (
    <div className="cam" style={{ flex: "1 1 auto", minHeight: mini ? 150 : 220 }} onClick={onExpand}>
      <div className={`rec${streaming ? "" : " off"}`}>
        <span className="dot" /> {streaming ? "LIVE" : "OFFLINE"}
      </div>

      {showVideo ? (
        <img
          key={reconnect}
          className="cam-video"
          src="/api/camera/stream"
          alt="live camera feed"
          onError={() => setImgError(true)}
        />
      ) : (
        <div className="placeholder">
          <div>{streaming ? "connecting to camera…" : "camera offline"}</div>
          <div style={{ fontSize: 11, opacity: 0.7 }}>
            {streaming ? (
              <>waiting for the first frame from the pipeline</>
            ) : (
              <>start the program to see the live feed;<br />
                the pipeline also opens its own window</>
            )}
          </div>
        </div>
      )}

      {/* Real detection boxes are already drawn onto the streamed frame; only
          show the decorative overlay boxes when there's no live video. */}
      {!showVideo && people.map((p, i) => (
        <div
          key={p.trackId}
          className={`detbox${p.stranger ? " stranger" : ""}`}
          style={{
            left: `${18 + (i * 27) % 55}%`,
            top: `${24 + (i * 19) % 40}%`,
            width: "22%",
            height: "34%",
          }}
        >
          <span className="tag">{p.name}</span>
        </div>
      ))}
      {onExpand && <div className="expand-hint">click to expand ⤢</div>}
    </div>
  );
}
