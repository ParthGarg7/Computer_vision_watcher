/**
 * App.tsx — the dashboard shell.
 *
 * Owns all data fetching (so a panel used in two layouts doesn't double its
 * requests) and the small amount of UI state: which layout, which theme,
 * which subject is selected, and the 3D head's look settings.
 *
 * Three layouts, switchable from the top bar:
 *   A · camera   — live feed leading, subject + charts alongside
 *   B · analytics— stats and history leading, camera docked slim
 *   C · focus    — the subject head centre stage
 */

import { useEffect, useMemo, useState } from "react";
import { api } from "./api/client";
import { usePoll, useLive } from "./api/hooks";
import { Panel } from "./components/Panel";
import MoodHead, { type HeadLook } from "./components/MoodHead";
import { TopBar, BackgroundFX, TweaksPanel } from "./components/chrome";
import {
  PresentNow, StatsPanel, SessionsPanel, EmotionBars, TrendChart,
  ValenceArousalPad, AlertFeed, CameraFeed, fmtDur,
} from "./components/panels";
import type { LivePerson } from "./types";
import { MOOD_HINT, moodColor } from "./mood";

type LayoutId = "A" | "B" | "C";

const DEFAULT_LOOK: HeadLook = {
  speed: 0.28,
  glow: 0.9,
  wire: 0.55,
  solid: false,
  colorSource: "mood",
};

export default function App() {
  const [layout, setLayout] = useState<LayoutId>("A");
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  const [look, setLook] = useState<HeadLook>(DEFAULT_LOOK);
  const [selectedTrack, setSelectedTrack] = useState<number | null>(null);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  // ── data ────────────────────────────────────────────────────────────────
  const live = useLive();
  const stats = usePoll((s) => api.stats(s), 8000);
  const sessions = usePoll((s) => api.sessions(12, s), 8000);
  const events = usePoll((s) => api.recentEvents(60, s), 6000);

  // Keep a valid selection as tracks come and go.
  useEffect(() => {
    if (live.people.length === 0) { setSelectedTrack(null); return; }
    if (selectedTrack == null || !live.people.some((p) => p.trackId === selectedTrack)) {
      setSelectedTrack(live.people[0].trackId);
    }
  }, [live.people, selectedTrack]);

  const subject: LivePerson | null = useMemo(
    () => live.people.find((p) => p.trackId === selectedTrack) ?? live.people[0] ?? null,
    [live.people, selectedTrack],
  );

  const subjectEvents = useMemo(
    () => (events.data ?? []).filter((e) => !subject || e.track_id === subject.trackId),
    [events.data, subject],
  );

  const patchLook = (p: Partial<HeadLook>) => setLook((l) => ({ ...l, ...p }));

  // ── the subject head, reused by every layout ───────────────────────────
  const headPanel = (className?: string, showTitle = true) => (
    <Panel
      title={showTitle ? "Subject · 3D mood" : undefined}
      count={showTitle ? (subject?.name ?? "—") : undefined}
      className={className}
      tight
    >
      <div className="head-stage" style={{ position: "relative", flex: "1 1 auto" }}>
        <MoodHead mood={subject?.mood ?? null} look={look} />
        <div className="mood-label">
          {subject ? (
            <>
              <b style={{ color: moodColor(subject.mood) }}>{subject.mood ?? "—"}</b>
              {subject.mood && <> · {MOOD_HINT[subject.mood]}</>}
              {subject.valence != null && (
                <> · v {subject.valence.toFixed(2)} / a {subject.arousal?.toFixed(2)}</>
              )}
            </>
          ) : "no subject on camera"}
        </div>
      </div>
    </Panel>
  );

  const camPanel = (title: string, mini: boolean, className?: string) => (
    <Panel title={title} className={className} tight>
      <div style={{ display: "flex", flexDirection: "column", padding: 10, flex: "1 1 auto", minHeight: 0 }}>
        <CameraFeed people={live.people} mini={mini} />
      </div>
    </Panel>
  );

  return (
    <>
      <BackgroundFX />
      <div id="app">
        <TopBar
          layout={layout} onLayout={setLayout}
          theme={theme} onTheme={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
          live={live.connected}
        />

        <div id="stage">
          {layout === "A" && (
            <div className="layout A">
              {camPanel("Live · camera", false, "a-cam")}
              <PresentNow
                people={live.people} selected={selectedTrack}
                onSelect={setSelectedTrack} className="a-now"
              />
              {headPanel("a-head")}
              <Panel title="Expression" count={subject?.name ?? ""} className="a-chart">
                <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                  <EmotionBars person={subject} bare />
                  <TrendChart events={subjectEvents} bare />
                </div>
              </Panel>
              <SessionsPanel
                sessions={sessions.data} error={sessions.error}
                loading={sessions.loading} className="a-sessions"
              />
            </div>
          )}

          {layout === "B" && (
            <div className="layout B">
              <StatsPanel
                stats={stats.data} error={stats.error}
                loading={stats.loading} className="b-stats"
              />
              <SessionsPanel
                sessions={sessions.data} error={sessions.error}
                loading={sessions.loading} className="b-sessions"
              />
              {headPanel("b-focus")}
              <TrendChart events={subjectEvents} className="b-chart" />
              <Panel title="Cam" className="b-cam" tight>
                <div style={{ display: "flex", flexDirection: "column", gap: 8, padding: 8 }}>
                  <CameraFeed people={live.people} mini />
                  {live.people.map((p) => (
                    <div
                      key={p.trackId}
                      className={`person${p.trackId === selectedTrack ? " sel" : ""}`}
                      onClick={() => setSelectedTrack(p.trackId)}
                    >
                      <div className="avatar">{p.name === "Unidentified" ? "??" : p.name[0]}</div>
                      <div className="meta">
                        <div className="nm" style={{ fontSize: 12 }}>{p.name.split(" ")[0]}</div>
                      </div>
                      {p.mood && (
                        <span className="chip mood" style={{ background: moodColor(p.mood) }}>
                          {p.mood}
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              </Panel>
            </div>
          )}

          {layout === "C" && (
            <div className="layout C">
              <PresentNow
                people={live.people} selected={selectedTrack}
                onSelect={setSelectedTrack} className="c-now"
              />
              <Panel
                title="Subject focus"
                count={subject ? `${subject.name} · ${fmtDur(subject.presentSeconds)}` : ""}
                className="c-focus"
              >
                <div style={{ display: "grid", gridTemplateColumns: "1.1fr 1fr", gap: 16, height: "100%", minHeight: 0 }}>
                  <div className="head-stage" style={{ position: "relative" }}>
                    <MoodHead mood={subject?.mood ?? null} look={look} />
                    <div className="mood-label">
                      <b style={{ color: moodColor(subject?.mood) }}>{subject?.mood ?? "—"}</b>
                    </div>
                  </div>
                  <div style={{ overflow: "auto" }}>
                    <EmotionBars person={subject} bare />
                  </div>
                </div>
              </Panel>
              {camPanel("Cam", true, "c-cam")}
              <ValenceArousalPad person={subject} className="c-pad" />
              <AlertFeed alerts={live.alerts} className="c-chart" />
            </div>
          )}
        </div>
      </div>

      <TweaksPanel look={look} onChange={patchLook} />
    </>
  );
}
