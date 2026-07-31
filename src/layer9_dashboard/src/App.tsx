/**
 * App.tsx — router + dashboard shell.
 *
 * A tiny hash router (no dependency, so nothing to npm-install offline) picks
 * between two pages that share one theme:
 *   #/       → the focus cockpit dashboard (Dashboard)
 *   #/logs   → the full-page live backend logs (LogsPage)
 *
 * The dashboard owns all data fetching so every panel shares one set of
 * requests, plus the small UI state: selected subject, 3D-head look, camera
 * expand, and the Tweaks panel.
 */

import { useEffect, useMemo, useState } from "react";
import { api } from "./api/client";
import { usePoll, useLive } from "./api/hooks";
import { Panel } from "./components/Panel";
import MoodHead, { type HeadLook } from "./components/MoodHead";
import LightPillar from "./components/LightPillar";
import { TopBar, TweaksPanel } from "./components/chrome";
import { LogsPage } from "./components/LogsPage";
import {
  PresentNow, EmotionBars, TrendChart, ValenceArousalPad, CameraFeed, fmtDur,
} from "./components/panels";
import type { LivePerson } from "./types";
import { MOOD_HINT, moodColor } from "./mood";

const DEFAULT_LOOK: HeadLook = {
  speed: 0.28,
  glow: 0.9,
  wire: 0.55,
  solid: false,
  colorSource: "mood",
};

/** Minimal hash route: "" = dashboard, "logs" = logs page. */
function useHashRoute(): string {
  const read = () => window.location.hash.replace(/^#\/?/, "");
  const [route, setRoute] = useState(read);
  useEffect(() => {
    const on = () => setRoute(read());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return route;
}

export default function App() {
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  const route = useHashRoute();
  const toggleTheme = () => setTheme((t) => (t === "dark" ? "light" : "dark"));

  if (route === "logs") {
    return (
      <LogsPage
        theme={theme}
        onTheme={toggleTheme}
        onBack={() => { window.location.hash = "#/"; }}
      />
    );
  }
  return <Dashboard theme={theme} onTheme={toggleTheme} />;
}

function Dashboard({ theme, onTheme }: { theme: "dark" | "light"; onTheme: () => void }) {
  const [look, setLook] = useState<HeadLook>(DEFAULT_LOOK);
  const [selectedTrack, setSelectedTrack] = useState<number | null>(null);
  const [camOpen, setCamOpen] = useState(false);
  const [tweaksOpen, setTweaksOpen] = useState(false);

  // ── data ────────────────────────────────────────────────────────────────
  const live = useLive();
  const events = usePoll((s) => api.recentEvents(60, s), 6000);

  // ── pipeline control ──────────────────────────────────────────────────────
  const pipeline = usePoll((s) => api.pipelineStatus(s), 4000);
  const [pipelineBusy, setPipelineBusy] = useState(false);
  const pipelineRunning = pipeline.data?.running ?? false;

  const togglePipeline = async (start: boolean) => {
    setPipelineBusy(true);
    try {
      await (start ? api.startPipeline() : api.stopPipeline());
      pipeline.refresh();
    } catch (err) {
      console.error("pipeline control failed", err);
    } finally {
      setPipelineBusy(false);
    }
  };

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

  return (
    <>
      {/* React Bits LightPillar behind everything. Rainbow mode sweeps the
          full spectrum (pink→yellow→cyan…) and rotates it over time. */}
      <div id="pillar-bg" aria-hidden>
        <LightPillar
          rainbow
          hueSpeed={0.25}
          hueSpread={0}
          saturation={1.0}
          intensity={0.95}
          rotationSpeed={0.25}
          glowAmount={0.006}
          pillarWidth={3.2}
          pillarHeight={0.4}
          noiseIntensity={0.35}
          mixBlendMode="screen"
        />
      </div>
      <div id="bggrid" />

      <div id="app">
        <TopBar
          theme={theme}
          onTheme={onTheme}
          live={live.connected}
          running={pipelineRunning}
          busy={pipelineBusy}
          onStart={() => togglePipeline(true)}
          onStop={() => togglePipeline(false)}
          onLogs={() => { window.location.hash = "#/logs"; }}
          tweaksOpen={tweaksOpen}
          onTweaks={() => setTweaksOpen((o) => !o)}
        />

        <div id="stage">
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
              <div className="focus-grid">
                <div className="head-stage" style={{ position: "relative" }}>
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
                <div className="exprmix">
                  <div className="exprmix-head">
                    <span>EXPRESSION MIX</span>
                    <span className="who">{subject?.name ?? "—"}</span>
                  </div>
                  <EmotionBars person={subject} bare />
                </div>
              </div>
            </Panel>

            <div className="c-cam" style={{ display: "flex", flexDirection: "column" }}>
              <Panel title="Cam · live" tight style={{ flex: "1 1 auto", minHeight: 0 }}>
                <div style={{ display: "flex", flexDirection: "column", padding: 10, flex: "1 1 auto", minHeight: 0 }}>
                  <CameraFeed
                    people={live.people}
                    streaming={pipelineRunning}
                    onExpand={() => setCamOpen(true)}
                  />
                </div>
              </Panel>
            </div>

            <ValenceArousalPad person={subject} className="c-pad" />

            <TrendChart events={subjectEvents} className="c-chart" />
          </div>
        </div>
      </div>

      <TweaksPanel
        look={look}
        onChange={patchLook}
        open={tweaksOpen}
        onToggle={() => setTweaksOpen((o) => !o)}
      />

      {camOpen && (
        <div className="cam-modal open" onClick={() => setCamOpen(false)}>
          <button className="closex" onClick={() => setCamOpen(false)}>✕ close</button>
          {pipelineRunning ? (
            <img className="cam-video big" src="/api/camera/stream" alt="live camera feed" />
          ) : (
            <div className="state">camera offline — start the program to see the feed</div>
          )}
        </div>
      )}
    </>
  );
}
