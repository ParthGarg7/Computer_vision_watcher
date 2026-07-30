/**
 * App.tsx — the dashboard shell.
 *
 * ONE view: the focus cockpit (the old "C" layout). The subject head sits
 * centre stage; Present Now on the left; camera + valence/arousal on the
 * right; mood-over-time across the bottom. No layout switcher — this is the
 * single intended front end.
 *
 * Owns all data fetching so every panel shares one set of requests, plus the
 * small UI state: theme, which subject is selected, and the 3D head's look.
 */

import { useEffect, useMemo, useState } from "react";
import { api } from "./api/client";
import { usePoll, useLive } from "./api/hooks";
import { Panel } from "./components/Panel";
import MoodHead, { type HeadLook } from "./components/MoodHead";
import LightPillar from "./components/LightPillar";
import { TopBar, TweaksPanel } from "./components/chrome";
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

export default function App() {
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  const [look, setLook] = useState<HeadLook>(DEFAULT_LOOK);
  const [selectedTrack, setSelectedTrack] = useState<number | null>(null);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  // ── data ────────────────────────────────────────────────────────────────
  const live = useLive();
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

  return (
    <>
      {/* React Bits LightPillar behind everything. Tuned to the Watcher
          palette; swap the colours here to re-skin the glow. */}
      <div id="pillar-bg" aria-hidden>
        <LightPillar
          topColor="#4dd8ff"
          bottomColor="#5227FF"
          intensity={0.85}
          rotationSpeed={0.22}
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
          onTheme={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
          live={live.connected}
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
                <div style={{ overflow: "auto", minWidth: 0 }}>
                  <EmotionBars person={subject} bare />
                </div>
              </div>
            </Panel>

            <div className="c-cam" style={{ display: "flex", flexDirection: "column" }}>
              <Panel title="Cam · live" tight style={{ flex: "1 1 auto", minHeight: 0 }}>
                <div style={{ display: "flex", flexDirection: "column", padding: 10, flex: "1 1 auto", minHeight: 0 }}>
                  <CameraFeed people={live.people} />
                </div>
              </Panel>
            </div>

            <ValenceArousalPad person={subject} className="c-pad" />

            <TrendChart events={subjectEvents} className="c-chart" />
          </div>
        </div>
      </div>

      <TweaksPanel look={look} onChange={patchLook} />
    </>
  );
}
