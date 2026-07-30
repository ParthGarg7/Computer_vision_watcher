/**
 * chrome.tsx — the app shell: top bar, animated background, and the Tweaks
 * panel that drives the 3D head's look.
 */

import { useEffect, useRef, useState } from "react";
import type { HeadLook } from "./MoodHead";

/* ── Top bar ─────────────────────────────────────────────────────────────── */

export function TopBar({
  theme, onTheme, live,
}: {
  theme: "dark" | "light";
  onTheme: () => void;
  live: boolean;
}) {
  const [clock, setClock] = useState(() => new Date().toLocaleTimeString());
  useEffect(() => {
    const id = setInterval(() => setClock(new Date().toLocaleTimeString()), 1000);
    return () => clearInterval(id);
  }, []);

  return (
    <div id="topbar">
      <div id="brand">
        <span className="dot" /> THE WATCHER <small>local · offline</small>
      </div>

      <div className={`livechip${live ? " on" : ""}`} style={{ marginLeft: "auto" }} title={
        live ? "receiving live events from the pipeline"
             : "no live connection — showing stored data"
      }>
        <span className="led" />
        {live ? "live" : "offline"}
      </div>

      {/* Single focus view — no layout switcher. Only the theme toggle remains. */}
      <div className="switcher">
        <button onClick={onTheme} title="Toggle theme">
          {theme === "dark" ? "◐ dark" : "◑ light"}
        </button>
      </div>

      <div className="clock">{clock}</div>
    </div>
  );
}

/* ── Background: slow light pillars behind the dot grid ──────────────────── */

export function BackgroundFX() {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;

    let raf = 0;
    const resize = () => {
      cv.width = innerWidth;
      cv.height = innerHeight;
    };
    resize();
    addEventListener("resize", resize);

    const bars = Array.from({ length: 7 }, (_, i) => ({
      x: Math.random(),
      w: 0.05 + Math.random() * 0.13,
      speed: 0.006 + Math.random() * 0.015,
      hue: 220 + i * 12,
      phase: Math.random() * Math.PI * 2,
    }));

    const tick = (t: number) => {
      raf = requestAnimationFrame(tick);
      ctx.clearRect(0, 0, cv.width, cv.height);
      ctx.globalCompositeOperation = "lighter";
      bars.forEach((b) => {
        const x = ((b.x + t * 0.00002 * b.speed * 60) % 1.3 - 0.15) * cv.width;
        const a = 0.05 + 0.04 * Math.sin(t * 0.0004 + b.phase);
        const g = ctx.createLinearGradient(x, 0, x + b.w * cv.width, 0);
        g.addColorStop(0, `hsla(${b.hue},90%,60%,0)`);
        g.addColorStop(0.5, `hsla(${b.hue},90%,62%,${a})`);
        g.addColorStop(1, `hsla(${b.hue},90%,60%,0)`);
        ctx.fillStyle = g;
        ctx.fillRect(x, 0, b.w * cv.width, cv.height);
      });
      ctx.globalCompositeOperation = "source-over";
    };
    raf = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(raf);
      removeEventListener("resize", resize);
    };
  }, []);

  return (
    <>
      <canvas id="bgfx" ref={ref} />
      <div id="bggrid" />
    </>
  );
}

/* ── Tweaks panel (3D head controls) ─────────────────────────────────────── */

export function TweaksPanel({
  look, onChange,
}: {
  look: HeadLook;
  onChange: (patch: Partial<HeadLook>) => void;
}) {
  const [collapsed, setCollapsed] = useState(true);

  const Slider = ({ label, k, min, max, step }: {
    label: string; k: "speed" | "glow" | "wire"; min: number; max: number; step: number;
  }) => (
    <div className="tw-row">
      <div className="lab"><span>{label}</span><span>{look[k].toFixed(2)}</span></div>
      <input
        type="range" min={min} max={max} step={step} value={look[k]}
        onChange={(e) => onChange({ [k]: Number(e.target.value) } as Partial<HeadLook>)}
      />
    </div>
  );

  const Toggle = ({ label, k }: { label: string; k: "solid" }) => (
    <div className="tw-toggle">
      <span>{label}</span>
      <div className={`sw${look[k] ? " on" : ""}`} onClick={() => onChange({ [k]: !look[k] } as Partial<HeadLook>)} />
    </div>
  );

  return (
    <div id="tweaks" className={collapsed ? "collapsed" : ""}>
      <div className="tw-head" onClick={() => setCollapsed((c) => !c)}>
        <span className="gear">⚙</span> Tweaks · 3D head
        <span className="chev">{collapsed ? "▴" : "▾"}</span>
      </div>
      <div className="tw-body">
        <Slider label="spin speed" k="speed" min={0} max={1.2} step={0.02} />
        <Slider label="rim glow" k="glow" min={0} max={2} step={0.05} />
        <Slider label="wire opacity" k="wire" min={0.05} max={1} step={0.05} />
        <Toggle label="lit surface" k="solid" />
        <div className="tw-row">
          <div className="lab"><span>colour</span></div>
          <div className="tw-seg">
            {(["mood", "cyan", "blue", "amber", "violet"] as const).map((c) => (
              <button
                key={c}
                className={look.colorSource === c ? "on" : ""}
                onClick={() => onChange({ colorSource: c })}
              >
                {c}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
