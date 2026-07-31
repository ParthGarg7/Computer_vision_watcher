/**
 * LogsPage.tsx — full-page live view of the backend log, per layer.
 *
 * Its own route (#/logs), same theme as the dashboard. Polls /api/logs
 * incrementally and renders logs/watcher.log in real time.
 *   • "All live logs" — every line the system is generating right now.
 *   • Layer 1–9       — filter to one pipeline layer and surface that layer's
 *                       condition (ok / warning / error / idle) from its lines.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client";

interface LogLine {
  time: string;
  level: string;
  layer: string;
  msg: string;
}

// Matches "2026-07-31 11:25:01,249 DEBUG    watcher.layer8: message"
const LINE_RE = /^([\d-]+ [\d:,]+)\s+(\w+)\s+([\w.]+):\s?(.*)$/;

function parse(input: string): LogLine {
  const raw = input.replace(/\r+$/, ""); // strip Windows CRLF remnants
  const m = LINE_RE.exec(raw);
  if (!m) return { time: "", level: "INFO", layer: "—", msg: raw };
  const layer = m[3].replace(/^watcher\.?/, "") || "root"; // watcher.layer8 → layer8
  return { time: m[1].slice(11), level: m[2].toUpperCase(), layer, msg: m[4] };
}

const MAX_LINES = 2000;
const LAYERS = [1, 2, 3, 4, 5, 6, 7, 8, 9];
const LAYER_NAMES: Record<number, string> = {
  1: "Ingestion", 2: "Preprocess", 3: "Detection", 4: "Identity", 5: "Expression",
  6: "Analytics", 7: "Storage", 8: "API", 9: "Dashboard",
};

/** Does a log line belong to pipeline layer N? (name is "layer4", "layer4.tracker", …) */
function inLayer(line: LogLine, n: number): boolean {
  return line.layer === `layer${n}` || line.layer.startsWith(`layer${n}.`);
}

type Condition = "ok" | "warn" | "error" | "idle";
function conditionOf(lines: LogLine[]): Condition {
  if (lines.length === 0) return "idle";
  let worst: Condition = "ok";
  for (const l of lines.slice(-250)) {
    if (l.level === "ERROR" || l.level === "CRITICAL") return "error";
    if (l.level === "WARNING") worst = "warn";
  }
  return worst;
}

const COND_LABEL: Record<Condition, string> = {
  ok: "active", warn: "warnings", error: "errors", idle: "idle",
};

export function LogsPage({
  theme, onTheme, onBack,
}: {
  theme: "dark" | "light";
  onTheme: () => void;
  onBack: () => void;
}) {
  const [lines, setLines] = useState<LogLine[]>([]);
  const [sel, setSel] = useState<"all" | number>("all");
  const [clock, setClock] = useState(() => new Date().toLocaleTimeString());
  const cursorRef = useRef(-1);
  const bodyRef = useRef<HTMLDivElement>(null);
  const stick = useRef(true); // stay pinned to newest line unless scrolled up

  useEffect(() => {
    const id = setInterval(() => setClock(new Date().toLocaleTimeString()), 1000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const r = await api.logs(cursorRef.current);
        cursorRef.current = r.cursor;
        if (!cancelled && r.lines.length) {
          setLines((prev) => {
            const next = prev.concat(r.lines.map(parse));
            return next.length > MAX_LINES ? next.slice(next.length - MAX_LINES) : next;
          });
        }
      } catch {
        /* API not up yet — keep polling */
      }
    };
    poll();
    const id = setInterval(poll, 1500);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  // Auto-scroll to newest while pinned to the bottom.
  useEffect(() => {
    const el = bodyRef.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [lines, sel]);

  const perLayer = useMemo(() => {
    const map = new Map<number, LogLine[]>();
    for (const n of LAYERS) map.set(n, []);
    for (const l of lines) for (const n of LAYERS) if (inLayer(l, n)) map.get(n)!.push(l);
    return map;
  }, [lines]);

  const shown = sel === "all" ? lines : (perLayer.get(sel) ?? []);
  const selCondition: Condition = sel === "all"
    ? conditionOf(lines)
    : conditionOf(perLayer.get(sel) ?? []);

  return (
    <>
      <div id="pillar-bg" aria-hidden />
      <div id="bggrid" />

      <div id="app">
        <div id="topbar">
          <div id="brand">
            <span className="dot" /> THE WATCHER <small>· backend logs</small>
          </div>
          <button className="logbtn" style={{ marginLeft: "auto" }} onClick={onBack}
            title="Back to the dashboard">← dashboard</button>
          <div className="switcher">
            <button onClick={onTheme} title="Toggle theme">
              {theme === "dark" ? "◐ dark" : "◑ light"}
            </button>
          </div>
          <div className="clock">{clock}</div>
        </div>

        <div className="logs-page">
          <aside className="logs-side">
            <button className={`logs-tab${sel === "all" ? " on" : ""}`} onClick={() => setSel("all")}>
              <span className={`cond cond-${conditionOf(lines)}`} />
              <span className="nm">All live logs</span>
              <span className="ct">{lines.length}</span>
            </button>
            {LAYERS.map((n) => {
              const ll = perLayer.get(n) ?? [];
              return (
                <button key={n} className={`logs-tab${sel === n ? " on" : ""}`} onClick={() => setSel(n)}>
                  <span className={`cond cond-${conditionOf(ll)}`} />
                  <span className="nm">Layer {n} · {LAYER_NAMES[n]}</span>
                  <span className="ct">{ll.length}</span>
                </button>
              );
            })}
          </aside>

          <section className="logs-main">
            <div className="logs-main-head">
              <span className="ttl">
                {sel === "all"
                  ? "ALL LIVE LOGS"
                  : `LAYER ${sel} · ${LAYER_NAMES[sel as number].toUpperCase()}`}
              </span>
              <span className={`cond-pill cond-${selCondition}`}>{COND_LABEL[selCondition]}</span>
              <span className="logs-count">{shown.length} lines</span>
            </div>
            <div
              className="logs-stream"
              ref={bodyRef}
              onScroll={(e) => {
                const el = e.currentTarget;
                stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
              }}
            >
              {shown.length === 0 ? (
                <div className="logcon-empty">
                  {sel === "all"
                    ? "no log lines yet — start the program (or the API) to see live backend logs"
                    : `no logs from Layer ${sel} yet`}
                </div>
              ) : (
                shown.map((l, i) => (
                  <div className={`logln lv-${l.level.toLowerCase()}`} key={i}>
                    <span className="t">{l.time}</span>
                    <span className="lv">{l.level}</span>
                    <span className="ly">{l.layer}</span>
                    <span className="mg">{l.msg}</span>
                  </div>
                ))
              )}
            </div>
          </section>
        </div>
      </div>
    </>
  );
}
