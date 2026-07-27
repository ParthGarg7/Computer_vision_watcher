/**
 * Panel.tsx — the glass card every widget sits in, plus the small shared
 * state views (loading / empty / error) so every panel handles a missing
 * backend the same way.
 */

import type { ReactNode, CSSProperties } from "react";

interface PanelProps {
  title?: ReactNode;
  count?: ReactNode;
  className?: string;
  style?: CSSProperties;
  /** Removes body padding — for tables and canvases that fill the panel. */
  tight?: boolean;
  children: ReactNode;
}

export function Panel({ title, count, className = "", style, tight, children }: PanelProps) {
  return (
    <div className={`panel ${className}`} style={style}>
      {title !== undefined && (
        <h3>
          {title}
          {count !== undefined && <span className="count">{count}</span>}
        </h3>
      )}
      <div className={`body${tight ? " tight" : ""}`}>{children}</div>
    </div>
  );
}

export function Loading({ label = "loading…" }: { label?: string }) {
  return (
    <div className="state">
      <div className="skeleton" style={{ width: "70%" }} />
      <div className="skeleton" style={{ width: "45%" }} />
      <div>{label}</div>
    </div>
  );
}

export function Empty({ label, hint }: { label: string; hint?: string }) {
  return (
    <div className="state">
      <div>{label}</div>
      {hint && <div className="hint">{hint}</div>}
    </div>
  );
}

export function ErrorState({ error }: { error: Error }) {
  // 503 means "the pipeline hasn't written anything yet" — that is an empty
  // state, not a failure, and saying "error" would be misleading.
  const isNoData = /no database|503/i.test(error.message);
  if (isNoData) {
    return <Empty label="no data yet" hint="run the pipeline: python main.py" />;
  }
  return (
    <div className="state bad">
      <div>API unreachable</div>
      <div className="hint">{error.message}</div>
    </div>
  );
}
