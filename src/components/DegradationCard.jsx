import React from 'react';

const SEVERITY_COLOR = {
  low: 'var(--teal)',
  moderate: 'var(--amber)',
  high: 'var(--accent)',
};

export default function DegradationCard({ data }) {
  if (!data) return <ReadoutSkeleton title="TYRE DEGRADATION" />;

  const { rate_pct_per_lap, severity } = data;
  const pct = Math.min(100, (rate_pct_per_lap / 8) * 100);

  return (
    <div className="readout" style={{ '--bar-color': SEVERITY_COLOR[severity] }}>
      <div className="readout-head">
        <span className="readout-title">TYRE DEGRADATION</span>
        <span className={`readout-badge badge-${severity}`}>{severity}</span>
      </div>
      <div className="readout-value tabular">
        {rate_pct_per_lap.toFixed(2)}<span className="unit">%/lap</span>
      </div>
      <p className="readout-sub">Pace loss rate relative to the stint's first flying lap.</p>
      <div className="readout-bar">
        <div className="readout-bar-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function ReadoutSkeleton({ title }) {
  return (
    <div className="readout">
      <div className="readout-head">
        <span className="readout-title">{title}</span>
      </div>
      <div className="skel" style={{ height: 38, width: '60%', marginBottom: 8 }} />
      <div className="skel" style={{ height: 12, width: '85%' }} />
    </div>
  );
}
