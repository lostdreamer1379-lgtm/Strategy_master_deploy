import React from 'react';

const REC_COLOR = {
  'stay out': 'var(--teal)',
  'consider pit window': 'var(--amber)',
  'pit now': 'var(--accent)',
};

const REC_BADGE = {
  'stay out': 'badge-stay',
  'consider pit window': 'badge-consider',
  'pit now': 'badge-pit',
};

export default function PitWindowCard({ data }) {
  if (!data) return <ReadoutSkeleton title="PIT WINDOW" />;

  const { pit_probability, recommendation } = data;
  const pct = Math.round(pit_probability * 100);

  return (
    <div className="readout" style={{ '--bar-color': REC_COLOR[recommendation] }}>
      <div className="readout-head">
        <span className="readout-title">PIT WINDOW</span>
        <span className={`readout-badge ${REC_BADGE[recommendation]}`}>{recommendation}</span>
      </div>
      <div className="readout-value tabular">
        {pct}<span className="unit">% likely</span>
      </div>
      <p className="readout-sub">Probability the optimal strategy calls for a stop within the next lap.</p>
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
      <div className="skel" style={{ height: 38, width: '55%', marginBottom: 8 }} />
      <div className="skel" style={{ height: 12, width: '80%' }} />
    </div>
  );
}
