import React from 'react';

export default function FuelCard({ data }) {
  if (!data) return <ReadoutSkeleton title="FUEL MODEL" />;

  const { estimated_kg, laps_remaining } = data;
  const pct = Math.min(100, (estimated_kg / 110) * 100);

  return (
    <div className="readout" style={{ '--bar-color': 'var(--teal)' }}>
      <div className="readout-head">
        <span className="readout-title">FUEL MODEL</span>
      </div>
      <div className="readout-value tabular">
        {estimated_kg.toFixed(1)}<span className="unit">kg</span>
      </div>
      <p className="readout-sub">≈ {laps_remaining} laps of fuel remaining at current consumption.</p>
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
      <div className="skel" style={{ height: 38, width: '50%', marginBottom: 8 }} />
      <div className="skel" style={{ height: 12, width: '75%' }} />
    </div>
  );
}
