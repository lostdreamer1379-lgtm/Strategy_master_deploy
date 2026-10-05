import React from 'react';

export default function StrategyCard({ strategy }) {
  if (!strategy) return null;

  return (
    <div className="panel">
      <div className="panel-head">
        <h2>Strategy options</h2>
        <span className="hint">pit by lap {strategy.must_pit_by_lap}</span>
      </div>
      <div className="strategy-list">
        {strategy.strategies.map((s) => (
          <div key={s.rank} className={`strategy-row ${s.rank === 1 ? 'primary' : ''}`}>
            <div className="strategy-rank">{s.rank}</div>
            <div>
              <div className="strategy-name">{s.name}</div>
              <div className="strategy-desc">{s.description}</div>
            </div>
            <div className="strategy-meta">
              <div className="strategy-laps tabular">{s.total_laps} laps</div>
              <div className="strategy-risk">{s.risk} risk</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
