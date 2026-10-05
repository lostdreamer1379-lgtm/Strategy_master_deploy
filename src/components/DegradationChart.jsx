import React, { useMemo } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from 'recharts';

export default function DegradationChart({ data, currentLap, pitLap }) {
  const series = useMemo(() => {
    if (!data) return [];
    const { rate_pct_per_lap } = data;
    const points = [];
    for (let lap = 1; lap <= 45; lap++) {
      // simple projected cumulative pace loss for visualization purposes
      const cumulative = rate_pct_per_lap * Math.sqrt(lap);
      points.push({ lap, degradation: Number(cumulative.toFixed(2)) });
    }
    return points;
  }, [data]);

  if (!data) {
    return (
      <div className="panel chart-card">
        <div className="panel-head"><h2>Projected degradation curve</h2></div>
        <div className="skel" style={{ height: 220, width: '100%' }} />
      </div>
    );
  }

  return (
    <div className="panel chart-card">
      <div className="panel-head">
        <h2>Projected degradation curve</h2>
        <span className="hint">cumulative pace loss, modeled</span>
      </div>
      <div className="chart-wrap">
        <ResponsiveContainer width="100%" height={220}>
          <AreaChart data={series} margin={{ top: 6, right: 12, left: -18, bottom: 0 }}>
            <defs>
              <linearGradient id="degFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--accent)" stopOpacity={0.35} />
                <stop offset="100%" stopColor="var(--accent)" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 5" stroke="var(--stroke)" vertical={false} />
            <XAxis
              dataKey="lap" stroke="var(--text-faint)" fontSize={11}
              tickLine={false} axisLine={{ stroke: 'var(--stroke)' }}
            />
            <YAxis
              stroke="var(--text-faint)" fontSize={11}
              tickLine={false} axisLine={false}
              width={34}
            />
            <Tooltip
              contentStyle={{
                background: 'var(--bg-raised)', border: '1px solid var(--stroke-bright)',
                borderRadius: 4, fontSize: 12, fontFamily: 'var(--mono)',
              }}
              labelFormatter={(l) => `Lap ${l}`}
              formatter={(v) => [`${v}%`, 'cum. pace loss']}
            />
            {currentLap && (
              <ReferenceLine x={currentLap} stroke="var(--teal)" strokeDasharray="4 3" />
            )}
            {pitLap && (
              <ReferenceLine x={pitLap} stroke="var(--accent)" strokeDasharray="4 3" />
            )}
            <Area
              type="monotone" dataKey="degradation"
              stroke="var(--accent)" strokeWidth={2}
              fill="url(#degFill)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
