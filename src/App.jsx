import React, { useEffect, useState, useCallback } from 'react';
import Header from './components/Header';
import Selectors from './components/Selectors';
import DegradationCard from './components/DegradationCard';
import PitWindowCard from './components/PitWindowCard';
import FuelCard from './components/FuelCard';
import ExplainabilityCard from './components/ExplainabilityCard';
import DegradationChart from './components/DegradationChart';
import StrategyCard from './components/StrategyCard';

const DEFAULT_FORM = {
  circuit: 'Silverstone',
  compound: 'MEDIUM',
  tyre_life: 12,
  stint_lap: 12,
  lap_number: 24,
  track_temp: 34,
  air_temp: 23,
  humidity: 48,
  wind_speed: 2,
  rainfall: 0,
  fuel_kg: 62,
  total_laps: 52,
};

// Deterministic mock so the UI is fully demoable without a live API —
// used only if /api/predict is unreachable (e.g. local `npm start` with no backend).
function mockPredict(form) {
  const circAbrasive = {
    Monaco: 5, Monza: 7, Silverstone: 8, Baku: 9, Singapore: 8,
  }[form.circuit] || 7;
  const compDur = { SOFT: 3, MEDIUM: 5, HARD: 8, INTERMEDIATE: 6, WET: 7 }[form.compound] || 5;

  const baseRate = (circAbrasive / compDur) * (1 + form.track_temp / 100) * (form.tyre_life / 20);
  const rate = Math.max(0.3, Math.min(9, baseRate));
  const severity = rate < 2 ? 'low' : rate < 5 ? 'moderate' : 'high';

  const pitProb = Math.max(0, Math.min(1, (form.tyre_life / 35) + (rate / 20)));
  const recommendation = pitProb > 0.65 ? 'pit now' : pitProb > 0.35 ? 'consider pit window' : 'stay out';

  const fuelEst = Math.max(2, form.fuel_kg - form.lap_number * 1.6);
  const lapsRemaining = Math.max(0, Math.round(fuelEst / 1.6));

  const optimalStint = Math.round(Math.max(8, Math.min(42, (100 / Math.max(rate, 0.5)) * 0.6)));
  const recommendedPitLap = form.lap_number + Math.max(1, optimalStint - form.stint_lap);
  const mustPitBy = form.lap_number + Math.max(1, Math.round(100 / Math.max(rate, 0.5)));

  const degFactors = [
    { feature: 'track_temp_c', impact: (form.track_temp - 30) * 0.04 },
    { feature: 'tyre_life', impact: form.tyre_life * 0.03 },
    { feature: 'abrasiveness', impact: (circAbrasive - 7) * 0.08 },
    { feature: 'comp_durability', impact: -(compDur - 5) * 0.09 },
    { feature: 'humidity_pct', impact: (form.humidity - 50) * 0.01 },
  ];
  const pitFactors = [
    { feature: 'tyre_life', impact: form.tyre_life * 0.02 },
    { feature: 'deg_pct', impact: rate * 0.05 },
    { feature: 'stint_lap', impact: form.stint_lap * 0.015 },
    { feature: 'fuel_estimate_kg', impact: -fuelEst * 0.004 },
    { feature: 'rainfall', impact: form.rainfall ? 0.3 : -0.05 },
  ];
  const fuelFactors = [
    { feature: 'lap_number', impact: -form.lap_number * 0.4 },
    { feature: 'stint_lap', impact: -form.stint_lap * 0.1 },
    { feature: 'lap_length_km', impact: 0.6 },
    { feature: 'comp_durability', impact: compDur * 0.05 },
    { feature: 'air_temp_c', impact: (form.air_temp - 20) * 0.02 },
  ];

  return {
    success: true,
    input: form,
    predictions: {
      degradation: { rate_pct_per_lap: rate, severity, shap_factors: degFactors },
      pit_stop: { pit_probability: pitProb, recommendation, shap_factors: pitFactors },
      fuel: { estimated_kg: fuelEst, laps_remaining: lapsRemaining, shap_factors: fuelFactors },
    },
    strategy: {
      optimal_stint_laps: optimalStint,
      recommended_pit_lap: recommendedPitLap,
      pit_probability_pct: Math.round(pitProb * 100 * 10) / 10,
      fuel_laps_remaining: lapsRemaining,
      must_pit_by_lap: mustPitBy,
      strategies: [
        {
          rank: 1, name: 'Balanced (recommended)',
          description: `Pit around lap ${recommendedPitLap} for fresh rubber before degradation compounds.`,
          total_laps: optimalStint, risk: 'low',
          time_saved_s: Math.max(0, Math.round((5 - rate) * 1.8 * 10) / 10),
        },
        {
          rank: 2, name: 'Extended stint',
          description: 'Push the current tyres further to undercut-proof against late-race traffic.',
          total_laps: optimalStint + 6, risk: 'moderate',
          time_saved_s: Math.round((3 - rate) * 1.2 * 10) / 10,
        },
        {
          rank: 3, name: 'Aggressive undercut',
          description: 'Pit early to jump cars still on an older set in the pit window.',
          total_laps: Math.max(5, optimalStint - 5), risk: 'high',
          time_saved_s: Math.round((6 - rate) * 0.9 * 10) / 10,
        },
      ],
    },
    model_sources: { degradation: 'xgboost', pitstop: 'xgboost', fuel: 'xgboost' },
    _mock: true,
  };
}

export default function App() {
  const [theme, setTheme] = useState(() => {
    if (typeof window === 'undefined') return 'dark';
    return window.localStorage.getItem('sm-theme') || 'dark';
  });
  const [form, setForm] = useState(DEFAULT_FORM);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [usingMock, setUsingMock] = useState(false);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    window.localStorage.setItem('sm-theme', theme);
  }, [theme]);

  const toggleTheme = () => setTheme((t) => (t === 'dark' ? 'light' : 'dark'));

  const run = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      });
      if (!res.ok) throw new Error('API unavailable');
      const json = await res.json();
      if (!json.success) throw new Error(json.error || 'Prediction failed');
      setResult(json);
      setUsingMock(false);
    } catch (e) {
      // Graceful fallback so the UI is always demoable
      setResult(mockPredict(form));
      setUsingMock(true);
    } finally {
      setLoading(false);
    }
  }, [form]);

  useEffect(() => {
    run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="app">
      <div className="shell">
        <Header theme={theme} onToggleTheme={toggleTheme} />

        <div className="layout">
          <Selectors form={form} setForm={setForm} onRun={run} loading={loading} />

          <div className="data-zone">
            {!result ? (
              <EmptyState />
            ) : (
              <>
                <div className="readout-strip">
                  <DegradationCard data={result.predictions.degradation} />
                  <PitWindowCard data={result.predictions.pit_stop} />
                  <FuelCard data={result.predictions.fuel} />
                </div>

                <DegradationChart
                  data={result.predictions.degradation}
                  currentLap={form.lap_number}
                  pitLap={result.strategy.recommended_pit_lap}
                />

                <StrategyCard strategy={result.strategy} />

                <ExplainabilityCard predictions={result.predictions} />
              </>
            )}

            <Footer usingMock={usingMock} sources={result?.model_sources} />
          </div>
        </div>
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="empty-state">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
        <path d="M3 12h4l2-6h6l2 6h4" />
        <path d="M5 12v5a1 1 0 001 1h12a1 1 0 001-1v-5" />
      </svg>
      <h3>No run yet</h3>
      <p>Set circuit, compound and conditions on the left, then run the strategy model.</p>
    </div>
  );
}

function Footer({ usingMock, sources }) {
  return (
    <div className="footer">
      <span>
        {usingMock
          ? 'demo mode — local heuristic (API not reached)'
          : sources
            ? `models: deg=${sources.degradation} · pit=${sources.pitstop} · fuel=${sources.fuel}`
            : ''}
      </span>
      <span>Strategy Master · Project 1 of 6 · F1 Analytics Platform</span>
    </div>
  );
}
