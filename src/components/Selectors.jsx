import React from 'react';

const CIRCUITS = [
  'Bahrain', 'Jeddah', 'Melbourne', 'Baku', 'Miami', 'Imola', 'Monaco',
  'Barcelona', 'Montreal', 'Silverstone', 'Budapest', 'Spa', 'Zandvoort',
  'Monza', 'Singapore', 'Suzuka', 'Austin', 'Mexico City', 'São Paulo',
  'Las Vegas', 'Abu Dhabi', 'Lusail',
];

const COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET'];

export default function Selectors({ form, setForm, onRun, loading }) {
  const update = (key, value) => setForm((f) => ({ ...f, [key]: value }));

  return (
    <div className="rail">
      <div className="rail-section">
        <p className="rail-label">CIRCUIT</p>
        <div className="select-wrap">
          <select value={form.circuit} onChange={(e) => update('circuit', e.target.value)}>
            {CIRCUITS.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="rail-section">
        <p className="rail-label">COMPOUND</p>
        <div className="compound-grid">
          {COMPOUNDS.map((c) => (
            <button
              key={c}
              className={`compound-chip ${form.compound === c ? `active-${c}` : ''}`}
              onClick={() => update('compound', c)}
              type="button"
            >
              {c.slice(0, 4)}
            </button>
          ))}
        </div>
      </div>

      <div className="rail-section">
        <div className="slider-row">
          <p className="rail-label">TYRE LIFE</p>
          <span className="slider-value tabular">{form.tyre_life} laps</span>
        </div>
        <input
          type="range" min="1" max="45" value={form.tyre_life}
          onChange={(e) => update('tyre_life', Number(e.target.value))}
        />
      </div>

      <div className="rail-section">
        <div className="slider-row">
          <p className="rail-label">LAP NUMBER</p>
          <span className="slider-value tabular">{form.lap_number}</span>
        </div>
        <input
          type="range" min="1" max="78" value={form.lap_number}
          onChange={(e) => update('lap_number', Number(e.target.value))}
        />
      </div>

      <div className="rail-section">
        <div className="slider-row">
          <p className="rail-label">TRACK TEMP</p>
          <span className="slider-value tabular">{form.track_temp}°C</span>
        </div>
        <input
          type="range" min="10" max="60" value={form.track_temp}
          onChange={(e) => update('track_temp', Number(e.target.value))}
        />
      </div>

      <div className="rail-section">
        <div className="slider-row">
          <p className="rail-label">AIR TEMP</p>
          <span className="slider-value tabular">{form.air_temp}°C</span>
        </div>
        <input
          type="range" min="5" max="45" value={form.air_temp}
          onChange={(e) => update('air_temp', Number(e.target.value))}
        />
      </div>

      <div className="rail-section">
        <div className="slider-row">
          <p className="rail-label">HUMIDITY</p>
          <span className="slider-value tabular">{form.humidity}%</span>
        </div>
        <input
          type="range" min="10" max="100" value={form.humidity}
          onChange={(e) => update('humidity', Number(e.target.value))}
        />
      </div>

      <div className="rail-section">
        <div className="slider-row">
          <p className="rail-label">FUEL LOAD</p>
          <span className="slider-value tabular">{form.fuel_kg} kg</span>
        </div>
        <input
          type="range" min="5" max="110" value={form.fuel_kg}
          onChange={(e) => update('fuel_kg', Number(e.target.value))}
        />
      </div>

      <div className="rail-section">
        <div className="toggle-row">
          <span>Rainfall</span>
          <button
            type="button"
            className={`mini-switch ${form.rainfall ? 'on' : ''}`}
            onClick={() => update('rainfall', form.rainfall ? 0 : 1)}
            aria-label="Toggle rainfall"
          />
        </div>
      </div>

      <button className="run-btn" onClick={onRun} disabled={loading}>
        {loading ? (
          'CALCULATING…'
        ) : (
          <>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="5 3 19 12 5 21 5 3" />
            </svg>
            RUN STRATEGY
          </>
        )}
      </button>
    </div>
  );
}
