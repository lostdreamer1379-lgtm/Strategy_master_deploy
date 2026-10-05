import React from 'react';
import ThemeToggle from './ThemeToggle';

export default function Header({ theme, onToggleTheme }) {
  return (
    <header className="tower-header">
      <div className="brand">
        <div className="brand-mark">
          <svg viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M3 12h4l2-6h6l2 6h4" />
            <path d="M5 12v5a1 1 0 001 1h12a1 1 0 001-1v-5" />
            <circle cx="8" cy="18" r="1.5" />
            <circle cx="16" cy="18" r="1.5" />
          </svg>
        </div>
        <div className="brand-text">
          <h1>Strategy Master</h1>
          <p>tyre degradation · pit window · fuel model</p>
        </div>
      </div>
      <div className="header-right">
        <div className="live-pill">
          <span className="live-dot" />
          MODEL ONLINE
        </div>
        <ThemeToggle theme={theme} onToggle={onToggleTheme} />
      </div>
    </header>
  );
}
