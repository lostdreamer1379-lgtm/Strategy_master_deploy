"""
STRATEGY MASTER - Prediction API (Vercel serverless function)
POST /api/predict

Lazy-loads models from models/production/ on first invocation (checking
for XGBoost .json first, then falling back to .pkl for LR/RF), builds
feature vectors via api/utils.py, runs all 3 predictors, computes SHAP
top factors, and returns a strategy recommendation.
"""

import os
import sys
import json
import traceback
from http.server import BaseHTTPRequestHandler

import numpy as np
import joblib
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, '../models')
PRODUCTION_DIR = os.path.join(MODELS_DIR, 'production')

_state = {'loaded': False}


def _try_load(path_json, path_pkl):
    """Prefer XGBoost .json, fall back to sklearn .pkl. Returns (model, kind)."""
    if os.path.exists(path_json):
        model = xgb.Booster()
        model.load_model(path_json)
        return model, 'xgboost'
    if os.path.exists(path_pkl):
        return joblib.load(path_pkl), 'sklearn'
    return None, None


def load_models():
    if _state['loaded']:
        return

    encoders_path = os.path.join(MODELS_DIR, 'label_encoders.pkl')
    _state['encoders'] = joblib.load(encoders_path) if os.path.exists(encoders_path) else None

    selected_path = os.path.join(PRODUCTION_DIR, 'SELECTED_MODELS.json')
    selected = {}
    if os.path.exists(selected_path):
        with open(selected_path, 'r') as f:
            selected = json.load(f)

    model_sources = {}

    # Degradation
    deg_json = os.path.join(PRODUCTION_DIR, 'model_degradation_BEST.json')
    deg_pkl = os.path.join(PRODUCTION_DIR, 'model_degradation_BEST.pkl')
    if not os.path.exists(deg_json) and not os.path.exists(deg_pkl):
        deg_json = os.path.join(MODELS_DIR, 'xgb_degradation.json')  # fallback default
    deg_model, deg_kind = _try_load(deg_json, deg_pkl)
    _state['degradation_model'] = deg_model
    _state['degradation_kind'] = deg_kind
    deg_scaler_path = os.path.join(PRODUCTION_DIR, 'scaler_degradation_BEST.pkl')
    _state['degradation_scaler'] = joblib.load(deg_scaler_path) if os.path.exists(deg_scaler_path) else None
    model_sources['degradation'] = selected.get('degradation', 'xgboost (default)')

    # Pit stop
    pit_json = os.path.join(PRODUCTION_DIR, 'model_pitstop_BEST.json')
    pit_pkl = os.path.join(PRODUCTION_DIR, 'model_pitstop_BEST.pkl')
    if not os.path.exists(pit_json) and not os.path.exists(pit_pkl):
        pit_json = os.path.join(MODELS_DIR, 'xgb_pitstop.json')
    pit_model, pit_kind = _try_load(pit_json, pit_pkl)
    _state['pitstop_model'] = pit_model
    _state['pitstop_kind'] = pit_kind
    pit_scaler_path = os.path.join(PRODUCTION_DIR, 'scaler_pitstop_BEST.pkl')
    _state['pitstop_scaler'] = joblib.load(pit_scaler_path) if os.path.exists(pit_scaler_path) else None
    model_sources['pitstop'] = selected.get('pitstop', 'xgboost (default)')

    # Fuel
    fuel_json = os.path.join(PRODUCTION_DIR, 'model_fuel_BEST.json')
    fuel_pkl = os.path.join(PRODUCTION_DIR, 'model_fuel_BEST.pkl')
    if not os.path.exists(fuel_json) and not os.path.exists(fuel_pkl):
        fuel_json = os.path.join(MODELS_DIR, 'xgb_fuel.json')
    fuel_model, fuel_kind = _try_load(fuel_json, fuel_pkl)
    _state['fuel_model'] = fuel_model
    _state['fuel_kind'] = fuel_kind
    fuel_scaler_path = os.path.join(PRODUCTION_DIR, 'scaler_fuel_BEST.pkl')
    _state['fuel_scaler'] = joblib.load(fuel_scaler_path) if os.path.exists(fuel_scaler_path) else None
    model_sources['fuel'] = selected.get('fuel', 'xgboost (default)')

    # SHAP explainers (optional — only exist for the tree/XGBoost models we trained SHAP on)
    for key, fname in [('degradation', 'shap_explainer_deg.pkl'),
                        ('pitstop', 'shap_explainer_pit.pkl'),
                        ('fuel', 'shap_explainer_fuel.pkl')]:
        p = os.path.join(MODELS_DIR, fname)
        _state[f'{key}_explainer'] = joblib.load(p) if os.path.exists(p) else None

    _state['model_sources'] = model_sources
    _state['loaded'] = True


def _predict_value(kind, model, scaler, X):
    X = np.array(X).reshape(1, -1)
    if kind == 'sklearn' and scaler is not None:
        X_in = scaler.transform(X)
    else:
        X_in = X
    pred = model.predict(xgb.DMatrix(X_in))[0] if kind == 'xgboost' else model.predict(X_in)[0]
    return float(pred), X


def _shap_top_factors(explainer, X_raw, feature_names, top_n=5):
    if explainer is None:
        return []
    try:
        shap_values = explainer(np.array(X_raw).reshape(1, -1))
        vals = shap_values.values[0]
        pairs = sorted(zip(feature_names, vals), key=lambda p: abs(p[1]), reverse=True)[:top_n]
        return [{'feature': name, 'impact': round(float(v), 4)} for name, v in pairs]
    except Exception:
        return []


def run_prediction(payload):
    load_models()
    meta = utils.get_metadata()
    encoders = _state['encoders']

    circuit = payload.get('circuit', 'Silverstone')
    compound = str(payload.get('compound', 'MEDIUM')).upper()
    tyre_life = float(payload.get('tyre_life', 10))
    stint_lap = float(payload.get('stint_lap', tyre_life))
    track_temp = float(payload.get('track_temp', 30))
    air_temp = float(payload.get('air_temp', 20))
    humidity = float(payload.get('humidity', 50))
    wind_speed = float(payload.get('wind_speed', 2))
    rainfall = int(bool(payload.get('rainfall', 0)))
    fuel_kg = float(payload.get('fuel_kg', 60))
    total_laps = float(payload.get('total_laps', 55))
    lap_number = float(payload.get('lap_number', stint_lap))

    norm_payload = {
        'circuit': circuit, 'compound': compound, 'tyre_life': tyre_life,
        'stint_lap': stint_lap, 'track_temp': track_temp, 'air_temp': air_temp,
        'humidity': humidity, 'wind_speed': wind_speed, 'rainfall': rainfall,
        'fuel_kg': fuel_kg, 'total_laps': total_laps, 'lap_number': lap_number,
    }

    # --- Degradation ---
    X_deg = utils.build_feature_vector('degradation', norm_payload, encoders)
    deg_rate, X_deg_arr = _predict_value(_state['degradation_kind'], _state['degradation_model'],
                                          _state['degradation_scaler'], X_deg)
    deg_rate = max(0.0, deg_rate)
    severity = 'low' if deg_rate < 2 else ('moderate' if deg_rate < 5 else 'high')
    deg_factors = _shap_top_factors(_state['degradation_explainer'], X_deg, meta['degradation_features'])

    # --- Pit stop ---
    X_pit = utils.build_feature_vector('pitstop', norm_payload, encoders)
    pit_model, pit_kind, pit_scaler = _state['pitstop_model'], _state['pitstop_kind'], _state['pitstop_scaler']
    X_pit_arr = np.array(X_pit).reshape(1, -1)
    X_pit_in = pit_scaler.transform(X_pit_arr) if (pit_kind == 'sklearn' and pit_scaler is not None) else X_pit_arr
    if pit_kind == 'xgboost':
        pit_proba = float(pit_model.predict(xgb.DMatrix(X_pit_in))[0])
    else:
        pit_proba = float(pit_model.predict_proba(X_pit_in)[0][1])
    pit_recommendation = 'pit now' if pit_proba > 0.65 else ('consider pit window' if pit_proba > 0.35 else 'stay out')
    pit_factors = _shap_top_factors(_state['pitstop_explainer'], X_pit, meta['pitstop_features'])

    # --- Fuel ---
    X_fuel = utils.build_feature_vector('fuel', norm_payload, encoders)
    fuel_est, X_fuel_arr = _predict_value(_state['fuel_kind'], _state['fuel_model'],
                                           _state['fuel_scaler'], X_fuel)
    fuel_est = max(0.0, fuel_est)
    laps_remaining = max(0, int(total_laps - lap_number))
    fuel_factors = _shap_top_factors(_state['fuel_explainer'], X_fuel, meta['fuel_features'])

    # --- Strategy synthesis ---
    optimal_stint = int(max(5, min(45, (100 / max(deg_rate, 0.5)) * 0.6)))
    recommended_pit_lap = int(lap_number + max(1, optimal_stint - stint_lap))
    must_pit_by = int(lap_number + max(1, (100 / max(deg_rate, 0.5))))

    strategies = [
        {
            'rank': 1, 'name': 'Balanced (recommended)',
            'description': f'Pit around lap {recommended_pit_lap} for fresh rubber before degradation compounds.',
            'total_laps': optimal_stint, 'risk': 'low',
            'time_saved_s': round(max(0, (5 - deg_rate) * 1.8), 1),
        },
        {
            'rank': 2, 'name': 'Extended stint',
            'description': 'Push the current tyres further to undercut-proof against late-race traffic.',
            'total_laps': optimal_stint + 6, 'risk': 'moderate',
            'time_saved_s': round(max(-2, (3 - deg_rate) * 1.2), 1),
        },
        {
            'rank': 3, 'name': 'Aggressive undercut',
            'description': 'Pit early to jump cars still on an older set in the pit window.',
            'total_laps': max(5, optimal_stint - 5), 'risk': 'high',
            'time_saved_s': round(max(-3, (6 - deg_rate) * 0.9), 1),
        },
    ]

    return {
        'success': True,
        'input': norm_payload,
        'predictions': {
            'degradation': {
                'rate_pct_per_lap': round(deg_rate, 3),
                'severity': severity,
                'shap_factors': deg_factors,
            },
            'pit_stop': {
                'pit_probability': round(pit_proba, 3),
                'recommendation': pit_recommendation,
                'shap_factors': pit_factors,
            },
            'fuel': {
                'estimated_kg': round(fuel_est, 2),
                'laps_remaining': laps_remaining,
                'shap_factors': fuel_factors,
            },
        },
        'strategy': {
            'optimal_stint_laps': optimal_stint,
            'recommended_pit_lap': recommended_pit_lap,
            'pit_probability_pct': round(pit_proba * 100, 1),
            'fuel_laps_remaining': laps_remaining,
            'must_pit_by_lap': must_pit_by,
            'strategies': strategies,
        },
        'model_sources': _state['model_sources'],
    }


class handler(BaseHTTPRequestHandler):
    def _set_headers(self, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(204)

    def do_GET(self):
        self._set_headers(200)
        self.wfile.write(json.dumps({
            'status': 'ok',
            'message': 'Strategy Master predict API. POST a JSON payload to get predictions.',
            'circuits': utils.list_circuits(),
            'compounds': utils.list_compounds(),
        }).encode())

    def do_POST(self):
        try:
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length) if length else b'{}'
            payload = json.loads(body or b'{}')
            result = run_prediction(payload)
            self._set_headers(200)
            self.wfile.write(json.dumps(result).encode())
        except Exception as e:
            self._set_headers(500)
            self.wfile.write(json.dumps({
                'success': False,
                'error': str(e),
                'trace': traceback.format_exc()[:2000],
            }).encode())
