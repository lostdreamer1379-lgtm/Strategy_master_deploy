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
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils  # noqa: E402
from rf_native import NativeRandomForest  # noqa: E402
from xgb_native import NativeXGBModel  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, '../models')
PRODUCTION_DIR = os.path.join(MODELS_DIR, 'production')

_state = {'loaded': False}


def _try_load(path_json, path_pkl):
    """Load a native XGBoost model. Production deployment uses JSON models."""
    if os.path.exists(path_json):
        model = NativeXGBModel(path_json)
        return model, 'xgboost'
    return None, None


def load_models():
    if _state['loaded']:
        return

    encoders_path = os.path.join(MODELS_DIR, 'label_encoders.json')
    with open(encoders_path, 'r') as f:
        _state['encoders'] = json.load(f)

    selected_path = os.path.join(PRODUCTION_DIR, 'SELECTED_MODELS.json')
    selected = {}
    if os.path.exists(selected_path):
        with open(selected_path, 'r') as f:
            selected = json.load(f)

    model_sources = {}

    # Degradation
    deg_json = os.path.join(PRODUCTION_DIR, 'model_degradation_BEST.json')
    deg_rf = os.path.join(PRODUCTION_DIR, 'model_degradation_BEST.rfbin')
    if selected.get('degradation') == 'random_forest' and os.path.exists(deg_rf):
        deg_model, deg_kind = NativeRandomForest(deg_rf), 'random_forest'
    elif not os.path.exists(deg_json):
        deg_json = os.path.join(MODELS_DIR, 'xgb_degradation.json')  # fallback default
        deg_model, deg_kind = _try_load(deg_json, None)
    else:
        deg_model, deg_kind = _try_load(deg_json, None)
    _state['degradation_model'] = deg_model
    _state['degradation_kind'] = deg_kind
    _state['degradation_scaler'] = None
    model_sources['degradation'] = selected.get('degradation', 'xgboost (default)')

    # Pit stop
    pit_json = os.path.join(PRODUCTION_DIR, 'model_pitstop_BEST.json')
    if not os.path.exists(pit_json):
        pit_json = os.path.join(MODELS_DIR, 'xgb_pitstop.json')
    pit_model, pit_kind = _try_load(pit_json, None)
    _state['pitstop_model'] = pit_model
    _state['pitstop_kind'] = pit_kind
    _state['pitstop_scaler'] = None
    model_sources['pitstop'] = selected.get('pitstop', 'xgboost (default)')

    # Fuel
    fuel_json = os.path.join(PRODUCTION_DIR, 'model_fuel_BEST.json')
    if not os.path.exists(fuel_json):
        fuel_json = os.path.join(MODELS_DIR, 'xgb_fuel.json')
    fuel_model, fuel_kind = _try_load(fuel_json, None)
    _state['fuel_model'] = fuel_model
    _state['fuel_kind'] = fuel_kind
    _state['fuel_scaler'] = None
    model_sources['fuel'] = selected.get('fuel', 'xgboost (default)')

    _state['model_sources'] = model_sources
    _state['loaded'] = True


def _predict_value(kind, model, scaler, X):
    X = list(X)
    pred = model.predict(X)
    return float(pred), X


def _shap_top_factors(model, X_raw, feature_names, top_n=5):
    if model is None:
        return []
    try:
        vals = model.shap_values(X_raw)
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
    deg_factors = _shap_top_factors(_state['degradation_model'], X_deg, meta['degradation_features'])

    # --- Pit stop ---
    X_pit = utils.build_feature_vector('pitstop', norm_payload, encoders)
    pit_model, pit_kind, pit_scaler = _state['pitstop_model'], _state['pitstop_kind'], _state['pitstop_scaler']
    pit_proba = float(pit_model.predict(X_pit))
    pit_recommendation = 'pit now' if pit_proba > 0.65 else ('consider pit window' if pit_proba > 0.35 else 'stay out')
    pit_factors = _shap_top_factors(_state['pitstop_model'], X_pit, meta['pitstop_features'])

    # --- Fuel ---
    X_fuel = utils.build_feature_vector('fuel', norm_payload, encoders)
    fuel_est, X_fuel_arr = _predict_value(_state['fuel_kind'], _state['fuel_model'],
                                           _state['fuel_scaler'], X_fuel)
    fuel_est = max(0.0, fuel_est)
    laps_remaining = max(0, int(total_laps - lap_number))
    fuel_factors = _shap_top_factors(_state['fuel_model'], X_fuel, meta['fuel_features'])

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


if __name__ == '__main__':
    server = HTTPServer(('localhost', 5000), handler)
    print('Strategy Master API running at http://localhost:5000')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
