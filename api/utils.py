"""
STRATEGY MASTER - API Utilities
Loads circuit/compound characteristics from the SAME JSON config files
used by training (../training/circuits_config.json, compounds_config.json)
so there is a single source of truth, and assembles feature vectors for
each of the 3 predictors in the exact column order the models expect.
"""

import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
CIRCUITS_PATH = os.path.join(HERE, '../training/circuits_config.json')
COMPOUNDS_PATH = os.path.join(HERE, '../training/compounds_config.json')
METADATA_PATH = os.path.join(HERE, '../models/model_metadata.json')

DEFAULT_CIRCUIT = {'corner_ratio': 0.55, 'abrasiveness': 7, 'speed_index': 6, 'lap_length_km': 5.0}
DEFAULT_COMPOUND = {'durability': 5, 'grip': 5, 'optimal_temp_c': 80, 'fuel_mult': 1.0}

_circuits_config = None
_compounds_config = None
_metadata = None


def _load_configs():
    global _circuits_config, _compounds_config
    if _circuits_config is None:
        with open(CIRCUITS_PATH, 'r') as f:
            _circuits_config = json.load(f)
    if _compounds_config is None:
        with open(COMPOUNDS_PATH, 'r') as f:
            _compounds_config = json.load(f)


def get_metadata():
    global _metadata
    if _metadata is None:
        with open(METADATA_PATH, 'r') as f:
            _metadata = json.load(f)
    return _metadata


def get_circuit_features(circuit):
    _load_configs()
    return _circuits_config.get(circuit, DEFAULT_CIRCUIT)


def get_compound_features(compound):
    _load_configs()
    return _compounds_config.get(str(compound).upper(), DEFAULT_COMPOUND)


def list_circuits():
    _load_configs()
    return sorted(_circuits_config.keys())


def list_compounds():
    _load_configs()
    return sorted(k for k in _compounds_config.keys() if k != 'UNKNOWN')


def encode_label(value, encoder_classes):
    """Mimic sklearn LabelEncoder.transform for a single value using the
    fitted classes_ array, falling back to 0 for unseen labels."""
    classes = list(encoder_classes)
    try:
        return classes.index(value)
    except ValueError:
        return 0


def dur_x_abrasive_norm_deg(circ, comp, tyre_life, track_temp):
    """Rough physical estimate of cumulative % pace loss so far in the stint,
    used only to seed rolling/stint-aggregate features at inference time
    (training computes these from real lap history instead)."""
    temp_mismatch = abs(track_temp - comp['optimal_temp_c'])
    wear_rate = (circ['abrasiveness'] / max(comp['durability'], 1)) * (1 + temp_mismatch / 100)
    return wear_rate * 0.15 * tyre_life


def build_feature_vector(predictor, payload, encoders):
    """Assemble the ordered numeric feature vector for one predictor.

    payload: dict from the request body (circuit, compound, tyre_life,
             stint_lap, track_temp, air_temp, humidity, wind_speed,
             rainfall, fuel_kg, total_laps, lap_number)
    encoders: dict of fitted LabelEncoders loaded from label_encoders.pkl
    """
    meta = get_metadata()
    circuit = payload['circuit']
    compound = str(payload['compound']).upper()

    circ = get_circuit_features(circuit)
    comp = get_compound_features(compound)

    circuit_enc = encode_label(circuit, encoders['circuit'].classes_)
    compound_enc = encode_label(compound, encoders['compound'].classes_)

    tyre_life = float(payload.get('tyre_life', 1))
    stint_lap = float(payload.get('stint_lap', tyre_life))
    lap_number = float(payload.get('lap_number', stint_lap))
    track_temp = float(payload.get('track_temp', 30.0))
    air_temp = float(payload.get('air_temp', 20.0))
    humidity = float(payload.get('humidity', 50.0))
    rainfall = int(bool(payload.get('rainfall', 0)))
    fuel_kg = float(payload.get('fuel_kg', 55.0))
    total_laps = float(payload.get('total_laps', 55))

    temp_mismatch = abs(track_temp - comp['optimal_temp_c'])
    dur_x_abrasive = comp['durability'] * circ['abrasiveness']
    grip_x_corners = comp['grip'] * circ['corner_ratio']
    is_wet = 1 if compound in ('INTERMEDIATE', 'WET') else 0
    fuel_x_stint = fuel_kg * stint_lap
    abrasive_x_lap = circ['abrasiveness'] * lap_number

    # At inference time we don't have actual lap-by-lap history for this
    # stint, so rolling/stint-aggregate features are approximated from the
    # single-point physical model rather than left at zero — zero would read
    # as "no degradation yet", which skews both the prediction and SHAP.
    approx_deg_pct = min(20.0, dur_x_abrasive_norm_deg(circ, comp, tyre_life, track_temp))
    rolling_deg_3 = approx_deg_pct
    stint_max_lap = max(stint_lap, tyre_life)
    stint_avg_deg = approx_deg_pct * 0.6

    feature_values = {
        'tyre_life': tyre_life, 'stint_lap': stint_lap, 'compound_enc': compound_enc,
        'circuit_enc': circuit_enc, 'track_temp_c': track_temp, 'air_temp_c': air_temp,
        'humidity_pct': humidity, 'comp_durability': comp['durability'],
        'comp_grip': comp['grip'], 'comp_opt_temp': comp['optimal_temp_c'],
        'temp_mismatch': temp_mismatch, 'abrasiveness': circ['abrasiveness'],
        'corner_ratio': circ['corner_ratio'], 'speed_index': circ['speed_index'],
        'dur_x_abrasive': dur_x_abrasive, 'grip_x_corners': grip_x_corners,
        'rolling_deg_3': rolling_deg_3, 'is_wet_compound': is_wet, 'rainfall': rainfall,
        'deg_pct': approx_deg_pct, 'fuel_estimate_kg': fuel_kg, 'lap_number': lap_number,
        'stint_max_lap': stint_max_lap, 'stint_avg_deg': stint_avg_deg,
        'fuel_x_stint': fuel_x_stint, 'lap_length_km': circ['lap_length_km'],
        'abrasive_x_lap': abrasive_x_lap,
    }

    key = {'degradation': 'degradation_features', 'pitstop': 'pitstop_features', 'fuel': 'fuel_features'}[predictor]
    ordered = [feature_values[name] for name in meta[key]]
    return ordered
