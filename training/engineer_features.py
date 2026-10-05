"""
STRATEGY MASTER - Feature Engineering
Takes the raw per-lap CSV from collect_data.py and builds the final
model-ready feature set: encodes categoricals, adds rolling/stint stats,
interaction terms, and the pit_next_lap classification target.
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.preprocessing import LabelEncoder

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(HERE, '../data/raw/f1_strategy_data.csv')
OUT_PATH = os.path.join(HERE, '../data/processed/f1_strategy_features.csv')
ENCODERS_PATH = os.path.join(HERE, '../models/label_encoders.pkl')
METADATA_PATH = os.path.join(HERE, '../models/model_metadata.json')
ALIASES_PATH = os.path.join(HERE, 'circuit_aliases.json')


def engineer():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(
            f"Raw data not found at {RAW_PATH}. Run collect_data.py first."
        )

    print("Loading raw data...")
    df = pd.read_csv(RAW_PATH)
    print(f"   {len(df):,} rows loaded")

    # --- Encode categoricals ---
    print("Encoding categorical features...")
    with open(ALIASES_PATH, 'r', encoding='utf-8') as f:
        circuit_aliases = json.load(f)
    circuit_model_names = {
        name: details['model'] for name, details in circuit_aliases.items()
    }
    encoders = {}
    for col in ['circuit', 'compound', 'driver']:
        le = LabelEncoder()
        values = df[col].astype(str)
        if col == 'circuit':
            values = values.map(circuit_model_names).fillna(values)
        df[f'{col}_enc'] = le.fit_transform(values)
        encoders[col] = le

    os.makedirs(os.path.dirname(ENCODERS_PATH), exist_ok=True)
    joblib.dump(encoders, ENCODERS_PATH)

    # --- Sort for rolling/stint computations ---
    df = df.sort_values(['season', 'race', 'driver', 'lap_number']).reset_index(drop=True)

    # --- Rolling degradation / lap time features (per driver-race-stint) ---
    print("Computing rolling and stint features...")
    group_keys = ['season', 'race', 'driver', 'compound']
    df['rolling_deg_3'] = (
        df.groupby(group_keys)['deg_pct']
        .transform(lambda s: s.rolling(3, min_periods=1).mean())
    )
    df['lap_time_roll3'] = (
        df.groupby(group_keys)['lap_time_s']
        .transform(lambda s: s.rolling(3, min_periods=1).mean())
    )

    # --- Stint-level stats ---
    stint_keys = ['season', 'race', 'driver', 'compound']
    df['stint_max_lap'] = df.groupby(stint_keys)['stint_lap'].transform('max')
    df['stint_avg_deg'] = df.groupby(stint_keys)['deg_pct'].transform('mean')

    # --- Pit-next-lap target (classification) ---
    df = df.sort_values(['season', 'race', 'driver', 'lap_number']).reset_index(drop=True)
    df['pit_next_lap'] = (
        df.groupby(['season', 'race', 'driver'])['pit_in']
        .shift(-1)
        .fillna(0)
        .astype(int)
    )

    # --- Interaction features ---
    print("Building interaction features...")
    df['temp_x_dur'] = df['temp_mismatch'] * df['comp_durability']
    df['fuel_x_stint'] = df['fuel_estimate_kg'] * df['stint_lap']
    df['abrasive_x_lap'] = df['abrasiveness'] * df['lap_number']

    # --- Clean NaNs introduced by rolling/shift ops at group edges ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # --- Feature lists per predictor (used by train/evaluate/predict) ---
    degradation_features = [
        'tyre_life', 'stint_lap', 'compound_enc', 'circuit_enc',
        'track_temp_c', 'air_temp_c', 'humidity_pct', 'comp_durability',
        'comp_grip', 'comp_opt_temp', 'temp_mismatch', 'abrasiveness',
        'corner_ratio', 'speed_index', 'dur_x_abrasive', 'grip_x_corners',
        'rolling_deg_3', 'is_wet_compound', 'rainfall',
    ]
    pitstop_features = [
        'tyre_life', 'stint_lap', 'deg_pct', 'rolling_deg_3', 'compound_enc',
        'circuit_enc', 'fuel_estimate_kg', 'lap_number', 'stint_max_lap',
        'stint_avg_deg', 'track_temp_c', 'abrasiveness', 'is_wet_compound',
        'rainfall',
    ]
    fuel_features = [
        'lap_number', 'stint_lap', 'fuel_x_stint', 'comp_durability',
        'circuit_enc', 'compound_enc', 'air_temp_c', 'lap_length_km',
        'abrasive_x_lap',
    ]

    metadata = {
        'degradation_features': degradation_features,
        'pitstop_features': pitstop_features,
        'fuel_features': fuel_features,
        'targets': {
            'degradation': 'deg_pct',
            'pitstop': 'pit_next_lap',
            'fuel': 'fuel_estimate_kg',
        },
        'n_rows': len(df),
        'circuits': sorted(df['circuit'].unique().tolist()),
        'compounds': sorted(df['compound'].unique().tolist()),
    }
    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"\nFeature engineering complete")
    print(f"   Rows: {len(df):,}")
    print(f"   Saved to: {OUT_PATH}")
    print(f"   Encoders: {ENCODERS_PATH}")
    print(f"   Metadata: {METADATA_PATH}")


if __name__ == '__main__':
    engineer()
