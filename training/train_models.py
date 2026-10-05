"""
STRATEGY MASTER - Model Training
Trains 3 candidate models for each of the 3 predictors:
  - Degradation (regression):      Linear Regression | XGBoost | Random Forest
  - Pit Stop (classification):     Logistic Regression | XGBoost | Random Forest
  - Fuel Consumption (regression): Linear Regression | XGBoost | Random Forest

All models + scalers are saved to ../models/. evaluate_models.py then
scores each trio and promotes the best to ../models/production/.
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, '../data/processed/f1_strategy_features.csv')
METADATA_PATH = os.path.join(HERE, '../models/model_metadata.json')
MODELS_DIR = os.path.join(HERE, '../models')

RANDOM_STATE = 42


def load_data_and_metadata():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found. Run engineer_features.py first.")
    df = pd.read_csv(DATA_PATH)
    with open(METADATA_PATH, 'r') as f:
        meta = json.load(f)
    return df, meta


def train_degradation(df, features):
    print("\n[1/3] Degradation models (regression)")
    X = df[features].values
    y = df['deg_pct'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler_degradation.pkl'))

    lr = LinearRegression()
    lr.fit(X_train_s, y_train)
    joblib.dump(lr, os.path.join(MODELS_DIR, 'lr_degradation.pkl'))
    print("   Linear Regression trained")

    xgb_model = xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE
    )
    xgb_model.fit(X_train, y_train)
    xgb_model.save_model(os.path.join(MODELS_DIR, 'xgb_degradation.json'))
    print("   XGBoost trained")

    rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    joblib.dump(rf, os.path.join(MODELS_DIR, 'rf_degradation.pkl'))
    print("   Random Forest trained")

    joblib.dump(
        {'X_test': X_test, 'X_test_s': X_test_s, 'y_test': y_test},
        os.path.join(MODELS_DIR, 'test_split_degradation.pkl')
    )


def train_pitstop(df, features):
    print("\n[2/3] Pit stop models (classification)")
    X = df[features].values
    y = df['pit_next_lap'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler_pitstop.pkl'))

    logr = LogisticRegression(max_iter=1000, class_weight='balanced')
    logr.fit(X_train_s, y_train)
    joblib.dump(logr, os.path.join(MODELS_DIR, 'logr_pitstop.pkl'))
    print("   Logistic Regression trained")

    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pos_weight,
        random_state=RANDOM_STATE, eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    xgb_model.save_model(os.path.join(MODELS_DIR, 'xgb_pitstop.json'))
    print("   XGBoost trained")

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12, random_state=RANDOM_STATE,
        n_jobs=-1, class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    joblib.dump(rf, os.path.join(MODELS_DIR, 'rf_pitstop.pkl'))
    print("   Random Forest trained")

    joblib.dump(
        {'X_test': X_test, 'X_test_s': X_test_s, 'y_test': y_test},
        os.path.join(MODELS_DIR, 'test_split_pitstop.pkl')
    )


def train_fuel(df, features):
    print("\n[3/3] Fuel consumption models (regression)")
    X = df[features].values
    y = df['fuel_estimate_kg'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler_fuel.pkl'))

    lr = LinearRegression()
    lr.fit(X_train_s, y_train)
    joblib.dump(lr, os.path.join(MODELS_DIR, 'lr_fuel.pkl'))
    print("   Linear Regression trained")

    xgb_model = xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE
    )
    xgb_model.fit(X_train, y_train)
    xgb_model.save_model(os.path.join(MODELS_DIR, 'xgb_fuel.json'))
    print("   XGBoost trained")

    rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    joblib.dump(rf, os.path.join(MODELS_DIR, 'rf_fuel.pkl'))
    print("   Random Forest trained")

    joblib.dump(
        {'X_test': X_test, 'X_test_s': X_test_s, 'y_test': y_test},
        os.path.join(MODELS_DIR, 'test_split_fuel.pkl')
    )


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("=" * 70)
    print("STRATEGY MASTER - MODEL TRAINING")
    print("=" * 70)

    df, meta = load_data_and_metadata()
    print(f"\nLoaded {len(df):,} rows")

    train_degradation(df, meta['degradation_features'])
    train_pitstop(df, meta['pitstop_features'])
    train_fuel(df, meta['fuel_features'])

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE — 9 models saved to ../models/")
    print("Run evaluate_models.py next to compare and promote the best.")
    print("=" * 70)


if __name__ == '__main__':
    main()
