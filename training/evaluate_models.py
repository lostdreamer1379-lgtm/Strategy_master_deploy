"""
STRATEGY MASTER - Model Evaluation
Evaluates all 3 candidate models per predictor, generates SHAP explainability
plots, picks the best model by holdout metric, and copies it (plus its
scaler, if needed) into ../models/production/ for the API to serve.
"""

import os
import json
import numpy as np
import joblib
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, roc_auc_score, precision_score, recall_score,
    confusion_matrix, roc_curve
)
import xgboost as xgb
import shap

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, '../models')
PRODUCTION_DIR = os.path.join(MODELS_DIR, 'production')
REPORT_DIR = os.path.join(HERE, '../evaluation_report')
METADATA_PATH = os.path.join(MODELS_DIR, 'model_metadata.json')

os.makedirs(PRODUCTION_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


def load_metadata():
    with open(METADATA_PATH, 'r') as f:
        return json.load(f)


def load_xgb_model(filename):
    """Load JSON models without invoking XGBoost's sklearn compatibility layer."""
    model = xgb.Booster()
    model.load_model(os.path.join(MODELS_DIR, filename))
    return model


def xgb_predict(model, X):
    return model.predict(xgb.DMatrix(X))


def evaluate_regression(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"   {name:<18} MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")
    return {'mae': mae, 'rmse': rmse, 'r2': r2}


def evaluate_classification(name, y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        auc = float('nan')
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    print(f"   {name:<18} Acc={acc:.4f}  AUC={auc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}")
    return {'accuracy': acc, 'auc': auc, 'precision': prec, 'recall': rec}


def evaluate_degradation(meta):
    print("\n[1/3] Degradation (regression) — lower MAE/RMSE, higher R2 is better")
    split = joblib.load(os.path.join(MODELS_DIR, 'test_split_degradation.pkl'))
    X_test, X_test_s, y_test = split['X_test'], split['X_test_s'], split['y_test']

    results = {}

    lr = joblib.load(os.path.join(MODELS_DIR, 'lr_degradation.pkl'))
    results['linear_regression'] = evaluate_regression('Linear Regression', y_test, lr.predict(X_test_s))

    xgb_model = load_xgb_model('xgb_degradation.json')
    results['xgboost'] = evaluate_regression('XGBoost', y_test, xgb_predict(xgb_model, X_test))

    rf = joblib.load(os.path.join(MODELS_DIR, 'rf_degradation.pkl'))
    results['random_forest'] = evaluate_regression('Random Forest', y_test, rf.predict(X_test))

    best_name = min(results, key=lambda k: results[k]['mae'])
    print(f"   -> Best: {best_name} (MAE={results[best_name]['mae']:.4f})")

    # SHAP on the tree model (XGBoost) as it's generally most faithful/fast
    try:
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer(X_test[:200])
        joblib.dump(explainer, os.path.join(MODELS_DIR, 'shap_explainer_deg.pkl'))

        plt.figure()
        shap.summary_plot(shap_values, X_test[:200], feature_names=meta['degradation_features'], show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(REPORT_DIR, 'shap_degradation.png'), dpi=120)
        plt.close()
    except Exception as e:
        print(f"   SHAP skipped: {str(e)[:60]}")

    try:
        plt.figure(figsize=(6, 5))
        preds = xgb_predict(xgb_model, X_test)
        plt.scatter(y_test, preds, alpha=0.3, s=10)
        lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
        plt.plot(lims, lims, 'r--', linewidth=1)
        plt.xlabel('Actual degradation %')
        plt.ylabel('Predicted degradation %')
        plt.title('Degradation: Actual vs Predicted (XGBoost)')
        plt.tight_layout()
        plt.savefig(os.path.join(REPORT_DIR, 'degradation_predictions.png'), dpi=120)
        plt.close()
    except Exception as e:
        print(f"   Plot skipped: {str(e)[:60]}")

    _promote(best_name, 'degradation')
    return results, best_name


def evaluate_pitstop(meta):
    print("\n[2/3] Pit Stop (classification) — higher Acc/AUC/Prec/Rec is better")
    split = joblib.load(os.path.join(MODELS_DIR, 'test_split_pitstop.pkl'))
    X_test, X_test_s, y_test = split['X_test'], split['X_test_s'], split['y_test']

    results = {}

    logr = joblib.load(os.path.join(MODELS_DIR, 'logr_pitstop.pkl'))
    proba = logr.predict_proba(X_test_s)[:, 1]
    results['logistic_regression'] = evaluate_classification('Logistic Reg', y_test, (proba > 0.5).astype(int), proba)

    xgb_model = load_xgb_model('xgb_pitstop.json')
    proba_x = xgb_predict(xgb_model, X_test)
    results['xgboost'] = evaluate_classification('XGBoost', y_test, (proba_x > 0.5).astype(int), proba_x)

    rf = joblib.load(os.path.join(MODELS_DIR, 'rf_pitstop.pkl'))
    proba_r = rf.predict_proba(X_test)[:, 1]
    results['random_forest'] = evaluate_classification('Random Forest', y_test, (proba_r > 0.5).astype(int), proba_r)

    best_name = max(results, key=lambda k: (results[k]['auc'] if results[k]['auc'] == results[k]['auc'] else -1))
    print(f"   -> Best: {best_name} (AUC={results[best_name]['auc']:.4f})")

    try:
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer(X_test[:200])
        joblib.dump(explainer, os.path.join(MODELS_DIR, 'shap_explainer_pit.pkl'))

        plt.figure()
        shap.summary_plot(shap_values, X_test[:200], feature_names=meta['pitstop_features'], show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(REPORT_DIR, 'shap_pitstop.png'), dpi=120)
        plt.close()
    except Exception as e:
        print(f"   SHAP skipped: {str(e)[:60]}")

    try:
        cm = confusion_matrix(y_test, (proba_x > 0.5).astype(int))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].imshow(cm, cmap='Blues')
        axes[0].set_title('Confusion Matrix (XGBoost)')
        for i in range(2):
            for j in range(2):
                axes[0].text(j, i, str(cm[i, j]), ha='center', va='center')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')

        fpr, tpr, _ = roc_curve(y_test, proba_x)
        axes[1].plot(fpr, tpr, label=f"AUC={results['xgboost']['auc']:.3f}")
        axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1)
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(REPORT_DIR, 'pitstop_evaluation.png'), dpi=120)
        plt.close()
    except Exception as e:
        print(f"   Plot skipped: {str(e)[:60]}")

    _promote(best_name, 'pitstop')
    return results, best_name


def evaluate_fuel(meta):
    print("\n[3/3] Fuel Consumption (regression) — lower MAE/RMSE, higher R2 is better")
    split = joblib.load(os.path.join(MODELS_DIR, 'test_split_fuel.pkl'))
    X_test, X_test_s, y_test = split['X_test'], split['X_test_s'], split['y_test']

    results = {}

    lr = joblib.load(os.path.join(MODELS_DIR, 'lr_fuel.pkl'))
    results['linear_regression'] = evaluate_regression('Linear Regression', y_test, lr.predict(X_test_s))

    xgb_model = load_xgb_model('xgb_fuel.json')
    results['xgboost'] = evaluate_regression('XGBoost', y_test, xgb_predict(xgb_model, X_test))

    rf = joblib.load(os.path.join(MODELS_DIR, 'rf_fuel.pkl'))
    results['random_forest'] = evaluate_regression('Random Forest', y_test, rf.predict(X_test))

    best_name = min(results, key=lambda k: results[k]['mae'])
    print(f"   -> Best: {best_name} (MAE={results[best_name]['mae']:.4f})")

    try:
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer(X_test[:200])
        joblib.dump(explainer, os.path.join(MODELS_DIR, 'shap_explainer_fuel.pkl'))

        plt.figure()
        shap.summary_plot(shap_values, X_test[:200], feature_names=meta['fuel_features'], show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(REPORT_DIR, 'shap_fuel.png'), dpi=120)
        plt.close()
    except Exception as e:
        print(f"   SHAP skipped: {str(e)[:60]}")

    try:
        plt.figure(figsize=(6, 5))
        preds = xgb_predict(xgb_model, X_test)
        plt.scatter(y_test, preds, alpha=0.3, s=10, color='teal')
        lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
        plt.plot(lims, lims, 'r--', linewidth=1)
        plt.xlabel('Actual fuel (kg)')
        plt.ylabel('Predicted fuel (kg)')
        plt.title('Fuel: Actual vs Predicted (XGBoost)')
        plt.tight_layout()
        plt.savefig(os.path.join(REPORT_DIR, 'fuel_predictions.png'), dpi=120)
        plt.close()
    except Exception as e:
        print(f"   Plot skipped: {str(e)[:60]}")

    _promote(best_name, 'fuel')
    return results, best_name


MODEL_FILE_MAP = {
    'degradation': {
        'linear_regression': ('lr_degradation.pkl', 'scaler_degradation.pkl'),
        'xgboost': ('xgb_degradation.json', None),
        'random_forest': ('rf_degradation.pkl', None),
    },
    'pitstop': {
        'logistic_regression': ('logr_pitstop.pkl', 'scaler_pitstop.pkl'),
        'xgboost': ('xgb_pitstop.json', None),
        'random_forest': ('rf_pitstop.pkl', None),
    },
    'fuel': {
        'linear_regression': ('lr_fuel.pkl', 'scaler_fuel.pkl'),
        'xgboost': ('xgb_fuel.json', None),
        'random_forest': ('rf_fuel.pkl', None),
    },
}


def _promote(best_name, predictor):
    model_file, scaler_file = MODEL_FILE_MAP[predictor][best_name]
    src = os.path.join(MODELS_DIR, model_file)
    ext = os.path.splitext(model_file)[1]
    dst = os.path.join(PRODUCTION_DIR, f'model_{predictor}_BEST{ext}')
    shutil.copy2(src, dst)

    if scaler_file:
        shutil.copy2(
            os.path.join(MODELS_DIR, scaler_file),
            os.path.join(PRODUCTION_DIR, f'scaler_{predictor}_BEST.pkl')
        )


def main():
    print("=" * 70)
    print("STRATEGY MASTER - MODEL EVALUATION")
    print("=" * 70)

    meta = load_metadata()

    deg_results, deg_best = evaluate_degradation(meta)
    pit_results, pit_best = evaluate_pitstop(meta)
    fuel_results, fuel_best = evaluate_fuel(meta)

    selected = {
        'degradation': deg_best,
        'pitstop': pit_best,
        'fuel': fuel_best,
    }
    with open(os.path.join(PRODUCTION_DIR, 'SELECTED_MODELS.json'), 'w') as f:
        json.dump(selected, f, indent=2)

    report = {
        'degradation': deg_results,
        'pitstop': pit_results,
        'fuel': fuel_results,
        'selected_models': selected,
    }
    with open(os.path.join(HERE, '../evaluation_report/evaluation_report.json'), 'w') as f:
        json.dump(report, f, indent=2, default=float)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\nSelected models:")
    print(f"   Degradation: {deg_best}")
    print(f"   Pit Stop:    {pit_best}")
    print(f"   Fuel:        {fuel_best}")
    print(f"\nProduction models in: {PRODUCTION_DIR}")
    print(f"Report + SHAP plots in: {REPORT_DIR}\n")


if __name__ == '__main__':
    main()
