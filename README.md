# Strategy Master

F1 race-strategy predictor — tyre degradation, pit window and fuel-consumption
models trained on real FastF1 data, served through a Vercel serverless API,
with a React timing-tower UI (dark/light theme) and SHAP explainability.

Project 1 of the 6-part F1 Analytics Platform.

---

## What it predicts

| Predictor | Type | Models compared | Output |
|---|---|---|---|
| Tyre degradation | regression | Linear Regression, XGBoost, Random Forest | % pace loss per lap |
| Pit stop timing | classification | Logistic Regression, XGBoost, Random Forest | probability of pitting next lap |
| Fuel consumption | regression | Linear Regression, XGBoost, Random Forest | kg remaining, laps of fuel left |

The best of the 3 candidate models per predictor (by holdout MAE for
regression, AUC for classification) is auto-promoted to `models/production/`
and is what the live API serves.

---

## Project structure

```
strategy-master/
├── training/
│   ├── circuits_config.json     # 22 circuits — corner ratio, abrasiveness, speed, length
│   ├── compounds_config.json    # 5 tyre compounds — durability, grip, opt. temp
│   ├── collect_data.py          # FastF1 data pipeline (cross-platform timeout)
│   ├── engineer_features.py     # encoding, rolling stats, interaction terms
│   ├── train_models.py          # trains all 9 candidate models
│   └── evaluate_models.py       # scores, SHAP plots, promotes best to production/
├── api/
│   ├── predict.py               # POST /api/predict — Vercel serverless function
│   └── utils.py                 # feature-vector builder (reads the same JSON configs)
├── models/
│   └── production/              # auto-populated — best model per predictor
├── src/
│   ├── components/              # Header, Selectors, 3 readout cards, chart, strategy, SHAP
│   ├── App.jsx
│   └── App.css                  # design tokens, dark/light theme
├── evaluation_report/           # auto-generated SHAP plots + metrics JSON
├── vercel.json
├── requirements.txt
└── package.json
```

---

## Run the training pipeline

```bash
cd training
pip install -r ../requirements.txt

python collect_data.py        # pulls race data from FastF1 (~20-40 min for 4 seasons)
python engineer_features.py   # builds the model-ready feature set
python train_models.py        # trains all 9 models
python evaluate_models.py     # evaluates, generates SHAP plots, promotes best models
```

`collect_data.py` is resilient by design: each race session loads under a
90-second thread-based timeout (works on Windows/macOS/Linux — no `signal`
module dependency), and if a session's weather data can't be fetched it
falls back to sensible defaults rather than failing the whole run. A race
that can't be loaded is skipped and logged, not fatal.

Circuit and compound characteristics live in `training/circuits_config.json`
and `training/compounds_config.json` — edit these to add circuits or update
tyre-compound assumptions without touching any code. `api/utils.py` reads
the *same* files, so there's one source of truth between training and
inference.

---

## Run the frontend locally

```bash
npm install
npm start
```

`npm start` runs the React dev server and the local Python API together. The
frontend proxies `/api/predict` to the Python process, so it uses the locally
trained models in `models/production/` instead of the browser demo heuristic.
Install the Python dependencies first:

```bash
cd training
pip install -r ../requirements.txt
cd ..
npm start
```

The browser footer should show the model sources, for example
`models: deg=random_forest · pit=xgboost · fuel=xgboost`. If you only want to
run the React frontend without the local Python API, use
`npm run start:frontend`; it will fall back to demo mode.

---

## Deploy

```bash
vercel login
vercel --prod
```

`.vercelignore` excludes the local FastF1 cache, generated datasets, training
artifacts, and non-production models from the upload. The deployment keeps the
promoted models in `models/production/` and the shared training configuration
files required by the API.

Vercel builds the React app and deploys `api/predict.py` as a serverless
function automatically per `vercel.json`. Make sure `models/production/`
contains your trained models before deploying (it's gitignored — copy it
up or adjust `.gitignore` if you want models in version control).

---

## API contract

```
POST /api/predict
Content-Type: application/json

{
  "circuit": "Silverstone",
  "compound": "MEDIUM",
  "tyre_life": 12,
  "stint_lap": 12,
  "lap_number": 24,
  "track_temp": 34,
  "air_temp": 23,
  "humidity": 48,
  "wind_speed": 2,
  "rainfall": 0,
  "fuel_kg": 62,
  "total_laps": 52
}
```

Returns predictions for all 3 models, a synthesized strategy with 3 ranked
options, SHAP top-5 factors per predictor, and which model (LR/XGBoost/RF)
is serving each prediction.

---

## UI notes

The interface is built around a timing-tower metaphor rather than generic
dashboard cards — big tabular-figure numbers are the focus, with a single
accent color (F1 red) spent deliberately on the primary recommendation,
teal for "optimal" states, and amber for caution. Toggle dark/light mode
top-right; the preference is remembered across visits.
