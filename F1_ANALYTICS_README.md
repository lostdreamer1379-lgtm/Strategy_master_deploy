# 🏁 F1 Analytics

> **A modular, explainable Formula 1 analytics platform built on FastF1, machine learning, simulation, and interactive race intelligence.**

F1 Analytics is a suite of **six focused analytics products** sharing one common FastF1 data layer, feature-engineering pipeline, model infrastructure, caching layer, and explainability system.

The architecture is designed around a simple principle:

**Raw race data → engineered race intelligence → specialized ML/simulation models → explainable predictions → actionable recommendations**

---

## 🚀 Platform Architecture

```text
                         ┌──────────────────────────┐
                         │     FastF1 Data Layer    │
                         │   Historical + Live-ish  │
                         │       Cached Data        │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │ Shared Feature Pipeline  │
                         │ Telemetry • Laps • Weather│
                         │ Results • Circuits • Fuel │
                         └────────────┬─────────────┘
                                      │
              ┌───────────────────────┼────────────────────────┐
              │                       │                        │
              ▼                       ▼                        ▼
      ┌───────────────┐       ┌───────────────┐       ┌───────────────┐
      │ Strategy      │       │ Driver        │       │ Overtake      │
      │ Master        │       │ Profiler      │       │ Analyzer      │
      └───────────────┘       └───────────────┘       └───────────────┘
              │                       │                        │
              ▼                       ▼                        ▼
      ┌───────────────┐       ┌───────────────┐       ┌───────────────┐
      │ Race Sim      │       │ Setup         │       │ Championship  │
      │ Engine        │       │ Calculator    │       │ Tracker       │
      └───────────────┘       └───────────────┘       └───────────────┘
              │                       │                        │
              └───────────────────────┼────────────────────────┘
                                      ▼
                         ┌──────────────────────────┐
                         │ Explainability Layer    │
                         │ SHAP • Feature Effects  │
                         │ Scenario Analysis        │
                         └────────────┬─────────────┘
                                      ▼
                         ┌──────────────────────────┐
                         │ React Analytics Frontend │
                         │ Charts • Telemetry • UX  │
                         └──────────────────────────┘
```

### Core architecture

Each project contains:

- **3 candidate predictive models**
- **1 selected production model**
- **SHAP/explainability where the selected model supports it**
- Scenario/risk analysis
- A React-based analytical interface
- Shared FastF1 data and caching infrastructure

The three-model approach intentionally compares a simple baseline, a stronger nonlinear/sequence model, and an alternative model before selecting the production approach.

---

# 📊 The Six Products

| Project | Primary Purpose | Predictors | Candidate Models | Production Pick |
|---|---|---:|---|---|
| 🎯 **Strategy Master** | Pit windows, tire degradation, fuel strategy | 4 | Linear, XGBoost, LSTM | **XGBoost** |
| 👥 **Driver Profiler** | Driver style, telemetry, consistency | 4 | Linear, Random Forest, LSTM | **LSTM** |
| 🏁 **Overtake Analyzer** | Overtaking opportunity and pass probability | 2 | Linear, XGBoost, Random Forest | **XGBoost** |
| 🎬 **Race Sim Engine** | Race outcomes and what-if scenarios | 3 | Linear, XGBoost, Neural Net | **XGBoost** |
| 🔧 **Setup Calculator** | Setup-to-pace impact estimation | 1 | Linear, XGBoost, Random Forest | **Linear Regression** |
| 🏆 **Championship Tracker** | Championship probabilities and scenarios | 1 | Linear, Monte Carlo, XGBoost | **Monte Carlo** |

---

# 1. 🎯 Strategy Master

### Purpose

**Real-time pit-stop planning, tire degradation analysis, and fuel strategy.**

The system estimates the optimal pit window by combining tire condition, fuel state, track conditions, race position, and expected pace loss.

### Predictors

1. **Tire Degradation Predictor** — primary
2. **Pit Stop Strategy Optimizer** — primary
3. **Fuel Consumption & Range Predictor** — secondary
4. **Weather Impact Model** — contextual

### Candidate models

- Linear Regression — baseline
- **XGBoost — production model**
- LSTM — sequence alternative

### Why XGBoost?

Pit strategy contains nonlinear interactions such as:

- Tire degradation × compound
- Fuel load × lap time
- Track temperature × tire performance
- Gap × pit-loss cost
- Safety-car probability × pit timing

XGBoost provides a strong balance of nonlinear feature interaction, inference speed, and deployment simplicity.

### Example output

```text
PIT WINDOW
Optimal: Lap 19–22
Recommended: Lap 21
Estimated pit loss: 23.4s

TIRE STATE
Soft degradation: 1.8% / lap
Expected pace loss: +0.8s / lap

FUEL
Current: 45 kg
Estimated consumption: 1.2 kg/lap
Safe range: 32.5 laps

RECOMMENDATION
→ Pit on Lap 21
→ Alternative: Lap 22 if traffic remains clear
```

### Data

- Lap data
- Tire compound
- Tire life/stint length
- Weather
- Telemetry
- Fuel/consumption features
- Track characteristics

---

# 2. 👥 Driver Profiler

### Purpose

**Understand how two drivers differ in pace, braking, throttle application, cornering, sector performance, and consistency.**

### Predictors

1. **Telemetry Comparison**
2. **Sector Performance Breakdown**
3. **Driver Performance Statistics**
4. **Driver Consistency Score**

### Candidate models

- Linear Regression — baseline
- Random Forest — nonlinear alternative
- **LSTM — production model**

### Why LSTM?

Driver telemetry is inherently sequential.

A driver's:

- throttle application
- braking behavior
- speed trace
- corner entry
- apex speed
- corner exit

must be interpreted across a lap and across multiple laps.

The LSTM therefore evaluates temporal patterns rather than treating every telemetry point as independent.

### Example output

```text
HEAD-TO-HEAD

Driver A
├── Braking efficiency: 82%
├── Apex speed: 208 km/h
├── Consistency: 7.9/10
└── Sector 2: +0.5s

Driver B
├── Braking efficiency: 61%
├── Apex speed: 212 km/h
├── Consistency: 8.7/10
└── Sector 2: -0.5s

PROFILE
Driver A → stronger braking/corner entry
Driver B → higher-speed specialist
```

### Data

- `laps.get_telemetry()`
- Throttle
- Brake
- Speed
- RPM
- Gear
- Position coordinates
- Sector times
- Lap times
- Multi-race historical data

### Visualization

- D3.js comparison charts
- Telemetry overlays
- Sector delta charts
- Racing-line visualization
- Optional Three.js track visualization

---

# 3. 🏁 Overtake Analyzer

### Purpose

**Estimate overtaking opportunities using pure race dynamics, gap behavior, relative pace, braking performance, corner exits, and track positioning.**

## ⚠️ 2026 Architecture Change: No DRS Predictor

The Overtake Analyzer **does not use DRS as a predictive feature**.

The previous DRS-effectiveness concept has been removed from the ML architecture.

Instead, the model focuses on the underlying race dynamics:

### Overtake factors

1. **Gap to car ahead**
2. **Relative pace delta**
3. **Braking-zone efficiency**
4. **Corner-exit speed**
5. **Track positioning**

This makes the analyzer centered on the actual conditions required to create an overtaking opportunity rather than relying on DRS availability as a model feature.

### Candidate models

- Linear Regression — baseline
- **XGBoost — production model**
- Random Forest — alternative

### Why XGBoost?

Overtaking probability depends on nonlinear interactions between:

- closing rate
- gap
- speed differential
- braking performance
- corner-exit performance
- circuit characteristics
- track position

XGBoost handles these interactions efficiently and is suitable for frequent prediction updates.

### Example output

```text
OVERTAKE OPPORTUNITY

Current gap:              0.8s
Gap closure:              0.15s/lap
Relative pace delta:     -0.21s/lap

Braking efficiency:       High
Corner-exit advantage:    Moderate
Track positioning:        Favorable

Overtake probability:     84%
Estimated attempt:        Lap 12–14

Recommendation:
→ Maintain pressure through the next sector
→ Prioritize the next braking zone
→ Avoid compromising corner exit
```

### Data

- Lap gaps
- Lap times
- Relative pace
- Telemetry speed traces
- Braking-zone performance
- Corner-exit speed
- Track position
- Circuit characteristics

---

# 4. 🎬 Race Sim Engine

### Purpose

**Simulate race outcomes and evaluate strategic what-if scenarios.**

### Predictors

1. **Race Outcome Predictor**
2. **Qualifying Simulator**
3. **Safety-Car Impact Analyzer**

### Candidate models

- Linear Regression — baseline
- **XGBoost — production predictive model**
- Neural Network — alternative

### Why XGBoost?

Race performance depends on nonlinear interactions involving:

- starting position
- tire strategy
- pace
- fuel
- pit stops
- weather
- safety cars
- track characteristics

XGBoost provides fast inference while remaining easier to deploy than a larger neural architecture.

### Simulation layer

The production system combines ML predictions with scenario simulation.

```text
Historical Data
      ↓
Feature Engineering
      ↓
XGBoost Race Model
      ↓
Scenario Generator
      ↓
Monte Carlo / Batch Simulation
      ↓
Race Outcome Distribution
```

### Example scenarios

```text
BASELINE
VER → P1
LEC → P2
HAM → P3

WHAT IF?
Safety Car at Lap 15
→ LEC gains 2 positions
→ Pit timing advantage increases

WHAT IF?
Rain at Lap 30
→ Intermediate-tire strategy becomes favorable
→ Race outcome distribution changes
```

### Deployment consideration

Large simulations should be treated as asynchronous/batched workloads rather than assuming every simulation fits inside a short serverless request.

---

# 5. 🔧 Setup Calculator

### Purpose

**Estimate how setup changes affect lap time and sector performance.**

### Predictor

1. **Setup Impact Estimator**

### Candidate models

- **Linear Regression — production model**
- XGBoost
- Random Forest

### Why Linear Regression?

The initial model intentionally prioritizes:

- interpretability
- coefficient-level understanding
- fast inference
- engineering usability

The model estimates the relationship between setup parameters and measured pace while allowing the more complex models to serve as validation alternatives.

### Example

```text
SETUP COMPARISON

Parameter          Setup A    Setup B
--------------------------------------
Front Wing            4          6
Rear Wing             3          4
Brake Bias           55.2       56.0
Suspension          Soft       Stiff

ESTIMATED IMPACT

Front Wing       +0.8s
Rear Wing        +0.3s
Brake Bias       +0.1s
Suspension       -0.4s
--------------------------------------
NET EFFECT       +0.8s

→ Setup A estimated faster
```

### Data

- Setup configurations
- Lap times
- Sector times
- Telemetry
- Track conditions
- Historical setup/pace relationships

---

# 6. 🏆 Championship Tracker

### Purpose

**Estimate championship probabilities and identify the races and scenarios that matter most.**

### Predictor

1. **Championship Points / Outcome Predictor**

### Candidate approaches

- Linear Regression — baseline
- XGBoost — predictive alternative
- **Monte Carlo — production simulation**

### Why Monte Carlo?

A championship is not a single deterministic prediction.

It is a distribution of possible future outcomes.

The engine can simulate thousands of combinations of:

- finishing positions
- driver pace
- race results
- scenario outcomes
- remaining races
- championship points

### Example

```text
CHAMPIONSHIP PROJECTION

Driver       Win Probability    Expected Points
------------------------------------------------
VER              82%                450
LEC              15%                428
HAM               3%                405

KEY SCENARIOS

VER wins next 2 races
→ Championship probability increases

LEC wins + VER scores 0
→ Championship gap collapses

KEY RACES
→ Highest-impact remaining events
→ Biggest probability swings
```

### Simulation

```text
Current Standings
       ↓
Remaining Calendar
       ↓
Race Outcome Distributions
       ↓
10,000+ Championship Simulations
       ↓
Win / Podium / Points Probabilities
```

---

# 🧠 Explainability Architecture

Explainability is a **platform-wide requirement**, but the implementation depends on the selected model.

## Tree-based models

For XGBoost / Random Forest:

- SHAP TreeExplainer
- Local feature contributions
- Global feature importance
- Waterfall plots
- Dependence plots

## Linear models

For Linear Regression:

- Model coefficients
- Standardized feature effects
- Local contribution analysis

SHAP can also be used where appropriate, but coefficients remain the primary engineering explanation.

## LSTM / Neural models

For sequence models:

- Sequence-aware SHAP/explanation methods
- Feature attribution across telemetry sequences
- Temporal contribution visualization

A generic `shap.TreeExplainer()` should **not** be used for LSTM or Linear Regression models.

## Monte Carlo systems

For simulation-driven products:

- Scenario sensitivity
- Feature perturbation
- Probability delta
- Race-impact analysis

SHAP is not the primary explanation mechanism for the simulation itself.

---

# 🔍 Standard Prediction Response

Every project should expose a consistent explanation contract.

```json
{
  "prediction": "...",
  "confidence": 0.94,
  "top_factors": [
    {
      "factor": "Tire Degradation Rate",
      "contribution": 1.2,
      "direction": "increases pit urgency"
    },
    {
      "factor": "Current Gap",
      "contribution": 0.8,
      "direction": "increases pit urgency"
    }
  ],
  "recommendation": "...",
  "alternatives": [],
  "risk": "medium"
}
```

### Standard UI

```text
┌──────────────────────────────────────┐
│ PREDICTION                           │
│ Pit Lap 21                           │
│ Confidence: 94%                      │
├──────────────────────────────────────┤
│ WHY?                                 │
│                                      │
│ ↑ Tire degradation       +1.2s       │
│   Major                                │
│                                      │
│ ↑ Current gap            +0.8s       │
│   High                                 │
│                                      │
│ ↑ Track temperature      +0.3s       │
│   Medium                               │
│                                      │
│ ↓ Fuel level             -0.5s       │
│   Medium                               │
├──────────────────────────────────────┤
│ SHAP / Attribution Waterfall         │
│ Feature Importance                   │
│ Scenario Comparison                  │
├──────────────────────────────────────┤
│ RECOMMENDATION                       │
│ → Pit on Lap 21                      │
│                                      │
│ ALTERNATIVE                          │
│ → Delay to Lap 22 if gap remains     │
│                                      │
│ RISK: Medium                         │
└──────────────────────────────────────┘
```

---

# 🏗️ Shared Data Architecture

All six products use the same underlying data infrastructure.

```text
FastF1
  ↓
Session Loader
  ↓
Data Normalization
  ↓
Feature Engineering
  ↓
Redis Cache
  ↓
Project-specific Feature Pipeline
  ↓
Model / Simulation
```

### Core FastF1 sources

Depending on the project:

- `session.laps`
- `session.laps.get_telemetry()`
- `session.weather_data`
- `session.results`
- Session metadata
- Circuit information
- Historical race/session data

### Shared cache

```python
class F1CacheService:

    def get_session_data(circuit, season, session_type):
        key = f"f1:{circuit}:{season}:{session_type}"

        cached = redis.get(key)

        if cached:
            return cached

        data = fastf1.get_session(
            season,
            circuit,
            session_type
        )

        data.load()

        redis.set(key, data, ttl=86400)

        return data
```

Projects should consume the shared cache rather than independently downloading the same FastF1 sessions.

---

# 📁 Repository Architecture

```text
f1-analytics/
│
├── apps/
│   ├── strategy-master/
│   │   ├── api/
│   │   ├── src/
│   │   └── models/
│   │
│   ├── driver-profiler/
│   │   ├── api/
│   │   ├── src/
│   │   └── models/
│   │
│   ├── overtake-analyzer/
│   │   ├── api/
│   │   ├── src/
│   │   └── models/
│   │
│   ├── race-sim-engine/
│   │   ├── api/
│   │   ├── src/
│   │   └── models/
│   │
│   ├── setup-calculator/
│   │   ├── api/
│   │   ├── src/
│   │   └── models/
│   │
│   └── championship-tracker/
│       ├── api/
│       ├── src/
│       └── models/
│
├── shared/
│   ├── fastf1/
│   ├── cache/
│   ├── features/
│   ├── explainability/
│   ├── circuits/
│   └── constants/
│
├── training/
│   ├── strategy/
│   ├── driver/
│   ├── overtake/
│   ├── race/
│   ├── setup/
│   └── championship/
│
└── docs/
    ├── architecture.md
    ├── api.md
    └── modeling.md
```

---

# 🤖 Model Selection Matrix

| Project | Baseline | Alternative | Primary | Why |
|---|---|---|---|---|
| Strategy Master | Linear | LSTM | **XGBoost** | Nonlinear pit/tire/fuel interactions |
| Driver Profiler | Linear | Random Forest | **LSTM** | Temporal telemetry patterns |
| Overtake Analyzer | Linear | Random Forest | **XGBoost** | Gap/speed/braking interactions |
| Race Sim Engine | Linear | Neural Net | **XGBoost** | Fast nonlinear race prediction |
| Setup Calculator | XGBoost | Random Forest | **Linear** | Interpretability and baseline relationship |
| Championship Tracker | Linear | XGBoost | **Monte Carlo** | Probabilistic future outcomes |

> **Important:** The "best model" is selected through validation metrics, not assumed solely from the architecture. The listed production pick is the target architecture; validation should confirm it.

---

# 📈 Evaluation Framework

Each project should use metrics appropriate to its prediction task.

### Regression

- MAE
- RMSE
- R²

### Classification

- ROC-AUC
- PR-AUC
- F1
- Precision / Recall
- Brier score
- Calibration

### Sequence models

- MAE / RMSE for continuous targets
- Sequence-level validation
- Temporal holdout testing

### Probability / simulation

- Calibration error
- Brier score
- Log loss
- Distribution comparison
- Scenario sensitivity

### Critical validation rule

**Avoid random leakage across laps from the same race/session.**

Where appropriate:

```text
TRAIN → Earlier races / seasons
VALIDATION → Later races
TEST → Completely held-out races
```

This gives a much more realistic measure of whether the system can generalize to unseen races.

---

# 🔄 Standard Project Pipeline

Every project follows the same high-level lifecycle:

```text
1. Collect
   ↓
2. Clean
   ↓
3. Normalize
   ↓
4. Engineer features
   ↓
5. Split by race/session/time
   ↓
6. Train 3 candidate models
   ↓
7. Evaluate
   ↓
8. Select production model
   ↓
9. Build explanation layer
   ↓
10. Expose API
   ↓
11. Build React interface
   ↓
12. Deploy
   ↓
13. Monitor predictions
```

---

# ☁️ Deployment Architecture

### Frontend

- React
- Vite / Next.js where appropriate
- Plotly
- D3.js
- Canvas for live visualizations
- Three.js for advanced track/telemetry visualization

### Backend

- Python model services
- FastF1 data processing
- Serverless API endpoints where suitable
- Redis/KV caching
- Background/asynchronous jobs for expensive simulations

### Shared infrastructure

```text
                    ┌───────────────┐
                    │   React Apps  │
                    └───────┬───────┘
                            │
                    ┌───────▼───────┐
                    │ API / Backend │
                    └───────┬───────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
          ML Models     Explainability   Cache
              │             │             │
              └─────────────┼─────────────┘
                            ▼
                      FastF1 Data
```

### Authentication

A shared authentication layer should provide:

- One account
- Access to all six applications
- Shared user preferences
- Shared prediction history

---

# ⚡ Compute Strategy

Not every workload should run synchronously.

### Low compute

- Setup Calculator
- Championship dashboard updates
- Simple feature predictions

### Medium compute

- Strategy Master
- Driver Profiler
- Overtake Analyzer

### High compute

- Race Sim Engine
- Large Monte Carlo championship simulations

For expensive workloads:

```text
User Request
    ↓
Create Simulation Job
    ↓
Queue / Background Worker
    ↓
Run Simulation
    ↓
Cache Result
    ↓
Frontend Polls / Receives Result
```

This avoids forcing long simulations through short serverless request limits.

---

# 📱 Product UX Principles

All six applications should share a consistent design language.

### Primary principles

- Dark, telemetry-inspired interface
- High information density without visual clutter
- Clear hierarchy
- Mobile-responsive layouts
- Explain predictions instead of only displaying numbers
- Make uncertainty visible
- Always provide actionable interpretation

### Every major prediction should answer:

1. **What is predicted?**
2. **How confident is the system?**
3. **Why did it predict this?**
4. **What should the user do?**
5. **What happens under alternative scenarios?**
6. **What is the risk?**

---

# 🗺️ Development Roadmap

| Phase | Project | Priority | Target |
|---|---|---|---|
| W1–2 | 🎯 Strategy Master | P1 | Real-time pit strategy |
| W3–4 | 👥 Driver Profiler | P2 | Head-to-head telemetry |
| W5–6 | 🏁 Overtake Analyzer | P3 | DRS-independent overtaking intelligence |
| W7–10 | 🎬 Race Sim Engine | P4 | Scenario/race simulation |
| W11 | 🔧 Setup Calculator | Quick Win | Setup impact model |
| W12 | 🏆 Championship Tracker | Integration | Championship dashboard |

---

# 💰 Optional Monetization Architecture

The six applications can operate independently while sharing one account and infrastructure.

### FREE

- Limited daily predictions
- Current-race data
- Basic explanations
- Limited history

### PRO

- Unlimited predictions
- Historical data
- Real-time alerts
- Advanced explanations
- No ads

### EXPERT / ALL-ACCESS

- All six applications
- Extended historical analysis
- CSV/PDF exports
- Advanced scenario simulation
- API access

---

# 🏁 Definition of Done

A project is considered complete when it has:

```text
DATA
□ FastF1 ingestion
□ Feature engineering
□ Cached data
□ Historical validation dataset

ML
□ 3 candidate models trained
□ Baseline established
□ Evaluation completed
□ Production model selected
□ Model version recorded

EXPLAINABILITY
□ Local prediction explanation
□ Global feature analysis
□ Appropriate attribution method
□ Scenario/risk interpretation

PRODUCT
□ React interface
□ Responsive/mobile UI
□ Prediction API
□ Loading/error states
□ Confidence/uncertainty display
□ Recommendation layer

DEPLOYMENT
□ Production deployment
□ Cache configured
□ Model artifacts versioned
□ API documented
□ Basic monitoring
□ End-to-end testing
```

---

# 🧩 Final Architecture

```text
                         🏎️ F1 ANALYTICS
                              │
                              ▼
                     ┌─────────────────┐
                     │    FastF1       │
                     │  Shared Data    │
                     └────────┬────────┘
                              ▼
                     ┌─────────────────┐
                     │ Feature Engine  │
                     └────────┬────────┘
                              ▼
       ┌──────────────┬───────┴────────┬──────────────┐
       ▼              ▼                ▼              ▼
   STRATEGY        DRIVER          OVERTAKE        RACE SIM
   MASTER          PROFILER        ANALYZER        ENGINE
       │              │                │              │
       └──────────────┴───────┬────────┴──────────────┘
                              ▼
                    ┌──────────────────┐
                    │ SETUP CALCULATOR │
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │ CHAMPIONSHIP     │
                    │ TRACKER          │
                    └────────┬─────────┘
                             ▼
                  ┌──────────────────────┐
                  │ EXPLAINABILITY       │
                  │ SHAP / Attribution   │
                  │ Sensitivity / Risk   │
                  └──────────┬───────────┘
                             ▼
                  ┌──────────────────────┐
                  │ REACT PRODUCT LAYER │
                  │ Prediction → Why →  │
                  │ Action → Scenario   │
                  └──────────────────────┘
```

## 🎯 Vision

F1 Analytics is not intended to be six disconnected dashboards.

It is a **single analytical platform expressed through six specialized products**.

The shared data layer keeps the system efficient.  
The three-model architecture makes model selection measurable.  
The explainability layer makes predictions understandable.  
The scenario engines turn predictions into decisions.

The end goal is a system that answers not only:

> **“What is likely to happen?”**

but also:

> **“Why is it likely to happen, what could change it, and what should we do about it?”**

---

## ⚠️ Important Architecture Notes

- **DRS is intentionally excluded from the Overtake Analyzer's predictive feature set.**
- The Overtake Analyzer therefore focuses on **gap, relative pace, braking-zone efficiency, corner-exit speed, and track positioning**.
- **SHAP implementation is model-specific**; `TreeExplainer` should only be used for compatible tree models.
- Monte Carlo projects use **probabilistic sensitivity/scenario analysis** rather than pretending that SHAP is universally applicable.
- Model selections are architectural targets and must ultimately be confirmed through proper validation.
- Historical race/session splits should be used to reduce temporal leakage.
