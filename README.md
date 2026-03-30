<<<<<<< HEAD
# forexguard
Real-Time User/Trader Anomaly Detection Engine
=======
# ForexGuard Prototype

Student project for real-time anomaly detection in a forex platform using unsupervised ML.

The pipeline is simple and straightforward:
1. Generate synthetic user/trader events
2. Build per-user behavioral features
3. Train baseline and advanced anomaly models
4. Compare models and expose scoring through FastAPI
5. Add SHAP-based explanations for flagged users
6. Generate LLM compliance alerts for high-risk users

## Quick Start

### 1) Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```


### 2) Run full pipeline
```bash
python -m data.generator
python -m features.engineering
python -m models.isolation_forest
python -m models.lstm_autoencoder
python -m models.compare
python -m explainability.shap_explainer
```

### 3) Start API
```bash
uvicorn api.main:app --reload --port 8000
```

Docs: http://localhost:8000/docs

### 4) Other demos
```bash
python -m streaming.simulator
python -m mlflow_tracking.tracker
```

## Project Layout

- `data/generator.py`: synthetic event generation (~50k events)
- `features/engineering.py`: per-user feature engineering (25 features)
- `models/isolation_forest.py`: Isolation Forest + LOF baseline training
- `models/lstm_autoencoder.py`: LSTM autoencoder training (PyTorch)
- `models/compare.py`: model comparison + simple ensemble
- `explainability/shap_explainer.py`: SHAP explanations for flagged users
- `api/main.py`: FastAPI scoring service
- `api/schemas.py`: API request/response schemas
- `api/llm_alerts.py`: Anthropic LLM alert generation (with fallback template)
- `streaming/simulator.py`: async streaming simulation (batch-based)
- `mlflow_tracking/tracker.py`: optional MLflow logging

## Architecture (High Level)

Event Stream (synthetic CSV) -> Feature Engineering -> Model Scoring -> Explanations -> API/Alerts

The current streaming demo uses asyncio queues. In production, the same flow can move to Kafka topics.

## Model Explanation

### Baseline
- Isolation Forest: Extremely fast; handles huge datasets well; doesn't require data scaling; great out-of-the-box. Struggles to find complex, hidden relationships; ignores the sequence/time of events.

- Local Outlier Factor (LOF): Great at finding subtle anomalies hidden near clusters of normal data. Computationally heavy (slow on massive datasets); hard to pick the right anomaly threshold.

### Advanced
- LSTM Autoencoder: Excellent at capturing time-based patterns and sequences; standard for time-series anomalies. Slow to train; can "forget" patterns if the sequence window is too long.

Note: sequence length is 1 (aggregated user profile, not full event sequence), which is a deliberate simplification for this prototype.

- Transformer: State-of-the-art accuracy; understands complex, long-range context better than any other model. Extremely data-hungry; computationally expensive; highly complex to build from scratch.

- VAE: Handles noisy, chaotic data beautifully; provides actual mathematical probability scores. Can be tricky to train (balancing the loss function); might miss sharp, sudden spikes because it focuses on smoothed probabilities.

I built and tested the LSTM Autoencoder but I couldn't get the time to try out the Transformer and VAE.

## Features Used

25 features grouped into:
- Login behavior (IP/country spread, night login ratio, rapid re-login pattern)
- Financial behavior (deposit/withdraw totals, structuring score, withdrawal anomaly)
- Trading behavior (volume spikes, frequency, instrument diversity)
- Session behavior (session duration and rapid navigation)
- Device/IP behavior (device count, IP switching, mismatch score)
- PnL behavior (volatility and consistency)

## API Endpoints

- `GET /health`: service + model status
- `POST /predict`: score one user profile and return risk level + top features
- `GET /alerts`: top suspicious users from latest scoring file
- `GET /stats`: risk-level distribution and model metrics

## Important Outputs

Generated after running the pipeline:
- `data/raw_events.csv`
- `data/features.csv`
- `data/if_scores.csv`
- `data/lof_scores.csv`
- `data/lstm_scores.csv`
- `data/model_comparison.csv`
- `data/all_scores.csv`
- `data/shap_explanations.csv`

Saved models:
- `models/saved/isolation_forest.pkl`
- `models/saved/lof.pkl`
- `models/saved/lstm_autoencoder.pt`

## Assumptions and Trade-offs

- Data is synthetic because real brokerage data is private.
- Labels are only used for evaluation, not for model training.
- Async simulator is used instead of full Kafka setup to keep the repo lightweight.
- Feature computation is batch-based from CSV, not from a real-time feature store.

### Limitations

- LSTM currently uses aggregated user-level features (sequence length = 1), so temporal sequence behavior is not fully modeled.
- Alert thresholding is fixed and not yet calibrated for false-positive control per risk tier.
- LLM alerts depend on external API availability; fallback templates are used if the API call fails.
- The simulator demonstrates streaming flow, but it is not a production event bus with exactly-once guarantees.

## Docker

```bash
docker-compose up --build
```

- API: http://localhost:8000
- MLflow: http://localhost:5000

>>>>>>> 223484a (Commit)
