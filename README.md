# Aletheia — Explainable AI & Model Governance Platform

A production-grade governance platform for machine-learning models. It registers external
artifacts, scores with them, explains them with mathematically verified SHAP attributions, audits outcome
calibration and fairness over time, and records an immutable audit trail with cryptographic lineage so a
reviewer can reconstruct how any score or attribution was produced.

The platform's central rule: **an explanation that does not reconstruct the model's own output is not
published.** Additivity ($\left|\text{base} + \sum \phi_i - f(x)\right| \le \epsilon$) is strictly checked before
publication, and a fidelity violation is an error rather than a footnote.

---

## Status

| Area | State |
|---|---|
| Stage 0 — discovery, requirements, architecture | **Complete** (11 documents, `docs/00`–`09` + remediation register `docs/10`) |
| Stage 1a — foundation: config, logging, error model, health, CI, container | **Complete and verified** |
| Stage 1b — walking skeleton: artifact → prediction → SHAP → additivity → grouped view | **Complete and verified** |
| Stage 2 — Model Registry & Version Lifecycle | **Complete**: database persistence, SHA-256 verification, positional transformer checks |
| Stage 3 — Scoring & Inference Service | **Complete**: single & batch prediction endpoints, threshold evaluation, subject pseudonymisation |
| Stage 4 — Local Explainability & Additivity Gate | **Complete**: Linear / Tree / Kernel SHAP with strict fidelity checks, content-addressed caching |
| Stage 5 — Late-Arriving Labels & Outcome Calibration | **Complete**: ground truth outcome ingestion, empirical accuracy, Brier scores, calibration gap |
| Stage 6 — Governance Audit Ledger | **Complete**: immutable audit event trail tracking all sensitive operations |
| Stage 7 — Interactive Governance Dashboard | **Complete**: multi-page Streamlit dashboard (`src/aletheia/dashboard/app.py`) |
| Delivered software | **~3,200 lines of Python across 43 source files, 119 hermetic tests (100% pass), strict mypy, clean ruff** |

---

## What is implemented

- **Model Registry & Versioning** (`db/models.py`, `services/model_service.py`, `api/routers/models.py`):
  Registers models and versioned artifacts. Pins SHA-256 digests, caps file sizes, extracts framework versions,
  decision thresholds, and maps transformer feature spaces.
- **Scoring & Prediction API** (`services/prediction_service.py`, `api/routers/predictions.py`):
  Single (`POST /api/v1/models/{id}/predict`) and batch (`POST /api/v1/models/{id}/predict/batch`) scoring with
  automatic subject ID pseudonymisation (SHA-256 hashing) and threshold-based label classification.
- **Local SHAP with Strict Additivity Gate** (`ml/explain/shap_local.py`, `api/routers/explanations.py`):
  Evaluates local contributions in the model's native output space (`log_odds` or `probability`).
  Enforces $\left|\text{base} + \sum \phi_i - \text{output}\right| \le 10^{-6}$; unfaithful explanations raise
  `ExplanationFidelityError` rather than returning unverified numbers.
- **Content-Addressed Explanation Caching** (`services/explanation_service.py`):
  Hashes `(artifact_sha256, sorted_features, explainer_type)` to instantly serve identical explanation requests
  from the database without redundant CPU cycles.
- **Late-Arriving Labels & Outcome Calibration** (`services/label_service.py`, `api/routers/labels.py`):
  Ingests ground truth outcomes as they arrive (`POST /api/v1/models/{id}/labels`) and calculates empirical accuracy,
  precision, recall, F1, Brier score ($\frac{1}{N}\sum(p_i - y_i)^2$), and calibration gap (`mean_prob - prevalence`).
- **Governance Audit Ledger** (`db/models.py`, `api/routers/audit.py`):
  Records immutable timestamped events for model creation, version registration, and label ingestion.
- **Interactive Streamlit Dashboard** (`src/aletheia/dashboard/app.py`, `scripts/run_dashboard.py`):
  A full 3-page vertical slice providing:
  1. *Model Governance & Lineage*: cryptographic hashes, schema metadata, framework versions, audit trail.
  2. *Live Scoring & Local SHAP*: real-time input sliders, probability vs threshold, live Additivity Gate Pass badge, Plotly attribution charts, and grouped feature views.
  3. *Fairness & Calibration Audit*: late-arriving label registration form, Brier score, and reliability diagrams.
- **Database Layer** (`db/connection.py`, `db/models.py`, `db/repositories.py`):
  SQLAlchemy 2.0 dual-mode engine supporting PostgreSQL 16 in production and SQLite in local/testing environments.
- **Error Taxonomy** (`core/errors.py`):
  RFC 9457 `application/problem+json` standard with stable machine-readable error codes. No leaked credentials or stack traces.
- **Architectural Scaffolding & AST Gate** (`tests/unit/test_architecture.py`):
  Static AST-based unit test enforcing strict layer separation (`core` $\leftarrow$ `ml` $\leftarrow$ `db` $\leftarrow$ `services` $\leftarrow$ `api` $\leftarrow$ `dashboard`).

---

## Verified on this machine

| Check | Command | Result |
|---|---|---|
| Tests | `pytest tests` | **119 passed**, 0 failed |
| Lint | `ruff check .` | **clean** (0 errors) |
| Types | `mypy` | **clean** (43 source files, strict mode) |
| Architecture Guard | `pytest tests/unit/test_architecture.py` | **3 passed**, 0 layer violations |
| REST API Contracts | `pytest tests/unit/test_api_endpoints.py` | **passed**, full end-to-end HTTP lifecycle |
| Liveness Probe | `GET /health` | `200 {"status":"ok","version":"0.1.0"}` |
| Readiness Probe | `GET /health/ready` | `200 {"status":"ready","checks":{"database":{"ok":true}}}` |
| Additivity Residual | `python scripts/demo_explain.py` | `2.22e-16` (tolerance `1e-06`), **PASS** |

---

## Quickstart

### 1. Setup Environment
```bash
python -m venv .venv
source .venv/Scripts/activate          # Windows Git Bash (or .venv\Scripts\Activate.ps1 in PowerShell)
pip install -e ".[dev,dashboard]"
```

### 2. Launch the Interactive Dashboard
```bash
python scripts/run_dashboard.py
# Or directly via streamlit:
streamlit run src/aletheia/dashboard/app.py
```
Open your browser at `http://localhost:8501`.

### 3. Run the FastAPI REST Server
```bash
uvicorn aletheia.api.app:app --port 8000 --reload
```
- API Docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`
- Capabilities discovery: `http://localhost:8000/api/v1/meta`

### 4. Run the Full Test Suite
```bash
pytest tests -v
ruff check .
mypy
```

---

## API Summary

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe (process up) |
| `GET` | `/health/ready` | Readiness probe (database connectivity) |
| `GET` | `/api/v1/meta` | Capability and limit discovery |
| `GET` | `/api/v1/models` | List all registered models |
| `POST` | `/api/v1/models` | Register a new model entity |
| `GET` | `/api/v1/models/{id}` | Get model details and version history |
| `POST` | `/api/v1/models/{id}/versions` | Register a new version artifact |
| `POST` | `/api/v1/models/{id}/predict` | Score single observation |
| `POST` | `/api/v1/models/{id}/predict/batch` | Batch score multiple observations |
| `POST` | `/api/v1/models/{id}/explain` | Compute local SHAP with additivity verification |
| `POST` | `/api/v1/models/{id}/labels` | Ingest late-arriving ground truth outcome |
| `GET` | `/api/v1/models/{id}/metrics` | Outcome calibration, accuracy, and Brier score |
| `GET` | `/api/v1/audit` | Query immutable governance audit trail |

---

## Layout

```
src/aletheia/
  core/        config, logging, errors, health probes
  ml/          schema, feature map, adapters, explainers (strict: no HTTP, no DB)
  db/          SQLAlchemy 2.0 models, connection pooling, typed repositories
  schemas/     Pydantic v2 request & response schemas
  services/    business services (models, predictions, explainability, labels, audit)
  api/         FastAPI app factory, middleware, problem handlers, routers
  dashboard/   multi-page Streamlit governance & explainability dashboard
tests/unit/    119 hermetic tests (architecture, ML, database, services, REST API)
docs/          complete design package (00–10)
scripts/       demo execution & dashboard launcher
```

---

## Licence

Proprietary.
"# Explainable-AI-Dashboard" 
