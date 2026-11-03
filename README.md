# Aletheia — Explainable AI & Model Governance

A governance platform for machine-learning models that were not built here. It registers an external
artifact, scores with it, explains it, audits it for fairness and drift, and records everything with enough
lineage that a reviewer can reconstruct how a number was produced.

The product's central rule: **an explanation that does not reconstruct the model's own output is not
published.** Additivity is checked, not assumed, and a failure is an error rather than a footnote.

---

## Status

| Area | State |
|---|---|
| Stage 0 — discovery, requirements, architecture | **Complete** (11 documents, `docs/00`–`09` + remediation register `docs/10`) |
| Stage 1a — foundation: config, logging, error model, health, CI, container | **Complete and verified** |
| Stage 1b — walking skeleton: artifact → prediction → SHAP → additivity → grouped view | **Core complete** (CLI path). The single dashboard page is **not built** |
| Stages 2–14 — training CLI, registry, evaluations, global explanations, fairness, drift, persistence, dashboard, hardening | **Not started** |
| Delivered software | ~1,100 lines of Python, 111 tests, no database yet |

Nothing above claims more than it can show. Where a check was not run, it says so below.

---

## What is implemented

- **Configuration** (`core/config.py`) — typed settings, production refuses to start without a database URL
  and signing key, worker count bounded by measurement rather than core count.
- **Structured logging** (`core/logging.py`) — JSON in deployed environments, request ids bound into context.
- **Error taxonomy** (`core/errors.py`) — stable machine-readable codes rendered as RFC 9457 problem
  documents; no stack traces or credentials in responses.
- **Health** (`core/health.py`, `api/routers/system.py`) — liveness independent of dependencies, readiness
  that reports *why* it is not ready and never echoes a connection string.
- **Composite artifact adapter** (`ml/adapters/sklearn_composite.py`) — loads the
  `{model, preprocessor, feature_names, metadata}` shape used by an existing churn model, including its
  known quirk of declaring feature names *without* scikit-learn's `num__`/`cat__` prefixes. Names are
  validated positionally against the fitted transformer; a mismatch is a rejection with the offending index.
- **Feature map** (`ml/feature_map.py`) — transformed column → source field, so a one-hot contribution can
  be shown as `Contract = Month-to-month` and aggregated back to one field, explicitly labelled as an
  aggregation of the same attributions.
- **Local SHAP** (`ml/explain/shap_local.py`) — Linear / Tree / Kernel, chosen from an allowlist, never
  guessed. Contributions are returned in the model's output space (`log_odds`), never as probability changes.
- **Additivity gate** (`ml/explain/additivity.py`) — `|base + Σφ − output| ≤ tolerance` on every explanation;
  failing explanations raise instead of being returned.
- **API shell** (`api/app.py`) — request ids, access logs, problem documents, `/health`, `/health/ready`,
  `/api/v1/meta` declaring capabilities, limits and caveats.
- **Engineering scaffolding** — pytest markers and fixtures, ruff, strict mypy on `core` + `ml`, an
  AST-based architecture test that fails on a layer violation, multi-stage Dockerfile, Compose stack, CI
  with lint, types, tests, a coverage floor, `pip-audit` and an SBOM.

## Verified on this machine

Every figure below came from a command run in this repository, not from a plan.

| Check | Command | Result |
|---|---|---|
| Tests | `pytest tests` | **111 passed**, 0 failed |
| Coverage (ML core) | `pytest tests/unit --cov=aletheia.ml --cov=aletheia.core` | **94%** — 562 statements, 26 missed |
| Lint | `ruff check .` | clean |
| Types | `mypy` | clean, 15 source files |
| Liveness | `curl /health` | `200 {"status":"ok","version":"0.1.0"}` |
| Readiness | `curl /health/ready` | `503 application/problem+json`, `database: not_configured` |
| Capabilities | `curl /api/v1/meta` | `200`, capabilities/limits/caveats |
| Skeleton path | `python scripts/demo_explain.py` | additivity `2.22e-16` (tolerance `1e-06`), `PASS` |
| Container image | `docker compose build api` | built, `1.14 GB`, runtime user `aletheia` (non-root) |
| Container probes | `curl :8010/health` against the image | `200 {"status":"ok","version":"0.1.0"}`; readiness `503` without a database |

**Not verified here, and therefore not claimed:** the GitHub Actions workflow (never executed on GitHub),
`pip-audit` and the SBOM step (not run locally). The Stage 1b dashboard page does not exist yet.

## Known gaps

- **The image is 1.14 GB**, dominated by `shap` → `numba` → `llvmlite` plus `scipy`. Acceptable for a
  governance service that explains models in-process; a split worker image is the obvious next step.
- **No persistence.** There is no database schema applied yet, so `/health/ready` correctly reports
  `not_configured` and nothing is stored. Stage 3 adds migrations and the registry.
- **Kernel SHAP is approximate.** Its additivity gate runs at `5e-3` instead of `1e-6`, and the tolerance
  travels with the result rather than being hidden.
- **One artifact family is proven.** `sklearn-composite-v1` is tested; the in-house XGBoost that would prove
  generality is Stage 2.

## Quickstart

```bash
python -m venv .venv
source .venv/Scripts/activate          # Windows (Git Bash); use .venv/bin/activate elsewhere
pip install -e ".[dev]"
cp .env.example .env                   # local values only; never commit .env

PYTHONPATH=src python scripts/demo_explain.py       # artifact → prediction → SHAP → additivity
PYTHONPATH=src uvicorn aletheia.api.app:app --port 8000
curl http://127.0.0.1:8000/health
```

The demo trains a small synthetic model on first run and writes it to `artifacts/` (git-ignored). It prints
the probability, the model output in log-odds, the additivity residual, the top contributions, the grouped
view and the caveats — the same code path the API will use, with no database required.

With Docker:

```bash
docker compose up --build              # Postgres 16 + API on :8000
```

## Layout

```
src/aletheia/
  core/      config, logging, errors, health probes          (imports nothing from the app)
  ml/        schema, feature map, adapters, explainers       (no HTTP, no database, no framework)
  api/       FastAPI factory, routers                        (composes core + ml)
tests/unit/  111 hermetic tests, no network, no binary fixtures
docs/        the design package (00–10)
scripts/     demo path, runnable without infrastructure
docker/      multi-stage image; docker-compose.yml at the root
```

The layer rule is enforced by `tests/unit/test_architecture.py`, which parses every module and fails on an
import that points backwards. A rule that lives only in a diagram is a suggestion.

## Design package

| Doc | Contents |
|---|---|
| [`docs/00`](docs/00-stage-0-report.md) | Stage 0 report, gate result, verified findings, open questions |
| [`docs/01`](docs/01-product-requirements.md) | Personas, use cases, 61 functional + 14 non-functional requirements |
| [`docs/02`](docs/02-existing-artifact-assessment.md) | Measured assessment of the reused churn artifact |
| [`docs/03`](docs/03-system-architecture.md) | Layers, contracts, model-load path, security, execution model |
| [`docs/04`](docs/04-database-design.md) | 22-table ER model, constraints, indexes, retention |
| [`docs/05`](docs/05-data-flow.md) | Eleven flows including request lifecycle and cache lifecycle |
| [`docs/06`](docs/06-api-specification.md) | Endpoints, error catalogue, rate limits, labels interface |
| [`docs/07`](docs/07-technology-decisions.md) | 18 ADRs with rejected alternatives and revisit triggers |
| [`docs/08`](docs/08-risk-assessment.md) | 40 scored risks, residual risk, accepted risks |
| [`docs/09`](docs/09-development-plan.md) | Stages, gates, effort with contingency |
| [`docs/10`](docs/10-review-remediation.md) | Design-review findings and how each was closed |

## Principles the code enforces

1. **No fabricated results.** Numbers come from an executed script. Examples in the API spec are labelled
   illustrative.
2. **Association is not causation.** Every explanation carries caveats saying so.
3. **SHAP values are not probabilities.** The output space travels with every value.
4. **Fairness is reported, not adjudicated.** Group metrics include sample sizes and limitations.
5. **Drift is not degradation.** Performance snapshots require labels and are nullable by schema.
6. **Untrusted artifacts are handled as untrusted.** The API process never unpickles a model; loading is
   worker-side, hash-pinned and size-capped.

## Open decisions for the project owner

1. **Fairness depth.** The Telco dataset has no race or real age field, so a strong fairness audit is not
   possible on it. Add a second, richer dataset, or state the limitation as the v1 boundary? Current answer:
   state it, and score the risk high.
2. **Second model family.** Training an in-house XGBoost (Stage 2) is what proves the adapter generalises
   beyond a single logistic regression. Worth the extra stage?
3. **Scope of the dashboard.** Eight pages is a large surface for one developer. A three-page vertical slice
   (prediction, local explanation, global overview) is the smaller alternative.

## Licence

Proprietary. The reused churn artifact and its dataset are not committed here; artifacts are git-ignored.
