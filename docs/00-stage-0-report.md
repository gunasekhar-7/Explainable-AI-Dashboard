# Stage 0 Completion Report — Discovery, Requirements and Architecture

| Field | Value |
|---|---|
| Stage | 0 — Product Discovery, Requirements and Architecture |
| Date | 2026-11-03 |
| Status | **Complete, pending architecture validation** |
| Implementation started | **No** (deliberate: gate requires approval) |

---

## 1. What was done

1. **Inspected the workspace.** The project directory contained no code and no git repository — this is a
   greenfield build.
2. **Located and audited the pre-existing churn project** at
   `C:\Projects\Customer-Churn-Prediction-Retention-Actions` (7,043-row Apache-2.0 dataset, verified
   `v1.0.0` artifact, FastAPI/Streamlit/Postgres implementation, drift + retention + pseudonymisation,
   tests, CI, model card, provenance).
3. **Ran the model artifact** rather than trusting its documentation: loaded it under the current
   environment (Python 3.11.7, scikit-learn 1.9.1, pandas 3.0.6, shap 0.51.0), transformed all 7,043 rows,
   scored them, and verified SHAP additivity.
4. **Audited the secondary credit-risk project** and ruled it out of v1 on evidence (bare estimators, no
   preprocessor or schema bundled, Kaggle-gated data).
5. **Produced the Stage 0 deliverable set** (below), including a measurable acceptance framework for
   explainability, fairness and drift quality.

---

## 2. Verified findings (evidence, not assumption)

| Finding | Evidence |
|---|---|
| Artifact structure is a composite dict (`model`, `preprocessor`, `feature_names`, `metadata`), not a `Pipeline` | Direct inspection of `model_v1.joblib` |
| Artifact loads and scores in the current environment | 7,043 × 46 transform + `predict_proba` succeeded |
| SHAP additivity holds exactly | max \|base + Σφ − logit\| = **2.22e-16** over 3 rows |
| SHAP background sample is embedded | `metadata.shap_background`, shape (100, 46) |
| Feature names need normalisation | stored names unprefixed; `get_feature_names_out()` adds `num__`/`cat__`; strip → exact match |
| No library versions in artifact metadata | `training_config` holds hyperparameters only |
| Model is deliberately miscalibrated | mean predicted risk 0.4172 vs observed prevalence 0.2654 (`class_weight="balanced"`) |
| Documented held-out metrics | ROC-AUC 0.8320, PR-AUC 0.6235, precision 0.5588, recall 0.6604, F1 0.6054, threshold 0.6478 |
| Credit-risk artifacts are not registrable as-is | bare `LogisticRegression` / `XGBClassifier` with no preprocessor or schema |

These findings directly changed the design: composite-artifact support, a feature-map layer, dependency-version
recording at registration, and a calibration module that has a real phenomenon to report.

---

## 3. Deliverables produced

| # | Deliverable | File |
|---|---|---|
| 1 | Product Requirements Document | `docs/01-product-requirements.md` |
| 2 | Existing-artifact assessment (integration decision) | `docs/02-existing-artifact-assessment.md` |
| 3 | System architecture + diagrams | `docs/03-system-architecture.md` |
| 4 | Database design + ER diagram | `docs/04-database-design.md` |
| 5 | Data-flow diagrams | `docs/05-data-flow.md` |
| 6 | API specification | `docs/06-api-specification.md` |
| 7 | Technology decision record (18 ADRs) | `docs/07-technology-decisions.md` |
| 8 | Risk assessment | `docs/08-risk-assessment.md` |
| 9 | Stage-wise development plan (Stages 1–14) | `docs/09-development-plan.md` |
| 10 | Project README | `README.md` |
| 11 | This report | `docs/00-stage-0-report.md` |
| 12 | Design-review remediation register | `docs/10-review-remediation.md` |

---

## 4. Key architecture decisions (summary)

| Decision | Rationale | Rejected alternative |
|---|---|---|
| Reuse the churn artifact + train a second XGBoost model in-house | Preserves validated work **and** proves the adapter abstraction | Copying the old single-model code |
| SHAP primary, LIME secondary, never merged | Faithfulness and honesty | Treating them as interchangeable |
| Fairness metrics implemented, Fairlearn as test oracle | Definitions must be visible in the product | Fairlearn in the runtime path |
| DB-backed job queue, no Celery/Redis | One node, bounded workload, fewer services to operate | Broker-based queue |
| Content-addressed explanation cache | Immutable inputs make caching provably safe | TTL cache |
| Hash-pinned artifact trust boundary + subprocess load | Pickle is code execution | Unrestricted uploads |
| No MLflow in v1 | The platform's registry is the product; MLflow adds a second source of truth | Half-integration |
| Streamlit over Dash/React | Depth over front-end infrastructure; API remains the real boundary | SPA |
| ReportLab for PDF | Wheel-only, works on Windows and Linux | WeasyPrint (GTK), Chromium |
| PostgreSQL everywhere incl. tests | JSONB, partial indexes, `SKIP LOCKED` are used | SQLite in tests |

Full reasoning, including consequences and revisit triggers, is in `docs/07-technology-decisions.md`.

---

## 5. Acceptance criteria status

| Stage 0 expectation | Status |
|---|---|
| Requirements analysis with functional/non-functional requirements | **Met** — 61 functional + 14 non-functional requirements, each with acceptance criteria |
| Target users and real-world use cases | **Met** — 5 personas, 8 use cases |
| Supported model types defined | **Met** — sklearn pipeline/composite, XGBoost native |
| Ingestion and explanation workflows designed | **Met** — data-flow diagrams + adapter/explainer contracts |
| Global and local explainability requirements | **Met** — FR-G-1..7, FR-L-1..5, fidelity gates |
| Fairness methodology established | **Met** — declared inputs, metric set, small-sample rules, no mitigation |
| Database schema designed | **Met** — 22 tables, ER diagram, indexes, retention |
| Application architecture designed | **Met** — layered diagram, contracts, execution model, deployment topologies |
| Security/scalability/reliability requirements | **Met** — trust boundaries, RBAC, budgets, NFR table |
| Dependencies and infrastructure cost | **Met** — ADRs + costed topologies (estimates flagged) |
| Measurable success criteria | **Met** — 10 criteria in the PRD §7 |
| Demonstration use case selected | **Met** — churn primary, credit risk documented as future |
| Existing churn project assessed for reuse | **Met** — with measured evidence |

**Gate decision: PASS (conditional).** All Stage 0 deliverables exist and are internally consistent. The
condition is the user's architecture validation, explicitly required before implementation.

---

## 6. Open questions for the user (non-blocking assumptions stated)

None of these block Stage 1, but the chosen defaults are recorded here so they can be changed cheaply now:

1. **Project name.** Default: *Aletheia*. Trivial to rename before Stage 1.
2. **Target cloud for the deployment reference.** Default: AWS reference documented, cheap container-host demo
   as the default path. Confirm if a different provider is preferred.
3. **Authentication depth.** Default: local JWT + API keys with three roles. No OIDC/SSO in v1 (would add an
   external dependency without demonstrating anything the portfolio needs).
4. **Full dataset in git.** Default: no — fetch script + checksum. Confirm if a self-contained repository is
   preferred despite the licensing/bloat trade-off.
5. **Live public demo.** Default: build it, but publish only with explicit approval at Stage 13.

---

## 7. Known limitations of this stage

- This report describes Stages 0 and 1 only. Stages 2–14 have no code, so nothing in those sections is
  runtime-verified. What is implemented (configuration, logging, error model, health, composite adapter,
  feature map, local SHAP with the additivity gate) is covered by 111 passing tests, with 94% line coverage on
  `aletheia.ml` + `aletheia.core`, and was exercised over real HTTP; see `docs/10-review-remediation.md`.
- The Stage 1b dashboard page does not exist yet, so the walking skeleton is verified as a command-line path
  rather than through the UI.
- Cost figures are order-of-magnitude estimates, not quotes, and must be re-verified before any deployment is
  described as live.
- The daily-effort estimates are planning aids; the gates, not the calendar, determine completion.
- The churn dataset's fairness analysis is inherently limited (no race, no age beyond a senior flag). This is
  disclosed in the PRD, the fairness design and the risk register rather than worked around.
- Multiclass support is specified at the adapter level but deliberately not designed into the dashboard for v1.

---

## 8. Next step

Stage 1a (foundation) and the core of Stage 1b (artifact → prediction → SHAP → additivity → grouped view) are
implemented and verified. The remaining work in order:

1. **Close Stage 1b** — one dashboard page over the existing skeleton path, plus the screenshot the plan's
gate requires.
2. **Stage 2** — data fetch and the in-house training CLI, which is what proves the adapter generalises
   beyond a single logistic regression.
3. **Stage 3** — schema migrations and the registry, at which point the API stops being an empty shell.

Stage boundaries still apply: no persistence before the migration policy exists, and no fairness claim before
there is a labelled dataset to support it.
