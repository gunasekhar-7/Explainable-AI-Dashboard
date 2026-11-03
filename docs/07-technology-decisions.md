# Technology Decisions (ADRs)

Format: Context → Decision → Alternatives rejected → Consequences. Decisions marked **Final** are fixed for
v1; **Provisional** may be revisited at the stated trigger.

---

## ADR-001 — Reuse the existing churn model as the first registered model
**Status:** Final · **Stage:** 2–3

**Context.** A verified logistic-regression artifact exists (`v1.0.0`, threshold 0.6478, test ROC-AUC 0.8320),
carrying threshold metadata, test metrics, provenance and a 100×46 SHAP background. It loads and scores under
the current environment and is SHAP-additive to 2.2e-16 (measured — see
`docs/02-existing-artifact-assessment.md`).

**Decision.** Register it unchanged as the first model version, translating its metadata into the platform
manifest. Separately train an XGBoost model through Aletheia's own pipeline so the adapter abstraction is
proven across two estimator families.

**Rejected.** (a) Retrain everything and discard the artifact — wastes validated work and loses the
calibration finding (mean score 0.417 vs 26.5% prevalence), which is a genuine teaching case. (b) Copy the
existing FastAPI/DB code — it is hard-coded to one model and would defeat the registry design.

**Consequences.** The adapter must support composite dicts (not just `Pipeline`) and a feature-name
normalisation layer (stored names are unprefixed; `get_feature_names_out()` prefixes `num__`/`cat__`).
Registration must add library versions, which the artifact lacks.

---

## ADR-002 — FastAPI with synchronous SQLAlchemy and `def` endpoints
**Status:** Final · **Stage:** 3

**Context.** ML work is CPU-bound pandas/numpy/sklearn. The expected load is one operator and a handful of
concurrent reviewers.

**Decision.** SQLAlchemy 2.0 sync `Session` per request; endpoints declared `def` so FastAPI runs them in its
threadpool; heavy CPU work dispatched to a process pool via the job system.

**Rejected.** Async SQLAlchemy + `asyncpg` (real concurrency benefit only for I/O-bound DB work mixed with
async ML, which this is not; it adds session-lifetime pitfalls and forces every ML call through
`run_in_executor` anyway). Django (heavier, ORM-centric, no benefit here).

**Consequences.** Throughput is bounded by threadpool size; for this workload that ceiling is far above
demand. Blocking calls in `async def` would stall the loop, so the rule "`async def` only for pure I/O
coordination" is enforced in review and by a lint check for `time.sleep`/sync DB in async functions.

---

## ADR-003 — PostgreSQL 16 as the only database
**Status:** Final · **Stage:** 1

**Context.** Requirement states PostgreSQL; the data model leans on JSONB, partial unique indexes and
`FOR UPDATE SKIP LOCKED`.

**Decision.** PostgreSQL 16 in all environments, including tests (disposable database per session).

**Rejected.** SQLite for tests (diverges on JSONB semantics, partial indexes and `SKIP LOCKED`, so tests would
pass on behaviour production does not have). Adding a separate vector/document store (no requirement).

**Consequences.** Local development requires Docker (or a local Postgres). CI uses a Postgres service
container. Unit tests that need no DB stay pure so the fast suite remains fast.

---

## ADR-004 — Streamlit for the dashboard, API-only communication
**Status:** Final · **Stage:** 10

**Context.** Eight analytic surfaces, heavy Plotly use, one developer, and a need to ship depth rather than
front-end infrastructure.

**Decision.** Streamlit multi-page app with a single `api_client.py`; the dashboard performs no DB access and
no model loading.

**Rejected.** Dash (more control, more code for routing/state), React SPA (best UX, disproportionate effort
and a second deployment surface), Gradio (poor fit for multi-page analytics).

**Consequences.** Some UI patterns are awkward (deep-linking, complex cross-page state), accepted as a
trade-off. Because the API is the product boundary, the dashboard can be replaced without touching the
backend. Server-rendered matplotlib images are used where native SHAP plots are required; Plotly handles
interactive charts the API exposes as data.

---

## ADR-005 — SHAP primary, LIME secondary and never merged
**Status:** Final · **Stage:** 4–5

**Context.** SHAP gives additive, model-consistent attributions; LIME gives a local surrogate approximation.
Presenting them as equivalent would be scientifically wrong.

**Decision.** SHAP is the primary engine with explainer selection by estimator family
(`LinearExplainer`/`TreeExplainer`, `KernelExplainer` as a flagged fallback). LIME is computed for the same
prediction, stored separately, and always displayed with measured fidelity (R², local agreement) and a
"surrogate" label. Additivity is checked before persisting; failure withholds the explanation.

**Rejected.** LIME-only (unstable, not additive), KernelExplainer everywhere (slow, sampled, approximate),
`shap.Explainer` auto-selection (hides which algorithm ran; explicit selection is auditable).

**Consequences.** Per-model explainer coverage must be tested; multiclass requires explicit handling
(per-class outputs); `KernelExplainer` paths must be row-limited and clearly labelled as approximate.

---

## ADR-006 — Fairlearn for cross-validation, own metrics for the product
**Status:** Final · **Stage:** 6

**Context.** Fairness metrics must be correct and defensible, and their definitions must be visible in the
product.

**Decision.** Implement per-group metrics and disparity metrics explicitly (confusion-matrix cells stored),
and use Fairlearn's `MetricFrame` in tests as an independent oracle (parity within 1e-9). No mitigation
algorithms in v1.

**Rejected.** Fairlearn as the runtime engine (its abstractions would leak into API contracts and obscure
definitions the product must state). Any automatic threshold adjustment (would silently change decisions
under review).

**Consequences.** More code and more tests, but every number is explainable line by line and the definitions
can be printed in the report. Metric conflicts (demographic parity vs equal opportunity) are stated, not
resolved by the software.

---

## ADR-007 — No MLflow in v1
**Status:** Provisional (revisit at Stage 13) · **Stage:** 2

**Context.** The prompt allows MLflow "if justified". The platform's own registry is a product feature that
must expose versioning, metrics and lineage to end users.

**Decision.** Implement a first-class internal experiment/lineage record (training run rows + artifact
manifest with params, metrics, dataset version, library versions). Do not run an MLflow server.

**Rejected.** MLflow as system of record (duplicates the registry, adds a service to operate, and its UI is
not the deliverable), MLflow as a half-integration (worse than none: two sources of truth, neither tested).

**Consequences.** No MLflow UI screenshots. The artifact manifest is shaped so an MLflow export adapter can
be added later without schema change. Trigger to revisit: training experiments exceed what the run table can
express (e.g. sweeps over dozens of configurations needing comparison UI).

---

## ADR-008 — DB-backed job queue with an in-process worker, no Celery/Redis
**Status:** Final · **Stage:** 3, revisited at 11 · **Trigger:** job p95 > 60 s, or >1 API replica

**Context.** Long operations (global SHAP, batch scoring, drift, PDF) must not block requests.

**Decision.** `jobs` table + `FOR UPDATE SKIP LOCKED` claiming + `ProcessPoolExecutor` for CPU work; job
payloads versioned; graceful drain on SIGTERM; bounded retries with error classification.

**Rejected.** Celery + Redis (a second service, serialisation boundary, and visibility problems for a
single-node deployment), FastAPI `BackgroundTasks` (dies with the process, no state, no retry), in-memory
`Queue` (loses work on restart).

**Consequences.** No horizontal job scaling in v1; the `JobQueue` interface is broker-agnostic so the swap is
contained. On Windows, `resource` limits are unavailable — the sandboxed artifact loader and memory caps are
therefore only fully effective in the Linux container, which is stated in the security doc.

---

## ADR-009 — Artifact trust model: hash-pinned, sandboxed, no untrusted pickle execution
**Status:** Final · **Stage:** 3, hardened at 11

**Context.** `joblib`/`pickle` deserialisation executes arbitrary code. A platform that accepts model uploads
is a code-execution surface.

**Decision.**
1. Production registration reads only from a server-controlled import directory or a hash-pinned manifest.
2. Uploads are disabled unless `ALLOW_ARTIFACT_UPLOAD=true` (development/demo only) and are still parsed in a
   subprocess with timeout and memory cap.
3. SHA-256 is verified before load; mismatch aborts.
4. XGBoost native `.json`/`.ubj` is preferred where available; sklearn artifacts must originate from this
   project's training CLI or a known producer.
5. `LoadedModel` is constructed only inside the loader subprocess boundary; the API process never unpickles
   untrusted bytes directly.

**Rejected.** Unrestricted upload (unacceptable), `picklescan`-style heuristics as the only control (bypassable;
used at most as defence in depth), ONNX-only (excludes the required scikit-learn support).

**Consequences.** Residual risk remains: a determined pickle payload in an allowed import directory executes
inside the sandbox. Documented explicitly with the assumption that the import directory is operator-controlled.

---

## ADR-010 — ReportLab for PDF, Jinja2 for HTML
**Status:** Final · **Stage:** 12

**Context.** Reports must be downloadable PDFs containing charts and tables, generated in a Docker image and on
Windows for local development.

**Decision.** Jinja2 renders HTML; ReportLab renders PDF with embedded PNG charts. CSV is written directly.

**Rejected.** WeasyPrint (excellent CSS, but needs native GTK libraries that complicate Windows development and
enlarge the image), headless Chromium via Playwright (~300 MB, slow cold start), wkhtmltopdf (unmaintained
binary dependency).

**Consequences.** More explicit layout code; fully deterministic output and wheel-only installation. HTML
output is available as a lighter alternative and is what the UI previews.

---

## ADR-011 — Content-addressed caching for explanations
**Status:** Final · **Stage:** 4

**Context.** Global SHAP over 7k rows takes seconds; the same analysis is requested repeatedly while a user
explores.

**Decision.** Cache key = hash of artifact SHA-256, reference dataset hash, input hash, method, canonicalised
parameters and library versions; stored in `explanation_runs.cache_key` with a unique index. Immutability makes
indefinite caching sound. Failed runs delete their key.

**Rejected.** TTL cache (can serve stale results against changed inputs; TTL expiry is unrelated to validity),
no cache (wastes CPU and hurts the UX budget), Redis (ADR-008).

**Consequences.** A library upgrade changes the key (correct: numerics may change). Storage grows with distinct
analyses; the retention job prunes explanation artifacts.

---

## ADR-012 — Storage abstraction: local filesystem in dev, S3-compatible in production
**Status:** Final · **Stage:** 1, wired at 13

**Context.** Artifacts, datasets (Parquet), plots and reports are binary objects; the demo must run with zero
cloud setup.

**Decision.** `ArtifactStore` protocol (`put`, `get`, `exists`, `url`, `delete`) with `LocalFileStore` and
`S3Store` implementations selected by configuration. Keys are content-agnostic paths
(`models/{id}/{version}/artifact.joblib`), integrity checked by SHA-256.

**Rejected.** DB large objects (poor fit, bloats backups), committing artifacts to git (repository bloat and
secret-risk), direct `boto3` calls in services (couples business logic to AWS).

**Consequences.** Two implementations to test; a contract test suite runs against both (S3 via MinIO in CI is
optional and documented).

---

## ADR-013 — Configuration and secrets via environment only
**Status:** Final · **Stage:** 1

**Decision.** `pydantic-settings` models, `.env` for local development (gitignored), `.env.example` committed
with non-secret keys, fail-fast validation at startup, redaction filter in the logger. No secret is ever read
from a file committed to git or from the database.

**Consequences.** Startup fails loudly on missing required configuration rather than degrading silently.

---

## ADR-014 — Modular monolith, explicit layer boundaries
**Status:** Final

**Context.** One developer, one deployment, strong internal contracts.

**Decision.** One installable package (`aletheia`) with `api → services → ml` and a separate `dashboard`
client package. Boundaries enforced by an import-graph test.

**Rejected.** Microservices (network boundaries, deployment overhead, distributed debugging — all cost, no
benefit at this scale), single flat module tree (no enforceable boundaries, ML code would accrete HTTP and DB
concerns).

**Consequences.** Refactoring across layers is cheap; the whole app deploys as two containers (api, dashboard)
plus Postgres.

---

## ADR-015 — Datasets are fetched, not committed
**Status:** Final · **Stage:** 2

**Context.** The IBM Telco file is ~948 KB, Apache-2.0, and reproducible from a public URL with a known
SHA-256. Model artifacts are binary and can embed data.

**Decision.** `scripts/fetch_datasets.py` downloads and verifies checksums; `data/` and `artifacts/` are
gitignored; a small committed fixture (≤200 rows, synthetic, clearly labelled) supports tests. The served
model artifact is built by the training CLI in the deployment pipeline and stored in the artifact store.

**Rejected.** Committing the dataset (licensing clarity aside, it invites silent modification and bloats the
repo), committing the joblib (binary, unreviewable, version-fragile).

**Consequences.** A fresh clone must run one script before training (documented in the README quickstart). CI
uses the committed fixture, and any metric computed on it is labelled as a smoke-test number, never a
performance claim.

---

## ADR-016 — Testing: pytest with layered markers, Postgres for integration
**Status:** Final · **Stage:** 1, expanded each stage

**Decision.** `unit` (pure, no DB/network), `integration` (Postgres + artifact store), `e2e` (API + dashboard
client workflow), `contract` (OpenAPI snapshot), `slow` (stability sweeps). Coverage gate ≥85% on `ml/` and
`services/`; property-style checks for metric identities (e.g. confusion-matrix cell sums, additivity,
PSI symmetry-not-required but PSI(x,x)=0).

**Rejected.** Coverage as a global vanity number (excludes thin API glue from the gate deliberately), snapshot
tests for charts (brittle; charts are verified by asserting the data behind them).

**Consequences.** CI needs a Postgres service and ~6–8 minutes; the fast unit suite stays under 60 s for local
iteration.

---

## ADR-017 — No LLM in the explanation path
**Status:** Final · **Stage:** 5

**Context.** Natural-language explanations are requested by stakeholders; an LLM could phrase them fluently.

**Decision.** Deterministic templates that consume only stored contributions and feature labels. The narrative
is generated once, stored with the explanation, and reproduced verbatim in exports.

**Rejected.** LLM paraphrasing (non-deterministic, may hallucinate a factor that is not in the explanation,
untestable faithfulness, adds a credential and a cost per explanation).

**Consequences.** Language is formulaic but always faithful and testable; templating covers direction, rank,
raw value and unit wording.

---

## ADR-018 — Deployment: Compose for demo, ECS/RDS as the production reference
**Status:** Provisional · **Stage:** 13

**Decision.** Local/demo: `docker compose` (api, dashboard, postgres). Public demo: one small container host +
managed Postgres + S3-compatible storage (estimated $0–25/month at demo scale). Production reference: ECS
Fargate + RDS + S3 + ALB + CloudWatch (estimated $70–130/month idle) with a cheaper single-instance Compose
variant documented.

**Rejected.** Kubernetes (no requirement, high operational cost), Lambda (cold starts and heavy native
dependencies in the ML stack are a poor fit).

**Consequences.** Idle cost of the "full" cloud path is stated plainly; the demo path is the default because a
portfolio project should not bill for idle infrastructure. All figures are estimates to be re-verified before
any deployment is claimed as live.
