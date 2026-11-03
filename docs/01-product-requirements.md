# Product Requirements Document — Aletheia

| Field | Value |
|---|---|
| Product | Aletheia — Explainable AI & Model Governance Platform |
| Version | 1.0 (Stage 0 baseline) |
| Date | 2026-11-03 |
| Owner | Project author (solo build) |
| Status | Awaiting architecture validation |

---

## 1. Problem statement

Teams can train a classifier in minutes but cannot answer the questions that decide whether it may be
used: *why did it produce this output, does it behave differently for different groups, is it calibrated,
has its input distribution moved, and can any of this be shown to a reviewer?* Answering those questions
usually means ad-hoc notebooks that are neither reproducible nor auditable, and whose figures cannot be
traced back to a specific model version and dataset.

Aletheia turns those ad-hoc answers into a product: a governed model registry with explanation, fairness,
reliability and drift analysis, all persisted with lineage and exportable as reports.

**Non-problem:** Aletheia is not a training platform, not a feature store, and not a general BI tool. It
consumes already-trained models.

---

## 2. Target users

| Persona | Primary need | Technical level | Key surfaces |
|---|---|---|---|
| **ML Engineer** (primary) | Verify that explanations are faithful and that a model version is reproducible | High | Model registration, global explanations, calibration, drift |
| **Data Scientist** (primary) | Investigate individual predictions and compare model versions | High | Prediction explorer, local SHAP/LIME, model comparison |
| **AI QA / Model Validator** (primary) | Independently test the model and produce evidence of the tests | High | Evaluation, fairness report, explanation-fidelity checks, audit trail |
| **Compliance / Risk Reviewer** (secondary) | Read a defensible summary of model behaviour and disparities | Medium | Fairness dashboard, PDF reports |
| **Business Stakeholder** (secondary) | Understand why a specific customer is flagged and what to do | Low | Prediction explanation card, readable narrative |

The interface must serve the first three without hiding evidence, and serve the last two without
exposing them to raw metric soup. This is handled by a technical/executive split on every page
(see FR-UI-3).

---

## 3. Real-world use cases

| ID | Use case | Actor | Outcome |
|---|---|---|---|
| UC-1 | Register a new model version and confirm it is compatible with the operational dataset | ML Engineer | Versioned model with validated feature schema and lineage |
| UC-2 | Explain overall model behaviour for a stakeholder deck | Data Scientist | Global importance, beeswarm, dependence plots, exportable figures |
| UC-3 | Explain one high-risk customer's prediction to a retention agent | Business Stakeholder | Local SHAP waterfall + plain-language factors + probability |
| UC-4 | Check whether recall differs by gender or senior-citizen status before launch | AI QA | Per-group metrics with sample sizes and disparity metrics |
| UC-5 | Determine whether a 0.6478 threshold is appropriate for the intervention budget | ML Engineer | Threshold/ROC/PR analysis with precision-recall trade-offs |
| UC-6 | Detect that last month's applications look different from the training reference | ML Engineer | PSI/KS drift report, per-feature, with explicit limits |
| UC-7 | Produce an audit pack for a model review | Compliance Reviewer | PDF/CSV reports with model version, dataset version, metrics and caveats |
| UC-8 | Compare two model versions on identical evaluation data | Data Scientist | Side-by-side metrics, calibration and fairness deltas |

---

## 4. Scope

### 4.1 In scope (v1)

- Model registry with immutable versions, artifact integrity checks and feature-schema validation.
- Single and batch prediction with threshold control, prediction history and pseudonymised subject refs.
- Global explainability: SHAP importance, summary/beeswarm, dependence, interaction, stability analysis,
  comparison against model-native importance.
- Local explainability: SHAP waterfall/force-equivalent, LIME cross-check, signed contributions,
  human-readable narrative generated deterministically from the numeric explanation.
- Fairness: per-group performance, demographic parity, equal opportunity, equalised odds, selection
  rates, FPR/FNR gaps, small-sample warnings, explicit definitions and limitations.
- Evaluation: precision/recall/F1/ROC-AUC/PR-AUC, confusion matrix, calibration (Brier, reliability
  curve), error analysis, subgroup performance, threshold sweep, version comparison.
- Monitoring: feature drift (PSI, KS, chi-square), prediction drift, missing-rate drift, label-based
  performance tracking when ground truth arrives, historical monitoring runs.
- Reporting: PDF/CSV/HTML exports for explanations, fairness, evaluation and single-prediction reports.
- Governance: audit events, request IDs, retention controls, roles (admin/analyst/viewer).
- Interactive dashboard covering all of the above.

### 4.2 Explicit non-goals (v1)

| Non-goal | Reason |
|---|---|
| Model training UI | Training is a CLI/notebook concern; the platform consumes artifacts (Stage 2 CLI only) |
| Automatic fairness mitigation (pre/in/post-processing) | Mitigation changes the model under review; it must be a separate, separately evaluated workflow |
| Causal inference | Attribution ≠ causation; claiming otherwise would be scientifically false |
| LLM-generated narratives | Deterministic templates keep explanations faithful and testable |
| Real-time streaming inference | Batch/request-response covers the demonstrated use cases |
| Multi-tenant isolation | Single-organisation deployment; RBAC only |
| Multiclass UI depth | Adapter supports multiclass; dashboard depth is binary-first in v1 (documented) |
| Deep-learning model support | Adapters are open for extension; no CNN/transformer path in v1 |

---

## 5. Functional requirements

Each requirement has an ID, a statement and an acceptance criterion that a test or a recorded manual
check can verify. "Verified by" references the stage gate in `docs/09-development-plan.md`.

### 5.1 Dataset management

| ID | Requirement | Acceptance criterion |
|---|---|---|
| FR-D-1 | Upload a CSV dataset with a declared target column and task type | Valid file → dataset version created with checksum, row count, schema; invalid file → structured 422 |
| FR-D-2 | Validate schema, dtypes, target domain, duplicate keys and missingness on ingest | Validation report persisted; hard failures block ingest, warnings are surfaced |
| FR-D-3 | Compute and persist descriptive statistics and per-column quality metrics | Missing rate, cardinality, min/max/quantiles, outlier counts stored per version |
| FR-D-4 | Version datasets immutably with content hash and provenance (source, license, timestamp) | Re-upload of identical content produces identical hash; versions are never mutated |
| FR-D-5 | Register a dataset version as the drift/explanation reference for a model version | Reference link persisted and used by default in drift and SHAP background selection |
| FR-D-6 | Reject group/fairness analysis when the requested attribute is absent | Explicit capability error, never a fabricated group dimension |

### 5.2 Model management

| ID | Requirement | Acceptance criterion |
|---|---|---|
| FR-M-1 | Register a model artifact (scikit-learn `Pipeline`, composite `{preprocessor, model}`, or XGBoost native JSON/UBJ) | Adapter loads it, extracts feature schema, and reports the adapter used |
| FR-M-2 | Enforce artifact integrity (SHA-256) and record framework/library versions | Mismatch or unreadable artifact → registration rejected with a specific error code |
| FR-M-3 | Validate model↔dataset compatibility (feature presence, dtypes, unseen categories policy) | Incompatibility reported per feature before any prediction is attempted |
| FR-M-4 | Immutable model versions with semantic version labels and status (candidate/active/archived) | Only one `active` version per model; version rows cannot be edited after registration |
| FR-M-5 | Compare two or more model versions on identical evaluation data | Comparison endpoint returns aligned metric tables and deltas |
| FR-M-6 | Preserve model-native feature importance alongside SHAP importance | Where the estimator exposes importances/coefficients, both are stored and displayed |
| FR-M-7 | Persist the feature-schema diff produced by a compatibility check | Diff stored on the model version and returned by the compatibility endpoint; identical inputs produce an identical diff |

### 5.3 Prediction

| ID | Requirement | Acceptance criterion |
|---|---|---|
| FR-P-1 | Single-record prediction returning probability, label at the configured threshold, and model version | Response includes `model_version_id`, `artifact_sha256`, `threshold` |
| FR-P-2 | Batch prediction over a dataset version or uploaded CSV, executed as a background job | Job is pollable; partial failures are reported per row, not silently dropped |
| FR-P-3 | Threshold override per request without mutating the stored model version | Override echoed in the prediction record |
| FR-P-4 | Persist prediction history with minimised inputs and pseudonymised subject reference | No raw identifier or excluded field is stored when minimisation is on |
| FR-P-5 | Reproducibility metadata per prediction | Re-running the same input against the same version yields the same probability (tested) |
| FR-P-6 | Meaningful error for invalid input (missing feature, wrong dtype, out-of-domain category) | 422 with per-field detail; 409/422 for schema mismatch, never a 500 |
| FR-P-7 | Attach ground-truth labels to already-scored predictions (late-arriving outcomes) | Labels join by `prediction_id` or subject reference; duplicate or conflicting labels rejected with a code; coverage returned; unlabelled predictions are reported as excluded, never assumed negative |

### 5.4 Global explainability

| ID | Requirement | Acceptance criterion |
|---|---|---|
| FR-G-1 | Global SHAP feature importance (mean \|SHAP\|) with feature-name mapping back to raw columns | Names map one-to-one to raw/encoded features; mapping is unit-tested |
| FR-G-2 | SHAP summary/beeswarm visualisation | Plot renders and is downloadable as PNG/SVG |
| FR-G-3 | SHAP dependence plots per selected feature, including interaction colouring where computed | Values consistent with the stored SHAP matrix |
| FR-G-4 | Explainability additivity check on a sample of rows | \|base + Σφ − model log-odds\| ≤ 1e-6 for linear/tree explainers |
| FR-G-5 | Explanation stability report across background/reference resamples | Rank correlation and top-k overlap reported, not asserted |
| FR-G-6 | Explainability cache keyed by (artifact hash, reference data hash, parameters) | Identical inputs hit cache; changed inputs miss it |
| FR-G-7 | Explainer selection documented per model type, with limitations surfaced in the UI | Response includes `explainer_type` and a `caveats` list |

### 5.5 Local explainability

| ID | Requirement | Acceptance criterion |
|---|---|---|
| FR-L-1 | SHAP local explanation for a chosen prediction with per-feature contributions | Contributions sum to the model output gap within tolerance (FR-G-4) |
| FR-L-2 | LIME explanation for the same prediction, stored separately and labelled as a surrogate | Both stored; UI shows fidelity (R²/local agreement), never merged |
| FR-L-3 | Display raw feature values next to contributions, with direction | Raw values come from the stored input, not the encoded vector |
| FR-L-4 | Deterministic plain-language narrative derived only from stored contributions | Template output references only features present in the explanation; unit-tested |
| FR-L-5 | Export a single-prediction explanation as PDF/CSV | Export contains model version, threshold, timestamp, method and caveats |

### 5.6 Fairness

| ID | Requirement | Acceptance criterion |
|---|---|---|
| FR-F-1 | Compute per-group accuracy, precision, recall, F1, FPR, FNR, selection rate, support | Values match an independent hand-computed fixture |
| FR-F-2 | Compute demographic parity difference/ratio, equal opportunity difference, equalised odds difference | Values match Fairlearn on the same inputs within 1e-9 |
| FR-F-3 | Require explicit declarations: protected attribute, privileged group, favourable label, threshold | Assessment refuses to run without them |
| FR-F-4 | Flag groups below a configurable minimum support and downgrade conclusions | Small-group results marked insufficient; no disparity verdict asserted |
| FR-F-5 | Report metric definitions, conflicts between criteria and dataset limitations in the report | Report template includes a mandatory limitations section |
| FR-F-6 | Never modify predictions to satisfy a fairness criterion | No code path mutates stored predictions for fairness |
| FR-F-7 | Handle missing group values transparently | Missing counts reported; excluded rows never silently dropped |

### 5.7 Evaluation and reliability

| ID | Requirement | Acceptance criterion |
|---|---|---|
| FR-E-1 | Reproducible metric suite on a fixed evaluation split (accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix) | Recompute in two runs produces identical values |
| FR-E-2 | Calibration analysis (reliability curve, Brier score, expected calibration error) stored separately from discrimination | Both reported distinctly |
| FR-E-3 | Threshold sweep showing precision/recall/F1/flag-volume trade-offs | Table and chart generated for configurable thresholds |
| FR-E-4 | Error analysis: false positives/negatives with feature context and exportable samples | Subset counts reconcile with the confusion matrix |
| FR-E-5 | Subgroup performance consistent with the fairness engine | Same group metrics; cross-checked by a test |
| FR-E-6 | Model-version comparison on identical evaluation data | Differences attributable only to the model, verified by shared dataset version ID |

### 5.8 Monitoring and drift

| ID | Requirement | Acceptance criterion |
|---|---|---|
| FR-R-1 | Feature-level drift with PSI and KS (numeric) / chi-square (categorical) and configurable thresholds | Implementations match reference fixtures |
| FR-R-2 | Prediction-distribution drift against the stored reference | PSI on scores persisted per run |
| FR-R-3 | Missing-rate and cardinality change detection | Reported per feature |
| FR-R-4 | Performance tracking when labels arrive, explicitly separated from drift | Report separates the three signals with distinct wording |
| FR-R-5 | Historical monitoring runs with trend view | Runs listable and comparable over time |
| FR-R-6 | Drift output must never assert accuracy loss | Report text and tests enforce the distinction |

### 5.9 Reporting and governance

| ID | Requirement | Acceptance criterion |
|---|---|---|
| FR-RP-1 | PDF/CSV export for global explanation, local explanation, fairness, evaluation and monitoring | Files generated, checksummed and downloadable |
| FR-RP-2 | Reports embed model version, dataset version, artifact hash, generation timestamp, method and caveats | A report can be traced to exact inputs |
| FR-RP-3 | Audit trail of privileged actions (registration, activation, de-pseudonymisation, export, deletion) | Audit rows are append-only and queryable by resource |
| FR-RP-4 | Data retention policy applied by a scheduled, bounded, dry-runnable job | Expired rows removed in batches; dry-run reports counts without deleting |
| FR-RP-5 | Role-based access: viewer (read), analyst (compute), admin (register/manage/delete) | 403 on insufficient role, tested per endpoint class |

### 5.10 Dashboard

| ID | Requirement | Acceptance criterion |
|---|---|---|
| FR-UI-1 | Eight surfaces: Executive Overview, Model Explorer, Global Explainability, Prediction Explorer, Fairness, Reliability, Monitoring, Reports | All reachable, each backed only by API calls |
| FR-UI-2 | Dashboard never talks to the database directly | Import-level test/architecture test forbids DB imports in the dashboard package |
| FR-UI-3 | Every page separates executive summary from technical detail | Executive panel present on each analysis page |
| FR-UI-4 | Explicit, non-dismissible caveats where interpretation is risky | Caveats present on explanation, fairness and drift pages |
| FR-UI-5 | Graceful degradation when a model or dataset lacks a capability | Clear "not available for this model" state, never a crash or a fake chart |

---

## 6. Non-functional requirements

| ID | Requirement | Target |
|---|---|---|
| NFR-1 | Single-record prediction latency | p95 ≤ 150 ms server-side on the reference dataset |
| NFR-2 | Local explanation latency (SHAP + LIME) | p95 ≤ 2.5 s |
| NFR-3 | Global explanation over 7,043 rows | ≤ 20 s uncached; cached retrieval ≤ 300 ms |
| NFR-4 | Fairness assessment | ≤ 10 s for 7k rows / 2 attributes |
| NFR-5 | Availability of read paths | Health + readiness endpoints; degraded-mode responses when the artifact store is unavailable |
| NFR-6 | Reproducibility | Fixed seeds; content-hash-keyed caches; identical outputs on re-run |
| NFR-7 | Security | No committed secrets; hashed API keys; artifact trust boundary enforced; rate limits per identity |
| NFR-8 | Privacy | Pseudonymised subject references; minimised stored inputs; configurable retention; de-pseudonymisation audited |
| NFR-9 | Observability | Structured JSON logs with request IDs; no PII or credentials in logs; job failure visibility |
| NFR-10 | Maintainability | Typed public interfaces; MyPy clean on `ml/` and `services/`; Ruff clean; coverage ≥ 85% on `ml/` and `services/` |
| NFR-11 | Portability | `docker compose up` brings up API + Postgres + dashboard on Windows and Linux |
| NFR-12 | CI | Lint, type-check, unit + integration tests, image build — under 10 minutes |
| NFR-13 | Accessibility/usability | Consistent navigation, labelled controls, colour-blind-safe palettes, units on every axis |
| NFR-14 | Supply chain | Every release produces an SBOM; `pip-audit` runs in CI and fails on a known-vulnerable direct dependency; a new runtime dependency requires an ADR entry with a rejected alternative |

---

## 7. Success criteria (measurable)

The project is portfolio-ready only when all of the following hold with recorded evidence:

1. Registered churn artifact reproduces its documented held-out metrics (test ROC-AUC 0.8320 ± 0.005) when
   evaluated by the platform on a reconstructed split, with the split definition documented.
2. SHAP additivity error ≤ 1e-6 across a 200-row sample, enforced as a test.
3. Global feature-importance stability: Spearman rank correlation ≥ 0.90 across 10 differing background
   samples (measured and reported; if it fails, the failure is reported, not hidden).
4. LIME local fidelity reported numerically (surrogate R² and local agreement), with an explicit statement
   that it is a surrogate and not interchangeable with SHAP.
5. Fairness metrics match Fairlearn to ≤ 1e-9 on a shared fixture; small-group warnings demonstrated on a
   real subgroup with support below the configured minimum.
6. Calibration reported separately from discrimination for the shipped model, with Brier score and
   reliability curve.
7. Drift detection validated on a synthetic shifted dataset (known shift → detected; identical data → not
   detected), and the report never equates drift with accuracy loss.
8. Every persisted artifact (prediction, explanation, assessment, report) resolves to a model version and
   dataset version, verified by an integration test.
9. `pytest` suite green; coverage ≥ 85% on `ml/` and `services/`; MyPy and Ruff clean; CI green.
10. `docker compose up` yields a healthy stack; dashboard completes the full workflow: register → predict →
    explain globally → explain locally → fairness → drift → export report.

Anything not measured is not claimed. Where a metric is expected to be weak (e.g. LIME stability, fairness
power on small subgroups), it is reported as weak.

---

## 8. Terminology discipline (enforced in UI copy and tests)

| Term | Meaning in this product | What it must never be called |
|---|---|---|
| SHAP value | Additive attribution in model output space (log-odds for linear models) | "Probability contribution", "percentage risk", "cause" |
| Feature importance | Mean absolute SHAP across a reference sample | "The reason customers churn" |
| LIME weight | Local surrogate coefficient | "SHAP", "the model's reasoning" |
| Demographic parity difference | Gap in favourable-outcome rates between groups | "Fairness score", "bias percentage" |
| Data drift | Change in input distribution vs reference | "Model degradation", "accuracy drop" |
| Calibration | Agreement between predicted probability and observed frequency | "Accuracy" |

The three fairness criteria can be mutually unsatisfiable; the product states this and never declares a
model "fair" or "unfair" from a single number.
