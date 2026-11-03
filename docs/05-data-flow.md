# Data Flow Diagrams — Aletheia

Notation: rectangles are processes, cylinders are stores, parallelograms are external actors. Every flow
that writes data names the store it writes to. Trust boundaries are marked explicitly.

---

## 1. Context (level 0)

```mermaid
flowchart LR
    U["Analyst / ML Engineer<br/>Admin / Stakeholder"] -->|"credentials, datasets,<br/>requests"| S(["Aletheia platform"])
    S -->|"predictions, explanations,<br/>fairness & drift reports"| U
    M["Trained model artifacts<br/>(external, hash-pinned)"] --> S
    D["Churn dataset<br/>(IBM Telco, Apache-2.0)"] --> S
    S -->|"PDF / CSV audit packs"| R["Reviewer / recruiter"]
```

---

## 2. Dataset onboarding

```mermaid
flowchart TD
    A["CSV upload or file reference"] --> B{"Size / extension /<br/>encoding checks"}
    B -->|fail| E1["422 DATASET_INVALID<br/>(no state written)"]
    B -->|pass| C["Parse with pandas<br/>(bounded rows, dtypes inferred)"]
    C --> D{"Schema + target<br/>validation"}
    D -->|error| E1
    D -->|warning| W["Persist quality warnings"]
    D --> F["Canonicalise to Parquet<br/>compute SHA-256"]
    F --> G[("Object store<br/>datasets/{dataset_id}/{version}/data.parquet")]
    F --> H[("DB: dataset_versions<br/>schema, statistics, quality_report,<br/>content_sha256, provenance")]
    W --> H
    H --> I["Register as reference<br/>for a model version"]
```

Key rules: content hash is computed on the canonical Parquet (stable column order, UTC timestamps), so
re-uploading identical data cannot create a second version; the raw upload is discarded after
canonicalisation unless `KEEP_RAW_UPLOADS=true`.

---

## 3. Model registration and activation

```mermaid
flowchart TD
    A["Artifact source<br/>import dir | allowlisted upload"] --> B{"Source allowed?<br/>ALLOW_ARTIFACT_UPLOAD"}
    B -->|no| E1["403 ARTIFACT_SOURCE_NOT_ALLOWED"]
    B -->|yes| C["Copy to staging,<br/>compute SHA-256"]
    C --> D{"Hash matches<br/>manifest / declared value?"}
    D -->|no| E2["422 ARTIFACT_INTEGRITY_FAILED"]
    D -->|yes| F["Sandbox load in subprocess<br/>timeout + memory cap, no network"]
    F -->|crash/timeout| E3["422 ARTIFACT_UNLOADABLE"]
    F --> G["AdapterRegistry.select<br/>→ adapter + LoadedModel"]
    G --> H["Extract feature_schema,<br/>feature_map, classes, framework versions"]
    H --> I{"Compatible with<br/>declared dataset?"}
    I -->|no| E4["409 MODEL_SCHEMA_MISMATCH<br/>(per-feature diff returned)"]
    I -->|yes| J["Persist background sample<br/>deterministic from reference dataset"]
    J --> K[("DB: model_versions<br/>status=candidate")]
    K --> L["POST /activate"]
    L --> M[("DB: active flag swapped atomically;<br/>audit_events: model.activate")]
```

The artifact and its metadata are the *only* source of the feature contract. No schema is hand-written for a
model; it is derived from the fitted transformer and then frozen into `model_versions.feature_schema`.

---

## 4. Prediction (single and batch)

```mermaid
flowchart TD
    A["POST /model-versions/{id}/predict"] --> B["Load model version<br/>(in-memory LRU keyed by sha256)"]
    B --> C["Validate against feature_schema<br/>(missing / dtype / unknown category)"]
    C -->|invalid| E1["422 PREDICTION_INPUT_INVALID<br/>per-field detail"]
    C --> D["preprocessor.transform → estimator.predict_proba"]
    D --> F["Apply threshold (version default or request override)"]
    F --> G[("DB: predictions<br/>minimised inputs, pseudonymised subject,<br/>probability, label, threshold")]
    G --> H["Response: probability, label,<br/>model_version, artifact_sha256, threshold"]

    subgraph BATCH["Batch path"]
        BA["POST .../predict:batch<br/>(dataset_version_id or upload)"] --> BJ[("DB: job queued")]
        BJ --> BW["Worker: stream rows in chunks of 1000"]
        BW --> BP[("DB: prediction_batches + predictions")]
        BP --> BR["Job progress / failures per row"]
    end
```

Reproducibility: the response echoes `model_version_id`, `artifact_sha256`, `threshold` and the
`preprocessor`/`estimator` versions, so any prediction can be re-derived later even after a new version is
activated.

---

## 5. Global explanation

```mermaid
flowchart TD
    A["POST /model-versions/{id}/explanations/global<br/>{dataset_version_id, method, params}"] --> B["Compute cache_key"]
    B --> C{"Cache hit?<br/>(explanation_runs.cache_key)"}
    C -->|yes| R["Return stored run + artifact URIs<br/>(≤ 300 ms)"]
    C -->|no| D[("DB: explanation_runs status=running")]
    D --> E["Load dataset sample (n rows, seed)<br/>+ background sample"]
    E --> F["ExplainerRegistry.select(adapter, estimator)"]
    F --> G["SHAP values matrix (n × features)"]
    G --> H["Additivity check on sample rows"]
    H -->|fail| X["EXPLANATION_FIDELITY_ERROR<br/>run marked failed, nothing persisted as valid"]
    H --> I["Aggregate: mean|SHAP|, mean, std, rank<br/>derive source-column grouping"]
    I --> J["Render PNG+SVG: summary, beeswarm,<br/>dependence per selected feature"]
    J --> K["Stability: resample background (k=10),<br/>rank correlation + top-k overlap"]
    K --> L[("DB: global_explanations,<br/>feature_importances")]
    L --> M[("Object store: explanation artifacts")]
    M --> N["Run succeeded; UI poll returns result"]
```

Caching is content-addressed (`artifact_sha256 | reference_hash | input_hash | method | params |
library_versions`), so cached explanations are never stale — they are keyed on immutable inputs.

---

## 6. Local explanation

```mermaid
flowchart TD
    A["POST /predictions/{id}/explanation<br/>{methods: [shap, lime]}"] --> B["Load prediction + model version"]
    B --> C["SHAP local: base value + per-feature φ"]
    C --> D["Additivity check<br/>|base + Σφ − logit| ≤ 1e-6"]
    D -->|fail| E["EXPLANATION_FIDELITY_ERROR"]
    D --> F["Map encoded features → source columns<br/>attach raw values + readable labels"]
    F --> G["LIME: local surrogate,<br/>measure R² and local agreement"]
    G --> H["Narrative: deterministic template<br/>from ranked contributions only"]
    H --> I[("DB: explanation_runs + local_explanations<br/>(contributions, fidelity, narrative, caveats)")]
    I --> J["UI: waterfall, raw values, narrative,<br/>method labels, caveats"]
    J --> K["Export PDF/CSV (Stage 10/12)"]
```

SHAP and LIME are stored as **separate rows with separate methods**; the UI shows them side by side with
their fidelity numbers and never averages or merges them. Negative contributions are phrased as
"would have reduced the predicted risk, according to the model", never as prescriptions.

---

## 7. Fairness assessment

```mermaid
flowchart TD
    A["POST /model-versions/{id}/fairness-assessments<br/>{dataset_version_id, protected_attribute,<br/>privileged_group, favorable_label, threshold}"] --> B{"Attribute exists in dataset?"}
    B -->|no| E1["422 FAIRNESS_ATTRIBUTE_UNAVAILABLE<br/>(no fabricated groups)"]
    B -->|yes| C["Join predictions (same dataset version)<br/>with ground-truth labels"]
    C --> D["Group counts; drop missing groups,<br/>record excluded_missing"]
    D --> E["Per group: confusion matrix,<br/>selection rate, TPR/FPR, precision/recall/F1"]
    E --> F["Disparity metrics:<br/>DP diff/ratio, EO diff, equalised-odds diff"]
    F --> G["Flags: support < MIN_GROUP_SUPPORT,<br/>small-sample, conflicting criteria"]
    G --> H["Persist definitions + limitations<br/>(verbatim, rendered in reports)"]
    H --> I[("DB: fairness_assessments +<br/>fairness_group_metrics")]
    I --> J["Fairness dashboard + PDF report"]
```

The distribution join requires that the assessment dataset version has both labels and the group attribute.
If labels are absent, the API returns `FAIRNESS_LABELS_UNAVAILABLE` rather than computing metrics against
predictions.

---

## 8. Evaluation and calibration

```mermaid
flowchart TD
    A["POST /model-versions/{id}/evaluations<br/>{dataset_version_id, threshold}"] --> B["Score dataset with the pinned version"]
    B --> C["Discrimination: accuracy, precision, recall,<br/>F1, ROC-AUC, PR-AUC, confusion matrix"]
    B --> D["Calibration: Brier, ECE, reliability bins"]
    B --> E["Threshold sweep: precision/recall/F1/volume"]
    B --> F["Error analysis: FP/FN rows + feature context"]
    C --> G[("DB: evaluations<br/>metrics | confusion | calibration | curves")]
    D --> G
    E --> H[("DB: evaluation_metrics<br/>one row per threshold")]
    F --> I["Exportable FP/FN sample (respecting minimisation)"]
    G --> J["Reliability dashboard;<br/>version comparison joins two evaluations<br/>on the same dataset_version_id"]
```

Evaluation always uses an explicitly chosen dataset version and records it, so comparisons cannot silently
use different data.

---

## 9. Drift monitoring

```mermaid
flowchart TD
    A["POST /model-versions/{id}/drift-reports<br/>{current_dataset_version_id}"] --> B["Load reference + current samples"]
    B --> C["Per numeric feature: PSI, KS statistic and p-value"]
    C --> D["Per categorical feature: PSI, chi-square"]
    D --> E["Missing-rate delta, cardinality delta"]
    B --> F["Score current data → prediction PSI<br/>vs stored reference scores"]
    F --> G{"Ground-truth labels<br/>available?"}
    G -->|yes| H["Performance snapshot:<br/>metrics on current labelled data"]
    G -->|no| I["performance_snapshot = null<br/>(stated in report)"]
    E --> J["Status per feature:<br/>stable / warning / drift"]
    H --> K[("DB: drift_reports +<br/>drift_feature_results")]
    I --> K
    J --> K
    K --> L["Monitoring dashboard + trend view;<br/>report wording separates drift from degradation"]
```

---

## 10. Reporting and export

```mermaid
flowchart TD
    A["POST /reports {kind, source_id, format}"] --> B[("DB: job queued")]
    B --> C["Worker loads persisted run/assessment<br/>(never recomputes silently)"]
    C --> D["Render Jinja2 HTML<br/>embed PNG charts + tables"]
    D --> E["ReportLab PDF (pure-python) or CSV"]
    E --> F[("Object store: reports/{id}/file")]
    F --> G[("DB: reports + content_sha256 + expires_at")]
    G --> H["GET /reports/{id}/download<br/>(signed URL in prod, streamed in dev)"]
    C --> I[("audit_events: report.generate")]
```

Reports render from stored findings, so a report regenerated months later reproduces the original numbers
and caveats rather than recomputing against changed data.

---

## 11. Cross-cutting flows

### 11.1 Request lifecycle

```mermaid
sequenceDiagram
    participant C as Client
    participant MW as Middleware
    participant AU as Auth
    participant S as Service
    participant ML as ML core
    participant DB as Postgres
    participant ST as Artifact store
    C->>MW: HTTP request
    MW->>MW: assign/accept X-Request-ID, start timer
    MW->>AU: authenticate (JWT or API key), authorise role
    AU-->>MW: principal or 401/403 problem+json
    MW->>S: call use case
    S->>DB: read model/dataset metadata
    S->>ST: read artifact (hash-checked, cached)
    S->>ML: transform / predict / explain
    ML-->>S: result + fidelity check
    S->>DB: persist (transaction)
    S-->>MW: DTO
    MW-->>C: JSON + X-Request-ID + timing
    MW->>DB: audit event (mutations only)
```

### 11.2 Explanation cache lifecycle

```text
key = sha256(artifact_sha256 | reference_dataset_hash | input_row_hash | method | params | lib_versions)
INSERT ... ON CONFLICT (cache_key) DO NOTHING   → winner computes
loser polls explanation_runs.status             → returns stored result on success
failed run deletes its cache_key row            → next request recomputes
```

This prevents duplicate expensive work under concurrent requests without introducing a cache service.

### 11.3 Graceful shutdown

```text
SIGTERM → stop accepting new requests (readiness=false)
        → worker stops claiming jobs
        → in-flight jobs marked 'interrupted' with progress preserved
        → process pool drained, sessions closed
        → exit 0
On restart, 'interrupted' jobs are re-queued if attempts < max_attempts.
```

---

## 12. Data classification (drives masking and export rules)

| Class | Examples | Storage | Export |
|---|---|---|---|
| Direct identifier | `customerID` | Not stored raw; salted hash only | Never exported |
| Sensitive attribute | `gender`, `SeniorCitizen` | Stored for fairness only; excluded from minimised prediction payloads | Aggregated only, never per-row in reports |
| Quasi-identifier | `tenure`, `Contract`, `MonthlyCharges` | Stored according to minimisation policy | Allowed with retention policy applied |
| Model artefact | joblib/UBJ, background sample | Artifact store, hash-pinned | Not exportable via public API |
| Derived evidence | SHAP matrices, metrics, curves | JSONB + object store | Exportable (PDF/CSV) |
