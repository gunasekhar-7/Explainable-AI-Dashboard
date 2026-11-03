# Assessment of Existing Artifacts (Stage 0 investigation)

Two pre-existing local projects were inspected before designing Aletheia. This document records what was
measured, what is reusable, and what must be rebuilt. Nothing here is assumed: each claim was produced by
running the commands shown or by reading the cited file.

---

## 1. Customer Churn Prediction & Retention Intelligence Actions

Path: `C:\Projects\Customer-Churn-Prediction-Retention-Actions` (separate git repository, not modified).

### 1.1 What exists

| Artifact | Location | Status |
|---|---|---|
| Full dataset, 7,043 × 21, IBM Telco, Apache-2.0 | `data/Telco-Customer-Churn-FULL.csv` (gitignored) | Present, checksum-verified in its own docs |
| 100-row synthetic smoke sample | `data/Telco-Customer-Churn.csv` (tracked) | Present; explicitly non-production |
| Trained model artifact v1.0.0 | `models/model_v1.joblib` | Present and loadable |
| Preprocessing pipeline | `src/preprocessing.py` (`TotalChargesCleaner`, `ColumnTransformer` with impute/scale/one-hot) | Present |
| Training + serialisation script with quality gate | `scripts/train_and_serialize.py` | Present |
| FastAPI service, SQLAlchemy models, Alembic migrations, Streamlit dashboard, PSI/KS drift service, retention + pseudonymisation | `src/`, `dashboard/`, `alembic/`, `monitoring/` | Present |
| Tests (~20 files incl. integration) + CI workflow + Docker/Compose + CloudFormation | `tests/`, `.github/workflows/ci.yml`, `Dockerfile.*`, `cloudformation/` | Present |
| Model card, metrics, provenance, ADRs (6) | `docs/` | Present and detailed |

### 1.2 Measured properties of the model artifact

Command: load `models/model_v1.joblib`, inspect keys, transform the full dataset, score it, and check SHAP
additivity with `shap.LinearExplainer` against the stored background sample.

```
TYPE: <class 'dict'>
  model: LogisticRegression
  preprocessor: ColumnTransformer
  feature_names: list (46, unprefixed, e.g. 'tenure', 'gender_Female', 'Contract_Month-to-month')
  metadata: dict keys = version, threshold, timestamp, threshold_metadata, test_metrics,
                       reference_predictions, shap_background, data_provenance,
                       training_config, artifact_sha256
```

| Property | Measured value |
|---|---|
| Version / threshold | `1.0.0` / `0.6477892911405992` |
| Documented threshold rule | max precision s.t. recall ≥ 0.70, selected on **validation only** |
| Test metrics (stored in artifact) | ROC-AUC 0.8320003102, PR-AUC 0.6234734501, precision 0.5588235294, recall 0.6604278075, F1 0.6053921569, CM `[[840,195],[127,247]]` |
| Transformed matrix | 7,043 × 46; `predict_proba` succeeds |
| Mean predicted risk vs observed prevalence | 0.4172 vs 0.2654 (expected: `class_weight="balanced"` inflates probabilities) |
| SHAP background | stored, shape (100, 46) |
| SHAP additivity (3 rows, log-odds) | max \|base + Σφ − logit\| = **2.22e-16** |
| Top local drivers, row 0 | `tenure`, `TotalCharges`, `MonthlyCharges`, `InternetService_DSL`, `Contract_Month-to-month` |
| Feature-name alignment | Stored names are unprefixed; `get_feature_names_out()` returns `num__`/`cat__` prefixes. Stripping the prefix yields an exact one-to-one match. |

### 1.3 Findings that shape the design

1. **The artifact is not a single sklearn `Pipeline`.** It is a dict of `preprocessor` + `model` + explicit
   `feature_names`. The platform's adapter contract must therefore support **composite artifacts**, not only
   `Pipeline`. This is now a first-class requirement (FR-M-1).
2. **Feature-name mapping needs an explicit normalisation layer.** SHAP is computed on the 46 transformed
   columns, but the UI must show the original 21 raw fields with readable values. The platform will keep
   both: `transformed_feature` ↔ `source_column` mapping, derived from `ColumnTransformer` metadata and
   unit-tested against this artifact.
3. **The SHAP background sample is already stored in the artifact** (100 × 46), which makes global and local
   explanations reproducible without re-selecting a reference sample. Aletheia will instead own background
   selection (dataset version → deterministic sample → content hash) and treat the embedded sample as
   imported metadata, so caching keys remain under the platform's control.
4. **No library versions are recorded in the artifact metadata.** `training_config` holds model
   hyperparameters but not the scikit-learn/numpy versions used to pickle it. Pickled artifacts are
   version-fragile. Aletheia must record `framework_version`, `python_version` and dependency hashes at
   registration, warn on mismatch, and prefer re-serialisation into a manifest at Stage 2/3.
5. **The model is probability-miscalibrated by construction** (balanced class weights, mean score 0.417 vs
   26.5% prevalence). This is a genuine, useful demonstration for the reliability module: discrimination
   (ROC-AUC 0.832) is good while calibration is poor. The platform must report that contrast rather than
   hide it.
6. **Metrics are honest and internally consistent.** The project separates production metrics (full dataset)
   from notebook benchmarks and from 100-row sample runs, and warns that the sample fails its own gate. This
   discipline is adopted here.
7. **Fairness attributes are limited.** Telco provides `gender` (2 groups), `SeniorCitizen` (0/1),
   `Partner`, `Dependents`. There is no race/ethnicity, no age beyond the senior flag, no geography beyond
   none. Fairness analysis must therefore be presented as *group-difference analysis on available
   attributes*, with an explicit statement that absent attributes cannot be analysed and must not be
   invented.

### 1.4 Reuse decision

| Component | Decision | Rationale |
|---|---|---|
| Dataset + provenance/checksums | **Reuse** | Apache-2.0, documented, checksum-verified. Re-fetched by script, not committed. |
| `model_v1.joblib` + its metadata | **Reuse as first registered model version** | Loads and scores correctly; SHAP-additive; carries threshold, metrics and provenance. |
| Preprocessing definition (`src/preprocessing.py`) | **Reuse as reference implementation**, re-implemented in the platform's own module | The platform must own preprocessing for schema inference and retraining. |
| Training/threshold/quality-gate logic | **Reuse the methodology** (val-only threshold, held-out test, ROC-AUC gate) | Sound and already validated; re-implemented as the Stage 2 CLI. |
| FastAPI/SQLAlchemy/dashboard/DB code | **Do not copy** | Built for a single fixed model; Aletheia needs a generic registry. Copying would import a hard-coded schema and defeat the product's purpose. |
| Drift service, retention, pseudonymisation | **Reuse the designs and lessons**, re-implemented | Good patterns (PSI/KS thresholds, hashed IDs, bounded retention batches); the code is model-specific. |
| Tests, CI, Docker, model card | **Reuse as patterns** | Proven structure; new tests are written against the new contracts. |
| Original repository | **Leave untouched** | It remains a standalone portfolio project; Aletheia does not import from it at runtime. |

**Migration path:** at Stage 3, the artifact is copied into Aletheia's artifact store, hashed, and registered
through the public registration API with its metadata translated into the platform's manifest. At Stage 2, a
second model (XGBoost) is trained through Aletheia's own pipeline from the same dataset, giving the platform
two genuinely different model types (linear + tree ensemble) to validate adapter and explainer coverage, and
enabling honest model comparison.

---

## 2. Explainable Credit Risk Scoring (secondary candidate)

Path: `C:\Projects\Explainable Credit Risk Scoring`.

| Item | Status |
|---|---|
| Data | Home Credit `application_train.csv` is Kaggle-gated; only a 200-row synthetic sample is committed |
| Artifacts | `scorecard_model.pkl` → bare `LogisticRegression`; `xgboost_model.pkl` → bare `XGBClassifier` |
| Preprocessing / schema | **Not bundled with the artifacts** (WoE maps and feature engineering live in notebooks) |
| Explainability | SHAP notebook + `src/explain/shap_explainer.py`, no persisted explanations |
| Tests | A single `tests/test_pipeline.py` |

**Assessment:** the pickles are bare estimators with no attached preprocessor, feature schema, or training
split, so they cannot be registered directly without reverse-engineering the notebook pipeline and without
the real dataset. Doing so would require fabricating a feature contract — exactly the kind of unverified
claim this project forbids.

**Decision:** credit risk is documented as a *supported extension point* (the platform is model-agnostic and
its fairness engine is arguably more valuable there, since gender/age/family status are present), but it is
not part of v1's delivered scope. If added later, it must come with a re-trained, self-contained artifact
and its own provenance record.

---

## 3. Consequences for the roadmap

1. Stage 2 must produce a model through Aletheia's own pipeline (XGBoost) in addition to registering the
   existing logistic regression — this is the only way to prove the adapter abstraction rather than assume it.
2. Stage 3's registration API must accept composite artifacts, and record dependency versions.
3. Stage 4/5 must prove feature-name mapping against the 46-column one-hot layout, including the `num__`/
   `cat__` prefix quirk found here.
4. Stage 7's calibration module has a guaranteed real finding to report (good discrimination, poor
   calibration), so the module is validated on a genuine phenomenon rather than a synthetic one.
5. Fairness documentation must state the Telco attribute limitations up front, and demonstrate the
   insufficient-sample path using a real subgroup rather than a contrived one.
