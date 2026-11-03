# Risk Assessment — Aletheia

Scoring: likelihood (L) and impact (I) on 1–5. `Score = L × I`. Owner = the stage that must address it.
Residual = exposure after the stated mitigation.

---

## 1. Technical risks

| ID | Risk | L | I | Score | Mitigation | Stage | Residual |
|---|---|---|---|---|---|---|---|
| T1 | **Pickle artifact executes arbitrary code on load** | 3 | 5 | 15 | Hash-pinned imports, uploads off by default, subprocess load with timeout/memory cap, native XGBoost formats preferred, explicit trust-boundary documentation | 3, 11 | Medium — a malicious artifact in the operator-controlled import directory still executes inside the sandbox |
| T2 | **Artifact fails to load after a library upgrade** (sklearn pickle fragility) | 4 | 4 | 16 | Record framework/python versions at registration; pin runtime in `requirements.lock`; startup check reports incompatibility; re-serialise into a manifest during Stage 2 | 2, 3 | Low–Medium |
| T3 | **Feature-name mapping wrong** (one-hot expansion, `num__`/`cat__` prefixes) | 3 | 4 | 12 | Explicit `FeatureMap` derived from fitted transformer; golden-fixture test on the real 46-column artifact; raw-value joins asserted by test | 3, 4 | Low |
| T4 | **SHAP additivity violated or misapplied** (output space confusion: probability vs log-odds) | 3 | 5 | 15 | Additivity check before persistence; `output_space` recorded and displayed; tests on both linear and tree models | 4, 5 | Low |
| T5 | **Explanation instability presented as fact** | 4 | 3 | 12 | Stability report across background resamples, reported numerically; caveats persisted with results; no "global truth" wording | 4 | Medium — instability is inherent; the product reports it rather than removing it |
| T6 | **LIME interpreted as the model's reasoning** | 3 | 4 | 12 | Separate storage/endpoint/visual; fidelity numbers displayed; explicit non-interchangeability caveat; UI wording tests | 5 | Low |
| T7 | **KernelExplainer too slow / times out on unsupported estimators** | 3 | 3 | 9 | Row limits, sampling, hard timeout, `EXPLAINER_UNAVAILABLE` when parameters cannot be met | 4, 5 | Low |
| T8 | **Fairness conclusions from tiny groups** | 4 | 4 | 16 | Minimum support threshold, per-group support shown, `insufficient_support` flags, mandatory limitations text, no aggregate verdict | 6 | Medium — statistical power cannot be manufactured; the product states it |
| T9 | **Fairness metric implemented incorrectly** | 2 | 5 | 10 | Hand-computed fixtures + Fairlearn oracle parity ≤1e-9; confusion cells persisted for recomputation | 6 | Low |
| T10 | **Drift reported as accuracy loss** | 3 | 4 | 12 | Schema-level separation (`performance_snapshot` nullable), fixed report wording, tests asserting phrasing and null behaviour without labels | 8 | Low |
| T11 | **PSI/KS implementation error** (binning, empty bins, divergences) | 3 | 3 | 9 | Reference fixtures with known values; identical-distribution test (PSI=0); shifted-distribution detection test | 8 | Low |
| T12 | **Long jobs block or die mid-run** | 3 | 3 | 9 | DB-backed queue, chunked writes, progress, retries with classification, `interrupted` recovery on restart, graceful drain | 3, 11 | Low |
| T13 | **In-memory model cache serves a stale artifact after activation** | 2 | 4 | 8 | Cache keyed by `artifact_sha256`; activation clears the key; health check verifies artifact hash | 3 | Low |
| T14 | **Report PDF layout/dependency problems** (native libs, Windows) | 3 | 2 | 6 | ReportLab (wheel-only), HTML fallback, golden-file size/page-count checks | 12 | Low |
| T15 | **Dashboard bypasses the API** (direct DB access creeping in) | 2 | 4 | 8 | Architecture test forbidding DB/model imports in `dashboard/`; single `api_client` module | 1, 10 | Low |
| T16 | **Semantic drift between docs and code** | 4 | 3 | 12 | Stage gate requires doc update; OpenAPI snapshot; ADR updates when a decision changes | all | Medium — requires discipline; reviewed at each gate |

## 2. Data and scientific risks

| ID | Risk | L | I | Score | Mitigation | Stage | Residual |
|---|---|---|---|---|---|---|---|
| D1 | **Telco dataset lacks meaningful protected attributes** | 5 | 4 | 20 | State the limitation in PRD, fairness UI and reports; analyse only `gender`/`SeniorCitizen`/`Partner`/`Dependents`; never invent demographics; credit risk documented as a future extension | 0, 6 | High — inherent to the dataset; disclosed rather than hidden |
| D2 | **Dataset licence misused** | 2 | 4 | 8 | Apache-2.0 verified with checksum and attribution; dataset not committed; provenance persisted with the model version | 2 | Low |
| D3 | **Synthetic/smoke data presented as production metrics** | 3 | 5 | 15 | Metrics labelled by source dataset; CI fixture results never appear as performance claims; README separates the two explicitly | 2, 14 | Low |
| D4 | **Leakage in training/splits** | 2 | 5 | 10 | Split before fitting any transformer; preprocessing fitted on train only; test set untouched during selection; leakage test asserting transformer fit scope | 2 | Low |
| D5 | **Class imbalance distorts default-threshold conclusions** | 4 | 3 | 12 | Balanced class weights, threshold selected on validation by a stated rule, PR-AUC reported alongside ROC-AUC, calibration assessed separately | 2, 7 | Low |
| D6 | **Calibration claims confused with discrimination** | 3 | 4 | 12 | Separate columns/sections/tests; Brier + reliability curve; explicit sentence that the shipped model is miscalibrated by design | 7 | Low |
| D7 | **Fairness criteria conflict and the product implies a verdict** | 4 | 4 | 16 | Product never outputs "fair/unfair"; limitations text lists the conflict; no single-number badge | 6, 10 | Medium |

## 3. Engineering and delivery risks

| ID | Risk | L | I | Score | Mitigation | Stage | Residual |
|---|---|---|---|---|---|---|---|
| E1 | **Scope creep across 14 stages** | 4 | 4 | 16 | Stage gates with fixed acceptance criteria; explicit non-goals; no new technology without an ADR | all | Medium |
| E2 | **"Looks done" but unverified** | 3 | 5 | 15 | Every gate requires executed tests and recorded output; no completion claim without command evidence | all | Low |
| E3 | **Windows/Linux divergence** (file paths, `resource`, signals) | 4 | 3 | 12 | Docker is the reference environment; CI on Linux; Windows-specific limitations documented (sandbox limits, signal handling) | 1, 11 | Medium |
| E4 | **CI runtime creep** (>10 min) | 3 | 3 | 9 | Layered markers, Postgres service container, cached wheels, slow tests on a separate job | 1, 13 | Low |
| E5 | **Dependency breakage (pandas 3.x / sklearn 1.9 / shap)** | 4 | 4 | 16 | Locked requirements, version matrix check at Stage 1, integration smoke test on the real artifact early, adapter isolation from library churn | 1, 3 | Medium |
| E6 | **Secret leakage into git or logs** | 2 | 5 | 10 | `.env` gitignored, `.env.example` only, redaction filter, secret-scan in CI, no credentials in fixtures | 1, 11 | Low |
| E7 | **Coverage theatre** (tests execute, assertions weak) | 3 | 3 | 9 | Fixtures with known answers, Fairlearn oracle, mutation-style spot checks on key metrics, review of assertions at each gate | 12 | Medium |
| E8 | **Public demo cost or exposure** | 3 | 3 | 9 | Demo path costed and capped; rate limits; no real personal data (public dataset); seeded demo account with least privilege | 13 | Low |
| E9 | **No path for late-arriving labels** (found in design review) | 4 | 4 | 16 | `prediction_labels` table + `POST /model-versions/{id}/labels` + coverage endpoint; unlabelled rows reported as excluded | 04 §2.8, 06 §6.4.1 | Low |
| E10 | **Model memory and unpickling multiplied by the process pool** | 3 | 3 | 9 | One unpickle per artifact per worker, LRU ≤ 2, scrubbed child env, parent never unpickles, measured at Stage 4 | 03 §5.1 | Low |
| E11 | **Vulnerable dependency shipped inside a governance product** | 3 | 4 | 12 | SBOM per release, `pip-audit` gate in CI, ADR required for a new runtime dependency | NFR-14 | Low |
| E12 | **Breadth-first plan reaches late stages with nothing demonstrable** | 4 | 3 | 12 | Stage 1 split into foundation + walking skeleton; gate 1b requires a scored and explained real artifact from a cold start | 09 §0.1 | Low |
| E13 | **Illustrative API examples quoted back as measurements** | 2 | 3 | 6 | Examples labelled illustrative; fixtures carry `"example": true`; claims only from executed scripts | 06 §6.3 | Low |

## 4. Product and interpretation risks

| ID | Risk | L | I | Score | Mitigation | Stage | Residual |
|---|---|---|---|---|---|---|---|
| P1 | **Users read attributions as causal claims** | 5 | 4 | 20 | Fixed caveat on every explanation surface; wording rules ("according to the model"); glossary; no causal verbs in narratives | 5, 10 | Medium — the caveat reduces, not eliminates, misreading |
| P2 | **Executive UI hides model limitations** | 3 | 4 | 12 | Executive panels must include a limitations line; caveats non-dismissible | 10 | Low |
| P3 | **Stakeholder workflow too technical to use** | 3 | 3 | 9 | Two-layer pages (executive/technical), tooltips, plain labels, prediction card export | 10 | Low |
| P4 | **Portfolio claims outrun the implementation** | 4 | 4 | 16 | README/resume bullets only reference measured, reproducible results; "known limitations" section mandatory | 14 | Low |

---

## 5. Top five risks and what is actually done about them

1. **D1 — the Telco dataset cannot support a strong fairness audit (score 20).** Cannot be mitigated away.
   Handled by disclosure at three points (PRD, fairness UI, exported report) and by refusing to invent
   demographic fields. The credit-risk use case is documented as the place where fairness would matter more,
   and it is excluded from v1 rather than faked.
2. **P1 — causal misreading of SHAP (score 20).** Handled by fixed caveat text stored with each explanation,
   narrative templates that never use causal verbs, and a glossary that defines attribution. Residual risk
   remains because readers interpret; the platform's obligation is accurate framing, not mind control.
3. **T2/E5 — artifact and dependency fragility (score 16).** Handled by recording versions, pinning, an early
   integration smoke test against the real artifact (done in Stage 0), and re-serialisation at Stage 2.
4. **T8/D7 — fairness overreach (score 16).** Handled by minimum support, flags, mandatory limitations, and no
   fair/unfair verdict. Statistical power is a property of the data, not something code can fix.
5. **E1 — scope creep (score 16).** Handled by non-goals in the PRD, per-stage gates, and a rule that new
   technology requires an ADR with a rejected alternative.

---

## 6. Risks deliberately accepted

| Accepted risk | Why |
|---|---|
| Single-node deployment, no HA | Portfolio-scale product; HA would add cost and complexity without demonstrating anything new |
| In-process rate limiter is per-replica | Documented; correct at one replica; the interface allows a shared backend |
| Pickle support retained | Required by the scikit-learn support requirement; mitigated by the trust policy, not removed |
| Fairness analysis limited to available attributes | Truthful limitation; fabricating attributes would be worse |
| No automated fairness mitigation | Deliberate: mitigation changes the system under review and must be separately evaluated |
| Windows sandbox limits | Development convenience; production reference is Linux where limits are enforceable |
