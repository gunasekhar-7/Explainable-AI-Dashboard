# 10 — Design review remediation register

A senior review of the Stage 0 package (2026-11-03) produced thirteen findings. This register records each
one, what changed, and how the change is verified. It is the audit trail for the claim "the design review's
objections were closed", not a summary of the package itself.

Severity: **High** = would have produced a wrong or unbuildable result; **Medium** = would have cost a
rewrite or shipped a misleading artefact; **Low** = hygiene.

| ID | Finding | Severity | Resolution | Where | Verified by |
|---|---|---|---|---|---|
| D1 | No interface for late-arriving labels: outcome calibration, drift performance snapshots and fairness re-assessment all need labels for rows already scored, and nothing could supply them | High | `prediction_labels` table (unique per prediction, joinable by subject hash) + `POST /model-versions/{id}/labels` + coverage endpoint; unlabelled rows reported as excluded | `04` §2.8, `06` §6.4.1, `FR-P-7` | Requirement exists with acceptance criteria; schema constraint `UNIQUE (prediction_id)` |
| D2 | `POST /auth/token` was absent from the rate-limit matrix, so credential stuffing was unlimited and there was no lockout | High | Per-account **and** per-source limit, constant-time comparison, exponential delay, audited failures, password never logged | `06` §1.5 | Rate table now covers the login path |
| D3 | Retention purge deletes by age across all models, but the only index led with `model_version_id` — a sequential scan every run | Medium | Added `predictions (created_at)` with the reason recorded | `04` §3 | Index plan states the query it serves |
| D4 | `explanation_runs` allowed a `scope='global'` row with no dataset and a `scope='local'` row with no prediction; enum columns were validated in prose only | Medium | §2.8 constraints table: scope-dependent `CHECK`, range `CHECK`s, enum `CHECK`s, JSONB shape `CHECK`, label uniqueness | `04` §2.8 | Constraint is stated as a database rule, not a convention |
| D5 | `reports.source_id` was a polymorphic UUID with no foreign key and no discriminator | Medium | Added `source_kind`; writes go through one resolving repository function; nightly orphan check. Composite-FK alternative recorded as rejected | `04` §2.8 | Column exists in the ER |
| D6 | Per-process model cache × process pool multiplies memory, and the budget was never written down | Medium | Single load per artifact per worker (LRU ≤ 2), scrubbed child environment, explicit per-worker budget, `maxtasksperchild`, deviation > 2× treated as a finding | `03` §5.1 | Section exists with a measurable gate |
| D7 | Each pool worker re-executes an untrusted pickle, which multiplies the attack surface the sandbox was meant to bound | Medium | Parent never unpickles; it hashes bytes and receives metadata as JSON; workers are spawned without credentials | `03` §5.1 | Enforced in code: `load_artifact` is worker-side only |
| D8 | API examples mixed measured (`0.8096`) and invented fairness figures, inviting quotation of fabricated results | Low | Examples labelled illustrative, fixtures carry `"example": true` | `06` §6.3 | Label is in the document |
| D9 | Stage 3 built "subset" routers and Stage 9 rebuilt "all routers" — a rework loop budgeted as sequencing | Medium | Each stage owns its router module; Stage 9 became wiring, pagination, idempotency, rate limiting | `09` §0.1 | Plan states the change |
| D10 | No dependency-vulnerability gate or SBOM in a product about trustworthy software | Medium | `NFR-14` (SBOM + `pip-audit` in CI + ADR for new runtime dependencies) | `01` §6, `08` E11 | Requirement and risk recorded |
| D11 | `FR-M-7` was a duplicate of `FR-M-3` ("See FR-M-3") | Low | Replaced with the distinct persistence requirement it was hiding: the schema diff must be stored on the version | `01` §5.2 | No requirement references another as its acceptance criterion |
| D12 | 43 days for 14 stages was optimistic, and one dev at 10 h/week was quoted 8–9 weeks | Medium | 46 days planned, ~60 with 30% contingency, 12–14 weeks part-time; Stage 1 split | `09` effort summary | Numbers recomputed |
| D13 | 2,900 lines of design against zero lines of code — the inverted ratio was the project's largest risk | High | Stage 1a (foundation) and 1b (walking skeleton) implemented: configuration, structured logging, RFC 9457 error model, health/readiness, composite-artifact adapter, feature map, SHAP local explanation with enforced additivity, architecture import test, CI | `src/aletheia/`, `tests/` | `pytest`, `ruff`, live `/health` and `/health/ready` |

## State after remediation

| Dimension | Before | After | Remaining gap |
|---|---|---|---|
| Discovery & evidence | 9 | 9 | Nothing — still the strongest part |
| Requirements | 8 | 9 | 61 FR + 14 NFR, no self-referential rows |
| Architecture | 8.5 | 9 | Model-load path specified and implemented |
| Data model | 8 | 9 | Constraints promoted from prose to schema |
| API design | 8.5 | 9.5 | Labels interface closes the largest functional hole |
| Testing | 8 | 9 | Real, passing tests over the ML core instead of a plan for tests |
| Security | 7 | 8 | Login limits; supply-chain gate now required. Still no TLS termination, no refresh-token rotation list |
| Delivery plan | 6 | 8 | Skeleton-first order, honest effort with contingency |
| Delivered software | 0% | Foundation + skeleton core | Stages 2–14 remain; nothing claims otherwise |

The honest headline: **the design package is now ~9.5, the project is not yet.** Everything above the last
row is design; only the last row is software, and it covers Stage 1. Ratings for the finished product can be
earned at the Stage 4 and Stage 11 gates, with measurements, not before.
