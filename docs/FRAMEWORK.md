# FRAMEWORK.md

Formal mathematical framework for simulating mixed AI agent/human developer software teams. This document defines the stochastic system whose dynamics are implemented in the Python Monte Carlo simulation. All variables are defined in VARIABLES.md; this document assembles them into equations of motion, a detection pipeline, a queuing model, stability conditions, and the Pareto optimization procedure.

---

## Notation Conventions

| Convention | Meaning |
|---|---|
| t ∈ {0, 1, …, T−1} | Sprint index (integer); each step is exactly one sprint |
| k | Replication index within a Monte Carlo run |
| S(t) | Full system state vector at start of sprint t |
| (N, M, **A**, R) | Configuration tuple (fixed per simulation run) |
| a_k | Count of agents of role type k (k = 1 … 13) |
| X ~ Dist(·) | X is drawn from the specified distribution |
| ⌈x⌉ | Ceiling of x |
| ⌊x⌋ | Floor of x |
| clip(x, a, b) | max(a, min(b, x)) |
| `X(t+1) = f(·)` | Between-sprint state transition: defines the value written at the end of sprint t / start of sprint t+1 |
| `X(t) ← g(·)` | Within-sprint update: mutates the working value of X during sprint t; the final value written into state at the end of the sprint is the last `←` assignment |
| Scenario-level | Sampled once per replication; constant across all T sprints |
| Step-level | Sampled at each sprint t |
| Derived | Computed deterministically from state and parameters |

---

## Section 1 — System Overview and Modeling Assumptions

The model represents a software development team as a discrete-time stochastic dynamical system. At each sprint t, features are implemented, bugs are injected and detected, human capacity is allocated, queues evolve, and costs accumulate. The simulation runs K independent Monte Carlo replications; each replication draws a fresh set of scenario-level parameters and evolves the full state over T sprints.

The team configuration (N, M, **A**, R) is fixed for all replications within a single simulation run and is the unit of Pareto analysis.

### 1.1 Modeling Assumptions

**A1 — Sprint atomicity.** All arrivals, completions, detections, fixes, and queue updates within sprint t are computed as aggregate counts. Within-sprint ordering is not tracked.

**A2 — Agent homogeneity within type.** All a_k agents of role type k are statistically identical per sprint. Individual agent heterogeneity is captured by inter-scenario variation in the shared per-sprint rate distributions.

**A3 — Within-sprint detection (Stage A) vs. delayed detection (Stage B).** Bugs injected in sprint t are subject to **Stage A** detection — Code Review (CR) and Security Audit (SA) — within the same sprint t, since these layers operate directly on the code change. Bugs that survive Stage A enter a latency buffer and become visible to the **Stage B** test layers (Unit, Integration, System/Acceptance tests) only after ⌈τ_CI(t)⌉ sprints have elapsed. Throughout this document, *Stage A* = CR + SA (synchronous), *Stage B* = UT + IT + ST (delayed by CI/CD latency).

**A4 — Human capacity is pooled per developer and prioritized.** Each Human Developer has a total budget H_capacity hours per sprint. Competing demands are served in strict priority order: (1) peer communication overhead, (2) agent monitoring, (3) architecture review queue, (4) escalation queue, (5) direct coding. Residual capacity falls to zero if demand exceeds supply; queues grow accordingly.

**A5 — Release cadence is fixed.** A release event occurs exactly every R sprints, at sprint indices R, 2R, 3R, … All hidden bugs surviving the detection pipeline at a release event escape to production.

**A6 — Regression bugs skip same-sprint detection.** Bugs introduced as regressions (from fixing or refactoring) are treated as having already survived the code review of the triggering change and enter the latency buffer directly without passing through Stage A.

**A7 — Severity and security class are orthogonal, independent per-bug labels.** Each injected bug is independently classified as major with probability θ (minor with probability 1−θ) **and independently** as security-class with probability `p_sec` (non-security with probability 1−`p_sec`), regardless of which phase or agent produced it. A single bug can therefore fall into any of the four combinations (security/non-security × major/minor); the Stage A detection pipeline in §4.2 relies on this independence when assigning bugs to Code Review (non-security) or Security Audit (security).

**A8 — Code Review throughput is modeled as unbounded.** The number of changes reviewed per sprint by the Code Review Agent is **not** rate-limited by `a_6 × r_review`. All injected non-security bugs pass through Stage A Code Review; detection quality is captured entirely by `d_CR_eff(t)`, which degrades with change size (via `α_size`) and with the architecture-review queue overload (via `η_rev(t)`). Nevertheless, each deployed Code Review Agent accrues cost through `c_token_6` in §8.1, so an oversized `a_6` is penalized by the cost objective `C_F`. Pareto optimality therefore places an **implicit upper bound on `a_6`**: configurations with excess CR capacity are dominated in `C_F` by lighter configurations of equal or better quality. The scenario parameter `r_review` is retained for potential future refinements and cost-model calibration; it does not appear in any sprint-transition equation in §4. Implementers should not add a throughput gate for CR.

---

## Section 2 — Team Configuration and State Vector

### 2.1 Team Configuration

A team configuration is the tuple (N, M, **A**, R) where:

- **N ∈ ℕ≥1** — number of Human Developers
- **M ∈ ℕ≥1** — span of control (agents per Human Developer)
- **A = (a_1, …, a_13) ∈ ℕ^13** — agent type composition, subject to Σ_{k=1}^{13} a_k = N×M
- **R ∈ ℕ≥1** — release cadence in sprints (fixed integer, not sampled)

Agent type indices follow ROLES.md:

| k | Role |
|---|---|
| 1 | Requirements Refinement Agent |
| 2 | Architecture & Design Proposal Agent |
| 3 | Feature Implementation Agent |
| 4 | Bug Fixing Agent |
| 5 | Refactoring Agent |
| 6 | Code Review Agent |
| 7 | Security Audit Agent |
| 8 | Unit Test Development Agent |
| 9 | Integration Test Development Agent |
| 10 | System & Acceptance Test Development Agent |
| 11 | DevOps/CI-CD Agent |
| 12 | Developer Documentation Agent |
| 13 | User Documentation Agent |

The vector **A** indexes **AI agent roles only**. Human Developers (ROLES.md role 15) are specified by the scalar `N` and allocated across tasks by the equations in Section 3. The Product Owner (ROLES.md role 14) is modeled as a fixed singleton whose cost enters via `c_PO` in Section 8.1 and is not a component of **A**.

### 2.2 State Vector

The complete system state at sprint t is:

```
S(t) = (
  F_backlog(t),      -- features awaiting implementation
  F_completed(t),    -- cumulative features shipped
  B_hid_min(t),      -- hidden minor bugs (post-Stage-A, in buffer or undetected)
  B_hid_maj(t),      -- hidden major bugs
  B_kno_min(t),      -- known minor bugs queued for fixing
  B_kno_maj(t),      -- known major bugs queued for fixing
  B_escaped(t),      -- cumulative bugs shipped to production
  D(t),              -- technical debt level ∈ [0, 1]
  Q_rev(t),          -- architecture review queue length
  Q_esc(t),          -- escalation queue length
  tests_UT(t),       -- cumulative unit tests written
  tests_IT(t),       -- cumulative integration tests written
  tests_ST(t),       -- cumulative system/acceptance tests written
  doc_cov(t)         -- developer documentation coverage ∈ [0, 1]
)
```

The following derived quantities are computed from S(t) and sampled parameters at each sprint and are not stored as explicit state:

- τ_CI(t) — CI/CD pipeline latency (Section 4.10)
- cov_UT(t), cov_IT(t), cov_ST(t) — test coverage levels (Section 4.9)
- n_tests(t) = β_UT × tests_UT(t) + β_IT × tests_IT(t) + β_ST × tests_ST(t)

**Latency buffer.** In addition to S(t), the simulation maintains per-severity FIFO buffers recording surviving bug cohorts by sprint of injection:

```
buf_min[t_inj] -- minor bugs surviving Stage A, injected at sprint t_inj
buf_maj[t_inj] -- major bugs surviving Stage A, injected at sprint t_inj
```

Each sprint t, the cohort from sprint t_inj becomes detectable by test layers when t ≥ t_inj + ⌈τ_CI(t_inj)⌉. The maximum buffer length needed is L_buf = ⌈τ_CI_max⌉ + 1, where τ_CI_max is bounded analytically (see Lemma 6.1).

**Implementation.** The natural representation is a pair of dictionaries `buf_maj: {t_inj → count}`, `buf_min: {t_inj → count}` keyed by injection sprint. At each sprint t:
1. New cohorts are inserted at key `t_inj = t` (§4.2).
2. All keys `t_inj` with `t ≥ t_inj + ⌈τ_CI(t_inj)⌉` are *popped* from the dictionary, summed into `(C_maj, C_min)`, and passed through Stage B (§4.3). Note that the `τ_CI(t_inj)` used for this check is the *injection-time* latency snapshot, stored alongside the cohort at insertion (see Definition 6.1).
3. Regression cohorts from §4.7 are added to `buf_min[t]` with `τ_CI` set to the current-sprint value.
4. At a release event (§4.11), all remaining dictionary entries are summed into `B_escaped` and the dictionaries are cleared.

An equivalent array-based implementation pre-allocates `L_buf` slots and treats the structure as a circular buffer indexed by `t_inj mod L_buf`.

### 2.3 Initial Conditions

At t = 0:

| Variable | Initial Value |
|---|---|
| F_backlog(0) | 0 (or warm-start F_0 ~ Poisson(λ_F × R_warmup); see note below) |
| F_completed(0) | 0 |
| B_hid_min(0), B_hid_maj(0) | 0 |
| B_kno_min(0), B_kno_maj(0) | 0 |
| B_escaped(0) | 0 |
| D(0) | ~ Be(α_D0, β_D0) [greenfield: α_D0 >> β_D0; legacy: reverse] |
| Q_rev(0), Q_esc(0) | 0 |
| tests_UT(0), tests_IT(0), tests_ST(0) | 0 |
| doc_cov(0) | 0 |
| buf_min[·], buf_maj[·] | all zero |

`R_warmup ∈ ℕ≥0` is a scenario-level parameter representing the number of pre-simulation sprints assumed to have generated the starting backlog. `R_warmup = 0` yields a cold start (empty backlog at t=0); `R_warmup > 0` represents a running system whose feature pipeline was already in flight when the simulation window begins.

---

## Section 3 — Human Capacity Allocation

Human capacity allocation must be computed before the sprint transition equations since feature and bug processing rates depend on the residual capacity available. All quantities are per-developer per sprint (hours), then multiplied by N for team totals.

### 3.1 Gross Capacity After Communication Overhead

```
H_net(t) = max(0, H_capacity − c_comm × (N − 1))
```

If H_net(t) = 0, the team has collapsed under communication overhead. This is an infeasible configuration (see Stability Condition 3, Section 7).

### 3.2 Monitoring Overhead

Documentation coverage reduces the per-agent monitoring burden:

```
H_mon_eff(t) = H_monitoring_overhead × (1 − Γ_doc × doc_cov(t))
H_mon_total(t) = min(H_net(t), M × H_mon_eff(t))
H_after_mon(t) = H_net(t) − H_mon_total(t)
```

### 3.3 Architecture Review Service

```
H_for_rev(t) = min(H_after_mon(t), Q_rev(t) × H_review_time / N)
H_after_rev(t) = H_after_mon(t) − H_for_rev(t)
```

Reviews completed this sprint by the whole team:

```
rev_done(t) = ⌊H_for_rev(t) × N / H_review_time⌋
```

Review quality degradation (applied to architecture proposals processed this sprint):

```
η_rev(t) = max(η_min_rev, 1 − α_η_rev × max(0, Q_rev(t) − Q_thr_rev))
```

### 3.4 Escalation Service

```
H_for_esc(t) = min(H_after_rev(t), Q_esc(t) × H_escalation_time / N)
H_after_esc(t) = H_after_rev(t) − H_for_esc(t)
```

Escalations resolved this sprint:

```
esc_done(t) = ⌊H_for_esc(t) × N / H_escalation_time⌋
```

Escalation quality degradation:

```
η_esc(t) = max(η_min_esc, 1 − α_η_esc × max(0, Q_esc(t) − Q_thr_esc))
```

### 3.5 Residual Coding Capacity

```
H_code(t) = f_code × H_after_esc(t)    [hours per developer]
```

Human-implemented features this sprint:

```
impl_human(t) = ⌊H_code(t) × N / h_feature⌋
```

where h_feature ~ LN(μ_hf, σ_hf) is the mean human hours required per feature (scenario-level parameter). Human-implemented code has a lower bug injection rate than agent code; this is handled via a modifier λ_human_scale ∈ (0, 1] applied to the injection rate for the impl_human(t) portion (see Section 4.2).

---

## Section 4 — Sprint Transition Equations

The following steps are executed in strict dependency order within each sprint t. All quantities indexed by t that appear on the right-hand side of an equation refer to their **start-of-sprint** value; quantities indexed by t+1 are the post-update values written into state at the end of the sprint.

**Canonical within-sprint order.** Readers and implementers should process §4.1 through §4.11 in exactly this sequence:

1. §4.1 — Feature arrivals, architecture blocking from `Q_rev(t)`, agent+human implementation.
2. §4.2 — Bug injection using `q_arch_eff(t)` computed from `Q_rev(t)` (start of sprint, before §4.4 updates the queue), Stage A detection.
3. §4.3 — Stage B detection using `τ_CI(t)` (last sprint's value), `tests_k(t)`, `F_completed(t)`.
4. §4.4 — Architecture review queue update (`Q_rev(t)` → `Q_rev(t+1)`).
5. §4.5 — Escalation queue update (`Q_esc(t)` → `Q_esc(t+1)`).
6. §4.6 — Bug fixing against `B_kno` after Stage A/B arrivals.
7. §4.7 — Regression injection into `buf_min[t]`.
8. §4.8 — Technical debt update (`D(t)` → `D(t+1)`).
9. §4.9 — Test suite and documentation update (`tests_k(t+1)`, `doc_cov(t+1)`).
10. §4.10 — CI/CD latency update using post-update test counts (`τ_CI(t+1)`).
11. §4.11 — Release event if `(t+1) mod R = 0`.

In particular, `q_arch_eff(t)` in §4.2 uses `Q_rev(t)` (start of sprint), **not** the post-update queue `Q_rev(t+1)` from §4.4. This avoids a within-sprint circular dependency between §4.2 (bug injection) and §4.4 (queue update).

### 4.1 Feature Arrival and Implementation

New features arrive from the Product Owner:

```
arrivals(t) ~ Poisson(λ_F)
```

Architecture blocking: when the review queue is long, Feature Implementation Agents cannot safely implement against unreviewed designs:

```
b_F(t) = min(1, Q_rev(t) / Q_thr_block)
```

Agent-implemented features (bounded by backlog and blocked by pending reviews):

```
impl_agents(t) = min(F_backlog(t), ⌊r_F × ψ_D(D(t)) × a_3 × (1 − b_F(t))⌋)
```

Total features implemented this sprint:

```
impl(t) = impl_agents(t) + impl_human(t)
```

Backlog and completion updates:

```
F_backlog(t+1) = max(0, F_backlog(t) + arrivals(t) − impl(t))
F_completed(t+1) = F_completed(t) + impl(t)
```

### 4.2 Bug Injection

Effective bug injection rate per feature (modulated by debt, specification quality, and architecture quality):

```
λ_impl_eff(t) = λ_impl × φ_D(D(t)) × (1 + α_spec × ε_spec) × (2 − q_arch_eff(t))
```

The factor (2 − q_arch_eff(t)) spans [1, 2]: perfect effective architecture quality (q_arch_eff = 1) gives no amplification; zero-quality architecture (q_arch_eff = 0) doubles injection. This factor should be clipped to [1, 2]. q_arch_eff(t) is defined in §4.4 as q_arch × η_rev(t) + (1 − η_rev(t)) × q_arch_floor; it degrades when the review queue is overloaded. Both q_arch_eff and η_rev(t) are computed from Q_rev(t) at the **start** of the sprint before any queue update.

**Interpretation of ε_spec.** `ε_spec` is sampled once per replication and applied uniformly to every feature implemented in that replication — it represents the *scenario-level average* specification-error rate, not a per-feature Bernoulli. All features in the same replication therefore experience the same `(1 + α_spec × ε_spec)` multiplier; per-feature variation in spec-induced injection is absorbed into the Poisson noise in `bugs_raw(t)`.

**Simplification on spec-bug detectability.** Bugs injected via specification errors are *not* tagged separately in this model; they are subject to the same Stage A (`d_CR`, `d_SA`) and Stage B (`d_UT`, `d_IT`, `d_ST`) detection rates as any other implementation bug. In reality, specification bugs are typically harder to catch in Code Review (the code is correct against the flawed spec) but more likely to fail System/Acceptance tests (which encode the intended behavior). Modeling this class-specific detection profile is deferred to a calibration variant; the aggregate rates here capture the first-order effect.

Total bugs injected from agent-implemented features:

```
bugs_raw(t) ~ Poisson(impl_agents(t) × λ_impl_eff(t))
```

Bugs from human-implemented features (lower injection rate):

```
bugs_human_raw(t) ~ Poisson(impl_human(t) × λ_impl_eff(t) × λ_human_scale)
```

Combined:

```
bugs_total(t) = bugs_raw(t) + bugs_human_raw(t)
bugs_maj(t) ~ Binomial(bugs_total(t), θ)
bugs_min(t) = bugs_total(t) − bugs_maj(t)
```

**Stage A detection** is applied immediately (same sprint) before bugs enter the latency buffer. CR detects non-security-class bugs; SA detects security-class bugs. Per Assumption A7, severity (θ) and security class (`p_sec`) are orthogonal: each bug is independently major/minor and independently security/non-security. A fraction `p_sec` of all bugs are security-class:

```
bugs_sec_maj(t) ~ Binomial(bugs_maj(t), p_sec)
bugs_sec_min(t) ~ Binomial(bugs_min(t), p_sec)
bugs_nonsec_maj(t) = bugs_maj(t) − bugs_sec_maj(t)
bugs_nonsec_min(t) = bugs_min(t) − bugs_sec_min(t)
```

Effective Code Review detection rate (degrades with change size). Per Assumption A8, CR is not throughput-limited by `a_6`; degradation arises only from review-queue overload (`η_rev(t)`) and change size (`α_size`):

```
d_CR_eff(t) = d_CR × η_rev(t) × max(0, 1 − α_size × log((impl(t) + 1) / baseline_size))
```

CR detections (non-security bugs):

```
CR_det_maj(t) ~ Binomial(bugs_nonsec_maj(t), d_CR_eff(t))
CR_det_min(t) ~ Binomial(bugs_nonsec_min(t), d_CR_eff(t))
```

SA detections (security bugs):

```
SA_det_maj(t) ~ Binomial(bugs_sec_maj(t), d_SA)
SA_det_min(t) ~ Binomial(bugs_sec_min(t), d_SA)
```

All Stage A detections flow to the known bug queue:

```
B_kno_maj(t) ← B_kno_maj(t) + CR_det_maj(t) + SA_det_maj(t)
B_kno_min(t) ← B_kno_min(t) + CR_det_min(t) + SA_det_min(t)
```

Bugs surviving Stage A enter the latency buffer at position t:

```
buf_maj[t] = bugs_nonsec_maj(t) − CR_det_maj(t) + bugs_sec_maj(t) − SA_det_maj(t)
buf_min[t] = bugs_nonsec_min(t) − CR_det_min(t) + bugs_sec_min(t) − SA_det_min(t)
```

### 4.3 Stage B Detection (Test Layers)

Compute current τ_CI (see Section 4.10 for the derivation):

```
τ_CI(t) = τ_CI_base × (1 + α_F_CI × F_completed(t)) × (1 + α_test_CI × n_tests(t))
τ_CI(t) = min(τ_CI(t), T)          -- cap to prevent exceeding simulation horizon
```

Determine the detectable cohort: all cohorts injected at sprint t_inj such that t ≥ t_inj + ⌈τ_CI(t_inj)⌉. Let this cohort be (C_maj, C_min) = sum of buf_maj[t_inj] and buf_min[t_inj] over all eligible t_inj, then clear those buffer slots.

Compute test coverage levels:

```
cov_UT(t) = min(c_UT_ceil, tests_UT(t) / max(1, F_completed(t)))
cov_IT(t) = min(c_IT_ceil, tests_IT(t) / max(1, F_completed(t)))
cov_ST(t) = min(c_ST_ceil, tests_ST(t) / max(1, F_completed(t)))
```

Apply Stage B detection sequentially (each layer sees the survivors of all previous layers). The parameters `d_UT`, `d_IT`, `d_ST` represent the **maximum** per-layer detection rate (achieved at full coverage); the effective per-layer rate is `d_k × cov_k(t)`, so that `cov_k(t) = 0` yields no detection and `cov_k(t) = c_k_ceil` yields `d_k × c_k_ceil` (see VARIABLES.md §4 for the definition of `d_k` as a maximum).

**Unit tests:**

```
UT_det_maj ~ Binomial(C_maj, d_UT × cov_UT(t))
UT_det_min ~ Binomial(C_min, d_UT × cov_UT(t))
C_maj ← C_maj − UT_det_maj
C_min ← C_min − UT_det_min
```

**Integration tests:**

```
IT_det_maj ~ Binomial(C_maj, d_IT × cov_IT(t))
IT_det_min ~ Binomial(C_min, d_IT × cov_IT(t))
C_maj ← C_maj − IT_det_maj
C_min ← C_min − IT_det_min
```

**System/acceptance tests:**

```
ST_det_maj ~ Binomial(C_maj, d_ST × cov_ST(t))
ST_det_min ~ Binomial(C_min, d_ST × cov_ST(t))
C_maj ← C_maj − ST_det_maj
C_min ← C_min − ST_det_min
```

Stage B detections flow to the known bug queue:

```
B_kno_maj(t) ← B_kno_maj(t) + UT_det_maj + IT_det_maj + ST_det_maj
B_kno_min(t) ← B_kno_min(t) + UT_det_min + IT_det_min + ST_det_min
```

Bugs remaining in (C_maj, C_min) after all Stage B layers are still hidden. Update hidden stock:

```
B_hid_maj(t) = B_hid_maj(t) + buf_maj[t] − (total Stage B major detections from cohort processed)
B_hid_min(t) = B_hid_min(t) + buf_min[t] − (total Stage B minor detections from cohort processed)
```

**Note on hidden stock tracking.** B_hid_s(t) is the total stock of hidden bugs: bugs in the latency buffer (not yet testable) plus bugs that have been through Stage B but survived all test layers. The B_hid values are useful for reporting; the operative distinction for release is "bugs in the buffer or surviving Stage B that have not yet been fixed."

### 4.4 Architecture Review Queue Dynamics

New architecture proposals from Design Agents this sprint:

```
arch_new(t) ~ Poisson(r_arch × a_2)
```

Queue dynamics:

```
Q_rev(t+1) = max(0, Q_rev(t) + arch_new(t) − rev_done(t))
```

where rev_done(t) was computed in Section 3.3. When η_rev(t) < 1, proposals reviewed during an overloaded sprint are approved with degraded quality. The architectural quality for sprint t proposals is modeled as:

```
q_arch_eff(t) = q_arch × η_rev(t) + (1 − η_rev(t)) × q_arch_floor
```

where q_arch_floor ~ Be(α_qf, β_qf) is a scenario-level floor representing the worst-case quality when reviews are entirely rushed. q_arch_eff(t) replaces q_arch in λ_impl_eff(t) for features implemented against designs reviewed in sprint t.

### 4.5 Escalation Queue Dynamics

New escalations from agents across the whole team:

```
esc_new(t) ~ Poisson(N × M × ρ_esc)
```

Queue dynamics:

```
Q_esc(t+1) = max(0, Q_esc(t) + esc_new(t) − esc_done(t))
```

where esc_done(t) was computed in Section 3.4. Escalations handled within a sprint are always resolved successfully — there is no re-escalation.

### 4.6 Bug Fixing

Fix rates per severity, slowed by technical debt:

```
r_fix_min_eff(t) = r_fix_minor × ψ_D(D(t))
r_fix_maj_eff(t) = r_fix_major × ψ_D(D(t))
```

Fixes this sprint (bounded by known queue):

```
fixes_min(t) = min(B_kno_min(t), ⌊r_fix_min_eff(t) × a_4 × (1 − θ)⌋)
fixes_maj(t) = min(B_kno_maj(t), ⌊r_fix_maj_eff(t) × a_4 × θ⌋)
```

Known bug queue update (post-fixing):

```
B_kno_min(t+1) = B_kno_min(t) − fixes_min(t)   [plus Stage A/B arrivals above]
B_kno_maj(t+1) = B_kno_maj(t) − fixes_maj(t)
```

### 4.7 Regression Injection

Fixes and refactoring introduce regression bugs (all classified as **minor**; severe refactoring regressions are handled via the ordinary bug-injection pathway in §4.2 and are not remodeled here) that enter the latency buffer directly (skipping Stage A, per Assumption A6):

```
reg_from_fixes(t) ~ Binomial(fixes_min(t) + fixes_maj(t), p_regress)
reg_from_refactor(t) ~ Binomial(round(a_5 × r_refactor), p_regress_refactor)
regressions(t) = reg_from_fixes(t) + reg_from_refactor(t)
```

**Modeling note on refactoring regressions.** `r_refactor` is a continuous work-rate (units of debt reduction per sprint per agent; see VARIABLES.md §3). The binomial treats each unit of refactoring work as an independent Bernoulli trial with success probability `p_regress_refactor`. The rounding in `round(a_5 × r_refactor)` is required because the binomial's `n` parameter is an integer; for typical `r_refactor` in the 1–5 range and `p_regress_refactor` ≪ 1, rounding introduces negligible bias in the expected regression count `a_5 × r_refactor × p_regress_refactor`.

Regressions are added to buf_min[t], so they are processed in Stage B after τ_CI(t) sprints:

```
buf_min[t] ← buf_min[t] + regressions(t)
```

**Simplification on regression detectability.** Regression bugs are subject to the same Stage B detection rates (`d_UT × cov_UT`, etc.) as newly-injected bugs. In reality, regressions are often easier to catch (previously-passing tests now fail), so the model is conservative on regression detection. A regression-specific detection multiplier could be added as a future calibration extension.

### 4.8 Technical Debt Update

```
D(t+1) = clip(
  D(t)
  + δ_impl × impl(t)
  + δ_fix × (fixes_min(t) + fixes_maj(t))
  − δ_refactor × a_5 × r_refactor,
  0, 1
)
```

**Units.** `D(t)` is dimensionless on `[0, 1]`. `δ_impl` is debt increment per feature (dimensionless/feature), multiplied by `impl(t)` (features/sprint) → debt/sprint. `δ_fix` is debt increment per fix, multiplied by the fix count. `δ_refactor` is debt reduction per unit of refactoring work, multiplied by `a_5 × r_refactor` (units of refactoring work per sprint across all refactoring agents) → debt reduction per sprint. All terms therefore have units of `debt/sprint`, matching the LHS increment `D(t+1) − D(t)`.

### 4.9 Test Suite and Documentation Update

New tests written this sprint:

```
Δtests_UT(t) = r_UT × a_8
Δtests_IT(t) = r_IT × a_9
Δtests_ST(t) = r_ST × a_10
```

Cumulative test counts:

```
tests_UT(t+1) = tests_UT(t) + Δtests_UT(t)
tests_IT(t+1) = tests_IT(t) + Δtests_IT(t)
tests_ST(t+1) = tests_ST(t) + Δtests_ST(t)
```

Coverage (derived, used in Stage B at sprint t+1):

```
cov_k(t+1) = min(c_k_ceil, tests_k(t+1) / max(1, F_completed(t+1)))   for k ∈ {UT, IT, ST}
```

Documentation coverage update:

```
doc_cov(t+1) = clip(
  doc_cov(t) + (r_doc × a_12 − ζ_doc × impl(t)) / max(1, F_completed(t+1)),
  0, 1
)
```

The term ζ_doc × impl(t) models coverage erosion: each new feature that ships without documentation dilutes the fraction of the codebase that is documented.

### 4.10 CI/CD Pipeline Latency Update

```
n_tests(t+1) = β_UT × tests_UT(t+1) + β_IT × tests_IT(t+1) + β_ST × tests_ST(t+1)

τ_CI(t+1) = τ_CI_base × (1 + α_F_CI × F_completed(t+1)) × (1 + α_test_CI × n_tests(t+1))
τ_CI(t+1) = min(τ_CI(t+1), T)
```

Flag τ_CI(t) > R as pathological: the pipeline takes longer than one full release cycle, meaning bugs injected in one release window cannot be detected before the release.

### 4.11 Release Event

If (t+1) mod R = 0:

```
B_escaped(t+1) = B_escaped(t) + all bugs remaining in buffer slots and surviving Stage B
                               that have not been added to B_kno
```

Concretely, the production-escaped bugs are the union of:
1. Bugs still in the latency buffer (injected too recently to have been processed by Stage B), and
2. Bugs that exited Stage B but were not detected (the residual (C_maj, C_min) after all three test layers).

---

## Section 5 — Degradation Function Definitions

### 5.1 Technical Debt Bug Amplifier: φ_D(D)

```
φ_D(D) = 1 + (φ_max − 1) × D^κ
```

| Property | Value |
|---|---|
| φ_D(0) | 1 (no amplification at zero debt) |
| φ_D(1) | φ_max (maximum amplification) |
| Monotonicity | Strictly increasing on [0, 1] |
| Convexity | Convex for κ > 1; linear for κ = 1 |

Parameters: φ_max ~ LN(μ_φ, σ_φ) [scenario-level]; κ ~ LN(μ_κ, σ_κ) [scenario-level].

### 5.2 Technical Debt Slowdown Multiplier: ψ_D(D)

```
ψ_D(D) = 1 − (1 − ψ_min) × D^γ
```

| Property | Value |
|---|---|
| ψ_D(0) | 1 (no slowdown at zero debt) |
| ψ_D(1) | ψ_min (maximum slowdown) |
| Monotonicity | Strictly decreasing on [0, 1] |
| Inverse | ψ_D^{−1}(v) = ((1−v)/(1−ψ_min))^{1/γ} for v ∈ [ψ_min, 1] |

Parameters: ψ_min ~ Be(α_ψ, β_ψ) [scenario-level]; γ ~ LN(μ_γ, σ_γ) [scenario-level].

### 5.3 Review Quality Degradation: η_rev(Q_rev)

```
η_rev(Q) = max(η_min_rev, 1 − α_η_rev × max(0, Q − Q_thr_rev))
```

The function is identically 1 for Q ≤ Q_thr_rev, then decreases linearly, floored at η_min_rev.

Parameters: η_min_rev ~ Be(α_ηr, β_ηr); α_η_rev ~ Ga(α_a, β_a); Q_thr_rev ~ Poisson(λ_thr_rev) [all scenario-level].

### 5.4 Escalation Quality Degradation: η_esc(Q_esc)

```
η_esc(Q) = max(η_min_esc, 1 − α_η_esc × max(0, Q − Q_thr_esc))
```

Parameters: η_min_esc ~ Be(α_ηe, β_ηe); α_η_esc ~ Ga(α_b, β_b); Q_thr_esc ~ Poisson(λ_thr_esc) [all scenario-level].

---

## Section 6 — CI/CD Latency and Delayed Detection

### Definition 6.1 — Detection Delay

A bug surviving Stage A (CR/SA), injected in sprint t_inj, first becomes visible to test layers at the earliest sprint t' satisfying:

```
t' ≥ t_inj + ⌈τ_CI(t_inj)⌉
```

where τ_CI is evaluated at injection time. Bugs currently in the latency buffer are not detectable by UT/IT/ST until this condition is satisfied.

**Remark.** Using injection-time τ_CI is the natural model because the developer submits code to the CI pipeline at sprint t_inj; the latency they experience is the pipeline state at that moment. Using current-time τ_CI would be more conservative (since τ_CI grows) and is noted as a sensitivity variant.

### Lemma 6.1 — Latency Buffer Size

The runtime cap `τ_CI(t) ← min(τ_CI(t), T)` enforced in §4.3 and §4.10 is the authoritative bound on the latency used by the detection pipeline. The maximum buffer length needed before simulation start is therefore:

```
L_buf = ⌈min(τ_CI_analytic_max, T)⌉ + 1
```

where `τ_CI_analytic_max` is an analytical pre-simulation upper bound on the uncapped CI/CD latency:

```
τ_CI_analytic_max = τ_CI_base_max
                    × (1 + α_F_CI_max × T × r_F_max × a_3)
                    × (1 + α_test_CI_max × T × (β_UT_max × r_UT_max × a_8
                                                + β_IT_max × r_IT_max × a_9
                                                + β_ST_max × r_ST_max × a_10))
```

where subscript `_max` denotes the 99th percentile of each scenario-level distribution. This can be computed before the simulation loop to pre-allocate the buffer. Because the runtime cap clamps `τ_CI(t)` at T, the buffer needs at most T+1 slots regardless of how large `τ_CI_analytic_max` gets — `L_buf = T + 1` is always safe.

### Proposition 6.1 — Steady-State Hidden Bug Stock

Under simplifying steady-state assumptions (constant injection rate Λ_inj per sprint after Stage A, constant detection rate d_B across all Stage B layers, constant τ_CI = τ), the expected hidden bug stock in the latency buffer is:

```
E[B_buf_ss] = Λ_inj × τ
```

The expected hidden stock of bugs that have exited the buffer but survived Stage B is:

```
E[B_post_buf_ss] = Λ_inj × (1 − d_B) / d_B
```

Total expected hidden stock:

```
E[B_hid_ss] = Λ_inj × (τ + (1 − d_B) / d_B)
```

**Corollary.** Each additional sprint of CI/CD latency adds exactly Λ_inj hidden bugs in expectation. This quantifies the value of the DevOps/CI-CD Agent: a reduction Δτ_CI in latency reduces the expected hidden bug stock by Λ_inj × Δτ_CI bugs per sprint.

---

## Section 7 — Stability Conditions

This section derives **necessary conditions** (in the mean-field / deterministic-approximation sense) for the simulation to produce meaningful output. Configurations violating these conditions are infeasible: queues diverge, development halts, or the system enters a degenerate state; they are recorded and excluded from the Pareto sweep.

**Necessary, not sufficient.** These conditions are derived by replacing random variables with their expectations and analyzing the resulting difference equations for steady-state boundedness. They are therefore **necessary for stability but not strictly sufficient** — a configuration that passes all checks may still exhibit stochastic instability in individual Monte Carlo replications (queue growth driven by right-tail events, rare cascading debt spirals, etc.). Such residual stochastic instability is detected post-hoc by the convergence check in §9.5 and by per-run queue-growth diagnostics during the sweep; affected replications are flagged and excluded from the Pareto aggregation in §9.3.

The stability analysis uses **deterministic approximations**: random variables are replaced by their expected values, and steady-state values D* and doc_cov* are treated as given by Conditions 4 and derived self-consistently. Monte Carlo simulation then tests stochastic stability around these analytic bounds.

---

### Condition 1 — Feature Backlog Stability

**Theorem 7.1.** The expected feature backlog E[F_backlog(t)] is bounded in t if and only if the mean implementation rate meets or exceeds the mean arrival rate:

```
r_F × ψ_D(D*) × a_3 + E[impl_human] ≥ λ_F
```

where D* is the steady-state debt level (Condition 4) and E[impl_human] = H_code* × N / E[h_feature] is the expected human feature output per sprint.

**Proof sketch.** Model F_backlog as a reflected random walk with mean per-step increment λ_F − (r_F × ψ_D(D*) × a_3 + E[impl_human]). Stability (non-positive drift) requires the stated condition. The Foster-Lyapunov function V(F) = F satisfies E[V(F(t+1)) − V(F(t)) | F(t) > 0] < 0 iff the condition holds.

**Minimum implementation agent constraint:**

```
a_3 ≥ (λ_F − E[impl_human]) / (r_F × ψ_D(D*))
```

**Coupling note.** This condition depends on D* from Condition 4 and must be checked jointly.

---

### Condition 2 — Bug Queue Stability

**Theorem 7.2.** The known bug queue B_kno(t) = B_kno_min(t) + B_kno_maj(t) is bounded in expectation if and only if the traffic intensity ρ_bugs < 1, where:

```
ρ_bugs = Λ_net / μ_fix
```

with:

```
Λ_known  = (r_F × ψ_D(D*) × a_3 + E[impl_human]) × λ_impl_eff* × p_det_total
Λ_refact = a_5 × r_refactor × p_regress_refactor × p_det_B
μ_fix    = a_4 × (r_fix_minor × ψ_D(D*) × (1 − θ) + r_fix_major × ψ_D(D*) × θ)
```

where:
- λ_impl_eff* = λ_impl × φ_D(D*) × (1 + α_spec × ε_spec) × (2 − q_arch) is the steady-state effective injection rate
- p_det_A = 1 − (1 − d_CR_eff × (1 − p_sec)) × (1 − d_SA × p_sec) is the Stage A compound detection probability
- p_det_B = 1 − (1 − d_UT × cov_UT*) × (1 − d_IT × cov_IT*) × (1 − d_ST × cov_ST*) is the Stage B compound detection probability  
- p_det_total = 1 − (1 − p_det_A) × (1 − p_det_B) is the overall detection probability across both stages

**Regression feedback.** Each bug fix introduces a new regression with probability p_regress; that regression eventually reaches B_known with probability p_det_total_B = p_det_A + (1 − p_det_A) × p_det_B (it re-enters the pipeline at Stage A equivalent since regressions go directly to the buffer, so effectively at Stage B-only detection rate p_det_B). The net arrival rate accounting for this feedback is:

```
Λ_net = (Λ_known + Λ_refact) / (1 − p_regress × p_det_B)
```

**Infeasibility condition.** If p_regress × p_det_B ≥ 1, a self-sustaining bug population exists: every fix generates at least one detectable bug on average, making the queue unbounded regardless of Bug Fixing Agent capacity. Configurations satisfying this condition are infeasible.

**Minimum Bug Fixing Agent constraint:**

```
a_4 ≥ Λ_net / (r_fix_minor × ψ_D(D*) × (1 − θ) + r_fix_major × ψ_D(D*) × θ)
```

---

### Condition 3 — Development Does Not Come to a Standstill

**Theorem 7.3a (Monitoring feasibility).** Human Developers retain positive capacity after monitoring (development can proceed at all) if and only if:

```
H_capacity − c_comm × (N − 1) > M × H_monitoring_overhead × (1 − Γ_doc × doc_cov*)
```

**Maximum span of control (basic):**

```
M ≤ (H_capacity − c_comm × (N − 1)) / (H_monitoring_overhead × (1 − Γ_doc × doc_cov*))
```

**Theorem 7.3b (Full queue stability — primary filter).** For both queues to remain bounded and for escalations to be resolved, the combined per-developer demand must not exceed supply:

```
M × H_mon_eff* + (r_arch × a_2 / N) × H_review_time + M × ρ_esc × H_escalation_time
  < H_capacity − c_comm × (N − 1)
```

Rearranging to the tightest span-of-control bound:

```
M ≤ (H_capacity − c_comm × (N − 1) − (r_arch × a_2 / N) × H_review_time)
    / (H_mon_eff* + ρ_esc × H_escalation_time)
```

where H_mon_eff* = H_monitoring_overhead × (1 − Γ_doc × doc_cov*).

This is the **primary feasibility filter** for the configuration sweep. It must be evaluated first, before checking Conditions 1, 2, and 4.

**Minimum human developer count (fixing M):**

```
N × (H_capacity − M × H_mon_eff* − M × ρ_esc × H_escalation_time)
  > c_comm × (N − 1) + r_arch × a_2 × H_review_time
```

Solving approximately for N ≥ N_min gives a lower bound linear in a_2, M, and H_mon_eff*.

---

### Condition 4 — Technical Debt Does Not Spiral to 1

**Theorem 7.4 (Debt fixed-point existence and stability).**

In steady state (E[D(t+1) − D(t)] = 0), the debt level satisfies the balance equation:

```
δ_refactor × a_5 × r_refactor
  = δ_impl × (r_F × ψ_D(D*) × a_3 + E[impl_human])
  + δ_fix × a_4 × (r_fix_minor × ψ_D(D*) × (1 − θ) + r_fix_major × ψ_D(D*) × θ)
```

Since ψ_D(D) is strictly decreasing, the right-hand side is a decreasing function of D*, while the left-hand side is constant. When the right-hand side exceeds the left-hand side at D=0 (i.e., debt generation exceeds refactoring capacity at zero debt), D* > 0 and the system accumulates debt. Instability (D* → 1) occurs when the right-hand side also exceeds the left-hand side at D=1 (see the stability condition below).

**Debt accumulation condition (D* > 0):**

```
δ_refactor × a_5 × r_refactor
  < δ_impl × (r_F × a_3 + E[impl_human])
  + δ_fix × a_4 × (r_fix_minor × (1 − θ) + r_fix_major × θ)
```

When the above holds, debt will accumulate toward a positive steady-state D*. Whether D* stays below 1 depends on the stability condition below.

**Stability condition (debt remains bounded below 1):** Refactoring capacity must exceed the debt accumulation rate even at maximum debt slowdown (D=1, ψ_D = ψ_min). Note that E[impl_human] is NOT multiplied by ψ_min because human coding capacity is unaffected by technical debt in this model:

```
δ_refactor × a_5 × r_refactor
  > δ_impl × (r_F × ψ_min × a_3 + E[impl_human])
  + δ_fix × a_4 × (r_fix_minor × ψ_min × (1 − θ) + r_fix_major × ψ_min × θ)
```

**Minimum Refactoring Agent constraint:**

```
a_5 ≥ ψ_min × [δ_impl × (r_F × a_3 + E[impl_human])
               + δ_fix × a_4 × (r_fix_minor × (1 − θ) + r_fix_major × θ)]
      / (δ_refactor × r_refactor)
```

**Steady-state debt.** When the stability condition holds, D* is the unique solution to the balance equation and lies in (0, 1). The analytical expression is obtained by solving the balance equation for D* given the form of ψ_D:

```
ψ_D(D*) = (δ_refactor × a_5 × r_refactor − δ_impl × E[impl_human])
           / (δ_impl × r_F × a_3 + δ_fix × a_4 × r_fix_mean)
```

where r_fix_mean = r_fix_minor × (1−θ) + r_fix_major × θ. Inverting ψ_D yields D* = ψ_D^{−1}(·).

**Linear stability.** The fixed point D* is linearly stable iff the absolute derivative of the right-hand side at D* is less than 1:

```
|δ_impl × r_F × a_3 × (1 − ψ_min) × γ × (D*)^{γ−1}| < 1
```

This is satisfied for all practical parameter ranges when δ_impl is small.

---

### Condition 5 — Architecture Review and Escalation Queues Are Stable

**Theorem 7.5 (Review queue stability).**

The architecture review queue Q_rev is stable (bounded in expectation) iff the review service rate across all N Human Developers exceeds the proposal arrival rate:

```
H_after_mon* × N / H_review_time ≥ r_arch × a_2
```

where H_after_mon* = max(0, H_capacity − c_comm × (N − 1) − M × H_mon_eff*) is the steady-state post-monitoring capacity.

**Maximum Architecture & Design Agent constraint:**

```
a_2 ≤ H_after_mon* × N / (r_arch × H_review_time)
```

**Theorem 7.6 (Escalation queue stability).**

The escalation queue Q_esc is stable iff:

```
H_remaining* × N / H_escalation_time ≥ N × M × ρ_esc
```

where H_remaining* = max(0, H_after_mon* − (r_arch × a_2 / N) × H_review_time) is the capacity remaining after reviews.

Simplifying (dividing both sides by N):

```
H_remaining* / H_escalation_time ≥ M × ρ_esc
```

This gives the **tightest upper bound on M** in the model:

```
M ≤ (H_capacity − c_comm × (N − 1) − M × H_mon_eff* − r_arch × a_2 × H_review_time / N)
    / H_escalation_time / ρ_esc
```

Solving for M:

```
M × (H_mon_eff* + ρ_esc × H_escalation_time)
  ≤ H_capacity − c_comm × (N − 1) − r_arch × a_2 × H_review_time / N
```

This is identical to the compound constraint of Theorem 7.3b, confirming that escalation queue stability is the binding span-of-control constraint.

---

### Stability Summary Table

| # | Condition | Stability Criterion | Binding Constraint |
|---|---|---|---|
| 7.1 | Feature backlog bounded | r_F × ψ_D(D*) × a_3 + E[impl_human] ≥ λ_F | min(a_3) |
| 7.2 | Bug queue bounded | ρ_bugs < 1 AND p_regress × p_det_B < 1 | min(a_4); infeasibility check |
| 7.3 | No standstill | H_total_demand < H_capacity per developer | max(M) via compound constraint |
| 7.4 | Debt bounded below 1 | refactoring exceeds accumulation at D = ψ_min | min(a_5) |
| 7.5 | Q_rev stable | r_arch × a_2 < μ_review | max(a_2) given N, M |
| 7.6 | Q_esc stable | M × ρ_esc < μ_esc | tightest max(M) = Theorem 7.3b |

**Checking order in the sweep.** For each candidate (N, M, **A**, R):

0. **Hard infeasibility gate.** Check the regression-feedback infeasibility condition from Theorem 7.2: if `p_regress × p_det_B ≥ 1`, the bug queue is unbounded regardless of Bug Fixing Agent capacity — discard the configuration without evaluating any other condition. This check depends only on scenario-level parameters and steady-state coverage, so it can be evaluated before any agent-count-dependent analysis.
1. Check Theorem 7.3b (compound span-of-control constraint); discard if violated.
2. Check Theorem 7.5 (architecture review queue stability); discard if violated.
3. Compute D* from Condition 4; check if the debt-spiral condition holds; discard if so.
4. Check Theorem 7.1 (feature backlog stability) using D*.
5. Check Theorem 7.2 (bug queue traffic intensity `ρ_bugs < 1`) using D* and steady-state coverage levels.

Configurations passing Step 0 **and** all five subsequent checks are feasible and enter the Monte Carlo simulation sweep. Feasibility here is in the mean-field sense (see Section 7 preamble) — residual stochastic instability is flagged at aggregation time (§9.3, §9.5).

---

## Section 8 — Cost Model

### 8.1 Cost Components

```
C_human = (N × c_salary + N_PO × c_PO) × T / 52

C_agents = Σ_{k=1}^{13} a_k × c_token_k × T

C_bugs = Σ_{t=0}^{T−1} [C_fix_minor × fixes_min(t) + C_fix_major × fixes_maj(t)]

C_production = Σ_{releases r} [B_escaped_min(r) × C_prod_min + B_escaped_maj(r) × C_prod_maj]

C_devops = C_DevOps_setup + Σ_{t=0}^{T−1} C_DevOps_maint(t)

C_userdoc = Σ_{releases r} C_user_doc × features_in_release(r)

C_total = C_human + C_agents + C_bugs + C_production + C_devops + C_userdoc
```

where N_PO is the number of Product Owners (typically 1, fixed parameter) and T/52 converts sprint-based time to years for salary calculations (assuming 1 sprint = 1 week; adjust if sprint length differs).

### 8.2 Pareto Objectives

| Symbol | Definition | Direction |
|---|---|---|
| V | F_completed(T) / T | maximize |
| Q | B_escaped(T) / max(1, F_completed(T)) | minimize |
| C_F | C_total / max(1, F_completed(T)) | minimize |

If F_completed(T) = 0, the configuration is degenerate (no features delivered) and is excluded from Pareto analysis.

### 8.3 Pareto Dominance

Configuration *c* dominates *c'* if V_c ≥ V_c', Q_c ≤ Q_c', C_F_c ≤ C_F_c' with at least one strict inequality.

The Pareto frontier P* is the set of non-dominated configurations, computed over the Monte Carlo mean estimates (V̄, Q̄, C̄_F) from K replications.

**Statistical tie-breaking.** Because V̄, Q̄, C̄_F are Monte Carlo estimates with non-zero variance, strict dominance on point estimates can be spurious. Two configurations `c` and `c'` are treated as **statistically equivalent** if the 95% bootstrap confidence intervals from §9.3 overlap on all three objectives simultaneously. When computing the frontier, a candidate `c` is discarded only if some other configuration `c'` dominates it *and* the dominance is statistically significant (the CIs on at least one objective do not overlap in the dominating direction). Configurations that tie statistically are both retained on the frontier and reported together.

---

## Section 9 — Monte Carlo Execution Protocol

### 9.0 Reproducibility and Random Seeding

The simulation is fully reproducible from a single integer master seed `s_0`. Every random draw — scenario-parameter sampling, sprint-level Poisson/Binomial draws, bootstrap resampling in §9.3 — is obtained from a PRNG whose state is derived deterministically from `s_0` and the enclosing loop indices.

**PRNG family.** Use a counter-based PRNG with strong statistical properties and independent sub-streams, such as **PCG-64** (`numpy.random.PCG64`) or **Philox** (`numpy.random.Philox`). Avoid legacy MT19937 as the default; it does not support independent sub-streams cleanly.

**Per-replication seed derivation.** For each configuration `c = (N, M, A, R)` and replication index `k ∈ {1, …, K}`, derive a replication seed:

```
s_{c,k} = SeedSequence(s_0, spawn_key=(hash(c), k))
```

Then instantiate an independent PCG-64 generator from `s_{c,k}` for that replication. Using SeedSequence (or equivalent) rather than a raw `s_0 + k` arithmetic scheme ensures that independent configurations/replications get statistically independent streams.

**Scope rules.**
- Scenario-level parameters (§9.2 Step 1) are drawn from the replication-level generator `s_{c,k}` before the sprint loop begins. This means the same `(c, k)` always reproduces the same scenario parameters.
- Sprint-level random draws inside the sprint loop use the same generator, advanced sequentially.
- Sobol sensitivity sampling (§9.6) uses a separate, independently-seeded stream derived from `s_0` so that varying `K` in the Monte Carlo sweep does not perturb Sobol draws.

**Reporting.** Every run logs `s_0`, the PRNG family, and the library version alongside the Pareto results so any frontier point can be re-simulated exactly.

### 9.1 Configuration Space

The feasible configuration space is formally:

```
Ω = {(N, M, A, R) :
      N ∈ ℕ≥1,
      M ∈ ℕ≥1,
      A ∈ ℕ^13, Σ_k a_k = N×M, a_3 ≥ 1,
      R ∈ ℕ≥1,
      all feasibility checks in §7 pass — Step 0 hard gate plus Steps 1–5 —
      using prior means for scenario parameters}
```

Infeasible configurations are recorded with their violated condition but not simulated.

**Exhaustive enumeration is infeasible.** The number of valid agent compositions A for a team of t = N×M total agents across 13 roles (with a_3 ≥ 1) equals C(t+11, 12) — the stars-and-bars count after reserving one agent for role 3. For t = 6 this is ≈18 564; for t = 12 it is ≈2.7 M; combined with the (N, M, R) axes the full space is intractable for any practical upper bound on M. Section 9.7 describes the two stochastic search strategies used to find near-Pareto-optimal configurations.

**M is jointly determined with A.** In the search procedure, M is not chosen independently: given N and a sampled M ∈ [1, M_max], the total agent budget is total = N × M, and A is drawn as a composition of `total` into 13 non-negative integer parts with a_3 ≥ 1. The resulting TeamConfiguration(N, M, A, R) is valid by construction because sum(A) = N × M.

### 9.2 Per-Configuration Monte Carlo Loop

For each (N, M, **A**, R) ∈ Ω, run K independent replications:

**Step 1 — Scenario parameter sampling.** Draw all scenario-level parameters once and hold fixed for T sprints. The full list of scenario-level parameters includes:

λ_F, r_req, r_arch, r_F, r_fix_minor, r_fix_major, r_refactor, r_review, r_security, r_UT, r_IT, r_ST, r_doc, H_capacity, H_monitoring_overhead, λ_impl, φ_max, κ, ψ_min, γ, θ, ε_spec, α_spec, q_arch, q_arch_floor, p_regress, p_regress_refactor, d_CR, d_SA, d_UT, d_IT, d_ST, c_UT_ceil, c_IT_ceil, c_ST_ceil, τ_CI_base, α_F_CI, α_test_CI, β_UT, β_IT, β_ST, H_review_time, H_escalation_time, ρ_esc, δ_impl, δ_fix, δ_refactor, Γ_doc, ζ_doc, η_min_rev, α_η_rev, Q_thr_rev, η_min_esc, α_η_esc, Q_thr_esc, c_comm, f_code, h_feature, Q_thr_block, α_size, baseline_size, p_sec, λ_human_scale, c_token_k (k = 1…13), c_salary, c_PO, C_fix_minor, C_fix_major, C_prod_min, C_prod_maj, C_DevOps_setup, D(0), R_warmup.

**Step 2 — State initialization.** Initialize S(0) per Section 2.3. Compute L_buf = ⌈τ_CI_max⌉ + 1 and allocate the per-severity latency buffers.

**Step 3 — Sprint loop.** For t = 0 to T−1:

1. Compute human capacity allocation (Section 3): H_net, H_mon_total, H_for_rev, rev_done, η_rev, H_for_esc, esc_done, η_esc, H_code, impl_human.
2. Feature dynamics (Section 4.1): arrivals, b_F, impl_agents, impl.
3. Bug injection (Section 4.2): λ_impl_eff, bugs_total, Stage A detection (CR, SA), update B_kno, fill buf.
4. Stage B detection (Section 4.3): compute τ_CI, pop detectable cohort, apply UT/IT/ST, update B_kno and B_hid.
5. Queue dynamics (Section 4.4–4.5): update Q_rev, Q_esc, re-escalations.
6. Bug fixing (Section 4.6): fixes_min, fixes_maj, update B_kno.
7. Regression injection (Section 4.7): regressions, append to buf.
8. Technical debt update (Section 4.8): D(t+1).
9. Test suite and documentation update (Section 4.9): tests_k, doc_cov.
10. CI/CD latency update (Section 4.10): n_tests, τ_CI.
11. Release event check (Section 4.11): if (t+1) mod R = 0, accumulate B_escaped.
12. Accumulate per-sprint costs into C_bugs, C_agents, C_devops.

**Step 4 — Compute objectives.**

```
V_k  = F_completed(T) / T
Q_k  = B_escaped(T) / max(1, F_completed(T))
CF_k = C_total_k / max(1, F_completed(T))
```

**Step 5 — Store replication output.** Record (V_k, Q_k, CF_k) and optionally the full state trajectory S_k(0..T) for diagnostics.

### 9.3 Aggregation Across Replications

For each configuration:

```
V̄  = (1/K) Σ_k V_k           [sample mean]
Q̄  = (1/K) Σ_k Q_k
C̄_F = (1/K) Σ_k CF_k
CI_V, CI_Q, CI_CF              [95% bootstrap confidence intervals]
```

### 9.4 Pareto Frontier Construction

Apply non-dominated sorting to all configurations using (V̄, Q̄, C̄_F). Report:
- The Pareto frontier as a table of (N, M, **A**, R, V̄, Q̄, C̄_F, CIs)
- The hypervolume indicator relative to a reference point (V_ref = 0, Q_ref = Q_max, C_F_ref = C_F_max)
- Anchor configurations maximizing V̄, minimizing Q̄, and minimizing C̄_F individually

**Search-based Pareto analysis.** In practice Ω is explored by stochastic search (§9.7) rather than exhaustive enumeration. The Pareto frontier is therefore an approximation computed over the set of discovered, feasibility-checked configurations. Random search provides broad coverage of the composition simplex; evolutionary search concentrates evaluations near Pareto-promising regions. Running both strategies and taking the union of their frontiers improves coverage.

### 9.5 Convergence Check

Compute the fraction of Pareto frontier configurations that change when K is halved (K/2 vs. K). If this fraction exceeds 5%, double K and re-run.

### 9.6 Sensitivity Analysis Hook

After the Pareto sweep, a Sobol sensitivity analysis over scenario-level parameters identifies which parameters most influence each Pareto objective. Use the Saltelli sampling scheme with N_sobol = 2^m × (D+2) samples, where D is the number of scenario parameters and m is chosen to achieve acceptable CI width on the Sobol indices (typically m ≥ 10). Parameters with high total-order Sobol indices require empirical calibration; others are treated as nuisance parameters.

### 9.7 Configuration Search Strategies

Because exhaustive enumeration of Ω is infeasible (§9.1), near-optimal configurations are found by stochastic search. Two strategies are implemented.

#### 9.7.1 Uniform Composition Sampling

To draw A uniformly from all non-negative integer compositions of `total` into 13 parts with a_3 ≥ 1:

1. Reserve 1 agent for role 3 (Feature Implementation).
2. Draw 12 cut-points u_1, …, u_12 uniformly from {0, 1, …, total−1} with replacement.
3. Sort: u_(1) ≤ … ≤ u_(12).
4. Form boundaries: 0, u_(1), …, u_(12), total−1.
5. Role counts: a_k = boundary_k − boundary_{k−1} for k = 1…12; a_3 ← a_3 + 1.

This is the stars-and-bars cut-point method and produces a uniform distribution over all valid compositions of `total` into 13 non-negative integer parts with a_3 ≥ 1.

#### 9.7.2 Random Search

Given search parameters N_values, M_max, R_values, and a budget n_samples:

For i = 1, …, n_samples:
1. Draw N uniformly from N_values.
2. Draw M uniformly from {1, …, M_max}.
3. Compute total = N × M.
4. Draw A from the uniform composition distribution (§9.7.1) with sum total.
5. Draw R uniformly from R_values.
6. Check feasibility (§7 Steps 0–5) using prior means for scenario parameters. Record (N, M, A, R) as infeasible with failed step, or add to the feasible pool.
7. If feasible, run K replications (§9.2) and aggregate objectives (§9.3).

All evaluated configurations are collected; the Pareto frontier is computed once over the full feasible pool at the end (§9.4).

**Coverage.** Random search with n_samples evaluations covers a fraction n_samples / |Ω| of the composition space. For t = 6 and n_samples = 500 this is ~2.7%; for t = 12 it is ~0.018%. Coverage decreases rapidly with team size.

#### 9.7.3 Evolutionary Search

Evolutionary search concentrates evaluations near Pareto-promising regions by using already-evaluated configurations to guide the generation of new candidates.

**Initialization.** Draw n_initial configurations using the random search procedure (§9.7.2). Evaluate all feasible ones.

**Generations.** Repeat for g = 1, …, n_generations:

a. Compute the current Pareto frontier F_g over all evaluated feasible configurations.
b. For j = 1, …, n_offspring:
   i.  Select a parent uniformly from F_g.
   ii. Apply the mutation operator (§9.7.4) to produce a child configuration.
   iii. If the child has not been evaluated before, check feasibility and — if feasible — simulate it and add to the evaluated pool.

**Pool.** All configurations evaluated across all generations (including initialization) are retained. The final Pareto frontier is computed once over the complete pool.

**Diversity vs. exploitation.** Evolutionary search exploits promising regions but is more likely to produce near-neighbour configurations, reducing diversity relative to random search. Running both strategies and taking the union of their frontiers is recommended when budget permits.

#### 9.7.4 Mutation Operator

The mutation operator preserves (N, M, R) and changes only A, keeping sum(A) = N × M and a_3 ≥ 1:

Repeat n_shifts ∈ {1, 2, 3} times (drawn uniformly):
1. Pick a donor role: sample src uniformly from {k : a_k > 0 and (k ≠ 3 or a_k > 1)}.
2. Pick a recipient role: sample dst uniformly from {1, …, 13}.
3. a_src ← a_src − 1;  a_dst ← a_dst + 1.

If no valid donor exists (all agents are concentrated in the single required impl slot a_3 = 1), the mutation is a no-op and the parent is returned unchanged.

---

## Section 10 — Release Policy

Releases occur exactly every **R** sprints, where R is a fixed integer component of the team configuration. A release event at sprint t_rel = jR (for j = 1, 2, …) has the following effects:

1. All hidden bugs (bugs in the latency buffer and bugs surviving Stage B) that have not entered B_kno are released to production and added to B_escaped.
2. User Documentation Agent (role 13) produces documentation for the release at cost C_user_doc × features_in_release.
3. Cumulative cost metrics are updated.

The simulation does not enforce a quality gate on releases. Product Owner release decisions and acceptance criteria are modeled implicitly through the detection pipeline (features are "accepted" once implemented and tested) rather than as an explicit gate.

---

## Section 11 — Calibration Strategy

### 11.1 Primary Data Sources

| Parameter group | Recommended source |
|---|---|
| r_F, r_fix (agent rates) | SWE-Bench, HumanEval agent benchmarks |
| r_req, r_arch | LLM specification/design task benchmarks |
| r_UT, r_IT, r_ST | Test generation benchmarks (e.g., EvalPlus) |
| λ_impl, θ, p_regress | Empirical defect density databases (NASA, PROMISE, industrial studies) |
| H_capacity, H_monitoring_overhead | Time-study data from engineering teams; typical range H_capacity ∈ [20, 60] hours/sprint |
| ρ_esc | Empirical escalation rate from AI pair-programming studies |
| c_token_k | Current LLM API pricing; recalibrate each year |
| δ_impl, δ_refactor, δ_fix | Technical debt studies; SonarQube "technical debt ratio" data |

### 11.2 Calibration Method

For each parameter with distributional uncertainty:
1. Identify the canonical dataset.
2. Fit the distribution family using maximum likelihood estimation.
3. Report the fitted (α, β) or (μ, σ) hyperparameters as the simulation baseline.

### 11.3 Sobol-Guided Calibration Priority

Run the Sobol sensitivity analysis (Section 9.6) on the full parameter set before deep empirical calibration. Parameters with total-order Sobol index S_Ti < 0.01 can be fixed at their prior mean without loss of accuracy.

---

## Section 12 — Notation Summary

Full reference table of all symbols used in this document.

| Symbol | Definition | Domain | Type |
|---|---|---|---|
| N | Human Developer count | ℕ≥1 | Config |
| M | Span of control | ℕ≥1 | Config |
| A = (a_1,…,a_13) | Agent type composition | ℕ^13, Σ=NM | Config |
| R | Release cadence (sprints) | ℕ≥1 | Config |
| T | Simulation horizon (sprints) | ℕ≥1 | Config |
| K | Monte Carlo replications | ℕ≥1 | Config |
| t | Sprint index | {0,…,T−1} | Index |
| k | Replication index | {1,…,K} | Index |
| F_backlog(t) | Feature backlog | ℕ≥0 | State |
| F_completed(t) | Cumulative shipped features | ℕ≥0 | State |
| B_hid_min(t) | Hidden minor bugs | ℕ≥0 | State |
| B_hid_maj(t) | Hidden major bugs | ℕ≥0 | State |
| B_kno_min(t) | Known minor bugs | ℕ≥0 | State |
| B_kno_maj(t) | Known major bugs | ℕ≥0 | State |
| B_escaped(t) | Cumulative production bugs | ℕ≥0 | State |
| D(t) | Technical debt level | [0,1] | State |
| Q_rev(t) | Architecture review queue | ℕ≥0 | State |
| Q_esc(t) | Escalation queue | ℕ≥0 | State |
| tests_k(t) | Cumulative tests (k=UT/IT/ST) | ℕ≥0 | State |
| doc_cov(t) | Documentation coverage | [0,1] | State |
| buf_s[t_inj] | Latency buffer by injection sprint | ℕ≥0 | Buffer |
| τ_CI(t) | CI/CD pipeline latency (sprints) | ℝ>0 | Derived |
| cov_k(t) | Test coverage (k=UT/IT/ST) | [0,c_k_ceil] | Derived |
| n_tests(t) | Weighted cumulative test count | ℝ≥0 | Derived |
| H_net(t) | Net human capacity after comm | ℝ≥0 | Derived |
| H_mon_eff(t) | Monitoring overhead per agent | ℝ≥0 | Derived |
| H_for_rev(t) | Capacity used for reviews | ℝ≥0 | Derived |
| H_for_esc(t) | Capacity used for escalations | ℝ≥0 | Derived |
| H_code(t) | Capacity remaining for coding | ℝ≥0 | Derived |
| η_rev(t) | Review quality degradation | [η_min_rev,1] | Derived |
| η_esc(t) | Escalation quality degradation | [η_min_esc,1] | Derived |
| b_F(t) | Architecture blocking fraction | [0,1] | Derived |
| λ_F | Feature arrival rate | ℝ>0 | Scenario |
| r_req | Requirements refinement rate (cost-only; see A8 for r_review) | ℝ>0 | Scenario |
| r_arch | Architecture proposal rate | ℝ>0 | Scenario |
| r_F | Feature implementation rate | ℝ>0 | Scenario |
| r_fix_minor | Minor bug fix rate | ℝ>0 | Scenario |
| r_fix_major | Major bug fix rate | ℝ>0 | Scenario |
| r_refactor | Refactoring work rate | ℝ>0 | Scenario |
| r_review | Code review throughput (unused in §4; retained for cost calibration, see A8) | ℝ>0 | Scenario |
| r_security | Security audit throughput (cost-only parameter) | ℝ>0 | Scenario |
| r_UT | Unit test development rate | ℝ>0 | Scenario |
| r_IT | Integration test rate | ℝ>0 | Scenario |
| r_ST | System test rate | ℝ>0 | Scenario |
| r_doc | Documentation rate | ℝ>0 | Scenario |
| H_capacity | Human attention budget | ℝ>0 | Scenario |
| H_monitoring_overhead | Per-agent monitoring cost | ℝ≥0 | Scenario |
| H_review_time | Per-review time (hours) | ℝ>0 | Scenario |
| H_escalation_time | Per-escalation time (hours) | ℝ>0 | Scenario |
| h_feature | Human hours per feature | ℝ>0 | Scenario |
| c_comm | Per-peer comm overhead | ℝ≥0 | Scenario |
| f_code | Human coding fraction | [0,1] | Scenario |
| λ_human_scale | Human bug injection scale vs. agent | (0,1] | Scenario |
| λ_impl | Base bug injection rate | ℝ≥0 | Scenario |
| φ_max | Max debt injection amplifier | ℝ>1 | Scenario |
| κ | Debt amplifier shape exponent | ℝ>0 | Scenario |
| ψ_min | Min debt slowdown multiplier | (0,1) | Scenario |
| γ | Debt slowdown shape exponent | ℝ>0 | Scenario |
| φ_D(D) | Debt bug amplifier | [1,φ_max] | Function |
| ψ_D(D) | Debt slowdown multiplier | [ψ_min,1] | Function |
| θ | Major bug fraction | [0,1] | Scenario |
| p_sec | Security-class bug fraction | [0,1] | Scenario |
| ε_spec | Specification error rate | [0,1] | Scenario |
| α_spec | Spec error amplifier | ℝ>0 | Scenario |
| q_arch | Architecture quality score | [0,1] | Scenario |
| q_arch_floor | Rushed review architecture quality | [0,1] | Scenario |
| p_regress | Fix regression probability | [0,1] | Scenario |
| p_regress_refactor | Refactoring regression probability | [0,1] | Scenario |
| d_CR | Code review detection rate | [0,1] | Scenario |
| d_SA | Security audit detection rate | [0,1] | Scenario |
| d_UT | Unit test max detection rate | [0,1] | Scenario |
| d_IT | Integration test max detection rate | [0,1] | Scenario |
| d_ST | System test max detection rate | [0,1] | Scenario |
| c_UT_ceil | Unit test coverage ceiling | (0,1] | Scenario |
| c_IT_ceil | Integration test coverage ceiling | (0,1] | Scenario |
| c_ST_ceil | System test coverage ceiling | (0,1] | Scenario |
| τ_CI_base | Baseline CI/CD latency | ℝ>0 | Scenario |
| α_F_CI | Latency growth per feature | ℝ≥0 | Scenario |
| α_test_CI | Latency growth per test | ℝ≥0 | Scenario |
| β_UT | Unit test execution time weight | ℝ>0 | Scenario |
| β_IT | Integration test execution time weight | ℝ>0 | Scenario |
| β_ST | System test execution time weight | ℝ>0 | Scenario |
| ρ_esc | Per-agent escalation rate | ℝ≥0 | Scenario |
| δ_impl | Debt increment per feature | ℝ>0 | Scenario |
| δ_fix | Debt increment per fix | ℝ>0 | Scenario |
| δ_refactor | Debt decrement per refactoring unit | ℝ>0 | Scenario |
| Γ_doc | Documentation overhead reduction | [0,1] | Scenario |
| ζ_doc | Coverage erosion rate | [0,1] | Scenario |
| η_min_rev | Minimum review quality | (0,1) | Scenario |
| α_η_rev | Review degradation slope | ℝ>0 | Scenario |
| Q_thr_rev | Review queue degradation threshold | ℕ≥0 | Scenario |
| η_min_esc | Minimum escalation quality | (0,1) | Scenario |
| α_η_esc | Escalation degradation slope | ℝ>0 | Scenario |
| Q_thr_esc | Escalation queue degradation threshold | ℕ≥0 | Scenario |
| Q_thr_block | Blocking threshold for architecture reviews | ℕ≥1 | Scenario |
| α_size | Code review size degradation rate | ℝ≥0 | Scenario |
| baseline_size | Baseline PR size (features/sprint) | ℝ>0 | Scenario |
| D(0) | Initial technical debt | [0,1] | Scenario |
| c_token_k | Token cost per agent of type k | ℝ≥0 | Scenario |
| c_salary | Human Developer annual cost | ℝ>0 | Scenario |
| c_PO | Product Owner annual cost | ℝ>0 | Scenario |
| C_fix_minor | Minor bug fix cost | ℝ≥0 | Scenario |
| C_fix_major | Major bug fix cost | ℝ≥0 | Scenario |
| C_prod_min | Production impact per minor escaped bug | ℝ≥0 | Scenario |
| C_prod_maj | Production impact per major escaped bug | ℝ≥0 | Scenario |
| C_DevOps_setup | One-time DevOps setup cost | ℝ≥0 | Scenario |
| C_DevOps_maint | Recurring DevOps maintenance cost | ℝ≥0 | Scenario |
| C_user_doc | User documentation cost per release | ℝ≥0 | Scenario |
| p_det_A | Stage A compound detection probability | [0,1] | Derived |
| p_det_B | Stage B compound detection probability | [0,1] | Derived |
| p_det_total | Overall detection probability | [0,1] | Derived |
| D* | Steady-state technical debt | [0,1] | Derived |
| ρ_bugs | Bug queue traffic intensity | ℝ≥0 | Derived |
| V | Feature throughput (features/sprint) | ℝ≥0 | Objective |
| Q | Escaped bug rate (bugs/feature) | ℝ≥0 | Objective |
| C_F | Cost per feature | ℝ>0 | Objective |
| P* | Pareto frontier | subset ℝ^3 | Output |
| M_optimal | Optimal span of control | ℕ≥1 | Output |
| M_max | Maximum span of control (search bound) | ℕ≥1 | Search param |
| n_samples | Random search evaluation budget | ℕ≥1 | Search param |
| n_initial | Evolutionary search initialisation size | ℕ≥1 | Search param |
| n_generations | Number of evolutionary generations | ℕ≥0 | Search param |
| n_offspring | New candidates per generation | ℕ≥1 | Search param |
| n_shifts | Agent moves per mutation | {1,2,3} | Search param |
