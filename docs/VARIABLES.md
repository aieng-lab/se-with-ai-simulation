# VARIABLES.md

This document defines every random variable, parameter, and derived quantity required to simulate mixed AI-agent/human-developer software teams. It is the direct precursor to Framework.md, which assembles these variables into equations of motion, queuing models, and the Pareto optimization procedure.

**Scope of this document:** variable definitions, domains, probability distribution families with justification, role applicability, and inter-variable dependencies.

**Deferred to Framework.md:** difference equations for state evolution, queuing model formalism, compound detection chain probabilities, the release decision policy, calibration strategy, and sensitivity analysis design.

---

## Notation Conventions

Each variable entry follows this schema:

| Field | Content |
|---|---|
| **Symbol** | Notation used throughout Framework.md |
| **Definition** | Precise prose definition |
| **Domain** | Mathematical domain (e.g., ℝ≥0, [0,1], ℕ) |
| **Distribution** | Recommended family for Monte Carlo sampling, with justification |
| **Applies To** | Role(s) this variable characterizes (referencing ROLES.md role numbers) |
| **Interactions** | Key variables this one drives or is driven by |

Distribution abbreviations: LN = LogNormal, Be = Beta, Ga = Gamma, N = Normal, Po = Poisson.

---

## Section 1 — Team Composition Parameters

These are the **control variables** swept across simulation runs to find Pareto-optimal configurations. They are fixed within a single run and are not sampled from probability distributions.

---

**N — Number of Human Developers**

| | |
|---|---|
| **Symbol** | N |
| **Definition** | Total number of Human Developers employed. Determines total human review capacity, the number of independent agent sub-teams, and the fixed salary cost component. |
| **Domain** | ℕ≥1 |
| **Distribution** | Sweep parameter. Enumerated across N ∈ {1, 2, …, N_max}. |
| **Applies To** | Human Developer (role 15) |
| **Interactions** | Combined with M, determines total agent pool size N×M. Linearly scales total salary cost N×c_salary and total review throughput. As N grows, per-developer communication overhead H_comm(N) = c_comm×(N−1) also grows linearly, creating a countervailing bottleneck on H_capacity alongside the throughput gain. |

---

**M — Span of Control**

| | |
|---|---|
| **Symbol** | M |
| **Definition** | Number of AI agents assigned to each Human Developer. Each Human Developer oversees exactly M agents; total agent count = N×M. The central trade-off parameter of the simulation. |
| **Domain** | ℕ≥1 |
| **Distribution** | Sweep parameter. Enumerated across M ∈ {1, 2, …, M_max}. |
| **Applies To** | Human Developer (role 15) |
| **Interactions** | As M grows, monitoring overhead M×H_monitoring_overhead grows linearly, crowding out H_capacity available for reviews and escalations. Drives the bottleneck degradation model (see Section 6). |

---

**A — Agent Type Composition Vector**

| | |
|---|---|
| **Symbol** | A = (a_1, a_2, …, a_13) |
| **Definition** | Integer vector where a_k is the count of agents of type k deployed (k indexes the 13 agent roles). Subject to Σa_k = N×M. |
| **Domain** | ℕ^13, with Σa_k = N×M and minimum staffing constraints (e.g., a_3 ≥ 1, i.e., at least one Feature Implementation Agent) |
| **Distribution** | Sweep parameter. The simulation enumerates feasible compositions subject to the total agent count constraint. |
| **Applies To** | All agent roles (roles 1–13) |
| **Interactions** | Determines which productivity and quality variables are active. Trades off speed (larger a_3) vs. quality (larger a_6, a_7, a_8, a_9, a_10) vs. support overhead (a_11, a_12, a_13). |

---

**R — Release Cadence**

| | |
|---|---|
| **Symbol** | R ∈ ℕ≥1 |
| **Definition** | Number of sprints between consecutive releases. A release event occurs exactly at sprint indices R, 2R, 3R, …, flushing all hidden bugs (latency buffer and post-Stage-B survivors) to production. |
| **Domain** | ℕ≥1 (typically 1–12 for weekly/quarterly cadences) |
| **Distribution** | Sweep parameter — not sampled; enumerated alongside (N, M, **A**). Short cadence (R=1) approximates continuous delivery; long cadence (R=12) approximates quarterly release. |
| **Applies To** | Whole team (all roles); in particular, affects User Documentation Agent (role 13) cost (`C_user_doc × features_in_release`) and the rate at which hidden bugs convert to escaped bugs. |
| **Interactions** | Shorter R reduces `B_escaped` per release (fewer bugs accumulate in the buffer) but increases release-overhead costs (e.g., DevOps and user documentation per release). Longer R allows Stage B detection more time to catch latent bugs before release, but concentrates more bugs per release when they do escape. Interacts with τ_CI: if τ_CI(t) > R, the CI pipeline cannot complete a full cycle within one release window and the release effectively ships unchecked changes. |

---

## Section 2 — State Variables

These variables represent the **dynamic system state**, updated at each discrete time step (one sprint). They are not directly sampled; their evolution is governed by difference equations defined in Framework.md.

**Notation bridge to FRAMEWORK.md.** FRAMEWORK.md writes per-severity splits with abbreviated subscripts: `B_hid_min(t)`, `B_hid_maj(t)` for `B_hidden_minor(t)`, `B_hidden_major(t)`, and `B_kno_min(t)`, `B_kno_maj(t)` for `B_known(t)` split by severity. The symbols are interchangeable; the abbreviated forms are canonical inside FRAMEWORK.md equations.

---

**F_backlog(t) — Feature Backlog**

| | |
|---|---|
| **Symbol** | F_backlog(t) |
| **Definition** | Number of features in the backlog awaiting implementation at the start of sprint t. Features arrive from the Product Owner as a stochastic process and are consumed by Feature Implementation Agents. |
| **Domain** | ℕ≥0 |
| **Distribution** | Arrivals per sprint: Poisson(λ_F). State evolves as F_backlog(t+1) = F_backlog(t) + arrivals(t) − implementations(t). |
| **Applies To** | Product Owner (role 14), Requirements Refinement Agent (role 1), Feature Implementation Agent (role 3) |
| **Interactions** | Driven by λ_F. Drained by r_F × a_3. If F_backlog = 0, Feature Implementation Agents are idle. |

---

**F_completed(t) — Cumulative Completed Features**

| | |
|---|---|
| **Symbol** | F_completed(t) |
| **Definition** | Total count of features fully implemented, reviewed, tested, and accepted as of the end of sprint t. Non-decreasing in t. The primary speed metric. |
| **Domain** | ℕ≥0, non-decreasing |
| **Distribution** | Derived — integral of effective implementation throughput over sprints, gated by queue lengths and blocking events. |
| **Applies To** | Feature Implementation Agent (role 3), Human Developer (role 15) |
| **Interactions** | Driven by r_F_eff (debt-adjusted implementation rate). Each unit increments D by δ_impl and injects bugs at rate λ_impl_eff. Used in the speed metric V and cost-per-feature metric C_F. |

---

**B_hidden_s(t) — Hidden Bug Stock by Severity**

| | |
|---|---|
| **Symbol** | B_hidden_minor(t), B_hidden_major(t) |
| **Definition** | Number of bugs of severity s ∈ {minor, major} that exist in the codebase but have not yet been detected by any review or test layer. This is the latent quality state of the system. |
| **Domain** | ℕ≥0 for each severity level |
| **Distribution** | Derived via difference equation: B_hidden_s(t+1) = B_hidden_s(t) + injections_s(t) − detections_s(t). Injections and detections are sampled each sprint using injection and detection rate variables. |
| **Applies To** | Feature Implementation Agent (role 3), Bug Fixing Agent (role 4), Refactoring Agent (role 5) |
| **Interactions** | Fed by λ_impl_eff × θ (major) and λ_impl_eff × (1−θ) (minor). Drained by detection rates d_CR, d_SA, d_UT, d_IT, d_ST. Residual at release time becomes B_escaped. |

---

**B_known(t) — Known Bug Backlog**

| | |
|---|---|
| **Symbol** | B_known(t) |
| **Definition** | Number of bugs that have been detected (by any review or test layer) and are queued for resolution but not yet fixed. |
| **Domain** | ℕ≥0 |
| **Distribution** | Derived: B_known(t+1) = B_known(t) + detections(t) − fixes(t). |
| **Applies To** | Bug Fixing Agent (role 4) |
| **Interactions** | Driven by all detection layers. Drained by r_fix × a_4. Competes with F_backlog for Bug Fixing Agent capacity when both are non-zero. High B_known is a leading indicator of quality degradation. |

---

**B_escaped(t) — Cumulative Bugs Escaped to Production**

| | |
|---|---|
| **Symbol** | B_escaped(t) |
| **Definition** | Total count of bugs released to production without detection at any pre-release stage. Non-decreasing. The primary quality metric (lower is better). |
| **Domain** | ℕ≥0, non-decreasing |
| **Distribution** | Derived — accumulates the residual B_hidden_s surviving all detection layers at each release event. |
| **Applies To** | System and Acceptance Test Development Agent (role 10), Product Owner (role 14) |
| **Interactions** | Accumulates at each release from bugs in B_hidden_s that survive the full sequential detection chain: Code Review (d_CR) → Security Audit (d_SA) → Unit Test (d_UT) → Integration Test (d_IT) → System Test (d_ST). d_ST is the final pre-release gate; any bug surviving all five layers escapes to production. Drives C_production_impact. Used in the quality metric Q. |

---

**D(t) — Technical Debt Level**

| | |
|---|---|
| **Symbol** | D(t) ∈ [0, 1] |
| **Definition** | Normalized continuous measure of codebase technical debt. D = 0 is debt-free; D = 1 is maximally degraded (implementation and fixes are extremely difficult). Captures accumulated shortcuts, missing abstractions, and structural degradation. |
| **Domain** | [0, 1] |
| **Distribution** | Derived via difference equation: D(t+1) = clip(D(t) + δ_impl×implementations(t) + δ_fix×fixes(t) − δ_refactor×refactor_work(t), 0, 1). The δ parameters are sampled from distributions defined in Section 4. |
| **Applies To** | Feature Implementation Agent (role 3), Bug Fixing Agent (role 4), Refactoring Agent (role 5) |
| **Interactions** | Multiplicatively amplifies bug injection rate via φ_D(D). Multiplicatively slows implementation and fix throughput via ψ_D(D). Reduced by Refactoring Agent activity (δ_refactor). The Refactoring Agent's economic value is measured precisely by the debt reduction it achieves relative to its cost. |

---

**Q_review(t) — Architecture Review Queue**

| | |
|---|---|
| **Symbol** | Q_review(t) |
| **Definition** | Number of architecture or design proposals submitted by Architecture and Design Proposal Agents awaiting Human Developer review and approval at the start of sprint t. |
| **Domain** | ℕ≥0 |
| **Distribution** | Derived from queue dynamics: arrivals at rate r_arch × a_2, service rate bounded by Human Developer review capacity. |
| **Applies To** | Architecture and Design Proposal Agent (role 2), Human Developer (role 15) |
| **Interactions** | When Q_review > 0 and Human Developer capacity is saturated, Feature Implementation Agents are blocked (cannot implement against unreviewed designs). Q_review > 0 also triggers review quality degradation η_review. |

---

**Q_escalation(t) — Escalation Queue**

| | |
|---|---|
| **Symbol** | Q_escalation(t) |
| **Definition** | Number of outstanding escalations submitted to Human Developers from any agent (implementation blockers, complex bugs, critical security findings) at the start of sprint t. |
| **Domain** | ℕ≥0 |
| **Distribution** | Derived from queue dynamics: arrivals at rate λ_esc = M × ρ_esc, service rate bounded by remaining H_capacity after monitoring and reviews. |
| **Applies To** | Human Developer (role 15), Feature Implementation Agent (role 3), Bug Fixing Agent (role 4), Security Audit Agent (role 7) |
| **Interactions** | Each waiting escalation blocks one agent task. Triggers escalation quality degradation η_escalation. Grows with M (more agents → more escalations). |

---

## Section 3 — Productivity Variables

These variables characterize **work output rates** per role per sprint. They are sampled at the start of each Monte Carlo replication (scenario-level parameters) or per sprint (step-level random variables), as specified.

---

**λ_F — Feature Arrival Rate**

| | |
|---|---|
| **Symbol** | λ_F |
| **Definition** | Mean number of new feature requests provided by the Product Owner per sprint. |
| **Domain** | ℝ>0 |
| **Distribution** | Ga(α_F, β_F) — λ_F is the rate parameter of the Poisson arrival process for features. Gamma is the natural conjugate prior for a Poisson rate and captures across-scenario variation in product owner throughput. |
| **Applies To** | Product Owner (role 14) |
| **Interactions** | Determines steady-state backlog size. If λ_F exceeds r_F × a_3, the backlog grows unboundedly. Feeds F_backlog. |

---

**r_req — Requirements Refinement Rate**

| | |
|---|---|
| **Symbol** | r_req |
| **Definition** | Number of feature specifications fully refined per sprint by one Requirements Refinement Agent. |
| **Domain** | ℝ>0 |
| **Distribution** | LN(μ_req, σ_req) — task completion rates for LLM agents exhibit right-skewed distributions: most tasks complete quickly, but occasionally complex requirements require much longer processing. LogNormal captures this multiplicative variability. |
| **Applies To** | Requirements Refinement Agent (role 1) |
| **Interactions** | Must track λ_F to prevent a backlog from forming upstream of Feature Implementation. Quality of output modulates ε_spec (specification error rate). |

---

**r_arch — Architecture Proposal Rate**

| | |
|---|---|
| **Symbol** | r_arch |
| **Definition** | Number of architecture or design documents produced per sprint by one Architecture and Design Proposal Agent. |
| **Domain** | ℝ>0 |
| **Distribution** | LN(μ_arch, σ_arch) — architecture task time is highly variable (simple module design vs. distributed system design). |
| **Applies To** | Architecture and Design Proposal Agent (role 2) |
| **Interactions** | Feeds Q_review. If r_arch × a_2 exceeds Human Developer review throughput, Q_review grows. Design quality parameter q_arch modulates downstream bug injection rate. |

---

**r_F — Feature Implementation Rate**

| | |
|---|---|
| **Symbol** | r_F |
| **Definition** | Number of features implemented per sprint by one Feature Implementation Agent at baseline technical debt D = 0. |
| **Domain** | ℝ>0 |
| **Distribution** | LN(μ_F, σ_F) — feature implementation time is multiplicatively composed of many sub-tasks; LogNormal follows from the multiplicative central limit theorem. High variance is characteristic of software implementation tasks. |
| **Applies To** | Feature Implementation Agent (role 3) |
| **Interactions** | Effective rate r_F_eff = r_F × ψ_D(D). Each implemented feature injects bugs at rate λ_impl_eff and increments D by δ_impl. Gated by F_backlog > 0 and Q_review = 0 (no pending unreviewed designs). |

---

**ψ_D — Technical Debt Slowdown Multiplier**

| | |
|---|---|
| **Symbol** | ψ_D(D) ∈ [ψ_min, 1] |
| **Definition** | Multiplicative factor applied to implementation and fix rates as a function of current debt level D. ψ_D(0) = 1 (no slowdown); ψ_D(1) = ψ_min < 1 (maximum slowdown). Functional form: ψ_D(D) = 1 − (1 − ψ_min) × D^γ. |
| **Domain** | [ψ_min, 1] |
| **Distribution** | ψ_min ~ Be(α_ψ, β_ψ); γ ~ LN(μ_γ, σ_γ). Beta is appropriate for a bounded minimum multiplier; LogNormal for a positive shape exponent. |
| **Applies To** | Feature Implementation Agent (role 3), Bug Fixing Agent (role 4) |
| **Interactions** | Applied to r_F and r_fix to yield effective rates. When D is high and ψ_D is low, the ROI of the Refactoring Agent increases substantially. |

---

**r_fix — Bug Fix Rate**

| | |
|---|---|
| **Symbol** | r_fix_minor, r_fix_major |
| **Definition** | Number of bugs of each severity resolved per sprint by one Bug Fixing Agent, at baseline technical debt D = 0. r_fix_minor > r_fix_major because major bugs require more diagnostic effort. |
| **Domain** | ℝ>0 for each severity |
| **Distribution** | LN(μ_fix_s, σ_fix_s) for each severity s — bug fix times are empirically log-normally distributed. |
| **Applies To** | Bug Fixing Agent (role 4) |
| **Interactions** | Effective rate r_fix_eff = r_fix × ψ_D(D). Drains B_known. Each fix has probability p_regress of injecting a new minor bug and increments D by δ_fix. |

---

**r_refactor — Refactoring Work Rate**

| | |
|---|---|
| **Symbol** | r_refactor |
| **Definition** | Units of debt reduction accomplished per sprint by one Refactoring Agent, measured in Δ_D units (i.e., the amount by which D would decrease if operating on debt-saturated code). |
| **Domain** | ℝ>0 |
| **Distribution** | LN(μ_refactor, σ_refactor) — refactoring task complexity varies widely from trivial renaming to major architectural restructuring. |
| **Applies To** | Refactoring Agent (role 5) |
| **Interactions** | Reduces D by r_refactor × δ_refactor per sprint. Has no direct effect on F_completed or B_known but improves r_F_eff and r_fix_eff via ψ_D. Has probability p_regress_refactor of introducing regression bugs. |

---

**r_review — Code Review Throughput**

| | |
|---|---|
| **Symbol** | r_review |
| **Definition** | Number of code changes (pull requests) reviewed per sprint by one Code Review Agent. |
| **Domain** | ℝ>0 |
| **Distribution** | LN(μ_review, σ_review) — review time per change is log-normally distributed (small patches reviewed quickly; large changes take much longer). |
| **Applies To** | Code Review Agent (role 6) |
| **Interactions** | If r_review × a_6 falls below combined output of implementation, fix, and refactoring agents, a review queue forms. Interacts with d_CR (effective detection rate). Detection rate d_CR decreases with change size (see Section 4). |

---

**r_security — Security Audit Throughput**

| | |
|---|---|
| **Symbol** | r_security |
| **Definition** | Number of code changes or design documents security-audited per sprint by one Security Audit Agent. |
| **Domain** | ℝ>0 |
| **Distribution** | LN(μ_security, σ_security) — security analysis time is right-skewed (most changes are quick; complex authentication or authorization flows take much longer). |
| **Applies To** | Security Audit Agent (role 7) |
| **Interactions** | Operates in parallel with Code Review Agent on complementary bug classes. Detects security-class bugs at rate d_SA. Generates escalations to Human Developer for critical findings. |

---

**r_UT — Unit Test Development Rate**

| | |
|---|---|
| **Symbol** | r_UT |
| **Definition** | Number of unit test cases written per sprint by one Unit Test Development Agent. Unit tests target individual functions and modules; their authoring rate depends on the complexity and modularity of the implemented code. Sampled independently from r_IT and r_ST. |
| **Domain** | ℝ>0 |
| **Distribution** | LN(μ_UT, σ_UT) — unit test authoring time is right-skewed: simple functions are tested quickly, but complex logic with many branches requires substantially more test cases. |
| **Applies To** | Unit Test Development Agent (role 8) |
| **Interactions** | Coverage cov_UT accumulates as cumulative (r_UT × a_8) relative to the implemented feature set. Effective unit test detection rate d_UT_eff = d_UT_max × cov_UT. r_UT contributes to the total test count n_tests(t), which drives τ_CI growth. |

---

**r_IT — Integration Test Development Rate**

| | |
|---|---|
| **Symbol** | r_IT |
| **Definition** | Number of integration test suites written per sprint by one Integration Test Development Agent. Integration tests target interactions between modules and services; they take longer per test than unit tests due to the need to coordinate multiple components. Sampled independently from r_UT and r_ST. |
| **Domain** | ℝ>0 |
| **Distribution** | LN(μ_IT, σ_IT) — integration test authoring time is more variable than unit tests, reflecting the diversity of component interaction patterns (simple API contracts vs. complex stateful workflows). |
| **Applies To** | Integration Test Development Agent (role 9) |
| **Interactions** | Coverage cov_IT accumulates as cumulative (r_IT × a_9) relative to the implemented interface set. Effective integration test detection rate d_IT_eff = d_IT_max × cov_IT. r_IT contributes to n_tests(t) and thus to τ_CI growth; integration tests typically have higher per-test execution time than unit tests. |

---

**r_ST — System and Acceptance Test Development Rate**

| | |
|---|---|
| **Symbol** | r_ST |
| **Definition** | Number of system or acceptance test scenarios written per sprint by one System and Acceptance Test Development Agent. System tests cover end-to-end user workflows and are the slowest to both author and execute. Sampled independently from r_UT and r_IT. |
| **Domain** | ℝ>0 |
| **Distribution** | LN(μ_ST, σ_ST) — system test authoring time has the highest variance of the three test layers, as scenario complexity ranges from simple happy-path flows to multi-actor adversarial scenarios. |
| **Applies To** | System and Acceptance Test Development Agent (role 10) |
| **Interactions** | Coverage cov_ST accumulates as cumulative (r_ST × a_10) relative to the implemented workflow set. Effective system test detection rate d_ST_eff = d_ST_max × cov_ST. r_ST contributes the largest per-test execution time to n_tests(t), making it the dominant driver of τ_CI growth among the three test types. |

---

**r_doc — Developer Documentation Production Rate**

| | |
|---|---|
| **Symbol** | r_doc |
| **Definition** | Number of implemented features fully documented per sprint by one Developer Documentation Agent, where "fully documented" means the corresponding code changes, interfaces, and design decisions are covered in the internal developer documentation. Determines the rate at which doc_coverage accumulates relative to the growing codebase. |
| **Domain** | ℝ>0 |
| **Distribution** | LN(μ_doc, σ_doc) — documentation authoring time is right-skewed: simple CRUD features are documented quickly while complex features with intricate state machines or cross-module interactions require substantially more effort. |
| **Applies To** | Developer Documentation Agent (role 12) |
| **Interactions** | Drives doc_coverage(t) upward at rate r_doc × a_12 / F_completed(t). When r_doc × a_12 < r_F_eff × a_3, new features are implemented faster than they are documented and doc_coverage declines. High doc_coverage amplifies the monitoring overhead reduction via Γ_doc, so r_doc indirectly reduces the span-of-control penalty on Human Developer capacity. |

---

**H_capacity — Human Developer Attention Budget**

| | |
|---|---|
| **Symbol** | H_capacity |
| **Definition** | Total productive hours per sprint available to one Human Developer for oversight, review, escalation resolution, and direct code contribution. Excludes meetings, administration, and interruptions. |
| **Domain** | ℝ>0 |
| **Distribution** | N(μ_H, σ_H) truncated to (0, 80] — human work capacity has a roughly symmetric distribution around a typical value; truncated Normal avoids physically impossible values. |
| **Applies To** | Human Developer (role 15) |
| **Interactions** | Consumed by H_comm(N) (peer communication with N−1 other developers), M×H_monitoring_overhead (agent monitoring), Q_review resolution at rate H_review_time, Q_escalation resolution at rate H_escalation_time, and direct coding at fraction f_code. When total demand exceeds H_capacity, queues form. This is the central bottleneck variable. |

---

**H_monitoring_overhead — Per-Agent Monitoring Cost**

| | |
|---|---|
| **Symbol** | H_monitoring_overhead |
| **Definition** | Hours per sprint consumed by one Human Developer monitoring the work of one assigned agent (status checks, output sampling, course corrections that do not rise to full escalation). |
| **Domain** | ℝ≥0 |
| **Distribution** | Ga(α_mon, β_mon) — monitoring time is always non-negative and right-skewed (usually small but occasionally large when an agent goes off-track). Gamma is the natural non-negative right-skewed distribution. |
| **Applies To** | Human Developer (role 15) |
| **Interactions** | Total monitoring cost per Human Developer = M × H_monitoring_overhead. As M increases, this grows linearly, squeezing the budget available for reviews and escalations — the core span-of-control degradation mechanism. Can be reduced by Γ_doc when Developer Documentation coverage is high. |

---

**h_feature — Human Hours per Feature**

| | |
|---|---|
| **Symbol** | h_feature |
| **Definition** | Mean human-developer hours required to implement one feature directly (used when a Human Developer — not an agent — writes the code for a complex-logic feature). Acts as the denominator converting residual coding capacity H_code × N into a per-sprint human feature output: impl_human(t) = ⌊H_code(t) × N / h_feature⌋. |
| **Domain** | ℝ>0 |
| **Distribution** | LN(μ_hf, σ_hf) — feature implementation time is strictly positive and right-skewed (a few features take much longer than the median). LogNormal is the canonical distribution for task completion times. |
| **Applies To** | Human Developer (role 15). |
| **Interactions** | Inversely proportional to E[impl_human] used in Stability Condition 1 (Theorem 7.1). Higher h_feature reduces the human contribution to feature throughput and therefore tightens the minimum-a_3 constraint on Feature Implementation Agents. Human-implemented features inject bugs at rate λ_impl_eff × λ_human_scale (see λ_human_scale), typically lower than agent-implemented features. |

---

## Section 4 — Quality Variables

These variables govern **bug injection, detection across pipeline layers, escape to production, and technical debt dynamics**.

---

**λ_impl — Bug Injection Rate**

| | |
|---|---|
| **Symbol** | λ_impl |
| **Definition** | Mean number of bugs introduced per feature implemented by the Feature Implementation Agent, at baseline technical debt D = 0 and without requirements or architecture quality modifiers. |
| **Domain** | ℝ≥0 |
| **Distribution** | Ga(α_λ, β_λ) — the number of defects per unit of code follows a right-skewed distribution; most features introduce few bugs but occasionally complex features introduce many. Gamma is well-supported for defect density modeling. |
| **Applies To** | Feature Implementation Agent (role 3) |
| **Interactions** | Effective injection rate: λ_impl_eff = λ_impl × φ_D(D) × (1 + α_spec×ε_spec) × (2 − q_arch). Each injected bug is classified as major with probability θ and minor with probability 1−θ. |

---

**φ_D — Technical Debt Bug Injection Amplifier**

| | |
|---|---|
| **Symbol** | φ_D(D) ∈ [1, φ_max] |
| **Definition** | Multiplicative factor by which technical debt amplifies the bug injection rate. φ_D(0) = 1 (no amplification); φ_D(1) = φ_max (maximum amplification). Functional form: φ_D(D) = 1 + (φ_max − 1) × D^κ. |
| **Domain** | [1, φ_max] |
| **Distribution** | φ_max ~ LN(μ_φ, σ_φ); κ ~ LN(μ_κ, σ_κ). LogNormal for both: φ_max is a positive multiplier; κ is a positive shape exponent. |
| **Applies To** | Feature Implementation Agent (role 3), Bug Fixing Agent (role 4) |
| **Interactions** | Driven by D(t). High φ_D combined with high D makes the quality cost of neglecting refactoring very high. Works in tandem with ψ_D — high D simultaneously slows work (ψ_D) and amplifies bugs (φ_D). |

---

**θ — Bug Severity Mixing Parameter**

| | |
|---|---|
| **Symbol** | θ ∈ [0, 1] |
| **Definition** | Fraction of all injected bugs that are classified as major severity. The complementary fraction (1−θ) are minor. Major bugs have higher fix cost, higher production impact, and different detection probabilities across layers. |
| **Domain** | [0, 1] |
| **Distribution** | Be(α_θ, β_θ) — θ is a proportion; Beta is the canonical distribution for proportions. |
| **Applies To** | Feature Implementation Agent (role 3), Bug Fixing Agent (role 4), Security Audit Agent (role 7) |
| **Interactions** | Determines the split of B_hidden into B_hidden_minor and B_hidden_major. Major bugs disproportionately drive C_production_impact and trigger Human Developer escalations (via Security Audit Agent for critical security findings). |

---

**ε_spec — Specification Error Rate**

| | |
|---|---|
| **Symbol** | ε_spec ∈ [0, 1] |
| **Definition** | Fraction of features where the Requirements Refinement Agent produces a specification with one or more misinterpretations or gaps. Specification errors propagate downstream as implementation bugs that are harder to detect (the implementation may appear correct against the flawed specification). |
| **Domain** | [0, 1] |
| **Distribution** | Be(α_ε, β_ε) — a proportion of events; Beta is appropriate. Higher values reflect poor LLM instruction-following or highly ambiguous raw requirements. |
| **Applies To** | Requirements Refinement Agent (role 1) |
| **Interactions** | Increases λ_impl_eff multiplicatively for affected features via the α_spec coefficient. Specification bugs are more likely to be classified as major than the baseline θ would suggest, since they typically reflect missing or contradictory requirements rather than coding errors. |

---

**α_spec — Specification Error Injection Amplifier**

| | |
|---|---|
| **Symbol** | α_spec |
| **Definition** | Coefficient governing how severely a specification error amplifies the bug injection rate for the affected feature. When ε_spec fraction of features have specification errors, the effective injection rate for those features is multiplied by (1 + α_spec × ε_spec). A value of α_spec = 1 means a fully erroneous specification doubles expected bugs; larger values reflect greater LLM sensitivity to ambiguous or contradictory requirements. |
| **Domain** | ℝ>0 |
| **Distribution** | LN(μ_αs, σ_αs) — a positive scaling coefficient; LogNormal ensures positivity and captures right-skew in how severely different agent or project types respond to poor specifications. |
| **Applies To** | Feature Implementation Agent (role 3), Requirements Refinement Agent (role 1) |
| **Interactions** | Appears directly in the effective injection rate formula: λ_impl_eff = λ_impl × φ_D(D) × (1 + α_spec × ε_spec) × (2 − q_arch). When ε_spec is low (good specifications), α_spec has negligible effect regardless of its value. When ε_spec is high, α_spec determines the magnitude of the quality penalty, making it a critical sensitivity parameter for the value of investing in Requirements Refinement Agent capacity. |

---

**p_sec — Security-Class Bug Fraction**

| | |
|---|---|
| **Symbol** | p_sec ∈ [0, 1] |
| **Definition** | Fraction of all injected bugs that belong to the **security class** (injection, authentication, authorization, data-exposure, and related security-sensitive categories). The complementary fraction (1 − p_sec) are functional (non-security) bugs. Security class is assigned **independently of severity** (θ): any bug can be security-minor, security-major, non-security-minor, or non-security-major. |
| **Domain** | [0, 1] |
| **Distribution** | Be(α_psec, β_psec) — a proportion of events; Beta is canonical. Mean is typically low (empirical CVE/CWE rates in modern codebases are a small single-digit percentage of total defects), right-skewed toward higher values in safety-critical or internet-facing systems. |
| **Applies To** | Security Audit Agent (role 7) as the detection target; Feature Implementation Agent (role 3), Bug Fixing Agent (role 4), Refactoring Agent (role 5) as sources of injection. |
| **Interactions** | Partitions the injected bug stream into two disjoint streams that are routed to different Stage A detectors: non-security bugs (fraction 1 − p_sec) are subject to d_CR (Code Review Agent); security bugs (fraction p_sec) are subject to d_SA (Security Audit Agent). Stage B detection (UT, IT, ST) applies to both classes identically. Higher p_sec shifts the optimal agent mix toward Security Audit Agents; lower p_sec makes Code Review Agents the dominant quality investment. |

---

**q_arch — Architecture Quality Score**

| | |
|---|---|
| **Symbol** | q_arch ∈ [0, 1] |
| **Definition** | Quality of the architecture and design proposal produced by the Architecture and Design Proposal Agent per design task, where 1.0 is a perfectly clear, correct, and complete design and 0.0 is a design that causes systematic implementation failures. |
| **Domain** | [0, 1] |
| **Distribution** | Be(α_q, β_q) with mean toward the higher end. Beta models a quality proportion. |
| **Applies To** | Architecture and Design Proposal Agent (role 2) |
| **Interactions** | Low q_arch multiplicatively increases λ_impl_eff. Human Developer review can correct low q_arch designs before they propagate — but only if Q_review queue is short and η_review is high. The interaction between q_arch, Q_review, and η_review models the quality benefit of Human Developer oversight. |

---

**q_arch_floor — Rushed-Review Architecture Quality Floor**

| | |
|---|---|
| **Symbol** | q_arch_floor ∈ [0, 1] |
| **Definition** | Worst-case architecture quality attained when the review queue is fully overloaded and review quality degradation `η_rev` collapses to `η_min_rev`. Represents the residual quality of an architecture proposal that was reviewed purely cursorily (rubber-stamped) rather than fully scrutinized. Appears in the effective architecture quality expression `q_arch_eff(t) = q_arch × η_rev(t) + (1 − η_rev(t)) × q_arch_floor`. |
| **Domain** | [0, 1] |
| **Distribution** | Be(α_qf, β_qf) with mean well below `q_arch` — rushed reviews produce meaningfully degraded quality. |
| **Applies To** | Architecture and Design Proposal Agent (role 2); Human Developer (role 15) — determines how severely a Human Developer bottleneck on reviews damages downstream quality. |
| **Interactions** | Interacts with `η_rev(t)`: when the review queue is below `Q_thr_rev`, `η_rev = 1` and `q_arch_eff = q_arch` (floor has no effect). When the queue grows, `η_rev → η_min_rev` and `q_arch_eff → q_arch_floor`. This models the quality-dilution effect of human review overload on the downstream bug injection rate via λ_impl_eff. |

---

**R_warmup — Warm-Start Horizon**

| | |
|---|---|
| **Symbol** | R_warmup ∈ ℕ≥0 |
| **Definition** | Number of pre-simulation sprints assumed to have already populated the feature backlog at `t = 0`. `R_warmup = 0` represents a cold start (empty backlog); `R_warmup > 0` represents a running system where features were already arriving before the simulation window began. Used only in the initial-conditions draw `F_backlog(0) ~ Poisson(λ_F × R_warmup)`. |
| **Domain** | ℕ≥0 |
| **Distribution** | Scenario-level; may be fixed per simulation run (cold-start vs. warm-start experiments) or treated as a discrete sweep. When sampled, a Poisson or uniform discrete distribution is adequate. |
| **Applies To** | Product Owner (role 14); system initialization. |
| **Interactions** | A nonzero `R_warmup` means Feature Implementation Agents begin with a pending workload, which can mask transient startup dynamics and gives a different reading of Theorem 7.1 stability. Combined with `D(0)` (initial debt), `R_warmup` parameterizes the two orthogonal "starting state" conditions: backlog pressure (`R_warmup`) and legacy-code quality (`D(0)`). |

---

**p_regress — Fix-Induced Regression Probability**

| | |
|---|---|
| **Symbol** | p_regress ∈ [0, 1] |
| **Definition** | Probability that fixing one bug introduces at least one new minor bug (a regression). Applies per fix operation, not per feature. |
| **Domain** | [0, 1] |
| **Distribution** | Be(α_reg, β_reg) — a per-event probability; Beta is appropriate. |
| **Applies To** | Bug Fixing Agent (role 4) |
| **Interactions** | Each resolved bug has probability p_regress of generating one new B_hidden_minor entry. High p_regress combined with large B_known can sustain a self-perpetuating bug population. Increments D by δ_fix per fix regardless of regression. |

---

**p_regress_refactor — Refactoring Regression Probability**

| | |
|---|---|
| **Symbol** | p_regress_refactor ∈ [0, 1] |
| **Definition** | Probability that one unit of refactoring work introduces a regression bug. Expected to be lower than p_regress because refactoring does not change behavior, but non-zero because structural changes can break implicit behavioral contracts. |
| **Domain** | [0, 1] |
| **Distribution** | Be(α_rr, β_rr) with mean lower than p_regress. |
| **Applies To** | Refactoring Agent (role 5) |
| **Interactions** | Generates B_hidden_minor entries. The net value of the Refactoring Agent depends on the balance between regression introduction cost and D-reduction benefit (via ψ_D and φ_D). |

---

**d_CR — Code Review Detection Rate**

| | |
|---|---|
| **Symbol** | d_CR ∈ [0, 1] |
| **Definition** | Fraction of functional bugs present in reviewed code that are detected by the Code Review Agent per review pass. Does not apply to security-class bugs (that is d_SA's domain). |
| **Domain** | [0, 1] |
| **Distribution** | Be(α_CR, β_CR) — a detection fraction; Beta is appropriate. Effective rate is modulated by change size: d_CR_eff = d_CR × max(0, 1 − α_size × log(change_size / baseline_size)). The size-degradation parameters α_size ~ Ga(…) are sampled per scenario. |
| **Applies To** | Code Review Agent (role 6) |
| **Interactions** | Bugs detected here are routed directly to B_known without reaching test layers. Detection reduces the load on Unit, Integration, and System test agents. d_CR_eff decreases as change size increases, modeling review fatigue. |

---

**d_SA — Security Audit Detection Rate**

| | |
|---|---|
| **Symbol** | d_SA ∈ [0, 1] |
| **Definition** | Fraction of security-class bugs present in audited code that are detected by the Security Audit Agent per audit pass. Applies only to security-class bugs (injection, authentication, authorization, data exposure categories). |
| **Domain** | [0, 1] |
| **Distribution** | Be(α_SA, β_SA) — proportion; Beta. Security audit detection rates for LLM-based tools have higher variance than functional review, justifying wider Beta spread. |
| **Applies To** | Security Audit Agent (role 7) |
| **Interactions** | Complementary to d_CR — operates on a disjoint bug class. Critical security findings trigger Q_escalation increments. Detected security bugs are routed to B_known with elevated major-severity probability. |

---

**d_UT, d_IT, d_ST — Test Layer Detection Rates**

| | |
|---|---|
| **Symbol** | d_UT, d_IT, d_ST ∈ [0, 1] |
| **Definition** | Fraction of bugs that survive all earlier detection layers and are detected by the Unit Test, Integration Test, and System/Acceptance Test layers, respectively, per pipeline execution. Each layer detects a complementary class: d_UT targets unit-level logic errors; d_IT targets interface and contract violations; d_ST targets end-to-end workflow failures. |
| **Domain** | [0, 1] for each |
| **Distribution** | Be(α_k, β_k) for k ∈ {UT, IT, ST} — detection fractions; Beta. Effective rate for each: d_k_eff = d_k_max × cov_k, where cov_k is the current test coverage level. |
| **Applies To** | Unit Test Development Agent (role 8), Integration Test Development Agent (role 9), System and Acceptance Test Development Agent (role 10), DevOps/CI-CD Agent (role 11) |
| **Interactions** | Applied sequentially to B_hidden_s after Code Review and Security Audit. Each layer's detected bugs are routed to B_known. Residual after all layers at release time becomes B_escaped. Detection at later layers (d_ST) is more expensive per bug than earlier layers (d_UT), modeling the increasing cost of late defect detection. |

---

**cov_UT, cov_IT, cov_ST — Test Coverage Levels**

| | |
|---|---|
| **Symbol** | cov_k ∈ [0, c_k_ceiling] for k ∈ {UT, IT, ST} |
| **Definition** | Fraction of the codebase or feature set covered by each test layer. Coverage accumulates over time as test agents write more tests, with a ceiling c_k < 1 for integration and system tests reflecting practical completeness limits. |
| **Domain** | [0, c_k_ceiling] |
| **Distribution** | Be(α_cov_k, β_cov_k) characterizes the steady-state coverage distribution across scenario replications. Coverage at time t grows with cumulative r_k output relative to the size of the implemented codebase, with diminishing returns. |
| **Applies To** | Unit Test Development Agent (role 8), Integration Test Development Agent (role 9), System and Acceptance Test Development Agent (role 10) |
| **Interactions** | cov_k scales d_k_max to yield effective detection rate d_k_eff. More test agent capacity (higher a_8, a_9, a_10) yields higher cov_k, but with diminishing returns near the ceiling. |

---

**τ_CI_base — CI/CD Baseline Pipeline Latency**

| | |
|---|---|
| **Symbol** | τ_CI_base |
| **Definition** | Minimum time (in sprint fractions) between code being committed and test results being available, corresponding to a pipeline with no features yet implemented and no tests yet written. Represents the irreducible infrastructure latency (build system startup, environment provisioning, framework overhead). |
| **Domain** | ℝ>0 |
| **Distribution** | LN(μ_CI, σ_CI) — baseline pipeline time is right-skewed: typically fast but with occasional long-tail failures (flaky infrastructure, cold caches). The DevOps Agent investment primarily reduces μ_CI and σ_CI. |
| **Applies To** | DevOps/CI-CD Agent (role 11) |
| **Interactions** | Serves as the intercept of the derived τ_CI(t) function. Combined with α_F_CI and α_test_CI to produce the sprint-by-sprint pipeline latency. |

---

**α_F_CI — Pipeline Latency Growth Rate per Feature**

| | |
|---|---|
| **Symbol** | α_F_CI |
| **Definition** | Incremental increase in CI/CD pipeline time (in sprint fractions) per additional completed feature in the codebase. Captures the cost of compiling, linking, and statically analyzing a larger codebase as features accumulate. |
| **Domain** | ℝ≥0 |
| **Distribution** | Ga(α_aF, β_aF) — always non-negative; Gamma captures the right-skewed distribution of per-feature compile/analysis overhead across project types (interpreted vs. compiled, monolith vs. microservice). |
| **Applies To** | DevOps/CI-CD Agent (role 11), Feature Implementation Agent (role 3) |
| **Interactions** | Multiplied by F_completed(t) in the τ_CI(t) formula. As more features are shipped, τ_CI grows, delaying bug detection and increasing B_hidden accumulation per sprint. |

---

**α_test_CI — Pipeline Latency Growth Rate per Test**

| | |
|---|---|
| **Symbol** | α_test_CI |
| **Definition** | Incremental increase in CI/CD pipeline time (in sprint fractions) per additional test case in the test suite, averaged across test types. Because system and acceptance tests execute more slowly than unit tests, the effective weight of an r_ST-produced test is larger than an r_UT-produced test; Framework.md defines per-type weights β_UT, β_IT, β_ST applied to n_tests(t). |
| **Domain** | ℝ≥0 |
| **Distribution** | Ga(α_aT, β_aT) — always non-negative and right-skewed (fast unit tests contribute little; slow end-to-end scenarios contribute substantially). |
| **Applies To** | DevOps/CI-CD Agent (role 11), Unit Test Development Agent (role 8), Integration Test Development Agent (role 9), System and Acceptance Test Development Agent (role 10) |
| **Interactions** | Multiplied by the weighted cumulative test count n_tests(t) = β_UT × cumulative(r_UT×a_8) + β_IT × cumulative(r_IT×a_9) + β_ST × cumulative(r_ST×a_10) in the τ_CI(t) formula, where β_UT, β_IT, β_ST are per-type execution time weights (scenario-level constants, not sampled per sprint). The ordering constraint β_UT ≤ β_IT ≤ β_ST holds because system tests execute substantially slower than unit tests. Each β_k ~ Ga(α_βk, β_βk) with the prior means calibrated to reflect typical ratios of test execution times across the three layers. Represents the principal mechanism by which growing test suites slow the feedback loop. |

---

**τ_CI(t) — CI/CD Pipeline Latency (Derived)**

| | |
|---|---|
| **Symbol** | τ_CI(t) |
| **Definition** | Time (in sprint fractions) between code being committed and test results being available at sprint t. Derived from baseline latency and the accumulated size of the codebase and test suite. Functional form: τ_CI(t) = τ_CI_base × (1 + α_F_CI × F_completed(t)) × (1 + α_test_CI × n_tests(t)), where n_tests(t) is the weighted cumulative test count across all three test layers. |
| **Domain** | ℝ>0, non-decreasing over time as F_completed and n_tests grow |
| **Distribution** | Derived — not sampled. Computed each sprint from τ_CI_base, α_F_CI, α_test_CI, F_completed(t), and n_tests(t). |
| **Applies To** | DevOps/CI-CD Agent (role 11) |
| **Interactions** | Bugs injected at time t are undetectable by test layers until t + τ_CI(t). As τ_CI(t) grows, B_hidden accumulates for longer before detection, worsening quality. This creates a long-run tension: investing in more tests improves detection rates d_k but simultaneously increases τ_CI(t), delaying the detection of new bugs. The DevOps Agent mitigates this by reducing τ_CI_base and the scaling coefficients (e.g., via parallelization and test sharding). |

---

**H_review_time — Architecture Review Time**

| | |
|---|---|
| **Symbol** | H_review_time |
| **Definition** | Hours consumed by one Human Developer to review and disposition one architecture or design proposal. |
| **Domain** | ℝ>0 |
| **Distribution** | LN(μ_Hrev, σ_Hrev) — review time is right-skewed: simple designs take minutes, complex distributed system designs may require several hours. |
| **Applies To** | Human Developer (role 15), Architecture and Design Proposal Agent (role 2) |
| **Interactions** | Consumes H_capacity. High r_arch combined with high H_review_time is the primary driver of Q_review growth. The review throughput per Human Developer = H_capacity_remaining / H_review_time. |

---

**H_escalation_time — Escalation Resolution Time**

| | |
|---|---|
| **Symbol** | H_escalation_time |
| **Definition** | Hours consumed by one Human Developer to resolve one agent escalation (diagnose the problem, provide guidance, confirm the agent can proceed). |
| **Domain** | ℝ>0 |
| **Distribution** | LN(μ_Hesc, σ_Hesc) with mean higher than H_review_time — escalations are definitionally hard problems that agents could not resolve independently. |
| **Applies To** | Human Developer (role 15) |
| **Interactions** | Consumes H_capacity. While an escalation is pending in Q_escalation, the escalating agent is blocked. High H_escalation_time combined with high λ_esc can saturate Human Developer capacity. |

---

**λ_esc — Escalation Arrival Rate**

| | |
|---|---|
| **Symbol** | λ_esc |
| **Definition** | Mean number of escalations per sprint submitted to one Human Developer by all M agents under their supervision, aggregated across all agent types. |
| **Domain** | ℝ≥0 |
| **Distribution** | The per-agent escalation rate ρ_esc ~ Ga(α_esc, β_esc); the total arrival rate λ_esc = M × ρ_esc is a Poisson process with this Gamma-distributed rate (forming a negative binomial count distribution). |
| **Applies To** | Human Developer (role 15), all agent roles |
| **Interactions** | λ_esc grows linearly with M. Combined with H_escalation_time, determines expected escalation queue service demand. When λ_esc × H_escalation_time + M × H_monitoring_overhead exceeds available H_capacity, the Human Developer is overloaded and both queues grow. |

---

**δ_impl — Debt Accumulation Rate per Feature**

| | |
|---|---|
| **Symbol** | δ_impl |
| **Definition** | Increment to technical debt D per feature implemented. Represents shortcuts, inconsistencies, and suboptimal patterns introduced under time pressure. |
| **Domain** | ℝ>0, small relative to the [0,1] scale of D |
| **Distribution** | Ga(α_δi, β_δi) — Gamma is appropriate for a positive, right-skewed quantity. Most features accumulate a small, similar debt increment, but occasionally a poorly structured feature adds substantially more. Beta scaled to a small range would be an alternative, but Gamma is more natural for an unbounded positive increment and consistent with the distribution used for δ_refactor. |
| **Applies To** | Feature Implementation Agent (role 3) |
| **Interactions** | Drives D(t) upward per sprint as features are completed. Must be offset by δ_refactor × refactor_work to prevent D from growing monotonically. |

---

**δ_fix — Debt Accumulation Rate per Bug Fix**

| | |
|---|---|
| **Symbol** | δ_fix |
| **Definition** | Increment to technical debt D per bug fixed. Fixes often introduce local patches without proper structural improvement, incrementally adding to debt. Expected smaller than δ_impl. |
| **Domain** | ℝ>0, smaller than δ_impl |
| **Distribution** | Ga(α_δf, β_δf) with mean less than that of δ_impl — Gamma is appropriate for a positive, right-skewed debt increment, consistent with δ_impl. The prior on (α_δf, β_δf) is set so that E[δ_fix] < E[δ_impl], reflecting that fixes are narrower interventions than full feature implementations. |
| **Applies To** | Bug Fixing Agent (role 4) |
| **Interactions** | Drives D(t) upward per sprint as bugs are fixed. Particularly significant when B_known is high and Bug Fixing Agents operate at maximum throughput. |

---

**δ_refactor — Debt Reduction Rate per Refactoring Unit**

| | |
|---|---|
| **Symbol** | δ_refactor |
| **Definition** | Decrement to technical debt D per sprint of one Refactoring Agent. Captures the effectiveness of refactoring work at reducing codebase complexity. |
| **Domain** | ℝ>0 |
| **Distribution** | LN(μ_δr, σ_δr) — refactoring effectiveness varies widely (trivial renaming vs. major architectural restructuring). Right-skewed distribution is appropriate. |
| **Applies To** | Refactoring Agent (role 5) |
| **Interactions** | Drives D(t) downward. If a_5 × δ_refactor > a_3 × δ_impl × r_F_eff + a_4 × δ_fix × r_fix_eff, debt is declining over time; otherwise it is growing. This inequality defines the minimum refactoring investment to maintain debt stability. |

---

**Γ_doc — Documentation Overhead Reduction Factor**

| | |
|---|---|
| **Symbol** | Γ_doc ∈ [0, 1] |
| **Definition** | Maximum fractional reduction in H_monitoring_overhead attributable to comprehensive Developer Documentation. When documentation is current and complete, Human Developers need less time to understand agent outputs and verify their work. |
| **Domain** | [0, 1] |
| **Distribution** | Be(α_Γ, β_Γ) — a proportional reduction; Beta. The achievable reduction is bounded below 1 because documentation never eliminates all monitoring overhead. |
| **Applies To** | Developer Documentation Agent (role 12), Human Developer (role 15) |
| **Interactions** | Effective monitoring overhead per agent = H_monitoring_overhead × (1 − Γ_doc × doc_coverage). High doc_coverage amplifies the value of Γ_doc; low doc_coverage renders Developer Documentation Agent investment ineffective at reducing human oversight burden. |

---

**doc_coverage(t) — Developer Documentation Coverage**

| | |
|---|---|
| **Symbol** | doc_coverage(t) ∈ [0, 1] |
| **Definition** | Fraction of the implemented codebase for which current, accurate Developer Documentation exists at sprint t. Coverage accumulates as the Developer Documentation Agent writes and updates documentation and erodes as new features are implemented without corresponding documentation updates. |
| **Domain** | [0, 1] |
| **Distribution** | Derived — doc_coverage(t) evolves as a function of Developer Documentation Agent throughput (r_doc × a_12) relative to the pace of new feature implementation (r_F_eff × a_3). The exact difference equation is defined in Framework.md. The per-feature documentation erosion rate ζ_doc ~ Be(α_ζ, β_ζ) is a scenario-level parameter capturing how quickly new implementations outpace existing documentation. |
| **Applies To** | Developer Documentation Agent (role 12), Human Developer (role 15) |
| **Interactions** | Scales the monitoring overhead reduction in the Γ_doc formula: H_monitoring_overhead_eff = H_monitoring_overhead × (1 − Γ_doc × doc_coverage). When r_doc × a_12 < r_F_eff × a_3 (new features arrive faster than they are documented), doc_coverage declines toward zero and the H_monitoring_overhead reduction benefit is lost. Sufficient Developer Documentation Agent capacity is required to sustain high doc_coverage in a rapidly shipping team. |

---

## Section 5 — Cost Variables

These variables drive the **total cost objective** in the Pareto optimization.

---

**c_token_k — Token Cost per Agent Type**

| | |
|---|---|
| **Symbol** | c_token_k |
| **Definition** | Cost per sprint (in USD) of operating one agent of type k, based on LLM API token consumption for both input and output, weighted by current API pricing. |
| **Domain** | ℝ≥0 |
| **Distribution** | LN(μ_ck, σ_ck) for each agent type k — token consumption per task is log-normally distributed. Different agent types have meaningfully different expected token costs (e.g., Code Review and Security Audit agents process large contexts; User Documentation agents produce large outputs). |
| **Applies To** | Each agent role (roles 1–13) independently |
| **Interactions** | Sum across all a_k agents of type k across all 13 types gives C_agent_total per sprint. The ratio c_salary / Σ(c_token_k) is the primary driver of the optimal human-to-agent trade-off. |

---

**c_salary — Human Developer Annual Salary**

| | |
|---|---|
| **Symbol** | c_salary |
| **Definition** | Annual fully-loaded cost (salary + benefits + equipment + overhead) of one Human Developer, in USD. |
| **Domain** | ℝ>0 |
| **Distribution** | LN(μ_sal, σ_sal) — engineering salaries are log-normally distributed across organizations and geographies. May be treated as a fixed scenario parameter for a specific organization. |
| **Applies To** | Human Developer (role 15) |
| **Interactions** | N × c_salary is the fixed annual human cost, independent of agent activity. The higher c_salary is relative to c_token_k, the more attractive higher M (fewer humans, more agents per human) becomes, despite span-of-control quality degradation. |

---

**c_PO — Product Owner Annual Salary**

| | |
|---|---|
| **Symbol** | c_PO |
| **Definition** | Annual fully-loaded cost of one Product Owner. Treated separately because Product Owners scale with product breadth, not agent count. |
| **Domain** | ℝ>0 |
| **Distribution** | LN(μ_PO, σ_PO). Typically treated as a fixed scenario parameter. |
| **Applies To** | Product Owner (role 14) |
| **Interactions** | Adds a fixed cost per Product Owner to each configuration. Does not directly affect quality or speed but constrains λ_F through Product Owner availability. |

---

**C_fix_minor, C_fix_major — Bug Fix Cost by Severity**

| | |
|---|---|
| **Symbol** | C_fix_minor, C_fix_major |
| **Definition** | Total cost (Bug Fixing Agent token costs plus Human Developer escalation time when applicable) to resolve one bug of each severity class. C_fix_major > C_fix_minor because major bugs require more complex root cause analysis and, more frequently, Human Developer involvement. |
| **Domain** | ℝ>0 for each |
| **Distribution** | LN(μ_Cf_s, σ_Cf_s) for each severity s — bug fix costs are empirically log-normally distributed. |
| **Applies To** | Bug Fixing Agent (role 4), Human Developer (role 15) |
| **Interactions** | Total bug fixing cost per sprint = C_fix_minor × fixes_minor + C_fix_major × fixes_major. Drives the economic value of earlier detection: a bug caught at unit test costs C_fix_minor; the same bug reaching production costs C_production_impact_minor >> C_fix_minor. |

---

**C_production_impact_k — Production Bug Impact Cost**

| | |
|---|---|
| **Symbol** | C_production_impact_minor, C_production_impact_major |
| **Definition** | Expected total cost of one bug of severity k that escapes to production, including customer support burden, emergency patching, SLA penalties, reputation damage, and post-production fix cost. Security-class major bugs may carry an additional severity multiplier. |
| **Domain** | ℝ≥0 for each |
| **Distribution** | LN(μ_CI_s, σ_CI_s) — production incident costs are extremely right-skewed (most minor bugs cost little; major bugs, especially security incidents, can be catastrophically expensive). The fat-tailed LogNormal captures this. |
| **Applies To** | System and Acceptance Test Development Agent (role 10), Product Owner (role 14) |
| **Interactions** | Total production impact per release = Σ_s B_escaped_s × C_production_impact_s. This is typically the dominant quality cost in the model. Heavily penalizes configurations that allow high B_escaped_major. |

---

**C_DevOps_setup — DevOps Pipeline Setup Cost**

| | |
|---|---|
| **Symbol** | C_DevOps_setup |
| **Definition** | One-time token and time cost incurred by the DevOps/CI-CD Agent to establish the initial build, test automation, and deployment pipeline. Paid once at simulation start. |
| **Domain** | ℝ≥0 |
| **Distribution** | LN(μ_DevOps, σ_DevOps) — setup complexity varies widely across project types. |
| **Applies To** | DevOps/CI-CD Agent (role 11) |
| **Interactions** | Without this investment, automated test layers have infinite detection latency (τ_CI → ∞), meaning d_UT, d_IT, and d_ST are effectively zero. Amortized over the simulation horizon T. |

---

**C_DevOps_maintenance — DevOps Pipeline Maintenance Cost**

| | |
|---|---|
| **Symbol** | C_DevOps_maintenance |
| **Definition** | Recurring cost per sprint of the DevOps/CI-CD Agent maintaining the pipeline (dependency updates, flaky test remediation, configuration drift correction). |
| **Domain** | ℝ≥0 |
| **Distribution** | Ga(α_DM, β_DM) — maintenance cost is always non-negative and right-skewed (most sprints are low-cost; occasional major pipeline failures require substantial rework). |
| **Applies To** | DevOps/CI-CD Agent (role 11) |
| **Interactions** | Recurring component of C_agent_total. Contributes to the ongoing cost of the test detection infrastructure. |

---

**C_user_doc — User Documentation Cost per Release**

| | |
|---|---|
| **Symbol** | C_user_doc |
| **Definition** | Token and time cost of the User Documentation Agent producing documentation for one release, proportional to the number of features in that release. |
| **Domain** | ℝ≥0 |
| **Distribution** | LN(μ_UD, σ_UD) scaled by features_in_release. |
| **Applies To** | User Documentation Agent (role 13) |
| **Interactions** | Contributes to the overhead cost per release. Scales with release cadence and feature volume. Does not affect quality or speed. |

---

**C_total — Total Cost**

| | |
|---|---|
| **Symbol** | C_total |
| **Definition** | Total cost over the simulation horizon T, combining all human salary costs, agent token costs, bug fix costs, and production impact costs. The primary cost objective for Pareto optimization. |
| **Domain** | ℝ≥0 |
| **Distribution** | Derived — not sampled directly. Emerges as an output distribution from Monte Carlo replications. Formula defined in Framework.md. |
| **Applies To** | All roles |
| **Interactions** | C_total = N×c_salary×(T/52) + N_PO×c_PO×(T/52) + Σ_k Σ_t a_k×c_token_k + Σ_t [C_fix_minor×fixes_minor(t) + C_fix_major×fixes_major(t)] + Σ_releases C_production_impact(r) + C_DevOps_setup + Σ_t C_DevOps_maintenance(t) + Σ_releases C_user_doc(r). |

---

## Section 6 — Human Developer Bottleneck Variables

These variables specifically govern the Human Developer as a **constrained resource** — the central bottleneck in scaling AI agent teams.

---

**f_code — Human Direct Coding Fraction**

| | |
|---|---|
| **Symbol** | f_code ∈ [0, 1] |
| **Definition** | Fraction of H_capacity that a Human Developer devotes to writing code directly (for tasks beyond agent capability) as opposed to oversight, review, and escalation resolution. |
| **Domain** | [0, 1] |
| **Distribution** | Be(α_fc, β_fc) — a proportion of time. Expected to decrease as M increases (more agents means fewer tasks requiring human code contribution, but more oversight). |
| **Applies To** | Human Developer (role 15) |
| **Interactions** | H_capacity_code = H_capacity × f_code. Code produced directly by Human Developers has a lower bug injection rate than agent-produced code, so the fraction f_code partially offsets the quality cost of high M by providing higher-quality output for complex tasks. |

---

**η_review — Review Quality Degradation**

| | |
|---|---|
| **Symbol** | η_review(Q_review) ∈ [η_min_rev, 1] |
| **Definition** | Factor by which Human Developer review quality degrades when the architecture review queue is long. When Q_review is large, reviewers spend less time per proposal; the probability of approving a low-quality design without catching its flaws increases. η_review = 1 means full review quality; lower values mean degraded quality. Functional form: η_review = max(η_min_rev, 1 − α_η_rev × max(0, Q_review − Q_threshold_rev)). |
| **Domain** | [η_min_rev, 1] |
| **Distribution** | η_min_rev ~ Be(α_ηr, β_ηr); α_η_rev ~ Ga(α_a, β_a); Q_threshold_rev ~ Po(λ_thresh). |
| **Applies To** | Human Developer (role 15) |
| **Interactions** | When η_review < 1, proposals with low q_arch are approved without correction, and their full negative effect on λ_impl_eff propagates. This is the mechanism by which high M degrades downstream code quality even before direct throughput bottleneck effects appear. |

---

**η_escalation — Escalation Resolution Quality Degradation**

| | |
|---|---|
| **Symbol** | η_escalation(Q_escalation) ∈ [η_min_esc, 1] |
| **Definition** | Factor by which escalation resolution quality degrades when the escalation queue is large. A rushed resolution may be incorrect or incomplete, causing the agent to remain blocked or to proceed with faulty guidance. Functional form: η_escalation = max(η_min_esc, 1 − α_η_esc × max(0, Q_escalation − Q_threshold_esc)). |
| **Domain** | [η_min_esc, 1] |
| **Distribution** | η_min_esc ~ Be(α_ηe, β_ηe); α_η_esc ~ Ga(α_b, β_b); Q_threshold_esc ~ Po(λ_thresh_esc). |
| **Applies To** | Human Developer (role 15) |
| **Interactions** | When η_escalation < 1, some escalation resolutions are incorrect, causing re-escalation (increasing Q_escalation further) or incorrect agent behavior (increasing λ_impl_eff for the affected task). Models the compounding failure mode of an overloaded Human Developer. |

---

**H_comm — Human Developer Communication Overhead**

| | |
|---|---|
| **Symbol** | H_comm(N) |
| **Definition** | Hours per sprint consumed by one Human Developer in coordination and communication with other Human Developers — team meetings, design discussions, code handoffs, and conflict resolution. Communication overhead grows with team size because each additional developer introduces another communication channel. Functional form: H_comm(N) = c_comm × (N − 1), where c_comm is the mean hours per sprint per communication partner and the (N − 1) factor counts the number of distinct peer relationships each developer maintains. |
| **Domain** | ℝ≥0 |
| **Distribution** | c_comm ~ Ga(α_comm, β_comm) — per-partner communication time is always non-negative and right-skewed (brief daily check-ins most sprints, but occasionally lengthy design reviews or conflict resolution sessions). N is a sweep parameter, so H_comm(N) is a deterministic linear function of the sampled c_comm. |
| **Applies To** | Human Developer (role 15) |
| **Interactions** | Consumes H_capacity alongside M×H_monitoring_overhead, Q_review service, and Q_escalation service. As N grows, H_comm(N) = c_comm×(N−1) grows linearly, creating a compounding overhead that counteracts the throughput benefit of hiring additional Human Developers (Brooks' Law). When N is large and c_comm is high, H_comm(N) can become the dominant H_capacity consumer, rendering further human hires net-negative for review and escalation throughput. This creates an upper bound on the optimal N within the Pareto sweep. |

---

## Section 7 — Derived Metrics and Optimization Targets

These are **computed outputs** of the simulation, not sampled variables. The Pareto frontier is constructed over these three quantities.

---

**V — Feature Throughput (Speed)**

| | |
|---|---|
| **Symbol** | V = F_completed(T) / T |
| **Definition** | Average number of features shipped per sprint over the simulation horizon T. The speed axis of the Pareto frontier. Higher V is preferred. |
| **Domain** | ℝ≥0 |
| **Notes** | Primarily driven by r_F_eff × a_3, subject to F_backlog availability, Human Developer review bottleneck, and D(t) evolution. Represents the aggregate productivity of the team configuration. |

---

**Q — Escaped Bug Rate (Quality)**

| | |
|---|---|
| **Symbol** | Q = B_escaped(T) / F_completed(T) |
| **Definition** | Bugs escaped to production per completed feature over the simulation horizon. The quality axis. Lower Q is preferred. |
| **Domain** | ℝ≥0 |
| **Notes** | Determined by the compound survival probability across the detection chain (Code Review → Security Audit → Unit Test → Integration Test → System Test) and the effective injection rate λ_impl_eff. Framework.md derives the formal expression. |

---

**C_F — Cost per Feature**

| | |
|---|---|
| **Symbol** | C_F = C_total / F_completed(T) |
| **Definition** | Total cost per delivered feature, including all human, agent, fix, and production impact costs, normalized by productivity. The cost axis. Lower C_F is preferred. |
| **Domain** | ℝ>0 |
| **Notes** | Normalizing by F_completed makes this a unit-cost metric, enabling fair comparison between configurations with very different throughput. An expensive team that ships many features may have a lower C_F than a cheap team that ships few. |

---

**P* — Pareto Frontier**

| | |
|---|---|
| **Symbol** | P*(V, Q, C_F) |
| **Definition** | The set of team configurations (N, M, A) for which no other feasible configuration strictly dominates on all three metrics simultaneously. The Pareto frontier is the primary output of the simulation — it defines the efficient frontier of team design choices. |
| **Domain** | Subset of ℝ^3 |
| **Notes** | Computed from Monte Carlo mean estimates of (V, Q, C_F) for each team configuration, with uncertainty intervals from simulation replications. Different stakeholder preferences (e.g., cost-constrained vs. quality-first) select different points on the frontier. |

---

**M_optimal — Optimal Span of Control**

| | |
|---|---|
| **Symbol** | M_optimal |
| **Definition** | The span of control value M that maximizes a specified utility function (e.g., features delivered per dollar of C_total) for a given agent type mix A and Human Developer count N. This is a simulation **output**, not a parameter — it emerges from the Pareto analysis. |
| **Domain** | ℕ≥1 |
| **Notes** | Derived from the simulation results across the (N, M, A) sweep. The Monte Carlo distribution of M_optimal across scenario replications characterizes uncertainty in the optimal span. M_optimal depends jointly on the ratio c_salary / Σc_token_k, the escalation rate λ_esc, the monitoring overhead H_monitoring_overhead, the communication overhead H_comm(N), the review time H_review_time, and the quality degradation functions η_review and η_escalation. M_optimal is the central finding of the simulation. |

---

## Scope Boundary: What Belongs in Framework.md

The following are deliberately deferred and must not appear in this document:

1. **Difference equations** for D(t), B_hidden_s(t), B_known(t): the precise mathematical recurrences that assemble the δ and rate variables into state evolution equations.

2. **Queuing model formalism**: the M/G/1 or M/G/N queuing analysis for Q_review and Q_escalation, including steady-state queue length and wait time distributions as functions of H_capacity, H_review_time, r_arch, λ_esc.

3. **Detection chain compound probability**: the joint probability that a bug survives all detection layers and escapes to production, derived from d_CR, d_SA, d_UT, d_IT, d_ST.

4. **Release policy**: the decision rule (B_known threshold, minimum F_completed, system test pass rate) governing when the Product Owner approves a release.

5. **Calibration strategy**: how to fit distribution parameters (μ, σ, α, β) from empirical data on LLM agent performance and human developer productivity.

6. **Sensitivity analysis specification**: which variables to subject to global sensitivity analysis (e.g., Sobol indices) and the experimental design for parameter sweeps.

---

## Summary Table

| Symbol | Name | Section | Distribution |
|---|---|---|---|
| N | Human Developer count | 1 | Sweep |
| M | Span of control | 1 | Sweep |
| A | Agent type composition | 1 | Sweep |
| R | Release cadence (sprints) | 1 | Sweep |
| F_backlog(t) | Feature backlog | 2 | Poisson arrivals |
| F_completed(t) | Completed features | 2 | Derived |
| B_hidden_s(t) | Hidden bug stock (per severity) | 2 | Derived (difference eq.) |
| B_known(t) | Known bug backlog | 2 | Derived |
| B_escaped(t) | Production bugs | 2 | Derived |
| D(t) | Technical debt | 2 | Derived (difference eq.) |
| Q_review(t) | Architecture review queue | 2 | Derived (queue) |
| Q_escalation(t) | Escalation queue | 2 | Derived (queue) |
| λ_F | Feature arrival rate | 3 | Gamma |
| r_req | Requirements refinement rate | 3 | LogNormal |
| r_arch | Architecture proposal rate | 3 | LogNormal |
| r_F | Feature implementation rate | 3 | LogNormal |
| ψ_D | Debt slowdown multiplier | 3 | Beta (ψ_min), LogNormal (γ) |
| r_fix | Bug fix rate (per severity) | 3 | LogNormal |
| r_refactor | Refactoring work rate | 3 | LogNormal |
| r_review | Code review throughput | 3 | LogNormal |
| r_security | Security audit throughput | 3 | LogNormal |
| r_UT | Unit test development rate | 3 | LogNormal |
| r_IT | Integration test development rate | 3 | LogNormal |
| r_ST | System/acceptance test development rate | 3 | LogNormal |
| r_doc | Developer documentation production rate | 3 | LogNormal |
| H_capacity | Human attention budget | 3 | Normal (truncated) |
| H_monitoring_overhead | Per-agent monitoring cost | 3 | Gamma |
| h_feature | Human hours per feature | 3 | LogNormal |
| λ_impl | Bug injection rate | 4 | Gamma |
| φ_D | Debt injection amplifier | 4 | LogNormal (φ_max, κ) |
| θ | Bug severity mixing fraction | 4 | Beta |
| ε_spec | Specification error rate | 4 | Beta |
| α_spec | Specification error injection amplifier | 4 | LogNormal |
| p_sec | Security-class bug fraction | 4 | Beta |
| q_arch | Architecture quality score | 4 | Beta |
| q_arch_floor | Rushed-review architecture quality floor | 4 | Beta |
| R_warmup | Warm-start horizon (sprints) | 4 | Discrete |
| p_regress | Fix regression probability | 4 | Beta |
| p_regress_refactor | Refactoring regression probability | 4 | Beta |
| d_CR | Code review detection rate | 4 | Beta (+ size modifier) |
| d_SA | Security audit detection rate | 4 | Beta |
| d_UT, d_IT, d_ST | Test layer detection rates | 4 | Beta (× coverage) |
| cov_UT, cov_IT, cov_ST | Test coverage levels | 4 | Beta |
| τ_CI_base | CI/CD baseline pipeline latency | 4 | LogNormal |
| α_F_CI | Pipeline latency growth per feature | 4 | Gamma |
| α_test_CI | Pipeline latency growth per test (weighted) | 4 | Gamma |
| β_UT, β_IT, β_ST | Per-type test execution time weights | 4 | Gamma |
| τ_CI(t) | CI/CD pipeline latency | 4 | Derived |
| H_review_time | Architecture review time | 4 | LogNormal |
| H_escalation_time | Escalation resolution time | 4 | LogNormal |
| λ_esc | Escalation arrival rate | 4 | Poisson(M × ρ_esc), ρ_esc ~ Gamma |
| δ_impl | Debt per feature | 4 | Gamma |
| δ_fix | Debt per fix | 4 | Gamma |
| δ_refactor | Debt reduction per refactoring unit | 4 | LogNormal |
| Γ_doc | Documentation overhead reduction | 4 | Beta |
| doc_coverage(t) | Developer documentation coverage | 4 | Derived |
| c_token_k | Token cost per agent type | 5 | LogNormal |
| c_salary | Human Developer salary | 5 | LogNormal |
| c_PO | Product Owner salary | 5 | LogNormal |
| C_fix_minor, C_fix_major | Bug fix cost (per severity) | 5 | LogNormal |
| C_production_impact_k | Production impact cost (per severity) | 5 | LogNormal |
| C_DevOps_setup | DevOps setup cost | 5 | LogNormal |
| C_DevOps_maintenance | DevOps maintenance cost | 5 | Gamma |
| C_user_doc | User documentation cost | 5 | LogNormal |
| C_total | Total cost | 5 | Derived |
| f_code | Human coding fraction | 6 | Beta |
| η_review | Review quality degradation | 6 | Deterministic fn + Beta/Gamma params |
| η_escalation | Escalation quality degradation | 6 | Deterministic fn + Beta/Gamma params |
| H_comm(N) | Human developer communication overhead | 6 | Gamma (c_comm) |
| V | Feature throughput | 7 | Derived |
| Q | Escaped bug rate | 7 | Derived |
| C_F | Cost per feature | 7 | Derived |
| P* | Pareto frontier | 7 | Derived |
| M_optimal | Optimal span of control | 7 | Derived (simulation output) |
