"""Sprint transition equations.

Within-sprint ordering:
  Step 1  Human capacity allocation
  Step 2  Debt amplifier: D_arch, D_eff, delta_D
  Step 3  Feature arrival and implementation (priority drain Q_arch_impl, then F_backlog)
  Step 4  Test suite update
  Step 5  Bug injection + detection (Test → CR → SA; all synchronous)
  Step 6  Architecture queue dynamics (Q_rev, Q_arch_impl, Q_arch)
  Step 7  Bug fixing
  Step 8  Escalation queue dynamics
  Step 9  General maintenance (technical debt + documentation update)
  Step 10 Release event

Bug fixing precedes the escalation queue because the escalation arrival rate is
driven by the agent work completed this sprint, which includes the agent-phase
bug fixes (paper §3.6, Eq. qesc).

Detection is unified: Code Review, Security Audit, and the Test suite all act
within the same sprint as injection. There is no CI/CD latency buffer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .config import AGENT_ROLE_COUNT, TeamConfiguration, net_capacity, supervised_agent_count
from .parameters import ScenarioParameters
from .state import SystemState

if TYPE_CHECKING:
    from numpy.random import Generator


#: Token cost of implementing one approved architecture proposal, in units of one
#: feature implementation (paper §3.7, Table tab:work_units).  Implementing a proposal
#: needs the same code-base-wide analysis that makes producing one five times as
#: expensive as a feature (c_token,2 = 5 * c_token,1 across every tier of §4.1), so it
#: is billed at five features.  It still consumes only one unit of the implementation
#: agent's *capacity* in Eq. (arch_impl) — this factor is a cost weight, not a rate.
ARCH_IMPL_COST_UNITS = 5.0


@dataclass(frozen=True, slots=True)
class SprintActions:
    """Work completed in one sprint: the billable agent vector plus who-did-what counts.

    `units` holds the work units of each agent role, indexed k=1..8.  Each entry is
    denominated in the same unit as that role's throughput rate r_k
    (features for k=1, change requests for k=2, bugs for k=3, debt for k=4, deployed
    instances for the flat-rate services k=5,6, tests for k=7, documents for k=8), so
    the agent cost of paper Eq. (cost_agents) is the plain inner product with
    `c_token`.  The state vector of §3.4 stays 21-dimensional: these counts are
    returned rather than stored.

    The four scalars are *contribution accounting*, not billing — nothing reads them
    for cost, and `units` stays the sole billable vector.  They exist because the
    human/agent split of the work is otherwise unrecoverable downstream: `units[0]`
    conflates agent features with implemented architecture proposals, and human work
    appears in no entry at all (human cost is the headcount term of Eq. (cost_human),
    independent of what the humans produced).  Paper symbols, all §3.6:

        impl_human   impl_human'(t), Eq. (impl_human_capped) — features *delivered* by
                     humans, i.e. after the demand cap, not the budget of Eq. (implhuman)
        impl_agents  impl_agents(t), Eq. (impl_agents)
        fixes_human  sum of the per-tier n^H_i(t) of Eq. (human_fixes)
        fixes_agent  fixes_agent(t) — deliberately duplicates `units[2]`, so that the
                     human/agent pair reads symmetrically at the call site

    Recording `impl_agents` also un-conflates `units[0]`: the implemented proposals come
    back as `(units[0] - impl_agents) / ARCH_IMPL_COST_UNITS`.
    """

    units: tuple[float, ...]
    impl_human: float = 0.0
    impl_agents: float = 0.0
    fixes_human: float = 0.0
    fixes_agent: float = 0.0

    def __post_init__(self) -> None:
        if len(self.units) != AGENT_ROLE_COUNT:
            raise ValueError(
                f"units must have length {AGENT_ROLE_COUNT}, got {len(self.units)}"
            )


def _volume_factor(impl: int, a: int, alpha_size: float) -> float:
    """Log-linear detection degradation in the change volume per detection agent.

    The `a` deployed agents of a background detection service share the sprint's
    change volume, so each inspects impl/a features.  Returns 0.0 when the
    service is not deployed, which gates the corresponding detection stage.
    The reference volume is a single feature, so the result is bounded in [0, 1]
    and the product with a detection probability needs no outer clamp.
    """
    if a <= 0:
        return 0.0
    return max(0.0, 1.0 - alpha_size * math.log(impl / a + 1.0))


def step_sprint(
    t: int,
    state: SystemState,
    config: TeamConfiguration,
    params: ScenarioParameters,
    rng: "Generator",
) -> SprintActions:
    """Advance `state` from sprint t-1 to sprint t (in place).

    Implements the ten-step transition in canonical order and returns the work each
    agent role actually completed, which is what Eq. (cost_agents) bills.
    """

    # ------------------------------------------------------------------
    # Feature-freeze flag and active agent composition
    # ------------------------------------------------------------------
    is_frozen = (t % config.R) >= (config.R - config.F_freeze)
    A_active = config.active_A(is_frozen)
    # Only the supervised roles consume human oversight; the background detection
    # services (k=5,6) are exempt (see config.SUPERVISED_ROLE_INDICES).
    n_supervised = supervised_agent_count(A_active)

    # ------------------------------------------------------------------
    # Degradation functions at current debt D(t)
    # Architecture and escalation debts are normalised by the team's maximum
    # throughput for the corresponding human activity (review / escalation
    # handling) in a single sprint: H_capacity * N / H_x_time items/sprint.
    # Both debts use the same structural formula; no threshold magic numbers.
    # ------------------------------------------------------------------
    cap_review = params.H_capacity * config.N / params.H_review_time
    cap_esc    = params.H_capacity * config.N / params.H_escalation_time
    Q_arch_total = state.Q_arch + state.Q_rev + state.Q_arch_impl
    D_arch = min(1.0, Q_arch_total / cap_review)
    D_esc  = min(1.0, state.Q_esc  / cap_esc)
    D_eff  = min(1.0, state.D_tech + D_arch + D_esc)
    delta_D = 1.0 - (1.0 - params.delta_min) * (D_eff ** params.delta_exp)

    # ------------------------------------------------------------------
    # Human capacity allocation
    # ------------------------------------------------------------------
    # H_net is the net budget of *one FTE*; every downstream term scales it by config.N.
    H_net = net_capacity(params.H_capacity, params.c_comm, config.N)
    doc_cov = min(1.0, state.N_doc / max(1.0, state.F_completed))
    H_mon_eff = params.H_monitoring_overhead * (1.0 - params.Gamma_doc * doc_cov)
    # Monitoring overhead is proportional to actual supervised agents per FTE.
    # The feasibility gate only guarantees a positive remainder at doc_cov = 1, so
    # monitoring can still exhaust the budget early in a run; the clamp at 0 means
    # such a sprint produces no productive human work (paper §3.6, Eq. hafter_mon)
    # rather than a negative budget that would run the queues backwards.
    agents_per_fte = n_supervised / config.N
    H_after_mon = max(0.0, H_net - agents_per_fte * H_mon_eff)

    # Architecture reviews served this sprint (eq. rev_done, H_after_rev).
    # The residuals are clamped at 0: mathematically they cannot be negative, since the
    # floor bounds the served item count by the budget, but `k * H_x_time / N` is not
    # exactly the quantity that was floored and can exceed the budget by an ulp — with
    # H_after_mon = 40, N = 0.75, H_review_time = 4, Q_rev = 5 and H_escalation_time = 2
    # the escalation residual lands at -1.8e-15.  That is not a rounding-sized error
    # downstream, because both consumers take its floor and so lose a whole item:
    # `impl_human = -1` (an invalid Poisson rate, which raises) and
    # `fix_human_budget = -1` (a negative drain, which puts bugs *back* into the queues).
    rev_done = min(math.floor(H_after_mon * config.N / params.H_review_time), state.Q_rev)
    H_after_rev = max(0.0, H_after_mon - rev_done * params.H_review_time / config.N)

    # Escalations served this sprint (eq. esc_done, H_after_esc)
    esc_done = min(math.floor(H_after_rev * config.N / params.H_escalation_time), state.Q_esc)
    H_after_esc = max(0.0, H_after_rev - esc_done * params.H_escalation_time / config.N)

    # f_fix fraction of H_after_esc goes to bug fixing; (1-f_fix) to feature coding.
    # During feature freeze, the coding fraction is redirected to bug fixing as well.
    H_fix_human = H_after_esc if is_frozen else params.f_fix * H_after_esc
    fix_human_budget = math.floor(H_fix_human * config.N / params.h_fix)
    impl_human = 0 if is_frozen else math.floor(
        (1.0 - params.f_fix) * H_after_esc * config.N / params.h_feature
    )

    # ------------------------------------------------------------------
    # Feature arrival and implementation
    # ------------------------------------------------------------------
    arrivals = int(rng.poisson(params.r_arrival))
    a_1 = A_active[0]   # Feature Implementation Agent (k=1, index 0)
    total_impl_cap = int(params.r_F * delta_D * a_1)
    # Approved architecture proposals are implemented first (priority over backlog).
    # arch_impl_done consumes agent capacity but does NOT count toward F_completed.
    arch_impl_done = min(state.Q_arch_impl, total_impl_cap)
    # Demand is served by the humans first; the agents fill whatever is left. Capping
    # each stream at the residual backlog (rather than capping only their sum) keeps
    # impl_agents + impl_human == impl, so the two bug-injection streams of Step 5
    # cannot inject defects for features that were never implemented.
    demand = state.F_backlog + arrivals
    impl_human = min(impl_human, demand)
    impl_agents = min(demand - impl_human, total_impl_cap - arch_impl_done)
    impl = impl_agents + impl_human
    state.F_backlog = max(0, state.F_backlog + arrivals - impl)
    state.F_completed += impl

    # ------------------------------------------------------------------
    # Test suite update (before detection so new tests count this sprint)
    # ------------------------------------------------------------------
    # Beyond c_test_ceil * F_completed no further test raises cov_test, so the
    # Test Development Agent (k=7, index 6) has no work there and is not billed for
    # it.  The max(0, .) guards an initial state loaded above the cap.
    test_cap = params.c_test_ceil * state.F_completed
    tests_written = max(0.0, min(params.r_test * A_active[6], test_cap - state.N_test))
    state.N_test += tests_written

    # ------------------------------------------------------------------
    # Bug injection + detection (Test → CR → SA; all synchronous)
    # ------------------------------------------------------------------
    r_bugs_eff = params.r_bugs * (2.0 - delta_D)

    bugs_raw = int(rng.poisson(impl_agents * r_bugs_eff))
    bugs_human_raw = int(rng.poisson(impl_human * r_bugs_eff * params.r_bugs_human))
    bugs_total = bugs_raw + bugs_human_raw

    # Three mutually exclusive types: security drawn first, then major/minor split of remainder
    theta_min = 1.0 - params.theta_maj - params.theta_sec
    bugs_sec = int(rng.binomial(bugs_total, params.theta_sec)) if bugs_total > 0 else 0
    remaining = bugs_total - bugs_sec
    theta_maj_cond = (params.theta_maj / (theta_min + params.theta_maj)
                      if (theta_min + params.theta_maj) > 0 else 0.0)
    bugs_maj = int(rng.binomial(remaining, theta_maj_cond)) if remaining > 0 else 0
    bugs_min = remaining - bugs_maj

    # Stage 1 — Test suite (acts on all three types; coverage from tests written so far)
    cov_test = min(params.c_test_ceil, state.N_test / max(1, state.F_completed))
    p_test = min(1.0, params.d_test * cov_test)
    test_det_min = int(rng.binomial(bugs_min, p_test)) if bugs_min > 0 else 0
    test_det_maj = int(rng.binomial(bugs_maj, p_test)) if bugs_maj > 0 else 0
    test_det_sec = int(rng.binomial(bugs_sec, p_test)) if bugs_sec > 0 else 0

    surv_min_t = bugs_min - test_det_min
    surv_maj_t = bugs_maj - test_det_maj
    surv_sec_t = bugs_sec - test_det_sec

    # Stage 2 — Code Review (acts on all survivors).  Zero review agents means
    # no code review takes place; more agents share the change volume and thus
    # each degrade less.
    a_CR = A_active[4]   # Code Review Agent (k=5, index 4) — background service
    d_CR_eff = params.d_CR * delta_D * _volume_factor(impl, a_CR, params.alpha_size)
    CR_det_min = int(rng.binomial(surv_min_t, d_CR_eff)) if surv_min_t > 0 else 0
    CR_det_maj = int(rng.binomial(surv_maj_t, d_CR_eff)) if surv_maj_t > 0 else 0
    CR_det_sec = int(rng.binomial(surv_sec_t, d_CR_eff)) if surv_sec_t > 0 else 0

    surv_sec_cr = surv_sec_t - CR_det_sec

    # Stage 3 — Security Audit (acts only on security bugs that survived test + CR).
    # Same volume split and debt sensitivity as code review.
    a_SA = A_active[5]   # Security Audit Agent (k=6, index 5) — background service
    d_SA_eff = params.d_SA * delta_D * _volume_factor(impl, a_SA, params.alpha_size)
    SA_det_sec = int(rng.binomial(surv_sec_cr, d_SA_eff)) if surv_sec_cr > 0 else 0

    state.B_kno_min += test_det_min + CR_det_min
    state.B_kno_maj += test_det_maj + CR_det_maj
    state.B_kno_sec += test_det_sec + CR_det_sec + SA_det_sec
    state.B_hid_min += surv_min_t - CR_det_min
    state.B_hid_maj += surv_maj_t - CR_det_maj
    state.B_hid_sec += surv_sec_cr - SA_det_sec

    # ------------------------------------------------------------------
    # Architecture queues update
    # ------------------------------------------------------------------
    # a) New change requests arrive from this sprint's implementations
    arch_chg_new = int(rng.poisson(params.f_arch_chg * impl))
    state.Q_arch += arch_chg_new
    # b) Architecture agents drain Q_arch (resolving change requests)
    #    and forward each resolved item as a design proposal into Q_rev
    arch_chg_done = min(state.Q_arch, int(params.r_arch_chg * A_active[1]))
    state.Q_arch = max(0, state.Q_arch - arch_chg_done)
    state.Q_rev = max(0, state.Q_rev + arch_chg_done - rev_done)
    # c) Approved proposals (rev_done) move into the arch-impl queue
    state.Q_arch_impl = max(0, state.Q_arch_impl + rev_done - arch_impl_done)

    # ------------------------------------------------------------------
    # Bug fixing — strict priority: sec > maj > min; rel > kno
    # Both phases spend a count budget of bugs, one unit per fix regardless of
    # severity. Humans fix first (debt-independent budget), then agents (budget
    # scaled by headcount and slowed by debt).
    # ------------------------------------------------------------------
    a_3 = A_active[2]   # Bug Fixing Agent (k=3, index 2)

    def _drain_count(queue: int, budget: int) -> int:
        """Fix as many bugs as the count budget allows; return fixes."""
        return min(queue, budget)

    # Human fixes (debt-independent, same priority order)
    hb = fix_human_budget
    h_rel_fixes_sec = _drain_count(state.B_rel_sec, hb); hb -= h_rel_fixes_sec; state.B_rel_sec -= h_rel_fixes_sec
    h_fixes_sec     = _drain_count(state.B_kno_sec, hb); hb -= h_fixes_sec;     state.B_kno_sec -= h_fixes_sec
    h_rel_fixes_maj = _drain_count(state.B_rel_maj, hb); hb -= h_rel_fixes_maj; state.B_rel_maj -= h_rel_fixes_maj
    h_fixes_maj     = _drain_count(state.B_kno_maj, hb); hb -= h_fixes_maj;     state.B_kno_maj -= h_fixes_maj
    h_rel_fixes_min = _drain_count(state.B_rel_min, hb); hb -= h_rel_fixes_min; state.B_rel_min -= h_rel_fixes_min
    h_fixes_min     = _drain_count(state.B_kno_min, hb); hb -= h_fixes_min;     state.B_kno_min -= h_fixes_min

    # Agent fixes (debt-slowed count budget, drain remaining queues)
    ab = int(a_3 * delta_D * params.r_fix)
    rel_fixes_sec = _drain_count(state.B_rel_sec, ab); ab -= rel_fixes_sec; state.B_rel_sec -= rel_fixes_sec
    fixes_sec     = _drain_count(state.B_kno_sec, ab); ab -= fixes_sec;     state.B_kno_sec -= fixes_sec
    rel_fixes_maj = _drain_count(state.B_rel_maj, ab); ab -= rel_fixes_maj; state.B_rel_maj -= rel_fixes_maj
    fixes_maj     = _drain_count(state.B_kno_maj, ab); ab -= fixes_maj;     state.B_kno_maj -= fixes_maj
    rel_fixes_min = _drain_count(state.B_rel_min, ab); ab -= rel_fixes_min; state.B_rel_min -= rel_fixes_min
    fixes_min     = _drain_count(state.B_kno_min, ab); ab -= fixes_min;     state.B_kno_min -= fixes_min

    # The agent-phase total is tracked separately from total_fixes: escalations are
    # raised by agents, so only agent work feeds the escalation rate below, whereas
    # the debt update below that is driven by all repairs regardless of who made them.
    fixes_agent = (rel_fixes_min + rel_fixes_maj + rel_fixes_sec
                 + fixes_min     + fixes_maj     + fixes_sec)
    # Sum^6_i n^H_i(t) of Eq. (human_fixes), written out rather than derived as
    # total_fixes - fixes_agent so that it mirrors the agent sum above.
    fixes_human = (h_rel_fixes_min + h_rel_fixes_maj + h_rel_fixes_sec
                 + h_fixes_min    + h_fixes_maj    + h_fixes_sec)
    total_fixes = fixes_human + fixes_agent

    # ------------------------------------------------------------------
    # Escalation queue update
    # ------------------------------------------------------------------
    # Escalations arrive per completed *agent* work item, not per deployed agent:
    # rho_esc is a per-task escalation probability.  Only the roles whose throughput
    # is denominated in comparable task units enter W_agent — the refactoring, test
    # and documentation rates are in index units and would swamp the sum.  Human work
    # is excluded because a developer does not escalate to themselves, and the
    # background detection services (k=5,6) file findings rather than escalating.
    # The (2 - delta_D) factor makes debt increase the escalation rate, matching the
    # convention of r_bugs_eff above.
    W_agent = impl_agents + arch_chg_done + fixes_agent
    esc_new = int(rng.poisson(params.rho_esc * (2.0 - delta_D) * W_agent))
    state.Q_esc = max(0, state.Q_esc + esc_new - esc_done)

    # ------------------------------------------------------------------
    # Technical debt and documentation update (General Maintenance)
    # ------------------------------------------------------------------
    D_raw = (state.D_tech
             + params.delta_impl * impl
             + params.delta_fix * total_fixes)
    D_new = max(0.0, min(1.0,
        D_raw - params.r_refactor * A_active[3],   # Refactoring Agent (k=4, index 3)
    ))
    # Debt the refactoring actually removed: where it would have landed without any
    # refactoring, minus where it landed with it.  Bounded by r_refactor * a_4 and zero
    # when the role is unstaffed, so it is the billable work unit for k=4.  Computed
    # beside the update rather than by restructuring it — the clip order matters.
    refactor_applied = min(1.0, D_raw) - D_new
    state.D_tech = D_new

    # Documentation coverage is capped at 1, i.e. one document per completed feature;
    # beyond that the Developer Documentation Agent (k=8, index 7) has no work, exactly
    # as for the test suite above.
    doc_cap = float(state.F_completed)
    docs_written = max(0.0, min(params.r_doc * A_active[7], doc_cap - state.N_doc))
    state.N_doc += docs_written

    # ------------------------------------------------------------------
    # Release event
    # ------------------------------------------------------------------
    new_escaped_min = new_escaped_maj = new_escaped_sec = 0
    if (t + 1) % config.R == 0:
        # Hidden bugs (undetected) and known bugs (detected but unfixed) both ship
        new_escaped_min = state.B_hid_min + state.B_kno_min
        new_escaped_maj = state.B_hid_maj + state.B_kno_maj
        new_escaped_sec = state.B_hid_sec + state.B_kno_sec
        state.B_escaped_min += new_escaped_min
        state.B_escaped_maj += new_escaped_maj
        state.B_escaped_sec += new_escaped_sec
        state.B_rel_min += new_escaped_min
        state.B_rel_maj += new_escaped_maj
        state.B_rel_sec += new_escaped_sec
        state.B_hid_min = state.B_hid_maj = state.B_hid_sec = 0
        state.B_kno_min = state.B_kno_maj = state.B_kno_sec = 0

    # ------------------------------------------------------------------
    # Billable work completed this sprint, by role (paper Eq. cost_agents)
    # ------------------------------------------------------------------
    # Every role except the two background services is demand-driven and idle when no
    # work reaches it: roles 1-4 through the queues they drain, roles 7-8 through the
    # coverage caps above.  The background services have no discrete unit of output, so
    # they are charged per deployed instance per sprint (r_5 = r_6 = 1).
    return SprintActions(
        units=(
            # k=1 features implemented, plus proposals at ARCH_IMPL_COST_UNITS each
            float(impl_agents + ARCH_IMPL_COST_UNITS * arch_impl_done),
            float(arch_chg_done),                  # k=2 change requests resolved
            float(fixes_agent),                    # k=3 bugs repaired by agents
            float(refactor_applied),               # k=4 technical debt removed
            float(A_active[4]),                    # k=5 deployed review instances
            float(A_active[5]),                    # k=6 deployed audit instances
            tests_written,                         # k=7 tests written
            docs_written,                          # k=8 documentation items written
        ),
        # Contribution accounting, not billed — see the class docstring.
        impl_human=float(impl_human),
        impl_agents=float(impl_agents),
        fixes_human=float(fixes_human),
        fixes_agent=float(fixes_agent),
    )

