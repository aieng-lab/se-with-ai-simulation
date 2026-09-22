"""The sprint transition: one step of the discrete-time system.

`step_sprint` advances the state through nine stages, in this order.  
Each reads state the ones before it wrote — so the blocks below must not be reordered,
and the random draws they make must not be moved, since a stream is addressed by
index:

  debt penalties        the amplifier delta_D, from the previous sprint's queues
  capacity allocation   monitoring, reviews, escalations, QA, coding, fixing
  arrival               feature arrival and implementation
  tests                 test suite growth, before detection so new tests count now
  bugs                  injection and the detection pipeline
  architecture          the three architecture queues
  fixing                bug repair, humans first
  escalations           the escalation queue, fed by the agent work completed above
  maintenance           technical debt and documentation
  release               everything unrepaired ships

A team with no agents (`config.is_human_only`) diverges in exactly two places, both
guarded by that predicate: it reserves f_qa of its capacity for its own review, and its
detection is a single human review stage instead of the test/review/audit pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .config import (
    AGENT_ROLE_COUNT,
    TeamConfiguration,
    is_human_only,
    net_capacity,
    supervised_agent_count,
)
from .parameters import ScenarioParameters
from .state import SystemState

if TYPE_CHECKING:
    from numpy.random import Generator


#: What implementing one approved architecture proposal costs, in units of one feature.
#: It consumes that much implementation capacity and is billed at that much.
ARCH_IMPL_COST_UNITS = 5.0


@dataclass(frozen=True, slots=True)
class SprintActions:
    """Work completed in one sprint.

    `units` stores what each agent role completed, each entry in that role's own
    throughput unit: features for k=1, change requests for k=2, bugs for k=3, deployed
    instances for k=4,5,6, tests for k=7, documents for k=8.

    The four scalars are reporting, since `units[0]` conflates agent features with
    implemented architecture proposals, and human work appears in no entry at all.

        impl_human   features delivered by humans, after the demand cap
        impl_agents  features delivered by agents; `units[0]` minus
                     ARCH_IMPL_COST_UNITS * (proposals implemented)
        fixes_human  bugs repaired by humans, over all severities
        fixes_agent  bugs repaired by agents; the same number as `units[2]`
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

    The `a` deployed instances share the sprint's `impl` features, so each inspects
    `impl / a` of them.  Returns 0.0 when the service is not deployed.
    The result is in [0, 1].
    """
    if a <= 0:
        return 0.0
    return max(0.0, 1.0 - alpha_size * math.log(impl / a + 1.0))


#: Bug queues in the order they are repaired: security before major before minor, and
#: within a severity the bugs already in production before the ones still unreleased.
_FIX_PRIORITY: tuple[str, ...] = (
    "B_rel_sec", "B_kno_sec", "B_rel_maj", "B_kno_maj", "B_rel_min", "B_kno_min",
)


def _detect(rng: Generator, n: int, p: float) -> int:
    """How many of `n` bugs a detection stage catches at probability `p`.

    The `n > 0` guard keeps a stage with nothing to inspect from consuming
    a draw, which would shift every draw after it.
    """
    return int(rng.binomial(n, p)) if n > 0 else 0


def _drain_bug_queues(state: SystemState, budget: int) -> int:
    """Repair bugs from `state` in priority order until `budget` runs out.

    The budget counts bugs, one per repair whatever the severity.  Returns how many
    were repaired; `state` is updated in place.
    """
    repaired = 0
    for queue in _FIX_PRIORITY:
        if budget <= 0:
            break
        available = getattr(state, queue)
        if available <= 0:
            continue
        n = min(available, budget)
        setattr(state, queue, available - n)
        budget -= n
        repaired += n
    return repaired


def step_sprint(
    t: int,
    state: SystemState,
    config: TeamConfiguration,
    params: ScenarioParameters,
    rng: Generator,
) -> SprintActions:
    """Advance `state` from sprint t-1 to sprint t, in place.

    Returns the work each agent role completed, which the agent bill is computed from.
    """

    # --- Active composition -------------------------------------------------
    is_frozen = (t % config.R) >= (config.R - config.F_freeze)
    A_active = config.active_A(is_frozen)
    n_supervised = supervised_agent_count(A_active)

    # --- Debt penalties ------------------------------------------------------
    # The two queue debts are normalised by what the team could clear in one sprint:
    # H_capacity * N / H_x_time items.
    cap_review = params.H_capacity * config.N / params.H_review_time
    cap_esc    = params.H_capacity * config.N / params.H_escalation_time
    Q_arch_total = state.Q_arch + state.Q_rev + state.Q_arch_impl
    D_arch = min(1.0, Q_arch_total / cap_review)
    D_esc  = min(1.0, state.Q_esc  / cap_esc)
    D_eff  = min(1.0, state.D_tech + D_arch + D_esc)
    delta_D = 1.0 - (1.0 - params.delta_min) * (D_eff ** params.delta_exp)

    # --- Human capacity allocation -------------------------------------------
    # Every budget here is per FTE, the terms that consume it scale by config.N.
    H_net = net_capacity(params.H_capacity, params.c_comm, config.N)
    doc_cov = min(1.0, state.N_doc / max(1.0, state.F_completed))
    H_mon_eff = params.H_monitoring_overhead * (1.0 - params.Gamma_doc * doc_cov)
    # The feasibility gate only guarantees a positive remainder at doc_cov = 1,
    # so monitoring can still exhaust the budget early in a run.
    agents_per_fte = n_supervised / config.N
    H_after_mon = max(0.0, H_net - agents_per_fte * H_mon_eff)

    # The served count is floored against
    # the budget, but re-multiplying by the per-item cost can overshoot it by an ulp,
    # and both consumers floor the result and would lose a whole item to it.
    rev_done = min(math.floor(H_after_mon * config.N / params.H_review_time), state.Q_rev)
    H_after_rev = max(0.0, H_after_mon - rev_done * params.H_review_time / config.N)

    esc_done = min(math.floor(H_after_rev * config.N / params.H_escalation_time), state.Q_esc)
    H_after_esc = max(0.0, H_after_rev - esc_done * params.H_escalation_time / config.N)

    # A team with no agents reserves f_qa of what is left to review its own work.
    human_only = is_human_only(config)
    H_after_qa = H_after_esc
    if human_only:
        H_after_qa = (1.0 - params.f_qa) * H_after_esc

    # f_fix of the remainder goes to bug fixing, the rest to coding; a freeze sprint
    # redirects the coding share to fixing too.
    H_fix_human = H_after_qa if is_frozen else params.f_fix * H_after_qa
    fix_human_budget = math.floor(H_fix_human * config.N / params.h_fix)
    impl_human = 0 if is_frozen else math.floor(
        (1.0 - params.f_fix) * H_after_qa * config.N / params.h_feature
    )

    # --- Feature arrival and implementation ----------------------------------
    arrivals = int(rng.poisson(params.r_arrival))
    a_1 = A_active[0]   # Feature Implementation Agent (k=1, index 0)
    total_impl_cap = int(params.r_F * delta_D * a_1)
    # Approved proposals take priority over the backlog.  They consume implementation
    # capacity but do not count toward F_completed.
    arch_impl_cap = int(total_impl_cap // ARCH_IMPL_COST_UNITS)
    arch_impl_done = min(state.Q_arch_impl, arch_impl_cap)
    # Humans serve demand first, agents fill the rest.  Each stream is capped at the
    # residual backlog rather than only their sum, so impl_human + impl_agents == impl
    # and the bug injection below cannot charge for features nobody implemented.
    demand = state.F_backlog + arrivals
    impl_human = min(impl_human, demand)
    impl_agents = min(demand - impl_human,
                      total_impl_cap - int(ARCH_IMPL_COST_UNITS * arch_impl_done))
    impl = impl_agents + impl_human
    state.F_backlog = max(0, state.F_backlog + arrivals - impl)
    state.F_completed += impl
    state.F_cycle += impl

    # --- Test suite growth, before detection so new tests count now ----------
    # Past the coverage ceiling a further test buys nothing, so the agent (k=7) writes
    # nothing and is billed nothing.  max(0, .) guards an initial state above the cap.
    test_cap = params.c_test_ceil * state.F_completed
    tests_written = max(0.0, min(params.r_test * A_active[6], test_cap - state.N_test))
    state.N_test += tests_written

    # --- Bug injection and detection -----------------------------------------
    # Only the agent stream is debt-amplified.
    r_bugs_eff = params.r_bugs * (2.0 - delta_D)
    bugs_raw = int(rng.poisson(impl_agents * r_bugs_eff))
    bugs_human_raw = int(rng.poisson(impl_human * params.r_bugs_human))
    bugs_total = bugs_raw + bugs_human_raw

    # Three mutually exclusive severities: security first, then major/minor of the rest.
    theta_min = 1.0 - params.theta_maj - params.theta_sec
    bugs_sec = _detect(rng, bugs_total, params.theta_sec)
    remaining = bugs_total - bugs_sec
    theta_maj_cond = (params.theta_maj / (theta_min + params.theta_maj)
                      if (theta_min + params.theta_maj) > 0 else 0.0)
    bugs_maj = _detect(rng, remaining, theta_maj_cond)
    bugs_min = remaining - bugs_maj

    volume = state.F_cycle if is_frozen else impl
    if is_frozen:
        bugs_min += state.B_hid_min
        bugs_maj += state.B_hid_maj
        bugs_sec += state.B_hid_sec
        state.B_hid_min = state.B_hid_maj = state.B_hid_sec = 0

    if human_only:
        # A single human review stage, replacing the three agentic ones below.  cov_qa
        # is the share of features the reserved capacity can review in full, and the
        # same detection probability applies to all three severities.
        cov_qa = min(1.0, params.f_qa * config.N * H_net
                          / (params.h_qa * max(1, volume)))
        p_qa = cov_qa * params.d_qa
        det_min = _detect(rng, bugs_min, p_qa)
        det_maj = _detect(rng, bugs_maj, p_qa)
        det_sec = _detect(rng, bugs_sec, p_qa)
    else:
        # Stage 1 — the test suite, acting on all three severities.
        cov_test = min(params.c_test_ceil, state.N_test / max(1, state.F_completed))
        p_test = 0.0 if is_frozen else min(1.0, params.d_test * cov_test)
        test_det_min = _detect(rng, bugs_min, p_test)
        test_det_maj = _detect(rng, bugs_maj, p_test)
        test_det_sec = _detect(rng, bugs_sec, p_test)

        surv_min_t = bugs_min - test_det_min
        surv_maj_t = bugs_maj - test_det_maj
        surv_sec_t = bugs_sec - test_det_sec

        # Stage 2 — code review, acting on every survivor.  Zero agents means no review
        # happens at all; more agents share the change volume and each degrade less.
        a_CR = A_active[4]   # Code Review Agent (k=5, index 4)
        d_CR_eff = params.d_CR * delta_D * _volume_factor(volume, a_CR, params.alpha_size)
        CR_det_min = _detect(rng, surv_min_t, d_CR_eff)
        CR_det_maj = _detect(rng, surv_maj_t, d_CR_eff)
        CR_det_sec = _detect(rng, surv_sec_t, d_CR_eff)

        surv_sec_cr = surv_sec_t - CR_det_sec

        # Stage 3 — security audit, on the security bugs that survived stages 1 and 2.
        a_SA = A_active[5]   # Security Audit Agent (k=6, index 5)
        d_SA_eff = params.d_SA * delta_D * _volume_factor(volume, a_SA, params.alpha_size)
        SA_det_sec = _detect(rng, surv_sec_cr, d_SA_eff)

        det_min = test_det_min + CR_det_min
        det_maj = test_det_maj + CR_det_maj
        det_sec = test_det_sec + CR_det_sec + SA_det_sec

    # Detected bugs join the known queues, the rest the hidden ones.
    state.B_kno_min += det_min
    state.B_kno_maj += det_maj
    state.B_kno_sec += det_sec
    state.B_hid_min += bugs_min - det_min
    state.B_hid_maj += bugs_maj - det_maj
    state.B_hid_sec += bugs_sec - det_sec

    # --- Architecture queues -------------------------------------------------
    # Q_arch (change requests) -> Q_rev (proposals awaiting review) -> Q_arch_impl
    # (approved, awaiting implementation).
    arch_chg_new = int(rng.poisson(params.f_arch_chg * impl))
    state.Q_arch += arch_chg_new
    arch_chg_done = min(state.Q_arch, int(params.r_arch_chg * A_active[1]))
    state.Q_arch = max(0, state.Q_arch - arch_chg_done)
    state.Q_rev = max(0, state.Q_rev + arch_chg_done - rev_done)
    state.Q_arch_impl = max(0, state.Q_arch_impl + rev_done - arch_impl_done)

    # --- Bug fixing -----------------------------------------------------------
    # Humans spend their budget first, then the agents spend theirs on what is left.
    fixes_human = _drain_bug_queues(state, fix_human_budget)
    a_3 = A_active[2]   # Bug Fixing Agent (k=3, index 2)
    fixes_agent = _drain_bug_queues(state, int(a_3 * delta_D * params.r_fix))
    total_fixes = fixes_human + fixes_agent

    # --- Escalation queue ------------------------------------------------------
    # The rate is per completed agent task, so W_agent sums the three roles whose
    # output is counted in tasks.
    W_agent = impl_agents + arch_chg_done + fixes_agent
    esc_new = int(rng.poisson(params.rho_esc * (2.0 - delta_D) * W_agent))
    state.Q_esc = max(0, state.Q_esc + esc_new - esc_done)

    # --- Maintenance: technical debt and documentation -------------------------
    D_raw = (state.D_tech
             + params.delta_impl * impl
             + params.delta_fix * total_fixes)
    # The refactoring term is in the same D_tech units as the two increments above,
    # and nothing throttles the removal but the clip into [0, 1].
    state.D_tech = max(0.0, min(1.0,
        D_raw - params.r_refactor * A_active[3],   # Refactoring Agent (k=4, index 3)
    ))

    # One document per completed feature is full coverage; past that the agent (k=8)
    # writes nothing and is billed nothing, as for the test suite above.
    doc_cap = float(state.F_completed)
    docs_written = max(0.0, min(params.r_doc * A_active[7], doc_cap - state.N_doc))
    state.N_doc += docs_written

    # --- Release event ----------------------------------------------------------
    if (t + 1) % config.R == 0:
        # Everything unrepaired ships, whether or not it was ever detected.
        escaped_min = state.B_hid_min + state.B_kno_min
        escaped_maj = state.B_hid_maj + state.B_kno_maj
        escaped_sec = state.B_hid_sec + state.B_kno_sec
        state.B_escaped_min += escaped_min
        state.B_escaped_maj += escaped_maj
        state.B_escaped_sec += escaped_sec
        state.B_rel_min += escaped_min
        state.B_rel_maj += escaped_maj
        state.B_rel_sec += escaped_sec
        state.B_hid_min = state.B_hid_maj = state.B_hid_sec = 0
        state.B_kno_min = state.B_kno_maj = state.B_kno_sec = 0
        state.F_cycle = 0

    # --- Billable work completed this sprint, by role ---------------------------
    # Roles 1-3 and 7-8 bill for what they produced, so one that no work reached bills
    # nothing; roles 4-6 bill per deployed instance whether or not they had work.
    return SprintActions(
        units=(
            # k=1 features implemented, plus proposals at ARCH_IMPL_COST_UNITS each
            float(impl_agents + ARCH_IMPL_COST_UNITS * arch_impl_done),
            float(arch_chg_done),                  # k=2 change requests resolved
            float(fixes_agent),                    # k=3 bugs repaired by agents
            float(A_active[3]),                    # k=4 deployed refactoring instances
            float(A_active[4]),                    # k=5 deployed review instances
            float(A_active[5]),                    # k=6 deployed audit instances
            tests_written,                         # k=7 tests written
            docs_written,                          # k=8 documentation items written
        ),
        impl_human=float(impl_human),
        impl_agents=float(impl_agents),
        fixes_human=float(fixes_human),
        fixes_agent=float(fixes_agent),
    )
