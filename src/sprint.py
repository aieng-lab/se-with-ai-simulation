"""Sprint transition equations.

Within-sprint ordering:
  Step 1  Human capacity allocation
  Step 2  Debt amplifier: D_arch, D_eff, delta_D
  Step 3  Feature arrival and implementation (priority drain Q_arch_impl, then F_backlog)
  Step 4  Test suite update
  Step 5  Bug injection + detection (Test → CR → SA; all synchronous)
  Step 6  Architecture queue dynamics (Q_rev, Q_arch_impl, Q_arch)
  Step 7  Escalation queue dynamics
  Step 8  Bug fixing
  Step 9  General maintenance (technical debt + documentation update)
  Step 10 Release event

Detection is unified: Code Review, Security Audit, and the Test suite all act
within the same sprint as injection. There is no CI/CD latency buffer.
Regression bugs bypass detection and go directly to the hidden stock (A6).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .config import TeamConfiguration
from .parameters import ScenarioParameters
from .state import SystemState

if TYPE_CHECKING:
    from numpy.random import Generator


def step_sprint(
    t: int,
    state: SystemState,
    config: TeamConfiguration,
    params: ScenarioParameters,
    rng: "Generator",
) -> None:
    """Advance `state` from sprint t-1 to sprint t (in place).

    Implements the ten-step transition in canonical order.
    """

    # ------------------------------------------------------------------
    # Feature-freeze flag and active agent composition
    # ------------------------------------------------------------------
    is_frozen = (t % config.R) >= (config.R - config.F_freeze)
    A_active = config.active_A(is_frozen)
    n_agents = sum(A_active)   # actual deployed agents (may be < N*M)

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
    H_net = params.H_capacity - params.c_comm * (config.N - 1)
    doc_cov = min(params.c_doc_ceil, state.N_doc / max(1.0, state.F_completed))
    H_mon_eff = params.H_monitoring_overhead * (1.0 - params.Gamma_doc * doc_cov)
    # Monitoring overhead is proportional to actual agents per developer.
    # Feasibility guarantees H_net - agents_per_dev * H_mon_eff >= 0.
    agents_per_dev = n_agents / config.N
    H_after_mon = H_net - agents_per_dev * H_mon_eff

    # Architecture reviews served this sprint (eq. rev_done, H_after_rev)
    rev_done = min(int(H_after_mon * config.N / params.H_review_time), state.Q_rev)
    H_after_rev = H_after_mon - rev_done * params.H_review_time / config.N

    # Escalations served this sprint (eq. esc_done, H_after_esc)
    esc_done = min(int(H_after_rev * config.N / params.H_escalation_time), state.Q_esc)
    H_after_esc = H_after_rev - esc_done * params.H_escalation_time / config.N

    # f_fix fraction of H_after_esc goes to bug fixing; (1-f_fix) to feature coding.
    # During feature freeze, the coding fraction is redirected to bug fixing as well.
    H_fix_human = H_after_esc if is_frozen else params.f_fix * H_after_esc
    fix_human_budget = int(H_fix_human * config.N / params.h_fix)
    impl_human = 0 if is_frozen else int((1.0 - params.f_fix) * H_after_esc * config.N / params.h_feature)

    # ------------------------------------------------------------------
    # Feature arrival and implementation
    # ------------------------------------------------------------------
    arrivals = int(rng.poisson(params.r_arrival))
    a_3 = A_active[0]   # Feature Implementation Agent (k=1, index 0)
    total_impl_cap = int(params.r_F * delta_D * a_3)
    # Approved architecture proposals are implemented first (priority over backlog).
    # arch_impl_done consumes agent capacity but does NOT count toward F_completed.
    arch_impl_done = min(state.Q_arch_impl, total_impl_cap)
    impl_agents = min(
        state.F_backlog + arrivals,
        total_impl_cap - arch_impl_done,
    )
    impl = min(impl_agents + impl_human, state.F_backlog + arrivals)
    state.F_backlog = max(0, state.F_backlog + arrivals - impl)
    state.F_completed += impl

    # ------------------------------------------------------------------
    # Test suite update (before detection so new tests count this sprint)
    # ------------------------------------------------------------------
    state.N_test += params.r_test * A_active[6]   # Test Development Agent (k=7, index 6)

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

    # Stage 2 — Code Review (acts on all survivors)
    size_factor = max(
        0.0,
        1.0 - params.alpha_size * math.log((impl + 1) / params.baseline_size),
    )
    d_CR_eff = min(1.0, params.d_CR * delta_D * size_factor)
    CR_det_min = int(rng.binomial(surv_min_t, d_CR_eff)) if surv_min_t > 0 else 0
    CR_det_maj = int(rng.binomial(surv_maj_t, d_CR_eff)) if surv_maj_t > 0 else 0
    CR_det_sec = int(rng.binomial(surv_sec_t, d_CR_eff)) if surv_sec_t > 0 else 0

    surv_sec_cr = surv_sec_t - CR_det_sec

    # Stage 3 — Security Audit (acts only on security bugs that survived test + CR)
    SA_det_sec = int(rng.binomial(surv_sec_cr, params.d_SA)) if surv_sec_cr > 0 else 0

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
    # Escalation queue update
    # ------------------------------------------------------------------
    esc_new = int(rng.poisson(n_agents * params.rho_esc))
    state.Q_esc = max(0, state.Q_esc + esc_new - esc_done)

    # ------------------------------------------------------------------
    # Bug fixing — strict priority: sec > maj > min; rel > kno
    # Humans fix first (debt-independent count budget), then agents (work budget).
    # ------------------------------------------------------------------
    a_4 = A_active[2]   # Bug Fixing Agent (k=3, index 2)

    def _drain(queue: int, rate: float, budget: float) -> tuple[int, float]:
        """Fix as many bugs as agent work budget allows; return (fixes, budget_used)."""
        n = min(queue, int(budget * rate))
        return n, (n / rate if rate > 0 else 0.0)

    def _drain_count(queue: int, budget: int) -> int:
        """Fix as many bugs as human count budget allows; return fixes."""
        return min(queue, budget)

    # Human fixes (debt-independent, same priority order)
    hb = fix_human_budget
    h_rel_fixes_sec = _drain_count(state.B_rel_sec, hb); hb -= h_rel_fixes_sec; state.B_rel_sec -= h_rel_fixes_sec
    h_fixes_sec     = _drain_count(state.B_kno_sec, hb); hb -= h_fixes_sec;     state.B_kno_sec -= h_fixes_sec
    h_rel_fixes_maj = _drain_count(state.B_rel_maj, hb); hb -= h_rel_fixes_maj; state.B_rel_maj -= h_rel_fixes_maj
    h_fixes_maj     = _drain_count(state.B_kno_maj, hb); hb -= h_fixes_maj;     state.B_kno_maj -= h_fixes_maj
    h_rel_fixes_min = _drain_count(state.B_rel_min, hb); hb -= h_rel_fixes_min; state.B_rel_min -= h_rel_fixes_min
    h_fixes_min     = _drain_count(state.B_kno_min, hb); hb -= h_fixes_min;     state.B_kno_min -= h_fixes_min

    # Agent fixes (debt-slowed work budget, drain remaining queues)
    W = float(a_4) * delta_D
    rel_fixes_sec, w = _drain(state.B_rel_sec, params.r_fix_sec,   W); W -= w; state.B_rel_sec -= rel_fixes_sec
    fixes_sec,     w = _drain(state.B_kno_sec, params.r_fix_sec,   W); W -= w; state.B_kno_sec -= fixes_sec
    rel_fixes_maj, w = _drain(state.B_rel_maj, params.r_fix_major, W); W -= w; state.B_rel_maj -= rel_fixes_maj
    fixes_maj,     w = _drain(state.B_kno_maj, params.r_fix_major, W); W -= w; state.B_kno_maj -= fixes_maj
    rel_fixes_min, w = _drain(state.B_rel_min, params.r_fix_minor, W); W -= w; state.B_rel_min -= rel_fixes_min
    fixes_min,     w = _drain(state.B_kno_min, params.r_fix_minor, W); W -= w; state.B_kno_min -= fixes_min

    total_fixes = (h_rel_fixes_min + h_rel_fixes_maj + h_rel_fixes_sec
                 + h_fixes_min    + h_fixes_maj    + h_fixes_sec
                 + rel_fixes_min  + rel_fixes_maj  + rel_fixes_sec
                 + fixes_min      + fixes_maj      + fixes_sec)

    # ------------------------------------------------------------------
    # Technical debt and documentation update (General Maintenance)
    # ------------------------------------------------------------------
    state.D_tech = max(0.0, min(1.0,
        state.D_tech
        + params.delta_impl * impl
        + params.delta_fix * total_fixes
        - params.r_refactor * A_active[3],      # Refactoring Agent (k=4, index 3)
    ))

    state.N_doc += params.r_doc * A_active[7]   # Developer Documentation Agent (k=8)

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

