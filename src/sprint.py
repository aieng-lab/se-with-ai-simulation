"""Sprint transition equations per FRAMEWORK.md §4.

Within-sprint ordering (from the §4 preamble):
  Prelim  Degradation functions, τ_CI(t), human capacity allocation (§3.1–3.5)
  4.1     Feature arrival and implementation
  4.2     Bug injection + Stage A detection
  4.3     Stage B detection (test layers)
  4.4     Architecture review queue dynamics
  4.5     Escalation queue dynamics
  4.6     Bug fixing
  4.7     Regression injection
  4.8     Technical debt update
  4.9     Test suite and documentation update
  4.10    CI/CD pipeline latency (τ_CI(t+1) — implicit, derived from updated state)
  4.11    Release event

`q_arch_eff(t)` in §4.2 uses `Q_rev(t)` (start of sprint), *not* the post-4.4 queue.
τ_CI is not stored in SystemState; it is re-derived from F_completed and tests_k each sprint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .config import TeamConfiguration
from .parameters import ScenarioParameters
from .state import LatencyBuffer, SystemState

if TYPE_CHECKING:
    from numpy.random import Generator


@dataclass(frozen=True, slots=True)
class SprintRecord:
    """Per-sprint output quantities needed for cost accounting (§8.1)."""

    impl: int               # total features implemented this sprint
    fixes_min: int          # minor bugs fixed
    fixes_maj: int          # major bugs fixed
    new_escaped_min: int    # minor bugs escaped to production (0 unless release sprint)
    new_escaped_maj: int    # major bugs escaped to production (0 unless release sprint)


def step_sprint(
    t: int,
    state: SystemState,
    buffer: LatencyBuffer,
    config: TeamConfiguration,
    params: ScenarioParameters,
    rng: "Generator",
    T: int,
) -> SprintRecord:
    """Advance `state` and `buffer` from sprint t to sprint t+1 (in place).

    Implements §4.1–§4.11 in the canonical order documented above.
    `T` is the simulation horizon used to cap τ_CI per §4.10.
    """

    # ------------------------------------------------------------------
    # Preliminary: feature-freeze flag and active agent composition
    # Sprint t is frozen iff (t % R) >= (R - F_freeze); use A_qa instead of A.
    # ------------------------------------------------------------------
    is_frozen = (t % config.R) >= (config.R - config.F_freeze)
    A_active = config.active_A(is_frozen)

    # ------------------------------------------------------------------
    # Preliminary: degradation functions at current debt D(t)
    # ------------------------------------------------------------------
    phi_D = 1.0 + (params.phi_max - 1.0) * (state.D ** params.kappa)
    psi_D = 1.0 - (1.0 - params.psi_min) * (state.D ** params.gamma)

    # ------------------------------------------------------------------
    # Preliminary: τ_CI(t) from current state (§4.3 / §4.10)
    # Not stored; re-derived from F_completed and tests each sprint.
    # ------------------------------------------------------------------
    n_tests_now = (
        params.beta_UT * state.tests_UT
        + params.beta_IT * state.tests_IT
        + params.beta_ST * state.tests_ST
    )
    tau_CI_now = min(
        params.tau_CI_base
        * (1.0 + params.alpha_F_CI * state.F_completed)
        * (1.0 + params.alpha_test_CI * n_tests_now),
        float(T),
    )
    tau_ceil_now = math.ceil(tau_CI_now)

    # ------------------------------------------------------------------
    # Preliminary: human capacity allocation (§3.1–3.5)
    # ------------------------------------------------------------------
    H_net = max(0.0, params.H_capacity - params.c_comm * (config.N - 1))
    H_mon_eff = params.H_monitoring_overhead * (1.0 - params.Gamma_doc * state.doc_cov)
    H_after_mon = max(0.0, H_net - config.M * H_mon_eff)

    # Quality degradation factors from start-of-sprint queue sizes
    eta_rev = max(
        params.eta_min_rev,
        1.0 - params.alpha_eta_rev * max(0, state.Q_rev - params.Q_thr_rev),
    )
    eta_esc = max(
        params.eta_min_esc,
        1.0 - params.alpha_eta_esc * max(0, state.Q_esc - params.Q_thr_esc),
    )

    # Architecture reviews served this sprint
    H_for_rev = min(H_after_mon, state.Q_rev * params.H_review_time / config.N)
    rev_done = int(H_for_rev * config.N / params.H_review_time)
    H_after_rev = H_after_mon - H_for_rev

    # Escalations served this sprint
    H_for_esc = min(H_after_rev, state.Q_esc * params.H_escalation_time / config.N)
    esc_done = int(H_for_esc * config.N / params.H_escalation_time)
    H_after_esc = H_after_rev - H_for_esc

    # Human coding capacity and feature output (zero during feature freeze)
    H_code = params.f_code * H_after_esc
    impl_human = 0 if is_frozen else int(H_code * config.N / params.h_feature)

    # Effective architecture quality (degraded when review queue is overloaded)
    q_arch_eff = params.q_arch * eta_rev + (1.0 - eta_rev) * params.q_arch_floor

    # ------------------------------------------------------------------
    # §4.1 Feature arrival and implementation
    # ------------------------------------------------------------------
    arrivals = int(rng.poisson(params.lambda_F))
    b_F = min(1.0, state.Q_rev / params.Q_thr_block) if params.Q_thr_block > 0 else 0.0
    a_3 = A_active[2]
    impl_agents = min(
        state.F_backlog,
        int(params.r_F * psi_D * a_3 * (1.0 - b_F)),
    )
    impl = impl_agents + impl_human
    state.F_backlog = max(0, state.F_backlog + arrivals - impl)
    state.F_completed += impl

    # ------------------------------------------------------------------
    # §4.2 Bug injection + Stage A detection
    # ------------------------------------------------------------------
    q_arch_factor = max(1.0, min(2.0, 2.0 - q_arch_eff))
    lambda_impl_eff = (
        params.lambda_impl
        * phi_D
        * (1.0 + params.alpha_spec * params.epsilon_spec)
        * q_arch_factor
    )

    bugs_raw = int(rng.poisson(impl_agents * lambda_impl_eff))
    bugs_human_raw = int(rng.poisson(impl_human * lambda_impl_eff * params.lambda_human_scale))
    bugs_total = bugs_raw + bugs_human_raw
    bugs_maj = int(rng.binomial(bugs_total, params.theta)) if bugs_total > 0 else 0
    bugs_min = bugs_total - bugs_maj

    # Orthogonal security split (§4.2 / A7)
    bugs_sec_maj = int(rng.binomial(bugs_maj, params.p_sec)) if bugs_maj > 0 else 0
    bugs_sec_min = int(rng.binomial(bugs_min, params.p_sec)) if bugs_min > 0 else 0
    bugs_nonsec_maj = bugs_maj - bugs_sec_maj
    bugs_nonsec_min = bugs_min - bugs_sec_min

    # d_CR_eff: code review detection (A8: not throughput-limited by a_6)
    size_factor = max(
        0.0,
        1.0 - params.alpha_size * math.log((impl + 1) / params.baseline_size),
    )
    d_CR_eff = min(1.0, params.d_CR * eta_rev * size_factor)

    CR_det_maj = int(rng.binomial(bugs_nonsec_maj, d_CR_eff)) if bugs_nonsec_maj > 0 else 0
    CR_det_min = int(rng.binomial(bugs_nonsec_min, d_CR_eff)) if bugs_nonsec_min > 0 else 0
    SA_det_maj = int(rng.binomial(bugs_sec_maj, params.d_SA)) if bugs_sec_maj > 0 else 0
    SA_det_min = int(rng.binomial(bugs_sec_min, params.d_SA)) if bugs_sec_min > 0 else 0

    state.B_kno_maj += CR_det_maj + SA_det_maj
    state.B_kno_min += CR_det_min + SA_det_min

    buf_maj = bugs_nonsec_maj - CR_det_maj + bugs_sec_maj - SA_det_maj
    buf_min = bugs_nonsec_min - CR_det_min + bugs_sec_min - SA_det_min
    buffer.add(t, buf_min, buf_maj, tau_ceil_now)

    # ------------------------------------------------------------------
    # §4.3 Stage B detection (sequential test layers)
    # ------------------------------------------------------------------
    C_min, C_maj = buffer.pop_detectable(t)

    cov_UT = min(params.c_UT_ceil, state.tests_UT / max(1, state.F_completed))
    cov_IT = min(params.c_IT_ceil, state.tests_IT / max(1, state.F_completed))
    cov_ST = min(params.c_ST_ceil, state.tests_ST / max(1, state.F_completed))

    # Each layer sees survivors of the previous layer; clamp to [0,1]
    p_UT = min(1.0, params.d_UT * cov_UT)
    UT_det_maj = int(rng.binomial(C_maj, p_UT)) if C_maj > 0 else 0
    UT_det_min = int(rng.binomial(C_min, p_UT)) if C_min > 0 else 0
    C_maj -= UT_det_maj
    C_min -= UT_det_min

    p_IT = min(1.0, params.d_IT * cov_IT)
    IT_det_maj = int(rng.binomial(C_maj, p_IT)) if C_maj > 0 else 0
    IT_det_min = int(rng.binomial(C_min, p_IT)) if C_min > 0 else 0
    C_maj -= IT_det_maj
    C_min -= IT_det_min

    p_ST = min(1.0, params.d_ST * cov_ST)
    ST_det_maj = int(rng.binomial(C_maj, p_ST)) if C_maj > 0 else 0
    ST_det_min = int(rng.binomial(C_min, p_ST)) if C_min > 0 else 0
    C_maj -= ST_det_maj
    C_min -= ST_det_min

    state.B_kno_maj += UT_det_maj + IT_det_maj + ST_det_maj
    state.B_kno_min += UT_det_min + IT_det_min + ST_det_min
    state.B_hid_maj += C_maj   # survivors of all 3 test layers (hidden, escape at release)
    state.B_hid_min += C_min

    # ------------------------------------------------------------------
    # §4.4 Architecture review queue update
    # ------------------------------------------------------------------
    arch_new = int(rng.poisson(params.r_arch * A_active[1]))
    state.Q_rev = max(0, state.Q_rev + arch_new - rev_done)

    # ------------------------------------------------------------------
    # §4.5 Escalation queue update
    # ------------------------------------------------------------------
    esc_new = int(rng.poisson(config.N * config.M * params.rho_esc))
    state.Q_esc = max(0, state.Q_esc + esc_new - esc_done)
    re_esc = int(rng.binomial(esc_done, 1.0 - eta_esc)) if esc_done > 0 else 0
    state.Q_esc += re_esc

    # ------------------------------------------------------------------
    # §4.6 Bug fixing
    # ------------------------------------------------------------------
    a_4 = A_active[3]
    r_fix_min_eff = params.r_fix_minor * psi_D
    r_fix_maj_eff = params.r_fix_major * psi_D
    fixes_min = min(state.B_kno_min, int(r_fix_min_eff * a_4 * (1.0 - params.theta)))
    fixes_maj = min(state.B_kno_maj, int(r_fix_maj_eff * a_4 * params.theta))
    state.B_kno_min -= fixes_min
    state.B_kno_maj -= fixes_maj

    # ------------------------------------------------------------------
    # §4.7 Regression injection (all regressions are minor, §4.7)
    # ------------------------------------------------------------------
    total_fixes = fixes_min + fixes_maj
    reg_from_fixes = int(rng.binomial(total_fixes, params.p_regress)) if total_fixes > 0 else 0
    refactor_units = round(A_active[4] * params.r_refactor)
    reg_from_refactor = (
        int(rng.binomial(refactor_units, params.p_regress_refactor))
        if refactor_units > 0 else 0
    )
    regressions = reg_from_fixes + reg_from_refactor
    buffer.add(t, regressions, 0, tau_ceil_now)

    # ------------------------------------------------------------------
    # §4.8 Technical debt update
    # ------------------------------------------------------------------
    state.D = max(0.0, min(1.0,
        state.D
        + params.delta_impl * impl
        + params.delta_fix * total_fixes
        - params.delta_refactor * A_active[4] * params.r_refactor,
    ))

    # ------------------------------------------------------------------
    # §4.9 Test suite and documentation update
    # ------------------------------------------------------------------
    state.tests_UT += params.r_UT * A_active[7]
    state.tests_IT += params.r_IT * A_active[8]
    state.tests_ST += params.r_ST * A_active[9]
    state.doc_cov = max(0.0, min(1.0,
        state.doc_cov
        + (params.r_doc * A_active[11] - params.zeta_doc * impl)
        / max(1, state.F_completed),
    ))

    # §4.10 τ_CI(t+1) is implicit: derived from updated F_completed and tests_k
    # at the start of the next step_sprint call. No state storage required.

    # ------------------------------------------------------------------
    # §4.11 Release event
    # ------------------------------------------------------------------
    new_escaped_min = new_escaped_maj = 0
    if (t + 1) % config.R == 0:
        late_min, late_maj = buffer.drain_all()
        new_escaped_min = state.B_hid_min + late_min
        new_escaped_maj = state.B_hid_maj + late_maj
        state.B_escaped_min += new_escaped_min
        state.B_escaped_maj += new_escaped_maj
        state.B_hid_min = 0
        state.B_hid_maj = 0

    return SprintRecord(
        impl=impl,
        fixes_min=fixes_min,
        fixes_maj=fixes_maj,
        new_escaped_min=new_escaped_min,
        new_escaped_maj=new_escaped_maj,
    )
