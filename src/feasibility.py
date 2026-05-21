"""Feasibility gating per FRAMEWORK.md §7 (Stability Conditions).

Applies the mean-field checks in the order documented in §7:
  Step 0 — Hard infeasibility gate (p_regress × p_det_B >= 1)
  Step 1 — Theorem 7.3b compound span-of-control
  Step 2 — Theorem 7.5 architecture review queue stability
  Step 3 — Condition 4 debt-spiral check; solve for D*
  Step 4 — Theorem 7.1 feature backlog stability using D*
  Step 5 — Theorem 7.2 bug queue traffic intensity using D* and steady-state coverage
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import TeamConfiguration
from .parameters import ScenarioParameters


@dataclass(frozen=True, slots=True)
class FeasibilityReport:
    feasible: bool
    failed_step: int | None = None   # 0..5 (None when feasible)
    reason: str = ""
    D_star: float | None = None      # Steady-state debt from Condition 4


def check_feasibility(
    config: TeamConfiguration,
    params: ScenarioParameters,
) -> FeasibilityReport:
    """Run steps 0–5 from FRAMEWORK.md §7 in order; return the first failure or OK."""
    a_2  = config.agent_count(2)
    a_3  = config.agent_count(3)
    a_4  = config.agent_count(4)
    a_5  = config.agent_count(5)
    a_8  = config.agent_count(8)
    a_9  = config.agent_count(9)
    a_10 = config.agent_count(10)

    # doc_cov_star = 0: conservative approximation (breaks circular coupling);
    # maximises H_mon_eff_star → hardest possible span-of-control constraint.
    H_mon_eff_star = params.H_monitoring_overhead * (1.0 - params.Gamma_doc * 0.0)

    # ------------------------------------------------------------------
    # Step 0 — Hard infeasibility gate
    # Evaluate p_det_B at coverage ceilings (worst-case regression feedback).
    # ------------------------------------------------------------------
    p_det_B_max = 1.0 - (
        (1.0 - params.d_UT * params.c_UT_ceil)
        * (1.0 - params.d_IT * params.c_IT_ceil)
        * (1.0 - params.d_ST * params.c_ST_ceil)
    )
    if params.p_regress * p_det_B_max >= 1.0:
        return FeasibilityReport(
            feasible=False, failed_step=0,
            reason="p_regress × p_det_B_max ≥ 1: regression feedback loop self-sustaining",
        )

    # ------------------------------------------------------------------
    # Step 1 — Theorem 7.3b (compound span-of-control)
    # ------------------------------------------------------------------
    H_net = params.H_capacity - params.c_comm * (config.N - 1)
    review_load_per_dev = (params.r_arch * a_2 / config.N) * params.H_review_time
    numerator = H_net - review_load_per_dev
    denominator = H_mon_eff_star + params.rho_esc * params.H_escalation_time
    if denominator <= 0.0 or config.M > numerator / denominator:
        return FeasibilityReport(
            feasible=False, failed_step=1,
            reason="Theorem 7.3b: span-of-control M exceeds capacity",
        )
    H_after_mon_star = max(0.0, H_net - config.M * H_mon_eff_star)

    # ------------------------------------------------------------------
    # Step 2 — Theorem 7.5 (architecture review queue stability)
    # ------------------------------------------------------------------
    if params.r_arch * params.H_review_time > 0.0:
        a_2_max = H_after_mon_star * config.N / (params.r_arch * params.H_review_time)
        if a_2 > a_2_max:
            return FeasibilityReport(
                feasible=False, failed_step=2,
                reason="Theorem 7.5: architecture review queue unstable (too many a_2 agents)",
            )
    H_remaining_per_dev = max(
        0.0,
        H_after_mon_star - (params.r_arch * a_2 / config.N) * params.H_review_time,
    )

    # E[impl_human]: human coding capacity at steady state
    H_code_per_dev = params.f_code * max(
        0.0,
        H_remaining_per_dev - config.M * params.rho_esc * params.H_escalation_time,
    )
    E_impl_human = H_code_per_dev * config.N / params.h_feature

    # ------------------------------------------------------------------
    # Step 3 — Theorem 7.4 (debt stability; solve D*)
    # The balance equation is:
    #   δ_refactor × a_5 × r_refactor
    #     = δ_impl × (r_F × ψ_D(D*) × a_3 + E[impl_human])
    #     + δ_fix  × a_4 × r_fix_mean × ψ_D(D*)
    # Note: E[impl_human] is NOT multiplied by ψ_D (human capacity is debt-independent).
    # Infeasibility: stability condition violated at D=1 (FRAMEWORK.md §7 Condition 4).
    # ------------------------------------------------------------------
    r_fix_mean = params.r_fix_minor * (1.0 - params.theta) + params.r_fix_major * params.theta
    refactor_cap = params.delta_refactor * a_5 * params.r_refactor

    # Debt generation at D=1 (maximum slowdown, ψ_D = ψ_min)
    debt_gen_at_D1 = (
        params.delta_impl * (params.r_F * params.psi_min * a_3 + E_impl_human)
        + params.delta_fix * a_4 * r_fix_mean * params.psi_min
    )
    if refactor_cap <= debt_gen_at_D1:
        return FeasibilityReport(
            feasible=False, failed_step=3,
            reason="Theorem 7.4: debt spirals to 1 (stability condition violated at D=1)",
        )

    # Solve for D* using corrected formula (FRAMEWORK.md §7 balance equation):
    #   ψ_D(D*) = (refactor_cap − δ_impl × E[impl_human])
    #             / (δ_impl × r_F × a_3 + δ_fix × a_4 × r_fix_mean)
    agent_debt_denom = (
        params.delta_impl * params.r_F * a_3
        + params.delta_fix * a_4 * r_fix_mean
    )
    if agent_debt_denom <= 0.0:
        D_star = 0.0
    else:
        psi_D_star = (refactor_cap - params.delta_impl * E_impl_human) / agent_debt_denom
        if psi_D_star >= 1.0:
            D_star = 0.0  # refactoring outpaces debt even at D=0
        else:
            D_star = ((1.0 - psi_D_star) / (1.0 - params.psi_min)) ** (1.0 / params.gamma)
            D_star = min(D_star, 1.0)  # numerical safety clamp

    # ------------------------------------------------------------------
    # Step 4 — Theorem 7.1 (feature backlog stability)
    # Scale by the fraction of non-frozen sprints: features are only implemented
    # during the (R - F_freeze) development sprints out of each R-sprint cycle.
    # ------------------------------------------------------------------
    psi_D_at_star = 1.0 - (1.0 - params.psi_min) * (D_star ** params.gamma)
    active_fraction = (config.R - config.F_freeze) / config.R
    impl_rate_star = (params.r_F * psi_D_at_star * a_3 + E_impl_human) * active_fraction
    if impl_rate_star < params.lambda_F:
        return FeasibilityReport(
            feasible=False, failed_step=4,
            reason="Theorem 7.1: feature backlog unstable (impl_rate < λ_F)",
            D_star=D_star,
        )

    # ------------------------------------------------------------------
    # Step 5 — Theorem 7.2 (bug queue traffic intensity ρ_bugs < 1)
    # ------------------------------------------------------------------
    # Steady-state coverage: rate-ratio approximation, capped at ceiling
    safe_impl = max(1e-12, impl_rate_star)
    cov_UT_star = min(params.c_UT_ceil, a_8  * params.r_UT / safe_impl)
    cov_IT_star = min(params.c_IT_ceil, a_9  * params.r_IT / safe_impl)
    cov_ST_star = min(params.c_ST_ceil, a_10 * params.r_ST / safe_impl)

    p_det_B = 1.0 - (
        (1.0 - params.d_UT * cov_UT_star)
        * (1.0 - params.d_IT * cov_IT_star)
        * (1.0 - params.d_ST * cov_ST_star)
    )

    # d_CR_eff at steady state: assume Q_rev=0 (η_rev=1); size degradation from impl rate
    size_factor = max(
        0.0,
        1.0 - params.alpha_size * math.log((impl_rate_star + 1.0) / params.baseline_size),
    )
    d_CR_eff_star = params.d_CR * size_factor

    p_det_A = 1.0 - (
        (1.0 - d_CR_eff_star * (1.0 - params.p_sec))
        * (1.0 - params.d_SA * params.p_sec)
    )
    p_det_total = 1.0 - (1.0 - p_det_A) * (1.0 - p_det_B)

    phi_D_star = 1.0 + (params.phi_max - 1.0) * (D_star ** params.kappa)
    q_arch_factor = max(1.0, min(2.0, 2.0 - params.q_arch))
    lambda_impl_eff_star = (
        params.lambda_impl
        * phi_D_star
        * (1.0 + params.alpha_spec * params.epsilon_spec)
        * q_arch_factor
    )

    Lambda_known = impl_rate_star * lambda_impl_eff_star * p_det_total
    Lambda_refact = a_5 * params.r_refactor * params.p_regress_refactor * p_det_B
    # Step 0 guarantees (1 - p_regress × p_det_B) > 0 here
    Lambda_net = (Lambda_known + Lambda_refact) / (1.0 - params.p_regress * p_det_B)

    mu_fix = a_4 * (
        params.r_fix_minor * psi_D_at_star * (1.0 - params.theta)
        + params.r_fix_major * psi_D_at_star * params.theta
    )
    rho_bugs = Lambda_net / max(1e-12, mu_fix)
    if rho_bugs >= 1.0:
        return FeasibilityReport(
            feasible=False, failed_step=5,
            reason=f"Theorem 7.2: bug queue unstable (ρ_bugs={rho_bugs:.4f} ≥ 1)",
            D_star=D_star,
        )

    return FeasibilityReport(feasible=True, D_star=D_star)
