"""Scenario-level parameters per FRAMEWORK.md §9.2 Step 1.

Holds the concrete numeric values drawn once per replication and held fixed across
all T sprints. Distribution families and hyperparameters live in VARIABLES.md and
are materialized by `src.sampling.sample_scenario_parameters`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import AGENT_ROLE_COUNT


@dataclass(frozen=True, slots=True)
class ScenarioParameters:
    """All scenario-level random variables, drawn once per replication.

    Field order mirrors FRAMEWORK.md §9.2 Step 1 for cross-referencing.
    """

    # --- Arrival / productivity rates (§3, §4.1) ---
    lambda_F: float                  # Feature arrival rate (features/sprint)
    r_req: float                     # Requirements refinement rate
    r_arch: float                    # Architecture proposal rate
    r_F: float                       # Feature implementation rate (per agent)
    r_fix_minor: float               # Minor bug fix rate
    r_fix_major: float               # Major bug fix rate
    r_refactor: float                # Refactoring work rate (debt units/sprint/agent)
    r_review: float                  # Code review throughput (cost-only, see A8)
    r_security: float                # Security audit throughput (cost-only)
    r_UT: float                      # Unit test development rate
    r_IT: float                      # Integration test development rate
    r_ST: float                      # System/acceptance test development rate
    r_doc: float                     # Developer documentation rate

    # --- Human capacity (§3) ---
    H_capacity: float                # Hours/sprint per developer
    H_monitoring_overhead: float     # Monitoring hours per agent
    H_review_time: float             # Hours per architecture review
    H_escalation_time: float         # Hours per escalation
    h_feature: float                 # Human hours per feature (§3.5)
    c_comm: float                    # Per-peer communication overhead
    f_code: float                    # Fraction of residual capacity used for coding

    # --- Bug injection (§4.2, §5.1) ---
    lambda_impl: float               # Base bug injection rate per feature
    lambda_human_scale: float        # Human vs agent bug injection scale (0, 1]
    phi_max: float                   # Max debt bug amplifier
    kappa: float                     # Debt amplifier shape exponent
    theta: float                     # Major bug fraction [0, 1]
    p_sec: float                     # Security-class bug fraction [0, 1]
    epsilon_spec: float              # Specification error rate [0, 1]
    alpha_spec: float                # Spec error injection amplifier
    q_arch: float                    # Architecture quality score [0, 1]
    q_arch_floor: float              # Rushed-review architecture quality floor

    # --- Regressions (§4.7) ---
    p_regress: float                 # Fix regression probability
    p_regress_refactor: float        # Refactoring regression probability

    # --- Detection rates (§4.2, §4.3) ---
    d_CR: float                      # Code review detection rate
    d_SA: float                      # Security audit detection rate
    d_UT: float                      # Unit test max detection rate
    d_IT: float                      # Integration test max detection rate
    d_ST: float                      # System test max detection rate
    c_UT_ceil: float                 # Unit test coverage ceiling
    c_IT_ceil: float                 # Integration test coverage ceiling
    c_ST_ceil: float                 # System test coverage ceiling
    alpha_size: float                # Code review size degradation rate
    baseline_size: float             # Baseline PR size (features/sprint)

    # --- CI/CD pipeline latency (§4.10) ---
    tau_CI_base: float               # Baseline CI/CD latency (sprints)
    alpha_F_CI: float                # Latency growth per feature
    alpha_test_CI: float             # Latency growth per weighted test
    beta_UT: float                   # Unit test execution time weight
    beta_IT: float                   # Integration test execution time weight
    beta_ST: float                   # System test execution time weight

    # --- Technical debt (§4.8, §5.2) ---
    psi_min: float                   # Min debt slowdown multiplier (0, 1)
    gamma: float                     # Debt slowdown shape exponent
    delta_impl: float                # Debt increment per feature
    delta_fix: float                 # Debt increment per fix
    delta_refactor: float            # Debt decrement per refactoring unit

    # --- Documentation (§4.9) ---
    Gamma_doc: float                 # Documentation overhead reduction [0, 1]
    zeta_doc: float                  # Coverage erosion rate [0, 1]

    # --- Review/escalation quality degradation (§5.3, §5.4) ---
    eta_min_rev: float               # Minimum review quality floor
    alpha_eta_rev: float             # Review degradation slope
    Q_thr_rev: int                   # Review queue degradation threshold
    eta_min_esc: float               # Minimum escalation quality floor
    alpha_eta_esc: float             # Escalation degradation slope
    Q_thr_esc: int                   # Escalation queue degradation threshold
    Q_thr_block: int                 # Architecture-review blocking threshold

    # --- Escalation (§4.5) ---
    rho_esc: float                   # Per-agent escalation rate

    # --- Initial conditions (§2.3) ---
    D0: float                        # Initial technical debt [0, 1]
    R_warmup: int                    # Warm-start horizon (sprints)

    # --- Costs (§8.1) ---
    c_token: tuple[float, ...] = field(default_factory=lambda: (0.0,) * AGENT_ROLE_COUNT)
    c_salary: float = 0.0            # Human developer annual cost
    c_PO: float = 0.0                # Product Owner annual cost
    C_fix_minor: float = 0.0         # Minor bug fix cost
    C_fix_major: float = 0.0         # Major bug fix cost
    C_prod_min: float = 0.0          # Production impact per minor escaped bug
    C_prod_maj: float = 0.0          # Production impact per major escaped bug
    C_DevOps_setup: float = 0.0      # One-time DevOps setup cost
    C_DevOps_maint: float = 0.0      # Per-sprint DevOps maintenance cost
    C_user_doc: float = 0.0          # User documentation cost per release

    def __post_init__(self) -> None:
        if len(self.c_token) != AGENT_ROLE_COUNT:
            raise ValueError(
                f"c_token must have length {AGENT_ROLE_COUNT}, got {len(self.c_token)}"
            )
