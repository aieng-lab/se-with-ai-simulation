"""Scenario-level parameters: the numeric values held fixed across all T sprints.

Composed from the five per-group scenario JSON files by `io.load_scenario_params`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import AGENT_ROLE_COUNT


@dataclass(frozen=True, slots=True)
class ScenarioParameters:
    """Every scenario-level parameter, grouped by what it governs."""

    # --- Arrival / productivity rates  ---
    r_arrival: float                 # Feature arrival rate (features/sprint)
    f_arch_chg: float                # Fraction of implemented features generating architecture change requests
    r_arch_chg: float                # Architecture change request resolution rate (per agent per sprint)
    r_F: float                       # Feature implementation rate (per agent)
    r_fix: float                     # Bug fix rate (per Bug Fixing Agent per sprint at zero debt)
    r_refactor: float                # Debt reduction per Refactoring Agent per sprint
    r_test: float                    # Test development rate (per agent)
    r_doc: float                     # Developer documentation rate

    # --- Human capacity ---
    H_capacity: float                # Hours/sprint per FTE (gross, before communication)
    H_monitoring_overhead: float     # Monitoring hours per agent
    H_review_time: float             # Hours per architecture review
    H_escalation_time: float         # Hours per escalation
    h_feature: float                 # Human hours per feature
    h_fix: float                     # Human hours per bug fixed
    c_comm: float                    # Per-peer communication overhead
    f_fix: float                     # Share of residual capacity spent on bug fixing
    # Read only by a team with no agents (`config.is_human_only`).
    f_qa: float                      # Share of net capacity reserved for QA in dev sprints
    h_qa: float                      # Human hours to review one implemented feature in full
    d_qa: float                      # Probability a fully reviewed bug is detected

    # --- Bug injection  ---
    r_bugs: float                    # Bug injection rate per agent-implemented feature
    # Bugs per human-implemented feature: an absolute rate, not a factor on r_bugs.
    r_bugs_human: float
    theta_maj: float                 # Major bug fraction [0, 1)
    theta_sec: float                 # Security bug fraction [0, 1)

    # --- Detection rates ---
    d_CR: float                      # Code review detection rate
    d_SA: float                      # Security audit detection rate
    d_test: float                    # Test suite maximum detection rate
    c_test_ceil: float               # Test coverage ceiling
    alpha_size: float                # Detection degradation slope in change volume per agent

    # --- Debt amplifier ---
    delta_min: float                 # Debt amplifier floor at D_eff=1 (applied to throughput, review, escalation)
    delta_exp: float                 # Debt amplifier shape exponent
    delta_impl: float                # Debt increment per feature
    delta_fix: float                 # Debt increment per fix

    # --- Documentation ---
    Gamma_doc: float                 # Documentation overhead reduction [0, 1]

    # --- Escalation ---
    rho_esc: float                   # Per-agent escalation rate

    # --- Costs ---
    # Price of one unit of each role's throughput, k=1..8, in that role's own unit.
    # Roles 4-6 are counted per deployed instance per sprint, the rest per artefact.
    c_token: tuple[float, ...] = field(default_factory=lambda: (0.0,) * AGENT_ROLE_COUNT)
    c_salary: float = 0.0            # Annual fully-loaded cost of one FTE
    # Dollar cost of one escaped bug, by severity.  The defaults here and on c_token /
    # c_salary above are for direct construction in tests; every scenario file sets them.
    omega_min: float = 150.0         # Cost of one escaped minor bug
    omega_maj: float = 300_000.0     # Cost of one escaped major bug
    omega_sec: float = 1_000_000.0   # Cost of one escaped security bug

    def __post_init__(self) -> None:
        if len(self.c_token) != AGENT_ROLE_COUNT:
            raise ValueError(
                f"c_token must have length {AGENT_ROLE_COUNT}, got {len(self.c_token)}"
            )
        if not (0.0 < self.theta_maj < 1.0):
            raise ValueError(f"theta_maj must be in (0, 1), got {self.theta_maj}")
        if not (0.0 < self.theta_sec < 1.0):
            raise ValueError(f"theta_sec must be in (0, 1), got {self.theta_sec}")
        if not (0.0 <= self.f_fix <= 1.0):
            raise ValueError(f"f_fix must be in [0, 1], got {self.f_fix}")
        if self.r_bugs_human <= 0.0:
            raise ValueError(f"r_bugs_human must be > 0, got {self.r_bugs_human}")
        if not (0.0 <= self.f_qa < 1.0):
            raise ValueError(f"f_qa must be in [0, 1), got {self.f_qa}")
        if self.h_qa <= 0.0:
            raise ValueError(f"h_qa must be > 0, got {self.h_qa}")
        if not (0.0 <= self.d_qa <= 1.0):
            raise ValueError(f"d_qa must be in [0, 1], got {self.d_qa}")
        if self.theta_maj + self.theta_sec >= 1.0:
            raise ValueError(
                f"theta_maj + theta_sec must be < 1 (leaves room for minor), "
                f"got {self.theta_maj + self.theta_sec}"
            )
