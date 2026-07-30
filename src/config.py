"""Team configuration (N, M, A_dev, R, F_freeze, A_qa)"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

AGENT_ROLE_COUNT = 8

# The five orthogonal groups of scenario parameters (paper §4.1).  Each group is
# supplied by its own JSON file; together they cover every ScenarioParameters
# field exactly once.
SCENARIO_GROUPS: tuple[str, ...] = ("human", "project", "debt", "agent", "bugimpact")

AGENT_ROLE_NAMES: tuple[str, ...] = (
    "Feature Implementation Agent",           # k=1
    "Architecture & Design Proposal Agent",   # k=2
    "Bug Fixing Agent",                       # k=3
    "Refactoring Agent",                      # k=4
    "Code Review Agent",                      # k=5
    "Security Audit Agent",                   # k=6
    "Test Development Agent",                 # k=7
    "Developer Documentation Agent",          # k=8
)

# Roles are split into two tiers with respect to human oversight (paper §3.3).
# Supervised roles consume monitoring capacity and are bounded by the span of
# control N*M.  The two background detection services (Code Review, Security
# Audit) are triggered automatically by the CI infrastructure and report their
# findings asynchronously, so they need no human supervision and are therefore
# neither counted against N*M nor charged monitoring overhead.
SUPERVISED_ROLE_INDICES: tuple[int, ...] = (0, 1, 2, 3, 6, 7)   # k = 1,2,3,4,7,8
BACKGROUND_ROLE_INDICES: tuple[int, ...] = (4, 5)               # k = 5,6


def supervised_agent_count(A: tuple[int, ...]) -> int:
    """Number of agents in `A` that consume human supervision (n_sup, paper §3.3)."""
    return sum(A[i] for i in SUPERVISED_ROLE_INDICES)


@dataclass(frozen=True, slots=True)
class TeamConfiguration:
    """Configuration tuple (N, M, A_dev, R, F_freeze, A_qa) fixed across replications.

    A_dev = (a_1, ..., a_8) — development-phase composition, a_1 >= 1.
    A_qa  = (a_1, ..., a_8) — QA/stabilisation-phase composition.
            Defaults to A_dev when omitted.  Expected to have a_1=a_2=0 so the
            deployed agents are redeployed to bug-fixing, testing, and
            refactoring during the feature-freeze window.
    The span-of-control bound sum <= N*M applies to the *supervised* roles only
    (see SUPERVISED_ROLE_INDICES); the two background detection services
    (a_5, a_6) are unbounded, since their only brake is their token cost
    weighed against concave detection returns.
    F_freeze — number of sprints at the end of each release cycle in which the
            team uses A_qa instead of A_dev.  Sprints satisfying
            (t % R) >= (R - F_freeze) are "frozen".  Range: [0, R-1].
            F_freeze = 0 means no feature freeze (A_qa is never used).
    """

    N: int                              # Human Developer count
    M: int                              # Span of control (agents per human)
    A_dev: tuple[int, ...]              # Dev-phase composition, length == AGENT_ROLE_COUNT
    R: int                              # Release cadence (sprints)
    F_freeze: int = 1                   # QA-phase length per release cycle, in [0, R-1]
    A_qa: Optional[tuple[int, ...]] = None  # QA-phase composition; None → defaults to A_dev

    def __post_init__(self) -> None:
        # Resolve A_qa before validation so subsequent checks can use it.
        if self.A_qa is None:
            object.__setattr__(self, "A_qa", self.A_dev)

        if self.N < 1:
            raise ValueError(f"N must be >= 1, got {self.N}")
        if self.M < 1:
            raise ValueError(f"M must be >= 1, got {self.M}")
        if self.R < 1:
            raise ValueError(f"R must be >= 1, got {self.R}")

        # Validate development composition A_dev
        if len(self.A_dev) != AGENT_ROLE_COUNT:
            raise ValueError(f"A_dev must have length {AGENT_ROLE_COUNT}, got {len(self.A_dev)}")
        if any(a < 0 for a in self.A_dev):
            raise ValueError(f"A_dev entries must be >= 0, got {self.A_dev}")
        if supervised_agent_count(self.A_dev) > self.N * self.M:
            raise ValueError(
                f"supervised agents in A_dev must be <= N*M = {self.N * self.M}, "
                f"got {supervised_agent_count(self.A_dev)}"
            )
        if self.A_dev[0] < 1:
            raise ValueError("a_1 (Feature Implementation Agent) must be >= 1")

        # Validate F_freeze
        if not (0 <= self.F_freeze <= self.R - 1):
            raise ValueError(
                f"F_freeze must be in [0, R-1={self.R - 1}], got {self.F_freeze}"
            )

        # Validate QA composition A_qa
        if len(self.A_qa) != AGENT_ROLE_COUNT:  # type: ignore[arg-type]
            raise ValueError(
                f"A_qa must have length {AGENT_ROLE_COUNT}, got {len(self.A_qa)}"  # type: ignore[arg-type]
            )
        if any(a < 0 for a in self.A_qa):  # type: ignore[union-attr]
            raise ValueError(f"A_qa entries must be >= 0, got {self.A_qa}")
        if supervised_agent_count(self.A_qa) > self.N * self.M:  # type: ignore[arg-type]
            raise ValueError(
                f"supervised agents in A_qa must be <= N*M = {self.N * self.M}, "
                f"got {supervised_agent_count(self.A_qa)}"  # type: ignore[arg-type]
            )
        if self.F_freeze > 0:
            if self.A_qa[0] != 0:  # type: ignore[index]
                raise ValueError(
                    "a_qa,1 (Feature Implementation Agent) must be 0 during feature freeze"
                )
            if self.A_qa[1] != 0:  # type: ignore[index]
                raise ValueError(
                    "a_qa,2 (Architecture & Design Agent) must be 0 during feature freeze"
                )

    def agent_count_dev(self, role_k: int) -> int:
        """1-indexed access to the development composition A_dev: role_k in {1, ..., 8}."""
        if not 1 <= role_k <= AGENT_ROLE_COUNT:
            raise ValueError(f"role_k must be in [1, {AGENT_ROLE_COUNT}], got {role_k}")
        return self.A_dev[role_k - 1]

    def agent_count_qa(self, role_k: int) -> int:
        """1-indexed access to the QA/freeze composition A_qa: role_k in {1, ..., 8}."""
        if not 1 <= role_k <= AGENT_ROLE_COUNT:
            raise ValueError(f"role_k must be in [1, {AGENT_ROLE_COUNT}], got {role_k}")
        return self.A_qa[role_k - 1]  # type: ignore[index]

    def active_A(self, is_frozen: bool) -> tuple[int, ...]:
        """Return A_qa during freeze sprints, A_dev otherwise."""
        return self.A_qa if is_frozen else self.A_dev  # type: ignore[return-value]

    def n_supervised(self, is_frozen: bool) -> int:
        """Supervised agent count n_sup of the composition active in this sprint."""
        return supervised_agent_count(self.active_A(is_frozen))


@dataclass(frozen=True, slots=True)
class SearchConfiguration:
    """Parameters controlling a configuration-space search."""

    scenario_paths: dict[str, str]          # one path per entry of SCENARIO_GROUPS
    horizon_path: str
    strategy: str                           # "random" or "evolutionary"
    N_values: tuple[int, ...]
    M_max: int
    R_values: tuple[int, ...]
    F_freeze_values: tuple[int, ...] = (1,)
    n_samples: int = 500                    # random: total samples
    n_samples_initial: int = 100            # evolutionary: initial population
    n_generations: int = 15                 # evolutionary: generation count
    n_offspring: int = 20                   # evolutionary: offspring per generation
    seed: int = 0                           # RNG seed for search

    def __post_init__(self) -> None:
        if set(self.scenario_paths) != set(SCENARIO_GROUPS):
            missing = sorted(set(SCENARIO_GROUPS) - self.scenario_paths.keys())
            unknown = sorted(self.scenario_paths.keys() - set(SCENARIO_GROUPS))
            raise ValueError(
                f"scenario_paths must have exactly one entry per scenario group "
                f"{list(SCENARIO_GROUPS)}; missing={missing}, unknown={unknown}"
            )
        if self.strategy not in ("random", "evolutionary"):
            raise ValueError(f"strategy must be 'random' or 'evolutionary', got {self.strategy!r}")
        if not self.N_values:
            raise ValueError("N_values must be non-empty")
        if self.M_max < 1:
            raise ValueError(f"M_max must be >= 1, got {self.M_max}")
        if not self.R_values:
            raise ValueError("R_values must be non-empty")
        if not self.F_freeze_values:
            raise ValueError("F_freeze_values must be non-empty")
        if self.n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {self.n_samples}")
        if self.n_samples_initial < 1:
            raise ValueError(f"n_samples_initial must be >= 1, got {self.n_samples_initial}")
        if self.n_generations < 1:
            raise ValueError(f"n_generations must be >= 1, got {self.n_generations}")
        if self.n_offspring < 1:
            raise ValueError(f"n_offspring must be >= 1, got {self.n_offspring}")


@dataclass(frozen=True, slots=True)
class SimulationHorizon:
    """Top-level simulation horizon and Monte Carlo settings."""

    T: int                      # Sprints per replication
    K: int                      # Monte Carlo replications per configuration
    master_seed: int            # Root seed s_0

    def __post_init__(self) -> None:
        if self.T < 1:
            raise ValueError(f"T must be >= 1, got {self.T}")
        if self.K < 1:
            raise ValueError(f"K must be >= 1, got {self.K}")
