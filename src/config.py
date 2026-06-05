"""Team configuration (N, M, A, R, F_freeze, A_qa) per FRAMEWORK.md §2.1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

AGENT_ROLE_COUNT = 8

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


@dataclass(frozen=True, slots=True)
class TeamConfiguration:
    """Configuration tuple (N, M, A, R, F_freeze, A_qa) fixed across replications.

    A     = (a_1, ..., a_8) — development-phase composition (sum <= N*M, a_1 >= 1).
    A_qa  = (a_1, ..., a_8) — QA/stabilisation-phase composition (sum <= N*M).
            Defaults to A when omitted.  Expected to have a_1=a_2=0 so the
            deployed agents are redeployed to bug-fixing, testing, and
            refactoring during the feature-freeze window.
    F_freeze — number of sprints at the end of each release cycle in which the
            team uses A_qa instead of A.  Sprints satisfying
            (t % R) >= (R - F_freeze) are "frozen".  Range: [1, R].
    """

    N: int                              # Human Developer count
    M: int                              # Span of control (agents per human)
    A: tuple[int, ...]                  # Dev-phase composition, length == AGENT_ROLE_COUNT
    R: int                              # Release cadence (sprints)
    F_freeze: int = 1                   # QA-phase length per release cycle, in [1, R]
    A_qa: Optional[tuple[int, ...]] = None  # QA-phase composition; None → defaults to A

    def __post_init__(self) -> None:
        # Resolve A_qa before validation so subsequent checks can use it.
        if self.A_qa is None:
            object.__setattr__(self, "A_qa", self.A)

        if self.N < 1:
            raise ValueError(f"N must be >= 1, got {self.N}")
        if self.M < 1:
            raise ValueError(f"M must be >= 1, got {self.M}")
        if self.R < 1:
            raise ValueError(f"R must be >= 1, got {self.R}")

        # Validate development composition A
        if len(self.A) != AGENT_ROLE_COUNT:
            raise ValueError(f"A must have length {AGENT_ROLE_COUNT}, got {len(self.A)}")
        if any(a < 0 for a in self.A):
            raise ValueError(f"A entries must be >= 0, got {self.A}")
        if sum(self.A) > self.N * self.M:
            raise ValueError(
                f"sum(A) must be <= N*M = {self.N * self.M}, got {sum(self.A)}"
            )
        if self.A[0] < 1:
            raise ValueError("a_1 (Feature Implementation Agent) must be >= 1")

        # Validate F_freeze
        if not (1 <= self.F_freeze <= self.R):
            raise ValueError(
                f"F_freeze must be in [1, R={self.R}], got {self.F_freeze}"
            )

        # Validate QA composition A_qa
        if len(self.A_qa) != AGENT_ROLE_COUNT:  # type: ignore[arg-type]
            raise ValueError(
                f"A_qa must have length {AGENT_ROLE_COUNT}, got {len(self.A_qa)}"  # type: ignore[arg-type]
            )
        if any(a < 0 for a in self.A_qa):  # type: ignore[union-attr]
            raise ValueError(f"A_qa entries must be >= 0, got {self.A_qa}")
        if sum(self.A_qa) > self.N * self.M:  # type: ignore[union-attr]
            raise ValueError(
                f"sum(A_qa) must be <= N*M = {self.N * self.M}, got {sum(self.A_qa)}"  # type: ignore[union-attr]
            )

    def agent_count(self, role_k: int) -> int:
        """1-indexed access to the DEVELOPMENT composition A: role_k in {1, ..., 8}."""
        if not 1 <= role_k <= AGENT_ROLE_COUNT:
            raise ValueError(f"role_k must be in [1, {AGENT_ROLE_COUNT}], got {role_k}")
        return self.A[role_k - 1]

    def active_A(self, is_frozen: bool) -> tuple[int, ...]:
        """Return A_qa during freeze sprints, A otherwise."""
        return self.A_qa if is_frozen else self.A  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class SweepSettings:
    """Top-level simulation horizon and Monte Carlo settings."""

    T: int                      # Sprints per replication
    K: int                      # Monte Carlo replications per configuration
    master_seed: int            # Root seed s_0 (FRAMEWORK §9.0)

    def __post_init__(self) -> None:
        if self.T < 1:
            raise ValueError(f"T must be >= 1, got {self.T}")
        if self.K < 1:
            raise ValueError(f"K must be >= 1, got {self.K}")
