"""Team configuration (N, M, A, R) per FRAMEWORK.md §2.1."""

from __future__ import annotations

from dataclasses import dataclass

AGENT_ROLE_COUNT = 13

AGENT_ROLE_NAMES: tuple[str, ...] = (
    "Requirements Refinement Agent",          # k=1
    "Architecture & Design Proposal Agent",   # k=2
    "Feature Implementation Agent",           # k=3
    "Bug Fixing Agent",                       # k=4
    "Refactoring Agent",                      # k=5
    "Code Review Agent",                      # k=6
    "Security Audit Agent",                   # k=7
    "Unit Test Development Agent",            # k=8
    "Integration Test Development Agent",     # k=9
    "System & Acceptance Test Development Agent",  # k=10
    "DevOps/CI-CD Agent",                     # k=11
    "Developer Documentation Agent",          # k=12
    "User Documentation Agent",               # k=13
)


@dataclass(frozen=True, slots=True)
class TeamConfiguration:
    """Configuration tuple (N, M, A, R) fixed across replications of one sweep point.

    A = (a_1, ..., a_13) counts AI agents only. Human developers are specified by N;
    the Product Owner is a fixed singleton (cost via c_PO) and is not part of A.
    """

    N: int              # Human Developer count
    M: int              # Span of control (agents per human)
    A: tuple[int, ...]  # Agent composition, length == AGENT_ROLE_COUNT, sum == N*M
    R: int              # Release cadence (sprints)

    def __post_init__(self) -> None:
        if self.N < 1:
            raise ValueError(f"N must be >= 1, got {self.N}")
        if self.M < 1:
            raise ValueError(f"M must be >= 1, got {self.M}")
        if self.R < 1:
            raise ValueError(f"R must be >= 1, got {self.R}")
        if len(self.A) != AGENT_ROLE_COUNT:
            raise ValueError(f"A must have length {AGENT_ROLE_COUNT}, got {len(self.A)}")
        if any(a < 0 for a in self.A):
            raise ValueError(f"A entries must be >= 0, got {self.A}")
        if sum(self.A) != self.N * self.M:
            raise ValueError(
                f"sum(A) must equal N*M = {self.N * self.M}, got {sum(self.A)}"
            )
        if self.A[2] < 1:
            raise ValueError("a_3 (Feature Implementation Agent) must be >= 1")

    def agent_count(self, role_k: int) -> int:
        """1-indexed access: role_k in {1, ..., 13}."""
        if not 1 <= role_k <= AGENT_ROLE_COUNT:
            raise ValueError(f"role_k must be in [1, {AGENT_ROLE_COUNT}], got {role_k}")
        return self.A[role_k - 1]


@dataclass(frozen=True, slots=True)
class SweepSettings:
    """Top-level simulation horizon and Monte Carlo settings."""

    T: int                      # Sprints per replication
    K: int                      # Monte Carlo replications per configuration
    master_seed: int            # Root seed s_0 (FRAMEWORK §9.0)
    N_PO: int = 1               # Product Owner count (fixed)

    def __post_init__(self) -> None:
        if self.T < 1:
            raise ValueError(f"T must be >= 1, got {self.T}")
        if self.K < 1:
            raise ValueError(f"K must be >= 1, got {self.K}")
        if self.N_PO < 1:
            raise ValueError(f"N_PO must be >= 1, got {self.N_PO}")
