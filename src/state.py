"""System state vector and latency buffer per FRAMEWORK.md §2.2."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SystemState:
    """Full per-sprint state S(t). Mutated in place by the sprint transition."""

    F_backlog: int = 0
    F_completed: int = 0
    B_hid_min: int = 0
    B_hid_maj: int = 0
    B_kno_min: int = 0
    B_kno_maj: int = 0
    B_escaped_min: int = 0
    B_escaped_maj: int = 0
    D: float = 0.0
    Q_rev: int = 0
    Q_esc: int = 0
    tests_UT: float = 0.0
    tests_IT: float = 0.0
    tests_ST: float = 0.0
    doc_cov: float = 0.0

    @property
    def B_escaped(self) -> int:
        return self.B_escaped_min + self.B_escaped_maj

    @property
    def B_kno(self) -> int:
        return self.B_kno_min + self.B_kno_maj

    @property
    def B_hid(self) -> int:
        return self.B_hid_min + self.B_hid_maj


@dataclass(slots=True)
class LatencyBuffer:
    """Per-severity cohort store keyed by injection sprint (FRAMEWORK §2.2, §6.1).

    Each cohort records the count of surviving bugs and the injection-time τ_CI(t_inj)
    so that Definition 6.1's release criterion (t >= t_inj + ceil(τ_CI(t_inj))) can be
    evaluated per cohort.
    """

    # maps t_inj -> (count, ceil_tau_CI_at_injection)
    minor: dict[int, tuple[int, int]] = field(default_factory=dict)
    major: dict[int, tuple[int, int]] = field(default_factory=dict)

    def add(self, t_inj: int, n_minor: int, n_major: int, tau_ceil: int) -> None:
        if n_minor:
            prior = self.minor.get(t_inj, (0, tau_ceil))
            self.minor[t_inj] = (prior[0] + n_minor, tau_ceil)
        if n_major:
            prior = self.major.get(t_inj, (0, tau_ceil))
            self.major[t_inj] = (prior[0] + n_major, tau_ceil)

    def pop_detectable(self, t: int) -> tuple[int, int]:
        """Remove and return (minor, major) cohorts with t >= t_inj + ceil(τ)."""
        c_min = _pop_matured(self.minor, t)
        c_maj = _pop_matured(self.major, t)
        return c_min, c_maj

    def drain_all(self) -> tuple[int, int]:
        c_min = sum(count for count, _ in self.minor.values())
        c_maj = sum(count for count, _ in self.major.values())
        self.minor.clear()
        self.major.clear()
        return c_min, c_maj


def _pop_matured(cohorts: dict[int, tuple[int, int]], t: int) -> int:
    matured = [t_inj for t_inj, (_, tau) in cohorts.items() if t >= t_inj + tau]
    total = sum(cohorts.pop(t_inj)[0] for t_inj in matured)
    return total
