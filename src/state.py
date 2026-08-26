"""The 21-dimensional system state carried from sprint to sprint."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SystemState:
    """Full per-sprint state S(t). Mutated in place by the sprint transition."""

    F_backlog: int = 0
    F_completed: int = 0
    B_hid_min: int = 0    # injected but undetected, by severity
    B_hid_maj: int = 0
    B_hid_sec: int = 0
    B_kno_min: int = 0    # detected but not yet repaired, by severity
    B_kno_maj: int = 0
    B_kno_sec: int = 0
    B_escaped_min: int = 0  # cumulative count that reached production
    B_escaped_maj: int = 0
    B_escaped_sec: int = 0
    B_rel_min: int = 0    # of those, the ones still open
    B_rel_maj: int = 0
    B_rel_sec: int = 0
    D_tech: float = 0.0   # normalised technical debt, in [0, 1]
    Q_rev: int = 0        # architecture proposals awaiting human review
    Q_arch: int = 0       # architecture change requests awaiting an agent
    Q_arch_impl: int = 0  # approved proposals awaiting implementation
    Q_esc: int = 0        # escalations awaiting a human
    N_test: float = 0.0   # cumulative tests written
    N_doc: float = 0.0    # cumulative documentation items written
