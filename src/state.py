"""System state vector per paper/main.tex (§3)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SystemState:
    """Full per-sprint state S(t). Mutated in place by the sprint transition."""

    F_backlog: int = 0
    F_completed: int = 0
    B_hid_min: int = 0     # hidden bugs (not detected; escape at next release)
    B_hid_maj: int = 0
    B_hid_sec: int = 0
    B_kno_min: int = 0
    B_kno_maj: int = 0
    B_kno_sec: int = 0
    B_escaped_min: int = 0
    B_escaped_maj: int = 0
    B_escaped_sec: int = 0
    B_rel_min: int = 0     # escaped bugs still open in production
    B_rel_maj: int = 0
    B_rel_sec: int = 0
    D_tech: float = 0.0
    Q_rev: int = 0
    Q_arch: int = 0       # pending architecture change requests (impl-generated, arch-drained)
    Q_arch_impl: int = 0  # approved architecture proposals awaiting implementation (rev_done-filled)
    Q_esc: int = 0
    N_test: float = 0.0     # cumulative tests written by Test Development Agent
    N_doc: float = 0.0     # cumulative documentation items written

    @property
    def B_escaped(self) -> int:
        return self.B_escaped_min + self.B_escaped_maj + self.B_escaped_sec

    @property
    def B_rel(self) -> int:
        return self.B_rel_min + self.B_rel_maj + self.B_rel_sec

    @property
    def B_kno(self) -> int:
        return self.B_kno_min + self.B_kno_maj + self.B_kno_sec

    @property
    def B_hid(self) -> int:
        return self.B_hid_min + self.B_hid_maj + self.B_hid_sec
