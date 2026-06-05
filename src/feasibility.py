"""Feasibility gating: one mean-field check applied before Monte Carlo simulation.

  Step 1 — Human capacity: monitoring must not exhaust H_net; compound
            span-of-control bound (monitoring + escalation vs. available capacity).
            Evaluated at doc_cov = c_doc_ceil (perfect documentation), giving the
            minimum possible monitoring overhead.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import TeamConfiguration
from .parameters import ScenarioParameters


@dataclass(frozen=True, slots=True)
class FeasibilityReport:
    feasible: bool
    failed_step: int | None = None   # 1 (None when feasible)
    reason: str = ""


def check_feasibility(
    config: TeamConfiguration,
    params: ScenarioParameters,
) -> FeasibilityReport:
    """Run the single feasibility check; return failure or OK."""
    # doc_cov = c_doc_ceil (perfect documentation) minimises monitoring overhead.
    H_mon_eff_star = params.H_monitoring_overhead * (1.0 - params.Gamma_doc * params.c_doc_ceil)

    # ------------------------------------------------------------------
    # Step 1 — Human capacity (monitoring + escalation vs. available capacity)
    # ------------------------------------------------------------------
    H_net = params.H_capacity - params.c_comm * (config.N - 1)
    actual_agents_per_dev = sum(config.A) / config.N

    # Monitoring alone must not exhaust developer capacity.
    if actual_agents_per_dev * H_mon_eff_star >= H_net:
        return FeasibilityReport(
            feasible=False, failed_step=1,
            reason="Monitoring overhead exhausts H_net even at perfect documentation",
        )

    return FeasibilityReport(feasible=True)
