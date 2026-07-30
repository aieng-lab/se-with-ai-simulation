"""Feasibility gating: one mean-field check applied before Monte Carlo simulation.

  Step 1 — Human capacity: the monitoring of the supervised agents must not
            exhaust H_net.  Evaluated at doc_cov = 1 (perfect documentation),
            giving the minimum possible monitoring overhead, which makes the
            check a necessary but not a sufficient condition.

No analogous bound is imposed on architecture-review or escalation time: leaving
those queues unserved only increases the debt penalties, so implementation
remains possible (paper §3.6, Step 1).
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import TeamConfiguration, supervised_agent_count
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
    # doc_cov = 1 (perfect documentation) minimises monitoring overhead.
    H_mon_eff_star = params.H_monitoring_overhead * (1.0 - params.Gamma_doc)

    # ------------------------------------------------------------------
    # Step 1 — Human capacity (monitoring vs. available capacity).  The two
    # background detection services are exempt from supervision and therefore
    # do not enter this bound.
    # ------------------------------------------------------------------
    H_net = params.H_capacity - params.c_comm * (config.N - 1)
    actual_agents_per_dev = supervised_agent_count(config.A_dev) / config.N

    # Monitoring alone must not exhaust developer capacity.
    if actual_agents_per_dev * H_mon_eff_star >= H_net:
        return FeasibilityReport(
            feasible=False, failed_step=1,
            reason="Monitoring overhead exhausts H_net even at perfect documentation",
        )

    return FeasibilityReport(feasible=True)
