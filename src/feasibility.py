"""Feasibility gating: one mean-field check applied before Monte Carlo simulation.

  Step 1 — Human capacity: the monitoring of the supervised agents (K_sup, i.e.
            not the three autonomous roles) must not exhaust H_net.  Evaluated at
            doc_cov = 1 (perfect documentation), giving the minimum possible
            monitoring overhead, which makes the check a necessary but not a
            sufficient condition.  Applied to A_dev, and to A_qa when the
            configuration has a freeze window.

No analogous bound is imposed on architecture-review or escalation time: leaving
those queues unserved only increases the debt penalties, so implementation
remains possible (paper §3.6, Step 1).
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import TeamConfiguration, net_capacity, supervised_agent_count
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
    # Step 1 — Human capacity (monitoring vs. available capacity).  Only the
    # supervised roles K_sup = {1,2,3,7,8} enter this bound: the two detection
    # services and the Refactoring Agent run autonomously (paper §3.3), so they
    # are exempt from supervision.
    # ------------------------------------------------------------------
    H_net = net_capacity(params.H_capacity, params.c_comm, config.N)

    # Both compositions must clear the bound, but A_qa only when a freeze window
    # exists: with F_freeze == 0 it is never applied and the search leaves an
    # arbitrary unexercised vector in those rows.
    compositions = [("A_dev", config.A_dev)]
    if config.F_freeze > 0:
        compositions.append(("A_qa", config.A_qa))

    # Monitoring alone must not exhaust the capacity of one FTE.
    for name, A in compositions:
        actual_agents_per_fte = supervised_agent_count(A) / config.N
        if actual_agents_per_fte * H_mon_eff_star >= H_net:
            return FeasibilityReport(
                feasible=False, failed_step=1,
                reason=f"Monitoring overhead of {name} exhausts H_net even at "
                       f"perfect documentation",
            )

    return FeasibilityReport(feasible=True)
