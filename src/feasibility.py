"""Human-capacity gate, applied to a configuration before it is simulated.

Supervising the supervised agent roles must not exhaust one FTE's net capacity.  The
check is evaluated at perfect documentation, i.e. at the smallest monitoring overhead
the configuration can ever incur, so passing it is necessary but not sufficient.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import TeamConfiguration, net_capacity, supervised_agent_count
from .parameters import ScenarioParameters


@dataclass(frozen=True, slots=True)
class FeasibilityReport:
    feasible: bool
    reason: str = ""


def check_feasibility(
    config: TeamConfiguration,
    params: ScenarioParameters,
) -> FeasibilityReport:
    """Check the human-capacity condition; return the outcome and, on failure, why."""
    # Monitoring overhead per supervised agent at perfect documentation coverage.
    H_mon_eff_star = params.H_monitoring_overhead * (1.0 - params.Gamma_doc)
    H_net = net_capacity(params.H_capacity, params.c_comm, config.N)

    # A_qa is only checked when a freeze window exists; without one it is never applied.
    compositions = [("A_dev", config.A_dev)]
    if config.F_freeze > 0:
        compositions.append(("A_qa", config.A_qa))

    for name, A in compositions:
        agents_per_fte = supervised_agent_count(A) / config.N
        if agents_per_fte * H_mon_eff_star >= H_net:
            return FeasibilityReport(
                feasible=False,
                reason=f"Monitoring overhead of {name} exhausts H_net even at "
                       f"perfect documentation",
            )

    return FeasibilityReport(feasible=True)
