"""Human-capacity gate, applied to a configuration before it is simulated."""

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
    """Check that supervising the agents leaves an FTE some capacity; say why if not.

    Evaluated at perfect documentation coverage, the lowest monitoring overhead the
    configuration can incur, so passing here is necessary but not sufficient: the sprint
    transition can still exhaust the budget while coverage is being built up.
    """
    H_mon_eff_star = params.H_monitoring_overhead * (1.0 - params.Gamma_doc)
    H_net = net_capacity(params.H_capacity, params.c_comm, config.N)

    # A_qa is applied only when there is a freeze window to apply it in.
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
