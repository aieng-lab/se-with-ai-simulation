"""Monte Carlo execution protocol.

Pass `capture_traces=True` to `run_configuration` to record the full per-sprint
state evolution as a `ConfigTrace`; the default returns the lightweight `ConfigResult`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass

from .config import AGENT_ROLE_COUNT, SimulationHorizon, TeamConfiguration
from .parameters import ScenarioParameters
from .rng import generator_for
from .sprint import step_sprint
from .state import SystemState


@dataclass(frozen=True, slots=True)
class ReplicationResult:
    """Summary of one replication over the whole horizon."""

    value: float            # features per sprint, F_completed(T) / T
    cost: float             # dollars per completed feature
    # The three drivers of `cost`, sharing its denominator, so they sum to it.
    cost_quality: float = 0.0
    cost_human: float = 0.0
    cost_agent: float = 0.0
    # Work completed per role, k=1..8, each in that role's own throughput unit.
    actions: tuple[float, ...] = ()
    # The human/agent split of the work; `actions` cannot answer this on its own.
    impl_human: float = 0.0
    impl_agents: float = 0.0
    fixes_human: float = 0.0
    fixes_agent: float = 0.0


@dataclass(frozen=True, slots=True)
class ConfigResult:
    config: TeamConfiguration
    replications: tuple[ReplicationResult, ...]


@dataclass(frozen=True, slots=True)
class SprintSnapshot:
    """The system state at the end of sprint `t`: every SystemState field, plus `t`.

    The field order is the column order of `results.to_dataframe`, so it is part of the
    persisted trace format and is not the same as SystemState's.
    """

    t: int
    F_backlog: int
    F_completed: int
    B_kno_min: int
    B_kno_maj: int
    B_kno_sec: int
    B_hid_min: int
    B_hid_maj: int
    B_hid_sec: int
    B_escaped_min: int
    B_escaped_maj: int
    B_escaped_sec: int
    B_rel_min: int
    B_rel_maj: int
    B_rel_sec: int
    D_tech: float
    Q_rev: int
    Q_arch: int
    Q_arch_impl: int
    Q_esc: int
    N_test: float
    N_doc: float


@dataclass(frozen=True, slots=True)
class ReplicationTrace:
    """Full per-sprint history for one replication."""

    k: int
    snapshots: tuple[SprintSnapshot, ...]


@dataclass(frozen=True, slots=True)
class ConfigTrace:
    """A ConfigResult plus the per-sprint history of each of its replications."""

    config: TeamConfiguration
    replications: tuple[ReplicationResult, ...]
    traces: tuple[ReplicationTrace, ...]


def phase_sprint_counts(config: TeamConfiguration, T: int) -> tuple[int, int]:
    """(dev_sprint_count, qa_sprint_count) over a T-sprint horizon."""
    qa_sprint_count = sum(
        1 for t in range(T)
        if (t % config.R) >= (config.R - config.F_freeze)
    )
    return T - qa_sprint_count, qa_sprint_count


def escaped_bug_cost(params: ScenarioParameters, B_min: int, B_maj: int, B_sec: int) -> float:
    """Dollar cost of the bugs that reached production, by severity."""
    return (params.omega_min * B_min
            + params.omega_maj * B_maj
            + params.omega_sec * B_sec)


def cost_components(
    config: TeamConfiguration,
    params: ScenarioParameters,
    T: int,
    actions: Sequence[float],
) -> tuple[float, float]:
    """(C_human, C_agents) over a T-sprint horizon.

    `actions` is the per-role work completed over the horizon, accumulated from the
    `SprintActions` of each sprint; `c_token[k]` prices one unit of it.
    """
    if len(actions) != AGENT_ROLE_COUNT:
        raise ValueError(
            f"actions must have length {AGENT_ROLE_COUNT}, got {len(actions)}"
        )
    # c_salary is annual and a sprint is one week, so T sprints are T/52 of a year.
    C_human = config.N * params.c_salary * T / 52
    C_agents = sum(actions[k] * params.c_token[k] for k in range(AGENT_ROLE_COUNT))
    return C_human, C_agents


def run_configuration(
    config: TeamConfiguration,
    settings: SimulationHorizon,
    params: ScenarioParameters,
    *,
    capture_traces: bool = False,
    initial_state: SystemState | None = None,
    K: int | None = None,
    k_offset: int = 0,
) -> ConfigResult | ConfigTrace:
    """Run K independent replications of `config` under `settings` and `params`.

    With `capture_traces=False` (default) returns a lightweight `ConfigResult`.
    With `capture_traces=True` returns a `ConfigTrace` with full per-sprint history.
    `initial_state` sets the starting system state; defaults to all-zero if not provided.

    `K` overrides the replication count, which otherwise is `settings.K_search`, and
    `k_offset` shifts the block of replication indices from `[0, K)` to
    `[k_offset, k_offset + K)`, which is how a caller requests a disjoint block.
    """
    rep_results: list[ReplicationResult] = []
    rep_traces: list[ReplicationTrace] = []

    n_replications = settings.K_search if K is None else K
    if n_replications < 1:
        raise ValueError(f"K must be >= 1, got {n_replications}")
    if k_offset < 0:
        raise ValueError(f"k_offset must be >= 0, got {k_offset}")

    for k in range(k_offset, k_offset + n_replications):
        result, trace = _run_replication(config, settings, params, k, capture_traces, initial_state)
        rep_results.append(result)
        if trace is not None:
            rep_traces.append(trace)

    if capture_traces:
        return ConfigTrace(
            config=config,
            replications=tuple(rep_results),
            traces=tuple(rep_traces),
        )
    return ConfigResult(config=config, replications=tuple(rep_results))


def _run_replication(
    config: TeamConfiguration,
    settings: SimulationHorizon,
    params: ScenarioParameters,
    k: int,
    capture_traces: bool,
    initial_state: SystemState | None = None,
) -> tuple[ReplicationResult, ReplicationTrace | None]:
    rng = generator_for(settings.master_seed, config, k)
    # Copied, so a shared initial_state is not mutated across replications.
    state = dataclasses.replace(initial_state) if initial_state is not None else SystemState()

    snapshots: list[SprintSnapshot] = []
    actions = [0.0] * AGENT_ROLE_COUNT
    impl_human = impl_agents = fixes_human = fixes_agent = 0.0

    for t in range(settings.T):
        sprint_actions = step_sprint(t, state, config, params, rng)
        for role in range(AGENT_ROLE_COUNT):
            actions[role] += sprint_actions.units[role]
        impl_human  += sprint_actions.impl_human
        impl_agents += sprint_actions.impl_agents
        fixes_human += sprint_actions.fixes_human
        fixes_agent += sprint_actions.fixes_agent

        if capture_traces:
            snapshots.append(SprintSnapshot(t=t, **dataclasses.asdict(state)))

    C_human, C_agents = cost_components(config, params, settings.T, actions)
    C_quality = escaped_bug_cost(
        params, state.B_escaped_min, state.B_escaped_maj, state.B_escaped_sec
    )
    C_total = C_quality + C_human + C_agents

    F_done = state.F_completed
    # A horizon can legitimately complete no feature at all, so floor the denominator.
    denom = max(1, F_done)
    result = ReplicationResult(
        value=F_done / settings.T,
        cost=C_total / denom,
        cost_quality=C_quality / denom,
        cost_human=C_human / denom,
        cost_agent=C_agents / denom,
        actions=tuple(actions),
        impl_human=impl_human,
        impl_agents=impl_agents,
        fixes_human=fixes_human,
        fixes_agent=fixes_agent,
    )

    trace = ReplicationTrace(k=k, snapshots=tuple(snapshots)) if capture_traces else None
    return result, trace
