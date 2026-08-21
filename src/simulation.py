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
    value: float    # F_completed(T) / T, in features per sprint
    # The objective: (C_quality + C_human + C_agents) / F_completed(T), formed *within*
    # this replication, so it is charged for the bugs it let through and the agent work it
    # consumed, and credited with the features it delivered.
    cost: float
    # The three cost drivers, sharing `cost`'s denominator, so they sum to `cost`.  Stored
    # rather than reconstructed: the quality and agent terms depend on the realised
    # trajectory, so no fixed ratio recovers them from `cost` alone.
    cost_quality: float = 0.0
    cost_human: float = 0.0
    cost_agent: float = 0.0
    # Work completed per role over the horizon, in each role's own throughput unit, so the
    # analysis can tell a role that worked from one that was merely deployed.
    # `cost_agent` is its inner product with c_token.
    actions: tuple[float, ...] = ()
    # Who did the work, summed over the horizon: features delivered and bugs repaired,
    # split by human developers and agents.  Reporting only — not billable, and invisible
    # to the selection rule.  `impl_human + impl_agents == F_completed(T)` exactly.
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
    """Full system state captured at the end of sprint t."""

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
    """Full trace for all K replications; includes summary objectives and per-sprint history."""

    config: TeamConfiguration
    replications: tuple[ReplicationResult, ...]   # same as ConfigResult.replications
    traces: tuple[ReplicationTrace, ...]


def phase_sprint_counts(config: TeamConfiguration, T: int) -> tuple[int, int]:
    """(dev_sprint_count, qa_sprint_count) over a T-sprint horizon.

    A sprint belongs to the feature freeze when it falls in the last `F_freeze`
    sprints of its release cycle of length `R`.
    """
    qa_sprint_count = sum(
        1 for t in range(T)
        if (t % config.R) >= (config.R - config.F_freeze)
    )
    return T - qa_sprint_count, qa_sprint_count


def escaped_bug_cost(params: ScenarioParameters, B_min: int, B_maj: int, B_sec: int) -> float:
    """What the bugs that reached production cost.

    The severity weights are absolute expected dollar costs of a single escaped minor,
    major or security bug, not relative quality weights.  The counts cover everything that
    shipped: bugs never detected, and bugs detected but not fixed before their release.
    """
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

    `actions` is the per-role work completed over the whole horizon, accumulated from the
    `SprintActions` that `step_sprint` returns.  Agents are billed for the work they
    performed, so one that no work reaches costs nothing; `c_token[k]` prices one unit of
    role k's throughput, which is exactly the unit `actions[k]` is counted in.  C_human is
    deterministic given `(config, T)`.
    """
    if len(actions) != AGENT_ROLE_COUNT:
        raise ValueError(
            f"actions must have length {AGENT_ROLE_COUNT}, got {len(actions)}"
        )
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

    `K` overrides the replication count, which otherwise is `settings.K_search`.
    `k_offset` shifts the block of replication indices used, from `[0, K)` to
    `[k_offset, k_offset + K)`.  Both exist for the re-simulation stage of
    `search.two_stage_selection`, which runs a block disjoint from the search's.
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
    # ------------------------------------------------------------------
    # RNG (sprint-level randomness only; params are fixed constants)
    # ------------------------------------------------------------------
    rng = generator_for(settings.master_seed, config, k)

    # ------------------------------------------------------------------
    # State initialisation
    # ------------------------------------------------------------------
    state = dataclasses.replace(initial_state) if initial_state is not None else SystemState()

    snapshots: list[SprintSnapshot] = []

    # ------------------------------------------------------------------
    # Sprint loop
    # ------------------------------------------------------------------
    # Work completed per role over the horizon; the agent bill is its inner product
    # with c_token.
    actions = [0.0] * AGENT_ROLE_COUNT
    # The human/agent contribution split over the same horizon (reporting only).
    impl_human = impl_agents = fixes_human = fixes_agent = 0.0

    for t in range(settings.T):
        sprint_actions = step_sprint(t, state, config, params, rng)
        for role in range(AGENT_ROLE_COUNT):
            actions[role] += sprint_actions.units[role]
        impl_human  += sprint_actions.impl_human
        impl_agents += sprint_actions.impl_agents
        fixes_human += sprint_actions.fixes_human
        fixes_agent += sprint_actions.fixes_agent

        # Optional trace capture (after all state updates)
        if capture_traces:
            snapshots.append(SprintSnapshot(
                t=t,
                F_backlog=state.F_backlog,
                F_completed=state.F_completed,
                B_kno_min=state.B_kno_min,
                B_kno_maj=state.B_kno_maj,
                B_kno_sec=state.B_kno_sec,
                B_hid_min=state.B_hid_min,
                B_hid_maj=state.B_hid_maj,
                B_hid_sec=state.B_hid_sec,
                B_escaped_min=state.B_escaped_min,
                B_escaped_maj=state.B_escaped_maj,
                B_escaped_sec=state.B_escaped_sec,
                B_rel_min=state.B_rel_min,
                B_rel_maj=state.B_rel_maj,
                B_rel_sec=state.B_rel_sec,
                D_tech=state.D_tech,
                Q_rev=state.Q_rev,
                Q_arch=state.Q_arch,
                Q_arch_impl=state.Q_arch_impl,
                Q_esc=state.Q_esc,
                N_test=state.N_test,
                N_doc=state.N_doc,
            ))

    # ------------------------------------------------------------------
    # Compute Costs
    # ------------------------------------------------------------------
    C_human, C_agents = cost_components(config, params, settings.T, actions)
    C_quality = escaped_bug_cost(
        params, state.B_escaped_min, state.B_escaped_maj, state.B_escaped_sec
    )
    C_total = C_quality + C_human + C_agents

    # ------------------------------------------------------------------
    # Compute the objective and the constraint quantity
    # ------------------------------------------------------------------
    F_done = state.F_completed
    # A horizon that completes no feature at all is reachable (a tiny r_arrival whose
    # Poisson draws all come up zero), so the denominator is floored at one.
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
