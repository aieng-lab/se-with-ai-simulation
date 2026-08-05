"""Monte Carlo execution protocol.

Pass `capture_traces=True` to `run_configuration` to record the full per-sprint
state evolution as a `ConfigTrace`; the default returns the lightweight `ConfigResult`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from .config import AGENT_ROLE_COUNT, SimulationHorizon, TeamConfiguration
from .parameters import ScenarioParameters
from .rng import generator_for
from .sprint import step_sprint
from .state import SystemState


@dataclass(frozen=True, slots=True)
class ReplicationResult:
    value: float    # F_completed(T) / T — features/sprint
    quality: float  # weighted B_escaped(T) / max(1, F_completed(T))
    cost: float     # (C_human + C_agents) / max(1, F_completed(T)) — cost/feature


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
    result: ReplicationResult
    snapshots: tuple[SprintSnapshot, ...]   # length T (production sprints only)


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


def cost_components(
    config: TeamConfiguration,
    params: ScenarioParameters,
    T: int,
) -> tuple[float, float]:
    """(C_human, C_agents) over a T-sprint horizon — Eqs. (cost_human), (cost_agents).

    Both are deterministic given `(config, params, T)`; only the per-feature
    denominator of the cost objective is stochastic. `src.results_analysis` relies
    on this to split a persisted `cost_mean` into its human and agent shares
    without re-simulating.
    """
    C_human = config.N * params.c_salary * T / 52
    # Agent costs are split by phase: dev sprints use A_dev, freeze sprints use A_qa.
    dev_sprint_count, qa_sprint_count = phase_sprint_counts(config, T)
    # c_token is a cost per unit of role-specific throughput; Code Review and
    # Security Audit (indices 4, 5) are background services with no discrete unit
    # of output, so they keep rate 1 — a flat per-sprint subscription cost.
    agent_rates = (
        params.r_F, params.r_arch_chg, params.r_fix, params.r_refactor,
        1.0, 1.0, params.r_test, params.r_doc,
    )
    C_agents = (
        sum(config.A_dev[r] * params.c_token[r] * agent_rates[r] for r in range(AGENT_ROLE_COUNT)) * dev_sprint_count
        + sum(config.A_qa[r] * params.c_token[r] * agent_rates[r] for r in range(AGENT_ROLE_COUNT)) * qa_sprint_count  # type: ignore[index]
    )
    return C_human, C_agents


def run_configuration(
    config: TeamConfiguration,
    settings: SimulationHorizon,
    params: ScenarioParameters,
    *,
    capture_traces: bool = False,
    initial_state: SystemState | None = None,
) -> ConfigResult | ConfigTrace:
    """Run K independent replications of `config` under `settings` and `params`.

    With `capture_traces=False` (default) returns a lightweight `ConfigResult`.
    With `capture_traces=True` returns a `ConfigTrace` with full per-sprint history.
    `initial_state` sets the starting system state; defaults to all-zero if not provided.
    """
    rep_results: list[ReplicationResult] = []
    rep_traces: list[ReplicationTrace] = []

    for k in range(settings.K):
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

    snapshots: list[SprintSnapshot] = [] if capture_traces else []  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Sprint loop
    # ------------------------------------------------------------------
    for t in range(settings.T):
        step_sprint(t, state, config, params, rng)

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
    C_human, C_agents = cost_components(config, params, settings.T)
    C_total = C_human + C_agents

    # ------------------------------------------------------------------
    # Compute Pareto objectives
    # ------------------------------------------------------------------
    F_done = state.F_completed
    denom = max(1, F_done)
    Q_weighted = (params.omega_min * state.B_escaped_min
                + params.omega_maj * state.B_escaped_maj
                + params.omega_sec * state.B_escaped_sec)
    result = ReplicationResult(
        value=F_done / settings.T,
        quality=Q_weighted / denom,
        cost=C_total / denom,
    )

    trace = ReplicationTrace(k=k, result=result, snapshots=tuple(snapshots)) if capture_traces else None
    return result, trace
