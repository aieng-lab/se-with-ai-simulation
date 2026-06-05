"""Monte Carlo execution protocol per FRAMEWORK.md §9.2 (per-configuration loop).

For each (N, M, A, R) ∈ Ω, runs K independent replications. Each replication:
  1. Initialize state S(0) to zero (all components fixed at their initial values)
  2. Step T sprints via `sprint.step_sprint` (all scenario parameters are fixed constants)
  3. Compute Pareto objectives (§8.2): V, Q, C_F

Pass `capture_traces=True` to `run_configuration` to record the full per-sprint
state evolution as a `ConfigTrace`; the default returns the lightweight `ConfigResult`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, overload

from .config import AGENT_ROLE_COUNT, SweepSettings, TeamConfiguration
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
    # Aggregated means + 95% bootstrap CIs are computed by src.pareto.


@dataclass(frozen=True, slots=True)
class SprintSnapshot:
    """Full system state captured at the end of production sprint t (§9.2 Step 5)."""

    t: int
    # SystemState fields — end-of-sprint values after all §4 updates
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
    D: float
    Q_rev: int
    Q_arch: int
    Q_arch_impl: int
    Q_esc: int
    tests: float
    N_doc: float
    # SprintRecord fields — what happened during this sprint
    impl: int
    fixes_min: int
    fixes_maj: int
    fixes_sec: int
    rel_fixes_min: int
    rel_fixes_maj: int
    rel_fixes_sec: int
    h_fixes_min: int
    h_fixes_maj: int
    h_fixes_sec: int
    h_rel_fixes_min: int
    h_rel_fixes_maj: int
    h_rel_fixes_sec: int
    new_escaped_min: int
    new_escaped_maj: int
    new_escaped_sec: int


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


@overload
def run_configuration(
    config: TeamConfiguration,
    settings: SweepSettings,
    params: ScenarioParameters,
    *,
    capture_traces: Literal[False] = ...,
) -> ConfigResult: ...

@overload
def run_configuration(
    config: TeamConfiguration,
    settings: SweepSettings,
    params: ScenarioParameters,
    *,
    capture_traces: Literal[True],
) -> ConfigTrace: ...

def run_configuration(
    config: TeamConfiguration,
    settings: SweepSettings,
    params: ScenarioParameters,
    *,
    capture_traces: bool = False,
) -> ConfigResult | ConfigTrace:
    """Run K independent replications of `config` under `settings` and `params`.

    With `capture_traces=False` (default) returns a lightweight `ConfigResult`.
    With `capture_traces=True` returns a `ConfigTrace` with full per-sprint history.
    """
    rep_results: list[ReplicationResult] = []
    rep_traces: list[ReplicationTrace] = []

    for k in range(settings.K):
        result, trace = _run_replication(config, settings, params, k, capture_traces)
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
    settings: SweepSettings,
    params: ScenarioParameters,
    k: int,
    capture_traces: bool,
) -> tuple[ReplicationResult, ReplicationTrace | None]:
    # ------------------------------------------------------------------
    # §9.2 Step 1 — RNG (sprint-level randomness only; params are fixed constants)
    # ------------------------------------------------------------------
    rng = generator_for(settings.master_seed, config, k)

    # ------------------------------------------------------------------
    # §9.2 Step 2 — State initialisation
    # ------------------------------------------------------------------
    state = SystemState()

    snapshots: list[SprintSnapshot] = [] if capture_traces else []  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # §9.2 Step 3 — Sprint loop
    # ------------------------------------------------------------------
    for t in range(settings.T):
        rec = step_sprint(t, state, config, params, rng)

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
                D=state.D,
                Q_rev=state.Q_rev,
                Q_arch=state.Q_arch,
                Q_arch_impl=state.Q_arch_impl,
                Q_esc=state.Q_esc,
                tests=state.tests,
                N_doc=state.N_doc,
                impl=rec.impl,
                fixes_min=rec.fixes_min,
                fixes_maj=rec.fixes_maj,
                fixes_sec=rec.fixes_sec,
                rel_fixes_min=rec.rel_fixes_min,
                rel_fixes_maj=rec.rel_fixes_maj,
                rel_fixes_sec=rec.rel_fixes_sec,
                h_fixes_min=rec.h_fixes_min,
                h_fixes_maj=rec.h_fixes_maj,
                h_fixes_sec=rec.h_fixes_sec,
                h_rel_fixes_min=rec.h_rel_fixes_min,
                h_rel_fixes_maj=rec.h_rel_fixes_maj,
                h_rel_fixes_sec=rec.h_rel_fixes_sec,
                new_escaped_min=rec.new_escaped_min,
                new_escaped_maj=rec.new_escaped_maj,
                new_escaped_sec=rec.new_escaped_sec,
            ))

    # ------------------------------------------------------------------
    # §8.1 Fixed costs (scenario-level, computed once after sprint loop)
    # ------------------------------------------------------------------
    C_human = config.N * params.c_salary * settings.T / 52
    # Agent costs are split by phase: dev sprints use A, freeze sprints use A_qa.
    qa_sprint_count = sum(
        1 for t in range(settings.T)
        if (t % config.R) >= (config.R - config.F_freeze)
    )
    dev_sprint_count = settings.T - qa_sprint_count
    C_agents = (
        sum(config.A[r] * params.c_token[r] for r in range(AGENT_ROLE_COUNT)) * dev_sprint_count
        + sum(config.A_qa[r] * params.c_token[r] for r in range(AGENT_ROLE_COUNT)) * qa_sprint_count  # type: ignore[index]
    )
    C_total = C_human + C_agents

    # ------------------------------------------------------------------
    # §9.2 Step 4 — Pareto objectives (§8.2)
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
