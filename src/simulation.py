"""Monte Carlo execution protocol per FRAMEWORK.md §9.2 (per-configuration loop).

For each (N, M, A, R) ∈ Ω, runs K independent replications. Each replication:
  1. Draw scenario parameters from priors (§9.2 Step 1)
  2. Initialize state (§2.3); optional warmup (R_warmup sprints)
  3. Step T sprints via `sprint.step_sprint`
  4. Compute Pareto objectives (§8.2): V, Q, C_F

Pass `capture_traces=True` to `run_configuration` to record the full per-sprint
state evolution as a `ConfigTrace`; the default returns the lightweight `ConfigResult`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, overload

from .config import AGENT_ROLE_COUNT, SweepSettings, TeamConfiguration
from .rng import generator_for
from .sampling import ScenarioPriors, sample_scenario_parameters
from .sprint import step_sprint
from .state import LatencyBuffer, SystemState


@dataclass(frozen=True, slots=True)
class ReplicationResult:
    V: float     # F_completed(T) / T — features/sprint
    Q: float     # B_escaped(T) / max(1, F_completed(T)) — bugs/feature
    C_F: float   # C_total / max(1, F_completed(T)) — cost/feature


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
    B_hid_min: int
    B_hid_maj: int
    B_escaped_min: int
    B_escaped_maj: int
    D: float
    Q_rev: int
    Q_esc: int
    tests_UT: float
    tests_IT: float
    tests_ST: float
    doc_cov: float
    # SprintRecord fields — what happened during this sprint
    impl: int
    fixes_min: int
    fixes_maj: int
    new_escaped_min: int
    new_escaped_maj: int


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
    priors: ScenarioPriors,
    *,
    capture_traces: Literal[False] = ...,
) -> ConfigResult: ...

@overload
def run_configuration(
    config: TeamConfiguration,
    settings: SweepSettings,
    priors: ScenarioPriors,
    *,
    capture_traces: Literal[True],
) -> ConfigTrace: ...

def run_configuration(
    config: TeamConfiguration,
    settings: SweepSettings,
    priors: ScenarioPriors,
    *,
    capture_traces: bool = False,
) -> ConfigResult | ConfigTrace:
    """Run K independent replications of `config` under `settings` and `priors`.

    With `capture_traces=False` (default) returns a lightweight `ConfigResult`.
    With `capture_traces=True` returns a `ConfigTrace` with full per-sprint history.
    """
    rep_results: list[ReplicationResult] = []
    rep_traces: list[ReplicationTrace] = []

    for k in range(settings.K):
        result, trace = _run_replication(config, settings, priors, k, capture_traces)
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
    priors: ScenarioPriors,
    k: int,
    capture_traces: bool,
) -> tuple[ReplicationResult, ReplicationTrace | None]:
    # ------------------------------------------------------------------
    # §9.2 Step 1 — RNG + scenario parameters
    # ------------------------------------------------------------------
    rng = generator_for(settings.master_seed, config, k)
    params = sample_scenario_parameters(rng, priors)

    # ------------------------------------------------------------------
    # §9.2 Step 2 — State initialisation
    # ------------------------------------------------------------------
    state = SystemState(D=params.D0)
    buffer = LatencyBuffer()

    # Optional warmup: sprint indices -(R_warmup)..-1; state carries over,
    # costs are not accumulated.
    for t_w in range(-params.R_warmup, 0):
        step_sprint(t_w, state, buffer, config, params, rng, settings.T)

    # ------------------------------------------------------------------
    # Cost accumulators (§8.1)
    # C_devops is constant per replication (setup + T × maintenance).
    # ------------------------------------------------------------------
    C_bugs = 0.0
    C_production = 0.0
    C_userdoc = 0.0
    C_devops = params.C_DevOps_setup + settings.T * params.C_DevOps_maint
    features_since_release = 0

    snapshots: list[SprintSnapshot] = [] if capture_traces else []  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # §9.2 Step 3 — Sprint loop
    # ------------------------------------------------------------------
    for t in range(settings.T):
        rec = step_sprint(t, state, buffer, config, params, rng, settings.T)

        # Per-sprint bug-fix costs
        C_bugs += (params.C_fix_minor * rec.fixes_min
                   + params.C_fix_major * rec.fixes_maj)
        features_since_release += rec.impl

        # Per-release production + user-doc costs
        if (t + 1) % config.R == 0:
            C_production += (rec.new_escaped_min * params.C_prod_min
                             + rec.new_escaped_maj * params.C_prod_maj)
            C_userdoc += params.C_user_doc * features_since_release
            features_since_release = 0

        # Optional trace capture (after all state updates)
        if capture_traces:
            snapshots.append(SprintSnapshot(
                t=t,
                F_backlog=state.F_backlog,
                F_completed=state.F_completed,
                B_kno_min=state.B_kno_min,
                B_kno_maj=state.B_kno_maj,
                B_hid_min=state.B_hid_min,
                B_hid_maj=state.B_hid_maj,
                B_escaped_min=state.B_escaped_min,
                B_escaped_maj=state.B_escaped_maj,
                D=state.D,
                Q_rev=state.Q_rev,
                Q_esc=state.Q_esc,
                tests_UT=state.tests_UT,
                tests_IT=state.tests_IT,
                tests_ST=state.tests_ST,
                doc_cov=state.doc_cov,
                impl=rec.impl,
                fixes_min=rec.fixes_min,
                fixes_maj=rec.fixes_maj,
                new_escaped_min=rec.new_escaped_min,
                new_escaped_maj=rec.new_escaped_maj,
            ))

    # ------------------------------------------------------------------
    # §8.1 Fixed costs (scenario-level, computed once after sprint loop)
    # ------------------------------------------------------------------
    C_human = (config.N * params.c_salary + settings.N_PO * params.c_PO) * settings.T / 52
    C_agents = (
        sum(config.agent_count(role + 1) * params.c_token[role]
            for role in range(AGENT_ROLE_COUNT))
        * settings.T
    )
    C_total = C_human + C_agents + C_bugs + C_production + C_devops + C_userdoc

    # ------------------------------------------------------------------
    # §9.2 Step 4 — Pareto objectives (§8.2)
    # ------------------------------------------------------------------
    F_done = state.F_completed
    denom = max(1, F_done)
    result = ReplicationResult(
        V=F_done / settings.T,
        Q=state.B_escaped / denom,
        C_F=C_total / denom,
    )

    trace = ReplicationTrace(k=k, result=result, snapshots=tuple(snapshots)) if capture_traces else None
    return result, trace
