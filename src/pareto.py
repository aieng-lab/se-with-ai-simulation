"""Pareto frontier construction

Dominance: `c` dominates `c'` iff value_c >= value_c', quality_c <= quality_c',
C_F_c <= C_F_c' with at least one strict inequality. Statistical tie-breaking
uses 95% bootstrap CIs: candidates whose CIs overlap on all three
objectives are co-retained.

The resulting CI-aware relation is *not transitive* — point-dominance chains, CI
significance does not — so `pareto_frontier` compares every ordered pair rather than
pruning eliminated candidates out of the dominator pool.  See its docstring.

`pareto_frontier` also implements the parsimony tie-break of §3.10 when given per-point
`ParsimonyKey`s: among statistically indistinguishable configurations running the same
process, only teams that are componentwise minimal in their agent counts are reported,
since the larger ones buy no measurable improvement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .rng import _config_hash

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .config import TeamConfiguration
    from .simulation import ConfigResult


@dataclass(frozen=True, slots=True)
class AggregatedObjectives:
    value_mean: float
    quality_mean: float
    cost_mean: float
    value_ci: tuple[float, float]
    quality_ci: tuple[float, float]
    cost_ci: tuple[float, float]
    # Reporting only — the §3.10 dominance relation stays three-objective and must not
    # read these.  They are persisted because C_agents depends on the realised
    # trajectory and can no longer be reconstructed from `cost_mean` by a fixed ratio.
    cost_human_mean: float = 0.0
    cost_agent_mean: float = 0.0
    actions_mean: tuple[float, ...] = ()


def aggregate_config_result(
    result: "ConfigResult",
    master_seed: int,
    n_bootstrap: int = 1000,
) -> AggregatedObjectives:
    """Compute means and 95% percentile bootstrap CIs over K replications.
    """
    value_arr   = np.array([r.value   for r in result.replications])
    quality_arr = np.array([r.quality for r in result.replications])
    cost_arr      = np.array([r.cost    for r in result.replications])

    K = len(value_arr)
    boot_seed = (master_seed ^ _config_hash(result.config) ^ 0xB007_5EED) & 0xFFFF_FFFF_FFFF_FFFF
    rng = np.random.default_rng(boot_seed)

    idx = rng.integers(0, K, size=(n_bootstrap, K))
    value_boot   = value_arr[idx].mean(axis=1)
    quality_boot = quality_arr[idx].mean(axis=1)
    cost_boot    = cost_arr[idx].mean(axis=1)

    reps = result.replications
    actions_arr = np.array([r.actions for r in reps], dtype=float) if reps and reps[0].actions else None

    return AggregatedObjectives(
        value_mean=float(value_arr.mean()),
        quality_mean=float(quality_arr.mean()),
        cost_mean=float(cost_arr.mean()),
        cost_human_mean=float(np.mean([r.cost_human for r in reps])),
        cost_agent_mean=float(np.mean([r.cost_agent for r in reps])),
        actions_mean=tuple(actions_arr.mean(axis=0)) if actions_arr is not None else (),
        value_ci=(float(np.percentile(value_boot,     2.5)), float(np.percentile(value_boot,   97.5))),
        quality_ci=(float(np.percentile(quality_boot, 2.5)), float(np.percentile(quality_boot, 97.5))),
        cost_ci=(float(np.percentile(cost_boot,       2.5)), float(np.percentile(cost_boot,    97.5))),
    )


def dominates_strict(a: AggregatedObjectives, b: AggregatedObjectives) -> bool:
    """Point-estimate strict dominance. Does not account for CI overlap."""
    return (
        a.value_mean >= b.value_mean
        and a.quality_mean <= b.quality_mean
        and a.cost_mean <= b.cost_mean
        and (a.value_mean > b.value_mean or a.quality_mean < b.quality_mean or a.cost_mean < b.cost_mean)
    )


def _ci_significantly_better(
    better: AggregatedObjectives,
    worse: AggregatedObjectives,
) -> bool:
    """True if at least one objective has non-overlapping CIs in the dominating direction.

    Dominance is statistically significant when the CI of the better
    configuration does not overlap with the CI of the worse configuration on at
    least one objective.
    """
    # value (maximize): better's lower CI strictly exceeds worse's upper CI
    value_sig = better.value_ci[0] > worse.value_ci[1]
    # quality (minimize): better's upper CI strictly below worse's lower CI
    quality_sig = better.quality_ci[1] < worse.quality_ci[0]
    # CF (minimize): better's upper CI strictly below worse's lower CI
    cf_sig = better.cost_ci[1] < worse.cost_ci[0]
    return value_sig or quality_sig or cf_sig


@dataclass(frozen=True, slots=True)
class ParsimonyKey:
    """What the parsimony tie-break of §3.10 compares: team size at a fixed process.

    `process` is (N, R, F_freeze) and must match exactly for two candidates to be
    comparable — without that, the rule could prefer a team with fewer agents but *more*
    humans, which is more expensive rather than more parsimonious.  `agents` is the
    concatenation of A_dev and A_qa, so the comparison covers both phases.
    """

    process: tuple[float, int, int]
    agents: tuple[int, ...]


def parsimony_key(config: "TeamConfiguration") -> ParsimonyKey:
    """Build the ParsimonyKey of a configuration."""
    return ParsimonyKey(
        process=(config.N, config.R, config.F_freeze),
        agents=tuple(config.A_dev) + tuple(config.A_qa),  # type: ignore[arg-type]
    )


def _statistically_tied(a: AggregatedObjectives, b: AggregatedObjectives) -> bool:
    """True when the CIs of `a` and `b` overlap on all three objectives.

    Equivalently: neither is CI-significantly better than the other in either direction,
    so §3.10 has no grounds to separate them.
    """
    return not _ci_significantly_better(a, b) and not _ci_significantly_better(b, a)


def parsimony_dominates(
    key_a: ParsimonyKey, key_b: ParsimonyKey,
    a: AggregatedObjectives, b: AggregatedObjectives,
) -> bool:
    """True when `a` is the same process, no larger anywhere, smaller somewhere, and tied.

    The three conditions of §3.10's parsimony tie-break.  Antisymmetric by construction:
    the componentwise condition cannot hold in both directions, so two candidates can
    never eliminate each other and a componentwise-minimal member of any tied family
    always survives.
    """
    if key_a.process != key_b.process:
        return False
    if key_a.agents == key_b.agents:
        return False
    if any(x > y for x, y in zip(key_a.agents, key_b.agents)):
        return False
    return _statistically_tied(a, b)


def _parsimony_prune(
    points: list[AggregatedObjectives],
    keys: "Sequence[ParsimonyKey]",
    frontier: list[int],
) -> list[int]:
    """Drop frontier members that are parsimony-dominated by another frontier member.

    Deliberately restricted to the frontier rather than the whole candidate pool: the
    CI-aware relation is not transitive, so a pool-wide pass would let an already
    eliminated candidate remove a frontier point.  Every removal here is justified by a
    point that itself survived §3.10.  Single pass, so the result does not depend on
    ordering.
    """
    return [
        i for i in frontier
        if not any(
            parsimony_dominates(keys[j], keys[i], points[j], points[i])
            for j in frontier if j != i
        )
    ]


def pareto_frontier(
    points: list[AggregatedObjectives],
    ci_aware: bool = True,
    keys: "Sequence[ParsimonyKey] | None" = None,
) -> list[int]:
    """Return indices of non-dominated points.

    When ci_aware=True (default): a candidate i is eliminated only if some j
    both point-dominates it (dominates_strict) AND is statistically significantly
    better (_ci_significantly_better). Configurations whose CIs overlap on all
    objectives are co-retained on the frontier.

    When ci_aware=False: standard point-estimate Pareto dominance — i is
    eliminated whenever some j strictly dominates it on mean values alone.

    Every candidate is tested against every other, including candidates that have
    themselves already been eliminated.  That is deliberate and must not be "optimised"
    away: the usual pruning (skip j once it is dominated) is only sound because ordinary
    Pareto dominance is transitive, and the CI-aware relation is not.  `k` can CI-dominate
    `j` and `j` CI-dominate `i` while `k` CI-dominates `i` on no objective at all, because
    point-dominance chains but CI significance does not — see
    tests/test_pareto.py::TestParetoFrontierNonTransitive.  Skipping an eliminated `j`
    would then leave `i` on the frontier although §3.10 excludes it, and would make the
    result depend on the order in which candidates were evaluated.  The cost of the full
    comparison is ~1.7 s at the §4.2 budget of 10,000 candidates (measured).

    `keys` optionally enables the parsimony tie-break of §3.10: given one ParsimonyKey per
    point, the frontier is reduced afterwards so that of several statistically
    indistinguishable teams running the same process, only the smallest is reported.  With
    `keys=None` (the default) the result is exactly the CI-aware frontier and nothing about
    the behaviour above changes.
    """
    n = len(points)
    if n == 0:
        return []
    if keys is not None and len(keys) != n:
        raise ValueError(f"keys must have one entry per point: {len(keys)} != {n}")
    dominated = [False] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates_strict(points[j], points[i]):
                if not ci_aware or _ci_significantly_better(points[j], points[i]):
                    dominated[i] = True
                    break
    frontier = [i for i in range(n) if not dominated[i]]
    if keys is not None:
        frontier = _parsimony_prune(points, keys, frontier)
    return frontier
