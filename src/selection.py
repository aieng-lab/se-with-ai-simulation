"""Scoring a configuration and selecting the one to report.

A configuration is scored by its mean cost per feature, and is admissible only when
`value >= (1 - eps) * r_arrival`.  `preference_key` combines the two into a total preorder,
so a candidate outside the admissible region is still ranked usefully.

The mean cost is a Monte Carlo estimate, so the reported result is a pair: the minimiser
`c*`, and the set `I*` of admissible candidates whose cost CI overlaps its own, reduced by
the parsimony tie-break.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .rng import config_hash

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .config import TeamConfiguration
    from .parameters import ScenarioParameters
    from .simulation import ConfigResult


@dataclass(frozen=True, slots=True)
class AggregatedObjectives:
    """Monte Carlo summary of one configuration over its replications.

    `cost_mean` is the objective and `value_mean` the quantity the throughput constraint
    bounds; both carry a 95 % percentile bootstrap CI.  Every other field is reporting only
    and takes no part in ranking or in `I*` membership.
    """

    value_mean: float
    value_ci: tuple[float, float]
    cost_mean: float
    cost_ci: tuple[float, float]
    # The three drivers of `cost_mean`, summing to it.
    cost_quality_mean: float = 0.0
    cost_human_mean: float = 0.0
    cost_agent_mean: float = 0.0
    # Mean work completed per role, k=1..8, each in that role's own throughput unit.
    actions_mean: tuple[float, ...] = ()
    # The human/agent split of the work.
    impl_human_mean: float = 0.0
    impl_agents_mean: float = 0.0
    fixes_human_mean: float = 0.0
    fixes_agent_mean: float = 0.0


#: Resamples drawn for the percentile bootstrap CIs of `cost` and `value`.
N_BOOTSTRAP = 1000

#: Mixed into the bootstrap seed so it cannot collide with the simulation's own streams.
_BOOTSTRAP_SALT = 0xB007_5EED


def aggregate_config_result(
    result: ConfigResult,
    master_seed: int,
) -> AggregatedObjectives:
    """Means and 95 % percentile bootstrap CIs over the replications of `result`."""
    reps = result.replications

    def mean_of(field: str) -> float:
        return float(np.mean([getattr(r, field) for r in reps]))

    value_arr = np.array([r.value for r in reps])
    cost_arr  = np.array([r.cost  for r in reps])

    K = len(value_arr)
    boot_seed = (master_seed ^ config_hash(result.config) ^ _BOOTSTRAP_SALT) & 0xFFFF_FFFF_FFFF_FFFF
    rng = np.random.default_rng(boot_seed)

    idx = rng.integers(0, K, size=(N_BOOTSTRAP, K))
    value_boot = value_arr[idx].mean(axis=1)
    cost_boot  = cost_arr[idx].mean(axis=1)

    # `actions` is optional on ReplicationResult, so it may be absent entirely.
    actions = (tuple(np.array([r.actions for r in reps], dtype=float).mean(axis=0))
               if reps and reps[0].actions else ())

    value_lo, value_hi = np.percentile(value_boot, [2.5, 97.5])
    cost_lo, cost_hi = np.percentile(cost_boot, [2.5, 97.5])

    return AggregatedObjectives(
        value_mean=float(value_arr.mean()),
        value_ci=(float(value_lo), float(value_hi)),
        cost_mean=float(cost_arr.mean()),
        cost_ci=(float(cost_lo), float(cost_hi)),
        cost_quality_mean=mean_of("cost_quality"),
        cost_human_mean=mean_of("cost_human"),
        cost_agent_mean=mean_of("cost_agent"),
        actions_mean=actions,
        impl_human_mean=mean_of("impl_human"),
        impl_agents_mean=mean_of("impl_agents"),
        fixes_human_mean=mean_of("fixes_human"),
        fixes_agent_mean=mean_of("fixes_agent"),
    )


# ---------------------------------------------------------------------------
# Throughput constraint
# ---------------------------------------------------------------------------

def throughput_floor(params: ScenarioParameters, epsilon: float) -> float:
    """The value a configuration must reach: `(1 - epsilon) * r_arrival`."""
    if not 0.0 <= epsilon < 1.0:
        raise ValueError(f"epsilon must be in [0, 1), got {epsilon}")
    return (1.0 - epsilon) * params.r_arrival


def violation(agg: AggregatedObjectives, floor: float) -> float:
    """How far short of `floor` this configuration falls; 0 when it is admissible."""
    return max(0.0, floor - agg.value_mean)


def satisfies_throughput(agg: AggregatedObjectives, floor: float) -> bool:
    """True when this configuration reaches the throughput floor."""
    return agg.value_mean >= floor


# ---------------------------------------------------------------------------
# Preference relation
# ---------------------------------------------------------------------------

def preference_key(agg: AggregatedObjectives, floor: float) -> tuple[float, float]:
    """Ranking key, smaller is better, compared lexicographically.

    Constraint violation first, cost second, so an admissible configuration (violation 0)
    precedes every violator and the remaining ties are broken on cost alone.
    """
    return (violation(agg, floor), agg.cost_mean)


def rank(aggs: Sequence[AggregatedObjectives], floor: float) -> list[int]:
    """Indices of `aggs` ordered best first, ties keeping their input order."""
    return sorted(range(len(aggs)), key=lambda i: preference_key(aggs[i], floor))


def best_index(aggs: Sequence[AggregatedObjectives], floor: float) -> int | None:
    """Index of `c*`, the best-ranked configuration; None for an empty pool.

    When no candidate satisfies the throughput constraint the least-violating one is
    returned, so a caller that reports it must check `satisfies_throughput` on the entry.
    """
    if not aggs:
        return None
    return min(range(len(aggs)), key=lambda i: preference_key(aggs[i], floor))


# ---------------------------------------------------------------------------
# Parsimony tie-break
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ParsimonyKey:
    """Comparison key for teams.

    `process` is (N, R, F_freeze, f_fix, f_qa) and must match exactly for two candidates to be
    comparable.  `agents` concatenates A_dev and A_qa, so both phases are covered.
    """

    process: tuple[float, int, int, float, float]
    agents: tuple[int, ...]


def parsimony_key(config: TeamConfiguration) -> ParsimonyKey:
    """Build the ParsimonyKey of a configuration."""
    return ParsimonyKey(
        process=(config.N, config.R, config.F_freeze, config.f_fix, config.f_qa),
        agents=tuple(config.A_dev) + tuple(config.A_qa),
    )


def parsimony_dominates(
    key_a: ParsimonyKey,
    key_b: ParsimonyKey,
    cost_a: float,
    cost_b: float,
) -> bool:
    """True when `a` runs `b`'s process, is no larger anywhere, smaller somewhere, and cheaper.

    Only meaningful between candidates already known to be statistically tied.
    Requiring `cost_a < cost_b` makes this a strict partial order.
    """
    if key_a.process != key_b.process:
        return False
    if key_a.agents == key_b.agents:
        return False
    if not cost_a < cost_b:
        return False
    return all(x <= y for x, y in zip(key_a.agents, key_b.agents))


def report_optimum(
    aggs: Sequence[AggregatedObjectives],
    floor: float,
    keys: Sequence[ParsimonyKey] | None = None,
) -> tuple[int | None, list[int]]:
    """The `(c*, I*)` pair to report: the index of `c*` and the indices of `I*`.

    A candidate belongs to `I*` when it satisfies the throughput constraint and its cost CI
    overlaps `c*`'s.  With `keys` given, the parsimony tie-break then drops any member that
    another member dominates.  `c*` itself can never be dropped: domination requires a
    strictly lower mean cost, and `c*` is the minimiser.

    `I*` is empty for an empty pool, and for one in which no candidate is admissible; in
    that case the least-violating candidate is still returned beside it, and callers must
    mark the result inadmissible with `satisfies_throughput`.
    """
    best = best_index(aggs, floor)
    if best is None:
        return None, []
    if keys is not None and len(keys) != len(aggs):
        raise ValueError(f"keys must have one entry per point: {len(keys)} != {len(aggs)}")
    if not satisfies_throughput(aggs[best], floor):
        return best, []

    lo, hi = aggs[best].cost_ci
    members = [
        i for i, agg in enumerate(aggs)
        if i == best
        or (satisfies_throughput(agg, floor)
            and agg.cost_ci[0] <= hi and lo <= agg.cost_ci[1])
    ]
    if keys is None:
        return best, members
    # Domination requires an identical process triple, so only members sharing one can
    # dominate each other.  Bucketing on it first keeps the scan quadratic in the bucket
    # rather than in the whole preliminary set, which can run to thousands of members.
    buckets: dict = {}
    for i in members:
        buckets.setdefault(keys[i].process, []).append(i)
    return best, [
        i for i in members
        if not any(
            parsimony_dominates(keys[j], keys[i], aggs[j].cost_mean, aggs[i].cost_mean)
            for j in buckets[keys[i].process] if j != i
        )
    ]
