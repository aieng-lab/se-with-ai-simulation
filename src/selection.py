"""Scoring a configuration and selecting the one to report.

A configuration is scored by its mean cost per feature and is admissible only when
`value >= (1 - eps) * r_arrival`.  `preference_key` combines the two into a total
preorder, so a candidate outside the admissible region is still ranked usefully.

Because the mean cost is a Monte Carlo estimate, the reported result is a pair: the
minimiser `c*` and the set `I*` of admissible candidates whose cost CI overlaps its own,
reduced by the parsimony tie-break.
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
    bounds; both carry a 95 % percentile bootstrap CI.  Every other field is reporting
    only and takes no part in ranking or in `I*` membership.
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
    result: "ConfigResult",
    master_seed: int,
) -> AggregatedObjectives:
    """Means and 95 % percentile bootstrap CIs over the replications of `result`.

    Only `cost` and `value` are bootstrapped; the remaining fields take part in no
    comparison and are reported as plain means.
    """
    value_arr = np.array([r.value for r in result.replications])
    cost_arr  = np.array([r.cost  for r in result.replications])

    K = len(value_arr)
    boot_seed = (master_seed ^ config_hash(result.config) ^ _BOOTSTRAP_SALT) & 0xFFFF_FFFF_FFFF_FFFF
    rng = np.random.default_rng(boot_seed)

    idx = rng.integers(0, K, size=(N_BOOTSTRAP, K))
    value_boot = value_arr[idx].mean(axis=1)
    cost_boot  = cost_arr[idx].mean(axis=1)

    reps = result.replications
    actions_arr = np.array([r.actions for r in reps], dtype=float) if reps and reps[0].actions else None

    return AggregatedObjectives(
        value_mean=float(value_arr.mean()),
        value_ci=(float(np.percentile(value_boot, 2.5)), float(np.percentile(value_boot, 97.5))),
        cost_mean=float(cost_arr.mean()),
        cost_ci=(float(np.percentile(cost_boot, 2.5)), float(np.percentile(cost_boot, 97.5))),
        cost_quality_mean=float(np.mean([r.cost_quality for r in reps])),
        cost_human_mean=float(np.mean([r.cost_human for r in reps])),
        cost_agent_mean=float(np.mean([r.cost_agent for r in reps])),
        actions_mean=tuple(actions_arr.mean(axis=0)) if actions_arr is not None else (),
        impl_human_mean=float(np.mean([r.impl_human for r in reps])),
        impl_agents_mean=float(np.mean([r.impl_agents for r in reps])),
        fixes_human_mean=float(np.mean([r.fixes_human for r in reps])),
        fixes_agent_mean=float(np.mean([r.fixes_agent for r in reps])),
    )


# ---------------------------------------------------------------------------
# Throughput constraint
# ---------------------------------------------------------------------------

def throughput_floor(params: "ScenarioParameters", epsilon: float) -> float:
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
# Preference relation (Deb's feasibility rule)
# ---------------------------------------------------------------------------

def preference_key(agg: AggregatedObjectives, floor: float) -> tuple[float, float]:
    """Ranking key, smaller is better, compared lexicographically.

    Constraint violation first, cost second, so an admissible configuration (violation 0)
    precedes every violator and the remaining ties are broken on cost alone.
    """
    return (violation(agg, floor), agg.cost_mean)


def rank(aggs: "Sequence[AggregatedObjectives]", floor: float) -> list[int]:
    """Indices of `aggs` ordered best first, ties keeping their input order."""
    return sorted(range(len(aggs)), key=lambda i: preference_key(aggs[i], floor))


def best_index(aggs: "Sequence[AggregatedObjectives]", floor: float) -> int | None:
    """Index of `c*`, the best-ranked configuration; None for an empty pool.

    `c*` violates the throughput constraint when no candidate satisfies it, so a caller
    that reports it must check `satisfies_throughput` on the entry.
    """
    if not aggs:
        return None
    return min(range(len(aggs)), key=lambda i: preference_key(aggs[i], floor))


# ---------------------------------------------------------------------------
# Parsimony tie-break
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ParsimonyKey:
    """What the parsimony tie-break compares: team size at a fixed process.

    `process` is (N, R, F_freeze) and must match exactly for two candidates to be
    comparable.  `agents` concatenates A_dev and A_qa, so both phases are covered.
    """

    process: tuple[float, int, int]
    agents: tuple[int, ...]


def parsimony_key(config: "TeamConfiguration") -> ParsimonyKey:
    """Build the ParsimonyKey of a configuration."""
    return ParsimonyKey(
        process=(config.N, config.R, config.F_freeze),
        agents=tuple(config.A_dev) + tuple(config.A_qa),
    )


def parsimony_dominates(
    key_a: ParsimonyKey,
    key_b: ParsimonyKey,
    cost_a: float,
    cost_b: float,
) -> bool:
    """True when `a` runs `b`'s process, is no larger anywhere, smaller somewhere, and cheaper.

    Only meaningful between candidates already known to be statistically tied, which is
    the caller's precondition.  Requiring `cost_a < cost_b` makes this a strict partial
    order, so a single filtering pass suffices and its result is order-independent.
    """
    if key_a.process != key_b.process:
        return False
    if key_a.agents == key_b.agents:
        return False
    if not cost_a < cost_b:
        return False
    return all(x <= y for x, y in zip(key_a.agents, key_b.agents))


def _members(
    aggs: "Sequence[AggregatedObjectives]",
    floor: float,
    best: int,
    keys: "Sequence[ParsimonyKey] | None",
) -> list[int]:
    """`I*` given a known `c*` index; empty when `c*` itself is inadmissible."""
    if not satisfies_throughput(aggs[best], floor):
        return []

    lo, hi = aggs[best].cost_ci
    members = [
        i for i, agg in enumerate(aggs)
        if i == best
        or (satisfies_throughput(agg, floor)
            and agg.cost_ci[0] <= hi and lo <= agg.cost_ci[1])
    ]
    if keys is None:
        return members
    return [
        i for i in members
        if not any(
            parsimony_dominates(keys[j], keys[i], aggs[j].cost_mean, aggs[i].cost_mean)
            for j in members if j != i
        )
    ]


def indistinguishable_set(
    aggs: "Sequence[AggregatedObjectives]",
    floor: float,
    keys: "Sequence[ParsimonyKey] | None" = None,
) -> list[int]:
    """Indices of `I*`: the admissible candidates statistically tied with `c*`.

    A candidate belongs to `I*` when it satisfies the throughput constraint and its cost
    CI overlaps `c*`'s.  With `keys` given, the parsimony tie-break then drops any member
    another member dominates; `c*` cannot be dropped, since domination requires a
    strictly lower mean cost.

    Empty for an empty pool, and for a pool in which no candidate is admissible — there
    is no `c*` for a set to form around, though one is still reported.
    """
    best = best_index(aggs, floor)
    if best is None:
        return []
    if keys is not None and len(keys) != len(aggs):
        raise ValueError(f"keys must have one entry per point: {len(keys)} != {len(aggs)}")
    return _members(aggs, floor, best, keys)


def report_optimum(
    aggs: "Sequence[AggregatedObjectives]",
    floor: float,
    keys: "Sequence[ParsimonyKey] | None" = None,
) -> tuple[int | None, list[int]]:
    """The `(c*, I*)` pair to report; see :func:`indistinguishable_set`.

    When no candidate is admissible `I*` is empty and the least-violating candidate is
    returned beside it — callers must mark such a result as inadmissible, for which
    `satisfies_throughput` on the returned entry is the check.
    """
    best = best_index(aggs, floor)
    if best is None:
        return None, []
    if keys is not None and len(keys) != len(aggs):
        raise ValueError(f"keys must have one entry per point: {len(keys)} != {len(aggs)}")
    return best, _members(aggs, floor, best, keys)
