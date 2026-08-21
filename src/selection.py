"""Scoring a configuration and selecting the one to report.

A configuration is scored by one number, the expected cost per feature, and is admissible
only if it keeps up with the work the project generates, `value >= (1 - eps) * r_arrival`.
The two are combined into the total preorder `preference_key` — Deb's feasibility rule —
so that a candidate outside the admissible region is still ranked usefully.

That preorder ranks by the *mean* cost, which is a Monte Carlo estimate, so the reported
result is the minimiser `c*` together with the indistinguishable set `I*`: the admissible
candidates whose 95 % bootstrap CI overlaps `c*`'s, reduced by the parsimony tie-break.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .rng import _config_hash

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .config import TeamConfiguration
    from .parameters import ScenarioParameters
    from .simulation import ConfigResult


@dataclass(frozen=True, slots=True)
class AggregatedObjectives:
    """Per-configuration Monte Carlo summary: the objective, the constraint, the split.

    `cost_mean` is the objective and `value_mean` the quantity the throughput constraint
    bounds; both carry a 95 % percentile bootstrap CI.  Every other field is reporting
    only and takes no part in ranking or in `I*` membership.
    """

    value_mean: float
    value_ci: tuple[float, float]
    cost_mean: float
    cost_ci: tuple[float, float]
    # The three cost drivers, summing to `cost_mean`.  Stored rather than reconstructed:
    # the quality and agent terms both depend on the realised trajectory.
    cost_quality_mean: float = 0.0
    cost_human_mean: float = 0.0
    cost_agent_mean: float = 0.0
    # Work completed per role over the horizon; `cost_agent_mean` is its inner product
    # with c_token, so this tells a role that worked from one merely deployed.
    actions_mean: tuple[float, ...] = ()
    # Who did the work; the human share appears in no `actions_mean` entry at all.
    impl_human_mean: float = 0.0
    impl_agents_mean: float = 0.0
    fixes_human_mean: float = 0.0
    fixes_agent_mean: float = 0.0


#: Resamples drawn for the percentile bootstrap CIs of `cost` and `value`.
N_BOOTSTRAP = 1000


def aggregate_config_result(
    result: "ConfigResult",
    master_seed: int,
) -> AggregatedObjectives:
    """Means and 95 % percentile bootstrap CIs over the replications of `result`.

    Only `cost` and `value` are bootstrapped.  The cost drivers and contribution counts
    take part in no comparison, so they are reported as plain means.
    """
    value_arr = np.array([r.value for r in result.replications])
    cost_arr  = np.array([r.cost  for r in result.replications])

    K = len(value_arr)
    boot_seed = (master_seed ^ _config_hash(result.config) ^ 0xB007_5EED) & 0xFFFF_FFFF_FFFF_FFFF
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
    """The value a configuration must reach: (1 - eps) * r_arrival.

    Features cannot be implemented faster than they arrive, so `r_arrival` is the expected
    maximum attainable value and `epsilon` the tolerance for the Poisson arrival noise.
    """
    if not 0.0 <= epsilon < 1.0:
        raise ValueError(f"epsilon must be in [0, 1), got {epsilon}")
    return (1.0 - epsilon) * params.r_arrival


def violation(agg: AggregatedObjectives, floor: float) -> float:
    """How far short of the throughput floor this configuration falls; 0 when admissible."""
    return max(0.0, floor - agg.value_mean)


def satisfies_throughput(agg: AggregatedObjectives, floor: float) -> bool:
    """True when the configuration delivers the work the project generates."""
    return violation(agg, floor) <= 0.0


# ---------------------------------------------------------------------------
# Preference relation (Deb's feasibility rule)
# ---------------------------------------------------------------------------

def preference_key(agg: AggregatedObjectives, floor: float) -> tuple[float, float]:
    """Ranking key, smaller is better, compared lexicographically.

    Constraint violation first, cost second.  That encodes all three of Deb's rules at
    once: two violators are separated by whose violation is smaller, an admissible
    configuration has violation 0 and precedes every violator, and two admissible ones
    are separated by cost alone.
    """
    return (violation(agg, floor), agg.cost_mean)


def rank(aggs: "Sequence[AggregatedObjectives]", floor: float) -> list[int]:
    """Indices of `aggs` ordered best first.

    A plain stable sort: the key is a total preorder, so ties keep input order and the
    result does not depend on the comparison sequence.
    """
    return sorted(range(len(aggs)), key=lambda i: preference_key(aggs[i], floor))


def best_index(aggs: "Sequence[AggregatedObjectives]", floor: float) -> int | None:
    """Index of `c*`, the best-ranked configuration; None for an empty pool.

    `c*` may violate the throughput constraint when *no* candidate satisfies it, so a
    caller that reports it must say so; `satisfies_throughput` on the entry is the check.
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


def parsimony_dominates(
    key_a: ParsimonyKey,
    key_b: ParsimonyKey,
    cost_a: float,
    cost_b: float,
) -> bool:
    """True when `a` runs `b`'s process, is no larger anywhere, smaller somewhere, and cheaper.

    Statistical indistinguishability is the caller's precondition: apply this only within
    `I*`, whose members are tied on cost by construction.  The cost condition
    `cost_a < cost_b` is checked here, which has two consequences the module relies on:
    `c*` is the cost minimiser and therefore can never be dominated, and the relation is a
    strict partial order, so one filtering pass suffices and its result is
    order-independent.
    """
    if key_a.process != key_b.process:
        return False
    if key_a.agents == key_b.agents:
        return False
    if not cost_a < cost_b:
        return False
    return all(x <= y for x, y in zip(key_a.agents, key_b.agents))


def indistinguishable_set(
    aggs: "Sequence[AggregatedObjectives]",
    floor: float,
    keys: "Sequence[ParsimonyKey] | None" = None,
) -> list[int]:
    """Indices of `I*`: the admissible candidates statistically tied with `c*`.

    A candidate belongs to `I*` when it satisfies the throughput constraint and its cost
    CI overlaps `c*`'s, i.e. the two are not significantly different at the 95 % level.

    With `keys` given, the parsimony tie-break is applied afterwards: a member that
    another member parsimony-dominates is dropped.  `c*` needs no exemption — domination
    requires a strictly lower mean cost — so it is a member of every non-empty result.

    Empty in exactly two cases: an empty pool, and a pool in which no candidate satisfies
    the throughput constraint.  In the latter there is no `c*` for a set to be built
    around; the caller still reports the least-violating candidate, marked inadmissible.
    """
    best = best_index(aggs, floor)
    if best is None:
        return []
    if keys is not None and len(keys) != len(aggs):
        raise ValueError(f"keys must have one entry per point: {len(keys)} != {len(aggs)}")
    if not satisfies_throughput(aggs[best], floor):
        # Nothing in the pool keeps up with the project, so there is no `c*` here.
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


def report_optimum(
    aggs: "Sequence[AggregatedObjectives]",
    floor: float,
    keys: "Sequence[ParsimonyKey] | None" = None,
) -> tuple[int | None, list[int]]:
    """The `(c*, I*)` pair to report.

    `c*` is the best-ranked candidate of the whole pool and `I*` is
    :func:`indistinguishable_set`.  The parsimony tie-break cannot remove `c*`, so it is
    a member of every non-empty `I*`.

    The exception is the inadmissible case: when no candidate satisfies the throughput
    constraint `I*` is empty and this returns the least-violating candidate beside it.
    Callers must mark such a result as inadmissible; `satisfies_throughput` on the
    returned entry is the check.  Both halves are None/empty only for an empty pool.
    """
    best = best_index(aggs, floor)
    if best is None:
        return None, []
    return best, indistinguishable_set(aggs, floor, keys)
