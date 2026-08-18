"""Scoring a configuration and selecting the best one (paper §3.8-3.10).

A configuration is scored by *one* number, the expected cost per feature of
Eq. (cost_rep), and is admissible only if it keeps up with the work the project
generates, `value >= (1 - eps) * r_arrival` (Eq. throughput_constraint).  Because a
candidate that violates the constraint still carries information — the search approaches
the admissible region from outside — the two are combined into the total preorder `prefers`
of Deb's feasibility rule rather than into a penalty.

That preorder ranks by the *mean* cost, which is a Monte Carlo estimate.  We therefore
report the minimiser `c*` together with the *indistinguishable set* `I*` of admissible
configurations whose 95% bootstrap CI overlaps `c*`'s (Eq. topset), reduced by the
parsimony tie-break: of two equally good teams running the same process, only the one that
uses no more agents anywhere is reported.  `c*` is exempt from that tie-break and stays in
`I*` unconditionally, so the best measured performance is never under-reported.

Nothing here is a Pareto frontier.  With a single objective the preference relation is a
total preorder, so ranking is a sort, `c*` is its first element, and none of the
non-transitivity care that a CI-aware multi-objective dominance rule needs applies.
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

    `cost_mean` is the objective of Eq. (objective) and `value_mean` the quantity the
    throughput constraint of Eq. (throughput_constraint) bounds; both carry a 95%
    percentile bootstrap CI.  Everything below them is reporting only and must stay
    invisible to `prefers` and to `I*` membership.
    """

    value_mean: float
    value_ci: tuple[float, float]
    cost_mean: float
    cost_ci: tuple[float, float]
    # The three cost drivers of §3.8, summing to `cost_mean`.  Persisted rather than
    # reconstructed: C_quality and C_agents both depend on the realised trajectory.
    cost_quality_mean: float = 0.0
    cost_human_mean: float = 0.0
    cost_agent_mean: float = 0.0
    # Work completed per role over the horizon; `cost_agent_mean` is its inner product
    # with c_token, so this tells a role that worked from one merely deployed.
    actions_mean: tuple[float, ...] = ()
    # Who did the work — the human share appears in no `actions_mean` entry at all
    # (see SprintActions).
    impl_human_mean: float = 0.0
    impl_agents_mean: float = 0.0
    fixes_human_mean: float = 0.0
    fixes_agent_mean: float = 0.0


def aggregate_config_result(
    result: "ConfigResult",
    master_seed: int,
    n_bootstrap: int = 1000,
) -> AggregatedObjectives:
    """Compute means and 95% percentile bootstrap CIs over K replications (§3.8).

    Only `cost` and `value` are bootstrapped — the objective and the constrained
    quantity.  The cost drivers and contribution counts are reported as plain means: they
    take part in no comparison, so a CI for them would never be read.
    """
    value_arr = np.array([r.value for r in result.replications])
    cost_arr  = np.array([r.cost  for r in result.replications])

    K = len(value_arr)
    boot_seed = (master_seed ^ _config_hash(result.config) ^ 0xB007_5EED) & 0xFFFF_FFFF_FFFF_FFFF
    rng = np.random.default_rng(boot_seed)

    idx = rng.integers(0, K, size=(n_bootstrap, K))
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
# Throughput constraint (§3.9, Eq. throughput_constraint)
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
# Preference relation (§3.11.3, Deb's feasibility rule)
# ---------------------------------------------------------------------------

def _preference_key(agg: AggregatedObjectives, floor: float) -> tuple[float, float]:
    """Sort key realising `prefers`: smaller is better, lexicographically.

    Constraint violation first, cost second.  This encodes all three of Deb's rules at
    once: two violators are separated by whose violation is smaller (equivalently, whose
    value is higher), an admissible configuration has violation 0 and therefore precedes
    every violator, and two admissible ones are separated by cost alone.
    """
    return (violation(agg, floor), agg.cost_mean)


def prefers(a: AggregatedObjectives, b: AggregatedObjectives, floor: float) -> bool:
    """True when `a` is preferred to `b`, i.e. `a` ≺ `b` (§3.11.3).

    Strict: two configurations with the same key are equally preferred and neither
    `prefers` the other.
    """
    return _preference_key(a, floor) < _preference_key(b, floor)


def rank(aggs: "Sequence[AggregatedObjectives]", floor: float) -> list[int]:
    """Indices of `aggs` ordered best first under `prefers`.

    A plain stable sort: `prefers` is a total preorder, so ties keep input order and the
    result does not depend on the comparison sequence.
    """
    return sorted(range(len(aggs)), key=lambda i: _preference_key(aggs[i], floor))


def best_index(aggs: "Sequence[AggregatedObjectives]", floor: float) -> int | None:
    """Index of `c*`, the `prefers`-minimal configuration; None for an empty pool.

    Note that `c*` may violate the throughput constraint when *no* candidate satisfies it.
    Callers that report it must say so — `satisfies_throughput` on the returned entry is
    the check.
    """
    if not aggs:
        return None
    return min(range(len(aggs)), key=lambda i: _preference_key(aggs[i], floor))


# ---------------------------------------------------------------------------
# Parsimony tie-break (§3.10)
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


def parsimony_dominates(key_a: ParsimonyKey, key_b: ParsimonyKey) -> bool:
    """True when `a` runs the same process as `b`, is no larger anywhere, and smaller somewhere.

    The statistical half of the §3.10 rule is the caller's precondition: this relation is
    only applied *within* `I*`, whose members are equally good by construction, so the
    additional agents of the larger team buy nothing measurable.  Antisymmetric — the
    componentwise condition cannot hold in both directions — so two candidates can never
    eliminate each other and a componentwise-minimal member of any family always survives.
    """
    if key_a.process != key_b.process:
        return False
    if key_a.agents == key_b.agents:
        return False
    return all(x <= y for x, y in zip(key_a.agents, key_b.agents))


def indistinguishable_set(
    aggs: "Sequence[AggregatedObjectives]",
    floor: float,
    keys: "Sequence[ParsimonyKey] | None" = None,
) -> list[int]:
    """Indices of `I*` (Eq. topset): admissible candidates statistically tied with `c*`.

    A candidate belongs to `I*` when it satisfies the throughput constraint and its cost CI
    overlaps `c*`'s, i.e. the two are not significantly different at the 95% level.

    With `keys` given, the parsimony tie-break of §3.10 is applied afterwards: a member
    that another member parsimony-dominates is dropped.  A single pass suffices and the
    result does not depend on ordering, because `parsimony_dominates` is antisymmetric and
    every remover is itself a member.

    `c*` itself is *exempt* from the tie-break and can never be dropped, even when a
    smaller team running the same process is statistically tied with it: §3.10 keeps it in
    so that the best measured performance is not under-reported.  `c*` is therefore always
    a member of a non-empty result, and `report_optimum` returns the raw minimiser.

    Empty in exactly two cases: an empty pool, and a pool in which *no* candidate satisfies
    the throughput constraint.  In the latter §3.10 says no `c*` exists and `I*` is empty —
    the caller still reports the least-violating candidate and marks it inadmissible
    (`report_optimum` returns its index beside the empty set).
    """
    best = best_index(aggs, floor)
    if best is None:
        return []
    if keys is not None and len(keys) != len(aggs):
        raise ValueError(f"keys must have one entry per point: {len(keys)} != {len(aggs)}")
    if not satisfies_throughput(aggs[best], floor):
        # Nothing in the pool keeps up with the project, so there is no `c*` for the set to
        # be built around (§3.10).  `report_optimum` reports the least-violating candidate.
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
        if i == best
        or not any(parsimony_dominates(keys[j], keys[i]) for j in members if j != i)
    ]


def report_optimum(
    aggs: "Sequence[AggregatedObjectives]",
    floor: float,
    keys: "Sequence[ParsimonyKey] | None" = None,
) -> tuple[int | None, list[int]]:
    """The `(c*, I*)` pair to report (§3.10).

    `c*` is the `prefers`-minimal candidate of the whole pool and `I*` is
    `indistinguishable_set`.  The parsimony tie-break never removes `c*` — §3.10 exempts it
    so that the best measured performance is not under-reported — so `c*` is a member of
    every non-empty `I*`.

    The one case where it is not is the inadmissible one: when no candidate satisfies the
    throughput constraint, §3.10 says `I*` is empty, and this returns the least-violating
    candidate (equivalently the highest-value one) beside it.  Callers must mark that
    result as inadmissible; `satisfies_throughput` on the returned entry is the check.
    Both halves are None/empty only for an empty pool.
    """
    best = best_index(aggs, floor)
    if best is None:
        return None, []
    return best, indistinguishable_set(aggs, floor, keys)
