"""Pareto frontier construction per FRAMEWORK.md §8.3 and §9.4.

Dominance: `c` dominates `c'` iff value_c >= value_c', quality_c <= quality_c',
C_F_c <= C_F_c' with at least one strict inequality. Statistical tie-breaking
uses 95% bootstrap CIs (§9.3): candidates whose CIs overlap on all three
objectives are co-retained.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .rng import _config_hash

if TYPE_CHECKING:
    from .simulation import ConfigResult


@dataclass(frozen=True, slots=True)
class AggregatedObjectives:
    value_mean: float
    quality_mean: float
    cost_mean: float
    value_ci: tuple[float, float]
    quality_ci: tuple[float, float]
    cost_ci: tuple[float, float]


def aggregate_config_result(
    result: "ConfigResult",
    master_seed: int,
    n_bootstrap: int = 1000,
) -> AggregatedObjectives:
    """Compute means and 95% percentile bootstrap CIs over K replications (§9.3).

    The bootstrap RNG is derived deterministically from master_seed and the config,
    consistent with §9.0 reproducibility requirements.
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
    cost_boot      = cost_arr[idx].mean(axis=1)

    return AggregatedObjectives(
        value_mean=float(value_arr.mean()),
        quality_mean=float(quality_arr.mean()),
        cost_mean=float(cost_arr.mean()),
        value_ci=(float(np.percentile(value_boot, 2.5)),   float(np.percentile(value_boot, 97.5))),
        quality_ci=(float(np.percentile(quality_boot, 2.5)), float(np.percentile(quality_boot, 97.5))),
        cost_ci=(float(np.percentile(cost_boot, 2.5)),          float(np.percentile(cost_boot, 97.5))),
    )


def dominates_strict(a: AggregatedObjectives, b: AggregatedObjectives) -> bool:
    """Point-estimate strict dominance (§8.3). Does not account for CI overlap."""
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

    Per §8.3: dominance is statistically significant when the CI of the better
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


def pareto_frontier(points: list[AggregatedObjectives]) -> list[int]:
    """Return indices of non-dominated points with CI-aware tie-breaking (§8.3, §9.4).

    A candidate i is eliminated only if some j both point-dominates it
    (dominates_strict) AND is statistically significantly better
    (_ci_significantly_better). Configurations that are statistically equivalent
    (all CIs overlap) are co-retained on the frontier.
    """
    n = len(points)
    if n == 0:
        return []
    dominated = [False] * n
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[j]:
                continue
            if (dominates_strict(points[j], points[i])
                    and _ci_significantly_better(points[j], points[i])):
                dominated[i] = True
                break
    return [i for i in range(n) if not dominated[i]]
