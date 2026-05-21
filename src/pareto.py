"""Pareto frontier construction per FRAMEWORK.md §8.3 and §9.4.

Dominance: `c` dominates `c'` iff V_c >= V_c', Q_c <= Q_c', C_F_c <= C_F_c' with
at least one strict inequality. Statistical tie-breaking uses 95% bootstrap CIs
(§9.3): candidates whose CIs overlap on all three objectives are co-retained.
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
    V_mean: float
    Q_mean: float
    CF_mean: float
    V_ci: tuple[float, float]
    Q_ci: tuple[float, float]
    CF_ci: tuple[float, float]


def aggregate_config_result(
    result: "ConfigResult",
    master_seed: int,
    n_bootstrap: int = 1000,
) -> AggregatedObjectives:
    """Compute means and 95% percentile bootstrap CIs over K replications (§9.3).

    The bootstrap RNG is derived deterministically from master_seed and the config,
    consistent with §9.0 reproducibility requirements.
    """
    V_arr = np.array([r.V  for r in result.replications])
    Q_arr = np.array([r.Q  for r in result.replications])
    CF_arr = np.array([r.C_F for r in result.replications])

    K = len(V_arr)
    boot_seed = (master_seed ^ _config_hash(result.config) ^ 0xB007_5EED) & 0xFFFF_FFFF_FFFF_FFFF
    rng = np.random.default_rng(boot_seed)

    idx = rng.integers(0, K, size=(n_bootstrap, K))
    V_boot  = V_arr[idx].mean(axis=1)
    Q_boot  = Q_arr[idx].mean(axis=1)
    CF_boot = CF_arr[idx].mean(axis=1)

    return AggregatedObjectives(
        V_mean=float(V_arr.mean()),
        Q_mean=float(Q_arr.mean()),
        CF_mean=float(CF_arr.mean()),
        V_ci=(float(np.percentile(V_boot, 2.5)),  float(np.percentile(V_boot, 97.5))),
        Q_ci=(float(np.percentile(Q_boot, 2.5)),  float(np.percentile(Q_boot, 97.5))),
        CF_ci=(float(np.percentile(CF_boot, 2.5)), float(np.percentile(CF_boot, 97.5))),
    )


def dominates_strict(a: AggregatedObjectives, b: AggregatedObjectives) -> bool:
    """Point-estimate strict dominance (§8.3). Does not account for CI overlap."""
    return (
        a.V_mean >= b.V_mean
        and a.Q_mean <= b.Q_mean
        and a.CF_mean <= b.CF_mean
        and (a.V_mean > b.V_mean or a.Q_mean < b.Q_mean or a.CF_mean < b.CF_mean)
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
    # V (maximize): better's lower CI strictly exceeds worse's upper CI
    v_sig = better.V_ci[0] > worse.V_ci[1]
    # Q (minimize): better's upper CI strictly below worse's lower CI
    q_sig = better.Q_ci[1] < worse.Q_ci[0]
    # CF (minimize): better's upper CI strictly below worse's lower CI
    cf_sig = better.CF_ci[1] < worse.CF_ci[0]
    return v_sig or q_sig or cf_sig


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
