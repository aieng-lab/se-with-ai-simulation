"""Time-series analysis functions for simulation trace DataFrames.

All functions accept a `pd.DataFrame` as returned by `results.to_dataframe()` or
`results.load_traces()`, which has a (replication, sprint) MultiIndex and one column
per SprintSnapshot field.

Public API:
    mean_trajectory(df, field)                      -> pd.Series
    quantile_band(df, field, q_lo, q_hi)            -> pd.DataFrame
    final_distribution(df, field)                   -> pd.Series
    convergence_check(results, master_seed, floor)  -> dict
"""

from __future__ import annotations

import pandas as pd

from .selection import aggregate_config_result, best_index
from .simulation import ConfigResult


def mean_trajectory(df: pd.DataFrame, field: str) -> pd.Series:
    """Mean of `field` across replications at each sprint.

    Returns a Series indexed by sprint (0 .. T-1).
    """
    return df.groupby("sprint")[field].mean()


def quantile_band(
    df: pd.DataFrame,
    field: str,
    q_lo: float = 0.10,
    q_hi: float = 0.90,
) -> pd.DataFrame:
    """Per-sprint mean, lower and upper quantiles for `field`.

    Returns a DataFrame with columns `mean`, `lo`, `hi` indexed by sprint.
    """
    g = df.groupby("sprint")[field]
    return pd.DataFrame({
        "mean": g.mean(),
        "lo":   g.quantile(q_lo),
        "hi":   g.quantile(q_hi),
    })


def final_distribution(df: pd.DataFrame, field: str) -> pd.Series:
    """Values of `field` at the last sprint across all replications.

    Useful for characterising the spread of end-state quantities (e.g. debt at T-1).
    Returns a Series indexed by replication.
    """
    last_sprint = df.index.get_level_values("sprint").max()
    return df.xs(last_sprint, level="sprint")[field]


def convergence_check(
    results: list[ConfigResult],
    master_seed: int,
    floor: float = 0.0,
) -> dict[str, object]:
    """Is the reported optimum stable when the replication count is halved?

    `c*` is chosen by a mean over K Monte Carlo replications (§3.10), so the honest check
    on K is whether the same configuration still wins at K//2.  Aggregates every
    ConfigResult at full K and at K//2 under the throughput floor `floor`, then compares
    the two minimisers and reports how wide the winner's cost CI is relative to its mean —
    the quantity that decides how large the indistinguishable set comes out.

    Returns:
        stable          bool  — True iff the same configuration wins at both K
        best_K          int | None — index of c* at full K
        best_half_K     int | None — index of c* at K//2
        rel_ci_width_K  float — (cost_ci_hi - cost_ci_lo) / cost_mean at full K
    """
    full_aggs = [aggregate_config_result(r, master_seed) for r in results]

    # Half-K aggregated objectives (first K//2 replications only)
    half_results = [
        ConfigResult(config=r.config, replications=r.replications[: max(1, len(r.replications) // 2)])
        for r in results
    ]
    half_aggs = [aggregate_config_result(r, master_seed) for r in half_results]

    best_K = best_index(full_aggs, floor)
    best_half_K = best_index(half_aggs, floor)

    rel_ci_width = 0.0
    if best_K is not None:
        agg = full_aggs[best_K]
        if agg.cost_mean:
            rel_ci_width = (agg.cost_ci[1] - agg.cost_ci[0]) / agg.cost_mean

    return {
        "stable": best_K == best_half_K,
        "best_K": best_K,
        "best_half_K": best_half_K,
        "rel_ci_width_K": rel_ci_width,
    }
