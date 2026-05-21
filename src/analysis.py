"""Time-series analysis functions for simulation trace DataFrames.

All functions accept a `pd.DataFrame` as returned by `results.to_dataframe()` or
`results.load_traces()`, which has a (replication, sprint) MultiIndex and one column
per SprintSnapshot field.

Public API:
    mean_trajectory(df, field)               -> pd.Series
    quantile_band(df, field, q_lo, q_hi)     -> pd.DataFrame
    final_distribution(df, field)            -> pd.Series
    convergence_check(results, master_seed)  -> dict
"""

from __future__ import annotations

import pandas as pd

from .pareto import aggregate_config_result, pareto_frontier
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
) -> dict[str, object]:
    """§9.5 Pareto stability: compare frontier at K vs K//2 replications.

    For each ConfigResult, aggregate objectives at full K and at K//2, build Pareto
    frontiers for both, then compute the fraction of frontier slots that differ.

    Returns:
        stable          bool  — True iff change_fraction ≤ 0.05
        change_fraction float — fraction of frontier positions that changed
        frontier_K      list[int] — indices on full-K frontier
        frontier_half_K list[int] — indices on half-K frontier
    """
    # Full-K aggregated objectives
    full_aggs = [aggregate_config_result(r, master_seed) for r in results]

    # Half-K aggregated objectives (first K//2 replications only)
    half_results = [
        ConfigResult(config=r.config, replications=r.replications[: max(1, len(r.replications) // 2)])
        for r in results
    ]
    half_aggs = [aggregate_config_result(r, master_seed) for r in half_results]

    frontier_K = set(pareto_frontier(full_aggs))
    frontier_half_K = set(pareto_frontier(half_aggs))

    # Fraction of positions that differ between the two frontiers
    total = max(1, len(frontier_K | frontier_half_K))
    changed = len(frontier_K.symmetric_difference(frontier_half_K))
    change_fraction = changed / total

    return {
        "stable": change_fraction <= 0.05,
        "change_fraction": change_fraction,
        "frontier_K": sorted(frontier_K),
        "frontier_half_K": sorted(frontier_half_K),
    }
