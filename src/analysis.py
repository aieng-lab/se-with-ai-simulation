"""Time-series helpers for simulation trace DataFrames.

Every function takes a DataFrame as returned by `results.to_dataframe` or
`results.load_traces`: a (replication, sprint) MultiIndex with one column per
SprintSnapshot field.
"""

from __future__ import annotations

import pandas as pd


def mean_trajectory(df: pd.DataFrame, field: str) -> pd.Series:
    """Mean of `field` across replications at each sprint, indexed by sprint."""
    return df.groupby("sprint")[field].mean()


def final_distribution(df: pd.DataFrame, field: str) -> pd.Series:
    """Values of `field` at the last sprint, indexed by replication."""
    last_sprint = df.index.get_level_values("sprint").max()
    return df.xs(last_sprint, level="sprint")[field]
