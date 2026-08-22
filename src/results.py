"""Persistence and DataFrame conversion for simulation results."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd

from .config import TeamConfiguration
from .selection import AggregatedObjectives
from .simulation import ConfigResult, ConfigTrace, ReplicationResult

_T = TypeVar("_T")

# JSON has no tuples, so these come back as lists and have to be converted on load.
_TUPLE_FIELDS = frozenset({
    "actions", "actions_mean", "cost_ci", "value_ci", "A_dev", "A_qa",
})


def _from_json(cls: "type[_T]", data: dict[str, Any]) -> _T:
    """Rebuild a dataclass from its `asdict` form, restoring the tuple-typed fields."""
    kwargs = {
        f.name: tuple(data[f.name]) if f.name in _TUPLE_FIELDS else data[f.name]
        for f in dataclasses.fields(cls)          # type: ignore[arg-type]
    }
    return cls(**kwargs)                          # type: ignore[operator]


def to_dataframe(trace: ConfigTrace) -> pd.DataFrame:
    """Convert a ConfigTrace to a DataFrame indexed by (replication, sprint).

    One row per sprint per replication, one column per SprintSnapshot field, in that
    dataclass's declaration order — which is the column order of a persisted trace CSV.
    `SprintSnapshot.t` is called `sprint` here.
    """
    rows = [
        {"replication": rep_trace.k, **dataclasses.asdict(snap)}
        for rep_trace in trace.traces
        for snap in rep_trace.snapshots
    ]
    return (pd.DataFrame(rows)
            .rename(columns={"t": "sprint"})
            .set_index(["replication", "sprint"]))


def save_result(
    result: ConfigResult,
    agg: AggregatedObjectives,
    path: str | Path,
) -> None:
    """Save ConfigResult summary and AggregatedObjectives to a JSON file."""
    data = {
        "config": dataclasses.asdict(result.config),
        "replications": [dataclasses.asdict(r) for r in result.replications],
        "aggregated": dataclasses.asdict(agg),
    }
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_result(path: str | Path) -> tuple[ConfigResult, AggregatedObjectives]:
    """Load a JSON file written by save_result."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    result = ConfigResult(
        config=_from_json(TeamConfiguration, data["config"]),
        replications=tuple(_from_json(ReplicationResult, r)
                           for r in data["replications"]),
    )
    return result, _from_json(AggregatedObjectives, data["aggregated"])


def save_traces(trace: ConfigTrace, path: str | Path) -> None:
    """Save the full per-sprint time-series trace to a CSV file."""
    to_dataframe(trace).reset_index().to_csv(path, index=False)


def load_traces(path: str | Path) -> pd.DataFrame:
    """Load a CSV written by save_traces; restores the (replication, sprint) MultiIndex."""
    return pd.read_csv(path).set_index(["replication", "sprint"])
