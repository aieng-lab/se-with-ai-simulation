"""Persistence and DataFrame conversion for simulation results.

Public API:
    to_dataframe(trace)          -> pd.DataFrame   (K × T rows, MultiIndex)
    save_result(result, agg, path)                  JSON summary
    load_result(path)            -> (ConfigResult, AggregatedObjectives)
    save_traces(trace, path)                        CSV of full time series
    load_traces(path)            -> pd.DataFrame
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import AGENT_ROLE_COUNT, SimulationHorizon, TeamConfiguration
from .pareto import AggregatedObjectives
from .simulation import ConfigResult, ConfigTrace, ReplicationResult, SprintSnapshot


# All SprintSnapshot field names except 't' (stored as 'sprint' in the DataFrame)
_SNAPSHOT_FIELDS: tuple[str, ...] = tuple(
    f.name for f in SprintSnapshot.__dataclass_fields__.values() if f.name != "t"
)


def to_dataframe(trace: ConfigTrace) -> pd.DataFrame:
    """Convert a ConfigTrace to a tidy DataFrame indexed by (replication, sprint).

    Each row corresponds to one sprint in one replication. Columns include all
    SystemState fields (end-of-sprint values).
    """
    rows = []
    for rep_trace in trace.traces:
        for snap in rep_trace.snapshots:
            row = {"replication": rep_trace.k, "sprint": snap.t}
            for field in _SNAPSHOT_FIELDS:
                row[field] = getattr(snap, field)
            rows.append(row)
    return pd.DataFrame(rows).set_index(["replication", "sprint"])


def save_result(
    result: ConfigResult,
    agg: AggregatedObjectives,
    path: str | Path,
) -> None:
    """Save ConfigResult summary and AggregatedObjectives to a JSON file."""
    data = {
        "config": {
            "N": result.config.N,
            "A_dev": list(result.config.A_dev),
            "R": result.config.R,
            "F_freeze": result.config.F_freeze,
            "A_qa": list(result.config.A_qa),  # type: ignore[arg-type]
        },
        "replications": [
            {"value": r.value, "quality": r.quality, "cost": r.cost,
             "cost_human": r.cost_human, "cost_agent": r.cost_agent,
             "actions": list(r.actions)}
            for r in result.replications
        ],
        "aggregated": {
            "value_mean": agg.value_mean,
            "quality_mean": agg.quality_mean,
            "cost_mean": agg.cost_mean,
            "value_ci": list(agg.value_ci),
            "quality_ci": list(agg.quality_ci),
            "cost_ci": list(agg.cost_ci),
            "cost_human_mean": agg.cost_human_mean,
            "cost_agent_mean": agg.cost_agent_mean,
            "actions_mean": list(agg.actions_mean),
        },
    }
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_result(path: str | Path) -> tuple[ConfigResult, AggregatedObjectives]:
    """Load a JSON file written by save_result."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    cfg = data["config"]
    # Reconstruct TeamConfiguration; handle files written before F_freeze/A_qa existed.
    A_dev = tuple(cfg["A_dev"])
    if len(A_dev) != AGENT_ROLE_COUNT:
        raise ValueError(f"A_dev has length {len(A_dev)}, expected {AGENT_ROLE_COUNT}")
    F_freeze = cfg.get("F_freeze", 1)
    A_qa = tuple(cfg["A_qa"]) if "A_qa" in cfg else None  # None → defaults to A_dev
    config = TeamConfiguration(N=cfg["N"], A_dev=A_dev, R=cfg["R"],
                               F_freeze=F_freeze, A_qa=A_qa)

    # The cost split and the action counts postdate the usage-based cost model; files
    # written before it carry neither, so both default to empty rather than failing.
    replications = tuple(
        ReplicationResult(value=r["value"], quality=r["quality"], cost=r["cost"],
                          cost_human=r.get("cost_human", 0.0),
                          cost_agent=r.get("cost_agent", 0.0),
                          actions=tuple(r.get("actions", ())))
        for r in data["replications"]
    )
    result = ConfigResult(config=config, replications=replications)

    ag = data["aggregated"]
    agg = AggregatedObjectives(
        value_mean=ag["value_mean"],
        quality_mean=ag["quality_mean"],
        cost_mean=ag["cost_mean"],
        value_ci=tuple(ag["value_ci"]),
        quality_ci=tuple(ag["quality_ci"]),
        cost_ci=tuple(ag["cost_ci"]),
        cost_human_mean=ag.get("cost_human_mean", 0.0),
        cost_agent_mean=ag.get("cost_agent_mean", 0.0),
        actions_mean=tuple(ag.get("actions_mean", ())),
    )
    return result, agg


def save_traces(trace: ConfigTrace, path: str | Path) -> None:
    """Save the full per-sprint time-series trace to a CSV file."""
    df = to_dataframe(trace)
    df.reset_index().to_csv(path, index=False)


def load_traces(path: str | Path) -> pd.DataFrame:
    """Load a CSV written by save_traces; restores the (replication, sprint) MultiIndex."""
    df = pd.read_csv(path)
    return df.set_index(["replication", "sprint"])
