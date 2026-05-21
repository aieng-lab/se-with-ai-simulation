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

from .config import AGENT_ROLE_COUNT, SweepSettings, TeamConfiguration
from .pareto import AggregatedObjectives
from .simulation import ConfigResult, ConfigTrace, ReplicationResult, SprintSnapshot


# All SprintSnapshot field names except 't' (stored as 'sprint' in the DataFrame)
_SNAPSHOT_FIELDS: tuple[str, ...] = tuple(
    f.name for f in SprintSnapshot.__dataclass_fields__.values() if f.name != "t"
)


def to_dataframe(trace: ConfigTrace) -> pd.DataFrame:
    """Convert a ConfigTrace to a tidy DataFrame indexed by (replication, sprint).

    Each row corresponds to one sprint in one replication. Columns include all
    SystemState fields (end-of-sprint values) and SprintRecord fields (sprint output).
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
            "M": result.config.M,
            "A": list(result.config.A),
            "R": result.config.R,
            "F_freeze": result.config.F_freeze,
            "A_qa": list(result.config.A_qa),  # type: ignore[arg-type]
        },
        "replications": [
            {"V": r.V, "Q": r.Q, "C_F": r.C_F}
            for r in result.replications
        ],
        "aggregated": {
            "V_mean": agg.V_mean,
            "Q_mean": agg.Q_mean,
            "CF_mean": agg.CF_mean,
            "V_ci": list(agg.V_ci),
            "Q_ci": list(agg.Q_ci),
            "CF_ci": list(agg.CF_ci),
        },
    }
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_result(path: str | Path) -> tuple[ConfigResult, AggregatedObjectives]:
    """Load a JSON file written by save_result."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    cfg = data["config"]
    # Reconstruct TeamConfiguration; handle files written before F_freeze/A_qa existed.
    A = tuple(cfg["A"])
    if len(A) != AGENT_ROLE_COUNT:
        raise ValueError(f"A has length {len(A)}, expected {AGENT_ROLE_COUNT}")
    F_freeze = cfg.get("F_freeze", 1)
    A_qa = tuple(cfg["A_qa"]) if "A_qa" in cfg else None  # None → defaults to A
    config = TeamConfiguration(N=cfg["N"], M=cfg["M"], A=A, R=cfg["R"],
                               F_freeze=F_freeze, A_qa=A_qa)

    replications = tuple(
        ReplicationResult(V=r["V"], Q=r["Q"], C_F=r["C_F"])
        for r in data["replications"]
    )
    result = ConfigResult(config=config, replications=replications)

    ag = data["aggregated"]
    agg = AggregatedObjectives(
        V_mean=ag["V_mean"],
        Q_mean=ag["Q_mean"],
        CF_mean=ag["CF_mean"],
        V_ci=tuple(ag["V_ci"]),
        Q_ci=tuple(ag["Q_ci"]),
        CF_ci=tuple(ag["CF_ci"]),
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
