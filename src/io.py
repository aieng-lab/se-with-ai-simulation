"""Load configuration from JSON files.

All scenario parameters are fixed constants; no sampling occurs.

Public API:
    load_simulation_horizon(path)    -> SimulationHorizon
    load_team_config(path)     -> TeamConfiguration
    load_scenario_params(path) -> ScenarioParameters
    load_initial_state(path)   -> SystemState
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from .config import AGENT_ROLE_COUNT, SearchConfiguration, SimulationHorizon, TeamConfiguration
from .parameters import ScenarioParameters
from .state import SystemState


# Fields of ScenarioParameters that must be stored as int
_INT_FIELDS: frozenset[str] = frozenset()


def _parse_value(key: str, value: Any) -> float:
    """Extract a plain float from a JSON value (scalar or legacy fixed-dict)."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        # Legacy {"dist": "fixed", "value": x} or {"value": x}
        if "value" in value:
            return float(value["value"])
        raise ValueError(
            f"Parameter {key!r}: dict must have a 'value' key, got {list(value.keys())}"
        )
    raise TypeError(
        f"Cannot parse value for {key!r}: expected number or dict, "
        f"got {type(value).__name__}"
    )


def _flatten_scenario(scenario_dict: dict) -> dict[str, Any]:
    """Merge all nested section dicts into a single {field_name: raw_value} mapping."""
    flat: dict[str, Any] = {}
    for section, entries in scenario_dict.items():
        if not isinstance(entries, dict):
            raise ValueError(
                f"Scenario section {section!r} must be a JSON object, not {type(entries).__name__}"
            )
        for key, val in entries.items():
            if key in flat:
                raise ValueError(
                    f"Duplicate parameter key {key!r} found in scenario section {section!r}"
                )
            flat[key] = val
    return flat


def load_simulation_horizon(path: str | Path) -> SimulationHorizon:
    """Parse a simulation horizon JSON file and return SimulationHorizon."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return SimulationHorizon(
        T=int(data["T"]),
        K=int(data["K"]),
        master_seed=int(data["master_seed"]),
    )


def load_team_config(path: str | Path) -> TeamConfiguration:
    """Parse a team configuration JSON file and return TeamConfiguration."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    A_dev = tuple(int(x) for x in data["A_dev"])
    A_qa_raw = data.get("A_qa")
    A_qa = tuple(int(x) for x in A_qa_raw) if A_qa_raw is not None else None
    return TeamConfiguration(
        N=int(data["N"]),
        M=int(data["M"]),
        A_dev=A_dev,
        R=int(data["R"]),
        F_freeze=int(data.get("F_freeze", 1)),
        A_qa=A_qa,
    )


def load_scenario_params(path: str | Path) -> ScenarioParameters:
    """Parse a scenario parameters JSON file and return ScenarioParameters.

    Every value must be a plain number (or a legacy ``{"value": x}`` dict).
    No distribution metadata is accepted.
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    flat = _flatten_scenario(data)

    # c_token is always a list — extract before building kwargs
    raw_c_token = flat.pop("c_token")
    if not isinstance(raw_c_token, list) or len(raw_c_token) != AGENT_ROLE_COUNT:
        raise ValueError(
            f"c_token must be a list of exactly {AGENT_ROLE_COUNT} numbers, "
            f"got {type(raw_c_token).__name__} of length "
            f"{len(raw_c_token) if isinstance(raw_c_token, list) else '?'}"
        )
    c_token = tuple(float(x) for x in raw_c_token)

    # Validate coverage
    required = {f.name for f in dataclasses.fields(ScenarioParameters) if f.name != "c_token"}
    missing = required - flat.keys()
    extra = flat.keys() - required
    if missing:
        raise ValueError(f"JSON is missing parameters: {sorted(missing)}")
    if extra:
        raise ValueError(f"JSON contains unknown parameters: {sorted(extra)}")

    kwargs: dict = {}
    for key, raw in flat.items():
        val = _parse_value(key, raw)
        if key in _INT_FIELDS:
            kwargs[key] = max(1, round(val))
        else:
            kwargs[key] = val

    return ScenarioParameters(**kwargs, c_token=c_token)


def load_search_config(path: str | Path) -> SearchConfiguration:
    """Parse a search configuration JSON file and return SearchConfiguration."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    s = data["search"]
    return SearchConfiguration(
        scenario_path=data["scenario"],
        horizon_path=data["horizon"],
        strategy=s["strategy"],
        N_values=tuple(int(x) for x in s["N_values"]),
        M_max=int(s["M_max"]),
        R_values=tuple(int(x) for x in s["R_values"]),
        F_freeze_values=tuple(int(x) for x in s.get("F_freeze_values", [1])),
        n_samples=int(s.get("n_samples", 500)),
        n_samples_initial=int(s.get("n_samples_initial", 100)),
        n_generations=int(s.get("n_generations", 15)),
        n_offspring=int(s.get("n_offspring", 20)),
        seed=int(s.get("seed", 0)),
    )


def load_initial_state(path: str | Path) -> SystemState:
    """Load initial SystemState values from a flat JSON file.

    Only the fields listed in the JSON are set; all others default to 0/0.0.
    Unknown field names raise ValueError.
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    valid = {f.name for f in dataclasses.fields(SystemState)}
    unknown = set(data) - valid
    if unknown:
        raise ValueError(f"Unknown SystemState fields in {path}: {sorted(unknown)}")
    kwargs: dict = {}
    for f in dataclasses.fields(SystemState):
        if f.name in data:
            coerce = type(f.default)  # int for int fields, float for float fields
            kwargs[f.name] = coerce(data[f.name])
    return SystemState(**kwargs)
