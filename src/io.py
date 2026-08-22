"""Load configuration from JSON files."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .config import (
    AGENT_ROLE_COUNT,
    SCENARIO_GROUPS,
    SearchConfiguration,
    SimulationHorizon,
    TeamConfiguration,
)
from .parameters import ScenarioParameters
from .state import SystemState


def _parse_value(key: str, value: Any) -> float:
    """Coerce a scenario parameter to a float, rejecting anything that is not a number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"Cannot parse value for {key!r}: expected a number, "
            f"got {type(value).__name__}"
        )
    return float(value)


def _flatten_scenario(scenario_dict: dict) -> dict[str, Any]:
    """Merge all nested section dicts into a single {field_name: raw_value} mapping.

    A top-level ``"description"`` string is treated as documentation and skipped;
    every other top-level value must be a JSON object.
    """
    flat: dict[str, Any] = {}
    for section, entries in scenario_dict.items():
        if section == "description" and isinstance(entries, str):
            continue
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
        K_search=int(data["K_search"]),
        K_final=int(data["K_final"]),
        master_seed=int(data["master_seed"]),
    )


def load_team_config(path: str | Path) -> TeamConfiguration:
    """Parse a team configuration JSON file and return TeamConfiguration."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return TeamConfiguration(
        N=float(data["N"]),
        A_dev=tuple(int(x) for x in data["A_dev"]),
        R=int(data["R"]),
        F_freeze=int(data.get("F_freeze", 1)),
        A_qa=tuple(int(x) for x in data.get("A_qa") or ()),
    )


def load_scenario_params(group_paths: Mapping[str, str | Path]) -> ScenarioParameters:
    """Compose ScenarioParameters from one JSON file per scenario group.

    ``group_paths`` must have exactly one entry per element of SCENARIO_GROUPS.  The
    groups partition the parameter set, so each field must be defined in exactly one
    file, and every value must be a plain number.
    """
    missing_groups = sorted(set(SCENARIO_GROUPS) - group_paths.keys())
    unknown_groups = sorted(group_paths.keys() - set(SCENARIO_GROUPS))
    if missing_groups or unknown_groups:
        raise ValueError(
            f"Expected exactly one file per scenario group {list(SCENARIO_GROUPS)}; "
            f"missing groups: {missing_groups}, unknown groups: {unknown_groups}"
        )

    flat: dict[str, Any] = {}
    defined_by: dict[str, str] = {}
    for group in SCENARIO_GROUPS:
        with open(group_paths[group], encoding="utf-8") as fh:
            data = json.load(fh)
        for key, val in _flatten_scenario(data).items():
            if key in flat:
                raise ValueError(
                    f"Parameter {key!r} is defined in both the {defined_by[key]!r} and "
                    f"{group!r} group files; the groups must partition the parameters"
                )
            flat[key] = val
            defined_by[key] = group

    # c_token is always a list — extract before building kwargs
    if "c_token" not in flat:
        raise ValueError("JSON is missing parameters: ['c_token']")
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

    kwargs = {key: _parse_value(key, raw) for key, raw in flat.items()}

    return ScenarioParameters(**kwargs, c_token=c_token)


#: The optional `search` keys, with the cast each takes.  A key the JSON omits is left
#: to the dataclass default rather than restated here, so there is one source of truth
#: for every default.
_SEARCH_FIELDS: dict = {
    "F_freeze_values": lambda v: tuple(int(x) for x in v),
    "sigma": float,
    "epsilon": float,
    "budget": int,
    "n_samples_initial": int,
    "n_generations": int,
    "n_offspring": int,
    "n_elite": int,
    "human_only": bool,
    "samples": str,
    "seed": int,
}


def load_search_config(path: str | Path) -> SearchConfiguration:
    """Parse a search configuration JSON file and return SearchConfiguration."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    s = data["search"]
    raw_scenario = data["scenario"]
    if not isinstance(raw_scenario, dict):
        raise ValueError(
            f"'scenario' must be a JSON object mapping each of the scenario groups "
            f"{list(SCENARIO_GROUPS)} to a file path, not {type(raw_scenario).__name__}"
        )
    optional = {name: cast(s[name]) for name, cast in _SEARCH_FIELDS.items() if name in s}
    return SearchConfiguration(
        scenario_paths={str(g): str(p) for g, p in raw_scenario.items()},
        horizon_path=data["horizon"],
        N_values=tuple(float(x) for x in s["N_values"]),
        R_values=tuple(int(x) for x in s["R_values"]),
        scenario_id=str(data.get("scenario_id", "")),
        **optional,
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
