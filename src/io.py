"""Load scenario configuration from a JSON file.

All scenario parameters are fixed constants; no sampling occurs.

Public API:
    load_scenario_config(path)   -> (SweepSettings, ScenarioParameters)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import AGENT_ROLE_COUNT, SweepSettings
from .parameters import ScenarioParameters


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


def _load_sweep(sweep_dict: dict) -> SweepSettings:
    return SweepSettings(
        T=int(sweep_dict["T"]),
        K=int(sweep_dict["K"]),
        master_seed=int(sweep_dict["master_seed"]),
    )


def load_scenario_config(path: str | Path) -> tuple[SweepSettings, ScenarioParameters]:
    """Parse a scenario JSON file and return sweep settings and fixed scenario parameters.

    Every value in the ``scenario`` section must be a plain number (or a legacy
    ``{"value": x}`` dict).  No distribution metadata is accepted.
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    sweep = _load_sweep(data["sweep"])
    flat = _flatten_scenario(data["scenario"])

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
    from dataclasses import fields as dc_fields
    required = {f.name for f in dc_fields(ScenarioParameters) if f.name != "c_token"}
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

    return sweep, ScenarioParameters(**kwargs, c_token=c_token)
