"""Load scenario configuration from a JSON file.

Public API:
    load_scenario_config(path)   -> (SweepSettings, ScenarioPriors)
    load_mean_parameters(path)   -> (SweepSettings, ScenarioParameters)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import AGENT_ROLE_COUNT, SweepSettings
from .parameters import ScenarioParameters
from .sampling import ParameterSpec, ScenarioPriors


# Fields of ScenarioParameters that must be returned as int
_INT_FIELDS: frozenset[str] = frozenset({"Q_thr_rev", "Q_thr_esc", "Q_thr_block", "R_warmup"})


def _parse_spec(key: str, value: Any) -> ParameterSpec:
    """Convert a JSON value (scalar or distribution dict) to a ParameterSpec."""
    if isinstance(value, (int, float)):
        return ParameterSpec(dist="fixed", mean=float(value))
    if isinstance(value, dict):
        dist = value.get("dist", "fixed")
        if dist == "lognormal":
            return ParameterSpec(
                dist="lognormal",
                mean=float(value["mean"]),
                cv=float(value["cv"]),
            )
        if dist == "beta":
            return ParameterSpec(
                dist="beta",
                mean=float(value["mean"]),
                std=float(value["std"]),
            )
        if dist == "uniform":
            lo, hi = float(value["lo"]), float(value["hi"])
            return ParameterSpec(
                dist="uniform",
                mean=(lo + hi) / 2.0,
                lo=lo,
                hi=hi,
            )
        if dist == "fixed":
            return ParameterSpec(dist="fixed", mean=float(value["value"]))
        raise ValueError(f"Unknown distribution {dist!r} for parameter {key!r}")
    raise TypeError(
        f"Cannot parse parameter spec for {key!r}: expected number or dict, "
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
        N_PO=int(sweep_dict.get("N_PO", 1)),
    )


def load_scenario_config(path: str | Path) -> tuple[SweepSettings, ScenarioPriors]:
    """Parse a scenario JSON file and return sweep settings and per-field priors.

    The returned `ScenarioPriors` can be passed to `sample_scenario_parameters` inside
    the Monte Carlo replication loop.
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    sweep = _load_sweep(data["sweep"])
    flat = _flatten_scenario(data["scenario"])

    # c_token is always a fixed list — extract before building specs
    raw_c_token = flat.pop("c_token")
    if not isinstance(raw_c_token, list) or len(raw_c_token) != AGENT_ROLE_COUNT:
        raise ValueError(
            f"c_token must be a list of exactly {AGENT_ROLE_COUNT} numbers, "
            f"got {type(raw_c_token).__name__} of length {len(raw_c_token) if isinstance(raw_c_token, list) else '?'}"
        )
    c_token = tuple(float(x) for x in raw_c_token)

    specs = {key: _parse_spec(key, val) for key, val in flat.items()}

    # Validate that specs cover every non-c_token field of ScenarioParameters
    from dataclasses import fields as dc_fields
    required = {f.name for f in dc_fields(ScenarioParameters) if f.name != "c_token"}
    missing = required - specs.keys()
    extra = specs.keys() - required
    if missing:
        raise ValueError(f"JSON is missing parameters: {sorted(missing)}")
    if extra:
        raise ValueError(f"JSON contains unknown parameters: {sorted(extra)}")

    return sweep, ScenarioPriors(specs=specs, c_token=c_token)


def load_mean_parameters(path: str | Path) -> tuple[SweepSettings, ScenarioParameters]:
    """Parse a scenario JSON file and return the mean-value ScenarioParameters.

    Uses the `mean` of each distribution spec (midpoint for uniform) as the point
    estimate. Useful for the feasibility pre-filter before any Monte Carlo sampling.
    """
    sweep, priors = load_scenario_config(path)
    kwargs: dict = {}
    for key, spec in priors.specs.items():
        val = spec.mean  # midpoint stored in mean for uniform; plain mean for others
        if key in _INT_FIELDS:
            kwargs[key] = max(1, round(val))
        elif key == "D0":
            kwargs[key] = max(0.0, min(1.0, val))
        else:
            kwargs[key] = val
    return sweep, ScenarioParameters(**kwargs, c_token=priors.c_token)
