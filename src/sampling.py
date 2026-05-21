"""Scenario parameter sampling per FRAMEWORK.md §9.2 Step 1.

`ScenarioPriors` holds one `ParameterSpec` per field of `ScenarioParameters`.
`sample_scenario_parameters` draws a fresh `ScenarioParameters` for each Monte Carlo
replication using the provided numpy Generator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields as dataclass_fields
from typing import TYPE_CHECKING

from .parameters import ScenarioParameters

if TYPE_CHECKING:
    from numpy.random import Generator


_INT_FIELDS: frozenset[str] = frozenset({"Q_thr_rev", "Q_thr_esc", "Q_thr_block", "R_warmup"})


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """Distribution specification for one scenario parameter.

    dist    "fixed"     → always return mean
            "lognormal" → exp(N(μ, σ)) parameterised by mean and cv
            "beta"      → Beta(α, β) parameterised by mean and std
            "uniform"   → U(lo, hi); mean stores the midpoint for load_mean_parameters
    """

    dist: str
    mean: float = 0.0
    cv: float = 0.0    # lognormal: coefficient of variation (std/mean)
    std: float = 0.0   # beta: standard deviation in natural units
    lo: float = 0.0    # uniform: lower bound
    hi: float = 1.0    # uniform: upper bound

    def sample(self, rng: "Generator") -> float:
        """Draw one value from this distribution using `rng`."""
        if self.dist == "fixed":
            return self.mean
        if self.dist == "lognormal":
            # E[Y] = mean, CV[Y] = cv  →  σ² = log(1+cv²),  μ = log(mean) − σ²/2
            sigma_sq = math.log(1.0 + self.cv ** 2)
            mu = math.log(self.mean) - sigma_sq / 2.0
            return float(rng.lognormal(mu, math.sqrt(sigma_sq)))
        if self.dist == "beta":
            # Method of moments: k = mean*(1-mean)/std² - 1; α = mean*k, β = (1-mean)*k
            variance = self.std ** 2
            k = self.mean * (1.0 - self.mean) / variance - 1.0
            if k <= 0.0:
                raise ValueError(
                    f"Beta std={self.std} too large for mean={self.mean}; "
                    f"requires std < {math.sqrt(self.mean * (1.0 - self.mean)):.4f}"
                )
            return float(rng.beta(self.mean * k, (1.0 - self.mean) * k))
        if self.dist == "uniform":
            return float(rng.uniform(self.lo, self.hi))
        raise ValueError(f"Unknown distribution: {self.dist!r}")


@dataclass(frozen=True)
class ScenarioPriors:
    """Per-field distribution specs for one scenario configuration.

    `specs` maps every ScenarioParameters field name (except c_token) to a ParameterSpec.
    `c_token` is always a fixed tuple of 13 floats (token costs do not vary per replication).
    """

    specs: dict[str, ParameterSpec]
    c_token: tuple[float, ...]


def sample_scenario_parameters(
    rng: "Generator",
    priors: ScenarioPriors,
) -> ScenarioParameters:
    """Draw one ScenarioParameters from priors (FRAMEWORK.md §9.2 Step 1)."""
    drawn: dict = {}
    for f in dataclass_fields(ScenarioParameters):
        name = f.name
        if name == "c_token":
            continue
        spec = priors.specs[name]
        value: float = spec.sample(rng)
        if name in _INT_FIELDS:
            drawn[name] = max(1, round(value))
        elif name == "D0":
            drawn[name] = max(0.0, min(1.0, value))
        else:
            drawn[name] = value
    return ScenarioParameters(**drawn, c_token=priors.c_token)
