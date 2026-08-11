"""Reproducible random-number generation per paper/main.tex (§3).

Every stream is derived deterministically from an integer master seed via
numpy's SeedSequence + PCG-64. Independent configurations/replications get
statistically independent sub-streams via SeedSequence.spawn_key.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.random import Generator

    from .config import TeamConfiguration


def _config_hash(config: "TeamConfiguration") -> int:
    """Stable integer hash of (N, A_dev, R, F_freeze, A_qa) independent of PYTHONHASHSEED."""
    h = 1469598103934665603  # FNV-1a 64-bit offset basis
    prime = 1099511628211
    # N is a fractional FTE staffing level; fold it in at 1/100 FTE resolution so the
    # mixing stays integer-only.  The §4.2 search range uses quarter-FTE steps.
    N_key = round(config.N * 100)
    for value in (N_key, config.R, config.F_freeze, *config.A_dev, *config.A_qa):  # type: ignore[union-attr]
        h = ((h ^ (value & 0xFFFFFFFFFFFFFFFF)) * prime) & 0xFFFFFFFFFFFFFFFF
    return h


def replication_seed(master_seed: int, config: "TeamConfiguration", k: int):
    """Derive a SeedSequence for configuration `config`, replication index `k`.

    Uses the configuration hash and replication index as the spawn_key, so identical
    configurations share a stream across scenarios (common random numbers) while
    replications stay independent.
    """
    from numpy.random import SeedSequence

    return SeedSequence(master_seed, spawn_key=(_config_hash(config), k))


def generator_for(master_seed: int, config: "TeamConfiguration", k: int) -> "Generator":
    """Return a fresh PCG-64 Generator for replication (config, k)."""
    from numpy.random import PCG64, Generator

    return Generator(PCG64(replication_seed(master_seed, config, k)))
