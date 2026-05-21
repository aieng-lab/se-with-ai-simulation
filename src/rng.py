"""Reproducible random-number generation per FRAMEWORK.md §9.0.

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
    """Stable integer hash of (N, M, A, R, F_freeze, A_qa) independent of PYTHONHASHSEED."""
    h = 1469598103934665603  # FNV-1a 64-bit offset basis
    prime = 1099511628211
    for value in (config.N, config.M, config.R, config.F_freeze, *config.A, *config.A_qa):  # type: ignore[union-attr]
        h = ((h ^ (value & 0xFFFFFFFFFFFFFFFF)) * prime) & 0xFFFFFFFFFFFFFFFF
    return h


def replication_seed(master_seed: int, config: "TeamConfiguration", k: int):
    """Derive a SeedSequence for configuration `config`, replication index `k`.

    Uses the configuration hash and replication index as the spawn_key, matching
    FRAMEWORK §9.0's derivation scheme.
    """
    from numpy.random import SeedSequence

    return SeedSequence(master_seed, spawn_key=(_config_hash(config), k))


def generator_for(master_seed: int, config: "TeamConfiguration", k: int) -> "Generator":
    """Return a fresh PCG-64 Generator for replication (config, k)."""
    from numpy.random import PCG64, Generator

    return Generator(PCG64(replication_seed(master_seed, config, k)))


def sobol_seed_sequence(master_seed: int):
    """Independent stream for §9.6 Sobol sampling.

    Derived from the master seed but with a distinct spawn_key so that varying K
    in the Monte Carlo sweep does not perturb Sobol draws.
    """
    from numpy.random import SeedSequence

    return SeedSequence(master_seed, spawn_key=(0xDEAD_BEEF, 0))
