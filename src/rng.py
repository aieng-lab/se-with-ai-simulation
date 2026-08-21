"""Reproducible random-number generation.

Every stream is derived deterministically from an integer master seed via numpy's
SeedSequence + PCG-64, so a (configuration, replication) pair is addressable by index
and independent of PYTHONHASHSEED or of how many replications are run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.random import PCG64, Generator, SeedSequence

if TYPE_CHECKING:
    from .config import TeamConfiguration


def config_hash(config: "TeamConfiguration") -> int:
    """FNV-1a 64-bit hash of (N, R, F_freeze, A_dev, A_qa)."""
    h = 1469598103934665603      # offset basis
    prime = 1099511628211
    # N is a float, so it is folded in at 1/100 FTE resolution to keep the mixing
    # integer-only.
    N_key = round(config.N * 100)
    for value in (N_key, config.R, config.F_freeze, *config.A_dev, *config.A_qa):
        h = ((h ^ (value & 0xFFFFFFFFFFFFFFFF)) * prime) & 0xFFFFFFFFFFFFFFFF
    return h


def generator_for(master_seed: int, config: "TeamConfiguration", k: int) -> Generator:
    """A fresh PCG-64 Generator for replication `k` of `config`.

    The configuration hash and the replication index form the spawn key, so identical
    configurations share a stream across scenarios while replications stay independent.
    """
    return Generator(PCG64(SeedSequence(master_seed, spawn_key=(config_hash(config), k))))
