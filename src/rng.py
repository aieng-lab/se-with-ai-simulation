"""Reproducible random-number generation.

Every stream is derived from an integer master seed via numpy's SeedSequence and PCG-64.
A (configuration, replication) pair therefore addresses one stream by index, independently
of PYTHONHASHSEED and of how many replications are run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.random import PCG64, Generator, SeedSequence

if TYPE_CHECKING:
    from .config import TeamConfiguration


def config_hash(config: TeamConfiguration) -> int:
    """FNV-1a 64-bit hash of (N, R, F_freeze, A_dev, A_qa, f_fix, f_qa)."""
    h = 1469598103934665603      # offset basis
    prime = 1099511628211
    # N and the two shares are floats, so they are folded in at 1/100 resolution to keep
    # the mixing integer-only.
    N_key = round(config.N * 100)
    for value in (N_key, config.R, config.F_freeze, *config.A_dev, *config.A_qa,
                  round(config.f_fix * 100), round(config.f_qa * 100)):
        h = ((h ^ (value & 0xFFFFFFFFFFFFFFFF)) * prime) & 0xFFFFFFFFFFFFFFFF
    return h


def generator_for(master_seed: int, config: TeamConfiguration, k: int) -> Generator:
    """A fresh PCG-64 Generator for replication `k` of `config`.

    The configuration hash and the replication index form the spawn key, so replications
    are independent and a given configuration draws the same stream in every scenario.
    """
    return Generator(PCG64(SeedSequence(master_seed, spawn_key=(config_hash(config), k))))
