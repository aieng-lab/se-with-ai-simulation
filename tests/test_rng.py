"""Tests for src/rng.py — hash stability and generator reproducibility."""

import numpy as np

from src.rng import _config_hash, generator_for, replication_seed


class TestConfigHash:
    def test_deterministic(self, base_config):
        assert _config_hash(base_config) == _config_hash(base_config)

    def test_differs_on_N(self, base_config):
        from src.config import TeamConfiguration
        cfg2 = TeamConfiguration(N=3, M=2, A=base_config.A, R=base_config.R)
        assert _config_hash(base_config) != _config_hash(cfg2)

    def test_differs_on_M(self, base_config):
        from src.config import AGENT_ROLE_COUNT, TeamConfiguration
        A = [0] * AGENT_ROLE_COUNT
        A[2] = 4; A[3] = 2; A[5] = 2
        cfg2 = TeamConfiguration(N=2, M=4, A=tuple(A), R=base_config.R)
        assert _config_hash(base_config) != _config_hash(cfg2)

    def test_differs_on_R(self, base_config):
        from src.config import TeamConfiguration
        cfg2 = TeamConfiguration(N=base_config.N, M=base_config.M, A=base_config.A, R=2)
        assert _config_hash(base_config) != _config_hash(cfg2)

    def test_differs_on_A(self, base_config):
        from src.config import AGENT_ROLE_COUNT, TeamConfiguration
        A = list(base_config.A)
        A[2] -= 1; A[3] += 1  # swap one agent between roles; sum unchanged
        cfg2 = TeamConfiguration(N=base_config.N, M=base_config.M, A=tuple(A), R=base_config.R)
        assert _config_hash(base_config) != _config_hash(cfg2)

    def test_returns_int(self, base_config):
        assert isinstance(_config_hash(base_config), int)


class TestGeneratorFor:
    def test_reproducible(self, base_config):
        g1 = generator_for(42, base_config, 0)
        g2 = generator_for(42, base_config, 0)
        vals1 = g1.random(10).tolist()
        vals2 = g2.random(10).tolist()
        assert vals1 == vals2

    def test_k_independence(self, base_config):
        g0 = generator_for(42, base_config, 0)
        g1 = generator_for(42, base_config, 1)
        assert g0.random(10).tolist() != g1.random(10).tolist()

    def test_config_independence(self, base_config):
        from src.config import AGENT_ROLE_COUNT, TeamConfiguration
        A = list(base_config.A); A[2] -= 1; A[3] += 1
        cfg2 = TeamConfiguration(N=base_config.N, M=base_config.M, A=tuple(A), R=base_config.R)
        g1 = generator_for(42, base_config, 0)
        g2 = generator_for(42, cfg2, 0)
        assert g1.random(10).tolist() != g2.random(10).tolist()
