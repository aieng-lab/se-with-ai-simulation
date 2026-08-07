"""Tests for src/rng.py — hash stability and generator reproducibility."""

import numpy as np

from src.rng import _config_hash, generator_for, replication_seed


class TestConfigHash:
    def test_deterministic(self, base_config):
        assert _config_hash(base_config) == _config_hash(base_config)

    def test_differs_on_N(self, base_config):
        from src.config import TeamConfiguration
        cfg2 = TeamConfiguration(N=3, M=2, A_dev=base_config.A_dev, A_qa=base_config.A_qa, R=base_config.R)
        assert _config_hash(base_config) != _config_hash(cfg2)

    def test_differs_on_a_fractional_N(self, base_config):
        """N is a staffing level in FTE, so quarter steps must give distinct streams."""
        from src.config import TeamConfiguration
        cfgs = [
            TeamConfiguration(N=N, M=24, A_dev=base_config.A_dev, A_qa=base_config.A_qa,
                              R=base_config.R)
            for N in (0.25, 0.5, 0.75, 1.0)
        ]
        hashes = [_config_hash(c) for c in cfgs]
        assert len(set(hashes)) == len(hashes)

    def test_differs_on_M(self, base_config):
        from src.config import AGENT_ROLE_COUNT, TeamConfiguration
        A = [0] * AGENT_ROLE_COUNT
        A[0] = 4; A[2] = 2; A[4] = 2   # a_1=4, a_3=2, a_5=2, sum=8
        cfg2 = TeamConfiguration(N=2, M=4, A_dev=tuple(A), A_qa=base_config.A_qa, R=base_config.R)
        assert _config_hash(base_config) != _config_hash(cfg2)

    def test_differs_on_R(self, base_config):
        from src.config import TeamConfiguration
        cfg2 = TeamConfiguration(N=base_config.N, M=base_config.M, A_dev=base_config.A_dev, A_qa=base_config.A_qa, R=2)
        assert _config_hash(base_config) != _config_hash(cfg2)

    def test_differs_on_A(self, base_config):
        from src.config import AGENT_ROLE_COUNT, TeamConfiguration
        A = list(base_config.A_dev)
        A[0] -= 1; A[2] += 1  # swap one agent between roles; sum unchanged
        cfg2 = TeamConfiguration(N=base_config.N, M=base_config.M, A_dev=tuple(A), A_qa=base_config.A_qa, R=base_config.R)
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
        A = list(base_config.A_dev); A[2] -= 1; A[3] += 1
        cfg2 = TeamConfiguration(N=base_config.N, M=base_config.M, A_dev=tuple(A), A_qa=base_config.A_qa, R=base_config.R)
        g1 = generator_for(42, base_config, 0)
        g2 = generator_for(42, cfg2, 0)
        assert g1.random(10).tolist() != g2.random(10).tolist()
