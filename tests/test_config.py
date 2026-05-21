"""Tests for src/config.py — TeamConfiguration and SweepSettings validation."""

import pytest

from src.config import AGENT_ROLE_COUNT, SweepSettings, TeamConfiguration


def _A(*pairs) -> tuple[int, ...]:
    A = [0] * AGENT_ROLE_COUNT
    for role_k, count in pairs:
        A[role_k - 1] = count
    return tuple(A)


_VALID_A = _A((3, 2), (4, 1), (5, 1), (6, 1), (8, 1))  # N=2, M=3 → sum=6


class TestTeamConfigurationValidation:
    def test_valid_config(self):
        cfg = TeamConfiguration(N=2, M=3, A=_VALID_A, R=4)
        assert cfg.N == 2 and cfg.M == 3 and cfg.R == 4

    def test_N_zero_raises(self):
        with pytest.raises(ValueError, match="N"):
            TeamConfiguration(N=0, M=3, A=_VALID_A, R=4)

    def test_M_zero_raises(self):
        with pytest.raises(ValueError, match="M"):
            TeamConfiguration(N=2, M=0, A=_VALID_A, R=4)

    def test_R_zero_raises(self):
        with pytest.raises(ValueError, match="R"):
            TeamConfiguration(N=2, M=3, A=_VALID_A, R=0)

    def test_A_wrong_length_raises(self):
        short_A = (1,) * (AGENT_ROLE_COUNT - 1)
        with pytest.raises(ValueError):
            TeamConfiguration(N=1, M=AGENT_ROLE_COUNT - 1, A=short_A, R=1)

    def test_A_negative_raises(self):
        A = list(_VALID_A)
        A[5] = -1
        with pytest.raises(ValueError):
            TeamConfiguration(N=2, M=3, A=tuple(A), R=4)

    def test_A_sum_mismatch_raises(self):
        A = list(_VALID_A)
        A[2] += 1  # sum(A) = 7 ≠ N*M = 6
        with pytest.raises(ValueError):
            TeamConfiguration(N=2, M=3, A=tuple(A), R=4)

    def test_no_impl_agent_raises(self):
        A = list(_VALID_A)
        A[2] = 0   # role 3 (Feature Implementation) must be ≥ 1
        A[3] += 2  # keep sum correct
        with pytest.raises(ValueError):
            TeamConfiguration(N=2, M=3, A=tuple(A), R=4)

    def test_single_impl_agent_valid(self):
        A = _A((3, 1), (4, 1), (5, 1), (6, 1), (8, 2))  # sum=6, a_3=1
        cfg = TeamConfiguration(N=2, M=3, A=A, R=4)
        assert cfg.agent_count(3) == 1

    def test_frozen(self, base_config):
        with pytest.raises(Exception):
            base_config.N = 99  # type: ignore[misc]


class TestAgentCount:
    def test_1indexed(self, base_config):
        assert base_config.agent_count(3) == base_config.A[2]
        assert base_config.agent_count(1) == base_config.A[0]
        assert base_config.agent_count(13) == base_config.A[12]

    def test_role_0_raises(self, base_config):
        with pytest.raises(ValueError):
            base_config.agent_count(0)

    def test_role_14_raises(self, base_config):
        with pytest.raises(ValueError):
            base_config.agent_count(14)


class TestSweepSettings:
    def test_valid(self):
        s = SweepSettings(T=52, K=100, master_seed=42)
        assert s.T == 52 and s.K == 100 and s.N_PO == 1

    def test_T_zero_raises(self):
        with pytest.raises(ValueError):
            SweepSettings(T=0, K=10, master_seed=0)

    def test_K_zero_raises(self):
        with pytest.raises(ValueError):
            SweepSettings(T=10, K=0, master_seed=0)

    def test_N_PO_default(self):
        assert SweepSettings(T=1, K=1, master_seed=0).N_PO == 1

    def test_N_PO_custom(self):
        assert SweepSettings(T=1, K=1, master_seed=0, N_PO=2).N_PO == 2

    def test_N_PO_zero_raises(self):
        with pytest.raises(ValueError, match="N_PO"):
            SweepSettings(T=1, K=1, master_seed=0, N_PO=0)
