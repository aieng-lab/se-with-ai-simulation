"""Tests for src/config.py — TeamConfiguration and SimulationHorizon validation."""

import pytest

from src.config import AGENT_ROLE_COUNT, SimulationHorizon, TeamConfiguration


def _A(*pairs) -> tuple[int, ...]:
    A = [0] * AGENT_ROLE_COUNT
    for role_k, count in pairs:
        A[role_k - 1] = count
    return tuple(A)


_VALID_A = _A((1, 2), (3, 1), (4, 1), (5, 1), (7, 1))  # N=2, M=3 → sum=6
_VALID_A_QA = _A((3, 1), (4, 1), (5, 1), (7, 2))       # a_qa,1=a_qa,2=0, sum=5


class TestTeamConfigurationValidation:
    def test_valid_config(self):
        cfg = TeamConfiguration(N=2, M=3, A_dev=_VALID_A, A_qa=_VALID_A_QA, R=4)
        assert cfg.N == 2 and cfg.M == 3 and cfg.R == 4

    def test_N_zero_raises(self):
        with pytest.raises(ValueError, match="N"):
            TeamConfiguration(N=0, M=3, A_dev=_VALID_A, R=4)

    def test_M_zero_raises(self):
        with pytest.raises(ValueError, match="M"):
            TeamConfiguration(N=2, M=0, A_dev=_VALID_A, R=4)

    def test_R_zero_raises(self):
        with pytest.raises(ValueError, match="R"):
            TeamConfiguration(N=2, M=3, A_dev=_VALID_A, R=0)

    def test_A_wrong_length_raises(self):
        short_A = (1,) * (AGENT_ROLE_COUNT - 1)
        with pytest.raises(ValueError):
            TeamConfiguration(N=1, M=AGENT_ROLE_COUNT - 1, A_dev=short_A, R=1)

    def test_A_negative_raises(self):
        A = list(_VALID_A)
        A[5] = -1
        with pytest.raises(ValueError):
            TeamConfiguration(N=2, M=3, A_dev=tuple(A), R=4)

    def test_A_sum_exceeds_NM_raises(self):
        A = list(_VALID_A)
        A[2] += 1  # sum(A_dev) = 7 > N*M = 6
        with pytest.raises(ValueError):
            TeamConfiguration(N=2, M=3, A_dev=tuple(A), R=4)

    def test_A_sum_less_than_NM_valid(self):
        # Fewer agents than budget (sum=4 < N*M=6) is now allowed
        A = _A((1, 2), (3, 1), (7, 1))  # sum=4
        cfg = TeamConfiguration(N=2, M=3, A_dev=A, A_qa=_A((3, 1), (7, 1)), R=4)
        assert sum(cfg.A_dev) == 4

    def test_no_impl_agent_raises(self):
        A = list(_VALID_A)
        A[0] = 0   # role 1 (Feature Implementation) must be ≥ 1
        A[2] += 2  # keep sum correct
        with pytest.raises(ValueError):
            TeamConfiguration(N=2, M=3, A_dev=tuple(A), R=4)

    def test_single_impl_agent_valid(self):
        A = _A((1, 1), (3, 1), (4, 1), (5, 1), (7, 2))  # sum=6, a_1=1
        cfg = TeamConfiguration(N=2, M=3, A_dev=A, A_qa=_VALID_A_QA, R=4)
        assert cfg.agent_count_dev(1) == 1

    def test_qa_impl_agent_nonzero_raises(self):
        A_qa = list(_VALID_A_QA)
        A_qa[0] = 1  # a_qa,1 must be 0
        with pytest.raises(ValueError, match="a_qa,1"):
            TeamConfiguration(N=2, M=3, A_dev=_VALID_A, A_qa=tuple(A_qa), R=4)

    def test_qa_arch_agent_nonzero_raises(self):
        A_qa = list(_VALID_A_QA)
        A_qa[1] = 1  # a_qa,2 must be 0
        with pytest.raises(ValueError, match="a_qa,2"):
            TeamConfiguration(N=2, M=3, A_dev=_VALID_A, A_qa=tuple(A_qa), R=4)

    def test_qa_constraint_skipped_when_no_freeze(self):
        A_qa = _VALID_A  # a_qa,1 != 0, but F_freeze=0 so it should be allowed
        cfg = TeamConfiguration(N=2, M=3, A_dev=_VALID_A, A_qa=A_qa, R=4, F_freeze=0)
        assert cfg.A_qa[0] == _VALID_A[0]

    def test_frozen(self, base_config):
        with pytest.raises(Exception):
            base_config.N = 99  # type: ignore[misc]


class TestAgentCountDev:
    def test_1indexed(self, base_config):
        assert base_config.agent_count_dev(2) == base_config.A_dev[1]
        assert base_config.agent_count_dev(1) == base_config.A_dev[0]
        assert base_config.agent_count_dev(8) == base_config.A_dev[7]

    def test_role_0_raises(self, base_config):
        with pytest.raises(ValueError):
            base_config.agent_count_dev(0)

    def test_role_too_large_raises(self, base_config):
        with pytest.raises(ValueError):
            base_config.agent_count_dev(AGENT_ROLE_COUNT + 1)


class TestAgentCountQa:
    def test_1indexed(self, base_config):
        assert base_config.agent_count_qa(2) == base_config.A_qa[1]
        assert base_config.agent_count_qa(1) == base_config.A_qa[0]
        assert base_config.agent_count_qa(8) == base_config.A_qa[7]

    def test_role_0_raises(self, base_config):
        with pytest.raises(ValueError):
            base_config.agent_count_qa(0)

    def test_role_too_large_raises(self, base_config):
        with pytest.raises(ValueError):
            base_config.agent_count_qa(AGENT_ROLE_COUNT + 1)

    def test_impl_and_arch_are_zero(self, base_config):
        assert base_config.agent_count_qa(1) == 0
        assert base_config.agent_count_qa(2) == 0


class TestSimulationHorizon:
    def test_valid(self):
        s = SimulationHorizon(T=52, K=100, master_seed=42)
        assert s.T == 52 and s.K == 100

    def test_T_zero_raises(self):
        with pytest.raises(ValueError):
            SimulationHorizon(T=0, K=10, master_seed=0)

    def test_K_zero_raises(self):
        with pytest.raises(ValueError):
            SimulationHorizon(T=10, K=0, master_seed=0)
