"""Tests for src/config.py — TeamConfiguration and SimulationHorizon validation."""

import pytest

from src.config import (
    AGENT_ROLE_COUNT,
    SearchConfiguration,
    SCENARIO_GROUPS,
    SimulationHorizon,
    TeamConfiguration,
    supervised_agent_count,
)


def _A(*pairs) -> tuple[int, ...]:
    A = [0] * AGENT_ROLE_COUNT
    for role_k, count in pairs:
        A[role_k - 1] = count
    return tuple(A)


# Roles 5 and 6 are background services and consume no human monitoring capacity.
_VALID_A = _A((1, 2), (3, 1), (4, 1), (5, 1), (7, 1))  # 5 supervised
_VALID_A_QA = _A((3, 1), (4, 1), (5, 1), (7, 2))       # a_qa,1=a_qa,2=0, 4 supervised


class TestTeamConfigurationValidation:
    def test_valid_config(self):
        cfg = TeamConfiguration(N=2, A_dev=_VALID_A, A_qa=_VALID_A_QA, R=4)
        assert cfg.N == 2 and cfg.R == 4

    def test_N_zero_raises(self):
        with pytest.raises(ValueError, match="N"):
            TeamConfiguration(N=0, A_dev=_VALID_A, R=4)

    def test_N_negative_raises(self):
        with pytest.raises(ValueError, match="N"):
            TeamConfiguration(N=-0.5, A_dev=_VALID_A, R=4)

    def test_fractional_N_is_a_valid_staffing_level(self):
        """N is a staffing level in FTE (paper §3.1), so it need not be an integer."""
        cfg = TeamConfiguration(N=0.25, A_dev=_VALID_A, A_qa=_VALID_A_QA, R=4)
        assert cfg.N == 0.25

    def test_R_zero_raises(self):
        with pytest.raises(ValueError, match="R"):
            TeamConfiguration(N=2, A_dev=_VALID_A, R=0)

    def test_A_wrong_length_raises(self):
        short_A = (1,) * (AGENT_ROLE_COUNT - 1)
        with pytest.raises(ValueError):
            TeamConfiguration(N=1, A_dev=short_A, R=1)

    def test_A_negative_raises(self):
        A = list(_VALID_A)
        A[5] = -1
        with pytest.raises(ValueError):
            TeamConfiguration(N=2, A_dev=tuple(A), R=4)

    def test_no_structural_bound_on_the_supervised_count(self):
        """`TeamConfiguration` no longer caps n_sup — that is Eq. (feas_monitoring)'s job.

        The condition needs the scenario parameters, so it lives in `src.feasibility`
        and a heavily staffed composition is structurally valid here.
        """
        A = _A((1, 20), (3, 20), (7, 20))
        cfg = TeamConfiguration(N=0.25, A_dev=A, A_qa=_A((3, 20), (7, 20)), R=4)
        assert supervised_agent_count(cfg.A_dev) == 60


class TestSearchConfigurationValidation:
    def _cfg(self, **overrides):
        kwargs = dict(
            scenario_paths={g: f"config/{g}_medium.json" for g in SCENARIO_GROUPS},
            horizon_path="config/horizon_default.json",
            strategy="random",
            N_values=(0.25, 1.0),
            R_values=(2,),
        )
        return SearchConfiguration(**(kwargs | overrides))

    def test_fractional_staffing_levels_are_accepted(self):
        assert self._cfg().N_values == (0.25, 1.0)

    def test_non_positive_staffing_level_raises(self):
        with pytest.raises(ValueError, match="N_values must all be > 0"):
            self._cfg(N_values=(0.0, 1.0))

    def test_sigma_defaults_to_the_paper_value(self):
        assert self._cfg().sigma == 2.0

    def test_sigma_below_one_raises(self):
        with pytest.raises(ValueError, match="sigma must be >= 1"):
            self._cfg(sigma=0.9)


class TestBackgroundServiceBudget:
    """Roles 5 and 6 need no supervision, so they never enter n_sup."""

    def test_background_agents_do_not_count_toward_n_sup(self):
        A = _A((1, 2), (3, 2), (4, 1), (7, 1), (5, 3), (6, 2))
        cfg = TeamConfiguration(N=2, A_dev=A, A_qa=_A((3, 1), (5, 1)), R=4)
        assert supervised_agent_count(cfg.A_dev) == 6
        assert sum(cfg.A_dev) == 11

    def test_large_background_count_accepted(self):
        A = _A((1, 1), (5, 50), (6, 25))
        cfg = TeamConfiguration(N=1, A_dev=A, A_qa=_A((5, 7)), R=2)
        assert cfg.agent_count_dev(5) == 50
        assert cfg.agent_count_dev(6) == 25

    def test_n_supervised_selects_active_composition(self):
        cfg = TeamConfiguration(
            N=2,
            A_dev=_A((1, 2), (3, 1), (5, 4)),
            A_qa=_A((3, 1), (6, 4)),
            R=4, F_freeze=1,
        )
        assert cfg.n_supervised(is_frozen=False) == 3
        assert cfg.n_supervised(is_frozen=True) == 1

    def test_no_impl_agent_raises(self):
        A = list(_VALID_A)
        A[0] = 0   # role 1 (Feature Implementation) must be ≥ 1
        A[2] += 2  # keep sum correct
        with pytest.raises(ValueError):
            TeamConfiguration(N=2, A_dev=tuple(A), R=4)

    def test_single_impl_agent_valid(self):
        A = _A((1, 1), (3, 1), (4, 1), (5, 1), (7, 2))  # sum=6, a_1=1
        cfg = TeamConfiguration(N=2, A_dev=A, A_qa=_VALID_A_QA, R=4)
        assert cfg.agent_count_dev(1) == 1

    def test_qa_impl_agent_nonzero_raises(self):
        A_qa = list(_VALID_A_QA)
        A_qa[0] = 1  # a_qa,1 must be 0
        with pytest.raises(ValueError, match="a_qa,1"):
            TeamConfiguration(N=2, A_dev=_VALID_A, A_qa=tuple(A_qa), R=4)

    def test_qa_arch_agent_nonzero_raises(self):
        A_qa = list(_VALID_A_QA)
        A_qa[1] = 1  # a_qa,2 must be 0
        with pytest.raises(ValueError, match="a_qa,2"):
            TeamConfiguration(N=2, A_dev=_VALID_A, A_qa=tuple(A_qa), R=4)

    def test_qa_constraint_skipped_when_no_freeze(self):
        A_qa = _VALID_A  # a_qa,1 != 0, but F_freeze=0 so it should be allowed
        cfg = TeamConfiguration(N=2, A_dev=_VALID_A, A_qa=A_qa, R=4, F_freeze=0)
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
