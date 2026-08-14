"""Tests for src/config.py — TeamConfiguration and SimulationHorizon validation."""

from pathlib import Path

import pytest

from src.config import (
    AGENT_ROLE_COUNT,
    ALL_SCENARIO_TIERS,
    TIER_INITIALS,
    SearchConfiguration,
    SCENARIO_GROUPS,
    SCENARIO_TIERS,
    SEARCH_STRATEGIES,
    SimulationHorizon,
    TeamConfiguration,
    scenario_id,
    scenario_tier_combinations,
    scenario_tiers,
    supervised_agent_count,
)
from src.rng import _config_hash

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


class TestScenarioTierGrid:
    """The §4.2 grid is not a cube: `project` and `agent` each carry a fourth tier."""

    def test_project_has_the_extra_very_high_tier(self):
        assert scenario_tiers("project") == ("low", "medium", "high", "veryhigh")

    def test_agent_has_the_extra_pessimistic_tier(self):
        assert scenario_tiers("agent") == ("low", "medium", "high", "pessimistic")

    def test_every_other_group_has_the_three_base_tiers(self):
        for group in SCENARIO_GROUPS:
            if group not in ("project", "agent"):
                assert scenario_tiers(group) == SCENARIO_TIERS

    def test_no_group_carries_both_fourth_tiers(self):
        # The visual encodings of `src.visualization` give "veryhigh" and "pessimistic"
        # the same fourth slot (colour, shape, size), which is only sound because no
        # channel can ever be asked to draw both at once.
        for group in SCENARIO_GROUPS:
            tiers = set(scenario_tiers(group))
            assert not {"veryhigh", "pessimistic"} <= tiers

    def test_every_tier_has_a_distinct_initial(self):
        assert set(TIER_INITIALS) == set(ALL_SCENARIO_TIERS)
        assert len(set(TIER_INITIALS.values())) == len(ALL_SCENARIO_TIERS)

    def test_unknown_group_raises(self):
        with pytest.raises(ValueError, match="unknown scenario group"):
            scenario_tiers("nonsense")

    def test_combination_count_is_the_per_group_product(self):
        combos = scenario_tier_combinations()
        assert len(combos) == len(set(combos)) == 3 ** 3 * 4 * 4 == 432

    def test_scenario_ids_are_unique_over_the_grid(self):
        assert len({scenario_id(t) for t in scenario_tier_combinations()}) == 432

    def test_scenario_id_rejects_a_wrong_length_tuple(self):
        with pytest.raises(ValueError, match="expected 5 tiers"):
            scenario_id(("low", "medium"))

    def test_every_group_tier_has_a_parameter_file(self):
        for group in SCENARIO_GROUPS:
            for tier in scenario_tiers(group):
                assert (_CONFIG_DIR / f"{group}_{tier}.json").is_file()


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
            strategy="evolutionary",
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

    @pytest.mark.parametrize("strategy", SEARCH_STRATEGIES)
    def test_both_paper_strategies_are_accepted(self, strategy):
        assert self._cfg(strategy=strategy).strategy == strategy

    def test_the_removed_random_strategy_is_rejected(self):
        """Uniform random sampling is the shared initialiser, not a strategy (§3.11.2)."""
        with pytest.raises(ValueError, match="strategy must be one of"):
            self._cfg(strategy="random")

    def test_epsilon_defaults_to_the_paper_value(self):
        assert self._cfg().epsilon == 0.02

    @pytest.mark.parametrize("epsilon", [-0.01, 1.0, 1.5])
    def test_epsilon_outside_the_unit_interval_raises(self, epsilon):
        with pytest.raises(ValueError, match=r"epsilon must be in \[0, 1\)"):
            self._cfg(epsilon=epsilon)

    def test_non_positive_elite_raises(self):
        with pytest.raises(ValueError, match="n_elite must be >= 1"):
            self._cfg(n_elite=0)

    def test_non_positive_restart_count_raises(self):
        with pytest.raises(ValueError, match="n_starts must be >= 1"):
            self._cfg(n_starts=0)

    def test_restarts_are_not_bounded_by_the_budget(self):
        """The descent runs every restart to convergence, so `budget` does not cap it."""
        assert self._cfg(budget=4, n_starts=20).n_starts == 20


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

    def test_qa_canonicalised_when_no_freeze(self):
        # a_qa,1 != 0 does not raise at F_freeze=0 — the vector is never applied.  It is
        # canonicalised to zero rather than kept, so that a configuration is identified
        # only by the parameters that affect it.
        cfg = TeamConfiguration(N=2, A_dev=_VALID_A, A_qa=_VALID_A, R=4, F_freeze=0)
        assert cfg.A_qa == (0,) * AGENT_ROLE_COUNT

    def test_qa_malformed_still_raises_when_no_freeze(self):
        # Canonicalisation runs after the shape checks, so a broken A_qa is not swallowed.
        with pytest.raises(ValueError, match="A_qa"):
            TeamConfiguration(N=2, A_dev=_VALID_A, A_qa=(1, 2, 3), R=4, F_freeze=0)
        with pytest.raises(ValueError, match="A_qa"):
            TeamConfiguration(N=2, A_dev=_VALID_A, A_qa=(0,) * 7 + (-1,), R=4, F_freeze=0)

    def test_inert_qa_variants_are_one_configuration(self):
        """The regression test for the behavioural-duplicate finding.

        Configurations differing only in an unexercised A_qa are the same model, and must
        therefore be the same object: equal, equally hashed, and — via _config_hash —
        scored on the same RNG stream.  Otherwise each variant lands at a different cost
        and the search spends budget distinguishing teams that are not distinct.
        """
        variants = [
            (0,) * AGENT_ROLE_COUNT,
            _A((3, 1)),
            _A((3, 1), (4, 1), (5, 2), (6, 2), (7, 1), (8, 1)),
            _A((4, 1), (5, 5), (7, 1)),
        ]
        cfgs = [TeamConfiguration(N=1.0, A_dev=_VALID_A, A_qa=v, R=4, F_freeze=0)
                for v in variants]
        assert len({c.A_qa for c in cfgs}) == 1
        assert len(set(cfgs)) == 1
        assert len({_config_hash(c) for c in cfgs}) == 1

    def test_qa_preserved_when_freeze_present(self):
        cfg = TeamConfiguration(N=2, A_dev=_VALID_A, A_qa=_VALID_A_QA, R=4, F_freeze=1)
        assert cfg.A_qa == _VALID_A_QA

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
