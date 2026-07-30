"""Tests for src/search_configurations.py — composition sampling and mutation.

The two-tier budget is the invariant under test: the supervised roles share the
span-of-control bound N*M, while the background detection services (Code Review,
Security Audit) are not bounded by it and sampled separately, uniformly from
{0, ..., floor(r_arrival)}.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import (
    BACKGROUND_ROLE_INDICES,
    AGENT_ROLE_COUNT,
    SCENARIO_GROUPS,
    SearchConfiguration,
    TeamConfiguration,
    supervised_agent_count,
)
from src.search_configurations import (
    _mutate,
    _random_configs,
    _sample_composition_with_impl,
    _sample_qa_composition,
    background_count_max,
)

BG_MAX = 12   # matches r_arrival of config/project_medium.json


@pytest.fixture
def search_cfg():
    return SearchConfiguration(
        scenario_paths={
            group: f"config/{group}_medium.json" for group in SCENARIO_GROUPS
        },
        horizon_path="config/horizon_default.json",
        strategy="random",
        N_values=(1, 2, 3),
        M_max=6,
        R_values=(2, 4),
        F_freeze_values=(1, 2),
        n_samples=50,
        seed=7,
    )


class TestCompositionSampling:
    def test_dev_composition_spends_budget_on_supervised_roles(self):
        rng = np.random.default_rng(0)
        for budget in (1, 3, 12):
            for _ in range(50):
                A = _sample_composition_with_impl(budget, rng, BG_MAX)
                assert len(A) == AGENT_ROLE_COUNT
                assert supervised_agent_count(A) == budget
                assert A[0] >= 1

    def test_qa_composition_pins_impl_and_arch_to_zero(self):
        rng = np.random.default_rng(1)
        for budget in (0, 2, 9):
            for _ in range(50):
                A = _sample_qa_composition(budget, rng, BG_MAX)
                assert A[0] == 0 and A[1] == 0
                assert supervised_agent_count(A) == budget

    def test_background_counts_are_uniform_up_to_the_arrival_rate(self):
        rng = np.random.default_rng(2)
        draws = [
            _sample_composition_with_impl(1, rng, BG_MAX)[i]
            for _ in range(400)
            for i in BACKGROUND_ROLE_INDICES
        ]
        assert min(draws) == 0, "the services must sometimes be switched off"
        assert max(draws) == BG_MAX, "the whole range up to r_arrival must be reachable"
        # Uniform on 13 values: every count must occur, none may dominate.
        counts = np.bincount(draws, minlength=BG_MAX + 1)
        assert counts.min() > 0
        assert counts.max() < len(draws) / 4

    def test_background_bound_is_the_floor_of_the_arrival_rate(self):
        class _Params:
            r_arrival = 12.7

        assert background_count_max(_Params()) == 12


class TestRandomConfigs:
    def test_all_sampled_configs_respect_the_supervised_budget(self, search_cfg):
        rng = np.random.default_rng(3)
        configs = _random_configs(search_cfg, rng, n_samples=300, bg_max=BG_MAX)
        assert len(configs) == 300
        for cfg in configs:
            budget = cfg.N * cfg.M
            assert supervised_agent_count(cfg.A_dev) <= budget
            assert supervised_agent_count(cfg.A_qa) <= budget
            assert cfg.A_dev[0] >= 1

    def test_background_services_may_exceed_the_budget(self, search_cfg):
        rng = np.random.default_rng(4)
        configs = _random_configs(search_cfg, rng, n_samples=300, bg_max=BG_MAX)
        # Total headcount is not bounded by N*M any more, so at least one config
        # must exceed it once the background services are counted in.
        assert any(sum(cfg.A_dev) > cfg.N * cfg.M for cfg in configs)


class TestMutation:
    def _parent(self):
        A_dev = [0] * AGENT_ROLE_COUNT
        A_dev[0], A_dev[2], A_dev[4] = 2, 1, 1     # 3 supervised of N*M = 6
        A_qa = [0] * AGENT_ROLE_COUNT
        A_qa[2], A_qa[5] = 2, 1
        return TeamConfiguration(N=2, M=3, A_dev=tuple(A_dev), A_qa=tuple(A_qa),
                                 R=4, F_freeze=1)

    def test_children_are_valid_and_within_the_supervised_budget(self, search_cfg):
        rng = np.random.default_rng(5)
        parent = self._parent()
        for _ in range(500):
            child = _mutate(parent, rng, search_cfg)
            budget = child.N * child.M
            assert supervised_agent_count(child.A_dev) <= budget
            assert supervised_agent_count(child.A_qa) <= budget
            assert child.A_dev[0] >= 1
            assert child.A_qa[0] == 0 and child.A_qa[1] == 0
            parent = child   # chain mutations to explore further from the start

    def test_mutation_reaches_the_background_services(self, search_cfg):
        """The +/-1 background operation must actually fire and move both roles."""
        rng = np.random.default_rng(6)
        parent = self._parent()
        seen = {i: set() for i in BACKGROUND_ROLE_INDICES}
        for _ in range(500):
            child = _mutate(parent, rng, search_cfg)
            for i in BACKGROUND_ROLE_INDICES:
                seen[i].add(child.A_dev[i])
            parent = child
        for i in BACKGROUND_ROLE_INDICES:
            assert len(seen[i]) > 1, f"role index {i} was never mutated"

    def test_background_services_can_grow_past_the_budget(self, search_cfg):
        """Nothing caps the services, so a mutation chain can push them past N*M."""
        rng = np.random.default_rng(7)
        parent = self._parent()
        peak = 0
        for _ in range(2000):
            parent = _mutate(parent, rng, search_cfg)
            peak = max(peak, max(parent.A_dev[i] for i in BACKGROUND_ROLE_INDICES))
        assert peak > 1

    def test_impl_agent_is_never_drained_to_zero(self, search_cfg):
        rng = np.random.default_rng(8)
        A_dev = [0] * AGENT_ROLE_COUNT
        A_dev[0] = 1   # already at the floor
        parent = TeamConfiguration(N=1, M=2, A_dev=tuple(A_dev), R=2, F_freeze=0)
        for _ in range(300):
            parent = _mutate(parent, rng, search_cfg)
            assert parent.A_dev[0] >= 1
