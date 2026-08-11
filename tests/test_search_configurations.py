"""Tests for src/search_configurations.py — composition sampling and mutation.

The invariant under test is the throughput saturation bound u_k (paper §3.13.1),
which caps each supervised role; the background detection services (Code Review,
Security Audit) are subject to no bound at all and are sampled separately,
uniformly from {0, ..., floor(r_arrival)}.  The one collective limit on the
supervised roles, the human-capacity condition of Eq. (feas_monitoring), is applied
by the feasibility gate after the draw, not by the sampler.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.config import (
    BACKGROUND_ROLE_INDICES,
    AGENT_ROLE_COUNT,
    SCENARIO_GROUPS,
    SUPERVISED_ROLE_INDICES,
    SearchConfiguration,
    SimulationHorizon,
    TeamConfiguration,
    supervised_agent_count,
)
from src.io import load_scenario_params
from src.pareto import pareto_frontier
from src.search_configurations import (
    _QA_SUPERVISED_INDICES,
    _mutate,
    _random_draws,
    _run_evolutionary_search,
    _run_random_search,
    _sample_dev_composition,
    _sample_qa_composition,
    background_count_max,
    saturation_bounds,
    write_samples,
)

BG_MAX = 12   # matches r_arrival of config/project_medium.json

#: Generous stand-in bounds: large enough that u_k is not the binding constraint in
#: the tests that are about something else.
WIDE_BOUNDS = {i: 4 for i in SUPERVISED_ROLE_INDICES}


@pytest.fixture
def search_cfg():
    return SearchConfiguration(
        scenario_paths={
            group: f"config/{group}_medium.json" for group in SCENARIO_GROUPS
        },
        horizon_path="config/horizon_default.json",
        strategy="random",
        N_values=(1.0, 2.0, 3.0),
        R_values=(2, 4),
        F_freeze_values=(1, 2),
        n_samples=50,
        seed=7,
    )


@pytest.fixture
def medium_params():
    return load_scenario_params(
        {group: f"config/{group}_medium.json" for group in SCENARIO_GROUPS}
    )


class TestSaturationBounds:
    def test_matches_the_paper_formula(self, medium_params):
        """u_k = ceil(sigma * W_k / rho_k), Eqs. (workload), (sat_rates), (sat_bound)."""
        p = medium_params
        sigma = 2.0
        W_3 = p.r_bugs * (2.0 - p.delta_min) * p.r_arrival
        expected = {
            0: math.ceil(sigma * (1.0 + p.f_arch_chg) * p.r_arrival / (p.r_F * p.delta_min)),
            1: math.ceil(sigma * p.f_arch_chg * p.r_arrival / p.r_arch_chg),
            2: math.ceil(sigma * W_3 / (p.r_fix * p.delta_min)),
            3: math.ceil(sigma * (p.delta_impl * p.r_arrival + p.delta_fix * W_3) / p.r_refactor),
            6: math.ceil(sigma * p.c_test_ceil * p.r_arrival / p.r_test),
            7: math.ceil(sigma * p.r_arrival / p.r_doc),
        }
        assert saturation_bounds(p, sigma) == {k: max(1, v) for k, v in expected.items()}

    def test_covers_exactly_the_supervised_roles(self, medium_params):
        assert set(saturation_bounds(medium_params, 2.0)) == set(SUPERVISED_ROLE_INDICES)

    def test_impl_role_is_always_reachable(self, medium_params):
        """u_1 >= 1 for every scenario, else a_dev,1 >= 1 could not be satisfied."""
        assert min(saturation_bounds(medium_params, 1.0).values()) >= 1

    def test_grows_with_the_arrival_rate(self):
        from dataclasses import replace
        low = load_scenario_params(
            {g: f"config/{g}_{'low' if g == 'project' else 'medium'}.json"
             for g in SCENARIO_GROUPS}
        )
        high = load_scenario_params(
            {g: f"config/{g}_{'high' if g == 'project' else 'medium'}.json"
             for g in SCENARIO_GROUPS}
        )
        assert replace(low, r_arrival=high.r_arrival) == high  # only r_arrival differs
        lo, hi = saturation_bounds(low, 2.0), saturation_bounds(high, 2.0)
        assert all(hi[k] >= lo[k] for k in lo)
        assert any(hi[k] > lo[k] for k in lo)

    def test_rejects_a_zero_throughput_rate(self, medium_params):
        from dataclasses import replace
        with pytest.raises(ValueError, match="rho"):
            saturation_bounds(replace(medium_params, r_doc=0.0), 2.0)

    def test_rejects_headroom_below_one(self, medium_params):
        with pytest.raises(ValueError, match="sigma"):
            saturation_bounds(medium_params, 0.5)


class TestCompositionSampling:
    def test_dev_composition_stays_within_the_saturation_bounds(self, medium_params):
        rng = np.random.default_rng(0)
        bounds = saturation_bounds(medium_params, 2.0)
        for _ in range(300):
            A = _sample_dev_composition(rng, bounds, BG_MAX)
            assert len(A) == AGENT_ROLE_COUNT
            assert A[0] >= 1
            assert all(A[i] <= bounds[i] for i in SUPERVISED_ROLE_INDICES)

    def test_qa_composition_pins_impl_and_arch_to_zero(self, medium_params):
        rng = np.random.default_rng(1)
        bounds = saturation_bounds(medium_params, 2.0)
        for _ in range(300):
            A = _sample_qa_composition(rng, bounds, BG_MAX)
            assert A[0] == 0 and A[1] == 0
            assert all(A[i] <= bounds[i] for i in _QA_SUPERVISED_INDICES)

    def test_every_role_count_up_to_its_bound_is_reachable(self):
        """The per-role draw is uniform on {0,...,u_k}, so it must span the range."""
        rng = np.random.default_rng(9)
        bounds = {i: 3 for i in SUPERVISED_ROLE_INDICES}
        seen = {i: set() for i in SUPERVISED_ROLE_INDICES}
        for _ in range(400):
            A = _sample_dev_composition(rng, bounds, BG_MAX)
            for i in SUPERVISED_ROLE_INDICES:
                seen[i].add(A[i])
        assert seen[0] == {1, 2, 3}, "the impl role is drawn from {1,...,u_1}"
        for i in SUPERVISED_ROLE_INDICES[1:]:
            assert seen[i] == {0, 1, 2, 3}

    def test_background_counts_are_uniform_up_to_the_arrival_rate(self, medium_params):
        rng = np.random.default_rng(2)
        bounds = saturation_bounds(medium_params, 2.0)
        draws = [
            _sample_dev_composition(rng, bounds, BG_MAX)[i]
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


class TestRandomDraws:
    def test_every_draw_is_structurally_valid_and_within_u_k(self, search_cfg,
                                                             medium_params):
        """The sampler builds only valid candidates, so nothing is discarded here."""
        rng = np.random.default_rng(3)
        bounds = saturation_bounds(medium_params, search_cfg.sigma)
        draws = _random_draws(search_cfg, rng, 300, medium_params)
        assert len(draws) == 300
        for cfg in draws:
            assert cfg.A_dev[0] >= 1
            assert cfg.A_qa[0] == 0 and cfg.A_qa[1] == 0
            assert 0 <= cfg.F_freeze <= cfg.R - 1
            assert all(cfg.A_dev[i] <= bounds[i] for i in SUPERVISED_ROLE_INDICES)
            assert all(cfg.A_qa[i] <= bounds[i] for i in SUPERVISED_ROLE_INDICES)

    def test_background_services_are_capped_by_nothing(self, search_cfg, medium_params):
        """The services answer to neither u_k nor the supervised count."""
        rng = np.random.default_rng(4)
        draws = _random_draws(search_cfg, rng, 300, medium_params)
        assert any(sum(cfg.A_dev) > supervised_agent_count(cfg.A_dev) * 2
                   for cfg in draws)


class TestMutation:
    def _parent(self):
        A_dev = [0] * AGENT_ROLE_COUNT
        A_dev[0], A_dev[2], A_dev[4] = 2, 1, 1     # 3 supervised + 1 background
        A_qa = [0] * AGENT_ROLE_COUNT
        A_qa[2], A_qa[5] = 2, 1
        return TeamConfiguration(N=2.0, A_dev=tuple(A_dev), A_qa=tuple(A_qa),
                                 R=4, F_freeze=1)

    def test_children_are_structurally_valid(self, search_cfg):
        rng = np.random.default_rng(5)
        parent = self._parent()
        for _ in range(500):
            child = _mutate(parent, rng, search_cfg, WIDE_BOUNDS)
            assert child.A_dev[0] >= 1
            assert child.A_qa[0] == 0 and child.A_qa[1] == 0
            parent = child   # chain mutations to explore further from the start

    def test_children_never_exceed_the_saturation_bounds(self, search_cfg):
        rng = np.random.default_rng(15)
        bounds = {i: 2 for i in SUPERVISED_ROLE_INDICES}
        parent = self._parent()
        for _ in range(1000):
            parent = _mutate(parent, rng, search_cfg, bounds)
            for i in SUPERVISED_ROLE_INDICES:
                assert parent.A_dev[i] <= bounds[i], f"A_dev role index {i} overran u_k"
                assert parent.A_qa[i] <= bounds[i], f"A_qa role index {i} overran u_k"

    def test_N_walks_the_discrete_staffing_ladder(self, search_cfg):
        """The N mutation moves to an adjacent element of N_values and stays in it."""
        rng = np.random.default_rng(16)
        parent = self._parent()
        seen_N = set()
        for _ in range(400):
            parent = _mutate(parent, rng, search_cfg, WIDE_BOUNDS)
            seen_N.add(parent.N)
        assert len(seen_N) > 1, "N must move at all"
        assert seen_N <= set(search_cfg.N_values)

    def test_mutation_reaches_the_background_services(self, search_cfg):
        """The +/-1 background operation must actually fire and move both roles."""
        rng = np.random.default_rng(6)
        parent = self._parent()
        seen = {i: set() for i in BACKGROUND_ROLE_INDICES}
        for _ in range(500):
            child = _mutate(parent, rng, search_cfg, WIDE_BOUNDS)
            for i in BACKGROUND_ROLE_INDICES:
                seen[i].add(child.A_dev[i])
            parent = child
        for i in BACKGROUND_ROLE_INDICES:
            assert len(seen[i]) > 1, f"role index {i} was never mutated"

    def test_background_services_can_grow_without_bound(self, search_cfg):
        """Nothing caps the services — not u_k, not the supervised count."""
        rng = np.random.default_rng(7)
        parent = self._parent()
        peak = 0
        for _ in range(2000):
            parent = _mutate(parent, rng, search_cfg, WIDE_BOUNDS)
            peak = max(peak, max(parent.A_dev[i] for i in BACKGROUND_ROLE_INDICES))
        assert peak > 1

    def test_impl_agent_is_never_drained_to_zero(self, search_cfg):
        rng = np.random.default_rng(8)
        A_dev = [0] * AGENT_ROLE_COUNT
        A_dev[0] = 1   # already at the floor
        parent = TeamConfiguration(N=1.0, A_dev=tuple(A_dev), R=2, F_freeze=0)
        for _ in range(300):
            parent = _mutate(parent, rng, search_cfg, WIDE_BOUNDS)
            assert parent.A_dev[0] >= 1


# ---------------------------------------------------------------------------
# Search drivers: budget accounting (paper §4.2) and sample retention
# ---------------------------------------------------------------------------

# A deliberately wasteful search range.  The debt-low tier has Gamma_doc = 0, so
# monitoring costs the full H_mon.overhead = 2.5 h per agent and exhausts the 40 h
# capacity from 16 supervised agents per FTE on.  At the quarter-FTE staffing level
# that threshold is already crossed by 4 supervised agents, which the saturation
# bounds comfortably allow, so a fair share of the draws is rejected by the
# human-capacity condition of Eq. (feas_monitoring) — the one rejection path, and
# what makes the budget accounting of §4.2 observable.
_WASTEFUL_TIERS = {group: "medium" for group in SCENARIO_GROUPS} | {"debt": "low"}


@pytest.fixture
def wasteful_cfg():
    return SearchConfiguration(
        scenario_paths={group: f"config/{group}_{tier}.json"
                        for group, tier in _WASTEFUL_TIERS.items()},
        horizon_path="config/horizon_default.json",
        strategy="random",
        N_values=(0.25, 1.0),
        R_values=(2, 4),
        F_freeze_values=(0, 1),
        n_samples=40,
        n_samples_initial=20,
        n_generations=3,
        n_offspring=10,
        seed=11,
    )


@pytest.fixture
def tiny_horizon():
    return SimulationHorizon(T=4, K=3, master_seed=1)


@pytest.fixture
def wasteful_params(wasteful_cfg):
    return load_scenario_params(wasteful_cfg.scenario_paths)


class TestSearchBudget:
    def test_random_search_charges_every_candidate(self, wasteful_cfg, tiny_horizon,
                                                   wasteful_params):
        outcome = _run_random_search(
            wasteful_cfg, tiny_horizon, wasteful_params, np.random.default_rng(0),
            wasteful_cfg.n_samples, progress_every=0,
        )
        assert outcome.n_evaluated == wasteful_cfg.n_samples
        assert outcome.n_simulated == len(outcome.aggs) <= outcome.n_evaluated
        assert outcome.n_evaluated - outcome.n_simulated == sum(
            1 for e in outcome.manifest.values() if not e["feasible"]
        )

    def test_infeasible_draws_are_charged_and_labelled(self, wasteful_cfg, tiny_horizon,
                                                       wasteful_params):
        """A draw rejected by Eq. (feas_monitoring) still consumes one evaluation (§4.2)."""
        from dataclasses import replace
        cfg = replace(wasteful_cfg, N_values=(0.25,), n_samples=30)
        outcome = _run_random_search(
            cfg, tiny_horizon, wasteful_params, np.random.default_rng(3),
            cfg.n_samples, progress_every=0,
        )
        assert outcome.n_evaluated == cfg.n_samples
        rejected = [e for e in outcome.manifest.values() if not e["feasible"]]
        assert rejected, "the wasteful range must produce rejections"
        assert all(e["failed_step"] == 1 for e in rejected)
        # The manifest must still describe the candidate that was thrown away.
        assert all({"N", "R", "F_freeze", "A_dev", "A_qa"} <= e.keys() for e in rejected)

    def test_evolutionary_search_stays_within_its_nominal_budget(
        self, wasteful_cfg, tiny_horizon, wasteful_params
    ):
        """Infeasible offspring consume budget too, so §4.2's E is a hard ceiling."""
        budget = (wasteful_cfg.n_samples_initial
                  + wasteful_cfg.n_generations * wasteful_cfg.n_offspring)
        outcome = _run_evolutionary_search(
            wasteful_cfg, tiny_horizon, wasteful_params, np.random.default_rng(0),
            progress_every=0,
        )
        assert outcome.n_evaluated <= budget
        # The wasteful staffing range guarantees rejections, so this run must have
        # charged some infeasible candidates against the budget.
        assert outcome.n_simulated < outcome.n_evaluated

    def test_evolutionary_search_never_evaluates_a_duplicate(
        self, wasteful_cfg, tiny_horizon, wasteful_params
    ):
        outcome = _run_evolutionary_search(
            wasteful_cfg, tiny_horizon, wasteful_params, np.random.default_rng(2),
            progress_every=0,
        )
        keys = [(e["N"], e["R"], e["F_freeze"],
                 tuple(e["A_dev"]), tuple(e["A_qa"]))
                for e in outcome.manifest.values()]
        assert len(keys) == len(set(keys))


class TestSampleRetention:
    def _outcome(self, cfg, horizon, params, samples):
        from dataclasses import replace
        return _run_random_search(
            replace(cfg, samples=samples), horizon, params, np.random.default_rng(1),
            cfg.n_samples, progress_every=0,
        )

    def test_frontier_policy_writes_one_file_per_frontier_config(
        self, wasteful_cfg, tiny_horizon, wasteful_params, tmp_path
    ):
        outcome = self._outcome(wasteful_cfg, tiny_horizon, wasteful_params, "frontier")
        frontier_keys = [outcome.aggs[i][0]
                         for i in pareto_frontier([a for _, _, a in outcome.aggs])]
        written = write_samples(outcome, tmp_path, frontier_keys)
        assert written == len(frontier_keys) > 0
        assert sorted(p.stem for p in (tmp_path / "samples").glob("*.json")) == \
            sorted(frontier_keys)

    def test_none_policy_keeps_no_results_in_memory(
        self, wasteful_cfg, tiny_horizon, wasteful_params, tmp_path
    ):
        outcome = self._outcome(wasteful_cfg, tiny_horizon, wasteful_params, "none")
        assert outcome.results == {}
        assert outcome.aggs, "objectives must still be aggregated"
        assert write_samples(outcome, tmp_path, []) == 0
        assert not (tmp_path / "samples").exists()

    def test_all_policy_buffers_every_simulated_candidate(
        self, wasteful_cfg, tiny_horizon, wasteful_params
    ):
        outcome = self._outcome(wasteful_cfg, tiny_horizon, wasteful_params, "all")
        assert set(outcome.results) == {key for key, _, _ in outcome.aggs}
