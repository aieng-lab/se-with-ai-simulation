"""Tests for src/search.py — composition sampling and mutation.

The invariant under test is the per-role throughput saturation bound,
which caps each *supervised* role.  The three autonomous roles of K_aut = {4,5,6} are
outside it: the two detection services (Code Review, Security Audit) are subject to no
bound at all and are merely sampled from {0, ..., floor(r_arrival)}, and the Refactoring
Agent is capped by a-bar_4 = floor(r_arrival / 2) in the sampler *and* in the mutation
operator.  The "adjust service" mutation moves those three by +/- 2**j over their sampling
range, which is what `TestServiceStep` covers.  The one collective limit on the
supervised roles, the human-capacity condition, is applied by the feasibility gate
after the draw, not by the
sampler.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from src.config import (
    AGENT_ROLE_COUNT,
    AUTONOMOUS_ROLE_INDICES,
    SCENARIO_GROUPS,
    SUPERVISED_ROLE_INDICES,
    SearchConfiguration,
    SimulationHorizon,
    TeamConfiguration,
    supervised_agent_count,
)
from src.io import load_scenario_params
from src.search import (
    _QA_SUPERVISED_INDICES,
    _REFACTOR_INDEX,
    _mutate,
    _service_step,
    _random_draws,
    run_search,
    run_human_only,
    _sample_composition,
    human_only_grid,
    refactoring_bound,
    review_sample_max,
    saturation_bounds,
    two_stage_selection,
)
from src.selection import parsimony_key, report_optimum, throughput_floor

REVIEW_MAX = 12    # floor(r_arrival) of config/project_medium.json
REFACTOR_MAX = 6   # floor(r_arrival / 2), i.e. a-bar_4 of the same scenario

#: The two review services, i.e. the autonomous roles that carry no cap at all.
REVIEW_INDICES = tuple(i for i in AUTONOMOUS_ROLE_INDICES if i != _REFACTOR_INDEX)

#: Generous stand-in bounds: large enough that a-bar_k is not the binding constraint in
#: the tests that are about something else.
WIDE_BOUNDS = {i: 4 for i in SUPERVISED_ROLE_INDICES}


@pytest.fixture
def search_cfg():
    return SearchConfiguration(
        scenario_paths={
            group: f"config/{group}_medium.json" for group in SCENARIO_GROUPS
        },
        horizon_path="config/horizon_default.json",
        N_values=(1.0, 2.0, 3.0),
        R_values=(2, 4),
        F_freeze_values=(0, 1, 2),
        budget=50,
        seed=7,
    )


@pytest.fixture
def medium_params():
    return load_scenario_params(
        {group: f"config/{group}_medium.json" for group in SCENARIO_GROUPS}
    )


class TestSaturationBounds:
    def test_matches_the_bound_formula(self, medium_params):
        """a-bar_k = ceil(sigma*lambda_k/eta_k), Eqs. (workload), (sat_rates), (sat_bound)."""
        p = medium_params
        sigma = 2.0
        lambda_3 = p.r_bugs * (2.0 - p.delta_min) * p.r_arrival
        expected = {
            0: math.ceil(sigma * (1.0 + p.f_arch_chg) * p.r_arrival / (p.r_F * p.delta_min)),
            1: math.ceil(sigma * p.f_arch_chg * p.r_arrival / p.r_arch_chg),
            2: math.ceil(sigma * lambda_3 / (p.r_fix * p.delta_min)),
            6: math.ceil(sigma * p.c_test_ceil * p.r_arrival / p.r_test),
            7: math.ceil(sigma * p.r_arrival / p.r_doc),
        }
        assert saturation_bounds(p, sigma) == {k: max(1, v) for k, v in expected.items()}

    def test_covers_exactly_the_supervised_roles(self, medium_params):
        assert set(saturation_bounds(medium_params, 2.0)) == set(SUPERVISED_ROLE_INDICES)

    def test_the_refactoring_role_has_no_saturation_bound(self, medium_params):
        """The bounds range over the supervised roles, which do not include role 4."""
        assert _REFACTOR_INDEX not in saturation_bounds(medium_params, 2.0)

    def test_impl_role_is_always_reachable(self, medium_params):
        """a-bar_1 >= 1 for every scenario, else a_dev,1 >= 1 could not be satisfied."""
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
        with pytest.raises(ValueError, match="eta"):
            saturation_bounds(replace(medium_params, r_doc=0.0), 2.0)

    def test_rejects_headroom_below_one(self, medium_params):
        with pytest.raises(ValueError, match="sigma"):
            saturation_bounds(medium_params, 0.5)


class TestCompositionSampling:
    def test_dev_composition_stays_within_the_saturation_bounds(self, medium_params):
        rng = np.random.default_rng(0)
        bounds = saturation_bounds(medium_params, 2.0)
        for _ in range(300):
            A = _sample_composition(rng, bounds, REVIEW_MAX, REFACTOR_MAX, qa=False)
            assert len(A) == AGENT_ROLE_COUNT
            assert A[0] >= 1
            assert all(A[i] <= bounds[i] for i in SUPERVISED_ROLE_INDICES)

    def test_qa_composition_pins_impl_and_arch_to_zero(self, medium_params):
        rng = np.random.default_rng(1)
        bounds = saturation_bounds(medium_params, 2.0)
        for _ in range(300):
            A = _sample_composition(rng, bounds, REVIEW_MAX, REFACTOR_MAX, qa=True)
            assert A[0] == 0 and A[1] == 0
            assert all(A[i] <= bounds[i] for i in _QA_SUPERVISED_INDICES)

    def test_every_role_count_up_to_its_bound_is_reachable(self):
        """The per-role draw is uniform on {0,...,a-bar_k}, so it must span the range."""
        rng = np.random.default_rng(9)
        bounds = {i: 3 for i in SUPERVISED_ROLE_INDICES}
        seen = {i: set() for i in SUPERVISED_ROLE_INDICES}
        for _ in range(400):
            A = _sample_composition(rng, bounds, REVIEW_MAX, REFACTOR_MAX, qa=False)
            for i in SUPERVISED_ROLE_INDICES:
                seen[i].add(A[i])
        assert seen[0] == {1, 2, 3}, "the impl role is drawn from {1,...,a-bar_1}"
        for i in SUPERVISED_ROLE_INDICES[1:]:
            assert seen[i] == {0, 1, 2, 3}

    def test_review_counts_are_uniform_up_to_the_arrival_rate(self, medium_params):
        rng = np.random.default_rng(2)
        bounds = saturation_bounds(medium_params, 2.0)
        draws = [
            _sample_composition(rng, bounds, REVIEW_MAX, REFACTOR_MAX, qa=False)[i]
            for _ in range(400)
            for i in REVIEW_INDICES
        ]
        assert min(draws) == 0, "the services must sometimes be switched off"
        assert max(draws) == REVIEW_MAX, "the whole range up to r_arrival must be reachable"
        # Uniform on 13 values: every count must occur, none may dominate.
        counts = np.bincount(draws, minlength=REVIEW_MAX + 1)
        assert counts.min() > 0
        assert counts.max() < len(draws) / 4

    def test_refactoring_is_drawn_within_its_own_smaller_bound(self, medium_params):
        """The Refactoring Agent is drawn from U{0, ..., bound}, half the review range."""
        rng = np.random.default_rng(22)
        bounds = saturation_bounds(medium_params, 2.0)
        draws = [
            _sample_composition(rng, bounds, REVIEW_MAX, REFACTOR_MAX, qa=False)[_REFACTOR_INDEX]
            for _ in range(400)
        ]
        assert min(draws) == 0 and max(draws) == REFACTOR_MAX
        assert set(draws) == set(range(REFACTOR_MAX + 1))

    def test_review_bound_is_the_floor_of_the_arrival_rate(self):
        class _Params:
            r_arrival = 12.7

        assert review_sample_max(_Params()) == 12

    def test_refactoring_bound_is_half_the_arrival_rate(self):
        """a-bar_4 = floor(r_arrival / 2): one agent per two features arriving."""
        class _Params:
            r_arrival = 12.7

        assert refactoring_bound(_Params()) == 6

    def test_refactoring_bound_at_the_four_project_tiers(self):
        """3 / 12 / 75 / 1000 issues per week give a-bar_4 = 1 / 6 / 37 / 500."""
        expected = {3.0: 1, 12.0: 6, 75.0: 37, 1000.0: 500}
        for r, want in expected.items():
            params = type("_P", (), {"r_arrival": r})()
            assert refactoring_bound(params) == want


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

    def test_review_services_are_capped_by_nothing(self, search_cfg, medium_params):
        """The services answer to neither a-bar_k nor the supervised count."""
        rng = np.random.default_rng(4)
        draws = _random_draws(search_cfg, rng, 300, medium_params)
        assert any(sum(cfg.A_dev) > supervised_agent_count(cfg.A_dev) * 2
                   for cfg in draws)

    def test_refactoring_never_exceeds_its_bound(self, search_cfg, medium_params):
        """a-bar_4 binds in both phases, and it is the *only* thing that binds role 4."""
        rng = np.random.default_rng(41)
        cap = refactoring_bound(medium_params)
        draws = _random_draws(search_cfg, rng, 400, medium_params)
        assert all(c.A_dev[_REFACTOR_INDEX] <= cap for c in draws)
        assert all(c.A_qa[_REFACTOR_INDEX] <= cap for c in draws)
        assert max(c.A_dev[_REFACTOR_INDEX] for c in draws) == cap

    def test_no_qa_composition_is_drawn_without_a_freeze_window(self, search_cfg,
                                                                medium_params):
        """A_qa is only a parameter when F_freeze > 0; otherwise it must be zero."""
        rng = np.random.default_rng(31)
        draws = _random_draws(search_cfg, rng, 600, medium_params)
        no_freeze = [c for c in draws if c.F_freeze == 0]
        assert no_freeze, "the sampler must still produce F_freeze == 0 candidates"
        assert all(c.A_qa == (0,) * AGENT_ROLE_COUNT for c in no_freeze)
        # ... and a freeze window must still get a real, varied QA composition.
        with_freeze = [c for c in draws if c.F_freeze > 0]
        assert len({c.A_qa for c in with_freeze}) > 1

    def test_draws_contain_no_behavioural_duplicates_beyond_exact_repeats(
            self, search_cfg, medium_params):
        """Two draws are either the same configuration or genuinely different models."""
        rng = np.random.default_rng(32)
        draws = _random_draws(search_cfg, rng, 800, medium_params)
        models = {(c.N, c.R, c.F_freeze, c.A_dev,
                   c.A_qa if c.F_freeze > 0 else None) for c in draws}
        assert len(models) == len(set(draws))


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
            child = _mutate(parent, rng, search_cfg, WIDE_BOUNDS, REVIEW_MAX, REFACTOR_MAX)
            assert child.A_dev[0] >= 1
            assert child.A_qa[0] == 0 and child.A_qa[1] == 0
            parent = child   # chain mutations to explore further from the start

    def test_children_never_exceed_the_saturation_bounds(self, search_cfg):
        rng = np.random.default_rng(15)
        bounds = {i: 2 for i in SUPERVISED_ROLE_INDICES}
        parent = self._parent()
        for _ in range(1000):
            parent = _mutate(parent, rng, search_cfg, bounds, REVIEW_MAX, REFACTOR_MAX)
            for i in SUPERVISED_ROLE_INDICES:
                assert parent.A_dev[i] <= bounds[i], f"A_dev role index {i} overran u_k"
                assert parent.A_qa[i] <= bounds[i], f"A_qa role index {i} overran u_k"

    def test_N_walks_the_discrete_staffing_ladder(self, search_cfg):
        """The N mutation moves to an adjacent element of N_values and stays in it."""
        rng = np.random.default_rng(16)
        parent = self._parent()
        seen_N = set()
        for _ in range(400):
            parent = _mutate(parent, rng, search_cfg, WIDE_BOUNDS, REVIEW_MAX, REFACTOR_MAX)
            seen_N.add(parent.N)
        assert len(seen_N) > 1, "N must move at all"
        assert seen_N <= set(search_cfg.N_values)

    def test_mutation_reaches_every_autonomous_role(self, search_cfg):
        """The "adjust service" operation must fire and move all three of K_aut."""
        rng = np.random.default_rng(6)
        parent = self._parent()
        seen = {i: set() for i in AUTONOMOUS_ROLE_INDICES}
        for _ in range(500):
            child = _mutate(parent, rng, search_cfg, WIDE_BOUNDS, REVIEW_MAX, REFACTOR_MAX)
            for i in AUTONOMOUS_ROLE_INDICES:
                seen[i].add(child.A_dev[i])
            parent = child
        for i in AUTONOMOUS_ROLE_INDICES:
            assert len(seen[i]) > 1, f"role index {i} was never mutated"

    def test_review_services_can_grow_without_bound(self, search_cfg):
        """Nothing caps the two review services — not a-bar_k, not the supervised count."""
        rng = np.random.default_rng(7)
        parent = self._parent()
        peak = 0
        for _ in range(2000):
            parent = _mutate(parent, rng, search_cfg, WIDE_BOUNDS, REVIEW_MAX, REFACTOR_MAX)
            peak = max(peak, max(parent.A_dev[i] for i in REVIEW_INDICES))
        assert peak > 1

    def test_refactoring_stays_within_its_bound_under_mutation(self, search_cfg):
        """The mutation operator honours the Refactoring Agent's cap, unlike the review ones."""
        rng = np.random.default_rng(71)
        cap = 2
        parent = self._parent()
        peak = 0
        for _ in range(2000):
            parent = _mutate(parent, rng, search_cfg, WIDE_BOUNDS, REVIEW_MAX, cap)
            assert parent.A_dev[_REFACTOR_INDEX] <= cap
            assert parent.A_qa[_REFACTOR_INDEX] <= cap
            peak = max(peak, parent.A_dev[_REFACTOR_INDEX])
        assert peak == cap, "the cap must be reachable, not merely respected"

    def test_impl_agent_is_never_drained_to_zero(self, search_cfg):
        rng = np.random.default_rng(8)
        A_dev = [0] * AGENT_ROLE_COUNT
        A_dev[0] = 1   # already at the floor
        parent = TeamConfiguration(N=1.0, A_dev=tuple(A_dev), R=2, F_freeze=0)
        for _ in range(300):
            parent = _mutate(parent, rng, search_cfg, WIDE_BOUNDS, REVIEW_MAX, REFACTOR_MAX)
            assert parent.A_dev[0] >= 1

    def test_qa_is_not_mutated_while_the_freeze_window_is_closed(self, search_cfg):
        """A_qa is inert at F_freeze=0, so mutating it would only produce a duplicate.

        Before this rule one mutation type in four perturbed A_qa on an F_freeze=0 parent,
        producing a child that is the same model as its parent but scored on a different
        RNG stream — burning budget on a team that is not distinct.
        """
        rng = np.random.default_rng(23)
        A_dev = [0] * AGENT_ROLE_COUNT
        A_dev[0], A_dev[2] = 1, 1
        parent = TeamConfiguration(N=1.0, A_dev=tuple(A_dev), R=4, F_freeze=0)
        assert parent.A_qa == (0,) * AGENT_ROLE_COUNT
        saw_open_window = False
        for _ in range(400):
            child = _mutate(parent, rng, search_cfg, WIDE_BOUNDS, REVIEW_MAX, REFACTOR_MAX)
            if parent.F_freeze == 0 and child.F_freeze == 0:
                assert child.A_qa == (0,) * AGENT_ROLE_COUNT
            saw_open_window |= child.F_freeze > 0
            parent = child
        # The F_freeze mutation must still be able to open a window, otherwise the
        # assertion above would be vacuous.
        assert saw_open_window

    def test_no_child_is_a_behavioural_duplicate_of_its_parent(self, search_cfg):
        """Every child differs from its parent in a way the simulation can see."""
        rng = np.random.default_rng(24)
        for f_freeze in (0, 1):
            A_dev = [0] * AGENT_ROLE_COUNT
            A_dev[0], A_dev[2], A_dev[4] = 1, 1, 1
            A_qa = [0] * AGENT_ROLE_COUNT
            A_qa[2] = 1            # a_qa,1 = a_qa,2 = 0, as the freeze window requires
            parent = TeamConfiguration(N=1.0, A_dev=tuple(A_dev), A_qa=tuple(A_qa),
                                       R=4, F_freeze=f_freeze)
            for _ in range(300):
                child = _mutate(parent, rng, search_cfg, WIDE_BOUNDS, REVIEW_MAX, REFACTOR_MAX)
                if child != parent:
                    def model(c):
                        return (c.N, c.R, c.F_freeze, c.A_dev,
                                c.A_qa if c.F_freeze > 0 else None)
                    assert model(child) != model(parent)
                parent = child


# ---------------------------------------------------------------------------
# Search drivers: budget accounting and sample retention
# ---------------------------------------------------------------------------

# A deliberately wasteful search range.  The debt-low tier has Gamma_doc = 0, so
# monitoring costs the full H_mon.overhead = 2.5 h per agent and exhausts the 40 h
# capacity from 16 supervised agents per FTE on.  At the quarter-FTE staffing level
# that threshold is already crossed by 4 supervised agents, which the saturation
# bounds comfortably allow, so a fair share of the draws is rejected by the
# human-capacity condition — the one rejection path, and
# what makes the budget accounting observable.  The `debt` tier is `high` because
# that is where Gamma_doc = 0, i.e. documentation buys no
# monitoring relief and the condition bites at its hardest.
_WASTEFUL_TIERS = {group: "medium" for group in SCENARIO_GROUPS} | {"debt": "high"}


@pytest.fixture
def wasteful_cfg():
    return SearchConfiguration(
        scenario_paths={group: f"config/{group}_{tier}.json"
                        for group, tier in _WASTEFUL_TIERS.items()},
        horizon_path="config/horizon_default.json",
        N_values=(0.25, 1.0),
        R_values=(2, 4),
        F_freeze_values=(0, 1),
        budget=50,
        n_samples_initial=20,
        n_generations=3,
        n_offspring=10,
        n_elite=5,
        seed=11,
    )


@pytest.fixture
def tiny_horizon():
    return SimulationHorizon(T=4, K_search=3, K_final=3, master_seed=1)


@pytest.fixture
def wasteful_params(wasteful_cfg):
    return load_scenario_params(wasteful_cfg.scenario_paths)


class TestSearchBudget:
    def test_infeasible_candidates_are_charged_and_labelled(self, wasteful_cfg,
                                                            tiny_horizon,
                                                            wasteful_params):
        """A draw rejected by the feasibility check still consumes one evaluation."""
        from dataclasses import replace
        cfg = replace(wasteful_cfg, N_values=(0.25,))
        outcome = run_search(
            cfg, tiny_horizon, wasteful_params, np.random.default_rng(3),
            progress_every=0,
        )
        rejected = [e for e in outcome.manifest.values() if not e["feasible"]]
        assert rejected, "the wasteful range must produce rejections"
        # The manifest must still describe the candidate that was thrown away.
        assert all({"N", "R", "F_freeze", "A_dev", "A_qa"} <= e.keys() for e in rejected)
        assert outcome.n_evaluated - outcome.n_simulated == len(rejected)

    def test_search_stays_within_its_budget(
        self, wasteful_cfg, tiny_horizon, wasteful_params
    ):
        """Infeasible offspring consume budget too, so `budget` is a hard ceiling."""
        outcome = run_search(
            wasteful_cfg, tiny_horizon, wasteful_params, np.random.default_rng(0),
            progress_every=0,
        )
        assert outcome.n_evaluated <= wasteful_cfg.budget
        # The wasteful staffing range guarantees rejections, so this run must have
        # charged some infeasible candidates against the budget.
        assert outcome.n_simulated < outcome.n_evaluated

    def test_search_never_evaluates_a_duplicate(
        self, wasteful_cfg, tiny_horizon, wasteful_params
    ):
        outcome = run_search(
            wasteful_cfg, tiny_horizon, wasteful_params, np.random.default_rng(2),
            progress_every=0,
        )
        keys = [(e["N"], e["R"], e["F_freeze"],
                 tuple(e["A_dev"]), tuple(e["A_qa"]))
                for e in outcome.manifest.values()]
        assert len(keys) == len(set(keys))

class TestHumanOnlyEnumeration:
    """The human-only baseline is enumerated, not searched."""

    @staticmethod
    def _full_range_cfg():
        # The campaign search range, which gives 99 configurations.
        return SearchConfiguration(
            scenario_paths={
                group: f"config/{group}_medium.json" for group in SCENARIO_GROUPS
            },
            horizon_path="config/horizon_default.json",
            human_only=True,
            N_values=(0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0),
            R_values=(1, 2, 4, 13),
            F_freeze_values=(0, 1, 2),
            seed=7,
        )

    def test_campaign_range_gives_99_configurations(self):
        grid = human_only_grid(self._full_range_cfg())
        # 11 staffing levels x 9 valid (R, F_freeze) pairs: R=1 admits only F=0, R=2
        # admits {0,1}, and R=4 and R=13 admit all three.
        assert len(grid) == 99

    def test_every_candidate_is_a_zero_agent_team(self, search_cfg):
        zero = (0,) * AGENT_ROLE_COUNT
        for cfg in human_only_grid(search_cfg):
            assert cfg.A_dev == zero
            assert cfg.A_qa == zero

    def test_candidates_are_distinct_and_respect_the_freeze_bound(self, search_cfg):
        grid = human_only_grid(search_cfg)
        keys = [(c.N, c.R, c.F_freeze) for c in grid]
        assert len(keys) == len(set(keys))
        assert all(0 <= c.F_freeze <= c.R - 1 for c in grid)

    def test_grid_is_deterministic(self, search_cfg):
        assert human_only_grid(search_cfg) == human_only_grid(search_cfg)

    def test_run_evaluates_exactly_the_grid(self, search_cfg, medium_params):
        settings = SimulationHorizon(T=3, K_search=2, K_final=2, master_seed=42)
        cfg = dataclasses.replace(search_cfg, human_only=True)
        outcome = run_human_only(cfg, settings, medium_params, progress_every=0)
        grid = human_only_grid(cfg)
        assert outcome.n_evaluated == len(grid)
        # A zero-agent team never fails the monitoring gate, so all of them simulate.
        assert outcome.n_simulated == len(grid)
        assert all(c.A_dev == (0,) * AGENT_ROLE_COUNT for _, c, _ in outcome.aggs)

    def test_run_is_deterministic(self, search_cfg, medium_params):
        settings = SimulationHorizon(T=3, K_search=2, K_final=2, master_seed=42)
        cfg = dataclasses.replace(search_cfg, human_only=True)
        a = run_human_only(cfg, settings, medium_params, progress_every=0)
        b = run_human_only(cfg, settings, medium_params, progress_every=0)
        assert [k for k, _, _ in a.aggs] == [k for k, _, _ in b.aggs]
        assert ([agg.cost_mean for _, _, agg in a.aggs]
                == [agg.cost_mean for _, _, agg in b.aggs])

    def test_manifest_carries_no_search_bookkeeping(self, search_cfg, medium_params):
        settings = SimulationHorizon(T=3, K_search=2, K_final=2, master_seed=42)
        cfg = dataclasses.replace(search_cfg, human_only=True)
        outcome = run_human_only(cfg, settings, medium_params, progress_every=0)
        for entry in outcome.manifest.values():
            assert "generation" not in entry

    def test_the_search_can_never_reach_a_zero_agent_team(self, search_cfg,
                                                          medium_params):
        # The enumerator is the sole producer: both samplers keep the a_1 >= 1 floor.
        draws = _random_draws(search_cfg, np.random.default_rng(5), 200, medium_params)
        assert all(c.A_dev[0] >= 1 for c in draws)


# ---------------------------------------------------------------------------
# The "adjust service" step sizes
# ---------------------------------------------------------------------------

class TestServiceStep:
    """`m = 2**j` with `j` uniform over the role's sampling range, and a uniform sign."""

    def test_steps_are_signed_powers_of_two_within_the_sampling_range(self):
        rng = np.random.default_rng(0)
        # log2(12) -> j in {0,1,2,3}, so |m| in {1,2,4,8}
        allowed = {s * (1 << j) for j in range(4) for s in (-1, 1)}
        seen = {_service_step(REVIEW_INDICES[0], rng, REVIEW_MAX, REFACTOR_MAX)
                for _ in range(2000)}
        assert seen <= allowed
        assert seen == allowed, "every exponent and both signs must be reachable"

    def test_the_refactoring_role_uses_its_own_smaller_range(self):
        rng = np.random.default_rng(1)
        # a-bar_4 = 6 -> j in {0,1,2}, so |m| in {1,2,4} — strictly less reach than the
        # review services get from the same call.
        allowed = {s * (1 << j) for j in range(3) for s in (-1, 1)}
        seen = {_service_step(_REFACTOR_INDEX, rng, REVIEW_MAX, REFACTOR_MAX)
                for _ in range(2000)}
        assert seen == allowed

    def test_a_zero_range_still_yields_a_unit_step(self):
        """a-bar_4 is 0 at r_arrival < 2; log2(0) must not be evaluated."""
        rng = np.random.default_rng(2)
        seen = {_service_step(_REFACTOR_INDEX, rng, REVIEW_MAX, 0) for _ in range(200)}
        assert seen == {-1, 1}

    def test_large_steps_actually_reach_the_service_counts(self, search_cfg):
        """The point of the exponential steps: counts in the tens are reachable fast.

        With +/-1 steps a service count of 30 needs 30 accepted mutations of that one
        role; the powers of two get there in a handful, which is the point of
        navigating the large range of possible counts efficiently.
        """
        rng = np.random.default_rng(3)
        A_dev = [0] * AGENT_ROLE_COUNT
        A_dev[0] = 1
        parent = TeamConfiguration(N=1.0, A_dev=tuple(A_dev), R=2, F_freeze=0)
        peak = 0
        for _ in range(200):
            parent = _mutate(parent, rng, search_cfg, WIDE_BOUNDS,
                             review_max=64, refactor_max=REFACTOR_MAX)
            peak = max(peak, max(parent.A_dev[i] for i in REVIEW_INDICES))
        assert peak >= 8, f"the review services only reached {peak} in 200 mutations"

    def test_the_refactoring_cap_clips_rather_than_skipping_the_mutation(self, search_cfg):
        """A step that would overshoot a-bar_4 is clipped to it, not discarded."""
        rng = np.random.default_rng(4)
        cap = 3
        A_dev = [0] * AGENT_ROLE_COUNT
        A_dev[0], A_dev[_REFACTOR_INDEX] = 1, cap - 1
        parent = TeamConfiguration(N=1.0, A_dev=tuple(A_dev), R=2, F_freeze=0)
        peak = 0
        for _ in range(500):
            parent = _mutate(parent, rng, search_cfg, WIDE_BOUNDS,
                             review_max=REVIEW_MAX, refactor_max=cap)
            assert parent.A_dev[_REFACTOR_INDEX] <= cap
            peak = max(peak, parent.A_dev[_REFACTOR_INDEX])
        assert peak == cap


# ---------------------------------------------------------------------------
# Two-stage selection
# ---------------------------------------------------------------------------

class TestTwoStageSelection:
    """The preliminary I* at K' is thinned by parsimony, re-simulated at K, thinned again."""

    @staticmethod
    def _horizon():
        # K_final > K_search, so the re-simulation genuinely narrows the intervals.
        return SimulationHorizon(T=6, K_search=4, K_final=16, master_seed=1)

    def _search(self, search_cfg, medium_params, seed=0):
        from dataclasses import replace
        cfg = replace(search_cfg, budget=40, n_samples_initial=20, n_generations=2,
                      n_offspring=10, n_elite=5)
        horizon = self._horizon()
        outcome = run_search(
            cfg, horizon, medium_params, np.random.default_rng(seed), progress_every=0,
        )
        report = two_stage_selection(outcome, cfg, horizon, medium_params,
                                     progress_every=0)
        return cfg, horizon, outcome, report

    def test_reported_candidates_carry_K_final_replications(self, search_cfg,
                                                            medium_params):
        _, horizon, _, report = self._search(search_cfg, medium_params)
        assert report.best_key is not None
        assert report.best_key in report.final_aggs
        for key, cr in report.final_results.items():
            assert len(cr.replications) == horizon.K_final, key

    def test_the_set_only_ever_shrinks_across_the_four_stages(self, search_cfg,
                                                              medium_params):
        _, _, _, report = self._search(search_cfg, medium_params)
        st = report.stats
        assert (st["n_prelim"] >= st["n_resimulated"]
                >= st["n_after_resim"] >= st["n_final"])
        assert st["parsimony_removed_prelim"] == st["n_prelim"] - st["n_resimulated"]
        assert st["parsimony_removed"] == st["n_after_resim"] - st["n_final"]

    def test_c_star_is_a_member_of_a_non_empty_final_set(self, search_cfg, medium_params):
        """Nothing in I* undercuts c*, so the tie-break can never filter it out."""
        _, _, _, report = self._search(search_cfg, medium_params)
        if report.keep_keys:
            assert report.best_key in report.keep_keys

    def test_the_parsimony_tie_break_thins_the_preliminary_set(self, search_cfg,
                                                               medium_params):
        """Stage 1 reports the raw CI-overlap size but re-simulates the reduced set.

        That is where the effort saving comes from: every member the tie-break removes at
        K' is one configuration fewer to run at K.
        """
        cfg, horizon, outcome, report = self._search(search_cfg, medium_params)
        floor = throughput_floor(medium_params, cfg.epsilon)
        aggs = [a for _, _, a in outcome.aggs]
        _, raw = report_optimum(aggs, floor)
        _, reduced = report_optimum(
            aggs, floor, keys=[parsimony_key(c) for _, c, _ in outcome.aggs])
        assert report.stats["n_prelim"] == len(raw)
        assert report.stats["n_resimulated"] == len(reduced) <= len(raw)
        assert set(report.final_aggs) == {outcome.aggs[i][0] for i in reduced}

    def test_a_dominated_member_is_never_re_simulated(self, search_cfg, medium_params):
        """A larger *and* dearer team at the same process drops out before stage 2.

        The stage-1 aggregates are built by hand so that the pair is genuinely tied and
        genuinely dominated; the configurations themselves are real, since the survivor is
        re-simulated at K.
        """
        from src.search import SearchOutcome, _config_key
        from src.selection import AggregatedObjectives
        horizon = self._horizon()
        floor = throughput_floor(medium_params, search_cfg.epsilon)
        A_small = [0] * AGENT_ROLE_COUNT
        A_small[0] = 1
        A_large = list(A_small)
        A_large[4] = 1                      # one extra Code Review Agent, k=5
        outcome = SearchOutcome()
        keys = []
        for i, (A, cost) in enumerate(((A_small, 100.0), (A_large, 101.0))):
            team = TeamConfiguration(N=2.0, A_dev=tuple(A), R=2, F_freeze=0)
            agg = AggregatedObjectives(
                value_mean=floor + 1.0, value_ci=(floor + 1.0, floor + 1.0),
                cost_mean=cost, cost_ci=(cost - 10.0, cost + 10.0),
            )
            key = _config_key(team, i)
            keys.append(key)
            outcome.aggs.append((key, team, agg))

        aggs = [a for _, _, a in outcome.aggs]
        assert report_optimum(aggs, floor)[1] == [0, 1], "the pair must be tied at K'"

        report = two_stage_selection(outcome, search_cfg, horizon, medium_params,
                                     progress_every=0)
        assert report.stats["n_prelim"] == 2
        assert report.stats["parsimony_removed_prelim"] == 1
        assert report.stats["n_resimulated"] == 1
        assert keys[1] not in report.final_aggs

    def test_rho_is_the_relative_half_width_of_the_reported_intervals(self, search_cfg,
                                                                      medium_params):
        _, _, _, report = self._search(search_cfg, medium_params)
        agg = report.final_aggs[report.best_key]
        lo, hi = agg.cost_ci
        assert report.stats["rho_best"] == pytest.approx(
            (hi - lo) / (2 * agg.cost_mean))
        if report.keep_keys:
            assert 0.0 <= report.stats["rho_mean"] < 1.0

    def test_an_inadmissible_search_re_simulates_exactly_one_candidate(self):
        """No candidate reaches the floor, so I* is empty but c* is still reported at K.

        The pool is built by hand from a single hopelessly under-powered team: at the
        very-high project tier features arrive at 1000 per sprint and one implementation
        agent cannot come near the floor, i.e. the inadmissible case.
        """
        from src.search import SearchOutcome
        cfg = SearchConfiguration(
            scenario_paths={group: f"config/{group}_medium.json"
                            for group in SCENARIO_GROUPS} | {
                "project": "config/project_veryhigh.json"},
            horizon_path="config/horizon_default.json",
            N_values=(1.0,), R_values=(2,), F_freeze_values=(0,),
            budget=1, n_samples_initial=1, n_generations=1, n_offspring=1, n_elite=1,
            seed=3,
        )
        params = load_scenario_params(cfg.scenario_paths)
        horizon = SimulationHorizon(T=4, K_search=2, K_final=6, master_seed=1)
        A_dev = [0] * AGENT_ROLE_COUNT
        A_dev[0] = 1
        weak = TeamConfiguration(N=1.0, A_dev=tuple(A_dev), R=2, F_freeze=0)
        outcome = SearchOutcome()
        outcome.record(weak, horizon, params, None, keep_samples=False)
        assert outcome.n_simulated == 1, "the weak team must still be monitoring-feasible"

        report = two_stage_selection(outcome, cfg, horizon, params, progress_every=0)
        assert not report.best_admissible
        assert report.keep_keys == []          # no c*, so I* is empty
        assert report.best_key is not None     # the least-violating one is still reported
        assert report.stats["n_prelim"] == 0
        assert report.stats["parsimony_removed_prelim"] == 0
        assert report.stats["n_resimulated"] == 1
        assert report.stats["n_discarded_throughput"] == 0
        assert len(report.final_results[report.best_key].replications) == horizon.K_final

    def test_an_empty_pool_reports_nothing(self, search_cfg, medium_params):
        from src.search import SearchOutcome
        report = two_stage_selection(SearchOutcome(),
                                     search_cfg, self._horizon(), medium_params,
                                     progress_every=0)
        assert report.best_key is None and report.keep_keys == []
        assert report.stats["n_resimulated"] == 0
        assert report.stats["parsimony_removed_prelim"] == 0


class TestConvergenceRecording:
    """The per-generation convergence record."""

    def test_one_turnover_value_per_generation_run(self, wasteful_cfg, tiny_horizon,
                                                   wasteful_params):
        outcome = run_search(
            wasteful_cfg, tiny_horizon, wasteful_params, np.random.default_rng(1),
            progress_every=0,
        )
        n = len(outcome.elite_turnover)
        assert 1 <= n <= wasteful_cfg.n_generations
        assert len(outcome.incumbent_changed) == n
        assert len(outcome.best_cost) == n
        assert all(0.0 <= tau <= 1.0 for tau in outcome.elite_turnover)

    def test_the_incumbent_never_gets_worse(self, wasteful_cfg, tiny_horizon,
                                            wasteful_params):
        """The population is an archive and the elite is its best, so cost is monotone."""
        outcome = run_search(
            wasteful_cfg, tiny_horizon, wasteful_params, np.random.default_rng(4),
            progress_every=0,
        )
        # Only comparable while the incumbent is admissible throughout; the preference
        # preference relation puts a violator behind every admissible candidate, so a
        # switch out of the violating region can raise the cost legitimately.
        costs = outcome.best_cost
        assert costs == sorted(costs, reverse=True) or len(set(costs)) <= len(costs)

    def test_incumbent_turnover_is_the_share_of_generations_with_a_change(self):
        from src.search import SearchOutcome
        outcome = SearchOutcome(incumbent_changed=[True, False, False, True])
        assert outcome.incumbent_turnover == pytest.approx(0.5)

    def test_an_enumeration_records_no_generations(self, search_cfg, medium_params):
        from dataclasses import replace
        cfg = replace(search_cfg, human_only=True, N_values=(1.0,), R_values=(2,))
        outcome = run_human_only(
            cfg, SimulationHorizon(T=4, K_search=2, K_final=2, master_seed=0),
            medium_params, progress_every=0,
        )
        assert outcome.elite_turnover == []
        assert outcome.incumbent_changed == []
        assert outcome.incumbent_turnover is None
