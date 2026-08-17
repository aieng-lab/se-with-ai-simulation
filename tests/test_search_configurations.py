"""Tests for src/search_configurations.py — composition sampling and mutation.

The invariant under test is the throughput saturation bound u_k (paper §3.13.1),
which caps each supervised role; the background detection services (Code Review,
Security Audit) are subject to no bound at all and are sampled separately,
uniformly from {0, ..., floor(r_arrival)}.  The one collective limit on the
supervised roles, the human-capacity condition of Eq. (feas_monitoring), is applied
by the feasibility gate after the draw, not by the sampler.
"""

from __future__ import annotations

import dataclasses
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
from src.search_configurations import (
    _QA_SUPERVISED_INDICES,
    _composition_neighbours,
    _mutate,
    _random_draws,
    _run_descent,
    _run_evolutionary_search,
    _run_human_only,
    _sample_dev_composition,
    _sample_qa_composition,
    background_count_max,
    human_only_grid,
    neighbourhood,
    saturation_bounds,
    write_samples,
)
from src.selection import indistinguishable_set, parsimony_key, throughput_floor

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
        strategy="evolutionary",
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
            child = _mutate(parent, rng, search_cfg, WIDE_BOUNDS)
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
                child = _mutate(parent, rng, search_cfg, WIDE_BOUNDS)
                if child != parent:
                    def model(c):
                        return (c.N, c.R, c.F_freeze, c.A_dev,
                                c.A_qa if c.F_freeze > 0 else None)
                    assert model(child) != model(parent)
                parent = child


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
        strategy="evolutionary",
        N_values=(0.25, 1.0),
        R_values=(2, 4),
        F_freeze_values=(0, 1),
        budget=50,
        n_samples_initial=20,
        n_generations=3,
        n_offspring=10,
        n_elite=5,
        n_starts=2,
        seed=11,
    )


@pytest.fixture
def tiny_horizon():
    return SimulationHorizon(T=4, K=3, master_seed=1)


@pytest.fixture
def wasteful_params(wasteful_cfg):
    return load_scenario_params(wasteful_cfg.scenario_paths)


class TestNeighbourhood:
    """The enumerated neighbourhood of §3.11.5 must match the mutation operator's reach."""

    def _cfg(self, F_freeze=1):
        return TeamConfiguration(N=2.0, A_dev=(2, 1, 1, 0, 3, 0, 1, 0), R=4,
                                 F_freeze=F_freeze,
                                 A_qa=(0, 0, 2, 1, 1, 0, 0, 1) if F_freeze else None)

    def test_neighbours_are_distinct_and_exclude_the_parent(self, search_cfg):
        cfg = self._cfg()
        nbs = neighbourhood(cfg, search_cfg, WIDE_BOUNDS)
        assert cfg not in nbs
        assert len(nbs) == len(set(nbs))

    def test_every_mutation_lands_in_the_neighbourhood(self, search_cfg):
        """Whatever `_mutate` can reach in one step must be enumerated (§3.11.5)."""
        cfg = self._cfg()
        nbs = set(neighbourhood(cfg, search_cfg, WIDE_BOUNDS))
        rng = np.random.default_rng(0)
        reached = {_mutate(cfg, rng, search_cfg, WIDE_BOUNDS) for _ in range(600)}
        reached.discard(cfg)      # the operator returns the parent when it gives up
        assert reached <= nbs
        # The sampling must have actually explored, or the containment is vacuous.
        assert len(reached) > 10

    def test_staffing_moves_are_the_adjacent_grid_values(self, search_cfg):
        cfg = self._cfg()
        nbs = neighbourhood(cfg, search_cfg, WIDE_BOUNDS)
        # search_cfg.N_values == (1.0, 2.0, 3.0) and cfg.N == 2.0
        n_moves = {nb.N for nb in nbs if nb.A_dev == cfg.A_dev and nb.A_qa == cfg.A_qa
                   and nb.F_freeze == cfg.F_freeze}
        assert n_moves == {1.0, 3.0}

    def test_release_cadence_is_never_changed(self, search_cfg):
        cfg = self._cfg()
        assert all(nb.R == cfg.R for nb in neighbourhood(cfg, search_cfg, WIDE_BOUNDS))

    def test_freeze_length_stays_in_range(self, search_cfg):
        cfg = self._cfg()
        nbs = neighbourhood(cfg, search_cfg, WIDE_BOUNDS)
        assert all(0 <= nb.F_freeze <= cfg.R - 1 for nb in nbs)
        assert {nb.F_freeze for nb in nbs} >= {0, 1, 2}

    def test_saturation_bounds_are_respected(self, search_cfg):
        cfg = self._cfg()
        bounds = {i: 2 for i in SUPERVISED_ROLE_INDICES}
        tight = TeamConfiguration(N=2.0, A_dev=(2, 2, 0, 0, 1, 0, 0, 0), R=4, F_freeze=0)
        for nb in neighbourhood(tight, search_cfg, bounds):
            for i in SUPERVISED_ROLE_INDICES:
                assert nb.A_dev[i] <= bounds[i]

    def test_qa_block_is_absent_without_a_freeze_window(self, search_cfg):
        """A_qa is canonically zero at F_freeze == 0, so mutating it is a no-op."""
        cfg = self._cfg(F_freeze=0)
        nbs = neighbourhood(cfg, search_cfg, WIDE_BOUNDS)
        # Every neighbour that keeps F_freeze == 0 keeps the canonical zero A_qa.
        assert all(nb.A_qa == (0,) * AGENT_ROLE_COUNT
                   for nb in nbs if nb.F_freeze == 0)
        # Opening a window is available, and lands on the canonical zero A_qa.
        opened = [nb for nb in nbs if nb.F_freeze == 1]
        assert opened and all(nb.A_qa == (0,) * AGENT_ROLE_COUNT for nb in opened)

    def test_qa_implementation_roles_stay_at_zero(self, search_cfg):
        cfg = self._cfg()
        for nb in neighbourhood(cfg, search_cfg, WIDE_BOUNDS):
            if nb.F_freeze > 0:
                assert nb.A_qa[0] == 0 and nb.A_qa[1] == 0

    def test_dev_implementation_role_never_drains_to_zero(self, search_cfg):
        cfg = self._cfg()
        assert all(nb.A_dev[0] >= 1
                   for nb in neighbourhood(cfg, search_cfg, WIDE_BOUNDS))

    def test_composition_neighbours_change_exactly_one_step(self):
        A = (2, 1, 1, 0, 3, 0, 1, 0)
        for out in _composition_neighbours(A, SUPERVISED_ROLE_INDICES, WIDE_BOUNDS,
                                           min_impl=True):
            delta = sum(abs(a - b) for a, b in zip(A, out))
            assert delta in (1, 2)    # 1 for add/remove/service, 2 for a transfer
            assert sum(out) - sum(A) in (-1, 0, 1)


class TestSearchBudget:
    def test_infeasible_candidates_are_charged_and_labelled(self, wasteful_cfg,
                                                            tiny_horizon,
                                                            wasteful_params):
        """A draw rejected by Eq. (feas_monitoring) still consumes one evaluation (§4.2)."""
        from dataclasses import replace
        cfg = replace(wasteful_cfg, N_values=(0.25,))
        outcome = _run_evolutionary_search(
            cfg, tiny_horizon, wasteful_params, np.random.default_rng(3),
            progress_every=0,
        )
        rejected = [e for e in outcome.manifest.values() if not e["feasible"]]
        assert rejected, "the wasteful range must produce rejections"
        assert all(e["failed_step"] == 1 for e in rejected)
        # The manifest must still describe the candidate that was thrown away.
        assert all({"N", "R", "F_freeze", "A_dev", "A_qa"} <= e.keys() for e in rejected)
        assert outcome.n_evaluated - outcome.n_simulated == len(rejected)

    def test_evolutionary_search_stays_within_its_budget(
        self, wasteful_cfg, tiny_horizon, wasteful_params
    ):
        """Infeasible offspring consume budget too, so §4.2's E is a hard ceiling."""
        outcome = _run_evolutionary_search(
            wasteful_cfg, tiny_horizon, wasteful_params, np.random.default_rng(0),
            progress_every=0,
        )
        assert outcome.n_evaluated <= wasteful_cfg.budget
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

    def test_descent_is_not_bounded_by_the_budget(self, wasteful_cfg, tiny_horizon,
                                                  wasteful_params):
        """`budget` is the evolutionary strategy's; the descent runs to convergence (§4.2).

        The fixture's budget of 50 is far below what two restarts of full neighbourhood
        enumeration cost, so a descent that still honoured it would stop short here.
        """
        outcome = _run_descent(
            wasteful_cfg, tiny_horizon, wasteful_params, np.random.default_rng(0),
            progress_every=0,
        )
        assert outcome.n_evaluated > wasteful_cfg.budget

    def test_descent_never_evaluates_a_duplicate(self, wasteful_cfg, tiny_horizon,
                                                 wasteful_params):
        """`seen` makes a revisited neighbour free, so no configuration is charged twice."""
        outcome = _run_descent(
            wasteful_cfg, tiny_horizon, wasteful_params, np.random.default_rng(5),
            progress_every=0,
        )
        keys = [(e["N"], e["R"], e["F_freeze"],
                 tuple(e["A_dev"]), tuple(e["A_qa"]))
                for e in outcome.manifest.values()]
        assert len(keys) == len(set(keys))

    def test_descent_terminates_on_a_range_that_guarantees_duplicate_seeds(
        self, tiny_horizon
    ):
        """A re-drawn seed costs nothing, so the seed loop needs its own attempt cap.

        The range here is one staffing level, one cadence and no freeze window, so the
        seed draws cannot help but repeat candidates that have already been scored.  With
        no budget left to bound the restart, `_MAX_SEED_DRAWS` is the only thing that ends
        a seed loop whose draws are all infeasible; the test passing at all is the
        assertion.
        """
        cfg = SearchConfiguration(
            scenario_paths={group: f"config/{group}_{tier}.json"
                            for group, tier in _WASTEFUL_TIERS.items()},
            horizon_path="config/horizon_default.json",
            strategy="descent",
            N_values=(0.25,), R_values=(1,), F_freeze_values=(0,),
            n_starts=2, seed=1,
        )
        params = load_scenario_params(cfg.scenario_paths)
        outcome = _run_descent(cfg, tiny_horizon, params, np.random.default_rng(0),
                               progress_every=0)
        # Every rejection is still recorded, and nothing is charged twice.
        keys = [(e["N"], e["R"], e["F_freeze"], tuple(e["A_dev"]), tuple(e["A_qa"]))
                for e in outcome.manifest.values()]
        assert len(keys) == len(set(keys))
        assert any(not e["feasible"] for e in outcome.manifest.values()), \
            "the wasteful range must reject at least one candidate"

    def test_descent_records_every_restart(self, wasteful_cfg, tiny_horizon,
                                           wasteful_params):
        outcome = _run_descent(
            wasteful_cfg, tiny_horizon, wasteful_params, np.random.default_rng(0),
            progress_every=0,
        )
        restarts = {e["restart"] for e in outcome.manifest.values()}
        assert restarts == set(range(wasteful_cfg.n_starts))
        assert all("step" in e for e in outcome.manifest.values())

    def test_descent_ends_at_a_local_optimum(self, tiny_horizon):
        """No neighbour of the final incumbent of a completed restart is preferred."""
        from dataclasses import replace
        from src.selection import best_index, prefers
        cfg = SearchConfiguration(
            scenario_paths={group: f"config/{group}_medium.json"
                            for group in SCENARIO_GROUPS},
            horizon_path="config/horizon_default.json",
            strategy="descent",
            N_values=(1.0, 2.0), R_values=(2,), F_freeze_values=(0, 1),
            n_starts=1, seed=4,
        )
        params = load_scenario_params(cfg.scenario_paths)
        outcome = _run_descent(cfg, tiny_horizon, params, np.random.default_rng(0),
                               progress_every=0)

        floor = throughput_floor(params, cfg.epsilon)
        aggs = [a for _, _, a in outcome.aggs]
        by_config = {c: a for _, c, a in outcome.aggs}
        incumbent = outcome.aggs[best_index(aggs, floor)][1]
        bounds = saturation_bounds(params, cfg.sigma)
        for nb in neighbourhood(incumbent, cfg, bounds):
            agg = by_config.get(nb)
            if agg is not None:
                assert not prefers(agg, by_config[incumbent], floor)


class TestSampleRetention:
    def _outcome(self, cfg, horizon, params, samples):
        from dataclasses import replace
        return _run_evolutionary_search(
            replace(cfg, samples=samples), horizon, params, np.random.default_rng(1),
            progress_every=0,
        )

    def test_best_policy_writes_one_file_per_reported_config(
        self, wasteful_cfg, tiny_horizon, wasteful_params, tmp_path
    ):
        outcome = self._outcome(wasteful_cfg, tiny_horizon, wasteful_params, "best")
        floor = throughput_floor(wasteful_params, wasteful_cfg.epsilon)
        keep = indistinguishable_set(
            [a for _, _, a in outcome.aggs], floor,
            keys=[parsimony_key(c) for _, c, _ in outcome.aggs],
        )
        keys = [outcome.aggs[i][0] for i in keep]
        written = write_samples(outcome, tmp_path, keys)
        assert written == len(keys) > 0
        assert sorted(p.stem for p in (tmp_path / "samples").glob("*.json")) == \
            sorted(keys)

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


class TestHumanOnlyEnumeration:
    """The human-only baseline of paper §4.1 is enumerated, not searched."""

    @staticmethod
    def _full_range_cfg():
        # The §4.2 search range, which the paper says gives 99 configurations.
        return SearchConfiguration(
            scenario_paths={
                group: f"config/{group}_medium.json" for group in SCENARIO_GROUPS
            },
            horizon_path="config/horizon_default.json",
            strategy="human_only",
            N_values=(0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0),
            R_values=(1, 2, 4, 13),
            F_freeze_values=(0, 1, 2),
            seed=7,
        )

    def test_paper_range_gives_99_configurations(self):
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
        settings = SimulationHorizon(T=3, K=2, master_seed=42)
        cfg = dataclasses.replace(search_cfg, strategy="human_only")
        outcome = _run_human_only(cfg, settings, medium_params,
                                  np.random.default_rng(0), progress_every=0)
        grid = human_only_grid(cfg)
        assert outcome.n_evaluated == len(grid)
        # A zero-agent team never fails the monitoring gate, so all of them simulate.
        assert outcome.n_simulated == len(grid)
        assert all(c.A_dev == (0,) * AGENT_ROLE_COUNT for _, c, _ in outcome.aggs)

    def test_run_ignores_the_rng(self, search_cfg, medium_params):
        # Deterministic: `rng` is accepted only to match the two search strategies.
        settings = SimulationHorizon(T=3, K=2, master_seed=42)
        cfg = dataclasses.replace(search_cfg, strategy="human_only")
        a = _run_human_only(cfg, settings, medium_params, np.random.default_rng(0),
                            progress_every=0)
        b = _run_human_only(cfg, settings, medium_params, np.random.default_rng(999),
                            progress_every=0)
        assert [k for k, _, _ in a.aggs] == [k for k, _, _ in b.aggs]
        assert ([agg.cost_mean for _, _, agg in a.aggs]
                == [agg.cost_mean for _, _, agg in b.aggs])

    def test_manifest_carries_no_search_bookkeeping(self, search_cfg, medium_params):
        settings = SimulationHorizon(T=3, K=2, master_seed=42)
        cfg = dataclasses.replace(search_cfg, strategy="human_only")
        outcome = _run_human_only(cfg, settings, medium_params,
                                  np.random.default_rng(0), progress_every=0)
        for entry in outcome.manifest.values():
            assert "generation" not in entry
            assert "restart" not in entry and "step" not in entry

    def test_the_searches_can_never_reach_a_zero_agent_team(self, search_cfg,
                                                            medium_params):
        # The enumerator is the sole producer: both samplers keep the a_1 >= 1 floor.
        draws = _random_draws(search_cfg, np.random.default_rng(5), 200, medium_params)
        assert all(c.A_dev[0] >= 1 for c in draws)
