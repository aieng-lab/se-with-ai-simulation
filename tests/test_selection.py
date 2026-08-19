"""Tests for src/selection.py — the objective, the constraint, and what gets reported.

Three things are under test.  The bootstrap aggregation of §3.8: means over K replications
with 95 % percentile CIs on the objective and the constrained quantity.  The preference
relation of §3.11.3 (Deb's feasibility rule): a total preorder over (constraint violation,
cost), which is what makes `rank` a sort and `c*` its first element.  And the reporting
rule of §3.10: `c*` plus the set of admissible configurations whose cost CI overlaps its
own, reduced by the parsimony tie-break.
"""

from __future__ import annotations

import dataclasses

import pytest

from src.config import TeamConfiguration
from src.selection import (
    AggregatedObjectives,
    aggregate_config_result,
    best_index,
    indistinguishable_set,
    parsimony_dominates,
    parsimony_key,
    prefers,
    rank,
    report_optimum,
    satisfies_throughput,
    throughput_floor,
    violation,
)
from src.simulation import ConfigResult, ReplicationResult, run_configuration


def _agg(value=10.0, cost=100.0, ci=None, **extra) -> AggregatedObjectives:
    """An AggregatedObjectives with a degenerate CI unless one is given."""
    lo, hi = ci if ci is not None else (cost, cost)
    return AggregatedObjectives(
        value_mean=value, value_ci=(value, value),
        cost_mean=cost, cost_ci=(lo, hi), **extra,
    )


# ---------------------------------------------------------------------------
# Throughput constraint (§3.9)
# ---------------------------------------------------------------------------

class TestThroughputConstraint:
    def test_floor_is_the_tolerated_fraction_of_the_arrival_rate(self, base_params):
        p = dataclasses.replace(base_params, r_arrival=50.0)
        assert throughput_floor(p, 0.02) == pytest.approx(49.0)
        assert throughput_floor(p, 0.0) == pytest.approx(50.0)

    @pytest.mark.parametrize("epsilon", [-0.01, 1.0, 2.0])
    def test_floor_rejects_a_tolerance_outside_the_unit_interval(self, base_params,
                                                                 epsilon):
        with pytest.raises(ValueError, match=r"epsilon must be in \[0, 1\)"):
            throughput_floor(base_params, epsilon)

    def test_violation_is_the_shortfall_and_zero_when_admissible(self):
        assert violation(_agg(value=8.0), floor=10.0) == pytest.approx(2.0)
        assert violation(_agg(value=10.0), floor=10.0) == pytest.approx(0.0)
        assert violation(_agg(value=12.0), floor=10.0) == pytest.approx(0.0)

    def test_the_floor_itself_is_admissible(self):
        """Eq. (throughput_constraint) is a weak inequality."""
        assert satisfies_throughput(_agg(value=10.0), floor=10.0)
        assert not satisfies_throughput(_agg(value=9.999), floor=10.0)


# ---------------------------------------------------------------------------
# Preference relation (§3.11.3, Deb's feasibility rule)
# ---------------------------------------------------------------------------

class TestPreferenceRelation:
    def test_admissible_beats_inadmissible_however_cheap(self):
        """Rule 2: satisfying the constraint always wins, even at a far higher cost."""
        admissible = _agg(value=10.0, cost=1e9)
        cheap_but_short = _agg(value=9.0, cost=1.0)
        assert prefers(admissible, cheap_but_short, floor=10.0)
        assert not prefers(cheap_but_short, admissible, floor=10.0)

    def test_two_admissible_are_separated_by_cost(self):
        """Rule 3: among admissible configurations, only cost matters."""
        cheap = _agg(value=10.0, cost=50.0)
        pricey = _agg(value=99.0, cost=60.0)   # far more value, but dearer
        assert prefers(cheap, pricey, floor=10.0)
        assert not prefers(pricey, cheap, floor=10.0)

    def test_two_violators_are_separated_by_the_smaller_violation(self):
        """Rule 1: the closer to admissibility (equivalently, the higher value) wins."""
        closer = _agg(value=9.0, cost=1e6)
        further = _agg(value=1.0, cost=1.0)    # much cheaper, but far short
        assert prefers(closer, further, floor=10.0)
        assert not prefers(further, closer, floor=10.0)

    def test_cost_never_separates_two_violators_with_equal_shortfall(self):
        """The lexicographic key falls through to cost only at equal violation."""
        a = _agg(value=9.0, cost=10.0)
        b = _agg(value=9.0, cost=20.0)
        assert prefers(a, b, floor=10.0)
        assert not prefers(b, a, floor=10.0)

    def test_the_relation_is_strict(self):
        identical = _agg()
        assert not prefers(identical, _agg(), floor=10.0)
        assert not prefers(_agg(), identical, floor=10.0)

    def test_relation_is_transitive_over_a_random_pool(self):
        """A total preorder, unlike the CI-aware multi-objective rule it replaced."""
        import numpy as np
        rng = np.random.default_rng(0)
        pool = [_agg(value=float(rng.uniform(5, 15)), cost=float(rng.uniform(1, 1e4)))
                for _ in range(40)]
        for a in pool:
            for b in pool:
                for c in pool:
                    if prefers(a, b, 10.0) and prefers(b, c, 10.0):
                        assert prefers(a, c, 10.0)

    def test_rank_orders_best_first_and_is_stable(self):
        pool = [_agg(value=10.0, cost=30.0),   # 0: admissible, dearest
                _agg(value=5.0, cost=1.0),     # 1: violates by 5
                _agg(value=10.0, cost=10.0),   # 2: admissible, cheapest
                _agg(value=9.0, cost=1.0),     # 3: violates by 1
                _agg(value=10.0, cost=10.0)]   # 4: ties with 2
        assert rank(pool, floor=10.0) == [2, 4, 0, 3, 1]

    def test_best_index_is_the_first_of_rank(self):
        import numpy as np
        rng = np.random.default_rng(1)
        pool = [_agg(value=float(rng.uniform(5, 15)), cost=float(rng.uniform(1, 1e4)))
                for _ in range(30)]
        assert best_index(pool, 10.0) == rank(pool, 10.0)[0]

    def test_best_index_on_an_empty_pool(self):
        assert best_index([], 10.0) is None

    def test_best_index_falls_back_to_the_least_violating(self):
        """When nothing is admissible, c* is the closest to it — never a cost minimum."""
        pool = [_agg(value=1.0, cost=1.0), _agg(value=9.0, cost=1e6)]
        best = best_index(pool, floor=10.0)
        assert best == 1
        assert not satisfies_throughput(pool[best], 10.0)

    def test_the_reported_means_never_affect_the_relation(self):
        """The cost drivers and contribution counts are reporting only (§3.8)."""
        plain = _agg(value=10.0, cost=100.0)
        loaded = _agg(value=10.0, cost=100.0, cost_quality_mean=99.0,
                      cost_human_mean=1.0, cost_agent_mean=0.0,
                      actions_mean=(1.0,) * 8, impl_human_mean=7.0,
                      fixes_agent_mean=3.0)
        assert not prefers(plain, loaded, 10.0)
        assert not prefers(loaded, plain, 10.0)


# ---------------------------------------------------------------------------
# Indistinguishable set (§3.10, Eq. topset)
# ---------------------------------------------------------------------------

class TestIndistinguishableSet:
    def test_overlapping_cis_are_retained_and_separated_ones_are_not(self):
        pool = [_agg(value=10.0, cost=100.0, ci=(90.0, 110.0)),   # 0: c*
                _agg(value=10.0, cost=115.0, ci=(105.0, 125.0)),  # 1: overlaps 0
                _agg(value=10.0, cost=300.0, ci=(290.0, 310.0))]  # 2: separated
        assert indistinguishable_set(pool, floor=10.0) == [0, 1]

    def test_best_is_a_member_without_the_tie_break(self):
        pool = [_agg(value=10.0, cost=100.0, ci=(99.0, 101.0))]
        assert indistinguishable_set(pool, 10.0) == [0]

    def test_inadmissible_configurations_are_excluded_even_when_tied(self):
        """I* is defined over admissible candidates only (Eq. topset)."""
        pool = [_agg(value=10.0, cost=100.0, ci=(90.0, 110.0)),   # 0: c*
                _agg(value=5.0, cost=100.0, ci=(90.0, 110.0))]    # 1: same cost, short
        assert indistinguishable_set(pool, floor=10.0) == [0]

    def test_an_inadmissible_pool_yields_an_empty_set(self):
        """§3.10: with nothing admissible there is no c*, so I* is empty.

        `report_optimum` still hands back the least-violating candidate beside it — see
        TestReportOptimum — but the set itself has no members to speak of.
        """
        pool = [_agg(value=9.0, cost=100.0, ci=(90.0, 110.0)),
                _agg(value=1.0, cost=100.0, ci=(90.0, 110.0))]
        assert indistinguishable_set(pool, floor=10.0) == []

    def test_touching_cis_count_as_overlapping(self):
        pool = [_agg(value=10.0, cost=100.0, ci=(90.0, 110.0)),
                _agg(value=10.0, cost=130.0, ci=(110.0, 150.0))]
        assert indistinguishable_set(pool, 10.0) == [0, 1]

    def test_empty_pool_yields_an_empty_set(self):
        assert indistinguishable_set([], 10.0) == []

    def test_key_length_mismatch_raises(self):
        pool = [_agg(), _agg()]
        with pytest.raises(ValueError, match="one entry per point"):
            indistinguishable_set(pool, 10.0, keys=[parsimony_key(_cfg())])

    def test_membership_does_not_depend_on_input_order(self):
        pool = [_agg(value=10.0, cost=300.0, ci=(290.0, 310.0)),
                _agg(value=10.0, cost=115.0, ci=(105.0, 125.0)),
                _agg(value=10.0, cost=100.0, ci=(90.0, 110.0))]
        assert indistinguishable_set(pool, 10.0) == [1, 2]


# ---------------------------------------------------------------------------
# Parsimony tie-break (§3.10)
# ---------------------------------------------------------------------------

def _cfg(N=2.0, R=4, F_freeze=1, A_dev=None, A_qa=None) -> TeamConfiguration:
    return TeamConfiguration(
        N=N, A_dev=A_dev if A_dev is not None else (1, 0, 1, 0, 1, 0, 0, 0),
        R=R, F_freeze=F_freeze,
        A_qa=A_qa if A_qa is not None else (0, 0, 1, 0, 0, 0, 0, 0),
    )


class TestParsimonyRelation:
    def test_key_covers_the_process_and_both_phases(self):
        key = parsimony_key(_cfg())
        assert key.process == (2.0, 4, 1)
        assert len(key.agents) == 16

    def test_inert_qa_half_is_zero(self):
        """A_qa is canonicalised away at F_freeze == 0, so it cannot separate two teams."""
        key = parsimony_key(_cfg(R=4, F_freeze=0, A_qa=(0, 0, 3, 0, 0, 0, 0, 0)))
        assert key.agents[8:] == (0,) * 8

    def test_smaller_team_dominates(self):
        small = parsimony_key(_cfg(A_dev=(1, 0, 1, 0, 1, 0, 0, 0)))
        large = parsimony_key(_cfg(A_dev=(1, 0, 2, 0, 1, 0, 0, 0)))
        assert parsimony_dominates(small, large)
        assert not parsimony_dominates(large, small)

    def test_identical_teams_do_not_dominate(self):
        key = parsimony_key(_cfg())
        assert not parsimony_dominates(key, key)

    def test_incomparable_counts_do_not_dominate(self):
        """Componentwise <= is a partial order: it cannot separate a swap."""
        a = parsimony_key(_cfg(A_dev=(1, 0, 3, 2, 1, 0, 0, 0)))
        b = parsimony_key(_cfg(A_dev=(1, 0, 2, 3, 1, 0, 0, 0)))
        assert not parsimony_dominates(a, b)
        assert not parsimony_dominates(b, a)

    def test_a_different_process_blocks_it(self):
        """Otherwise the rule could prefer fewer agents but more (dearer) humans."""
        small_team_more_people = parsimony_key(_cfg(N=5.0, A_dev=(1, 0, 1, 0, 1, 0, 0, 0)))
        large_team = parsimony_key(_cfg(N=2.0, A_dev=(1, 0, 2, 0, 1, 0, 0, 0)))
        assert not parsimony_dominates(small_team_more_people, large_team)

    def test_qa_counts_participate(self):
        small = parsimony_key(_cfg(A_qa=(0, 0, 1, 0, 0, 0, 0, 0)))
        large = parsimony_key(_cfg(A_qa=(0, 0, 1, 1, 0, 0, 0, 0)))
        assert parsimony_dominates(small, large)


class TestIndistinguishableSetParsimony:
    def test_tied_family_collapses_to_the_smallest(self):
        pool = [_agg(value=10.0, cost=100.0, ci=(90.0, 110.0)),
                _agg(value=10.0, cost=101.0, ci=(91.0, 111.0))]
        keys = [parsimony_key(_cfg(A_dev=(1, 0, 1, 0, 1, 0, 0, 0))),
                parsimony_key(_cfg(A_dev=(1, 0, 2, 0, 1, 0, 0, 0)))]
        assert indistinguishable_set(pool, 10.0) == [0, 1]
        assert indistinguishable_set(pool, 10.0, keys=keys) == [0]

    def test_incomparable_members_both_survive(self):
        pool = [_agg(value=10.0, cost=100.0, ci=(90.0, 110.0)),
                _agg(value=10.0, cost=101.0, ci=(91.0, 111.0))]
        keys = [parsimony_key(_cfg(A_dev=(1, 0, 3, 2, 1, 0, 0, 0))),
                parsimony_key(_cfg(A_dev=(1, 0, 2, 3, 1, 0, 0, 0)))]
        assert indistinguishable_set(pool, 10.0, keys=keys) == [0, 1]

    def test_the_set_is_never_emptied(self):
        """`parsimony_dominates` is antisymmetric, so a minimal member always survives."""
        import numpy as np
        rng = np.random.default_rng(3)
        pool, keys = [], []
        for _ in range(25):
            A = (1,) + tuple(int(x) for x in rng.integers(0, 3, size=7))
            pool.append(_agg(value=10.0, cost=100.0, ci=(90.0, 110.0)))
            keys.append(parsimony_key(_cfg(A_dev=A, A_qa=(0, 0, 1, 0, 0, 0, 0, 0))))
        assert indistinguishable_set(pool, 10.0, keys=keys)

    def test_the_tie_break_never_removes_the_raw_minimiser(self):
        """§3.10 exempts c* from the tie-break, so a smaller tied team cannot displace it.

        Both survive here: c* because it is exempt, and the smaller team because the larger
        one does not parsimony-dominate it.
        """
        pool = [_agg(value=10.0, cost=100.0, ci=(90.0, 110.0)),   # cheapest, but larger
                _agg(value=10.0, cost=101.0, ci=(91.0, 111.0))]   # dearer, but smaller
        keys = [parsimony_key(_cfg(A_dev=(1, 0, 2, 0, 1, 0, 0, 0))),
                parsimony_key(_cfg(A_dev=(1, 0, 1, 0, 1, 0, 0, 0)))]
        assert best_index(pool, 10.0) == 0
        assert indistinguishable_set(pool, 10.0, keys=keys) == [0, 1]

    def test_a_dominated_non_optimum_is_still_removed(self):
        """The exemption covers c* alone; every other member obeys the tie-break."""
        pool = [_agg(value=10.0, cost=100.0, ci=(90.0, 110.0)),   # c*, smallest anyway
                _agg(value=10.0, cost=101.0, ci=(91.0, 111.0)),   # tied, small
                _agg(value=10.0, cost=102.0, ci=(92.0, 112.0))]   # tied, dominated by 1
        keys = [parsimony_key(_cfg(N=2.0, A_dev=(1, 0, 1, 0, 1, 0, 0, 0))),
                parsimony_key(_cfg(N=3.0, A_dev=(1, 0, 1, 0, 1, 0, 0, 0))),
                parsimony_key(_cfg(N=3.0, A_dev=(1, 0, 2, 0, 1, 0, 0, 0)))]
        assert indistinguishable_set(pool, 10.0, keys=keys) == [0, 1]

    def test_parsimony_does_not_depend_on_input_order(self):
        """The dominance filter is order-independent once c* is pinned.

        Costs are made distinct so that the exempt c* is the same *configuration* in every
        ordering; with all costs equal, `best_index` would break the tie by index and the
        exemption would legitimately land on a different member after a shuffle.
        """
        import numpy as np
        rng = np.random.default_rng(4)
        pool, keys = [], []
        for i in range(12):
            A = (1,) + tuple(int(x) for x in rng.integers(0, 3, size=7))
            cost = 100.0 + i
            pool.append(_agg(value=10.0, cost=cost, ci=(cost - 30.0, cost + 30.0)))
            keys.append(parsimony_key(_cfg(A_dev=A, A_qa=(0, 0, 1, 0, 0, 0, 0, 0))))
        survivors = {keys[i].agents for i in indistinguishable_set(pool, 10.0, keys=keys)}
        order = list(rng.permutation(len(pool)))
        shuffled = [pool[i] for i in order], [keys[i] for i in order]
        survivors_shuffled = {
            shuffled[1][i].agents
            for i in indistinguishable_set(shuffled[0], 10.0, keys=shuffled[1])
        }
        assert survivors == survivors_shuffled


class TestReportOptimum:
    """`report_optimum` must never report a c* that is absent from the reported I*."""

    def test_best_is_always_a_member_of_a_non_empty_set(self):
        import numpy as np
        rng = np.random.default_rng(9)
        for _ in range(60):
            pool, keys = [], []
            for _ in range(int(rng.integers(1, 12))):
                cost = float(rng.uniform(50, 150))
                A = (1,) + tuple(int(x) for x in rng.integers(0, 3, size=7))
                pool.append(_agg(value=float(rng.uniform(9, 11)), cost=cost,
                                 ci=(cost - 30.0, cost + 30.0)))
                keys.append(parsimony_key(_cfg(A_dev=A,
                                               A_qa=(0, 0, 1, 0, 0, 0, 0, 0))))
            best, members = report_optimum(pool, 10.0, keys=keys)
            assert best is not None
            # I* is empty exactly when nothing in the pool is admissible (§3.10); whenever
            # it is not, c* is one of its members.
            if satisfies_throughput(pool[best], 10.0):
                assert best in members
            else:
                assert members == []

    def test_the_tie_break_never_displaces_the_optimum(self):
        """§3.10: c* is the raw minimiser even when a smaller team is tied with it."""
        pool = [_agg(value=10.0, cost=100.0, ci=(90.0, 110.0)),   # cheapest, but larger
                _agg(value=10.0, cost=101.0, ci=(91.0, 111.0))]   # dearer, but smaller
        keys = [parsimony_key(_cfg(A_dev=(1, 0, 2, 0, 1, 0, 0, 0))),
                parsimony_key(_cfg(A_dev=(1, 0, 1, 0, 1, 0, 0, 0)))]
        best, members = report_optimum(pool, 10.0, keys=keys)
        assert (best, members) == (0, [0, 1])

    def test_an_inadmissible_pool_reports_the_least_violating_with_an_empty_set(self):
        """§3.10: no c* exists, so I* is empty, but the result is still reported."""
        pool = [_agg(value=1.0, cost=10.0, ci=(9.0, 11.0)),      # cheapest, but far short
                _agg(value=9.0, cost=100.0, ci=(90.0, 110.0))]   # least-violating
        keys = [parsimony_key(_cfg(A_dev=(1, 0, 1, 0, 1, 0, 0, 0))),
                parsimony_key(_cfg(A_dev=(1, 0, 2, 0, 1, 0, 0, 0)))]
        best, members = report_optimum(pool, 10.0, keys=keys)
        assert best == 1
        assert members == []
        assert not satisfies_throughput(pool[best], 10.0)

    def test_matches_best_index_without_the_tie_break(self):
        pool = [_agg(value=10.0, cost=300.0, ci=(290.0, 310.0)),
                _agg(value=10.0, cost=100.0, ci=(90.0, 110.0))]
        best, members = report_optimum(pool, 10.0)
        assert best == best_index(pool, 10.0) == 1
        assert members == indistinguishable_set(pool, 10.0)

    def test_empty_pool(self):
        assert report_optimum([], 10.0) == (None, [])


# ---------------------------------------------------------------------------
# Bootstrap aggregation (§3.8)
# ---------------------------------------------------------------------------

def _result(config, values, costs) -> ConfigResult:
    return ConfigResult(config=config, replications=tuple(
        ReplicationResult(value=v, cost=c) for v, c in zip(values, costs)
    ))


class TestAggregateConfigResult:
    def test_means_are_the_replication_means(self):
        cfg = _cfg()
        agg = aggregate_config_result(_result(cfg, [1.0, 2.0, 3.0], [10.0, 20.0, 30.0]),
                                      master_seed=0)
        assert agg.value_mean == pytest.approx(2.0)
        assert agg.cost_mean == pytest.approx(20.0)

    def test_cis_contain_the_mean(self):
        cfg = _cfg()
        agg = aggregate_config_result(
            _result(cfg, list(range(1, 21)), [10.0 * i for i in range(1, 21)]),
            master_seed=0,
        )
        assert agg.value_ci[0] <= agg.value_mean <= agg.value_ci[1]
        assert agg.cost_ci[0] <= agg.cost_mean <= agg.cost_ci[1]

    def test_ci_width_shrinks_with_K(self):
        import numpy as np
        cfg = _cfg()
        rng = np.random.default_rng(0)
        small = rng.normal(100.0, 10.0, size=10).tolist()
        large = rng.normal(100.0, 10.0, size=400).tolist()
        w_small = aggregate_config_result(_result(cfg, [1.0] * 10, small), 0).cost_ci
        w_large = aggregate_config_result(_result(cfg, [1.0] * 400, large), 0).cost_ci
        assert (w_large[1] - w_large[0]) < (w_small[1] - w_small[0])

    def test_deterministic(self):
        cfg = _cfg()
        r = _result(cfg, [1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0])
        assert aggregate_config_result(r, 7) == aggregate_config_result(r, 7)

    def test_reported_extras_are_averaged(self):
        cfg = _cfg()
        reps = (
            ReplicationResult(value=1.0, cost=10.0, cost_quality=4.0, cost_human=5.0,
                              cost_agent=1.0, actions=(2.0,) * 8, impl_human=3.0,
                              impl_agents=1.0, fixes_human=2.0, fixes_agent=0.0),
            ReplicationResult(value=3.0, cost=20.0, cost_quality=8.0, cost_human=10.0,
                              cost_agent=2.0, actions=(4.0,) * 8, impl_human=5.0,
                              impl_agents=3.0, fixes_human=4.0, fixes_agent=2.0),
        )
        agg = aggregate_config_result(ConfigResult(config=cfg, replications=reps), 0)
        assert agg.cost_quality_mean == pytest.approx(6.0)
        assert agg.cost_human_mean == pytest.approx(7.5)
        assert agg.cost_agent_mean == pytest.approx(1.5)
        assert agg.actions_mean == pytest.approx((3.0,) * 8)
        assert agg.impl_human_mean == pytest.approx(4.0)
        assert agg.impl_agents_mean == pytest.approx(2.0)
        assert agg.fixes_human_mean == pytest.approx(3.0)
        assert agg.fixes_agent_mean == pytest.approx(1.0)

    def test_reported_extras_default_to_zero(self):
        cfg = _cfg()
        agg = aggregate_config_result(_result(cfg, [1.0], [10.0]), 0)
        assert agg.cost_quality_mean == 0.0
        assert agg.actions_mean == ()

    def test_the_three_cost_drivers_sum_to_the_objective(self, base_config, base_params):
        """Over a real simulation, not a hand-built fixture (§3.8)."""
        from src.config import SimulationHorizon
        s = SimulationHorizon(T=6, K_search=4, K_final=4, master_seed=0)
        agg = aggregate_config_result(
            run_configuration(base_config, s, base_params), master_seed=0
        )
        assert (agg.cost_quality_mean + agg.cost_human_mean + agg.cost_agent_mean
                == pytest.approx(agg.cost_mean))
