"""Tests for src/pareto.py — dominance, CI significance, frontier, aggregation."""

import itertools

import numpy as np
import pytest

from src.pareto import (
    AggregatedObjectives,
    ParsimonyKey,
    _ci_significantly_better,
    aggregate_config_result,
    dominates_strict,
    pareto_frontier,
    parsimony_dominates,
    parsimony_key,
)
from src.config import TeamConfiguration
from src.simulation import ConfigResult, ReplicationResult


def _obj(V, Q, CF, ci_w=0.0):
    """Build AggregatedObjectives with optional symmetric CI half-width."""
    return AggregatedObjectives(
        value_mean=V, quality_mean=Q, cost_mean=CF,
        value_ci=(V - ci_w, V + ci_w),
        quality_ci=(Q - ci_w, Q + ci_w),
        cost_ci=(CF - ci_w, CF + ci_w),
    )


def _obj_ci(V, Q, CF, value_ci, quality_ci, cost_ci):
    """Build AggregatedObjectives with an explicit, per-objective asymmetric CI."""
    return AggregatedObjectives(
        value_mean=V, quality_mean=Q, cost_mean=CF,
        value_ci=value_ci, quality_ci=quality_ci, cost_ci=cost_ci,
    )


def _naive_frontier(points, ci_aware=True):
    """Brute-force transcription of paper §3.10, used as the reference oracle.

    `i` is on the frontier iff *no* other candidate CI-dominates it.  Deliberately
    makes no use of dominance transitivity, which the CI-aware relation lacks.
    """
    keep = []
    for i, p in enumerate(points):
        if not any(
            dominates_strict(q, p)
            and (not ci_aware or _ci_significantly_better(q, p))
            for j, q in enumerate(points) if j != i
        ):
            keep.append(i)
    return keep


def _nontransitive_triple():
    """(k, j, i) with CIdom(k,j) and CIdom(j,i) but NOT CIdom(k,i).

    The two links must be significant on *different* objectives.  If both were
    significant on the same one, say quality, then q_hi[k] < q_lo[j] <= q_hi[j] < q_lo[i]
    would give q_hi[k] < q_lo[i] and k would dominate i as well — the chain is provably
    transitive in that case.  Here k beats j on quality and j beats i on cost, while k,
    whose cost CI is wide, is significantly better than i on nothing.
    """
    k = _obj_ci(10.0, 1.0, 100.0, (9.99, 10.01), (0.5, 1.5), (50.0, 260.0))
    j = _obj_ci(10.0, 2.0, 110.0, (9.99, 10.01), (1.9, 2.1), (105.0, 115.0))
    i = _obj_ci(10.0, 3.0, 250.0, (9.99, 10.01), (1.4, 4.6), (245.0, 255.0))
    return k, j, i


def _config_result(V_vals, Q_vals, CF_vals, config):
    reps = tuple(ReplicationResult(value=v, quality=q, cost=c) for v, q, c in zip(V_vals, Q_vals, CF_vals))
    return ConfigResult(config=config, replications=reps)


class TestDominatesStrict:
    def test_all_better(self):
        a = _obj(5, 0.1, 100)
        b = _obj(4, 0.2, 200)
        assert dominates_strict(a, b)

    def test_equal_fails(self):
        a = _obj(5, 0.1, 100)
        assert not dominates_strict(a, a)

    def test_one_strict_rest_equal(self):
        a = _obj(5.1, 0.1, 100)
        b = _obj(5.0, 0.1, 100)
        assert dominates_strict(a, b)

    def test_reverse_fails(self):
        a = _obj(4, 0.2, 200)
        b = _obj(5, 0.1, 100)
        assert not dominates_strict(a, b)

    def test_one_dimension_worse_fails(self):
        a = _obj(5, 0.1, 300)  # better value, quality but worse CF
        b = _obj(4, 0.2, 200)
        assert not dominates_strict(a, b)


class TestCISignificantlyBetter:
    def test_value_non_overlapping(self):
        b = _obj(V=5, Q=0.1, CF=100, ci_w=0.2)   # value_ci = (4.8, 5.2)
        w = _obj(V=4, Q=0.2, CF=200, ci_w=0.2)   # value_ci = (3.8, 4.2)
        # b.value_ci[0]=4.8 > w.value_ci[1]=4.2 → significant
        assert _ci_significantly_better(b, w)

    def test_value_overlapping(self):
        # value CIs overlap; quality and CF CIs also overlap → not significant
        b = AggregatedObjectives(value_mean=5, quality_mean=0.1, cost_mean=100,
                                 value_ci=(4.0, 6.0), quality_ci=(0.05, 0.15), cost_ci=(50.0, 150.0))
        w = AggregatedObjectives(value_mean=4, quality_mean=0.2, cost_mean=200,
                                 value_ci=(3.0, 5.0), quality_ci=(0.15, 0.25), cost_ci=(100.0, 300.0))
        # value: 4.0 > 5.0? No. quality: 0.15 < 0.15? No. CF: 150 < 100? No. → False
        assert not _ci_significantly_better(b, w)

    def test_quality_non_overlapping(self):
        b = _obj(V=5, Q=0.1, CF=100, ci_w=0.02)  # quality_ci = (0.08, 0.12)
        w = _obj(V=4, Q=0.2, CF=200, ci_w=0.02)  # quality_ci = (0.18, 0.22)
        # b.quality_ci[1]=0.12 < w.quality_ci[0]=0.18 → significant
        assert _ci_significantly_better(b, w)

    def test_CF_non_overlapping(self):
        b = _obj(V=5, Q=0.1, CF=100, ci_w=5)
        w = _obj(V=4, Q=0.2, CF=200, ci_w=5)
        # b.CF_ci[1]=105 < w.CF_ci[0]=195 → significant
        assert _ci_significantly_better(b, w)

    def test_all_overlap_false(self):
        b = _obj(V=5, Q=0.1, CF=100, ci_w=5)
        w = _obj(V=4, Q=0.2, CF=200, ci_w=100)
        assert not _ci_significantly_better(b, w)


class TestParetoFrontier:
    def test_empty(self):
        assert pareto_frontier([]) == []

    def test_single(self):
        assert pareto_frontier([_obj(3, 0.1, 100)]) == [0]

    def test_clear_dominance(self):
        dominant = _obj(V=5, Q=0.05, CF=50, ci_w=0.2)
        weak     = _obj(V=2, Q=0.2,  CF=200, ci_w=0.2)
        result = pareto_frontier([weak, dominant])
        assert result == [1]

    def test_statistical_tie_both_retained(self):
        # A point-dominates B but all CIs overlap → both retained
        a = _obj(V=5, Q=0.1, CF=100, ci_w=50)
        b = _obj(V=4, Q=0.2, CF=200, ci_w=50)
        assert dominates_strict(a, b)
        result = pareto_frontier([a, b])
        assert sorted(result) == [0, 1]

    def test_incomparable_both_retained(self):
        # A better on value, B better on quality → neither dominates
        a = _obj(V=5, Q=0.3, CF=100, ci_w=0.1)
        b = _obj(V=3, Q=0.1, CF=100, ci_w=0.1)
        assert sorted(pareto_frontier([a, b])) == [0, 1]

    def test_three_points_one_dominated(self):
        best = _obj(V=5, Q=0.05, CF=50, ci_w=0.2)
        mid  = _obj(V=3, Q=0.10, CF=80, ci_w=0.2)
        bad  = _obj(V=2, Q=0.20, CF=200, ci_w=0.2)
        result = pareto_frontier([best, mid, bad])
        assert 0 in result
        assert 2 not in result  # bad is dominated by best


class TestParetoFrontierNonTransitive:
    """The CI-aware relation of §3.10 is not transitive.

    A candidate that has itself been eliminated must still be allowed to eliminate
    others, so `pareto_frontier` may not skip already-dominated candidates when
    searching for a dominator.
    """

    def test_triple_is_a_genuine_witness(self):
        k, j, i = _nontransitive_triple()
        for a, b in ((k, j), (j, i), (k, i)):
            assert dominates_strict(a, b)          # point-dominance IS transitive
        assert _ci_significantly_better(k, j)      # link 1, significant on quality
        assert _ci_significantly_better(j, i)      # link 2, significant on cost
        assert not _ci_significantly_better(k, i)  # ... but the chain does not close

    def test_dominated_candidate_still_eliminates(self):
        # j is CI-dominated by k, and j is the only candidate that CI-dominates i.
        # i must therefore be off the frontier, leaving k alone on it.
        k, j, i = _nontransitive_triple()
        assert pareto_frontier([k, j, i]) == [0]

    def test_frontier_is_order_invariant(self):
        k, j, i = _nontransitive_triple()
        points = [k, j, i]
        expected = {0}   # only k survives, whatever the ordering
        for perm in itertools.permutations(range(3)):
            permuted = [points[p] for p in perm]
            front = {perm[idx] for idx in pareto_frontier(permuted)}
            assert front == expected, f"ordering {perm} changed the frontier"

    @pytest.mark.parametrize("ci_aware", [True, False])
    @pytest.mark.parametrize("seed", range(8))
    def test_matches_naive_reference_on_random_input(self, seed, ci_aware):
        rng = np.random.default_rng(seed)
        n = 40
        points = []
        for _ in range(n):
            v, q, c = rng.uniform(1, 10), rng.uniform(0.01, 5), rng.uniform(50, 500)
            # Deliberately heterogeneous CI widths: a homogeneous relative width makes
            # the relation accidentally transitive and hides the defect.
            wv, wq, wc = rng.uniform(0.01, 2), rng.uniform(0.01, 3), rng.uniform(1, 200)
            points.append(_obj_ci(v, q, c, (v - wv, v + wv), (q - wq, q + wq), (c - wc, c + wc)))
        assert pareto_frontier(points, ci_aware=ci_aware) == _naive_frontier(points, ci_aware=ci_aware)

    @pytest.mark.parametrize("seed", range(8))
    def test_random_input_order_invariant(self, seed):
        rng = np.random.default_rng(seed)
        points = []
        for _ in range(40):
            v, q, c = rng.uniform(1, 10), rng.uniform(0.01, 5), rng.uniform(50, 500)
            wv, wq, wc = rng.uniform(0.01, 2), rng.uniform(0.01, 3), rng.uniform(1, 200)
            points.append(_obj_ci(v, q, c, (v - wv, v + wv), (q - wq, q + wq), (c - wc, c + wc)))
        baseline = set(pareto_frontier(points))
        for shuffle_seed in range(5):
            perm = np.random.default_rng(shuffle_seed).permutation(len(points))
            front = {int(perm[idx]) for idx in pareto_frontier([points[p] for p in perm])}
            assert front == baseline


def _pk(agents, process=(1.0, 4, 0)):
    """ParsimonyKey from an 8- or 16-element agent tuple (padded to both phases)."""
    agents = tuple(agents)
    if len(agents) == 8:
        agents += (0,) * 8
    return ParsimonyKey(process=process, agents=agents)


class TestParsimonyKey:
    def test_key_covers_process_and_both_phases(self):
        cfg = TeamConfiguration(N=1.5, A_dev=(1, 0, 1, 0, 2, 0, 1, 0),
                                A_qa=(0, 0, 1, 1, 2, 0, 0, 0), R=4, F_freeze=1)
        key = parsimony_key(cfg)
        assert key.process == (1.5, 4, 1)
        assert key.agents == (1, 0, 1, 0, 2, 0, 1, 0) + (0, 0, 1, 1, 2, 0, 0, 0)

    def test_inert_qa_half_is_zero(self):
        """Follows from the canonicalisation in TeamConfiguration, not from this helper."""
        cfg = TeamConfiguration(N=1.0, A_dev=(1, 0, 1, 0, 2, 0, 1, 0),
                                A_qa=(0, 0, 3, 3, 3, 3, 3, 3), R=4, F_freeze=0)
        assert parsimony_key(cfg).agents[8:] == (0,) * 8


class TestParsimonyRelation:
    """The §3.10 tie-break: same process, no larger anywhere, smaller somewhere, tied."""

    def _tied_pair(self):
        # Wide CIs on every objective → statistically indistinguishable in both directions.
        return _obj(V=5, Q=1.0, CF=100, ci_w=50), _obj(V=5, Q=1.2, CF=110, ci_w=50)

    def test_smaller_team_dominates_when_tied(self):
        small, big = self._tied_pair()
        assert parsimony_dominates(_pk((1, 0, 0, 0, 1, 0, 0, 0)),
                                   _pk((1, 0, 0, 0, 3, 0, 0, 0)), small, big)

    def test_identical_agent_counts_do_not_dominate(self):
        a, b = self._tied_pair()
        assert not parsimony_dominates(_pk((1, 0, 0, 0, 2, 0, 0, 0)),
                                       _pk((1, 0, 0, 0, 2, 0, 0, 0)), a, b)

    def test_incomparable_counts_do_not_dominate(self):
        """Larger in one role, smaller in another — the partial order says nothing.

        This is why the tie-break cannot replace canonicalising an inert A_qa: half of
        the duplicate pairs on a real frontier look like this.
        """
        a, b = self._tied_pair()
        ka, kb = _pk((1, 0, 0, 0, 3, 2, 0, 0)), _pk((1, 0, 0, 0, 2, 3, 0, 0))
        assert not parsimony_dominates(ka, kb, a, b)
        assert not parsimony_dominates(kb, ka, b, a)

    def test_different_process_blocks_it(self):
        """A team with fewer agents but more humans is not more parsimonious."""
        small, big = self._tied_pair()
        assert not parsimony_dominates(_pk((1, 0, 0, 0, 1, 0, 0, 0), process=(2.0, 4, 0)),
                                       _pk((1, 0, 0, 0, 3, 0, 0, 0), process=(1.0, 4, 0)),
                                       small, big)

    def test_significant_difference_blocks_it(self):
        # Narrow CIs → the bigger team is significantly better on quality, so they are
        # not "the same result" and §3.10 already separates them.
        small = _obj(V=5, Q=2.0, CF=100, ci_w=0.01)
        big   = _obj(V=5, Q=1.0, CF=100, ci_w=0.01)
        assert not parsimony_dominates(_pk((1, 0, 0, 0, 1, 0, 0, 0)),
                                       _pk((1, 0, 0, 0, 3, 0, 0, 0)), small, big)

    def test_qa_counts_participate(self):
        small, big = self._tied_pair()
        ka = ParsimonyKey(process=(1.0, 4, 1), agents=(1,) * 8 + (0, 0, 1, 0, 0, 0, 0, 0))
        kb = ParsimonyKey(process=(1.0, 4, 1), agents=(1,) * 8 + (0, 0, 2, 0, 0, 0, 0, 0))
        assert parsimony_dominates(ka, kb, small, big)

    @pytest.mark.parametrize("seed", range(10))
    def test_antisymmetry(self, seed):
        rng = np.random.default_rng(seed)
        for _ in range(200):
            ka = _pk(tuple(int(x) for x in rng.integers(0, 4, 16)))
            kb = _pk(tuple(int(x) for x in rng.integers(0, 4, 16)))
            a = _obj(V=5, Q=rng.uniform(1, 2), CF=100, ci_w=50)
            b = _obj(V=5, Q=rng.uniform(1, 2), CF=100, ci_w=50)
            assert not (parsimony_dominates(ka, kb, a, b)
                        and parsimony_dominates(kb, ka, b, a))


class TestParetoFrontierParsimony:
    def test_keys_none_leaves_the_frontier_untouched(self):
        """The backward-compatibility guarantee for callers that do not opt in."""
        rng = np.random.default_rng(0)
        points, keys = [], []
        for _ in range(60):
            v, q, c = rng.uniform(1, 10), rng.uniform(0.01, 5), rng.uniform(50, 500)
            w = rng.uniform(0.01, 1)
            points.append(_obj(v, q, c, ci_w=w))
            keys.append(_pk(tuple(int(x) for x in rng.integers(0, 3, 16))))
        baseline = pareto_frontier(points)
        assert pareto_frontier(points, keys=None) == baseline
        assert set(pareto_frontier(points, keys=keys)) <= set(baseline)

    def test_tied_family_collapses_to_the_smallest(self):
        # Four statistically indistinguishable teams forming a chain in agent counts.
        pts = [_obj(V=5, Q=1.0 + 0.05 * i, CF=100 + i, ci_w=50) for i in range(4)]
        keys = [_pk((1, 0, 0, 0, i + 1, 0, 0, 0)) for i in range(4)]
        assert pareto_frontier(pts) == [0, 1, 2, 3]          # §3.10 alone keeps all four
        assert pareto_frontier(pts, keys=keys) == [0]        # ... parsimony keeps one

    def test_frontier_is_never_emptied(self):
        rng = np.random.default_rng(4)
        for _ in range(50):
            pts = [_obj(V=5, Q=1.0 + 0.01 * i, CF=100, ci_w=50) for i in range(6)]
            keys = [_pk(tuple(int(x) for x in rng.integers(0, 3, 16))) for _ in range(6)]
            assert pareto_frontier(pts, keys=keys)

    @pytest.mark.parametrize("seed", range(6))
    def test_order_invariant_with_keys(self, seed):
        rng = np.random.default_rng(seed)
        points, keys = [], []
        for _ in range(50):
            v, q, c = rng.uniform(1, 10), rng.uniform(0.5, 2), rng.uniform(90, 110)
            points.append(_obj(v, q, c, ci_w=rng.uniform(1, 10)))
            keys.append(_pk(tuple(int(x) for x in rng.integers(0, 3, 16))))
        baseline = set(pareto_frontier(points, keys=keys))
        for shuffle_seed in range(4):
            perm = np.random.default_rng(shuffle_seed).permutation(len(points))
            front = {int(perm[i]) for i in pareto_frontier([points[p] for p in perm],
                                                           keys=[keys[p] for p in perm])}
            assert front == baseline

    def test_length_mismatch_raises(self):
        pts = [_obj(5, 1, 100, ci_w=1), _obj(4, 2, 200, ci_w=1)]
        with pytest.raises(ValueError, match="one entry per point"):
            pareto_frontier(pts, keys=[_pk((1,) * 16)])

    def test_inert_qa_family_reduces_to_one_point(self):
        """End-to-end on the duplicate family from the review, via parsimony alone.

        With Part 1 in place these four are a single TeamConfiguration; this checks the
        independent path, for a chain of inert QA vectors built by hand.
        """
        pts = [_obj(V=11.96, Q=7.379, CF=250.48, ci_w=1.0),
               _obj(V=12.03, Q=7.971, CF=249.23, ci_w=1.0),
               _obj(V=12.05, Q=7.402, CF=248.90, ci_w=1.0),
               _obj(V=12.03, Q=7.920, CF=249.18, ci_w=1.0)]
        keys = [ParsimonyKey(process=(1.0, 4, 1), agents=(1,) * 8 + (0, 0, n, 0, 0, 0, 0, 0))
                for n in (0, 1, 2, 3)]
        assert len(pareto_frontier(pts, keys=keys)) == 1


class TestAggregateConfigResult:
    def test_means_correct(self, base_config):
        cr = _config_result([2.0, 4.0], [0.1, 0.3], [500.0, 1500.0], base_config)
        agg = aggregate_config_result(cr, master_seed=0)
        assert agg.value_mean == pytest.approx(3.0)
        assert agg.quality_mean == pytest.approx(0.2)
        assert agg.cost_mean == pytest.approx(1000.0)

    def test_CI_contains_mean(self, base_config):
        vals = list(range(1, 51))  # 50 values
        cr = _config_result(vals, [0.1] * 50, [100.0] * 50, base_config)
        agg = aggregate_config_result(cr, master_seed=0)
        assert agg.value_ci[0] <= agg.value_mean <= agg.value_ci[1]

    def test_CI_width_shrinks_with_K(self, base_config):
        import random
        random.seed(0)
        vals = [random.gauss(3, 1) for _ in range(100)]
        cr10  = _config_result(vals[:10],  [0.1]*10,  [100.0]*10,  base_config)
        cr100 = _config_result(vals[:100], [0.1]*100, [100.0]*100, base_config)
        agg10  = aggregate_config_result(cr10,  master_seed=0)
        agg100 = aggregate_config_result(cr100, master_seed=0)
        w10  = agg10.value_ci[1]  - agg10.value_ci[0]
        w100 = agg100.value_ci[1] - agg100.value_ci[0]
        assert w100 < w10

    def test_deterministic(self, base_config):
        cr = _config_result([1.0, 2.0, 3.0], [0.1]*3, [100.0]*3, base_config)
        a1 = aggregate_config_result(cr, master_seed=42)
        a2 = aggregate_config_result(cr, master_seed=42)
        assert a1.value_ci == a2.value_ci

    def test_contribution_counts_are_averaged(self, base_config):
        """The four who-did-what counters aggregate as plain means, without CIs."""
        reps = (
            ReplicationResult(value=1.0, quality=0.1, cost=10.0, impl_human=10.0,
                              impl_agents=90.0, fixes_human=2.0, fixes_agent=8.0),
            ReplicationResult(value=1.0, quality=0.1, cost=10.0, impl_human=20.0,
                              impl_agents=80.0, fixes_human=4.0, fixes_agent=6.0),
        )
        agg = aggregate_config_result(
            ConfigResult(config=base_config, replications=reps), master_seed=0)
        assert agg.impl_human_mean == pytest.approx(15.0)
        assert agg.impl_agents_mean == pytest.approx(85.0)
        assert agg.fixes_human_mean == pytest.approx(3.0)
        assert agg.fixes_agent_mean == pytest.approx(7.0)

    def test_contribution_counts_default_to_zero(self, base_config):
        """Replications built without them (as most of this module's helpers do)."""
        cr = _config_result([2.0, 4.0], [0.1, 0.3], [500.0, 1500.0], base_config)
        agg = aggregate_config_result(cr, master_seed=0)
        assert agg.impl_human_mean == 0.0
        assert agg.fixes_agent_mean == 0.0


class TestContributionCountsAreInvisibleToDominance:
    """§3.10 stays three-objective: the reported counts must not sway the frontier."""

    def test_frontier_ignores_the_contribution_counts(self):
        import dataclasses
        points = [_obj(5.0, 0.1, 100.0), _obj(4.0, 0.2, 200.0), _obj(4.5, 0.15, 150.0)]
        baseline = pareto_frontier(points)

        # Same objectives, wildly different contribution splits.
        loaded = [
            dataclasses.replace(points[0], impl_human_mean=1000.0,
                                fixes_human_mean=500.0),
            dataclasses.replace(points[1], impl_agents_mean=1000.0,
                                fixes_agent_mean=500.0),
            dataclasses.replace(points[2], impl_human_mean=7.0, impl_agents_mean=3.0),
        ]
        assert pareto_frontier(loaded) == baseline

    def test_ci_significance_ignores_the_contribution_counts(self):
        import dataclasses
        b = _obj(V=5, Q=0.1, CF=100, ci_w=0.2)
        w = _obj(V=4, Q=0.2, CF=200, ci_w=0.2)
        assert _ci_significantly_better(b, w)
        assert _ci_significantly_better(
            dataclasses.replace(b, impl_human_mean=0.0, impl_agents_mean=1e6),
            dataclasses.replace(w, impl_human_mean=1e6, impl_agents_mean=0.0),
        )
