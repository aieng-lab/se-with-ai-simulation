"""Tests for src/pareto.py — dominance, CI significance, frontier, aggregation."""

import numpy as np
import pytest

from src.pareto import (
    AggregatedObjectives,
    _ci_significantly_better,
    aggregate_config_result,
    dominates_strict,
    pareto_frontier,
)
from src.simulation import ConfigResult, ReplicationResult


def _obj(V, Q, CF, ci_w=0.0):
    """Build AggregatedObjectives with optional symmetric CI half-width."""
    return AggregatedObjectives(
        value_mean=V, quality_mean=Q, cost_mean=CF,
        value_ci=(V - ci_w, V + ci_w),
        quality_ci=(Q - ci_w, Q + ci_w),
        cost_ci=(CF - ci_w, CF + ci_w),
    )


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
