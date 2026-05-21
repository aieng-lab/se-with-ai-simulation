"""Tests for src/feasibility.py — all 6 failure steps + feasible path."""

from dataclasses import replace as dc_replace

import pytest

from src.config import AGENT_ROLE_COUNT, TeamConfiguration
from src.feasibility import check_feasibility


def _cfg(comp_dict, N=2, M=3, R=4):
    A = [0] * AGENT_ROLE_COUNT
    for k, v in comp_dict.items():
        A[k - 1] = v
    return TeamConfiguration(N=N, M=M, A=tuple(A), R=R)


class TestFeasiblePath:
    def test_feasible_returns_true(self, base_config, base_params):
        report = check_feasibility(base_config, base_params)
        assert report.feasible
        assert report.failed_step is None
        assert report.D_star is not None
        assert 0.0 <= report.D_star <= 1.0

    def test_D_star_zero_when_refactor_dominates(self, base_config, base_params):
        from dataclasses import replace
        p = replace(base_params, delta_impl=0.001, delta_fix=0.001, delta_refactor=1.0)
        report = check_feasibility(base_config, p)
        assert report.feasible
        assert report.D_star == 0.0


class TestStep0Gate:
    def test_regression_loop_infeasible(self, base_config, base_params):
        from dataclasses import replace
        p = replace(base_params, p_regress=1.0, c_UT_ceil=1.0, d_UT=1.0)
        report = check_feasibility(base_config, p)
        assert not report.feasible
        assert report.failed_step == 0


class TestStep1SpanControl:
    def test_M_too_large_infeasible(self, base_params):
        # Tight capacity: 10h, 2h monitoring/agent — M=6 needs 12h > H_net≈9.5h
        cfg = _cfg({3: 4, 4: 2, 5: 2, 6: 2, 8: 2}, N=2, M=6)
        from dataclasses import replace
        p = replace(base_params, H_capacity=10.0, H_monitoring_overhead=2.0)
        report = check_feasibility(cfg, p)
        assert not report.feasible
        assert report.failed_step == 1


class TestStep3DebtSpiral:
    def test_no_refactoring_infeasible(self, base_params):
        # No refactoring agents + high debt increment → debt spirals
        cfg = _cfg({3: 3, 4: 2, 6: 1})  # a_5 = 0
        from dataclasses import replace
        p = replace(base_params, delta_impl=0.1)
        report = check_feasibility(cfg, p)
        assert not report.feasible
        assert report.failed_step == 3
        assert report.D_star is None  # D_star not set when debt spirals


class TestStep4BacklogUnstable:
    def test_lambda_F_too_high(self, base_config, base_params):
        from dataclasses import replace
        p = replace(base_params, lambda_F=100.0)
        report = check_feasibility(base_config, p)
        assert not report.feasible
        assert report.failed_step == 4
        assert report.D_star is not None  # D_star computed before step 4


class TestStep5BugQueueUnstable:
    def test_high_bug_rate_high_fix_cost(self, base_config, base_params):
        # Force rho_bugs >= 1: no fix agents, high injection
        cfg = _cfg({3: 3, 5: 1, 6: 1, 8: 1})  # a_4 = 0
        from dataclasses import replace
        p = replace(base_params, lambda_impl=2.0, p_regress=0.0)
        report = check_feasibility(cfg, p)
        assert not report.feasible
        assert report.failed_step == 5
