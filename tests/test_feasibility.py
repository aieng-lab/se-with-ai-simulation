"""Tests for src/feasibility.py — single human-capacity check + feasible path."""

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


class TestStep1SpanControl:
    def test_M_too_large_infeasible(self, base_params):
        # Gamma_doc=0 so doc_cov assumption is irrelevant; 6 agents * 2h = 12h > H_net≈9h
        cfg = _cfg({1: 4, 3: 2, 4: 2, 5: 2, 7: 2}, N=2, M=6)
        from dataclasses import replace
        p = replace(base_params, H_capacity=10.0, H_monitoring_overhead=2.0, Gamma_doc=0.0)
        report = check_feasibility(cfg, p)
        assert not report.feasible
        assert report.failed_step == 1
