"""Tests for src/feasibility.py — single human-capacity check + feasible path."""

import pytest

from src.config import AGENT_ROLE_COUNT, TeamConfiguration
from src.feasibility import check_feasibility


def _cfg(comp_dict, N=2, M=3, R=4, F_freeze=0):
    A = [0] * AGENT_ROLE_COUNT
    for k, v in comp_dict.items():
        A[k - 1] = v
    return TeamConfiguration(N=N, M=M, A_dev=tuple(A), R=R, F_freeze=F_freeze)


class TestFeasiblePath:
    def test_feasible_returns_true(self, base_config, base_params):
        report = check_feasibility(base_config, base_params)
        assert report.feasible
        assert report.failed_step is None


class TestStep1SpanControl:
    def test_M_too_large_infeasible(self, base_params):
        # Gamma_doc=0 so doc_cov assumption is irrelevant; 12 supervised agents over
        # N=2 developers cost 6 * 2h = 12h of monitoring each > H_net ≈ 9h
        cfg = _cfg({1: 4, 3: 3, 4: 3, 7: 2}, N=2, M=6)
        from dataclasses import replace
        p = replace(base_params, H_capacity=10.0, H_monitoring_overhead=2.0, Gamma_doc=0.0)
        report = check_feasibility(cfg, p)
        assert not report.feasible
        assert report.failed_step == 1

    def test_background_services_never_cause_infeasibility(self, base_params):
        """Piling on review/audit agents must not exhaust the monitoring budget."""
        from dataclasses import replace
        p = replace(base_params, H_capacity=10.0, H_monitoring_overhead=2.0, Gamma_doc=0.0)
        # 4 supervised agents over N=2 developers = 2 * 2h = 4h < H_net ≈ 9h
        lean = _cfg({1: 2, 3: 1, 7: 1}, N=2, M=6)
        assert check_feasibility(lean, p).feasible
        # Same supervised load, but 40 background services on top
        loaded = _cfg({1: 2, 3: 1, 7: 1, 5: 20, 6: 20}, N=2, M=6)
        assert check_feasibility(loaded, p).feasible
