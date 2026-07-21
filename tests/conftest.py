"""Shared fixtures for the agent-ai-cost-sim test suite."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import AGENT_ROLE_COUNT, SimulationHorizon, TeamConfiguration
from src.parameters import ScenarioParameters


def _make_A(*pairs) -> tuple[int, ...]:
    """Build an 8-element A tuple from (role_k, count) pairs (1-indexed)."""
    A = [0] * AGENT_ROLE_COUNT
    for role_k, count in pairs:
        A[role_k - 1] = count
    return tuple(A)


@pytest.fixture
def base_config():
    """N=2, M=3, R=4; a_1=2 impl, a_3=1 fix, a_4=1 refactor, a_5=1 review, a_7=1 test."""
    return TeamConfiguration(
        N=2, M=3,
        A_dev=_make_A((1, 2), (3, 1), (4, 1), (5, 1), (7, 1)),
        A_qa=_make_A((3, 2), (4, 1), (5, 1), (7, 2)),
        R=4,
    )


@pytest.fixture
def base_params():
    """Concrete ScenarioParameters with clearly feasible, numerically safe values."""
    return ScenarioParameters(
        r_arrival=2.0, f_arch_chg=0.0, r_arch_chg=2.0, r_F=3.0,
        r_fix_minor=5.0, r_fix_major=2.0, r_fix_sec=1.5, r_refactor=0.2,
        r_test=3.0, r_doc=2.0,
        H_capacity=40.0, H_monitoring_overhead=2.0,
        H_review_time=4.0, H_escalation_time=2.0,
        h_feature=8.0, h_fix=4.0, c_comm=1.0, f_fix=0.10,
        r_bugs=0.02, r_bugs_human=0.8,
        theta_maj=0.18, theta_sec=0.10,
        d_CR=0.7, d_SA=0.85, d_test=0.6, c_test_ceil=2.0,
        alpha_size=0.1, baseline_size=2.0,
        delta_min=0.5, delta_exp=1.5,
        delta_impl=0.01, delta_fix=0.005,
        Gamma_doc=0.5, c_doc_ceil=1.0,
        rho_esc=0.01,
    )


@pytest.fixture
def small_settings():
    return SimulationHorizon(T=4, K=3, master_seed=42)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
