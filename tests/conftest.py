"""Shared fixtures for the agent-ai-cost-sim test suite."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import AGENT_ROLE_COUNT, SweepSettings, TeamConfiguration
from src.parameters import ScenarioParameters
from src.sampling import ParameterSpec, ScenarioPriors


def _make_A(*pairs) -> tuple[int, ...]:
    """Build a 13-element A tuple from (role_k, count) pairs (1-indexed)."""
    A = [0] * AGENT_ROLE_COUNT
    for role_k, count in pairs:
        A[role_k - 1] = count
    return tuple(A)


@pytest.fixture
def base_config():
    """N=2, M=3, R=4; a_3=2 impl, a_4=1 fix, a_5=1 refactor, a_6=1 review, a_8=1 UT."""
    return TeamConfiguration(
        N=2, M=3,
        A=_make_A((3, 2), (4, 1), (5, 1), (6, 1), (8, 1)),
        R=4,
    )


@pytest.fixture
def base_params():
    """Concrete ScenarioParameters with clearly feasible, numerically safe values."""
    return ScenarioParameters(
        lambda_F=2.0, r_req=2.0, r_arch=0.5, r_F=3.0,
        r_fix_minor=5.0, r_fix_major=2.0, r_refactor=4.0,
        r_review=5.0, r_security=3.0,
        r_UT=3.0, r_IT=1.5, r_ST=0.5, r_doc=2.0,
        H_capacity=40.0, H_monitoring_overhead=2.0,
        H_review_time=4.0, H_escalation_time=2.0,
        h_feature=8.0, c_comm=1.0, f_code=0.30,
        lambda_impl=0.02, lambda_human_scale=0.8,
        phi_max=1.5, kappa=1.0, theta=0.2, p_sec=0.1,
        epsilon_spec=0.05, alpha_spec=1.0,
        q_arch=0.8, q_arch_floor=0.5,
        p_regress=0.05, p_regress_refactor=0.03,
        d_CR=0.7, d_SA=0.85, d_UT=0.5, d_IT=0.4, d_ST=0.3,
        c_UT_ceil=2.0, c_IT_ceil=1.0, c_ST_ceil=0.5,
        alpha_size=0.1, baseline_size=2.0,
        tau_CI_base=1.0, alpha_F_CI=0.002, alpha_test_CI=0.0005,
        beta_UT=1.0, beta_IT=2.0, beta_ST=4.0,
        psi_min=0.3, gamma=1.5,
        delta_impl=0.01, delta_fix=0.005, delta_refactor=0.05,
        Gamma_doc=0.5, zeta_doc=0.02,
        eta_min_rev=0.5, alpha_eta_rev=0.05, Q_thr_rev=3,
        eta_min_esc=0.6, alpha_eta_esc=0.1, Q_thr_esc=5, Q_thr_block=10,
        rho_esc=0.01, D0=0.0, R_warmup=0,
    )


@pytest.fixture
def fixed_priors(base_params):
    """ScenarioPriors where every field is fixed to base_params values (no randomness)."""
    from dataclasses import fields as dc_fields
    specs = {
        f.name: ParameterSpec(dist="fixed", mean=float(getattr(base_params, f.name)))
        for f in dc_fields(ScenarioParameters)
        if f.name != "c_token"
    }
    return ScenarioPriors(specs=specs, c_token=base_params.c_token)


@pytest.fixture
def small_settings():
    return SweepSettings(T=4, K=3, master_seed=42)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
