"""Tests for src/simulation.py — objectives formula, trace capture, cost accounting."""

import pytest

from src.config import SweepSettings
from src.simulation import ConfigResult, ConfigTrace, run_configuration


class TestRunConfigurationBasic:
    def test_result_count(self, base_config, fixed_priors):
        s = SweepSettings(T=4, K=3, master_seed=0)
        result = run_configuration(base_config, s, fixed_priors)
        assert isinstance(result, ConfigResult)
        assert len(result.replications) == 3

    def test_objectives_non_negative(self, base_config, fixed_priors):
        s = SweepSettings(T=8, K=5, master_seed=0)
        result = run_configuration(base_config, s, fixed_priors)
        for rep in result.replications:
            assert rep.V >= 0
            assert rep.Q >= 0
            assert rep.C_F >= 0

    def test_no_trace_returns_ConfigResult(self, base_config, fixed_priors, small_settings):
        result = run_configuration(base_config, small_settings, fixed_priors)
        assert type(result) is ConfigResult


class TestTraceCapture:
    def test_trace_returns_ConfigTrace(self, base_config, fixed_priors, small_settings):
        trace = run_configuration(base_config, small_settings, fixed_priors, capture_traces=True)
        assert type(trace) is ConfigTrace

    def test_trace_shape(self, base_config, fixed_priors):
        s = SweepSettings(T=4, K=2, master_seed=0)
        trace = run_configuration(base_config, s, fixed_priors, capture_traces=True)
        assert len(trace.traces) == 2
        assert all(len(rt.snapshots) == 4 for rt in trace.traces)

    def test_trace_objectives_identical_to_no_trace(self, base_config, fixed_priors):
        s = SweepSettings(T=4, K=3, master_seed=42)
        no_trace = run_configuration(base_config, s, fixed_priors)
        with_trace = run_configuration(base_config, s, fixed_priors, capture_traces=True)
        for a, b in zip(no_trace.replications, with_trace.replications):
            assert a.V == b.V and a.Q == b.Q and a.C_F == b.C_F

    def test_warmup_not_in_snapshots(self, base_config, fixed_priors):
        from src.sampling import ParameterSpec, ScenarioPriors
        specs = dict(fixed_priors.specs)
        specs["R_warmup"] = ParameterSpec(dist="fixed", mean=3.0)
        priors_w = ScenarioPriors(specs=specs, c_token=fixed_priors.c_token)
        s = SweepSettings(T=4, K=1, master_seed=0)
        trace = run_configuration(base_config, s, priors_w, capture_traces=True)
        sprint_indices = [snap.t for snap in trace.traces[0].snapshots]
        assert sprint_indices == [0, 1, 2, 3]


class TestDeterminism:
    def test_same_seed_same_results(self, base_config, fixed_priors):
        s = SweepSettings(T=4, K=3, master_seed=7)
        r1 = run_configuration(base_config, s, fixed_priors)
        r2 = run_configuration(base_config, s, fixed_priors)
        assert r1.replications == r2.replications


class TestCostAccounting:
    def test_zero_costs_give_zero_CF(self, base_config, fixed_priors):
        from src.sampling import ParameterSpec, ScenarioPriors
        specs = dict(fixed_priors.specs)
        cost_fields = ["c_salary", "c_PO", "C_fix_minor", "C_fix_major",
                       "C_prod_min", "C_prod_maj", "C_DevOps_setup",
                       "C_DevOps_maint", "C_user_doc"]
        for f in cost_fields:
            specs[f] = ParameterSpec(dist="fixed", mean=0.0)
        zero_priors = ScenarioPriors(
            specs=specs,
            c_token=tuple(0.0 for _ in range(13)),
        )
        s = SweepSettings(T=4, K=2, master_seed=0)
        result = run_configuration(base_config, s, zero_priors)
        for rep in result.replications:
            assert rep.C_F == pytest.approx(0.0)
