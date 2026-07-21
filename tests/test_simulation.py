"""Tests for src/simulation.py — objectives formula, trace capture, cost accounting."""

import dataclasses

import pytest

from src.config import SimulationHorizon
from src.simulation import ConfigResult, ConfigTrace, run_configuration
from src.state import SystemState


class TestRunConfigurationBasic:
    def test_result_count(self, base_config, base_params):
        s = SimulationHorizon(T=4, K=3, master_seed=0)
        result = run_configuration(base_config, s, base_params)
        assert isinstance(result, ConfigResult)
        assert len(result.replications) == 3

    def test_objectives_non_negative(self, base_config, base_params):
        s = SimulationHorizon(T=8, K=5, master_seed=0)
        result = run_configuration(base_config, s, base_params)
        for rep in result.replications:
            assert rep.value >= 0
            assert rep.quality >= 0
            assert rep.cost >= 0

    def test_no_trace_returns_ConfigResult(self, base_config, base_params, small_settings):
        result = run_configuration(base_config, small_settings, base_params)
        assert type(result) is ConfigResult


class TestTraceCapture:
    def test_trace_returns_ConfigTrace(self, base_config, base_params, small_settings):
        trace = run_configuration(base_config, small_settings, base_params, capture_traces=True)
        assert type(trace) is ConfigTrace

    def test_trace_shape(self, base_config, base_params):
        s = SimulationHorizon(T=4, K=2, master_seed=0)
        trace = run_configuration(base_config, s, base_params, capture_traces=True)
        assert len(trace.traces) == 2
        assert all(len(rt.snapshots) == 4 for rt in trace.traces)

    def test_trace_objectives_identical_to_no_trace(self, base_config, base_params):
        s = SimulationHorizon(T=4, K=3, master_seed=42)
        no_trace = run_configuration(base_config, s, base_params)
        with_trace = run_configuration(base_config, s, base_params, capture_traces=True)
        for a, b in zip(no_trace.replications, with_trace.replications):
            assert a.value == b.value and a.quality == b.quality and a.cost == b.cost


class TestDeterminism:
    def test_same_seed_same_results(self, base_config, base_params):
        s = SimulationHorizon(T=4, K=3, master_seed=7)
        r1 = run_configuration(base_config, s, base_params)
        r2 = run_configuration(base_config, s, base_params)
        assert r1.replications == r2.replications


class TestInitialState:
    def test_non_zero_backlog_affects_results(self, base_config, base_params):
        s = SimulationHorizon(T=4, K=3, master_seed=0)
        zero_result = run_configuration(base_config, s, base_params)
        warm_result = run_configuration(
            base_config, s, base_params, initial_state=SystemState(F_backlog=50)
        )
        # More backlog gives more completed features on average
        zero_mean = sum(r.value for r in zero_result.replications) / len(zero_result.replications)
        warm_mean = sum(r.value for r in warm_result.replications) / len(warm_result.replications)
        assert warm_mean > zero_mean

    def test_initial_state_object_not_mutated(self, base_config, base_params):
        s = SimulationHorizon(T=4, K=3, master_seed=0)
        init = SystemState(F_backlog=20)
        run_configuration(base_config, s, base_params, initial_state=init)
        assert init.F_backlog == 20

    def test_same_seed_same_results_with_initial_state(self, base_config, base_params):
        s = SimulationHorizon(T=4, K=3, master_seed=7)
        init = SystemState(F_backlog=10, D_tech=1.0)
        r1 = run_configuration(base_config, s, base_params, initial_state=init)
        r2 = run_configuration(base_config, s, base_params, initial_state=init)
        assert r1.replications == r2.replications

    def test_none_initial_state_equals_default(self, base_config, base_params):
        s = SimulationHorizon(T=4, K=3, master_seed=5)
        r_none = run_configuration(base_config, s, base_params, initial_state=None)
        r_zero = run_configuration(base_config, s, base_params, initial_state=SystemState())
        assert r_none.replications == r_zero.replications


class TestCostAccounting:
    def test_zero_costs_give_zero_cost(self, base_config, base_params):
        from src.config import AGENT_ROLE_COUNT
        import dataclasses
        zero_params = dataclasses.replace(
            base_params,
            c_salary=0.0,
            c_token=tuple(0.0 for _ in range(AGENT_ROLE_COUNT)),
        )
        s = SimulationHorizon(T=4, K=2, master_seed=0)
        result = run_configuration(base_config, s, zero_params)
        for rep in result.replications:
            assert rep.cost == pytest.approx(0.0)

    def test_agent_cost_scaled_by_role_rate(self, base_config, base_params):
        from src.config import AGENT_ROLE_COUNT
        params = dataclasses.replace(
            base_params,
            c_salary=0.0,
            c_token=tuple(1.0 for _ in range(AGENT_ROLE_COUNT)),
        )
        s = SimulationHorizon(T=4, K=1, master_seed=0)
        trace = run_configuration(base_config, s, params, capture_traces=True)
        rep = trace.replications[0]
        F_completed = trace.traces[0].snapshots[-1].F_completed

        r_fix = params.r_fix_minor + params.r_fix_major + params.r_fix_sec
        agent_rates = (
            params.r_F, params.r_arch_chg, r_fix, params.r_refactor,
            1.0, 1.0, params.r_test, params.r_doc,
        )
        qa_sprint_count = sum(
            1 for t in range(s.T)
            if (t % base_config.R) >= (base_config.R - base_config.F_freeze)
        )
        dev_sprint_count = s.T - qa_sprint_count
        expected_C_agents = (
            sum(base_config.A_dev[r] * agent_rates[r] for r in range(AGENT_ROLE_COUNT)) * dev_sprint_count
            + sum(base_config.A_qa[r] * agent_rates[r] for r in range(AGENT_ROLE_COUNT)) * qa_sprint_count
        )
        expected_cost = expected_C_agents / max(1, F_completed)
        assert rep.cost == pytest.approx(expected_cost)
