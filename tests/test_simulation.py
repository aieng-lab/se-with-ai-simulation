"""Tests for src/simulation.py — objectives formula, trace capture, cost accounting."""

import dataclasses

import pytest

from src.config import SimulationHorizon, TeamConfiguration
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

    def test_agent_cost_is_the_work_performed_times_unit_cost(self, base_config, base_params):
        """Eq. (cost_agents): the bill is the inner product of work and c_token."""
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

        # Unit token costs, so C_agents is just the total work performed.
        assert rep.cost == pytest.approx(sum(rep.actions) / max(1, F_completed))
        assert rep.cost_agent == pytest.approx(rep.cost)   # c_salary = 0
        assert rep.cost_human == pytest.approx(0.0)

    def test_idle_agents_are_free(self, base_params):
        """With no work arriving, every role but the two flat services bills nothing."""
        from src.config import AGENT_ROLE_COUNT
        params = dataclasses.replace(
            base_params, c_salary=0.0, r_arrival=0.0, f_arch_chg=0.0,
            c_token=tuple(1.0 for _ in range(AGENT_ROLE_COUNT)),
        )
        config = TeamConfiguration(
            N=2, A_dev=tuple(1 for _ in range(AGENT_ROLE_COUNT)), R=4, F_freeze=0,
        )
        s = SimulationHorizon(T=6, K=1, master_seed=0)
        rep = run_configuration(config, s, params).replications[0]

        # k=1..4 idle: no features arrive, so nothing is implemented, no change
        # requests are raised, no bugs are injected, and no debt accrues to remove.
        # k=7,8 are idle too: with no completed features there is no coverage to buy,
        # so both caps are zero.
        for k in (0, 1, 2, 3, 6, 7):
            assert rep.actions[k] == pytest.approx(0.0)
        # k=5,6 are flat per deployed instance per sprint.
        assert rep.actions[4] == pytest.approx(s.T)
        assert rep.actions[5] == pytest.approx(s.T)

    def test_saturated_demand_reproduces_the_nominal_bill(self, base_params):
        """Regression anchor: when every agent is busy, usage billing meets the
        old nominal formula.  This pins the unit convention of each action count."""
        from src.config import AGENT_ROLE_COUNT
        # Demand far above capacity and zero debt, so the implementation agent runs
        # at its full rate every sprint.
        params = dataclasses.replace(
            base_params, c_salary=0.0, r_arrival=500.0, delta_min=1.0,
            delta_impl=0.0, delta_fix=0.0, r_refactor=0.0,
            c_token=tuple(1.0 for _ in range(AGENT_ROLE_COUNT)),
            f_fix=0.0, h_feature=1e9,      # humans implement nothing
        )
        config = TeamConfiguration(
            N=2, A_dev=(2, 0, 0, 0, 1, 1, 1, 1), R=4, F_freeze=0,
        )
        s = SimulationHorizon(T=5, K=1, master_seed=3)
        rep = run_configuration(config, s, params).replications[0]

        # k=1 saturates at floor(r_F * 1.0 * a_1) features every sprint.
        nominal_impl = int(params.r_F * 1.0 * config.A_dev[0]) * s.T
        assert rep.actions[0] == pytest.approx(nominal_impl)
        assert rep.actions[4] == pytest.approx(config.A_dev[4] * s.T)
        # Features complete fast enough here that the coverage cap never binds, so the
        # test agent also runs at its nominal rate.
        assert rep.actions[6] == pytest.approx(params.r_test * config.A_dev[6] * s.T)

    def test_architecture_implementation_bills_five_features(self, base_params):
        """Table tab:work_units: u_1 counts a proposal as ARCH_IMPL_COST_UNITS features."""
        from src.config import AGENT_ROLE_COUNT
        from src.sprint import ARCH_IMPL_COST_UNITS, step_sprint
        from src.state import SystemState
        import numpy as np

        # No arrivals and no backlog, so the implementation agent's only work is the
        # queued architecture proposals; u_1 is then purely the arch_impl term.
        params = dataclasses.replace(
            base_params, r_arrival=0.0, f_arch_chg=0.0, delta_min=1.0,
        )
        config = TeamConfiguration(
            N=2, A_dev=tuple(1 if k == 0 else 0 for k in range(AGENT_ROLE_COUNT)),
            R=8, F_freeze=0,
        )
        state = SystemState(F_backlog=0, Q_arch_impl=3)
        actions = step_sprint(0, state, config, params, np.random.default_rng(0))

        # Capacity is floor(r_F * 1.0 * 1) = 3 with the fixture's r_F, so all three
        # proposals are implemented this sprint and nothing else is.
        implemented = 3 - state.Q_arch_impl
        assert implemented == 3
        assert actions.units[0] == pytest.approx(ARCH_IMPL_COST_UNITS * implemented)
        assert ARCH_IMPL_COST_UNITS == 5.0

    def test_test_and_doc_output_is_capped_at_the_coverage_demand(self, base_params):
        """Eqs. (ntest_update)/(ndoc_update): neither role writes past its ceiling.

        Also pins the billing identity for the two capped roles: the total work billed
        is exactly the cumulative count the transition wrote.
        """
        from src.config import AGENT_ROLE_COUNT
        # r_test/r_doc far above the coverage demand, so the cap is what binds.
        params = dataclasses.replace(base_params, r_test=500.0, r_doc=500.0)
        config = TeamConfiguration(
            N=2, A_dev=(1, 0, 0, 0, 0, 0, 1, 1), R=4, F_freeze=0,
        )
        s = SimulationHorizon(T=10, K=1, master_seed=4)
        trace = run_configuration(config, s, params, capture_traces=True)
        rep = trace.replications[0]

        for snap in trace.traces[0].snapshots:
            assert snap.N_test <= params.c_test_ceil * snap.F_completed + 1e-9
            assert snap.N_doc <= snap.F_completed + 1e-9

        last = trace.traces[0].snapshots[-1]
        assert rep.actions[6] == pytest.approx(last.N_test)
        assert rep.actions[7] == pytest.approx(last.N_doc)
        # The cap really bit — otherwise this test would prove nothing.
        assert last.N_test < params.r_test * s.T

    def test_capped_output_still_reaches_full_coverage(self, base_params):
        """The cap is exactly where Eq. (coverage) already clipped, so it costs nothing.

        With the role staffed in every sprint the agent refills the deficit each sprint,
        so coverage sits at its ceiling and documentation coverage at 1.
        """
        params = dataclasses.replace(base_params, r_test=500.0, r_doc=500.0)
        config = TeamConfiguration(
            N=2, A_dev=(1, 0, 0, 0, 0, 0, 1, 1), R=4, F_freeze=0,
        )
        s = SimulationHorizon(T=10, K=1, master_seed=4)
        trace = run_configuration(config, s, params, capture_traces=True)

        for snap in trace.traces[0].snapshots:
            if snap.F_completed > 0:
                assert snap.N_test == pytest.approx(params.c_test_ceil * snap.F_completed)
                assert snap.N_doc == pytest.approx(snap.F_completed)

    def test_cost_split_sums_to_cost(self, base_config, base_params):
        params = dataclasses.replace(base_params, c_salary=120_000.0)
        s = SimulationHorizon(T=8, K=3, master_seed=11)
        for rep in run_configuration(base_config, s, params).replications:
            assert rep.cost_human + rep.cost_agent == pytest.approx(rep.cost)
            assert rep.cost_human > 0 and rep.cost_agent >= 0


class TestContributionAccounting:
    """The horizon totals of who did the work (see SprintActions)."""

    @pytest.mark.parametrize("r_arrival, h_feature", [(2.0, 8.0), (20.0, 2.0)])
    def test_the_implementation_split_accounts_for_every_feature(
        self, base_config, base_params, r_arrival, h_feature
    ):
        """impl_human + impl_agents == F_completed(T) == value * T, exactly."""
        params = dataclasses.replace(base_params, r_arrival=r_arrival,
                                     h_feature=h_feature)
        s = SimulationHorizon(T=12, K=3, master_seed=7)
        result = run_configuration(base_config, s, params)
        for rep in result.replications:
            assert rep.impl_human + rep.impl_agents == pytest.approx(rep.value * s.T)

    def test_bug_fixing_split_matches_the_billed_agent_unit(self, base_config, base_params):
        s = SimulationHorizon(T=12, K=3, master_seed=8)
        for rep in run_configuration(base_config, s, base_params).replications:
            assert rep.fixes_agent == pytest.approx(rep.actions[2])
            assert rep.fixes_human >= 0.0

    def test_implemented_proposals_are_recoverable_from_the_billed_unit(
        self, base_config, base_params
    ):
        """(u_1 - impl_agents) / ARCH_IMPL_COST_UNITS is a whole number of proposals."""
        from src.sprint import ARCH_IMPL_COST_UNITS
        # f_arch_chg > 0 so proposals actually flow through the review queue.
        params = dataclasses.replace(base_params, f_arch_chg=0.5, r_arrival=6.0)
        config = TeamConfiguration(
            N=2, A_dev=(2, 1, 0, 0, 1, 0, 0, 0), R=4, F_freeze=0,
        )
        s = SimulationHorizon(T=16, K=3, master_seed=9)
        for rep in run_configuration(config, s, params).replications:
            arch_impl = (rep.actions[0] - rep.impl_agents) / ARCH_IMPL_COST_UNITS
            assert arch_impl >= 0.0
            assert arch_impl == pytest.approx(round(arch_impl))

    def test_replications_keep_distinct_trace_labels(self, base_config, base_params):
        """Regression: the per-role accumulator used to shadow the replication index,
        so every ReplicationTrace was labelled with the last role instead of its k."""
        s = SimulationHorizon(T=4, K=3, master_seed=12)
        trace = run_configuration(base_config, s, base_params, capture_traces=True)
        assert {tr.k for tr in trace.traces} == set(range(s.K))
