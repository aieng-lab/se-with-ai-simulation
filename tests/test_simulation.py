"""Tests for src/simulation.py — objectives formula and cost accounting."""

import dataclasses
from pathlib import Path

import pytest

from src.config import AGENT_ROLE_COUNT, SimulationHorizon, TeamConfiguration
from src.rng import generator_for
from src.simulation import ConfigResult, run_configuration
from src.sprint import step_sprint
from src.state import SystemState


#: An all-zero QA composition.  Every team below runs F_freeze = 0 or is a
#: validation case, so A_qa is never applied; it is spelled out because
#: `TeamConfiguration` requires both compositions.
_NO_QA = (0,) * AGENT_ROLE_COUNT


class TestRunConfigurationBasic:
    def test_result_count(self, base_config, base_params):
        s = SimulationHorizon(T=4, K_search=3, K_final=3, master_seed=0)
        result = run_configuration(base_config, s, base_params)
        assert isinstance(result, ConfigResult)
        assert len(result.replications) == 3

    def test_objectives_non_negative(self, base_config, base_params):
        s = SimulationHorizon(T=8, K_search=5, K_final=5, master_seed=0)
        result = run_configuration(base_config, s, base_params)
        for rep in result.replications:
            assert rep.value >= 0
            assert rep.cost >= 0
            assert rep.cost_quality >= 0

    def test_K_override_sets_the_replication_count(self, base_config, base_params):
        s = SimulationHorizon(T=4, K_search=3, K_final=11, master_seed=0)
        assert len(run_configuration(base_config, s, base_params).replications) == 3
        result = run_configuration(base_config, s, base_params, K=s.K_final)
        assert len(result.replications) == 11

    def test_the_replication_block_is_reproducible_and_index_addressed(
        self, base_config, base_params
    ):
        """Every replication is seeded from `(config, k)` alone, so index k is index k."""
        s = SimulationHorizon(T=6, K_search=4, K_final=12, master_seed=17)
        short = run_configuration(base_config, s, base_params)
        long = run_configuration(base_config, s, base_params, K=s.K_final)
        assert long.replications[:s.K_search] == short.replications

    def test_k_offset_yields_a_disjoint_sample(self, base_config, base_params):
        """The re-simulation must not reuse the replications that selected the candidate.

        A member of the preliminary I* got there by having one of the lowest of thousands
        of noisy K'-replication estimates, so that estimate is biased low by the selection.
        Shifting the block past K_search makes the re-simulated estimate independent of it.
        """
        s = SimulationHorizon(T=6, K_search=4, K_final=8, master_seed=17)
        search_block = run_configuration(base_config, s, base_params)
        resim = run_configuration(base_config, s, base_params,
                                  K=s.K_final, k_offset=s.K_search)
        assert len(resim.replications) == s.K_final
        # Disjoint by construction: no replication of the search block reappears.
        long = run_configuration(base_config, s, base_params,
                                 K=s.K_search + s.K_final)
        assert resim.replications == long.replications[s.K_search:]
        assert search_block.replications == long.replications[:s.K_search]

    def test_a_negative_offset_raises(self, base_config, base_params, small_settings):
        with pytest.raises(ValueError, match="k_offset must be >= 0"):
            run_configuration(base_config, small_settings, base_params, k_offset=-1)

    def test_K_below_one_raises(self, base_config, base_params, small_settings):
        with pytest.raises(ValueError, match="K must be >= 1"):
            run_configuration(base_config, small_settings, base_params, K=0)


class TestDeterminism:
    def test_same_seed_same_results(self, base_config, base_params):
        s = SimulationHorizon(T=4, K_search=3, K_final=3, master_seed=7)
        r1 = run_configuration(base_config, s, base_params)
        r2 = run_configuration(base_config, s, base_params)
        assert r1.replications == r2.replications


class TestInitialState:
    def test_non_zero_backlog_affects_results(self, base_config, base_params):
        s = SimulationHorizon(T=4, K_search=3, K_final=3, master_seed=0)
        zero_result = run_configuration(base_config, s, base_params)
        warm_result = run_configuration(
            base_config, s, base_params, initial_state=SystemState(F_backlog=50)
        )
        # More backlog gives more completed features on average
        zero_mean = sum(r.value for r in zero_result.replications) / len(zero_result.replications)
        warm_mean = sum(r.value for r in warm_result.replications) / len(warm_result.replications)
        assert warm_mean > zero_mean

    def test_initial_state_object_not_mutated(self, base_config, base_params):
        s = SimulationHorizon(T=4, K_search=3, K_final=3, master_seed=0)
        init = SystemState(F_backlog=20)
        run_configuration(base_config, s, base_params, initial_state=init)
        assert init.F_backlog == 20

    def test_same_seed_same_results_with_initial_state(self, base_config, base_params):
        s = SimulationHorizon(T=4, K_search=3, K_final=3, master_seed=7)
        init = SystemState(F_backlog=10, D_tech=1.0)
        r1 = run_configuration(base_config, s, base_params, initial_state=init)
        r2 = run_configuration(base_config, s, base_params, initial_state=init)
        assert r1.replications == r2.replications

    def test_none_initial_state_equals_default(self, base_config, base_params):
        s = SimulationHorizon(T=4, K_search=3, K_final=3, master_seed=5)
        r_none = run_configuration(base_config, s, base_params, initial_state=None)
        r_zero = run_configuration(base_config, s, base_params, initial_state=SystemState())
        assert r_none.replications == r_zero.replications


class TestCostAccounting:
    def test_zero_costs_give_zero_cost(self, base_config, base_params):
        # All three cost drivers priced at zero, the escaped bugs included:
        # this fixture does let bugs through, so leaving the weights alone would price
        # them and the total would not be zero.
        zero_params = dataclasses.replace(
            base_params,
            c_salary=0.0,
            c_token=tuple(0.0 for _ in range(AGENT_ROLE_COUNT)),
            omega_min=0.0, omega_maj=0.0, omega_sec=0.0,
        )
        s = SimulationHorizon(T=4, K_search=2, K_final=2, master_seed=0)
        result = run_configuration(base_config, s, zero_params)
        for rep in result.replications:
            assert rep.cost == pytest.approx(0.0)

    def test_agent_cost_is_the_work_performed_times_unit_cost(self, base_config, base_params):
        """The agent bill is the inner product of the work vector and c_token."""
        # Only the agent bill is left standing: the salary and the escaped-bug weights
        # are zeroed so that `cost` is C_agents / F_completed and nothing else.
        params = dataclasses.replace(
            base_params,
            c_salary=0.0,
            c_token=tuple(1.0 for _ in range(AGENT_ROLE_COUNT)),
            omega_min=0.0, omega_maj=0.0, omega_sec=0.0,
        )
        s = SimulationHorizon(T=4, K_search=1, K_final=1, master_seed=0)
        rep = run_configuration(base_config, s, params).replications[0]
        # value is F_completed(T)/T, so the horizon recovers the count.
        F_completed = round(rep.value * s.T)

        # Unit token costs, so C_agents is just the total work performed.
        assert rep.cost == pytest.approx(sum(rep.actions) / max(1, F_completed))
        assert rep.cost_agent == pytest.approx(rep.cost)   # c_salary = 0
        assert rep.cost_human == pytest.approx(0.0)

    def test_idle_agents_are_free(self, base_params):
        """With no work arriving, every demand-driven role bills nothing."""
        params = dataclasses.replace(
            base_params, c_salary=0.0, r_arrival=0.0, f_arch_chg=0.0,
            c_token=tuple(1.0 for _ in range(AGENT_ROLE_COUNT)),
        )
        config = TeamConfiguration(
            N=2, A_dev=tuple(1 for _ in range(AGENT_ROLE_COUNT)), R=4, F_freeze=0,
            A_qa=_NO_QA,
        )
        s = SimulationHorizon(T=6, K_search=1, K_final=1, master_seed=0)
        rep = run_configuration(config, s, params).replications[0]

        # k=1..3 idle: no features arrive, so nothing is implemented, no change
        # requests are raised and no bugs are injected.  k=7,8 are idle too: with no
        # completed features there is no coverage to buy, so both caps are zero.
        for k in (0, 1, 2, 6, 7):
            assert rep.actions[k] == pytest.approx(0.0)
        # k=4,5,6 have no countable unit of output and are charged flat per deployed
        # instance per sprint, so they bill even with nothing to do.
        for k in (3, 4, 5):
            assert rep.actions[k] == pytest.approx(s.T)

    def test_saturated_demand_reproduces_the_nominal_bill(self, base_params):
        """When no agent is idle, the agent bill reaches its ceiling.

        Every role then completes a full sprint's worth of work, so the per-role action
        counts are exactly its nominal rates; this is what pins the unit convention of
        each count.
        """
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
            A_qa=_NO_QA,
        )
        s = SimulationHorizon(T=5, K_search=1, K_final=1, master_seed=3)
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
        from src.sprint import ARCH_IMPL_COST_UNITS, step_sprint
        from src.state import SystemState
        import numpy as np

        # No arrivals and no backlog, so the implementation agents' only work is the
        # queued architecture proposals; u_1 is then purely the arch_impl term.
        params = dataclasses.replace(
            base_params, r_arrival=0.0, f_arch_chg=0.0, delta_min=1.0,
        )
        # A proposal also *consumes* five features' worth of capacity, so
        # implementing three needs floor(r_F * a_1) >= 15: with the fixture's r_F = 3
        # that is a_1 = 5.
        config = TeamConfiguration(
            N=2, A_dev=tuple(5 if k == 0 else 0 for k in range(AGENT_ROLE_COUNT)),
            R=8, F_freeze=0,
            A_qa=_NO_QA,
        )
        state = SystemState(F_backlog=0, Q_arch_impl=3)
        actions = step_sprint(0, state, config, params, np.random.default_rng(0))

        implemented = 3 - state.Q_arch_impl
        assert implemented == 3
        assert actions.units[0] == pytest.approx(ARCH_IMPL_COST_UNITS * implemented)
        assert ARCH_IMPL_COST_UNITS == 5.0

    def test_architecture_implementation_capacity_costs_five_features(self, base_params):
        """The proposal cap is floor(r_F * delta_D * a_1 / 5)."""
        from src.sprint import ARCH_IMPL_COST_UNITS, step_sprint
        from src.state import SystemState
        import numpy as np

        params = dataclasses.replace(
            base_params, r_arrival=0.0, f_arch_chg=0.0, delta_min=1.0,
            f_fix=0.0, h_feature=1e9,      # humans implement nothing
        )
        # floor(r_F * a_1) = 3 * 4 = 12, so at five features per proposal only two fit,
        # and they leave 12 - 10 = 2 features' worth of capacity for the backlog.
        config = TeamConfiguration(
            N=2, A_dev=tuple(4 if k == 0 else 0 for k in range(AGENT_ROLE_COUNT)),
            R=8, F_freeze=0,
            A_qa=_NO_QA,
        )
        state = SystemState(F_backlog=10, Q_arch_impl=5)
        actions = step_sprint(0, state, config, params, np.random.default_rng(0))

        assert state.Q_arch_impl == 3, "only two of the five proposals fit"
        arch_impl = (actions.units[0] - actions.impl_agents) / ARCH_IMPL_COST_UNITS
        assert arch_impl == pytest.approx(2.0)
        assert actions.impl_agents == pytest.approx(2.0)

    def test_test_and_doc_output_is_capped_at_the_coverage_demand(self, base_params):
        """Eqs. (ntest_update)/(ndoc_update): neither role writes past its ceiling.

        Also pins the billing identity for the two capped roles: the total work billed
        is exactly the cumulative count the transition wrote.  Driven sprint by sprint,
        since the property is about the state the transition leaves behind.
        """
        # r_test/r_doc far above the coverage demand, so the cap is what binds.
        params = dataclasses.replace(base_params, r_test=500.0, r_doc=500.0)
        config = TeamConfiguration(
            N=2, A_dev=(1, 0, 0, 0, 0, 0, 1, 1), R=4, F_freeze=0,
            A_qa=_NO_QA,
        )
        T = 10
        state = SystemState()
        rng = generator_for(4, config, 0)
        billed_tests = billed_docs = 0.0
        for t in range(T):
            actions = step_sprint(t, state, config, params, rng)
            billed_tests += actions.units[6]
            billed_docs  += actions.units[7]
            assert state.N_test <= params.c_test_ceil * state.F_completed + 1e-9
            assert state.N_doc <= state.F_completed + 1e-9

        assert billed_tests == pytest.approx(state.N_test)
        assert billed_docs == pytest.approx(state.N_doc)
        # The cap really bit — otherwise this test would prove nothing.
        assert state.N_test < params.r_test * T

    def test_capped_output_still_reaches_full_coverage(self, base_params):
        """The cap sits exactly where the coverage formula already clipped, so it costs nothing.

        With the role staffed in every sprint the agent refills the deficit each sprint,
        so coverage sits at its ceiling and documentation coverage at 1.
        """
        params = dataclasses.replace(base_params, r_test=500.0, r_doc=500.0)
        config = TeamConfiguration(
            N=2, A_dev=(1, 0, 0, 0, 0, 0, 1, 1), R=4, F_freeze=0,
            A_qa=_NO_QA,
        )
        state = SystemState()
        rng = generator_for(4, config, 0)
        for t in range(10):
            step_sprint(t, state, config, params, rng)
            if state.F_completed > 0:
                assert state.N_test == pytest.approx(
                    params.c_test_ceil * state.F_completed)
                assert state.N_doc == pytest.approx(state.F_completed)

    def test_cost_split_sums_to_cost(self, base_config, base_params):
        """The three cost drivers partition the total cost exactly."""
        params = dataclasses.replace(base_params, c_salary=120_000.0)
        s = SimulationHorizon(T=8, K_search=3, K_final=3, master_seed=11)
        for rep in run_configuration(base_config, s, params).replications:
            assert (rep.cost_quality + rep.cost_human + rep.cost_agent
                    == pytest.approx(rep.cost))
            assert rep.cost_human > 0
            assert rep.cost_agent >= 0 and rep.cost_quality >= 0

    def test_escaped_bugs_are_priced_into_the_objective(self, base_params):
        """The omega weights are dollars, so they move `cost` directly."""
        from src.simulation import escaped_bug_cost
        s = SimulationHorizon(T=12, K_search=3, K_final=3, master_seed=13)
        # Plenty of bugs and nobody to catch or repair them, so every injected bug ships.
        leaky = dataclasses.replace(base_params, r_bugs=1.0, d_test=0.0)
        config = TeamConfiguration(
            N=2, A_dev=(2, 0, 0, 0, 0, 0, 0, 0), R=4, F_freeze=0,
            A_qa=_NO_QA,
        )
        cheap = dataclasses.replace(leaky, omega_min=1.0, omega_maj=1.0, omega_sec=1.0)
        pricey = dataclasses.replace(cheap, omega_min=1e4, omega_maj=1e4, omega_sec=1e4)
        cheap_reps = run_configuration(config, s, cheap).replications
        pricey_reps = run_configuration(config, s, pricey).replications

        # Identical trajectories (the weights enter no transition), so the only
        # difference is the price of the escaped bugs.
        for a, b in zip(cheap_reps, pricey_reps):
            assert a.value == b.value
            assert a.cost_human == pytest.approx(b.cost_human)
            assert a.cost_agent == pytest.approx(b.cost_agent)
            assert b.cost_quality == pytest.approx(1e4 * a.cost_quality)
        assert sum(r.cost_quality for r in pricey_reps) > 0, "bugs must have escaped"

        assert escaped_bug_cost(pricey, 1, 2, 3) == pytest.approx(6e4)

    def test_refactoring_is_billed_per_deployed_instance(self, base_params):
        """Table tab:work_units: u_4 = a_4, independent of the debt trajectory."""
        config = TeamConfiguration(
            N=3, A_dev=(1, 0, 0, 2, 0, 0, 0, 0), R=4, F_freeze=0,
            A_qa=_NO_QA,
        )
        s = SimulationHorizon(T=7, K_search=1, K_final=1, master_seed=5)
        # Two runs whose debt trajectories differ wildly: one where refactoring has
        # something to remove, one where debt never accrues at all.
        busy = dataclasses.replace(base_params, delta_impl=0.5, delta_fix=0.5)
        idle = dataclasses.replace(base_params, delta_impl=0.0, delta_fix=0.0)
        for params in (busy, idle):
            rep = run_configuration(config, s, params).replications[0]
            assert rep.actions[3] == pytest.approx(config.A_dev[3] * s.T)

    def test_unstaffed_refactoring_bills_nothing(self, base_params):
        config = TeamConfiguration(
            N=2, A_dev=(1, 0, 0, 0, 0, 0, 0, 0), R=4, F_freeze=0,
            A_qa=_NO_QA,
        )
        s = SimulationHorizon(T=5, K_search=1, K_final=1, master_seed=6)
        rep = run_configuration(config, s, base_params).replications[0]
        assert rep.actions[3] == pytest.approx(0.0)


class TestContributionAccounting:
    """The horizon totals of who did the work (see SprintActions)."""

    @pytest.mark.parametrize("r_arrival, h_feature", [(2.0, 8.0), (20.0, 2.0)])
    def test_the_implementation_split_accounts_for_every_feature(
        self, base_config, base_params, r_arrival, h_feature
    ):
        """impl_human + impl_agents == F_completed(T) == value * T, exactly."""
        params = dataclasses.replace(base_params, r_arrival=r_arrival,
                                     h_feature=h_feature)
        s = SimulationHorizon(T=12, K_search=3, K_final=3, master_seed=7)
        result = run_configuration(base_config, s, params)
        for rep in result.replications:
            assert rep.impl_human + rep.impl_agents == pytest.approx(rep.value * s.T)

    def test_bug_fixing_split_matches_the_billed_agent_unit(self, base_config, base_params):
        s = SimulationHorizon(T=12, K_search=3, K_final=3, master_seed=8)
        for rep in run_configuration(base_config, s, base_params).replications:
            assert rep.fixes_agent == pytest.approx(rep.actions[2])
            assert rep.fixes_human >= 0.0

    def test_implemented_proposals_are_recoverable_from_the_billed_unit(self, base_params):
        """(u_1 - impl_agents) / ARCH_IMPL_COST_UNITS is a whole number of proposals."""
        from src.sprint import ARCH_IMPL_COST_UNITS
        # f_arch_chg > 0 so proposals actually flow through the review queue.
        params = dataclasses.replace(base_params, f_arch_chg=0.5, r_arrival=6.0)
        config = TeamConfiguration(
            N=2, A_dev=(2, 1, 0, 0, 1, 0, 0, 0), R=4, F_freeze=0,
            A_qa=_NO_QA,
        )
        s = SimulationHorizon(T=16, K_search=3, K_final=3, master_seed=9)
        for rep in run_configuration(config, s, params).replications:
            arch_impl = (rep.actions[0] - rep.impl_agents) / ARCH_IMPL_COST_UNITS
            assert arch_impl >= 0.0
            assert arch_impl == pytest.approx(round(arch_impl))


class TestHumanOnlyBaselineCost:
    """A zero-agent team has no token bill at all."""

    @staticmethod
    def _cfg(N=2):
        zero = (0,) * AGENT_ROLE_COUNT
        return TeamConfiguration(N=N, A_dev=zero, A_qa=zero, R=4, F_freeze=1)

    def test_agent_cost_is_exactly_zero(self, base_params):
        params = dataclasses.replace(
            base_params, c_token=tuple(1000.0 for _ in range(AGENT_ROLE_COUNT)),
        )
        s = SimulationHorizon(T=6, K_search=2, K_final=2, master_seed=0)
        result = run_configuration(self._cfg(), s, params)
        for rep in result.replications:
            assert rep.cost_agent == 0.0
            assert sum(rep.actions) == 0.0

    def test_the_cost_split_still_sums_to_the_total(self, base_params):
        s = SimulationHorizon(T=6, K_search=2, K_final=2, master_seed=0)
        result = run_configuration(self._cfg(), s, base_params)
        for rep in result.replications:
            assert rep.cost == pytest.approx(
                rep.cost_quality + rep.cost_human + rep.cost_agent)

    def test_every_feature_is_implemented_by_a_human(self, base_params):
        s = SimulationHorizon(T=6, K_search=2, K_final=2, master_seed=0)
        result = run_configuration(self._cfg(), s, base_params)
        for rep in result.replications:
            assert rep.impl_agents == 0
            assert rep.impl_human == pytest.approx(rep.value * s.T)


# ---------------------------------------------------------------------------
# The persisted campaign as a golden record
# ---------------------------------------------------------------------------

#: Scenarios spanning both result modes and both stress tiers of the grid: the extreme
#: project velocity (`p-v`), the pessimistic agent tier (`a-p`), and a human-only
#: enumeration at each end of the bug-impact range.
_GOLDEN_SCENARIOS = (
    "h-l_p-l_d-l_a-l_b-l",
    "h-h_p-v_d-h_a-p_b-h",
    "h-h_p-m_d-h_a-m_b-m",
    "h-h_p-l_d-X_a-X_b-h",
)


class TestAgainstThePersistedCampaign:
    """The reported `c*` of a finished campaign, re-simulated and compared bit for bit.

    Every random stream is addressed by `(config_hash, replication index)`, so
    ``run_configuration(cfg, K=K_final, k_offset=K_search)`` reproduces *exactly* the block
    the two-stage selection already ran.  The persisted ``optima.csv`` is therefore an
    exact golden record of the transition, the cost accounting and the bootstrap, not an
    approximate one, and no search is involved, so the check costs seconds.

    This is what stands between an accidental edit to `step_sprint` and a campaign whose
    results have silently moved.  It catches a change in *behaviour* — dropping the debt
    amplifier from the effective bug rate, say, moves all three agent-supported scenarios
    below — but not a perturbation far under the discretisation of the Poisson and
    binomial draws, which is unobservable in the model anyway.  Skipped where no campaign
    has been consolidated, so a clean checkout is unaffected.
    """

    @pytest.fixture(scope="class")
    def optima(self):
        import pandas as pd
        path = Path("results/analysis/optima.csv")
        if not path.is_file():
            pytest.skip("no consolidated campaign in results/analysis")
        return pd.read_csv(path)

    @pytest.mark.parametrize("scenario_id", _GOLDEN_SCENARIOS)
    def test_the_reported_optimum_is_reproduced_exactly(self, optima, scenario_id):
        from src.campaign import scenario_params_for_id, team_config
        from src.io import load_simulation_horizon
        from src.selection import aggregate_config_result

        rows = optima[optima["scenario_id"] == scenario_id]
        if rows.empty:
            pytest.skip(f"{scenario_id} is not in this campaign")
        row = rows.iloc[0]

        horizon = load_simulation_horizon("config/horizon_default.json")
        params = scenario_params_for_id(scenario_id)
        # The re-simulation runs the disjoint block of replication indices
        # [K_search, K_search + K_final), never an extension of the search's own.
        result = run_configuration(
            team_config(row), horizon, params,
            K=horizon.K_final, k_offset=horizon.K_search,
        )
        agg = aggregate_config_result(result, master_seed=horizon.master_seed)

        assert agg.cost_mean == pytest.approx(row["cost_mean"], rel=0, abs=1e-9)
        assert agg.value_mean == pytest.approx(row["value_mean"], rel=0, abs=1e-12)
        assert agg.cost_ci[0] == pytest.approx(row["cost_ci_lo"], rel=0, abs=1e-9)
        assert agg.cost_ci[1] == pytest.approx(row["cost_ci_hi"], rel=0, abs=1e-9)
        # The three drivers are reported separately and must add back up to the objective.
        assert (agg.cost_quality_mean + agg.cost_human_mean
                + agg.cost_agent_mean) == pytest.approx(agg.cost_mean)
