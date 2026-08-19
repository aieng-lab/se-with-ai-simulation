"""Tests for src/analysis.py — trajectory shapes, quantile bands, convergence."""

import pytest

from src.config import SimulationHorizon
from src.analysis import convergence_check, final_distribution, mean_trajectory, quantile_band
from src.results import to_dataframe
from src.simulation import run_configuration


@pytest.fixture
def trace_df(base_config, base_params):
    s = SimulationHorizon(T=8, K_search=5, K_final=5, master_seed=0)
    trace = run_configuration(base_config, s, base_params, capture_traces=True)
    return to_dataframe(trace)


class TestMeanTrajectory:
    def test_length(self, trace_df):
        traj = mean_trajectory(trace_df, "F_completed")
        assert len(traj) == 8  # T sprints

    def test_F_completed_nondecreasing(self, trace_df):
        traj = mean_trajectory(trace_df, "F_completed")
        for i in range(len(traj) - 1):
            assert traj.iloc[i] <= traj.iloc[i + 1]

    def test_constant_field(self, trace_df):
        # D initialised to 0.0; may grow after sprint 0
        first_sprint = trace_df.xs(0, level="sprint")["D_tech"]
        assert (first_sprint >= 0).all()


class TestQuantileBand:
    def test_columns(self, trace_df):
        band = quantile_band(trace_df, "D_tech")
        assert set(band.columns) == {"mean", "lo", "hi"}

    def test_length(self, trace_df):
        band = quantile_band(trace_df, "D_tech")
        assert len(band) == 8

    def test_ordering(self, trace_df):
        band = quantile_band(trace_df, "F_completed")
        assert (band["lo"] <= band["mean"]).all()
        assert (band["mean"] <= band["hi"]).all()

    def test_extreme_quantiles(self, trace_df):
        band = quantile_band(trace_df, "F_completed", q_lo=0.0, q_hi=1.0)
        # lo should be min, hi should be max
        assert (band["lo"] <= band["hi"]).all()


class TestFinalDistribution:
    def test_length(self, trace_df):
        fd = final_distribution(trace_df, "F_completed")
        assert len(fd) == 5  # K=5 replications

    def test_is_last_sprint(self, trace_df):
        fd = final_distribution(trace_df, "F_completed")
        last = trace_df.xs(7, level="sprint")["F_completed"]
        assert (fd.values == last.values).all()


class TestConvergenceCheck:
    def test_single_config_stable(self, base_config, base_params):
        """With one candidate there is nothing for halving K to change."""
        s = SimulationHorizon(T=8, K_search=20, K_final=20, master_seed=0)
        result = run_configuration(base_config, s, base_params)
        check = convergence_check([result], master_seed=0)
        assert check["stable"]
        assert check["best_K"] == check["best_half_K"] == 0
        assert check["rel_ci_width_K"] >= 0.0

    def test_empty_pool_reports_no_optimum(self):
        check = convergence_check([], master_seed=0)
        assert check["best_K"] is None and check["best_half_K"] is None
        assert check["stable"]
        assert check["rel_ci_width_K"] == 0.0

    def test_picks_the_cheaper_of_two_configs(self, base_config, base_params):
        """The check is about c*, so a clearly cheaper candidate must win at both K."""
        import dataclasses
        s = SimulationHorizon(T=8, K_search=20, K_final=20, master_seed=0)
        cheap = run_configuration(base_config, s, base_params)
        # A second candidate that is otherwise identical but pays ten developers.
        pricey_cfg = dataclasses.replace(base_config, N=10.0)
        pricey = run_configuration(
            pricey_cfg, s, dataclasses.replace(base_params, c_salary=1e6)
        )
        check = convergence_check([pricey, cheap], master_seed=0)
        assert check["best_K"] == 1        # the cheap one, second in the list
        assert check["stable"]
