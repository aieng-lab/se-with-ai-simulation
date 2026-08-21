"""Tests for src/analysis.py — trajectory shapes and final-sprint distributions."""

import pytest

from src.config import SimulationHorizon
from src.analysis import final_distribution, mean_trajectory
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


class TestFinalDistribution:
    def test_length(self, trace_df):
        fd = final_distribution(trace_df, "F_completed")
        assert len(fd) == 5  # K=5 replications

    def test_is_last_sprint(self, trace_df):
        fd = final_distribution(trace_df, "F_completed")
        last = trace_df.xs(7, level="sprint")["F_completed"]
        assert (fd.values == last.values).all()
