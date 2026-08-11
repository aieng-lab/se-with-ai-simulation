"""Tests for src/results.py — DataFrame shape and save/load round-trips."""

import pytest

from src.config import SimulationHorizon
from src.results import load_result, load_traces, save_result, save_traces, to_dataframe
from src.simulation import run_configuration


@pytest.fixture
def small_trace(base_config, base_params):
    s = SimulationHorizon(T=4, K=2, master_seed=0)
    return run_configuration(base_config, s, base_params, capture_traces=True)


@pytest.fixture
def small_result(base_config, base_params):
    s = SimulationHorizon(T=4, K=2, master_seed=0)
    return run_configuration(base_config, s, base_params)


@pytest.fixture
def small_agg(small_result):
    from src.pareto import aggregate_config_result
    return aggregate_config_result(small_result, master_seed=0)


class TestToDataframe:
    def test_shape(self, small_trace):
        df = to_dataframe(small_trace)
        assert len(df) == 2 * 4  # K=2, T=4

    def test_multiindex(self, small_trace):
        df = to_dataframe(small_trace)
        assert list(df.index.names) == ["replication", "sprint"]

    def test_sprint_range(self, small_trace):
        df = to_dataframe(small_trace)
        sprints = df.index.get_level_values("sprint").unique().tolist()
        assert sorted(sprints) == [0, 1, 2, 3]

    def test_no_nans(self, small_trace):
        df = to_dataframe(small_trace)
        assert not df.isnull().any().any()

    def test_F_completed_present(self, small_trace):
        df = to_dataframe(small_trace)
        assert "F_completed" in df.columns

    def test_no_t_column(self, small_trace):
        df = to_dataframe(small_trace)
        assert "t" not in df.columns


class TestSaveLoadResult:
    def test_round_trip(self, tmp_path, small_result, small_agg):
        path = tmp_path / "result.json"
        save_result(small_result, small_agg, path)
        loaded_result, loaded_agg = load_result(path)
        assert len(loaded_result.replications) == len(small_result.replications)
        assert loaded_result.replications[0].value == pytest.approx(small_result.replications[0].value)
        assert loaded_agg.value_mean == pytest.approx(small_agg.value_mean)

    def test_json_created(self, tmp_path, small_result, small_agg):
        path = tmp_path / "result.json"
        save_result(small_result, small_agg, path)
        assert path.exists()

    def test_load_wrong_A_length_raises(self, tmp_path):
        import json
        bad = {"config": {"N": 1, "A_dev": [1] * 5, "R": 1},
               "replications": [], "aggregated": {
                   "value_mean": 0, "quality_mean": 0, "cost_mean": 0,
                   "value_ci": [0, 0], "quality_ci": [0, 0], "cost_ci": [0, 0]}}
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            load_result(p)


class TestSaveLoadTraces:
    def test_round_trip_shape(self, tmp_path, small_trace):
        path = tmp_path / "traces.csv"
        save_traces(small_trace, path)
        df = load_traces(path)
        assert len(df) == 2 * 4

    def test_round_trip_values(self, tmp_path, small_trace):
        path = tmp_path / "traces.csv"
        original_df = to_dataframe(small_trace)
        save_traces(small_trace, path)
        loaded_df = load_traces(path)
        assert list(loaded_df.columns) == list(original_df.columns)
        assert (loaded_df["F_completed"] == original_df["F_completed"]).all()
