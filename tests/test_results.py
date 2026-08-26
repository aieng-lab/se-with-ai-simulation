"""Tests for src/results.py — the JSON save/load round-trip."""

import pytest

from src.config import SimulationHorizon
from src.results import load_result, save_result
from src.simulation import run_configuration


@pytest.fixture
def small_result(base_config, base_params):
    s = SimulationHorizon(T=4, K_search=2, K_final=2, master_seed=0)
    return run_configuration(base_config, s, base_params)


@pytest.fixture
def small_agg(small_result):
    from src.selection import aggregate_config_result
    return aggregate_config_result(small_result, master_seed=0)


class TestSaveLoadResult:
    def test_round_trip(self, tmp_path, small_result, small_agg):
        path = tmp_path / "result.json"
        save_result(small_result, small_agg, path)
        loaded_result, loaded_agg = load_result(path)
        assert len(loaded_result.replications) == len(small_result.replications)
        assert loaded_result.replications[0].value == pytest.approx(small_result.replications[0].value)
        assert loaded_agg.value_mean == pytest.approx(small_agg.value_mean)

    def test_round_trip_keeps_the_reported_extras(self, tmp_path, small_result, small_agg):
        """The cost drivers, the per-role work and the contribution counts all survive."""
        path = tmp_path / "result.json"
        save_result(small_result, small_agg, path)
        loaded_result, loaded_agg = load_result(path)

        for loaded, original in zip(loaded_result.replications, small_result.replications):
            assert loaded.cost_quality == pytest.approx(original.cost_quality)
            assert loaded.cost_human == pytest.approx(original.cost_human)
            assert loaded.cost_agent == pytest.approx(original.cost_agent)
            assert loaded.actions == pytest.approx(original.actions)
            for field in ("impl_human", "impl_agents", "fixes_human", "fixes_agent"):
                assert getattr(loaded, field) == pytest.approx(getattr(original, field))

        for field in ("cost_quality_mean", "cost_human_mean", "cost_agent_mean",
                      "impl_human_mean", "impl_agents_mean",
                      "fixes_human_mean", "fixes_agent_mean"):
            assert getattr(loaded_agg, field) == pytest.approx(getattr(small_agg, field))
        assert loaded_agg.actions_mean == pytest.approx(small_agg.actions_mean)

    def test_load_result_with_a_missing_field_raises(self, tmp_path):
        """Every field save_result writes is required; there is no lean variant."""
        import json
        lean = {"config": {"N": 1, "A_dev": [1] * 8, "R": 4, "F_freeze": 0,
                           "A_qa": [0] * 8},
                "replications": [{"value": 1.0, "cost": 3.0}],
                "aggregated": {"value_mean": 1.0, "value_ci": [1.0, 1.0],
                               "cost_mean": 3.0, "cost_ci": [3.0, 3.0]}}
        p = tmp_path / "lean.json"
        p.write_text(json.dumps(lean))
        with pytest.raises(KeyError):
            load_result(p)

    def test_load_wrong_A_length_raises(self, tmp_path):
        import json
        bad = {"config": {"N": 1, "A_dev": [1] * 5, "R": 1, "F_freeze": 0,
                          "A_qa": []},
               "replications": [], "aggregated": {
                   "value_mean": 0, "value_ci": [0, 0],
                   "cost_mean": 0, "cost_ci": [0, 0]}}
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            load_result(p)
