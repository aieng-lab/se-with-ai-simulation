"""Tests for src/io.py — value parsing, schema validation, round-trips."""

import json
from pathlib import Path

import pytest

from src.io import _flatten_scenario, _load_sweep, _parse_value, load_scenario_config
from src.parameters import ScenarioParameters


class TestParseValue:
    def test_scalar_int(self):
        assert _parse_value("x", 5) == 5.0

    def test_scalar_float(self):
        assert _parse_value("x", 3.14) == pytest.approx(3.14)

    def test_legacy_value_dict(self):
        assert _parse_value("x", {"value": 2.5}) == pytest.approx(2.5)

    def test_dict_without_value_raises(self):
        with pytest.raises(ValueError, match="'value' key"):
            _parse_value("x", {"dist": "lognormal", "mean": 1.0})

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            _parse_value("x", "not_a_number")


class TestFlattenScenario:
    def test_merges_sections(self):
        flat = _flatten_scenario({"s1": {"a": 1, "b": 2}, "s2": {"c": 3}})
        assert flat == {"a": 1, "b": 2, "c": 3}

    def test_duplicate_key_raises(self):
        with pytest.raises(ValueError, match="Duplicate"):
            _flatten_scenario({"s1": {"a": 1}, "s2": {"a": 2}})

    def test_non_dict_section_raises(self):
        with pytest.raises(ValueError):
            _flatten_scenario({"s1": [1, 2, 3]})

    def test_empty_returns_empty(self):
        assert _flatten_scenario({}) == {}


class TestLoadSweep:
    def test_basic(self):
        s = _load_sweep({"T": 52, "K": 100, "master_seed": 42})
        assert s.T == 52 and s.K == 100 and s.master_seed == 42

    def test_missing_T_raises(self):
        with pytest.raises(KeyError):
            _load_sweep({"K": 10, "master_seed": 0})


class TestLoadScenarioConfig:
    def test_default_scenario_loads(self):
        sweep, params = load_scenario_config("config/default_scenario.json")
        assert sweep.T == 52 and sweep.K == 100
        assert isinstance(params, ScenarioParameters)
        assert len(params.c_token) == 8

    def test_returns_correct_values(self):
        _, params = load_scenario_config("config/default_scenario.json")
        assert params.r_arrival == pytest.approx(2.0)
        assert params.r_bugs == pytest.approx(0.1)
        assert params.r_bugs_human == pytest.approx(0.8)

    def test_c_token_wrong_length_raises(self, tmp_path):
        path = Path("config/default_scenario.json")
        data = json.loads(path.read_text())
        data["scenario"]["costs"]["c_token"] = [0.1] * 5  # wrong length
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(data))
        with pytest.raises(ValueError):
            load_scenario_config(bad)

    def test_missing_parameter_raises(self, tmp_path):
        path = Path("config/default_scenario.json")
        data = json.loads(path.read_text())
        del data["scenario"]["productivity"]["r_arrival"]
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="missing"):
            load_scenario_config(bad)

    def test_extra_parameter_raises(self, tmp_path):
        path = Path("config/default_scenario.json")
        data = json.loads(path.read_text())
        data["scenario"]["productivity"]["nonexistent_param"] = 1.0
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="unknown"):
            load_scenario_config(bad)
