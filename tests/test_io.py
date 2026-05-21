"""Tests for src/io.py — spec parsing, schema validation, round-trips."""

import json
from pathlib import Path

import pytest

from src.io import _flatten_scenario, _load_sweep, _parse_spec, load_mean_parameters, load_scenario_config
from src.sampling import ParameterSpec


class TestParseSpec:
    def test_scalar_int(self):
        spec = _parse_spec("x", 5)
        assert spec.dist == "fixed" and spec.mean == 5.0

    def test_scalar_float(self):
        spec = _parse_spec("x", 3.14)
        assert spec.dist == "fixed" and spec.mean == pytest.approx(3.14)

    def test_lognormal(self):
        spec = _parse_spec("x", {"dist": "lognormal", "mean": 2.0, "cv": 0.2})
        assert spec.dist == "lognormal"
        assert spec.mean == 2.0 and spec.cv == 0.2

    def test_beta(self):
        spec = _parse_spec("x", {"dist": "beta", "mean": 0.5, "std": 0.1})
        assert spec.dist == "beta"
        assert spec.mean == 0.5 and spec.std == 0.1

    def test_uniform(self):
        spec = _parse_spec("x", {"dist": "uniform", "lo": 1.0, "hi": 5.0})
        assert spec.dist == "uniform"
        assert spec.lo == 1.0 and spec.hi == 5.0
        assert spec.mean == pytest.approx(3.0)  # midpoint

    def test_unknown_dist_raises(self):
        with pytest.raises(ValueError, match="Unknown distribution"):
            _parse_spec("x", {"dist": "gamma", "shape": 1.0})

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            _parse_spec("x", "not_a_number")


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
        assert s.T == 52 and s.K == 100 and s.master_seed == 42 and s.N_PO == 1

    def test_N_PO_override(self):
        s = _load_sweep({"T": 10, "K": 5, "master_seed": 0, "N_PO": 3})
        assert s.N_PO == 3

    def test_missing_T_raises(self):
        with pytest.raises(KeyError):
            _load_sweep({"K": 10, "master_seed": 0})


class TestLoadScenarioConfig:
    def test_default_scenario_loads(self):
        sweep, priors = load_scenario_config("config/default_scenario.json")
        assert sweep.T == 52 and sweep.K == 100
        assert len(priors.c_token) == 13
        assert len(priors.specs) == 74  # all fields except c_token

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
        del data["scenario"]["productivity"]["lambda_F"]
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


class TestLoadMeanParameters:
    def test_returns_scenario_parameters(self):
        _, params = load_mean_parameters("config/default_scenario.json")
        assert params.lambda_F == pytest.approx(2.0)

    def test_integer_fields(self):
        _, params = load_mean_parameters("config/default_scenario.json")
        assert isinstance(params.Q_thr_rev, int)
        assert isinstance(params.R_warmup, int)
