"""Tests for src/io.py — value parsing, schema validation, round-trips."""

import json
from pathlib import Path

import pytest

from src.io import (
    _flatten_scenario,
    _parse_value,
    load_initial_state,
    load_scenario_params,
    load_simulation_horizon,
    load_team_config,
)
from src.parameters import ScenarioParameters
from src.state import SystemState


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


class TestLoadSimulationHorizon:
    def test_default_horizon_loads(self):
        s = load_simulation_horizon("config/horizon_default.json")
        assert s.T == 52 and s.K == 100 and s.master_seed == 20260423

    def test_basic(self, tmp_path):
        p = tmp_path / "h.json"
        p.write_text('{"T": 52, "K": 100, "master_seed": 42}')
        s = load_simulation_horizon(p)
        assert s.T == 52 and s.K == 100 and s.master_seed == 42

    def test_missing_T_raises(self, tmp_path):
        p = tmp_path / "h.json"
        p.write_text('{"K": 10, "master_seed": 0}')
        with pytest.raises(KeyError):
            load_simulation_horizon(p)


class TestLoadTeamConfig:
    def test_default_team_loads(self):
        team = load_team_config("config/team_default.json")
        assert team.N == 2 and team.M == 3 and team.R == 4 and team.F_freeze == 1
        assert len(team.A_dev) == 8
        assert len(team.A_qa) == 8


class TestLoadScenarioParams:
    def test_default_scenario_loads(self):
        params = load_scenario_params("config/scenario_default.json")
        assert isinstance(params, ScenarioParameters)
        assert len(params.c_token) == 8

    def test_returns_correct_values(self):
        params = load_scenario_params("config/scenario_default.json")
        assert params.r_arrival == pytest.approx(2.0)
        assert params.r_bugs == pytest.approx(0.1)
        assert params.r_bugs_human == pytest.approx(0.8)

    def test_high_load_scenario_loads(self):
        params = load_scenario_params("config/scenario_high_load.json")
        assert params.r_arrival == pytest.approx(4.0)

    def test_c_token_wrong_length_raises(self, tmp_path):
        path = Path("config/scenario_default.json")
        data = json.loads(path.read_text())
        data["costs"]["c_token"] = [0.1] * 5  # wrong length
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(data))
        with pytest.raises(ValueError):
            load_scenario_params(bad)

    def test_missing_parameter_raises(self, tmp_path):
        path = Path("config/scenario_default.json")
        data = json.loads(path.read_text())
        del data["productivity"]["r_arrival"]
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="missing"):
            load_scenario_params(bad)

    def test_extra_parameter_raises(self, tmp_path):
        path = Path("config/scenario_default.json")
        data = json.loads(path.read_text())
        data["productivity"]["nonexistent_param"] = 1.0
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="unknown"):
            load_scenario_params(bad)


class TestLoadInitialState:
    def test_default_config_gives_all_zero_state(self):
        state = load_initial_state("config/startstate_default.json")
        assert state == SystemState()

    def test_partial_config_sets_specified_fields(self, tmp_path):
        p = tmp_path / "start.json"
        p.write_text(json.dumps({"F_backlog": 10, "D_tech": 2.5}))
        state = load_initial_state(p)
        assert state.F_backlog == 10
        assert state.D_tech == pytest.approx(2.5)
        assert state.F_completed == 0
        assert state.B_hid_min == 0

    def test_int_fields_are_coerced_to_int(self, tmp_path):
        p = tmp_path / "start.json"
        p.write_text(json.dumps({"F_backlog": 5}))
        state = load_initial_state(p)
        assert isinstance(state.F_backlog, int)

    def test_float_fields_are_coerced_to_float(self, tmp_path):
        p = tmp_path / "start.json"
        p.write_text(json.dumps({"D_tech": 1}))
        state = load_initial_state(p)
        assert isinstance(state.D_tech, float)

    def test_unknown_field_raises(self, tmp_path):
        p = tmp_path / "start.json"
        p.write_text(json.dumps({"nonexistent": 1}))
        with pytest.raises(ValueError, match="Unknown"):
            load_initial_state(p)

    def test_empty_config_gives_default_state(self, tmp_path):
        p = tmp_path / "start.json"
        p.write_text("{}")
        state = load_initial_state(p)
        assert state == SystemState()
