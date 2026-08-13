"""Tests for src/io.py — value parsing, schema validation, round-trips."""

import dataclasses
import json
from pathlib import Path

import pytest

from src.io import (
    SCENARIO_GROUPS,
    _flatten_scenario,
    _parse_value,
    load_initial_state,
    load_scenario_params,
    load_search_config,
    load_simulation_horizon,
    load_team_config,
)
from src.parameters import ScenarioParameters
from src.state import SystemState


def tier_paths(tier: str, **overrides: str) -> dict[str, str]:
    """Paths to the five group files for `tier`, with optional per-group overrides."""
    paths = {group: f"config/{group}_{tier}.json" for group in SCENARIO_GROUPS}
    paths.update(overrides)
    return paths


def copy_group_file(group: str, tier: str, tmp_path: Path, mutate) -> Path:
    """Write a mutated copy of a group config to tmp_path and return its path."""
    data = json.loads(Path(f"config/{group}_{tier}.json").read_text())
    mutate(data[group])
    out = tmp_path / f"{group}_mutated.json"
    out.write_text(json.dumps(data))
    return out


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

    def test_description_string_is_skipped(self):
        flat = _flatten_scenario({"description": "docs", "s1": {"a": 1}})
        assert flat == {"a": 1}

    def test_non_string_description_still_validated(self):
        with pytest.raises(ValueError):
            _flatten_scenario({"description": [1, 2]})

    def test_empty_returns_empty(self):
        assert _flatten_scenario({}) == {}


class TestLoadSimulationHorizon:
    def test_default_horizon_loads(self):
        # The Monte Carlo protocol of paper §4.2, Table 6: two years of weekly sprints.
        s = load_simulation_horizon("config/horizon_default.json")
        assert s.T == 104 and s.K == 100 and s.master_seed == 42

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
        assert team.N == 2 and team.R == 4 and team.F_freeze == 1
        assert len(team.A_dev) == 8
        assert len(team.A_qa) == 8


class TestLoadScenarioParams:
    def test_medium_scenario_loads(self):
        params = load_scenario_params(tier_paths("medium"))
        assert isinstance(params, ScenarioParameters)
        assert len(params.c_token) == 8

    def test_returns_correct_values(self):
        params = load_scenario_params(tier_paths("medium"))
        assert params.r_arrival == pytest.approx(12.0)
        assert params.r_bugs == pytest.approx(0.03)
        assert params.r_bugs_human == pytest.approx(0.74)
        assert params.c_salary == pytest.approx(150_000.0)
        assert params.c_token[0] == pytest.approx(3.80)

    @pytest.mark.parametrize("tier", ["low", "medium", "high"])
    def test_every_tier_loads(self, tier):
        params = load_scenario_params(tier_paths(tier))
        assert isinstance(params, ScenarioParameters)

    def test_groups_can_be_mixed(self):
        """The five groups are orthogonal, so tiers combine freely (paper §4.1)."""
        params = load_scenario_params(
            tier_paths("medium", project="config/project_high.json")
        )
        assert params.r_arrival == pytest.approx(75.0)
        assert params.r_bugs_human == pytest.approx(0.74)   # human group still medium

    def test_groups_partition_all_parameters(self):
        """Union of the group files must be exactly the ScenarioParameters fields."""
        keys: set[str] = set()
        for group in SCENARIO_GROUPS:
            data = json.loads(Path(f"config/{group}_medium.json").read_text())
            group_keys = set(_flatten_scenario(data))
            assert not (keys & group_keys), f"{group} duplicates {keys & group_keys}"
            keys |= group_keys
        assert keys == {f.name for f in dataclasses.fields(ScenarioParameters)}

    def test_missing_group_raises(self):
        paths = tier_paths("medium")
        del paths["debt"]
        with pytest.raises(ValueError, match="missing groups"):
            load_scenario_params(paths)

    def test_unknown_group_raises(self):
        paths = tier_paths("medium")
        paths["nonexistent_group"] = "config/debt_medium.json"
        with pytest.raises(ValueError, match="unknown groups"):
            load_scenario_params(paths)

    def test_parameter_in_two_groups_raises(self, tmp_path):
        overlap = tmp_path / "debt_overlap.json"
        overlap.write_text(json.dumps({"debt": {"r_arrival": 1.0}}))
        with pytest.raises(ValueError, match="defined in both"):
            load_scenario_params(tier_paths("medium", debt=str(overlap)))

    def test_c_token_wrong_length_raises(self, tmp_path):
        def mutate(section):
            section["c_token"] = [0.1] * 5

        bad = copy_group_file("agent", "medium", tmp_path, mutate)
        with pytest.raises(ValueError, match="c_token"):
            load_scenario_params(tier_paths("medium", agent=str(bad)))

    def test_missing_parameter_raises(self, tmp_path):
        bad = copy_group_file(
            "project", "medium", tmp_path, lambda s: s.pop("r_arrival")
        )
        with pytest.raises(ValueError, match="missing"):
            load_scenario_params(tier_paths("medium", project=str(bad)))

    def test_extra_parameter_raises(self, tmp_path):
        def mutate(section):
            section["nonexistent_param"] = 1.0

        bad = copy_group_file("project", "medium", tmp_path, mutate)
        with pytest.raises(ValueError, match="unknown"):
            load_scenario_params(tier_paths("medium", project=str(bad)))


class TestLoadSearchConfig:
    def test_default_search_config_loads(self):
        cfg = load_search_config("config/search_default.json")
        assert set(cfg.scenario_paths) == set(SCENARIO_GROUPS)
        assert cfg.horizon_path == "config/horizon_default.json"
        assert cfg.strategy == "evolutionary"

    def test_default_search_config_matches_paper_search_range(self):
        # Paper §4.2, Table 6: search range, the evolutionary E = 10,000 evaluation
        # budget split 10 % / 90 % between the initial pool and the generation loop, and
        # the n_starts restarts of the descent, which is budgeted by convergence instead.
        cfg = load_search_config("config/search_default.json")
        # Staffing levels from a one-day-per-week part-timer to a two-pizza team;
        # N is a staffing level, so the values are floats.
        assert cfg.N_values == (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0)
        assert cfg.R_values == (1, 2, 4, 13)
        assert cfg.F_freeze_values == (0, 1, 2)
        assert cfg.sigma == 2.0
        assert cfg.epsilon == 0.02
        assert cfg.budget == 10_000
        assert cfg.n_samples_initial == 1_000
        assert cfg.n_generations * cfg.n_offspring == 9_000
        assert cfg.n_samples_initial + cfg.n_generations * cfg.n_offspring == cfg.budget
        assert cfg.n_elite == 100
        assert cfg.n_starts == 20
        assert cfg.samples == "best"
        assert cfg.seed == 42

    def test_sigma_defaults_and_is_validated(self, tmp_path):
        data = json.loads(Path("config/search_default.json").read_text())
        del data["search"]["sigma"]
        path = tmp_path / "search.json"
        path.write_text(json.dumps(data))
        assert load_search_config(path).sigma == 2.0

        data["search"]["sigma"] = 0.5
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="sigma must be >= 1"):
            load_search_config(path)

    def test_samples_policy_defaults_and_is_validated(self, tmp_path):
        data = json.loads(Path("config/search_default.json").read_text())
        del data["search"]["samples"]
        path = tmp_path / "search.json"
        path.write_text(json.dumps(data))
        assert load_search_config(path).samples == "best"

        data["search"]["samples"] = "everything"
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="samples must be one of"):
            load_search_config(path)

    def test_scenario_id_is_read_from_the_top_level(self):
        cfg = load_search_config("config/search_default.json")
        assert cfg.scenario_id == "h-m_p-m_d-m_a-m_b-m"

    def test_scenario_paths_resolve_to_params(self):
        cfg = load_search_config("config/search_default.json")
        params = load_scenario_params(cfg.scenario_paths)
        assert params.r_arrival == pytest.approx(12.0)

    def test_scalar_scenario_raises(self, tmp_path):
        data = json.loads(Path("config/search_default.json").read_text())
        data["scenario"] = "config/human_medium.json"
        bad = tmp_path / "search.json"
        bad.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="must be a JSON object"):
            load_search_config(bad)

    def test_incomplete_scenario_group_set_raises(self, tmp_path):
        data = json.loads(Path("config/search_default.json").read_text())
        del data["scenario"]["debt"]
        bad = tmp_path / "search.json"
        bad.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="scenario_paths"):
            load_search_config(bad)


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
