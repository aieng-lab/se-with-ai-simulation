"""Tests for scripts/generate_search_configs.py — the 432-scenario grid."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from src.config import SCENARIO_GROUPS, scenario_tier_combinations
from src.io import load_search_config

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "generate_search_configs.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("generate_search_configs", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)   # type: ignore[union-attr]
    return module


@pytest.fixture(scope="module")
def script():
    return _load_script()


@pytest.fixture(scope="module")
def generated(script, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("search_configs")
    assert script.main(["--out-dir", str(out_dir)]) == 0
    return out_dir


class TestScenarioId:
    def test_one_token_per_group_in_group_order(self, script):
        assert script.scenario_id(("low", "medium", "high", "low", "medium")) == \
            "h-l_p-m_d-h_a-l_b-m"

    def test_ids_are_unique_across_the_whole_grid(self, script):
        combos = scenario_tier_combinations()
        ids = {script.scenario_id(t) for t in combos}
        # 3^3 * 4 * 4: `project` carries a fourth "very high" tier and `agent` a fourth
        # "pessimistic" one.
        assert len(ids) == len(combos) == 432

    def test_the_very_high_project_tier_gets_its_own_initial(self, script):
        assert script.scenario_id(("low", "veryhigh", "high", "low", "medium")) == \
            "h-l_p-v_d-h_a-l_b-m"

    def test_the_pessimistic_agent_tier_gets_its_own_initial(self, script):
        # The tier initial `p` coincides with the *group* initial of `project`; the
        # `<group>-<tier>` token format keeps that unambiguous.
        assert script.scenario_id(("low", "medium", "high", "pessimistic", "medium")) == \
            "h-l_p-m_d-h_a-p_b-m"


class TestGeneratedGrid:
    def test_writes_one_config_per_tier_combination(self, generated):
        assert len(list(generated.glob("search_*.json"))) == 432
        # One quarter of them are the very high project tier, and one quarter the
        # pessimistic agent tier.
        assert len(list(generated.glob("search_*_p-v_*.json"))) == 108
        assert len(list(generated.glob("search_*_a-p_*.json"))) == 108

    def test_every_config_loads_and_keeps_the_shared_setup(self, generated):
        reference = load_search_config("config/search_default.json")
        for path in generated.glob("search_*.json"):
            cfg = load_search_config(path)
            assert set(cfg.scenario_paths) == set(SCENARIO_GROUPS)
            # Only the scenario differs; horizon, range, budget and seed are shared so
            # that result differences are attributable to the scenario alone.
            assert cfg.horizon_path == reference.horizon_path
            assert cfg.N_values == reference.N_values
            assert cfg.R_values == reference.R_values
            assert cfg.F_freeze_values == reference.F_freeze_values
            assert cfg.sigma == reference.sigma
            assert cfg.epsilon == reference.epsilon
            assert cfg.budget == reference.budget
            assert cfg.n_samples_initial == reference.n_samples_initial
            assert cfg.n_generations == reference.n_generations
            assert cfg.n_offspring == reference.n_offspring
            assert cfg.n_elite == reference.n_elite
            assert cfg.seed == reference.seed

    def test_tier_combinations_are_covered_exactly_once(self, generated):
        combos = set()
        for path in generated.glob("search_*.json"):
            cfg = load_search_config(path)
            combo = tuple(
                Path(cfg.scenario_paths[group]).stem.split("_")[-1]
                for group in SCENARIO_GROUPS
            )
            assert combo not in combos, f"{combo} generated twice"
            combos.add(combo)
        assert combos == set(scenario_tier_combinations())

    def test_filename_matches_the_scenario_id_inside(self, generated):
        for path in generated.glob("search_*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            assert path.stem == f"search_{data['scenario_id']}"

    def test_scenario_paths_point_at_existing_tier_files(self, generated):
        cfg = load_search_config(generated / "search_h-l_p-m_d-h_a-l_b-m.json")
        assert cfg.scenario_paths == {
            "human": "config/human_low.json",
            "project": "config/project_medium.json",
            "debt": "config/debt_high.json",
            "agent": "config/agent_low.json",
            "bugimpact": "config/bugimpact_medium.json",
        }
        for path in cfg.scenario_paths.values():
            assert Path(path).is_file()


class TestOverwriteGuard:
    def test_refuses_to_overwrite_without_force(self, script, tmp_path):
        assert script.main(["--out-dir", str(tmp_path)]) == 0
        assert script.main(["--out-dir", str(tmp_path)]) == 1
        assert script.main(["--out-dir", str(tmp_path), "--force"]) == 0


class TestBaselineGrid:
    """`--baseline` writes the 36 human-only scenarios"""

    @pytest.fixture(scope="class")
    def baseline_dir(self, script, tmp_path_factory):
        out_dir = tmp_path_factory.mktemp("baseline_configs")
        assert script.main(["--baseline", "--out-dir", str(out_dir)]) == 0
        return out_dir

    def test_writes_thirty_six_configs(self, baseline_dir):
        # 3 human x 4 project x 3 bugimpact; debt and agent are held at the sentinel.
        assert len(list(baseline_dir.glob("search_*.json"))) == 36

    def test_every_id_marks_debt_and_agent_unused(self, baseline_dir):
        for path in baseline_dir.glob("search_*.json"):
            sid = json.loads(path.read_text())["scenario_id"]
            assert "_d-X_" in sid and "_a-X_" in sid
            assert path.name == f"search_{sid}.json"

    def test_every_config_runs_in_human_only_mode(self, baseline_dir):
        for path in baseline_dir.glob("search_*.json"):
            assert json.loads(path.read_text())["search"]["human_only"] is True

    def test_configs_load_and_point_at_the_unused_placeholders(self, baseline_dir):
        for path in baseline_dir.glob("search_*.json"):
            cfg = load_search_config(str(path))
            assert cfg.human_only is True
            assert cfg.scenario_paths["debt"].endswith("debt_unused.json")
            assert cfg.scenario_paths["agent"].endswith("agent_unused.json")
            for group in SCENARIO_GROUPS:
                assert Path(cfg.scenario_paths[group]).is_file()

    def test_the_varying_tiers_are_covered_exactly_once(self, baseline_dir):
        seen = set()
        for path in baseline_dir.glob("search_*.json"):
            cfg = load_search_config(str(path))
            seen.add(tuple(cfg.scenario_paths[g]
                           for g in ("human", "project", "bugimpact")))
        assert len(seen) == 36

    def test_baseline_and_campaign_ids_never_collide(self, script):
        campaign = {script.scenario_id(t) for t in scenario_tier_combinations()}
        baseline = {script.scenario_id(t) for t in script.baseline_tier_combinations()}
        assert len(baseline) == 36
        assert not (campaign & baseline)

    def test_refuses_to_overwrite_without_force(self, script, tmp_path):
        assert script.main(["--baseline", "--out-dir", str(tmp_path)]) == 0
        assert script.main(["--baseline", "--out-dir", str(tmp_path)]) == 1
        assert script.main(["--baseline", "--out-dir", str(tmp_path), "--force"]) == 0
