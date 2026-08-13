"""Tests for src/campaign.py — consolidating a campaign into flat tables."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.campaign import (
    build_tables,
    optimum_row,
    scenario_params_for_id,
    scenario_row,
    search_dirs,
    team_config,
    tiers_from_scenario_id,
)
from src.config import AGENT_ROLE_COUNT, SCENARIO_GROUPS


class TestScenarioIds:
    def test_round_trip(self):
        tiers = tiers_from_scenario_id("h-l_p-m_d-h_a-l_b-m")
        assert tiers == {"human": "low", "project": "medium", "debt": "high",
                         "agent": "low", "bugimpact": "medium"}

    def test_reads_the_very_high_project_tier(self):
        assert tiers_from_scenario_id("h-m_p-v_d-m_a-m_b-m")["project"] == "veryhigh"

    @pytest.mark.parametrize("bad", [
        "h-l_p-m_d-h_a-l",              # too few tokens
        "h-l_p-m_d-h_a-l_b-m_x-y",      # too many
        "h-l_p-m_d-h_a-l_b-z",          # unknown tier initial
        "x-l_p-m_d-h_a-l_b-m",          # wrong group initial
        "hl_p-m_d-h_a-l_b-m",           # missing separator
    ])
    def test_rejects_malformed(self, bad):
        with pytest.raises(ValueError):
            tiers_from_scenario_id(bad)

    def test_params_read_the_right_tiers(self):
        params = scenario_params_for_id("h-l_p-m_d-h_a-l_b-h")
        # bugimpact high: omega_maj is the dollar cost of an escaped major bug (§4.1).
        assert params.omega_maj == 7_000_000.0
        assert params.omega_min == 250.0

    def test_params_are_cached(self):
        a = scenario_params_for_id("h-m_p-m_d-m_a-m_b-m")
        b = scenario_params_for_id("h-m_p-m_d-m_a-m_b-m")
        assert a is b


# ---------------------------------------------------------------------------
# A synthetic one-search campaign on disk
# ---------------------------------------------------------------------------

_SID = "h-m_p-m_d-m_a-m_b-m"


def _objective(cost=1000.0, value=12.0, admissible=True) -> dict:
    return {
        "N": 2.0, "R": 4, "F_freeze": 1,
        "A_dev": [2, 0, 1, 1, 3, 0, 1, 0],
        "A_qa": [0, 0, 2, 1, 1, 0, 1, 0],
        "cost_mean": cost, "cost_ci": [cost * 0.9, cost * 1.1],
        "value_mean": value, "value_ci": [value * 0.99, value * 1.01],
        "feasible_throughput": admissible,
        "cost_quality_mean": cost * 0.5,
        "cost_human_mean": cost * 0.3,
        "cost_agent_mean": cost * 0.2,
        "actions_mean": [10.0] * AGENT_ROLE_COUNT,
        "impl_human_mean": 4.0, "impl_agents_mean": 6.0,
        "fixes_human_mean": 1.0, "fixes_agent_mean": 2.0,
    }


@pytest.fixture
def campaign(tmp_path) -> Path:
    d = tmp_path / _SID / "evolutionary"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({
        "scenario_id": _SID, "strategy": "evolutionary",
        "horizon": {"path": "config/horizon_default.json", "T": 104, "K": 100,
                    "master_seed": 42},
        "n_evaluated": 100, "n_simulated": 85, "best_admissible": True,
        "wall_clock_s": 12.5,
    }))
    (d / "search_summary.json").write_text(json.dumps({
        "best": "N2_R4_00007",
        "indistinguishable": ["N2_R4_00007", "N2_R4_00011"],
        "throughput_floor": 11.76,
        "objectives": {"N2_R4_00007": _objective(1000.0),
                       "N2_R4_00011": _objective(1050.0)},
    }))
    return tmp_path


class TestSearchDirs:
    def test_finds_the_finished_search(self, campaign):
        found = search_dirs(campaign)
        assert [(sid, strategy) for sid, strategy, _ in found] == [(_SID, "evolutionary")]

    def test_ignores_a_directory_without_a_summary(self, campaign):
        (campaign / _SID / "descent").mkdir()
        assert len(search_dirs(campaign)) == 1


class TestOptimumRow:
    def _row(self, campaign):
        d = campaign / _SID / "evolutionary"
        manifest = json.loads((d / "manifest.json").read_text())
        summary = json.loads((d / "search_summary.json").read_text())
        return optimum_row(_SID, "evolutionary", manifest, summary)

    def test_describes_the_reported_optimum(self, campaign):
        row = self._row(campaign)
        assert row["config_key"] == "N2_R4_00007"
        assert row["cost_mean"] == 1000.0
        assert (row["cost_quality_mean"] + row["cost_human_mean"]
                + row["cost_agent_mean"]) == pytest.approx(row["cost_mean"])
        assert row["n_indistinguishable"] == 2

    def test_carries_every_tier_and_both_compositions(self, campaign):
        row = self._row(campaign)
        assert all(f"tier_{g}" in row for g in SCENARIO_GROUPS)
        assert [row[f"a_dev_{k + 1}"] for k in range(AGENT_ROLE_COUNT)] == \
            [2, 0, 1, 1, 3, 0, 1, 0]
        assert [row[f"a_qa_{k + 1}"] for k in range(AGENT_ROLE_COUNT)] == \
            [0, 0, 2, 1, 1, 0, 1, 0]

    def test_value_ratio_is_the_fraction_of_the_arrival_rate(self, campaign):
        row = self._row(campaign)
        assert row["value_ratio"] == pytest.approx(row["value_mean"] / row["r_arrival"])
        # The medium project tier arrives at 12 issues/week, matching the fixture.
        assert row["value_ratio"] == pytest.approx(1.0)

    def test_supervised_counts_exclude_the_background_services(self, campaign):
        row = self._row(campaign)
        assert row["n_sup_dev"] == 2 + 0 + 1 + 1 + 1 + 0    # a_5 = 3 excluded
        assert row["n_sup_qa"] == 0 + 0 + 2 + 1 + 1 + 0

    def test_phase_sprint_counts_partition_the_horizon(self, campaign):
        row = self._row(campaign)
        assert row["T_dev"] + row["T_qa"] == 104
        assert row["T_qa"] == 104 // 4     # one freeze sprint per release cycle

    def test_no_optimum_yields_no_row(self, campaign):
        d = campaign / _SID / "evolutionary"
        manifest = json.loads((d / "manifest.json").read_text())
        summary = json.loads((d / "search_summary.json").read_text())
        summary["best"] = None
        assert optimum_row(_SID, "evolutionary", manifest, summary) is None


class TestBuildTables:
    def test_one_row_per_search_in_each_table(self, campaign):
        optima, scenarios = build_tables(campaign)
        assert len(optima) == 1 and len(scenarios) == 1
        assert scenarios.iloc[0]["n_infeasible"] == 100 - 85
        assert scenarios.iloc[0]["K"] == 100

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="is not a directory"):
            build_tables(tmp_path / "nope")

    def test_empty_campaign_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no finished searches"):
            build_tables(tmp_path)


class TestTeamConfig:
    def test_rebuilds_the_configuration_from_a_row(self):
        row = {"N": 2.0, "R": 4, "F_freeze": 1}
        for k in range(AGENT_ROLE_COUNT):
            row[f"a_dev_{k + 1}"] = 1 if k == 0 else 0
            row[f"a_qa_{k + 1}"] = 1 if k == 2 else 0
        cfg = team_config(row)
        assert cfg.N == 2.0 and cfg.R == 4 and cfg.F_freeze == 1
        assert cfg.A_dev[0] == 1 and cfg.A_qa[2] == 1
