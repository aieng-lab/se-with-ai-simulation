"""Tests for src/campaign.py — consolidating a campaign into flat tables."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.campaign import (
    baseline_uplift,
    build_tables,
    convergence_summary,
    expected_modes,
    is_baseline_scenario,
    optimum_row,
    scenario_params_for_id,
    search_dirs,
    team_config,
    tiers_from_scenario_id,
)
from src.config import (
    AGENT_ROLE_COUNT,
    SEARCH_MODE,
    HUMAN_ONLY_MODE,
    SCENARIO_GROUPS,
)


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
        # bugimpact high: omega_maj is the dollar cost of an escaped major bug.
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
    d = tmp_path / _SID / "search"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({
        "scenario_id": _SID, "mode": "search",
        "horizon": {"path": "config/horizon_default.json", "T": 104,
                    "K_search": 100, "K_final": 1000, "master_seed": 42},
        "n_evaluated": 100, "n_simulated": 85, "best_admissible": True,
        "wall_clock_s": 12.5,
    }))
    (d / "search_summary.json").write_text(json.dumps({
        "best": "N2_R4_00007",
        "indistinguishable": ["N2_R4_00007", "N2_R4_00011"],
        "throughput_floor": 11.76,
        "K_search": 100, "K_final": 1000,
        # The search's own K'-replication estimates of every simulated candidate …
        "objectives": {"N2_R4_00007": _objective(1200.0),
                       "N2_R4_00011": _objective(1250.0)},
        # … and the K-replication estimates the report comes from.
        "final_objectives": {"N2_R4_00007": _objective(1000.0),
                             "N2_R4_00011": _objective(1050.0)},
    }))
    (d / "convergence.json").write_text(json.dumps({
        "scenario_id": _SID, "mode": "search",
        "n_elite": 10, "n_generations": 3, "n_generations_run": 3,
        "K_search": 100, "K_final": 1000,
        "elite_turnover": [0.5, 0.2, 0.1],
        "incumbent_changed": [True, False, False],
        "incumbent_turnover": 1 / 3,
        "best_cost": [1400.0, 1300.0, 1200.0],
        "i_star": {"n_prelim": 5, "parsimony_removed_prelim": 1, "n_resimulated": 4,
                   "n_discarded_throughput": 1, "n_after_resim": 3, "n_final": 2,
                   "parsimony_removed": 1, "best_changed_by_resim": True,
                   "rho_mean": 0.04, "rho_best": 0.03},
        "throughput_floor": 11.76, "best_admissible": True, "value_ratio": 0.99,
        "wall_clock_s": 12.5, "resimulation_s": 3.0,
    }))
    return tmp_path


class TestSearchDirs:
    def test_finds_the_finished_search(self, campaign):
        found = search_dirs(campaign)
        assert [(sid, mode) for sid, mode, _ in found] == [(_SID, "search")]

    def test_ignores_a_directory_without_a_summary(self, campaign):
        (campaign / _SID / HUMAN_ONLY_MODE).mkdir()
        assert len(search_dirs(campaign)) == 1


class TestOptimumRow:
    def _row(self, campaign):
        d = campaign / _SID / "search"
        manifest = json.loads((d / "manifest.json").read_text())
        summary = json.loads((d / "search_summary.json").read_text())
        return optimum_row(_SID, "search", manifest, summary)

    def test_describes_the_reported_optimum(self, campaign):
        row = self._row(campaign)
        assert row["config_key"] == "N2_R4_00007"
        assert row["cost_mean"] == 1000.0
        assert (row["cost_quality_mean"] + row["cost_human_mean"]
                + row["cost_agent_mean"]) == pytest.approx(row["cost_mean"])
        assert row["n_indistinguishable"] == 2

    def test_cost_shares_partition_the_total(self, campaign):
        row = self._row(campaign)
        assert row["cost_quality_share"] == pytest.approx(0.5)
        assert row["cost_human_share"] == pytest.approx(0.3)
        assert row["cost_agent_share"] == pytest.approx(0.2)
        assert (row["cost_quality_share"] + row["cost_human_share"]
                + row["cost_agent_share"]) == pytest.approx(1.0)

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

    def test_supervised_counts_exclude_the_autonomous_roles(self, campaign):
        """n_sup ranges over K_sup = {1,2,3,7,8}; a_4, a_5 and a_6 do not enter it."""
        row = self._row(campaign)
        assert row["n_sup_dev"] == 2 + 0 + 1 + 1 + 0        # a_4 = 1, a_5 = 3 excluded
        assert row["n_sup_qa"] == 0 + 0 + 2 + 1 + 0         # a_4 = 1, a_5 = 1 excluded

    def test_phase_sprint_counts_partition_the_horizon(self, campaign):
        row = self._row(campaign)
        assert row["T_dev"] + row["T_qa"] == 104
        assert row["T_qa"] == 104 // 4     # one freeze sprint per release cycle

    def test_no_optimum_yields_no_row(self, campaign):
        d = campaign / _SID / "search"
        manifest = json.loads((d / "manifest.json").read_text())
        summary = json.loads((d / "search_summary.json").read_text())
        summary["best"] = None
        assert optimum_row(_SID, "search", manifest, summary) is None


class TestBuildTables:
    def test_one_row_per_search_in_each_table(self, campaign):
        optima, scenarios, convergence, turnover = build_tables(campaign)
        assert len(optima) == 1 and len(scenarios) == 1
        assert scenarios.iloc[0]["n_infeasible"] == 100 - 85
        assert scenarios.iloc[0]["K_search"] == 100
        assert scenarios.iloc[0]["K_final"] == 1000

    def test_the_optimum_is_reported_at_K_not_at_K_prime(self, campaign):
        """`optima.csv` must carry the re-simulated estimates, not the search's."""
        optima, _, _, _ = build_tables(campaign)
        assert optima.iloc[0]["cost_mean"] == 1000.0        # final_objectives
        assert optima.iloc[0]["cost_mean"] != 1200.0        # objectives, at K'

    def test_convergence_tables_carry_the_5_1_measures(self, campaign):
        _, _, convergence, turnover = build_tables(campaign)
        assert len(convergence) == 1
        row = convergence.iloc[0]
        assert row["n_prelim"] == 5 and row["n_resimulated"] == 4
        assert row["n_after_resim"] == 3 and row["n_final"] == 2
        assert row["parsimony_removed_prelim"] == 1 and row["parsimony_removed"] == 1
        assert row["n_discarded_throughput"] == 1
        assert bool(row["best_changed_by_resim"]) is True
        assert row["tau_first"] == 0.5 and row["tau_last"] == 0.1
        assert row["n_generations_run"] == 3
        # One row per generation, in order, and the incumbent flag travels with it.
        assert list(turnover["generation"]) == [1, 2, 3]
        assert list(turnover["tau"]) == [0.5, 0.2, 0.1]
        assert list(turnover["incumbent_changed"]) == [True, False, False]

    def test_convergence_summary_reports_every_stage(self, campaign):
        """The summary reports all four sizes of I* and both parsimony applications."""
        _, _, convergence, _ = build_tables(campaign)
        text = convergence_summary(convergence)
        assert "Size of I* at the four stages" in text
        assert "preliminary, at K'" in text
        assert "after the parsimony tie-break" in text
        assert "after the re-simulation at K" in text
        assert "after the second tie-break" in text
        assert "parsimony removed at K'" in text
        assert "parsimony removed at K " in text
        # The exemption the tie-break no longer needs is gone from the report.
        assert "parsimony-dominated" not in text
        assert "Inadmissible searches: 0/1" in text

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


# ---------------------------------------------------------------------------
# The human-only baseline
# ---------------------------------------------------------------------------

_BASELINE_SID = "h-m_p-m_d-X_a-X_b-m"


def _baseline_objective(cost=5000.0, value=12.0) -> dict:
    obj = _objective(cost=cost, value=value)
    obj.update({
        "A_dev": [0] * AGENT_ROLE_COUNT,
        "A_qa": [0] * AGENT_ROLE_COUNT,
        "cost_quality_mean": cost * 0.8,
        "cost_human_mean": cost * 0.2,
        "cost_agent_mean": 0.0,          # no agents, so no token bill at all
        "actions_mean": [0.0] * AGENT_ROLE_COUNT,
        "impl_agents_mean": 0.0, "impl_human_mean": 10.0, "fixes_agent_mean": 0.0,
    })
    return obj


@pytest.fixture
def campaign_with_baseline(campaign) -> Path:
    d = campaign / _BASELINE_SID / "human_only"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({
        "scenario_id": _BASELINE_SID, "mode": "human_only",
        "horizon": {"path": "config/horizon_default.json", "T": 104,
                    "K_search": 100, "K_final": 1000, "master_seed": 42},
        "n_evaluated": 99, "n_simulated": 99, "best_admissible": True,
        "wall_clock_s": 20.0,
    }))
    (d / "search_summary.json").write_text(json.dumps({
        "best": "N2_R4_00003",
        "indistinguishable": ["N2_R4_00003"],
        "throughput_floor": 11.76,
        "objectives": {"N2_R4_00003": _baseline_objective()},
        "final_objectives": {"N2_R4_00003": _baseline_objective()},
    }))
    # An enumeration runs no generations, but it still writes the record.
    (d / "convergence.json").write_text(json.dumps({
        "scenario_id": _BASELINE_SID, "mode": "human_only",
        "n_elite": 10, "n_generations": 3, "n_generations_run": 0,
        "K_search": 100, "K_final": 1000,
        "elite_turnover": [], "incumbent_changed": [], "incumbent_turnover": None,
        "best_cost": [],
        "i_star": {"n_prelim": 1, "parsimony_removed_prelim": 0, "n_resimulated": 1,
                   "n_discarded_throughput": 0, "n_after_resim": 1, "n_final": 1,
                   "parsimony_removed": 0, "best_changed_by_resim": False,
                   "rho_mean": 0.02, "rho_best": 0.02},
        "throughput_floor": 11.76, "best_admissible": True, "value_ratio": 1.0,
        "wall_clock_s": 20.0, "resimulation_s": 4.0,
    }))
    return campaign


class TestScenarioIdSentinel:
    def test_round_trips_the_unused_marker(self):
        tiers = tiers_from_scenario_id(_BASELINE_SID)
        assert tiers["debt"] == "unused" and tiers["agent"] == "unused"
        assert tiers["human"] == "medium" and tiers["project"] == "medium"

    def test_is_baseline_scenario(self):
        assert is_baseline_scenario(_BASELINE_SID)
        assert not is_baseline_scenario(_SID)

    def test_params_resolve_to_the_unused_placeholders(self):
        # `<group>_unused.json` keeps the file-name convention, so no special case is
        # needed anywhere: the params load exactly like any other scenario's.
        params = scenario_params_for_id(_BASELINE_SID)
        assert params.r_arrival > 0 and params.r_bugs > 0

    def test_expected_modes_differ_by_scenario_kind(self):
        """A baseline is enumerated once; every other scenario gets one search."""
        assert expected_modes(_BASELINE_SID) == (HUMAN_ONLY_MODE,)
        assert expected_modes(_SID) == (SEARCH_MODE,)


class TestBaselineInTheTables:
    def test_search_dirs_picks_up_the_human_only_directory(self, campaign_with_baseline):
        found = {(sid, mode) for sid, mode, _ in search_dirs(campaign_with_baseline)}
        assert (_BASELINE_SID, "human_only") in found
        assert (_SID, "search") in found

    def test_optima_carries_the_baseline_row(self, campaign_with_baseline):
        optima, *_ = build_tables(campaign_with_baseline)
        row = optima[optima["mode"] == "human_only"].iloc[0]
        assert row["scenario_id"] == _BASELINE_SID
        assert row["cost_agent_mean"] == 0.0
        assert row["cost_agent_share"] == 0.0
        assert all(row[f"a_dev_{k + 1}"] == 0 for k in range(AGENT_ROLE_COUNT))

    def test_team_config_round_trips_an_all_zero_row(self, campaign_with_baseline):
        optima, *_ = build_tables(campaign_with_baseline)
        row = optima[optima["mode"] == "human_only"].iloc[0]
        cfg = team_config(row)
        assert cfg.A_dev == (0,) * AGENT_ROLE_COUNT
        assert cfg.A_qa == (0,) * AGENT_ROLE_COUNT


class TestBaselineUplift:
    def test_joins_on_the_three_varying_groups(self, campaign_with_baseline):
        optima, *_ = build_tables(campaign_with_baseline)
        uplift = baseline_uplift(optima)
        assert len(uplift) == 1
        row = uplift.iloc[0]
        assert row["scenario_id"] == _SID
        assert row["mode"] == "search"
        assert row["baseline_scenario_id"] == _BASELINE_SID
        # baseline 5000 against the search optimum's 1000
        assert row["cost_ratio"] == pytest.approx(5.0)

    def test_the_baseline_matches_every_debt_and_agent_combination(self):
        # One baseline is the reference for every (debt, agent) pair at its other tiers,
        # since neither group reaches a zero-agent team.
        import pandas as pd

        base = {"mode": "human_only", "scenario_id": _BASELINE_SID,
                "tier_human": "medium", "tier_project": "medium",
                "tier_bugimpact": "medium", "tier_debt": "unused",
                "tier_agent": "unused", "cost_mean": 800.0, "N": 2.0, "R": 4,
                "F_freeze": 1, "value_mean": 12.0}
        rows = [base]
        for debt in ("low", "medium", "high"):
            for agent in ("low", "medium", "high", "pessimistic"):
                rows.append({**base, "mode": "search",
                             "scenario_id": f"h-m_p-m_d-{debt[0]}_a-{agent[0]}_b-m",
                             "tier_debt": debt, "tier_agent": agent,
                             "cost_mean": 200.0})
        uplift = baseline_uplift(pd.DataFrame(rows))
        assert len(uplift) == 12
        assert uplift["cost_ratio"].tolist() == pytest.approx([4.0] * 12)

    def test_no_baseline_yields_an_empty_table(self, campaign):
        optima, *_ = build_tables(campaign)
        assert baseline_uplift(optima).empty
