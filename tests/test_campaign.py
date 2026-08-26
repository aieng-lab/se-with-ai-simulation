"""Tests for src/campaign.py — consolidating a campaign into flat tables."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.campaign import (
    baseline_uplift,
    build_tables,
    convergence_summary,
    optimum_row,
    scenario_params_for_id,
    search_dirs,
    team_config,
    tiers_from_scenario_id,
)
from src.config import (
    AGENT_ROLE_COUNT,
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


def _convergence_row(mode: str, *, n_prelim: int, n_final: int,
                     admissible: bool = True) -> dict:
    """One `convergence.csv` row, with the columns the summary reads."""
    return {
        "scenario_id": _SID, "mode": mode, "n_generations_run": 90 if mode == "search" else 0,
        "n_elite": 100, "tau_first": 0.4, "tau_last": 0.03, "tau_mean_last_10": 0.03,
        "n_prelim": n_prelim, "parsimony_removed_prelim": n_prelim // 2,
        "n_resimulated": n_prelim // 2, "n_discarded_throughput": 0,
        "n_after_resim": n_final * 2, "n_final": n_final, "parsimony_removed": n_final,
        "best_changed_by_resim": True, "rho_mean": 0.01, "rho_best": 0.01,
        "best_admissible": admissible, "value_ratio": 1.0 if admissible else 0.5,
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
        summary = json.loads((d / "search_summary.json").read_text())
        return optimum_row(_SID, "search", summary)

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

    def test_no_optimum_yields_no_row(self, campaign):
        d = campaign / _SID / "search"
        summary = json.loads((d / "search_summary.json").read_text())
        summary["best"] = None
        assert optimum_row(_SID, "search", summary) is None


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
        assert "Inadmissible runs: 0/1" in text

    def test_convergence_summary_reports_the_two_modes_separately(self):
        """Section 5.1 gives the searches and the baseline enumerations two columns.

        A search and an exhaustive enumeration differ both in how large the preliminary
        set is and in how far the selection narrows it, so a pooled median describes
        neither.  Pinned on frames whose two modes cannot be confused: the search sets
        are an order of magnitude larger.
        """
        rows = pd.DataFrame(
            [_convergence_row("search", n_prelim=1000, n_final=40)] * 3
            + [_convergence_row("human_only", n_prelim=17, n_final=7)] * 2
        )
        text = convergence_summary(rows)
        search_block = text.split("human_only")[0]
        baseline_block = text.split("human_only")[1]
        assert "search (3 run(s), 3 admissible)" in search_block
        assert "1e+03" in search_block and "40" in search_block
        assert "(2 run(s), 2 admissible)" in baseline_block
        assert "17" in baseline_block and "7" in baseline_block
        # The pooled median of the five would be 1000, i.e. the searches' alone.
        assert "1e+03" not in baseline_block

    def test_the_i_star_stage_sizes_exclude_the_inadmissible_runs(self):
        """An inadmissible run has an empty I* (Section 3.10).

        Section 5.1's table is over the admissible runs for exactly that reason: a zero
        at every stage is not a small set, it is the absence of one.
        """
        rows = pd.DataFrame([
            _convergence_row("search", n_prelim=100, n_final=10),
            _convergence_row("search", n_prelim=200, n_final=20),
            _convergence_row("search", n_prelim=0, n_final=0, admissible=False),
        ])
        text = convergence_summary(rows)
        assert "search (3 run(s), 2 admissible)" in text
        # The median of {100, 200} is 150; pooling the zero in would give 100.
        assert "150" in text

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

    def test_params_resolve_to_the_unused_placeholders(self):
        # `<group>_unused.json` keeps the file-name convention, so no special case is
        # needed anywhere: the params load exactly like any other scenario's.
        params = scenario_params_for_id(_BASELINE_SID)
        assert params.r_arrival > 0 and params.r_bugs > 0


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
                "F_freeze": 1, "value_mean": 12.0, "n_indistinguishable": 5}
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

    def test_it_carries_the_two_tiers_the_join_cannot_supply(self):
        """Section 5.2 reads the comparison by agent and by debt tier.

        Those are exactly the two groups the baseline holds at `UNUSED_TIER`, so they
        cannot come from the join keys and have to be carried from the search row.
        """
        import pandas as pd

        base = {"mode": "human_only", "scenario_id": _BASELINE_SID,
                "tier_human": "medium", "tier_project": "medium",
                "tier_bugimpact": "medium", "tier_debt": "unused",
                "tier_agent": "unused", "cost_mean": 800.0, "N": 2.0, "R": 4,
                "F_freeze": 1, "value_mean": 12.0, "n_indistinguishable": 5}
        search = {**base, "mode": "search", "scenario_id": "h-m_p-m_d-h_a-p_b-m",
                  "tier_debt": "high", "tier_agent": "pessimistic",
                  "cost_mean": 200.0, "n_indistinguishable": 3}
        uplift = baseline_uplift(pd.DataFrame([base, search]))
        row = uplift.iloc[0]
        assert row["tier_agent"] == "pessimistic" and row["tier_debt"] == "high"
        assert bool(row["admissible"]) and bool(row["baseline_admissible"])

    def test_an_inadmissible_side_is_flagged_but_kept(self):
        """An empty I* marks the run inadmissible (Section 3.10); the row still lands.

        Section 5.2 restricts the comparison to pairs where both sides are admissible,
        which is a filter on these two columns rather than a missing row.
        """
        import pandas as pd

        base = {"mode": "human_only", "scenario_id": _BASELINE_SID,
                "tier_human": "medium", "tier_project": "medium",
                "tier_bugimpact": "medium", "tier_debt": "unused",
                "tier_agent": "unused", "cost_mean": 800.0, "N": 2.0, "R": 4,
                "F_freeze": 1, "value_mean": 3.0, "n_indistinguishable": 0}
        search = {**base, "mode": "search", "scenario_id": "h-m_p-m_d-h_a-p_b-m",
                  "tier_debt": "high", "tier_agent": "pessimistic",
                  "cost_mean": 200.0, "n_indistinguishable": 3}
        uplift = baseline_uplift(pd.DataFrame([base, search]))
        assert len(uplift) == 1
        assert bool(uplift.iloc[0]["admissible"])
        assert not bool(uplift.iloc[0]["baseline_admissible"])

    def test_no_baseline_yields_an_empty_table(self, campaign):
        optima, *_ = build_tables(campaign)
        assert baseline_uplift(optima).empty


# ---------------------------------------------------------------------------
# The multiplicative cost model
# ---------------------------------------------------------------------------

def _multiplier_optima(effects: "dict[str, dict[str, float]] | None" = None,
                       noise: float = 0.0) -> pd.DataFrame:
    """A synthetic `optima.csv` whose cost is an exact product of per-tier factors.

    With `noise = 0` the additive log model is exact, so `cost_multipliers` must recover
    the factors it was built from and report `R^2 = 1`.
    """
    import numpy as np

    from src.config import scenario_id, scenario_tier_combinations

    effects = effects or {"bugimpact": {"low": 1.0, "medium": 10.0, "high": 100.0}}
    rng = np.random.default_rng(0)
    rows = []
    for combination in scenario_tier_combinations():
        tiers = dict(zip(SCENARIO_GROUPS, combination))
        cost = 100.0
        for group, factors in effects.items():
            cost *= factors.get(tiers[group], 1.0)
        cost *= 10.0 ** (noise * rng.standard_normal())
        rows.append({
            "scenario_id": scenario_id(combination), "mode": "search",
            **{f"tier_{g}": t for g, t in tiers.items()},
            "cost_mean": cost, "n_indistinguishable": 5, "feasible_throughput": True,
        })
    return pd.DataFrame(rows)


class TestCostMultipliers:
    def test_recovers_the_factors_a_noise_free_grid_was_built_from(self):
        from src.campaign import cost_multipliers
        table, stats = cost_multipliers(_multiplier_optima())
        assert stats["r_squared"] == pytest.approx(1.0)
        assert stats["residual_sd"] == pytest.approx(0.0, abs=1e-9)
        bug = table[table["group"] == "bugimpact"].set_index("tier")["multiplier"]
        assert bug["low"] == pytest.approx(1.0)
        assert bug["medium"] == pytest.approx(10.0)
        assert bug["high"] == pytest.approx(100.0)

    def test_the_reference_is_the_cheapest_tier_and_has_multiplier_one(self):
        from src.campaign import cost_multipliers
        table, _ = cost_multipliers(_multiplier_optima())
        for _, sub in table.groupby("group"):
            reference = sub[sub["is_reference"]]
            assert len(reference) == 1, "exactly one reference tier per group"
            # No tier is cheaper than the reference, so every multiplier is >= 1.  An
            # inert group ties all its tiers at 1.0, which reference it picks is moot.
            assert reference["multiplier"].iloc[0] == pytest.approx(1.0)
            assert sub["multiplier"].min() == pytest.approx(1.0)

    def test_an_inert_group_explains_no_variance(self):
        from src.campaign import cost_multipliers
        _, stats = cost_multipliers(_multiplier_optima())
        # Only `bugimpact` drives the synthetic cost, so it takes all of the variance.
        assert stats["eta_squared_bugimpact"] == pytest.approx(1.0)
        for group in ("human", "project", "debt", "agent"):
            assert stats[f"eta_squared_{group}"] == pytest.approx(0.0, abs=1e-9)

    def test_two_groups_combine_multiplicatively(self):
        from src.campaign import cost_multipliers
        table, stats = cost_multipliers(_multiplier_optima({
            "bugimpact": {"low": 1.0, "medium": 8.0, "high": 260.0},
            "debt": {"low": 1.0, "medium": 1.7, "high": 2.4},
        }))
        assert stats["r_squared"] == pytest.approx(1.0)
        debt = table[table["group"] == "debt"].set_index("tier")["multiplier"]
        assert debt["high"] == pytest.approx(2.4)
        assert stats["eta_squared_bugimpact"] > stats["eta_squared_debt"]

    def test_the_human_only_baseline_is_excluded(self):
        from src.campaign import admissible_optima, cost_multipliers
        optima = _multiplier_optima()
        baseline = optima.head(5).copy()
        baseline["mode"] = HUMAN_ONLY_MODE
        baseline["cost_mean"] = 1e9
        combined = pd.concat([optima, baseline], ignore_index=True)
        assert len(admissible_optima(combined)) == len(optima)
        table, stats = cost_multipliers(combined)
        assert stats["n"] == len(optima)
        assert stats["r_squared"] == pytest.approx(1.0)

    def test_an_inadmissible_optimum_is_excluded(self):
        from src.campaign import cost_multipliers
        optima = _multiplier_optima()
        optima.loc[0, "n_indistinguishable"] = 0
        _, stats = cost_multipliers(optima)
        assert stats["n"] == len(optima) - 1

    def test_an_empty_frame_yields_nothing(self):
        from src.campaign import cost_multipliers, multipliers_summary
        table, stats = cost_multipliers(pd.DataFrame())
        assert table.empty and stats == {}
        assert "no admissible" in multipliers_summary(table, stats)

    def test_the_summary_names_every_group_and_the_fit(self):
        from src.campaign import cost_multipliers, multipliers_summary
        table, stats = cost_multipliers(_multiplier_optima(noise=0.05))
        text = multipliers_summary(table, stats)
        for group in SCENARIO_GROUPS:
            assert group in text
        assert "R^2" in text and "residual sd" in text
