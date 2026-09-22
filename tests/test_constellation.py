"""Tests for src/constellation.py — what the optimal teams are made of.

The campaign tree these build is the same shape `tests/test_campaign.py` uses; what
differs is which block of ``search_summary.json`` is read, so the fixtures deliberately
make ``indistinguishable`` and ``final_objectives`` disagree.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.campaign import scenario_params_for_id, search_rows
from src.config import AGENT_ROLE_COUNT, HUMAN_ONLY_MODE, supervised_agent_count
from src.constellation import (
    AGENT_COST_PARAMETERS,
    BILL_PARAMETERS,
    C_STAR,
    CELL_GROUPS,
    CONSTELLATION_PARAMETERS,
    DEV_PARAMETERS,
    I_STAR,
    QA_PARAMETERS,
    build_constellation,
    build_indistinguishable,
    cell_statistics,
    constellation_summary,
    derive_totals,
    indistinguishable_rows,
    role_bill,
    scenario_means,
)

_SID = "h-m_p-m_d-m_a-m_b-m"
_OTHER = "h-l_p-m_d-h_a-l_b-m"       # same project/bugimpact cell, different free tiers
_BASELINE_SID = "h-m_p-m_d-X_a-X_b-m"


def _objective(A_dev, A_qa=(0,) * AGENT_ROLE_COUNT, N=2.0, R=4, F_freeze=1,
               cost=1000.0) -> dict:
    return {
        "N": N, "R": R, "F_freeze": F_freeze, "f_fix": 0.5, "f_qa": 0.0,
        "A_dev": list(A_dev), "A_qa": list(A_qa),
        "cost_mean": cost, "cost_ci": [cost * 0.9, cost * 1.1],
        "value_mean": 12.0, "value_ci": [11.9, 12.1],
        "feasible_throughput": True,
        "cost_quality_mean": cost * 0.5,
        "cost_human_mean": cost * 0.3,
        "cost_agent_mean": cost * 0.2,
        "actions_mean": [10.0] * AGENT_ROLE_COUNT,
        "impl_human_mean": 4.0, "impl_agents_mean": 6.0,
        "fixes_human_mean": 1.0, "fixes_agent_mean": 2.0,
    }


def _summary(best: str, keep: list[str], final: dict, objectives: dict | None = None):
    return {
        "best": best,
        "indistinguishable": keep,
        "throughput_floor": 11.76,
        "K_search": 100, "K_final": 1000,
        "objectives": objectives if objectives is not None else final,
        "final_objectives": final,
    }


#: Three configurations re-simulated, but only two of them survive to the final `I*`.
#: `k3` is the member the stage-2 filter dropped and that must never reach the tables.
_FINAL = {
    "k1": _objective([1, 0, 1, 0, 4, 2, 1, 0], N=1.0),
    "k2": _objective([3, 0, 1, 0, 8, 2, 1, 0], N=3.0),
    "k3": _objective([9, 9, 9, 9, 9, 9, 9, 9], N=9.0),
}


def _write(root: Path, scenario_id: str, mode: str, summary: dict) -> Path:
    d = root / scenario_id / mode
    d.mkdir(parents=True)
    (d / "search_summary.json").write_text(json.dumps(summary))
    (d / "manifest.json").write_text(json.dumps({
        "scenario_id": scenario_id, "mode": mode,
        "horizon": {"T": 104, "K_search": 100, "K_final": 1000, "master_seed": 42},
        "n_evaluated": 100, "n_simulated": 85, "best_admissible": True,
    }))
    return d


@pytest.fixture
def campaign(tmp_path) -> Path:
    _write(tmp_path, _SID, "search", _summary("k1", ["k1", "k2"], _FINAL))
    return tmp_path


# ---------------------------------------------------------------------------
# Extracting I*
# ---------------------------------------------------------------------------

class TestIndistinguishableRows:
    def test_the_key_list_drives_the_extraction_not_final_objectives(self):
        """`final_objectives` is the *preliminary* set; `indistinguishable` is `I*`."""
        summary = _summary("k1", ["k1", "k2"], _FINAL)
        rows = indistinguishable_rows(_SID, "search", summary)
        assert len(rows) == 2
        assert {r["config_key"] for r in rows} == {"k1", "k2"}
        assert len(summary["final_objectives"]) == 3       # the set it was reduced from

    def test_the_count_is_what_optima_csv_reports(self, campaign):
        from src.campaign import optimum_row
        d = campaign / _SID / "search"
        summary = json.loads((d / "search_summary.json").read_text())
        row = optimum_row(_SID, "search", summary)
        assert len(indistinguishable_rows(_SID, "search", summary)) == \
            row["n_indistinguishable"]

    def test_c_star_is_a_member_and_is_flagged(self):
        rows = indistinguishable_rows(_SID, "search",
                                      _summary("k2", ["k1", "k2"], _FINAL))
        assert [r["config_key"] for r in rows if r["is_best"]] == ["k2"]

    def test_an_inadmissible_search_contributes_nothing(self):
        assert indistinguishable_rows(_SID, "search",
                                      _summary("k1", [], _FINAL)) == []

    def test_the_tiers_come_from_the_scenario_id(self):
        row = indistinguishable_rows(_SID, "search",
                                     _summary("k1", ["k1"], _FINAL))[0]
        assert row["tier_project"] == "medium"
        assert row["tier_human"] == "medium"

    def test_the_composition_columns_are_one_per_role_and_1_based(self):
        row = indistinguishable_rows(_SID, "search",
                                     _summary("k1", ["k1"], _FINAL))[0]
        assert [row[c] for c in DEV_PARAMETERS] == [1, 0, 1, 0, 4, 2, 1, 0]
        assert [row[c] for c in QA_PARAMETERS] == [0] * AGENT_ROLE_COUNT
        assert row["n_sup_dev"] == supervised_agent_count((1, 0, 1, 0, 4, 2, 1, 0))

    def test_a_key_with_no_objective_is_an_error_not_a_silent_drop(self):
        with pytest.raises(KeyError, match="ghost"):
            indistinguishable_rows(_SID, "search",
                                   _summary("k1", ["k1", "ghost"], _FINAL))


class TestBuildIndistinguishable:
    def test_reads_the_tree(self, campaign):
        df = build_indistinguishable(campaign)
        assert len(df) == 2
        assert set(df["mode"]) == {"search"}

    def test_skips_a_directory_without_a_summary(self, campaign):
        (campaign / _SID / HUMAN_ONLY_MODE).mkdir()
        assert len(build_indistinguishable(campaign)) == 2

    def test_a_missing_tree_is_an_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_indistinguishable(tmp_path / "nope")

    def test_an_empty_tree_is_an_error(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError):
            build_indistinguishable(tmp_path / "empty")


# ---------------------------------------------------------------------------
# Derived columns and cell statistics
# ---------------------------------------------------------------------------

def _member_frame(records: list[dict]) -> pd.DataFrame:
    """A minimal `I*` member table: one dict per member, defaults filled in."""
    rows = []
    for i, record in enumerate(records):
        row = {"scenario_id": record.get("scenario_id", _SID),
               "mode": "search", "config_key": f"k{i}", "is_best": i == 0,
               "N": 1.0, "R": 4, "F_freeze": 1,
               "tier_human": "medium", "tier_project": "medium", "tier_debt": "medium",
               "tier_agent": "medium", "tier_bugimpact": "medium"}
        for k in range(AGENT_ROLE_COUNT):
            row[f"a_dev_{k + 1}"] = 0
            row[f"a_qa_{k + 1}"] = 0
        row.update(record)
        rows.append(row)
    return pd.DataFrame(rows)


class TestDeriveTotals:
    def test_team_size_is_the_row_sum(self):
        df = derive_totals(_member_frame([{"a_dev_1": 2, "a_dev_5": 7, "a_qa_3": 4}]))
        assert df["n_agents_dev"].iloc[0] == 9
        assert df["n_agents_qa"].iloc[0] == 4

    def test_supervised_counts_match_the_model(self):
        A = (2, 1, 3, 40, 50, 60, 1, 1)
        record = {f"a_dev_{k + 1}": A[k] for k in range(AGENT_ROLE_COUNT)}
        df = derive_totals(_member_frame([record]))
        assert df["n_sup_dev"].iloc[0] == supervised_agent_count(A)

class TestScenarioMeans:
    def test_one_row_per_scenario(self):
        frame = _member_frame([{"a_dev_1": 1}, {"a_dev_1": 3},
                               {"scenario_id": _OTHER, "a_dev_1": 8}])
        means = scenario_means(derive_totals(frame))
        assert len(means) == 2
        assert set(means["scenario_id"]) == {_SID, _OTHER}
        assert means.set_index("scenario_id").loc[_SID, "a_dev_1"] == 2.0

    def test_the_member_count_is_kept(self):
        frame = _member_frame([{"a_dev_1": 1}, {"a_dev_1": 3}])
        assert scenario_means(derive_totals(frame))["n_members"].iloc[0] == 2

    def test_the_tiers_survive(self):
        means = scenario_means(derive_totals(_member_frame([{"a_dev_1": 1}])))
        assert means["tier_project"].iloc[0] == "medium"


class TestCellStatistics:
    def _cells(self, frame, population=C_STAR):
        return cell_statistics(derive_totals(frame), population)

    def test_mean_and_sd_over_the_scenarios_in_the_cell(self):
        cells = self._cells(_member_frame([
            {"a_dev_5": 2}, {"scenario_id": _OTHER, "a_dev_5": 4}]))
        row = cells[cells["parameter"] == "a_dev_5"].iloc[0]
        assert row["mean"] == 3.0
        assert row["sd"] == pytest.approx(pd.Series([2.0, 4.0]).std())  # ddof = 1
        assert row["n_scenarios"] == 2

    def test_a_single_scenario_has_no_spread_to_report(self):
        cells = self._cells(_member_frame([{"a_dev_5": 2}]))
        assert pd.isna(cells[cells["parameter"] == "a_dev_5"]["sd"].iloc[0])

    def test_every_parameter_appears_once_per_cell(self):
        cells = self._cells(_member_frame([{"a_dev_5": 2}]))
        assert list(cells["parameter"]) == list(CONSTELLATION_PARAMETERS)

    def test_the_grid_is_project_by_bugimpact(self):
        """One fixed cut, not an argument.

        These two are expected to shape the optimal team most strongly; the other three
        stay free within a cell and become its spread.
        """
        assert CELL_GROUPS == ("project", "bugimpact")
        cells = self._cells(_member_frame([{"a_dev_5": 2}]))
        assert "tier_project" in cells.columns and "tier_bugimpact" in cells.columns
        assert "tier_human" not in cells.columns

    def test_a_missing_cell_column_is_rejected(self):
        frame = derive_totals(_member_frame([{"a_dev_5": 2}])).drop(
            columns=["tier_bugimpact"])
        with pytest.raises(KeyError, match="tier_bugimpact"):
            cell_statistics(frame, C_STAR)

    def test_an_empty_frame_yields_the_schema(self):
        cells = cell_statistics(pd.DataFrame(), C_STAR)
        assert list(cells.columns) == ["population", "tier_project", "tier_bugimpact",
                                       "parameter", "mean", "sd", "n_scenarios",
                                       "n_members"]


class TestScenarioBalance:
    def test_a_wide_tie_does_not_outweigh_a_narrow_one(self):
        """Two scenarios in one cell weigh the same however large their `I*` is."""
        members = ([{"a_dev_5": 10}] * 2
                   + [{"scenario_id": _OTHER, "a_dev_5": 20}] * 200)
        cells = cell_statistics(
            scenario_means(derive_totals(_member_frame(members))), I_STAR)
        row = cells[cells["parameter"] == "a_dev_5"].iloc[0]
        assert row["mean"] == 15.0            # not the pooled 19.9
        assert row["n_scenarios"] == 2
        assert row["n_members"] == 202


# ---------------------------------------------------------------------------
# The whole table
# ---------------------------------------------------------------------------

def _optima_frame(modes=("search",)) -> pd.DataFrame:
    rows = []
    for mode in modes:
        record = {"mode": mode, "a_dev_5": 5}
        rows.append(_member_frame([record]).assign(mode=mode).iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


class TestBuildConstellation:
    def test_both_populations_appear(self, campaign):
        istar = build_indistinguishable(campaign)
        cells = build_constellation(_optima_frame(), istar)
        assert set(cells["population"]) == {C_STAR, I_STAR}

    def test_the_human_only_baseline_is_excluded(self):
        frame = _optima_frame(("search", HUMAN_ONLY_MODE))
        assert len(search_rows(frame)) == 1
        cells = build_constellation(frame, pd.DataFrame())
        assert set(cells["population"]) == {C_STAR}
        assert cells["n_scenarios"].max() == 1

    def test_nothing_to_describe_yields_the_schema(self):
        cells = build_constellation(pd.DataFrame(), pd.DataFrame())
        assert cells.empty
        assert "population" in cells.columns


class TestConstellationSummary:
    def test_it_names_both_populations_and_the_grid(self, campaign):
        istar = build_indistinguishable(campaign)
        text = constellation_summary(build_constellation(_optima_frame(), istar))
        assert f"{CELL_GROUPS[0]} × {CELL_GROUPS[1]}" in text
        assert C_STAR in text and I_STAR in text

    def test_it_warns_about_the_structural_a_qa_zeros(self, campaign):
        istar = build_indistinguishable(campaign)
        text = constellation_summary(build_constellation(_optima_frame(), istar))
        assert "F_freeze = 0" in text

    def test_the_columns_line_up(self, campaign):
        istar = build_indistinguishable(campaign)
        text = constellation_summary(build_constellation(_optima_frame(), istar))
        # No two cells may run together: every printed value keeps a leading space.
        assert "  " in text
        assert not any(line.rstrip().endswith("±") for line in text.split("\n"))

    def test_an_empty_table_says_so(self):
        assert "no optimum" in constellation_summary(pd.DataFrame())


# ---------------------------------------------------------------------------
# What each role costs
# ---------------------------------------------------------------------------

class TestRoleBill:
    """`actions_k * c_token_k` is the exact per-role split of the agent bill."""

    def _frame(self, scenario_id=_SID, actions=None) -> pd.DataFrame:
        actions = actions if actions is not None else [10.0] * AGENT_ROLE_COUNT
        row = {"scenario_id": scenario_id, "mode": "search",
               "cost_agent_mean": 20.0, "cost_human_mean": 80.0}
        row.update({f"actions_{k + 1}": actions[k] for k in range(AGENT_ROLE_COUNT)})
        return pd.DataFrame([row])

    def test_one_bill_column_per_role(self):
        billed = role_bill(self._frame())
        assert all(c in billed.columns for c in BILL_PARAMETERS)

    def test_the_bill_is_the_work_times_the_price(self):
        prices = scenario_params_for_id(_SID).c_token
        billed = role_bill(self._frame())
        for k, column in enumerate(BILL_PARAMETERS):
            assert billed[column].iloc[0] == pytest.approx(10.0 * prices[k])

    def test_the_price_follows_the_scenario_s_agent_tier(self):
        """The `pessimistic` tier is the low tier's capability at 10x the high price."""
        cheap = role_bill(self._frame("h-m_p-m_d-m_a-h_b-m"))["bill_1"].iloc[0]
        dear = role_bill(self._frame("h-m_p-m_d-m_a-p_b-m"))["bill_1"].iloc[0]
        assert dear == pytest.approx(10.0 * cheap)

    def test_a_role_that_did_no_work_bills_nothing(self):
        """A demand-driven agent no work reaches costs nothing."""
        actions = [0.0] * AGENT_ROLE_COUNT
        billed = role_bill(self._frame(actions=actions))
        assert (billed[list(BILL_PARAMETERS)].iloc[0] == 0.0).all()

    def test_the_input_frame_is_not_mutated(self):
        frame = self._frame()
        role_bill(frame)
        assert not any(c in frame.columns for c in BILL_PARAMETERS)

    def test_the_per_feature_split_adds_to_the_reported_agent_cost(self):
        """`AGENT_COST_PARAMETERS` is an apportionment, so it closes exactly."""
        billed = role_bill(self._frame())
        assert billed[list(AGENT_COST_PARAMETERS)].sum(axis=1).iloc[0] == pytest.approx(
            billed["cost_agent_mean"].iloc[0])

    def test_a_team_that_bills_nothing_has_no_split_to_apportion(self):
        billed = role_bill(self._frame(actions=[0.0] * AGENT_ROLE_COUNT))
        assert (billed[list(AGENT_COST_PARAMETERS)].iloc[0] == 0.0).all()

    def test_the_team_cost_is_the_two_drivers_the_composition_controls(self):
        """Not the third: escaped-bug cost is a consequence, not a line item."""
        billed = role_bill(self._frame())
        assert billed["cost_team_mean"].iloc[0] == pytest.approx(100.0)

    def test_it_reproduces_the_reported_agent_cost(self):
        """Against the real campaign: the split must add back up to what optima.csv reports.

        A cross-check against consolidated results rather than a unit test — the formula
        itself is pinned above on synthetic data — so it is skipped where no campaign has
        been consolidated.  The horizon comes from `scenarios.csv` rather than a literal,
        since it is a property of the run and not of the code.
        """
        path = Path("results/analysis/optima.csv")
        horizon_path = Path("results/analysis/scenarios.csv")
        if not (path.is_file() and horizon_path.is_file()):
            pytest.skip("no consolidated campaign in results/analysis")
        horizons = pd.read_csv(horizon_path)["T"].unique()
        assert len(horizons) == 1, f"the campaign mixes horizons: {horizons}"
        optima = pd.read_csv(path)
        optima = optima[optima["mode"] == "search"]
        billed = role_bill(optima)
        features = billed["value_mean"] * horizons[0]
        recovered = billed[list(BILL_PARAMETERS)].sum(axis=1) / features
        # Not exact: `cost_agent_mean` is a mean of per-replication ratios, this is a
        # ratio of means.  Within a fraction of a per cent, and a *share* — what the
        # figure draws — has no such gap at all, being formed inside one row.
        error = ((recovered - billed["cost_agent_mean"]) / billed["cost_agent_mean"]).abs()
        assert error.max() < 0.01


# ---------------------------------------------------------------------------
# Role usage, the release process, and the process spread within I*
# ---------------------------------------------------------------------------

def _usage_optima(n: int = 12) -> pd.DataFrame:
    """A searched `optima.csv` with a known head count and a known bill per role.

    Role 5 (Code Review) is stacked and cheap, role 6 (Security) is scarce and dear, so
    the head-count and bill shares must disagree in a direction the test can pin.
    """
    rows = []
    for i in range(n):
        row = {
            "scenario_id": f"{_SID}_{i}", "mode": "search", "config_key": f"c{i}",
            "n_indistinguishable": 3, "feasible_throughput": True,
            "N": 1.0, "R": 1 if i % 2 == 0 else 13, "F_freeze": 0 if i % 3 else 1,
            "cost_agent_mean": 100.0, "cost_human_mean": 50.0,
            "tier_agent": "medium",
        }
        for k in range(1, AGENT_ROLE_COUNT + 1):
            row[f"a_dev_{k}"] = 0
            row[f"bill_{k}"] = 0.0
        row["a_dev_1"], row["bill_1"] = 1, 10.0
        row["a_dev_5"], row["bill_5"] = 90, 10.0     # 90 % of the team, 10 % of the bill
        row["a_dev_6"], row["bill_6"] = 9, 80.0      # 9 % of the team, 80 % of the bill
        rows.append(row)
    return pd.DataFrame(rows)


class TestRoleUsage:
    def test_head_count_and_bill_shares_disagree_as_constructed(self):
        from src.constellation import role_usage
        usage = role_usage(_usage_optima()).set_index("role")
        assert usage.loc[5, "head_count_share"] == pytest.approx(0.90)
        assert usage.loc[5, "bill_share"] == pytest.approx(0.10)
        assert usage.loc[6, "head_count_share"] == pytest.approx(0.09)
        assert usage.loc[6, "bill_share"] == pytest.approx(0.80)

    def test_each_set_of_shares_sums_to_one(self):
        from src.constellation import role_usage
        usage = role_usage(_usage_optima())
        assert usage["head_count_share"].sum() == pytest.approx(1.0)
        assert usage["bill_share"].sum() == pytest.approx(1.0)

    def test_staffed_share_counts_the_optima_that_use_the_role(self):
        from src.constellation import role_usage
        optima = _usage_optima(n=4)
        optima.loc[:1, "a_dev_2"] = 3          # staffed in half the optima
        usage = role_usage(optima).set_index("role")
        assert usage.loc[2, "staffed_share"] == pytest.approx(0.5)
        assert usage.loc[1, "staffed_share"] == pytest.approx(1.0)
        assert usage.loc[3, "staffed_share"] == pytest.approx(0.0)

    def test_shares_are_formed_within_a_scenario_not_over_the_pool(self):
        """A scenario 1000x larger than the rest must not dominate the mean share."""
        from src.constellation import role_usage
        optima = _usage_optima(n=4)
        optima.loc[0, ["a_dev_5", "a_dev_6", "a_dev_1"]] = [90_000, 9_000, 1_000]
        optima.loc[0, ["bill_5", "bill_6", "bill_1"]] = [10_000.0, 80_000.0, 10_000.0]
        usage = role_usage(optima).set_index("role")
        # The within-scenario shares are identical in every row, so the mean is unmoved.
        assert usage.loc[5, "head_count_share"] == pytest.approx(0.90)
        assert usage.loc[6, "bill_share"] == pytest.approx(0.80)

    def test_splitting_by_a_group_keeps_every_tier_present(self):
        from src.constellation import role_usage
        optima = _usage_optima(n=4)
        optima.loc[:1, "tier_agent"] = "low"
        usage = role_usage(optima, by="agent")
        assert set(usage["tier"]) == {"low", "medium"}
        assert set(usage["group"]) == {"agent"}
        assert usage.groupby("tier")["n"].first().tolist() == [2, 2]

    def test_an_empty_frame_yields_nothing(self):
        from src.constellation import role_usage
        assert role_usage(pd.DataFrame()).empty


class TestProcessUsage:
    def test_the_release_shares_sum_to_one_over_the_cadences_used(self):
        from src.constellation import process_usage
        process = process_usage(_usage_optima())
        cadences = process[process["quantity"] == "R"]
        assert cadences["share"].sum() == pytest.approx(1.0)
        assert set(cadences["value"]) == {1, 13}

    def test_the_freeze_share_counts_the_optima_without_one(self):
        from src.constellation import process_usage
        optima = _usage_optima(n=12)
        process = process_usage(optima)
        freeze = process[process["quantity"] == "F_freeze"].iloc[0]
        assert freeze["share"] == pytest.approx((optima["F_freeze"] == 0).mean())


class TestIstarProcessDiversity:
    def test_counts_the_distinct_triples_of_each_set(self):
        from src.constellation import istar_process_diversity
        istar = pd.DataFrame([
            {"scenario_id": "s1", "mode": "search", "N": 1.0, "R": 1, "F_freeze": 0},
            {"scenario_id": "s1", "mode": "search", "N": 1.0, "R": 1, "F_freeze": 0},
            {"scenario_id": "s1", "mode": "search", "N": 2.0, "R": 1, "F_freeze": 0},
            {"scenario_id": "s2", "mode": "search", "N": 1.0, "R": 4, "F_freeze": 0},
        ])
        diversity = istar_process_diversity(istar).set_index("scenario_id")
        assert diversity.loc["s1", "n_triples"] == 2
        assert diversity.loc["s1", "R_unique"]        # R agrees across the tie
        assert not diversity.loc["s1", "N_unique"]    # N does not
        assert diversity.loc["s2", "n_triples"] == 1

    def test_the_human_only_baseline_is_excluded(self):
        from src.constellation import istar_process_diversity
        istar = pd.DataFrame([
            {"scenario_id": "s1", "mode": HUMAN_ONLY_MODE, "N": 1.0, "R": 1,
             "F_freeze": 0},
        ])
        assert istar_process_diversity(istar).empty


class TestBuildProcessSummary:
    def test_it_carries_all_three_sections(self):
        from src.constellation import build_process_summary
        istar = pd.DataFrame([
            {"scenario_id": f"{_SID}_0", "mode": "search", "N": 1.0, "R": 1,
             "F_freeze": 0},
        ])
        table = build_process_summary(_usage_optima(), istar)
        assert set(table["section"]) == {"role", "process", "istar"}

    def test_the_baseline_and_the_inadmissible_are_both_dropped(self):
        from src.constellation import build_process_summary
        optima = _usage_optima(n=6)
        optima.loc[0, "mode"] = HUMAN_ONLY_MODE
        optima.loc[1, "n_indistinguishable"] = 0
        table = build_process_summary(optima, pd.DataFrame())
        roles = table[(table["section"] == "role") & (table["tier"] == "all")]
        assert roles["n"].iloc[0] == 4

    def test_the_summary_names_the_roles_and_the_process(self):
        from src.constellation import build_process_summary, process_summary
        table = build_process_summary(_usage_optima(), pd.DataFrame())
        text = process_summary(table)
        assert "Review" in text and "Security" in text
        assert "R = 1" in text and "no feature freeze" in text

    def test_an_empty_frame_yields_nothing(self):
        from src.constellation import build_process_summary, process_summary
        table = build_process_summary(pd.DataFrame(), pd.DataFrame())
        assert table.empty
        assert "no searched optimum" in process_summary(table)
