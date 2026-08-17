"""Tests for src/visualization.py — the campaign-level five-channel optima matrix.

The per-search figures are exercised only indirectly (they take the raw `objectives`
dicts and have no branching worth pinning); what needs guarding is the channel
assignment of `plot_optima_matrix`, since a duplicated or missing group silently
collapses two scenario dimensions into one.

Rendering the full 432-scenario grid costs a few seconds — one `ax.scatter` per
(size, shape, colour) subgroup per panel — so the tests that only inspect a figure's
structure share one module-scoped render, and the rest use a handful of scenarios.
"""

from __future__ import annotations

import inspect
import itertools

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.figure import Figure

from src.config import (
    SCENARIO_GROUPS,
    UNUSED_TIER,
    scenario_id,
    scenario_tier_combinations,
)
from src.visualization import (
    COST_SHARES,
    _BASELINE_LABEL,
    _baseline_cell,
    _log_floor,
    _tiers_present,
    plot_agent_human_ratio_matrix,
    plot_human_agent_matrix,
    plot_optima_matrix,
    plot_quality_development_matrix,
    save_figure,
    visualize_campaign,
)


def _optima(
    combinations: "list[tuple[str, ...]] | None" = None,
    strategies: tuple[str, ...] = ("evolutionary",),
) -> pd.DataFrame:
    """A synthetic `optima.csv`: one row per (scenario, strategy)."""
    combinations = (scenario_tier_combinations() if combinations is None
                    else combinations)
    rows = []
    for i, (tiers, strategy) in enumerate(itertools.product(combinations, strategies)):
        row = {"scenario_id": scenario_id(tiers), "strategy": strategy,
               "cost_mean": 100.0 * (i + 1)}
        row.update({f"tier_{g}": t for g, t in zip(SCENARIO_GROUPS, tiers)})
        # Shares that span three decades and reach zero, the way the real optima do.
        row["cost_agent_share"] = (i % 100) / 1000.0
        row["cost_human_share"] = 0.3
        row["cost_quality_share"] = 1.0 - row["cost_agent_share"] - 0.3
        row["cost_agent_mean"] = row["cost_mean"] * row["cost_agent_share"]
        row["cost_human_mean"] = row["cost_mean"] * 0.3
        row["cost_quality_mean"] = row["cost_mean"] * row["cost_quality_share"]
        rows.append(row)
    return pd.DataFrame(rows)


def _few(strategies: tuple[str, ...] = ("evolutionary",)) -> pd.DataFrame:
    """A handful of scenarios — enough for the tests that do not read tier structure."""
    return _optima(scenario_tier_combinations()[:6], strategies)


@pytest.fixture(scope="module")
def full_grid() -> pd.DataFrame:
    return _optima()


@pytest.fixture(scope="module")
def default_fig(full_grid) -> Figure:
    """The default channel assignment over the whole grid; read-only, shared."""
    return plot_optima_matrix(full_grid)


class TestTierHelpers:
    def test_tiers_come_back_in_mild_to_extreme_order(self, full_grid):
        assert _tiers_present(full_grid, "tier_human") == ["low", "medium", "high"]
        # `project` and `agent` are the two groups that reach a fourth tier.
        assert _tiers_present(full_grid, "tier_project") == \
            ["low", "medium", "high", "veryhigh"]
        assert _tiers_present(full_grid, "tier_agent") == \
            ["low", "medium", "high", "pessimistic"]

    def test_absent_tiers_are_skipped(self, full_grid):
        df = full_grid[full_grid["tier_debt"] != "medium"]
        assert _tiers_present(df, "tier_debt") == ["low", "high"]

    def test_log_floor_is_half_the_smallest_positive(self):
        assert _log_floor(pd.DataFrame({"x": [0.0, 0.004, 0.02]}), "x") \
            == pytest.approx(0.002)

    def test_no_floor_without_a_non_positive_value(self):
        assert _log_floor(pd.DataFrame({"x": [0.004, 0.02]}), "x") is None


class TestOptimaMatrix:
    def test_grid_is_the_two_facet_groups(self, default_fig):
        # agent (four tiers) x human (three).
        assert isinstance(default_fig, Figure)
        assert len(default_fig.axes) == 12

    def test_a_three_tier_group_on_the_x_facet_narrows_the_grid(self, full_grid):
        # Swapping the four-tier `agent` off the column axis for the three-tier `debt`
        # gives the 3x3 grid; the facet count follows the data, not a fixed layout.
        fig = plot_optima_matrix(full_grid, x_group="debt", color_group="agent")
        assert len(fig.axes) == 9

    def test_every_point_is_drawn_exactly_once(self, full_grid, default_fig):
        drawn = sum(len(coll.get_offsets()) for ax in default_fig.axes
                    for coll in ax.collections)
        assert drawn == len(full_grid)

    def test_three_proxy_legends_one_per_non_facet_channel(self, default_fig):
        titles = [lg.get_title().get_text() for lg in default_fig.legends]
        assert titles == ["debt (colour)", "project (shape)", "bugimpact (size)"]
        # The four-tier group gets a fourth entry; the three-tier ones do not.
        assert [len(lg.get_texts()) for lg in default_fig.legends] == [3, 4, 3]

    def test_the_pessimistic_agent_tier_draws_on_a_per_point_channel(self, full_grid):
        # `agent` is a facet by default, so its fourth tier never reaches the colour /
        # shape / size dicts there.  Moving it onto one exercises that entry.
        fig = plot_optima_matrix(full_grid, x_group="debt", color_group="agent")
        colour_legend = fig.legends[0]
        assert colour_legend.get_title().get_text() == "agent (colour)"
        assert [t.get_text() for t in colour_legend.get_texts()] == \
            ["low", "medium", "high", "pessimistic"]

    def test_both_axes_are_logarithmic(self, default_fig):
        assert default_fig.axes[0].get_xscale() == "log"
        assert default_fig.axes[0].get_yscale() == "log"

    def test_a_zero_share_is_clipped_onto_the_log_axis(self, full_grid, default_fig):
        assert (full_grid["cost_agent_share"] == 0.0).any()
        drawn_x = [x for ax in default_fig.axes for coll in ax.collections
                   for x, _ in coll.get_offsets()]
        assert min(drawn_x) > 0.0

    @pytest.mark.parametrize("share", list(COST_SHARES))
    def test_every_cost_driver_can_go_on_x(self, share):
        assert isinstance(plot_optima_matrix(_few(), share=share), Figure)

    def test_unknown_share_rejected(self):
        with pytest.raises(ValueError, match="unknown cost share"):
            plot_optima_matrix(_few(), share="tokens")

    def test_a_group_may_drive_only_one_channel(self):
        with pytest.raises(ValueError, match="only one visual channel"):
            plot_optima_matrix(_few(), x_group="agent", color_group="agent")

    def test_unknown_group_rejected(self):
        with pytest.raises(ValueError, match="unknown scenario group"):
            plot_optima_matrix(_few(), color_group="weather")

    def test_missing_tier_column_rejected(self):
        with pytest.raises(KeyError, match="tier_debt"):
            plot_optima_matrix(_few().drop(columns=["tier_debt"]))

    def test_missing_share_column_rejected(self):
        with pytest.raises(KeyError, match="cost_agent_share"):
            plot_optima_matrix(_few().drop(columns=["cost_agent_share"]))


class TestHumanAgentMatrix:
    """The variant shares `_draw_matrix`, so only what it owns needs its own tests."""

    def test_puts_the_two_team_cost_drivers_on_the_axes(self, full_grid):
        fig = plot_human_agent_matrix(full_grid)
        # Only the bottom row carries an x label and only the left column a y label.
        assert "agent cost" in fig.axes[-1].get_xlabel()
        assert "developer cost" in fig.axes[0].get_ylabel()
        assert (fig.axes[0].get_xscale(), fig.axes[0].get_yscale()) == ("log", "log")

    def test_same_facets_and_point_count_as_the_optima_matrix(self, full_grid):
        fig = plot_human_agent_matrix(full_grid)
        assert len(fig.axes) == 12
        drawn = sum(len(coll.get_offsets()) for ax in fig.axes for coll in ax.collections)
        assert drawn == len(full_grid)

    def test_channel_validation_is_shared(self):
        with pytest.raises(ValueError, match="only one visual channel"):
            plot_human_agent_matrix(_few(), x_group="agent", size_group="agent")

    def test_both_axes_carry_the_same_range(self, full_grid):
        """Same unit on both axes, so a point's side of the 1:1 line is the reading."""
        fig = plot_human_agent_matrix(full_grid)
        for ax in fig.axes:
            assert ax.get_xlim() == ax.get_ylim()
        # And the range must hold every point of both columns, not just of one.  A
        # non-positive value is drawn at its log floor, which the range must cover too.
        lo, hi = fig.axes[0].get_xlim()
        for column in ("cost_agent_mean", "cost_human_mean"):
            drawn = full_grid[column][full_grid[column] > 0]
            floor = _log_floor(full_grid, column)
            assert lo < min(drawn.min(), floor if floor is not None else drawn.min())
            assert drawn.max() < hi

    def test_equal_aspect_so_a_dollar_is_the_same_distance_on_both_axes(self, full_grid):
        fig = plot_human_agent_matrix(full_grid)
        assert all(ax.get_aspect() == 1.0 for ax in fig.axes)

    @pytest.mark.parametrize("column", ["cost_agent_mean", "cost_human_mean"])
    def test_missing_cost_column_rejected(self, column):
        with pytest.raises(KeyError, match=column):
            plot_human_agent_matrix(_few().drop(columns=[column]))


def _drawn(fig, axis: int) -> list[float]:
    """Every point's x (`axis=0`) or y (`axis=1`) coordinate, over all panels."""
    return sorted(float(point[axis])
                  for ax in fig.axes for coll in ax.collections
                  for point in coll.get_offsets())


class TestQualityDevelopmentMatrix:
    """Shares `_draw_matrix` with the other matrices; only its own columns need testing."""

    def test_x_is_the_sum_of_the_two_team_cost_drivers(self, full_grid):
        fig = plot_quality_development_matrix(full_grid)
        expected = sorted(full_grid["cost_human_mean"] + full_grid["cost_agent_mean"])
        assert _drawn(fig, 0) == pytest.approx(expected)
        assert _drawn(fig, 1) == pytest.approx(sorted(full_grid["cost_quality_mean"]))

    def test_both_axes_carry_the_same_range_at_an_equal_aspect(self, full_grid):
        """Same unit on both axes, so a point's side of the 1:1 line is the reading."""
        fig = plot_quality_development_matrix(full_grid)
        for ax in fig.axes:
            assert ax.get_xlim() == ax.get_ylim()
            assert ax.get_aspect() == 1.0

    def test_the_derived_column_does_not_leak_into_the_caller_s_table(self):
        df = _few()
        plot_quality_development_matrix(df)
        assert "cost_development_mean" not in df.columns

    @pytest.mark.parametrize(
        "column", ["cost_quality_mean", "cost_human_mean", "cost_agent_mean"]
    )
    def test_missing_cost_column_rejected(self, column):
        with pytest.raises(KeyError, match=column):
            plot_quality_development_matrix(_few().drop(columns=[column]))


class TestAgentHumanRatioMatrix:
    def test_a_rank_on_a_linear_x_against_the_ratio_on_a_log_y(self, full_grid):
        fig = plot_agent_human_ratio_matrix(full_grid)
        for ax in fig.axes:
            assert ax.get_xscale() == "linear"
            assert ax.get_yscale() == "log"

    def test_the_percentile_is_taken_over_the_whole_frame_not_per_panel(self, full_grid):
        """A per-panel ECDF would put one point per panel at x = 100 %, not one overall."""
        fig = plot_agent_human_ratio_matrix(full_grid)
        ratio = full_grid["cost_agent_mean"] / full_grid["cost_human_mean"]
        assert _drawn(fig, 0) == pytest.approx(
            sorted(ratio.rank(method="average", pct=True))
        )

    def test_y_is_the_ratio_with_zeros_on_the_log_floor(self, full_grid):
        fig = plot_agent_human_ratio_matrix(full_grid)
        ratio = full_grid["cost_agent_mean"] / full_grid["cost_human_mean"]
        floor = _log_floor(full_grid.assign(r=ratio), "r")
        assert _drawn(fig, 1) == pytest.approx(
            sorted(r if r > 0 else floor for r in ratio)
        )

    def test_a_parity_line_is_drawn_at_one(self, full_grid):
        fig = plot_agent_human_ratio_matrix(full_grid)
        for ax in fig.axes:
            assert any(line.get_ydata()[0] == 1.0 for line in ax.get_lines())

    @pytest.mark.parametrize("column", ["cost_agent_mean", "cost_human_mean"])
    def test_missing_cost_column_rejected(self, column):
        with pytest.raises(KeyError, match=column):
            plot_agent_human_ratio_matrix(_few().drop(columns=[column]))


class TestVisualizeCampaign:
    def test_all_matrices_per_strategy(self, tmp_path):
        visualize_campaign(_few(("evolutionary", "descent")), tmp_path)
        assert sorted(p.name for p in tmp_path.glob("*.png")) == [
            "agent_human_ratio_matrix_descent.png",
            "agent_human_ratio_matrix_evolutionary.png",
            "human_agent_matrix_descent.png", "human_agent_matrix_evolutionary.png",
            "optima_matrix_descent.png", "optima_matrix_evolutionary.png",
            "quality_development_matrix_descent.png",
            "quality_development_matrix_evolutionary.png",
        ]

    def test_channels_are_passed_through(self, tmp_path):
        # A duplicated channel must surface, not be swallowed by the loop.
        with pytest.raises(ValueError, match="only one visual channel"):
            visualize_campaign(_few(), tmp_path, color_group="agent")

    def test_a_table_without_strategies_is_rejected(self, tmp_path):
        with pytest.raises(KeyError, match="strategy"):
            visualize_campaign(_few().drop(columns=["strategy"]), tmp_path)


class TestSaveFigure:
    @pytest.mark.parametrize("fmt", ["png", "pdf"])
    def test_writes_the_requested_format(self, tmp_path, fmt):
        save_figure(plot_optima_matrix(_few()), tmp_path / "plots", "optima_matrix", fmt)
        assert (tmp_path / "plots" / f"optima_matrix.{fmt}").is_file()


# ---------------------------------------------------------------------------
# The human-only baseline overlay (paper §4.1)
# ---------------------------------------------------------------------------

def _baseline(n: int = 6) -> pd.DataFrame:
    """Synthetic baseline rows: `debt` and `agent` carry the UNUSED_TIER sentinel."""
    rows = []
    varying = [t for t in scenario_tier_combinations()[: n * 4]]
    seen = set()
    for tiers in varying:
        key = (tiers[0], tiers[1], tiers[4])   # human, project, bugimpact
        if key in seen:
            continue
        seen.add(key)
        sub = tuple(UNUSED_TIER if g in ("debt", "agent") else t
                    for g, t in zip(SCENARIO_GROUPS, tiers))
        cost = 1000.0 * (len(rows) + 1)
        rows.append({
            "scenario_id": scenario_id(sub), "strategy": "human_only",
            "cost_mean": cost,
            **{f"tier_{g}": t for g, t in zip(SCENARIO_GROUPS, sub)},
            # A team with no agents: zero bill, zero share.
            "cost_agent_share": 0.0, "cost_human_share": 0.25,
            "cost_quality_share": 0.75,
            "cost_agent_mean": 0.0,
            "cost_human_mean": cost * 0.25,
            "cost_quality_mean": cost * 0.75,
        })
        if len(rows) == n:
            break
    return pd.DataFrame(rows)


def _baseline_points(fig: "Figure") -> int:
    """How many points across the figure were drawn as the baseline overlay."""
    return sum(
        len(coll.get_offsets())
        for ax in fig.get_axes()
        for coll in ax.collections
        if coll.get_label() == _BASELINE_LABEL
    )


class TestBaselineCellMatching:
    def test_a_group_the_baseline_does_not_vary_matches_every_tier(self):
        base = _baseline()
        for agent_tier in ("low", "medium", "high", "pessimistic"):
            cell = _baseline_cell(base, "agent", agent_tier, "human", "low")
            expected = base[(base["tier_human"] == "low")]
            assert len(cell) == len(expected)

    def test_a_group_the_baseline_does_vary_selects_properly(self):
        base = _baseline()
        low = _baseline_cell(base, "agent", "medium", "human", "low")
        assert set(low["tier_human"]) <= {"low"}

    def test_no_baseline_row_survives_a_tier_it_does_not_carry(self):
        base = _baseline()
        cell = _baseline_cell(base, "human", "high", "project", "veryhigh")
        assert set(cell["tier_human"]) <= {"high"}


class TestBaselineOverlay:
    def test_optima_matrix_draws_the_baseline_in_every_agent_column(self):
        df, base = _few(), _baseline(3)
        without = plot_optima_matrix(df, baseline=None)
        with_base = plot_optima_matrix(df, baseline=base)
        assert _baseline_points(without) == 0
        assert _baseline_points(with_base) > 0
        plt.close(without), plt.close(with_base)

    def test_baseline_repeats_across_the_agent_facet(self, full_grid):
        # `agent` is a facet the baseline does not vary, so every column carries the same
        # reference points: 4 agent tiers x whatever each (human) row selects.
        base = _baseline(6)
        fig = plot_optima_matrix(full_grid, baseline=base)
        # 4 agent columns, so each baseline row is drawn once per column.
        assert _baseline_points(fig) == 4 * len(base)
        plt.close(fig)

    def test_human_agent_matrix_places_the_zero_agent_cost_on_the_axis(self):
        # cost_agent_mean is exactly 0 there; without a floor derived over both frames it
        # would drop silently off the log axis.
        df, base = _few(), _baseline(3)
        fig = plot_human_agent_matrix(df, baseline=base)
        assert _baseline_points(fig) > 0
        plt.close(fig)

    def test_quality_development_matrix_derives_the_column_for_the_baseline(self):
        # cost_development_mean is derived inside the plot function and must be derived
        # on the baseline frame too, or the overlay would raise a KeyError.
        df, base = _few(), _baseline(3)
        fig = plot_quality_development_matrix(df, baseline=base)
        assert _baseline_points(fig) > 0
        plt.close(fig)

    def test_ratio_matrix_takes_no_baseline(self):
        # A team with no agents has no agent/developer ratio.
        assert "baseline" not in inspect.signature(
            plot_agent_human_ratio_matrix).parameters


class TestVisualizeCampaignWithBaseline:
    def test_the_baseline_gets_no_figure_set_of_its_own(self, tmp_path):
        df = pd.concat([_few(("evolutionary", "descent")), _baseline(3)],
                       ignore_index=True)
        visualize_campaign(df, tmp_path, fmt="png")
        written = {p.name for p in tmp_path.glob("*.png")}
        assert not any("human_only" in name for name in written)
        for strategy in ("evolutionary", "descent"):
            assert f"optima_matrix_{strategy}.png" in written
            assert f"agent_human_ratio_matrix_{strategy}.png" in written

    def test_a_campaign_without_a_baseline_still_works(self, tmp_path):
        visualize_campaign(_few(("evolutionary",)), tmp_path, fmt="png")
        assert (tmp_path / "optima_matrix_evolutionary.png").is_file()
