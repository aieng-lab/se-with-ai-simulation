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

import itertools

import pandas as pd
import pytest
from matplotlib.figure import Figure

from src.config import SCENARIO_GROUPS, scenario_id, scenario_tier_combinations
from src.visualization import (
    COST_SHARES,
    _log_floor,
    _tiers_present,
    plot_human_agent_matrix,
    plot_optima_matrix,
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

    @pytest.mark.parametrize("column", ["cost_agent_mean", "cost_human_mean"])
    def test_missing_cost_column_rejected(self, column):
        with pytest.raises(KeyError, match=column):
            plot_human_agent_matrix(_few().drop(columns=[column]))


class TestVisualizeCampaign:
    def test_both_matrices_per_strategy(self, tmp_path):
        visualize_campaign(_few(("evolutionary", "descent")), tmp_path)
        assert sorted(p.name for p in tmp_path.glob("*.png")) == [
            "human_agent_matrix_descent.png", "human_agent_matrix_evolutionary.png",
            "optima_matrix_descent.png", "optima_matrix_evolutionary.png",
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
