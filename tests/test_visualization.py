"""Tests for src/visualization.py — the campaign-level five-channel optima matrix.

The per-search figures are exercised only indirectly (they take the raw `objectives`
dicts and have no branching worth pinning); what needs guarding is the channel
assignment the campaign matrices share, since a duplicated or missing group silently
collapses two scenario dimensions into one.  `plot_human_agent_matrix` stands in for
all of them there: they differ only in the pair of columns on the axes.

Rendering the full 432-scenario grid costs a few seconds — one `ax.scatter` per
(size, shape, colour) subgroup per panel — so the tests that only inspect a figure's
structure share one module-scoped render, and the rest use a handful of scenarios.
"""

from __future__ import annotations

import itertools

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from src.config import (
    AGENT_ROLE_ABBREV,
    SCENARIO_GROUPS,
    UNUSED_TIER,
    scenario_id,
    scenario_tier_combinations,
    scenario_tiers,
    tiers_present,
)
from src.constellation import BILL_PARAMETERS, role_bill
from src.visualization import (
    COMPOSITION_BAR_GROUP,
    COMPOSITION_BLOCK_GROUP,
    COMPOSITION_PANEL_GROUP,
    MATRIX_CHANNELS,
    _BUG_COST_LABEL,
    _compact_dollars,
    _BASELINE_COLOR,
    _BASELINE_LABEL,
    _TIER_STYLES,
    _baseline_cell,
    _baseline_style_groups,
    _log_floor,
    plot_agent_human_ratio_matrix,
    plot_human_agent_matrix,
    plot_elite_turnover,
    _SHARE_WEIGHTS,
    _TEAM_QUANTITIES,
    _team_quantities,
    plot_team_drivers,
    plot_team_shape,
    plot_quality_development_matrix,
    save_figure,
    visualize_campaign,
    visualize_team_summary,
)


def _optima(
    combinations: list[tuple[str, ...]] | None = None,
    modes: tuple[str, ...] = ("search",),
) -> pd.DataFrame:
    """A synthetic `optima.csv`: one row per (scenario, mode)."""
    combinations = (scenario_tier_combinations() if combinations is None
                    else combinations)
    rows = []
    for i, (tiers, mode) in enumerate(itertools.product(combinations, modes)):
        row = {"scenario_id": scenario_id(tiers), "mode": mode,
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


def _few(modes: tuple[str, ...] = ("search",)) -> pd.DataFrame:
    """A handful of scenarios — enough for the tests that do not read tier structure."""
    return _optima(scenario_tier_combinations()[:6], modes)


@pytest.fixture(scope="module")
def full_grid() -> pd.DataFrame:
    return _optima()


@pytest.fixture(scope="module")
def default_fig(full_grid) -> Figure:
    """`plot_human_agent_matrix` over the whole grid; read-only, shared.

    Rendering the 432-scenario grid costs two to four seconds, so the tests that only
    inspect a finished figure share one render instead of each paying for its own.
    """
    return plot_human_agent_matrix(full_grid)


@pytest.fixture(scope="module")
def quality_fig(full_grid) -> Figure:
    """`plot_quality_development_matrix` over the whole grid; read-only, shared."""
    return plot_quality_development_matrix(full_grid)


@pytest.fixture(scope="module")
def ratio_fig(full_grid) -> Figure:
    """`plot_agent_human_ratio_matrix` over the whole grid; read-only, shared."""
    return plot_agent_human_ratio_matrix(full_grid)


class TestTierHelpers:
    def test_tiers_come_back_in_mild_to_extreme_order(self, full_grid):
        assert tiers_present(full_grid["tier_human"]) == ["low", "medium", "high"]
        # `project` and `agent` are the two groups that reach a fourth tier.
        assert tiers_present(full_grid["tier_project"]) == \
            ["low", "medium", "high", "veryhigh"]
        assert tiers_present(full_grid["tier_agent"]) == \
            ["low", "medium", "high", "pessimistic"]

    def test_absent_tiers_are_skipped(self, full_grid):
        df = full_grid[full_grid["tier_debt"] != "medium"]
        assert tiers_present(df["tier_debt"]) == ["low", "high"]

    def test_log_floor_is_half_the_smallest_positive(self):
        assert _log_floor(pd.DataFrame({"x": [0.0, 0.004, 0.02]}), "x") \
            == pytest.approx(0.002)

    def test_no_floor_without_a_non_positive_value(self):
        assert _log_floor(pd.DataFrame({"x": [0.004, 0.02]}), "x") is None


class TestMatrixChannels:
    """The five-channel tier grid `_draw_matrix` owns, exercised through one figure.

    Every campaign matrix goes through it and differs only in the pair of columns on
    the axes, so the assignment is pinned once here rather than per figure.  It is the
    fixed `MATRIX_CHANNELS` and not an argument, so what is pinned is that constant and
    the way the grid follows the *data* within it.
    """

    def test_the_assignment_covers_every_group_exactly_once(self):
        """Five channels, five groups, no group driving two of them.

        This used to be checked at every call, when the channels were arguments.  With
        one fixed assignment it is a property of the constant, so it is checked once.
        """
        assert set(MATRIX_CHANNELS) == {
            "x_group", "y_group", "color_group", "marker_group", "size_group"}
        assert sorted(MATRIX_CHANNELS.values()) == sorted(SCENARIO_GROUPS)

    def test_project_is_on_the_shape_channel_and_agent_on_a_facet(self):
        """Where the two four-tier groups sit is deliberate, not incidental.

        Shape is the only per-point channel whose fourth level reads as a peer rather
        than as "more", and a facet axis absorbs `agent`'s fourth tier for one extra
        column — a fifth marker shape would not survive the smallest size tier.
        """
        assert MATRIX_CHANNELS["marker_group"] == "project"
        assert MATRIX_CHANNELS["x_group"] == "agent"

    def test_grid_is_the_two_facet_groups(self, default_fig):
        # agent (four tiers) x human (three).
        assert isinstance(default_fig, Figure)
        assert len(default_fig.axes) == 12

    def test_the_facet_count_follows_the_data(self, full_grid):
        # A campaign missing the fourth agent tier gets a 3x3 grid: the panels come from
        # the tiers actually present, not from a fixed layout.
        narrowed = full_grid[full_grid["tier_agent"] != "pessimistic"]
        assert len(plot_human_agent_matrix(narrowed).axes) == 9

    def test_every_point_is_drawn_exactly_once(self, full_grid, default_fig):
        drawn = sum(len(coll.get_offsets()) for ax in default_fig.axes
                    for coll in ax.collections)
        assert drawn == len(full_grid)

    def test_three_proxy_legends_one_per_non_facet_channel(self, default_fig):
        titles = [lg.get_title().get_text() for lg in default_fig.legends]
        assert titles == ["debt (colour)", "project (shape)", "bugimpact (size)"]
        # The four-tier group gets a fourth entry; the three-tier ones do not.
        assert [len(lg.get_texts()) for lg in default_fig.legends] == [3, 4, 3]

    def test_the_fourth_project_tier_draws_on_the_shape_channel(self, default_fig):
        """`veryhigh` is the one fourth tier that reaches a per-point channel.

        `agent` is a facet, so `pessimistic` never enters the colour / shape / size
        dicts; `project` is on the shape channel, so `_TIER_STYLES`' fourth slot — which
        the two share — is exercised through it.
        """
        shape_legend = default_fig.legends[1]
        assert shape_legend.get_title().get_text() == "project (shape)"
        assert [t.get_text() for t in shape_legend.get_texts()] == \
            ["low", "medium", "high", "veryhigh"]

    def test_a_zero_value_is_clipped_onto_the_log_axis(self, full_grid, default_fig):
        assert (full_grid["cost_agent_mean"] == 0.0).any()
        drawn_x = [x for ax in default_fig.axes for coll in ax.collections
                   for x, _ in coll.get_offsets()]
        assert min(drawn_x) > 0.0

    def test_missing_tier_column_rejected(self):
        with pytest.raises(KeyError, match="tier_debt"):
            plot_human_agent_matrix(_few().drop(columns=["tier_debt"]))


class TestHumanAgentMatrix:
    """`_draw_matrix` is pinned by `TestMatrixChannels`; only what this figure owns."""

    def test_puts_the_two_team_cost_drivers_on_the_axes(self, default_fig):
        fig = default_fig
        # Only the bottom row carries an x label and only the left column a y label.
        assert "agent cost" in fig.axes[-1].get_xlabel()
        assert "developer cost" in fig.axes[0].get_ylabel()
        assert (fig.axes[0].get_xscale(), fig.axes[0].get_yscale()) == ("log", "log")

    def test_both_axes_carry_the_same_range(self, full_grid, default_fig):
        """Same unit on both axes, so a point's side of the 1:1 line is the reading."""
        fig = default_fig
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

    def test_equal_aspect_so_a_dollar_is_the_same_distance_on_both_axes(self, default_fig):
        assert all(ax.get_aspect() == 1.0 for ax in default_fig.axes)

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

    def test_x_is_the_sum_of_the_two_team_cost_drivers(self, full_grid, quality_fig):
        fig = quality_fig
        expected = sorted(full_grid["cost_human_mean"] + full_grid["cost_agent_mean"])
        assert _drawn(fig, 0) == pytest.approx(expected)
        assert _drawn(fig, 1) == pytest.approx(sorted(full_grid["cost_quality_mean"]))

    def test_both_axes_carry_the_same_range_at_an_equal_aspect(self, quality_fig):
        """Same unit on both axes, so a point's side of the 1:1 line is the reading."""
        fig = quality_fig
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
    def test_a_rank_on_a_linear_x_against_the_ratio_on_a_log_y(self, ratio_fig):
        fig = ratio_fig
        for ax in fig.axes:
            assert ax.get_xscale() == "linear"
            assert ax.get_yscale() == "log"

    def test_the_percentile_is_taken_over_the_whole_frame_not_per_panel(self, full_grid,
                                                                        ratio_fig):
        """A per-panel ECDF would put one point per panel at x = 100 %, not one overall."""
        fig = ratio_fig
        ratio = full_grid["cost_agent_mean"] / full_grid["cost_human_mean"]
        assert _drawn(fig, 0) == pytest.approx(
            sorted(ratio.rank(method="average", pct=True))
        )

    def test_y_is_the_ratio_with_zeros_on_the_log_floor(self, full_grid, ratio_fig):
        fig = ratio_fig
        ratio = full_grid["cost_agent_mean"] / full_grid["cost_human_mean"]
        floor = _log_floor(full_grid.assign(r=ratio), "r")
        assert _drawn(fig, 1) == pytest.approx(
            sorted(r if r > 0 else floor for r in ratio)
        )

    def test_a_parity_line_is_drawn_at_one(self, ratio_fig):
        for ax in ratio_fig.axes:
            assert any(line.get_ydata()[0] == 1.0 for line in ax.get_lines())

    @pytest.mark.parametrize("column", ["cost_agent_mean", "cost_human_mean"])
    def test_missing_cost_column_rejected(self, column):
        with pytest.raises(KeyError, match=column):
            plot_agent_human_ratio_matrix(_few().drop(columns=[column]))


class TestVisualizeCampaign:
    def test_all_three_matrices_are_written(self, tmp_path):
        """One search, so three figures and no mode suffix on their names."""
        visualize_campaign(_few(("search",)), tmp_path)
        assert sorted(p.name for p in tmp_path.glob("*.png")) == [
            "agent_human_ratio_matrix.png",
            "human_agent_matrix.png",
            "quality_development_matrix.png",
        ]

    def test_the_turnover_figure_is_added_when_the_curves_are_given(self, tmp_path):
        visualize_campaign(_few(("search",)), tmp_path,
                           turnover=_turnover(("search",)))
        assert (tmp_path / "elite_turnover.png").is_file()

    def test_a_mode_without_curves_gets_no_turnover_figure(self, tmp_path):
        """The human-only enumeration has no generations, so it contributes no rows."""
        visualize_campaign(_few(("search",)), tmp_path,
                           turnover=_turnover(("human_only",)))
        assert not (tmp_path / "elite_turnover.png").is_file()
        assert (tmp_path / "human_agent_matrix.png").is_file()

    def test_every_matrix_is_drawn_on_the_one_fixed_layout(self, tmp_path):
        """All three matrices come out of `MATRIX_CHANNELS`; none takes a layout."""
        visualize_campaign(_few(("search",)), tmp_path)
        assert {p.name for p in tmp_path.glob("*.png")} == {
            "human_agent_matrix.png", "quality_development_matrix.png",
            "agent_human_ratio_matrix.png"}

    def test_a_table_without_a_mode_column_is_rejected(self, tmp_path):
        with pytest.raises(KeyError, match="mode"):
            visualize_campaign(_few().drop(columns=["mode"]), tmp_path)

    def test_a_baseline_only_table_draws_nothing(self, tmp_path):
        """Every row is a zero-agent baseline, so there is no search optimum to draw."""
        with pytest.warns(UserWarning, match="no search optimum"):
            visualize_campaign(_few(("human_only",)), tmp_path)
        assert not list(tmp_path.glob("*.png"))


def _turnover(modes=("search",), n_searches=3, n_generations=5):
    """A long-format elite_turnover table like the campaign's."""
    return pd.DataFrame([
        {"scenario_id": f"s{i}", "mode": mode, "generation": g,
         "tau": 1.0 / (g + i), "incumbent_changed": g <= 2, "best_cost": 1000.0 - g}
        for mode in modes
        for i in range(n_searches)
        for g in range(1, n_generations + 1)
    ])


def _curve(fig, label_prefix):
    """The one curve, on either axis, whose legend label starts with `label_prefix`."""
    lines = [line for ax in fig.axes for line in ax.lines
             if line.get_label().startswith(label_prefix)]
    assert len(lines) == 1, f"{label_prefix!r} matched {len(lines)} curves"
    return lines[0]


class TestEliteTurnover:
    """The convergence figure: mean and median of tau_g and of `c*`'s cost drop."""

    def test_draws_a_figure(self):
        fig = plot_elite_turnover(_turnover(), title="turnover")
        assert isinstance(fig, Figure)
        # Three curves: mean and median tau_g on the left axis, the mean drop on the
        # right one.  No shading, and no median drop — it is zero almost everywhere.
        assert [len(ax.lines) for ax in fig.axes] == [2, 1]
        assert all(len(ax.collections) == 0 for ax in fig.axes)
        assert not [line for ax in fig.axes for line in ax.lines
                    if line.get_label().startswith("median drop")]
        plt.close(fig)

    def test_the_drop_gets_a_logarithmic_axis_of_its_own(self):
        """It is a rate like tau_g, but decades below it — a shared linear axis buries it."""
        fig = plot_elite_turnover(_turnover())
        turnover_ax, drop_ax = fig.axes
        assert turnover_ax.get_yscale() == "linear"
        assert drop_ax.get_yscale() == "log"
        assert _curve(fig, "mean drop") in drop_ax.lines
        plt.close(fig)

    def test_the_mean_is_solid_and_the_median_dotted(self):
        """The two statistics are told apart by line style, the quantities by colour."""
        fig = plot_elite_turnover(_turnover())
        assert _curve(fig, r"mean $\tau_g$").get_linestyle() == "-"
        assert _curve(fig, "mean drop").get_linestyle() == "-"
        assert _curve(fig, r"median $\tau_g$").get_linestyle() == ":"
        assert (_curve(fig, "mean drop").get_color()
                != _curve(fig, r"mean $\tau").get_color())
        plt.close(fig)

    def test_the_x_axis_covers_every_generation(self):
        fig = plot_elite_turnover(_turnover(n_generations=7))
        assert fig.axes[0].get_xlim() == (1.0, 7.0)
        plt.close(fig)

    def test_the_cost_drop_is_relative_to_the_previous_generation(self):
        """And unsigned, which is what makes it the same kind of rate as tau_g."""
        fig = plot_elite_turnover(_turnover(n_generations=4))
        # 1000 - g, so the drop is 1/(1000-g+1) of the previous generation's cost.
        expected = [1.0 / (1000.0 - g + 1) for g in range(2, 5)]
        drawn = np.asarray(_curve(fig, "mean drop").get_ydata(), dtype=float)
        assert np.isnan(drawn[0])              # generation 1 has no predecessor
        assert drawn[1:] == pytest.approx(expected)
        plt.close(fig)

    def test_each_search_is_normalised_by_its_own_cost(self):
        """Costs span decades across scenarios; only the relative drop is comparable."""
        rows = [
            {"scenario_id": name, "mode": "search", "generation": g,
             "tau": 0.5, "incumbent_changed": False, "best_cost": cost}
            for name, costs in [("cheap", [10.0, 5.0]), ("dear", [1000.0, 500.0])]
            for g, cost in enumerate(costs, start=1)
        ]
        fig = plot_elite_turnover(pd.DataFrame(rows))
        assert _curve(fig, "mean drop").get_ydata()[1] == pytest.approx(0.5)
        plt.close(fig)

    def test_an_empty_table_is_rejected(self):
        with pytest.raises(ValueError, match="at least one generation"):
            plot_elite_turnover(_turnover().iloc[0:0])

    def test_a_table_without_tau_is_rejected(self):
        with pytest.raises(KeyError, match="tau"):
            plot_elite_turnover(_turnover().drop(columns=["tau"]))

    def test_a_curve_a_log_axis_cannot_show_says_so_in_its_label(self):
        """An incumbent that never moves gives a drop of 0, which a log axis omits."""
        rows = [
            {"scenario_id": "flat", "mode": "search", "generation": g,
             "tau": 0.5, "best_cost": 100.0}
            for g in (1, 2, 3)
        ]
        fig = plot_elite_turnover(pd.DataFrame(rows))
        assert "zero throughout" in _curve(fig, "mean drop").get_label()
        assert "zero throughout" not in _curve(fig, r"mean $\tau_g$").get_label()
        plt.close(fig)

    def test_the_mean_and_median_differ_on_a_skewed_sample(self):
        """One search improving while the others stand still must move only the mean."""
        rows = [
            {"scenario_id": name, "mode": "search", "generation": g,
             "tau": tau, "best_cost": cost}
            for name, taus, costs in [
                ("a", [0.5, 0.1], [100.0, 100.0]),
                ("b", [0.5, 0.1], [100.0, 100.0]),
                ("c", [0.5, 0.7], [100.0, 10.0]),
            ]
            for g, (tau, cost) in enumerate(zip(taus, costs), start=1)
        ]
        fig = plot_elite_turnover(pd.DataFrame(rows))
        assert _curve(fig, r"median $\tau_g$").get_ydata() == pytest.approx([0.5, 0.1])
        assert _curve(fig, r"mean $\tau_g$").get_ydata() == pytest.approx([0.5, 0.3])
        assert _curve(fig, "mean drop").get_ydata()[1] == pytest.approx(0.3)
        plt.close(fig)


class TestSaveFigure:
    def test_writes_a_png_and_creates_the_directory(self, tmp_path):
        """PNG throughout; there is no format knob."""
        save_figure(plot_human_agent_matrix(_few()), tmp_path / "plots",
                    "human_agent_matrix")
        assert (tmp_path / "plots" / "human_agent_matrix.png").is_file()


# ---------------------------------------------------------------------------
# The human-only baseline overlay
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
            "scenario_id": scenario_id(sub), "mode": "human_only",
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


def _baseline_points(fig: Figure) -> int:
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


@pytest.fixture(scope="module")
def overlay_baseline() -> pd.DataFrame:
    return _baseline(6)


@pytest.fixture(scope="module")
def overlay_fig(full_grid, overlay_baseline) -> Figure:
    """The whole grid with the baseline overlaid; read-only, shared."""
    return plot_human_agent_matrix(full_grid, baseline=overlay_baseline)


class TestBaselineOverlay:
    def test_a_matrix_draws_the_baseline_only_when_it_is_given_one(self):
        df, base = _few(), _baseline(3)
        without = plot_human_agent_matrix(df, baseline=None)
        with_base = plot_human_agent_matrix(df, baseline=base)
        assert _baseline_points(without) == 0
        assert _baseline_points(with_base) > 0
        plt.close(without), plt.close(with_base)

    def test_baseline_repeats_across_the_agent_facet(self, overlay_fig, overlay_baseline):
        # `agent` is a facet the baseline does not vary, so every column carries the same
        # reference points: 4 agent tiers x whatever each (human) row selects.
        assert _baseline_points(overlay_fig) == 4 * len(overlay_baseline)

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

    def test_baseline_takes_the_shape_and_size_of_the_groups_it_varies(
            self, overlay_fig, overlay_baseline):
        # `project` on the shape channel and `bugimpact` on the size channel are both
        # groups the baseline varies, so its points carry the tier styles rather than the
        # one fixed style — one collection per (project, bugimpact) pair present.
        styles = {
            (coll.get_paths()[0].vertices.tobytes(), float(coll.get_sizes()[0]))
            for ax in overlay_fig.get_axes()
            for coll in ax.collections
            if coll.get_label() == _BASELINE_LABEL
        }
        pairs = {(r.tier_project, r.tier_bugimpact)
                 for r in overlay_baseline.itertuples()}
        assert len(styles) == len(pairs) > 1

    def test_the_baseline_is_black_on_every_style(self, overlay_fig):
        # Colour is the one channel that stays fixed: with shape and size now shared with
        # the search optima it is what still separates the baseline from them.
        fig = overlay_fig
        black = matplotlib.colors.to_rgb(_BASELINE_COLOR)
        for ax in fig.get_axes():
            for coll in ax.collections:
                if coll.get_label() != _BASELINE_LABEL:
                    continue
                # The open shapes carry the colour on the edge, the filled ones on the
                # face; compared on RGB alone, since the overlay is drawn transparent.
                drawn = [tuple(c) for c in coll.get_edgecolors()] or \
                        [tuple(c) for c in coll.get_facecolors()]
                assert all(c[:3] == black for c in drawn)

    def test_the_baseline_is_drawn_transparent(self, overlay_fig):
        # It sits on top of the search optima, so an opaque marker would mask them.
        drawn = [coll for ax in overlay_fig.get_axes() for coll in ax.collections
                 if coll.get_label() == _BASELINE_LABEL]
        assert drawn and all(coll.get_alpha() < 1.0 for coll in drawn)

    def test_a_group_the_baseline_does_not_vary_falls_back_to_the_fixed_style(self):
        # `debt` on the shape channel: the baseline holds it at UNUSED_TIER, so there is
        # no tier to encode and the star stands in.
        base = _baseline(6)
        blocks = _baseline_style_groups(base, "debt", "bugimpact")
        assert [marker for marker, _, _ in blocks] == [None] * len(blocks)
        assert all(size is not None for _, size, _ in blocks)

    def test_style_blocks_are_drawn_largest_first(self):
        base = _baseline(6)
        areas = [_TIER_STYLES[size].area for _, size, _ in
                 _baseline_style_groups(base, "project", "bugimpact")]
        assert areas == sorted(areas, reverse=True)


class TestVisualizeCampaignWithBaseline:
    def test_the_baseline_gets_no_figure_set_of_its_own(self, tmp_path):
        df = pd.concat([_few(("search",)), _baseline(3)], ignore_index=True)
        visualize_campaign(df, tmp_path)
        written = {p.name for p in tmp_path.glob("*.png")}
        assert not any("human_only" in name for name in written)
        assert "human_agent_matrix.png" in written
        assert "agent_human_ratio_matrix.png" in written

    def test_a_campaign_without_a_baseline_still_works(self, tmp_path):
        visualize_campaign(_few(("search",)), tmp_path)
        assert (tmp_path / "human_agent_matrix.png").is_file()


# ---------------------------------------------------------------------------
# The two-figure team-structure summary
# ---------------------------------------------------------------------------

def _summary_optima(modes: tuple[str, ...] = ("search",)) -> pd.DataFrame:
    """A synthetic `optima.csv` with compositions the summary figures can compress."""
    df = _optima(modes=modes)
    rng = np.random.default_rng(0)
    for k in range(1, 9):
        df[f"a_dev_{k}"] = rng.integers(0, 4, len(df))
        df[f"a_qa_{k}"] = 0
    # The two detection services carry the team, as they do on the real campaign, and
    # Security scales with bugimpact so the figures have a signal to show.
    df["a_dev_5"] = 40
    df["a_dev_6"] = df["tier_bugimpact"].map({"low": 4, "medium": 12, "high": 30})
    df["a_dev_1"] = 1
    df["N"] = df["tier_agent"].map(
        {"low": 6.0, "medium": 1.0, "high": 0.5, "pessimistic": 7.0})
    # The two process controls the driver figure reports beside `N`.  `F_freeze` reaches 0,
    # as it does in 69 % of the real optima, so its panel must stay off a log axis.
    df["R"] = df["tier_debt"].map({"low": 1, "medium": 4, "high": 13})
    df["F_freeze"] = df["tier_human"].map({"low": 0, "medium": 1, "high": 2})
    df["r_arrival"] = df["tier_project"].map(
        {"low": 3.0, "medium": 12.0, "high": 75.0, "veryhigh": 1000.0})
    df["value_mean"] = df["r_arrival"] * 0.99
    for k in range(1, 9):
        df[f"actions_{k}"] = 100.0 * df[f"a_dev_{k}"]
    df["cost_agent_mean"] = 12.34
    df["cost_human_mean"] = df["N"] * 40.0
    # The real prices, from the scenario configs the ids name — `role_bill` is what the
    # cost weighting depends on, so the fixture goes through it rather than faking it.
    return role_bill(df)


def _share_widths(fig) -> list[float]:
    """Every stacked segment's width, in draw order; closes the figure."""
    widths = [p.get_width() for p in fig.axes[0].patches]
    plt.close(fig)
    return widths


class TestTeamShape:
    def test_one_stacked_bar_per_cell_with_one_segment_per_role(self):
        fig = plot_team_shape(_summary_optima())
        # 4 project tiers x 3 bugimpact tiers, eight segments each.
        assert len(fig.axes[0].patches) == 4 * 3 * 8
        plt.close(fig)

    def test_every_bar_sums_to_one(self):
        """A share figure that does not close to 100 % is lying about the parts."""
        fig = plot_team_shape(_summary_optima())
        widths = {}
        for patch in fig.axes[0].patches:
            widths.setdefault(round(patch.get_y(), 3), 0.0)
            widths[round(patch.get_y(), 3)] += patch.get_width()
        assert all(total == pytest.approx(1.0) for total in widths.values())
        plt.close(fig)

    def test_the_share_is_the_mean_of_the_ratios_not_the_ratio_of_the_means(self):
        """A cell's largest scenario must not dictate the composition it reports."""
        df = _summary_optima()
        # The first bar of the first panel: the (agent, project, bugimpact) = low cell.
        cell = df[(df["tier_agent"] == "low") & (df["tier_project"] == "low")
                  & (df["tier_bugimpact"] == "low")]
        bill = list(BILL_PARAMETERS)
        expected = (cell[bill].div(cell[bill].sum(axis=1), axis=0)).mean()["bill_5"]
        fig = plot_team_shape(df)
        first_bar = [p for p in fig.axes[0].patches
                     if round(p.get_y(), 3) == round(fig.axes[0].patches[0].get_y(), 3)]
        assert first_bar[4].get_width() == pytest.approx(expected)
        plt.close(fig)

    def test_a_tiny_segment_is_drawn_but_not_labelled(self):
        fig = plot_team_shape(_summary_optima())
        labels = [t.get_text() for t in fig.axes[0].texts if t.get_text().endswith("%")]
        assert 0 < len(labels) < 4 * 3 * 8
        plt.close(fig)

    def test_the_x_axis_is_the_full_share_range(self):
        fig = plot_team_shape(_summary_optima())
        assert fig.axes[0].get_xlim() == (0.0, 1.0)
        plt.close(fig)

class TestTeamShapeWeighting:
    """The default weighs a role by what it bills, not by how many of it there are."""

    def test_the_default_is_the_agent_bill(self):
        """The default weighs a role by what it bills, and heads the column accordingly."""
        fig = plot_team_shape(_summary_optima())
        assert _SHARE_WEIGHTS["agent"].total_label in [
            t.get_text() for t in fig.axes[0].texts]
        plt.close(fig)

    def test_the_panelled_form_carries_no_x_label(self):
        """Percent ticks and the figure title say what the axis is; four copies would not."""
        fig = plot_team_shape(_summary_optima(), title="Share of agent costs")
        assert all(ax.get_xlabel() == "" for ax in fig.axes)
        # The suptitle is the only figure-level text left.
        assert [t.get_text() for t in fig.texts] == ["Share of agent costs"]
        plt.close(fig)

    def test_the_two_weightings_disagree(self):
        """The agent bill and the whole team bill are not the same split."""
        df = _summary_optima()
        by_agent = _share_widths(plot_team_shape(df))
        by_team = _share_widths(plot_team_shape(df, weight="team"))
        assert by_agent != pytest.approx(by_team, abs=1e-6)

    def test_an_unknown_weight_is_rejected(self):
        with pytest.raises(ValueError, match="unknown weight"):
            plot_team_shape(_summary_optima(), weight="headcount")

    def test_the_cost_weighting_needs_the_bill_columns(self):
        df = _summary_optima().drop(columns=list(BILL_PARAMETERS))
        with pytest.raises(KeyError, match="role_bill"):
            plot_team_shape(df)



class TestTeamBillWeighting:
    """The `team` weighting puts the developers in beside the eight agent roles."""

    def test_the_developers_are_a_ninth_slice(self):
        fig = plot_team_shape(_summary_optima(), weight="team")
        assert len(fig.axes[0].patches) == 4 * 3 * (len(AGENT_ROLE_ABBREV) + 1)
        labels = [t.get_text() for legend in fig.legends for t in legend.get_texts()]
        assert labels[0] == "Developers"
        plt.close(fig)

    def test_the_developers_are_off_the_agent_palette(self):
        """A human slice that reads as a ninth agent would be the one real mistake here."""
        fig = plot_team_shape(_summary_optima(), weight="team")
        first_bar = sorted({round(p.get_y(), 3) for p in fig.axes[0].patches})[0]
        segments = [p for p in fig.axes[0].patches if round(p.get_y(), 3) == first_bar]
        human = matplotlib.colors.to_rgba(segments[0].get_facecolor())
        agents = {matplotlib.colors.to_rgba(p.get_facecolor()) for p in segments[1:]}
        assert human not in agents
        plt.close(fig)

    def test_escaped_bug_cost_is_not_a_line_item_of_the_team(self):
        """It is what the team's choices cost elsewhere, not what the team bills."""
        assert "cost_quality_mean" not in _SHARE_WEIGHTS["team"].columns


class TestCompositionAnnotations:
    """Two dollar figures per bar: what the team costs, and what its bugs cost."""

    def test_both_columns_are_annotated_on_every_bar(self):
        fig = plot_team_shape(_summary_optima(), weight="team")
        dollars = [t.get_text() for t in fig.axes[0].texts if t.get_text().startswith("$")]
        # 4 project x 3 bugimpact rows, two columns; the headers carry no leading "$".
        assert len(dollars) == 4 * 3 * 2
        plt.close(fig)

    def test_the_bug_column_is_the_third_driver_of_the_cost_equation(self):
        df = _summary_optima()
        cell = df[(df["tier_agent"] == "low") & (df["tier_project"] == "low")
                  & (df["tier_bugimpact"] == "low")]
        fig = plot_team_shape(df, weight="team")
        expected = _compact_dollars(cell["cost_quality_mean"].mean())
        assert expected in [t.get_text() for t in fig.axes[0].texts]
        plt.close(fig)

    def test_it_is_skipped_when_the_table_has_no_bug_cost(self):
        """The bar still describes the team; only the annotation needs the column."""
        df = _summary_optima().drop(columns=["cost_quality_mean"])
        fig = plot_team_shape(df, weight="team")
        assert _BUG_COST_LABEL not in [t.get_text() for t in fig.axes[0].texts]
        plt.close(fig)
        fig = plot_team_shape(_summary_optima(), weight="team")
        assert _BUG_COST_LABEL in [t.get_text() for t in fig.axes[0].texts]
        plt.close(fig)

    @pytest.mark.parametrize("value, expected", [
        (83.4, "$83"), (999.4, "$999"), (1_000.0, "$1.0k"), (6_749.0, "$6.7k"),
        (26_209.0, "$26k"), (516_645.0, "$517k"), (2_400_000.0, "$2.4M"),
    ])
    def test_the_bug_column_is_compact(self, value, expected):
        """Four orders of magnitude have to fit in a margin, so it says the decade."""
        assert _compact_dollars(value) == expected

class TestTeamShapePanels:
    """A third scenario group, read off the panels rather than off the rows."""

    def test_one_panel_per_tier_of_the_panel_group(self):
        fig = plot_team_shape(_summary_optima())
        titles = [ax.get_title() for ax in fig.axes]
        assert titles == [f"agent = {t}" for t in scenario_tiers("agent")]
        plt.close(fig)

    def test_every_panel_carries_the_same_rows(self):
        """The panels are only comparable if a row means the same thing in each."""
        fig = plot_team_shape(_summary_optima())
        rows = [sorted({round(p.get_y(), 3) for p in ax.patches}) for ax in fig.axes]
        assert all(r == rows[0] for r in rows[1:])
        assert all(ax.get_ylim() == fig.axes[0].get_ylim() for ax in fig.axes)
        plt.close(fig)

    def test_the_row_labels_are_written_once_per_grid_row(self):
        """`sharey` aligns the columns, so the right one is read off the left one's labels."""
        fig = plot_team_shape(_summary_optima())
        left, right = fig.axes[::2], fig.axes[1::2]
        assert all(any(t.get_visible() for t in ax.get_yticklabels()) for ax in left)
        assert not any(t.get_visible() for ax in right for t in ax.get_yticklabels())
        plt.close(fig)

    def test_the_panels_are_two_to_a_grid_row(self):
        """Four side by side is a 19-inch strip; stacked, the rows stay aligned."""
        fig = plot_team_shape(_summary_optima())
        assert len({round(ax.get_position().x0, 3) for ax in fig.axes}) == 2
        assert len({round(ax.get_position().y0, 3) for ax in fig.axes}) == 2
        plt.close(fig)

    def test_a_panel_holds_only_its_own_tier(self):
        """Otherwise it is the pooled figure drawn four times."""
        fig = plot_team_shape(_summary_optima())
        widths = [tuple(round(p.get_width(), 6) for p in ax.patches) for ax in fig.axes]
        assert len({widths[0], widths[1]}) == 2
        plt.close(fig)

    def test_the_three_composition_groups_are_distinct(self):
        """Blocks, bars and panels must be three different groups."""
        groups = (COMPOSITION_PANEL_GROUP, COMPOSITION_BLOCK_GROUP,
                  COMPOSITION_BAR_GROUP)
        assert len(set(groups)) == 3


class TestTeamDrivers:
    def test_one_panel_per_quantity(self):
        fig = plot_team_drivers(_summary_optima())
        assert len(fig.axes) == len(_TEAM_QUANTITIES)
        plt.close(fig)

    def test_one_row_per_group_and_tier(self):
        fig = plot_team_drivers(_summary_optima())
        expected = sum(len(scenario_tiers(g)) for g in SCENARIO_GROUPS)
        assert len(fig.axes[0].get_yticklabels()) == expected
        plt.close(fig)

    def test_each_row_gets_a_median_marker_and_an_iqr_bar(self):
        fig = plot_team_drivers(_summary_optima())
        expected = sum(len(scenario_tiers(g)) for g in SCENARIO_GROUPS)
        # Two Line2D per row plus one rule per block boundary, and nothing else: there
        # is no campaign-median rule.
        assert len(fig.axes[0].lines) == 2 * expected + len(SCENARIO_GROUPS) - 1
        plt.close(fig)

    def test_the_groups_are_separated_by_a_rule_in_every_panel(self):
        """Six panels wide, the eye has to carry a block boundary across the figure."""
        fig = plot_team_drivers(_summary_optima())
        rules = {tuple(line.get_ydata()) for line in fig.axes[0].lines
                 if len(set(line.get_ydata())) == 1 and len(line.get_ydata()) == 2
                 and tuple(line.get_xdata()) == (0.0, 1.0)}
        assert len(rules) == len(SCENARIO_GROUPS) - 1
        for ax in fig.axes[1:]:
            assert len(ax.lines) == len(fig.axes[0].lines)
        plt.close(fig)

    def test_the_legend_names_the_statistic_each_row_shows(self):
        fig = plot_team_drivers(_summary_optima())
        labels = [t.get_text() for legend in fig.legends for t in legend.get_texts()]
        assert "median with interquartile range" in labels
        plt.close(fig)

    def test_the_two_halves_of_the_team_get_a_panel_each(self):
        """Supervised and autonomous counts are bounded by different things."""
        frame = _team_quantities(_summary_optima())
        supervised = frame[[f"a_dev_{k}" for k in (1, 2, 3, 7, 8)]].sum(axis=1)
        autonomous = frame[[f"a_dev_{k}" for k in (4, 5, 6)]].sum(axis=1)
        assert np.allclose(frame["sup_agents_per_feature"],
                           supervised / frame["r_arrival"])
        assert np.allclose(frame["aut_agents_per_feature"],
                           autonomous / frame["r_arrival"])

    def test_the_two_agent_panels_are_not_forced_onto_one_range(self):
        """A shared range puts every autonomous row on one tick; see `_TEAM_QUANTITIES`."""
        fig = plot_team_drivers(_summary_optima())
        columns = [q.column for q in _TEAM_QUANTITIES]
        sup = fig.axes[columns.index("sup_agents_per_feature")]
        aut = fig.axes[columns.index("aut_agents_per_feature")]
        assert sup.get_xlim() != aut.get_xlim()
        plt.close(fig)

    def test_every_panel_is_linear(self):
        """Chosen deliberately: a log axis reads the small tiers as far apart as they are."""
        fig = plot_team_drivers(_summary_optima())
        assert {ax.get_xscale() for ax in fig.axes} == {"linear"}
        plt.close(fig)

    def test_the_fourth_tier_gets_one_legend_entry_not_two(self):
        """`veryhigh` and `pessimistic` share a colour slot; five groups share one axis."""
        fig = plot_team_drivers(_summary_optima())
        labels = [t.get_text() for legend in fig.legends for t in legend.get_texts()]
        assert "veryhigh / pessimistic" in labels
        assert "veryhigh" not in labels and "pessimistic" not in labels
        plt.close(fig)

    def test_a_team_with_no_detection_service_reports_no_split(self):
        """0/0 is not a Security share of zero; it drops out of the order statistics."""
        df = _summary_optima()
        df.loc[:, "a_dev_5"] = 0
        df.loc[:, "a_dev_6"] = 0
        assert _team_quantities(df)["security_share"].isna().all()


class TestVisualizeTeamSummary:
    def test_both_figures_are_written(self, tmp_path):
        visualize_team_summary(_summary_optima(), tmp_path)
        assert sorted(p.name for p in tmp_path.glob("*.png")) == [
            "composition_agent_bill.png", "composition_team_bill.png",
            "team_drivers.png",
        ]

    def test_the_human_only_baseline_is_excluded(self, tmp_path):
        """A zero-agent team has no composition; it would divide the shares by zero."""
        df = _summary_optima(("search", "human_only"))
        df.loc[df["mode"] == "human_only",
               [f"a_dev_{k}" for k in range(1, 9)]] = 0
        visualize_team_summary(df, tmp_path)
        assert (tmp_path / "composition_agent_bill.png").is_file()

    def test_nothing_to_summarise_warns(self, tmp_path):
        df = _summary_optima(("human_only",))
        with pytest.warns(UserWarning, match="summarise"):
            visualize_team_summary(df, tmp_path)
        assert not list(tmp_path.glob("*.png"))
