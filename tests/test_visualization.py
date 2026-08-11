"""Tests for the campaign-level plots of `src.visualization` (paper §5).

Only the 3×3 matrix figures are covered: they are the plots that carry a non-trivial
contract (five distinct scenario groups, one per visual channel, and — for the ratio
variant — a percentile computed per front) rather than just drawing a point cloud.
`src.visualization` forces the Agg backend at import time, so these run headless.
"""

from __future__ import annotations

import itertools

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.config import SCENARIO_GROUPS, SCENARIO_TIERS
from src.visualization import (
    plot_composition_archetypes,
    plot_composition_drift,
    plot_composition_tier_heatmap,
    plot_quality_cost_matrix,
    plot_ratio_percentile_matrix,
    save_figure,
)


def _row(tiers: dict[str, str], quality: float, cost: float) -> dict:
    """One frontier-table row: the tier columns plus the objective columns a plot reads."""
    row = {f"tier_{group}": tiers[group] for group in SCENARIO_GROUPS}
    row.update(
        scenario_id="_".join(f"{g[0]}-{tiers[g][0]}" for g in SCENARIO_GROUPS),
        strategy="random",
        quality_mean=quality,
        cost_mean=cost,
        cost_human_mean=cost * 0.4,
        cost_agent_mean=cost * 0.6,
    )
    return row


@pytest.fixture
def synthetic_front() -> pd.DataFrame:
    """Two points for every one of the 3^5 = 243 tier combinations."""
    rows = []
    for i, combo in enumerate(itertools.product(SCENARIO_TIERS, repeat=len(SCENARIO_GROUPS))):
        tiers = dict(zip(SCENARIO_GROUPS, combo))
        rows.append(_row(tiers, quality=1.0 + i, cost=100.0 + i))
        rows.append(_row(tiers, quality=2.0 + i, cost=50.0 + i))
    return pd.DataFrame(rows)


def test_matrix_default_layout(synthetic_front):
    fig = plot_quality_cost_matrix(synthetic_front)
    try:
        axes = fig.get_axes()
        assert len(axes) == 9
        assert all(ax.get_xscale() == "log" and ax.get_yscale() == "log" for ax in axes)

        # Columns are the agent tiers, rows the human tiers, both low → medium → high.
        top_row = [axes[i].get_title(loc="center") for i in range(3)]
        assert top_row == ["agent = low", "agent = medium", "agent = high"]
        left_column = [axes[3 * r].get_ylabel().splitlines()[0] for r in range(3)]
        assert left_column == ["human = low", "human = medium", "human = high"]

        # Every cell holds 2 points × 27 combinations of the other three groups.
        assert axes[0].get_title(loc="right") == "n = 54"

        # x-labels only on the bottom row.
        assert axes[0].get_xlabel() == ""
        assert "total cost / feature" in axes[6].get_xlabel()
    finally:
        plt.close(fig)


def test_matrix_cost_field_relabels_x_axis(synthetic_front):
    fig = plot_quality_cost_matrix(synthetic_front, cost_field="cost_agent_mean")
    try:
        assert "agent cost / feature" in fig.get_axes()[6].get_xlabel()
    finally:
        plt.close(fig)


def test_matrix_custom_channel_assignment(synthetic_front):
    fig = plot_quality_cost_matrix(
        synthetic_front,
        x_group="debt", y_group="bugimpact", color_group="agent",
        marker_group="human", size_group="project",
    )
    try:
        axes = fig.get_axes()
        assert axes[0].get_title(loc="center") == "debt = low"
        assert axes[0].get_ylabel().startswith("bugimpact = low")
    finally:
        plt.close(fig)


def test_matrix_rejects_duplicate_group(synthetic_front):
    with pytest.raises(ValueError, match="only one visual channel"):
        plot_quality_cost_matrix(synthetic_front, x_group="agent", y_group="agent")


def test_matrix_rejects_unknown_group(synthetic_front):
    with pytest.raises(ValueError, match="unknown scenario group"):
        plot_quality_cost_matrix(synthetic_front, x_group="nonsense")


def test_matrix_rejects_missing_tier_column(synthetic_front):
    with pytest.raises(KeyError, match="tier_debt"):
        plot_quality_cost_matrix(synthetic_front.drop(columns=["tier_debt"]))


def test_matrix_rejects_unknown_cost_field(synthetic_front):
    with pytest.raises(KeyError, match="cost_nonsense"):
        plot_quality_cost_matrix(synthetic_front, cost_field="cost_nonsense")


def test_matrix_tolerates_missing_tier(synthetic_front):
    """A partial campaign can lack a whole tier; the grid shrinks instead of failing."""
    partial = synthetic_front[synthetic_front["tier_agent"] != "medium"]
    fig = plot_quality_cost_matrix(partial)
    try:
        axes = fig.get_axes()
        assert len(axes) == 6
        assert [axes[i].get_title(loc="center") for i in range(2)] == [
            "agent = low", "agent = high",
        ]
    finally:
        plt.close(fig)


def test_matrix_y_is_the_quality_cost_product(synthetic_front):
    """The cost matrix plots the combined index on y, cost on x."""
    expected = {
        round(float(q * c), 9)
        for q, c in zip(synthetic_front["quality_mean"], synthetic_front["cost_mean"])
    }
    fig = plot_quality_cost_matrix(synthetic_front)
    try:
        drawn = {
            round(float(y), 9)
            for ax in fig.get_axes()
            for collection in ax.collections
            for _, y in collection.get_offsets()
        }
        assert drawn == expected
    finally:
        plt.close(fig)


def test_matrix_clips_zero_index_to_one_shared_floor(synthetic_front):
    """Zero-index points land at the same y in every panel, not per-panel minima."""
    zeroed = synthetic_front.copy()
    zeroed.loc[zeroed.index[::7], "quality_mean"] = 0.0
    index = zeroed["quality_mean"] * zeroed["cost_mean"]
    positive = index[index > 0].min()

    fig = plot_quality_cost_matrix(zeroed)
    try:
        drawn = set()
        for ax in fig.get_axes():
            for collection in ax.collections:
                for _, y in collection.get_offsets():
                    if y < positive:
                        drawn.add(round(float(y), 12))
        assert drawn == {round(positive * 0.5, 12)}
    finally:
        plt.close(fig)


def test_save_figure_writes_file(synthetic_front, tmp_path):
    fig = plot_quality_cost_matrix(synthetic_front)
    save_figure(fig, tmp_path / "plots", "quality_cost_matrix_random", "png")
    assert (tmp_path / "plots" / "quality_cost_matrix_random.png").is_file()


# ---------------------------------------------------------------------------
# The ratio / percentile variant of the same grid
# ---------------------------------------------------------------------------

def test_ratio_matrix_layout_and_scales(synthetic_front):
    """Same 3×3 facets as the quality/cost matrix, but a linear percentage x-axis."""
    fig = plot_ratio_percentile_matrix(synthetic_front)
    try:
        axes = fig.get_axes()
        assert len(axes) == 9
        assert all(ax.get_xscale() == "linear" and ax.get_yscale() == "log"
                   for ax in axes)
        assert all(ax.get_xlim() == (0.0, 1.0) for ax in axes)
        assert [axes[i].get_title(loc="center") for i in range(3)] == [
            "agent = low", "agent = medium", "agent = high",
        ]
    finally:
        plt.close(fig)


def test_ratio_matrix_x_is_the_within_front_percentile(synthetic_front):
    """Each front has two points, so its percentiles are exactly {0.5, 1.0}."""
    fig = plot_ratio_percentile_matrix(synthetic_front)
    try:
        drawn = {
            round(float(x), 12)
            for ax in fig.get_axes()
            for collection in ax.collections
            for x, _ in collection.get_offsets()
        }
        assert drawn == {0.5, 1.0}
    finally:
        plt.close(fig)


def test_ratio_matrix_y_is_the_quality_cost_product(synthetic_front):
    """y is the index — the product, not the ratio — of the chosen cost column."""
    expected = {
        round(float(q * c), 9)
        for q, c in zip(synthetic_front["quality_mean"],
                        synthetic_front["cost_agent_mean"])
    }
    fig = plot_ratio_percentile_matrix(synthetic_front, cost_field="cost_agent_mean")
    try:
        drawn = {
            round(float(y), 9)
            for ax in fig.get_axes()
            for collection in ax.collections
            for _, y in collection.get_offsets()
        }
        assert drawn == expected
    finally:
        plt.close(fig)


def test_ratio_matrix_rejects_duplicate_group(synthetic_front):
    with pytest.raises(ValueError, match="only one visual channel"):
        plot_ratio_percentile_matrix(synthetic_front, x_group="debt", color_group="debt")


def test_ratio_matrix_rejects_unknown_cost_field(synthetic_front):
    with pytest.raises(KeyError, match="cost_nonsense"):
        plot_ratio_percentile_matrix(synthetic_front, cost_field="cost_nonsense")


def test_ratio_matrix_tolerates_missing_tier(synthetic_front):
    partial = synthetic_front[synthetic_front["tier_human"] != "low"]
    fig = plot_ratio_percentile_matrix(partial)
    try:
        assert len(fig.get_axes()) == 6
    finally:
        plt.close(fig)


def test_ratio_matrix_save_figure_writes_file(synthetic_front, tmp_path):
    fig = plot_ratio_percentile_matrix(synthetic_front)
    save_figure(fig, tmp_path / "plots", "ratio_percentile_matrix_random", "png")
    assert (tmp_path / "plots" / "ratio_percentile_matrix_random.png").is_file()


# ---------------------------------------------------------------------------
# Team composition figures (paper §5)
# ---------------------------------------------------------------------------

def _composition_row(tiers: dict[str, str], A_dev, A_qa, **kwargs) -> dict:
    """A frontier row carrying the composition columns the §5 team plots read."""
    row = _row(tiers, kwargs.pop("quality", 10.0), kwargs.pop("cost", 100.0))
    row.update(
        N=2, R=4, F_freeze=1,
        n_sup_dev=sum(A_dev[i] for i in (0, 1, 2, 3, 6, 7)),
        n_sup_qa=sum(A_qa[i] for i in (0, 1, 2, 3, 6, 7)),
    )
    row.update({f"a_dev_{k}": A_dev[k - 1] for k in range(1, 9)})
    row.update({f"a_qa_{k}": A_qa[k - 1] for k in range(1, 9)})
    row.update(kwargs)
    return row


@pytest.fixture
def composition_front() -> pd.DataFrame:
    """Two archetypes per tier combination, for a single strategy."""
    lean = (1, 0, 0, 0, 6, 4, 0, 0)
    rich = (2, 1, 1, 1, 8, 6, 2, 1)
    qa_lean = (0, 0, 1, 0, 5, 3, 0, 0)
    qa_rich = (0, 0, 2, 1, 7, 5, 1, 1)

    rows = []
    for combo in itertools.product(SCENARIO_TIERS, repeat=len(SCENARIO_GROUPS)):
        tiers = dict(zip(SCENARIO_GROUPS, combo))
        rows.append(_composition_row(tiers, lean, qa_lean, quality=50.0, cost=100.0))
        rows.append(_composition_row(tiers, rich, qa_rich, quality=5.0, cost=900.0))
    return pd.DataFrame(rows)


@pytest.mark.parametrize("phase,expected_blocks", [("dev", 1), ("qa", 1), ("both", 2)])
def test_archetypes_layout_per_phase(composition_front, phase, expected_blocks):
    fig = plot_composition_archetypes(composition_front, phase=phase)
    try:
        # Four panels per phase block: shares, background, quality/cost, prevalence.
        assert len(fig.get_axes()) == 4 * expected_blocks
        share_ax = fig.get_axes()[0]
        assert share_ax.get_xlim() == (0.0, 1.0)
        # One row per archetype; the synthetic front has exactly two.
        assert len(share_ax.get_yticklabels()) == 2
        assert share_ax.get_ylabel() == f"{'dev' if phase != 'qa' else 'qa'} phase"
    finally:
        plt.close(fig)


def test_archetype_rows_are_quality_ordered(composition_front):
    """Best quality on top — the ladder the figure is built to show."""
    fig = plot_composition_archetypes(composition_front, phase="dev")
    try:
        labels = [t.get_text() for t in fig.get_axes()[0].get_yticklabels()]
        assert labels[0].startswith("Feature+Design")     # the rich, low-quality recipe
        assert labels[-1].replace("\n", "+") == "Feature+Review+Security"
    finally:
        plt.close(fig)


def test_archetypes_reject_mixed_strategies(composition_front):
    """One row per archetype would silently double if two strategies were mixed."""
    mixed = pd.concat([
        composition_front,
        composition_front.assign(strategy="evolutionary"),
    ], ignore_index=True)
    with pytest.raises(ValueError, match="single-strategy"):
        plot_composition_archetypes(mixed)
    with pytest.raises(ValueError, match="single-strategy"):
        plot_composition_tier_heatmap(mixed)


def test_unknown_phase_rejected(composition_front):
    with pytest.raises(ValueError, match="unknown phase"):
        plot_composition_archetypes(composition_front, phase="development")


def test_drift_facets_every_strategy(composition_front):
    mixed = pd.concat([
        composition_front,
        composition_front.assign(strategy="evolutionary"),
    ], ignore_index=True)
    fig = plot_composition_drift(mixed, phase="dev", bins=2)
    try:
        # Three panels per (phase, strategy) row; the drift figure faceting is its own.
        assert len(fig.get_axes()) == 3 * 2
        assert [t.get_text() for t in fig.get_axes()[0].get_xticklabels()] == ["1/2", "2/2"]
        assert fig.get_axes()[0].get_ylabel().startswith("dev / evolutionary")
    finally:
        plt.close(fig)


def test_tier_heatmap_has_one_row_per_tier(composition_front):
    fig = plot_composition_tier_heatmap(composition_front, phase="dev")
    try:
        axes = fig.get_axes()
        assert len(axes) == 3            # shares | background | controls
        rows = [t.get_text() for t in axes[0].get_yticklabels()]
        assert len(rows) == len(SCENARIO_GROUPS) * len(SCENARIO_TIERS)
        assert rows[0] == "human · low"
        assert rows[-1] == "bugimpact · high"
        # Column blocks: 6 supervised roles, 2 background roles, the control variables.
        assert len(axes[0].get_xticklabels()) == 6
        assert len(axes[1].get_xticklabels()) == 2
    finally:
        plt.close(fig)


def test_missing_role_column_rejected(composition_front):
    with pytest.raises(KeyError, match="a_dev_3"):
        plot_composition_archetypes(composition_front.drop(columns="a_dev_3"))


def test_composition_figures_are_saveable(composition_front, tmp_path):
    fig = plot_composition_archetypes(composition_front, phase="both")
    save_figure(fig, tmp_path / "plots", "composition_archetypes_both_random", "png")
    assert (tmp_path / "plots" / "composition_archetypes_both_random.png").is_file()
