"""Figures for a consolidated campaign.

Each `plot_*` takes a DataFrame derived from the tables of :mod:`src.campaign` and
:mod:`src.constellation` and returns a matplotlib Figure; the caller saves it.
`visualize_campaign` and `visualize_team_summary` drive them and write to a directory.

The three optima matrices share `_draw_matrix`, the five-channel tier grid, and differ
only in the pair of columns they put on the axes.  All but the ratio figure take an
optional `baseline` frame of zero-agent optima, drawn as an overlay in black.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe to override before pyplot import
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import (
    FuncFormatter,
    LogLocator,
    MaxNLocator,
    NullFormatter,
    PercentFormatter,
)

from .config import (
    AGENT_ROLE_ABBREV,
    AUTONOMOUS_ROLE_INDICES,
    EXTREME_TIERS,
    SCENARIO_GROUPS,
    SUPERVISED_ROLE_INDICES,
    UNUSED_TIER,
    tier_column,
    tiers_present,
)
from .campaign import baseline_rows, require_columns, search_rows
from .constellation import (
    AGENT_COST_PARAMETERS,
    BILL_PARAMETERS,
    DEV_PARAMETERS,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

#: The campaign matrices' titles.  `visualize_campaign` appends the scenario counts and
#: each plot function falls back to the bare form, so both are defined here.
HUMAN_AGENT_TITLE = "Developer vs. agent cost at $c^\\star$"
QUALITY_DEVELOPMENT_TITLE = ("Escaped-bug vs. development cost, developers + agents, "
                             "at $c^\\star$")
AGENT_HUMAN_RATIO_TITLE = "Agent-to-developer cost ratio at $c^\\star$"
ELITE_TURNOVER_TITLE = "Elite turnover and $c^\\star$ drop per generation"

#: Which scenario group drives which visual channel of the three optima matrices. 
# `project` is on the marker channel, the
#: only per-point channel whose fourth level reads as a peer rather than as "more", and
#: `agent` is on a facet axis, where its fourth tier costs one column rather than a fifth
#: marker shape.  `_TIER_STYLES` gives `veryhigh` and `pessimistic` one shared slot, which
#: works only while no single group holds both.
MATRIX_CHANNELS: dict[str, str] = {
    "x_group": "agent",
    "y_group": "human",
    "color_group": "debt",
    "marker_group": "project",
    "size_group": "bugimpact",
}

# Shared palette
_KEEP_COLOR  = cm.tab10(0)     # blue — elite turnover
_CHANGE_COLOR = cm.tab10(2)    # green — c*'s relative drop per generation
_STYLE       = "seaborn-v0_8-whitegrid"


# ---------------------------------------------------------------------------
# Search convergence
# ---------------------------------------------------------------------------

def _drop_label(values: np.ndarray) -> str:
    """Legend label for the drop curve, saying so when the log axis cannot draw it.

    A drop that is zero throughout has nothing for a log axis to place, and matplotlib
    omits the curve silently, which would leave an unexplained legend entry.
    """
    label = "mean drop in $c^\\star$ cost"
    positive = np.asarray(values, dtype=float) > 0.0
    if not positive.any():
        return f"{label} (zero throughout — not drawable on a log axis)"
    return label


def _incumbent_cost_drop(
    turnover: pd.DataFrame, generations: np.ndarray
) -> np.ndarray | None:
    """`c*`'s mean drop since the previous generation, relative and unsigned.

    `|best_cost(g) / best_cost(g-1) - 1|` per search, then the mean over searches, so that it
    is a rate like the turnover.  Generation 1 has no predecessor and no value.  Returns None
    when the column carries no usable value.
    """
    keys = ["scenario_id", "mode"]
    frame = (turnover[[*keys, "generation", "best_cost"]]
             .dropna(subset=["best_cost"])
             .sort_values("generation"))
    if frame.empty:
        return None
    costs = frame["best_cost"]
    previous = costs.groupby([frame[k] for k in keys]).shift(1)
    previous = previous.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        drop = np.where(previous > 0.0,
                        np.abs(costs.to_numpy(dtype=float) / previous - 1.0),
                        np.nan)
    mean = (frame.assign(_drop=drop).groupby("generation")["_drop"].mean()
            .reindex(generations).to_numpy(dtype=float))
    return None if np.all(np.isnan(mean)) else mean


def plot_elite_turnover(
    turnover: pd.DataFrame,
    title: str = "",
) -> Figure:
    """Elite turnover and the incumbent's cost drop, against the generation index.

    Takes the long-format ``elite_turnover.csv`` - one row per (scenario, mode, generation)
    - and draws three curves aggregated over searches: the turnover in blue on the left axis
    as a mean (solid) and a median (dotted), and the mean relative drop in `c*`'s cost in
    green on a logarithmic right-hand axis, which it needs because it runs one to two decades
    lower.  Carries no tier channels.
    """
    require_columns(turnover, ("scenario_id", "mode", "generation", "tau", "best_cost"),
                    "turnover table")
    if turnover.empty:
        raise ValueError("plot_elite_turnover needs at least one generation")

    grouped = turnover.groupby("generation")["tau"]
    generations = np.array(sorted(turnover["generation"].unique()), dtype=float)
    median = grouped.median().reindex(generations).to_numpy(dtype=float)
    mean = grouped.mean().reindex(generations).to_numpy(dtype=float)
    n_searches = turnover.groupby(["scenario_id", "mode"]).ngroups

    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        if title:
            ax.set_title(title, fontsize=11)

        ax.plot(generations, mean, color=_KEEP_COLOR, linewidth=1.8,
                label=r"mean $\tau_g$")
        ax.plot(generations, median, color=_KEEP_COLOR, linewidth=1.5, linestyle=":",
                label=r"median $\tau_g$")

        drop = _incumbent_cost_drop(turnover, generations)
        ax2 = None
        if drop is not None:
            ax2 = ax.twinx()
            ax2.grid(False)
            ax2.set_yscale("log")
            ax2.plot(generations, drop, color=_CHANGE_COLOR, linewidth=1.8,
                     label=_drop_label(drop))
            # An incumbent that never moves gives a drop of 0 everywhere, which a log
            # axis cannot autoscale over; the label already says the curve is missing.
            if not np.nanmax(drop) > 0.0:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", "Data has no positive values")
                    ax2.set_ylim(1e-4, 1.0)

            # Only two decades wide, so the 2/5 subdivisions are labelled as well.
            _decade_axis(ax2.yaxis, _percent_tick, label_minor=True)
            ax2.set_ylabel(r"relative drop in $c^\star$ cost per generation (log)",
                           fontsize=9, color=_CHANGE_COLOR)
            ax2.tick_params(axis="y", which="both", labelsize=8, colors=_CHANGE_COLOR)

        # The generation index is an integer; a shared locator would offer halves.
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("generation $g$", fontsize=9)
        ax.set_ylabel(r"elite turnover $\tau_g$", fontsize=9, color=_KEEP_COLOR)
        ax.tick_params(axis="y", labelsize=8, colors=_KEEP_COLOR)
        ax.set_ylim(bottom=0.0)
        ax.set_xlim(generations.min(), generations.max())
        # One legend for both axes; it goes on the twin, which `twinx` draws on top.
        handles, labels = ax.get_legend_handles_labels()
        legend_ax = ax if ax2 is None else ax2
        if ax2 is not None:
            extra = ax2.get_legend_handles_labels()
            handles += extra[0]
            labels += extra[1]
        legend_ax.legend(handles, labels, fontsize=8, loc="best")
        ax.annotate(f"{n_searches} search(es)", xy=(0.99, 0.02),
                    xycoords="axes fraction", ha="right", va="bottom", fontsize=7,
                    color="#666666")
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# The campaign in one figure: all five scenario groups at once
# ---------------------------------------------------------------------------

#: One visual style per tier, in mild -> extreme order so the legends read that way.
#: `veryhigh` and `pessimistic` share the fourth slot. No scenario group holds both, so
#: no channel ever has to draw them side by side.
class _TierStyle(NamedTuple):
    color: tuple
    marker: str
    filled: bool
    area: float                 # pt^2


_TIER_STYLES: dict[str, _TierStyle] = {
    "low":         _TierStyle(cm.tab10(0), "x", False, 8.0),
    "medium":      _TierStyle(cm.tab10(1), "o", False, 24.0),
    "high":        _TierStyle(cm.tab10(3), "^", True, 66.0),
    "veryhigh":    _TierStyle(cm.tab10(4), "s", True, 160.0),
    "pessimistic": _TierStyle(cm.tab10(4), "s", True, 160.0),
}

#: The baseline reuses the tier shape and size wherever the channel's group is one it
#: varies, and falls back to the star below on a group it holds at UNUSED_TIER.  Its
#: colour is always black, which is what identifies it.
_BASELINE_COLOR = "black"
_BASELINE_MARKER = "*"
_BASELINE_SIZE = 90.0
#: Wrapped, because on one line this is the longest colour-legend entry and overruns the
#: fixed-width legend column.
_BASELINE_LABEL = "human-only\nbaseline"


def _percent_tick(value: float, _pos: int | None = None) -> str:
    """Tick label for a fraction on a log axis: ``10%``, ``1%``, ``0.1%``, …

    Keeps as many decimals as the tick needs; one fixed number of them cannot serve an
    axis spanning several decades.
    """
    pct = 100.0 * value
    if pct <= 0:
        return "0"
    return f"{pct:.{max(0, int(np.ceil(-np.log10(pct))))}f}%"


def _dollar_tick(value: float, _pos: int | None = None) -> str:
    """Tick label for a dollar amount: ``$5``, ``$20``, ``$1.5k``, ``$1M``, …"""
    if value >= 1e6:
        return f"\\${value / 1e6:g}M"
    if value >= 1000:
        return f"\\${value / 1000:g}k"
    return f"\\${value:g}"


def _ratio_tick(value: float, _pos: int | None = None) -> str:
    """Tick label for a ratio on a log axis: ``0.001``, ``0.1``, ``1``, ``100``.

    Matplotlib's default writes the sub-unit decades in scientific mathtext instead.
    """
    return f"{value:g}"


def _log_floor(df: pd.DataFrame, column: str) -> float | None:
    """Where non-positive values are drawn on a log axis: half the smallest positive.

    One number for the whole table, so the same scenario lands at the same coordinate in
    every panel of a faceted figure.  `None` when nothing needs clipping.
    """
    x = df[column].to_numpy(dtype=float)
    positive = x[x > 0]
    if positive.size and (x <= 0).any():
        return float(positive.min()) * 0.5
    return None


def _clipped_xy(
    df: pd.DataFrame,
    x_col: str, y_col: str,
    x_floor: float | None, y_floor: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """The two plot columns as arrays, non-positive values moved onto their log floor."""
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    if x_floor is not None:
        x = np.where(x > 0, x, x_floor)
    if y_floor is not None:
        y = np.where(y > 0, y, y_floor)
    return x, y


def _padded_log_range(positive: np.ndarray, what: str) -> tuple[float, float]:
    """A log-axis range covering `positive`, padded by 5 % of the span in log space.

    Raises when nothing positive is left to scale.
    """
    positive = positive[positive > 0]
    if positive.size == 0:
        raise ValueError(f"no positive values in {what}; cannot scale a log axis")
    lo, hi = float(positive.min()), float(positive.max())
    pad = 0.05 * max(np.log10(hi / lo), 0.4)
    return 10.0 ** (np.log10(lo) - pad), 10.0 ** (np.log10(hi) + pad)


def _common_log_limits(
    df: pd.DataFrame,
    columns: tuple[str, ...],
    floors: tuple[float | None, ...],
) -> tuple[float, float]:
    """One padded (min, max) covering every column of `columns`, for a log axis.

    The `floors` of :func:`_log_floor` are included, because a clipped zero is drawn at its
    floor and would otherwise fall outside the limits.
    """
    parts = [df[c].to_numpy(dtype=float) for c in columns]
    parts.append(np.array([f for f in floors if f is not None], dtype=float))
    return _padded_log_range(np.concatenate(parts), str(list(columns)))


def _marker_kwargs(marker_tier: str, color, area: float) -> dict:
    """Scatter keywords drawing `marker_tier`'s shape in `color` at `area` pt².

    The stroked markers need a line width that grows with the marker; the filled ones must
    have none, or the edge swallows them at the smallest size tier.
    """
    marker = _TIER_STYLES[marker_tier].marker
    if _TIER_STYLES[marker_tier].filled:
        return {"marker": marker, "color": color, "linewidths": 0}
    width = max(0.35, 0.16 * float(np.sqrt(area)))
    if marker == "o":
        return {"marker": marker, "facecolors": "none", "edgecolors": color,
                "linewidths": width}
    return {"marker": marker, "color": color, "linewidths": width}


def _baseline_marker_kwargs(marker_tier: str | None, area: float) -> dict:
    """Scatter keywords for a baseline point of `marker_tier`'s shape at `area` pt².

    Black on every layout (see `_BASELINE_COLOR`).  `marker_tier` is None when the shape
    channel carries a group the baseline does not vary, and the fixed star stands in.
    """
    if marker_tier is None:
        return {"marker": _BASELINE_MARKER, "color": _BASELINE_COLOR, "linewidths": 0}
    return _marker_kwargs(marker_tier, _BASELINE_COLOR, area)


#: Style of the reference lines drawn on the cost planes and the ratio figure.
_REFERENCE_LINE = {"color": "0.6", "linewidth": 0.8, "linestyle": (0, (4, 3)),
                   "zorder": 0}


def _decade_axis(axis, formatter, label_minor: bool = False) -> None:
    """Put decade ticks on a log axis, formatted by `formatter`.

    The 2 and 5 subdivisions are always drawn; `label_minor` also labels them, which a
    range only two decades wide needs to carry more than two labels.
    """
    axis.set_major_locator(LogLocator(base=10.0))
    axis.set_minor_locator(LogLocator(base=10.0, subs=(2.0, 5.0)))
    axis.set_major_formatter(FuncFormatter(formatter))
    axis.set_minor_formatter(FuncFormatter(formatter) if label_minor
                             else NullFormatter())


def _density_style(n: int) -> tuple[float, float]:
    """(marker-size multiplier, alpha) for a point cloud of `n` points."""
    if n <= 2_000:
        return 1.0, 0.55
    if n <= 15_000:
        return 0.6, 0.40
    return 0.4, 0.30


#: Reserved to the right of the panels for the three channel legends, and the vertical
#: gap between two of them. In inches rather than figure fractions, because the legends
#: are the same size whatever the grid.
_LEGEND_WIDTH_IN = 2.1
_LEGEND_GAP_IN = 0.25

#: Font sizes (points) for the tier grid. The figure is 17-20 inches wide and is read
#: reduced to a text column's width, so these are sized against the printed result rather
#: than the canvas. Keep the panel geometry fixed when changing them.
_MATRIX_FONT: dict[str, int] = {
    "suptitle": 17,
    "column":   15,   # the facet tier heading, on the top row only
    "count":    12,   # the per-panel `n = ...`
    "axis":     14,   # the two axis labels
    "tick":     12,
    "legend":   13,
}


def _figure_bbox(fig: Figure, artist) -> matplotlib.transforms.Bbox:
    """`artist`'s bounding box in figure coordinates.

    A legend has no extent until a renderer has laid it out, and nothing has drawn the figure
    yet. The Agg canvas pinned at import time supplies one without a draw pass.
    """
    renderer = fig.canvas.get_renderer()
    return artist.get_window_extent(renderer).transformed(fig.transFigure.inverted())


def _require_tier_columns(df: pd.DataFrame) -> None:
    """Raise unless the optima table carries a tier column for every matrix channel."""
    require_columns(df, [tier_column(g) for g in MATRIX_CHANNELS.values()],
                    "optima table")


def _with_baseline(
    df: pd.DataFrame,
    baseline: pd.DataFrame | None,
) -> pd.DataFrame:
    """`df` and `baseline` stacked, for computing axis floors and ranges over both.

    A range derived from `df` alone would clip the baseline off the figure: its agent cost is
    exactly 0, which no log axis can place.
    """
    if baseline is None or baseline.empty:
        return df
    return pd.concat([df, baseline], ignore_index=True)


def _baseline_cell(
    baseline: pd.DataFrame,
    x_group: str, x_tier: str,
    y_group: str, y_tier: str,
) -> pd.DataFrame:
    """The baseline rows belonging in the panel at (`x_tier`, `y_tier`).

    A baseline row carries `UNUSED_TIER` for every group it does not vary:
    the row belongs in *every* tier of it, so a facet on one of them
    repeats the same reference points down each column.
    """
    cell = baseline
    for group, tier in ((x_group, x_tier), (y_group, y_tier)):
        column = tier_column(group)
        if column not in cell.columns:
            continue
        cell = cell[(cell[column] == tier) | (cell[column] == UNUSED_TIER)]
    return cell


def _baseline_style_groups(
    cell: pd.DataFrame,
    marker_group: str,
    size_group: str,
) -> list[tuple[str | None, str | None, pd.DataFrame]]:
    """`cell` split into (marker tier, size tier, rows) blocks, largest size first.

    A group held at `UNUSED_TIER` yields `None` for its channel, and the caller draws the
    fallback style instead.
    """
    def tiers(group: str) -> pd.Series:
        column = tier_column(group)
        if column in cell.columns:
            return cell[column]
        return pd.Series(UNUSED_TIER, index=cell.index)

    blocks = [
        (None if marker_tier == UNUSED_TIER else marker_tier,
         None if size_tier == UNUSED_TIER else size_tier,
         sub)
        for (marker_tier, size_tier), sub in
        cell.groupby([tiers(marker_group), tiers(size_group)], sort=False)
    ]
    blocks.sort(key=lambda b: _TIER_STYLES[b[1]].area if b[1] else _BASELINE_SIZE, reverse=True)
    return blocks


def _draw_matrix(
    df: pd.DataFrame,
    x_group: str,
    y_group: str,
    size_group: str,
    marker_group: str,
    color_group: str,
    x_col: str,
    y_col: str,
    xlabel: str,
    ylabel: str,
    title: str,
    xscale: str = "log",
    x_floor: float | None = None,
    y_floor: float | None = None,
    shared_limits: tuple[float, float] | None = None,
    baseline: pd.DataFrame | None = None,
) -> Figure:
    """The five-channel tier grid, for any pair of columns.

    The facet layout, the colour/shape/size encoding of the three non-facet groups and the
    three proxy legends are implemented here.
    Callers pass `MATRIX_CHANNELS` and the two plot columns.
    `x_floor` and `y_floor` clip non-positive values onto a log axis (`_log_floor`).
    `shared_limits` puts one range on both axes and squares the panels. Only valid where the
    two columns share a unit, never for a share or a rank. 
    `baseline` holds the zero-agent optima, matched to a panel by the facet groups
    alone (`_baseline_cell`).
    """
    columns = tiers_present(df[tier_column(x_group)])
    rows = tiers_present(df[tier_column(y_group)])
    if not columns or not rows:
        raise ValueError("no rows for any tier of the faceting groups")

    # Per channel, only the tiers the data has: the groups differ in tier count, so a
    # fixed list would print dead legend entries.
    size_tiers = tiers_present(df[tier_column(size_group)])
    marker_tiers = tiers_present(df[tier_column(marker_group)])
    color_tiers = tiers_present(df[tier_column(color_group)])

    with plt.style.context(_STYLE):
        # A panel carrying one range on both axes is square and needs its width back.
        panel_w = 4.2 if shared_limits is not None else 5.0
        fig, axes = plt.subplots(
            len(rows), len(columns),
            figsize=(panel_w * len(columns), 4.2 * len(rows)),
            sharex=True, sharey=True,
            squeeze=False,
        )
        fig.suptitle(title, fontsize=_MATRIX_FONT["suptitle"], y=1.0)

        channel_columns = [tier_column(g) for g in (size_group, marker_group, color_group)]
        for r, y_tier in enumerate(rows):
            y_sub = df[df[tier_column(y_group)] == y_tier]
            for c, x_tier in enumerate(columns):
                ax = axes[r][c]
                cell = y_sub[y_sub[tier_column(x_group)] == x_tier]
                scale, alpha = _density_style(len(cell))
                # Grouped once, then visited in tier order rather than the groupby's
                # own: the largest markers go down first, or the small ones end up buried
                # under them.
                blocks = dict(tuple(cell.groupby(channel_columns, sort=False)))
                for size_tier in reversed(size_tiers):
                    for marker_tier in marker_tiers:
                        for color_tier in color_tiers:
                            sub = blocks.get((size_tier, marker_tier, color_tier))
                            if sub is None or sub.empty:
                                continue
                            area = _TIER_STYLES[size_tier].area * scale
                            ax.scatter(
                                *_clipped_xy(sub, x_col, y_col, x_floor, y_floor),
                                s=area, alpha=alpha,
                                **_marker_kwargs(marker_tier, _TIER_STYLES[color_tier].color,
                                                 area),
                            )
                if baseline is not None and not baseline.empty:
                    b_cell = _baseline_cell(baseline, x_group, x_tier, y_group, y_tier)
                    for b_marker, b_size, b_sub in _baseline_style_groups(
                            b_cell, marker_group, size_group):
                        b_area = (_TIER_STYLES[b_size].area if b_size else _BASELINE_SIZE) * scale
                        # At the panel's alpha like every other point, so the overlay does
                        # not mask the optima it is the reference for.
                        ax.scatter(*_clipped_xy(b_sub, x_col, y_col, x_floor, y_floor),
                                   s=b_area, zorder=5, alpha=alpha,
                                   label=_BASELINE_LABEL,
                                   **_baseline_marker_kwargs(b_marker, b_area))
                ax.set_xscale(xscale)
                ax.set_yscale("log")
                if shared_limits is not None:
                    ax.set_xlim(*shared_limits)
                    ax.set_ylim(*shared_limits)
                    # On a log axis the aspect is read in decades, so this is what makes
                    # the two equal ranges equally long on the page.
                    ax.set_aspect("equal", adjustable="box")
                ax.tick_params(axis="both", which="both",
                               labelsize=_MATRIX_FONT["tick"])
                # Inside the panel rather than on the title line, which on the top row
                # already carries the centred facet heading.  Stacking two titles is not
                # an option: matplotlib repositions every title on each draw.
                ax.text(0.985, 0.015, f"n = {len(cell)}", transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=_MATRIX_FONT["count"],
                        color="0.35", zorder=6)
                if r == 0:
                    ax.set_title(f"{x_group} = {x_tier}",
                                 fontsize=_MATRIX_FONT["column"], loc="center")
                if r == len(rows) - 1:
                    ax.set_xlabel(xlabel, fontsize=_MATRIX_FONT["axis"])
                if c == 0:
                    ax.set_ylabel(f"{y_group} = {y_tier}\n{ylabel}",
                                  fontsize=_MATRIX_FONT["axis"])

        # Three proxy legends, one per non-facet channel.  Line2D takes a marker *width*
        # in points where scatter takes an area in pt², hence the sqrt.
        legend_specs = [
            (color_group, [
                Line2D([], [], linestyle="none", marker="o", markersize=8,
                       color=_TIER_STYLES[t].color, label=t) for t in color_tiers
            ]),
            (marker_group, [
                Line2D([], [], linestyle="none", marker=_TIER_STYLES[t].marker, markersize=9,
                       color="0.35", markeredgecolor="0.35", label=t,
                       markerfacecolor="0.35" if _TIER_STYLES[t].filled else "none")
                for t in marker_tiers
            ]),
            (size_group, [
                Line2D([], [], linestyle="none", marker="o", color="0.35", label=t,
                       markersize=float(np.sqrt(_TIER_STYLES[t].area)) * 1.5)
                for t in size_tiers
            ]),
        ]
        if baseline is not None and not baseline.empty:
            # In the colour legend rather than a block of its own: the baseline shares
            # the shape and size channels with the search optima, so only its colour
            # separates it from them.
            legend_specs[0][1].append(
                Line2D([], [], linestyle="none", marker="o", markersize=8,
                       color=_BASELINE_COLOR, label=_BASELINE_LABEL)
            )
        # A fixed width in inches. An extra facet column widens
        # the figure, and that inch must go to the panels rather than to the legends.
        right = 1.0 - _LEGEND_WIDTH_IN / fig.get_figwidth()
        fig.tight_layout(rect=(0.0, 0.0, right, 1.0))
        # Stacked by their measured heights. An entry is as tall as its marker and the
        # tier counts differ per channel, so a per-entry estimate would not place them.
        anchor_y = 0.97
        channel_names = ("colour", "shape", "size")
        for i, (group, handles) in enumerate(legend_specs):
            title = f"{group} ({channel_names[i]})"
            legend = fig.legend(
                handles=handles, loc="upper left", frameon=True,
                fontsize=_MATRIX_FONT["legend"], title=title,
                title_fontsize=_MATRIX_FONT["legend"], labelspacing=0.9,
                bbox_to_anchor=(right + 0.15 * _LEGEND_WIDTH_IN / fig.get_figwidth(),
                                anchor_y),
            )
            anchor_y = _figure_bbox(fig, legend).y0 - _LEGEND_GAP_IN / fig.get_figheight()
    return fig


def _draw_cost_plane(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    xlabel: str,
    ylabel: str,
    title: str,
    baseline: pd.DataFrame | None,
) -> Figure:
    """A tier grid whose two axes are both dollars per feature.

    Both columns share one log range at an equal aspect, with the 1:1 diagonal drawn.  The
    dollar signs in the labels must reach here escaped, or mathtext sees three delimiters in
    one label and renders it literally.
    """
    scaled = _with_baseline(df, baseline)
    x_floor = _log_floor(scaled, x_col)
    y_floor = _log_floor(scaled, y_col)
    fig = _draw_matrix(
        df, **MATRIX_CHANNELS,
        x_col=x_col, y_col=y_col, xlabel=xlabel, ylabel=ylabel, title=title,
        x_floor=x_floor, y_floor=y_floor,
        shared_limits=_common_log_limits(scaled, (x_col, y_col), (x_floor, y_floor)),
        baseline=baseline,
    )
    for ax in fig.get_axes():
        for axis in (ax.xaxis, ax.yaxis):
            _decade_axis(axis, _dollar_tick)
        ax.plot(ax.get_xlim(), ax.get_xlim(), **_REFERENCE_LINE)
    return fig


def plot_human_agent_matrix(
    df: pd.DataFrame,
    title: str = "",
    baseline: pd.DataFrame | None = None,
) -> Figure:
    """The two team cost drivers of each optimum against each other, on a tier grid.

    x is what the agents billed, y what the developers cost, both dollars per feature.
    """
    _require_tier_columns(df)
    require_columns(df, ("cost_agent_mean", "cost_human_mean"), "optima table")
    return _draw_cost_plane(
        df,
        x_col="cost_agent_mean", y_col="cost_human_mean",
        xlabel="↓  agent cost   (\\$ / feature)",
        ylabel="↓  developer cost   (\\$ / feature)",
        title=title or HUMAN_AGENT_TITLE,
        baseline=baseline,
    )


def plot_quality_development_matrix(
    df: pd.DataFrame,
    title: str = "",
    baseline: pd.DataFrame | None = None,
) -> Figure:
    """What the escaped bugs cost against what the team cost, on a tier grid.

    The two drivers :func:`plot_human_agent_matrix` separates are added back together on x, on
    a copy, since a raw ``optima.csv`` does not carry the sum.
    """
    _require_tier_columns(df)
    require_columns(df, ("cost_quality_mean", "cost_human_mean", "cost_agent_mean"),
                    "optima table")

    def add_team_cost(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.assign(
            cost_team_mean=frame["cost_human_mean"] + frame["cost_agent_mean"])

    df = add_team_cost(df)
    if baseline is not None and not baseline.empty:
        baseline = add_team_cost(baseline)
    return _draw_cost_plane(
        df,
        x_col="cost_team_mean", y_col="cost_quality_mean",
        xlabel="↓  development cost   (\\$ / feature)",
        ylabel="↓  escaped-bug cost   (\\$ / feature)",
        title=title or QUALITY_DEVELOPMENT_TITLE,
        baseline=baseline,
    )


def plot_agent_human_ratio_matrix(
    df: pd.DataFrame,
    title: str = "",
) -> Figure:
    """How lopsided the agent/developer split is at each optimum, against its percentile.

    y is `cost_agent_mean / cost_human_mean` on a log axis, with a reference line at 1.  x is
    that ratio's percentile rank within `df`, so the panels share one empirical distribution
    and each is a fragment of it.
    """
    _require_tier_columns(df)
    require_columns(df, ("cost_agent_mean", "cost_human_mean"), "optima table")

    ratio = df["cost_agent_mean"] / df["cost_human_mean"]
    df = df.assign(
        cost_agent_human_ratio=ratio,
        # Ties share their mean rank, so equal ratios land on one x rather than on an
        # arbitrary order of the rows.
        cost_agent_human_ratio_pct=ratio.rank(method="average", pct=True),
    )
    fig = _draw_matrix(
        df, **MATRIX_CHANNELS,
        x_col="cost_agent_human_ratio_pct", y_col="cost_agent_human_ratio",
        # Short enough not to run into the next panel's at `_MATRIX_FONT["axis"]`.
        xlabel="percentile rank of the ratio",
        # No "↓": unlike a cost, neither direction of the ratio is the better one.
        ylabel="agent cost / developer cost",
        title=title or AGENT_HUMAN_RATIO_TITLE,
        # x is a rank in [0, 1] — a log axis would only stretch the first percentile.
        xscale="linear",
        y_floor=_log_floor(df, "cost_agent_human_ratio"),
    )
    for ax in fig.get_axes():
        # A rank axis is uniform, so it takes the plain percent formatter rather than
        # the decade-aware `_percent_tick` a log axis of fractions needs.
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        _decade_axis(ax.yaxis, _ratio_tick)
        ax.axhline(1.0, **_REFERENCE_LINE)      # parity
    return fig


# ---------------------------------------------------------------------------
# The summary of team structure
# ---------------------------------------------------------------------------
#
# Two figures: what the optimal team is made of, and which scenario setting moves it.
# Both use `optima.csv`, because their shares and rates have to be formed within a 
# scenario before being averaged — a ratio of means is not the mean of the ratios.

#: The three scenario groups the composition figures lay out at once: blocks of bars, the
#: ladder repeated inside each block, and one panel per tier.
COMPOSITION_BLOCK_GROUP = "project"
COMPOSITION_BAR_GROUP = "bugimpact"
COMPOSITION_PANEL_GROUP = "agent"

#: One colour per agent role, k=1..8, for the composition bars.  Qualitative rather than
#: sequential, since the roles are categories.
_ROLE_COLORS = tuple(cm.tab10(i) for i in range(8))

#: Segments smaller than this are left unlabelled; their labels would overprint.
_SHARE_LABEL_MIN = 0.06

#: The developers' slice in the `team` weighting.  Off the eight-role palette, so that
#: they do not read as a ninth agent.
_HUMAN_COLOR = "0.30"


class _ShareWeight(NamedTuple):
    """One way of weighing a contributor's slice of the composition bar."""

    columns: tuple[str, ...]        # what to stack, in stacking order
    labels: tuple[str, ...]         # legend entry per column
    colors: tuple                   # colour per column
    total_column: str               # the per-feature dollar figure annotated at the bar's end
    total_label: str
    total_format: str
    filename: str                   # figure stem, naming what is split
    title: str                      # panelled-form figure title


#: Two views of one composition, selected by `plot_team_shape`'s `weight`:
#:
#: `agent` — each role's share of the agent bill.
#: `team`  — the same eight roles plus the developers.  The agent side is
#:   `cost_agent_mean` apportioned by the `agent` shares, so the nine slices add to
#:   `cost_team_mean` exactly.
#:
#: Both weigh a role by what it bills, not by how many of it there are: a head-count share
#: would overstate the three autonomous roles, which are billed flat and bounded only by
#: their own saturation.  Both annotate the reported per-feature figure, so a bar's end is
#: in the unit the tables report.
_SHARE_WEIGHTS: dict[str, _ShareWeight] = {
    "agent": _ShareWeight(
        columns=BILL_PARAMETERS, labels=AGENT_ROLE_ABBREV, colors=_ROLE_COLORS,
        total_column="cost_agent_mean", total_label="agent \\$ per\nfeature",
        total_format="${:,.2f}",
        filename="composition_agent_bill", title="Share of agent costs"),
    "team": _ShareWeight(
        columns=("cost_human_mean",) + AGENT_COST_PARAMETERS,
        labels=("Developers",) + AGENT_ROLE_ABBREV,
        colors=(_HUMAN_COLOR,) + _ROLE_COLORS,
        total_column="cost_team_mean", total_label="team \\$ per\nfeature",
        total_format="${:,.0f}",
        filename="composition_team_bill", title="Share of development costs"),
}


#: The escaped-bug cost, annotated beside every bar but never stacked into it: the bar
#: describes the team, and this is what the team's choices cost elsewhere.
_BUG_COST_COLUMN = "cost_quality_mean"
_BUG_COST_LABEL = "bug \\$ per\nfeature"

#: Bar-grid geometry, in inches except the inter-block gap, which is in bar rows.
_PANEL_WIDTH_IN = 4.4
_PANEL_MARGIN_IN = 1.4
_ROW_HEIGHT_IN = 0.30
_BLOCK_GAP = 0.6

#: Points to the right of a bar's end for the two annotated columns. Each has to clear
#: the widest figure the column before it can print.
_ANNOTATION_OFFSETS = (4.0, 54.0)


def _compact_dollars(value: float) -> str:
    """`$83`, `$851`, `$26.2k`, `$517k` — at most five characters, whatever the decade.

    The escaped-bug cost spans several orders of magnitude across the cells, where the
    plain format of the team column would print a seven-digit number into a margin.
    """
    if value < 1_000:
        return f"${value:,.0f}"
    if value < 10_000:
        return f"${value / 1_000:.1f}k"
    if value < 1_000_000:
        return f"${value / 1_000:,.0f}k"
    return f"${value / 1_000_000:.1f}M"


def _composition_shares(
    optima: pd.DataFrame,
    groups: tuple[str, ...],
    columns: tuple[str, ...],
    total_column: str,
    extra_column: str | None = None,
) -> pd.DataFrame:
    """Mean role share per cell, plus the mean total, over whichever `columns` weigh a role.

    The share is formed per scenario and then averaged, so a cell's shares are the mean
    composition of its optima, not the composition of its mean optimum.  A row whose total is
    zero contributes no share but still counts toward `n_scenarios`.
    """
    extras = [total_column] + ([extra_column] if extra_column else [])
    require_columns(optima, list(columns) + extras,
                    "optima table; the cost weighting needs `constellation.role_bill` "
                    "to have been applied first")
    columns = list(columns)
    frame = optima.copy()
    weight_total = frame[columns].sum(axis=1)
    frame["group_total"] = frame[total_column]
    # A per-scenario dollar figure like the first, averaged the same way.  It weighs
    # nothing and enters no share.
    annotations = ["group_total"]
    if extra_column is not None:
        frame["extra_total"] = frame[extra_column]
        annotations.append("extra_total")
    positive = weight_total > 0
    shares = frame.loc[positive, columns].div(weight_total[positive], axis=0)
    keys = [tier_column(g) for g in groups]
    shares = shares.join(frame[keys + annotations])
    out = shares.groupby(keys, sort=False)[columns + annotations].mean()
    out["n_scenarios"] = frame.groupby(keys, sort=False).size()
    return out.reset_index()


def _yaxis_note(ax, x: float, y: float, text: str, dx: float, **kwargs):
    """`text` beside data row `y`, `x` in axes fraction, offset `dx` points sideways.

    Offset in points rather than axes fractions: these notes sit outside a panel, where a
    fraction would place them differently on every layout.
    """
    kwargs.setdefault("va", "center")
    return ax.annotate(text, xy=(x, y), xycoords=ax.get_yaxis_transform(),
                       xytext=(dx, 0), textcoords="offset points", **kwargs)


def _bar_layout(
    blocks: list[str], bars: list[str]
) -> tuple[list[tuple[float, str, str]], list[float], list[str],
           list[tuple[str, float, float]], float]:
    """Row positions for one panel: bars grouped into blocks with a gap between them.

    Every panel carries the same rows, so this runs once.  Returns the (position, block, bar)
    triples, the tick positions and their labels, the (block, first, last) spans for the block
    headings, and the total height including the trailing gap.
    """
    positions: list[tuple[float, str, str]] = []
    ticks: list[float] = []
    tick_labels: list[str] = []
    block_spans: list[tuple[str, float, float]] = []
    y = 0.0
    for block in blocks:
        start = y
        for bar in bars:
            positions.append((y, block, bar))
            ticks.append(y)
            tick_labels.append(bar)
            y += 1.0
        block_spans.append((block, start, y - 1.0))
        y += _BLOCK_GAP
    return positions, ticks, tick_labels, block_spans, y


def plot_team_shape(
    optima: pd.DataFrame,
    weight: str = "agent",
    title: str = "",
) -> Figure:
    """What the optimal team is made of: one hundred-per-cent stacked bar per cell.

    `weight` selects an entry of `_SHARE_WEIGHTS`, which decides what a slice measures and
    supplies the labels, the palette and the annotation format.

    Three scenario groups are laid out at once: bars grouped into blocks by
    `COMPOSITION_BLOCK_GROUP`, the `COMPOSITION_BAR_GROUP` ladder repeating inside each block,
    and that whole structure repeated once per `COMPOSITION_PANEL_GROUP` tier, two panels to a
    grid row.  The per-feature dollar totals are annotated at the end of each bar rather than
    drawn, since an axis showing them would flatten the composition.
    """
    if weight not in _SHARE_WEIGHTS:
        raise ValueError(f"unknown weight {weight!r}; "
                         f"expected one of {list(_SHARE_WEIGHTS)}")
    spec = _SHARE_WEIGHTS[weight]
    columns, total_column, labels, colors = (
        spec.columns, spec.total_column, spec.labels, spec.colors)
    if len(labels) != len(columns):
        raise ValueError(f"{len(labels)} labels for {len(columns)} roles")

    grouping = (COMPOSITION_PANEL_GROUP, COMPOSITION_BLOCK_GROUP, COMPOSITION_BAR_GROUP)
    extra_column = _BUG_COST_COLUMN if _BUG_COST_COLUMN in optima.columns else None
    cells = _composition_shares(optima, grouping, columns, total_column, extra_column)
    panels = tiers_present(cells[tier_column(COMPOSITION_PANEL_GROUP)])
    blocks = tiers_present(cells[tier_column(COMPOSITION_BLOCK_GROUP)])
    bars = tiers_present(cells[tier_column(COMPOSITION_BAR_GROUP)])
    if not blocks or not bars or not panels:
        raise ValueError("no optima for any tier of the grouping variables")

    indexed = cells.set_index([tier_column(g) for g in grouping])
    positions, ticks, tick_labels, block_spans, height = _bar_layout(blocks, bars)

    n_panels = len(panels)
    ncols = min(2, n_panels)
    nrows = math.ceil(n_panels / ncols)
    with plt.style.context(_STYLE):
        # Two annotated columns hang off each panel's right edge; the inter-panel gap and
        # the right margin have to hold them, so they are budgeted in inches.
        fig = plt.figure(figsize=(1.45 + ncols * (_PANEL_WIDTH_IN + _PANEL_MARGIN_IN),
                                  nrows * _ROW_HEIGHT_IN * height + 2.6))
        grid = fig.add_gridspec(nrows, ncols)
        bar_axes = []
        for p in range(n_panels):
            r, c = divmod(p, ncols)
            bar_axes.append(fig.add_subplot(grid[r, c],
                                            sharey=bar_axes[0] if bar_axes else None))
        # Placed by hand: `suptitle`'s default `y` is a fraction, which would grow the
        # gap as the figure grows.
        fig.suptitle(title, fontsize=12, y=1.0 - 0.20 / fig.get_figheight())
        # Below this a segment is drawn but not labelled, or the labels overprint.
        label_min = max(_SHARE_LABEL_MIN, 0.24 / _PANEL_WIDTH_IN)
        for p, (panel, ax) in enumerate(zip(panels, bar_axes)):
            # A panel's column of totals overhangs its right edge, into the next panel's
            # cell of the grid.  Axes are drawn in `zorder` order, so a row is ordered
            # right to left and a neighbour's opaque background cannot cover it.
            ax.set_zorder(float(ncols - p % ncols))
            for position, block, bar in positions:
                key = (panel, block, bar)
                if key not in indexed.index:
                    continue
                record = indexed.loc[key]
                left = 0.0
                for k, parameter in enumerate(columns):
                    width = float(record[parameter])
                    ax.barh(position, width, left=left, height=0.72,
                            color=colors[k], edgecolor="white", linewidth=0.6,
                            label=labels[k] if (p == 0 and position == 0.0) else None)
                    if width >= label_min:
                        ax.text(left + width / 2.0, position, f"{width:.0%}",
                                ha="center", va="center", fontsize=7.5, color="white")
                    left += width
                _yaxis_note(ax, 1.0, position,
                            spec.total_format.format(record["group_total"]),
                            _ANNOTATION_OFFSETS[0], fontsize=8, ha="left", color="0.25")
                if extra_column is not None:
                    _yaxis_note(ax, 1.0, position,
                                _compact_dollars(record["extra_total"]),
                                _ANNOTATION_OFFSETS[1], fontsize=8, ha="left",
                                color="0.25")
            ax.set_xlim(0.0, 1.0)
            ax.set_xticks(np.linspace(0.0, 1.0, 6))
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
            ax.grid(axis="y", visible=False)
            if p % ncols:
                # `sharey` already aligns the rows, so the right column is read off the
                # left one's labels and needs no second copy of them.
                ax.tick_params(axis="y", labelleft=False)
            ax.set_title(f"{COMPOSITION_PANEL_GROUP} = {panel}", fontsize=10,
                         fontweight="bold")
            # One header per panel: each carries its own columns of absolute totals.
            _yaxis_note(ax, 1.0, -1.0, spec.total_label, _ANNOTATION_OFFSETS[0],
                        fontsize=8, ha="left", color="0.25", linespacing=0.95)
            if extra_column is not None:
                _yaxis_note(ax, 1.0, -1.0, _BUG_COST_LABEL, _ANNOTATION_OFFSETS[1],
                            fontsize=8, ha="left", color="0.25", linespacing=0.95)

        bar_axes[0].set_ylim(height - 0.4, -1.3)        # first block at the top
        bar_axes[0].set_yticks(ticks)
        bar_axes[0].set_yticklabels(tick_labels, fontsize=9)
        # Outside the axes, and on the first panel of every grid row rather than once per
        # figure.  These take the place of a y label, which would land left of them.
        for ax in bar_axes[::ncols]:
            for block, start, end in block_spans:
                _yaxis_note(ax, 0.0, (start + end) / 2.0,
                            f"{COMPOSITION_BLOCK_GROUP}\n= {block}",
                            -46, fontsize=9, ha="right", fontweight="bold")
            _yaxis_note(ax, 0.0, -1.0, COMPOSITION_BAR_GROUP, -4, fontsize=8, ha="right",
                        color="0.25")
        handles, legend_labels = bar_axes[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="lower center", ncol=len(labels),
                   fontsize=8, frameon=True, columnspacing=0.9, handlelength=1.1)
        # The legend strip and the title strip are reserved in inches, not in figure
        # fractions, which would grow with the grid.
        fig_h, fig_w = fig.get_figheight(), fig.get_figwidth()
        fig.tight_layout(rect=(1.35 / fig_w, 0.42 / fig_h,
                               1.0 - _PANEL_MARGIN_IN / fig_w, 1.0 - 0.32 / fig_h))
        if n_panels > 1:
            # Room for the columns of totals that overhang each panel's right edge.
            # `wspace` is a fraction of the axes width `tight_layout` has just decided, so
            # solve for the value that makes every gap exactly `_PANEL_MARGIN_IN` wide.
            first, last = bar_axes[0].get_position(), bar_axes[ncols - 1].get_position()
            span = (last.x1 - first.x0) * fig_w
            fig.subplots_adjust(wspace=_PANEL_MARGIN_IN / (
                (span - (ncols - 1) * _PANEL_MARGIN_IN) / ncols))
    return fig


class _Quantity(NamedTuple):
    """One panel of the driver figure: a column, its axis label, its tick formatter."""
    column: str
    label: str
    formatter: Callable[[float, int | None], str] | None = None


#: The quantities the driver figure summarises, one panel each: the four process
#: controls, the two agent counts per arriving feature, and the detection split.  Every
#: panel is linear, including the two that spread over decades.
_TEAM_QUANTITIES: tuple[_Quantity, ...] = (
    _Quantity("N", "human staffing $N$\n(FTE)"),
    _Quantity("R", "release cycle $R$\n(sprints)"),
    _Quantity("F_freeze", "feature freeze $F_{freeze}$\n(sprints)"),
    _Quantity("f_fix", "bug-fixing share\n$f_{fix}$", _percent_tick),
    _Quantity("sup_agents_per_feature", "supervised agents\nper arriving feature"),
    _Quantity("aut_agents_per_feature", "autonomous agents\nper arriving feature"),
    _Quantity("security_share", "Security's share of\nthe detection pair", _percent_tick),
)


def _team_quantities(optima: pd.DataFrame) -> pd.DataFrame:
    """Add the derived structural quantities of `_TEAM_QUANTITIES` to a copy of `optima`.

    Each is formed within a scenario before anything is averaged, so that the figure reports a
    typical ratio rather than the largest scenario's.
    """
    frame = optima.copy()
    # The supervised and autonomous halves get a column each: they are bounded by
    # different things and differ by more than an order of magnitude, so a combined column
    # would be the autonomous count with a rounding error.  Their panels also keep
    # separate x ranges, or every autonomous row lands on one tick.
    supervised = frame[[DEV_PARAMETERS[i] for i in SUPERVISED_ROLE_INDICES]].sum(axis=1)
    autonomous = frame[[DEV_PARAMETERS[i] for i in AUTONOMOUS_ROLE_INDICES]].sum(axis=1)
    detection = frame["a_dev_5"] + frame["a_dev_6"]
    frame["sup_agents_per_feature"] = supervised / frame["r_arrival"]
    frame["aut_agents_per_feature"] = autonomous / frame["r_arrival"]
    # A team with neither detection service has no split.  NaN drops out of the order
    # statistics rather than being charged to either service as a zero.
    frame["security_share"] = frame["a_dev_6"].where(detection > 0) / detection
    return frame


def _tier_rows(frame: pd.DataFrame, groups: tuple[str, ...]) -> list:
    """(group, tier, y position, subframe) for every tier present, with gaps between groups."""
    require_columns(frame, [tier_column(g) for g in groups], "optima table")
    rows, y = [], 0.0
    for group in groups:
        column = tier_column(group)
        for tier in tiers_present(frame[column]):
            rows.append((group, tier, y, frame[frame[column] == tier]))
            y += 1.0
        y += 0.8
    return rows


def plot_team_drivers(
    optima: pd.DataFrame,
    title: str = "",
) -> Figure:
    """Which scenario setting moves which part of the team, on one shared row axis.

    One row per (scenario group, tier) and one panel per structural quantity; each row is the
    median over the scenarios at that tier, with an interquartile bar.  Order statistics
    rather than mean and sd, because several of these quantities pile up on a search-range
    boundary, where a mean reports a value no optimum took.

    Putting all five groups on one axis requires the campaign to be a full factorial, or a row
    is confounded by the four groups it pools over.
    """
    frame = _team_quantities(optima)
    rows = _tier_rows(frame, SCENARIO_GROUPS)
    if not rows:
        raise ValueError("no optima for any tier of the scenario groups")
    height = max(r[2] for r in rows) + 1.0
    # Midway down the gap `_tier_rows` leaves between two groups, so the block boundary
    # lines up across all the panels.
    separators = [(prev[2] + cur[2]) / 2.0
                  for prev, cur in zip(rows, rows[1:]) if prev[0] != cur[0]]

    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(
            1, len(_TEAM_QUANTITIES),
            figsize=(2.5 * len(_TEAM_QUANTITIES) + 1.4, 0.25 * height + 1.9),
            sharey=True, squeeze=False,
        )
        fig.suptitle(title, fontsize=12, y=0.985)
        for a, quantity in enumerate(_TEAM_QUANTITIES):
            ax = axes[0][a]
            for y in separators:
                ax.axhline(y, color="0.75", linewidth=0.8, zorder=1)
            for _group, tier, y, sub in rows:
                values = sub[quantity.column].dropna()
                if values.empty:
                    continue
                q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
                color = _TIER_STYLES[tier].color
                ax.plot([q1, q3], [y, y], color=color, linewidth=2.4, alpha=0.45,
                        solid_capstyle="butt", zorder=2)
                ax.plot([median], [y], marker="o", markersize=6, color=color, zorder=3)
            ax.margins(x=0.06)
            if quantity.formatter is not None:
                ax.xaxis.set_major_formatter(FuncFormatter(quantity.formatter))
                ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel(quantity.label, fontsize=9)
            ax.grid(axis="y", visible=False)
            if a == 0:
                ax.set_yticks([y for _, _, y, _ in rows])
                ax.set_yticklabels([tier for _, tier, _, _ in rows], fontsize=8.5)
        axes[0][0].set_ylim(height - 0.4, -0.8)      # first group at the top

        # The group name once per block, outside the panels, so the rows stay short.
        ax = axes[0][0]
        for group in dict.fromkeys(g for g, _, _, _ in rows):
            ys = [y for g, _, y, _ in rows if g == group]
            ax.text(-0.62, (min(ys) + max(ys)) / 2.0, group,
                    transform=ax.get_yaxis_transform(), fontsize=9.5, ha="center",
                    va="center", fontweight="bold")
        # `veryhigh` and `pessimistic` share the fourth colour slot, and this is the one
        # figure that puts all five groups on one axis, so they get a single merged legend
        # entry rather than the same dot printed twice.
        present = tiers_present(tier for _, tier, _, _ in rows)
        fourth = [t for t in present if t in EXTREME_TIERS]
        legend = [Line2D([], [], linestyle="none", marker="o", markersize=6,
                         color=_TIER_STYLES[t].color, label=t)
                  for t in present if t not in fourth]
        if fourth:
            legend.append(Line2D([], [], linestyle="none", marker="o", markersize=6,
                                 color=_TIER_STYLES[fourth[0]].color,
                                 label=" / ".join(fourth)))
        legend.append(Line2D([], [], color="0.45", linewidth=2.4, alpha=0.6,
                             label="median with interquartile range"))
        fig.legend(handles=legend, loc="lower center", ncol=len(legend), fontsize=8,
                   frameon=True, columnspacing=1.2, handlelength=1.6)
        # Reserving only what the suptitle and the legend strip occupy; `tight_layout`
        # adds its own padding on top.
        fig.tight_layout(rect=(0.075, 0.035, 1.0, 0.965))
    return fig


def visualize_team_summary(
    optima: pd.DataFrame,
    out_dir: Path,
) -> None:
    """The team-structure summary figures, from the searched rows of ``optima.csv``.

    `optima` must carry the per-role bill of `constellation.role_bill`, which the caller
    applies because the prices come from the scenario configs this module does not read.
    Writes one composition figure per entry of `_SHARE_WEIGHTS`, plus the driver figure.
    """
    searched = search_rows(optima)
    if searched.empty:
        warnings.warn("no searched optimum to summarise; nothing to plot.", stacklevel=2)
        return

    print(f"\n── Team structure summary ({len(searched)} scenarios) ──")
    for weight in _SHARE_WEIGHTS:
        fig = plot_team_shape(searched, weight=weight,
                              title=_SHARE_WEIGHTS[weight].title)
        save_figure(fig, out_dir, _SHARE_WEIGHTS[weight].filename)

    fig = plot_team_drivers(
        searched,
        title="Impact of scenario on team composition",
    )
    save_figure(fig, out_dir, "team_drivers")


# ---------------------------------------------------------------------------
# Result-directory helpers
# ---------------------------------------------------------------------------

def save_figure(fig: Figure, out_dir: Path, name: str) -> None:
    """Write `fig` to ``<out_dir>/<name>.png``, close it, and report the path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{name}.png"
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {dest}")


def visualize_campaign(
    optima: pd.DataFrame,
    out_dir: Path,
    turnover: pd.DataFrame | None = None,
) -> None:
    """The campaign matrices, drawn from a consolidated ``optima.csv``.

    The human-only rows are split off and passed to the matrices as a `baseline` overlay
    rather than getting figures of their own.  `turnover` is the long-format
    ``elite_turnover.csv``; only its search rows drive the convergence figure.
    """
    require_columns(optima, ("mode",), "optima table")

    baseline = baseline_rows(optima)
    searches = search_rows(optima)
    if baseline.empty:
        baseline = None
    if searches.empty:
        warnings.warn("no search optimum to draw; skipping the campaign matrices.",
                      stacklevel=2)
        return

    suffix = f", search ({len(searches)} scenarios)"
    if baseline is not None:
        suffix += f" + {len(baseline)} human-only baselines"
    print(f"\n── Optima matrices ({len(searches)} scenarios) ──")

    fig = plot_human_agent_matrix(
        searches, baseline=baseline,
        title=f"{HUMAN_AGENT_TITLE}{suffix}",
    )
    save_figure(fig, out_dir, "human_agent_matrix")

    fig = plot_quality_development_matrix(
        searches, baseline=baseline,
        title=f"{QUALITY_DEVELOPMENT_TITLE}{suffix}",
    )
    save_figure(fig, out_dir, "quality_development_matrix")

    # No baseline: a team with no agents has no agent/developer ratio.
    fig = plot_agent_human_ratio_matrix(
        searches, title=f"{AGENT_HUMAN_RATIO_TITLE}{suffix}",
    )
    save_figure(fig, out_dir, "agent_human_ratio_matrix")

    # An enumeration has no generations, so it contributes no rows here.
    if turnover is not None and not turnover.empty:
        curves = search_rows(turnover)
        if not curves.empty:
            fig = plot_elite_turnover(
                curves, title=ELITE_TURNOVER_TITLE)
            save_figure(fig, out_dir, "elite_turnover")
