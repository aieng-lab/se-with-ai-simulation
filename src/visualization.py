"""Visualization functions for search results.

All functions accept plain Python dicts / pandas DataFrames derived from the persisted
result files and return matplotlib Figure objects. Callers decide how to save or display.

Public API — plot functions
---------------------------
    plot_cost_vs_value(objectives, best_key, keep_keys, floor, title)   -> Figure
    plot_cost_split(objectives, keys, title)                            -> Figure
    plot_trajectories(traces_per_config, highlight_keys, fields, title) -> Figure
    plot_comparison_optima(scenarios, title)                            -> Figure

Public API — campaign-level plots (paper §5)
---------------------------------------------
This one takes the ``optima.csv`` table of :mod:`src.campaign`, not a per-search
`objectives` dict.

    plot_optima_matrix(df, x_group, y_group, size_group, marker_group,
                       color_group, share, title)                       -> Figure
    plot_human_agent_matrix(df, x_group, y_group, size_group, marker_group,
                            color_group, title)                         -> Figure

Both share `_draw_matrix`, which owns the five-channel tier grid; they differ only in the
pair of columns they put on the axes.

Public API — result-directory helpers
--------------------------------------
    visualize_single(result_dir, out_dir, fmt)                    -> None
    visualize_multi(result_dirs, out_dir, fmt)                    -> None
    visualize_campaign(optima, out_dir, fmt, share, **channels)   -> None
    save_figure(fig, out_dir, name, fmt)                          -> None
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe to override before pyplot import
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from .config import ALL_SCENARIO_TIERS, SCENARIO_GROUPS

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure

# Shared palette
_BEST_COLOR  = cm.tab10(3)     # red — c*
_KEEP_COLOR  = cm.tab10(0)     # blue — the indistinguishable set I*
_OK_COLOR    = "#88bb88"       # green-grey — admissible but distinguishably worse
_OTHER_COLOR = "#cccccc"       # grey — violates the throughput constraint
_STYLE       = "seaborn-v0_8-whitegrid"

#: A §4.2 search evaluates up to 10,000 candidates, far past the point where per-point CI
#: whiskers or one bar per configuration stay legible.
_ERRORBAR_LIMIT = 200          # points drawn with CI whiskers
_BAR_LIMIT      = 60           # bars in plot_cost_split


# ---------------------------------------------------------------------------
# 1. The search landscape: cost against the constrained quantity
# ---------------------------------------------------------------------------

def plot_cost_vs_value(
    objectives: dict[str, dict],
    best_key: str | None,
    keep_keys: list[str],
    floor: float,
    title: str = "",
) -> "Figure":
    """Every evaluated candidate as (value, cost), with the throughput floor marked.

    This is the search problem of §3.9 drawn directly: the vertical line is
    `(1 - eps) * r_arrival`, everything left of it violates the throughput constraint, and
    the objective is to sit as low as possible to the right of it.  `c*` and the
    indistinguishable set `I*` are highlighted, and `c*` carries its cost CI — the interval
    whose overlap defines `I*`.
    """
    keep = set(keep_keys) - ({best_key} if best_key else set())
    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=(8, 5.5))
        if title:
            ax.set_title(title, fontsize=11)

        tiers: list[tuple[list[str], str, str, float, float]] = [
            ([k for k, o in objectives.items()
              if not o["feasible_throughput"]], _OTHER_COLOR,
             "violates throughput constraint", 12.0, 0.35),
            ([k for k, o in objectives.items()
              if o["feasible_throughput"] and k not in keep and k != best_key], _OK_COLOR,
             "admissible", 16.0, 0.55),
            (sorted(keep), _KEEP_COLOR,
             f"indistinguishable from $c^\\star$ ({len(keep_keys)})", 34.0, 0.9),
        ]
        for keys, color, label, size, alpha in tiers:
            if not keys:
                continue
            ax.scatter([objectives[k]["value_mean"] for k in keys],
                       [objectives[k]["cost_mean"] for k in keys],
                       s=size, c=[color], alpha=alpha, label=label,
                       edgecolors="none", zorder=2)

        ax.axvline(floor, color="black", linestyle="--", linewidth=1.1, zorder=1,
                   label=f"throughput floor ({floor:.2f})")

        if best_key is not None:
            obj = objectives[best_key]
            ax.errorbar(obj["value_mean"], obj["cost_mean"],
                        yerr=[[obj["cost_mean"] - obj["cost_ci"][0]],
                              [obj["cost_ci"][1] - obj["cost_mean"]]],
                        fmt="*", markersize=17, color=_BEST_COLOR, ecolor=_BEST_COLOR,
                        elinewidth=1.4, capsize=4, zorder=4,
                        label=f"$c^\\star$ = {best_key}")

        ax.set_xlabel("value   (features / sprint)", fontsize=9)
        ax.set_ylabel("cost   (\\$ / feature)", fontsize=9)
        ax.set_yscale("log")
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. What the cost is made of
# ---------------------------------------------------------------------------

_COST_DRIVERS: tuple[tuple[str, str], ...] = (
    ("cost_quality_mean", "escaped bugs"),
    ("cost_human_mean",   "developers"),
    ("cost_agent_mean",   "agents"),
)


def plot_cost_split(
    objectives: dict[str, dict],
    keys: list[str],
    title: str = "",
) -> "Figure":
    """Stacked bars of the three cost drivers of §3.8, one bar per key.

    The three shares sum to `cost_mean` by construction, so this reads as "where the money
    went" for the reported configurations.  Truncated at `_BAR_LIMIT` bars.
    """
    shown = keys[:_BAR_LIMIT]
    if not shown:
        raise ValueError("plot_cost_split needs at least one key")
    truncated = len(keys) - len(shown)

    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=(max(6.0, 0.42 * len(shown) + 3.0), 4.6))
        if title:
            ax.set_title(title, fontsize=11)

        x = np.arange(len(shown))
        bottom = np.zeros(len(shown))
        for i, (field, label) in enumerate(_COST_DRIVERS):
            vals = np.array([objectives[k][field] for k in shown], dtype=float)
            ax.bar(x, vals, bottom=bottom, label=label, color=cm.tab10(i), width=0.8)
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels(shown, rotation=90, fontsize=6)
        ax.set_ylabel("cost   (\\$ / feature)", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        if truncated:
            ax.set_xlabel(f"(+{truncated} further configuration(s) not shown)", fontsize=8)
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. State trajectories
# ---------------------------------------------------------------------------

_DEFAULT_TRAJECTORY_FIELDS = ["F_completed", "D_tech", "B_kno", "doc_cov"]
_TRAJECTORY_LABELS = {
    "F_completed": "Cumulative features\n(F_completed)",
    "D_tech":      "Technical debt\n(D_tech)",
    "B_kno":       "Known bugs\n(B_kno_min + B_kno_maj)",
    "doc_cov":     "Documentation coverage\n(doc_cov)",
}


def plot_trajectories(
    traces_per_config: "dict[str, pd.DataFrame]",
    highlight_keys: list[str],
    fields: "list[str] | None" = None,
    title: str = "",
) -> "Figure":
    """Mean trajectory per sprint for the given configs, with a shaded quantile band.

    Each trace DataFrame must have a (replication, sprint) MultiIndex and columns for
    each state field. B_kno is derived as B_kno_min + B_kno_maj if both are present.
    """
    if fields is None:
        fields = _DEFAULT_TRAJECTORY_FIELDS

    wanted = set(highlight_keys)
    palette = cm.tab10
    config_colors = {k: palette(i % 10) for i, k in enumerate(wanted)}

    with plt.style.context(_STYLE):
        nfields = len(fields)
        fig, axes = plt.subplots(nfields, 1, figsize=(10, 3 * nfields), sharex=True)
        if nfields == 1:
            axes = [axes]
        if title:
            fig.suptitle(title, fontsize=12)

        for key, df in traces_per_config.items():
            if key not in wanted:
                continue
            color = config_colors[key]

            # Derive B_kno if needed
            if "B_kno" in fields and "B_kno" not in df.columns:
                if "B_kno_min" in df.columns and "B_kno_maj" in df.columns:
                    df = df.copy()
                    df["B_kno"] = df["B_kno_min"] + df["B_kno_maj"]

            for ax, field in zip(axes, fields):
                if field not in df.columns:
                    continue
                g = df.groupby("sprint")[field]
                sprints = g.mean().index
                mean  = g.mean().values
                lo    = g.quantile(0.10).values
                hi    = g.quantile(0.90).values
                ax.plot(sprints, mean, color=color, linewidth=1.5, label=key)
                ax.fill_between(sprints, lo, hi, color=color, alpha=0.12)

        for ax, field in zip(axes, fields):
            ax.set_ylabel(_TRAJECTORY_LABELS.get(field, field), fontsize=8)
            ax.tick_params(labelsize=7)

        axes[-1].set_xlabel("Sprint", fontsize=9)
        axes[0].legend(fontsize=7, loc="upper left")
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Cross-experiment comparison of the reported optima
# ---------------------------------------------------------------------------

def plot_comparison_optima(
    scenarios: dict[str, dict],
    title: str = "",
) -> "Figure":
    """One point per scenario: the cost of its `c*` with the CI that defines its `I*`.

    `scenarios` maps a label to the `objectives` entry of that scenario's `c*`, so a
    scenario whose search simulated nothing simply has no row.
    """
    labels = list(scenarios)
    if not labels:
        raise ValueError("plot_comparison_optima needs at least one scenario")

    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=(8, max(3.0, 0.42 * len(labels) + 1.6)))
        if title:
            ax.set_title(title, fontsize=11)

        y = np.arange(len(labels))
        means = np.array([scenarios[l]["cost_mean"] for l in labels], dtype=float)
        lo = np.array([scenarios[l]["cost_ci"][0] for l in labels], dtype=float)
        hi = np.array([scenarios[l]["cost_ci"][1] for l in labels], dtype=float)
        admissible = [scenarios[l]["feasible_throughput"] for l in labels]

        ax.errorbar(means, y, xerr=[means - lo, hi - means], fmt="none",
                    ecolor="#888888", elinewidth=1.2, capsize=3, zorder=1)
        for ok, colour, label in ((True, _BEST_COLOR, "admissible $c^\\star$"),
                                  (False, _OTHER_COLOR, "no admissible candidate")):
            sel = [i for i, a in enumerate(admissible) if a == ok]
            if sel:
                ax.scatter(means[sel], y[sel], s=44, c=[colour], zorder=2, label=label)

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlabel("cost of $c^\\star$   (\\$ / feature)", fontsize=9)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. The campaign in one figure: all five scenario groups at once
# ---------------------------------------------------------------------------

# Ordered so the legends read low → medium → high → fourth tier rather than alphabetically.
# These are really *ordinal-position* encodings that happen to be keyed by tier name, so
# `veryhigh` and `pessimistic` share the fourth slot: they are the fourth tier of `project`
# and of `agent` respectively, no group holds both, and therefore no channel can ever have
# to draw them side by side.  `tests/test_config.py` pins that invariant; keeping the
# vocabulary at four levels is what lets the shape channel stay legible (see below).
_TIER_COLORS = {"low": cm.tab10(0), "medium": cm.tab10(1), "high": cm.tab10(3),
                "veryhigh": cm.tab10(4), "pessimistic": cm.tab10(4)}

# Marker shape per tier.  A cross, an open circle, a filled triangle and a filled square
# stay apart from each other even when overplotted at small sizes; four filled shapes do
# not, and a fifth distinct shape would not survive the smallest size tier at all.
_TIER_MARKERS = {"low": "x", "medium": "o", "high": "^", "veryhigh": "s",
                 "pessimistic": "s"}
_TIER_MARKER_FILLED = {"low": False, "medium": False, "high": True, "veryhigh": True,
                       "pessimistic": True}

# Marker area (pt²) per tier.  A 20× spread from low to high (~4.5× in marker width) so
# the size channel survives the alpha blending of a dense panel.
_TIER_SIZES = {"low": 3.0, "medium": 16.0, "high": 60.0, "veryhigh": 150.0,
               "pessimistic": 150.0}

#: The cost drivers of Eq. (cost_rep) that `plot_optima_matrix` can put on x, as the
#: ``cost_<driver>_share`` column of `optima.csv` and the axis label it prints.
COST_SHARES: dict[str, str] = {
    "agent":   "agent share of total cost",
    "human":   "developer share of total cost",
    "quality": "escaped-bug share of total cost",
}


def _percent_tick(value: float, _pos: int | None = None) -> str:
    """Tick label for a fraction on a log axis: ``10%``, ``1%``, ``0.1%``, …

    A fixed number of decimals cannot serve an axis spanning three decades; the label
    keeps exactly as many as the tick needs.
    """
    pct = 100.0 * value
    if pct <= 0:
        return "0"
    return f"{pct:.{max(0, int(np.ceil(-np.log10(pct))))}f}%"


def _dollar_tick(value: float, _pos: int | None = None) -> str:
    """Tick label for a dollar amount: ``$5``, ``$20``, ``$1.5k``, …"""
    if value >= 1000:
        return f"\\${value / 1000:g}k"
    return f"\\${value:g}"


def _tiers_present(df: "pd.DataFrame", column: str) -> list[str]:
    """The tiers occurring in `column`, in mild→extreme order.

    Only `project` reaches "veryhigh" and only `agent` reaches "pessimistic", and a campaign
    may be partial, so the count varies per group and callers must not assume a fixed set.
    """
    return [tier for tier in ALL_SCENARIO_TIERS if (df[column] == tier).any()]


def _tier_groups(df: "pd.DataFrame", column: str) -> "list[tuple[str, pd.DataFrame]]":
    """(tier, subframe) pairs in mild→extreme order, skipping empty tiers."""
    return [(tier, df[df[column] == tier]) for tier in _tiers_present(df, column)]


def _log_floor(df: "pd.DataFrame", column: str) -> float | None:
    """Where non-positive values are drawn on a log axis: half the smallest positive.

    Returned as one number for the whole table so the same scenario lands at the same
    coordinate in every panel of a faceted figure; `None` when nothing needs clipping.
    """
    x = df[column].to_numpy(dtype=float)
    positive = x[x > 0]
    if positive.size and (x <= 0).any():
        return float(positive.min()) * 0.5
    return None


def _marker_kwargs(marker_tier: str, color, area: float) -> dict:
    """Scatter keywords drawing `marker_tier`'s shape in `color` at `area` pt².

    The cross and the open circle are strokes, so they need a line width that grows with
    the marker; the filled shapes must have none or the edge swallows them at the smallest
    size tier.
    """
    marker = _TIER_MARKERS[marker_tier]
    if _TIER_MARKER_FILLED[marker_tier]:
        return {"marker": marker, "color": color, "linewidths": 0}
    width = max(0.35, 0.16 * float(np.sqrt(area)))
    if marker == "o":
        return {"marker": marker, "facecolors": "none", "edgecolors": color,
                "linewidths": width}
    return {"marker": marker, "color": color, "linewidths": width}


def _density_style(n: int) -> tuple[float, float]:
    """(marker-size multiplier, alpha) for a point cloud of `n` points."""
    if n <= 2_000:
        return 1.0, 0.55
    if n <= 15_000:
        return 0.6, 0.40
    return 0.4, 0.30


def _check_matrix_groups(df: "pd.DataFrame", channels: dict[str, str]) -> None:
    """Validate the five channel→group assignments of :func:`plot_optima_matrix`."""
    unknown = {c: g for c, g in channels.items() if g not in SCENARIO_GROUPS}
    if unknown:
        raise ValueError(
            "unknown scenario group(s) "
            + ", ".join(f"{c}={g!r}" for c, g in sorted(unknown.items()))
            + f"; expected one of {list(SCENARIO_GROUPS)}"
        )
    seen: dict[str, list[str]] = {}
    for channel, group in channels.items():
        seen.setdefault(group, []).append(channel)
    duplicated = {g: sorted(cs) for g, cs in seen.items() if len(cs) > 1}
    if duplicated:
        raise ValueError(
            "each scenario group may drive only one visual channel; "
            + ", ".join(f"{g!r} used by {cs}" for g, cs in sorted(duplicated.items()))
        )
    missing = [f"tier_{g}" for g in channels.values() if f"tier_{g}" not in df.columns]
    if missing:
        raise KeyError(f"{missing} are not columns of the optima table")


def _draw_matrix(
    df: "pd.DataFrame",
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
    yscale: str = "log",
    x_floor: float | None = None,
    y_floor: float | None = None,
) -> "Figure":
    """The five-channel tier grid, for any pair of columns.

    The facet layout, the colour/shape/size encoding of the three non-facet groups and the
    three proxy legends live here and nowhere else; callers validate the channel assignment
    (:func:`_check_matrix_groups`) and supply the two plot columns.  `x_floor` / `y_floor`
    clip non-positive values onto a log axis (see :func:`_log_floor`).
    """
    columns = _tier_groups(df, f"tier_{x_group}")
    rows = _tier_groups(df, f"tier_{y_group}")
    if not columns or not rows:
        raise ValueError("no rows for any tier of the faceting groups")

    # Per channel, only the tiers the data actually has: the groups differ in tier count
    # (`project` and `agent` have four), so a fixed list would print dead legend entries.
    size_tiers = _tiers_present(df, f"tier_{size_group}")
    marker_tiers = _tiers_present(df, f"tier_{marker_group}")
    color_tiers = _tiers_present(df, f"tier_{color_group}")

    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(
            len(rows), len(columns),
            figsize=(5.0 * len(columns), 4.2 * len(rows)),
            sharex=True, sharey=True,
            squeeze=False,
        )
        fig.suptitle(title, fontsize=12, y=1.0)

        for r, (y_tier, y_sub) in enumerate(rows):
            for c, (x_tier, _) in enumerate(columns):
                ax = axes[r][c]
                cell = y_sub[y_sub[f"tier_{x_group}"] == x_tier]
                scale, alpha = _density_style(len(cell))
                # Largest markers first, so the small ones are not buried under them.
                for size_tier in reversed(size_tiers):
                    for marker_tier in marker_tiers:
                        for color_tier in color_tiers:
                            sub = cell[
                                (cell[f"tier_{size_group}"] == size_tier)
                                & (cell[f"tier_{marker_group}"] == marker_tier)
                                & (cell[f"tier_{color_group}"] == color_tier)
                            ]
                            if sub.empty:
                                continue
                            x = sub[x_col].to_numpy(dtype=float)
                            y = sub[y_col].to_numpy(dtype=float)
                            if x_floor is not None:
                                x = np.where(x > 0, x, x_floor)
                            if y_floor is not None:
                                y = np.where(y > 0, y, y_floor)
                            area = _TIER_SIZES[size_tier] * scale
                            ax.scatter(
                                x, y, s=area, alpha=alpha,
                                **_marker_kwargs(marker_tier, _TIER_COLORS[color_tier],
                                                 area),
                            )
                ax.set_xscale(xscale)
                ax.set_yscale(yscale)
                ax.set_title(f"n = {len(cell)}", fontsize=8, loc="right")
                if r == 0:
                    ax.set_title(f"{x_group} = {x_tier}", fontsize=10, loc="center")
                if r == len(rows) - 1:
                    ax.set_xlabel(xlabel, fontsize=9)
                if c == 0:
                    ax.set_ylabel(f"{y_group} = {y_tier}\n{ylabel}", fontsize=9)

        # Three proxy legends, one per non-facet channel.  Line2D takes a marker *width*
        # in points where scatter takes an area in pt², hence the sqrt.
        legend_specs = (
            (color_group, [
                Line2D([], [], linestyle="none", marker="o", markersize=6,
                       color=_TIER_COLORS[t], label=t) for t in color_tiers
            ]),
            (marker_group, [
                Line2D([], [], linestyle="none", marker=_TIER_MARKERS[t], markersize=7,
                       color="0.35", markeredgecolor="0.35", label=t,
                       markerfacecolor="0.35" if _TIER_MARKER_FILLED[t] else "none")
                for t in marker_tiers
            ]),
            (size_group, [
                Line2D([], [], linestyle="none", marker="o", color="0.35", label=t,
                       markersize=float(np.sqrt(_TIER_SIZES[t])) * 1.5)
                for t in size_tiers
            ]),
        )
        fig.tight_layout(rect=(0.0, 0.0, 0.86, 1.0))
        # Stack the three legends by their own height: a four-tier channel (`project` or
        # `agent`) needs one more row than a three-tier one, or the next legend rides over it.
        anchor_y = 0.92
        for i, (group, handles) in enumerate(legend_specs):
            fig.legend(
                handles=handles, loc="upper left", frameon=True, fontsize=8,
                title=f"{group} (" + ("colour", "shape", "size")[i] + ")",
                title_fontsize=8, labelspacing=0.9,
                bbox_to_anchor=(0.875, anchor_y),
            )
            anchor_y -= 0.065 + 0.045 * len(handles)
    return fig


def plot_optima_matrix(
    df: "pd.DataFrame",
    x_group: str = "agent",
    y_group: str = "human",
    size_group: str = "bugimpact",
    marker_group: str = "project",
    color_group: str = "debt",
    share: str = "agent",
    title: str = "",
) -> "Figure":
    """The reported optimum of every scenario, all five scenario groups separated at once.

    `df` is the ``optima.csv`` table of :mod:`src.campaign`, filtered to a **single**
    strategy — mixing the two would draw each scenario twice.  One point per scenario:
    y is the cost of its `c*` in $ per feature, x is the share of that cost the `share`
    driver accounts for (`cost_<share>_share`).

    The grid facets the tiers of `x_group` (columns) against those of `y_group` (rows);
    inside a panel the remaining three groups become marker colour, shape and size.
    Two groups carry a fourth tier, `project` and `agent`, and they are placed accordingly:
    `project` belongs on the shape channel, the only *per-point* channel whose fourth level
    reads as a peer of the other three rather than as "more" of them, while `agent` sits on
    a facet axis, where a fourth tier costs nothing but one extra column (the default grid
    is therefore 4×3).

    Both axes are log-scaled and shared across all panels so the tiers compare directly.
    The x axis needs the log scale in particular because the agent bill is a low-single-digit
    percentage of the total at most optima while spanning some three orders of magnitude;
    a share of exactly zero is clipped onto :func:`_log_floor` so it stays visible.
    """
    channels = {
        "x_group": x_group, "y_group": y_group, "size_group": size_group,
        "marker_group": marker_group, "color_group": color_group,
    }
    _check_matrix_groups(df, channels)
    if share not in COST_SHARES:
        raise ValueError(f"unknown cost share {share!r}; expected one of "
                         f"{list(COST_SHARES)}")
    x_col = f"cost_{share}_share"
    for column in (x_col, "cost_mean"):
        if column not in df.columns:
            raise KeyError(f"{column!r} is not a column of the optima table")

    fig = _draw_matrix(
        df,
        x_group=x_group, y_group=y_group, size_group=size_group,
        marker_group=marker_group, color_group=color_group,
        x_col=x_col, y_col="cost_mean",
        xlabel=COST_SHARES[share],
        # The dollar sign needs escaping or mathtext sees three delimiters in the label
        # and renders the whole thing, `c*` included, literally.
        ylabel="↓  cost of $c^\\star$   (\\$ / feature)",
        title=title or f"Reported optima — {x_group} × {y_group}",
        x_floor=_log_floor(df, x_col),
    )
    # Shares read as percentages; the minor ticks stay unlabelled or a three-decade axis
    # carries eight labels per decade.
    for ax in fig.get_axes():
        ax.xaxis.set_major_formatter(FuncFormatter(_percent_tick))
        ax.xaxis.set_minor_formatter(NullFormatter())
    return fig


def plot_human_agent_matrix(
    df: "pd.DataFrame",
    x_group: str = "agent",
    y_group: str = "human",
    size_group: str = "bugimpact",
    marker_group: str = "project",
    color_group: str = "debt",
    title: str = "",
) -> "Figure":
    """The two team cost drivers of each optimum against each other, on the same tier grid.

    A variant of :func:`plot_optima_matrix` sharing its facets and its colour/shape/size
    channels, so the two figures read the same way; only the pair of columns differs.
    Here both axes are dollars per feature: x is what the agents billed
    (`cost_agent_mean`), y is what the developers cost (`cost_human_mean`).

    This is the part of the bill the team composition actually controls — the quality term
    is what the escaped bugs cost afterwards — so the panel is the human/agent trade-off
    itself rather than a view of the total.  Note the two axes are not comparable in
    magnitude: the agent bill sits in the tens of dollars per feature across the whole
    campaign while the human one spans four decades, which is why both are log-scaled and
    why a diagonal here is not a 1:1 exchange.
    """
    channels = {
        "x_group": x_group, "y_group": y_group, "size_group": size_group,
        "marker_group": marker_group, "color_group": color_group,
    }
    _check_matrix_groups(df, channels)
    for column in ("cost_agent_mean", "cost_human_mean"):
        if column not in df.columns:
            raise KeyError(f"{column!r} is not a column of the optima table")

    fig = _draw_matrix(
        df,
        x_group=x_group, y_group=y_group, size_group=size_group,
        marker_group=marker_group, color_group=color_group,
        x_col="cost_agent_mean", y_col="cost_human_mean",
        # Both labels escape the dollar sign; see plot_optima_matrix.
        xlabel="↓  agent cost   (\\$ / feature)",
        ylabel="↓  developer cost   (\\$ / feature)",
        title=title or f"Developer vs. agent cost at $c^\\star$ — "
                       f"{x_group} × {y_group}",
        x_floor=_log_floor(df, "cost_agent_mean"),
        y_floor=_log_floor(df, "cost_human_mean"),
    )
    # The agent axis spans barely a decade, so the default decade ticks label it once and
    # the panel becomes unreadable; tick it at 1-2-5 per decade in dollars instead.  The
    # developer axis spans four decades and its decade ticks are fine.
    for ax in fig.get_axes():
        ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_major_formatter(FuncFormatter(_dollar_tick))
        ax.xaxis.set_minor_formatter(NullFormatter())
    return fig


# ---------------------------------------------------------------------------
# Result-directory helpers
# ---------------------------------------------------------------------------

def _load_summary(result_dir: Path) -> dict:
    path = result_dir / "search_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"No search_summary.json in {result_dir}.")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_manifest(result_dir: Path) -> dict:
    path = result_dir / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _load_traces(
    result_dir: Path,
    config_keys: list[str],
) -> "dict[str, pd.DataFrame]":
    """Load traces.csv for each config that has one."""
    from .results import load_traces
    out = {}
    for key in config_keys:
        tfile = result_dir / "samples" / f"{key}_traces.csv"
        if tfile.exists():
            out[key] = load_traces(tfile)
    return out


def _scenario_label(result_dir: Path, manifest: dict) -> str:
    """Human-readable scenario name: the §4.2 scenario id if present, else a path stem."""
    scenario_id = manifest.get("scenario_id", "")
    if scenario_id:
        strategy = manifest.get("strategy", "")
        return f"{scenario_id} / {strategy}" if strategy else scenario_id
    scenario = manifest.get("scenario", "")
    if isinstance(scenario, dict):
        scenario = scenario.get("path", "")
    return Path(scenario).stem if scenario else result_dir.name


def save_figure(fig: "Figure", out_dir: Path, name: str, fmt: str) -> None:
    """Write `fig` to ``<out_dir>/<name>.<fmt>``, close it, and report the path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{name}.{fmt}"
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {dest}")


def visualize_single(result_dir: Path, out_dir: Path, fmt: str) -> None:
    """Generate all plots for a single search result directory."""
    summary  = _load_summary(result_dir)
    manifest = _load_manifest(result_dir)
    label    = _scenario_label(result_dir, manifest)

    objectives = summary["objectives"]
    best_key   = summary["best"]
    keep_keys  = summary["indistinguishable"]
    floor      = summary["throughput_floor"]
    title_base = f"Scenario: {label}"

    print(f"\n── {result_dir.name}  ({len(objectives)} simulated, "
          f"{len(keep_keys)} indistinguishable from c*) ──")

    if not objectives:
        warnings.warn(f"{result_dir} simulated no candidate; nothing to plot.",
                      stacklevel=2)
        return

    fig = plot_cost_vs_value(objectives, best_key, keep_keys, floor,
                             title=f"{title_base} — cost vs. throughput")
    save_figure(fig, out_dir, "cost_vs_value", fmt)

    if keep_keys:
        fig = plot_cost_split(objectives, keep_keys,
                              title=f"{title_base} — cost drivers of the reported optima")
        save_figure(fig, out_dir, "cost_split", fmt)

    traces = _load_traces(result_dir, keep_keys)
    if traces:
        fig = plot_trajectories(traces, keep_keys,
                                title=f"{title_base} — state trajectories")
        save_figure(fig, out_dir, "trajectories", fmt)
    else:
        warnings.warn(
            f"No traces.csv found in {result_dir}. Re-run with --traces to enable "
            f"trajectory plots.",
            stacklevel=2,
        )


def visualize_multi(result_dirs: list[Path], out_dir: Path, fmt: str) -> None:
    """Generate a cross-experiment comparison of the reported optima."""
    scenarios: dict[str, dict] = {}
    for rd in result_dirs:
        manifest = _load_manifest(rd)
        summary  = _load_summary(rd)
        best_key = summary["best"]
        if best_key is None:
            continue
        scenarios[_scenario_label(rd, manifest)] = summary["objectives"][best_key]

    if not scenarios:
        warnings.warn("None of the result directories reported an optimum.", stacklevel=2)
        return

    print(f"\n── Cross-experiment comparison ({len(scenarios)} scenarios) ──")
    fig = plot_comparison_optima(scenarios, title="Cost of $c^\\star$ across scenarios")
    save_figure(fig, out_dir, "comparison_optima", fmt)


def visualize_campaign(
    optima: "pd.DataFrame",
    out_dir: Path,
    fmt: str = "png",
    share: str = "agent",
    **channels: str,
) -> None:
    """Both campaign matrices, once per strategy in the campaign's ``optima.csv``.

    The strategies get their own figures rather than one shared pair: they report an
    optimum per scenario independently, so overlaying them would draw every scenario twice
    and hide exactly the disagreement between the two searches that is worth reading.
    `channels` passes `x_group` / `y_group` / `size_group` / `marker_group` /
    `color_group` through to both; `share` reaches only
    :func:`plot_optima_matrix`, the one with a share on an axis.
    """
    if "strategy" not in optima.columns:
        raise KeyError("'strategy' is not a column of the optima table")

    for strategy in sorted(optima["strategy"].unique()):
        sub = optima[optima["strategy"] == strategy]
        suffix = f", {strategy} search ({len(sub)} scenarios)"
        print(f"\n── Optima matrices: {strategy} ({len(sub)} scenarios) ──")

        fig = plot_optima_matrix(sub, share=share,
                                 title=f"Reported optima{suffix}", **channels)
        save_figure(fig, out_dir, f"optima_matrix_{strategy}", fmt)

        fig = plot_human_agent_matrix(
            sub, title=f"Developer vs. agent cost at $c^\\star${suffix}", **channels,
        )
        save_figure(fig, out_dir, f"human_agent_matrix_{strategy}", fmt)
