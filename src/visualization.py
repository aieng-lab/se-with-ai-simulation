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
pair of columns they put on the axes.  All but the ratio figure take an optional `baseline`
frame — the human-only optima of §4.1 — drawn as a fixed black marker outside the three
per-point tier channels; see `_draw_matrix` and `_baseline_cell`.

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
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter, PercentFormatter

from .config import ALL_SCENARIO_TIERS, SCENARIO_GROUPS, UNUSED_TIER

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

# The human-only baseline of paper §4.1 sits *outside* the three tier channels above: it
# is a reference level, not an observation at some tier combination, and it varies only
# three of the five groups.  It therefore gets one fixed style — black, one shape used by
# no tier, one size — rather than a fourth entry anywhere above.  Do not promote it into
# _TIER_MARKERS: a fifth tier shape would not survive the smallest size tier, which is
# exactly why the baseline is drawn on its own channel instead.
_BASELINE_COLOR = "black"
_BASELINE_MARKER = "*"
_BASELINE_SIZE = 90.0
_BASELINE_LABEL = "human-only baseline"

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
    """Tick label for a dollar amount: ``$5``, ``$20``, ``$1.5k``, ``$1M``, …"""
    if value >= 1e6:
        return f"\\${value / 1e6:g}M"
    if value >= 1000:
        return f"\\${value / 1000:g}k"
    return f"\\${value:g}"


def _ratio_tick(value: float, _pos: int | None = None) -> str:
    """Tick label for a dimensionless ratio on a log axis: ``0.001``, ``0.1``, ``1``, ``100``.

    Matplotlib's default would write the sub-unit decades in scientific mathtext; a ratio
    reads better as the plain number, and the axis is only five decades wide.
    """
    return f"{value:g}"


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


def _common_log_limits(
    df: "pd.DataFrame",
    columns: tuple[str, ...],
    floors: tuple[float | None, ...] = (),
) -> tuple[float, float]:
    """One padded (min, max) covering every column of `columns`, for a log axis.

    Used where two columns share a unit and must therefore share a range; the `floors` of
    :func:`_log_floor` are included, since a clipped zero is drawn at its floor and would
    otherwise fall outside the limits.
    """
    parts = [df[c].to_numpy(dtype=float) for c in columns]
    parts.append(np.array([f for f in floors if f is not None], dtype=float))
    values = np.concatenate(parts)
    positive = values[values > 0]
    if positive.size == 0:
        raise ValueError(f"no positive values in {list(columns)}; cannot scale a log axis")
    lo, hi = float(positive.min()), float(positive.max())
    # 5 % of the span in log space, so the extreme points do not sit on the frame.
    pad = 0.05 * max(np.log10(hi / lo), 0.4)
    return 10.0 ** (np.log10(lo) - pad), 10.0 ** (np.log10(hi) + pad)


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


#: Width (inches) reserved to the right of the panels for the three channel legends, and
#: the vertical gap (inches) between two of them.  Both are absolute: the legends are the
#: same objects whatever the grid, so their column must not grow with the facet count.
_LEGEND_WIDTH_IN = 1.9
_LEGEND_GAP_IN = 0.25


def _figure_bbox(fig: "Figure", artist) -> "matplotlib.transforms.Bbox":
    """`artist`'s bounding box in figure coordinates.

    A legend has no extent until a renderer has laid it out, and nothing has drawn the
    figure yet at the point the next legend needs placing; the Agg canvas this module
    pins at import time hands one over without a draw pass.
    """
    renderer = fig.canvas.get_renderer()
    return artist.get_window_extent(renderer).transformed(fig.transFigure.inverted())


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


def _with_baseline(
    df: "pd.DataFrame",
    baseline: "pd.DataFrame | None",
) -> "pd.DataFrame":
    """`df` and `baseline` stacked, for computing axis floors and ranges over both.

    The baseline points are drawn on the same axes as the search optima, so a floor or a
    shared range derived from `df` alone would clip them off the figure — its agent cost
    and agent share are exactly 0, which no log axis can place.
    """
    if baseline is None or baseline.empty:
        return df
    import pandas as pd    # runtime-only: the module-level import is TYPE_CHECKING

    return pd.concat([df, baseline], ignore_index=True)


def _baseline_cell(
    baseline: "pd.DataFrame",
    x_group: str, x_tier: str,
    y_group: str, y_tier: str,
) -> "pd.DataFrame":
    """The baseline rows belonging in the panel at (`x_tier`, `y_tier`).

    A baseline row carries `UNUSED_TIER` for every group it does not vary, and such a
    group constrains nothing: the row belongs in *every* tier of it.  So a facet on
    `agent` — which the baseline holds at `a-X` — repeats the same reference points down
    each column, while a facet on `human`, which it does vary, selects properly.
    """
    cell = baseline
    for group, tier in ((x_group, x_tier), (y_group, y_tier)):
        column = f"tier_{group}"
        if column not in cell.columns:
            continue
        cell = cell[(cell[column] == tier) | (cell[column] == UNUSED_TIER)]
    return cell


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
    shared_limits: tuple[float, float] | None = None,
    baseline: "pd.DataFrame | None" = None,
) -> "Figure":
    """The five-channel tier grid, for any pair of columns.

    The facet layout, the colour/shape/size encoding of the three non-facet groups and the
    three proxy legends live here and nowhere else; callers validate the channel assignment
    (:func:`_check_matrix_groups`) and supply the two plot columns.  `x_floor` / `y_floor`
    clip non-positive values onto a log axis (see :func:`_log_floor`).

    `shared_limits` puts the *same* range on both axes — only meaningful when the two
    columns are in the same unit.  The panels then also get an equal aspect, so a decade is
    the same length on both and the 1:1 diagonal runs at 45°; without that the equal range
    would be a claim the picture does not keep.

    `baseline` holds the human-only optima of §4.1, which carry no tier for the two groups
    they do not vary and are therefore drawn outside the three per-point channels: one
    fixed black marker, matched to a panel by the *facet* groups alone.  Where a facet
    group is one the baseline does not vary (`agent` on the default layout), every column
    gets the same points — the baseline is a reference level, not an observation at that
    tier, and the caption must say so.
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
        # A panel carrying one range on both axes is square, so it needs its width back:
        # 4.2 × 4.2 rather than the wide 5.0 × 4.2 of an unconstrained pair of columns.
        panel_w = 4.2 if shared_limits is not None else 5.0
        fig, axes = plt.subplots(
            len(rows), len(columns),
            figsize=(panel_w * len(columns), 4.2 * len(rows)),
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
                if baseline is not None and not baseline.empty:
                    # Matched on the facet groups only, and on a group the baseline does
                    # not vary every tier of it matches — which is what puts the same
                    # reference points in every agent column of the default layout.
                    b_cell = _baseline_cell(baseline, x_group, x_tier, y_group, y_tier)
                    if not b_cell.empty:
                        bx = b_cell[x_col].to_numpy(dtype=float)
                        by = b_cell[y_col].to_numpy(dtype=float)
                        if x_floor is not None:
                            bx = np.where(bx > 0, bx, x_floor)
                        if y_floor is not None:
                            by = np.where(by > 0, by, y_floor)
                        # Labelled so the overlay is identifiable on the finished axes —
                        # the figure legend is built from proxies, not from this handle.
                        ax.scatter(bx, by, s=_BASELINE_SIZE, marker=_BASELINE_MARKER,
                                   color=_BASELINE_COLOR, linewidths=0, zorder=5,
                                   label=_BASELINE_LABEL)
                ax.set_xscale(xscale)
                ax.set_yscale(yscale)
                if shared_limits is not None:
                    ax.set_xlim(*shared_limits)
                    ax.set_ylim(*shared_limits)
                    # On a log axis the aspect is read in decades, so this is what makes
                    # the two equal ranges equally long on the page.
                    ax.set_aspect("equal", adjustable="box")
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
        if baseline is not None and not baseline.empty:
            legend_specs = legend_specs + ((_BASELINE_LABEL, [
                Line2D([], [], linestyle="none", marker=_BASELINE_MARKER,
                       markersize=float(np.sqrt(_BASELINE_SIZE)) * 1.2,
                       color=_BASELINE_COLOR, label="no agents"),
            ]),)
        # The legend column is a fixed width in *inches*, not a figure fraction: a group
        # with four tiers on the column axis widens the figure, and a fraction would hand
        # the extra inch to the legends rather than to the panels.
        right = 1.0 - _LEGEND_WIDTH_IN / fig.get_figwidth()
        fig.tight_layout(rect=(0.0, 0.0, right, 1.0))
        # Stack the three legends by their *measured* heights.  A per-entry guess cannot
        # do it: an entry is as tall as its marker (the size channel's largest is 18 pt
        # against the 8 pt text), the tier counts differ per channel, and the conversion
        # to the figure fraction the anchor speaks depends on the number of facet rows.
        anchor_y = 0.97
        # The channel names run out after the three per-point channels; the baseline
        # legend that may follow them is not a channel and carries its label alone.
        channel_names = ("colour", "shape", "size")
        for i, (group, handles) in enumerate(legend_specs):
            title = (f"{group} ({channel_names[i]})" if i < len(channel_names)
                     else str(group))
            legend = fig.legend(
                handles=handles, loc="upper left", frameon=True, fontsize=8,
                title=title,
                title_fontsize=8, labelspacing=0.9,
                bbox_to_anchor=(right + 0.15 * _LEGEND_WIDTH_IN / fig.get_figwidth(),
                                anchor_y),
            )
            anchor_y = _figure_bbox(fig, legend).y0 - _LEGEND_GAP_IN / fig.get_figheight()
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
    baseline: "pd.DataFrame | None" = None,
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
    a facet axis, where a fourth tier costs nothing but one extra column (over the full
    §4.2 grid the default is therefore 4 columns × 3 rows, one column per agent tier).

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
        x_floor=_log_floor(_with_baseline(df, baseline), x_col),
        baseline=baseline,
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
    baseline: "pd.DataFrame | None" = None,
) -> "Figure":
    """The two team cost drivers of each optimum against each other, on the same tier grid.

    A variant of :func:`plot_optima_matrix` sharing its facets and its colour/shape/size
    channels, so the two figures read the same way; only the pair of columns differs.
    Here both axes are dollars per feature: x is what the agents billed
    (`cost_agent_mean`), y is what the developers cost (`cost_human_mean`).

    This is the part of the bill the team composition actually controls — the quality term
    is what the escaped bugs cost afterwards — so the panel is the human/agent trade-off
    itself rather than a view of the total.

    Both axes carry the **same range**, the union of the two columns padded in log space
    (:func:`_common_log_limits`), at an equal aspect.  The two are the same unit, so the
    comparison the figure exists for is the one *between* them: where a point sits relative
    to the 1:1 diagonal is which driver the optimum spends more on, and that reading is only
    available when a dollar is the same distance on both axes.  The price is that the agent
    axis is no longer scaled to its own spread — it holds one decade inside a column against
    the four the developer cost spans — so within-column agent differences read small, which
    is the honest depiction of their size.
    """
    channels = {
        "x_group": x_group, "y_group": y_group, "size_group": size_group,
        "marker_group": marker_group, "color_group": color_group,
    }
    _check_matrix_groups(df, channels)
    for column in ("cost_agent_mean", "cost_human_mean"):
        if column not in df.columns:
            raise KeyError(f"{column!r} is not a column of the optima table")

    # The baseline's agent cost is exactly 0, so the floor and the shared range are taken
    # over both frames: computed from `df` alone the floor would be None and every baseline
    # point would silently drop off the log axis.
    scaled = _with_baseline(df, baseline)
    x_floor = _log_floor(scaled, "cost_agent_mean")
    y_floor = _log_floor(scaled, "cost_human_mean")
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
        x_floor=x_floor, y_floor=y_floor,
        shared_limits=_common_log_limits(
            scaled, ("cost_agent_mean", "cost_human_mean"), (x_floor, y_floor)
        ),
        baseline=baseline,
    )
    # One range on both axes means one ticking on both: decades in dollars, with the 2 and
    # the 5 as unlabelled minor ticks so a point can still be placed inside its decade.
    for ax in fig.get_axes():
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(LogLocator(base=10.0))
            axis.set_major_formatter(FuncFormatter(_dollar_tick))
            axis.set_minor_locator(LogLocator(base=10.0, subs=(2.0, 5.0)))
            axis.set_minor_formatter(NullFormatter())
        # The 1:1 line: above it the developers cost more than the agents, below it less.
        ax.plot(ax.get_xlim(), ax.get_xlim(), color="0.6", linewidth=0.8,
                linestyle=(0, (4, 3)), zorder=0)
    return fig


def plot_quality_development_matrix(
    df: "pd.DataFrame",
    x_group: str = "agent",
    y_group: str = "human",
    size_group: str = "bugimpact",
    marker_group: str = "project",
    color_group: str = "debt",
    title: str = "",
    baseline: "pd.DataFrame | None" = None,
) -> "Figure":
    """What the escaped bugs cost against what the team cost, on the same tier grid.

    The other half of the split :func:`plot_human_agent_matrix` starts: there the two drivers
    the composition *controls* are separated from each other, here they are added back
    together (`cost_development_mean = cost_human_mean + cost_agent_mean`, x) and set against
    the one driver it only influences, the bill the defects that reached production sent
    (`cost_quality_mean`, y).  The sum is a column of neither `optima.csv` nor of any other
    table — it is derived here, on a copy, since it exists only for this figure.

    Both axes are dollars per feature, so the figure gets the same-unit treatment of
    :func:`plot_human_agent_matrix`: one shared log range, an equal aspect, and the 1:1
    diagonal.  Above the line the escaped bugs cost more than the whole team that built the
    features, which over the §4.2 campaign is almost everywhere — the quality term spans some
    four and a half decades where the development cost spans three.
    """
    channels = {
        "x_group": x_group, "y_group": y_group, "size_group": size_group,
        "marker_group": marker_group, "color_group": color_group,
    }
    _check_matrix_groups(df, channels)
    for column in ("cost_quality_mean", "cost_human_mean", "cost_agent_mean"):
        if column not in df.columns:
            raise KeyError(f"{column!r} is not a column of the optima table")

    def _with_development(frame):
        return frame.assign(
            cost_development_mean=frame["cost_human_mean"] + frame["cost_agent_mean"])

    df = _with_development(df)
    if baseline is not None and not baseline.empty:
        baseline = _with_development(baseline)
    scaled = _with_baseline(df, baseline)
    x_floor = _log_floor(scaled, "cost_development_mean")
    y_floor = _log_floor(scaled, "cost_quality_mean")
    fig = _draw_matrix(
        df,
        x_group=x_group, y_group=y_group, size_group=size_group,
        marker_group=marker_group, color_group=color_group,
        x_col="cost_development_mean", y_col="cost_quality_mean",
        # Both labels escape the dollar sign; see plot_optima_matrix.
        # Short enough not to collide with the next panel's; the title carries the
        # definition of "development cost".
        xlabel="↓  development cost   (\\$ / feature)",
        ylabel="↓  escaped-bug cost   (\\$ / feature)",
        title=title or f"Escaped-bug vs. development cost (developers + agents) at "
                       f"$c^\\star$ — {x_group} × {y_group}",
        x_floor=x_floor, y_floor=y_floor,
        shared_limits=_common_log_limits(
            scaled, ("cost_development_mean", "cost_quality_mean"), (x_floor, y_floor)
        ),
        baseline=baseline,
    )
    # One range on both axes means one ticking on both; see plot_human_agent_matrix.
    for ax in fig.get_axes():
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(LogLocator(base=10.0))
            axis.set_major_formatter(FuncFormatter(_dollar_tick))
            axis.set_minor_locator(LogLocator(base=10.0, subs=(2.0, 5.0)))
            axis.set_minor_formatter(NullFormatter())
        # The 1:1 line: above it the bugs that escaped cost more than building the features.
        ax.plot(ax.get_xlim(), ax.get_xlim(), color="0.6", linewidth=0.8,
                linestyle=(0, (4, 3)), zorder=0)
    return fig


def plot_agent_human_ratio_matrix(
    df: "pd.DataFrame",
    x_group: str = "agent",
    y_group: str = "human",
    size_group: str = "bugimpact",
    marker_group: str = "project",
    color_group: str = "debt",
    title: str = "",
) -> "Figure":
    """How lopsided the agent/developer split is at each optimum, against its own percentile.

    y is `cost_agent_mean / cost_human_mean`, the same trade-off
    :func:`plot_human_agent_matrix` draws as a plane, collapsed to the one number that says
    which side it fell on: above 1 the agents outbilled the developers.  Over the §4.2
    campaign it spans five decades, hence the log axis and the reference line at 1.

    x is that ratio's **percentile rank within `df`**, so the panels share one empirical
    distribution and each is a fragment of it: where a panel's points sit horizontally is
    where its scenarios sit in the campaign-wide ordering, and a panel that occupies the
    right end is one whose tiers push the bill onto the agents.  The rank is therefore taken
    over the frame as handed in — `visualize_campaign` passes one strategy's scenarios, and
    filtering the frame further before calling re-ranks what is left against itself.

    Ranking makes the x axis linear and uniform by construction; the reading is in the
    *vertical* spread of a panel and in which part of the axis it occupies, not in the
    slope, which is an artefact of every ECDF.
    """
    channels = {
        "x_group": x_group, "y_group": y_group, "size_group": size_group,
        "marker_group": marker_group, "color_group": color_group,
    }
    _check_matrix_groups(df, channels)
    for column in ("cost_agent_mean", "cost_human_mean"):
        if column not in df.columns:
            raise KeyError(f"{column!r} is not a column of the optima table")

    ratio = df["cost_agent_mean"] / df["cost_human_mean"]
    df = df.assign(
        cost_agent_human_ratio=ratio,
        # Ties share their mean rank, so equal ratios land on one x rather than on an
        # arbitrary order of the rows.
        cost_agent_human_ratio_pct=ratio.rank(method="average", pct=True),
    )
    fig = _draw_matrix(
        df,
        x_group=x_group, y_group=y_group, size_group=size_group,
        marker_group=marker_group, color_group=color_group,
        x_col="cost_agent_human_ratio_pct", y_col="cost_agent_human_ratio",
        xlabel="percentile of the agent / developer cost ratio",
        # No "↓" here: unlike a cost, neither direction of the ratio is the better one.
        ylabel="agent cost / developer cost",
        title=title or f"Agent-to-developer cost ratio at $c^\\star$ — "
                       f"{x_group} × {y_group}",
        # x is a rank in [0, 1] — a log axis would only stretch the first percentile.
        xscale="linear",
        y_floor=_log_floor(df, "cost_agent_human_ratio"),
    )
    for ax in fig.get_axes():
        # A rank axis is uniform, so it takes the plain percent formatter rather than the
        # decade-aware `_percent_tick` the log share axis of `plot_optima_matrix` needs.
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_major_formatter(FuncFormatter(_ratio_tick))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(2.0, 5.0)))
        ax.yaxis.set_minor_formatter(NullFormatter())
        # Parity: above the line the agents cost more than the developers they work for.
        ax.axhline(1.0, color="0.6", linewidth=0.8, linestyle=(0, (4, 3)), zorder=0)
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
    """All four campaign matrices, once per strategy in the campaign's ``optima.csv``.

    The strategies get their own figures rather than one shared set: they report an
    optimum per scenario independently, so overlaying them would draw every scenario twice
    and hide exactly the disagreement between the two searches that is worth reading.
    `channels` passes `x_group` / `y_group` / `size_group` / `marker_group` /
    `color_group` through to all four; `share` reaches only
    :func:`plot_optima_matrix`, the one with a share on an axis.

    Note that each figure is handed one strategy's slice, which is what makes the percentile
    axis of :func:`plot_agent_human_ratio_matrix` a within-strategy ranking.
    """
    if "strategy" not in optima.columns:
        raise KeyError("'strategy' is not a column of the optima table")

    # The human-only baseline is split out *before* the loop: it is a reference level
    # overlaid on each strategy's figures, not a strategy with a figure set of its own.
    is_baseline = optima["strategy"] == "human_only"
    baseline = optima[is_baseline]
    searches = optima[~is_baseline]
    if baseline.empty:
        baseline = None

    for strategy in sorted(searches["strategy"].unique()):
        sub = searches[searches["strategy"] == strategy]
        suffix = f", {strategy} search ({len(sub)} scenarios)"
        if baseline is not None:
            suffix += f" + {len(baseline)} human-only baselines"
        print(f"\n── Optima matrices: {strategy} ({len(sub)} scenarios) ──")

        fig = plot_optima_matrix(sub, share=share, baseline=baseline,
                                 title=f"Reported optima{suffix}", **channels)
        save_figure(fig, out_dir, f"optima_matrix_{strategy}", fmt)

        fig = plot_human_agent_matrix(
            sub, baseline=baseline,
            title=f"Developer vs. agent cost at $c^\\star${suffix}", **channels,
        )
        save_figure(fig, out_dir, f"human_agent_matrix_{strategy}", fmt)

        fig = plot_quality_development_matrix(
            sub, baseline=baseline, **channels,
            title=f"Escaped-bug vs. development cost, developers + agents, "
                  f"at $c^\\star${suffix}",
        )
        save_figure(fig, out_dir, f"quality_development_matrix_{strategy}", fmt)

        # No baseline here: a team with no agents has no agent/developer ratio, and 0 on a
        # five-decade log axis is not a position the figure can honestly give it.
        fig = plot_agent_human_ratio_matrix(
            sub, title=f"Agent-to-developer cost ratio at $c^\\star${suffix}", **channels,
        )
        save_figure(fig, out_dir, f"agent_human_ratio_matrix_{strategy}", fmt)
