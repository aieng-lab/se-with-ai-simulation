"""Visualization functions for sweep results.

All functions accept plain Python dicts / pandas DataFrames derived from the persisted
result files and return matplotlib Figure objects. Callers decide how to save or display.

Public API — plot functions
---------------------------
    plot_pareto_scatter(objectives, pareto_keys, title)          -> Figure
    plot_pareto_3d_scatter(objectives, pareto_keys, title)       -> Figure
    plot_objectives_bar(objectives, pareto_keys, title)          -> Figure
    plot_replication_distributions(per_config_replications, pareto_keys, title) -> Figure
    plot_trajectories(traces_per_config, pareto_keys, fields, title)    -> Figure
    plot_comparison_pareto(scenarios, title)                     -> Figure

Public API — campaign-level plots (paper §5)
--------------------------------------------
These take the derived frontier DataFrame of `src.results_analysis`, not the
per-search `objectives` dicts.

    plot_quality_cost_panels(df, colour_by, title)               -> Figure
    plot_quality_cost_facets(df, group, title)                   -> Figure
    plot_quality_cost_matrix(df, x_group, y_group, size_group,
                             marker_group, color_group, cost_field, title) -> Figure
    plot_ratio_percentile_matrix(df, x_group, y_group, size_group,
                                 marker_group, color_group, cost_field, title) -> Figure
    plot_value_saturation(df, threshold, title)                  -> Figure

The last two share `_draw_matrix`, which owns the 3×3 five-channel layout; they differ
only in the pair of columns they put on the axes.  Both put the combined
`quality_cost_index` on y — the *product* of the two minimised objectives, since their
ratio rises when cost improves and so cannot be read as a score.  The raw two-objective
view stays in `plot_quality_cost_panels` / `plot_quality_cost_facets`.

Public API — team composition plots (paper §5)
----------------------------------------------
These take the throughput-filtered front and derive their tables through
`src.composition_analysis`.  `phase` is "dev", "qa" or "both".

    plot_composition_archetypes(df, phase, top_k, title)          -> Figure
    plot_composition_drift(df, phase, bins, title)                -> Figure
    plot_composition_tier_heatmap(df, phase, title)               -> Figure

Public API — result-directory helpers
--------------------------------------
    visualize_single(result_dir, out_dir, fmt, mean_pareto)      -> None
    visualize_multi(result_dirs, out_dir, fmt, mean_pareto)      -> None
    save_figure(fig, out_dir, name, fmt)                         -> None
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
from matplotlib.ticker import PercentFormatter

from .composition_analysis import (
    CONTROL_COLUMNS,
    DEFAULT_ARCHETYPES,
    DEFAULT_BINS,
    OTHER_ARCHETYPE,
    PHASES,
    add_composition_columns,
    archetypes,
    composition_by_front_position,
    composition_by_tier,
    phase_columns,
)
from .config import AGENT_ROLE_ABBREV, AGENT_ROLE_COUNT, SCENARIO_GROUPS
from .pareto import AggregatedObjectives, pareto_frontier
from .results_analysis import add_front_position, add_quality_cost_index

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure

# Shared palette
_PARETO_COLOR = cm.tab10(0)    # blue
_OTHER_COLOR  = "#aaaaaa"      # grey
_STYLE        = "seaborn-v0_8-whitegrid"

# Scale guards.  A §4.2 search evaluates up to 10,000 candidates, which is far past
# the point where per-point CI whiskers, per-point labels, or one bar per config stay
# legible (and, for the bar chart, past the point where the figure can be rendered
# at all: its width grows with the config count).
_ERRORBAR_LIMIT = 200          # non-Pareto points drawn with CI whiskers
_ANNOTATE_LIMIT = 25           # Pareto points labelled with their config key
_BAR_LIMIT      = 60           # bars in plot_objectives_bar

_OBJ_META = {
    # field_suffix → (axis label, better direction symbol, is_lower_better)
    "value":   ("value   (features / sprint)",  "↑", False),
    "quality": ("quality (weighted bugs / feature)", "↓", True),
    "cost":      ("CF (cost / feature, $)",       "↓", True),
}


def _sorted_keys(objectives: dict[str, dict]) -> list[str]:
    """Sort config keys by descending value_mean for consistent x-axis ordering."""
    return sorted(objectives, key=lambda k: objectives[k]["value_mean"], reverse=True)


# ---------------------------------------------------------------------------
# 1. Pareto scatter
# ---------------------------------------------------------------------------

def _means_and_errors(
    objectives: dict[str, dict], keys: list[str], okey: str
) -> "tuple[np.ndarray, np.ndarray]":
    """Return (means, 2×n asymmetric CI errors) for `keys` on objective `okey`."""
    means = np.array([objectives[k][f"{okey}_mean"] for k in keys])
    lo = np.array([objectives[k][f"{okey}_ci"][0] for k in keys])
    hi = np.array([objectives[k][f"{okey}_ci"][1] for k in keys])
    return means, np.vstack([means - lo, hi - means])


def plot_pareto_scatter(
    objectives: dict[str, dict],
    pareto_keys: list[str],
    title: str = "",
) -> "Figure":
    """Three-panel 2-D scatter for every pair of objectives.

    Non-Pareto configs: grey circles.  Pareto configs: blue, larger.
    Error bars show 95% bootstrap CIs.  A §4.2-scale search evaluates ~10,000
    candidates, so the non-Pareto cloud is drawn as one vectorised scatter (and
    without CI whiskers, which would be solid ink at that density) once it exceeds
    _ERRORBAR_LIMIT points; annotations are dropped beyond _ANNOTATE_LIMIT.
    """
    pairs = [("value", "quality"), ("value", "cost"), ("quality", "cost")]
    pareto_set = set(pareto_keys)
    front = [k for k in objectives if k in pareto_set]
    rest = [k for k in objectives if k not in pareto_set]
    rest_errorbars = len(rest) <= _ERRORBAR_LIMIT
    annotate = len(front) <= _ANNOTATE_LIMIT

    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        if title:
            fig.suptitle(title, fontsize=12, y=1.01)

        for ax, (xkey, ykey) in zip(axes, pairs):
            xlabel, xarrow, x_lower_better = _OBJ_META[xkey]
            ylabel, yarrow, y_lower_better = _OBJ_META[ykey]

            for keys, color, size, zorder in ((rest, _OTHER_COLOR, 5, 2),
                                              (front, _PARETO_COLOR, 8, 3)):
                if not keys:
                    continue
                xm, xe = _means_and_errors(objectives, keys, xkey)
                ym, ye = _means_and_errors(objectives, keys, ykey)
                if keys is rest and not rest_errorbars:
                    ax.scatter(xm, ym, s=size ** 2 / 4, color=color, zorder=zorder,
                               alpha=0.35, linewidths=0)
                    continue
                ax.errorbar(xm, ym, xerr=xe, yerr=ye, fmt="o", color=color,
                            markersize=size, linewidth=0.8, capsize=2, zorder=zorder,
                            label=None)

            if annotate:
                for key in front:
                    ax.annotate(
                        key, (objectives[key][f"{xkey}_mean"], objectives[key][f"{ykey}_mean"]),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=6.5, color="black",
                    )

            ax.set_xlabel(f"{xarrow}  {xlabel}", fontsize=9)
            ax.set_ylabel(f"{yarrow}  {ylabel}", fontsize=9)

        # Shared legend
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=_PARETO_COLOR, markersize=8, label="Pareto"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=_OTHER_COLOR,  markersize=5, label="Other"),
        ]
        fig.legend(handles=handles, loc="upper right", fontsize=8)
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Three-objective scatter (V × Q × CF in one panel)
# ---------------------------------------------------------------------------

def plot_pareto_3d_scatter(
    objectives: dict[str, dict],
    pareto_keys: list[str],
    title: str = "",
) -> "Figure":
    """Single scatter encoding all three objectives: V (x), Q (y), CF (colour).

    Colour map: green = low CF (good) → red = high CF (bad).
    Pareto configs: large filled circles with a black edge + config-key label.
    Non-Pareto configs: small, semi-transparent circles, no label.
    95 % CI error bars are drawn for Pareto configs only to keep the plot readable.
    """
    pareto_set = set(pareto_keys)

    all_cf = [obj["cost_mean"] for obj in objectives.values()]
    cf_min, cf_max = min(all_cf), max(all_cf)
    norm = matplotlib.colors.Normalize(vmin=cf_min, vmax=cf_max)
    cmap = matplotlib.cm.get_cmap("RdYlGn_r")   # low CF → green (good), high CF → red

    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=(8, 6))
        if title:
            ax.set_title(title, fontsize=11)

        # Non-Pareto background layer, drawn in one vectorised call so that the
        # ~10,000 candidates of a §4.2 search stay cheap to render
        rest = [k for k in objectives if k not in pareto_set]
        if rest:
            ax.scatter(
                [objectives[k]["value_mean"] for k in rest],
                [objectives[k]["quality_mean"] for k in rest],
                c=[objectives[k]["cost_mean"] for k in rest], cmap=cmap, norm=norm,
                s=40, alpha=0.35, linewidths=0, zorder=2,
            )

        # Pareto configs with error bars and labels (foreground layer)
        annotate = len(pareto_keys) <= _ANNOTATE_LIMIT
        for key in pareto_keys:
            obj = objectives[key]
            xm, ym = obj["value_mean"], obj["quality_mean"]
            xci, yci = obj["value_ci"], obj["quality_ci"]
            color = cmap(norm(obj["cost_mean"]))

            ax.errorbar(
                xm, ym,
                xerr=[[xm - xci[0]], [xci[1] - xm]],
                yerr=[[ym - yci[0]], [yci[1] - ym]],
                fmt="none", color="black", linewidth=0.7, capsize=2, zorder=3,
            )
            ax.scatter(xm, ym, color=color, s=120, edgecolors="black",
                       linewidths=0.9, zorder=4)
            if annotate:
                ax.annotate(key, (xm, ym), textcoords="offset points",
                            xytext=(5, 4), fontsize=7)

        sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, pad=0.02)
        cb.set_label("↓  CF  (cost / feature, $)", fontsize=9)

        ax.set_xlabel("→  V  (features / sprint)", fontsize=9)
        ax.set_ylabel("↓  Q  (bugs / feature)",    fontsize=9)

        # Legend distinguishing Pareto from others
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
                   markeredgecolor="black", markersize=9, label="Pareto"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
                   markersize=6, alpha=0.4, label="Other"),
        ]
        ax.legend(handles=handles, fontsize=8, loc="upper right")
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Objectives bar chart
# ---------------------------------------------------------------------------

def plot_objectives_bar(
    objectives: dict[str, dict],
    pareto_keys: list[str],
    title: str = "",
    max_bars: int = _BAR_LIMIT,
) -> "Figure":
    """Three-row bar chart, one per objective, configs sorted by descending V.

    Restricted to at most `max_bars` configs: all Pareto-optimal ones first, then the
    highest-value remainder.  One bar per candidate is unreadable — and the figure
    unrenderable — at the ~10,000 candidates a §4.2 search evaluates.
    """
    pareto_set = set(pareto_keys)
    keys = _sorted_keys(objectives)
    if len(keys) > max_bars:
        front = [k for k in keys if k in pareto_set][:max_bars]
        rest = [k for k in keys if k not in pareto_set][:max_bars - len(front)]
        keys = [k for k in keys if k in set(front) | set(rest)]
        if title:
            title = f"{title}  (top {len(keys)} of {len(objectives)})"
    x = np.arange(len(keys))
    cols = [_PARETO_COLOR if k in pareto_set else _OTHER_COLOR for k in keys]

    obj_fields = [("value", "value_mean", "value_ci"), ("quality", "quality_mean", "quality_ci"), ("cost", "cost_mean", "cost_ci")]

    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(3, 1, figsize=(max(10, len(keys) * 0.9), 9), sharex=True)
        if title:
            fig.suptitle(title, fontsize=12)

        for ax, (okey, mean_field, ci_field) in zip(axes, obj_fields):
            label, arrow, _ = _OBJ_META[okey]
            means  = [objectives[k][mean_field] for k in keys]
            ci_lo  = [objectives[k][ci_field][0] for k in keys]
            ci_hi  = [objectives[k][ci_field][1] for k in keys]
            yerr   = [
                [m - lo for m, lo in zip(means, ci_lo)],
                [hi - m for m, hi in zip(means, ci_hi)],
            ]
            bars = ax.bar(x, means, color=cols, yerr=yerr, capsize=3,
                          error_kw={"linewidth": 0.8}, width=0.6, zorder=3)
            ax.set_ylabel(f"{arrow}  {label}", fontsize=8)
            ax.tick_params(axis="y", labelsize=7)

        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(keys, rotation=40, ha="right", fontsize=7)

        # Legend
        from matplotlib.patches import Patch
        fig.legend(
            handles=[Patch(facecolor=_PARETO_COLOR, label="Pareto"),
                     Patch(facecolor=_OTHER_COLOR,  label="Other")],
            loc="upper right", fontsize=8,
        )
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Replication distributions (box plots)
# ---------------------------------------------------------------------------

def plot_replication_distributions(
    per_config_replications: dict[str, list[tuple[float, float, float]]],
    pareto_keys: list[str],
    title: str = "",
    pareto_only: bool = True,
) -> "Figure":
    """Box plots of per-replication V, Q, CF values for each config.

    With `pareto_only=True` (default) only Pareto-frontier configs are shown,
    keeping the chart readable when the full pool contains hundreds of candidates.
    """
    pareto_set = set(pareto_keys)
    pool = (
        {k: v for k, v in per_config_replications.items() if k in pareto_set}
        if pareto_only
        else per_config_replications
    )
    keys = sorted(pool, key=lambda k: np.mean([r[0] for r in pool[k]]), reverse=True)
    x = np.arange(len(keys))
    obj_labels = [("V  (features/sprint)", "↑"), ("Q  (bugs/feature)", "↓"), ("CF  (cost/feature, $)", "↓")]

    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(3, 1, figsize=(max(8, len(keys) * 1.1), 9), sharex=True)
        if title:
            fig.suptitle(title, fontsize=12)

        for ax, idx, (ylabel, arrow) in zip(axes, [0, 1, 2], obj_labels):
            data = [[r[idx] for r in pool[k]] for k in keys]
            bp = ax.boxplot(
                data,
                positions=x,
                widths=0.5,
                patch_artist=True,
                whis=[5, 95],
                showfliers=False,
                medianprops={"color": "white", "linewidth": 1.5},
            )
            for patch, key in zip(bp["boxes"], keys):
                patch.set_facecolor(_PARETO_COLOR if key in pareto_set else _OTHER_COLOR)
                patch.set_alpha(0.85)
            ax.set_ylabel(f"{arrow}  {ylabel}", fontsize=8)
            ax.tick_params(axis="y", labelsize=7)

        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(keys, rotation=40, ha="right", fontsize=7)

        from matplotlib.patches import Patch
        fig.legend(
            handles=[Patch(facecolor=_PARETO_COLOR, label="Pareto"),
                     Patch(facecolor=_OTHER_COLOR,  label="Other")],
            loc="upper right", fontsize=8,
        )
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. State trajectories
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
    pareto_keys: list[str],
    fields: "list[str] | None" = None,
    title: str = "",
) -> "Figure":
    """Mean trajectory per sprint for Pareto configs, with a shaded quantile band.

    Each trace DataFrame must have a (replication, sprint) MultiIndex and columns for
    each state field. B_kno is derived as B_kno_min + B_kno_maj if both are present.
    """
    import pandas as pd

    if fields is None:
        fields = _DEFAULT_TRAJECTORY_FIELDS

    pareto_set = set(pareto_keys)
    palette = cm.tab10
    config_colors = {k: palette(i % 10) for i, k in enumerate(pareto_set)}

    with plt.style.context(_STYLE):
        nfields = len(fields)
        fig, axes = plt.subplots(nfields, 1, figsize=(10, 3 * nfields), sharex=True)
        if nfields == 1:
            axes = [axes]
        if title:
            fig.suptitle(title, fontsize=12)

        for key, df in traces_per_config.items():
            if key not in pareto_set:
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
# 5. Cross-experiment Pareto comparison scatter
# ---------------------------------------------------------------------------

def plot_comparison_pareto(
    scenarios: "dict[str, tuple[dict, list[str]]]",
    title: str = "",
) -> "Figure":
    """Three-panel 2-D scatter comparing Pareto frontiers across scenarios.

    Each scenario is a different colour; non-Pareto configs are shown faintly.
    """
    pairs = [("value", "quality"), ("value", "cost"), ("quality", "cost")]
    palette = cm.tab10
    scenario_colors = {label: palette(i % 10) for i, label in enumerate(scenarios)}

    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        if title:
            fig.suptitle(title, fontsize=12, y=1.01)

        for ax, (xkey, ykey) in zip(axes, pairs):
            xlabel, xarrow, _ = _OBJ_META[xkey]
            ylabel, yarrow, _ = _OBJ_META[ykey]

            for label, (objectives, pareto_keys) in scenarios.items():
                color = scenario_colors[label]
                pareto_set = set(pareto_keys)

                for key, obj in objectives.items():
                    xm  = obj[f"{xkey}_mean"]
                    ym  = obj[f"{ykey}_mean"]
                    xci = obj[f"{xkey}_ci"]
                    yci = obj[f"{ykey}_ci"]
                    xe  = [[xm - xci[0]], [xci[1] - xm]]
                    ye  = [[ym - yci[0]], [yci[1] - ym]]
                    is_pareto = key in pareto_set

                    ax.errorbar(
                        xm, ym,
                        xerr=xe, yerr=ye,
                        fmt="o",
                        color=color,
                        alpha=0.9 if is_pareto else 0.25,
                        markersize=7 if is_pareto else 4,
                        linewidth=0.6,
                        capsize=2,
                        zorder=3 if is_pareto else 2,
                    )

            ax.set_xlabel(f"{xarrow}  {xlabel}", fontsize=9)
            ax.set_ylabel(f"{yarrow}  {ylabel}", fontsize=9)

        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=scenario_colors[lbl], markersize=8, label=lbl)
            for lbl in scenarios
        ]
        fig.legend(handles=handles, loc="upper right", fontsize=8, title="Scenario")
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Campaign-level plots (paper §5)
# ---------------------------------------------------------------------------

# The three cost views §5 asks for: total, human developers only, agents only.
_COST_PANELS: tuple[tuple[str, str], ...] = (
    ("cost_mean",       "total cost / feature ($)"),
    ("cost_human_mean", "human developer cost / feature ($)"),
    ("cost_agent_mean", "agent cost / feature ($)"),
)

# Ordered so the legend reads low → medium → high rather than alphabetically.
_TIER_ORDER = ("low", "medium", "high")
_TIER_COLORS = {"low": cm.tab10(0), "medium": cm.tab10(1), "high": cm.tab10(3)}

# y-axis label shared by the two matrix figures: the product of the two minimised
# objectives, both of which already carry a per-feature denominator.
_INDEX_LABEL = "↓  quality × cost index (bugs · $ / feature²)"

# Marker shape per tier, for the 3×3 matrix plot.  A cross, an open circle and a filled
# triangle stay apart from each other even when overplotted at small sizes; three filled
# shapes do not.
_TIER_MARKERS = {"low": "x", "medium": "o", "high": "^"}
_TIER_MARKER_FILLED = {"low": False, "medium": False, "high": True}

# Marker area (pt²) per tier.  A 20× spread from low to high (~4.5× in marker width) so
# the size channel survives the alpha blending of a dense front.
_TIER_SIZES = {"low": 3.0, "medium": 16.0, "high": 60.0}


def _tier_groups(df: "pd.DataFrame", column: str) -> "list[tuple[str, pd.DataFrame]]":
    """(tier, subframe) pairs in low→medium→high order, skipping empty tiers.

    The campaign may be partial, so a tier can be absent entirely; callers must
    not assume all three are present.
    """
    return [
        (tier, df[df[column] == tier])
        for tier in _TIER_ORDER
        if (df[column] == tier).any()
    ]


def _log_floor(df: "pd.DataFrame", column: str = "quality_mean") -> float | None:
    """Where non-positive values are drawn on a log axis: half the smallest positive.

    Returned as one number for the whole table so the same configuration lands at the
    same height in every panel of a faceted figure; `None` when nothing needs clipping.
    """
    y = df[column].to_numpy(dtype=float)
    positive = y[y > 0]
    if positive.size and (y <= 0).any():
        return float(positive.min()) * 0.5
    return None


def _scatter_quality_cost(
    ax,
    df: "pd.DataFrame",
    cost_field: str,
    color,
    label: str | None = None,
    floor: float | None = None,
) -> None:
    """One quality-vs-cost point cloud on `ax`, log–log.

    `floor` overrides the y-position given to non-positive quality values; omit it to
    derive one from `df` alone (fine for a single-panel plot).
    """
    x = df[cost_field].to_numpy(dtype=float)
    y = df["quality_mean"].to_numpy(dtype=float)
    # A frontier point can have zero escaped bugs; log scale drops those, so
    # clip them to just under the smallest positive value to keep them visible.
    if floor is None:
        floor = _log_floor(df)
    if floor is not None:
        y = np.where(y > 0, y, floor)
    # Marker size tracks the point count: the throughput-filtered fronts of §5 are
    # an order of magnitude smaller than the full recorded frontiers, and markers
    # tuned for a 36,000-point cloud vanish on a 10,000-point one.
    n = len(x)
    size, alpha = (18, 0.55) if n <= 2_000 else (10, 0.45) if n <= 15_000 else (6, 0.35)
    ax.scatter(x, y, s=size, alpha=alpha, color=color, label=label, linewidths=0)


def plot_quality_cost_panels(
    df: "pd.DataFrame",
    colour_by: str | None = None,
    title: str = "",
) -> "Figure":
    """Quality against total / human / agent cost per feature — the three §5 plots.

    `df` is the derived frontier table of `src.results_analysis`.  Axes are
    log–log because both objectives span several orders of magnitude across the
    bugimpact and project tiers.  `colour_by` names a tier column (e.g.
    ``"tier_debt"``); when omitted all points share one colour.
    """
    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
        if title:
            fig.suptitle(title, fontsize=12, y=1.02)

        for ax, (field, xlabel) in zip(axes, _COST_PANELS):
            if colour_by and colour_by in df.columns:
                for tier, sub in _tier_groups(df, colour_by):
                    _scatter_quality_cost(ax, sub, field, _TIER_COLORS[tier], tier)
            else:
                _scatter_quality_cost(ax, df, field, _PARETO_COLOR)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(f"↓  {xlabel}", fontsize=9)

        axes[0].set_ylabel("↓  quality (weighted escaped bugs / feature)", fontsize=9)
        if colour_by and colour_by in df.columns:
            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="upper right", fontsize=8,
                           title=colour_by.replace("tier_", "") + " tier",
                           markerscale=3)
        fig.tight_layout()
    return fig


def plot_quality_cost_facets(
    df: "pd.DataFrame",
    group: str,
    title: str = "",
) -> "Figure":
    """The three quality-vs-cost panels, faceted by the tier of one scenario group.

    Rows are the tiers of `group` (low/medium/high, only those present in `df`),
    columns are the total / human / agent cost views.  All panels share axes so
    the tiers can be compared directly.
    """
    column = f"tier_{group}"
    if column not in df.columns:
        raise KeyError(f"{column!r} is not a column of the frontier table")
    tiers = _tier_groups(df, column)
    if not tiers:
        raise ValueError(f"no rows for any tier of {group!r}")

    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(
            len(tiers), 3,
            figsize=(15, 4.2 * len(tiers)),
            sharex="col", sharey=True,
            squeeze=False,
        )
        fig.suptitle(title or f"Quality vs. cost by {group} tier", fontsize=12, y=1.0)

        for row, (tier, sub) in enumerate(tiers):
            for col, (field, xlabel) in enumerate(_COST_PANELS):
                ax = axes[row][col]
                _scatter_quality_cost(ax, sub, field, _TIER_COLORS[tier])
                ax.set_xscale("log")
                ax.set_yscale("log")
                if row == len(tiers) - 1:
                    ax.set_xlabel(f"↓  {xlabel}", fontsize=9)
                if col == 0:
                    ax.set_ylabel(f"{group} = {tier}\n↓  quality (bugs / feature)", fontsize=9)
                ax.set_title(f"n = {len(sub)}", fontsize=8, loc="right")
        fig.tight_layout()
    return fig


def _marker_kwargs(marker_tier: str, color, area: float) -> dict:
    """Scatter keywords drawing `marker_tier`'s shape in `color` at `area` pt².

    The cross and the open circle are strokes, so they need a line width that grows
    with the marker; the filled triangle must have none or the edge swallows it at the
    smallest size tier.
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


def _check_matrix_groups(df: "pd.DataFrame", channels: "dict[str, str]") -> None:
    """Validate the five channel→group assignments of `plot_quality_cost_matrix`."""
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
        raise KeyError(f"{missing} are not columns of the frontier table")


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
    y_floor: float | None = None,
) -> "Figure":
    """The 3×3 five-channel grid, for any pair of columns.

    Shared by :func:`plot_quality_cost_matrix` and
    :func:`plot_ratio_percentile_matrix`: the facet layout, the colour/shape/size
    encoding of the three non-facet groups and the three proxy legends live here and
    nowhere else.  Callers validate the channel assignment (`_check_matrix_groups`)
    and supply the two plot columns; `y_floor` clips non-positive `y_col` values onto
    a log axis (see :func:`_log_floor`).
    """
    columns = _tier_groups(df, f"tier_{x_group}")
    rows = _tier_groups(df, f"tier_{y_group}")
    if not columns or not rows:
        raise ValueError("no rows for any tier of the faceting groups")

    floor = y_floor

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
                for size_tier in reversed(_TIER_ORDER):
                    for marker_tier in _TIER_ORDER:
                        for color_tier in _TIER_ORDER:
                            sub = cell[
                                (cell[f"tier_{size_group}"] == size_tier)
                                & (cell[f"tier_{marker_group}"] == marker_tier)
                                & (cell[f"tier_{color_group}"] == color_tier)
                            ]
                            if sub.empty:
                                continue
                            y = sub[y_col].to_numpy(dtype=float)
                            if floor is not None:
                                y = np.where(y > 0, y, floor)
                            area = _TIER_SIZES[size_tier] * scale
                            ax.scatter(
                                sub[x_col].to_numpy(dtype=float), y,
                                s=area, alpha=alpha,
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
                       color=_TIER_COLORS[t], label=t) for t in _TIER_ORDER
            ]),
            (marker_group, [
                Line2D([], [], linestyle="none", marker=_TIER_MARKERS[t], markersize=7,
                       color="0.35", markeredgecolor="0.35", label=t,
                       markerfacecolor="0.35" if _TIER_MARKER_FILLED[t] else "none")
                for t in _TIER_ORDER
            ]),
            (size_group, [
                Line2D([], [], linestyle="none", marker="o", color="0.35", label=t,
                       markersize=float(np.sqrt(_TIER_SIZES[t])) * 1.5)
                for t in _TIER_ORDER
            ]),
        )
        fig.tight_layout(rect=(0.0, 0.0, 0.86, 1.0))
        for i, (group, handles) in enumerate(legend_specs):
            fig.legend(
                handles=handles, loc="upper left", frameon=True, fontsize=8,
                title=f"{group} (" + ("colour", "shape", "size")[i] + ")",
                title_fontsize=8, labelspacing=0.9,
                bbox_to_anchor=(0.875, 0.92 - 0.20 * i),
            )
    return fig


def plot_quality_cost_matrix(
    df: "pd.DataFrame",
    x_group: str = "agent",
    y_group: str = "human",
    size_group: str = "bugimpact",
    marker_group: str = "project",
    color_group: str = "debt",
    cost_field: str = "cost_mean",
    title: str = "",
) -> "Figure":
    """The combined quality×cost index against cost, all five groups at once.

    The 3×3 grid of panels facets the tiers of `x_group` (columns) against those of
    `y_group` (rows); within a panel the remaining three groups are encoded as marker
    colour, marker shape and marker size.  Every point of `df` is drawn, so pass the
    front of a *single* search strategy — mixing strategies mixes two point clouds with
    very different densities into the same panel.

    y is `quality_cost_index` (:func:`src.results_analysis.add_quality_cost_index`),
    the product of the two minimised objectives — the ratio would *rise* when cost
    improves and so cannot be read as a score.  x is `cost_field`; since the index
    contains cost, the panel is the quality-vs-cost plane sheared by 45°, which is
    what makes the index directly readable per cost level.  The raw two-objective
    view lives in :func:`plot_quality_cost_panels` / :func:`plot_quality_cost_facets`.

    `df` is the throughput-filtered front of `src.results_analysis` (or any table with
    the same columns).  Axes are log–log and shared across all panels so the tiers can
    be compared directly.
    """
    channels = {
        "x_group": x_group, "y_group": y_group, "size_group": size_group,
        "marker_group": marker_group, "color_group": color_group,
    }
    _check_matrix_groups(df, channels)
    if cost_field not in df.columns:
        raise KeyError(f"{cost_field!r} is not a column of the frontier table")

    work = add_quality_cost_index(df, cost_field=cost_field)
    return _draw_matrix(
        work,
        x_group=x_group, y_group=y_group, size_group=size_group,
        marker_group=marker_group, color_group=color_group,
        x_col=cost_field, y_col="quality_cost_index",
        xlabel="↓  " + dict(_COST_PANELS).get(cost_field, cost_field),
        ylabel=_INDEX_LABEL,
        title=title or f"Quality×cost index vs. cost — {x_group} × {y_group}",
        y_floor=_log_floor(work, "quality_cost_index"),
    )


def plot_ratio_percentile_matrix(
    df: "pd.DataFrame",
    x_group: str = "agent",
    y_group: str = "human",
    size_group: str = "bugimpact",
    marker_group: str = "project",
    color_group: str = "debt",
    cost_field: str = "cost_mean",
    title: str = "",
) -> "Figure":
    """The quality×cost index against position along the front — same 3×3 grid.

    A variant of :func:`plot_quality_cost_matrix` for the throughput-filtered front,
    where every point delivers the same number of features so the two remaining
    objectives collapse into the single index this also puts on y.

    x is the percentile of the quality/cost *ratio* within the point's own
    (scenario, strategy) front (:func:`src.results_analysis.add_front_position`).
    That ratio is an exchange rate, not a score — which is precisely what makes it a
    position coordinate: it sweeps the front from the quality-dominant end (0 %) to
    the cost-dominant end (100 %).  Each front therefore spans the full axis
    regardless of whether it holds 13 or 1,190 points.

    The resulting curve is U-shaped rather than monotone, and its minimum is the
    front's best combined point — interior on 484 of the 486 §4.2 fronts.  Facets and
    the three marker channels are identical to the quality/cost matrix, so the two
    figures read the same way; as there, pass the front of a single search strategy.
    """
    channels = {
        "x_group": x_group, "y_group": y_group, "size_group": size_group,
        "marker_group": marker_group, "color_group": color_group,
    }
    _check_matrix_groups(df, channels)
    if cost_field not in df.columns:
        raise KeyError(f"{cost_field!r} is not a column of the frontier table")

    work = add_front_position(
        add_quality_cost_index(df, cost_field=cost_field), cost_field=cost_field,
    )
    fig = _draw_matrix(
        work,
        x_group=x_group, y_group=y_group, size_group=size_group,
        marker_group=marker_group, color_group=color_group,
        x_col="front_pct", y_col="quality_cost_index",
        xlabel="front position  (quality end → cost end)",
        ylabel=_INDEX_LABEL,
        title=title or f"Quality×cost index along the front — {x_group} × {y_group}",
        xscale="linear",
        y_floor=_log_floor(work, "quality_cost_index"),
    )
    for ax in fig.get_axes():
        ax.set_xlim(0.0, 1.0)
        # The panels touch, so a tick at 1.0 would print "100%" right against the next
        # panel's "0%"; the axis runs to 100 % regardless.
        ax.set_xticks([0.0, 0.25, 0.5, 0.75])
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    return fig


def plot_value_saturation(
    df: "pd.DataFrame",
    threshold: float = 0.99,
    title: str = "",
) -> "Figure":
    """Distribution of `value / r_arrival` over the frontiers, per scenario group.

    Supports the §5 claim that value sits at the feature arrival rate: one box
    per tier per group, with reference lines at 1.0 and at `threshold`.
    """
    groups = [c for c in df.columns if c.startswith("tier_")]
    if not groups:
        raise KeyError("frontier table has no tier_* columns")

    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(1, len(groups), figsize=(3.2 * len(groups), 4.5),
                                 sharey=True, squeeze=False)
        fig.suptitle(title or "value / r_arrival across the Pareto frontiers",
                     fontsize=12, y=1.02)

        for ax, column in zip(axes[0], groups):
            tiers = _tier_groups(df, column)
            data = [sub["value_ratio"].to_numpy(dtype=float) for _, sub in tiers]
            labels = [t for t, _ in tiers]
            bp = ax.boxplot(data, showfliers=True, patch_artist=True,
                            widths=0.55, flierprops={"markersize": 1.5, "alpha": 0.3})
            # Set tick labels separately: boxplot's own keyword was renamed from
            # `labels` to `tick_labels` in matplotlib 3.9.
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels)
            for patch, tier in zip(bp["boxes"], labels):
                patch.set_facecolor(_TIER_COLORS[tier])
                patch.set_alpha(0.45)
            for median in bp["medians"]:
                median.set_color("black")
            ax.axhline(1.0, color="black", linewidth=0.8, zorder=0)
            ax.axhline(threshold, color="crimson", linewidth=0.8, linestyle="--", zorder=0)
            ax.set_xlabel(column.replace("tier_", ""), fontsize=9)

        axes[0][0].set_ylabel("value / r_arrival", fontsize=9)
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Team composition on the throughput-filtered front (paper §5)
# ---------------------------------------------------------------------------

# One colour per agent role, k = 1..8 in the order of AGENT_ROLE_NAMES.  Roles keep
# their colour across every composition figure, so a reader who learns the legend once
# can read the drift bars, the archetype bars and the heat map interchangeably.
_ROLE_COLORS = {k: cm.tab10((k - 1) % 10) for k in range(1, AGENT_ROLE_COUNT + 1)}

_PHASE_CHOICES: tuple[str, ...] = ("dev", "qa", "both")

# Long enough to read, short enough for a tick label; the full signature label runs to
# ~56 characters for the all-roles archetype.
_ARCHETYPE_LABEL_WRAP = 34


def _phase_list(phase: str) -> list[str]:
    """Resolve a `phase` argument to the phases to draw."""
    if phase not in _PHASE_CHOICES:
        raise ValueError(f"unknown phase {phase!r}; expected one of {list(_PHASE_CHOICES)}")
    return list(PHASES) if phase == "both" else [phase]


def _role_color(column: str):
    """Colour for a role column such as ``a_dev_5`` or ``share_a_qa_7``."""
    return _ROLE_COLORS[int(column.rsplit("_", 1)[1])]


def _strategy_order(df: "pd.DataFrame") -> list[str]:
    """Strategies present in `df`, evolutionary first (it is the denser, primary front)."""
    present = set(df["strategy"].unique())
    ordered = [s for s in ("evolutionary", "random") if s in present]
    return ordered + sorted(present - set(ordered))


def _require_single_strategy(df: "pd.DataFrame", func: str) -> None:
    """Guard the figures that draw one row per archetype / tier."""
    n = df["strategy"].nunique()
    if n > 1:
        raise ValueError(
            f"{func} draws one row per archetype and would duplicate them across "
            f"strategies; pass a single-strategy front (got {n}: "
            f"{sorted(df['strategy'].unique())})"
        )


def _wrap_label(text: str, width: int = _ARCHETYPE_LABEL_WRAP) -> str:
    """Break a `+`-joined archetype label onto as few lines as `width` allows."""
    parts, lines, current = text.split("+"), [], ""
    for part in parts:
        candidate = f"{current}+{part}" if current else part
        if current and len(candidate) > width:
            lines.append(current)
            current = part
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


def _stacked_shares(ax, y, table: "pd.DataFrame", columns: list[str], horizontal: bool) -> None:
    """Stack `columns` of `table` along `y`, one colour per role."""
    offset = np.zeros(len(table))
    for column in columns:
        width = np.nan_to_num(table[column].to_numpy(dtype=float))
        if horizontal:
            ax.barh(y, width, left=offset, height=0.72,
                    color=_role_color(column), edgecolor="white", linewidth=0.4)
        else:
            ax.bar(y, width, bottom=offset, width=0.72,
                   color=_role_color(column), edgecolor="white", linewidth=0.4)
        offset += width


def _role_legend_handles(phase: str) -> list:
    """Proxy handles naming every role that can appear in `phase`'s panels."""
    columns: list[str] = []
    for one in _phase_list(phase):
        columns.extend(phase_columns(one)[0])
    seen, handles = set(), []
    for column in sorted(columns, key=lambda c: int(c.rsplit("_", 1)[1])):
        k = int(column.rsplit("_", 1)[1])
        if k in seen:
            continue
        seen.add(k)
        handles.append(
            Line2D([], [], linestyle="none", marker="s", markersize=7,
                   color=_ROLE_COLORS[k], label=AGENT_ROLE_ABBREV[k - 1])
        )
    return handles


def plot_composition_archetypes(
    df: "pd.DataFrame",
    phase: str = "dev",
    top_k: int = DEFAULT_ARCHETYPES,
    title: str = "",
) -> "Figure":
    """The recurring team recipes on the front, as a quality-ordered ladder.

    Rows are the `top_k` most common role-presence archetypes (see
    :func:`src.composition_analysis.archetypes`) plus an `other` residue, ordered by mean
    quality so the ladder reads best-first.  The four panels answer, left to right: what
    the supervised mix looks like, how many unbounded background agents come with it,
    where the recipe lands in quality/cost space, and how much of the front — and how
    many scenarios — it accounts for.

    `df` must hold a single search strategy; pass `phase="both"` for one row-block per
    phase.  Supervised roles are shares of `n_sup`, background roles absolute counts —
    see the `src.composition_analysis` docstring for why the two are never mixed.
    """
    _require_single_strategy(df, "plot_composition_archetypes")
    phases = _phase_list(phase)

    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(
            len(phases), 4,
            figsize=(17, 0.9 + 4.4 * len(phases)),
            gridspec_kw={"width_ratios": [1.35, 0.6, 1.0, 0.7]},
            squeeze=False,
        )
        fig.suptitle(title or "Team composition archetypes on the filtered front",
                     fontsize=12, y=1.0)

        for row, one in enumerate(phases):
            table = archetypes(df, one, top_k=top_k)
            share_ax, bg_ax, scatter_ax, prev_ax = axes[row]
            if table.empty:
                for ax in axes[row]:
                    ax.set_axis_off()
                share_ax.set_title(f"no {one}-phase points", fontsize=9, loc="left")
                continue

            _, sup_cols, bg_cols = phase_columns(one)
            y = np.arange(len(table))
            colours = _archetype_colors(table)

            _stacked_shares(share_ax, y, table, [f"share_{c}" for c in sup_cols], True)
            share_ax.set_xlim(0, 1)
            share_ax.set_yticks(y)
            share_ax.set_yticklabels(
                [_wrap_label(lab) for lab in table["label"]], fontsize=7,
            )
            for tick, colour in zip(share_ax.get_yticklabels(), colours):
                tick.set_color(colour)
            # A QA archetype is often nothing but the two background services, so its
            # supervised bar is legitimately empty — say so rather than leave what looks
            # like a missing row.
            staffed = np.nan_to_num(
                table[[f"share_{c}" for c in sup_cols]].to_numpy(dtype=float)
            ).sum(axis=1)
            for yi, n_sup, filled in zip(y, table["n_sup"], staffed):
                share_ax.text(1.01, yi, f"n_sup {n_sup:.1f}", fontsize=6.5,
                              va="center", ha="left", color="0.35")
                if filled <= 0:
                    share_ax.text(0.02, yi, "no supervised agents", fontsize=6.5,
                                  va="center", ha="left", color="0.45", style="italic")

            _stacked_shares(bg_ax, y, table, bg_cols, True)
            bg_ax.set_yticks(y)
            bg_ax.set_yticklabels([])

            _scatter_archetypes(scatter_ax, df, one, table, colours)

            # Drawn below the axis flip, so prevalence ends up the upper of the pair —
            # matching the order the panel title names them in.
            prev_ax.barh(y - 0.19, table["prevalence"], height=0.34,
                         color=colours, edgecolor="none")
            prev_ax.barh(y + 0.19, table["scenario_coverage"], height=0.34,
                         color=colours, alpha=0.45, edgecolor="none")
            prev_ax.set_xlim(0, 1)
            prev_ax.set_yticks(y)
            prev_ax.set_yticklabels([])

            for ax in (share_ax, bg_ax, prev_ax):
                ax.invert_yaxis()

            if row == 0:
                share_ax.set_title("supervised mix (share of n_sup)", fontsize=9, loc="left")
                bg_ax.set_title("background agents", fontsize=9, loc="left")
                scatter_ax.set_title("quality vs. cost", fontsize=9, loc="left")
                prev_ax.set_title("prevalence (solid) / scenario coverage (pale)",
                                  fontsize=8, loc="left")
            if row == len(phases) - 1:
                share_ax.set_xlabel("share of supervised capacity", fontsize=9)
                bg_ax.set_xlabel("agents", fontsize=9)
                prev_ax.set_xlabel("fraction", fontsize=9)
            share_ax.set_ylabel(f"{one} phase", fontsize=10)

        handles = _role_legend_handles(phase)
        fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8,
                   frameon=True, title="agent role", title_fontsize=8,
                   bbox_to_anchor=(0.5, 0.0))
    return fig


def _archetype_colors(table: "pd.DataFrame") -> list:
    """Ladder colours for the archetype rows; the `other` residue stays grey."""
    real = [a != OTHER_ARCHETYPE for a in table["archetype"]]
    n_real = max(1, sum(real))
    ramp = cm.viridis(np.linspace(0.08, 0.88, n_real))
    out, i = [], 0
    for is_real in real:
        if is_real:
            out.append(ramp[i])
            i += 1
        else:
            out.append(_OTHER_COLOR)
    return out


def _scatter_archetypes(
    ax, df: "pd.DataFrame", phase: str, table: "pd.DataFrame", colours: list,
) -> None:
    """Where each archetype sits in quality/cost space.

    Colouring all ~80,000 front points by archetype produces an unreadable mush — the
    clouds overlap almost completely and the ladder disappears.  So the points go down
    once in grey for context and each archetype is summarised by its median with
    interquartile whiskers, which is what actually separates the recipes.
    """
    work = add_composition_columns(df, phase)
    kept = {a for a in table["archetype"] if a != OTHER_ARCHETYPE}
    work = work.assign(
        archetype=work["signature"].where(work["signature"].isin(kept), OTHER_ARCHETYPE)
    )
    floor = _log_floor(work)

    def _quality(frame):
        y = frame["quality_mean"].to_numpy(dtype=float)
        return np.where(y > 0, y, floor) if floor is not None else y

    scale, _ = _density_style(len(work))
    ax.scatter(work["cost_mean"].to_numpy(dtype=float), _quality(work),
               s=6 * scale, alpha=0.10, color=_OTHER_COLOR, linewidths=0, zorder=1)

    for archetype, colour in zip(table["archetype"], colours):
        sub = work[work["archetype"] == archetype]
        if sub.empty:
            continue
        x, y = sub["cost_mean"].to_numpy(dtype=float), _quality(sub)
        xm, ym = np.median(x), np.median(y)
        xq, yq = np.percentile(x, [25, 75]), np.percentile(y, [25, 75])
        ax.errorbar(
            xm, ym,
            xerr=[[max(xm - xq[0], 0)], [max(xq[1] - xm, 0)]],
            yerr=[[max(ym - yq[0], 0)], [max(yq[1] - ym, 0)]],
            marker="D", markersize=8, color=colour, markeredgecolor="black",
            markeredgewidth=0.8, elinewidth=2.0, capsize=3, zorder=5,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("↓  cost / feature ($)   ·   median, IQR whiskers", fontsize=9)
    ax.set_ylabel("↓  quality (bugs / feature)", fontsize=9)


def plot_composition_drift(
    df: "pd.DataFrame",
    phase: str = "dev",
    bins: int = DEFAULT_BINS,
    title: str = "",
) -> "Figure":
    """How the team recipe changes along the cost axis of the front.

    Points are ranked by cost *within* their own (scenario, strategy) front and cut into
    `bins` quantiles, so the cheap end of a cheap scenario is compared against the cheap
    end of an expensive one.  One row-block per (phase, strategy); the three panels are
    the supervised mix, the background agent counts, and the control variables.

    Control variables share a log axis because they span `F_freeze` fractions of a sprint
    up to tens of agents; each line is annotated with its endpoint values so the axis is
    only a shape cue.
    """
    phases = _phase_list(phase)
    strategies = _strategy_order(df)
    rows = [(one, strategy) for one in phases for strategy in strategies]
    if not rows:
        raise ValueError("no strategies present in the front table")

    control_columns = ("n_sup", "n_bg", "N", "R", "F_freeze", "agents_per_fte")

    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(
            len(rows), 3, figsize=(15, 3.9 * len(rows)), squeeze=False,
        )
        fig.suptitle(title or "Composition along the filtered front, cheapest → priciest",
                     fontsize=12, y=1.0)

        for row, (one, strategy) in enumerate(rows):
            share_ax, bg_ax, ctl_ax = axes[row]
            sub = df[df["strategy"] == strategy]
            table = composition_by_front_position(sub, one, bins=bins)
            if table.empty:
                for ax in axes[row]:
                    ax.set_axis_off()
                share_ax.set_title(f"no {one}-phase points for {strategy}",
                                   fontsize=9, loc="left")
                continue

            _, sup_cols, bg_cols = phase_columns(one)
            labels = [str(b) for b in table["front_bin"]]
            x = np.arange(len(table))

            _stacked_shares(share_ax, x, table, [f"share_{c}" for c in sup_cols], False)
            share_ax.set_ylim(0, 1)
            _stacked_shares(bg_ax, x, table, bg_cols, False)

            # Six series in one panel: a legend box would sit on top of them wherever
            # it went, so the lines are labelled directly at their left end and carry
            # their final value at the right.
            for i, column in enumerate(control_columns):
                values = table[column].to_numpy(dtype=float)
                values = np.where(values > 0, values, np.nan)
                colour = cm.tab10(i % 10)
                ctl_ax.plot(x, values, marker="o", markersize=3.5, linewidth=1.3,
                            color=colour, label=column)
                finite = np.flatnonzero(np.isfinite(values))
                if finite.size:
                    first, last = finite[0], finite[-1]
                    # Series that start at the same value (n_sup and F_freeze both sit
                    # near 1 on the cheap end) would print their names on top of each
                    # other, so nudge alternate labels apart.
                    ctl_ax.annotate(column, (x[first], values[first]),
                                    textcoords="offset points",
                                    xytext=(-5, 5 if i % 2 else -5),
                                    fontsize=6.5, va="center", ha="right", color=colour)
                    ctl_ax.annotate(f"{values[last]:.2f}", (x[last], values[last]),
                                    textcoords="offset points", xytext=(5, 0),
                                    fontsize=6.5, va="center", ha="left", color=colour)
            ctl_ax.set_yscale("log")
            ctl_ax.set_xlim(-1.15, len(x) - 0.35)

            for ax in axes[row]:
                ax.set_xticks(x)
                ax.set_xticklabels(labels, fontsize=7)
            share_ax.set_ylabel(f"{one} / {strategy}\nshare of n_sup", fontsize=9)
            bg_ax.set_ylabel("background agents", fontsize=9)
            ctl_ax.set_ylabel("control variables (log)", fontsize=9)
            if row == 0:
                share_ax.set_title("supervised mix", fontsize=9, loc="left")
                bg_ax.set_title("background agents", fontsize=9, loc="left")
                ctl_ax.set_title("team controls", fontsize=9, loc="left")
            if row == len(rows) - 1:
                for ax in axes[row]:
                    ax.set_xlabel("within-front cost quantile", fontsize=9)

        handles = _role_legend_handles(phase)
        fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8,
                   frameon=True, title="agent role", title_fontsize=8,
                   bbox_to_anchor=(0.5, 0.0))
    return fig


def plot_composition_tier_heatmap(
    df: "pd.DataFrame",
    phase: str = "dev",
    title: str = "",
) -> "Figure":
    """Which scenario conditions push which roles.

    Rows are the fifteen scenario tiers (five groups × low/medium/high), grouped and
    separated; the three column blocks are the supervised shares, the background agent
    counts and the control variables.

    Cell **colour** is the deviation from the mean of that quantity over its own group's
    three tiers, scaled by the largest absolute deviation in the column — so colour shows
    the tier effect, comparably across quantities whose magnitudes differ by orders.
    Cell **text** is the actual value.  Pass a single strategy.
    """
    _require_single_strategy(df, "plot_composition_tier_heatmap")
    phases = _phase_list(phase)

    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(
            len(phases), 3,
            figsize=(16, 0.6 + 5.4 * len(phases)),
            gridspec_kw={"width_ratios": [6, 2, 8]},
            squeeze=False,
        )
        # Both sit above the axes; bbox_inches="tight" at save time keeps them on canvas.
        fig.suptitle(title or "Composition by scenario tier", fontsize=12, y=1.05)
        fig.text(
            0.5, 1.01,
            "cell text = value   ·   colour = deviation from that group's three-tier "
            "mean, scaled by the column's largest deviation (red above, blue below)",
            ha="center", va="top", fontsize=8, color="0.35",
        )

        for row, one in enumerate(phases):
            table = composition_by_tier(df, one)
            if table.empty:
                for ax in axes[row]:
                    ax.set_axis_off()
                axes[row][0].set_title(f"no {one}-phase points", fontsize=9, loc="left")
                continue

            _, sup_cols, bg_cols = phase_columns(one)
            blocks = (
                ([f"share_{c}" for c in sup_cols], "supervised share of n_sup"),
                (list(bg_cols), "background agents"),
                (list(CONTROL_COLUMNS), "team controls"),
            )
            index = [(g, t) for g in SCENARIO_GROUPS for t in _TIER_ORDER]
            values = table.pivot_table(
                index=["group", "tier"], columns="quantity",
                values="value", aggfunc="mean", observed=True,
            ).reindex(index)
            deviations = table.pivot_table(
                index=["group", "tier"], columns="quantity",
                values="deviation", aggfunc="mean", observed=True,
            ).reindex(index)

            for col, (columns, heading) in enumerate(blocks):
                ax = axes[row][col]
                _tier_heatmap_block(
                    ax, values[columns], deviations[columns],
                    labels=[_quantity_label(c) for c in columns],
                    row_labels=[f"{g} · {t}" for g, t in index] if col == 0 else None,
                )
                ax.set_title(heading, fontsize=9, loc="left")
                if col == 0:
                    ax.set_ylabel(f"{one} phase", fontsize=10)

        fig.tight_layout()
    return fig


def _quantity_label(column: str) -> str:
    """Column heading for a heat-map cell block."""
    if column.startswith("share_a_") or column.startswith("a_"):
        return AGENT_ROLE_ABBREV[int(column.rsplit("_", 1)[1]) - 1]
    return column


def _tier_heatmap_block(
    ax,
    values: "pd.DataFrame",
    deviations: "pd.DataFrame",
    labels: list[str],
    row_labels: list[str] | None,
) -> None:
    """One column block of the tier heat map, coloured by per-column scaled deviation."""
    raw = deviations.to_numpy(dtype=float)
    scale = np.nanmax(np.abs(raw), axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    normed = np.clip(raw / scale, -1.0, 1.0)

    ax.imshow(normed, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7.5, rotation=40, ha="right")
    ax.set_yticks(range(len(values)))
    if row_labels is None:
        ax.set_yticklabels([])
    else:
        ax.set_yticklabels(row_labels, fontsize=7.5)

    shown = values.to_numpy(dtype=float)
    for i in range(shown.shape[0]):
        for j in range(shown.shape[1]):
            cell = shown[i, j]
            if not np.isfinite(cell):
                continue
            text = f"{cell:.3f}" if abs(cell) < 10 else f"{cell:.1f}"
            # White text only where the diverging map is genuinely dark.
            colour = "white" if abs(normed[i, j]) > 0.62 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=6, color=colour)

    # Separate the five scenario groups; every group contributes three tier rows.
    for boundary in range(3, len(values), 3):
        ax.axhline(boundary - 0.5, color="black", linewidth=1.1)
    ax.grid(False)


# ---------------------------------------------------------------------------
# Result-directory helpers
# ---------------------------------------------------------------------------

def _load_summary(result_dir: Path) -> dict:
    path = result_dir / "search_summary.json"
    if not path.exists():
        path = result_dir / "sweep_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"No search_summary.json in {result_dir}.")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_manifest(result_dir: Path) -> dict:
    path = result_dir / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _load_per_config_replications(
    result_dir: Path,
    config_keys: list[str],
) -> dict[str, list[tuple[float, float, float]]]:
    """Load per-replication (V, Q, CF) from each config's result.json."""
    out = {}
    for key in config_keys:
        rfile = result_dir / "samples" / f"{key}.json"
        if not rfile.exists():
            continue
        data = json.loads(rfile.read_text(encoding="utf-8"))
        out[key] = [(r["value"], r["quality"], r["cost"]) for r in data["replications"]]
    return out


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


def _recompute_mean_pareto(objectives: dict) -> list[str]:
    """Recompute Pareto frontier using mean values only (no CI tie-breaking)."""
    keys = list(objectives.keys())
    points = [
        AggregatedObjectives(
            value_mean=obj["value_mean"],
            quality_mean=obj["quality_mean"],
            cost_mean=obj["cost_mean"],
            value_ci=tuple(obj["value_ci"]),
            quality_ci=tuple(obj["quality_ci"]),
            cost_ci=tuple(obj["cost_ci"]),
        )
        for obj in objectives.values()
    ]
    indices = pareto_frontier(points, ci_aware=False)
    return [keys[i] for i in indices]


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


_save = save_figure   # internal alias, kept for the existing call sites below


def visualize_single(result_dir: Path, out_dir: Path, fmt: str, mean_pareto: bool = False) -> None:
    """Generate all plots for a single sweep result directory."""
    summary  = _load_summary(result_dir)
    manifest = _load_manifest(result_dir)
    label    = _scenario_label(result_dir, manifest)

    objectives  = summary["objectives"]
    pareto_keys = _recompute_mean_pareto(objectives) if mean_pareto else summary["pareto_frontier"]
    config_keys = list(objectives.keys())
    title_base  = f"Scenario: {label}"

    print(f"\n── {result_dir.name}  ({len(config_keys)} configs, {len(pareto_keys)} Pareto) ──")

    fig = plot_pareto_scatter(objectives, pareto_keys, title=f"{title_base} — Pareto scatter")
    _save(fig, out_dir, "pareto_scatter", fmt)

    fig = plot_pareto_3d_scatter(objectives, pareto_keys, title=f"{title_base} — V / Q / CF")
    _save(fig, out_dir, "pareto_3d_scatter", fmt)

    fig = plot_objectives_bar(objectives, pareto_keys, title=f"{title_base} — Objectives")
    _save(fig, out_dir, "objectives_bar", fmt)

    reps = _load_per_config_replications(result_dir, config_keys)
    if reps:
        fig = plot_replication_distributions(reps, pareto_keys, title=f"{title_base} — Replication distributions")
        _save(fig, out_dir, "replication_distributions", fmt)

    traces = _load_traces(result_dir, pareto_keys)
    if traces:
        fig = plot_trajectories(traces, pareto_keys, title=f"{title_base} — State trajectories (Pareto configs)")
        _save(fig, out_dir, "trajectories", fmt)
    else:
        warnings.warn(
            f"No traces.csv found in {result_dir}. Re-run the sweep with --traces to enable trajectory plots.",
            stacklevel=2,
        )


def visualize_multi(result_dirs: list[Path], out_dir: Path, fmt: str, mean_pareto: bool = False) -> None:
    """Generate per-experiment plots and a cross-experiment comparison."""
    scenarios: dict[str, tuple[dict, list[str]]] = {}
    for rd in result_dirs:
        manifest = _load_manifest(rd)
        summary  = _load_summary(rd)
        label    = _scenario_label(rd, manifest)
        objectives = summary["objectives"]
        pareto_keys = _recompute_mean_pareto(objectives) if mean_pareto else summary["pareto_frontier"]
        scenarios[label] = (objectives, pareto_keys)

    print(f"\n── Cross-experiment comparison ({len(scenarios)} scenarios) ──")
    fig = plot_comparison_pareto(scenarios, title="Pareto comparison across scenarios")
    _save(fig, out_dir, "comparison_pareto", fmt)
