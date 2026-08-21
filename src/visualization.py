"""Figures for search results and for a consolidated campaign.

Every function takes plain dicts or pandas DataFrames derived from the persisted result
files and returns a matplotlib Figure; the caller decides how to save it.

Per-search plots
----------------
    plot_cost_vs_value, plot_cost_split, plot_trajectories, plot_comparison_optima,
    plot_elite_turnover

Campaign plots
--------------
These take the ``optima.csv`` table of :mod:`src.campaign` rather than a per-search
`objectives` dict:

    plot_human_agent_matrix, plot_quality_development_matrix,
    plot_agent_human_ratio_matrix, plot_constellation_profile,
    plot_constellation_heatmap, plot_team_shape, plot_team_drivers

The first three share `_draw_matrix`, the five-channel tier grid, and differ only in the
pair of columns they put on the axes.  All but the ratio figure take an optional
`baseline` frame of zero-agent optima, drawn as an overlay in black.

Directory-level helpers
-----------------------
    visualize_single, visualize_multi, visualize_campaign, visualize_constellation,
    visualize_team_summary, save_figure
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe to override before pyplot import
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
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
    ALL_SCENARIO_TIERS,
    AUTONOMOUS_ROLE_INDICES,
    HUMAN_ONLY_MODE,
    SCENARIO_GROUPS,
    require_distinct_groups,
    require_group_pair,
    SUPERVISED_ROLE_INDICES,
    UNUSED_TIER,
)
from .constellation import (
    AGENT_COST_PARAMETERS,
    BILL_PARAMETERS,
    C_STAR,
    DEV_PARAMETERS,
    I_STAR,
    PARAMETER_BLOCKS,
    POPULATIONS,
)

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure

# Shared palette
_BEST_COLOR  = cm.tab10(3)     # red — c*
_KEEP_COLOR  = cm.tab10(0)     # blue — the indistinguishable set I*
_OK_COLOR    = "#88bb88"       # green-grey — admissible but distinguishably worse
_OTHER_COLOR = "#cccccc"       # grey — violates the throughput constraint
_CHANGE_COLOR = cm.tab10(2)    # green — cumulative change in c*'s cost
_STYLE       = "seaborn-v0_8-whitegrid"

#: Most bars plot_cost_split will draw; a search evaluates far more candidates than fit.
_BAR_LIMIT = 60


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

    Everything left of the floor is inadmissible.  `c*` and the indistinguishable set
    `I*` are highlighted, and `c*` carries the cost CI that `I*` membership is decided on.
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
    """Stacked bars of the three cost drivers, one bar per key.

    The three sum to `cost_mean` by construction.  Truncated at `_BAR_LIMIT` bars.
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

#: Fields plot_trajectories draws.  All but `B_kno` are persisted snapshot columns;
#: `B_kno` is summed from the three severities, matching `SystemState.B_kno`.
_DEFAULT_TRAJECTORY_FIELDS = ["F_completed", "D_tech", "B_kno", "N_doc"]
_B_KNO_COLUMNS = ("B_kno_min", "B_kno_maj", "B_kno_sec")
_TRAJECTORY_LABELS = {
    "F_completed": "Cumulative features\n(F_completed)",
    "D_tech":      "Technical debt\n(D_tech)",
    "B_kno":       "Known bugs\n(B_kno, all severities)",
    "N_doc":       "Documentation written\n(N_doc)",
}


def plot_trajectories(
    traces_per_config: "dict[str, pd.DataFrame]",
    highlight_keys: list[str],
    title: str = "",
) -> "Figure":
    """Mean trajectory per sprint for the given configs, with a shaded quantile band.

    Each trace DataFrame must have a (replication, sprint) MultiIndex and one column per
    state field.  `B_kno` is summed from the per-severity columns when it is absent.
    """
    fields = _DEFAULT_TRAJECTORY_FIELDS
    wanted = set(highlight_keys)
    config_colors = {k: cm.tab10(i % 10) for i, k in enumerate(wanted)}

    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(len(fields), 1, figsize=(10, 3 * len(fields)),
                                 sharex=True)
        if title:
            fig.suptitle(title, fontsize=12)

        for key, df in traces_per_config.items():
            if key not in wanted:
                continue
            color = config_colors[key]

            if "B_kno" not in df.columns and all(c in df.columns for c in _B_KNO_COLUMNS):
                df = df.assign(B_kno=df[list(_B_KNO_COLUMNS)].sum(axis=1))

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
            ax.set_ylabel(_TRAJECTORY_LABELS[field], fontsize=8)
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
        means = np.array([scenarios[key]["cost_mean"] for key in labels], dtype=float)
        lo = np.array([scenarios[key]["cost_ci"][0] for key in labels], dtype=float)
        hi = np.array([scenarios[key]["cost_ci"][1] for key in labels], dtype=float)
        admissible = [scenarios[key]["feasible_throughput"] for key in labels]

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
# 4b. Search convergence
# ---------------------------------------------------------------------------

def _drop_label(values: np.ndarray) -> str:
    """Legend label for the drop curve, saying so when the log axis cannot draw it.

    A drop that is zero throughout has nothing for a log axis to place, and matplotlib
    would omit the curve silently, leaving an unexplained legend entry.
    """
    label = "mean drop in $c^\\star$ cost"
    positive = np.asarray(values, dtype=float) > 0.0
    if not positive.any():
        return f"{label} (zero throughout — not drawable on a log axis)"
    return label


def _incumbent_cost_drop(
    turnover: "pd.DataFrame", generations: np.ndarray
) -> "np.ndarray | None":
    """`c*`'s mean drop since the previous generation, relative and unsigned.

    `|best_cost(g) / best_cost(g-1) - 1|` per search, then the mean over searches.  It is
    taken relative because the optima span four decades, and unsigned per generation so
    that it is a rate like the turnover.  Generation 1 has no predecessor and no value.

    Returns None when the column carries no usable value; the figure then keeps its
    turnover curves alone.
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
    turnover: "pd.DataFrame",
    title: str = "",
) -> "Figure":
    """Elite turnover and the incumbent's cost drop, against the generation index.

    Takes the long-format ``elite_turnover.csv`` of :mod:`src.campaign` — one row per
    (scenario, mode, generation) — and draws three curves aggregated over searches: in
    blue on the left axis the turnover, as a mean (solid) and a median (dotted); in green
    on a logarithmic right-hand axis the mean relative drop in `c*`'s cost.

    Both are unsigned rates per generation, but the drop runs one to two decades lower,
    which is why it gets its own logarithmic axis.  A drop that is zero in every
    generation cannot be drawn at all, and :func:`_drop_label` says so in the legend.

    Carries no tier channels: convergence is a property of the search, not of the grid.
    """
    needed = {"scenario_id", "mode", "generation", "tau", "best_cost"}
    missing = needed - set(turnover.columns)
    if missing:
        raise KeyError(f"turnover table is missing column(s) {sorted(missing)}")
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
# 5. The campaign in one figure: all five scenario groups at once
# ---------------------------------------------------------------------------

#: One visual style per tier, in mild -> extreme order so the legends read that way.
#: `veryhigh` and `pessimistic` share the fourth slot: they are the fourth tier of
#: `project` and of `agent` respectively, and no group holds both, so no channel ever has
#: to draw them side by side.  The four marker shapes stay apart when overplotted at
#: small sizes, and the areas are spaced by a roughly constant ratio in marker *width*.
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
#: varies; on a group it holds at UNUSED_TIER it falls back to the fixed star below.
#: Its colour is fixed on every layout — with shape and size shared, colour is the only
#: channel left that says "no agents".
_BASELINE_COLOR = "black"
_BASELINE_MARKER = "*"
_BASELINE_SIZE = 90.0
#: Wrapped: on one line this is the longest colour-legend entry and overruns the
#: fixed-width legend column.
_BASELINE_LABEL = "human-only\nbaseline"


def _percent_tick(value: float, _pos: int | None = None) -> str:
    """Tick label for a fraction on a log axis: ``10%``, ``1%``, ``0.1%``, …

    Keeps exactly as many decimals as the tick needs, since one fixed number of them
    cannot serve an axis spanning several decades.
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

    Matplotlib's default writes the sub-unit decades in scientific mathtext.
    """
    return f"{value:g}"


def _tiers_present(df: "pd.DataFrame", column: str) -> list[str]:
    """The tiers occurring in `column`, in mild→extreme order.

    The count varies per group, and a partial campaign may narrow it further.
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


def _clipped_xy(
    df: "pd.DataFrame",
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


def _padded_log_range(
    positive: np.ndarray,
    what: str,
    extra_decades_below: float = 0.0,
) -> tuple[float, float]:
    """A log-axis range covering `positive`, padded by 5 % of the span in log space.

    `extra_decades_below` widens the low end further, for a figure whose marks grow from
    the frame.  Raises when nothing positive is left to scale.
    """
    positive = positive[positive > 0]
    if positive.size == 0:
        raise ValueError(f"no positive values in {what}; cannot scale a log axis")
    lo, hi = float(positive.min()), float(positive.max())
    pad = 0.05 * max(np.log10(hi / lo), 0.4)
    return (10.0 ** (np.log10(lo) - pad - extra_decades_below),
            10.0 ** (np.log10(hi) + pad))


def _common_log_limits(
    df: "pd.DataFrame",
    columns: tuple[str, ...],
    floors: tuple[float | None, ...],
) -> tuple[float, float]:
    """One padded (min, max) covering every column of `columns`, for a log axis.

    Used where two columns share a unit and must therefore share a range; the `floors` of
    :func:`_log_floor` are included, since a clipped zero is drawn at its floor and would
    otherwise fall outside the limits.
    """
    parts = [df[c].to_numpy(dtype=float) for c in columns]
    parts.append(np.array([f for f in floors if f is not None], dtype=float))
    return _padded_log_range(np.concatenate(parts), str(list(columns)))


def _marker_kwargs(marker_tier: str, color, area: float) -> dict:
    """Scatter keywords drawing `marker_tier`'s shape in `color` at `area` pt².

    The cross and the open circle are strokes, so they need a line width that grows with
    the marker; the filled shapes must have none or the edge swallows them at the smallest
    size tier.
    """
    marker = _TIER_STYLES[marker_tier].marker
    if _TIER_STYLES[marker_tier].filled:
        return {"marker": marker, "color": color, "linewidths": 0}
    width = max(0.35, 0.16 * float(np.sqrt(area)))
    if marker == "o":
        return {"marker": marker, "facecolors": "none", "edgecolors": color,
                "linewidths": width}
    return {"marker": marker, "color": color, "linewidths": width}


def _baseline_marker_kwargs(marker_tier: "str | None", area: float) -> dict:
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


#: Width (inches) reserved to the right of the panels for the three channel legends, and
#: the vertical gap (inches) between two of them.  Absolute rather than fractional: the
#: legends are the same objects whatever the grid, so the column must not grow with it.
_LEGEND_WIDTH_IN = 2.1
_LEGEND_GAP_IN = 0.25

#: Font sizes (points) for the tier grid.  Chosen against the size the figure is read at
#: rather than drawn at: it is 17-20 inches wide and is placed at a text column's width,
#: about a third of that.  Enlarging the figure instead would cancel out, so the type
#: grows inside the same inches and the panel geometry stays fixed.
_MATRIX_FONT: dict[str, int] = {
    "suptitle": 17,
    "column":   15,   # the facet tier heading, on the top row only
    "count":    12,   # the per-panel `n = ...`
    "axis":     14,   # the two axis labels
    "tick":     12,
    "legend":   13,
}



def _figure_bbox(fig: "Figure", artist) -> "matplotlib.transforms.Bbox":
    """`artist`'s bounding box in figure coordinates.

    A legend has no extent until a renderer has laid it out, and nothing has drawn the
    figure at the point the next legend needs placing; the Agg canvas this module pins at
    import time supplies a renderer without a draw pass.
    """
    renderer = fig.canvas.get_renderer()
    return artist.get_window_extent(renderer).transformed(fig.transFigure.inverted())


def _check_matrix_groups(df: "pd.DataFrame", channels: dict[str, str]) -> None:
    """Validate the five channel→group assignments of the campaign matrices."""
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


def _baseline_style_groups(
    cell: "pd.DataFrame",
    marker_group: str,
    size_group: str,
) -> "list[tuple[str | None, str | None, pd.DataFrame]]":
    """`cell` split into (marker tier, size tier, rows) blocks, largest size first.

    A channel is encodable exactly when its group is one the baseline varies; a group
    held at `UNUSED_TIER` yields `None` there and the caller draws the fallback style.
    Sorted largest area first, so the small markers are not buried.
    """
    import pandas as pd    # runtime-only: the module-level import is TYPE_CHECKING

    def tiers(group: str) -> "pd.Series":
        column = f"tier_{group}"
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
    x_floor: float | None = None,
    y_floor: float | None = None,
    shared_limits: tuple[float, float] | None = None,
    baseline: "pd.DataFrame | None" = None,
) -> "Figure":
    """The five-channel tier grid, for any pair of columns.

    The facet layout, the colour/shape/size encoding of the three non-facet groups and
    the three proxy legends live here; callers validate the channel assignment
    (:func:`_check_matrix_groups`) and supply the two plot columns.  `x_floor` / `y_floor`
    clip non-positive values onto a log axis (see :func:`_log_floor`).

    `shared_limits` puts the same range on both axes, which is only meaningful when the
    two columns are in the same unit.  The panels then also get an equal aspect, so a
    decade is the same length on both and the 1:1 diagonal runs at 45°.

    `baseline` holds the zero-agent optima.  They are matched to a panel by the facet
    groups alone, so a facet on a group the baseline does not vary repeats the same
    reference points down every column.
    """
    columns = _tiers_present(df, f"tier_{x_group}")
    rows = _tier_groups(df, f"tier_{y_group}")
    if not columns or not rows:
        raise ValueError("no rows for any tier of the faceting groups")

    # Per channel, only the tiers the data has: the groups differ in tier count, so a
    # fixed list would print dead legend entries.
    size_tiers = _tiers_present(df, f"tier_{size_group}")
    marker_tiers = _tiers_present(df, f"tier_{marker_group}")
    color_tiers = _tiers_present(df, f"tier_{color_group}")

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

        channel_columns = [f"tier_{g}" for g in (size_group, marker_group, color_group)]
        for r, (y_tier, y_sub) in enumerate(rows):
            for c, x_tier in enumerate(columns):
                ax = axes[r][c]
                cell = y_sub[y_sub[f"tier_{x_group}"] == x_tier]
                scale, alpha = _density_style(len(cell))
                # Grouped once, then visited in tier order rather than in the groupby's
                # own: the largest markers must be drawn first, or the small ones end up
                # buried under them.
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
                        # not mask the optima it is the reference for.  The label makes it
                        # identifiable on the axes; the figure legend uses proxies.
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
                # already carries the centred facet heading.  Two stacked titles are not
                # available: matplotlib repositions every title on each draw and `pad`
                # moves all of them together.
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
            # the shape and size channels with the search optima, so only black separates
            # it from them.
            legend_specs[0][1].append(
                Line2D([], [], linestyle="none", marker="o", markersize=8,
                       color=_BASELINE_COLOR, label=_BASELINE_LABEL)
            )
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
        # One legend per per-point channel, and the baseline rides in the colour one.
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


def _require_columns(df: "pd.DataFrame", columns: tuple[str, ...]) -> None:
    """Raise unless every one of `columns` is present."""
    for column in columns:
        if column not in df.columns:
            raise KeyError(f"{column!r} is not a column of the optima table")


def _draw_cost_plane(
    df: "pd.DataFrame",
    channels: dict[str, str],
    x_col: str,
    y_col: str,
    xlabel: str,
    ylabel: str,
    title: str,
    baseline: "pd.DataFrame | None",
) -> "Figure":
    """A tier grid whose two axes are both dollars per feature.

    Both columns share one log range — taken over `df` and `baseline` together, so a
    baseline zero does not drop off the axis — at an equal aspect, and the 1:1 diagonal
    is drawn.  Ticks are decades in dollars, with 2 and 5 as unlabelled minor ticks.
    """
    scaled = _with_baseline(df, baseline)
    x_floor = _log_floor(scaled, x_col)
    y_floor = _log_floor(scaled, y_col)
    fig = _draw_matrix(
        df, **channels,
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


#: The two cost planes: how each derives its columns, and how it labels them.  Both go
#: through `_draw_cost_plane`, so they differ only in this.
class _CostPlane(NamedTuple):
    x_col: str
    y_col: str
    xlabel: str                 # the dollar signs are escaped, or mathtext sees three
    ylabel: str                 # delimiters in the label and renders it literally
    title: str
    derive: object = None       # frame -> frame, for a column no table carries


def _add_development_cost(frame: "pd.DataFrame") -> "pd.DataFrame":
    """A copy of `frame` with the two team cost drivers added together."""
    return frame.assign(
        cost_development_mean=frame["cost_human_mean"] + frame["cost_agent_mean"])


_HUMAN_AGENT_PLANE = _CostPlane(
    x_col="cost_agent_mean", y_col="cost_human_mean",
    xlabel="↓  agent cost   (\\$ / feature)",
    ylabel="↓  developer cost   (\\$ / feature)",
    title="Developer vs. agent cost at $c^\\star$",
)

_QUALITY_DEVELOPMENT_PLANE = _CostPlane(
    x_col="cost_development_mean", y_col="cost_quality_mean",
    xlabel="↓  development cost   (\\$ / feature)",
    ylabel="↓  escaped-bug cost   (\\$ / feature)",
    title="Escaped-bug vs. development cost (developers + agents) at $c^\\star$",
    derive=_add_development_cost,
)


def _plot_cost_plane(
    plane: _CostPlane,
    df: "pd.DataFrame",
    required: tuple[str, ...],
    x_group: str, y_group: str, size_group: str, marker_group: str, color_group: str,
    title: str,
    baseline: "pd.DataFrame | None",
) -> "Figure":
    """Validate the channels and the columns, derive what is missing, and draw `plane`."""
    channels = {
        "x_group": x_group, "y_group": y_group, "size_group": size_group,
        "marker_group": marker_group, "color_group": color_group,
    }
    _check_matrix_groups(df, channels)
    _require_columns(df, required)
    if plane.derive is not None:
        df = plane.derive(df)
        if baseline is not None and not baseline.empty:
            baseline = plane.derive(baseline)
    return _draw_cost_plane(
        df, channels,
        x_col=plane.x_col, y_col=plane.y_col,
        xlabel=plane.xlabel, ylabel=plane.ylabel,
        title=title or f"{plane.title} — {x_group} × {y_group}",
        baseline=baseline,
    )


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
    """The two team cost drivers of each optimum against each other, on a tier grid.

    x is what the agents billed, y what the developers cost, both dollars per feature —
    the part of the bill the team composition controls.  The escaped-bug term is the
    subject of :func:`plot_quality_development_matrix`.
    """
    return _plot_cost_plane(
        _HUMAN_AGENT_PLANE, df, ("cost_agent_mean", "cost_human_mean"),
        x_group, y_group, size_group, marker_group, color_group, title, baseline,
    )


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
    """What the escaped bugs cost against what the team cost, on a tier grid.

    The two drivers :func:`plot_human_agent_matrix` separates are added back together on
    x and set against the escaped-bug bill on y.  The sum is derived on a copy; no table
    carries it.
    """
    return _plot_cost_plane(
        _QUALITY_DEVELOPMENT_PLANE, df,
        ("cost_quality_mean", "cost_human_mean", "cost_agent_mean"),
        x_group, y_group, size_group, marker_group, color_group, title, baseline,
    )


def plot_agent_human_ratio_matrix(
    df: "pd.DataFrame",
    x_group: str = "agent",
    y_group: str = "human",
    size_group: str = "bugimpact",
    marker_group: str = "project",
    color_group: str = "debt",
    title: str = "",
) -> "Figure":
    """How lopsided the agent/developer split is at each optimum, against its percentile.

    y is `cost_agent_mean / cost_human_mean`, the trade-off
    :func:`plot_human_agent_matrix` draws as a plane collapsed to one number: above 1 the
    agents outbilled the developers.  It spans several decades, hence the log axis and
    the reference line at 1.

    x is that ratio's percentile rank within `df`, so the panels share one empirical
    distribution and each is a fragment of it.  The rank is taken over the frame as handed
    in, so filtering it further before calling re-ranks what is left against itself.
    """
    channels = {
        "x_group": x_group, "y_group": y_group, "size_group": size_group,
        "marker_group": marker_group, "color_group": color_group,
    }
    _check_matrix_groups(df, channels)
    _require_columns(df, ("cost_agent_mean", "cost_human_mean"))

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
        # Short enough not to run into the next panel's at `_MATRIX_FONT["axis"]`.
        xlabel="percentile rank of the ratio",
        # No "↓": unlike a cost, neither direction of the ratio is the better one.
        ylabel="agent cost / developer cost",
        title=title or f"Agent-to-developer cost ratio at $c^\\star$ — "
                       f"{x_group} × {y_group}",
        # x is a rank in [0, 1] — a log axis would only stretch the first percentile.
        xscale="linear",
        y_floor=_log_floor(df, "cost_agent_human_ratio"),
    )
    for ax in fig.get_axes():
        # A rank axis is uniform, so it takes the plain percent formatter rather than the
        # decade-aware `_percent_tick` a log axis of fractions needs.
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        _decade_axis(ax.yaxis, _ratio_tick)
        ax.axhline(1.0, **_REFERENCE_LINE)      # parity
    return fig


# ---------------------------------------------------------------------------
# 6. What the optimal teams are made of, by two scenario groups
# ---------------------------------------------------------------------------

# The two populations of `src.constellation`, and the colours they keep across both
# figures: the same red and blue the per-search landscape gives `c*` and `I*`.
_POPULATION_COLORS = {C_STAR: _BEST_COLOR, I_STAR: _KEEP_COLOR}
_POPULATION_LABELS = {C_STAR: "$c^\\star$", I_STAR: "$\\mathcal{I}^\\star$ (mean per scenario)"}


def _check_cell_channels(cells: "pd.DataFrame", groups: tuple[str, ...]) -> tuple[str, str]:
    """Validate the two faceting groups of the constellation figures."""
    require_group_pair(groups)
    missing = [f"tier_{g}" for g in groups if f"tier_{g}" not in cells.columns]
    if missing:
        raise KeyError(f"{missing} are not columns of the constellation table")
    return groups[0], groups[1]


def _cell_lookup(cells: "pd.DataFrame", row_column: str, col_column: str) -> dict:
    """(population, row tier, col tier, parameter) -> (mean, sd), for O(1) panel lookup.

    Keyed on the two faceting columns by name, since a caller may hand in a wider table
    and a positional key would then line up with the wrong group.
    """
    return dict(zip(
        zip(cells["population"], cells[row_column], cells[col_column],
            cells["parameter"]),
        zip(cells["mean"], cells["sd"]),
    ))


def _profile_limits(cells: "pd.DataFrame", parameters: tuple[str, ...]) -> tuple[float, float]:
    """One padded log range covering every bar and whisker of the profile figure.

    Includes `mean + sd`, since the upper whisker is drawn.  A role no optimum staffs has
    a mean of exactly 0, which a log axis cannot place; `_zero_marks` annotates those.
    """
    sub = cells[cells["parameter"].isin(parameters)]
    upper = (sub["mean"].fillna(0.0) + sub["sd"].fillna(0.0)).to_numpy(dtype=float)
    means = sub["mean"].to_numpy(dtype=float)
    # The bars grow from the left frame, so the low end gets an extra decade of run-up:
    # without it the smallest mean is a bar of zero length and reads as an absent role.
    return _padded_log_range(np.concatenate([means, upper]), "the parameter means",
                             extra_decades_below=1.0)


def plot_constellation_profile(
    cells: "pd.DataFrame",
    rows_group: str = "bugimpact",
    cols_group: str = "project",
    title: str = "",
) -> "Figure":
    """The development-phase composition of the optimal team, per cell of a tier grid.

    One panel per (rows_group tier, cols_group tier) cell of the ``constellation_cells``
    table of :mod:`src.constellation`, each a horizontal bar chart over the eight agent
    roles with the two populations side by side and the between-scenario standard
    deviation as a whisker.

    The x axis is logarithmic and shared by every panel, since the counts span several
    decades and the panels have to be comparable.  A role with a mean of exactly 0 has no
    bar a log axis can draw and is marked `0` at the left frame; the whiskers are clipped
    at the frame, `sd` exceeding the mean wherever a role is staffed in only part of the
    cell.
    """
    rows_group, cols_group = _check_cell_channels(cells, (rows_group, cols_group))
    parameters, labels = DEV_PARAMETERS, AGENT_ROLE_ABBREV
    row_column, col_column = f"tier_{rows_group}", f"tier_{cols_group}"
    row_tiers = _tiers_present(cells, row_column)
    col_tiers = _tiers_present(cells, col_column)
    if not row_tiers or not col_tiers:
        raise ValueError("no rows for any tier of the faceting groups")
    populations = [p for p in POPULATIONS if (cells["population"] == p).any()]

    lookup = _cell_lookup(cells, row_column, col_column)
    x_lo, x_hi = _profile_limits(cells, parameters)
    y = np.arange(len(parameters), dtype=float)
    height = 0.8 / max(1, len(populations))

    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(
            len(row_tiers), len(col_tiers),
            figsize=(3.6 * len(col_tiers), 0.34 * len(parameters) * len(row_tiers) + 1.2),
            sharex=True, sharey=True, squeeze=False,
        )
        fig.suptitle(title, fontsize=12, y=1.0)

        for r, row_tier in enumerate(row_tiers):
            for c, col_tier in enumerate(col_tiers):
                ax = axes[r][c]
                for i, population in enumerate(populations):
                    offset = (i - (len(populations) - 1) / 2.0) * height
                    means, lower, upper, zeros = [], [], [], []
                    for j, parameter in enumerate(parameters):
                        mean, sd = lookup.get(
                            (population, row_tier, col_tier, parameter),
                            (np.nan, np.nan))
                        mean = 0.0 if np.isnan(mean) else float(mean)
                        sd = 0.0 if np.isnan(sd) else float(sd)
                        means.append(mean)
                        # Clipped at the frame; mean - sd is often negative here.
                        lower.append(max(0.0, mean - max(x_lo, mean - sd)))
                        upper.append(sd)
                        if mean <= 0:
                            zeros.append(j)
                    ax.barh(y + offset, means, height=height * 0.92,
                            color=_POPULATION_COLORS[population], alpha=0.75,
                            label=_POPULATION_LABELS[population],
                            xerr=[lower, upper],
                            error_kw={"elinewidth": 0.8, "ecolor": "0.3", "capsize": 1.5})
                    for j in zeros:
                        ax.text(x_lo, y[j] + offset, " 0", fontsize=6, va="center",
                                ha="left", color=_POPULATION_COLORS[population])
                ax.set_xscale("log")
                ax.set_xlim(x_lo, x_hi)
                ax.set_ylim(len(parameters) - 0.5, -0.5)   # role 1 at the top
                ax.set_yticks(y)
                ax.set_yticklabels(labels, fontsize=8)
                ax.grid(axis="y", visible=False)
                if r == 0:
                    ax.set_title(f"{cols_group} = {col_tier}", fontsize=10)
                if r == len(row_tiers) - 1:
                    ax.set_xlabel("agents per role (mean ± sd over scenarios)", fontsize=9)
                if c == 0:
                    ax.set_ylabel(f"{rows_group} = {row_tier}", fontsize=9)

        handles, legend_labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles[:len(populations)], legend_labels[:len(populations)],
                   loc="lower center", ncol=len(populations), frameon=True, fontsize=9,
                   bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.98))
    return fig


def _normalised_columns(
    cells: "pd.DataFrame",
    parameters: tuple[str, ...],
) -> "dict[str, tuple[float, float]]":
    """Per parameter, the (min, max) of ``log1p(mean)`` over both populations.

    Each parameter gets its own colour range: they share no unit and span four decades
    between them, so one range would paint every column but the largest flat.  The range
    covers both populations, so the heatmap's two panels stay comparable.  `log1p` rather
    than a plain min-max, since the counts are skewed within a column too and it maps an
    exact zero to an exact zero.
    """
    ranges = {}
    for parameter in parameters:
        values = cells.loc[cells["parameter"] == parameter, "mean"].to_numpy(dtype=float)
        values = np.log1p(np.clip(values[~np.isnan(values)], 0.0, None))
        if values.size == 0:
            ranges[parameter] = (0.0, 1.0)
        else:
            lo, hi = float(values.min()), float(values.max())
            ranges[parameter] = (lo, hi if hi > lo else lo + 1.0)
    return ranges


def _cell_annotation(mean: float, sd: float) -> str:
    """`mean` over `±sd`, three significant digits, for one heatmap cell."""
    if np.isnan(mean):
        return ""
    top = f"{mean:.3g}"
    return top if np.isnan(sd) else f"{top}\n±{sd:.2g}"


def plot_constellation_heatmap(
    cells: "pd.DataFrame",
    rows_group: str = "project",
    cols_group: str = "bugimpact",
    title: str = "",
) -> "Figure":
    """The whole constellation table as a picture, one panel per population.

    Rows are the cells of the tier grid, both faceting groups on one axis so that all of
    them are visible at once; columns are the control variables, grouped into the process
    triple, the two composition vectors and the derived totals, with a rule between the
    blocks.

    Colour is the mean normalised within its own column and across both populations (see
    :func:`_normalised_columns`).  The raw `mean ± sd` is printed in each cell, so the
    colour is a reading aid and the number is the result.
    """
    rows_group, cols_group = _check_cell_channels(cells, (rows_group, cols_group))
    row_column, col_column = f"tier_{rows_group}", f"tier_{cols_group}"
    row_tiers = _tiers_present(cells, row_column)
    col_tiers = _tiers_present(cells, col_column)
    if not row_tiers or not col_tiers:
        raise ValueError("no rows for any tier of the faceting groups")
    populations = [p for p in POPULATIONS if (cells["population"] == p).any()]

    parameters = tuple(p for _, block in PARAMETER_BLOCKS for p in block
                       if (cells["parameter"] == p).any())
    if not parameters:
        raise ValueError("none of the requested parameters is in the table")
    # Where each block ends, for the separator rules; the last one needs no rule.
    edges, seen = [], 0
    for _, block in PARAMETER_BLOCKS:
        seen += sum(1 for p in block if p in parameters)
        edges.append(seen)
    edges = [e for e in edges[:-1] if 0 < e < len(parameters)]

    grid = [(row_tier, col_tier) for row_tier in row_tiers for col_tier in col_tiers]
    lookup = _cell_lookup(cells, row_column, col_column)
    ranges = _normalised_columns(cells, parameters)
    cmap = plt.get_cmap("YlGnBu")

    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(
            len(populations), 1,
            figsize=(0.78 * len(parameters) + 2.6, 0.46 * len(grid) * len(populations) + 1.4),
            sharex=True, squeeze=False,
        )
        fig.suptitle(title, fontsize=12, y=1.0)

        for a, population in enumerate(populations):
            ax = axes[a][0]
            shades = np.full((len(grid), len(parameters)), np.nan)
            for i, (row_tier, col_tier) in enumerate(grid):
                for j, parameter in enumerate(parameters):
                    mean, sd = lookup.get(
                        (population, row_tier, col_tier, parameter), (np.nan, np.nan))
                    if np.isnan(mean):
                        continue
                    lo, hi = ranges[parameter]
                    shade = (np.log1p(max(mean, 0.0)) - lo) / (hi - lo)
                    shades[i, j] = shade
                    ax.text(j, i, _cell_annotation(mean, sd), fontsize=5.5,
                            ha="center", va="center", linespacing=0.95,
                            # The map runs light→dark, so the annotation has to flip
                            # before the background swallows it.
                            color="white" if shade > 0.62 else "0.12")
            ax.imshow(np.ma.masked_invalid(shades), cmap=cmap, vmin=0.0, vmax=1.0,
                      aspect="auto", interpolation="nearest")
            for edge in edges:
                ax.axvline(edge - 0.5, color="0.25", linewidth=1.2)
            ax.set_yticks(range(len(grid)))
            ax.set_yticklabels([f"{r} / {c}" for r, c in grid], fontsize=8)
            ax.set_ylabel(f"{rows_group} / {cols_group}", fontsize=9)
            ax.set_title(_POPULATION_LABELS[population], fontsize=10, loc="left")
            ax.grid(visible=False)
            ax.set_xticks(range(len(parameters)))
            if a == len(populations) - 1:
                ax.set_xticklabels(parameters, fontsize=7, rotation=90)
            ax.tick_params(length=0)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    return fig


def visualize_constellation(
    cells: "pd.DataFrame",
    out_dir: Path,
    fmt: str = "png",
    rows_group: str = "bugimpact",
    cols_group: str = "project",
) -> None:
    """Both constellation figures for one ``constellation_cells`` table.

    The file names carry the two faceting groups rather than the result mode: the table is
    built from the searched optima only — a human-only team has no composition — so the
    mode is fixed, while the grid is the thing the caller can change and must not silently
    overwrite.
    """
    if cells.empty:
        warnings.warn("the constellation table is empty; nothing to plot.", stacklevel=2)
        return

    populations = [p for p in POPULATIONS if (cells["population"] == p).any()]
    n_cells = cells.drop_duplicates(
        [f"tier_{rows_group}", f"tier_{cols_group}"]).shape[0]
    suffix = f"{cols_group}_{rows_group}"
    print(f"\n── Team constellation: {rows_group} × {cols_group} "
          f"({n_cells} cells, {len(populations)} population(s)) ──")

    fig = plot_constellation_profile(
        cells, rows_group=rows_group, cols_group=cols_group,
        title=f"Development-phase composition of the optimal team, "
              f"by {cols_group} and {rows_group}",
    )
    save_figure(fig, out_dir, f"constellation_profile_{suffix}", fmt)

    fig = plot_constellation_heatmap(
        cells, rows_group=cols_group, cols_group=rows_group,
        title=f"Every control variable of the optimal team, "
              f"by {cols_group} and {rows_group}",
    )
    save_figure(fig, out_dir, f"constellation_heatmap_{suffix}", fmt)


# ---------------------------------------------------------------------------
# 7. The two-figure summary of team structure
# ---------------------------------------------------------------------------
#
# `plot_constellation_profile` / `_heatmap` report every control variable; these two
# summarise, one on the composition of the team and one on what moves it.  Both take
# `optima.csv` rather than the aggregated cell table, because their shares and rates have
# to be formed within a scenario before being averaged: a ratio of means is not the mean
# of the ratios.

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
    xlabel: str
    total_column: "str | None"      # annotated at the bar's end; None = sum the columns
    total_label: str
    total_format: str
    filename: str                   # figure stem, naming what is split


#: Three views of one composition, selected by `plot_team_shape`'s `weight`:
#:
#: `agent` (the default) — each role's share of the agent bill.
#: `team`  — the same eight roles plus the developers, the whole bill the composition
#:   controls.  The agent side is `cost_agent_mean` apportioned by the `agent` shares, so
#:   the nine slices add to `cost_team_mean` exactly.  The escaped-bug cost is annotated
#:   beside the bar rather than stacked into it.
#: `count` — each role's share of the head count.  Not written by default; it disagrees
#:   with the bill, because the flat-rate autonomous roles are cheap and numerous.
#:
#: The two cost weightings annotate the reported per-feature figure rather than a summed
#: horizon bill, so the number at a bar's end is in the unit the tables report.
_SHARE_WEIGHTS: dict[str, _ShareWeight] = {
    "agent": _ShareWeight(
        columns=BILL_PARAMETERS, labels=AGENT_ROLE_ABBREV, colors=_ROLE_COLORS,
        xlabel="share of the agent bill of the optimal team",
        total_column="cost_agent_mean", total_label="agent \\$ per\nfeature",
        total_format="${:,.2f}",
        filename="composition_agent_bill"),
    "team": _ShareWeight(
        columns=("cost_human_mean",) + AGENT_COST_PARAMETERS,
        labels=("Developers",) + AGENT_ROLE_ABBREV,
        colors=(_HUMAN_COLOR,) + _ROLE_COLORS,
        xlabel="share of the team bill (developers + agents) of the optimal team",
        total_column="cost_team_mean", total_label="team \\$ per\nfeature",
        total_format="${:,.0f}",
        filename="composition_team_bill"),
    "count": _ShareWeight(
        columns=DEV_PARAMETERS, labels=AGENT_ROLE_ABBREV, colors=_ROLE_COLORS,
        xlabel="share of the agents in the optimal team (development phase)",
        total_column=None, total_label="team size", total_format="{:,.0f}",
        filename="composition_head_count"),
}


#: The escaped-bug cost, annotated beside every bar but never stacked into it: the bar
#: describes the team, and this is what the team's choices cost elsewhere.
_BUG_COST_COLUMN = "cost_quality_mean"
_BUG_COST_LABEL = "bug \\$ per\nfeature"

#: Points to the right of a bar's end for the two annotated columns; each has to clear
#: the widest figure the column before it can print.
_ANNOTATION_OFFSETS = (4.0, 54.0)


def _compact_dollars(value: float) -> str:
    """`$83`, `$851`, `$26.2k`, `$517k` — at most five characters, whatever the decade.

    The escaped-bug cost spans four orders of magnitude across the cells, where the plain
    format of the team column would print a seven-digit number into a margin.
    """
    if value < 1_000:
        return f"${value:,.0f}"
    if value < 10_000:
        return f"${value / 1_000:.1f}k"
    if value < 1_000_000:
        return f"${value / 1_000:,.0f}k"
    return f"${value / 1_000_000:.1f}M"


def _composition_shares(
    optima: "pd.DataFrame",
    groups: tuple[str, ...],
    columns: tuple[str, ...],
    total_column: "str | None" = None,
    extra_column: "str | None" = None,
) -> "pd.DataFrame":
    """Mean role share per cell, plus the mean total, over whichever `columns` weigh a role.

    The share is formed per scenario and then averaged, so a cell's shares are the mean
    composition of its optima rather than the composition of its mean optimum.  A row
    whose total is zero contributes no share but still counts toward `n_scenarios`.
    """
    extras = [c for c in (total_column, extra_column) if c]
    missing = [c for c in list(columns) + extras if c not in optima.columns]
    if missing:
        raise KeyError(
            f"{missing} are not columns of the optima table; the cost weighting needs "
            f"`constellation.role_bill` to have been applied first"
        )
    columns = list(columns)
    frame = optima.copy()
    weight_total = frame[columns].sum(axis=1)
    frame["group_total"] = (weight_total if total_column is None
                            else frame[total_column])
    # A per-scenario dollar figure like the first, averaged the same way; it weighs
    # nothing and enters no share.
    annotations = ["group_total"]
    if extra_column is not None:
        frame["extra_total"] = frame[extra_column]
        annotations.append("extra_total")
    positive = weight_total > 0
    shares = frame.loc[positive, columns].div(weight_total[positive], axis=0)
    keys = [f"tier_{g}" for g in groups]
    shares = shares.join(frame[keys + annotations])
    out = shares.groupby(keys, sort=False)[columns + annotations].mean()
    out["n_scenarios"] = frame.groupby(keys, sort=False).size()
    return out.reset_index()


def _yaxis_note(ax, x: float, y: float, text: str, dx: float, **kwargs):
    """`text` beside data row `y`, `x` in axes fraction, offset `dx` points sideways.

    Offset in points rather than in axes fractions: these notes sit outside a panel, and
    a fraction would place them differently on every layout.
    """
    kwargs.setdefault("va", "center")
    return ax.annotate(text, xy=(x, y), xycoords=ax.get_yaxis_transform(),
                       xytext=(dx, 0), textcoords="offset points", **kwargs)


def plot_team_shape(
    optima: "pd.DataFrame",
    block_group: str = "project",
    bar_group: str = "bugimpact",
    weight: str = "agent",
    title: str = "",
    panel_group: "str | None" = "agent",
) -> "Figure":
    """What the optimal team is made of: one hundred-per-cent stacked bar per cell.

    The absolute total is annotated at the end of each bar rather than drawn: it spans
    four decades, and an axis showing it would flatten the composition.

    `weight` chooses what a slice measures (see `_SHARE_WEIGHTS`) and brings its own
    labels and palette.

    Three scenario groups are laid out at once.  Bars are grouped into blocks by
    `block_group`, so the same `bar_group` ladder repeats in each; `panel_group` repeats
    that whole structure once per panel, two panels to a grid row.  `panel_group=None`
    gives the single-panel form and pools that group instead.
    """
    if weight not in _SHARE_WEIGHTS:
        raise ValueError(f"unknown weight {weight!r}; "
                         f"expected one of {list(_SHARE_WEIGHTS)}")
    spec = _SHARE_WEIGHTS[weight]
    columns, total_column = spec.columns, spec.total_column
    labels = spec.labels
    grouping = tuple(g for g in (panel_group, block_group, bar_group) if g is not None)
    require_distinct_groups(grouping)
    if len(labels) != len(columns):
        raise ValueError(f"{len(labels)} labels for {len(columns)} roles")

    extra_column = _BUG_COST_COLUMN if _BUG_COST_COLUMN in optima.columns else None
    cells = _composition_shares(optima, grouping, columns, total_column, extra_column)
    colors = spec.colors
    blocks = _tiers_present(cells, f"tier_{block_group}")
    bars = _tiers_present(cells, f"tier_{bar_group}")
    panels = _tiers_present(cells, f"tier_{panel_group}") if panel_group else [None]
    if not blocks or not bars or not panels:
        raise ValueError("no optima for any tier of the grouping variables")

    indexed = cells.set_index([f"tier_{g}" for g in grouping])
    # Every panel carries the same rows, so the layout is built once and each panel only
    # looks its own cells up.
    positions, ticks, tick_labels, block_spans = [], [], [], []
    y = 0.0
    for block in blocks:
        start = y
        for bar in bars:
            positions.append((y, block, bar))
            ticks.append(y)
            tick_labels.append(bar)
            y += 1.0
        block_spans.append((block, start, y - 1.0))
        y += 0.6                     # a gap, so the blocks read as blocks

    # The width is per panel, so a segment label still fits when panels are added, and
    # the figure grows rather than splitting one budget.  Two to a grid row keeps the
    # left column's rows aligned with the right column's.
    n_panels = len(panels)
    ncols = 1 if n_panels == 1 else 2
    nrows = -(-n_panels // ncols)
    panel_width = 11.0 if n_panels == 1 else 4.4
    row_height = 0.40 if n_panels == 1 else 0.30
    with plt.style.context(_STYLE):
        # Two annotated columns hang off each panel's right edge; the inter-panel gap and
        # the right margin have to hold them, so they are budgeted here in inches.
        margin = 1.4
        fig = plt.figure(figsize=(1.45 + ncols * (panel_width + margin),
                                  nrows * row_height * y + (2.1 if ncols == 1 else 2.6)))
        grid = fig.add_gridspec(nrows, ncols)
        bar_axes = []
        for p in range(n_panels):
            r, c = divmod(p, ncols)
            bar_axes.append(fig.add_subplot(grid[r, c],
                                            sharey=bar_axes[0] if bar_axes else None))
        # Placed by hand: `suptitle`'s default `y` is a fraction, which would grow the
        # gap as the figure grows.
        fig.suptitle(title, fontsize=12, y=1.0 - 0.20 / fig.get_figheight())
        # Below this a segment is drawn but not labelled.  In figure space, so the
        # threshold tightens as panels are added and the panels narrow.
        label_min = max(_SHARE_LABEL_MIN, 0.24 / panel_width)
        for p, (panel, ax) in enumerate(zip(panels, bar_axes)):
            # A panel's column of totals overhangs its right edge, into the next panel's
            # cell of the grid.  Axes are drawn in `zorder` order, so a row is ordered
            # right to left and a neighbour's opaque background cannot cover it.
            ax.set_zorder(float(ncols - p % ncols))
            for position, block, bar in positions:
                key = (block, bar) if panel is None else (panel, block, bar)
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
            # Panelled, the label is drawn once under the whole grid instead.
            ax.set_xlabel(spec.xlabel if n_panels == 1 else "", fontsize=9)
            ax.grid(axis="y", visible=False)
            if p % ncols:
                # `sharey` already aligns the rows, so the right column is read off the
                # left one's labels and needs no second copy of them.
                ax.tick_params(axis="y", labelleft=False)
            if panel is not None:
                ax.set_title(f"{panel_group} = {panel}", fontsize=10, fontweight="bold")
            # One header per panel: each carries its own columns of absolute totals.
            _yaxis_note(ax, 1.0, -1.0, spec.total_label, _ANNOTATION_OFFSETS[0],
                        fontsize=8, ha="left", color="0.25", linespacing=0.95)
            if extra_column is not None:
                _yaxis_note(ax, 1.0, -1.0, _BUG_COST_LABEL, _ANNOTATION_OFFSETS[1],
                            fontsize=8, ha="left", color="0.25", linespacing=0.95)

        bar_axes[0].set_ylim(y - 0.4, -1.3)        # first block at the top
        bar_axes[0].set_yticks(ticks)
        bar_axes[0].set_yticklabels(tick_labels, fontsize=9)
        # Outside the axes, and on the first panel of every grid row rather than once per
        # figure.  These take the place of a y label, which would land left of them.
        for ax in bar_axes[::ncols]:
            for block, start, end in block_spans:
                _yaxis_note(ax, 0.0, (start + end) / 2.0, f"{block_group}\n= {block}",
                            -46, fontsize=9, ha="right", fontweight="bold")
            # Two column headers, one over each margin, so both are named.
            _yaxis_note(ax, 0.0, -1.0, bar_group, -4, fontsize=8, ha="right",
                        color="0.25")
        ax = bar_axes[0]
        handles, legend_labels = ax.get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="lower center", ncol=len(labels),
                   fontsize=8, frameon=True, columnspacing=0.9, handlelength=1.1)
        panelled = n_panels > 1
        # The legend/x-label strip and the title strip are reserved in inches, not in
        # figure fractions, which would grow with the grid.
        height = fig.get_figheight()
        bottom = (0.42 if panelled else 0.55) / height
        fig.tight_layout(rect=(1.35 / fig.get_figwidth(), bottom,
                               1.0 - margin / fig.get_figwidth(), 1.0 - 0.32 / height))
        if panelled:
            # Room for the columns of totals that overhang each panel's right edge.
            # `wspace` is a fraction of the axes width `tight_layout` has just decided, so
            # solve for the value that makes every gap exactly `margin` inches.
            figure_width = fig.get_figwidth()
            first, last = bar_axes[0].get_position(), bar_axes[ncols - 1].get_position()
            span = (last.x1 - first.x0) * figure_width
            fig.subplots_adjust(
                wspace=margin / ((span - (ncols - 1) * margin) / ncols))
    return fig


class _Quantity(NamedTuple):
    """One panel of the driver figure: a column, its axis label, its tick formatter."""
    column: str
    label: str
    formatter: "object" = None


#: The quantities the driver figure summarises, one linear panel each: the three process
#: controls, the two agent counts per arriving feature, and the detection split.
_TEAM_QUANTITIES: tuple[_Quantity, ...] = (
    _Quantity("N", "human staffing $N$\n(FTE)"),
    _Quantity("R", "release cycle $R$\n(sprints)"),
    _Quantity("F_freeze", "feature freeze $F_{freeze}$\n(sprints)"),
    _Quantity("sup_agents_per_feature", "supervised agents\nper arriving feature"),
    _Quantity("aut_agents_per_feature", "autonomous agents\nper arriving feature"),
    _Quantity("security_share", "Security's share of\nthe detection pair", _percent_tick),
)



def _team_quantities(optima: "pd.DataFrame") -> "pd.DataFrame":
    """Add the derived structural quantities of `_TEAM_QUANTITIES` to a copy of `optima`.

    Each is formed within a scenario before anything is averaged, so that the figure
    reports a typical ratio rather than the largest scenario's.
    """
    frame = optima.copy()
    # The supervised and autonomous halves get a column each: they are bounded by
    # different things and differ by more than an order of magnitude, so a combined
    # column would be the autonomous count with a rounding error.
    supervised = frame[[DEV_PARAMETERS[i] for i in SUPERVISED_ROLE_INDICES]].sum(axis=1)
    autonomous = frame[[DEV_PARAMETERS[i] for i in AUTONOMOUS_ROLE_INDICES]].sum(axis=1)
    detection = frame["a_dev_5"] + frame["a_dev_6"]
    frame["sup_agents_per_feature"] = supervised / frame["r_arrival"]
    frame["aut_agents_per_feature"] = autonomous / frame["r_arrival"]
    # A team with neither detection service has no split; NaN drops out of the order
    # statistics rather than being charged to either service as a zero.
    frame["security_share"] = frame["a_dev_6"].where(detection > 0) / detection
    return frame


def _tier_rows(frame: "pd.DataFrame", groups: tuple[str, ...]) -> list:
    """(group, tier, y position, subframe) for every tier present, with gaps between groups."""
    rows, y = [], 0.0
    for group in groups:
        column = f"tier_{group}"
        if column not in frame.columns:
            raise KeyError(f"{column!r} is not a column of the optima table")
        for tier in _tiers_present(frame, column):
            rows.append((group, tier, y, frame[frame[column] == tier]))
            y += 1.0
        y += 0.8
    return rows


def plot_team_drivers(
    optima: "pd.DataFrame",
    title: str = "",
) -> "Figure":
    """Which scenario setting moves which part of the team, on one shared row axis.

    One row per (scenario group, tier) and one panel per structural quantity; each row is
    the median over the scenarios at that tier with a bar from the first to the third
    quartile.  A group whose rows sit on top of each other does not move that quantity;
    one whose rows march across the panel does.

    Putting all five groups on one axis assumes the campaign is a full factorial, so that
    the scenarios at any one tier are balanced over the other four groups and a row is
    not confounded by them.

    Order statistics rather than mean and sd: several of these quantities pile up on a
    search-range boundary, where a mean reports a value no optimum took.
    """
    frame = _team_quantities(optima)
    rows = _tier_rows(frame, SCENARIO_GROUPS)
    if not rows:
        raise ValueError("no optima for any tier of the scenario groups")
    height = max(r[2] for r in rows) + 1.0
    # Midway down the gap `_tier_rows` leaves between two groups, so the block boundary
    # carries across all six panels.
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
            for group, tier, y, sub in rows:
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
        # figure that puts all five groups on one axis, so they get a single merged
        # legend entry rather than the same dot printed twice.
        present = [t for t in ALL_SCENARIO_TIERS
                   if any(tier == t for _, tier, _, _ in rows)]
        fourth = [t for t in present if t in ("veryhigh", "pessimistic")]
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
    optima: "pd.DataFrame",
    out_dir: Path,
    fmt: str = "png",
    block_group: str = "project",
    bar_group: str = "bugimpact",
    weights: tuple[str, ...] = ("agent", "team"),
    panel_group: "str | None" = "agent",
) -> None:
    """The team-structure summary figures, from the searched rows of ``optima.csv``.

    `optima` is expected to carry the per-role bill of `constellation.role_bill`; without
    it the cost weightings have nothing to draw, so the caller applies it first — the
    prices come from the scenario configs, which this module does not read.

    Two compositions by default: `agent` is what the agent budget is spent on, and
    `team` puts the developers in beside it.  `_SHARE_WEIGHTS` has the third.

    The composition figures also panel on `panel_group`.  A re-cut grid may put that
    group on one of the other two axes, and a group cannot occupy two channels at once,
    so it falls back to the pooled single-panel form rather than failing.
    """
    if panel_group in (block_group, bar_group):
        warnings.warn(
            f"{panel_group!r} is already a composition axis; the composition figures "
            f"pool it instead of panelling on it.", stacklevel=2)
        panel_group = None
    searched = optima[optima["mode"] != HUMAN_ONLY_MODE]
    if searched.empty:
        warnings.warn("no searched optimum to summarise; nothing to plot.", stacklevel=2)
        return

    titles = {
        "agent": "Share of agent costs",
        "team": "Share of development costs",
        "count": "Share of the agent head count",
    }
    print(f"\n── Team structure summary ({len(searched)} scenarios) ──")
    for weight in weights:
        fig = plot_team_shape(
            searched, block_group=block_group, bar_group=bar_group,
            panel_group=panel_group, weight=weight, title=titles.get(weight, ""),
        )
        save_figure(fig, out_dir, _SHARE_WEIGHTS[weight].filename, fmt)

    fig = plot_team_drivers(
        searched,
        title="Impact of scenario on team composition",
    )
    save_figure(fig, out_dir, "team_drivers", fmt)


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
    """Human-readable scenario name: the scenario id if present, else the directory."""
    scenario_id = manifest.get("scenario_id", "")
    if scenario_id:
        mode = manifest.get("mode", "")
        return f"{scenario_id} / {mode}" if mode else scenario_id
    return result_dir.name


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

    # The whole evaluated pool at K_search, i.e. the landscape the search saw, with the
    # reported candidates overwritten by their K_final estimates so that c*'s CI is drawn
    # at the width it is reported with.
    objectives = dict(summary["objectives"])
    objectives.update(summary["final_objectives"])
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
        obj = summary["final_objectives"][best_key]
        scenarios[_scenario_label(rd, manifest)] = obj

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
    turnover: "pd.DataFrame | None" = None,
    **channels: str,
) -> None:
    """The campaign matrices, drawn from a consolidated ``optima.csv``.

    The human-only rows are split off first and passed to the matrices as a `baseline`
    overlay rather than getting figures of their own.  `channels` passes `x_group` /
    `y_group` / `size_group` / `marker_group` / `color_group` through to every matrix.

    `turnover` is the long-format ``elite_turnover.csv``; only its search rows drive the
    convergence figure, so an all-baseline campaign gets none.
    """
    if "mode" not in optima.columns:
        raise KeyError("'mode' is not a column of the optima table")

    is_baseline = optima["mode"] == HUMAN_ONLY_MODE
    baseline = optima[is_baseline]
    searches = optima[~is_baseline]
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
        title=f"Developer vs. agent cost at $c^\\star${suffix}", **channels,
    )
    save_figure(fig, out_dir, "human_agent_matrix", fmt)

    fig = plot_quality_development_matrix(
        searches, baseline=baseline, **channels,
        title=f"Escaped-bug vs. development cost, developers + agents, "
              f"at $c^\\star${suffix}",
    )
    save_figure(fig, out_dir, "quality_development_matrix", fmt)

    # No baseline: a team with no agents has no agent/developer ratio.
    fig = plot_agent_human_ratio_matrix(
        searches, title=f"Agent-to-developer cost ratio at $c^\\star${suffix}", **channels,
    )
    save_figure(fig, out_dir, "agent_human_ratio_matrix", fmt)

    # An enumeration has no generations, so it contributes no rows here.
    if turnover is not None and not turnover.empty:
        curves = turnover[turnover["mode"] != HUMAN_ONLY_MODE]
        if not curves.empty:
            fig = plot_elite_turnover(
                curves, title="Elite turnover and $c^\\star$ drop per generation")
            save_figure(fig, out_dir, "elite_turnover", fmt)
