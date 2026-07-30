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

Public API — result-directory helpers
--------------------------------------
    visualize_single(result_dir, out_dir, fmt, mean_pareto)      -> None
    visualize_multi(result_dirs, out_dir, fmt, mean_pareto)      -> None
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

from .pareto import AggregatedObjectives, pareto_frontier

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


def _save(fig: "Figure", out_dir: Path, name: str, fmt: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{name}.{fmt}"
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {dest}")


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
