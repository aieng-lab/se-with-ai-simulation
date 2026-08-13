"""Visualization functions for search results.

All functions accept plain Python dicts / pandas DataFrames derived from the persisted
result files and return matplotlib Figure objects. Callers decide how to save or display.

Public API — plot functions
---------------------------
    plot_cost_vs_value(objectives, best_key, keep_keys, floor, title)   -> Figure
    plot_cost_split(objectives, keys, title)                            -> Figure
    plot_trajectories(traces_per_config, highlight_keys, fields, title) -> Figure
    plot_comparison_optima(scenarios, title)                            -> Figure

Public API — result-directory helpers
--------------------------------------
    visualize_single(result_dir, out_dir, fmt)                    -> None
    visualize_multi(result_dirs, out_dir, fmt)                    -> None
    save_figure(fig, out_dir, name, fmt)                          -> None

The campaign-level §5 figures that used to live here were built on the three-objective
Pareto frontier the model no longer has (paper §3.9-3.10); they were removed with it and
will be rebuilt once §5 specifies its replacement.
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
        ax.set_ylabel("cost   ($ / feature)", fontsize=9)
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
        ax.set_ylabel("cost   ($ / feature)", fontsize=9)
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
        ax.set_xlabel("cost of $c^\\star$   ($ / feature)", fontsize=9)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
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
