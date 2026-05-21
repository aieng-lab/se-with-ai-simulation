#!/usr/bin/env python3
"""Generate visualizations from persisted sweep results.

Usage
-----
Single experiment (plots saved to RESULT_DIR/plots/):
    python experiments/visualize.py results/sweep_default [--format png] [--out DIR]

Multi-experiment comparison (comparison plot saved to --out or ./plots/):
    python experiments/visualize.py results/sweep_default results/sweep_high_load \\
        results/sweep_budget [--format png] [--out DIR]

Plots produced
--------------
Single mode:
    pareto_scatter.png           2-D Pareto projections for all config pairs
    objectives_bar.png           Mean ± CI bar chart for every config
    replication_distributions.png  Box plots of K per-replication values
    trajectories.png             Sprint-level state trajectories (Pareto configs only)
                                 — skipped if no traces.csv found

Multi mode (in addition to single plots for each directory):
    comparison_pareto.png        Combined Pareto scatter across all experiments
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from src.visualization import (
    plot_comparison_pareto,
    plot_objectives_bar,
    plot_pareto_3d_scatter,
    plot_pareto_scatter,
    plot_replication_distributions,
    plot_trajectories,
)


# ---------------------------------------------------------------------------
# Helpers to load result files
# ---------------------------------------------------------------------------

def _load_summary(result_dir: Path) -> dict:
    path = result_dir / "sweep_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"No sweep_summary.json in {result_dir}.")
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
        rfile = result_dir / key / "result.json"
        if not rfile.exists():
            continue
        data = json.loads(rfile.read_text(encoding="utf-8"))
        out[key] = [(r["V"], r["Q"], r["C_F"]) for r in data["replications"]]
    return out


def _load_traces(
    result_dir: Path,
    config_keys: list[str],
) -> "dict[str, 'pd.DataFrame']":
    """Load traces.csv for each config that has one."""
    from src.results import load_traces
    out = {}
    for key in config_keys:
        tfile = result_dir / key / "traces.csv"
        if tfile.exists():
            out[key] = load_traces(tfile)
    return out


def _scenario_label(result_dir: Path, manifest: dict) -> str:
    scenario = manifest.get("scenario", "")
    return Path(scenario).stem if scenario else result_dir.name


def _save(fig, out_dir: Path, name: str, fmt: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{name}.{fmt}"
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  Saved: {dest}")


# ---------------------------------------------------------------------------
# Single-experiment visualizations
# ---------------------------------------------------------------------------

def visualize_single(result_dir: Path, out_dir: Path, fmt: str) -> None:
    summary  = _load_summary(result_dir)
    manifest = _load_manifest(result_dir)
    label    = _scenario_label(result_dir, manifest)

    objectives  = summary["objectives"]
    pareto_keys = summary["pareto_frontier"]
    config_keys = list(objectives.keys())
    title_base  = f"Scenario: {label}"

    print(f"\n── {result_dir.name}  ({len(config_keys)} configs, {len(pareto_keys)} Pareto) ──")

    # 1a. Pareto scatter (three 2-D panels)
    fig = plot_pareto_scatter(objectives, pareto_keys, title=f"{title_base} — Pareto scatter")
    _save(fig, out_dir, "pareto_scatter", fmt)

    # 1b. Three-objective scatter (V × Q × CF in one panel)
    fig = plot_pareto_3d_scatter(objectives, pareto_keys, title=f"{title_base} — V / Q / CF")
    _save(fig, out_dir, "pareto_3d_scatter", fmt)

    # 2. Objectives bar
    fig = plot_objectives_bar(objectives, pareto_keys, title=f"{title_base} — Objectives")
    _save(fig, out_dir, "objectives_bar", fmt)

    # 3. Replication distributions
    reps = _load_per_config_replications(result_dir, config_keys)
    if reps:
        fig = plot_replication_distributions(reps, pareto_keys, title=f"{title_base} — Replication distributions")
        _save(fig, out_dir, "replication_distributions", fmt)

    # 4. Trajectories (optional — needs traces.csv)
    traces = _load_traces(result_dir, pareto_keys)
    if traces:
        fig = plot_trajectories(traces, pareto_keys, title=f"{title_base} — State trajectories (Pareto configs)")
        _save(fig, out_dir, "trajectories", fmt)
    else:
        warnings.warn(
            f"No traces.csv found in {result_dir}. Re-run the sweep with --traces to enable trajectory plots.",
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# Multi-experiment comparison
# ---------------------------------------------------------------------------

def visualize_multi(result_dirs: list[Path], out_dir: Path, fmt: str) -> None:
    scenarios: dict[str, tuple[dict, list[str]]] = {}
    for rd in result_dirs:
        manifest = _load_manifest(rd)
        summary  = _load_summary(rd)
        label    = _scenario_label(rd, manifest)
        scenarios[label] = (summary["objectives"], summary["pareto_frontier"])

    print(f"\n── Cross-experiment comparison ({len(scenarios)} scenarios) ──")
    fig = plot_comparison_pareto(scenarios, title="Pareto comparison across scenarios")
    _save(fig, out_dir, "comparison_pareto", fmt)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate visualizations from persisted sweep results.",
    )
    parser.add_argument(
        "result_dirs", nargs="+", metavar="RESULT_DIR",
        help="One or more result directories produced by run_sweep.py",
    )
    parser.add_argument(
        "--format", default="png", choices=["png", "pdf", "svg"],
        help="Output image format (default: png)",
    )
    parser.add_argument(
        "--out", default=None, metavar="DIR",
        help="Output directory for plots (default: RESULT_DIR/plots/ for single, ./plots/ for multi)",
    )
    args = parser.parse_args(argv)

    result_dirs = [Path(d) for d in args.result_dirs]
    for d in result_dirs:
        if not d.is_dir():
            print(f"Error: {d} is not a directory.", file=sys.stderr)
            sys.exit(1)

    fmt = args.format
    multi = len(result_dirs) > 1

    if multi:
        # Per-experiment plots always go to their own {result_dir}/plots/
        for rd in result_dirs:
            visualize_single(rd, rd / "plots", fmt)

        # Cross-experiment comparison goes to --out (default: ./plots/)
        cmp_out = Path(args.out) if args.out else Path("plots")
        visualize_multi(result_dirs, cmp_out, fmt)
    else:
        out = Path(args.out) if args.out else result_dirs[0] / "plots"
        visualize_single(result_dirs[0], out, fmt)

    print("\nDone.")


if __name__ == "__main__":
    main()
