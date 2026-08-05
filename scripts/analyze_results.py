#!/usr/bin/env python3
"""Produce the Section 5 results analysis from a finished §4.2 campaign.

Follows paper/main.tex §5, which analyses the frontiers by objective:

1. *Value.*  Check that ``value`` equals the feature arrival rate ``r_arrival``
   for the whole Pareto front of every scenario, and characterise the points
   where it does not.
2. *Throughput filter.*  Since that holds, restrict the candidates to those within
   98 % of ``T * r_arrival`` — the configurations that maximise throughput.
3. *Quality vs. cost.*  Compute a fresh Pareto front over the two remaining
   objectives on that filtered pool, and plot it against total cost, human
   developer cost, and agent cost.

Reads the campaign written by ``python -m src.main optimize`` and writes tables
and figures for the paper.  Frontiers are analysed per (scenario, strategy).

The filtered front is computed over *all* simulated candidates, not just the
recorded three-objective frontier: dropping the value objective can revive a
candidate that was eliminated in three dimensions on a value significance.  The
``--frontiers`` fast path only has the recorded frontier available and therefore
gives a slightly conservative filtered front; the script says so when used.

Usage
-----
    python scripts/analyze_results.py
    python scripts/analyze_results.py --frontiers results/experiments/frontiers.csv
    python scripts/analyze_results.py --out-dir /tmp/analysis --format png --per-scenario
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.config import SCENARIO_GROUPS                       # noqa: E402
from src.results_analysis import (                           # noqa: E402
    DEFAULT_THROUGHPUT_THRESHOLD,
    DEFAULT_VALUE_THRESHOLD,
    add_derived_columns,
    build_campaign,
    filtered_front_from_table,
    frontier_extremes,
    headline_summary,
    load_frontier_table,
    rank_frontier,
    value_deviation_patterns,
    value_saturation_summary,
)
from src.visualization import (                              # noqa: E402
    plot_quality_cost_facets,
    plot_quality_cost_panels,
    plot_value_saturation,
    save_figure,
)

DEFAULT_RESULTS_DIR = "results/experiments"
DEFAULT_OUT_DIR = "results/analysis"

#: Tier column used to colour the aggregate quality-vs-cost figure.  bugimpact
#: sets the ω weights and therefore the vertical spread of the quality axis.
_AGGREGATE_COLOUR_BY = "tier_bugimpact"


def _print_headline(summary: dict, threshold: float) -> None:
    print("\n── Section 5 headline numbers ──")
    print(f"  scenarios / searches / frontier points: "
          f"{summary['n_scenarios']} / {summary['n_searches']} / {summary['n_frontier_points']}")
    print("\n  value vs. arrival rate")
    print(f"    median value/r_arrival        {summary['value_ratio_median']:.4f}")
    print(f"    5th percentile                {summary['value_ratio_p05']:.4f}")
    print(f"    range                         [{summary['value_ratio_min']:.4f}, "
          f"{summary['value_ratio_max']:.4f}]")
    print(f"    share below {threshold:.2f}              {summary['frac_below_threshold']:.1%}")
    print(f"    share throughput-limited (CI) {summary['frac_throughput_limited']:.1%}")
    factor = summary.get("dominant_deviation_factor")
    if factor:
        print(f"\n  deviations are explained best by: {factor}")
        for level in summary["dominant_deviation_levels"]:
            print(f"    {str(level['level']):>8}  n={level['n_points']:>6}  "
                  f"throughput-limited {level['frac_throughput_limited']:6.1%}  "
                  f"median deficit {level['median_deficit']:+.4f}")
    if "throughput_threshold" in summary:
        print(f"\n  throughput filter (value >= {summary['throughput_threshold']:.2f}"
              f" * r_arrival)")
        print(f"    candidates passing            {summary['n_passed_total']:,} / "
              f"{summary['n_simulated_total']:,}  ({summary['frac_passed_overall']:.1%})")
        print(f"    lowest per-search pass rate   {summary['frac_passed_min']:.1%}")
        print(f"    filtered front size           median {summary['filtered_front_size_median']:.0f}"
              f"  (range {summary['filtered_front_size_min']}–"
              f"{summary['filtered_front_size_max']})")
        print(f"    vs recorded 3-D front         median "
              f"{summary['recorded_3d_front_size_median']:.0f}")
        print(f"    total filtered-front points   {summary['n_filtered_front_points']:,}")

    print(f"\n  quality / cost  (on the {summary.get('quality_cost_source', 'frontier')})")
    print(f"    ratio median                  {summary['quality_cost_ratio_median']:.4g}")
    print(f"    ratio range                   [{summary['quality_cost_ratio_min']:.4g}, "
          f"{summary['quality_cost_ratio_max']:.4g}]")
    print(f"    median agent share of cost    {summary['agent_cost_share_median']:.1%}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--results-dir", default=None, metavar="DIR",
                     help=f"Campaign output root (default: {DEFAULT_RESULTS_DIR})")
    src.add_argument("--frontiers", default=None, metavar="PATH",
                     help="Existing frontiers.csv from collect_results.py (fast path); "
                          "T is read from the sibling scenarios.csv")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, metavar="DIR",
                        help=f"Where to write tables and plots (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--config-dir", default="config", metavar="DIR",
                        help="Directory holding the <group>_<tier>.json scenario files "
                             "(default: config)")
    parser.add_argument("--value-threshold", type=float, default=DEFAULT_VALUE_THRESHOLD,
                        metavar="R",
                        help="value/r_arrival below which a frontier point counts as "
                             f"deviating (default: {DEFAULT_VALUE_THRESHOLD})")
    parser.add_argument("--throughput-threshold", type=float,
                        default=DEFAULT_THROUGHPUT_THRESHOLD, metavar="R",
                        help="value/r_arrival at or above which a candidate enters the "
                             "throughput-filtered quality/cost front "
                             f"(default: {DEFAULT_THROUGHPUT_THRESHOLD})")
    parser.add_argument("--extremes", type=int, default=5, metavar="K",
                        help="Configurations reported per end of each frontier (default: 5)")
    parser.add_argument("--format", default="pdf", choices=["png", "pdf", "svg"],
                        help="Output image format (default: pdf)")
    parser.add_argument("--per-scenario", action="store_true",
                        help="Also write one quality-vs-cost figure per (scenario, strategy); "
                             "up to 486 files")
    args = parser.parse_args(argv)

    filter_summary = None
    if args.frontiers:
        table = load_frontier_table(frontiers_csv=args.frontiers)
        source = args.frontiers
        df = add_derived_columns(table, config_dir=args.config_dir)
        print("  note: --frontiers only carries the recorded three-objective frontier, "
              "so the\n        filtered front is computed from that pool and may miss "
              "CI-retained points.\n        Use --results-dir for the exact front.")
        filtered = filtered_front_from_table(df, args.throughput_threshold)
    else:
        results_dir = Path(args.results_dir or DEFAULT_RESULTS_DIR)
        if not results_dir.is_dir():
            parser.error(f"{results_dir} is not a directory")
        print(f"Reading campaign from {results_dir} … (this parses every search_summary.json)")
        campaign = build_campaign(
            results_dir,
            throughput_threshold=args.throughput_threshold,
            config_dir=args.config_dir,
        )
        source = str(results_dir)
        table = campaign.frontiers
        df = add_derived_columns(table, config_dir=args.config_dir)
        filtered = campaign.filtered_front
        filter_summary = campaign.filter_summary

    if table.empty:
        print(f"error: no frontier points found in {source}", file=sys.stderr)
        return 1

    print(f"Loaded {len(table)} frontier points from {source}; "
          f"{len(filtered)} on the throughput-filtered quality/cost front")

    out_dir = Path(args.out_dir)
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- tables --
    # Bullet 1 describes the recorded three-objective frontier; the quality/cost
    # tables describe the throughput-filtered two-objective front.
    tables = {
        "frontiers_derived": df,
        "value_saturation": value_saturation_summary(df, args.value_threshold),
        "value_deviation_patterns": value_deviation_patterns(df, args.value_threshold),
        "filtered_front": filtered,
        "quality_cost_ranking": rank_frontier(filtered),
        "frontier_extremes": frontier_extremes(filtered, k=args.extremes),
    }
    if filter_summary is not None:
        tables["filtered_front_summary"] = filter_summary
    print()
    for name, tbl in tables.items():
        path = out_dir / f"{name}.csv"
        tbl.to_csv(path, index=False)
        print(f"  Saved: {path}  ({len(tbl)} rows)")

    summary = headline_summary(
        df, args.value_threshold,
        filtered_front=filtered, filter_summary=filter_summary,
    )
    summary_path = out_dir / "headline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"  Saved: {summary_path}")

    # -- plots --
    print()
    fig = plot_value_saturation(df, threshold=args.value_threshold)
    save_figure(fig, plots_dir, "value_saturation", args.format)

    thr = args.throughput_threshold
    front_caption = (f"throughput-filtered Pareto front "
                     f"(value ≥ {thr:.2f}·r_arrival)")

    fig = plot_quality_cost_panels(
        filtered,
        colour_by=_AGGREGATE_COLOUR_BY,
        title=f"Quality vs. cost — {front_caption}",
    )
    save_figure(fig, plots_dir, "quality_vs_cost", args.format)

    for group in SCENARIO_GROUPS:
        fig = plot_quality_cost_facets(
            filtered, group,
            title=f"Quality vs. cost by {group} tier — {front_caption}",
        )
        save_figure(fig, plots_dir, f"quality_vs_cost_by_{group}", args.format)

    if args.per_scenario:
        per_dir = plots_dir / "per_scenario"
        pairs = list(filtered.groupby(["scenario_id", "strategy"], sort=True))
        print(f"\n── Per-scenario figures ({len(pairs)}) ──")
        for (scenario_id, strategy), sub in pairs:
            fig = plot_quality_cost_panels(
                sub, title=f"{scenario_id} / {strategy} — {front_caption}",
            )
            save_figure(fig, per_dir, f"{scenario_id}_{strategy}", args.format)

    _print_headline(summary, args.value_threshold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
