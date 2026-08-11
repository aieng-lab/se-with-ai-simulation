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
Tables the paper typesets are additionally exported as LaTeX snippets under
``<out-dir>/tables/`` (see ``src/latex_tables.py``).

The filtered front is computed over *all* simulated candidates, not just the
recorded three-objective frontier: dropping the value objective can revive a
candidate that was eliminated in three dimensions on a value significance.  The
``--frontiers`` fast path only has the recorded frontier available and therefore
gives a slightly conservative filtered front; the script says so when used.

There are three input sources, in decreasing order of work:
``--results-dir`` parses the campaign, ``--frontiers`` reads a collected
frontiers.csv, and ``--filtered-front`` reads a filtered_front.csv this script
wrote earlier and only redraws the figures that depend on it.

Usage
-----
    python scripts/analyze_results.py
    python scripts/analyze_results.py --frontiers results/experiments/frontiers.csv
    python scripts/analyze_results.py --filtered-front results/analysis/filtered_front.csv
    python scripts/analyze_results.py --out-dir /tmp/analysis --format png --per-scenario
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.composition_analysis import (                       # noqa: E402
    DEFAULT_ARCHETYPES,
    DEFAULT_BINS,
    PHASES,
    archetypes,
    composition_by_front_position,
    composition_by_tier,
    composition_headline,
    role_prevalence,
)
from src.config import SCENARIO_GROUPS                       # noqa: E402
from src.latex_tables import index_optima_latex, save_latex_table   # noqa: E402
from src.results_analysis import (                           # noqa: E402
    DEFAULT_THROUGHPUT_THRESHOLD,
    DEFAULT_VALUE_THRESHOLD,
    RATIO_COST_FIELDS,
    STRATEGIES,
    add_derived_columns,
    build_campaign,
    filtered_front_from_table,
    frontier_extremes,
    headline_summary,
    index_optimum,
    load_frontier_table,
    rank_frontier,
    value_deviation_patterns,
    value_saturation_summary,
)
from src.visualization import (                              # noqa: E402
    plot_composition_archetypes,
    plot_composition_drift,
    plot_composition_tier_heatmap,
    plot_quality_cost_facets,
    plot_quality_cost_matrix,
    plot_quality_cost_panels,
    plot_ratio_percentile_matrix,
    plot_value_saturation,
    save_figure,
)

DEFAULT_RESULTS_DIR = "results/experiments"
DEFAULT_OUT_DIR = "results/analysis"

#: Tier column used to colour the aggregate quality-vs-cost figure.  bugimpact
#: sets the ω weights and therefore the vertical spread of the quality axis.
_AGGREGATE_COLOUR_BY = "tier_bugimpact"

#: Cost columns the quality/cost matrix figure can put on its x-axis.
_COST_FIELDS = ("cost_mean", "cost_human_mean", "cost_agent_mean")

#: Default channel → scenario-group assignment of the quality/cost matrix figure.
_MATRIX_DEFAULTS = {
    "x": "agent", "y": "human", "size": "bugimpact",
    "marker": "project", "color": "debt",
}


def _composition_phases(choice: str) -> list[str]:
    """Resolve --composition-phase to the `phase` arguments to draw."""
    return [*PHASES, "both"] if choice == "all" else [choice]


def _composition_tables(filtered, args) -> dict:
    """The composition CSVs, long-form with a `phase` column so each is a single file."""
    import pandas as pd

    out = {}
    for name, build in (
        ("composition_prevalence", lambda d, p: role_prevalence(d, p)),
        ("composition_by_front_position",
         lambda d, p: composition_by_front_position(d, p, bins=args.composition_bins)),
        ("composition_by_tier", lambda d, p: composition_by_tier(d, p)),
        ("composition_archetypes",
         lambda d, p: archetypes(d, p, top_k=args.composition_archetypes)),
    ):
        frames = [build(filtered, phase) for phase in PHASES]
        frames = [f for f in frames if not f.empty]
        out[name] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out


def _composition_figures(filtered, plots_dir: Path, front_caption: str, args) -> None:
    """Write the archetype, drift and tier figures for every requested phase/strategy."""
    phases = _composition_phases(args.composition_phase)
    strategies = (
        STRATEGIES if args.composition_strategy == "all" else (args.composition_strategy,)
    )
    # The drift figure facets the strategies itself, so it is written once per phase —
    # but it still has to honour a --composition-strategy restriction.
    restricted = args.composition_strategy != "all"
    drift_source = (
        filtered[filtered["strategy"] == args.composition_strategy] if restricted
        else filtered
    )
    drift_suffix = f"_{args.composition_strategy}" if restricted else ""

    print()
    for phase in phases:
        if drift_source.empty:
            print(f"  skipped composition_drift_{phase}{drift_suffix}: no front points")
        else:
            fig = plot_composition_drift(
                drift_source, phase=phase, bins=args.composition_bins,
                title=f"Composition along the front ({phase}) — {front_caption}",
            )
            save_figure(fig, plots_dir,
                        f"composition_drift_{phase}{drift_suffix}", args.format)

        # The archetype and tier figures draw one row per archetype / tier, so they must
        # not mix strategies — see plot_composition_archetypes.
        for strategy in strategies:
            sub = filtered[filtered["strategy"] == strategy]
            if sub.empty:
                print(f"  skipped composition_*_{phase}_{strategy}: no front points")
                continue
            fig = plot_composition_archetypes(
                sub, phase=phase, top_k=args.composition_archetypes,
                title=f"Team composition archetypes ({phase}) — {strategy} search "
                      f"— {front_caption}",
            )
            save_figure(fig, plots_dir,
                        f"composition_archetypes_{phase}_{strategy}", args.format)

            fig = plot_composition_tier_heatmap(
                sub, phase=phase,
                title=f"Composition by scenario tier ({phase}) — {strategy} search "
                      f"— {front_caption}",
            )
            save_figure(fig, plots_dir,
                        f"composition_tiers_{phase}_{strategy}", args.format)


def _write_plots(filtered, df, plots_dir: Path, args) -> None:
    """Every figure of §5.

    `filtered` is the throughput-filtered quality/cost front and drives all of them.
    `df` is the recorded three-objective frontier, which only the value-saturation
    figure needs; pass `None` on the `--filtered-front` fast path, where the recorded
    frontier is not available (and where the figure would be vacuous anyway — every
    point of the filtered front clears the threshold by construction).
    """
    print()
    if df is not None:
        fig = plot_value_saturation(df, threshold=args.value_threshold)
        save_figure(fig, plots_dir, "value_saturation", args.format)

    thr = args.throughput_threshold
    front_caption = f"throughput-filtered Pareto front (value ≥ {thr:.2f}·r_arrival)"

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

    # The 3×3 matrix separates all five groups at once, so it is drawn one strategy at
    # a time: random and evolutionary differ by an order of magnitude in point count.
    # Its ratio variant shares the layout and the channel assignment, so the two figures
    # read the same way.
    strategies = STRATEGIES if args.matrix_strategy == "all" else (args.matrix_strategy,)
    for strategy in strategies:
        sub = filtered[filtered["strategy"] == strategy]
        if sub.empty:
            print(f"  skipped quality_cost_matrix_{strategy}: no front points")
            continue
        channels = dict(
            x_group=args.matrix_x,
            y_group=args.matrix_y,
            size_group=args.matrix_size,
            marker_group=args.matrix_marker,
            color_group=args.matrix_color,
        )
        fig = plot_quality_cost_matrix(
            sub, **channels,
            cost_field=args.matrix_cost_field,
            title=f"Quality×cost index vs. cost — {strategy} search — {front_caption}",
        )
        save_figure(fig, plots_dir, f"quality_cost_matrix_{strategy}", args.format)

        fig = plot_ratio_percentile_matrix(
            sub, **channels,
            cost_field=args.ratio_cost_field,
            title=f"Quality×cost index along the front — {strategy} search "
                  f"— {front_caption}",
        )
        save_figure(fig, plots_dir, f"ratio_percentile_matrix_{strategy}", args.format)

    if not args.no_composition:
        _composition_figures(filtered, plots_dir, front_caption, args)

    if args.per_scenario:
        per_dir = plots_dir / "per_scenario"
        pairs = list(filtered.groupby(["scenario_id", "strategy"], sort=True))
        print(f"\n── Per-scenario figures ({len(pairs)}) ──")
        for (scenario_id, strategy), sub in pairs:
            fig = plot_quality_cost_panels(
                sub, title=f"{scenario_id} / {strategy} — {front_caption}",
            )
            save_figure(fig, per_dir, f"{scenario_id}_{strategy}", args.format)


def _write_index_optima(filtered, out_dir: Path, args):
    """The per-scenario quality×cost optimum, as a CSV and as a LaTeX snippet.

    Derived from the filtered front alone, so every input path can produce it.
    """
    optima = index_optimum(filtered, cost_field=args.ratio_cost_field,
                           by=("scenario_id",))
    if optima.empty:
        print("  skipped index_optima: no front points")
        return optima

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "index_optima.csv"
    optima.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}  ({len(optima)} rows)")
    save_latex_table(index_optima_latex(optima), out_dir / "tables", "index_optima")

    per_strategy = optima["strategy"].value_counts().to_dict()
    print(f"    optimum index median          "
          f"{optima['quality_cost_index'].median():.4g}")
    print("    scenarios won per strategy    "
          + ", ".join(f"{s} {n}" for s, n in sorted(per_strategy.items())))
    return optima


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
    print(f"    ratio median (front position) {summary['quality_cost_ratio_median']:.4g}")
    print(f"    ratio range                   [{summary['quality_cost_ratio_min']:.4g}, "
          f"{summary['quality_cost_ratio_max']:.4g}]")
    print(f"    index median (quality × cost) {summary['quality_cost_index_median']:.4g}")
    print(f"    index range                   [{summary['quality_cost_index_min']:.4g}, "
          f"{summary['quality_cost_index_max']:.4g}]")
    print(f"    median agent share of cost    {summary['agent_cost_share_median']:.1%}")

    for phase, entry in (summary.get("composition") or {}).items():
        if not entry.get("n_points"):
            continue
        print(f"\n  team composition ({phase} phase)")
        print(f"    front points                  {entry['n_points']:,}")
        print(f"    distinct role-presence sets   {entry['n_distinct_signatures']}")
        print(f"    mean supervised / background  {entry['mean_n_sup']:.1f} / "
              f"{entry['mean_n_bg']:.1f} agents "
              f"({entry['mean_agents_per_fte']:.1f} supervised per FTE)")
        for key, top in sorted(entry.items()):
            if not key.startswith("top_archetype_"):
                continue
            strategy = key[len("top_archetype_"):]
            print(f"    most common ({strategy}): {top['label']}")
            print(f"      {top['prevalence']:.1%} of front points, "
                  f"in {top['scenario_coverage']:.0%} of scenarios")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--results-dir", default=None, metavar="DIR",
                     help=f"Campaign output root (default: {DEFAULT_RESULTS_DIR})")
    src.add_argument("--frontiers", default=None, metavar="PATH",
                     help="Existing frontiers.csv from collect_results.py (fast path); "
                          "T is read from the sibling scenarios.csv")
    src.add_argument("--filtered-front", default=None, metavar="PATH",
                     help="Existing filtered_front.csv from an earlier run: redraws the "
                          "figures from it in seconds and writes no tables")
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

    matrix = parser.add_argument_group(
        "quality/cost matrix figures",
        "Two 3×3 grids separating all five scenario groups at once, both drawing the "
        "combined quality×cost index on y: against cost, and against position along "
        "the front.  Each of the five channel options must name a different group; "
        "both figures share the channel assignment and are written once per strategy.",
    )
    for channel, what in (("x", "column facet"), ("y", "row facet"),
                          ("color", "marker colour"), ("marker", "marker shape"),
                          ("size", "marker size")):
        matrix.add_argument(
            f"--matrix-{channel}", default=_MATRIX_DEFAULTS[channel],
            choices=SCENARIO_GROUPS, metavar="GROUP",
            help=f"Scenario group shown as the {what} "
                 f"(default: {_MATRIX_DEFAULTS[channel]})",
        )
    matrix.add_argument("--matrix-cost-field", default=_COST_FIELDS[0],
                        choices=_COST_FIELDS,
                        help=f"Cost column on the x-axis (default: {_COST_FIELDS[0]})")
    matrix.add_argument("--ratio-cost-field", default=RATIO_COST_FIELDS[0],
                        choices=RATIO_COST_FIELDS,
                        help="Cost term of the index and the front-position ratio in "
                             f"the ratio_percentile_matrix figure "
                             f"(default: {RATIO_COST_FIELDS[0]})")
    matrix.add_argument("--matrix-strategy", default="all",
                        choices=(*STRATEGIES, "all"),
                        help="Search strategy to plot; 'all' writes one figure per "
                             "strategy (default: all)")

    composition = parser.add_argument_group(
        "team composition figures",
        "What the teams on the filtered front look like: the recurring archetypes, how "
        "the mix drifts along the cost axis, and how it responds to the scenario tiers.",
    )
    composition.add_argument("--composition-phase", default="all",
                             choices=(*PHASES, "both", "all"),
                             help="Sprint phase whose composition is drawn; 'both' puts "
                                  "the two side by side and 'all' writes every variant "
                                  "(default: all)")
    composition.add_argument("--composition-strategy", default="all",
                             choices=(*STRATEGIES, "all"),
                             help="Search strategy for the archetype and tier figures; "
                                  "'all' writes one figure per strategy (default: all)")
    composition.add_argument("--composition-bins", type=int, default=DEFAULT_BINS,
                             metavar="N",
                             help=f"Within-front cost quantiles in the drift figure "
                                  f"(default: {DEFAULT_BINS})")
    composition.add_argument("--composition-archetypes", type=int,
                             default=DEFAULT_ARCHETYPES, metavar="K",
                             help=f"Archetypes kept before the rest are folded into "
                                  f"'other' (default: {DEFAULT_ARCHETYPES})")
    composition.add_argument("--no-composition", action="store_true",
                             help="Skip the composition tables and figures entirely")

    args = parser.parse_args(argv)

    # Fail before the multi-minute campaign parse, not after it.
    assigned = [args.matrix_x, args.matrix_y, args.matrix_color,
                args.matrix_marker, args.matrix_size]
    if len(set(assigned)) != len(assigned):
        parser.error("--matrix-x/-y/-color/-marker/-size must name five different "
                     f"scenario groups (got {assigned})")
    if args.composition_bins < 1:
        parser.error("--composition-bins must be >= 1")
    if args.composition_archetypes < 1:
        parser.error("--composition-archetypes must be >= 1")

    # Fast path: the filtered front has already been written, so everything derived
    # from it alone can be rebuilt without touching the campaign.
    if args.filtered_front:
        import pandas as pd

        filtered = pd.read_csv(args.filtered_front)
        if filtered.empty:
            print(f"error: {args.filtered_front} has no rows", file=sys.stderr)
            return 1
        print(f"Loaded {len(filtered)} filtered-front points from {args.filtered_front}")
        print("  note: --filtered-front rebuilds only what the filtered front alone "
              "determines —\n        the figures and index_optima.{csv,tex}.  The "
              "campaign-level CSVs and\n        headline_summary.json are not written, "
              "and value_saturation is skipped\n        (every point of this front "
              "clears the throughput threshold by construction).")
        out_dir = Path(args.out_dir)
        print()
        _write_index_optima(filtered, out_dir, args)
        _write_plots(filtered, None, out_dir / "plots", args)
        return 0

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
    if not args.no_composition:
        tables.update(_composition_tables(filtered, args))
    print()
    for name, tbl in tables.items():
        path = out_dir / f"{name}.csv"
        tbl.to_csv(path, index=False)
        print(f"  Saved: {path}  ({len(tbl)} rows)")

    # Writes its own CSV plus the LaTeX snippet, so it is not part of `tables`.
    _write_index_optima(filtered, out_dir, args)

    summary = headline_summary(
        df, args.value_threshold,
        filtered_front=filtered, filter_summary=filter_summary,
    )
    if not args.no_composition:
        summary["composition"] = composition_headline(
            filtered, bins=args.composition_bins, top_k=args.composition_archetypes,
        )
    summary_path = out_dir / "headline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"  Saved: {summary_path}")

    _write_plots(filtered, df, plots_dir, args)

    _print_headline(summary, args.value_threshold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
