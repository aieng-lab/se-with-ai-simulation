"""CLI entry point for agent-ai-cost-sim.

Usage
-----
Run a simulation (smoke-test defaults):
    python -m src.main run

Run with custom settings and save results:
    python -m src.main run --out results/

Run and also save per-sprint traces:
    python -m src.main run --out results/ --traces

Load and analyse persisted results:
    python -m src.main analyze results/result.json
    python -m src.main analyze results/result.json results/traces.csv

Search for the cheapest team that keeps up with the project:
    python -m src.main optimize config/search_default.json --out results/search_default
    python -m src.main optimize config/search_default.json --out results/search_descent \
        --strategy descent
    python -m src.main optimize config/search/search_h-l_p-m_d-h_a-l_b-m.json \
        --samples none --progress-every 500 --out results/one_scenario

Run all 432 scenarios of paper Section 4.2 with both strategies:
    ./run_experiments.sh          (see its --help; generate the configs first with
                                   scripts/generate_search_configs.py)

Generate visualizations from optimization results:
    python -m src.main visualize results/search_default [--out results/search_default/plots]
    python -m src.main visualize results/run1 results/run2 --out results/comparison

Consolidate a finished campaign into its tables and the five-dimension optima figure:
    python -m src.main campaign [results/experiments] [--out results/analysis]
    python -m src.main campaign --share human --x debt --color agent
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from .analysis import final_distribution, mean_trajectory
from .config import (
    AGENT_ROLE_NAMES,
    BACKGROUND_ROLE_INDICES,
    SAMPLE_POLICIES,
    SCENARIO_GROUPS,
    SEARCH_STRATEGIES,
    TeamConfiguration,
    supervised_agent_count,
)
from .io import load_initial_state, load_scenario_params, load_search_config, load_simulation_horizon, load_team_config
from .results import load_result, load_traces, save_result, save_traces
from .search_configurations import _run_descent, _run_evolutionary_search, write_samples
from .selection import (
    aggregate_config_result,
    parsimony_key,
    report_optimum,
    satisfies_throughput,
    throughput_floor,
)
from .campaign import build_tables, search_dirs
from .simulation import ConfigResult, ConfigTrace, run_configuration
from .visualization import COST_SHARES, visualize_campaign, visualize_multi, visualize_single


#: Default file per scenario group; every group defaults to its medium tier.
DEFAULT_SCENARIO_PATHS: dict[str, str] = {
    group: f"config/{group}_medium.json" for group in SCENARIO_GROUPS
}


def _add_scenario_arguments(parser: argparse.ArgumentParser) -> None:
    """Add one --<group> path option per scenario group (paper §4.1)."""
    for group, default in DEFAULT_SCENARIO_PATHS.items():
        parser.add_argument(
            f"--{group}", default=default, metavar="PATH",
            help=f"{group} scenario parameters JSON file (default: {default})",
        )


def _scenario_paths_from_args(args: argparse.Namespace) -> dict[str, str]:
    """Collect the five --<group> options into a group -> path mapping."""
    return {group: getattr(args, group) for group in SCENARIO_GROUPS}


def _print_config(config: TeamConfiguration) -> None:
    print(f"Team configuration:  N={config.N:g} FTE, R={config.R}, "
          f"F_freeze={config.F_freeze}")
    for phase, A, getter in (
        ("Dev phase ", config.A_dev, config.agent_count_dev),
        ("QA phase  ", config.A_qa, config.agent_count_qa),
    ):
        n_sup = supervised_agent_count(A)   # type: ignore[arg-type]
        n_bg = sum(A) - n_sup              # type: ignore[arg-type]
        print(f"  {phase} ({n_sup} supervised, {n_bg} background)")
        for k, name in enumerate(AGENT_ROLE_NAMES, start=1):
            count = getter(k)
            if count:
                tier = "bg " if (k - 1) in BACKGROUND_ROLE_INDICES else "   "
                print(f"    role {k:>2} {tier} {name:<48}  x{count}")


def _cmd_run(args: argparse.Namespace) -> None:
    print("agent-ai-cost-sim — Monte Carlo simulation")
    print("=" * 52)

    horizon = load_simulation_horizon(args.horizon)
    config = load_team_config(args.team)
    scenario_paths = _scenario_paths_from_args(args)
    params = load_scenario_params(scenario_paths)
    _print_config(config)
    print()

    initial_state = load_initial_state(args.start_state)

    print("Scenario:")
    for group, path in scenario_paths.items():
        print(f"  {group:<10} {path}")
    print(f"Team:       {args.team}")
    print(f"Start state:{args.start_state}")
    print(f"Horizon:   K={horizon.K} replications * T={horizon.T} sprints  "
          f"(seed={horizon.master_seed})")
    print(f"Traces:     {'yes' if args.traces else 'no'}")
    print()
    print("Running …")

    trace: ConfigTrace | None = None

    if args.traces:
        trace = run_configuration(config, horizon, params, capture_traces=True,
                                  initial_state=initial_state)
        cr = ConfigResult(config=config, replications=trace.replications)
    else:
        cr = run_configuration(config, horizon, params, initial_state=initial_state)

    agg = aggregate_config_result(cr, master_seed=horizon.master_seed)

    print()
    print("Objective and constraint (mean [95 % CI]):")
    print(f"  cost   $/feature         {agg.cost_mean:.1f}   [{agg.cost_ci[0]:.1f}, {agg.cost_ci[1]:.1f}]")
    print(f"  value  features/sprint   {agg.value_mean:.3f}   [{agg.value_ci[0]:.3f}, {agg.value_ci[1]:.3f}]")
    print("Cost drivers ($/feature):")
    print(f"  escaped bugs {agg.cost_quality_mean:.1f}   "
          f"developers {agg.cost_human_mean:.1f}   agents {agg.cost_agent_mean:.1f}")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        result_path = out / "result.json"
        save_result(cr, agg, result_path)
        print(f"\nSaved result  → {result_path}")
        if args.traces and trace is not None:
            traces_path = out / "traces.csv"
            save_traces(trace, traces_path)
            print(f"Saved traces  → {traces_path}")


def _cmd_analyze(args: argparse.Namespace) -> None:
    result, agg = load_result(args.result)
    config = result.config

    print("agent-ai-cost-sim — result analysis")
    print("=" * 52)
    _print_config(config)
    print()
    print(f"Source:      {args.result}")
    print(f"Replications: K={len(result.replications)}")
    print()
    print("Objective and constraint (mean [95 % CI]):")
    print(f"  cost   $/feature         {agg.cost_mean:.1f}   [{agg.cost_ci[0]:.1f}, {agg.cost_ci[1]:.1f}]")
    print(f"  value  features/sprint   {agg.value_mean:.3f}   [{agg.value_ci[0]:.3f}, {agg.value_ci[1]:.3f}]")
    print("Cost drivers ($/feature):")
    print(f"  escaped bugs {agg.cost_quality_mean:.1f}   "
          f"developers {agg.cost_human_mean:.1f}   agents {agg.cost_agent_mean:.1f}")

    if not args.traces:
        return

    df = load_traces(args.traces)
    T = df.index.get_level_values("sprint").max() + 1
    print()
    print(f"Trace file:  {args.traces}  ({len(result.replications)} reps × {T} sprints)")
    print()

    # Key trajectory fields to display
    fields = ["F_completed", "D_tech", "B_kno_min", "B_kno_maj", "doc_cov", "impl"]
    labels = ["F_done", "Debt", "B_kno_min", "B_kno_maj", "doc_cov", "impl/spr"]

    # Header
    col_w = 10
    header = f"{'sprint':>6}  " + "  ".join(f"{lbl:>{col_w}}" for lbl in labels)
    print(header)
    print("-" * len(header))

    # Print every sprint (or every 5th if T > 20)
    step = max(1, T // 20)
    sprints_to_show = list(range(0, T, step))
    if (T - 1) not in sprints_to_show:
        sprints_to_show.append(T - 1)

    means = {f: mean_trajectory(df, f) for f in fields}
    for s in sprints_to_show:
        row = f"{s:>6}  "
        row += "  ".join(f"{means[f].iloc[s]:>{col_w}.3f}" for f in fields)
        print(row)

    print()
    print("Final-sprint distribution (across replications):")
    for f, lbl in zip(fields, labels):
        fd = final_distribution(df, f)
        print(f"  {lbl:<12}  min={fd.min():.3f}  mean={fd.mean():.3f}  max={fd.max():.3f}")


def _cmd_optimize(args: argparse.Namespace) -> None:
    cfg_path = Path(args.search_config)
    search_cfg = load_search_config(cfg_path)
    strategy = args.strategy or search_cfg.strategy
    if args.samples is not None:
        search_cfg = replace(search_cfg, samples=args.samples)

    horizon = load_simulation_horizon(search_cfg.horizon_path)
    scenario_values = {
        group: json.loads(Path(path).read_text(encoding="utf-8"))
        for group, path in search_cfg.scenario_paths.items()
    }
    params = load_scenario_params(search_cfg.scenario_paths)
    initial_state = load_initial_state(args.start_state)

    print(f"Search:      {cfg_path}")
    print(f"Scenario id: {search_cfg.scenario_id or '(unnamed)'}")
    print("Scenario:")
    for group, path in search_cfg.scenario_paths.items():
        print(f"  {group:<10} {path}")
    print(f"Start state: {args.start_state}")
    print(f"Strategy:    {strategy}")
    print(f"Horizon:     K={horizon.K} replications * T={horizon.T} sprints  "
          f"(seed={horizon.master_seed})")
    if strategy == "evolutionary":
        print(f"Budget:      {search_cfg.budget} candidate evaluations")
        print(f"             {search_cfg.n_samples_initial} initial + "
              f"{search_cfg.n_generations} generations * {search_cfg.n_offspring} offspring, "
              f"elite {search_cfg.n_elite}")
    else:
        # The descent is not budgeted (§4.2): each restart runs to a local optimum and the
        # evaluations it took are reported at the end.
        print(f"Budget:      none; {search_cfg.n_starts} restarts, "
              f"each descending to a local optimum")
    print(f"Throughput:  value >= (1 - {search_cfg.epsilon}) * {params.r_arrival:g} = "
          f"{throughput_floor(params, search_cfg.epsilon):.3f} features/sprint")
    print(f"Samples:     {search_cfg.samples}")
    print()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(search_cfg.seed)

    t_start = time.perf_counter()
    runner = {
        "evolutionary": _run_evolutionary_search,
        "descent": _run_descent,
    }[strategy]
    outcome = runner(
        search_cfg, horizon, params, rng, initial_state,
        progress_every=args.progress_every,
    )
    wall_clock_s = time.perf_counter() - t_start

    aggs = outcome.aggs
    agg_objs = [a for _, _, a in aggs]
    floor = throughput_floor(params, search_cfg.epsilon)
    # c* and the set of admissible configurations it cannot be distinguished from (§3.10).
    # Parsimony keys are supplied, so of several equally good teams running the same
    # process only the one that uses no more agents anywhere is reported — including when
    # that removes the raw cost minimiser, which is why c* comes from report_optimum
    # rather than from best_index.
    best_idx, keep_idx = report_optimum(
        agg_objs, floor, keys=[parsimony_key(cfg) for _, cfg, _ in aggs]
    )
    best_key = aggs[best_idx][0] if best_idx is not None else None
    keep_keys = [aggs[i][0] for i in keep_idx]
    # c* may violate the throughput constraint when nothing in the pool satisfies it; say so
    # rather than presenting it as an admissible answer.
    best_admissible = (best_idx is not None
                       and satisfies_throughput(agg_objs[best_idx], floor))

    # Sample files are written only now, so that the "best" policy can restrict them to
    # the configurations that are actually reported.
    sample_keys = {
        "none": [],
        "best": keep_keys,
        "all": [key for key, _, _ in aggs],
    }[search_cfg.samples]
    n_samples_written = write_samples(outcome, out_dir, sample_keys)

    manifest_out = {
        "search_config": str(cfg_path),
        "scenario_id": search_cfg.scenario_id,
        "scenario": {"paths": search_cfg.scenario_paths, "values": scenario_values},
        "strategy": strategy,
        "horizon": {"path": search_cfg.horizon_path, "K": horizon.K, "T": horizon.T,
                    "master_seed": horizon.master_seed},
        "search": {
            "N_values": list(search_cfg.N_values),
            "R_values": list(search_cfg.R_values),
            "F_freeze_values": list(search_cfg.F_freeze_values),
            "sigma": search_cfg.sigma,
            "epsilon": search_cfg.epsilon,
            "budget": search_cfg.budget,
            "n_samples_initial": search_cfg.n_samples_initial,
            "n_generations": search_cfg.n_generations,
            "n_offspring": search_cfg.n_offspring,
            "n_elite": search_cfg.n_elite,
            "n_starts": search_cfg.n_starts,
            "samples": search_cfg.samples,
            "seed": search_cfg.seed,
        },
        # n_evaluated counts everything that consumed budget; n_simulated excludes the
        # candidates rejected by the human-capacity condition (paper §4.2).
        "n_evaluated": outcome.n_evaluated,
        "n_simulated": outcome.n_simulated,
        "n_infeasible": outcome.n_evaluated - outcome.n_simulated,
        "n_feasible": outcome.n_simulated,
        "throughput_floor": floor,
        "best_admissible": best_admissible,
        "n_indistinguishable": len(keep_keys),
        "n_samples_written": n_samples_written,
        "wall_clock_s": round(wall_clock_s, 1),
        "configs": outcome.manifest,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest_out), encoding="utf-8"
    )

    search_summary = {
        "best": best_key,
        "indistinguishable": keep_keys,
        "throughput_floor": floor,
        "objectives": {
            key: {
                "N": cfg.N, "R": cfg.R, "F_freeze": cfg.F_freeze,
                "A_dev": list(cfg.A_dev),
                "A_qa": list(cfg.A_qa),
                "cost_mean": agg.cost_mean, "cost_ci": list(agg.cost_ci),
                "value_mean": agg.value_mean, "value_ci": list(agg.value_ci),
                "feasible_throughput": satisfies_throughput(agg, floor),
                "cost_quality_mean": agg.cost_quality_mean,
                "cost_human_mean": agg.cost_human_mean,
                "cost_agent_mean": agg.cost_agent_mean,
                "actions_mean": list(agg.actions_mean),
                "impl_human_mean": agg.impl_human_mean,
                "impl_agents_mean": agg.impl_agents_mean,
                "fixes_human_mean": agg.fixes_human_mean,
                "fixes_agent_mean": agg.fixes_agent_mean,
            }
            for key, cfg, agg in aggs
        },
    }
    (out_dir / "search_summary.json").write_text(
        json.dumps(search_summary), encoding="utf-8"
    )

    print()
    print(f"Evaluated {outcome.n_evaluated} candidates "
          f"({outcome.n_simulated} simulated, "
          f"{outcome.n_evaluated - outcome.n_simulated} infeasible) "
          f"in {wall_clock_s:.1f} s")
    if best_key is None:
        print("No candidate was simulated — nothing to report.")
    else:
        obj = search_summary["objectives"][best_key]
        if not best_admissible:
            print(f"WARNING: no candidate reached the throughput floor of "
                  f"{floor:.3f} features/sprint; reporting the least-violating one.")
        print(f"Best configuration c* ({best_key}):")
        print(f"  N={obj['N']:g} R={obj['R']} F_freeze={obj['F_freeze']}  "
              f"cost={obj['cost_mean']:.0f} $/feature "
              f"[{obj['cost_ci'][0]:.0f}, {obj['cost_ci'][1]:.0f}]  "
              f"value={obj['value_mean']:.3f}")
        print(f"  cost split: escaped bugs {obj['cost_quality_mean']:.0f}   "
              f"developers {obj['cost_human_mean']:.0f}   "
              f"agents {obj['cost_agent_mean']:.0f}")
        print(f"  A_dev={obj['A_dev']}")
        print(f"  A_qa ={obj['A_qa']}")
        print(f"Statistically indistinguishable from c*: {len(keep_keys)} config(s) "
              f"(after the parsimony tie-break)")
    print()
    print(f"Results saved to: {out_dir}")


def _cmd_visualize(args: argparse.Namespace) -> None:
    result_dirs = [Path(d) for d in args.result_dirs]
    for d in result_dirs:
        if not d.is_dir():
            print(f"Error: {d} is not a directory.", file=sys.stderr)
            sys.exit(1)

    fmt = args.format
    multi = len(result_dirs) > 1

    if multi:
        for rd in result_dirs:
            visualize_single(rd, rd / "plots", fmt)
        cmp_out = Path(args.out) if args.out else Path("plots")
        visualize_multi(result_dirs, cmp_out, fmt)
    else:
        out = Path(args.out) if args.out else result_dirs[0] / "plots"
        visualize_single(result_dirs[0], out, fmt)

    print("\nDone.")


#: Which scenario group drives which visual channel of the §5 optima matrix, by default.
#: Two groups have a fourth tier: `project` sits on the shape channel, the only per-point
#: channel whose fourth level reads as a peer rather than as "more", and `agent` on a facet
#: axis, where a fourth tier costs only one extra column.
CAMPAIGN_CHANNEL_DEFAULTS: dict[str, str] = {
    "x": "agent", "y": "human", "color": "debt", "marker": "project", "size": "bugimpact",
}


def _cmd_campaign(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"Error: {results_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    searches = search_dirs(results_dir)
    if not searches:
        print(f"Error: no finished searches found under {results_dir}", file=sys.stderr)
        sys.exit(1)

    optima, scenarios = build_tables(results_dir, args.config_dir)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    optima_path = out_dir / "optima.csv"
    scenarios_path = out_dir / "scenarios.csv"
    optima.to_csv(optima_path, index=False)
    scenarios.to_csv(scenarios_path, index=False)

    n_scenarios = len({sid for sid, _, _ in searches})
    print(f"Collected {len(searches)} search(es) over {n_scenarios} scenario(s)")
    print(f"  {optima_path}   {len(optima)} reported optima")
    print(f"  {scenarios_path}   {len(scenarios)} (scenario, strategy) pairs")
    missing = len(SEARCH_STRATEGIES) * n_scenarios - len(searches)
    if missing:
        print(f"  note: {missing} (scenario, strategy) pair(s) have no results yet")
    if len(optima) < len(searches):
        print(f"  note: {len(searches) - len(optima)} search(es) simulated no candidate")

    if optima.empty:
        print("\nNo optimum was reported; skipping the figures.")
        return

    visualize_campaign(
        optima, out_dir, args.format, share=args.share,
        x_group=args.x, y_group=args.y, color_group=args.color,
        marker_group=args.marker, size_group=args.size,
    )
    print("\nDone.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.main",
        description="agent-ai-cost-sim: Monte Carlo simulation of AI+human dev teams.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", metavar="COMMAND")

    # -- run --
    p_run = sub.add_parser("run", help="Run a simulation.")
    _add_scenario_arguments(p_run)
    p_run.add_argument(
        "--team", default="config/team_default.json", metavar="PATH",
        help="Team configuration JSON file (default: config/team_default.json)",
    )
    p_run.add_argument(
        "--horizon", default="config/horizon_default.json", metavar="PATH",
        help="Simulation horizon JSON file (default: config/horizon_default.json)",
    )
    p_run.add_argument("--out", metavar="DIR",
                       help="Directory to write result.json [and traces.csv]")
    p_run.add_argument("--traces", action="store_true",
                       help="Capture and save full per-sprint time-series traces")
    p_run.add_argument(
        "--start-state", default="config/startstate_default.json", metavar="PATH",
        help="JSON file with initial SystemState values "
             "(default: config/startstate_default.json, which is all-zero)",
    )

    # -- analyze --
    p_ana = sub.add_parser("analyze", help="Analyse persisted results.")
    p_ana.add_argument("result", metavar="result.json",
                       help="Path to result.json written by 'run --out'")
    p_ana.add_argument("traces", nargs="?", metavar="traces.csv",
                       help="Path to traces.csv written by 'run --out --traces' (optional)")

    # -- optimize --
    p_optimize = sub.add_parser("optimize", help="Search for optimal team compositions.")
    p_optimize.add_argument("search_config", metavar="SEARCH_CONFIG",
                         help="Path to search config JSON (e.g. config/search_default.json)")
    p_optimize.add_argument("--out", required=True, metavar="DIR",
                         help="Output directory for optimization results")
    p_optimize.add_argument("--strategy", choices=list(SEARCH_STRATEGIES), default=None,
                         help="Override strategy from config")
    p_optimize.add_argument(
        "--samples", choices=list(SAMPLE_POLICIES), default=None,
        help="Which candidates get a samples/<key>.json file: 'all', 'best' "
             "(c* and the set indistinguishable from it), or 'none' "
             "(overrides the search config)",
    )
    p_optimize.add_argument(
        "--progress-every", type=int, default=100, metavar="N",
        help="Print a progress line every N candidates; 0 silences it (default: 100)",
    )
    p_optimize.add_argument(
        "--start-state", default="config/startstate_default.json", metavar="PATH",
        help="JSON file with initial SystemState values "
             "(default: config/startstate_default.json, which is all-zero)",
    )

    # -- visualize --
    p_vis = sub.add_parser("visualize", help="Generate plots from sweep results.")
    p_vis.add_argument("result_dirs", nargs="+", metavar="RESULT_DIR",
                       help="One or more result directories produced by 'optimize'")
    p_vis.add_argument("--format", default="png", choices=["png", "pdf", "svg"],
                       help="Output image format (default: png)")
    p_vis.add_argument("--out", default=None, metavar="DIR",
                       help="Output directory for plots (default: RESULT_DIR/plots/)")

    # -- campaign --
    p_camp = sub.add_parser(
        "campaign",
        help="Consolidate a whole campaign into optima.csv, scenarios.csv and the "
             "five-dimension optima figure.",
    )
    p_camp.add_argument("results_dir", nargs="?", default="results/experiments",
                        metavar="RESULTS_DIR",
                        help="Campaign output root (default: results/experiments)")
    p_camp.add_argument("--out", default="results/analysis", metavar="DIR",
                        help="Where to write the CSVs and figures "
                             "(default: results/analysis)")
    p_camp.add_argument("--config-dir", default="config", metavar="DIR",
                        help="Where the scenario JSONs live, for r_arrival "
                             "(default: config)")
    p_camp.add_argument("--format", default="png", choices=["png", "pdf", "svg"],
                        help="Output image format (default: png)")
    p_camp.add_argument("--share", default="agent", choices=list(COST_SHARES),
                        help="Which cost driver's share of the total goes on the x axis "
                             "(default: agent)")
    for channel, default in CAMPAIGN_CHANNEL_DEFAULTS.items():
        p_camp.add_argument(
            f"--{channel}", default=default, choices=list(SCENARIO_GROUPS),
            help=f"Scenario group on the {channel} channel (default: {default})",
        )

    args = parser.parse_args(argv)

    # Default: run smoke test if no subcommand given
    if args.cmd is None:
        args.cmd = "run"
        for group, default in DEFAULT_SCENARIO_PATHS.items():
            setattr(args, group, default)
        args.team = "config/team_default.json"
        args.horizon = "config/horizon_default.json"
        args.start_state = "config/startstate_default.json"
        args.out = None
        args.traces = False

    if args.cmd == "run":
        _cmd_run(args)
    elif args.cmd == "analyze":
        _cmd_analyze(args)
    elif args.cmd == "optimize":
        _cmd_optimize(args)
    elif args.cmd == "visualize":
        _cmd_visualize(args)
    elif args.cmd == "campaign":
        _cmd_campaign(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
