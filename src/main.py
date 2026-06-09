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

Search for optimal team compositions:
    python -m src.main optimize config/search_default.json --out results/search_default
    python -m src.main optimize config/search_default.json --out results/search_evo --strategy evolutionary

Generate visualizations from optimization results:
    python -m src.main visualize results/search_default [--out results/search_default/plots]
    python -m src.main visualize results/run1 results/run2 --out results/comparison
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .analysis import final_distribution, mean_trajectory
from .config import AGENT_ROLE_NAMES, TeamConfiguration
from .io import load_initial_state, load_scenario_params, load_search_config, load_simulation_horizon, load_team_config
from .pareto import aggregate_config_result, pareto_frontier
from .results import load_result, load_traces, save_result, save_traces
from .search_configurations import _run_evolutionary_search, _run_random_search
from .simulation import ConfigResult, ConfigTrace, run_configuration
from .visualization import visualize_multi, visualize_single


def _print_config(config: TeamConfiguration) -> None:
    print(f"Team configuration:  N={config.N}, M={config.M}, R={config.R}, "
          f"F_freeze={config.F_freeze}")
    print(f"  Dev phase  (total agents: {sum(config.A_dev)})")
    for k, name in enumerate(AGENT_ROLE_NAMES, start=1):
        count = config.agent_count_dev(k)
        if count:
            print(f"    role {k:>2}  {name:<48}  x{count}")
    print(f"  QA phase   (total agents: {sum(config.A_qa)})")
    for k, name in enumerate(AGENT_ROLE_NAMES, start=1):
        count = config.agent_count_qa(k)
        if count:
            print(f"    role {k:>2}  {name:<48}  x{count}")


def _cmd_run(args: argparse.Namespace) -> None:
    print("agent-ai-cost-sim — Monte Carlo simulation")
    print("=" * 52)

    horizon = load_simulation_horizon(args.horizon)
    config = load_team_config(args.team)
    priors = load_scenario_params(args.scenario)
    _print_config(config)
    print()

    initial_state = load_initial_state(args.start_state)

    print(f"Scenario:   {args.scenario}")
    print(f"Team:       {args.team}")
    print(f"Start state:{args.start_state}")
    print(f"Horizon:   K={horizon.K} replications * T={horizon.T} sprints  "
          f"(seed={horizon.master_seed})")
    print(f"Traces:     {'yes' if args.traces else 'no'}")
    print()
    print("Running …")

    trace: ConfigTrace | None = None

    if args.traces:
        trace = run_configuration(config, horizon, priors, capture_traces=True,
                                  initial_state=initial_state)
        cr = ConfigResult(config=config, replications=trace.replications)
    else:
        cr = run_configuration(config, horizon, priors, initial_state=initial_state)

    agg = aggregate_config_result(cr, master_seed=horizon.master_seed)

    print()
    print("Objectives (mean [95 % CI]):")
    print(f"  value   features/sprint       {agg.value_mean:.3f}   [{agg.value_ci[0]:.3f}, {agg.value_ci[1]:.3f}]")
    print(f"  quality weighted-bugs/feature {agg.quality_mean:.4f}   [{agg.quality_ci[0]:.4f}, {agg.quality_ci[1]:.4f}]")
    print(f"  CF  cost/feature      {agg.cost_mean:.1f}   [{agg.cost_ci[0]:.1f}, {agg.cost_ci[1]:.1f}]")

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
    print("Objectives (mean [95 % CI]):")
    print(f"  value   features/sprint       {agg.value_mean:.3f}   [{agg.value_ci[0]:.3f}, {agg.value_ci[1]:.3f}]")
    print(f"  quality weighted-bugs/feature {agg.quality_mean:.4f}   [{agg.quality_ci[0]:.4f}, {agg.quality_ci[1]:.4f}]")
    print(f"  CF  cost/feature      {agg.cost_mean:.1f}   [{agg.cost_ci[0]:.1f}, {agg.cost_ci[1]:.1f}]")

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

    horizon = load_simulation_horizon(search_cfg.horizon_path)
    scenario_values = json.loads(Path(search_cfg.scenario_path).read_text(encoding="utf-8"))
    params = load_scenario_params(search_cfg.scenario_path)
    initial_state = load_initial_state(args.start_state)

    print(f"Search:      {cfg_path}")
    print(f"Scenario:    {search_cfg.scenario_path}")
    print(f"Start state: {args.start_state}")
    print(f"Strategy:    {strategy}")
    print(f"Horizon:     K={horizon.K} replications * T={horizon.T} sprints  "
          f"(seed={horizon.master_seed})")
    if strategy == "random":
        print(f"Samples:     {search_cfg.n_samples}")
    print()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(search_cfg.seed)

    if strategy == "random":
        manifest, feasible, aggs = _run_random_search(
            search_cfg, horizon, params, out_dir, rng, search_cfg.n_samples, initial_state
        )
    else:
        manifest, feasible, aggs = _run_evolutionary_search(
            search_cfg, horizon, params, out_dir, rng, initial_state
        )

    agg_objs = [a for _, _, a in aggs]
    frontier_indices = pareto_frontier(agg_objs) if agg_objs else []
    frontier_keys = [aggs[i][0] for i in frontier_indices]

    manifest_out = {
        "search_config": str(cfg_path),
        "scenario": {"path": search_cfg.scenario_path, "values": scenario_values},
        "strategy": strategy,
        "horizon": {"path": search_cfg.horizon_path, "K": horizon.K, "T": horizon.T, "master_seed": horizon.master_seed},
        "n_evaluated": len(manifest),
        "n_feasible": len(aggs),
        "configs": manifest,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest_out, indent=2), encoding="utf-8"
    )

    search_summary = {
        "pareto_frontier": frontier_keys,
        "objectives": {
            key: {
                "N": cfg.N, "M": cfg.M, "R": cfg.R,
                "A_dev": list(cfg.A_dev),
                "value_mean": agg.value_mean, "quality_mean": agg.quality_mean, "cost_mean": agg.cost_mean,
                "value_ci": list(agg.value_ci), "quality_ci": list(agg.quality_ci), "cost_ci": list(agg.cost_ci),
            }
            for key, cfg, agg in aggs
        },
    }
    (out_dir / "search_summary.json").write_text(
        json.dumps(search_summary, indent=2), encoding="utf-8"
    )

    print()
    print(f"Pareto frontier ({len(frontier_keys)} config(s)):")
    for key in frontier_keys:
        obj = search_summary["objectives"][key]
        print(f"  {key}  N={obj['N']} M={obj['M']} R={obj['R']}  "
              f"value={obj['value_mean']:.3f}  quality={obj['quality_mean']:.4f}  CF={obj['cost_mean']:.0f}")
        print(f"         A_dev={obj['A_dev']}")
    print()
    print(f"Results saved to: {out_dir}")


def _cmd_visualize(args: argparse.Namespace) -> None:
    result_dirs = [Path(d) for d in args.result_dirs]
    for d in result_dirs:
        if not d.is_dir():
            print(f"Error: {d} is not a directory.", file=sys.stderr)
            sys.exit(1)

    fmt = args.format
    mean_pareto = args.mean_pareto
    multi = len(result_dirs) > 1

    if multi:
        for rd in result_dirs:
            visualize_single(rd, rd / "plots", fmt, mean_pareto=mean_pareto)
        cmp_out = Path(args.out) if args.out else Path("plots")
        visualize_multi(result_dirs, cmp_out, fmt, mean_pareto=mean_pareto)
    else:
        out = Path(args.out) if args.out else result_dirs[0] / "plots"
        visualize_single(result_dirs[0], out, fmt, mean_pareto=mean_pareto)

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
    p_run.add_argument(
        "--scenario", default="config/scenario_default.json", metavar="PATH",
        help="Scenario parameters JSON file (default: config/scenario_default.json)",
    )
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
        help="JSON file with initial SystemState values (default: all-zero)",
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
    p_optimize.add_argument("--strategy", choices=["random", "evolutionary"], default=None,
                         help="Override strategy from config")
    p_optimize.add_argument(
        "--start-state", default="config/startstate_default.json", metavar="PATH",
        help="JSON file with initial SystemState values (default: all-zero)",
    )

    # -- visualize --
    p_vis = sub.add_parser("visualize", help="Generate plots from sweep results.")
    p_vis.add_argument("result_dirs", nargs="+", metavar="RESULT_DIR",
                       help="One or more result directories produced by 'optimize'")
    p_vis.add_argument("--format", default="png", choices=["png", "pdf", "svg"],
                       help="Output image format (default: png)")
    p_vis.add_argument("--out", default=None, metavar="DIR",
                       help="Output directory for plots (default: RESULT_DIR/plots/)")
    p_vis.add_argument("--mean-pareto", action="store_true",
                       help="Use strict mean-value Pareto dominance instead of CI-aware frontier")

    args = parser.parse_args(argv)

    # Default: run smoke test if no subcommand given
    if args.cmd is None:
        args.cmd = "run"
        args.scenario = "config/scenario_default.json"
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
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
