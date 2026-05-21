"""CLI entry point for agent-ai-cost-sim.

Usage
-----
Run a simulation (smoke-test defaults):
    python -m src.main run

Run with custom settings and save results:
    python -m src.main run --K 100 --T 52 --out results/

Run and also save per-sprint traces:
    python -m src.main run --K 50 --T 52 --out results/ --traces

Load and analyse persisted results:
    python -m src.main analyze results/result.json
    python -m src.main analyze results/result.json results/traces.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .analysis import final_distribution, mean_trajectory, quantile_band
from .config import AGENT_ROLE_COUNT, AGENT_ROLE_NAMES, SweepSettings, TeamConfiguration
from .io import load_scenario_config
from .pareto import aggregate_config_result, pareto_frontier
from .results import load_result, load_traces, save_result, save_traces
from .simulation import ConfigResult, ConfigTrace, run_configuration


def _example_configuration() -> TeamConfiguration:
    """Example 2-human, span-3 team: 6 agents split across the most-used roles."""
    composition = [0] * AGENT_ROLE_COUNT
    composition[2] = 2   # Feature Implementation
    composition[3] = 1   # Bug Fixing
    composition[5] = 1   # Code Review
    composition[6] = 1   # Security Audit
    composition[7] = 1   # Unit Test Development
    return TeamConfiguration(N=2, M=3, A=tuple(composition), R=4)


def _print_config(config: TeamConfiguration) -> None:
    print(f"Team configuration:  N={config.N}, M={config.M}, R={config.R}  "
          f"(total agents: {sum(config.A)})")
    for k, name in enumerate(AGENT_ROLE_NAMES, start=1):
        count = config.agent_count(k)
        if count:
            print(f"  role {k:>2}  {name:<48}  x{count}")


def _cmd_run(args: argparse.Namespace) -> None:
    config = _example_configuration()

    print("agent-ai-cost-sim — Monte Carlo simulation")
    print("=" * 52)
    _print_config(config)
    print()

    sweep, priors = load_scenario_config(args.scenario)
    K = args.K if args.K is not None else sweep.K
    T = args.T if args.T is not None else sweep.T
    settings = SweepSettings(T=T, K=K, master_seed=sweep.master_seed)

    print(f"Scenario:   {args.scenario}")
    print(f"Settings:   K={settings.K} replications × T={settings.T} sprints  "
          f"(seed={settings.master_seed})")
    print(f"Traces:     {'yes' if args.traces else 'no'}")
    print()
    print("Running …")

    trace: ConfigTrace | None = None

    if args.traces:
        trace = run_configuration(config, settings, priors, capture_traces=True)
        cr = ConfigResult(config=config, replications=trace.replications)
    else:
        cr = run_configuration(config, settings, priors)

    agg = aggregate_config_result(cr, master_seed=settings.master_seed)
    frontier = pareto_frontier([agg])

    print()
    print("Objectives (mean [95 % CI]):")
    print(f"  V   features/sprint   {agg.V_mean:.3f}   [{agg.V_ci[0]:.3f}, {agg.V_ci[1]:.3f}]")
    print(f"  Q   bugs/feature      {agg.Q_mean:.4f}   [{agg.Q_ci[0]:.4f}, {agg.Q_ci[1]:.4f}]")
    print(f"  CF  cost/feature      {agg.CF_mean:.1f}   [{agg.CF_ci[0]:.1f}, {agg.CF_ci[1]:.1f}]")
    print()
    print(f"Pareto:  {'on frontier' if frontier else 'off frontier'}")

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
    print(f"  V   features/sprint   {agg.V_mean:.3f}   [{agg.V_ci[0]:.3f}, {agg.V_ci[1]:.3f}]")
    print(f"  Q   bugs/feature      {agg.Q_mean:.4f}   [{agg.Q_ci[0]:.4f}, {agg.Q_ci[1]:.4f}]")
    print(f"  CF  cost/feature      {agg.CF_mean:.1f}   [{agg.CF_ci[0]:.1f}, {agg.CF_ci[1]:.1f}]")

    if not args.traces:
        return

    df = load_traces(args.traces)
    T = df.index.get_level_values("sprint").max() + 1
    print()
    print(f"Trace file:  {args.traces}  ({len(result.replications)} reps × {T} sprints)")
    print()

    # Key trajectory fields to display
    fields = ["F_completed", "D", "B_kno_min", "B_kno_maj", "doc_cov", "impl"]
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
        "--scenario", default="config/default_scenario.json", metavar="PATH",
        help="Scenario JSON file (default: config/default_scenario.json)",
    )
    p_run.add_argument("--K", type=int, metavar="N",
                       help="Override K (replications) from scenario JSON")
    p_run.add_argument("--T", type=int, metavar="N",
                       help="Override T (sprints) from scenario JSON")
    p_run.add_argument("--out", metavar="DIR",
                       help="Directory to write result.json [and traces.csv]")
    p_run.add_argument("--traces", action="store_true",
                       help="Capture and save full per-sprint time-series traces")

    # -- analyze --
    p_ana = sub.add_parser("analyze", help="Analyse persisted results.")
    p_ana.add_argument("result", metavar="result.json",
                       help="Path to result.json written by 'run --out'")
    p_ana.add_argument("traces", nargs="?", metavar="traces.csv",
                       help="Path to traces.csv written by 'run --out --traces' (optional)")

    args = parser.parse_args(argv)

    # Default: run smoke test if no subcommand given
    if args.cmd is None:
        args.cmd = "run"
        args.scenario = "config/default_scenario.json"
        args.K = 10
        args.T = 8
        args.out = None
        args.traces = False

    if args.cmd == "run":
        _cmd_run(args)
    elif args.cmd == "analyze":
        _cmd_analyze(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
