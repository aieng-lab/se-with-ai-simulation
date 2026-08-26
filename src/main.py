"""CLI entry point for agent-ai-cost-sim.

Usage
-----
Search for the cheapest team that keeps up with the project:
    python -m src.main optimize config/search_default.json --out results/search_default
    python -m src.main optimize config/search/search_h-l_p-m_d-h_a-l_b-m.json \
        --samples none --progress-every 500 --out results/one_scenario

Enumerate the zero-agent baseline grid instead of searching (the baseline configs
carry `"human_only": true`, which is what selects the enumerator):
    python -m src.main optimize config/search_baseline/search_h-h_p-l_d-X_a-X_b-m.json \
        --out results/baseline_one

Run the whole campaign:
    ./run_experiments.sh          (see its --help; generate the configs first with
                                   scripts/generate_search_configs.py)

Consolidate a finished campaign into its tables and the five-channel optima figures:
    python -m src.main campaign [results/experiments] [--out results/analysis]
    python -m src.main campaign --figures-only    (redraw from an existing optima.csv)
    python -m src.main campaign --no-figures      (rewrite the CSVs only)

Describe what those optima are made of — the cell tables, the role/process shares and the
three team-structure figures:
    python -m src.main constellation [results/experiments] [--out results/analysis]
    python -m src.main constellation --figures-only

The figure layouts are fixed, not flags: `visualization.MATRIX_CHANNELS` for the three
optima matrices and `constellation.CELL_GROUPS` for the cell grid.  Section 5 reads every
figure off one layout, so there is nothing to choose.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    HUMAN_ONLY_MODE,
    RESULT_MODES,
    SAMPLE_POLICIES,
    SEARCH_MODE,
    SearchConfiguration,
    SimulationHorizon,
    TeamConfiguration,
)
from .io import (
    load_initial_state,
    load_scenario_params,
    load_search_config,
    load_simulation_horizon,
)
from .parameters import ScenarioParameters
from .search import (
    SearchOutcome,
    SelectionReport,
    human_only_grid,
    run_human_only,
    run_search,
    two_stage_selection,
    write_samples,
)
from .selection import (
    AggregatedObjectives,
    satisfies_throughput,
    throughput_floor,
)
from .campaign import (baseline_uplift, build_tables, convergence_summary,
                       cost_multipliers, multipliers_summary)
from .constellation import (build_constellation,
                            build_indistinguishable, build_process_summary,
                            constellation_summary, process_summary, role_bill)
from .visualization import visualize_campaign, visualize_team_summary


def _objective_entry(config: TeamConfiguration, agg: AggregatedObjectives,
                     floor: float) -> dict:
    """One `search_summary.json` objectives entry: the configuration and its aggregates.

    Used for both blocks of that file — `objectives` at `K_search` for every simulated
    candidate and `final_objectives` at `K_final` for the re-simulated ones — so the two
    have identical shape.
    """
    return {
        "N": config.N, "R": config.R, "F_freeze": config.F_freeze,
        "A_dev": list(config.A_dev),
        "A_qa": list(config.A_qa),
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


def _write_json(path: Path, payload: dict) -> None:
    """Write one of the three JSON artefacts a search leaves behind."""
    path.write_text(json.dumps(payload), encoding="utf-8")


def _print_search_banner(cfg_path: Path, search_cfg: SearchConfiguration,
                         horizon: SimulationHorizon, params: ScenarioParameters,
                         mode: str, start_state: str) -> None:
    """What is about to be run: inputs, protocol, budget and throughput floor."""
    print(f"Search:      {cfg_path}")
    print(f"Scenario id: {search_cfg.scenario_id or '(unnamed)'}")
    print("Scenario:")
    for group, path in search_cfg.scenario_paths.items():
        print(f"  {group:<10} {path}")
    print(f"Start state: {start_state}")
    print(f"Mode:        {mode}")
    print(f"Horizon:     K'={horizon.K_search} replications during the search, "
          f"K={horizon.K_final} for the re-simulation of I* * T={horizon.T} sprints  "
          f"(seed={horizon.master_seed})")
    if mode == HUMAN_ONLY_MODE:
        n_grid = len(human_only_grid(search_cfg))
        print(f"Budget:      none; exhaustive enumeration of the {n_grid}-point "
              f"zero-agent grid")
    else:
        print(f"Budget:      {search_cfg.budget} candidate evaluations")
        print(f"             {search_cfg.n_samples_initial} initial + "
              f"{search_cfg.n_generations} generations * {search_cfg.n_offspring} offspring, "
              f"elite {search_cfg.n_elite}")
    print(f"Throughput:  value >= (1 - {search_cfg.epsilon}) * {params.r_arrival:g} = "
          f"{throughput_floor(params, search_cfg.epsilon):.3f} features/sprint")
    print(f"Samples:     {search_cfg.samples}")
    print()


def _manifest_payload(cfg_path: Path, search_cfg: SearchConfiguration,
                      horizon: SimulationHorizon, mode: str, scenario_values: dict,
                      outcome: SearchOutcome, floor: float, best_admissible: bool,
                      wall_clock_s: float) -> dict:
    """`manifest.json`: what was run, and one entry per evaluated candidate."""
    return {
        "search_config": str(cfg_path),
        "scenario_id": search_cfg.scenario_id,
        "scenario": {"paths": search_cfg.scenario_paths, "values": scenario_values},
        "mode": mode,
        "horizon": {"path": search_cfg.horizon_path, "K_search": horizon.K_search,
                    "K_final": horizon.K_final, "T": horizon.T,
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
            "human_only": mode == HUMAN_ONLY_MODE,
            "samples": search_cfg.samples,
            "seed": search_cfg.seed,
        },
        # n_evaluated is the budget spend; n_simulated excludes the candidates the
        # human-capacity check rejected.
        "n_evaluated": outcome.n_evaluated,
        "n_simulated": outcome.n_simulated,
        "throughput_floor": floor,
        "best_admissible": best_admissible,
        "wall_clock_s": round(wall_clock_s, 1),
        "configs": outcome.manifest,
    }


def _search_summary_payload(outcome: SearchOutcome, report: SelectionReport,
                            horizon: SimulationHorizon, floor: float) -> dict:
    """`search_summary.json`: the reported pair, and the objectives behind it.

    `objectives` holds every simulated candidate at `K_search`, which is what the search
    ranked by; `final_objectives` holds the re-simulated ones at `K_final`, which is what
    `best` and `indistinguishable` key into and what every reported number comes from.
    """
    configs = outcome.configs_by_key
    return {
        "best": report.best_key,
        "indistinguishable": report.keep_keys,
        "throughput_floor": floor,
        "K_search": horizon.K_search,
        "K_final": horizon.K_final,
        "objectives": {
            key: _objective_entry(cfg, agg, floor) for key, cfg, agg in outcome.aggs
        },
        "final_objectives": {
            key: _objective_entry(configs[key], agg, floor)
            for key, agg in report.final_aggs.items()
        },
    }


def _convergence_payload(search_cfg: SearchConfiguration, horizon: SimulationHorizon,
                         mode: str, outcome: SearchOutcome, report: SelectionReport,
                         params: ScenarioParameters, final_best: dict | None,
                         wall_clock_s: float, resim_s: float) -> dict:
    """`convergence.json`: the per-generation record and the stages of the selection.

    Written separately from the manifest so that consolidating a campaign does not have
    to parse the manifest's per-candidate block for a handful of scalars.
    """
    return {
        "scenario_id": search_cfg.scenario_id,
        "mode": mode,
        "n_elite": search_cfg.n_elite,
        "n_generations_run": len(outcome.elite_turnover),
        "K_search": horizon.K_search,
        "K_final": horizon.K_final,
        "elite_turnover": outcome.elite_turnover,
        "incumbent_changed": [bool(b) for b in outcome.incumbent_changed],
        "incumbent_turnover": outcome.incumbent_turnover,
        "best_cost": outcome.best_cost,
        "i_star": report.stats,
        "best_admissible": report.best_admissible,
        "value_ratio": (final_best["value_mean"] / params.r_arrival
                        if final_best else None),
        "wall_clock_s": round(wall_clock_s, 1),
        "resimulation_s": round(resim_s, 1),
    }


def _print_optimum(report: SelectionReport, horizon: SimulationHorizon, floor: float,
                   final_best: dict | None) -> None:
    """The reported optimum and how the selection narrowed down to it."""
    if final_best is None:
        print("No candidate was simulated — nothing to report.")
        return

    obj = final_best
    if not report.best_admissible:
        print(f"WARNING: no candidate reached the throughput floor of "
              f"{floor:.3f} features/sprint; reporting the least-violating one.")
    print(f"Best configuration c* ({report.best_key}):")
    print(f"  N={obj['N']:g} R={obj['R']} F_freeze={obj['F_freeze']}  "
          f"cost={obj['cost_mean']:.0f} $/feature "
          f"[{obj['cost_ci'][0]:.0f}, {obj['cost_ci'][1]:.0f}]  "
          f"value={obj['value_mean']:.3f}")
    print(f"  cost split: escaped bugs {obj['cost_quality_mean']:.0f}   "
          f"developers {obj['cost_human_mean']:.0f}   "
          f"agents {obj['cost_agent_mean']:.0f}")
    print(f"  A_dev={obj['A_dev']}")
    print(f"  A_qa ={obj['A_qa']}")

    if not report.best_admissible:
        # With nothing admissible there is no c* for a set to form around, so I* is
        # empty rather than a set of one; the line below would read as a tie.
        print("Indistinguishable set I*: empty (the result is inadmissible)")
        return

    st = report.stats
    print(f"Statistically indistinguishable from c*: {len(report.keep_keys)} config(s)")
    print(f"  I*: {st['n_prelim']} at K'={horizon.K_search} → "
          f"{st['n_resimulated']} after the parsimony tie-break → "
          f"{st['n_after_resim']} after the re-simulation at K={horizon.K_final} "
          f"({st['n_discarded_throughput']} discarded on throughput) → "
          f"{st['n_final']} after the second tie-break")
    if st["best_changed_by_resim"]:
        print("  the re-simulation moved c* to a different configuration")


def _cmd_optimize(args: argparse.Namespace) -> None:
    cfg_path = Path(args.search_config)
    search_cfg = load_search_config(cfg_path)
    # The zero-agent enumerator is selected by the search config alone; the baseline
    # configs that scripts/generate_search_configs.py --baseline writes carry the flag.
    human_only = search_cfg.human_only
    mode = HUMAN_ONLY_MODE if human_only else SEARCH_MODE
    if args.samples is not None:
        search_cfg = replace(search_cfg, samples=args.samples)

    horizon = load_simulation_horizon(search_cfg.horizon_path)
    scenario_values = {
        group: json.loads(Path(path).read_text(encoding="utf-8"))
        for group, path in search_cfg.scenario_paths.items()
    }
    params = load_scenario_params(search_cfg.scenario_paths)
    initial_state = load_initial_state(args.start_state)

    _print_search_banner(cfg_path, search_cfg, horizon, params, mode, args.start_state)

    out_dir = Path(args.out)
    # `campaign.search_dirs` reads the result mode back off this directory's name, so a
    # run whose directory says one thing and whose config does another would be
    # mislabelled for the whole of the analysis — a search filed as a human-only
    # baseline, or the reverse.  Nothing downstream can detect that, so refuse here.
    if out_dir.name in RESULT_MODES and out_dir.name != mode:
        print(f"Error: {cfg_path} runs in {mode!r} mode, but --out names a "
              f"{out_dir.name!r} directory ({out_dir}).  The result mode is read back "
              f"from that name, so this would file the run under the wrong mode.",
              file=sys.stderr)
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    if human_only:
        outcome = run_human_only(search_cfg, horizon, params, initial_state,
                                 progress_every=args.progress_every)
    else:
        outcome = run_search(search_cfg, horizon, params,
                             np.random.default_rng(search_cfg.seed), initial_state,
                             progress_every=args.progress_every)
    wall_clock_s = time.perf_counter() - t_start

    floor = throughput_floor(params, search_cfg.epsilon)
    t_select = time.perf_counter()
    report = two_stage_selection(
        outcome, search_cfg, horizon, params, initial_state,
        progress_every=args.progress_every,
    )
    resim_s = time.perf_counter() - t_select

    # Samples are written only now, so the "best" policy can restrict them to the
    # configurations actually reported.  I* is empty when nothing reached the throughput
    # floor, so fall back to c*, which is still reported, just marked inadmissible.
    if search_cfg.samples == "none":
        sample_keys: list[str] = []
    elif search_cfg.samples == "best":
        sample_keys = report.keep_keys or (
            [report.best_key] if report.best_key is not None else [])
    else:  # unreachable: SearchConfiguration validates against SAMPLE_POLICIES
        raise ValueError(f"unknown sample policy {search_cfg.samples!r}")
    # The re-simulated results replace the search's for the keys they cover, so a
    # reported candidate's sample file carries the K_final replications behind its row.
    outcome.results.update(report.final_results)
    write_samples(outcome, out_dir, sample_keys, report.final_aggs)

    final_best = (
        _objective_entry(outcome.configs_by_key[report.best_key],
                         report.final_aggs[report.best_key], floor)
        if report.best_key is not None else None
    )

    _write_json(out_dir / "manifest.json",
                _manifest_payload(cfg_path, search_cfg, horizon, mode, scenario_values,
                                  outcome, floor, report.best_admissible, wall_clock_s))
    _write_json(out_dir / "search_summary.json",
                _search_summary_payload(outcome, report, horizon, floor))
    _write_json(out_dir / "convergence.json",
                _convergence_payload(search_cfg, horizon, mode, outcome, report, params,
                                     final_best, wall_clock_s, resim_s))

    print()
    print(f"Evaluated {outcome.n_evaluated} candidates "
          f"({outcome.n_simulated} simulated, "
          f"{outcome.n_evaluated - outcome.n_simulated} infeasible) "
          f"in {wall_clock_s:.1f} s")
    print(f"Re-simulated {report.stats['n_resimulated']} configuration(s) at "
          f"K={horizon.K_final} in {resim_s:.1f} s")
    _print_optimum(report, horizon, floor, final_best)
    print()
    print(f"Results saved to: {out_dir}")


def _read_table(path: Path, label: str, hint: str) -> "pd.DataFrame":
    """Load a consolidated CSV, or exit with the command that would produce it."""
    if not path.is_file():
        print(f"Error: {path} does not exist; {hint}", file=sys.stderr)
        sys.exit(1)
    frame = pd.read_csv(path)
    print(f"Loaded {len(frame)} {label} from {path}")
    return frame


def _cmd_campaign(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    optima_path = out_dir / "optima.csv"

    if args.figures_only:
        hint = "run the campaign without --figures-only first."
        optima = _read_table(optima_path, "reported optima", hint)
        if optima.empty:
            print("\nNo optimum was reported; skipping the figures.")
            return
        # An all-baseline campaign writes no turnover CSV, because an enumeration has
        # no generations; the convergence figure is simply skipped, as it is on the
        # path that rebuilds the tables.
        turnover_path = out_dir / "elite_turnover.csv"
        turnover = (_read_table(turnover_path, "turnover point(s)", hint)
                    if turnover_path.is_file() else pd.DataFrame())
        visualize_campaign(optima, out_dir, turnover=turnover)
        print("\nDone.")
        return

    try:
        optima, scenarios, convergence, turnover = build_tables(Path(args.results_dir))
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios_path = out_dir / "scenarios.csv"
    optima.to_csv(optima_path, index=False)
    scenarios.to_csv(scenarios_path, index=False)
    convergence_path = out_dir / "convergence.csv"
    turnover_path = out_dir / "elite_turnover.csv"
    if not convergence.empty:
        convergence.to_csv(convergence_path, index=False)
    if not turnover.empty:
        turnover.to_csv(turnover_path, index=False)

    uplift = baseline_uplift(optima)
    uplift_path = out_dir / "baseline_uplift.csv"
    if not uplift.empty:
        uplift.to_csv(uplift_path, index=False)

    multipliers, multiplier_stats = cost_multipliers(optima)
    multipliers_path = out_dir / "multipliers.csv"
    if not multipliers.empty:
        multipliers.to_csv(multipliers_path, index=False)

    # One `scenarios` row per finished run, so the two counts come off that table.
    n_searches = len(scenarios)
    print(f"Collected {n_searches} search(es) over "
          f"{scenarios['scenario_id'].nunique()} scenario(s)")
    print(f"  {optima_path}   {len(optima)} reported optima")
    print(f"  {scenarios_path}   {n_searches} (scenario, mode) pairs")
    if not uplift.empty:
        print(f"  {uplift_path}   {len(uplift)} optima with a human-only reference")
    if not convergence.empty:
        print(f"  {convergence_path}   {len(convergence)} search(es) with convergence measures")
    if not turnover.empty:
        print(f"  {turnover_path}   {len(turnover)} generation(s) of elite turnover")
    if not multipliers.empty:
        print(f"  {multipliers_path}   {len(multipliers)} (group, tier) cost multipliers")
    if len(optima) < n_searches:
        print(f"  note: {n_searches - len(optima)} search(es) simulated no candidate")

    if not convergence.empty:
        summary = convergence_summary(convergence)
        (out_dir / "convergence_summary.txt").write_text(summary, encoding="utf-8")
        print()
        print(summary, end="")

    if not multipliers.empty:
        summary = multipliers_summary(multipliers, multiplier_stats)
        (out_dir / "multipliers_summary.txt").write_text(summary, encoding="utf-8")
        print()
        print(summary, end="")

    if args.no_figures:
        print("\nDone.")
        return
    if optima.empty:
        print("\nNo optimum was reported; skipping the figures.")
        return

    visualize_campaign(optima, out_dir, turnover=turnover)
    print("\nDone.")


def _cmd_constellation(args: argparse.Namespace) -> None:
    """What the optimal teams are made of, cut along two scenario groups.

    A second pass over the same tree, reading each ``search_summary.json``'s final `I*`
    rather than just its `c*`, so it can be re-cut on a different grid without forcing a
    re-consolidation.
    """
    out_dir = Path(args.out)

    optima_path = out_dir / "optima.csv"
    optima_hint = "run `python -m src.main campaign` first."

    if args.figures_only:
        optima = _read_table(optima_path, "reported optima", optima_hint)
        visualize_team_summary(role_bill(optima), out_dir)
        print("\nDone.")
        return

    optima = _read_table(optima_path, "reported optima", optima_hint)
    cells_path = out_dir / "constellation_cells.csv"
    results_dir = Path(args.results_dir)
    print(f"Reading the indistinguishable sets under {results_dir} ...")
    try:
        istar = build_indistinguishable(results_dir)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"  {len(istar)} I* member(s) over "
          f"{istar[['scenario_id', 'mode']].drop_duplicates().shape[0]} search(es)")

    cells = build_constellation(optima, istar)
    if cells.empty:
        print("\nNo searched optimum to describe; nothing written.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    istar_path = out_dir / "indistinguishable.csv"
    istar.to_csv(istar_path, index=False)
    cells.to_csv(cells_path, index=False)
    print(f"  {istar_path}   {len(istar)} I* members")
    print(f"  {cells_path}   {len(cells)} (population, cell, parameter) rows")

    summary = constellation_summary(cells)
    summary_path = out_dir / "constellation_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")
    print(f"  {summary_path}")

    # What the optima staff and how they set the process — the composition statistics
    # that are quoted as shares rather than read off the cell grid.
    bills = role_bill(optima)
    usage = build_process_summary(bills, istar)
    if not usage.empty:
        usage_path = out_dir / "process_summary.csv"
        usage.to_csv(usage_path, index=False)
        usage_text = process_summary(usage)
        (out_dir / "process_summary.txt").write_text(usage_text, encoding="utf-8")
        print(f"  {usage_path}   {len(usage)} role/process/I* statistics")
        print()
        print(usage_text, end="")

    visualize_team_summary(bills, out_dir)
    print("\nDone.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.main",
        description="agent-ai-cost-sim: Monte Carlo simulation of AI+human dev teams.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(metavar="COMMAND")

    # -- optimize --
    p_optimize = sub.add_parser("optimize", help="Search for optimal team compositions.")
    p_optimize.add_argument("search_config", metavar="SEARCH_CONFIG",
                         help="Path to search config JSON (e.g. config/search_default.json)")
    p_optimize.add_argument("--out", required=True, metavar="DIR",
                         help="Output directory for optimization results")
    p_optimize.add_argument(
        "--samples", choices=list(SAMPLE_POLICIES), default=None,
        help="Which candidates get a samples/<key>.json file: 'best' "
             "(c* and the set indistinguishable from it) or 'none' "
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
    p_optimize.set_defaults(func=_cmd_optimize)

    # -- campaign --
    p_camp = sub.add_parser(
        "campaign",
        help="Consolidate a whole campaign into its CSV tables and the "
             "five-channel optima figures.",
    )
    p_camp.add_argument("results_dir", nargs="?", default="results/experiments",
                        metavar="RESULTS_DIR",
                        help="Campaign output root (default: results/experiments)")
    p_camp.add_argument("--out", default="results/analysis", metavar="DIR",
                        help="Where to write the CSVs and figures "
                             "(default: results/analysis)")
    p_camp.add_argument("--figures-only", action="store_true",
                        help="Redraw the figures from an existing OUT/optima.csv instead "
                             "of re-reading the campaign and rewriting the CSVs")
    p_camp.add_argument("--no-figures", action="store_true",
                        help="Write the CSVs and the convergence summary but draw no "
                             "figures")
    p_camp.set_defaults(func=_cmd_campaign)

    # -- constellation --
    p_const = sub.add_parser(
        "constellation",
        help="Describe what the optimal teams are made of, per cell of a two-group "
             "tier grid: mean and sd of every control variable, for c* and for I*.",
    )
    p_const.add_argument("results_dir", nargs="?", default="results/experiments",
                         metavar="RESULTS_DIR",
                         help="Campaign output root (default: results/experiments)")
    p_const.add_argument("--out", default="results/analysis", metavar="DIR",
                         help="Where OUT/optima.csv is read and the tables and figures "
                              "are written (default: results/analysis)")
    p_const.add_argument("--figures-only", action="store_true",
                         help="Redraw from an existing OUT/optima.csv instead of "
                              "re-walking the result tree for I*")
    p_const.set_defaults(func=_cmd_constellation)

    if argv is None:
        argv = sys.argv[1:]
    args = parser.parse_args(argv)
    # `add_subparsers` is not required, so that a bare invocation prints the usage
    # rather than a bare traceback.
    if not hasattr(args, "func"):
        parser.print_help(sys.stderr)
        sys.exit(2)
    args.func(args)


if __name__ == "__main__":
    main()
