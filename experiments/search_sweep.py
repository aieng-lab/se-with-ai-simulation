#!/usr/bin/env python3
"""Search for the optimal agent team composition using random or evolutionary search.

Unlike run_sweep.py (which fixes an agent allocation recipe), this script searches
over the full space of integer agent compositions A for given (N, M, R) values.
M is not a free parameter — it is derived from the sampled A: M = sum(A) / N.

Usage
-----
    # Random search (default)
    python experiments/search_sweep.py experiments/configs/search_default.json \\
        --out results/search_default [--K 50] [--T 52] [--n_samples 500]

    # Evolutionary search
    python experiments/search_sweep.py experiments/configs/search_default.json \\
        --out results/search_default --strategy evolutionary

Output layout
-------------
    RESULTS_DIR/
        manifest.json          metadata, per-config feasibility, A composition
        sweep_summary.json     Pareto frontier + all objectives including A
        N2_M3_R4_00042/
            result.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from src.config import AGENT_ROLE_COUNT, SweepSettings, TeamConfiguration
from src.feasibility import check_feasibility
from src.io import load_scenario_config
from src.pareto import aggregate_config_result, pareto_frontier
from src.results import save_result
from src.simulation import ConfigResult, run_configuration


# ---------------------------------------------------------------------------
# Composition sampling
# ---------------------------------------------------------------------------

def _sample_composition(total: int, n_roles: int, rng: np.random.Generator) -> tuple[int, ...]:
    """Uniformly sample a non-negative integer composition of `total` into `n_roles` parts.

    Uses the stars-and-bars cut-point method.
    """
    if total == 0:
        return (0,) * n_roles
    cuts = np.sort(rng.integers(0, total + 1, size=n_roles - 1))
    boundaries = np.concatenate([[0], cuts, [total]])
    return tuple(int(x) for x in np.diff(boundaries))


def _sample_composition_with_impl(
    total: int, n_roles: int, rng: np.random.Generator
) -> tuple[int, ...]:
    """Sample a composition guaranteeing a_2 (Feature Implementation, index 1) >= 1."""
    if total < 1:
        raise ValueError(f"total must be >= 1, got {total}")
    # Reserve 1 for impl, distribute the rest freely
    base = _sample_composition(total - 1, n_roles, rng)
    A = list(base)
    A[1] += 1   # role 2 = index 1 (Feature Implementation)
    return tuple(A)


def _sample_qa_composition(
    total: int, n_roles: int, rng: np.random.Generator
) -> tuple[int, ...]:
    """Sample a QA-phase composition: a_1=a_2=a_3=0; distribute total across roles 4–13."""
    # Roles 1-3 (indices 0-2) are forced to zero; distribute total across the remaining 10.
    qa_roles = n_roles - 3
    partial = _sample_composition(total, qa_roles, rng)
    return (0, 0, 0) + partial


# ---------------------------------------------------------------------------
# Mutation operator (for evolutionary search)
# ---------------------------------------------------------------------------

def _mutate(config: TeamConfiguration, rng: np.random.Generator) -> TeamConfiguration:
    """Perturb A by moves, additions, or removals; keep sum(A) <= N*M and a_2 >= 1.
    F_freeze and A_qa are inherited unchanged from the parent."""
    A = list(config.A)
    budget = config.N * config.M
    n_ops = int(rng.integers(1, 4))
    for _ in range(n_ops):
        op = int(rng.integers(3))   # 0=move, 1=add, 2=remove
        current_sum = sum(A)
        if op == 1 and current_sum < budget:
            A[int(rng.integers(AGENT_ROLE_COUNT))] += 1
        elif op == 2 and current_sum > 1:
            donors = [i for i in range(AGENT_ROLE_COUNT)
                      if A[i] > 0 and (i != 1 or A[i] > 1)]
            if donors:
                A[int(rng.choice(donors))] -= 1
        else:
            donors = [i for i in range(AGENT_ROLE_COUNT)
                      if A[i] > 0 and (i != 1 or A[i] > 1)]
            if donors:
                src = int(rng.choice(donors))
                dst = int(rng.integers(AGENT_ROLE_COUNT))
                A[src] -= 1
                A[dst] += 1
    try:
        return TeamConfiguration(N=config.N, M=config.M, A=tuple(A), R=config.R,
                                 F_freeze=config.F_freeze, A_qa=config.A_qa)
    except ValueError:
        return config   # mutation produced invalid config; keep parent


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

def _random_configs(
    search_spec: dict,
    rng: np.random.Generator,
    n_samples: int,
) -> list[TeamConfiguration]:
    """Draw `n_samples` random (N, M, A, R, F_freeze, A_qa) configurations."""
    f_freeze_values: list[int] = search_spec.get("F_freeze_values", [1])
    configs: list[TeamConfiguration] = []
    for _ in range(n_samples):
        N = int(rng.choice(search_spec["N_values"]))
        M = int(rng.integers(1, search_spec["M_max"] + 1))
        R = int(rng.choice(search_spec["R_values"]))
        budget = N * M
        # Sample actual agent totals uniformly in [1, budget] (dev) and [0, budget] (QA)
        B_dev = int(rng.integers(1, budget + 1))
        B_qa  = int(rng.integers(0, budget + 1))
        A = _sample_composition_with_impl(B_dev, AGENT_ROLE_COUNT, rng)
        # Sample F_freeze, clamped so it respects [1, R]
        F_freeze = int(rng.choice([v for v in f_freeze_values if 1 <= v <= R] or [1]))
        A_qa = _sample_qa_composition(B_qa, AGENT_ROLE_COUNT, rng)
        try:
            cfg = TeamConfiguration(N=N, M=M, A=A, R=R, F_freeze=F_freeze, A_qa=A_qa)
            configs.append(cfg)
        except ValueError:
            pass   # should not happen with correct sampling
    return configs


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _config_key(config: TeamConfiguration, idx: int) -> str:
    return f"N{config.N}_M{config.M}_R{config.R}_{idx:05d}"


def _evaluate(
    config: TeamConfiguration,
    key: str,
    settings: SweepSettings,
    params,
    out_dir: Path,
):
    """Run feasibility check + simulation.

    Returns (manifest_entry, agg_or_None).
    agg is None when the config is infeasible.
    """
    from src.pareto import AggregatedObjectives
    report = check_feasibility(config, params)
    entry: dict = {
        "feasible": report.feasible,
        "N": config.N, "M": config.M, "R": config.R,
        "A": list(config.A),
    }
    if report.D_star is not None:
        entry["D_star"] = report.D_star
    if not report.feasible:
        entry["failed_step"] = report.failed_step
        entry["reason"] = report.reason
        return entry, None

    cr = run_configuration(config, settings, params)
    agg = aggregate_config_result(cr, master_seed=settings.master_seed)
    config_dir = out_dir / key
    config_dir.mkdir(exist_ok=True)
    save_result(cr, agg, config_dir / "result.json")
    return entry, agg


# ---------------------------------------------------------------------------
# Search strategies
# ---------------------------------------------------------------------------

def _run_random_search(
    search_spec: dict,
    settings: SweepSettings,
    params,
    out_dir: Path,
    rng: np.random.Generator,
    n_samples: int,
) -> tuple[dict, list, list]:
    """Random search. Returns (manifest_configs, feasible_configs, all_aggs)."""
    configs = _random_configs(search_spec, rng, n_samples)
    manifest: dict[str, dict] = {}
    feasible: list[TeamConfiguration] = []
    aggs = []

    for i, cfg in enumerate(configs):
        key = _config_key(cfg, i)
        entry, agg = _evaluate(cfg, key, settings, params, out_dir)
        entry["generation"] = 0
        manifest[key] = entry
        if agg is not None:
            feasible.append(cfg)
            aggs.append((key, cfg, agg))
            print(f"  [{i+1}/{len(configs)}] {key}:  "
                  f"V={agg.value_mean:.3f}  Q={agg.quality_mean:.4f}  CF={agg.cost_mean:.0f}")
        else:
            print(f"  [{i+1}/{len(configs)}] {key}:  infeasible (step {entry.get('failed_step')})")

    return manifest, feasible, aggs


def _run_evolutionary_search(
    search_spec: dict,
    settings: SweepSettings,
    params,
    out_dir: Path,
    rng: np.random.Generator,
) -> tuple[dict, list, list]:
    """Evolutionary search: random init + guided mutation toward Pareto front."""
    n_initial   = search_spec.get("n_samples_initial", 100)
    pop_size    = search_spec.get("population_size", 30)
    n_gen       = search_spec.get("n_generations", 15)
    n_offspring = search_spec.get("n_offspring", 20)

    manifest: dict[str, dict] = {}
    all_aggs: list[tuple[str, TeamConfiguration, object]] = []
    eval_counter = [0]

    def evaluate_new(cfg: TeamConfiguration, generation: int) -> bool:
        idx = eval_counter[0]
        eval_counter[0] += 1
        key = _config_key(cfg, idx)
        entry, agg = _evaluate(cfg, key, settings, params, out_dir)
        entry["generation"] = generation
        manifest[key] = entry
        if agg is not None:
            all_aggs.append((key, cfg, agg))
            return True
        return False

    # Phase 0: random initialization
    print(f"  Phase 0 — random initialisation ({n_initial} samples) …")
    init_configs = _random_configs(search_spec, rng, n_initial)
    for cfg in init_configs:
        evaluate_new(cfg, generation=0)

    # Evolutionary generations
    seen: set[tuple] = {cfg.A for _, cfg, _ in all_aggs}

    for gen in range(1, n_gen + 1):
        # Select parents from current Pareto frontier
        agg_objs = [a for _, _, a in all_aggs]
        front_idx = pareto_frontier(agg_objs) if agg_objs else []
        parents = [all_aggs[i][1] for i in front_idx] if front_idx else [c for _, c, _ in all_aggs]

        n_new = 0
        attempts = 0
        while n_new < n_offspring and attempts < n_offspring * 10:
            attempts += 1
            parent = parents[int(rng.integers(len(parents)))]
            child = _mutate(parent, rng)
            if child.A in seen:
                continue
            seen.add(child.A)
            if evaluate_new(child, generation=gen):
                n_new += 1

        frontier_size = len(front_idx)
        print(f"  Gen {gen:2d}: Pareto frontier = {frontier_size}, "
              f"new offspring evaluated = {n_new}, total evaluated = {eval_counter[0]}")

    return manifest, [cfg for _, cfg, _ in all_aggs], all_aggs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Search for optimal agent team composition.",
    )
    parser.add_argument("search_config", metavar="SEARCH_CONFIG")
    parser.add_argument("--out", required=True, metavar="DIR")
    parser.add_argument("--K", type=int, default=None, metavar="N")
    parser.add_argument("--T", type=int, default=None, metavar="N")
    parser.add_argument("--n_samples", type=int, default=None, metavar="N",
                        help="Override n_samples from config (random search only)")
    parser.add_argument("--strategy", choices=["random", "evolutionary"], default=None,
                        help="Override strategy from config")
    args = parser.parse_args(argv)

    cfg_path = Path(args.search_config)
    exp = json.loads(cfg_path.read_text(encoding="utf-8"))
    search_spec = exp["search"]
    strategy = args.strategy or search_spec["strategy"]

    sweep_spec = exp["sweep"]
    K = args.K if args.K is not None else sweep_spec["K"]
    T = args.T if args.T is not None else sweep_spec["T"]
    n_samples = args.n_samples if args.n_samples is not None else search_spec.get("n_samples", 500)

    scenario_path = exp["scenario"]
    sweep_settings_from_json, params = load_scenario_config(scenario_path)
    settings = SweepSettings(T=T, K=K, master_seed=sweep_settings_from_json.master_seed)

    print(f"Search:      {cfg_path}")
    print(f"Scenario:    {scenario_path}")
    print(f"Strategy:    {strategy}")
    print(f"Sweep:       K={K}, T={T}")
    if strategy == "random":
        print(f"Samples:     {n_samples}")
    print()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(search_spec.get("seed", 0))

    if strategy == "random":
        manifest, feasible, aggs = _run_random_search(
            search_spec, settings, params, out_dir, rng, n_samples
        )
    else:
        manifest, feasible, aggs = _run_evolutionary_search(
            search_spec, settings, params, out_dir, rng
        )

    # Pareto frontier
    agg_objs = [a for _, _, a in aggs]
    frontier_indices = pareto_frontier(agg_objs) if agg_objs else []
    frontier_keys = [aggs[i][0] for i in frontier_indices]

    # Persist manifest
    manifest_out = {
        "search_config": str(cfg_path),
        "scenario": scenario_path,
        "strategy": strategy,
        "sweep": {"K": K, "T": T, "master_seed": settings.master_seed},
        "n_evaluated": len(manifest),
        "n_feasible": len(aggs),
        "configs": manifest,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest_out, indent=2), encoding="utf-8"
    )

    # Persist sweep summary (includes A for direct inspection)
    sweep_summary = {
        "pareto_frontier": frontier_keys,
        "objectives": {
            key: {
                "N": cfg.N, "M": cfg.M, "R": cfg.R,
                "A": list(cfg.A),
                "value_mean": agg.value_mean, "quality_mean": agg.quality_mean, "cost_mean": agg.cost_mean,
                "value_ci": list(agg.value_ci), "quality_ci": list(agg.quality_ci), "cost_ci": list(agg.cost_ci),
            }
            for key, cfg, agg in aggs
        },
    }
    (out_dir / "sweep_summary.json").write_text(
        json.dumps(sweep_summary, indent=2), encoding="utf-8"
    )

    print()
    print(f"Pareto frontier ({len(frontier_keys)} config(s)):")
    for key in frontier_keys:
        obj = sweep_summary["objectives"][key]
        print(f"  {key}  N={obj['N']} M={obj['M']} R={obj['R']}  "
              f"value={obj['value_mean']:.3f}  quality={obj['quality_mean']:.4f}  CF={obj['cost_mean']:.0f}")
        print(f"         A={obj['A']}")
    print()
    print(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
