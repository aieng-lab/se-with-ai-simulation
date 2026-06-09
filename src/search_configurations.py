"""Search for the optimal agent team composition using random or evolutionary search.

Unlike a fixed-recipe sweep, this module searches over the full space of integer agent
compositions A for given (N, M, R) values.

Output layout
-------------
    RESULTS_DIR/
        manifest.json          metadata, per-config feasibility, A composition
        sweep_summary.json     Pareto frontier + all objectives including A
        samples/
            N2_M3_R4_00042.json
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import AGENT_ROLE_COUNT, SearchConfiguration, SimulationHorizon, TeamConfiguration
from .feasibility import check_feasibility
from .pareto import aggregate_config_result, pareto_frontier
from .results import save_result
from .simulation import run_configuration
from .state import SystemState


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
    """Sample a composition guaranteeing a_1 (Feature Implementation, index 0) >= 1."""
    if total < 1:
        raise ValueError(f"total must be >= 1, got {total}")
    # Reserve 1 for impl, distribute the rest freely
    base = _sample_composition(total - 1, n_roles, rng)
    A_dev = list(base)
    A_dev[0] += 1   # role 1 = index 0 (Feature Implementation Agent)
    return tuple(A_dev)


def _sample_qa_composition(
    total: int, n_roles: int, rng: np.random.Generator
) -> tuple[int, ...]:
    """Sample a QA-phase composition: a_1=a_2=0; distribute total across roles 3–8."""
    # Roles 1-2 (indices 0-1) are forced to zero (no new features or architecture proposals
    # during feature freeze); Bug Fixing and other roles (indices 2-7) may be non-zero.
    qa_roles = n_roles - 2
    partial = _sample_composition(total, qa_roles, rng)
    return (0, 0) + partial


# ---------------------------------------------------------------------------
# Mutation operator (for evolutionary search)
# ---------------------------------------------------------------------------

def _mutate(
    config: TeamConfiguration,
    rng: np.random.Generator,
    search_cfg: SearchConfiguration,
) -> TeamConfiguration:
    """Perturb exactly one randomly chosen parameter from {N, M, F_freeze, A_dev, A_qa}.
    R is never mutated. Retries up to 10 times if the result is unchanged or invalid;
    returns the parent if all attempts fail."""
    for _ in range(10):
        mutation_type = int(rng.integers(5))  # 0=N, 1=M, 2=F_freeze, 3=A_dev, 4=A_qa

        N = config.N
        M = config.M
        A_dev = list(config.A_dev)
        A_qa = list(config.A_qa)
        F_freeze = config.F_freeze
        budget = N * M

        if mutation_type == 0:
            # Mutate N: pick an adjacent value in the sorted N_values list
            n_vals = sorted(search_cfg.N_values)
            idx = n_vals.index(N) if N in n_vals else 0
            candidates = []
            if idx > 0:
                candidates.append(n_vals[idx - 1])
            if idx < len(n_vals) - 1:
                candidates.append(n_vals[idx + 1])
            if candidates:
                N = int(rng.choice(candidates))

        elif mutation_type == 1:
            # Mutate M: increase by 1 and add one agent to A so the extra capacity is used.
            # Decreasing M is skipped — it cannot change behavior, only tighten the budget.
            if M < search_cfg.M_max:
                M += 1
                A_dev[int(rng.integers(AGENT_ROLE_COUNT))] += 1

        elif mutation_type == 2:
            # Mutate F_freeze: ±1 clamped to [0, R-1]
            F_freeze = int(np.clip(F_freeze + int(rng.choice([-1, 1])), 0, config.R - 1))

        elif mutation_type == 3:
            # Mutate A_dev: 1-3 add/remove/move ops, maintaining sum(A_dev) <= N*M and A_dev[0] >= 1
            op = int(rng.integers(3))   # 0=move, 1=add, 2=remove
            current_sum = sum(A_dev)
            if op == 1 and current_sum < budget:
                A_dev[int(rng.integers(AGENT_ROLE_COUNT))] += 1
            elif op == 2 and current_sum > 1:
                donors = [i for i in range(AGENT_ROLE_COUNT)
                            if A_dev[i] > 0 and (i != 1 or A_dev[i] > 1)]
                if donors:
                    A_dev[int(rng.choice(donors))] -= 1
            else:
                donors = [i for i in range(AGENT_ROLE_COUNT)
                            if A_dev[i] > 0 and (i != 1 or A_dev[i] > 1)]
                if donors:
                    src = int(rng.choice(donors))
                    dst = int(rng.integers(AGENT_ROLE_COUNT))
                    A_dev[src] -= 1
                    A_dev[dst] += 1

        else:
            # Mutate A_qa: 1-2 ops restricted to QA roles (indices 2-7; 0-1 stay at 0)
            _QA_MUTABLE = list(range(2, AGENT_ROLE_COUNT))
            op = int(rng.integers(3))   # 0=move, 1=add, 2=remove
            current_sum = sum(A_qa)
            if op == 1 and current_sum < budget:
                A_qa[_QA_MUTABLE[int(rng.integers(len(_QA_MUTABLE)))]] += 1
            elif op == 2 and current_sum > 0:
                donors = [i for i in _QA_MUTABLE if A_qa[i] > 0]
                if donors:
                    A_qa[int(rng.choice(donors))] -= 1
            else:
                donors = [i for i in _QA_MUTABLE if A_qa[i] > 0]
                if donors:
                    src = int(rng.choice(donors))
                    dst = _QA_MUTABLE[int(rng.integers(len(_QA_MUTABLE)))]
                    A_qa[src] -= 1
                    A_qa[dst] += 1

        try:
            child = TeamConfiguration(N=N, M=M, A_dev=tuple(A_dev), R=config.R,
                                      F_freeze=F_freeze, A_qa=tuple(A_qa))
        except ValueError:
            continue   # invalid; retry
        if child != config:
            return child

    return config   # all attempts were unchanged or invalid


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

def _random_configs(
    search_cfg: SearchConfiguration,
    rng: np.random.Generator,
    n_samples: int,
) -> list[TeamConfiguration]:
    """Draw `n_samples` random (N, M, A_dev, R, F_freeze, A_qa) configurations."""
    configs: list[TeamConfiguration] = []
    for _ in range(n_samples):
        N = int(rng.choice(search_cfg.N_values))
        M = int(rng.integers(1, search_cfg.M_max + 1))
        R = int(rng.choice(search_cfg.R_values))
        budget = N * M
        # Sample actual agent totals uniformly in [1, budget] (dev) and [0, budget] (QA)
        B_dev = int(rng.integers(1, budget + 1))
        B_qa  = int(rng.integers(0, budget + 1))
        A_dev = _sample_composition_with_impl(B_dev, AGENT_ROLE_COUNT, rng)
        # Sample F_freeze, clamped so it respects [0, R-1]
        F_freeze = int(rng.choice([v for v in search_cfg.F_freeze_values if 0 <= v <= R - 1] or [1]))
        A_qa = _sample_qa_composition(B_qa, AGENT_ROLE_COUNT, rng)
        try:
            cfg = TeamConfiguration(N=N, M=M, A_dev=A_dev, R=R, F_freeze=F_freeze, A_qa=A_qa)
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
    settings: SimulationHorizon,
    params,
    out_dir: Path,
    initial_state: SystemState | None = None,
):
    """Run feasibility check + simulation.

    Returns (manifest_entry, agg_or_None).
    agg is None when the config is infeasible.
    """
    from .pareto import AggregatedObjectives
    report = check_feasibility(config, params)
    entry: dict = {
        "feasible": report.feasible,
        "N": config.N, "M": config.M, "R": config.R, "F_freeze": config.F_freeze,
        "A_dev": list(config.A_dev),
        "A_qa": list(config.A_qa),
    }
    if not report.feasible:
        entry["failed_step"] = report.failed_step
        entry["reason"] = report.reason
        return entry, None

    cr = run_configuration(config, settings, params, initial_state=initial_state)
    agg = aggregate_config_result(cr, master_seed=settings.master_seed)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(exist_ok=True, parents=True)
    save_result(cr, agg, samples_dir / f"{key}.json")
    return entry, agg


# ---------------------------------------------------------------------------
# Search strategies
# ---------------------------------------------------------------------------

def _run_random_search(
    search_cfg: SearchConfiguration,
    settings: SimulationHorizon,
    params,
    out_dir: Path,
    rng: np.random.Generator,
    n_samples: int,
    initial_state: SystemState | None = None,
) -> tuple[dict, list, list]:
    """Random search. Returns (manifest_configs, feasible_configs, all_aggs)."""
    configs = _random_configs(search_cfg, rng, n_samples)
    manifest: dict[str, dict] = {}
    feasible: list[TeamConfiguration] = []
    aggs = []

    for i, cfg in enumerate(configs):
        key = _config_key(cfg, i)
        entry, agg = _evaluate(cfg, key, settings, params, out_dir, initial_state)
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
    search_cfg: SearchConfiguration,
    settings: SimulationHorizon,
    params,
    out_dir: Path,
    rng: np.random.Generator,
    initial_state: SystemState | None = None,
) -> tuple[dict, list, list]:
    """Evolutionary search: random init + guided mutation toward Pareto front."""
    n_initial   = search_cfg.n_samples_initial
    n_gen       = search_cfg.n_generations
    n_offspring = search_cfg.n_offspring

    manifest: dict[str, dict] = {}
    all_aggs: list[tuple[str, TeamConfiguration, object]] = []
    eval_counter = [0]

    def evaluate_new(cfg: TeamConfiguration, generation: int) -> bool:
        idx = eval_counter[0]
        eval_counter[0] += 1
        key = _config_key(cfg, idx)
        entry, agg = _evaluate(cfg, key, settings, params, out_dir, initial_state)
        entry["generation"] = generation
        manifest[key] = entry
        if agg is not None:
            all_aggs.append((key, cfg, agg))
            return True
        return False

    print(f"Random initialisation ({n_initial} samples) …")
    init_configs = _random_configs(search_cfg, rng, n_initial)
    for cfg in init_configs:
        evaluate_new(cfg, generation=0)

    # Evolutionary generations
    seen: set[TeamConfiguration] = {cfg for _, cfg, _ in all_aggs}

    for gen in range(1, n_gen + 1):
        # Build Pareto and non-Pareto parent pools (80% / 20% sampling)
        agg_objs = [a for _, _, a in all_aggs]
        front_idx = pareto_frontier(agg_objs) if agg_objs else []
        front_set = set(front_idx)
        pareto_parents = [all_aggs[i][1] for i in front_idx]
        non_pareto_parents = [all_aggs[i][1] for i in range(len(all_aggs)) if i not in front_set]
        # Fallback: if one pool is empty, use the other exclusively
        if not pareto_parents:
            pareto_parents = non_pareto_parents

        n_non_pareto_slots = round(n_offspring * 0.2) if non_pareto_parents else 0
        n_pareto_slots = n_offspring - n_non_pareto_slots

        n_new = 0
        for pool, slots in [(non_pareto_parents, n_non_pareto_slots),
                            (pareto_parents, n_pareto_slots)]:
            if not pool or slots == 0:
                continue
            count = 0
            attempts = 0
            while count < slots and attempts < slots * 10:
                attempts += 1
                parent = pool[int(rng.integers(len(pool)))]
                child = _mutate(parent, rng, search_cfg)
                if child in seen:
                    continue
                seen.add(child)
                if evaluate_new(child, generation=gen):
                    count += 1
            n_new += count

        frontier_size = len(front_idx)
        print(f"  Gen {gen:2d}: Pareto frontier = {frontier_size}, "
              f"new offspring evaluated = {n_new}, total evaluated = {eval_counter[0]}")

    return manifest, [cfg for _, cfg, _ in all_aggs], all_aggs

