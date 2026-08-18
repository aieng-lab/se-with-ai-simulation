"""Search for the cheapest team composition that keeps up with the project (paper §3.11).

`_run_evolutionary_search` solves the constrained single-objective problem of
Eq. (objective): it maintains a population and mutates parents drawn from an elite ranked
by the preference relation of `src.selection`.  It is the paper's only search strategy.
Uniform random sampling (`_random_draws`) is not a strategy of its own — it is the
initialiser the search seeds from (§3.11.2).

`_run_human_only` sits beside it but is not a second strategy: the human-only baseline of
§4.1 fixes both composition vectors at zero, leaving 99 configurations that are enumerated
exhaustively.  It is the only producer of zero-agent teams — the samplers and `_mutate`
keep their `a_1 >= 1` floor, so the search can never reach one.

Output layout
-------------
    RESULTS_DIR/
        manifest.json          metadata, per-config feasibility, A composition
        search_summary.json    c*, the indistinguishable set I*, and the objectives of
                               every simulated candidate including A
        samples/
            N2_R4_00042.json   per-replication objectives; which candidates get a
                               file is governed by SearchConfiguration.samples
                               ("all" / "best" / "none")

Supervised role counts are bounded per role by the throughput saturation bound a-bar_k
of `saturation_bounds` (Eq. sat_bound) — beyond a-bar_k an agent has no work to do —
and collectively by the human-capacity condition of Eq. (feas_monitoring), which
`src.feasibility` checks after the draw.  The three autonomous roles are outside both:
the two detection services are capped by nothing at all (only sampled within a range),
and the Refactoring Agent is capped by `refactoring_bound` alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import (
    AGENT_ROLE_COUNT,
    AUTONOMOUS_ROLE_INDICES,
    SUPERVISED_ROLE_INDICES,
    SearchConfiguration,
    SimulationHorizon,
    TeamConfiguration,
)
from .feasibility import check_feasibility
from .results import save_result
from .selection import (
    AggregatedObjectives,
    aggregate_config_result,
    rank,
    throughput_floor,
)
from .simulation import ConfigResult, run_configuration
from .state import SystemState


# ---------------------------------------------------------------------------
# Composition sampling
# ---------------------------------------------------------------------------

#: The 0-based index of the Refactoring Agent (k=4), the one autonomous role that carries
#: an upper bound.  Named because it is special-cased in the sampler and in the mutation
#: operator alike, while the two detection services next to it are capped by nothing.
_REFACTOR_INDEX: int = 3

_QA_SUPERVISED_INDICES: tuple[int, ...] = tuple(
    i for i in SUPERVISED_ROLE_INDICES if i >= 2
)   # QA phase pins roles 1-2 (indices 0-1) to zero


def saturation_bounds(params, sigma: float) -> dict[int, int]:
    """Per-role upper bounds a-bar_k of paper §3.11.1, Eq. (sat_bound), by 0-based index.

    Every *supervised* role is fed by work that ultimately derives from the feature
    arrival rate, so deploying more agents than the inflow can keep busy buys
    nothing.  `lambda_k` is that steady-state inflow (Eq. workload) and `eta_k` the
    throughput one agent of the role sustains (Eq. sat_rates); the debt amplifier
    enters at its floor `delta_min`, the least favourable and therefore most
    permissive case.  `sigma >= 1` is the headroom factor that still lets the search
    over-allocate against the Poisson peaks of the arrival process.

    The three autonomous roles of K_aut = {4,5,6} are deliberately absent: Eq. (sat_bound)
    ranges over K_sup only.  The Refactoring Agent's cap comes from `refactoring_bound`
    instead, and the two detection services have none.

    Every workload is a positive multiple of r_arrival, so the ceiling is at least 1
    for every role and a_dev,1 >= 1 stays reachable without an explicit floor.
    """
    if sigma < 1:
        raise ValueError(f"sigma must be >= 1, got {sigma}")

    r = params.r_arrival
    # lambda_3 counts only the agent-injected bug stream of Eq. (Binj-agent), matching the
    # paper's Eq. (workload).  Since Eq. (Binj-human) became an absolute rate the human
    # stream is no longer proportional to r_bugs and is therefore not represented here;
    # that under-counts the inflow, but harmlessly — eta_3 = r_fix * delta_min is in the
    # hundreds, so a-bar_3 = 1 either way, even at r_arrival = 1000.
    lambda_3 = params.r_bugs * (2.0 - params.delta_min) * r
    workload = {
        0: (1.0 + params.f_arch_chg) * r,                       # k=1 Feature Implementation
        1: params.f_arch_chg * r,                               # k=2 Architecture & Design
        2: lambda_3,                                            # k=3 Bug Fixing
        6: params.c_test_ceil * r,                              # k=7 Test Development
        7: r,                                                   # k=8 Documentation
    }
    rates = {
        0: params.r_F * params.delta_min,
        1: params.r_arch_chg,
        2: params.r_fix * params.delta_min,
        6: params.r_test,
        7: params.r_doc,
    }
    bounds: dict[int, int] = {}
    for idx in SUPERVISED_ROLE_INDICES:
        rate = rates[idx]
        if rate <= 0:
            raise ValueError(
                f"saturation rate eta for role index {idx} must be > 0, got {rate}"
            )
        bounds[idx] = math.ceil(sigma * workload[idx] / rate)
    return bounds


def review_sample_max(params) -> int:
    """Sampling range of the two detection services: features arriving per sprint.

    Paper §3.11.2, step 5: a_.,5 and a_.,6 are drawn uniformly from {0, ..., floor(
    r_arrival)}, i.e. at most one service instance per feature that arrives in a sprint.
    The model imposes no cap on these roles at all — this is a sampling choice that
    scales the candidate counts with the review volume the scenario generates, and the
    mutation operator may take them beyond it.
    """
    return int(params.r_arrival)


def refactoring_bound(params) -> int:
    """a-bar_4 of paper §3.11.2, step 6: floor(r_arrival / 2).

    Unlike `review_sample_max` this is a genuine cap, honoured by the sampler *and* by
    the mutation operator (§3.11.4): "the refactoring service is restricted by a-bar_4".
    It allows up to one Refactoring Agent per two features that arrive in a sprint, on the
    grounds that the debt to remove is generated by features and by the bug fixes those
    features cause.  It is not scaled by `sigma` — the paper defines it outside
    Eq. (sat_bound) — and it is 0 for r_arrival < 2, which pins the role to zero there.
    """
    return int(params.r_arrival // 2)


def _sample_count(rng: np.random.Generator, upper: int) -> int:
    """Draw one autonomous-role count uniformly from {0, ..., upper}."""
    return int(rng.integers(0, upper + 1))


def _assemble_composition(
    supervised: dict[int, int],
    rng: np.random.Generator,
    review_max: int,
    refactor_max: int,
) -> tuple[int, ...]:
    """Build a full 8-role vector from supervised counts plus fresh autonomous draws."""
    A = [0] * AGENT_ROLE_COUNT
    for idx, count in supervised.items():
        A[idx] = count
    for idx in AUTONOMOUS_ROLE_INDICES:
        upper = refactor_max if idx == _REFACTOR_INDEX else review_max
        A[idx] = _sample_count(rng, upper)
    return tuple(A)


def _sample_supervised(
    rng: np.random.Generator,
    bounds: dict[int, int],
    indices: tuple[int, ...],
    min_impl: bool,
) -> dict[int, int]:
    """Draw each supervised role independently and uniformly within its bound u_k.

    Paper §3.11.2, steps 3-4: a_k ~ U{0, ..., a-bar_k}, except that the dev-phase
    Feature Implementation role (index 0) is drawn from U{1, ..., a-bar_1} so that
    a_dev,1 >= 1 holds by construction.  The human-capacity condition is *not*
    enforced here — a draw that violates it is rejected by the feasibility gate.
    """
    supervised: dict[int, int] = {}
    for idx in indices:
        lo = 1 if (min_impl and idx == 0) else 0
        supervised[idx] = int(rng.integers(lo, bounds[idx] + 1))
    return supervised


def _sample_dev_composition(
    rng: np.random.Generator, bounds: dict[int, int],
    review_max: int, refactor_max: int,
) -> tuple[int, ...]:
    """Sample a dev-phase composition: every supervised role within a-bar_k, a_1 >= 1."""
    supervised = _sample_supervised(rng, bounds, SUPERVISED_ROLE_INDICES, min_impl=True)
    return _assemble_composition(supervised, rng, review_max, refactor_max)


def _sample_qa_composition(
    rng: np.random.Generator, bounds: dict[int, int],
    review_max: int, refactor_max: int,
) -> tuple[int, ...]:
    """Sample a QA-phase composition: a_1=a_2=0, the rest within their bound a-bar_k."""
    # Roles 1-2 (indices 0-1) are forced to zero (no new features or architecture
    # proposals during feature freeze).
    supervised = _sample_supervised(rng, bounds, _QA_SUPERVISED_INDICES, min_impl=False)
    return _assemble_composition(supervised, rng, review_max, refactor_max)


# ---------------------------------------------------------------------------
# Mutation operator (for evolutionary search)
# ---------------------------------------------------------------------------

def _pick(indices: tuple[int, ...], rng: np.random.Generator) -> int:
    """Uniformly pick one role index from `indices`."""
    return indices[int(rng.integers(len(indices)))]


def _mutate_composition(
    A: list[int],
    mutable: tuple[int, ...],
    bounds: dict[int, int],
    rng: np.random.Generator,
    min_impl: bool,
    refactor_max: int,
) -> None:
    """Apply one elementary mutation to `A` in place.

    `mutable` are the supervised role indices that may be changed; each is capped by
    its throughput saturation bound `bounds[i]` (paper §3.11.1).  The three autonomous
    roles of K_aut = {4,5,6} are mutated by a separate +/-1 "adjust service" operation
    instead, because they need no supervision and are not part of the pool the other
    three operations redistribute (§3.11.4).  The two detection services have no upper
    bound there; the Refactoring Agent is capped by `refactor_max` = a-bar_4, and a +1
    that would exceed it is left as a no-op for `_mutate` to retry.
    With `min_impl`, index 0 (Feature Implementation) is protected from being drained
    to zero.
    """
    op = int(rng.integers(4))   # 0=move, 1=add, 2=remove, 3=adjust service +/-1
    if op == 3:
        idx = _pick(AUTONOMOUS_ROLE_INDICES, rng)
        delta = int(rng.choice([-1, 1]))
        target = A[idx] + delta
        if idx == _REFACTOR_INDEX and target > refactor_max:
            return          # a-bar_4 reached; leave the parent unchanged
        A[idx] = max(0, target)
        return

    floor = 1 if min_impl else 0
    donors = [i for i in mutable if A[i] > 0 and (i != 0 or A[i] > floor)]
    # Only roles still below their saturation bound can receive an agent.
    recipients = tuple(i for i in mutable if A[i] < bounds[i])
    if op == 1 and recipients:
        A[_pick(recipients, rng)] += 1
    elif op == 2 and donors:
        A[int(rng.choice(donors))] -= 1
    elif op == 0 and donors and recipients:
        src = int(rng.choice(donors))
        A[src] -= 1
        # Re-check the bound, excluding the donor: moving an agent to the role it came
        # from is a no-op, and the donor is always below its bound after the decrement.
        recipients = tuple(i for i in mutable if i != src and A[i] < bounds[i])
        if recipients:
            A[_pick(recipients, rng)] += 1
        else:
            A[src] += 1   # nothing else can receive; undo and let the caller retry


def _mutate(
    config: TeamConfiguration,
    rng: np.random.Generator,
    search_cfg: SearchConfiguration,
    bounds: dict[int, int],
    refactor_max: int,
) -> TeamConfiguration:
    """Perturb exactly one randomly chosen parameter from {N, F_freeze, A_dev, A_qa}.
    R is never mutated. Retries up to 10 times if the result is unchanged or invalid;
    returns the parent if all attempts fail.

    When the parent has F_freeze == 0 the A_qa option is dropped and the choice is over
    the remaining three.  A_qa is canonically zero there (see TeamConfiguration), so
    mutating it would produce a child that is canonicalised straight back to its parent —
    burning a retry every time.  Before that canonicalisation existed this was the single
    largest source of behavioural duplicates: the mutation produced a child that differed
    from its parent in name only and was then scored on a different RNG stream.
    """
    n_mutation_types = 3 if config.F_freeze == 0 else 4
    for _ in range(10):
        mutation_type = int(rng.integers(n_mutation_types))  # 0=N, 1=F_freeze, 2=A_dev, 3=A_qa

        N = config.N
        A_dev = list(config.A_dev)
        A_qa = list(config.A_qa)
        F_freeze = config.F_freeze

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
                N = float(rng.choice(candidates))

        elif mutation_type == 1:
            # Mutate F_freeze: ±1 clamped to [0, R-1].  Going 0 -> 1 opens a freeze window
            # onto the canonical zero A_qa, i.e. a window staffed by nobody.  That is a
            # valid configuration (a_qa,1 = a_qa,2 = 0 holds trivially) that later A_qa
            # mutations can grow; re-sampling A_qa here would perturb two parameters at
            # once and break this operator's one-parameter contract.
            F_freeze = int(np.clip(F_freeze + int(rng.choice([-1, 1])), 0, config.R - 1))

        elif mutation_type == 2:
            # Mutate A_dev: exactly one elementary op, maintaining a_k <= u_k and
            # A_dev[0] >= 1
            _mutate_composition(A_dev, SUPERVISED_ROLE_INDICES, bounds,
                                rng, min_impl=True, refactor_max=refactor_max)

        else:
            # Mutate A_qa: exactly one elementary op restricted to the supervised QA
            # roles (indices 0-1 stay at 0) plus the autonomous services
            _mutate_composition(A_qa, _QA_SUPERVISED_INDICES, bounds,
                                rng, min_impl=False, refactor_max=refactor_max)

        try:
            child = TeamConfiguration(N=N, A_dev=tuple(A_dev), R=config.R,
                                      F_freeze=F_freeze, A_qa=tuple(A_qa))
        except ValueError:
            continue   # invalid; retry
        if child != config:
            return child

    return config   # all attempts were unchanged or invalid


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

def _random_draws(
    search_cfg: SearchConfiguration,
    rng: np.random.Generator,
    n_samples: int,
    params,
) -> list[TeamConfiguration]:
    """Draw `n_samples` random (N, A_dev, R, F_freeze, A_qa) candidates.

    Every draw satisfies the structural constraints of `TeamConfiguration` by
    construction, so the samplers never produce a configuration that has to be
    discarded here.  The only rejection path left is the human-capacity condition of
    Eq. (feas_monitoring), applied by the feasibility gate in `_evaluate`
    (paper §3.11.2, step 6).
    """
    review_max = review_sample_max(params)
    refactor_max = refactoring_bound(params)
    bounds = saturation_bounds(params, search_cfg.sigma)
    draws: list[TeamConfiguration] = []
    for _ in range(n_samples):
        N = float(rng.choice(search_cfg.N_values))
        R = int(rng.choice(search_cfg.R_values))
        # Sample F_freeze before the compositions, because it decides whether A_qa is a
        # real parameter at all.  Restricted to the admissible values for this R; the list
        # is never empty, since SearchConfiguration requires 0 in F_freeze_values.
        F_freeze = int(rng.choice([v for v in search_cfg.F_freeze_values if 0 <= v <= R - 1]))
        # Each supervised role is drawn independently within its saturation bound; the
        # three autonomous roles are drawn separately inside the samplers, the detection
        # services within a sampling range that bounds nothing and the Refactoring Agent
        # within a-bar_4 (§3.11.2, steps 5-6).
        A_dev = _sample_dev_composition(rng, bounds, review_max, refactor_max)
        # With no freeze window the QA composition is never applied, so drawing one would
        # only attach meaningless coordinates to the candidate (paper §3.11.2, step 4).
        A_qa = (_sample_qa_composition(rng, bounds, review_max, refactor_max)
                if F_freeze > 0 else (0,) * AGENT_ROLE_COUNT)
        draws.append(TeamConfiguration(N=N, A_dev=A_dev, R=R,
                                       F_freeze=F_freeze, A_qa=A_qa))
    return draws


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _config_key(config: TeamConfiguration, idx: int) -> str:
    """Key of a candidate, used as its manifest entry and its `samples/` filename.

    `:g` keeps a whole-FTE staffing level as ``N2`` and a fractional one as ``N0.25``.
    """
    return f"N{config.N:g}_R{config.R}_{idx:05d}"


def _evaluate(
    config: TeamConfiguration,
    settings: SimulationHorizon,
    params,
    initial_state: SystemState | None = None,
):
    """Run feasibility check + simulation.

    Returns (manifest_entry, agg_or_None, config_result_or_None); the latter two are
    None when the configuration is infeasible and therefore never simulated.
    Persistence is left to the caller, which knows the sample retention policy.
    """
    report = check_feasibility(config, params)
    entry: dict = {
        "feasible": report.feasible,
        "N": config.N, "R": config.R, "F_freeze": config.F_freeze,
        "A_dev": list(config.A_dev),
        "A_qa": list(config.A_qa),
    }
    if not report.feasible:
        entry["failed_step"] = report.failed_step
        entry["reason"] = report.reason
        return entry, None, None

    cr = run_configuration(config, settings, params, initial_state=initial_state)
    agg = aggregate_config_result(cr, master_seed=settings.master_seed)
    return entry, agg, cr


@dataclass(slots=True)
class SearchOutcome:
    """Everything a search or enumeration produces, independent of how it is persisted.

    `results` holds the per-candidate ConfigResults kept for possible sample export;
    it is empty when the retention policy is "none" and holds every simulated
    candidate otherwise (~2.5 KB per candidate, so ~25 MB at the §4.2 budget).
    """

    manifest: dict[str, dict]
    aggs: list[tuple[str, TeamConfiguration, "AggregatedObjectives"]]
    results: dict[str, ConfigResult]

    @property
    def n_evaluated(self) -> int:
        """Candidates that consumed budget, including infeasible ones (paper §4.2)."""
        return len(self.manifest)

    @property
    def n_simulated(self) -> int:
        """Candidates actually simulated, i.e. those that passed the feasibility check."""
        return len(self.aggs)


def write_samples(outcome: SearchOutcome, out_dir: Path, keys: list[str]) -> int:
    """Persist `samples/<key>.json` for the requested keys; returns the file count."""
    if not keys:
        return 0
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(exist_ok=True, parents=True)
    aggs_by_key = {key: agg for key, _, agg in outcome.aggs}
    written = 0
    for key in keys:
        cr = outcome.results.get(key)
        if cr is None:
            continue
        save_result(cr, aggs_by_key[key], samples_dir / f"{key}.json")
        written += 1
    return written


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def _progress(i: int, total: int, outcome: SearchOutcome, every: int) -> None:
    """Print a compact heartbeat every `every` candidates (`every == 0` silences it)."""
    if every and (i % every == 0 or i == total):
        n_infeasible = outcome.n_evaluated - outcome.n_simulated
        print(f"  [{i}/{total}] simulated={outcome.n_simulated}  infeasible={n_infeasible}",
              flush=True)



def _run_evolutionary_search(
    search_cfg: SearchConfiguration,
    settings: SimulationHorizon,
    params,
    rng: np.random.Generator,
    initial_state: SystemState | None = None,
    progress_every: int = 100,
) -> SearchOutcome:
    """Evolutionary search: random init + guided mutation toward the optimum (§3.11.4).

    Budget accounting follows paper §4.2: every candidate that reaches the
    feasibility check consumes one evaluation, whether or not it is simulated.  A
    generation can therefore fall short of `n_offspring` only when the mutation
    operator keeps producing duplicates until its attempt limit is exhausted.  The loop
    also stops early once `budget` evaluations have been spent, so the nominal
    `n_samples_initial + n_generations * n_offspring` can never overrun it.
    """
    n_initial   = search_cfg.n_samples_initial
    n_gen       = search_cfg.n_generations
    n_offspring = search_cfg.n_offspring
    n_elite     = search_cfg.n_elite
    budget      = search_cfg.budget
    keep_samples = search_cfg.samples != "none"
    bounds = saturation_bounds(params, search_cfg.sigma)
    refactor_max = refactoring_bound(params)
    floor = throughput_floor(params, search_cfg.epsilon)

    outcome = SearchOutcome(manifest={}, aggs=[], results={})
    # Every candidate that has consumed budget, feasible or not, so a duplicate is
    # never charged twice.
    seen: set[TeamConfiguration] = set()

    def evaluate_new(cfg: TeamConfiguration, generation: int) -> None:
        key = _config_key(cfg, outcome.n_evaluated)
        entry, agg, cr = _evaluate(cfg, settings, params, initial_state)
        entry["generation"] = generation
        outcome.manifest[key] = entry
        seen.add(cfg)
        if agg is not None:
            outcome.aggs.append((key, cfg, agg))
            if keep_samples:
                outcome.results[key] = cr    # type: ignore[assignment]

    print(f"Random initialisation ({n_initial} samples) …", flush=True)
    init_draws = _random_draws(search_cfg, rng, min(n_initial, budget), params)
    for i, cfg in enumerate(init_draws):
        evaluate_new(cfg, generation=0)
        _progress(i + 1, len(init_draws), outcome, progress_every)

    for gen in range(1, n_gen + 1):
        if outcome.n_evaluated >= budget:
            break
        # Rank the population by the preference relation of §3.11.3 and split it into the
        # elite that most offspring come from and the remainder that supplies exploration.
        # A single objective makes this a sort, so there is no frontier to compute.
        order = rank([a for _, _, a in outcome.aggs], floor)
        elite_parents = [outcome.aggs[i][1] for i in order[:n_elite]]
        rest_parents  = [outcome.aggs[i][1] for i in order[n_elite:]]
        # Fallback: if one pool is empty, use the other exclusively
        if not elite_parents:
            elite_parents = rest_parents

        n_rest_slots = round(n_offspring * 0.2) if rest_parents else 0
        n_elite_slots = n_offspring - n_rest_slots

        n_new = 0
        for pool, slots in [(rest_parents, n_rest_slots),
                            (elite_parents, n_elite_slots)]:
            if not pool or slots == 0:
                continue
            count = 0
            attempts = 0
            while count < slots and attempts < slots * 10:
                if outcome.n_evaluated >= budget:
                    break
                attempts += 1
                parent = pool[int(rng.integers(len(pool)))]
                child = _mutate(parent, rng, search_cfg, bounds, refactor_max)
                if child in seen:
                    continue
                evaluate_new(child, generation=gen)
                count += 1        # infeasible children consume budget too (§4.2)
            n_new += count

        if progress_every:
            best = outcome.aggs[order[0]][2] if order else None
            best_txt = f"{best.cost_mean:.0f}" if best is not None else "n/a"
            print(f"  Gen {gen:3d}: best cost = {best_txt}, elite = {len(elite_parents)}, "
                  f"offspring evaluated = {n_new}, total evaluated = {outcome.n_evaluated}, "
                  f"simulated = {outcome.n_simulated}", flush=True)

    return outcome


def human_only_grid(search_cfg: SearchConfiguration) -> list[TeamConfiguration]:
    """Every zero-agent team in the search range, in a deterministic order.

    The human-only baseline of paper §4.1 fixes both composition vectors at zero, so only
    `(N, R, F_freeze)` remain free and the configuration space collapses to
    `|N_values| * |{(R, F) : F in F_freeze_values, 0 <= F <= R-1}|` — 99 points at the
    §4.2 range (11 staffing levels x 9 valid `(R, F_freeze)` pairs).
    """
    zero = (0,) * AGENT_ROLE_COUNT
    return [
        TeamConfiguration(N=N, A_dev=zero, R=R, F_freeze=F, A_qa=zero)
        for N in search_cfg.N_values
        for R in search_cfg.R_values
        for F in search_cfg.F_freeze_values
        if 0 <= F <= R - 1
    ]


def _run_human_only(
    search_cfg: SearchConfiguration,
    settings: SimulationHorizon,
    params,
    rng: np.random.Generator,
    initial_state: SystemState | None = None,
    progress_every: int = 100,
) -> SearchOutcome:
    """Exhaustive enumeration of the human-only baseline grid (paper §4.1).

    Not a search: at 99 candidates there is nothing to search and no budget to spend, so
    `budget` and the evolutionary settings are all ignored and the result is deterministic
    — `rng` is accepted only to match the signature of `_run_evolutionary_search`, and is
    unused.

    `saturation_bounds` is likewise never called: it bounds supervised roles that the
    baseline does not staff.  Every candidate still passes through `_evaluate`, so an
    infeasible one counts as an evaluation exactly as in the search.
    """
    keep_samples = search_cfg.samples != "none"
    outcome = SearchOutcome(manifest={}, aggs=[], results={})
    grid = human_only_grid(search_cfg)

    for cfg in grid:
        key = _config_key(cfg, outcome.n_evaluated)
        entry, agg, cr = _evaluate(cfg, settings, params, initial_state)
        outcome.manifest[key] = entry
        if agg is not None:
            outcome.aggs.append((key, cfg, agg))
            if keep_samples:
                outcome.results[key] = cr    # type: ignore[assignment]
        _progress(outcome.n_evaluated, len(grid), outcome, progress_every)

    if progress_every:
        print(f"  Enumerated {outcome.n_evaluated} zero-agent configuration(s), "
              f"simulated = {outcome.n_simulated}", flush=True)

    return outcome

