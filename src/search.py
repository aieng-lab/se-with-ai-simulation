"""Search for the cheapest team that keeps up with the work the project generates.

`run_search` maintains a population seeded by `_random_draws` and mutates parents drawn
from an elite ranked by `src.selection`.  `run_human_only` sits beside it and enumerates
the zero-agent grid instead; it is the only producer of teams with no agents, since the
samplers and `_mutate` keep their `a_1 >= 1` floor.

`two_stage_selection` turns either outcome into the reported result: the candidates the
search ranked at `K_search` replications are thinned by the parsimony tie-break, and what
is left is re-simulated at `K_final` before `c*` and `I*` are settled.

Output layout
-------------
    RESULTS_DIR/
        manifest.json          metadata and one entry per evaluated candidate
        search_summary.json    c*, the indistinguishable set I*, the objectives of every
                               simulated candidate at K_search, and the re-simulated ones
                               at K_final in `final_objectives`
        convergence.json       the per-generation convergence record and the size of I*
                               at each stage of the selection
        samples/
            N2_R4_00042.json   per-replication objectives, for the candidates
                               SearchConfiguration.samples selects

Supervised role counts are bounded per role by `saturation_bounds` — beyond that an agent
has no work to do — and collectively by the human-capacity check in `src.feasibility`,
which runs after the draw.  The three autonomous roles are outside both: the two detection
services are capped by nothing at all and are merely sampled within a range, and the
Refactoring Agent is capped by `refactoring_bound`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
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
    parsimony_key,
    rank,
    report_optimum,
    satisfies_throughput,
    throughput_floor,
)
from .simulation import ConfigResult, run_configuration
from .state import SystemState


# ---------------------------------------------------------------------------
# Composition sampling
# ---------------------------------------------------------------------------

#: The Refactoring Agent (k=4), the one autonomous role that carries an upper bound; the
#: sampler and the mutation operator both special-case it.
_REFACTOR_INDEX: int = 3

_QA_SUPERVISED_INDICES: tuple[int, ...] = tuple(
    i for i in SUPERVISED_ROLE_INDICES if i >= 2
)   # QA phase pins roles 1-2 (indices 0-1) to zero


def saturation_bounds(params, sigma: float) -> dict[int, int]:
    """Per-role upper bounds for the supervised roles, keyed by 0-based role index.

    Every supervised role is fed by work that derives from the feature arrival rate, so
    deploying more agents than the inflow can keep busy buys nothing.  The bound is
    `ceil(sigma * workload / rate)`: `workload` is the steady-state inflow that reaches
    the role, `rate` the throughput one agent sustains with the debt amplifier at its
    floor, and `sigma >= 1` the headroom against the Poisson peaks of the arrivals.

    The autonomous roles are absent — see `refactoring_bound` and `review_sample_max`.
    Every workload is a positive multiple of r_arrival, so every bound is at least 1.
    """
    if sigma < 1:
        raise ValueError(f"sigma must be >= 1, got {sigma}")

    r = params.r_arrival
    # The bug-fixing inflow counts the agent-injected stream only; the human stream is an
    # absolute per-feature rate and is not represented here.  That under-counts, but
    # harmlessly: the bug-fixing rate is in the hundreds, so the bound is 1 either way.
    bug_inflow = params.r_bugs * (2.0 - params.delta_min) * r
    workload = {
        0: (1.0 + params.f_arch_chg) * r,                       # k=1 Feature Implementation
        1: params.f_arch_chg * r,                               # k=2 Architecture & Design
        2: bug_inflow,                                          # k=3 Bug Fixing
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

    The two are drawn uniformly from {0, ..., floor(r_arrival)}, i.e. at most one service
    instance per arriving feature.  This is a sampling choice, not a bound — nothing caps
    these roles, and the mutation operator may take them beyond it.
    """
    return int(params.r_arrival)


def refactoring_bound(params) -> int:
    """Upper bound on the Refactoring Agent: floor(r_arrival / 2).

    Unlike `review_sample_max` this is a genuine cap, honoured by the sampler and by the
    mutation operator alike: up to one Refactoring Agent per two arriving features.  It is
    not scaled by `sigma`, and it is 0 below r_arrival = 2, which pins the role to zero.
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
    """Draw each supervised role independently and uniformly within its bound.

    `a_k ~ U{0, ..., bound_k}`, except that with `min_impl` the dev-phase Feature
    Implementation role is drawn from `U{1, ..., bound_1}` so `a_dev,1 >= 1` holds by
    construction.  The human-capacity condition is not enforced here; a draw that
    violates it is rejected later by the feasibility gate.
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
# Mutation operator
# ---------------------------------------------------------------------------

def _pick(indices: tuple[int, ...], rng: np.random.Generator) -> int:
    """Uniformly pick one role index from `indices`."""
    return indices[int(rng.integers(len(indices)))]


def _service_step(idx: int, rng: np.random.Generator,
                  review_max: int, refactor_max: int) -> int:
    """One signed "adjust service" step for autonomous role `idx`.

    The size is `2**j` with `j` uniform over `{0, ..., floor(log2 range)}`, where `range`
    is the role's sampling range — `review_max` for the detection services, `refactor_max`
    for the Refactoring Agent — and the sign is uniform.  Exponentially spaced steps let
    the search traverse counts in the hundreds, which a +/-1 walk could not.  A range of 0
    leaves only `j = 0`.
    """
    s = refactor_max if idx == _REFACTOR_INDEX else review_max
    max_exponent = int(math.log2(s)) if s >= 1 else 0
    step = 1 << int(rng.integers(max_exponent + 1))
    return step * int(rng.choice([-1, 1]))


def _mutate_composition(
    A: list[int],
    mutable: tuple[int, ...],
    bounds: dict[int, int],
    rng: np.random.Generator,
    min_impl: bool,
    review_max: int,
    refactor_max: int,
) -> None:
    """Apply one elementary mutation to `A` in place.

    `mutable` are the supervised role indices that may be changed, each capped by
    `bounds[i]`; the three operations over them move, add or remove one agent.  The
    autonomous roles are not part of that pool and get a separate "adjust service"
    operation instead: `+/- 2**j` from `_service_step`, floored at zero and additionally
    capped at `refactor_max` for the Refactoring Agent.  The two detection services stay
    uncapped, so a mutation may take them past `review_max`.

    A step that lands back on the parent value is a no-op for `_mutate` to retry.  With
    `min_impl`, the Feature Implementation role is protected from being drained to zero.
    """
    op = int(rng.integers(4))   # 0=move, 1=add, 2=remove, 3=adjust service +/-2**j
    if op == 3:
        idx = _pick(AUTONOMOUS_ROLE_INDICES, rng)
        target = A[idx] + _service_step(idx, rng, review_max, refactor_max)
        if idx == _REFACTOR_INDEX:
            target = min(target, refactor_max)
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
    review_max: int,
    refactor_max: int,
) -> TeamConfiguration:
    """Perturb exactly one randomly chosen parameter from {N, F_freeze, A_dev, A_qa}.

    R is never mutated.  Retries up to ten times if the result is unchanged or invalid,
    and returns the parent if every attempt fails.

    With F_freeze == 0 the A_qa option is dropped and the choice is over the other three:
    A_qa is normalised to zero there, so mutating it would produce a child normalised
    straight back to its parent.
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
            # ±1, clamped to [0, R-1].  Going 0 -> 1 opens a freeze window onto the
            # normalised zero A_qa, i.e. one staffed by nobody, which later A_qa mutations
            # can grow.  Re-sampling A_qa here would perturb two parameters at once.
            F_freeze = int(np.clip(F_freeze + int(rng.choice([-1, 1])), 0, config.R - 1))

        elif mutation_type == 2:
            # Exactly one elementary op, maintaining the bounds and A_dev[0] >= 1.
            _mutate_composition(A_dev, SUPERVISED_ROLE_INDICES, bounds,
                                rng, min_impl=True, review_max=review_max,
                                refactor_max=refactor_max)

        else:
            # One elementary op restricted to the supervised QA roles (indices 0-1 stay
            # at 0) plus the autonomous services.
            _mutate_composition(A_qa, _QA_SUPERVISED_INDICES, bounds,
                                rng, min_impl=False, review_max=review_max,
                                refactor_max=refactor_max)

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

    Every draw satisfies `TeamConfiguration`'s structural constraints by construction, so
    nothing is discarded here; the only rejection path is the human-capacity check that
    `_evaluate` applies.
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
        A_dev = _sample_dev_composition(rng, bounds, review_max, refactor_max)
        # With no freeze window the QA composition is never applied, so drawing one would
        # only attach meaningless coordinates to the candidate.
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
    """Check feasibility, then simulate if it passes.

    Returns (manifest_entry, agg_or_None, config_result_or_None); the latter two are None
    for an infeasible configuration, which is never simulated.  Persistence is left to the
    caller, which knows the sample retention policy.
    """
    report = check_feasibility(config, params)
    entry: dict = {
        "feasible": report.feasible,
        "N": config.N, "R": config.R, "F_freeze": config.F_freeze,
        "A_dev": list(config.A_dev),
        "A_qa": list(config.A_qa),
    }
    if not report.feasible:
        entry["reason"] = report.reason
        return entry, None, None

    cr = run_configuration(config, settings, params, initial_state=initial_state)
    agg = aggregate_config_result(cr, master_seed=settings.master_seed)
    return entry, agg, cr


@dataclass(slots=True)
class SearchOutcome:
    """Everything a search or enumeration produces, independent of how it is persisted.

    `results` holds the per-candidate ConfigResults kept for possible sample export: empty
    under the "none" retention policy, and every simulated candidate otherwise (~2.5 KB
    each, so ~25 MB at the campaign budget).
    """

    manifest: dict[str, dict]
    aggs: list[tuple[str, TeamConfiguration, "AggregatedObjectives"]]
    results: dict[str, ConfigResult]
    # One entry per generation actually run.  `run_human_only` leaves all three empty:
    # an enumeration has no generations.
    elite_turnover: list[float] = field(default_factory=list)
    incumbent_changed: list[bool] = field(default_factory=list)
    best_cost: list[float] = field(default_factory=list)

    @property
    def incumbent_turnover(self) -> float | None:
        """Share of generations in which the incumbent changed; None if none ran.

        Reported beside the elite turnover because that one has a noise floor: candidates
        whose costs are indistinguishable at K_search swap ranks inside the elite without
        any improvement taking place.
        """
        if not self.incumbent_changed:
            return None
        return sum(self.incumbent_changed) / len(self.incumbent_changed)

    def record(
        self,
        config: TeamConfiguration,
        settings: SimulationHorizon,
        params,
        initial_state: SystemState | None,
        keep_samples: bool,
        generation: int | None = None,
    ) -> None:
        """Evaluate one candidate and file it under a fresh key.

        Infeasible candidates land in the manifest only; they consume budget but are
        never simulated.  `generation` is recorded when there is one.
        """
        key = _config_key(config, self.n_evaluated)
        entry, agg, cr = _evaluate(config, settings, params, initial_state)
        if generation is not None:
            entry["generation"] = generation
        self.manifest[key] = entry
        if agg is not None:
            self.aggs.append((key, config, agg))
            if keep_samples:
                self.results[key] = cr    # type: ignore[assignment]

    @property
    def n_evaluated(self) -> int:
        """Candidates that consumed budget, including the infeasible ones."""
        return len(self.manifest)

    @property
    def n_simulated(self) -> int:
        """Candidates actually simulated, i.e. those that passed the feasibility check."""
        return len(self.aggs)


def write_samples(
    outcome: SearchOutcome,
    out_dir: Path,
    keys: list[str],
    aggs_override: "dict[str, AggregatedObjectives] | None" = None,
) -> int:
    """Persist `samples/<key>.json` for the requested keys; returns the file count.

    `aggs_override` replaces the search's `K_search` aggregates for the keys it holds, so
    a re-simulated candidate's sample file carries the `K_final` estimates it is reported
    from.
    """
    if not keys:
        return 0
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(exist_ok=True, parents=True)
    aggs_by_key = {key: agg for key, _, agg in outcome.aggs}
    if aggs_override:
        aggs_by_key.update(aggs_override)
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



def run_search(
    search_cfg: SearchConfiguration,
    settings: SimulationHorizon,
    params,
    rng: np.random.Generator,
    initial_state: SystemState | None = None,
    progress_every: int = 100,
) -> SearchOutcome:
    """Random initialisation followed by generations of guided mutation.

    Every candidate that reaches the feasibility check consumes one evaluation, whether or
    not it is simulated.  A generation therefore falls short of `n_offspring` only when
    the mutation operator keeps producing duplicates until its attempt limit runs out.
    The loop also stops as soon as `budget` evaluations have been spent, so the nominal
    `n_samples_initial + n_generations * n_offspring` can never overrun it.
    """
    n_initial   = search_cfg.n_samples_initial
    n_gen       = search_cfg.n_generations
    n_offspring = search_cfg.n_offspring
    n_elite     = search_cfg.n_elite
    budget      = search_cfg.budget
    keep_samples = search_cfg.samples != "none"
    bounds = saturation_bounds(params, search_cfg.sigma)
    review_max = review_sample_max(params)
    refactor_max = refactoring_bound(params)
    floor = throughput_floor(params, search_cfg.epsilon)

    outcome = SearchOutcome(manifest={}, aggs=[], results={})
    # Every candidate that has consumed budget, feasible or not, so a duplicate is
    # never charged twice.
    seen: set[TeamConfiguration] = set()

    def evaluate_new(cfg: TeamConfiguration, generation: int) -> None:
        outcome.record(cfg, settings, params, initial_state, keep_samples, generation)
        seen.add(cfg)

    print(f"Random initialisation ({n_initial} samples) …", flush=True)
    init_draws = _random_draws(search_cfg, rng, min(n_initial, budget), params)
    for i, cfg in enumerate(init_draws):
        evaluate_new(cfg, generation=0)
        _progress(i + 1, len(init_draws), outcome, progress_every)

    # Rank the population and split it into the elite most offspring come from and the
    # remainder that supplies exploration.  The ranking is carried across the loop body —
    # computed here for the initial pool and refreshed at the end of every generation — so
    # one sort both supplies a generation's parents and measures the elite turnover.
    order = rank([a for _, _, a in outcome.aggs], floor)
    prev_elite = {outcome.aggs[i][1] for i in order[:n_elite]}
    prev_incumbent = outcome.aggs[order[0]][1] if order else None

    for gen in range(1, n_gen + 1):
        if outcome.n_evaluated >= budget:
            break
        elite_parents = [outcome.aggs[i][1] for i in order[:n_elite]]
        rest_parents  = [outcome.aggs[i][1] for i in order[n_elite:]]
        # If one pool is empty, use the other exclusively.
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
                child = _mutate(parent, rng, search_cfg, bounds, review_max, refactor_max)
                if child in seen:
                    continue
                evaluate_new(child, generation=gen)
                count += 1        # infeasible children consume budget too
            n_new += count

        # Re-rank over parents and offspring together, and record the convergence
        # measures, before the next generation draws its parents from this order.
        order = rank([a for _, _, a in outcome.aggs], floor)
        elite = {outcome.aggs[i][1] for i in order[:n_elite]}
        incumbent = outcome.aggs[order[0]][1] if order else None
        # The denominator is `n_elite`, not the realised elite size; with
        # n_samples_initial >= n_elite the elite is full from the first generation anyway.
        outcome.elite_turnover.append(len(elite - prev_elite) / n_elite)
        outcome.incumbent_changed.append(incumbent != prev_incumbent)
        outcome.best_cost.append(
            outcome.aggs[order[0]][2].cost_mean if order else float("nan")
        )
        prev_elite, prev_incumbent = elite, incumbent

        if progress_every:
            best_txt = f"{outcome.best_cost[-1]:.0f}" if order else "n/a"
            print(f"  Gen {gen:3d}: best cost = {best_txt}, elite = {len(elite_parents)}, "
                  f"turnover = {outcome.elite_turnover[-1]:.2f}, "
                  f"offspring evaluated = {n_new}, total evaluated = {outcome.n_evaluated}, "
                  f"simulated = {outcome.n_simulated}", flush=True)

    return outcome


def human_only_grid(search_cfg: SearchConfiguration) -> list[TeamConfiguration]:
    """Every zero-agent team in the search range, in a deterministic order.

    With both composition vectors fixed at zero only `(N, R, F_freeze)` remain free, so
    the space is `|N_values| * |{(R, F) : F in F_freeze_values, 0 <= F <= R-1}|` — 99
    points at the campaign range (11 staffing levels x 9 valid `(R, F_freeze)` pairs).
    """
    zero = (0,) * AGENT_ROLE_COUNT
    return [
        TeamConfiguration(N=N, A_dev=zero, R=R, F_freeze=F, A_qa=zero)
        for N in search_cfg.N_values
        for R in search_cfg.R_values
        for F in search_cfg.F_freeze_values
        if 0 <= F <= R - 1
    ]


def run_human_only(
    search_cfg: SearchConfiguration,
    settings: SimulationHorizon,
    params,
    initial_state: SystemState | None = None,
    progress_every: int = 100,
) -> SearchOutcome:
    """Exhaustively enumerate the zero-agent grid.

    Not a search: at 99 candidates there is nothing to search, so `budget` and the
    generation settings are ignored and the result is deterministic.  `saturation_bounds`
    is never called either — it bounds supervised roles this grid does not staff.  Every
    candidate still passes through `_evaluate`, so an infeasible one counts as an
    evaluation exactly as in the search.
    """
    keep_samples = search_cfg.samples != "none"
    outcome = SearchOutcome(manifest={}, aggs=[], results={})
    grid = human_only_grid(search_cfg)

    for cfg in grid:
        outcome.record(cfg, settings, params, initial_state, keep_samples)
        _progress(outcome.n_evaluated, len(grid), outcome, progress_every)

    if progress_every:
        print(f"  Enumerated {outcome.n_evaluated} zero-agent configuration(s), "
              f"simulated = {outcome.n_simulated}", flush=True)

    return outcome



# ---------------------------------------------------------------------------
# Final selection
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SelectionReport:
    """The `(c*, I*)` pair to report, and how the two-stage selection arrived at it.

    `final_aggs` / `final_results` hold the `K_final` estimates of the re-simulated
    candidates — the numbers every reported figure comes from.  The search's own
    `K_search` aggregates stay in `SearchOutcome.aggs`, which is what it ranked by and
    what a plot of the whole evaluated pool shows.  `stats` carries the stage counters.
    """

    best_key: str | None
    best_admissible: bool
    keep_keys: list[str]
    final_aggs: dict[str, AggregatedObjectives]
    final_results: dict[str, ConfigResult]
    stats: dict[str, object]


def _relative_half_width(agg: AggregatedObjectives) -> float:
    """The cost CI half-width relative to the mean."""
    lo, hi = agg.cost_ci
    return (hi - lo) / (2.0 * agg.cost_mean) if agg.cost_mean else float("nan")


def two_stage_selection(
    outcome: SearchOutcome,
    search_cfg: SearchConfiguration,
    settings: SimulationHorizon,
    params,
    initial_state: SystemState | None = None,
    progress_every: int = 100,
) -> SelectionReport:
    """Determine the reported `(c*, I*)` pair in two stages.

    Stage 1 ranks the `K_search` estimates the search produced and applies the parsimony
    tie-break, giving a preliminary `I*`.  Applying the tie-break this early is what makes
    it worth doing: a member it removes is both structurally larger and dearer in
    expectation, so it can be dropped before it costs a `K_final` run.

    Stage 2 re-simulates that reduced set at `K_final > K_search` and repeats the rule on
    the narrower confidence intervals, which is what makes `I*` shrink further.  Three
    things happen there, in this order: a member that no longer satisfies the throughput
    constraint is discarded (`indistinguishable_set` does this by filtering on
    `satisfies_throughput`); `c*` is re-determined and may move; and the tie-break runs a
    second time on the better estimates.  It cannot move `c*`, which is the cost minimiser.

    When nothing reached the throughput floor there is no `c*` and `I*` is empty, so there
    is no set to narrow — the least-violating candidate is re-simulated alone, which keeps
    every reported value a `K_final` number.

    The re-simulation runs a **disjoint** block of replication indices rather than
    extending the search's.  A candidate is in the preliminary set because its `K_search`
    estimate came out at or near the minimum of thousands of noisy estimates, so that
    estimate is biased low by the selection itself; a fresh block removes the bias at the
    same simulation cost.
    """
    floor = throughput_floor(params, search_cfg.epsilon)
    agg_objs = [a for _, _, a in outcome.aggs]
    prelim_keys = [parsimony_key(cfg) for _, cfg, _ in outcome.aggs]
    # Both calls return the same `c*`; the un-keyed one only supplies the set size before
    # the tie-break.  The extra scan is O(n), and the O(n^2) parsimony pass runs over the
    # CI-overlap members alone rather than over the whole evaluated pool.
    prelim_best, prelim_raw = report_optimum(agg_objs, floor)
    _, prelim_set = report_optimum(agg_objs, floor, keys=prelim_keys)

    stats: dict[str, object] = {
        "n_prelim": len(prelim_raw),
        "parsimony_removed_prelim": len(prelim_raw) - len(prelim_set),
        "n_resimulated": 0,
        "n_discarded_throughput": 0,
        "n_after_resim": 0,
        "n_final": 0,
        "parsimony_removed": 0,
        "best_changed_by_resim": False,
        "rho_mean": None,
        "rho_best": None,
        "K_search": settings.K_search,
        "K_final": settings.K_final,
    }
    if prelim_best is None:
        # Nothing was simulated at all; there is nothing to report or re-simulate.
        return SelectionReport(None, False, [], {}, {}, stats)

    prelim_best_key = outcome.aggs[prelim_best][0]
    resim_idx = prelim_set or [prelim_best]
    if progress_every:
        label = ("preliminary I*" if prelim_set
                 else "the least-violating candidate (the search is inadmissible)")
        print(f"Re-simulating {label}: {len(resim_idx)} configuration(s) with "
              f"K={settings.K_final} replications …", flush=True)

    resim_keys: list[str] = []
    resim_cfgs: list[TeamConfiguration] = []
    resim_aggs: list[AggregatedObjectives] = []
    final_results: dict[str, ConfigResult] = {}
    for n, i in enumerate(resim_idx, start=1):
        key, cfg, _ = outcome.aggs[i]
        cr = run_configuration(cfg, settings, params, initial_state=initial_state,
                               K=settings.K_final, k_offset=settings.K_search)
        resim_keys.append(key)
        resim_cfgs.append(cfg)
        resim_aggs.append(aggregate_config_result(cr, master_seed=settings.master_seed))
        final_results[key] = cr           # type: ignore[assignment]
        if progress_every and (n % progress_every == 0 or n == len(resim_idx)):
            print(f"  [{n}/{len(resim_idx)}] re-simulated", flush=True)

    keys = [parsimony_key(cfg) for cfg in resim_cfgs]
    best, members = report_optimum(resim_aggs, floor)          # before the tie-break
    final_best, keep = report_optimum(resim_aggs, floor, keys=keys)
    assert final_best == best, "the parsimony tie-break cannot move c*"

    best_key = resim_keys[best]
    best_admissible = satisfies_throughput(resim_aggs[best], floor)
    keep_keys = [resim_keys[i] for i in keep]

    stats["n_resimulated"] = len(resim_idx)
    # Only a preliminary member can be discarded: in the inadmissible case the single
    # re-simulated candidate is the reported result, not a member of any set.
    stats["n_discarded_throughput"] = (
        sum(1 for a in resim_aggs if not satisfies_throughput(a, floor))
        if prelim_set else 0
    )
    stats["n_after_resim"] = len(members)
    stats["n_final"] = len(keep)
    stats["parsimony_removed"] = len(members) - len(keep)
    stats["best_changed_by_resim"] = best_key != prelim_best_key
    if keep:
        rhos = [_relative_half_width(resim_aggs[i]) for i in keep]
        stats["rho_mean"] = sum(rhos) / len(rhos)
    stats["rho_best"] = _relative_half_width(resim_aggs[best])

    return SelectionReport(
        best_key=best_key,
        best_admissible=best_admissible,
        keep_keys=keep_keys,
        final_aggs=dict(zip(resim_keys, resim_aggs)),
        final_results=final_results,
        stats=stats,
    )
