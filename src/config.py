"""Team configurations, scenario groups and tiers, and the search settings."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Optional

AGENT_ROLE_COUNT = 8

# The five groups of scenario parameters.  Each is supplied by its own JSON file, and
# together they cover every ScenarioParameters field exactly once.
SCENARIO_GROUPS: tuple[str, ...] = ("human", "project", "debt", "agent", "bugimpact")

# The three tiers every scenario group is defined at.
SCENARIO_TIERS: tuple[str, ...] = ("low", "medium", "high")

# `project` and `agent` carry a fourth tier on top of those three, so the tier grid is
# not a cube: the free combination of the per-group tiers is 3^3 * 4 * 4 = 432 scenarios.
# Use `scenario_tiers` / `scenario_tier_combinations` rather than assuming SCENARIO_TIERS
# applies to every group.
SCENARIO_GROUP_TIERS: dict[str, tuple[str, ...]] = {
    "human": SCENARIO_TIERS,
    "project": SCENARIO_TIERS + ("veryhigh",),
    "debt": SCENARIO_TIERS,
    "agent": SCENARIO_TIERS + ("pessimistic",),
    "bugimpact": SCENARIO_TIERS,
}

# Every tier name in use, in mild -> extreme order; plot legends and ordinal encodings
# iterate this.  A group that lacks a tier simply contributes no rows for it.
ALL_SCENARIO_TIERS: tuple[str, ...] = (
    "low", "medium", "high", "veryhigh", "pessimistic",
)

# One-letter tier code used by the scenario ids (`p-v`, `a-p`).  Distinct per tier, so a
# scenario id round-trips to its tiers.  A tier initial may coincide with a *group*
# initial — the id token format is `<group>-<tier>`, so `a-p` is unambiguous.
TIER_INITIALS: dict[str, str] = {
    "low": "l", "medium": "m", "high": "h", "veryhigh": "v", "pessimistic": "p",
}

# Sentinel "tier" for a group a scenario does not use: no `debt`- or `agent`-group
# parameter reaches a zero-agent team, so a human-only baseline marks them `d-X` / `a-X`
# rather than pinning them to a tier that would imply a choice.  Kept out of
# TIER_INITIALS and ALL_SCENARIO_TIERS so it enters neither the scenario product nor the
# figure tier channels; `config/<group>_unused.json` supplies placeholder values, which
# keeps the `<group>_<tier>.json` file-name convention intact.
UNUSED_TIER: str = "unused"
UNUSED_TIER_INITIAL: str = "X"


def scenario_tiers(group: str) -> tuple[str, ...]:
    """The tiers `group` is defined at, in mild -> extreme order."""
    try:
        return SCENARIO_GROUP_TIERS[group]
    except KeyError:
        raise ValueError(f"unknown scenario group {group!r}") from None


def require_scenario_groups(groups: "tuple[str, ...]") -> None:
    """Raise unless every name in `groups` is a scenario group."""
    unknown = [g for g in groups if g not in SCENARIO_GROUPS]
    if unknown:
        raise ValueError(f"unknown scenario group(s) {unknown}; "
                         f"expected one of {list(SCENARIO_GROUPS)}")


def require_distinct_groups(groups: "tuple[str, ...]") -> None:
    """Raise unless `groups` are all scenario groups and all different."""
    require_scenario_groups(groups)
    if len(set(groups)) != len(groups):
        raise ValueError(f"the scenario groups must differ, got {list(groups)}")


def require_group_pair(groups: "tuple[str, ...]") -> tuple[str, ...]:
    """Validate a two-group faceting pair; returns it unchanged."""
    groups = tuple(groups)
    if len(groups) != 2:
        raise ValueError(f"expected exactly two faceting groups, got {list(groups)}")
    require_distinct_groups(groups)
    return groups


def scenario_tier_combinations() -> list[tuple[str, ...]]:
    """Every tier combination, one tuple per group in SCENARIO_GROUPS order.

    The product of the *per-group* tiers, so 3^3 * 4 * 4 = 432 combinations rather than
    3^5: `project` and `agent` each carry a fourth tier.
    """
    return list(itertools.product(*(scenario_tiers(g) for g in SCENARIO_GROUPS)))


def scenario_id(tiers: tuple[str, ...]) -> str:
    """Compact label of a tier combination, e.g. ``h-l_p-v_d-h_a-l_b-m``.

    One ``<group initial>-<tier initial>`` token per group, in SCENARIO_GROUPS order.
    Both initial sets are distinct, so the label round-trips through
    `campaign.tiers_from_scenario_id`.  UNUSED_TIER is accepted and renders as `X`.
    """
    if len(tiers) != len(SCENARIO_GROUPS):
        raise ValueError(f"expected {len(SCENARIO_GROUPS)} tiers, got {len(tiers)}")
    initials = {**TIER_INITIALS, UNUSED_TIER: UNUSED_TIER_INITIAL}
    return "_".join(
        f"{group[0]}-{initials[tier]}"
        for group, tier in zip(SCENARIO_GROUPS, tiers)
    )


# Retention policy for the per-candidate result.json files a search writes:
#   "all"  — one file per simulated candidate (10,000 per search at campaign scale)
#   "best" — only c* and the indistinguishable set I* it is reported with
#   "none" — no per-candidate files; the aggregates live in search_summary.json
SAMPLE_POLICIES: tuple[str, ...] = ("all", "best", "none")

# How a result directory was produced.  `SEARCH_MODE` runs the search; `HUMAN_ONLY_MODE`
# enumerates the zero-agent baseline grid, which is small enough not to need searching.
# These name the subdirectory under a scenario, the `mode` column of the campaign tables,
# and the value `search_dirs` looks for.
SEARCH_MODE: str = "search"
HUMAN_ONLY_MODE: str = "human_only"
RESULT_MODES: tuple[str, ...] = (SEARCH_MODE, HUMAN_ONLY_MODE)

AGENT_ROLE_NAMES: tuple[str, ...] = (
    "Feature Implementation Agent",           # k=1
    "Architecture & Design Proposal Agent",   # k=2
    "Bug Fixing Agent",                       # k=3
    "Refactoring Agent",                      # k=4
    "Code Review Agent",                      # k=5
    "Security Audit Agent",                   # k=6
    "Test Development Agent",                 # k=7
    "Developer Documentation Agent",          # k=8
)

# The same roles in the same k=1..8 order, short enough to use as a plot tick label.
AGENT_ROLE_ABBREV: tuple[str, ...] = (
    "Feature", "Design", "BugFix", "Refactor", "Review", "Security", "Test", "Docs",
)

# Roles split by human oversight.  Only the supervised roles consume monitoring capacity
# and enter the feasibility check; the two detection services and the Refactoring Agent
# run autonomously and are charged no monitoring overhead.
SUPERVISED_ROLE_INDICES: tuple[int, ...] = (0, 1, 2, 6, 7)      # k = 1,2,3,7,8
AUTONOMOUS_ROLE_INDICES: tuple[int, ...] = (3, 4, 5)            # k = 4,5,6


def supervised_agent_count(A: tuple[int, ...]) -> int:
    """Number of agents in `A` that consume human supervision."""
    return sum(A[i] for i in SUPERVISED_ROLE_INDICES)


def net_capacity(H_capacity: float, c_comm: float, N: float) -> float:
    """One FTE's capacity budget after communication overhead.

    Communication is charged per peer, and peers are counted by headcount rather than by
    staffing level: ceil(N) is the smallest headcount that can supply N FTE, which leaves
    ceil(N) - 1 = ceil(N - 1) peers.
    """
    return H_capacity - c_comm * math.ceil(N - 1.0)


@dataclass(frozen=True, slots=True)
class TeamConfiguration:
    """The configuration tuple (N, A_dev, R, F_freeze, A_qa), fixed across replications.

    N       staffing level in full-time equivalents, not a headcount, so it may be
            fractional: N = 0.25 is one developer investing a quarter of their time.
    A_dev   (a_1, ..., a_8), the development-phase composition.  Either all zero (a team
            with no agents) or a_1 >= 1.
    R       release cadence, in sprints.
    F_freeze
            how many sprints at the end of each release cycle use A_qa instead of A_dev.
            Sprints with (t % R) >= (R - F_freeze) are frozen.  Range [0, R-1]; 0 means
            no feature freeze at all.
    A_qa    (a_1, ..., a_8), the freeze-phase composition; defaults to A_dev when
            omitted.  Requires a_1 = a_2 = 0 whenever there is a freeze window.  With
            F_freeze == 0 it is never applied and is normalised to the zero vector, so
            that a configuration is identified only by what affects its behaviour.

    No collective bound on the supervised roles is checked here: that needs the scenario
    parameters and lives in `src.feasibility`.  The three autonomous roles are bounded
    only by the search-side sampling limits in `src.search`.
    """

    N: float                            # Human staffing level in FTE
    A_dev: tuple[int, ...]              # Dev-phase composition, length == AGENT_ROLE_COUNT
    R: int                              # Release cadence (sprints)
    F_freeze: int = 1                   # QA-phase length per release cycle, in [0, R-1]
    A_qa: Optional[tuple[int, ...]] = None  # QA-phase composition; None → defaults to A_dev

    def __post_init__(self) -> None:
        # Resolve A_qa before validation so subsequent checks can use it.
        if self.A_qa is None:
            object.__setattr__(self, "A_qa", self.A_dev)

        if self.N <= 0:
            raise ValueError(f"N must be > 0, got {self.N}")
        if self.R < 1:
            raise ValueError(f"R must be >= 1, got {self.R}")

        if len(self.A_dev) != AGENT_ROLE_COUNT:
            raise ValueError(f"A_dev must have length {AGENT_ROLE_COUNT}, got {len(self.A_dev)}")
        if any(a < 0 for a in self.A_dev):
            raise ValueError(f"A_dev entries must be >= 0, got {self.A_dev}")
        # A_dev is either all zero (no agents at all) or has at least one Feature
        # Implementation Agent; a team with agents but no implementer is rejected.
        if any(self.A_dev):
            if self.A_dev[0] < 1:
                raise ValueError("a_1 (Feature Implementation Agent) must be >= 1")
        elif self.A_qa is not None and any(self.A_qa):
            raise ValueError(
                "A_qa must be all-zero when A_dev is all-zero (human-only baseline)"
            )

        if not (0 <= self.F_freeze <= self.R - 1):
            raise ValueError(
                f"F_freeze must be in [0, R-1={self.R - 1}], got {self.F_freeze}"
            )

        if len(self.A_qa) != AGENT_ROLE_COUNT:  # type: ignore[arg-type]
            raise ValueError(
                f"A_qa must have length {AGENT_ROLE_COUNT}, got {len(self.A_qa)}"  # type: ignore[arg-type]
            )
        if any(a < 0 for a in self.A_qa):  # type: ignore[union-attr]
            raise ValueError(f"A_qa entries must be >= 0, got {self.A_qa}")
        if self.F_freeze > 0:
            if self.A_qa[0] != 0:  # type: ignore[index]
                raise ValueError(
                    "a_qa,1 (Feature Implementation Agent) must be 0 during feature freeze"
                )
            if self.A_qa[1] != 0:  # type: ignore[index]
                raise ValueError(
                    "a_qa,2 (Architecture & Design Agent) must be 0 during feature freeze"
                )
        else:
            # With no freeze window A_qa is never applied, so normalise it to zero and
            # two teams that differ only in it compare, hash and simulate as one.  After
            # the shape checks above, so a malformed A_qa still raises.
            object.__setattr__(self, "A_qa", (0,) * AGENT_ROLE_COUNT)

    def agent_count_dev(self, role_k: int) -> int:
        """1-indexed access to the development composition A_dev: role_k in {1, ..., 8}."""
        if not 1 <= role_k <= AGENT_ROLE_COUNT:
            raise ValueError(f"role_k must be in [1, {AGENT_ROLE_COUNT}], got {role_k}")
        return self.A_dev[role_k - 1]

    def agent_count_qa(self, role_k: int) -> int:
        """1-indexed access to the QA/freeze composition A_qa: role_k in {1, ..., 8}."""
        if not 1 <= role_k <= AGENT_ROLE_COUNT:
            raise ValueError(f"role_k must be in [1, {AGENT_ROLE_COUNT}], got {role_k}")
        return self.A_qa[role_k - 1]  # type: ignore[index]

    def active_A(self, is_frozen: bool) -> tuple[int, ...]:
        """Return A_qa during freeze sprints, A_dev otherwise."""
        return self.A_qa if is_frozen else self.A_dev  # type: ignore[return-value]


def is_human_only(config: TeamConfiguration) -> bool:
    """True for a team with no agents at all.

    `__post_init__` forces A_qa to zero whenever A_dev is zero, so A_dev alone decides.
    `src.sprint` uses this to switch to the human quality-assurance path.
    """
    return not any(config.A_dev)


@dataclass(frozen=True, slots=True)
class SearchConfiguration:
    """Parameters controlling a configuration-space search."""

    scenario_paths: dict[str, str]          # one path per entry of SCENARIO_GROUPS
    horizon_path: str
    N_values: tuple[float, ...]             # staffing levels in FTE; may be fractional
    R_values: tuple[int, ...]
    F_freeze_values: tuple[int, ...] = (0, 1)   # must contain 0; see __post_init__
    sigma: float = 2.0                      # headroom factor on the saturation bounds
    epsilon: float = 0.02                   # throughput-constraint tolerance
    budget: int = 500                       # candidate evaluations the search may spend
    n_samples_initial: int = 100            # initial population
    n_generations: int = 15                 # generation count
    n_offspring: int = 20                   # offspring per generation
    n_elite: int = 20                       # elite that parents are drawn from
    # Enumerate the zero-agent grid instead of searching; every field above is then unused.
    human_only: bool = False
    samples: str = "best"                   # per-candidate result.json retention policy
    seed: int = 0                           # RNG seed for search
    scenario_id: str = ""                   # label of the tier combination, e.g. "h-l_p-m_d-h_a-l_b-m"

    def __post_init__(self) -> None:
        if set(self.scenario_paths) != set(SCENARIO_GROUPS):
            missing = sorted(set(SCENARIO_GROUPS) - self.scenario_paths.keys())
            unknown = sorted(self.scenario_paths.keys() - set(SCENARIO_GROUPS))
            raise ValueError(
                f"scenario_paths must have exactly one entry per scenario group "
                f"{list(SCENARIO_GROUPS)}; missing={missing}, unknown={unknown}"
            )
        if not self.N_values:
            raise ValueError("N_values must be non-empty")
        if any(n <= 0 for n in self.N_values):
            raise ValueError(f"N_values must all be > 0, got {self.N_values}")
        if self.sigma < 1:
            raise ValueError(f"sigma must be >= 1, got {self.sigma}")
        if not self.R_values:
            raise ValueError("R_values must be non-empty")
        if not self.F_freeze_values:
            raise ValueError("F_freeze_values must be non-empty")
        if 0 not in self.F_freeze_values:
            # F_freeze is capped at R-1, so without 0 a drawn R=1 would leave no
            # admissible freeze length at all.
            raise ValueError(
                f"F_freeze_values must contain 0 so that every R (in particular R=1) "
                f"has an admissible feature-freeze length, got {self.F_freeze_values}"
            )
        if not 0.0 <= self.epsilon < 1.0:
            raise ValueError(f"epsilon must be in [0, 1), got {self.epsilon}")
        if self.budget < 1:
            raise ValueError(f"budget must be >= 1, got {self.budget}")
        if self.n_samples_initial < 1:
            raise ValueError(f"n_samples_initial must be >= 1, got {self.n_samples_initial}")
        if self.n_generations < 1:
            raise ValueError(f"n_generations must be >= 1, got {self.n_generations}")
        if self.n_offspring < 1:
            raise ValueError(f"n_offspring must be >= 1, got {self.n_offspring}")
        if self.n_elite < 1:
            raise ValueError(f"n_elite must be >= 1, got {self.n_elite}")
        if self.samples not in SAMPLE_POLICIES:
            raise ValueError(
                f"samples must be one of {list(SAMPLE_POLICIES)}, got {self.samples!r}"
            )


@dataclass(frozen=True, slots=True)
class SimulationHorizon:
    """Simulation horizon and Monte Carlo settings.

    The two replication counts are the two stages of selection: `K_search` scores every
    candidate the search evaluates, and the larger `K_final` re-simulates the preliminary
    indistinguishable set before it is reported.  The campaign uses 100 and 1,000.
    """

    T: int                      # sprints per replication
    K_search: int               # replications per candidate during the search
    K_final: int                # replications for the re-simulation of I*
    master_seed: int            # root seed

    def __post_init__(self) -> None:
        if self.T < 1:
            raise ValueError(f"T must be >= 1, got {self.T}")
        if self.K_search < 1:
            raise ValueError(f"K_search must be >= 1, got {self.K_search}")
        if self.K_final < self.K_search:
            # The re-simulation exists to narrow the confidence intervals, which it can
            # only do with at least as many replications as the search used.
            raise ValueError(
                f"K_final must be >= K_search, got {self.K_final} < {self.K_search}"
            )
