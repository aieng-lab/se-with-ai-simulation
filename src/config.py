"""Team configuration (N, A_dev, R, F_freeze, A_qa)"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Optional

AGENT_ROLE_COUNT = 8

# The five orthogonal groups of scenario parameters (paper §4.1).  Each group is
# supplied by its own JSON file; together they cover every ScenarioParameters
# field exactly once.
SCENARIO_GROUPS: tuple[str, ...] = ("human", "project", "debt", "agent", "bugimpact")

# The three tiers every scenario group is defined at (paper §4.1).
SCENARIO_TIERS: tuple[str, ...] = ("low", "medium", "high")

# Two groups carry a fourth tier on top of those three (paper §4.1), each a stress
# case rather than an observed operating point:
#   `project`  "veryhigh"     — r_arrival = 1000 issues/week; the high tier is already
#                               managed by two FTE, so it does not stress the simulation.
#   `agent`    "pessimistic"  — the capability of the low tier at ten times the cost of
#                               the high tier; both halves of the agent grounding (the
#                               SWE-bench scores and the token prices) are optimistic.
# The tier grid is therefore *not* a cube — combining the per-group tiers freely yields
# 3^3 * 4 * 4 = 432 scenarios (paper §4.2).  Read tiers through `scenario_tiers` /
# `scenario_tier_combinations` rather than assuming SCENARIO_TIERS applies to every group.
SCENARIO_GROUP_TIERS: dict[str, tuple[str, ...]] = {
    "human": SCENARIO_TIERS,
    "project": SCENARIO_TIERS + ("veryhigh",),
    "debt": SCENARIO_TIERS,
    "agent": SCENARIO_TIERS + ("pessimistic",),
    "bugimpact": SCENARIO_TIERS,
}

# Every tier name in use anywhere, in canonical mild -> extreme order.  Plot legends
# and ordinal encodings iterate this; a group that lacks a tier simply has no rows.
# The two fourth tiers are appended rather than interleaved: no group holds both, so
# their relative order is arbitrary, and appending keeps the base tiers' positions.
ALL_SCENARIO_TIERS: tuple[str, ...] = (
    "low", "medium", "high", "veryhigh", "pessimistic",
)

# One-letter tier code, used by the scenario ids (`p-v`, `a-p`) and the LaTeX scenario
# codes (uppercased).  Distinct per tier, so a scenario id round-trips to its tiers.
# A tier initial may coincide with a *group* initial — the id format is
# `<group>-<tier>`, so `a-p` is unambiguous.
TIER_INITIALS: dict[str, str] = {
    "low": "l", "medium": "m", "high": "h", "veryhigh": "v", "pessimistic": "p",
}

# Sentinel "tier" for a group a scenario does not use.  The human-only baseline of §4.1
# has no agents, and after Eq. (Binj-human) became an absolute rate no `agent`- or
# `debt`-group parameter reaches it at all, so those two groups are marked `a-X` / `d-X`
# rather than pinned to a tier that would imply a choice was made.  Deliberately kept out
# of TIER_INITIALS and ALL_SCENARIO_TIERS: it must never enter the 432-scenario product
# (`scenario_tier_combinations`) nor the figure tier channels.  `config/<group>_unused.json`
# supplies the placeholder values, so the `<group>_<tier>.json` convention still holds and
# `campaign.scenario_params_for_id` needs no special case.
UNUSED_TIER: str = "unused"
UNUSED_TIER_INITIAL: str = "X"


def scenario_tiers(group: str) -> tuple[str, ...]:
    """The tiers `group` is defined at, in mild -> extreme order."""
    try:
        return SCENARIO_GROUP_TIERS[group]
    except KeyError:
        raise ValueError(f"unknown scenario group {group!r}") from None


def scenario_tier_combinations() -> list[tuple[str, ...]]:
    """Every tier combination of paper §4.2, one tuple per group in SCENARIO_GROUPS order.

    The product of the *per-group* tiers, so 3^3 * 4 * 4 = 432 combinations rather than
    3^5: `project` and `agent` each carry a fourth tier.
    """
    return list(itertools.product(*(scenario_tiers(g) for g in SCENARIO_GROUPS)))


def scenario_id(tiers: tuple[str, ...]) -> str:
    """Compact label of a tier combination, e.g. ``h-l_p-v_d-h_a-l_b-m``.

    One ``<group initial>-<tier initial>`` token per group, in SCENARIO_GROUPS order.
    Both initial sets are distinct, so the label round-trips to the tier tuple
    (`src.campaign.tiers_from_scenario_id`).  UNUSED_TIER is accepted and renders as `X`,
    which is how the human-only baseline marks the groups it does not use.
    """
    if len(tiers) != len(SCENARIO_GROUPS):
        raise ValueError(f"expected {len(SCENARIO_GROUPS)} tiers, got {len(tiers)}")
    initials = {**TIER_INITIALS, UNUSED_TIER: UNUSED_TIER_INITIAL}
    return "_".join(
        f"{group[0]}-{initials[tier]}"
        for group, tier in zip(SCENARIO_GROUPS, tiers)
    )


# Retention policy for the per-candidate result.json files a search writes:
#   "all"  — one file per feasible candidate (10,000 per search at §4.2 scale)
#   "best" — only c* and the indistinguishable set I* it is reported with (§3.10)
#   "none" — no per-candidate files; aggregates live in search_summary.json
SAMPLE_POLICIES: tuple[str, ...] = ("all", "best", "none")

# Paper §3.11 describes exactly one search: the evolutionary search over an elite ranked
# by the preference relation.  Uniform random sampling is not a strategy of its own — it
# is the initialiser the evolutionary search seeds from (§3.11.2) — and there is no longer
# a second strategy to compare against, so the search is not configurable.
EVOLUTIONARY_MODE: str = "evolutionary"

# The human-only baseline of §4.1 is *not* a search: its configuration space is 99 points
# and is enumerated exhaustively, so there is nothing to search and no budget to spend.
# It is selected by `SearchConfiguration.human_only` rather than by a strategy name.
HUMAN_ONLY_MODE: str = "human_only"

# Everything a result directory can be produced by.  `campaign.search_dirs`, the result
# tree, the `strategy` column of the campaign CSVs and the per-mode figure names all range
# over these two.
RESULT_MODES: tuple[str, ...] = (EVOLUTIONARY_MODE, HUMAN_ONLY_MODE)

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

# Compact labels for the same roles, in the same k=1..8 order.  AGENT_ROLE_NAMES runs
# to 37 characters, which is unusable as a plot tick label or in a composition
# signature; these are the short forms used by the §5 composition figures.
AGENT_ROLE_ABBREV: tuple[str, ...] = (
    "Feature", "Design", "BugFix", "Refactor", "Review", "Security", "Test", "Docs",
)

# Roles are split into two tiers with respect to human oversight (paper §3.3):
# K_sup = {1,2,3,7,8} and K_aut = {4,5,6}.  Supervised roles consume monitoring capacity,
# and how many of them a staffing level can carry follows from the human-capacity
# condition of §3.6, Eq. (feas_monitoring) — there is no separate span-of-control budget.
# The three autonomous roles consume none: the two detection services (Code Review,
# Security Audit) are triggered automatically by the CI infrastructure and report their
# findings asynchronously, and the Refactoring Agent does low-stakes internal improvements
# that the developers do not review (§3.1).  None of them is charged monitoring overhead.
SUPERVISED_ROLE_INDICES: tuple[int, ...] = (0, 1, 2, 6, 7)      # k = 1,2,3,7,8
AUTONOMOUS_ROLE_INDICES: tuple[int, ...] = (3, 4, 5)            # k = 4,5,6


def supervised_agent_count(A: tuple[int, ...]) -> int:
    """Number of agents in `A` that consume human supervision (n_sup, paper §3.3)."""
    return sum(A[i] for i in SUPERVISED_ROLE_INDICES)


def net_capacity(H_capacity: float, c_comm: float, N: float) -> float:
    """H_net of paper §3.6, Eq. (hnet): one FTE's budget after communication overhead.

    Communication is a per-person cost that does not shrink when a person invests only
    part of their time in the project, so the number of peers is driven by the
    headcount rather than by the staffing level: ceil(N) is the smallest headcount
    that can supply N FTE, leaving ceil(N) - 1 = ceil(N - 1) peers.
    """
    return H_capacity - c_comm * math.ceil(N - 1.0)


@dataclass(frozen=True, slots=True)
class TeamConfiguration:
    """Configuration tuple (N, A_dev, R, F_freeze, A_qa) fixed across replications.

    A_dev = (a_1, ..., a_8) — development-phase composition, a_1 >= 1.
    A_qa  = (a_1, ..., a_8) — QA/stabilisation-phase composition.
            Defaults to A_dev when omitted.  Expected to have a_1=a_2=0 so the
            deployed agents are redeployed to bug-fixing, testing, and
            refactoring during the feature-freeze window.  When F_freeze == 0 it is
            never applied and is therefore canonicalised to the zero vector, so that
            a configuration is identified only by the parameters that affect it.
    N is a staffing level in full-time equivalents, not a headcount (paper §3.1),
    so it may be fractional: N = 0.25 is one developer investing a quarter of
    their working time.  How many *supervised* roles (see SUPERVISED_ROLE_INDICES)
    a staffing level can carry is not a structural property of the configuration:
    it follows from the human-capacity condition of §3.6, Eq. (feas_monitoring),
    which needs the scenario parameters and is therefore checked by
    `src.feasibility`, not here.  The two detection services (a_5, a_6) are unbounded,
    since their only brake is their token cost weighed against concave detection
    returns; the Refactoring Agent (a_4) is unsupervised in the same way, and the only
    cap on it is the sampling bound of §3.11.2 (`search_configurations.refactoring_bound`).
    F_freeze — number of sprints at the end of each release cycle in which the
            team uses A_qa instead of A_dev.  Sprints satisfying
            (t % R) >= (R - F_freeze) are "frozen".  Range: [0, R-1].
            F_freeze = 0 means no feature freeze (A_qa is never used, and is
            canonicalised to zero — see A_qa above).
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

        # Validate development composition A_dev
        if len(self.A_dev) != AGENT_ROLE_COUNT:
            raise ValueError(f"A_dev must have length {AGENT_ROLE_COUNT}, got {len(self.A_dev)}")
        if any(a < 0 for a in self.A_dev):
            raise ValueError(f"A_dev entries must be >= 0, got {self.A_dev}")
        # A_dev is either the all-zero vector — the human-only baseline of paper §4.1,
        # where the developers implement and review everything themselves — or a team
        # with at least one Feature Implementation Agent.  A composition that deploys
        # agents but no implementer is neither and stays rejected.
        if any(self.A_dev):
            if self.A_dev[0] < 1:
                raise ValueError("a_1 (Feature Implementation Agent) must be >= 1")
        elif self.A_qa is not None and any(self.A_qa):
            raise ValueError(
                "A_qa must be all-zero when A_dev is all-zero (human-only baseline)"
            )

        # Validate F_freeze
        if not (0 <= self.F_freeze <= self.R - 1):
            raise ValueError(
                f"F_freeze must be in [0, R-1={self.R - 1}], got {self.F_freeze}"
            )

        # Validate QA composition A_qa
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
            # F_freeze == 0 means the QA composition is never applied (paper §3.3), so any
            # vector here describes nothing.  Canonicalise it to zero *after* the shape
            # checks above, so a malformed A_qa still raises rather than being silently
            # replaced.  This is what keeps two configurations that differ only in an
            # inert A_qa from becoming two points: `rng._config_hash` folds A_qa in, so
            # without canonicalisation they would draw different RNG streams and land at
            # different objective values while being the same model — multiplying the
            # search space and the reported set alike.  Equality and hashing follow
            # the fields, so the collapse also makes the evolutionary `seen` set reject
            # such children for free.
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

    def n_supervised(self, is_frozen: bool) -> int:
        """Supervised agent count n_sup of the composition active in this sprint."""
        return supervised_agent_count(self.active_A(is_frozen))


def is_human_only(config: TeamConfiguration) -> bool:
    """True for the human-only baseline of paper §4.1: a team with no agents at all.

    `__post_init__` forces A_qa to zero whenever A_dev is zero, so A_dev alone decides.
    This is the gate that switches `src.sprint` onto the human quality-assurance path of
    Eqs. (hafter_qa), (cov_qa) and (qa_human), and that `src.visualization` reads to draw
    the baseline outside the tier channels — do not re-derive it inline.
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
    sigma: float = 2.0                      # throughput-saturation headroom factor (§3.11.1)
    epsilon: float = 0.02                   # throughput-constraint tolerance (§3.9)
    budget: int = 500                       # E: candidate evaluations (§4.2)
    n_samples_initial: int = 100            # initial population
    n_generations: int = 15                 # generation count
    n_offspring: int = 20                   # offspring per generation
    n_elite: int = 20                       # elite that parents are drawn from
    # The human-only baseline of §4.1 is enumerated exhaustively rather than searched, so
    # every field above is ignored when this is set.
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
    """Top-level simulation horizon and Monte Carlo settings."""

    T: int                      # Sprints per replication
    K: int                      # Monte Carlo replications per configuration
    master_seed: int            # Root seed s_0

    def __post_init__(self) -> None:
        if self.T < 1:
            raise ValueError(f"T must be >= 1, got {self.T}")
        if self.K < 1:
            raise ValueError(f"K must be >= 1, got {self.K}")
