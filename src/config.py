"""Team configurations, scenario groups and tiers, and the search settings."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path

AGENT_ROLE_COUNT = 8

# The five groups of scenario parameters.  Each is supplied by its own JSON file, and
# together they cover every ScenarioParameters field exactly once.
SCENARIO_GROUPS: tuple[str, ...] = ("human", "project", "debt", "agent", "bugimpact")

# The three tiers every scenario group is defined at.
SCENARIO_TIERS: tuple[str, ...] = ("low", "medium", "high")

# The tiers each group is defined at.  `project` and `agent` carry a fourth one.
# `scenario_tiers` is the accessor.
SCENARIO_GROUP_TIERS: dict[str, tuple[str, ...]] = {
    "human": SCENARIO_TIERS,
    "project": SCENARIO_TIERS + ("veryhigh",),
    "debt": SCENARIO_TIERS,
    "agent": SCENARIO_TIERS + ("pessimistic",),
    "bugimpact": SCENARIO_TIERS,
}

# Every tier name in use, in mild -> extreme order, for legends and ordinal encodings.
ALL_SCENARIO_TIERS: tuple[str, ...] = (
    "low", "medium", "high", "veryhigh", "pessimistic",
)

# One-letter tier code used by the scenario ids.  Distinct per tier, so an id round-trips
# to its tiers.  A tier initial may coincide with a group initial: an id token is always
# `<group>-<tier>`, so the two never have to be told apart.
TIER_INITIALS: dict[str, str] = {
    "low": "l", "medium": "m", "high": "h", "veryhigh": "v", "pessimistic": "p",
}

# Sentinel "tier" for a group a scenario does not use, rendered `X` in a scenario id.
# Kept out of TIER_INITIALS and ALL_SCENARIO_TIERS, so it enters neither the scenario
# product nor the figure tier channels.  `config/<group>_unused.json` supplies placeholder
# values, so the `<group>_<tier>.json` naming convention still holds.
UNUSED_TIER: str = "unused"
UNUSED_TIER_INITIAL: str = "X"

# The tiers beyond the three base ones.  Derived rather than listed, so it cannot drift
# from SCENARIO_GROUP_TIERS.  The plotting channels give these a single shared style slot,
# which works only while no one group holds two of them.
EXTREME_TIERS: tuple[str, ...] = tuple(
    t for t in ALL_SCENARIO_TIERS if t not in SCENARIO_TIERS
)

# The scenario-id tier codec, forward and inverse, defined together so the two cannot
# fall out of step.  UNUSED_TIER is folded in here rather than special-cased by callers.
SCENARIO_ID_INITIALS: dict[str, str] = {**TIER_INITIALS, UNUSED_TIER: UNUSED_TIER_INITIAL}
SCENARIO_ID_TIERS: dict[str, str] = {v: k for k, v in SCENARIO_ID_INITIALS.items()}

#: Directory the scenario JSON files are read from.
CONFIG_DIR: str = "config"

#: The scenario groups a human-only baseline varies.  `debt` and `agent` are inert for a
#: zero-agent team, so one baseline row is the reference for every combination of those.
#: The config generator writes the baseline grid over these groups and
#: `campaign.baseline_uplift` joins on them, so the two must stay in agreement.
BASELINE_VARYING_GROUPS: tuple[str, ...] = ("human", "project", "bugimpact")


def scenario_tiers(group: str) -> tuple[str, ...]:
    """The tiers `group` is defined at, in mild -> extreme order."""
    try:
        return SCENARIO_GROUP_TIERS[group]
    except KeyError:
        raise ValueError(f"unknown scenario group {group!r}") from None


def tiers_present(values) -> list[str]:
    """The tiers occurring in `values`, in mild -> extreme order.

    Takes any iterable of tier names: a DataFrame column, a generator over table rows.
    """
    seen = set(values)
    return [tier for tier in ALL_SCENARIO_TIERS if tier in seen]


def tier_column(group: str) -> str:
    """Name of the tier column `group` occupies in the campaign tables."""
    return f"tier_{group}"


def scenario_config_path(group: str, tier: str) -> Path:
    """Path of the scenario JSON holding `group` at `tier`."""
    return Path(CONFIG_DIR) / f"{group}_{tier}.json"


def scenario_tier_combinations() -> list[tuple[str, ...]]:
    """Every tier combination, one tuple per group in SCENARIO_GROUPS order."""
    return list(itertools.product(*(scenario_tiers(g) for g in SCENARIO_GROUPS)))


def scenario_id(tiers: tuple[str, ...]) -> str:
    """Compact label of a tier combination, e.g. ``h-l_p-v_d-h_a-l_b-m``.

    One ``<group initial>-<tier initial>`` token per group, in SCENARIO_GROUPS order;
    `campaign.tiers_from_scenario_id` is the inverse.  UNUSED_TIER renders as `X`.
    """
    if len(tiers) != len(SCENARIO_GROUPS):
        raise ValueError(f"expected {len(SCENARIO_GROUPS)} tiers, got {len(tiers)}")
    return "_".join(
        f"{group[0]}-{SCENARIO_ID_INITIALS[tier]}"
        for group, tier in zip(SCENARIO_GROUPS, tiers)
    )


# Retention policy for the per-candidate result.json files a search writes:
#   "best" — only c* and the indistinguishable set I* it is reported with
#   "none" — no per-candidate files; the aggregates live in search_summary.json
SAMPLE_POLICIES: tuple[str, ...] = ("best", "none")

# How a result directory was produced.  Names the subdirectory under a scenario, the
# `mode` column of the campaign tables, and the value `search_dirs` looks for.
SEARCH_MODE: str = "search"
HUMAN_ONLY_MODE: str = "human_only"
RESULT_MODES: tuple[str, ...] = (SEARCH_MODE, HUMAN_ONLY_MODE)

# The eight roles in k=1..8 order, short enough to use as a plot tick label.
AGENT_ROLE_ABBREV: tuple[str, ...] = (
    "Feature", "Design", "BugFix", "Refactor", "Review", "Security", "Test", "Docs",
)

# Roles split by human oversight; only the supervised ones enter the feasibility check.
SUPERVISED_ROLE_INDICES: tuple[int, ...] = (0, 1, 2, 6, 7)      # k = 1,2,3,7,8
AUTONOMOUS_ROLE_INDICES: tuple[int, ...] = (3, 4, 5)            # k = 4,5,6


def supervised_agent_count(A: tuple[int, ...]) -> int:
    """Number of agents in `A` that consume human supervision."""
    return sum(A[i] for i in SUPERVISED_ROLE_INDICES)


def net_capacity(H_capacity: float, c_comm: float, N: float) -> float:
    """One FTE's capacity budget after communication overhead.

    Charged per peer, counted by headcount rather than by staffing level: ceil(N) is the
    smallest headcount supplying N FTE, leaving ceil(N) - 1 = ceil(N - 1) peers.
    """
    return H_capacity - c_comm * math.ceil(N - 1.0)


@dataclass(frozen=True, slots=True)
class TeamConfiguration:
    """The configuration tuple (N, A_dev, A_qa, R, F_freeze), fixed across replications.

    N       staffing level in full-time equivalents, so it may be fractional.
    A_dev   (a_1, ..., a_8), the development-phase composition.  Either all zero (a team
            with no agents) or a_1 >= 1.
    A_qa    (a_1, ..., a_8), the freeze-phase composition.  Requires a_1 = a_2 = 0
            whenever there is a freeze window, and is normalised to all-zero at
            F_freeze == 0, where it is never applied — so a configuration is identified
            only by what affects its behaviour.
    R       release cadence, in sprints.
    F_freeze
            how many sprints at the end of each release cycle use A_qa instead of A_dev.
            Sprints with (t % R) >= (R - F_freeze) are frozen.  Range [0, R-1].

    Only the per-configuration constraints are checked here.  The collective bound on the
    supervised roles needs the scenario parameters and lives in `src.feasibility`.
    """

    N: float
    A_dev: tuple[int, ...]
    A_qa: tuple[int, ...]
    R: int
    F_freeze: int = 1

    def __post_init__(self) -> None:
        if self.N <= 0:
            raise ValueError(f"N must be > 0, got {self.N}")
        if self.R < 1:
            raise ValueError(f"R must be >= 1, got {self.R}")

        if len(self.A_dev) != AGENT_ROLE_COUNT:
            raise ValueError(f"A_dev must have length {AGENT_ROLE_COUNT}, got {len(self.A_dev)}")
        if any(a < 0 for a in self.A_dev):
            raise ValueError(f"A_dev entries must be >= 0, got {self.A_dev}")
        if any(self.A_dev):
            if self.A_dev[0] < 1:
                raise ValueError("a_1 (Feature Implementation Agent) must be >= 1")
        elif any(self.A_qa):
            raise ValueError(
                "A_qa must be all-zero when A_dev is all-zero (human-only baseline)"
            )

        if not (0 <= self.F_freeze <= self.R - 1):
            raise ValueError(
                f"F_freeze must be in [0, R-1={self.R - 1}], got {self.F_freeze}"
            )

        if len(self.A_qa) != AGENT_ROLE_COUNT:
            raise ValueError(
                f"A_qa must have length {AGENT_ROLE_COUNT}, got {len(self.A_qa)}"
            )
        if any(a < 0 for a in self.A_qa):
            raise ValueError(f"A_qa entries must be >= 0, got {self.A_qa}")
        if self.F_freeze > 0:
            if self.A_qa[0] != 0:
                raise ValueError(
                    "a_qa,1 (Feature Implementation Agent) must be 0 during feature freeze"
                )
            if self.A_qa[1] != 0:
                raise ValueError(
                    "a_qa,2 (Architecture & Design Agent) must be 0 during feature freeze"
                )
        else:
            # Never applied without a freeze window, so normalise it: two teams that
            # differ only here must hash and simulate as one.  Normalising after the
            # shape checks keeps a malformed A_qa an error.
            object.__setattr__(self, "A_qa", (0,) * AGENT_ROLE_COUNT)

    def active_A(self, is_frozen: bool) -> tuple[int, ...]:
        """Return A_qa during freeze sprints, A_dev otherwise."""
        return self.A_qa if is_frozen else self.A_dev


def is_human_only(config: TeamConfiguration) -> bool:
    """True for a team with no agents at all.

    A_dev alone decides, because `__post_init__` forces A_qa to zero whenever A_dev is.
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
    n_samples_initial: int = 100
    n_generations: int = 15
    n_offspring: int = 20
    n_elite: int = 20                       # top of the ranking that parents come from
    human_only: bool = False                # enumerate the zero-agent grid, do not search
    samples: str = "best"                   # one of SAMPLE_POLICIES
    seed: int = 0
    scenario_id: str = ""                   # tier combination, e.g. "h-l_p-m_d-h_a-l_b-m"

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
    candidate the search evaluates, `K_final` re-simulates the set it reports.
    """

    T: int                      # sprints per replication
    K_search: int               # replications per candidate during the search
    K_final: int                # replications for the re-simulation of I*
    master_seed: int

    def __post_init__(self) -> None:
        if self.T < 1:
            raise ValueError(f"T must be >= 1, got {self.T}")
        if self.K_search < 1:
            raise ValueError(f"K_search must be >= 1, got {self.K_search}")
        if self.K_final < self.K_search:
            raise ValueError(
                f"K_final must be >= K_search, got {self.K_final} < {self.K_search}"
            )
