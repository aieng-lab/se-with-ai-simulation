"""Consolidate a §4.2 campaign into flat tables.

A campaign is a tree of finished searches, one per (scenario, strategy):

    <results-dir>/<scenario_id>/<strategy>/{manifest,search_summary}.json

`build_tables` reads all of them into two DataFrames that `scripts/collect_results.py`
writes out:

``optima.csv``
    One row per (scenario, strategy): the reported optimum `c*` — its five control
    variables, the objective with its CI, the value it achieved, the three cost drivers
    in dollars and as shares of the total, the per-role work counts and the contribution
    split — plus the size of the indistinguishable set `I*` it is reported with (§3.10).

``scenarios.csv``
    One row per (scenario, strategy) describing the search effort: budget spend,
    feasibility rejections, wall clock, and the Monte Carlo protocol it ran under.

This is deliberately thin.  The §5 results analysis that used to live beside it was built
on the three-objective Pareto frontier the model no longer has, and the paper does not yet
specify its replacement; it will be rebuilt against the single-objective results.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .config import (
    AGENT_ROLE_COUNT,
    RESULT_MODES,
    SCENARIO_GROUPS,
    SEARCH_STRATEGIES,
    TIER_INITIALS,
    UNUSED_TIER,
    UNUSED_TIER_INITIAL,
    TeamConfiguration,
    supervised_agent_count,
)
from .io import load_scenario_params
from .parameters import ScenarioParameters
from .simulation import phase_sprint_counts

_INITIAL_TO_TIER = {v: k for k, v in TIER_INITIALS.items()}
# `X` marks a group the scenario does not use — the human-only baseline's `d-X` / `a-X`.
_INITIAL_TO_TIER[UNUSED_TIER_INITIAL] = UNUSED_TIER


# ----------------------------------------------------------------------------
# Scenario ids and parameters
# ----------------------------------------------------------------------------

def tiers_from_scenario_id(scenario_id: str) -> dict[str, str]:
    """Recover ``{group: tier}`` from a scenario id like ``h-l_p-m_d-h_a-l_b-m``."""
    tokens = scenario_id.split("_")
    if len(tokens) != len(SCENARIO_GROUPS):
        raise ValueError(f"scenario id {scenario_id!r} does not have "
                         f"{len(SCENARIO_GROUPS)} tokens")
    tiers = {}
    for group, token in zip(SCENARIO_GROUPS, tokens):
        initial, sep, tier_initial = token.partition("-")
        if not sep or initial != group[0] or tier_initial not in _INITIAL_TO_TIER:
            raise ValueError(f"unknown token {token!r} in scenario id {scenario_id!r}")
        tiers[group] = _INITIAL_TO_TIER[tier_initial]
    return tiers


@lru_cache(maxsize=None)
def scenario_params_for_id(
    scenario_id: str,
    config_dir: str = "config",
) -> ScenarioParameters:
    """Rebuild the `ScenarioParameters` a scenario id stands for.

    Reads ``<config_dir>/<group>_<tier>.json`` for the five groups.  Cached, so the 432
    scenarios cost at most 432 loads however often they are asked for.
    """
    tiers = tiers_from_scenario_id(scenario_id)
    return load_scenario_params({
        group: Path(config_dir) / f"{group}_{tier}.json"
        for group, tier in tiers.items()
    })


# ----------------------------------------------------------------------------
# Flat tables over the campaign artifacts
# ----------------------------------------------------------------------------

def is_baseline_scenario(scenario_id: str) -> bool:
    """True for a human-only baseline scenario, i.e. one carrying the `X` sentinel.

    The baseline varies only `human`, `project` and `bugimpact`; `debt` and `agent` are
    provably inert for a zero-agent team and are marked `d-X` / `a-X` rather than pinned.
    """
    return UNUSED_TIER in tiers_from_scenario_id(scenario_id).values()


def expected_modes(scenario_id: str) -> tuple[str, ...]:
    """The result modes a scenario is supposed to carry.

    A baseline scenario is enumerated once and has no search counterpart; every other
    scenario is solved by both search strategies (paper §4.2).  `build_tables` reports
    missing results against this, so the 432 campaign scenarios are not counted as missing
    a baseline nor the 36 baseline scenarios as missing two searches.
    """
    return ("human_only",) if is_baseline_scenario(scenario_id) else SEARCH_STRATEGIES


def search_dirs(results_dir: Path) -> list[tuple[str, str, Path]]:
    """Find every (scenario_id, mode, dir) that holds a finished search or enumeration."""
    found = []
    for scenario_dir in sorted(p for p in Path(results_dir).iterdir() if p.is_dir()):
        for mode in RESULT_MODES:
            d = scenario_dir / mode
            if (d / "search_summary.json").is_file():
                found.append((scenario_dir.name, mode, d))
    return found


def team_config(row: "pd.Series | dict") -> TeamConfiguration:
    """Rebuild the TeamConfiguration a table row describes."""
    A_dev = tuple(int(row[f"a_dev_{k + 1}"]) for k in range(AGENT_ROLE_COUNT))
    A_qa = tuple(int(row[f"a_qa_{k + 1}"]) for k in range(AGENT_ROLE_COUNT))
    return TeamConfiguration(N=float(row["N"]), A_dev=A_dev, R=int(row["R"]),
                             F_freeze=int(row["F_freeze"]), A_qa=A_qa)


def optimum_row(
    scenario_id: str,
    strategy: str,
    manifest: dict,
    summary: dict,
    config_dir: str = "config",
) -> dict | None:
    """The `optima.csv` row for one search; None when nothing was simulated.

    `value_ratio` is the achieved throughput as a fraction of the arrival rate, so the
    throughput constraint of Eq. (throughput_constraint) reads `value_ratio >= 1 - eps`
    directly, independently of the scenario's `r_arrival`.

    The three ``cost_*_share`` columns are the drivers of Eq. (cost_rep) as fractions of
    the total; they sum to 1 exactly, since the three means do sum to `cost_mean`.
    """
    best = summary.get("best")
    if best is None:
        return None
    obj = summary["objectives"][best]
    tiers = tiers_from_scenario_id(scenario_id)
    params = scenario_params_for_id(scenario_id, config_dir)
    T = manifest["horizon"]["T"]

    row: dict = {"scenario_id": scenario_id}
    row.update({f"tier_{group}": tiers[group] for group in SCENARIO_GROUPS})
    row.update({
        "strategy": strategy,
        "config_key": best,
        "N": obj["N"], "R": obj["R"], "F_freeze": obj["F_freeze"],
    })
    A_dev, A_qa = obj["A_dev"], obj["A_qa"]
    for k in range(AGENT_ROLE_COUNT):
        row[f"a_dev_{k + 1}"] = A_dev[k]
    for k in range(AGENT_ROLE_COUNT):
        row[f"a_qa_{k + 1}"] = A_qa[k]
    T_dev, T_qa = phase_sprint_counts(team_config(row), T)
    # A configuration with N > 0 always costs something, but a share column is the wrong
    # place to discover otherwise: divide by NaN rather than raising here.
    total = obj["cost_mean"] or float("nan")
    row.update({
        "n_sup_dev": supervised_agent_count(tuple(A_dev)),
        "n_sup_qa": supervised_agent_count(tuple(A_qa)),
        "T_dev": T_dev,
        "T_qa": T_qa,
        "cost_mean": obj["cost_mean"],
        "cost_ci_lo": obj["cost_ci"][0],
        "cost_ci_hi": obj["cost_ci"][1],
        "cost_quality_mean": obj["cost_quality_mean"],
        "cost_human_mean": obj["cost_human_mean"],
        "cost_agent_mean": obj["cost_agent_mean"],
        "cost_quality_share": obj["cost_quality_mean"] / total,
        "cost_human_share": obj["cost_human_mean"] / total,
        "cost_agent_share": obj["cost_agent_mean"] / total,
        "value_mean": obj["value_mean"],
        "value_ci_lo": obj["value_ci"][0],
        "value_ci_hi": obj["value_ci"][1],
        "r_arrival": params.r_arrival,
        "value_ratio": obj["value_mean"] / params.r_arrival,
        "feasible_throughput": obj["feasible_throughput"],
        "throughput_floor": summary["throughput_floor"],
        "n_indistinguishable": len(summary["indistinguishable"]),
        "impl_human_mean": obj["impl_human_mean"],
        "impl_agents_mean": obj["impl_agents_mean"],
        "fixes_human_mean": obj["fixes_human_mean"],
        "fixes_agent_mean": obj["fixes_agent_mean"],
    })
    for k, units in enumerate(obj["actions_mean"], start=1):
        row[f"actions_{k}"] = units
    return row


def scenario_row(
    scenario_id: str,
    strategy: str,
    manifest: dict,
    summary: dict,
) -> dict:
    """Search-effort figures for a single (scenario, strategy) pair."""
    tiers = tiers_from_scenario_id(scenario_id)
    n_evaluated = manifest["n_evaluated"]
    n_simulated = manifest["n_simulated"]
    row: dict = {"scenario_id": scenario_id}
    row.update({f"tier_{group}": tiers[group] for group in SCENARIO_GROUPS})
    row.update({
        "strategy": strategy,
        "n_evaluated": n_evaluated,
        "n_simulated": n_simulated,
        "n_infeasible": n_evaluated - n_simulated,
        "n_indistinguishable": len(summary["indistinguishable"]),
        "best_admissible": manifest.get("best_admissible"),
        "wall_clock_s": manifest.get("wall_clock_s"),
        "T": manifest["horizon"]["T"],
        "K": manifest["horizon"]["K"],
        "master_seed": manifest["horizon"]["master_seed"],
    })
    return row


def build_tables(
    results_dir: str | Path,
    config_dir: str = "config",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read every finished search under `results_dir` into ``(optima, scenarios)``."""
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"{results_dir} is not a directory")
    searches = search_dirs(results_dir)
    if not searches:
        raise FileNotFoundError(f"no finished searches found under {results_dir}")

    o_rows: list[dict] = []
    s_rows: list[dict] = []
    for scenario_id, strategy, d in searches:
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        summary = json.loads((d / "search_summary.json").read_text(encoding="utf-8"))
        row = optimum_row(scenario_id, strategy, manifest, summary, config_dir)
        if row is not None:
            o_rows.append(row)
        s_rows.append(scenario_row(scenario_id, strategy, manifest, summary))
    return pd.DataFrame(o_rows), pd.DataFrame(s_rows)


#: The scenario groups a human-only baseline actually varies.  `debt` and `agent` are
#: excluded because they are inert for a zero-agent team, which is what lets one baseline
#: row serve as the reference for every (debt, agent) combination at the same tiers.
BASELINE_JOIN_GROUPS: tuple[str, ...] = ("human", "project", "bugimpact")


def baseline_uplift(optima: "pd.DataFrame") -> "pd.DataFrame":
    """Cost of each search optimum against the human-only baseline at the same tiers.

    One row per (campaign scenario, search strategy) that has a baseline sharing its
    `human` / `project` / `bugimpact` tiers.  `cost_ratio = cost_mean(baseline) /
    cost_mean(c*)` is the factor by which the agent-supported optimum changes the cost per
    feature: above 1 the agents are cheaper per feature than the human-only team.

    Since the baseline is inert in `debt` and `agent`, each baseline row is the reference
    for every combination of those two — with the full campaign that is 36 baselines
    against 432 scenarios x 2 strategies.  `best_admissible` is carried through from both
    sides, because a ratio against an infeasible team is a different statement.
    """
    if optima.empty or "strategy" not in optima:
        return pd.DataFrame()

    base = optima[optima["strategy"] == "human_only"]
    search = optima[optima["strategy"] != "human_only"]
    if base.empty or search.empty:
        return pd.DataFrame()

    join = [f"tier_{g}" for g in BASELINE_JOIN_GROUPS]
    keep = join + ["scenario_id", "cost_mean", "N", "R", "F_freeze", "value_mean"]
    if "best_admissible" in optima:
        keep.append("best_admissible")

    merged = search.merge(
        base[keep].rename(columns=lambda c: c if c in join else f"baseline_{c}"),
        on=join, how="inner", validate="many_to_one",
    )
    merged["cost_ratio"] = merged["baseline_cost_mean"] / merged["cost_mean"]

    cols = (["scenario_id", "strategy"] + join
            + ["cost_mean", "baseline_cost_mean", "cost_ratio", "baseline_scenario_id",
               "baseline_N", "baseline_R", "baseline_F_freeze", "baseline_value_mean"])
    if "best_admissible" in optima:
        cols += ["best_admissible", "baseline_best_admissible"]
    return merged[[c for c in cols if c in merged]].sort_values(
        ["scenario_id", "strategy"]).reset_index(drop=True)
