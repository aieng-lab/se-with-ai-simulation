"""Consolidate a campaign into flat tables.

A campaign is a tree of finished runs, one per (scenario, result mode):

    <results-dir>/<scenario_id>/<mode>/{manifest,search_summary,convergence}.json

`build_tables` reads all of them into four DataFrames:

``optima.csv``
    One row per (scenario, mode): the reported optimum `c*` — its five control variables,
    the objective with its CI, the value it achieved, the three cost drivers in dollars
    and as shares of the total, the per-role work counts and the contribution split —
    plus the size of the indistinguishable set `I*` it is reported with.

``scenarios.csv``
    One row per (scenario, mode) describing the search effort: budget spend, feasibility
    rejections, wall clock, and the Monte Carlo protocol it ran under.

``convergence.csv``
    One row per (scenario, mode): the elite and incumbent turnover, the size of `I*` at
    each stage of the selection with the counters that explain the shrinkage, and the
    relative CI half-width `rho`.

``elite_turnover.csv``
    Long format — one row per (scenario, mode, generation) — carrying the per-generation
    turnover and whether the incumbent changed.  Long rather than one column per
    generation, since a search that exhausts its budget early runs fewer of them.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .config import (
    AGENT_ROLE_COUNT,
    HUMAN_ONLY_MODE,
    RESULT_MODES,
    SCENARIO_GROUPS,
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

def search_dirs(results_dir: Path) -> list[tuple[str, str, Path]]:
    """Find every (scenario_id, mode, dir) that holds a finished search or enumeration."""
    found = []
    for scenario_dir in sorted(p for p in Path(results_dir).iterdir() if p.is_dir()):
        for mode in RESULT_MODES:
            d = scenario_dir / mode
            if (d / "search_summary.json").is_file():
                found.append((scenario_dir.name, mode, d))
    return found


def require_search_dirs(results_dir: "str | Path") -> list[tuple[str, str, Path]]:
    """`search_dirs`, raising FileNotFoundError on a missing or empty results tree."""
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"{results_dir} is not a directory")
    searches = search_dirs(results_dir)
    if not searches:
        raise FileNotFoundError(f"no finished searches found under {results_dir}")
    return searches


def team_config(row: "pd.Series | dict") -> TeamConfiguration:
    """Rebuild the TeamConfiguration a table row describes."""
    A_dev = tuple(int(row[f"a_dev_{k + 1}"]) for k in range(AGENT_ROLE_COUNT))
    A_qa = tuple(int(row[f"a_qa_{k + 1}"]) for k in range(AGENT_ROLE_COUNT))
    return TeamConfiguration(N=float(row["N"]), A_dev=A_dev, R=int(row["R"]),
                             F_freeze=int(row["F_freeze"]), A_qa=A_qa)


def scenario_columns(scenario_id: str, mode: str) -> dict:
    """The identifying columns every campaign table starts with."""
    tiers = tiers_from_scenario_id(scenario_id)
    row: dict = {"scenario_id": scenario_id}
    row.update({f"tier_{group}": tiers[group] for group in SCENARIO_GROUPS})
    row["mode"] = mode
    return row


def composition_columns(obj: dict) -> dict:
    """The two composition vectors of one objectives entry, unrolled one column per role."""
    row = {f"a_dev_{k + 1}": obj["A_dev"][k] for k in range(AGENT_ROLE_COUNT)}
    row.update({f"a_qa_{k + 1}": obj["A_qa"][k] for k in range(AGENT_ROLE_COUNT)})
    row["n_sup_dev"] = supervised_agent_count(tuple(obj["A_dev"]))
    row["n_sup_qa"] = supervised_agent_count(tuple(obj["A_qa"]))
    return row


def optimum_row(
    scenario_id: str,
    mode: str,
    manifest: dict,
    summary: dict,
    config_dir: str = "config",
) -> dict | None:
    """The `optima.csv` row for one run; None when nothing was simulated.

    `value_ratio` is the achieved throughput as a fraction of the arrival rate, so the
    throughput constraint reads `value_ratio >= 1 - eps` whatever the scenario's
    `r_arrival`.  The three ``cost_*_share`` columns sum to 1 exactly.

    Every number is read from ``final_objectives``, the `K_final` block, and
    `n_indistinguishable` is the final size of `I*`; ``convergence.csv`` has the sizes at
    the earlier stages.
    """
    best = summary.get("best")
    if best is None:
        return None
    obj = summary["final_objectives"][best]
    params = scenario_params_for_id(scenario_id, config_dir)
    T = manifest["horizon"]["T"]

    row = scenario_columns(scenario_id, mode)
    row.update({
        "config_key": best,
        "N": obj["N"], "R": obj["R"], "F_freeze": obj["F_freeze"],
    })
    composition = composition_columns(obj)
    row.update({k: v for k, v in composition.items() if k.startswith("a_")})
    T_dev, T_qa = phase_sprint_counts(team_config(row), T)
    total = obj["cost_mean"]
    row.update({
        "n_sup_dev": composition["n_sup_dev"],
        "n_sup_qa": composition["n_sup_qa"],
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
    mode: str,
    manifest: dict,
    summary: dict,
) -> dict:
    """Search-effort figures for a single (scenario, mode) pair."""
    n_evaluated = manifest["n_evaluated"]
    n_simulated = manifest["n_simulated"]
    row = scenario_columns(scenario_id, mode)
    row.update({
        "n_evaluated": n_evaluated,
        "n_simulated": n_simulated,
        "n_infeasible": n_evaluated - n_simulated,
        "n_indistinguishable": len(summary["indistinguishable"]),
        "best_admissible": manifest.get("best_admissible"),
        "wall_clock_s": manifest.get("wall_clock_s"),
        "T": manifest["horizon"]["T"],
        "K_search": manifest["horizon"]["K_search"],
        "K_final": manifest["horizon"]["K_final"],
        "master_seed": manifest["horizon"]["master_seed"],
    })
    return row


#: The `i_star` counters of `SelectionReport.stats` that become `convergence.csv` columns,
#: in the order the selection stages produce them.  The four set sizes are
#: `n_prelim` -> `n_resimulated` -> `n_after_resim` -> `n_final`.
_I_STAR_COLUMNS: tuple[str, ...] = (
    "n_prelim",
    "parsimony_removed_prelim",
    "n_resimulated",
    "n_discarded_throughput",
    "n_after_resim",
    "n_final",
    "parsimony_removed",
    "best_changed_by_resim",
    "rho_mean",
    "rho_best",
)


def convergence_row(scenario_id: str, mode: str, convergence: dict) -> dict:
    """The `convergence.csv` row for one run: the convergence measures as scalars.

    The `tau_*` columns summarise the curve `turnover_rows` carries in full: its first and
    last value and the mean over the last ten generations, i.e. the level the elite
    turnover settles at.  They are None for an enumeration, which has no generations.
    """
    tau = list(convergence.get("elite_turnover") or [])
    row = scenario_columns(scenario_id, mode)
    row.update({
        "n_generations_run": convergence.get("n_generations_run", len(tau)),
        "n_elite": convergence.get("n_elite"),
        "incumbent_turnover": convergence.get("incumbent_turnover"),
        "tau_first": tau[0] if tau else None,
        "tau_last": tau[-1] if tau else None,
        "tau_mean_last_10": (sum(tau[-10:]) / len(tau[-10:])) if tau else None,
    })
    i_star = convergence.get("i_star") or {}
    row.update({col: i_star.get(col) for col in _I_STAR_COLUMNS})
    row.update({
        "best_admissible": convergence.get("best_admissible"),
        "value_ratio": convergence.get("value_ratio"),
        "K_search": convergence.get("K_search"),
        "K_final": convergence.get("K_final"),
        "wall_clock_s": convergence.get("wall_clock_s"),
        "resimulation_s": convergence.get("resimulation_s"),
    })
    return row


def turnover_rows(scenario_id: str, mode: str, convergence: dict) -> list[dict]:
    """The `elite_turnover.csv` rows for one run: the turnover per generation, long format."""
    tau = convergence.get("elite_turnover") or []
    changed = convergence.get("incumbent_changed") or []
    best_cost = convergence.get("best_cost") or []
    return [
        {
            "scenario_id": scenario_id,
            "mode": mode,
            "generation": g,
            "tau": value,
            "incumbent_changed": changed[g - 1] if g <= len(changed) else None,
            "best_cost": best_cost[g - 1] if g <= len(best_cost) else None,
        }
        for g, value in enumerate(tau, start=1)
    ]


def build_tables(
    results_dir: str | Path,
    config_dir: str = "config",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read every finished run under `results_dir`.

    Returns ``(optima, scenarios, convergence, turnover)``.
    """
    searches = require_search_dirs(results_dir)

    o_rows: list[dict] = []
    s_rows: list[dict] = []
    c_rows: list[dict] = []
    t_rows: list[dict] = []
    for scenario_id, mode, d in searches:
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        summary = json.loads((d / "search_summary.json").read_text(encoding="utf-8"))
        row = optimum_row(scenario_id, mode, manifest, summary, config_dir)
        if row is not None:
            o_rows.append(row)
        s_rows.append(scenario_row(scenario_id, mode, manifest, summary))
        convergence = json.loads((d / "convergence.json").read_text(encoding="utf-8"))
        c_rows.append(convergence_row(scenario_id, mode, convergence))
        t_rows.extend(turnover_rows(scenario_id, mode, convergence))
    return (pd.DataFrame(o_rows), pd.DataFrame(s_rows),
            pd.DataFrame(c_rows), pd.DataFrame(t_rows))


#: The scenario groups a human-only baseline varies; `debt` and `agent` are inert for a
#: zero-agent team, so one baseline row is the reference for every combination of those.
BASELINE_JOIN_GROUPS: tuple[str, ...] = ("human", "project", "bugimpact")


def baseline_uplift(optima: "pd.DataFrame") -> "pd.DataFrame":
    """Cost of each search optimum against the human-only baseline at the same tiers.

    One row per (campaign scenario, search mode) that has a baseline sharing its
    `human` / `project` / `bugimpact` tiers.  `cost_ratio = cost_mean(baseline) /
    cost_mean(c*)`, so above 1 the agent-supported optimum is the cheaper of the two.
    """
    if optima.empty or "mode" not in optima.columns:
        return pd.DataFrame()

    base = optima[optima["mode"] == HUMAN_ONLY_MODE]
    search = optima[optima["mode"] != HUMAN_ONLY_MODE]
    if base.empty or search.empty:
        return pd.DataFrame()

    join = [f"tier_{g}" for g in BASELINE_JOIN_GROUPS]
    keep = join + ["scenario_id", "cost_mean", "N", "R", "F_freeze", "value_mean"]

    merged = search.merge(
        base[keep].rename(columns=lambda c: c if c in join else f"baseline_{c}"),
        on=join, how="inner", validate="many_to_one",
    )
    merged["cost_ratio"] = merged["baseline_cost_mean"] / merged["cost_mean"]

    cols = (["scenario_id", "mode"] + join
            + ["cost_mean", "baseline_cost_mean", "cost_ratio", "baseline_scenario_id",
               "baseline_N", "baseline_R", "baseline_F_freeze", "baseline_value_mean"])
    return merged[cols].sort_values(
        ["scenario_id", "mode"]).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Search convergence summary
# ----------------------------------------------------------------------------

def _median_iqr_max(values: "pd.Series") -> str:
    """Render a column as `median [q1, q3], max = …`.

    Order statistics rather than mean and sd, because these distributions are skewed.
    """
    v = pd.to_numeric(values, errors="coerce").dropna()
    if v.empty:
        return "n/a"
    return (f"{v.median():.3g} [{v.quantile(0.25):.3g}, {v.quantile(0.75):.3g}], "
            f"max = {v.max():.3g}")


def convergence_summary(convergence: "pd.DataFrame") -> str:
    """Render `convergence.csv` as a text block, one section per group of measures."""
    if convergence.empty:
        return "Search convergence: no search carries a convergence record.\n"
    lines = [f"Search convergence over {len(convergence)} search(es)", "=" * 64]

    searched = convergence[convergence["n_generations_run"] > 0]
    lines.append("")
    lines.append("Elite and incumbent turnover")
    if searched.empty:
        lines.append("  no generational search in this campaign (enumeration only)")
    else:
        lines.append(f"  generations run                  {_median_iqr_max(searched['n_generations_run'])}")
        lines.append(f"  tau_1  (first generation)        {_median_iqr_max(searched['tau_first'])}")
        lines.append(f"  tau_G  (last generation)         {_median_iqr_max(searched['tau_last'])}")
        lines.append(f"  tau over the last 10 generations {_median_iqr_max(searched['tau_mean_last_10'])}")
        lines.append(f"  incumbent turnover               {_median_iqr_max(searched['incumbent_turnover'])}")

    lines.append("")
    lines.append("Size of I* at the four stages of the selection")
    lines.append(f"  preliminary, at K'              {_median_iqr_max(convergence['n_prelim'])}")
    lines.append(f"  after the parsimony tie-break   {_median_iqr_max(convergence['n_resimulated'])}")
    lines.append(f"  after the re-simulation at K    {_median_iqr_max(convergence['n_after_resim'])}")
    lines.append(f"  after the second tie-break      {_median_iqr_max(convergence['n_final'])}")

    n = len(convergence)
    changed = int(convergence["best_changed_by_resim"].astype(bool).sum())
    discarded = convergence["n_discarded_throughput"]
    discarded_total = int(discarded.sum())
    discarded_searches = int((discarded > 0).sum())
    removed_prelim = convergence["parsimony_removed_prelim"].sum()
    prelim = convergence["n_prelim"].sum()
    removed = convergence["parsimony_removed"].sum()
    after = convergence["n_after_resim"].sum()
    lines.append("")
    lines.append("What the three reduction steps did")
    # The first tie-break is the one that saves effort: a member removed there is one
    # fewer configuration to re-simulate at K replications.
    if prelim:
        lines.append(f"  parsimony removed at K'         {removed_prelim:.0f}/{prelim:.0f} members "
                     f"({removed_prelim / prelim:.1%} of the preliminary set)")
    else:
        # Every search was inadmissible, so there was no preliminary set to reduce.
        lines.append("  parsimony removed at K'         n/a (no admissible set)")
    lines.append(f"  re-simulation moved c*          {changed}/{n} searches "
                 f"({changed / n:.1%})")
    lines.append(f"  members discarded on throughput {discarded_total} configuration(s) "
                 f"in {discarded_searches}/{n} searches")
    if after:
        lines.append(f"  parsimony removed at K          {removed:.0f}/{after:.0f} members "
                     f"({removed / after:.1%} of the set at K)")
    else:
        lines.append("  parsimony removed at K          n/a (no admissible set)")

    inadmissible = convergence[~convergence["best_admissible"].astype(bool)]
    lines.append("")
    lines.append(f"Inadmissible searches: {len(inadmissible)}/{n}")
    for _, row in inadmissible.sort_values("scenario_id").iterrows():
        ratio = row["value_ratio"]
        ratio_txt = "n/a" if pd.isna(ratio) else f"{ratio:.3f} x r_arrival"
        lines.append(f"  {row['scenario_id']:<24} {row['mode']:<12} value = {ratio_txt}")

    lines.append("")
    lines.append("Relative CI half-width rho after the re-simulation")
    lines.append(f"  mean over the final I*          {_median_iqr_max(convergence['rho_mean'])}")
    lines.append(f"  at c*                           {_median_iqr_max(convergence['rho_best'])}")
    lines.append("")
    return "\n".join(lines)
