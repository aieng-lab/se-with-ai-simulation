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

`cost_multipliers` fits the additive log-cost model on top of ``optima.csv`` and yields
``multipliers.csv``: what each scenario tier multiplies the cost of the optimum by, and
how much of the variance its group explains on its own.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    AGENT_ROLE_COUNT,
    BASELINE_VARYING_GROUPS,
    HUMAN_ONLY_MODE,
    RESULT_MODES,
    SCENARIO_GROUPS,
    SCENARIO_ID_TIERS,
    SEARCH_MODE,
    TeamConfiguration,
    scenario_config_path,
    supervised_agent_count,
    tier_column,
    tiers_present,
)
from .io import load_scenario_params
from .parameters import ScenarioParameters

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
        if not sep or initial != group[0] or tier_initial not in SCENARIO_ID_TIERS:
            raise ValueError(f"unknown token {token!r} in scenario id {scenario_id!r}")
        tiers[group] = SCENARIO_ID_TIERS[tier_initial]
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
        group: scenario_config_path(group, tier, config_dir)
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
    row.update({tier_column(group): tiers[group] for group in SCENARIO_GROUPS})
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

    row = scenario_columns(scenario_id, mode)
    row.update({
        "config_key": best,
        "N": obj["N"], "R": obj["R"], "F_freeze": obj["F_freeze"],
    })
    row.update(composition_columns(obj))
    total = obj["cost_mean"]
    row.update({
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
        row = optimum_row(scenario_id, mode, summary, config_dir)
        if row is not None:
            o_rows.append(row)
        s_rows.append(scenario_row(scenario_id, mode, manifest, summary))
        convergence = json.loads((d / "convergence.json").read_text(encoding="utf-8"))
        c_rows.append(convergence_row(scenario_id, mode, convergence))
        t_rows.extend(turnover_rows(scenario_id, mode, convergence))
    return (pd.DataFrame(o_rows), pd.DataFrame(s_rows),
            pd.DataFrame(c_rows), pd.DataFrame(t_rows))


def search_rows(optima: "pd.DataFrame") -> "pd.DataFrame":
    """The rows of a campaign table that describe a search, not the human-only baseline.

    The single definition of "is a search": every consumer that excludes the baseline
    goes through this, so adding a third result mode is one edit rather than a hunt for
    the sites that spell the predicate out.
    """
    if optima.empty or "mode" not in optima.columns:
        return optima
    return optima[optima["mode"] != HUMAN_ONLY_MODE]


def baseline_uplift(optima: "pd.DataFrame") -> "pd.DataFrame":
    """Cost of each search optimum against the human-only baseline at the same tiers.

    One row per (campaign scenario, search mode) that has a baseline sharing its
    `human` / `project` / `bugimpact` tiers.  `cost_ratio = cost_mean(baseline) /
    cost_mean(c*)`, so above 1 the agent-supported optimum is the cheaper of the two.

    The join is on three groups, not five, because the baseline is inert in `debt` and
    `agent`: one baseline row is the reference for all 12 of their combinations.  All
    five tier columns are carried anyway — the comparison is *read* by agent and by debt
    tier, which is where its strongest interaction lies, and those two are exactly the
    ones the join cannot supply.

    `admissible` is true only where both sides reported an admissible optimum, which is
    the population the comparison is meaningful over: a ratio against a baseline that
    never kept up with the arriving work compares a cost to a shortfall.
    """
    if optima.empty or "mode" not in optima.columns:
        return pd.DataFrame()

    base = optima[optima["mode"] == HUMAN_ONLY_MODE]
    search = search_rows(optima)
    if base.empty or search.empty:
        return pd.DataFrame()

    join = [tier_column(g) for g in BASELINE_VARYING_GROUPS]
    tiers = [tier_column(g) for g in SCENARIO_GROUPS]
    keep = join + ["scenario_id", "cost_mean", "N", "R", "F_freeze", "value_mean",
                   "n_indistinguishable"]

    merged = search.merge(
        base[keep].rename(columns=lambda c: c if c in join else f"baseline_{c}"),
        on=join, how="inner", validate="many_to_one",
    )
    merged["cost_ratio"] = merged["baseline_cost_mean"] / merged["cost_mean"]
    # An inadmissible run has an empty I* (Section 3.10), which is what marks it here.
    merged["admissible"] = (merged["n_indistinguishable"] > 0)
    merged["baseline_admissible"] = (merged["baseline_n_indistinguishable"] > 0)

    cols = (["scenario_id", "mode"] + tiers
            + ["cost_mean", "baseline_cost_mean", "cost_ratio",
               "admissible", "baseline_admissible", "baseline_scenario_id",
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


def _i_star_block(rows: "pd.DataFrame", label: str) -> list[str]:
    """The four `I*` stage sizes and the three reduction steps, for one result mode.

    The stage sizes describe the searches that ended admissible, which is the population
    Section 5.1's table reports: an inadmissible search has an empty preliminary `I*`, so
    including it would put a zero into every stage.  The reduction-step totals below them
    are over *all* of `rows`, since a search that ends inadmissible still spends
    re-simulation effort.
    """
    n = len(rows)
    if not n:
        return [f"  {label}: none in this campaign"]

    admissible = rows[rows["best_admissible"].astype(bool)]
    lines = [f"  {label} ({n} run(s), {len(admissible)} admissible)"]
    if admissible.empty:
        lines.append("    every run ended inadmissible; I* is empty throughout")
    else:
        for caption, column in (("preliminary, at K'", "n_prelim"),
                                ("after the parsimony tie-break", "n_resimulated"),
                                ("after the re-simulation at K", "n_after_resim"),
                                ("after the second tie-break", "n_final")):
            lines.append(f"    {caption:<32}{_median_iqr_max(admissible[column])}")

    removed_prelim = rows["parsimony_removed_prelim"].sum()
    prelim = rows["n_prelim"].sum()
    removed = rows["parsimony_removed"].sum()
    after = rows["n_after_resim"].sum()
    # Counted over the admissible runs: an inadmissible one re-simulates a single
    # candidate, so its `c*` cannot move and would only dilute the rate.
    changed = int(admissible["best_changed_by_resim"].astype(bool).sum())
    discarded = rows["n_discarded_throughput"]
    lines.append("")
    if prelim:
        lines.append(f"    parsimony removed at K'         {removed_prelim:.0f}/{prelim:.0f} "
                     f"members ({removed_prelim / prelim:.1%}), leaving "
                     f"{prelim - removed_prelim:.0f} to re-simulate")
    else:
        lines.append("    parsimony removed at K'         n/a (no admissible set)")
    if len(admissible):
        lines.append(f"    re-simulation moved c*          {changed}/{len(admissible)} "
                     f"admissible runs ({changed / len(admissible):.1%})")
    lines.append(f"    members discarded on throughput {int(discarded.sum())} "
                 f"configuration(s) in {int((discarded > 0).sum())}/{n} runs")
    if after:
        lines.append(f"    parsimony removed at K          {removed:.0f}/{after:.0f} "
                     f"members ({removed / after:.1%})")
    else:
        lines.append("    parsimony removed at K          n/a (no admissible set)")
    lines.append(f"    rho over the final I*           {_median_iqr_max(rows['rho_mean'])}")
    lines.append(f"    rho at c*                       {_median_iqr_max(rows['rho_best'])}")
    return lines


def convergence_summary(convergence: "pd.DataFrame") -> str:
    """Render `convergence.csv` as a text block, one section per group of measures.

    The two result modes are reported **separately**, as Section 5.1 does: a search and an
    exhaustive enumeration differ in what they spend and in how far the selection narrows
    them, so pooling the 36 baselines into the 432 searches misstates both.
    """
    if convergence.empty:
        return "Search convergence: no search carries a convergence record.\n"
    lines = [f"Search convergence over {len(convergence)} run(s)", "=" * 64]

    searched = convergence[convergence["n_generations_run"] > 0]
    lines.append("")
    lines.append("Elite turnover (searches only; an enumeration has no generations)")
    if searched.empty:
        lines.append("  no generational search in this campaign (enumeration only)")
    else:
        lines.append(f"  generations run                  {_median_iqr_max(searched['n_generations_run'])}")
        lines.append(f"  tau_1  (first generation)        {_median_iqr_max(searched['tau_first'])}")
        lines.append(f"  tau_G  (last generation)         {_median_iqr_max(searched['tau_last'])}")
        lines.append(f"  tau over the last 10 generations {_median_iqr_max(searched['tau_mean_last_10'])}")

    lines.append("")
    lines.append("Size of I* at the four stages of the selection, by result mode")
    lines.append("  (stage sizes over the admissible runs; the counters below over all)")
    for mode in RESULT_MODES:
        rows = convergence[convergence["mode"] == mode]
        lines.append("")
        lines.extend(_i_star_block(rows, mode))

    inadmissible = convergence[~convergence["best_admissible"].astype(bool)]
    lines.append("")
    lines.append(f"Inadmissible runs: {len(inadmissible)}/{len(convergence)}")
    for _, row in inadmissible.sort_values("scenario_id").iterrows():
        ratio = row["value_ratio"]
        ratio_txt = "n/a" if pd.isna(ratio) else f"{ratio:.3f} x r_arrival"
        lines.append(f"  {row['scenario_id']:<24} {row['mode']:<12} value = {ratio_txt}")
    lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# The multiplicative effect of each scenario group on the cost
# ----------------------------------------------------------------------------

def admissible_optima(optima: "pd.DataFrame", mode: str = SEARCH_MODE) -> "pd.DataFrame":
    """The rows of `optima` that `mode` reported an admissible `c*` for.

    An inadmissible search has an empty `I*` (Section 3.10), so `n_indistinguishable`
    is the marker; `feasible_throughput` says the same thing and is checked with it.
    """
    if optima.empty:
        return optima
    return optima[(optima["mode"] == mode)
                  & (optima["n_indistinguishable"] > 0)
                  & optima["feasible_throughput"].astype(bool)]


def _tier_design(frame: "pd.DataFrame", groups: "tuple[str, ...]",
                 references: dict) -> "tuple[np.ndarray, list[tuple[str, str]]]":
    """Intercept plus one 0/1 dummy per non-reference tier of each group in `groups`."""
    columns: list[tuple[str, str]] = []
    for group in groups:
        for tier in sorted(frame[tier_column(group)].unique()):
            if tier != references[group]:
                columns.append((group, tier))
    design = np.column_stack(
        [np.ones(len(frame))]
        + [(frame[tier_column(g)] == t).to_numpy(dtype=float) for g, t in columns]
    )
    return design, columns


def _r_squared(design: "np.ndarray", y: "np.ndarray") -> "tuple[np.ndarray, float]":
    """Least-squares fit of `y` on `design`; returns the coefficients and R^2."""
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coef
    ss_res = float(residual @ residual)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return coef, (1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0)


def cost_multipliers(optima: "pd.DataFrame") -> "tuple[pd.DataFrame, dict]":
    """How much each scenario tier multiplies the cost per feature of the optimum.

    An additive model on ``log10(cost_mean)`` with one dummy per tier and no
    interactions, fitted over the admissible search optima.  A coefficient is a
    multiplier on the cost once exponentiated, so the five groups are reported as
    independent multipliers relative to the *cheapest* tier of their own group — which
    is what makes the trends readable one group at a time.

    Returns the per-(group, tier) table and a dict of fit statistics: `r_squared`,
    `residual_sd` (in decades) and one `eta_squared_<group>` per group, the R^2 of the
    single-group model, i.e. the share of the variance of the log cost that the group
    explains on its own.
    """
    frame = admissible_optima(optima)
    if frame.empty:
        return pd.DataFrame(), {}

    y = np.log10(frame["cost_mean"].to_numpy(dtype=float))
    groups = tuple(g for g in SCENARIO_GROUPS if frame[tier_column(g)].nunique() > 1)

    # The reference is the tier with the lowest mean log cost, so every multiplier is
    # >= 1 and the table reads as "how much dearer than the cheapest tier".
    references = {
        g: frame.groupby(tier_column(g))["cost_mean"].apply(lambda c: np.log10(c).mean()).idxmin()
        for g in groups
    }

    design, columns = _tier_design(frame, groups, references)
    coef, r_squared = _r_squared(design, y)
    residual = y - design @ coef
    dof = max(1, len(frame) - design.shape[1])
    stats = {
        "n": len(frame),
        "r_squared": r_squared,
        "residual_sd": float(np.sqrt((residual @ residual) / dof)),
    }

    multiplier = {(g, t): float(10.0 ** c) for (g, t), c in zip(columns, coef[1:])}
    rows = []
    for group in groups:
        _, eta = _r_squared(_tier_design(frame, (group,), references)[0], y)
        stats[f"eta_squared_{group}"] = eta
        for tier in sorted(frame[tier_column(group)].unique()):
            rows.append({
                "group": group,
                "tier": tier,
                "is_reference": tier == references[group],
                "multiplier": 1.0 if tier == references[group]
                              else multiplier[(group, tier)],
                "eta_squared": eta,
                "n": int((frame[tier_column(group)] == tier).sum()),
            })
    return pd.DataFrame(rows), stats


def multipliers_summary(table: "pd.DataFrame", stats: dict) -> str:
    """Render `multipliers.csv` as a text block, groups ordered by the variance they explain."""
    if table.empty:
        return "Cost multipliers: no admissible search optimum to fit.\n"

    tiers = tiers_present(table["tier"])
    width = max(12, *(len(g) for g in table["group"]))
    header = f"  {'group':<{width}}" + "".join(f"{t:>13}" for t in tiers) + f"{'eta^2':>9}"
    lines = [
        "Multiplicative effect of each scenario group on the cost per feature",
        "=" * len(header),
        f"  additive model on log10(cost) over {stats['n']} admissible optima:  "
        f"R^2 = {stats['r_squared']:.3f}, residual sd = {stats['residual_sd']:.2f} decades",
        "",
        header,
        "-" * len(header),
    ]
    order = table.groupby("group")["eta_squared"].first().sort_values(ascending=False)
    for group in order.index:
        sub = table[table["group"] == group].set_index("tier")
        cells = ""
        for tier in tiers:
            if tier not in sub.index:
                cells += f"{'—':>13}"
            else:
                value = f"{sub.loc[tier, 'multiplier']:.1f}"
                cells += f"{value + (' *' if sub.loc[tier, 'is_reference'] else ''):>13}"
        lines.append(f"  {group:<{width}}{cells}{order[group]:>9.2f}")
    lines.append("")
    lines.append("  * the reference tier of its group, i.e. the cheapest one.")
    lines.append("")
    return "\n".join(lines)
