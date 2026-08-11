"""Campaign-level analysis of the Pareto frontiers (paper §5).

Section 5 analyses the campaign of §4.2 along the three Pareto objectives:

1. *Value.*  The claim is that ``value`` sits at the feature arrival rate
   ``r_arrival`` for essentially every point of every frontier, i.e. the optimal
   configurations are arrival-limited rather than throughput-limited.
   :func:`value_saturation_summary` quantifies that per (scenario, strategy) and
   :func:`value_deviation_patterns` characterises the exceptions.

2. *Quality vs. cost.*  Because value is pinned, the candidates can be filtered to
   those within 98 % of ``T * r_arrival`` and a fresh two-objective Pareto front
   computed over the remainder — the *throughput-filtered quality/cost front*.
   :func:`build_campaign` produces it, :func:`quality_cost_frontier` is the
   dominance pass, :func:`add_derived_columns` splits the scalar ``cost_mean``
   into its human and agent shares, and :func:`rank_frontier` /
   :func:`frontier_extremes` rank the surviving points.

The cost split is read from the simulation rather than reconstructed. Agents are
billed for the work they actually performed (:func:`src.simulation.cost_components`),
so ``C_agents`` depends on the realised trajectory and no fixed ``C_human / C_total``
ratio can recover the split from ``cost_mean``. ``cost_human_mean`` and
``cost_agent_mean`` are therefore persisted per candidate. Campaign files written
before the usage-based cost model lack those columns; :func:`add_derived_columns`
falls back to the old deterministic reconstruction for them and warns.

This module also owns the flat-table schema that ``scripts/collect_results.py``
and ``scripts/analyze_results.py`` share, so the two tools cannot drift apart.

Public API:
    tiers_from_scenario_id(scenario_id)            -> dict[str, str]
    scenario_params_for_id(scenario_id, config_dir)-> ScenarioParameters
    search_dirs(results_dir)                       -> list[(scenario_id, strategy, Path)]
    frontier_rows(scenario_id, strategy, manifest, summary)  -> list[dict]
    candidate_rows(scenario_id, strategy, manifest, summary) -> list[dict]
    scenario_row(scenario_id, strategy, manifest, summary)   -> dict
    build_tables(results_dir)                      -> (frontiers_df, scenarios_df)
    build_campaign(results_dir, ...)               -> CampaignTables
    load_frontier_table(...)                       -> pd.DataFrame
    quality_cost_frontier(rows, ci_aware)          -> list[int]
    filtered_front_from_table(df, threshold)       -> pd.DataFrame
    add_derived_columns(df, config_dir)            -> pd.DataFrame
    value_saturation_summary(df, threshold)        -> pd.DataFrame
    value_deviation_patterns(df, threshold)        -> pd.DataFrame
    rank_frontier(df)                              -> pd.DataFrame
    add_quality_cost_index(df, cost_field)         -> pd.DataFrame
    add_front_position(df, cost_field)             -> pd.DataFrame
    frontier_extremes(df, k)                       -> pd.DataFrame
    index_optimum(df, cost_field, by)              -> pd.DataFrame
    headline_summary(df, threshold)                -> dict
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    AGENT_ROLE_COUNT,
    SCENARIO_GROUPS,
    SCENARIO_TIERS,
    TeamConfiguration,
    supervised_agent_count,
)
from .io import load_scenario_params
from .parameters import ScenarioParameters
from .pareto import AggregatedObjectives, ParsimonyKey, pareto_frontier
from .simulation import phase_sprint_counts

STRATEGIES: tuple[str, ...] = ("random", "evolutionary")

#: ``value_ratio`` below this counts as a deviation from the arrival rate (§5,
#: bullet 1 — classifying the exceptions).
DEFAULT_VALUE_THRESHOLD = 0.99

#: ``value_ratio`` at or above this keeps a candidate in the throughput-filtered
#: pool (§5, bullet 2 — "within 98 % of T * r_arrival").  Deliberately distinct
#: from DEFAULT_VALUE_THRESHOLD: one classifies, the other selects.
DEFAULT_THROUGHPUT_THRESHOLD = 0.98

#: Cost terms the quality/cost ratio of :func:`ratio_percentiles` can be taken
#: against — total, human developers only, agents only.
RATIO_COST_FIELDS: tuple[str, ...] = ("cost_mean", "cost_human_mean", "cost_agent_mean")

_TIER_INITIALS = {"low": "l", "medium": "m", "high": "h"}
_INITIAL_TO_TIER = {v: k for k, v in _TIER_INITIALS.items()}

#: Control variables the deviation-pattern breakdown groups by, alongside the
#: five tier columns.
CONTROL_FACTORS: tuple[str, ...] = ("N", "R", "F_freeze", "a_dev_1", "n_sup_dev")


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

    Reads ``<config_dir>/<group>_<tier>.json`` for the five groups.  Cached, so
    the 243 scenarios cost at most 243 loads regardless of frontier size.
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
    """Find every (scenario_id, strategy, dir) that holds a finished search."""
    found = []
    for scenario_dir in sorted(p for p in Path(results_dir).iterdir() if p.is_dir()):
        for strategy in STRATEGIES:
            d = scenario_dir / strategy
            if (d / "search_summary.json").is_file():
                found.append((scenario_dir.name, strategy, d))
    return found


def _objective_row(
    scenario_id: str,
    strategy: str,
    key: str,
    obj: dict,
    manifest: dict,
    tiers: dict[str, str],
) -> dict:
    """The flat-table row for one entry of ``search_summary.json["objectives"]``."""
    row: dict = {"scenario_id": scenario_id}
    row.update({f"tier_{group}": tiers[group] for group in SCENARIO_GROUPS})
    row.update({
        "strategy": strategy,
        "config_key": key,
        "N": obj["N"], "R": obj["R"],
        # F_freeze and A_qa were added to search_summary.json alongside the campaign;
        # fall back to the manifest for searches written before that.
        "F_freeze": obj.get("F_freeze", manifest["configs"][key]["F_freeze"]),
    })
    A_dev = obj["A_dev"]
    A_qa = obj.get("A_qa", manifest["configs"][key]["A_qa"])
    for k in range(AGENT_ROLE_COUNT):
        row[f"a_dev_{k + 1}"] = A_dev[k]
    for k in range(AGENT_ROLE_COUNT):
        row[f"a_qa_{k + 1}"] = A_qa[k]
    for name in ("value", "quality", "cost"):
        row[f"{name}_mean"] = obj[f"{name}_mean"]
        row[f"{name}_ci_lo"] = obj[f"{name}_ci"][0]
        row[f"{name}_ci_hi"] = obj[f"{name}_ci"][1]
    # The recorded human/agent cost split and the per-role work counts postdate the
    # usage-based cost model; searches written before it carry neither, and
    # add_derived_columns falls back to the deterministic reconstruction for them.
    for name in ("cost_human_mean", "cost_agent_mean"):
        if name in obj:
            row[name] = obj[name]
    for k, units in enumerate(obj.get("actions_mean", ()), start=1):
        row[f"actions_{k}"] = units
    return row


def frontier_rows(
    scenario_id: str,
    strategy: str,
    manifest: dict,
    summary: dict,
) -> list[dict]:
    """One row per Pareto-optimal configuration of a single search."""
    tiers = tiers_from_scenario_id(scenario_id)
    return [
        _objective_row(scenario_id, strategy, key, summary["objectives"][key], manifest, tiers)
        for key in summary["pareto_frontier"]
    ]


def candidate_rows(
    scenario_id: str,
    strategy: str,
    manifest: dict,
    summary: dict,
) -> list[dict]:
    """One row per *simulated* candidate of a single search — the full pool.

    Same schema as :func:`frontier_rows` plus ``on_3d_frontier``, which flags the
    candidates the search recorded on its three-objective Pareto frontier.  The
    throughput-filtered quality/cost front of §5 is computed over this pool rather
    than over the recorded frontier: dropping the value objective can revive a
    candidate that was eliminated in three dimensions on a *value* significance.
    """
    tiers = tiers_from_scenario_id(scenario_id)
    on_frontier = set(summary["pareto_frontier"])
    rows = []
    for key, obj in summary["objectives"].items():
        row = _objective_row(scenario_id, strategy, key, obj, manifest, tiers)
        row["on_3d_frontier"] = key in on_frontier
        rows.append(row)
    return rows


def scenario_row(
    scenario_id: str,
    strategy: str,
    manifest: dict,
    summary: dict,
) -> dict:
    """Search-effort figures for a single (scenario, strategy) pair."""
    tiers = tiers_from_scenario_id(scenario_id)
    n_evaluated = manifest["n_evaluated"]
    n_simulated = manifest.get("n_simulated", manifest.get("n_feasible", 0))
    row: dict = {"scenario_id": scenario_id}
    row.update({f"tier_{group}": tiers[group] for group in SCENARIO_GROUPS})
    row.update({
        "strategy": strategy,
        "n_evaluated": n_evaluated,
        "n_simulated": n_simulated,
        "n_infeasible": n_evaluated - n_simulated,
        "n_frontier": len(summary["pareto_frontier"]),
        "wall_clock_s": manifest.get("wall_clock_s"),
        "T": manifest["horizon"]["T"],
        "K": manifest["horizon"]["K"],
        "master_seed": manifest["horizon"]["master_seed"],
    })
    return row


def build_tables(results_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read every finished search under `results_dir` into the two flat tables.

    Returns ``(frontiers, scenarios)`` with the schemas documented in
    ``scripts/collect_results.py``.
    """
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"{results_dir} is not a directory")
    searches = search_dirs(results_dir)
    if not searches:
        raise FileNotFoundError(f"no finished searches found under {results_dir}")

    f_rows: list[dict] = []
    s_rows: list[dict] = []
    for scenario_id, strategy, d in searches:
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        summary = json.loads((d / "search_summary.json").read_text(encoding="utf-8"))
        f_rows.extend(frontier_rows(scenario_id, strategy, manifest, summary))
        s_rows.append(scenario_row(scenario_id, strategy, manifest, summary))
    return pd.DataFrame(f_rows), pd.DataFrame(s_rows)


@dataclass(frozen=True)
class CampaignTables:
    """Everything one pass over a campaign produces.

    frontiers      the recorded three-objective Pareto frontiers (§5 bullet 1)
    scenarios      per-search effort figures
    filtered_front the throughput-filtered quality/cost front (§5 bullet 2)
    filter_summary per-search filter and front sizes
    """

    frontiers: pd.DataFrame
    scenarios: pd.DataFrame
    filtered_front: pd.DataFrame
    filter_summary: pd.DataFrame


def build_campaign(
    results_dir: str | Path,
    throughput_threshold: float = DEFAULT_THROUGHPUT_THRESHOLD,
    config_dir: str = "config",
    ci_aware: bool = True,
    progress: bool = False,
) -> CampaignTables:
    """Read a campaign once and produce every §5 table.

    Per search the two JSON artifacts are parsed a single time, then:

    1. the recorded three-objective frontier rows are kept for bullet 1;
    2. *all* simulated candidates are filtered to
       ``value_mean >= throughput_threshold * r_arrival``;
    3. the survivors get a two-objective (quality, cost) Pareto pass;
    4. only the resulting front rows are retained and enriched.

    Step 4 is deliberately last: :func:`add_derived_columns` builds a
    `TeamConfiguration` and calls `cost_components` per row, which is far too slow
    to run over the ~10,000 candidates of a §4.2 search but trivial over the ~50
    that survive.  Memory stays bounded for the same reason.
    """
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"{results_dir} is not a directory")
    searches = search_dirs(results_dir)
    if not searches:
        raise FileNotFoundError(f"no finished searches found under {results_dir}")

    f_rows: list[dict] = []
    s_rows: list[dict] = []
    front_frames: list[pd.DataFrame] = []
    filter_rows: list[dict] = []

    for n, (scenario_id, strategy, d) in enumerate(searches, start=1):
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        summary = json.loads((d / "search_summary.json").read_text(encoding="utf-8"))
        tiers = tiers_from_scenario_id(scenario_id)
        params = scenario_params_for_id(scenario_id, config_dir)
        T = manifest["horizon"]["T"]

        f_rows.extend(frontier_rows(scenario_id, strategy, manifest, summary))
        s_rows.append(scenario_row(scenario_id, strategy, manifest, summary))

        candidates = candidate_rows(scenario_id, strategy, manifest, summary)
        cutoff = throughput_threshold * params.r_arrival
        passing = [r for r in candidates if r["value_mean"] >= cutoff]

        front: list[dict] = []
        if passing:
            front = [passing[i] for i in quality_cost_frontier(passing, ci_aware=ci_aware)]
            front_frames.append(add_derived_columns(
                pd.DataFrame(front).assign(T=T), config_dir=config_dir,
            ))

        row: dict = {"scenario_id": scenario_id}
        row.update({f"tier_{group}": tiers[group] for group in SCENARIO_GROUPS})
        row.update({
            "strategy": strategy,
            "r_arrival": params.r_arrival,
            "throughput_threshold": throughput_threshold,
            "n_simulated": len(candidates),
            "n_passed": len(passing),
            "frac_passed": len(passing) / len(candidates) if candidates else 0.0,
            "n_filtered_front": len(front),
            "n_recorded_3d_front": len(summary["pareto_frontier"]),
        })
        filter_rows.append(row)

        if progress:
            print(f"  [{n}/{len(searches)}] {scenario_id} / {strategy}: "
                  f"{len(passing)}/{len(candidates)} passed → {len(front)} on front")

    frontiers = pd.DataFrame(f_rows)
    scenarios = pd.DataFrame(s_rows)
    if "T" not in frontiers.columns and not scenarios.empty:
        frontiers = frontiers.merge(
            scenarios[["scenario_id", "strategy", "T"]],
            on=["scenario_id", "strategy"], how="left",
        )
    filtered_front = (
        pd.concat(front_frames).reset_index(drop=True) if front_frames else pd.DataFrame()
    )
    return CampaignTables(
        frontiers=frontiers,
        scenarios=scenarios,
        filtered_front=filtered_front,
        filter_summary=pd.DataFrame(filter_rows),
    )


def load_frontier_table(
    results_dir: str | Path | None = None,
    frontiers_csv: str | Path | None = None,
    scenarios_csv: str | Path | None = None,
    default_T: int | None = None,
) -> pd.DataFrame:
    """Load the frontier table, with the horizon `T` joined on.

    Either point at a campaign directory (`results_dir`, rebuilt from the raw
    JSON) or at an existing ``frontiers.csv`` (`frontiers_csv`, the fast path).
    When reading the CSV, `T` is taken from the sibling ``scenarios.csv`` unless
    `scenarios_csv` or `default_T` says otherwise.
    """
    if (results_dir is None) == (frontiers_csv is None):
        raise ValueError("pass exactly one of results_dir or frontiers_csv")

    if results_dir is not None:
        frontiers, scenarios = build_tables(results_dir)
    else:
        frontiers = pd.read_csv(frontiers_csv)
        path = Path(scenarios_csv) if scenarios_csv else Path(frontiers_csv).with_name("scenarios.csv")
        scenarios = pd.read_csv(path) if path.is_file() else pd.DataFrame()

    if "T" in frontiers.columns:
        return frontiers
    if not scenarios.empty and "T" in scenarios.columns:
        return frontiers.merge(
            scenarios[["scenario_id", "strategy", "T"]],
            on=["scenario_id", "strategy"],
            how="left",
        )
    if default_T is None:
        raise ValueError(
            "horizon T is unavailable: no scenarios.csv alongside the frontier "
            "table and no default_T given"
        )
    return frontiers.assign(T=default_T)


# ----------------------------------------------------------------------------
# §5 — the throughput-filtered quality/cost Pareto front
# ----------------------------------------------------------------------------

def _flatten_value(rows: "list[dict] | pd.DataFrame") -> list[AggregatedObjectives]:
    """Objectives with the value dimension held constant, for a 2-D Pareto pass.

    With `value_mean` and `value_ci` identical across every point,
    `pareto.dominates_strict` reduces exactly to two-objective dominance (the
    value comparison is always a non-strict tie) and `_ci_significantly_better`
    can only fire on quality or cost.  So the §3.10 CI-aware rule applies
    unchanged in two dimensions and there is no second dominance implementation
    to keep in sync with `src/pareto.py`.
    """
    if isinstance(rows, pd.DataFrame):
        rows = rows.to_dict("records")
    return [
        AggregatedObjectives(
            value_mean=0.0,
            quality_mean=float(r["quality_mean"]),
            cost_mean=float(r["cost_mean"]),
            value_ci=(0.0, 0.0),
            quality_ci=(float(r["quality_ci_lo"]), float(r["quality_ci_hi"])),
            cost_ci=(float(r["cost_ci_lo"]), float(r["cost_ci_hi"])),
        )
        for r in rows
    ]


def _parsimony_keys(rows: "list[dict] | pd.DataFrame") -> list[ParsimonyKey]:
    """ParsimonyKeys for flat-table rows, for the §3.10 tie-break.

    Built straight from the columns rather than via :func:`_team_config`, which would
    construct and validate a TeamConfiguration per row for no benefit here.
    """
    if isinstance(rows, pd.DataFrame):
        rows = rows.to_dict("records")
    return [
        ParsimonyKey(
            process=(float(r["N"]), int(r["R"]), int(r["F_freeze"])),
            agents=tuple(int(r[f"a_dev_{k + 1}"]) for k in range(AGENT_ROLE_COUNT))
            + tuple(int(r[f"a_qa_{k + 1}"]) for k in range(AGENT_ROLE_COUNT)),
        )
        for r in rows
    ]


def quality_cost_frontier(
    rows: "list[dict] | pd.DataFrame",
    ci_aware: bool = True,
    parsimony: bool = False,
) -> list[int]:
    """Positional indices of the (quality, cost) Pareto front, ignoring value.

    Both objectives are minimised.  Delegates to the tested
    `src.pareto.pareto_frontier` with the value dimension flattened — see
    :func:`_flatten_value`.

    `parsimony=True` additionally applies the §3.10 tie-break, so among statistically
    indistinguishable teams running the same process only the smallest is kept.  It is
    opt-in rather than the default because it needs the 19 configuration columns
    (`N`, `R`, `F_freeze`, `a_dev_*`, `a_qa_*`) on top of the six objective columns this
    function otherwise reads; `filtered_front_from_table` always has them and turns it on.
    The flattened value dimension is inert for the tie-break too: with value_ci = (0, 0)
    everywhere no pair can be value-significant, so ties are decided on quality and cost
    alone — exactly the two-objective question this front asks.
    """
    keys = _parsimony_keys(rows) if parsimony else None
    return pareto_frontier(_flatten_value(rows), ci_aware=ci_aware, keys=keys)


def filtered_front_from_table(
    df: pd.DataFrame,
    threshold: float = DEFAULT_THROUGHPUT_THRESHOLD,
    ci_aware: bool = True,
    parsimony: bool = True,
) -> pd.DataFrame:
    """Throughput-filter a frontier table, then take its quality/cost front.

    Filters to ``value_ratio >= threshold`` — i.e. the paper's
    ``F_completed >= threshold * T * r_arrival`` — and computes the two-objective
    front per (scenario, strategy).  `df` must already carry `value_ratio`, so run
    :func:`add_derived_columns` first.

    `parsimony=True` (the default) applies the §3.10 tie-break within each front, so a
    reported point is the smallest team achieving its trade-off rather than one of several
    statistically indistinguishable variants of it.  This is the §5 path and the flat
    table always carries the configuration columns it needs.

    Note this operates on whatever pool `df` holds.  Given only the recorded
    three-objective frontier it is correct for strict dominance but can miss a
    point the CI-aware rule would retain; :func:`build_campaign` uses the full
    candidate pool instead.
    """
    if df.empty:
        return df.copy()
    if "value_ratio" not in df.columns:
        raise KeyError("expected a 'value_ratio' column; call add_derived_columns() first")

    pieces = []
    for _, sub in df.groupby(["scenario_id", "strategy"], sort=True):
        passing = sub[sub["value_ratio"] >= threshold]
        if passing.empty:
            continue
        idx = quality_cost_frontier(passing, ci_aware=ci_aware, parsimony=parsimony)
        pieces.append(passing.iloc[idx])
    if not pieces:
        return df.iloc[0:0].copy()
    return pd.concat(pieces).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Derived columns
# ----------------------------------------------------------------------------

def _team_config(row: pd.Series) -> TeamConfiguration:
    return TeamConfiguration(
        N=float(row["N"]),
        A_dev=tuple(int(row[f"a_dev_{k + 1}"]) for k in range(AGENT_ROLE_COUNT)),
        R=int(row["R"]),
        F_freeze=int(row["F_freeze"]),
        A_qa=tuple(int(row[f"a_qa_{k + 1}"]) for k in range(AGENT_ROLE_COUNT)),
    )


def _legacy_cost_components(
    config: TeamConfiguration, params: ScenarioParameters, T: int
) -> tuple[float, float]:
    """(C_human, C_agents) under the pre-usage-based cost model.

    Charged every deployed agent its *nominal* throughput every sprint, regardless of
    the work that reached it.  Kept only so that campaign tables written before the
    switch still load; new results carry the recorded split instead.
    """
    C_human = config.N * params.c_salary * T / 52
    dev_sprint_count, qa_sprint_count = phase_sprint_counts(config, T)
    agent_rates = (
        params.r_F, params.r_arch_chg, params.r_fix, params.r_refactor,
        1.0, 1.0, params.r_test, params.r_doc,
    )
    C_agents = (
        sum(config.A_dev[r] * params.c_token[r] * agent_rates[r]
            for r in range(AGENT_ROLE_COUNT)) * dev_sprint_count
        + sum(config.A_qa[r] * params.c_token[r] * agent_rates[r]  # type: ignore[index]
              for r in range(AGENT_ROLE_COUNT)) * qa_sprint_count
    )
    return C_human, C_agents


def add_derived_columns(df: pd.DataFrame, config_dir: str = "config") -> pd.DataFrame:
    """Add the §5 quantities to a frontier table.

    Adds, per row:
        r_arrival, value_ratio (+ CI bounds), value_deficit
        T_dev, T_qa, C_human, C_agents, C_total, agent_cost_share
        cost_human_mean / cost_agent_mean (+ CI bounds)
        quality_cost_ratio, quality_cost_index, n_sup_dev, n_sup_qa
    """
    if df.empty:
        return df.copy()
    if "T" not in df.columns:
        raise KeyError("frontier table needs a 'T' column; use load_frontier_table()")

    out = df.copy()
    # The simulator persists the split; only pre-usage-model campaigns lack it.
    has_persisted_split = {"cost_human_mean", "cost_agent_mean"} <= set(out.columns)

    records = []
    for _, row in out.iterrows():
        params = scenario_params_for_id(row["scenario_id"], config_dir)
        config = _team_config(row)
        T = int(row["T"])
        T_dev, T_qa = phase_sprint_counts(config, T)
        records.append({
            "r_arrival": params.r_arrival,
            "T_dev": T_dev,
            "T_qa": T_qa,
            "n_sup_dev": supervised_agent_count(config.A_dev),
            "n_sup_qa": supervised_agent_count(config.A_qa),  # type: ignore[arg-type]
        })
    derived = pd.DataFrame(records, index=out.index)
    out = pd.concat([out, derived], axis=1)

    if has_persisted_split:
        # cost_human_mean and cost_agent_mean already carry the max(1, F_completed)
        # denominator, so the per-horizon totals come back by multiplying it out via
        # the ratio to cost_mean, which shares that denominator exactly.
        cost_mean = out["cost_mean"].replace(0.0, np.nan)
        agent_share = out["cost_agent_mean"] / cost_mean
        human_share = out["cost_human_mean"] / cost_mean
        out["agent_cost_share"] = agent_share
        # The CI bounds are not split by the simulator; apportion them by the mean
        # share, which is the best available estimate now that the split is stochastic.
        for suffix in ("ci_lo", "ci_hi"):
            out[f"cost_human_{suffix}"] = out[f"cost_{suffix}"] * human_share
            out[f"cost_agent_{suffix}"] = out[f"cost_{suffix}"] * agent_share
    else:
        warnings.warn(
            "frontier table has no cost_human_mean/cost_agent_mean columns; falling "
            "back to the deterministic cost split of the pre-usage-based cost model. "
            "Re-run the campaign to get the recorded split.",
            RuntimeWarning, stacklevel=2,
        )
        legacy = []
        for _, row in out.iterrows():
            params = scenario_params_for_id(row["scenario_id"], config_dir)
            config = _team_config(row)
            legacy.append(_legacy_cost_components(config, params, int(row["T"])))
        out["C_human"] = [c[0] for c in legacy]
        out["C_agents"] = [c[1] for c in legacy]
        total = (out["C_human"] + out["C_agents"]).replace(0.0, np.nan)
        human_share = out["C_human"] / total
        agent_share = out["C_agents"] / total
        out["agent_cost_share"] = agent_share
        for suffix in ("mean", "ci_lo", "ci_hi"):
            out[f"cost_human_{suffix}"] = out[f"cost_{suffix}"] * human_share
            out[f"cost_agent_{suffix}"] = out[f"cost_{suffix}"] * agent_share

    if "C_human" not in out.columns:
        # Per-horizon totals, recovered from the per-feature means.
        features = out["value_mean"] * out["T"]
        out["C_human"] = out["cost_human_mean"] * features.clip(lower=1.0)
        out["C_agents"] = out["cost_agent_mean"] * features.clip(lower=1.0)
    out["C_total"] = out["C_human"] + out["C_agents"]

    out["value_ratio"] = out["value_mean"] / out["r_arrival"]
    out["value_ratio_ci_lo"] = out["value_ci_lo"] / out["r_arrival"]
    out["value_ratio_ci_hi"] = out["value_ci_hi"] / out["r_arrival"]
    out["value_deficit"] = 1.0 - out["value_ratio"]

    # Two scalar views of the quality/cost pair: the ratio fixes the position along
    # the front, the product ranks points by goodness.  See rank_frontier and
    # add_quality_cost_index.
    out["quality_cost_ratio"] = out["quality_mean"] / out["cost_mean"].replace(0.0, np.nan)
    out["quality_cost_index"] = out["quality_mean"] * out["cost_mean"]
    return out


# ----------------------------------------------------------------------------
# §5 bullet 1 — value sits at the arrival rate
# ----------------------------------------------------------------------------

def _throughput_limited(df: pd.DataFrame, threshold: float) -> pd.Series:
    """CI-aware flag: the whole value CI lies below `threshold * r_arrival`.

    Using the CI upper bound rather than the mean keeps Monte Carlo noise from
    being reported as a genuine throughput shortfall.
    """
    return df["value_ratio_ci_hi"] < threshold


def value_saturation_summary(
    df: pd.DataFrame,
    threshold: float = DEFAULT_VALUE_THRESHOLD,
) -> pd.DataFrame:
    """Per (scenario, strategy): how close the frontier sits to `r_arrival`.

    Columns: the five tier columns, `r_arrival`, `n_frontier`, min/median/max
    `value_ratio`, `frac_below_threshold` (point estimate) and
    `n_throughput_limited` / `frac_throughput_limited` (CI-aware).
    """
    if df.empty:
        return df.copy()
    work = df.copy()
    work["_below"] = work["value_ratio"] < threshold
    work["_tl"] = _throughput_limited(work, threshold)

    tier_cols = [f"tier_{g}" for g in SCENARIO_GROUPS]
    grouped = work.groupby(["scenario_id", "strategy"], sort=True)
    summary = grouped.agg(
        **{c: (c, "first") for c in tier_cols},
        r_arrival=("r_arrival", "first"),
        n_frontier=("value_ratio", "size"),
        value_ratio_min=("value_ratio", "min"),
        value_ratio_median=("value_ratio", "median"),
        value_ratio_max=("value_ratio", "max"),
        frac_below_threshold=("_below", "mean"),
        n_throughput_limited=("_tl", "sum"),
        frac_throughput_limited=("_tl", "mean"),
    ).reset_index()
    return summary


def value_deviation_patterns(
    df: pd.DataFrame,
    threshold: float = DEFAULT_VALUE_THRESHOLD,
) -> pd.DataFrame:
    """Which factors explain the frontier points that miss the arrival rate.

    Returns a tidy table with one row per (factor, level):
        n_points, n_throughput_limited, frac_throughput_limited,
        median_deficit, max_deficit, spearman_rho

    Factors are the five scenario tiers (levels low/medium/high, ordered) and the
    control variables in `CONTROL_FACTORS`.  `spearman_rho` is the rank
    correlation between `value_deficit` and the factor across the *whole* table
    and is therefore repeated on every level of a numeric factor; it is NaN for
    the categorical tier factors, which use the ordered tier index instead.
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "factor", "level", "n_points", "n_throughput_limited",
            "frac_throughput_limited", "median_deficit", "max_deficit", "spearman_rho",
        ])

    work = df.copy()
    work["_tl"] = _throughput_limited(work, threshold)

    tier_factors = [f"tier_{g}" for g in SCENARIO_GROUPS]
    factors = tier_factors + [c for c in CONTROL_FACTORS if c in work.columns]

    rows: list[dict] = []
    for factor in factors:
        if factor in tier_factors:
            # Ordered tier index makes the rank correlation meaningful.
            numeric = work[factor].map({t: i for i, t in enumerate(SCENARIO_TIERS)})
        else:
            numeric = pd.to_numeric(work[factor], errors="coerce")
        rho = _spearman(numeric, work["value_deficit"])

        for level, sub in work.groupby(factor, sort=True, observed=True):
            rows.append({
                "factor": factor,
                "level": level,
                "n_points": len(sub),
                "n_throughput_limited": int(sub["_tl"].sum()),
                "frac_throughput_limited": float(sub["_tl"].mean()),
                "median_deficit": float(sub["value_deficit"].median()),
                "max_deficit": float(sub["value_deficit"].max()),
                "spearman_rho": rho,
            })
    return pd.DataFrame(rows)


def _spearman(x: pd.Series, y: pd.Series) -> float:
    """Spearman rank correlation, NaN when either side is constant or too short."""
    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return float("nan")
    xr = x[mask].rank()
    yr = y[mask].rank()
    if xr.nunique() < 2 or yr.nunique() < 2:
        return float("nan")
    return float(xr.corr(yr))


# ----------------------------------------------------------------------------
# §5 bullet 2 — quality/cost ratio and ranking
# ----------------------------------------------------------------------------

def rank_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """Rank each frontier by `quality_cost_ratio`, ascending.

    `quality_cost_ratio` is an *exchange rate* between the two objectives, not a
    score: both are minimised, so the ratio falls when quality improves but rises
    when cost improves.  What it does fix is the **position along the front** — on
    the §4.2 campaign it correlates +1.00 with `quality_mean` and −0.97 with
    `cost_mean` — which is what this ranking and :func:`frontier_extremes` need.
    The scalar that ranks points by *goodness* is `quality_cost_index`
    (:func:`add_quality_cost_index`), and it is deliberately not used here: it is
    U-shaped along the front and would fold the two ends onto each other.

    A low ratio is the quality-dominant end of the front, a high ratio the
    cost-dominant end.  Adds `ratio_rank` (1 = lowest ratio), `ratio_pct` in [0, 1], and the
    per-objective ranks `quality_rank` / `cost_rank` (both ascending, 1 = best)
    and `value_rank` (descending, 1 = best).  Rows are returned sorted by
    (scenario_id, strategy, ratio_rank).
    """
    if df.empty:
        return df.copy()
    out = df.copy()
    grouped = out.groupby(["scenario_id", "strategy"], sort=False)
    out["ratio_rank"] = grouped["quality_cost_ratio"].rank(method="min").astype("Int64")
    out["ratio_pct"] = grouped["quality_cost_ratio"].rank(pct=True)
    out["quality_rank"] = grouped["quality_mean"].rank(method="min").astype("Int64")
    out["cost_rank"] = grouped["cost_mean"].rank(method="min").astype("Int64")
    out["value_rank"] = grouped["value_mean"].rank(method="min", ascending=False).astype("Int64")
    return out.sort_values(["scenario_id", "strategy", "ratio_rank"]).reset_index(drop=True)


def _check_cost_field(df: pd.DataFrame, cost_field: str) -> None:
    """Guard the `cost_field` argument shared by the two scalar helpers."""
    if cost_field not in RATIO_COST_FIELDS:
        raise KeyError(f"{cost_field!r} is not one of {list(RATIO_COST_FIELDS)}")
    if not df.empty and cost_field not in df.columns:
        raise KeyError(f"{cost_field!r} is not a column of the frontier table")


def add_quality_cost_index(
    df: pd.DataFrame,
    cost_field: str = "cost_mean",
    column: str = "quality_cost_index",
) -> pd.DataFrame:
    """Add the combined quality/cost scalar ``quality_mean * cost_field``.

    Both objectives are minimised, so their *product* — not their ratio — is the
    number that falls whenever either improves.  On the throughput-filtered front,
    where every point delivers the same features, it is the whole trade-off in one
    figure: ``(quality / feature) · (cost / feature)``, weighted escaped bugs times
    dollars per feature².  On the log–log quality/cost plane its iso-lines are the
    −45° diagonals, so it weighs a relative gain in one objective against an equal
    relative loss in the other.

    It is *not* monotone along a Pareto front — on the §4.2 campaign the minimum is
    interior for 484 of the 486 fronts — which is exactly what makes it worth
    plotting, and why :func:`rank_frontier` keeps using `quality_cost_ratio` to fix
    front position instead.

    `cost_field` selects the cost term (`RATIO_COST_FIELDS`: total, human-only or
    agent-only).  The column is recomputed rather than reused, so a frontier table
    persisted before this column existed is handled the same as a fresh one.
    """
    _check_cost_field(df, cost_field)
    if df.empty:
        return df.copy()
    out = df.copy()
    out[column] = out["quality_mean"] * out[cost_field]
    return out


def add_front_position(
    df: pd.DataFrame,
    cost_field: str = "cost_mean",
    ratio_col: str = "quality_cost_ratio",
    pct_col: str = "front_pct",
) -> pd.DataFrame:
    """Add the quality/cost exchange rate and the point's position along its front.

    `ratio_col` is ``quality_mean / cost_field`` — see :func:`rank_frontier` for why
    that is a position coordinate rather than a score — and `pct_col` its percentile
    *within its own (scenario, strategy) front*, running from the quality-dominant
    end (0) to the cost-dominant end (1).  Front sizes span two orders of magnitude,
    so the per-front percentile is the only basis on which a 13-point and a
    1,190-point front are comparable.
    """
    _check_cost_field(df, cost_field)
    if df.empty:
        return df.copy()
    out = df.copy()
    out[ratio_col] = out["quality_mean"] / out[cost_field].replace(0.0, np.nan)
    out[pct_col] = out.groupby(["scenario_id", "strategy"], sort=False)[ratio_col].rank(
        pct=True
    )
    return out


def frontier_extremes(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """The `k` lowest- and `k` highest-ratio configurations per frontier.

    Adds an `end` column with values ``"quality_dominant"`` / ``"cost_dominant"``
    so §5 can characterise what the two ends of each front look like.  Frontiers
    with fewer than 2k points contribute every point they have; a point that
    qualifies for both ends is reported once, as quality-dominant.
    """
    if df.empty:
        return df.assign(end=pd.Series(dtype=object))
    ranked = rank_frontier(df)
    pieces = []
    for _, sub in ranked.groupby(["scenario_id", "strategy"], sort=False):
        sub = sub.sort_values("ratio_rank")
        low = sub.head(k).assign(end="quality_dominant")
        high = sub.tail(k).assign(end="cost_dominant")
        high = high[~high["config_key"].isin(low["config_key"])]
        pieces.append(pd.concat([low, high]))
    return pd.concat(pieces).reset_index(drop=True)


#: Grouping of :func:`index_optimum` that yields one row per *front*.
FRONT_KEYS: tuple[str, ...] = ("scenario_id", "strategy")


def index_optimum(
    df: pd.DataFrame,
    cost_field: str = "cost_mean",
    by: "tuple[str, ...] | list[str]" = FRONT_KEYS,
) -> pd.DataFrame:
    """The lowest-`quality_cost_index` row of each group.

    The index is U-shaped along a filtered front — on the §4.2 campaign its minimum
    is interior for 484 of the 486 fronts — so this names the configuration sitting
    at that knee: the best combined quality/cost point the search found.

    `by` defaults to `FRONT_KEYS`, one row per front (486 on the campaign).  Pass
    ``("scenario_id",)`` to pool the search strategies and get one row per scenario
    (243), which is what the §5 table reports.  All columns of the winning row are
    kept, so the caller still has the strategy, the config key and the full
    configuration.

    The index is recomputed via :func:`add_quality_cost_index`, so a frontier table
    persisted before that column existed works unchanged.
    """
    if df.empty:
        return df.copy()
    keys = list(by)
    work = add_quality_cost_index(df, cost_field=cost_field)
    # idxmin takes the first occurrence of a tie, so sort first to make the choice
    # depend on the data rather than on row order.
    work = work.sort_values([*keys, "quality_cost_index", "cost_mean"])
    winners = work.loc[work.groupby(keys, sort=True)["quality_cost_index"].idxmin()]
    return winners.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Headline numbers for the §5 prose
# ----------------------------------------------------------------------------

def headline_summary(
    df: pd.DataFrame,
    threshold: float = DEFAULT_VALUE_THRESHOLD,
    filtered_front: pd.DataFrame | None = None,
    filter_summary: pd.DataFrame | None = None,
) -> dict[str, object]:
    """The handful of numbers §5 states in prose.

    `df` is the recorded three-objective frontier and drives the bullet-1 figures:
    frontier size, the `value_ratio` distribution, the share of throughput-limited
    points, and the factor whose levels spread `frac_throughput_limited` most.

    Passing `filtered_front` / `filter_summary` adds the bullet-2 figures: how much
    of the candidate pool survives the throughput filter, how large the resulting
    quality/cost fronts are, and the spread of the two quality/cost scalars —
    `quality_cost_ratio` (front position) and `quality_cost_index` (goodness).
    """
    if df.empty:
        return {"n_frontier_points": 0}

    tl = _throughput_limited(df, threshold)
    patterns = value_deviation_patterns(df, threshold)
    # The factor whose levels disagree most about how often value falls short is
    # the one that "explains" the deviations.
    spread = (
        patterns.groupby("factor")["frac_throughput_limited"]
        .agg(lambda s: s.max() - s.min())
        .sort_values(ascending=False)
    )
    top_factor = str(spread.index[0]) if len(spread) else None

    out: dict[str, object] = {
        "n_scenarios": int(df["scenario_id"].nunique()),
        "n_searches": int(df.groupby(["scenario_id", "strategy"]).ngroups),
        "n_frontier_points": int(len(df)),
        "value_ratio_median": float(df["value_ratio"].median()),
        "value_ratio_p05": float(df["value_ratio"].quantile(0.05)),
        "value_ratio_min": float(df["value_ratio"].min()),
        "value_ratio_max": float(df["value_ratio"].max()),
        "frac_below_threshold": float((df["value_ratio"] < threshold).mean()),
        "frac_throughput_limited": float(tl.mean()),
        "threshold": threshold,
        "dominant_deviation_factor": top_factor,
        "dominant_deviation_levels": (
            patterns[patterns["factor"] == top_factor][
                ["level", "n_points", "frac_throughput_limited", "median_deficit"]
            ].to_dict("records")
            if top_factor else []
        ),
    }

    # The quality/cost figures describe the throughput-filtered front when one is
    # supplied, and fall back to the recorded frontier otherwise.
    qc = filtered_front if filtered_front is not None and not filtered_front.empty else df
    # Derived here rather than read off `qc`, so a table persisted before the column
    # existed still summarises.
    index = add_quality_cost_index(qc)
    out.update({
        "quality_cost_source": (
            "filtered_front" if qc is not df else "recorded_3d_frontier"
        ),
        "quality_cost_ratio_median": float(qc["quality_cost_ratio"].median()),
        "quality_cost_ratio_min": float(qc["quality_cost_ratio"].min()),
        "quality_cost_ratio_max": float(qc["quality_cost_ratio"].max()),
        "quality_cost_index_median": float(index["quality_cost_index"].median()),
        "quality_cost_index_min": float(index["quality_cost_index"].min()),
        "quality_cost_index_max": float(index["quality_cost_index"].max()),
        "agent_cost_share_median": float(qc["agent_cost_share"].median()),
    })

    if filtered_front is not None:
        out["n_filtered_front_points"] = int(len(filtered_front))
    if filter_summary is not None and not filter_summary.empty:
        fs = filter_summary
        out.update({
            "throughput_threshold": float(fs["throughput_threshold"].iloc[0]),
            "n_simulated_total": int(fs["n_simulated"].sum()),
            "n_passed_total": int(fs["n_passed"].sum()),
            "frac_passed_overall": float(fs["n_passed"].sum() / max(1, fs["n_simulated"].sum())),
            "frac_passed_min": float(fs["frac_passed"].min()),
            "filtered_front_size_median": float(fs["n_filtered_front"].median()),
            "filtered_front_size_min": int(fs["n_filtered_front"].min()),
            "filtered_front_size_max": int(fs["n_filtered_front"].max()),
            "recorded_3d_front_size_median": float(fs["n_recorded_3d_front"].median()),
        })
    return out
