"""Team constellation of the reported optima, faceted by two scenario groups.

`campaign.optima_csv` answers *what an optimum costs*; this module answers *what it is
made of*.  It reduces the campaign to a mean and a standard deviation of every control
variable of §3.3 — the staffing level `N`, the release process `(R, F_freeze)` and the two
8-element composition vectors — within each cell of a two-group tier grid.  The default
grid is `project` × `bugimpact`, the two groups expected to shape the optimal team most
strongly: 4 × 3 = 12 cells, each holding the 3 · 3 · 4 = 36 scenarios whose `human`, `debt`
and `agent` tiers are free.

Two populations are tabulated **separately**, because the campaign reports one optimum per
scenario but does not claim it is the only one:

``c_star``
    The reported `c*` of each scenario — one row of ``optima.csv`` per scenario.

``i_star``
    The whole indistinguishable set `I*` of §3.10, whose members are by construction
    statistically tied with `c*` on cost.  `|I*|` runs into the hundreds, so a `c*`-only
    table partly tabulates an arbitrary pick inside a tie; the `i_star` table says whether
    the pattern survives that.

`I*` is **not** ``search_summary.json``'s ``final_objectives``.  That block holds the
*preliminary* set re-simulated at `K_final`, before the stage-2 throughput filter and the
second parsimony tie-break of §3.10 — in `h-m_p-m_d-m_a-m_b-m` it has 115 entries against
the final 29.  The final set is the key list under ``indistinguishable``, which is also
what `campaign.optimum_row` counts into `n_indistinguishable`; the objective dicts are
then looked up in ``final_objectives``.  Extracting the wrong one would inflate every
`i_star` cell with configurations the selection rule rejected.

The `i_star` statistics are **scenario-balanced**: each scenario's `I*` is reduced to one
mean vector first (:func:`scenario_means`), and the cell mean and standard deviation are
taken over those 36 per-scenario vectors.  Both populations therefore share a unit of
observation and an `n`, which is what makes their two `sd` columns comparable.  Pooling
members instead would let a scenario with `|I*| = 338` outweigh one with `|I*| = 7` by 48×.

The human-only baseline of §4.1 is excluded from both populations: `config.is_human_only`
teams carry no agents at all, so they have no constellation to describe.

Outputs, all under the campaign's analysis directory:

``indistinguishable.csv``
    The extraction — one row per `I*` member, with the same column names `optima.csv`
    uses, so the two tables join on ``(scenario_id, strategy, config_key)``.  This one is
    written for **every** result mode in the tree, baselines included; it is the raw
    reading of the campaign, and :func:`search_optima` does the filtering the cell
    statistics need.

``constellation_cells.csv``
    Long format: one row per (population, cell, parameter) with `mean`, `sd`,
    `n_scenarios` and `n_members`.

``constellation_summary.txt``
    The same numbers rendered wide as `mean ± sd`, the way `campaign.convergence_summary`
    renders ``convergence.csv``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .campaign import (scenario_params_for_id, search_dirs,
                       tiers_from_scenario_id)
from .config import (
    AGENT_ROLE_COUNT,
    ALL_SCENARIO_TIERS,
    HUMAN_ONLY_MODE,
    SCENARIO_GROUPS,
    SUPERVISED_ROLE_INDICES,
    supervised_agent_count,
)

#: The two scenario groups whose tier grid the cells are cut along, by default.  Every
#: other group is free within a cell and contributes to its spread.
CELL_GROUPS: tuple[str, str] = ("project", "bugimpact")

#: The control variables of §3.3, in four blocks.  `N`, `R` and `F_freeze` are the process
#: triple; the two composition vectors are one column per role, 1-based on `k` exactly as
#: `campaign.optimum_row` writes them; the totals are derived here.
PROCESS_PARAMETERS: tuple[str, ...] = ("N", "R", "F_freeze")
DEV_PARAMETERS: tuple[str, ...] = tuple(
    f"a_dev_{k}" for k in range(1, AGENT_ROLE_COUNT + 1))
QA_PARAMETERS: tuple[str, ...] = tuple(
    f"a_qa_{k}" for k in range(1, AGENT_ROLE_COUNT + 1))
TOTAL_PARAMETERS: tuple[str, ...] = (
    "n_agents_dev", "n_agents_qa", "n_sup_dev", "n_sup_qa")

CONSTELLATION_PARAMETERS: tuple[str, ...] = (
    PROCESS_PARAMETERS + DEV_PARAMETERS + QA_PARAMETERS + TOTAL_PARAMETERS)

#: Human-readable block boundaries, for the text rendering and the heatmap separators.
PARAMETER_BLOCKS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("process", PROCESS_PARAMETERS),
    ("A_dev", DEV_PARAMETERS),
    ("A_qa", QA_PARAMETERS),
    ("totals", TOTAL_PARAMETERS),
)

#: Width of the parameter-name column in the text rendering; `n_agents_dev` is the
#: longest name and needs a space after it.
_LABEL_WIDTH = 16

C_STAR = "c_star"
I_STAR = "i_star"
POPULATIONS: tuple[str, ...] = (C_STAR, I_STAR)


def _check_cell_groups(groups: "tuple[str, ...]") -> tuple[str, ...]:
    """Validate the two faceting groups; returns them unchanged."""
    groups = tuple(groups)
    if len(groups) != 2:
        raise ValueError(f"expected exactly two faceting groups, got {list(groups)}")
    unknown = [g for g in groups if g not in SCENARIO_GROUPS]
    if unknown:
        raise ValueError(f"unknown scenario group(s) {unknown}; "
                         f"expected one of {list(SCENARIO_GROUPS)}")
    if groups[0] == groups[1]:
        raise ValueError(f"the two faceting groups must differ, got {groups[0]!r} twice")
    return groups


# ----------------------------------------------------------------------------
# Extracting the indistinguishable set
# ----------------------------------------------------------------------------

def indistinguishable_rows(scenario_id: str, strategy: str, summary: dict) -> list[dict]:
    """One row per member of the *final* `I*` of one search.

    The members are ``summary["indistinguishable"]`` — the key list the selection rule of
    §3.10 ends on — looked up in ``final_objectives``, the `K_final` re-simulation those
    keys were re-ranked on.  A result written before the two-stage selection existed has
    no ``final_objectives``, and the search's own `K_search` ``objectives`` stand in, as
    they do in `campaign.optimum_row`.

    `is_best` flags `c*`, which the rule guarantees is a member of every non-empty `I*`
    (it is the cost minimiser, and the parsimony tie-break cannot remove it).  An
    inadmissible search has an empty `I*` and contributes no rows at all.
    """
    keys = summary.get("indistinguishable") or []
    final = summary.get("final_objectives") or {}
    fallback = summary.get("objectives") or {}
    best = summary.get("best")
    tiers = tiers_from_scenario_id(scenario_id)

    rows: list[dict] = []
    for key in keys:
        obj = final.get(key) or fallback.get(key)
        if obj is None:
            raise KeyError(
                f"{scenario_id}/{strategy}: {key!r} is in 'indistinguishable' but in "
                f"neither 'final_objectives' nor 'objectives'"
            )
        row: dict = {"scenario_id": scenario_id}
        row.update({f"tier_{group}": tiers[group] for group in SCENARIO_GROUPS})
        row.update({
            "strategy": strategy,
            "config_key": key,
            "is_best": key == best,
            "N": obj["N"], "R": obj["R"], "F_freeze": obj["F_freeze"],
        })
        A_dev, A_qa = obj["A_dev"], obj["A_qa"]
        for k in range(AGENT_ROLE_COUNT):
            row[f"a_dev_{k + 1}"] = A_dev[k]
        for k in range(AGENT_ROLE_COUNT):
            row[f"a_qa_{k + 1}"] = A_qa[k]
        row["n_sup_dev"] = supervised_agent_count(tuple(A_dev))
        row["n_sup_qa"] = supervised_agent_count(tuple(A_qa))
        row["cost_mean"] = obj["cost_mean"]
        rows.append(row)
    return rows


def build_indistinguishable(results_dir: str | Path) -> pd.DataFrame:
    """Read every finished search under `results_dir` into one `I*` member table.

    Same tree and same discovery as `campaign.build_tables` — :func:`campaign.search_dirs`
    — so a partially finished campaign simply contributes fewer rows.  Only the
    ``search_summary.json`` is read; the 10,000-entry ``objectives`` block is untouched
    unless a pre-re-simulation result forces the fallback.
    """
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"{results_dir} is not a directory")
    searches = search_dirs(results_dir)
    if not searches:
        raise FileNotFoundError(f"no finished searches found under {results_dir}")

    rows: list[dict] = []
    for scenario_id, strategy, d in searches:
        summary = json.loads((d / "search_summary.json").read_text(encoding="utf-8"))
        rows.extend(indistinguishable_rows(scenario_id, strategy, summary))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# What each role costs
# ----------------------------------------------------------------------------

#: The per-role agent bill over the horizon, in dollars — `actions_k · c_token,k` of
#: Eq. (cost_agents), one column per role.
BILL_PARAMETERS: tuple[str, ...] = tuple(
    f"bill_{k}" for k in range(1, AGENT_ROLE_COUNT + 1))

#: The same split rescaled to **dollars per feature**, the unit `optima.csv` reports costs
#: in: `cost_agent_mean` apportioned by each role's share of the bill.  Rescaled rather
#: than divided by the feature count directly, so the eight columns sum to the reported
#: `cost_agent_mean` *exactly* and a figure mixing them with `cost_human_mean` is adding
#: like for like.
AGENT_COST_PARAMETERS: tuple[str, ...] = tuple(
    f"agent_cost_{k}" for k in range(1, AGENT_ROLE_COUNT + 1))


def role_bill(optima: pd.DataFrame, config_dir: str = "config") -> pd.DataFrame:
    """Add the per-role agent bill to a copy of an ``optima.csv`` frame.

    `C_agents = Σ_k c_token,k · Σ_t u_k(t)` (Eq. cost_agents) is already split by role in
    the data: ``actions_k`` is the work role `k` completed over the whole horizon in its
    own throughput unit, and `c_token,k` is that unit's price at the scenario's `agent`
    tier.  Their product is therefore the *exact* per-role bill, not an apportionment —
    summing the eight columns reproduces the reported `cost_agent_mean · F_completed` to
    within 0.3 % on the current campaign, the residual being that `cost_agent_mean` is a
    mean of per-replication ratios while this is a ratio of means.

    That distinction does not reach a **share**, which is what the composition figure
    draws: `bill_k / Σ_j bill_j` is formed inside one scenario from one row, and the
    horizon and the feature count cancel exactly.  `AGENT_COST_PARAMETERS` carries the
    same split in the reported dollars-per-feature unit, and `cost_team_mean` the
    developers and the agents together.

    Why this exists at all: a share of the *head count* systematically overstates the
    three autonomous roles.  They are flat-rate (`r_k = 1`, so `u_k = a_k`), they consume
    no monitoring capacity, and nothing but their own saturation bounds them — so the
    search stacks them freely.  Code Review is 69 % of the mean optimal team and 20 % of
    its bill; Feature Implementation is 2 % of the team and 21 % of the bill.
    """
    frame = optima.copy()
    prices = {}
    for scenario_id in frame["scenario_id"].unique():
        prices[scenario_id] = scenario_params_for_id(scenario_id, config_dir).c_token
    price = np.array([prices[s] for s in frame["scenario_id"]], dtype=float)
    actions = frame[[f"actions_{k}" for k in range(1, AGENT_ROLE_COUNT + 1)]].to_numpy(
        dtype=float)
    bill = actions * price
    for k, column in enumerate(BILL_PARAMETERS):
        frame[column] = bill[:, k]

    # …and the same split in dollars per feature.  A team that bills nothing has no split
    # to apportion; its eight shares are zero rather than a division by zero.
    total = bill.sum(axis=1)
    share = np.divide(bill, total[:, None], out=np.zeros_like(bill), where=total[:, None] > 0)
    reported = frame["cost_agent_mean"].to_numpy(dtype=float)[:, None]
    for k, column in enumerate(AGENT_COST_PARAMETERS):
        frame[column] = share[:, k] * reported[:, 0]
    # The team's own bill, the two drivers of Eq. (cost_rep) the composition controls.
    # The third, `cost_quality_mean`, is what the team's choices *cost elsewhere* and is
    # deliberately not folded in: it is a consequence, not a line item of the team.
    frame["cost_team_mean"] = frame["cost_human_mean"] + frame["cost_agent_mean"]
    return frame


# ----------------------------------------------------------------------------
# Cell statistics
# ----------------------------------------------------------------------------

def derive_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Add the four `TOTAL_PARAMETERS` to a frame that has the per-role columns.

    `n_agents_*` is the whole team size in that phase; `n_sup_*` is the supervised subset
    of §3.3 that Eq. (feas_monitoring) charges against the humans.  ``optima.csv`` already
    carries the latter two, and they are left alone where present so this is idempotent.
    """
    out = df.copy()
    out["n_agents_dev"] = out[list(DEV_PARAMETERS)].sum(axis=1)
    out["n_agents_qa"] = out[list(QA_PARAMETERS)].sum(axis=1)
    for phase, columns in (("dev", DEV_PARAMETERS), ("qa", QA_PARAMETERS)):
        column = f"n_sup_{phase}"
        if column not in out.columns:
            # The parameter tuples are in k=1..8 order, so the composition
            # indices of §3.3 index them directly.
            supervised = [columns[i] for i in SUPERVISED_ROLE_INDICES]
            out[column] = out[supervised].sum(axis=1)
    return out


def scenario_means(istar: pd.DataFrame) -> pd.DataFrame:
    """Reduce each scenario's `I*` to one mean vector, keeping its tiers.

    This is what makes the `i_star` cell statistics scenario-balanced: the cell mean is
    then a mean over scenarios, exactly as the `c_star` one is, rather than over members —
    which would weight a scenario by how wide its tie happened to be.  `n_members` records
    `|I*|` so the pooled size stays visible in the output.
    """
    if istar.empty:
        return istar
    tier_columns = [f"tier_{group}" for group in SCENARIO_GROUPS]
    parameters = [c for c in CONSTELLATION_PARAMETERS if c in istar.columns]
    grouped = istar.groupby(["scenario_id", "strategy"], sort=True)
    means = grouped[parameters].mean()
    means["n_members"] = grouped.size()
    tiers = grouped[tier_columns].first()
    return pd.concat([tiers, means], axis=1).reset_index()


def cell_statistics(
    df: pd.DataFrame,
    population: str,
    groups: tuple[str, ...] = CELL_GROUPS,
) -> pd.DataFrame:
    """Mean and standard deviation of every parameter within each cell of the tier grid.

    Long format — one row per (population, cell, parameter) — because the parameter set is
    a list the caller may shorten and a wide table would then change shape.  `sd` is the
    sample standard deviation (ddof = 1) over the scenarios in the cell, so it measures
    how much the three free scenario groups move the optimum, **not** Monte Carlo error;
    a cell with one scenario reports NaN.
    """
    groups = _check_cell_groups(groups)
    cell_columns = [f"tier_{group}" for group in groups]
    if df.empty:
        return pd.DataFrame(columns=["population", *cell_columns, "parameter",
                                     "mean", "sd", "n_scenarios", "n_members"])
    missing = [c for c in cell_columns if c not in df.columns]
    if missing:
        raise KeyError(f"{missing} are not columns of the table")

    parameters = [c for c in CONSTELLATION_PARAMETERS if c in df.columns]
    grouped = df.groupby(cell_columns, sort=False)
    stats = grouped[parameters].agg(["mean", "std"])
    counts = grouped.size().rename("n_scenarios")
    # `c_star` contributes one configuration per scenario, so the two counts coincide
    # there; for `i_star` the frame handed in carries the pooled `|I*|` per scenario.
    members = (grouped["n_members"].sum().rename("n_members")
               if "n_members" in df.columns else counts.rename("n_members"))

    long = (stats.stack(level=0, future_stack=True)
                 .rename(columns={"std": "sd"})
                 .rename_axis(index=[*cell_columns, "parameter"])
                 .reset_index())
    long = long.merge(counts.reset_index(), on=cell_columns, how="left")
    long = long.merge(members.reset_index(), on=cell_columns, how="left")
    long.insert(0, "population", population)
    return _sorted_cells(long, groups, parameters)


def _sorted_cells(
    long: pd.DataFrame,
    groups: tuple[str, ...],
    parameters: list[str],
) -> pd.DataFrame:
    """Order the long table by tier (mild→extreme) and by parameter block."""
    order = {name: i for i, name in enumerate(ALL_SCENARIO_TIERS)}
    keys = []
    for group in groups:
        keys.append(long[f"tier_{group}"].map(order))
    keys.append(long["parameter"].map({p: i for i, p in enumerate(parameters)}))
    out = long.assign(**{f"_k{i}": k for i, k in enumerate(keys)})
    sort_columns = [f"_k{i}" for i in range(len(keys))]
    out = out.sort_values(sort_columns).drop(columns=sort_columns)
    return out[["population", *[f"tier_{g}" for g in groups], "parameter",
                "mean", "sd", "n_scenarios", "n_members"]].reset_index(drop=True)


def search_optima(optima: pd.DataFrame) -> pd.DataFrame:
    """The rows of ``optima.csv`` that describe a searched, agent-supported team.

    The human-only baseline of §4.1 is dropped: `config.is_human_only` holds for every one
    of its teams, so its whole composition is the zero vector and it has no constellation.
    Filtered on the mode rather than on `is_human_only` so a search that happened to report
    a zero-agent optimum would still be counted — the samplers cannot reach one, but the
    table should say what the search found, not what it was supposed to find.
    """
    if optima.empty or "strategy" not in optima.columns:
        return optima
    return optima[optima["strategy"] != HUMAN_ONLY_MODE]


def build_constellation(
    optima: pd.DataFrame,
    istar: pd.DataFrame,
    groups: tuple[str, ...] = CELL_GROUPS,
) -> pd.DataFrame:
    """The two populations' cell statistics, stacked into one long table."""
    groups = _check_cell_groups(groups)
    frames = []

    c_rows = search_optima(optima)
    if not c_rows.empty:
        frames.append(cell_statistics(derive_totals(c_rows), C_STAR, groups))

    i_rows = search_optima(istar)
    if not i_rows.empty:
        frames.append(cell_statistics(
            scenario_means(derive_totals(i_rows)), I_STAR, groups))

    if not frames:
        return cell_statistics(pd.DataFrame(), C_STAR, groups)
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------------
# Text rendering
# ----------------------------------------------------------------------------

def _mean_sd(mean: float, sd: float) -> str:
    """`mean ± sd`, four significant digits on the mean and three on the spread."""
    if pd.isna(mean):
        return "n/a"
    sd_text = "n/a" if pd.isna(sd) else f"{sd:.3g}"
    return f"{mean:.4g} ± {sd_text}"


def constellation_summary(
    cells: pd.DataFrame,
    groups: tuple[str, ...] = CELL_GROUPS,
) -> str:
    """The cell statistics as a text block, one section per population and outer tier.

    Laid out with the *second* group across the columns and the parameters down the rows,
    so a block stays inside a terminal width whatever the parameter count; the first
    group's tiers become the blocks.  Everything here is a re-reading of
    ``constellation_cells.csv``; nothing is computed that the table does not already hold.
    """
    groups = _check_cell_groups(groups)
    if cells.empty:
        return f"Team constellation by {groups[0]} × {groups[1]}: no optimum reported.\n"

    row_group, col_group = groups
    row_column, col_column = f"tier_{row_group}", f"tier_{col_group}"
    col_tiers = [t for t in ALL_SCENARIO_TIERS if (cells[col_column] == t).any()]
    row_tiers = [t for t in ALL_SCENARIO_TIERS if (cells[row_column] == t).any()]
    populations = [p for p in POPULATIONS if (cells["population"] == p).any()]

    lines = [f"Team constellation of the reported optima, by {row_group} × {col_group}",
             "=" * 78,
             "",
             "Each cell aggregates the scenarios whose remaining scenario groups are free;",
             "`sd` is their spread, not Monte Carlo error.  Populations:",
             f"  {C_STAR:<8} the reported c* of each scenario, one configuration per scenario",
             f"  {I_STAR:<8} the indistinguishable set I* of §3.10, each scenario reduced to",
             "           its own mean first, so both populations weight scenarios equally",
             "",
             "Structural zeros to read past: a_qa_1 and a_qa_2 are 0 by construction, and the",
             "whole A_qa vector is 0 wherever F_freeze = 0 (TeamConfiguration canonicalises it),",
             "so a zero A_qa cell means 'no feature freeze', not 'a freeze with no agents'.",
             ""]

    # One column width for the whole file, measured on the widest cell it will print:
    # a `mean ± sd` is as wide as its significant digits make it, and a per-block width
    # would stop the blocks lining up under each other.
    width = max([len(_mean_sd(m, s)) for m, s in zip(cells["mean"], cells["sd"])]
                + [len(t) for t in col_tiers]) + 2
    for population in populations:
        block = cells[cells["population"] == population]
        for row_tier in row_tiers:
            sub = block[block[row_column] == row_tier]
            if sub.empty:
                continue
            head = sub.drop_duplicates(col_column).set_index(col_column)
            n_text = ", ".join(
                f"{t}: n = {int(head.loc[t, 'n_scenarios'])}"
                + (f" ({int(head.loc[t, 'n_members'])} members)"
                   if population == I_STAR else "")
                for t in col_tiers if t in head.index
            )
            lines.append("-" * 78)
            lines.append(f"{population}   {row_group} = {row_tier}")
            lines.append(f"  {n_text}")
            lines.append("-" * 78)
            lines.append(" " * _LABEL_WIDTH + "".join(f"{t:>{width}}" for t in col_tiers))
            table = sub.set_index(["parameter", col_column])
            for _, parameters in PARAMETER_BLOCKS:
                printed = False
                for parameter in parameters:
                    if parameter not in sub["parameter"].values:
                        continue
                    cell_texts = []
                    for tier in col_tiers:
                        if (parameter, tier) in table.index:
                            record = table.loc[(parameter, tier)]
                            cell_texts.append(
                                f"{_mean_sd(record['mean'], record['sd']):>{width}}")
                        else:
                            cell_texts.append(f"{'':>{width}}")
                    lines.append(f"  {parameter:<{_LABEL_WIDTH - 2}}" + "".join(cell_texts))
                    printed = True
                if printed:
                    lines.append("")
    return "\n".join(lines) + "\n"
