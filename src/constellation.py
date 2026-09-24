"""What the optimal teams are made of, faceted by `CELL_GROUPS`.

`src.campaign` answers what an optimum costs; this module answers what it is made of.  It
reduces the campaign to a mean and a standard deviation of every control variable — the
staffing level `N`, the release process `(R, F_freeze)`, the bug-fixing share `f_fix` and
the two 8-element composition vectors — within each cell of the `CELL_GROUPS` tier grid.  The scenario groups outside
that grid are free within a cell and become its spread.  Teams with no agents are excluded
throughout: they have no composition to describe.

Two populations are tabulated separately, since a campaign reports one optimum per
scenario without claiming it is the only one:

``c_star``
    The reported `c*` of each scenario — one row of ``optima.csv`` per scenario.

``i_star``
    The whole indistinguishable set `I*`, whose members are statistically tied with `c*`
    on cost.  Reduced to one mean vector per scenario first (:func:`scenario_means`), so
    both populations share a unit of observation and their `sd` columns are comparable.

Outputs, all under the campaign's analysis directory:

``indistinguishable.csv``
    One row per `I*` member, with the column names ``optima.csv`` uses, so the two join
    on ``(scenario_id, mode, config_key)``.  Written for every result mode in the tree.

``constellation_cells.csv``
    Long format: one row per (population, cell, parameter) with `mean`, `sd`,
    `n_scenarios` and `n_members`.

``constellation_summary.txt``
    The same numbers rendered wide as `mean +/- sd`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .campaign import (ACTION_PARAMETERS, DEV_PARAMETERS, QA_PARAMETERS,
                       admissible_optima, composition_columns, process_columns,
                       require_columns, require_search_dirs, scenario_columns,
                       scenario_params_for_id, search_rows)
from .config import (
    AGENT_ROLE_ABBREV,
    AGENT_ROLE_COUNT,
    ALL_SCENARIO_TIERS,
    SCENARIO_GROUPS,
    SUPERVISED_ROLE_INDICES,
    tier_column,
    tiers_present,
)
from .io import read_json

#: The two scenario groups whose tier grid the cells are cut along; every other group is
#: free within a cell and contributes to its spread.
CELL_GROUPS: tuple[str, str] = ("project", "bugimpact")

#: The scenario group the role statistics are additionally split by.
ROLE_SPLIT_GROUP: str = "agent"

#: The control variables in four blocks: the process controls, the two composition vectors
#: one column per role, and the totals derived here.
PROCESS_PARAMETERS: tuple[str, ...] = ("N", "R", "F_freeze", "f_fix")
TOTAL_PARAMETERS: tuple[str, ...] = (
    "n_agents_dev", "n_agents_qa", "n_sup_dev", "n_sup_qa")

CONSTELLATION_PARAMETERS: tuple[str, ...] = (
    PROCESS_PARAMETERS + DEV_PARAMETERS + QA_PARAMETERS + TOTAL_PARAMETERS)

#: Human-readable block boundaries, for the text rendering.
PARAMETER_BLOCKS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("process", PROCESS_PARAMETERS),
    ("A_dev", DEV_PARAMETERS),
    ("A_qa", QA_PARAMETERS),
    ("totals", TOTAL_PARAMETERS),
)

#: Width of the parameter-name column in the text rendering; the longest name is
#: `n_agents_dev`, which needs a space after it.
_LABEL_WIDTH = 16

C_STAR = "c_star"
I_STAR = "i_star"
POPULATIONS: tuple[str, ...] = (C_STAR, I_STAR)


# ----------------------------------------------------------------------------
# Extracting the indistinguishable set
# ----------------------------------------------------------------------------

def indistinguishable_rows(scenario_id: str, mode: str, summary: dict) -> list[dict]:
    """One row per member of the final `I*` of one run.

    The members are ``summary["indistinguishable"]``, the key list the selection ends on,
    looked up in ``final_objectives``.  Reading ``final_objectives`` directly would
    pick up the wider preliminary set, which the selection rule had already rejected.
    `is_best` flags `c*`. An inadmissible run has an empty `I*` and contributes no rows.
    """
    keys = summary["indistinguishable"]
    final = summary["final_objectives"]
    best = summary["best"]

    rows: list[dict] = []
    for key in keys:
        obj = final[key]
        row = scenario_columns(scenario_id, mode)
        row["config_key"] = key
        row["is_best"] = key == best
        row.update(process_columns(obj))
        row.update(composition_columns(obj))
        row["cost_mean"] = obj["cost_mean"]
        rows.append(row)
    return rows


def build_indistinguishable(results_dir: str | Path) -> pd.DataFrame:
    """Read every finished search under `results_dir` into one `I*` member table.

    Same tree and same discovery as `campaign.build_tables`.
    """
    searches = require_search_dirs(results_dir)

    rows: list[dict] = []
    for scenario_id, mode, d in searches:
        summary = read_json(d / "search_summary.json")
        rows.extend(indistinguishable_rows(scenario_id, mode, summary))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# What each role costs
# ----------------------------------------------------------------------------

#: The per-role agent bill over in dollars, `actions_k * c_token_k`, one
#: column per role.
BILL_PARAMETERS: tuple[str, ...] = tuple(
    f"bill_{k}" for k in range(1, AGENT_ROLE_COUNT + 1))

#: The same split in dollars per feature: `cost_agent_mean` by each role's
#: share of the bill. Rescaling rather than dividing by the feature count makes the eight
#: columns sum to `cost_agent_mean` exactly, so a figure may mix them with
#: `cost_human_mean`.
AGENT_COST_PARAMETERS: tuple[str, ...] = tuple(
    f"agent_cost_{k}" for k in range(1, AGENT_ROLE_COUNT + 1))


def role_bill(optima: pd.DataFrame) -> pd.DataFrame:
    """Add the per-role agent bill to a copy of an ``optima.csv`` frame.

    ``actions_k`` is the work role `k` completed over the horizon in its own throughput unit
    and `c_token_k` prices that unit.
    Summing the eight reproduces `cost_agent_mean * F_completed` only to
    within a fraction of a per cent, because `cost_agent_mean` is a mean of per-replication
    ratios while this is a ratio of means. `AGENT_COST_PARAMETERS` avoids that gap by rescaling
    onto the reported figure.
    """
    frame = optima.copy()
    prices = {scenario_id: scenario_params_for_id(scenario_id).c_token
              for scenario_id in frame["scenario_id"].unique()}
    price = np.array([prices[s] for s in frame["scenario_id"]], dtype=float)
    actions = frame[list(ACTION_PARAMETERS)].to_numpy(dtype=float)
    bill = actions * price
    for k, column in enumerate(BILL_PARAMETERS):
        frame[column] = bill[:, k]

    # The same split in dollars per feature.  A team that bills nothing has no split to
    # apportion, so its eight shares are zero rather than a division by zero.
    total = bill.sum(axis=1)
    share = np.divide(bill, total[:, None], out=np.zeros_like(bill), where=total[:, None] > 0)
    reported = frame["cost_agent_mean"].to_numpy(dtype=float)
    for k, column in enumerate(AGENT_COST_PARAMETERS):
        frame[column] = share[:, k] * reported
    # The two cost drivers the composition controls.  `cost_quality_mean` is not one of
    # them: it is what the team's choices cost elsewhere.
    frame["cost_team_mean"] = frame["cost_human_mean"] + frame["cost_agent_mean"]
    return frame


# ----------------------------------------------------------------------------
# Cell statistics
# ----------------------------------------------------------------------------

def derive_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Add the four `TOTAL_PARAMETERS` to a frame that has the per-role columns.

    `n_agents_*` is the whole team size in that phase; `n_sup_*` is the supervised subset.
    Idempotent: ``optima.csv`` already carries the latter.
    """
    out = df.copy()
    out["n_agents_dev"] = out[list(DEV_PARAMETERS)].sum(axis=1)
    out["n_agents_qa"] = out[list(QA_PARAMETERS)].sum(axis=1)
    # The parameter tuples run k=1..8, so a role index addresses them directly.
    for phase, columns in (("dev", DEV_PARAMETERS), ("qa", QA_PARAMETERS)):
        supervised = [columns[i] for i in SUPERVISED_ROLE_INDICES]
        out[f"n_sup_{phase}"] = out[supervised].sum(axis=1)
    return out


def scenario_means(istar: pd.DataFrame) -> pd.DataFrame:
    """Reduce each scenario's `I*` to one mean vector, keeping its tiers.

    This is what makes the `i_star` cell statistics scenario-balanced: a cell mean is then a
    mean over scenarios, as the `c_star` one is, rather than over members.  `n_members` records
    `|I*|`, so the pooled size stays visible.
    """
    if istar.empty:
        return istar
    tier_columns = [tier_column(group) for group in SCENARIO_GROUPS]
    parameters = list(CONSTELLATION_PARAMETERS)
    grouped = istar.groupby(["scenario_id", "mode"], sort=True)
    means = grouped[parameters].mean()
    means["n_members"] = grouped.size()
    tiers = grouped[tier_columns].first()
    return pd.concat([tiers, means], axis=1).reset_index()


def cell_statistics(
    df: pd.DataFrame,
    population: str,
) -> pd.DataFrame:
    """Mean and standard deviation of every parameter within each cell of the tier grid.

    Long format — one row per (population, cell, parameter) — so the table keeps its shape
    whatever the parameter set contains.  `sd` is the sample standard deviation over the
    scenarios in the cell, i.e. their spread rather than Monte Carlo error.
    """
    cell_columns = [tier_column(group) for group in CELL_GROUPS]
    if df.empty:
        return pd.DataFrame(columns=["population", *cell_columns, "parameter",
                                     "mean", "sd", "n_scenarios", "n_members"])
    require_columns(df, cell_columns, "table")

    parameters = list(CONSTELLATION_PARAMETERS)
    grouped = df.groupby(cell_columns, sort=False)
    stats = grouped[parameters].agg(["mean", "std"])
    counts = grouped.size().rename("n_scenarios")
    # `c_star` has one configuration per scenario, so its two counts coincide; `i_star`
    # contains the pooled `|I*|` per scenario in `n_members`.
    members = (grouped["n_members"].sum().rename("n_members")
               if "n_members" in df.columns else counts.rename("n_members"))

    long = (stats.stack(level=0, future_stack=True)
                 .rename(columns={"std": "sd"})
                 .rename_axis(index=[*cell_columns, "parameter"])
                 .reset_index())
    long = long.merge(counts.reset_index(), on=cell_columns, how="left")
    long = long.merge(members.reset_index(), on=cell_columns, how="left")
    long.insert(0, "population", population)
    return _sorted_cells(long)


def _sorted_cells(long: pd.DataFrame) -> pd.DataFrame:
    """Order the long table by tier (mild→extreme) and by parameter block."""
    tier_order = {name: i for i, name in enumerate(ALL_SCENARIO_TIERS)}
    parameter_order = {name: i for i, name in enumerate(CONSTELLATION_PARAMETERS)}
    tier_columns = [tier_column(g) for g in CELL_GROUPS]
    out = long.sort_values(
        tier_columns + ["parameter"],
        key=lambda column: column.map(
            parameter_order if column.name == "parameter" else tier_order),
    )
    return out[["population", *tier_columns, "parameter",
                "mean", "sd", "n_scenarios", "n_members"]].reset_index(drop=True)


def build_constellation(
    optima: pd.DataFrame,
    istar: pd.DataFrame,
) -> pd.DataFrame:
    """The two populations' cell statistics, stacked into one long table."""
    frames = []
    for frame, population, balance in ((optima, C_STAR, False), (istar, I_STAR, True)):
        rows = search_rows(frame)
        if rows.empty:
            continue
        rows = derive_totals(rows)
        frames.append(cell_statistics(
            scenario_means(rows) if balance else rows, population))

    if not frames:
        return cell_statistics(pd.DataFrame(), C_STAR)
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


def constellation_summary(cells: pd.DataFrame) -> str:
    """The cell statistics as a text block, one section per population and outer tier.

    The second cell group runs across the columns and the parameters down the rows, so a block
    stays inside a terminal width.
    """
    if cells.empty:
        return (f"Team constellation by {CELL_GROUPS[0]} × {CELL_GROUPS[1]}: "
                f"no optimum reported.\n")

    row_group, col_group = CELL_GROUPS
    row_column, col_column = tier_column(row_group), tier_column(col_group)
    col_tiers = tiers_present(cells[col_column])
    row_tiers = tiers_present(cells[row_column])
    populations = [p for p in POPULATIONS if (cells["population"] == p).any()]

    lines = [f"Team constellation of the reported optima, by {row_group} × {col_group}",
             "=" * 78,
             "",
             "Each cell aggregates the scenarios whose remaining scenario groups are free;",
             "`sd` is their spread, not Monte Carlo error.  Populations:",
             f"  {C_STAR:<8} the reported c* of each scenario, one configuration per scenario",
             f"  {I_STAR:<8} the indistinguishable set I*, each scenario reduced to",
             "           its own mean first, so both populations weight scenarios equally",
             "",
             "Structural zeros to ignore: a_qa_1 and a_qa_2 are 0 by construction, and the",
             "whole A_qa vector is 0 wherever F_freeze = 0, so a zero A_qa cell means",
             "'no feature freeze', not 'a freeze with no agents'.",
             ""]

    # One column width for the whole file, so the blocks line up under each other.
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


# ----------------------------------------------------------------------------
# Role usage and process diversity
# ----------------------------------------------------------------------------

def role_usage(bills: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Per role: how often it is staffed, and its share of the head count and the bill.

    `bills` is a `role_bill` restricted to the searched optima.  Both are formed
    **within** a scenario before being averaged: the optimal team size spans orders of
    magnitude, so a ratio of aggregates would describe the largest scenario rather than the
    typical one.  With `by` given, the statistics are additionally split by that group's tier.
    """
    if bills.empty:
        return pd.DataFrame()

    counts = bills[list(DEV_PARAMETERS)].to_numpy(float)
    bill = bills[list(BILL_PARAMETERS)].to_numpy(float)
    count_total = counts.sum(axis=1)
    bill_total = bill.sum(axis=1)
    count_share = np.divide(counts, count_total[:, None],
                            out=np.zeros_like(counts), where=count_total[:, None] > 0)
    bill_share = np.divide(bill, bill_total[:, None],
                           out=np.zeros_like(bill), where=bill_total[:, None] > 0)

    groups = (bills[tier_column(by)].to_numpy() if by is not None
              else np.full(len(bills), "all", dtype=object))
    rows = []
    for tier in (ALL_SCENARIO_TIERS if by is not None else ("all",)):
        mask = groups == tier
        if not mask.any():
            continue
        for k in range(AGENT_ROLE_COUNT):
            rows.append({
                "group": by or "all",
                "tier": tier,
                "role": k + 1,
                "staffed_share": float((counts[mask, k] > 0).mean()),
                "head_count_share": float(count_share[mask, k].mean()),
                "bill_share": float(bill_share[mask, k].mean()),
                "mean_count": float(counts[mask, k].mean()),
                "median_count": float(np.median(counts[mask, k])),
                "n": int(mask.sum()),
            })
    return pd.DataFrame(rows)


def process_usage(optima: pd.DataFrame) -> pd.DataFrame:
    """How the searched optima set the release process: the `R` spread and the freeze.

    One row per release cadence the optima actually use, plus one for "no feature freeze", each
    a share of the reported optima.
    """
    if optima.empty:
        return pd.DataFrame()
    n = len(optima)
    rows = [
        {"quantity": "R", "value": cadence,
         "share": float((optima["R"] == cadence).mean()), "n": n}
        for cadence in sorted(optima["R"].unique())
    ]
    rows.extend(
        {"quantity": "f_fix", "value": share,
         "share": float((optima["f_fix"] == share).mean()), "n": n}
        for share in sorted(optima["f_fix"].unique())
    )
    rows.append({"quantity": "F_freeze", "value": 0,
                 "share": float((optima["F_freeze"] == 0).mean()), "n": n})
    return pd.DataFrame(rows)


def istar_process_diversity(istar: pd.DataFrame) -> pd.DataFrame:
    """Per search, how many distinct `(N, R, F_freeze, f_fix)` tuples its `I*` spans.

    A reported optimum is one member of a statistical tie, so `R` or `N` being unique across
    the set means the tie agrees on it, and the tuple count says how much of the process grid
    it covers.
    """
    if istar.empty:
        return pd.DataFrame()
    searched = search_rows(istar)
    if searched.empty:
        return pd.DataFrame()
    rows = []
    for scenario_id, members in searched.groupby("scenario_id"):
        rows.append({
            "scenario_id": scenario_id,
            "n_members": len(members),
            "n_triples": int(members[list(PROCESS_PARAMETERS)].drop_duplicates().shape[0]),
            "R_unique": bool(members["R"].nunique() == 1),
            "N_unique": bool(members["N"].nunique() == 1),
        })
    return pd.DataFrame(rows)


def build_process_summary(bills: pd.DataFrame, istar: pd.DataFrame) -> pd.DataFrame:
    """The long table behind `process_summary`: role usage, the process, and `I*` spread.

    `bills` is a `role_bill` frame with only its admissible rows,
    since neither a zero-agent team nor a team that fails to deliver describes an optimum.
    """
    searched = admissible_optima(bills)
    if searched.empty:
        return pd.DataFrame()

    parts = [frame.assign(section="role")
             for frame in (role_usage(searched), role_usage(searched, by=ROLE_SPLIT_GROUP))
             if not frame.empty]

    process = process_usage(searched)
    if not process.empty:
        parts.append(process.assign(section="process", group="all", tier="all"))

    diversity = istar_process_diversity(istar)
    if not diversity.empty:
        parts.append(pd.DataFrame([
            {"section": "istar", "group": "all", "tier": "all",
             "quantity": "n_triples_median",
             "value": float(diversity["n_triples"].median()), "n": len(diversity)},
            {"section": "istar", "group": "all", "tier": "all",
             "quantity": "R_unique", "share": float(diversity["R_unique"].mean()),
             "n": len(diversity)},
            {"section": "istar", "group": "all", "tier": "all",
             "quantity": "N_unique", "share": float(diversity["N_unique"].mean()),
             "n": len(diversity)},
        ]))

    columns = ["section", "group", "tier", "role", "quantity", "value", "share",
               "staffed_share", "head_count_share", "bill_share",
               "mean_count", "median_count", "n"]
    combined = pd.concat(parts, ignore_index=True)
    return combined.reindex(columns=columns)


def process_summary(table: pd.DataFrame) -> str:
    """Render `process_summary.csv` as a text block.

    The group the per-tier block is split by is read off the table's own `group` column,
    so the header cannot disagree with the numbers under it.
    """
    if table.empty:
        return "Role usage: no searched optimum to describe.\n"

    roles = table[(table["section"] == "role") & (table["tier"] == "all")]
    lines = [
        "What the optimal teams staff, and what it costs them",
        "=" * 78,
        f"  over {int(roles['n'].iloc[0])} admissible searched optima; every share "
        "formed within a scenario first",
        "",
        f"  {'role':<22}{'staffed':>9}{'head count':>12}{'bill':>9}{'median':>9}",
        "-" * 78,
    ]
    for _, row in roles.iterrows():
        label = AGENT_ROLE_ABBREV[int(row["role"]) - 1]
        lines.append(
            f"  {label:<22}{row['staffed_share']:>8.1%}{row['head_count_share']:>12.1%}"
            f"{row['bill_share']:>9.1%}{row['median_count']:>9.0f}")

    by_tier = table[(table["section"] == "role") & (table["tier"] != "all")]
    if not by_tier.empty:
        tiers = tiers_present(by_tier["tier"])
        lines.append("")
        lines.append(f"  Share of the agent bill, by {by_tier['group'].iloc[0]} tier")
        lines.append(f"  {'role':<22}" + "".join(f"{t:>13}" for t in tiers))
        lines.append("-" * 78)
        for k in range(1, AGENT_ROLE_COUNT + 1):
            sub = by_tier[by_tier["role"] == k].set_index("tier")
            cells = "".join(
                f"{sub.loc[t, 'bill_share']:>13.1%}" if t in sub.index else f"{'—':>13}"
                for t in tiers)
            lines.append(f"  {AGENT_ROLE_ABBREV[k - 1]:<22}{cells}")

    process = table[table["section"] == "process"]
    if not process.empty:
        lines.append("")
        lines.append("  The release process and the bug-fixing share")
        for _, row in process.iterrows():
            if row["quantity"] == "R":
                lines.append(f"    R = {int(row['value']):<24}{row['share']:>7.1%}")
            elif row["quantity"] == "f_fix":
                lines.append(f"    f_fix = {row['value']:<20.1f}{row['share']:>7.1%}")
            else:
                lines.append(f"    {'no feature freeze':<28}{row['share']:>7.1%}")

    istar = table[table["section"] == "istar"]
    if not istar.empty:
        stats = istar.set_index("quantity")
        lines.append("")
        lines.append(f"  Process spread within I* ({int(istar['n'].iloc[0])} searches)")
        lines.append(f"    {'distinct (N, R, F, f_fix)':<28}"
                     f"{stats.loc['n_triples_median', 'value']:>7.0f} (median)")
        lines.append(f"    {'R unique across the set':<28}"
                     f"{stats.loc['R_unique', 'share']:>7.1%}")
        lines.append(f"    {'N unique across the set':<28}"
                     f"{stats.loc['N_unique', 'share']:>7.1%}")
    lines.append("")
    return "\n".join(lines)
