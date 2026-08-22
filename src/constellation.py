"""What the optimal teams are made of, faceted by two scenario groups.

`src.campaign` answers what an optimum costs; this module answers what it is made of.  It
reduces the campaign to a mean and a standard deviation of every control variable — the
staffing level `N`, the release process `(R, F_freeze)` and the two 8-element composition
vectors — within each cell of a two-group tier grid.  The default grid is
`project` x `bugimpact`: 4 x 3 = 12 cells, each holding the 36 scenarios whose `human`,
`debt` and `agent` tiers are free.

Two populations are tabulated separately, since a campaign reports one optimum per
scenario without claiming it is the only one:

``c_star``
    The reported `c*` of each scenario — one row of ``optima.csv`` per scenario.

``i_star``
    The whole indistinguishable set `I*`, whose members are statistically tied with `c*`
    on cost.  This is the key list under ``indistinguishable``, *not* the wider
    ``final_objectives`` block; the members' objectives are looked up in that block.

The `i_star` statistics are scenario-balanced: each scenario's `I*` is reduced to one mean
vector first (:func:`scenario_means`), so both populations share a unit of observation and
an `n`, which makes their two `sd` columns comparable.

Teams with no agents are excluded from both populations: they have no composition to
describe.

Outputs, all under the campaign's analysis directory:

``indistinguishable.csv``
    One row per `I*` member, with the column names ``optima.csv`` uses, so the two join
    on ``(scenario_id, mode, config_key)``.  Written for every result mode in the tree;
    :func:`search_optima` does the filtering the cell statistics need.

``constellation_cells.csv``
    Long format: one row per (population, cell, parameter) with `mean`, `sd`,
    `n_scenarios` and `n_members`.

``constellation_summary.txt``
    The same numbers rendered wide as `mean +/- sd`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .campaign import (admissible_optima, composition_columns, require_search_dirs,
                       scenario_columns, scenario_params_for_id)
from .config import (
    AGENT_ROLE_COUNT,
    ALL_SCENARIO_TIERS,
    HUMAN_ONLY_MODE,
    SCENARIO_GROUPS,
    require_group_pair,
    SUPERVISED_ROLE_INDICES,
)

#: The two scenario groups whose tier grid the cells are cut along, by default.  Every
#: other group is free within a cell and contributes to its spread.
CELL_GROUPS: tuple[str, str] = ("project", "bugimpact")

#: The control variables in four blocks: the process triple, the two composition vectors
#: as one column per role (1-based on `k`, exactly as `campaign.optimum_row` writes them),
#: and the totals derived here.
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


# ----------------------------------------------------------------------------
# Extracting the indistinguishable set
# ----------------------------------------------------------------------------

def indistinguishable_rows(scenario_id: str, mode: str, summary: dict) -> list[dict]:
    """One row per member of the final `I*` of one run.

    The members are ``summary["indistinguishable"]``, the key list the selection ends on,
    looked up in ``final_objectives``, the `K_final` re-simulation they were re-ranked on.
    `is_best` flags `c*`, which is a member of every non-empty `I*`.  An inadmissible run
    has an empty `I*` and contributes no rows at all.
    """
    keys = summary["indistinguishable"]
    final = summary["final_objectives"]
    best = summary["best"]

    rows: list[dict] = []
    for key in keys:
        obj = final[key]
        row = scenario_columns(scenario_id, mode)
        row.update({
            "config_key": key,
            "is_best": key == best,
            "N": obj["N"], "R": obj["R"], "F_freeze": obj["F_freeze"],
        })
        row.update(composition_columns(obj))
        row["cost_mean"] = obj["cost_mean"]
        rows.append(row)
    return rows


def build_indistinguishable(results_dir: str | Path) -> pd.DataFrame:
    """Read every finished search under `results_dir` into one `I*` member table.

    Same tree and same discovery as `campaign.build_tables`, so a partially finished
    campaign simply contributes fewer rows.  Only ``search_summary.json`` is read, and
    only its `final_objectives` block.
    """
    searches = require_search_dirs(results_dir)

    rows: list[dict] = []
    for scenario_id, mode, d in searches:
        summary = json.loads((d / "search_summary.json").read_text(encoding="utf-8"))
        rows.extend(indistinguishable_rows(scenario_id, mode, summary))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# What each role costs
# ----------------------------------------------------------------------------

#: The per-role agent bill over the horizon in dollars, `actions_k * c_token_k`, one
#: column per role.
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

    ``actions_k`` is the work role `k` completed over the horizon in its own throughput
    unit and `c_token_k` prices that unit, so `BILL_PARAMETERS` is an exact per-role split
    rather than an apportionment.  Summing the eight reproduces
    `cost_agent_mean * F_completed` only to within a fraction of a per cent, since
    `cost_agent_mean` is a mean of per-replication ratios and this is a ratio of means;
    `AGENT_COST_PARAMETERS` sidesteps that by rescaling the split onto the reported
    figure.  `cost_team_mean` is the developers and the agents together.
    """
    frame = optima.copy()
    prices = {scenario_id: scenario_params_for_id(scenario_id, config_dir).c_token
              for scenario_id in frame["scenario_id"].unique()}
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
    reported = frame["cost_agent_mean"].to_numpy(dtype=float)
    for k, column in enumerate(AGENT_COST_PARAMETERS):
        frame[column] = share[:, k] * reported
    # The two cost drivers the composition controls; `cost_quality_mean` is not one.
    frame["cost_team_mean"] = frame["cost_human_mean"] + frame["cost_agent_mean"]
    return frame


# ----------------------------------------------------------------------------
# Cell statistics
# ----------------------------------------------------------------------------

def derive_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Add the four `TOTAL_PARAMETERS` to a frame that has the per-role columns.

    `n_agents_*` is the whole team size in that phase; `n_sup_*` is the supervised subset
    charged against the humans.  Idempotent — ``optima.csv`` already carries the latter.
    """
    out = df.copy()
    out["n_agents_dev"] = out[list(DEV_PARAMETERS)].sum(axis=1)
    out["n_agents_qa"] = out[list(QA_PARAMETERS)].sum(axis=1)
    # The parameter tuples run k=1..8, so the role indices address them directly.
    for phase, columns in (("dev", DEV_PARAMETERS), ("qa", QA_PARAMETERS)):
        supervised = [columns[i] for i in SUPERVISED_ROLE_INDICES]
        out[f"n_sup_{phase}"] = out[supervised].sum(axis=1)
    return out


def scenario_means(istar: pd.DataFrame) -> pd.DataFrame:
    """Reduce each scenario's `I*` to one mean vector, keeping its tiers.

    This is what makes the `i_star` cell statistics scenario-balanced: the cell mean is
    then a mean over scenarios, as the `c_star` one is, rather than over members.
    `n_members` records `|I*|`, so the pooled size stays visible in the output.
    """
    if istar.empty:
        return istar
    tier_columns = [f"tier_{group}" for group in SCENARIO_GROUPS]
    parameters = list(CONSTELLATION_PARAMETERS)
    grouped = istar.groupby(["scenario_id", "mode"], sort=True)
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

    Long format — one row per (population, cell, parameter) — because the parameter set
    is a list the caller may shorten and a wide table would then change shape.  `sd` is
    the sample standard deviation (ddof = 1) over the scenarios in the cell, i.e. their
    spread rather than Monte Carlo error; a cell with one scenario reports NaN.
    """
    groups = require_group_pair(groups)
    cell_columns = [f"tier_{group}" for group in groups]
    if df.empty:
        return pd.DataFrame(columns=["population", *cell_columns, "parameter",
                                     "mean", "sd", "n_scenarios", "n_members"])
    missing = [c for c in cell_columns if c not in df.columns]
    if missing:
        raise KeyError(f"{missing} are not columns of the table")

    parameters = list(CONSTELLATION_PARAMETERS)
    grouped = df.groupby(cell_columns, sort=False)
    stats = grouped[parameters].agg(["mean", "std"])
    counts = grouped.size().rename("n_scenarios")
    # `c_star` has one configuration per scenario, so its two counts coincide; `i_star`
    # carries the pooled `|I*|` per scenario in `n_members`.
    members = (grouped["n_members"].sum().rename("n_members")
               if "n_members" in df.columns else counts.rename("n_members"))

    long = (stats.stack(level=0, future_stack=True)
                 .rename(columns={"std": "sd"})
                 .rename_axis(index=[*cell_columns, "parameter"])
                 .reset_index())
    long = long.merge(counts.reset_index(), on=cell_columns, how="left")
    long = long.merge(members.reset_index(), on=cell_columns, how="left")
    long.insert(0, "population", population)
    return _sorted_cells(long, groups)


def _sorted_cells(long: pd.DataFrame, groups: tuple[str, ...]) -> pd.DataFrame:
    """Order the long table by tier (mild→extreme) and by parameter block."""
    tier_order = {name: i for i, name in enumerate(ALL_SCENARIO_TIERS)}
    parameter_order = {name: i for i, name in enumerate(CONSTELLATION_PARAMETERS)}
    tier_columns = [f"tier_{g}" for g in groups]
    out = long.sort_values(
        tier_columns + ["parameter"],
        key=lambda column: column.map(
            parameter_order if column.name == "parameter" else tier_order),
    )
    return out[["population", *tier_columns, "parameter",
                "mean", "sd", "n_scenarios", "n_members"]].reset_index(drop=True)


def search_optima(optima: pd.DataFrame) -> pd.DataFrame:
    """The rows of ``optima.csv`` that describe a searched, agent-supported team.

    The human-only baseline is dropped: its composition is the zero vector, so it has no
    constellation.  Filtered on the mode rather than on `is_human_only`, so a search that
    happened to report a zero-agent optimum would still be counted.
    """
    if optima.empty or "mode" not in optima.columns:
        return optima
    return optima[optima["mode"] != HUMAN_ONLY_MODE]


def build_constellation(
    optima: pd.DataFrame,
    istar: pd.DataFrame,
    groups: tuple[str, ...] = CELL_GROUPS,
) -> pd.DataFrame:
    """The two populations' cell statistics, stacked into one long table."""
    groups = require_group_pair(groups)
    frames = []
    for frame, population, balance in ((optima, C_STAR, False), (istar, I_STAR, True)):
        rows = search_optima(frame)
        if rows.empty:
            continue
        rows = derive_totals(rows)
        frames.append(cell_statistics(
            scenario_means(rows) if balance else rows, population, groups))

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
    group's tiers become the blocks.
    """
    groups = require_group_pair(groups)
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
             f"  {I_STAR:<8} the indistinguishable set I*, each scenario reduced to",
             "           its own mean first, so both populations weight scenarios equally",
             "",
             "Structural zeros to read past: a_qa_1 and a_qa_2 are 0 by construction, and the",
             "whole A_qa vector is 0 wherever F_freeze = 0 (TeamConfiguration canonicalises it),",
             "so a zero A_qa cell means 'no feature freeze', not 'a freeze with no agents'.",
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

#: The release cadences the search range offers, in the order the summary reports them.
_RELEASE_CADENCES: tuple[int, ...] = (1, 2, 4, 13)

#: The (N, R, F_freeze) triple that the parsimony tie-break treats as one process.
_PROCESS_KEY: tuple[str, ...] = ("N", "R", "F_freeze")


def role_usage(bills: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Per role: how often it is staffed, and its share of the head count and the bill.

    `bills` is a `role_bill` frame restricted to the searched optima.  Both shares are
    formed **within** a scenario before being averaged, since the optimal team size spans
    four orders of magnitude and a ratio of aggregates would describe the largest scenario
    rather than the typical one.  The two disagree sharply and systematically: the three
    autonomous roles are billed flat per sprint and bounded by nothing but their own
    saturation, so the search stacks them freely.

    With `by` given, the statistics are additionally split by that scenario group's tier.
    """
    if bills.empty:
        return pd.DataFrame()

    counts = bills[[f"a_dev_{k}" for k in range(1, AGENT_ROLE_COUNT + 1)]].to_numpy(float)
    bill = bills[list(BILL_PARAMETERS)].to_numpy(float)
    count_total = counts.sum(axis=1)
    bill_total = bill.sum(axis=1)
    count_share = np.divide(counts, count_total[:, None],
                            out=np.zeros_like(counts), where=count_total[:, None] > 0)
    bill_share = np.divide(bill, bill_total[:, None],
                           out=np.zeros_like(bill), where=bill_total[:, None] > 0)

    groups = (bills[f"tier_{by}"].to_numpy() if by is not None
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

    One row per release cadence plus one for "no feature freeze", each a share of the
    reported optima.  These are the two panels of the driver figure in which the tier rows
    barely move, so the marginal distribution is what says whether the process is being
    chosen at all or merely tolerated.
    """
    if optima.empty:
        return pd.DataFrame()
    n = len(optima)
    rows = [
        {"quantity": "R", "value": cadence,
         "share": float((optima["R"] == cadence).mean()), "n": n}
        for cadence in _RELEASE_CADENCES
        if (optima["R"] == cadence).any()
    ]
    rows.append({"quantity": "F_freeze", "value": 0,
                 "share": float((optima["F_freeze"] == 0).mean()), "n": n})
    return pd.DataFrame(rows)


def istar_process_diversity(istar: pd.DataFrame) -> pd.DataFrame:
    """Per search, how many distinct `(N, R, F_freeze)` triples its `I*` spans.

    A reported optimum is one member of a statistical tie, so the spread of the *process*
    within that tie is what says whether the search identified a process at all.  `R` or
    `N` being unique across the whole set means the tie agrees on it; the triple count
    says how much of the 99-point process grid the tie covers.
    """
    if istar.empty:
        return pd.DataFrame()
    searched = istar[istar["mode"] != HUMAN_ONLY_MODE]
    if searched.empty:
        return pd.DataFrame()
    rows = []
    for scenario_id, members in searched.groupby("scenario_id"):
        rows.append({
            "scenario_id": scenario_id,
            "n_members": len(members),
            "n_triples": int(members[list(_PROCESS_KEY)].drop_duplicates().shape[0]),
            "R_unique": bool(members["R"].nunique() == 1),
            "N_unique": bool(members["N"].nunique() == 1),
        })
    return pd.DataFrame(rows)


def build_process_summary(
    bills: pd.DataFrame,
    istar: pd.DataFrame,
    by: str = "agent",
) -> pd.DataFrame:
    """The long table behind `process_summary`: role usage, the process, and `I*` spread.

    `bills` is a `role_bill` frame; only its **admissible searched** rows are described.
    A zero-agent team has no composition and no agent bill, and an inadmissible search
    reports a team that does not deliver the work, so neither belongs in a description of
    what an optimal team is made of.
    """
    searched = admissible_optima(search_optima(bills))
    if searched.empty:
        return pd.DataFrame()

    parts = [frame.assign(section="role")
             for frame in (role_usage(searched), role_usage(searched, by=by))
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


def process_summary(table: pd.DataFrame, by: str = "agent") -> str:
    """Render `process_summary.csv` as a text block."""
    if table.empty:
        return "Role usage: no searched optimum to describe.\n"
    from .config import AGENT_ROLE_ABBREV

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
        tiers = [t for t in ALL_SCENARIO_TIERS if (by_tier["tier"] == t).any()]
        lines.append("")
        lines.append(f"  Share of the agent bill, by {by} tier")
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
        lines.append("  The release process")
        for _, row in process.iterrows():
            if row["quantity"] == "R":
                lines.append(f"    R = {int(row['value']):<24}{row['share']:>7.1%}")
            else:
                lines.append(f"    {'no feature freeze':<28}{row['share']:>7.1%}")

    istar = table[table["section"] == "istar"]
    if not istar.empty:
        stats = istar.set_index("quantity")
        lines.append("")
        lines.append(f"  Process spread within I* ({int(istar['n'].iloc[0])} searches)")
        lines.append(f"    {'distinct (N, R, F) triples':<28}"
                     f"{stats.loc['n_triples_median', 'value']:>7.0f} (median of 99)")
        lines.append(f"    {'R unique across the set':<28}"
                     f"{stats.loc['R_unique', 'share']:>7.1%}")
        lines.append(f"    {'N unique across the set':<28}"
                     f"{stats.loc['N_unique', 'share']:>7.1%}")
    lines.append("")
    return "\n".join(lines)
