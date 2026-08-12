"""Team compositions on the throughput-filtered quality/cost front (paper §5).

`src.results_analysis` answers what the front looks like in *objective* space; this
module answers what the teams *on* it look like.  It consumes the same flat table —
`CampaignTables.filtered_front`, i.e. `results/analysis/filtered_front.csv` — which
already carries `N`, `R`, `F_freeze`, `a_dev_1..8`, `a_qa_1..8`, `n_sup_dev` and
`n_sup_qa` next to the objectives, so nothing here re-simulates or re-derives.

Two normalisation rules shape every table below.

*Supervised and background roles are normalised differently.*  Per §3.3 the two
background detection services (k=5 Code Review, k=6 Security Audit) are unbounded —
their only brake is token cost — while the six supervised roles are held down by the
human-capacity condition of Eq. (feas_monitoring) and by their saturation bounds
``u_k``.  Background counts reach 77 on the front and would take ~77 % of any
stacked bar drawn over all eight roles, flattening the supervised roles into invisibility.
So supervised roles are reported as a **share of n_sup** (where the signal lives: the
Feature Implementation share falls from ~0.81 to ~0.26 between the cheapest and the
most expensive fifth of the front) and background roles as **absolute counts**.

*Every aggregate is computed per (scenario_id, strategy) front first, then averaged
across fronts with equal weight.*  Front sizes span 13 to 1,190 points and the
evolutionary strategy contributes an order of magnitude more points than random, so a
flat ``groupby.mean()`` over raw rows would describe the largest evolutionary fronts
rather than the average front.  :func:`_front_mean` is the single place that rule lives.

One correctness guard is worth stating loudly: **the QA composition is only meaningful
when ``F_freeze > 0``**.  When ``F_freeze == 0`` the freeze window is empty, ``T_qa`` is
zero and ``A_qa`` is never applied.  ``TeamConfiguration`` now canonicalises those rows to
the zero vector, so on a fresh campaign they carry no phantom agents — but the guard must
stay: campaigns written before that change hold an arbitrary unexercised vector there (on
the §4.2 campaign, 50,680 of the 50,727 ``F_freeze == 0`` front points), and averaging a
canonical zero into the QA panels would be just as wrong as averaging in a phantom.
:func:`phase_frame` drops them either way, and every public function calls it first.

Public API:
    phase_columns(phase)                        -> (all, supervised, background)
    phase_frame(df, phase)                      -> pd.DataFrame
    signature_label(signature, phase)           -> str
    add_composition_columns(df, phase, bins)    -> pd.DataFrame
    role_prevalence(df, phase)                  -> pd.DataFrame
    composition_by_front_position(df, phase, bins) -> pd.DataFrame
    composition_by_tier(df, phase)              -> pd.DataFrame
    archetypes(df, phase, top_k)                -> pd.DataFrame
    composition_headline(df, bins, top_k)       -> dict
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    AGENT_ROLE_ABBREV,
    AGENT_ROLE_COUNT,
    BACKGROUND_ROLE_INDICES,
    SCENARIO_GROUPS,
    SUPERVISED_ROLE_INDICES,
)

PHASES: tuple[str, ...] = ("dev", "qa")

DEFAULT_ARCHETYPES = 6
DEFAULT_BINS = 5

#: Label for the catch-all archetype holding everything past `top_k`.
OTHER_ARCHETYPE = "other"

#: Control variables reported alongside the role mix.  These are counts and lengths in
#: different units, so they are never mixed into the share bars — they get their own
#: panel in every figure.
CONTROL_COLUMNS: tuple[str, ...] = (
    "N", "R", "F_freeze", "n_sup", "n_bg", "n_agents", "agents_per_fte",
)

#: Roles 1 (Feature Implementation) and 2 (Architecture & Design) are structurally
#: absent from the freeze phase (`TeamConfiguration` enforces a_qa,1 = a_qa,2 = 0), so
#: the QA panels show six roles rather than eight.
_QA_EXCLUDED_ROLES: tuple[int, ...] = (1, 2)

_PHASE_BIN_LABELS = ("cheapest", "2nd", "3rd", "4th", "priciest")


def _check_phase(phase: str) -> None:
    if phase not in PHASES:
        raise ValueError(f"unknown phase {phase!r}; expected one of {list(PHASES)}")


def phase_columns(phase: str) -> tuple[list[str], list[str], list[str]]:
    """Return `(all_roles, supervised, background)` column names for `phase`.

    Columns are the `a_dev_k` / `a_qa_k` names of the flat table, k = 1..8 in role
    order.  The QA phase drops roles 1 and 2, which are identically zero there.
    """
    _check_phase(phase)
    excluded = _QA_EXCLUDED_ROLES if phase == "qa" else ()
    roles = [k for k in range(1, AGENT_ROLE_COUNT + 1) if k not in excluded]
    supervised = [k + 1 for k in SUPERVISED_ROLE_INDICES if k + 1 in roles]
    background = [k + 1 for k in BACKGROUND_ROLE_INDICES if k + 1 in roles]
    fmt = f"a_{phase}_%d"
    return (
        [fmt % k for k in roles],
        [fmt % k for k in supervised],
        [fmt % k for k in background],
    )


def role_labels(columns: list[str]) -> list[str]:
    """Compact role labels for `columns`, e.g. `a_dev_5` -> `Review`."""
    return [AGENT_ROLE_ABBREV[int(c.rsplit("_", 1)[1]) - 1] for c in columns]


def phase_frame(df: "pd.DataFrame", phase: str) -> "pd.DataFrame":
    """Restrict `df` to the rows in which `phase`'s composition is actually applied.

    For the development phase that is every row.  For the QA phase it is the rows with
    `F_freeze > 0`: with no freeze window the QA vector is never used, and the search
    leaves an arbitrary unexercised vector behind (see the module docstring).
    """
    _check_phase(phase)
    if phase == "dev":
        return df
    if "F_freeze" not in df.columns:
        raise KeyError("'F_freeze' is not a column of the front table")
    return df[df["F_freeze"] > 0]


def signature_label(signature: str, phase: str = "dev") -> str:
    """Render a role-presence signature as `Feature+Review+Security`.

    `signature` is the bitstring produced by :func:`add_composition_columns`, one
    character per role of `phase` in role order.
    """
    if signature == OTHER_ARCHETYPE:
        return OTHER_ARCHETYPE
    columns, _, _ = phase_columns(phase)
    labels = role_labels(columns)
    if len(signature) != len(labels):
        raise ValueError(
            f"signature {signature!r} has {len(signature)} bits, but the {phase} phase "
            f"has {len(labels)} roles"
        )
    present = [lab for bit, lab in zip(signature, labels) if bit == "1"]
    return "+".join(present) if present else "none"


def add_composition_columns(
    df: "pd.DataFrame", phase: str, bins: int = DEFAULT_BINS,
) -> "pd.DataFrame":
    """Attach the derived composition columns for `phase` to a copy of `df`.

    Adds, for the roles of `phase`:
        `n_sup`, `n_bg`, `n_agents`  — supervised / background / total agent counts
        `agents_per_fte`            — n_sup / N, the monitoring load one FTE carries
        `share_<role column>`       — supervised role share of n_sup (NaN when n_sup = 0)
        `signature`                 — role-presence bitstring, one bit per role
        `front_pos`                 — percentile rank of cost_mean within the row's front
        `front_bin`                 — `front_pos` cut into `bins` equal-width quantiles

    `n_sup` is taken from the persisted `n_sup_dev` / `n_sup_qa` column when present
    rather than recomputed, so this stays consistent with `supervised_agent_count`.
    """
    all_cols, sup_cols, bg_cols = phase_columns(phase)
    out = phase_frame(df, phase).copy()
    missing = [c for c in all_cols if c not in out.columns]
    if missing:
        raise KeyError(f"{missing} are not columns of the front table")

    persisted_n_sup = f"n_sup_{phase}"
    if persisted_n_sup in out.columns:
        out["n_sup"] = out[persisted_n_sup].astype(float)
    else:
        out["n_sup"] = out[sup_cols].sum(axis=1).astype(float)
    out["n_bg"] = out[bg_cols].sum(axis=1).astype(float)
    out["n_agents"] = out["n_sup"] + out["n_bg"]
    # Supervised agents per FTE: exactly the quantity the human-capacity condition of
    # Eq. (feas_monitoring) bounds, so it is comparable across staffing levels without
    # needing the scenario parameters that fix where that bound sits.
    out["agents_per_fte"] = out["n_sup"] / out["N"]

    # Shares are of the supervised roles only; a team with no supervised agents has no
    # meaningful mix, so it contributes NaN rather than a zero that would drag means down.
    denom = out["n_sup"].replace(0.0, np.nan)
    for col in sup_cols:
        out[f"share_{col}"] = out[col] / denom

    out["signature"] = _signature_series(out, all_cols)
    out["front_pos"] = out.groupby(["scenario_id", "strategy"])["cost_mean"].rank(pct=True)
    out["front_bin"] = _bin_front_position(out["front_pos"], bins)
    return out


def _signature_series(df: "pd.DataFrame", columns: list[str]) -> "pd.Series":
    """Role-presence bitstring per row, one character per column of `columns`."""
    bits = (df[columns].to_numpy() > 0).astype(np.uint8)
    return pd.Series(
        ["".join(map(str, row)) for row in bits], index=df.index, dtype="string",
    )


def _bin_front_position(front_pos: "pd.Series", bins: int) -> "pd.Series":
    """Cut a percentile rank into `bins` ordered bins labelled cheapest -> priciest."""
    if bins < 1:
        raise ValueError(f"bins must be >= 1, got {bins}")
    if bins == len(_PHASE_BIN_LABELS):
        labels = list(_PHASE_BIN_LABELS)
    else:
        labels = [f"{i + 1}/{bins}" for i in range(bins)]
    edges = np.linspace(0.0, 1.0, bins + 1)
    return pd.cut(front_pos, bins=edges, labels=labels, include_lowest=True)


def _front_mean(
    df: "pd.DataFrame", value_columns: list[str], by: list[str],
) -> "pd.DataFrame":
    """Mean of `value_columns` over `by`, weighting every front equally.

    Each (scenario_id, strategy) front is averaged first, so a 1,190-point front and a
    13-point front contribute the same amount to the result.
    """
    keys = ["scenario_id", "strategy"] + [c for c in by if c not in ("scenario_id", "strategy")]
    per_front = df.groupby(keys, observed=True)[value_columns].mean()
    return per_front.groupby(by, observed=True).mean().reset_index()


def role_prevalence(df: "pd.DataFrame", phase: str) -> "pd.DataFrame":
    """How often each role is staffed at all, and how heavily when it is.

    One row per (strategy, role) with:
        `presence_rate`     — fraction of front points with a nonzero count
        `mean_count`        — mean agent count over all front points
        `mean_count_present`— mean agent count over the points that staff the role
        `mean_share`        — mean share of n_sup, supervised roles only (NaN otherwise)

    A role can be near-universal but tiny (Feature Implementation) or rare but large,
    which is exactly the distinction a stacked bar of shares cannot show.
    """
    all_cols, sup_cols, _ = phase_columns(phase)
    work = add_composition_columns(df, phase)
    if work.empty:
        return pd.DataFrame(
            columns=["strategy", "role", "role_column", "supervised", "presence_rate",
                     "mean_count", "mean_count_present", "mean_share"]
        )

    present = work[all_cols].gt(0).astype(float).add_prefix("present_")
    staffed = work[all_cols].where(work[all_cols] > 0).add_prefix("staffed_")
    work = pd.concat([work, present, staffed], axis=1)

    share_cols = [f"share_{c}" for c in sup_cols]
    agg = _front_mean(
        work,
        [f"present_{c}" for c in all_cols]
        + all_cols
        + [f"staffed_{c}" for c in all_cols]
        + share_cols,
        by=["strategy"],
    )

    rows = []
    for _, rec in agg.iterrows():
        for col, label in zip(all_cols, role_labels(all_cols)):
            rows.append({
                "phase": phase,
                "strategy": rec["strategy"],
                "role": label,
                "role_column": col,
                "supervised": col in sup_cols,
                "presence_rate": rec[f"present_{col}"],
                "mean_count": rec[col],
                "mean_count_present": rec[f"staffed_{col}"],
                "mean_share": rec[f"share_{col}"] if col in sup_cols else np.nan,
            })
    return pd.DataFrame(rows)


def composition_by_front_position(
    df: "pd.DataFrame", phase: str, bins: int = DEFAULT_BINS,
) -> "pd.DataFrame":
    """Mean composition per within-front cost quantile, per strategy.

    Points are ranked by `cost_mean` *within* their own (scenario, strategy) front and
    cut into `bins` quantiles, so the cheap end of a cheap scenario is compared with the
    cheap end of an expensive one rather than across scenarios.
    """
    _, sup_cols, bg_cols = phase_columns(phase)
    work = add_composition_columns(df, phase, bins=bins)
    if work.empty:
        return pd.DataFrame()
    share_cols = [f"share_{c}" for c in sup_cols]
    value_cols = share_cols + bg_cols + list(CONTROL_COLUMNS) + ["quality_mean", "cost_mean"]
    out = _front_mean(work, value_cols, by=["strategy", "front_bin"])
    out.insert(0, "phase", phase)
    return out


def composition_by_tier(df: "pd.DataFrame", phase: str) -> "pd.DataFrame":
    """Mean composition per scenario group tier, long-form, with grand-mean deviations.

    One row per (strategy, group, tier, quantity).  `deviation` is the value minus the
    mean of that quantity over the tiers of its group, which is what the heat map
    colours: it isolates the tier effect from the wildly different magnitudes of, say, a
    role share and an agent count.
    """
    _, sup_cols, bg_cols = phase_columns(phase)
    work = add_composition_columns(df, phase)
    if work.empty:
        return pd.DataFrame()

    share_cols = [f"share_{c}" for c in sup_cols]
    value_cols = share_cols + bg_cols + list(CONTROL_COLUMNS)

    frames = []
    for group in SCENARIO_GROUPS:
        column = f"tier_{group}"
        if column not in work.columns:
            raise KeyError(f"{column!r} is not a column of the front table")
        agg = _front_mean(work, value_cols, by=["strategy", column])
        agg = agg.rename(columns={column: "tier"})
        agg.insert(0, "group", group)
        frames.append(agg)

    wide = pd.concat(frames, ignore_index=True)
    long = wide.melt(
        id_vars=["group", "tier", "strategy"],
        value_vars=value_cols,
        var_name="quantity",
        value_name="value",
    )
    long["kind"] = np.select(
        [long["quantity"].isin(share_cols), long["quantity"].isin(bg_cols)],
        ["share", "background"],
        default="control",
    )
    grand = long.groupby(["strategy", "group", "quantity"], observed=True)["value"].transform("mean")
    long["deviation"] = long["value"] - grand
    long.insert(0, "phase", phase)
    return long


def archetypes(
    df: "pd.DataFrame", phase: str, top_k: int = DEFAULT_ARCHETYPES,
) -> "pd.DataFrame":
    """The recurring team recipes on the front, ranked by prevalence.

    An archetype is a *role-presence signature*: which roles are staffed at all,
    ignoring how many.  On the §4.2 campaign only 43 distinct dev-phase signatures occur
    and the six most common cover ~95 % of front points, so this partitions the front
    into a handful of named recipes without clustering — deterministic, dependency-free
    and directly readable.  Everything past `top_k` is folded into `OTHER_ARCHETYPE`.

    Returns one row per (strategy, archetype) carrying `prevalence` (share of that
    strategy's front points), `scenario_coverage` (fraction of the scenarios in which the
    archetype appears at all — a recipe that shows up everywhere is a different claim
    from one that only wins in a corner), the mean objectives, and the mean composition.
    Rows are sorted by mean quality so the ladder reads best-quality first.
    """
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    _, sup_cols, bg_cols = phase_columns(phase)
    work = add_composition_columns(df, phase)
    if work.empty:
        return pd.DataFrame()

    share_cols = [f"share_{c}" for c in sup_cols]
    value_cols = share_cols + bg_cols + list(CONTROL_COLUMNS) + ["quality_mean", "cost_mean"]

    frames = []
    for strategy, sub in work.groupby("strategy", observed=True):
        counts = sub["signature"].value_counts()
        kept = list(counts.index[:top_k])
        labelled = sub.assign(
            archetype=sub["signature"].where(sub["signature"].isin(kept), OTHER_ARCHETYPE)
        )
        n_scenarios = labelled["scenario_id"].nunique()

        # Prevalence and coverage are point counts, not front means: "how much of the
        # front is this recipe" and "in how many scenarios does it occur".
        stats = labelled.groupby("archetype", observed=True).agg(
            n_points=("archetype", "size"),
            n_scenarios=("scenario_id", "nunique"),
        )
        stats["prevalence"] = stats["n_points"] / len(labelled)
        stats["scenario_coverage"] = stats["n_scenarios"] / max(1, n_scenarios)

        means = _front_mean(labelled, value_cols, by=["strategy", "archetype"])
        merged = means.merge(stats.reset_index(), on="archetype", how="left")
        frames.append(merged)

    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "phase", phase)
    out.insert(3, "label", [
        signature_label(sig, phase) for sig in out["archetype"]
    ])
    # `other` is a residue, not a recipe — keep it last regardless of its mean quality.
    out["_is_other"] = out["archetype"].eq(OTHER_ARCHETYPE)
    out = out.sort_values(
        ["strategy", "_is_other", "quality_mean"], ascending=[True, True, True]
    ).drop(columns="_is_other")
    return out.reset_index(drop=True)


def composition_headline(
    df: "pd.DataFrame", bins: int = DEFAULT_BINS, top_k: int = DEFAULT_ARCHETYPES,
) -> dict[str, object]:
    """The composition numbers §5 would quote in prose, for `headline_summary.json`."""
    out: dict[str, object] = {}
    for phase in PHASES:
        work = add_composition_columns(df, phase, bins=bins)
        if work.empty:
            out[phase] = {"n_points": 0}
            continue
        _, sup_cols, _ = phase_columns(phase)
        arch = archetypes(df, phase, top_k=top_k)
        drift = composition_by_front_position(df, phase, bins=bins)

        entry: dict[str, object] = {
            "n_points": int(len(work)),
            "n_distinct_signatures": int(work["signature"].nunique()),
            "mean_n_sup": float(work["n_sup"].mean()),
            "mean_n_bg": float(work["n_bg"].mean()),
            "mean_agents_per_fte": float(work["agents_per_fte"].mean()),
        }
        for strategy, sub in arch.groupby("strategy", observed=True):
            real = sub[sub["archetype"] != OTHER_ARCHETYPE]
            top = real.nlargest(1, "prevalence")
            if top.empty:
                continue
            row = top.iloc[0]
            entry[f"top_archetype_{strategy}"] = {
                "signature": str(row["archetype"]),
                "label": str(row["label"]),
                "prevalence": float(row["prevalence"]),
                "scenario_coverage": float(row["scenario_coverage"]),
                "quality_mean": float(row["quality_mean"]),
                "cost_mean": float(row["cost_mean"]),
            }
            entry[f"top_{top_k}_coverage_{strategy}"] = float(real["prevalence"].sum())

        # The headline drift claim: how the supervised mix tilts from the cheap end of
        # the front to the expensive end.
        if not drift.empty:
            first, last = drift["front_bin"].cat.categories[0], drift["front_bin"].cat.categories[-1]
            tilt = {}
            for col in sup_cols:
                lo = drift.loc[drift["front_bin"] == first, f"share_{col}"].mean()
                hi = drift.loc[drift["front_bin"] == last, f"share_{col}"].mean()
                tilt[AGENT_ROLE_ABBREV[int(col.rsplit("_", 1)[1]) - 1]] = {
                    "cheapest_bin": float(lo), "priciest_bin": float(hi),
                }
            entry["supervised_share_drift"] = tilt
        out[phase] = entry
    return out
