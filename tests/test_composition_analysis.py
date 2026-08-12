"""Tests for `src.composition_analysis` (paper §5, team compositions).

The module derives every composition table from the throughput-filtered front, and the
two rules it has to get right are easy to break silently: QA compositions are only
meaningful when a freeze window exists, and every aggregate weights fronts equally
rather than points equally.  Both get a dedicated regression test below.
"""

from __future__ import annotations


import numpy as np
import pandas as pd
import pytest

from src.composition_analysis import (
    CONTROL_COLUMNS,
    OTHER_ARCHETYPE,
    add_composition_columns,
    archetypes,
    composition_by_front_position,
    composition_by_tier,
    composition_headline,
    phase_columns,
    phase_frame,
    role_prevalence,
    signature_label,
)
from src.config import SCENARIO_GROUPS, scenario_tier_combinations, scenario_tiers


def _row(
    tiers: dict[str, str],
    A_dev: tuple[int, ...],
    A_qa: tuple[int, ...],
    *,
    scenario_id: str,
    strategy: str = "random",
    N: int = 2,
    R: int = 4,
    F_freeze: int = 1,
    quality: float = 10.0,
    cost: float = 100.0,
) -> dict:
    """One filtered-front row with the columns the composition tables read."""
    row = {f"tier_{group}": tiers[group] for group in SCENARIO_GROUPS}
    row.update(
        scenario_id=scenario_id, strategy=strategy,
        N=N, R=R, F_freeze=F_freeze,
        quality_mean=quality, cost_mean=cost,
        n_sup_dev=sum(A_dev[i] for i in (0, 1, 2, 3, 6, 7)),
        n_sup_qa=sum(A_qa[i] for i in (0, 1, 2, 3, 6, 7)),
    )
    row.update({f"a_dev_{k}": A_dev[k - 1] for k in range(1, 9)})
    row.update({f"a_qa_{k}": A_qa[k - 1] for k in range(1, 9)})
    return row


@pytest.fixture
def synthetic_front() -> pd.DataFrame:
    """Four points per tier combination, spanning two archetypes and both strategies."""
    lean = (1, 0, 0, 0, 6, 4, 0, 0)          # Feature + Review + Security
    rich = (2, 1, 1, 1, 8, 6, 2, 1)          # every role
    qa_lean = (0, 0, 1, 0, 5, 3, 0, 0)
    qa_rich = (0, 0, 2, 1, 7, 5, 1, 1)

    rows = []
    for combo in scenario_tier_combinations():
        tiers = dict(zip(SCENARIO_GROUPS, combo))
        sid = "_".join(f"{g[0]}-{tiers[g][0]}" for g in SCENARIO_GROUPS)
        for strategy in ("random", "evolutionary"):
            rows.append(_row(tiers, lean, qa_lean, scenario_id=sid, strategy=strategy,
                             quality=50.0, cost=100.0))
            rows.append(_row(tiers, rich, qa_rich, scenario_id=sid, strategy=strategy,
                             quality=5.0, cost=900.0))
    return pd.DataFrame(rows)


# -- phase handling ---------------------------------------------------------

def test_phase_columns_dev_covers_all_eight_roles():
    all_cols, sup, bg = phase_columns("dev")
    assert all_cols == [f"a_dev_{k}" for k in range(1, 9)]
    assert sup == ["a_dev_1", "a_dev_2", "a_dev_3", "a_dev_4", "a_dev_7", "a_dev_8"]
    assert bg == ["a_dev_5", "a_dev_6"]


def test_phase_columns_qa_drops_the_structurally_absent_roles():
    all_cols, sup, bg = phase_columns("qa")
    # a_qa,1 and a_qa,2 are pinned to zero by TeamConfiguration, so they carry no signal.
    assert all_cols == [f"a_qa_{k}" for k in (3, 4, 5, 6, 7, 8)]
    assert sup == ["a_qa_3", "a_qa_4", "a_qa_7", "a_qa_8"]
    assert bg == ["a_qa_5", "a_qa_6"]


def test_unknown_phase_rejected():
    with pytest.raises(ValueError, match="unknown phase"):
        phase_columns("qa_phase")


def test_phase_frame_drops_unexercised_qa_vectors(synthetic_front):
    """The regression guard: with F_freeze = 0 the QA vector is never applied.

    The search still writes an arbitrary vector into those rows, so counting them would
    describe compositions the simulation never ran.
    """
    df = synthetic_front.copy()
    df.loc[df.index[: len(df) // 2], "F_freeze"] = 0

    assert len(phase_frame(df, "dev")) == len(df)

    qa = phase_frame(df, "qa")
    assert len(qa) == len(df) - len(df) // 2
    assert (qa["F_freeze"] > 0).all()


def test_phase_frame_requires_f_freeze_for_qa(synthetic_front):
    with pytest.raises(KeyError, match="F_freeze"):
        phase_frame(synthetic_front.drop(columns="F_freeze"), "qa")


# -- derived columns --------------------------------------------------------

def test_supervised_shares_sum_to_one_and_exclude_background(synthetic_front):
    work = add_composition_columns(synthetic_front, "dev")
    _, sup_cols, bg_cols = phase_columns("dev")

    shares = work[[f"share_{c}" for c in sup_cols]]
    assert np.allclose(shares.sum(axis=1), 1.0)
    # Background roles are counts, never shares — they must not appear in the mix.
    assert not any(f"share_{c}" in work.columns for c in bg_cols)

    lean = work[work["a_dev_2"] == 0].iloc[0]
    assert lean["n_sup"] == 1          # only the Feature agent is supervised
    assert lean["n_bg"] == 10          # 6 review + 4 security
    # n_sup / N: one supervised agent carried by a 2-FTE team.
    assert lean["agents_per_fte"] == pytest.approx(1 / 2)


def test_n_sup_is_taken_from_the_persisted_column(synthetic_front):
    """`n_sup_dev` is authoritative; recomputing it would let the two drift apart."""
    df = synthetic_front.copy()
    df["n_sup_dev"] = 99.0
    assert (add_composition_columns(df, "dev")["n_sup"] == 99.0).all()


def test_missing_role_column_rejected(synthetic_front):
    with pytest.raises(KeyError, match="a_dev_4"):
        add_composition_columns(synthetic_front.drop(columns="a_dev_4"), "dev")


# -- weighting --------------------------------------------------------------

def test_fronts_are_weighted_equally_not_points():
    """A 100-point front and a 2-point front must move a mean by the same amount.

    Real fronts range from 13 to 1,190 points, so point-weighting would report the
    largest fronts rather than the average front.
    """
    tiers = {g: "low" for g in SCENARIO_GROUPS}
    lean = (1, 0, 0, 0, 6, 4, 0, 0)
    rich = (0, 0, 0, 0, 6, 4, 1, 0)      # supervised capacity entirely on Test

    rows = [_row(tiers, lean, lean, scenario_id="big") for _ in range(100)]
    rows += [_row(tiers, rich, rich, scenario_id="small") for _ in range(2)]
    df = pd.DataFrame(rows)

    prevalence = role_prevalence(df, "dev").set_index("role")
    # Point-weighted this would be 100/102 ≈ 0.98; front-weighted it is (1 + 0) / 2.
    assert prevalence.loc["Feature", "presence_rate"] == pytest.approx(0.5)
    assert prevalence.loc["Test", "presence_rate"] == pytest.approx(0.5)


# -- archetypes -------------------------------------------------------------

def test_signature_label_names_the_present_roles():
    assert signature_label("10001100") == "Feature+Review+Security"
    assert signature_label("11111111") == (
        "Feature+Design+BugFix+Refactor+Review+Security+Test+Docs"
    )
    assert signature_label(OTHER_ARCHETYPE) == OTHER_ARCHETYPE


def test_signature_label_checks_width_against_the_phase():
    assert signature_label("001100", "qa") == "Review+Security"
    with pytest.raises(ValueError, match="has 8 bits"):
        signature_label("10001100", "qa")


def test_archetypes_partition_the_front(synthetic_front):
    table = archetypes(synthetic_front, "dev")
    for _, sub in table.groupby("strategy"):
        assert sub["prevalence"].sum() == pytest.approx(1.0)
        assert set(sub["label"]) == {
            "Feature+Review+Security",
            "Feature+Design+BugFix+Refactor+Review+Security+Test+Docs",
        }
        # Both recipes occur in every scenario of the synthetic campaign.
        assert np.allclose(sub["scenario_coverage"].to_numpy(), 1.0)


def test_archetypes_sort_best_quality_first(synthetic_front):
    table = archetypes(synthetic_front, "dev")
    for _, sub in table.groupby("strategy"):
        assert sub["quality_mean"].is_monotonic_increasing


def test_top_k_folds_the_remainder_into_other(synthetic_front):
    table = archetypes(synthetic_front, "dev", top_k=1)
    for _, sub in table.groupby("strategy"):
        assert len(sub) == 2
        assert sub["archetype"].iloc[-1] == OTHER_ARCHETYPE   # residue stays last
        assert sub["prevalence"].sum() == pytest.approx(1.0)


def test_top_k_must_be_positive(synthetic_front):
    with pytest.raises(ValueError, match="top_k"):
        archetypes(synthetic_front, "dev", top_k=0)


# -- the position and tier tables ------------------------------------------

def test_front_position_bins_span_the_front(synthetic_front):
    table = composition_by_front_position(synthetic_front, "dev", bins=2)
    assert list(table["front_bin"].astype(str).unique()) == ["1/2", "2/2"]
    for column in CONTROL_COLUMNS:
        assert column in table.columns

    # The cheap half is the lean recipe, the expensive half the rich one.
    cheap = table[table["front_bin"].astype(str) == "1/2"]
    pricey = table[table["front_bin"].astype(str) == "2/2"]
    assert cheap["share_a_dev_1"].mean() > pricey["share_a_dev_1"].mean()
    assert cheap["n_sup"].mean() < pricey["n_sup"].mean()


def test_tier_table_covers_every_group_and_tier(synthetic_front):
    table = composition_by_tier(synthetic_front, "dev")
    assert set(table["group"]) == set(SCENARIO_GROUPS)
    assert set(table["tier"]) == {t for g in SCENARIO_GROUPS for t in scenario_tiers(g)}
    assert set(table["kind"]) == {"share", "background", "control"}
    # Deviations are taken against the group's own tier mean, so they cancel out.
    summed = table.groupby(["strategy", "group", "quantity"])["deviation"].sum()
    assert np.allclose(summed.to_numpy(), 0.0, atol=1e-12)


def test_headline_reports_both_phases(synthetic_front):
    summary = composition_headline(synthetic_front)
    assert set(summary) == {"dev", "qa"}
    assert summary["dev"]["n_distinct_signatures"] == 2
    top = summary["dev"]["top_archetype_random"]
    assert top["prevalence"] == pytest.approx(0.5)
    assert "supervised_share_drift" in summary["dev"]


def test_empty_front_is_handled(synthetic_front):
    empty = synthetic_front.iloc[:0]
    assert archetypes(empty, "dev").empty
    assert composition_by_tier(empty, "dev").empty
    assert composition_by_front_position(empty, "dev").empty
    assert role_prevalence(empty, "dev").empty
    assert composition_headline(empty)["dev"] == {"n_points": 0}
