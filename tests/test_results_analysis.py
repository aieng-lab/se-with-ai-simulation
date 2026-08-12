"""Tests for the §5 campaign analysis (`src.results_analysis`)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd
import pytest

from src.config import AGENT_ROLE_COUNT, SCENARIO_GROUPS, SimulationHorizon, TeamConfiguration
from src.pareto import AggregatedObjectives, pareto_frontier
from src.results_analysis import (
    _legacy_cost_components,
    DEFAULT_THROUGHPUT_THRESHOLD,
    add_derived_columns,
    add_front_position,
    add_quality_cost_index,
    build_campaign,
    build_tables,
    candidate_rows,
    filtered_front_from_table,
    frontier_extremes,
    frontier_rows,
    headline_summary,
    index_optimum,
    load_frontier_table,
    quality_cost_frontier,
    rank_frontier,
    scenario_params_for_id,
    tiers_from_scenario_id,
    value_deviation_patterns,
    value_saturation_summary,
)
from src.simulation import cost_components, phase_sprint_counts, run_configuration

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CAMPAIGN_DIR = _REPO_ROOT / "results" / "experiments"


# ---------------------------------------------------------------------------
# cost_components / phase_sprint_counts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("R, F_freeze, T, expected", [
    (4, 0, 12, (12, 0)),      # no freeze: every sprint is a dev sprint
    (4, 1, 12, (9, 3)),       # last sprint of each of the 3 cycles
    (4, 2, 12, (6, 6)),
    (1, 0, 10, (10, 0)),      # continuous delivery admits no freeze
    (13, 2, 26, (22, 4)),
    (4, 1, 10, (8, 2)),       # horizon not a multiple of R
])
def test_phase_sprint_counts(base_config, R, F_freeze, T, expected):
    config = dataclasses.replace(base_config, R=R, F_freeze=F_freeze)
    assert phase_sprint_counts(config, T) == expected
    assert sum(phase_sprint_counts(config, T)) == T


def test_cost_components_is_the_inner_product_of_work_and_unit_cost(base_config, base_params):
    """Eq. (cost_agents): agents are billed for the work they performed."""
    T = 26
    params = dataclasses.replace(
        base_params,
        c_salary=150_000.0,
        c_token=(3.8, 19.0, 0.9, 0.9, 0.9, 9.0, 0.9, 0.9),
    )
    actions = (10.0, 2.0, 7.0, 0.5, 26.0, 26.0, 300.0, 120.0)
    C_human, C_agents = cost_components(base_config, params, T, actions)

    assert C_human == pytest.approx(base_config.N * params.c_salary * T / 52)
    assert C_agents == pytest.approx(
        sum(actions[k] * params.c_token[k] for k in range(AGENT_ROLE_COUNT))
    )


def test_cost_components_rejects_a_wrong_length_action_vector(base_config, base_params):
    with pytest.raises(ValueError, match="actions must have length"):
        cost_components(base_config, base_params, 10, (1.0, 2.0))


@pytest.mark.parametrize("R, F_freeze", [(1, 0), (4, 0), (4, 1), (4, 2), (13, 2)])
def test_cost_components_reproduces_simulated_cost(base_config, base_params, R, F_freeze):
    """The extracted helper must charge exactly what run_configuration charges."""
    config = dataclasses.replace(base_config, R=R, F_freeze=F_freeze)
    params = dataclasses.replace(
        base_params,
        c_salary=120_000.0,
        c_token=(2.0, 5.0, 1.0, 1.0, 0.5, 4.0, 1.0, 0.5),
    )
    settings = SimulationHorizon(T=12, K=4, master_seed=7)

    result = run_configuration(config, settings, params)
    for rep in result.replications:
        # Each replication carries the work it actually did, so the helper must
        # reproduce that replication's own bill.
        C_human, C_agents = cost_components(config, params, settings.T, rep.actions)
        F_completed = round(rep.value * settings.T)
        assert rep.cost == pytest.approx((C_human + C_agents) / max(1, F_completed))
        assert rep.cost_human + rep.cost_agent == pytest.approx(rep.cost)


# ---------------------------------------------------------------------------
# scenario ids and parameters
# ---------------------------------------------------------------------------

def test_tiers_from_scenario_id_round_trip():
    assert tiers_from_scenario_id("h-l_p-m_d-h_a-l_b-m") == {
        "human": "low", "project": "medium", "debt": "high",
        "agent": "low", "bugimpact": "medium",
    }


def test_tiers_from_scenario_id_reads_the_very_high_project_tier():
    assert tiers_from_scenario_id("h-l_p-v_d-h_a-l_b-m")["project"] == "veryhigh"


@pytest.mark.parametrize("bad", [
    "h-l_p-m_d-h_a-l",          # too few tokens
    "h-l_p-m_d-h_a-l_b-m_x-l",  # too many tokens
    "h-x_p-m_d-h_a-l_b-m",      # unknown tier initial
    "h-l_p-m_d-h_a-l_z-m",      # group initial does not match position
    "hl_p-m_d-h_a-l_b-m",       # missing separator
])
def test_tiers_from_scenario_id_rejects_malformed(bad):
    with pytest.raises(ValueError):
        tiers_from_scenario_id(bad)


def test_scenario_params_for_id_reads_the_right_tiers():
    params = scenario_params_for_id("h-h_p-l_d-m_a-m_b-h", str(_REPO_ROOT / "config"))
    assert params.r_arrival == 3.0          # project low
    assert params.c_salary == 250_000.0     # human high
    assert params.omega_maj == 28_000.0     # bugimpact high


def test_scenario_params_for_id_is_cached():
    a = scenario_params_for_id("h-m_p-m_d-m_a-m_b-m", str(_REPO_ROOT / "config"))
    b = scenario_params_for_id("h-m_p-m_d-m_a-m_b-m", str(_REPO_ROOT / "config"))
    assert a is b


# ---------------------------------------------------------------------------
# Synthetic frontier table
# ---------------------------------------------------------------------------

def _row(scenario_id, strategy, key, *, N=1, R=4, F_freeze=0,
         value, quality, cost, T=104, a_dev_1=1, ci_width=0.0,
         agent_share=0.25, legacy=False, contributions=None):
    """One flat-table row.

    `agent_share` sets the recorded human/agent cost split that the simulator now
    persists; `legacy=True` omits those columns, reproducing a campaign written before
    the usage-based cost model so the fallback path can be tested.

    `contributions` adds the four who-did-what counters as
    ``(impl_human, impl_agents, fixes_human, fixes_agent)``.  They are omitted by
    default, which reproduces the §4.2 campaign — written before those counters existed
    — so the "no shares without inputs" path stays covered.
    """
    tiers = tiers_from_scenario_id(scenario_id)
    row = {"scenario_id": scenario_id, "strategy": strategy, "config_key": key,
           "N": N, "R": R, "F_freeze": F_freeze, "T": T}
    row.update({f"tier_{g}": tiers[g] for g in SCENARIO_GROUPS})
    for k in range(1, AGENT_ROLE_COUNT + 1):
        row[f"a_dev_{k}"] = a_dev_1 if k == 1 else 0
        row[f"a_qa_{k}"] = 1 if k == 3 else 0
    for name, mean in (("value", value), ("quality", quality), ("cost", cost)):
        row[f"{name}_mean"] = mean
        row[f"{name}_ci_lo"] = mean - ci_width
        row[f"{name}_ci_hi"] = mean + ci_width
    if not legacy:
        row["cost_agent_mean"] = cost * agent_share
        row["cost_human_mean"] = cost * (1.0 - agent_share)
    if contributions is not None:
        row["impl_human_mean"], row["impl_agents_mean"] = contributions[:2]
        row["fixes_human_mean"], row["fixes_agent_mean"] = contributions[2:]
    return row


@pytest.fixture
def synthetic_frontier():
    """Two searches over one scenario; `r_arrival` is 12.0 (project medium).

    The two `low` points sit clearly below the arrival rate, the rest sit on it.
    """
    sid = "h-m_p-m_d-m_a-m_b-m"
    rows = [
        _row(sid, "random", "hi_1", value=12.0, quality=100.0, cost=1000.0),
        _row(sid, "random", "hi_2", value=12.0, quality=200.0, cost=500.0),
        _row(sid, "random", "low_1", value=9.6, quality=50.0, cost=2000.0, F_freeze=2),
        _row(sid, "evolutionary", "hi_3", value=12.0, quality=400.0, cost=250.0),
        _row(sid, "evolutionary", "low_2", value=10.8, quality=80.0, cost=1500.0, F_freeze=2),
    ]
    return pd.DataFrame(rows)


def test_add_derived_columns(synthetic_frontier):
    df = add_derived_columns(synthetic_frontier, config_dir=str(_REPO_ROOT / "config"))

    assert (df["r_arrival"] == 12.0).all()
    assert df.loc[df.config_key == "hi_1", "value_ratio"].item() == pytest.approx(1.0)
    assert df.loc[df.config_key == "low_1", "value_ratio"].item() == pytest.approx(0.8)
    assert df.loc[df.config_key == "low_1", "value_deficit"].item() == pytest.approx(0.2)

    # F_freeze=0 -> all dev sprints; F_freeze=2 with R=4 -> half and half.
    assert df.loc[df.config_key == "hi_1", ["T_dev", "T_qa"]].iloc[0].tolist() == [104, 0]
    assert df.loc[df.config_key == "low_1", ["T_dev", "T_qa"]].iloc[0].tolist() == [52, 52]

    assert df.loc[df.config_key == "hi_1", "quality_cost_ratio"].item() == pytest.approx(0.1)
    assert df.loc[df.config_key == "hi_1", "quality_cost_index"].item() == pytest.approx(1e5)
    assert (df["n_sup_dev"] == 1).all()   # only a_dev_1
    # a_qa_3 = 1 in every row, but n_sup_qa counts agents that are actually deployed:
    # TeamConfiguration canonicalises A_qa to zero when F_freeze == 0, so only the rows
    # with a freeze window report a QA agent.  Before that, no-freeze rows reported a
    # phantom agent that no sprint ever ran.
    assert (df.loc[df.F_freeze == 0, "n_sup_qa"] == 0).all()
    assert (df.loc[df.F_freeze > 0, "n_sup_qa"] == 1).all()


def test_cost_split_is_exact(synthetic_frontier):
    df = add_derived_columns(synthetic_frontier, config_dir=str(_REPO_ROOT / "config"))
    for suffix in ("mean", "ci_lo", "ci_hi"):
        recombined = df[f"cost_human_{suffix}"] + df[f"cost_agent_{suffix}"]
        pd.testing.assert_series_equal(
            recombined, df[f"cost_{suffix}"], check_names=False, rtol=1e-12,
        )
    assert ((df["agent_cost_share"] >= 0) & (df["agent_cost_share"] <= 1)).all()


def test_cost_split_uses_the_recorded_shares(synthetic_frontier):
    """The split is read from the simulation, not reconstructed from the config."""
    df = add_derived_columns(synthetic_frontier, config_dir=str(_REPO_ROOT / "config"))
    # _row records a 25 % agent share; a config-derived reconstruction would not
    # reproduce it, since the fixture's compositions bear no relation to these costs.
    assert df["agent_cost_share"].tolist() == pytest.approx([0.25] * len(df))
    for _, row in df.iterrows():
        assert row["cost_agent_mean"] == pytest.approx(row["cost_mean"] * 0.25)
        assert row["C_human"] + row["C_agents"] == pytest.approx(row["C_total"])


def test_cost_split_falls_back_for_pre_usage_model_campaigns():
    """A campaign without the recorded split still loads, via the old formula."""
    sid = "h-m_p-m_d-m_a-m_b-m"
    legacy_df = pd.DataFrame([
        _row(sid, "random", "hi_1", value=12.0, quality=100.0, cost=1000.0, legacy=True),
    ])
    assert "cost_agent_mean" not in legacy_df.columns

    with pytest.warns(RuntimeWarning, match="pre-usage-based cost model"):
        df = add_derived_columns(legacy_df, config_dir=str(_REPO_ROOT / "config"))

    params = scenario_params_for_id(sid, str(_REPO_ROOT / "config"))
    config = TeamConfiguration(
        N=1, A_dev=tuple(1 if k == 0 else 0 for k in range(AGENT_ROLE_COUNT)),
        R=4, F_freeze=0,
        A_qa=tuple(1 if k == 2 else 0 for k in range(AGENT_ROLE_COUNT)),
    )
    C_human, C_agents = _legacy_cost_components(config, params, 104)
    row = df.iloc[0]
    assert row["C_human"] == pytest.approx(C_human)
    assert row["C_agents"] == pytest.approx(C_agents)
    assert row["cost_human_mean"] == pytest.approx(
        row["cost_mean"] * C_human / (C_human + C_agents)
    )


def _summary_pair(objective_extras: dict) -> tuple[dict, dict]:
    """A minimal (manifest, search_summary) pair holding one candidate."""
    key = "N1_R4_00001"
    objectives = {
        "N": 1.0, "R": 4, "F_freeze": 0,
        "A_dev": [1] + [0] * (AGENT_ROLE_COUNT - 1),
        "A_qa": [0] * AGENT_ROLE_COUNT,
        "value_mean": 12.0, "quality_mean": 100.0, "cost_mean": 1000.0,
        "value_ci": [12.0, 12.0], "quality_ci": [100.0, 100.0],
        "cost_ci": [1000.0, 1000.0],
        **objective_extras,
    }
    manifest = {"configs": {key: {"F_freeze": 0, "A_qa": [0] * AGENT_ROLE_COUNT}}}
    return manifest, {"pareto_frontier": [key], "objectives": {key: objectives}}


def test_flat_table_carries_the_contribution_counts():
    """The four counters reach the flat table when search_summary.json has them."""
    manifest, summary = _summary_pair({
        "impl_human_mean": 300.0, "impl_agents_mean": 900.0,
        "fixes_human_mean": 5.0, "fixes_agent_mean": 15.0,
    })
    row = frontier_rows("h-m_p-m_d-m_a-m_b-m", "random", manifest, summary)[0]
    assert row["impl_human_mean"] == 300.0
    assert row["impl_agents_mean"] == 900.0
    assert row["fixes_human_mean"] == 5.0
    assert row["fixes_agent_mean"] == 15.0


def test_flat_table_omits_the_contribution_counts_when_absent():
    """A pre-counter search yields no such columns rather than zeros."""
    manifest, summary = _summary_pair({})
    row = frontier_rows("h-m_p-m_d-m_a-m_b-m", "random", manifest, summary)[0]
    for name in ("impl_human_mean", "impl_agents_mean",
                 "fixes_human_mean", "fixes_agent_mean"):
        assert name not in row


def test_contribution_shares_are_derived_when_recorded():
    """human_feature_share / human_fix_share come from the recorded counters."""
    sid = "h-m_p-m_d-m_a-m_b-m"
    df = pd.DataFrame([
        _row(sid, "random", "mixed", value=12.0, quality=100.0, cost=1000.0,
             contributions=(300.0, 900.0, 5.0, 15.0)),
        _row(sid, "random", "human_only", value=12.0, quality=200.0, cost=500.0,
             contributions=(1248.0, 0.0, 20.0, 0.0)),
    ])
    out = add_derived_columns(df, config_dir=str(_REPO_ROOT / "config"))

    shares = out.set_index("config_key")
    assert shares.loc["mixed", "human_feature_share"] == pytest.approx(0.25)
    assert shares.loc["mixed", "human_fix_share"] == pytest.approx(0.25)
    assert shares.loc["human_only", "human_feature_share"] == pytest.approx(1.0)
    assert shares.loc["human_only", "human_fix_share"] == pytest.approx(1.0)


def test_contribution_shares_absent_without_the_counters(synthetic_frontier):
    """The §4.2 campaign predates the counters: no shares, no warning, no crash."""
    assert "impl_human_mean" not in synthetic_frontier.columns
    df = add_derived_columns(synthetic_frontier, config_dir=str(_REPO_ROOT / "config"))
    assert "human_feature_share" not in df.columns
    assert "human_fix_share" not in df.columns


def test_no_fixes_leaves_the_fix_share_undefined():
    """A team that repaired nothing has no share to report, rather than a 0/0."""
    sid = "h-m_p-m_d-m_a-m_b-m"
    df = pd.DataFrame([
        _row(sid, "random", "clean", value=12.0, quality=0.0, cost=1000.0,
             contributions=(1248.0, 0.0, 0.0, 0.0)),
    ])
    out = add_derived_columns(df, config_dir=str(_REPO_ROOT / "config"))
    assert out["human_feature_share"].item() == pytest.approx(1.0)
    assert pd.isna(out["human_fix_share"].item())


def test_add_derived_columns_requires_T(synthetic_frontier):
    with pytest.raises(KeyError, match="T"):
        add_derived_columns(synthetic_frontier.drop(columns=["T"]))


def test_add_derived_columns_on_empty_frame():
    assert add_derived_columns(pd.DataFrame()).empty


# ---------------------------------------------------------------------------
# Bullet 1 — value saturation
# ---------------------------------------------------------------------------

def test_value_saturation_summary(synthetic_frontier):
    df = add_derived_columns(synthetic_frontier, config_dir=str(_REPO_ROOT / "config"))
    summary = value_saturation_summary(df, threshold=0.99).set_index("strategy")

    assert summary.loc["random", "n_frontier"] == 3
    assert summary.loc["random", "value_ratio_max"] == pytest.approx(1.0)
    assert summary.loc["random", "value_ratio_min"] == pytest.approx(0.8)
    # 1 of 3 below the threshold; with zero-width CIs it is also throughput-limited.
    assert summary.loc["random", "frac_below_threshold"] == pytest.approx(1 / 3)
    assert summary.loc["random", "n_throughput_limited"] == 1
    assert summary.loc["evolutionary", "frac_throughput_limited"] == pytest.approx(0.5)
    assert (summary["r_arrival"] == 12.0).all()


def test_throughput_limited_is_ci_aware(synthetic_frontier):
    """A wide CI reaching up to the arrival rate must not count as a shortfall."""
    df = synthetic_frontier.copy()
    df.loc[df.config_key == "low_1", "value_ci_hi"] = 12.0
    df = add_derived_columns(df, config_dir=str(_REPO_ROOT / "config"))
    summary = value_saturation_summary(df, threshold=0.99).set_index("strategy")

    # Still below threshold on the point estimate, but no longer throughput-limited.
    assert summary.loc["random", "frac_below_threshold"] == pytest.approx(1 / 3)
    assert summary.loc["random", "n_throughput_limited"] == 0


def test_value_deviation_patterns_finds_the_factor(synthetic_frontier):
    df = add_derived_columns(synthetic_frontier, config_dir=str(_REPO_ROOT / "config"))
    patterns = value_deviation_patterns(df, threshold=0.99)

    ff = patterns[patterns.factor == "F_freeze"].set_index("level")
    assert ff.loc[0, "n_points"] == 3
    assert ff.loc[0, "frac_throughput_limited"] == pytest.approx(0.0)
    assert ff.loc[2, "n_points"] == 2
    assert ff.loc[2, "frac_throughput_limited"] == pytest.approx(1.0)
    # Deficit rises with F_freeze, so the rank correlation is positive.
    assert ff["spearman_rho"].iloc[0] > 0

    # Every tier factor and every control factor present gets rows.
    assert {f"tier_{g}" for g in SCENARIO_GROUPS} <= set(patterns.factor)
    assert "n_sup_dev" in set(patterns.factor)


def test_value_deviation_patterns_on_empty_frame():
    out = value_deviation_patterns(pd.DataFrame())
    assert out.empty
    assert "frac_throughput_limited" in out.columns


# ---------------------------------------------------------------------------
# Bullet 2 — quality/cost ranking
# ---------------------------------------------------------------------------

def test_rank_frontier_orders_by_ratio(synthetic_frontier):
    df = add_derived_columns(synthetic_frontier, config_dir=str(_REPO_ROOT / "config"))
    ranked = rank_frontier(df)

    rnd = ranked[ranked.strategy == "random"].set_index("config_key")
    # ratios: hi_1 0.1, hi_2 0.4, low_1 0.025 -> low_1 < hi_1 < hi_2
    assert rnd.loc["low_1", "ratio_rank"] == 1
    assert rnd.loc["hi_1", "ratio_rank"] == 2
    assert rnd.loc["hi_2", "ratio_rank"] == 3
    assert rnd["ratio_pct"].max() == pytest.approx(1.0)

    # Ranks are computed within each (scenario, strategy), not globally.
    evo = ranked[ranked.strategy == "evolutionary"]
    assert sorted(evo["ratio_rank"].tolist()) == [1, 2]

    # cost_rank ascending (1 = cheapest), value_rank descending (1 = most value)
    assert rnd.loc["hi_2", "cost_rank"] == 1
    assert rnd.loc["low_1", "cost_rank"] == 3
    assert rnd.loc["low_1", "value_rank"] == 3


def test_rank_frontier_handles_ties():
    sid = "h-m_p-m_d-m_a-m_b-m"
    df = pd.DataFrame([
        _row(sid, "random", "a", value=12.0, quality=100.0, cost=1000.0),
        _row(sid, "random", "b", value=12.0, quality=200.0, cost=2000.0),   # same ratio
    ])
    ranked = rank_frontier(add_derived_columns(df, config_dir=str(_REPO_ROOT / "config")))
    assert ranked["quality_cost_ratio"].nunique() == 1
    assert ranked["ratio_rank"].tolist() == [1, 1]   # method="min"


def test_quality_cost_index_is_the_product(synthetic_frontier):
    """Both objectives are minimised, so the combined scalar is their product."""
    df = add_derived_columns(synthetic_frontier, config_dir=str(_REPO_ROOT / "config"))
    out = add_quality_cost_index(df)
    expected = df["quality_mean"] * df["cost_mean"]
    assert out["quality_cost_index"].tolist() == pytest.approx(expected.tolist())


def test_quality_cost_index_alternative_cost_field(synthetic_frontier):
    df = add_derived_columns(synthetic_frontier, config_dir=str(_REPO_ROOT / "config"))
    out = add_quality_cost_index(df, cost_field="cost_agent_mean")
    expected = df["quality_mean"] * df["cost_agent_mean"]
    assert out["quality_cost_index"].tolist() == pytest.approx(expected.tolist())


def test_quality_cost_index_is_not_monotone_along_a_front():
    """The point of the index: it has an interior optimum where the ratio has none."""
    sid = "h-m_p-m_d-m_a-m_b-m"
    # A textbook trade-off front: quality falls as cost rises, with a knee in between.
    df = pd.DataFrame([
        _row(sid, "random", "a", value=12.0, quality=1000.0, cost=100.0),   # index 1e5
        _row(sid, "random", "b", value=12.0, quality=100.0, cost=200.0),    # index 2e4
        _row(sid, "random", "c", value=12.0, quality=50.0, cost=2000.0),    # index 1e5
    ])
    out = add_quality_cost_index(
        add_derived_columns(df, config_dir=str(_REPO_ROOT / "config"))
    ).set_index("config_key")
    assert out.loc["b", "quality_cost_index"] < out.loc["a", "quality_cost_index"]
    assert out.loc["b", "quality_cost_index"] < out.loc["c", "quality_cost_index"]
    # The ratio, by contrast, falls monotonically from the quality to the cost end.
    assert out["quality_cost_ratio"].tolist() == sorted(
        out["quality_cost_ratio"].tolist(), reverse=True
    )


def test_front_position_is_per_front(synthetic_frontier):
    """Percentiles are ranked within each (scenario, strategy), not across the table."""
    df = add_derived_columns(synthetic_frontier, config_dir=str(_REPO_ROOT / "config"))
    out = add_front_position(df).set_index("config_key")

    # random front: ratios 0.1 (hi_1), 0.4 (hi_2), 0.025 (low_1) → 3 points
    assert out.loc["low_1", "front_pct"] == pytest.approx(1 / 3)
    assert out.loc["hi_1", "front_pct"] == pytest.approx(2 / 3)
    assert out.loc["hi_2", "front_pct"] == pytest.approx(1.0)
    # evolutionary front: 2 points, so its own top point is at 1.0 as well
    assert out.loc["hi_3", "front_pct"] == pytest.approx(1.0)
    assert out.loc["low_2", "front_pct"] == pytest.approx(0.5)


def test_front_position_ratio_matches_derived_column(synthetic_frontier):
    df = add_derived_columns(synthetic_frontier, config_dir=str(_REPO_ROOT / "config"))
    out = add_front_position(df)
    assert out["quality_cost_ratio"].tolist() == pytest.approx(
        df["quality_cost_ratio"].tolist()
    )


@pytest.mark.parametrize("fn", [add_quality_cost_index, add_front_position])
def test_scalar_helpers_reject_unknown_cost_field(synthetic_frontier, fn):
    df = add_derived_columns(synthetic_frontier, config_dir=str(_REPO_ROOT / "config"))
    with pytest.raises(KeyError, match="value_mean"):
        fn(df, cost_field="value_mean")


@pytest.mark.parametrize("fn", [add_quality_cost_index, add_front_position])
def test_scalar_helpers_on_empty_frame(fn):
    empty = pd.DataFrame(columns=["scenario_id", "strategy", "quality_mean", "cost_mean"])
    assert fn(empty).empty


def test_index_optimum_per_front(synthetic_frontier):
    """One row per (scenario, strategy), each the lowest-index point of its front."""
    df = add_derived_columns(synthetic_frontier, config_dir=str(_REPO_ROOT / "config"))
    best = index_optimum(df)

    assert len(best) == 2                       # two fronts in the fixture
    by_strategy = best.set_index("strategy")
    # random: 1e5 (hi_1), 1e5 (hi_2), 1e5 (low_1) … indices are 100*1000, 200*500,
    # 50*2000 — all 1e5, so the tie is broken by the lower cost.
    assert by_strategy.loc["random", "config_key"] == "hi_2"
    # evolutionary: 400*250 = 1e5 vs 80*1500 = 1.2e5
    assert by_strategy.loc["evolutionary", "config_key"] == "hi_3"


def test_index_optimum_per_scenario_pools_strategies(synthetic_frontier):
    df = add_derived_columns(synthetic_frontier, config_dir=str(_REPO_ROOT / "config"))
    best = index_optimum(df, by=("scenario_id",))

    assert len(best) == 1
    row = best.iloc[0]
    assert row["quality_cost_index"] == df["quality_mean"].mul(df["cost_mean"]).min()
    # The full configuration of the winning point comes along.
    for column in ("strategy", "config_key", "N", "R", "F_freeze", "a_dev_1"):
        assert column in best.columns


def test_index_optimum_picks_the_interior_minimum():
    """The optimum is not an end of the front — the whole reason the index exists."""
    sid = "h-m_p-m_d-m_a-m_b-m"
    df = pd.DataFrame([
        _row(sid, "random", "quality_end", value=12.0, quality=1000.0, cost=100.0),
        _row(sid, "random", "knee", value=12.0, quality=100.0, cost=200.0),
        _row(sid, "random", "cost_end", value=12.0, quality=50.0, cost=2000.0),
    ])
    best = index_optimum(add_derived_columns(df, config_dir=str(_REPO_ROOT / "config")))
    assert best["config_key"].tolist() == ["knee"]


def test_index_optimum_on_empty_frame():
    empty = pd.DataFrame(columns=["scenario_id", "strategy", "quality_mean", "cost_mean"])
    assert index_optimum(empty).empty


def test_frontier_extremes(synthetic_frontier):
    df = add_derived_columns(synthetic_frontier, config_dir=str(_REPO_ROOT / "config"))
    ex = frontier_extremes(df, k=1)

    rnd = ex[ex.strategy == "random"].set_index("config_key")
    assert rnd.loc["low_1", "end"] == "quality_dominant"
    assert rnd.loc["hi_2", "end"] == "cost_dominant"
    assert len(rnd) == 2


def test_frontier_extremes_does_not_double_count_small_frontiers(synthetic_frontier):
    """With k larger than the frontier, each point is reported exactly once."""
    df = add_derived_columns(synthetic_frontier, config_dir=str(_REPO_ROOT / "config"))
    ex = frontier_extremes(df, k=10)
    assert len(ex) == len(df)
    assert (ex["end"] == "quality_dominant").all()


# ---------------------------------------------------------------------------
# The throughput-filtered quality/cost front (§5 bullet 2)
# ---------------------------------------------------------------------------

def _pt(quality, cost, *, q_lo=None, q_hi=None, c_lo=None, c_hi=None, value=12.0):
    """A frontier-table row carrying only what the 2-D pass reads."""
    return {
        "value_mean": value,
        "quality_mean": quality,
        "quality_ci_lo": quality if q_lo is None else q_lo,
        "quality_ci_hi": quality if q_hi is None else q_hi,
        "cost_mean": cost,
        "cost_ci_lo": cost if c_lo is None else c_lo,
        "cost_ci_hi": cost if c_hi is None else c_hi,
    }


def test_quality_cost_frontier_basic():
    pts = [
        _pt(10.0, 100.0),   # 0: cheap, poor quality  -> on front
        _pt(1.0, 500.0),    # 1: expensive, good      -> on front
        _pt(5.0, 300.0),    # 2: in between           -> on front
        _pt(20.0, 600.0),   # 3: dominated by all     -> off front
    ]
    assert quality_cost_frontier(pts, ci_aware=False) == [0, 1, 2]


def test_quality_cost_frontier_ignores_value():
    """A point kept in 3-D only by its value must fall off the 2-D front."""
    pts = [
        _pt(1.0, 100.0, value=5.0),    # best on both remaining objectives
        _pt(9.0, 900.0, value=99.0),   # only its huge value saved it in 3-D
    ]
    assert quality_cost_frontier(pts, ci_aware=False) == [0]

    # Confirm the 3-D pass would indeed have kept both.
    three_d = [
        AggregatedObjectives(p["value_mean"], p["quality_mean"], p["cost_mean"],
                             (p["value_mean"],) * 2, (p["quality_mean"],) * 2,
                             (p["cost_mean"],) * 2)
        for p in pts
    ]
    assert pareto_frontier(three_d, ci_aware=False) == [0, 1]


def test_quality_cost_frontier_matches_flattened_pareto_frontier():
    """The 2-D helper must equal pareto_frontier with value held constant."""
    pts = [_pt(10.0, 100.0), _pt(1.0, 500.0), _pt(5.0, 300.0), _pt(20.0, 600.0)]
    flat = [
        AggregatedObjectives(0.0, p["quality_mean"], p["cost_mean"], (0.0, 0.0),
                             (p["quality_ci_lo"], p["quality_ci_hi"]),
                             (p["cost_ci_lo"], p["cost_ci_hi"]))
        for p in pts
    ]
    for ci in (True, False):
        assert quality_cost_frontier(pts, ci_aware=ci) == pareto_frontier(flat, ci_aware=ci)


def test_quality_cost_frontier_ci_aware_co_retains_overlapping_points():
    """Overlapping CIs keep a dominated point; disjoint CIs drop it."""
    best = _pt(1.0, 100.0, q_lo=0.5, q_hi=1.5, c_lo=95.0, c_hi=105.0)
    overlapping = _pt(1.4, 110.0, q_lo=0.9, q_hi=1.9, c_lo=99.0, c_hi=121.0)
    assert quality_cost_frontier([best, overlapping], ci_aware=True) == [0, 1]
    assert quality_cost_frontier([best, overlapping], ci_aware=False) == [0]

    disjoint = _pt(9.0, 900.0, q_lo=8.0, q_hi=10.0, c_lo=880.0, c_hi=920.0)
    assert quality_cost_frontier([best, disjoint], ci_aware=True) == [0]


def test_quality_cost_frontier_accepts_a_dataframe(synthetic_frontier):
    df = add_derived_columns(synthetic_frontier, config_dir=str(_REPO_ROOT / "config"))
    idx = quality_cost_frontier(df, ci_aware=False)
    assert all(0 <= i < len(df) for i in idx)
    assert len(set(idx)) == len(idx)


def test_quality_cost_frontier_parsimony_keeps_the_smallest_tied_team():
    """The §3.10 tie-break, opted in on rows that carry the configuration columns."""
    sid = "h-m_p-m_d-m_a-m_b-m"
    # Same process, statistically indistinguishable, growing implementation-agent counts.
    rows = [_row(sid, "random", f"k{n}", value=12.0, quality=100.0 + n, cost=1000.0 + n,
                 a_dev_1=n, ci_width=500.0) for n in (1, 2, 3)]
    assert quality_cost_frontier(rows, parsimony=False) == [0, 1, 2]
    assert quality_cost_frontier(rows, parsimony=True) == [0]


def test_quality_cost_frontier_parsimony_respects_a_significant_difference():
    """A bigger team that demonstrably buys something is not pruned.

    The two trade off quality against cost, so neither CI-dominates the other and both
    are on the front.  The bigger one is CI-significantly better on quality, so they are
    not "the same result" and the tie-break has no grounds to drop it.
    """
    sid = "h-m_p-m_d-m_a-m_b-m"
    rows = [_row(sid, "random", "small", value=12.0, quality=200.0, cost=900.0,
                 a_dev_1=1, ci_width=1.0),
            _row(sid, "random", "big", value=12.0, quality=100.0, cost=1000.0,
                 a_dev_1=3, ci_width=1.0)]
    assert sorted(quality_cost_frontier(rows, parsimony=False)) == [0, 1]
    assert sorted(quality_cost_frontier(rows, parsimony=True)) == [0, 1]


def test_filtered_front_applies_parsimony_by_default(synthetic_frontier):
    df = add_derived_columns(synthetic_frontier, config_dir=str(_REPO_ROOT / "config"))
    with_tb = filtered_front_from_table(df)
    without = filtered_front_from_table(df, parsimony=False)
    assert len(with_tb) <= len(without)
    assert set(with_tb["config_key"]) <= set(without["config_key"])


def test_filtered_front_respects_the_threshold(synthetic_frontier):
    df = add_derived_columns(synthetic_frontier, config_dir=str(_REPO_ROOT / "config"))
    # value_ratios present: 1.0, 1.0, 0.8 (random) and 1.0, 0.9 (evolutionary)
    front = filtered_front_from_table(df, threshold=0.98)
    assert (front["value_ratio"] >= 0.98).all()
    assert set(front["config_key"]) <= {"hi_1", "hi_2", "hi_3"}


def test_filtered_front_threshold_boundary():
    """A point exactly at the threshold is kept; just under is dropped."""
    sid = "h-m_p-m_d-m_a-m_b-m"          # r_arrival = 12.0
    df = pd.DataFrame([
        _row(sid, "random", "exact", value=12.0 * 0.98, quality=1.0, cost=10.0),
        _row(sid, "random", "under", value=12.0 * 0.98 - 1e-6, quality=0.5, cost=5.0),
    ])
    df = add_derived_columns(df, config_dir=str(_REPO_ROOT / "config"))
    front = filtered_front_from_table(df, threshold=0.98)
    assert front["config_key"].tolist() == ["exact"]


def test_filtered_front_is_per_scenario_strategy(synthetic_frontier):
    df = add_derived_columns(synthetic_frontier, config_dir=str(_REPO_ROOT / "config"))
    front = filtered_front_from_table(df, threshold=0.98)
    # hi_3 is the only evolutionary survivor and must be kept even though the
    # random search has points that dominate it outright.
    assert "hi_3" in set(front[front.strategy == "evolutionary"]["config_key"])


def test_filtered_front_drops_scenarios_with_no_survivors():
    sid = "h-m_p-m_d-m_a-m_b-m"
    df = pd.DataFrame([_row(sid, "random", "slow", value=1.0, quality=1.0, cost=10.0)])
    df = add_derived_columns(df, config_dir=str(_REPO_ROOT / "config"))
    assert filtered_front_from_table(df, threshold=0.98).empty


def test_filtered_front_requires_value_ratio(synthetic_frontier):
    with pytest.raises(KeyError, match="value_ratio"):
        filtered_front_from_table(synthetic_frontier)


def test_filtered_front_on_empty_frame():
    assert filtered_front_from_table(pd.DataFrame()).empty


# ---------------------------------------------------------------------------
# Headline summary
# ---------------------------------------------------------------------------

def test_headline_summary(synthetic_frontier):
    df = add_derived_columns(synthetic_frontier, config_dir=str(_REPO_ROOT / "config"))
    s = headline_summary(df, threshold=0.99)

    assert s["n_scenarios"] == 1
    assert s["n_searches"] == 2
    assert s["n_frontier_points"] == 5
    assert s["value_ratio_max"] == pytest.approx(1.0)
    assert s["frac_below_threshold"] == pytest.approx(0.4)
    assert s["frac_throughput_limited"] == pytest.approx(0.4)
    # F_freeze separates the deviating points perfectly, so it must win.
    assert s["dominant_deviation_factor"] == "F_freeze"


def test_headline_summary_on_empty_frame():
    assert headline_summary(pd.DataFrame())["n_frontier_points"] == 0


# ---------------------------------------------------------------------------
# End-to-end over the real campaign (skipped when it has not been run)
# ---------------------------------------------------------------------------

def _first_real_scenario() -> Path | None:
    if not _CAMPAIGN_DIR.is_dir():
        return None
    for d in sorted(p for p in _CAMPAIGN_DIR.iterdir() if p.is_dir()):
        if any((d / s / "search_summary.json").is_file()
               for s in ("random", "evolutionary")):
            return d
    return None


@pytest.fixture
def one_scenario_campaign(tmp_path):
    """A campaign root holding just the first finished scenario, via symlink."""
    scenario = _first_real_scenario()
    root = tmp_path / "campaign"
    root.mkdir()
    (root / scenario.name).symlink_to(scenario, target_is_directory=True)
    return root, scenario.name


@pytest.mark.skipif(_first_real_scenario() is None,
                    reason="no finished searches in results/experiments")
def test_pipeline_over_one_real_scenario(one_scenario_campaign):
    root, scenario_name = one_scenario_campaign

    frontiers, scenarios = build_tables(root)
    assert not frontiers.empty
    assert len(scenarios) >= 1
    assert scenarios["scenario_id"].unique().tolist() == [scenario_name]

    table = load_frontier_table(results_dir=root)
    df = add_derived_columns(table, config_dir=str(_REPO_ROOT / "config"))

    # The whole point of §5 bullet 1: the frontier sits at the arrival rate.
    assert df["value_ratio"].median() == pytest.approx(1.0, abs=0.01)
    # And the cost split reconstructs the persisted cost exactly.
    assert (df["cost_human_mean"] + df["cost_agent_mean"] - df["cost_mean"]).abs().max() < 1e-6

    assert len(value_saturation_summary(df)) == len(scenarios)
    assert not value_deviation_patterns(df).empty
    assert len(rank_frontier(df)) == len(df)
    assert headline_summary(df)["n_frontier_points"] == len(df)


@pytest.mark.skipif(_first_real_scenario() is None,
                    reason="no finished searches in results/experiments")
def test_candidate_rows_over_a_real_search(one_scenario_campaign):
    import json
    root, scenario_name = one_scenario_campaign
    d = next(p for p in (root / scenario_name).iterdir()
             if (p / "search_summary.json").is_file())
    manifest = json.loads((d / "manifest.json").read_text())
    summary = json.loads((d / "search_summary.json").read_text())

    cands = candidate_rows(scenario_name, d.name, manifest, summary)
    fronts = frontier_rows(scenario_name, d.name, manifest, summary)

    # Every simulated candidate, and the frontier ones correctly flagged.
    assert len(cands) == len(summary["objectives"])
    assert sum(r["on_3d_frontier"] for r in cands) == len(summary["pareto_frontier"])
    # Same schema as frontier_rows, plus the flag.
    assert set(cands[0]) == set(fronts[0]) | {"on_3d_frontier"}


@pytest.mark.skipif(_first_real_scenario() is None,
                    reason="no finished searches in results/experiments")
def test_build_campaign_over_one_real_scenario(one_scenario_campaign):
    root, scenario_name = one_scenario_campaign
    thr = DEFAULT_THROUGHPUT_THRESHOLD
    c = build_campaign(root, throughput_threshold=thr,
                       config_dir=str(_REPO_ROOT / "config"))

    assert not c.frontiers.empty
    assert not c.filtered_front.empty
    assert len(c.filter_summary) == len(c.scenarios)

    # Counts are nested: front <= passed <= simulated.
    fs = c.filter_summary
    assert (fs["n_filtered_front"] <= fs["n_passed"]).all()
    assert (fs["n_passed"] <= fs["n_simulated"]).all()
    assert fs["n_filtered_front"].sum() == len(c.filtered_front)

    # Everything on the front cleared the throughput filter …
    assert (c.filtered_front["value_ratio"] >= thr).all()
    # … carries the derived columns …
    assert (c.filtered_front["cost_human_mean"] + c.filtered_front["cost_agent_mean"]
            - c.filtered_front["cost_mean"]).abs().max() < 1e-6
    # … and is internally non-dominated on (quality, cost).
    for _, sub in c.filtered_front.groupby(["scenario_id", "strategy"]):
        assert quality_cost_frontier(sub) == list(range(len(sub)))

    # The filtered front should be markedly smaller than the recorded 3-D one —
    # that is the point of dropping the value dimension.
    assert fs["n_filtered_front"].sum() < fs["n_recorded_3d_front"].sum()


@pytest.mark.skipif(_first_real_scenario() is None,
                    reason="no finished searches in results/experiments")
def test_build_campaign_beats_filtering_the_recorded_frontier(one_scenario_campaign):
    """The all-candidate pool must be a superset of the recorded-frontier pool.

    Filtering only the recorded three-objective frontier is correct for strict
    dominance but can miss a point the CI-aware rule retains, which is why
    build_campaign uses every simulated candidate.
    """
    root, _ = one_scenario_campaign
    c = build_campaign(root, config_dir=str(_REPO_ROOT / "config"))

    df = add_derived_columns(load_frontier_table(results_dir=root),
                             config_dir=str(_REPO_ROOT / "config"))
    degraded = filtered_front_from_table(df, DEFAULT_THROUGHPUT_THRESHOLD)

    full_keys = set(zip(c.filtered_front["strategy"], c.filtered_front["config_key"]))
    degraded_keys = set(zip(degraded["strategy"], degraded["config_key"]))
    assert degraded_keys <= full_keys


@pytest.mark.skipif(_first_real_scenario() is None,
                    reason="no finished searches in results/experiments")
def test_headline_summary_with_filtered_front(one_scenario_campaign):
    root, _ = one_scenario_campaign
    c = build_campaign(root, config_dir=str(_REPO_ROOT / "config"))
    df = add_derived_columns(c.frontiers, config_dir=str(_REPO_ROOT / "config"))

    s = headline_summary(df, filtered_front=c.filtered_front,
                         filter_summary=c.filter_summary)
    assert s["quality_cost_source"] == "filtered_front"
    assert s["n_filtered_front_points"] == len(c.filtered_front)
    assert 0.0 < s["frac_passed_overall"] <= 1.0
    assert s["throughput_threshold"] == DEFAULT_THROUGHPUT_THRESHOLD

    # Without a filtered front it falls back to the recorded frontier.
    assert headline_summary(df)["quality_cost_source"] == "recorded_3d_frontier"
