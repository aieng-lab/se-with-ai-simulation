"""Tests for src/io.py — value parsing, schema validation, round-trips."""

import dataclasses
import json
from pathlib import Path

import pytest

from src.config import SCENARIO_GROUPS, scenario_tiers
from src.io import (
    _flatten_scenario,
    _parse_value,
    load_initial_state,
    load_scenario_params,
    load_search_config,
    load_simulation_horizon,
)
from src.parameters import ScenarioParameters
from src.state import SystemState


def tier_paths(tier: str, **overrides: str) -> dict[str, str]:
    """Paths to the five group files for `tier`, with optional per-group overrides."""
    paths = {group: f"config/{group}_{tier}.json" for group in SCENARIO_GROUPS}
    paths.update(overrides)
    return paths


def copy_group_file(group: str, tier: str, tmp_path: Path, mutate) -> Path:
    """Write a mutated copy of a group config to tmp_path and return its path."""
    data = json.loads(Path(f"config/{group}_{tier}.json").read_text())
    mutate(data[group])
    out = tmp_path / f"{group}_mutated.json"
    out.write_text(json.dumps(data))
    return out


class TestParseValue:
    def test_scalar_int(self):
        assert _parse_value("x", 5) == 5.0

    def test_scalar_float(self):
        assert _parse_value("x", 3.14) == pytest.approx(3.14)

    def test_dict_raises(self):
        with pytest.raises(TypeError):
            _parse_value("x", {"value": 2.5})

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            _parse_value("x", "not_a_number")


class TestFlattenScenario:
    def test_merges_sections(self):
        flat = _flatten_scenario({"s1": {"a": 1, "b": 2}, "s2": {"c": 3}})
        assert flat == {"a": 1, "b": 2, "c": 3}

    def test_duplicate_key_raises(self):
        with pytest.raises(ValueError, match="Duplicate"):
            _flatten_scenario({"s1": {"a": 1}, "s2": {"a": 2}})

    def test_non_dict_section_raises(self):
        with pytest.raises(ValueError):
            _flatten_scenario({"s1": [1, 2, 3]})

    def test_description_string_is_skipped(self):
        flat = _flatten_scenario({"description": "docs", "s1": {"a": 1}})
        assert flat == {"a": 1}

    def test_non_string_description_still_validated(self):
        with pytest.raises(ValueError):
            _flatten_scenario({"description": [1, 2]})

    def test_empty_returns_empty(self):
        assert _flatten_scenario({}) == {}


class TestLoadSimulationHorizon:
    def test_default_horizon_loads(self):
        # The campaign's Monte Carlo protocol: two years of weekly sprints.
        s = load_simulation_horizon("config/horizon_default.json")
        assert (s.T == 104 and s.K_search == 100 and s.K_final == 1000
                and s.master_seed == 42)

    def test_basic(self, tmp_path):
        p = tmp_path / "h.json"
        p.write_text('{"T": 52, "K_search": 100, "K_final": 500, "master_seed": 42}')
        s = load_simulation_horizon(p)
        assert (s.T == 52 and s.K_search == 100 and s.K_final == 500
                and s.master_seed == 42)

    def test_missing_T_raises(self, tmp_path):
        p = tmp_path / "h.json"
        p.write_text('{"K_search": 10, "K_final": 10, "master_seed": 0}')
        with pytest.raises(KeyError):
            load_simulation_horizon(p)

    def test_a_horizon_with_only_one_replication_count_raises(self, tmp_path):
        """Both counts are required; the two-stage selection needs K > K'.

        Defaulting the missing one would silently collapse the re-simulation into the
        search, which is the one thing the disjoint-block design exists to prevent.
        """
        p = tmp_path / "h.json"
        p.write_text('{"T": 52, "K_search": 100, "master_seed": 42}')
        with pytest.raises(KeyError):
            load_simulation_horizon(p)
        p.write_text('{"T": 52, "K_final": 1000, "master_seed": 42}')
        with pytest.raises(KeyError):
            load_simulation_horizon(p)


class TestLoadScenarioParams:
    def test_medium_scenario_loads(self):
        params = load_scenario_params(tier_paths("medium"))
        assert isinstance(params, ScenarioParameters)
        assert len(params.c_token) == 8

    def test_returns_correct_values(self):
        params = load_scenario_params(tier_paths("medium"))
        assert params.r_arrival == pytest.approx(12.0)
        assert params.r_bugs == pytest.approx(0.05)
        assert params.r_bugs_human == pytest.approx(0.12)
        assert params.c_salary == pytest.approx(150_000.0)
        assert params.c_token[0] == pytest.approx(3.80)

    @pytest.mark.parametrize("tier", ["low", "medium", "high"])
    def test_every_tier_loads(self, tier):
        params = load_scenario_params(tier_paths(tier))
        assert isinstance(params, ScenarioParameters)

    def test_pessimistic_agent_tier_is_low_capability_at_ten_times_the_high_price(self):
        """The `agent` group's fourth tier is defined, not measured.

        Its whole content is "the low tier's capability at ten times the high tier's
        price", so pinning it against the two files it is derived from is what catches a
        hand-edited number drifting away from the definition.
        """
        pess = load_scenario_params(tier_paths("medium", agent="config/agent_pessimistic.json"))
        low = load_scenario_params(tier_paths("medium", agent="config/agent_low.json"))
        high = load_scenario_params(tier_paths("medium", agent="config/agent_high.json"))

        assert pess.c_token == pytest.approx([10.0 * c for c in high.c_token])
        # `c_token` is the only cost in the group; every other agent field is the low
        # tier's, including the ones that happen to be tier-invariant.
        agent_fields = set(json.loads(
            Path("config/agent_low.json").read_text())["agent"]) - {"c_token"}
        assert agent_fields   # guard against the key set silently going empty
        for name in agent_fields:
            assert getattr(pess, name) == pytest.approx(getattr(low, name)), name

    def test_agent_capability_tiers_follow_the_swe_bench_scores(self):
        """Paper tab:scenario:agents, r_bugs / d_test rows.

        The footnote there resolves the leaderboard's near-tie in favour of the high
        tier, so both rows are monotone: the more expensive the tier, the fewer bugs it
        injects and the better it tests.  The values are 1 - score and score - 0.2 for
        scores 0.80 / 0.95 / 0.97, and the pair has been swapped between medium and high
        in the config once already, which is what this pins.
        """
        low, med, high = (
            load_scenario_params(tier_paths("medium", agent=f"config/agent_{t}.json"))
            for t in ("low", "medium", "high")
        )
        assert (low.r_bugs, med.r_bugs, high.r_bugs) == pytest.approx((0.20, 0.05, 0.03))
        assert (low.d_test, med.d_test, high.d_test) == pytest.approx((0.60, 0.75, 0.77))

    def test_refactoring_throughput_tracks_the_debt_a_work_item_adds(self):
        """One Refactoring Agent removes the debt of one feature or one bug fix.

        `r_refactor` is therefore calibrated against `delta_impl` / `delta_fix` and takes
        the same 0.02 / 0.06 / 0.13 by agent tier — it is *not* the tier-invariant 0.10
        it was while the role's throughput was denominated differently.
        """
        rates = [
            load_scenario_params(tier_paths("medium", agent=f"config/agent_{t}.json")).r_refactor
            for t in ("low", "medium", "high")
        ]
        assert rates == pytest.approx([0.02, 0.06, 0.13])
        debt = [
            load_scenario_params(tier_paths("medium", debt=f"config/debt_{t}.json")).delta_impl
            for t in ("low", "medium", "high")
        ]
        assert rates == pytest.approx(debt)

    def test_documentation_relief_falls_as_the_debt_tier_rises(self):
        """Paper tab:scenario:debt, Gamma_doc row: 0.40 / 0.34 / 0.0.

        The debt group's tiers run mild -> severe, and Gamma_doc is a *relief* factor, so
        it decreases where delta_impl increases.  Low and high were swapped in the config
        once, which is what this pins.
        """
        gammas = [
            load_scenario_params(tier_paths("medium", debt=f"config/debt_{t}.json")).Gamma_doc
            for t in ("low", "medium", "high")
        ]
        assert gammas == pytest.approx([0.40, 0.34, 0.0])

    def test_groups_can_be_mixed(self):
        """The five groups are orthogonal, so tiers combine freely."""
        params = load_scenario_params(
            tier_paths("medium", project="config/project_high.json")
        )
        assert params.r_arrival == pytest.approx(75.0)
        assert params.r_bugs_human == pytest.approx(0.12)   # human group still medium

    def test_groups_partition_all_parameters(self):
        """Union of the group files must be exactly the ScenarioParameters fields."""
        keys: set[str] = set()
        for group in SCENARIO_GROUPS:
            data = json.loads(Path(f"config/{group}_medium.json").read_text())
            group_keys = set(_flatten_scenario(data))
            assert not (keys & group_keys), f"{group} duplicates {keys & group_keys}"
            keys |= group_keys
        assert keys == {f.name for f in dataclasses.fields(ScenarioParameters)}

    def test_missing_group_raises(self):
        paths = tier_paths("medium")
        del paths["debt"]
        with pytest.raises(ValueError, match="missing groups"):
            load_scenario_params(paths)

    def test_unknown_group_raises(self):
        paths = tier_paths("medium")
        paths["nonexistent_group"] = "config/debt_medium.json"
        with pytest.raises(ValueError, match="unknown groups"):
            load_scenario_params(paths)

    def test_parameter_in_two_groups_raises(self, tmp_path):
        overlap = tmp_path / "debt_overlap.json"
        overlap.write_text(json.dumps({"debt": {"r_arrival": 1.0}}))
        with pytest.raises(ValueError, match="defined in both"):
            load_scenario_params(tier_paths("medium", debt=str(overlap)))

    def test_c_token_wrong_length_raises(self, tmp_path):
        def mutate(section):
            section["c_token"] = [0.1] * 5

        bad = copy_group_file("agent", "medium", tmp_path, mutate)
        with pytest.raises(ValueError, match="c_token"):
            load_scenario_params(tier_paths("medium", agent=str(bad)))

    def test_missing_parameter_raises(self, tmp_path):
        bad = copy_group_file(
            "project", "medium", tmp_path, lambda s: s.pop("r_arrival")
        )
        with pytest.raises(ValueError, match="missing"):
            load_scenario_params(tier_paths("medium", project=str(bad)))

    def test_extra_parameter_raises(self, tmp_path):
        def mutate(section):
            section["nonexistent_param"] = 1.0

        bad = copy_group_file("project", "medium", tmp_path, mutate)
        with pytest.raises(ValueError, match="unknown"):
            load_scenario_params(tier_paths("medium", project=str(bad)))


class TestLoadSearchConfig:
    def test_default_search_config_loads(self):
        cfg = load_search_config("config/search_default.json")
        assert set(cfg.scenario_paths) == set(SCENARIO_GROUPS)
        assert cfg.horizon_path == "config/horizon_default.json"
        # The template runs the search; the baseline is opted into per config.
        assert cfg.human_only is False

    def test_default_search_config_matches_the_campaign_range(self):
        # The campaign search range, and the 10,000-evaluation budget split
        # 10 % / 90 % between the initial pool and the generation loop.
        cfg = load_search_config("config/search_default.json")
        # Staffing levels from a one-day-per-week part-timer to a two-pizza team;
        # N is a staffing level, so the values are floats.
        assert cfg.N_values == (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0)
        assert cfg.R_values == (1, 2, 4, 13)
        assert cfg.F_freeze_values == (0, 1, 2)
        assert cfg.sigma == 2.0
        assert cfg.epsilon == 0.02
        assert cfg.budget == 10_000
        assert cfg.n_samples_initial == 1_000
        assert cfg.n_generations * cfg.n_offspring == 9_000
        assert cfg.n_samples_initial + cfg.n_generations * cfg.n_offspring == cfg.budget
        assert cfg.n_elite == 100
        assert cfg.samples == "best"
        assert cfg.seed == 42

    def test_sigma_defaults_and_is_validated(self, tmp_path):
        data = json.loads(Path("config/search_default.json").read_text())
        del data["search"]["sigma"]
        path = tmp_path / "search.json"
        path.write_text(json.dumps(data))
        assert load_search_config(path).sigma == 2.0

        data["search"]["sigma"] = 0.5
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="sigma must be >= 1"):
            load_search_config(path)

    def test_samples_policy_defaults_and_is_validated(self, tmp_path):
        data = json.loads(Path("config/search_default.json").read_text())
        del data["search"]["samples"]
        path = tmp_path / "search.json"
        path.write_text(json.dumps(data))
        assert load_search_config(path).samples == "best"

        data["search"]["samples"] = "everything"
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="samples must be one of"):
            load_search_config(path)

    def test_scenario_id_is_read_from_the_top_level(self):
        cfg = load_search_config("config/search_default.json")
        assert cfg.scenario_id == "h-m_p-m_d-m_a-m_b-m"

    def test_scenario_paths_resolve_to_params(self):
        cfg = load_search_config("config/search_default.json")
        params = load_scenario_params(cfg.scenario_paths)
        assert params.r_arrival == pytest.approx(12.0)

    def test_scalar_scenario_raises(self, tmp_path):
        data = json.loads(Path("config/search_default.json").read_text())
        data["scenario"] = "config/human_medium.json"
        bad = tmp_path / "search.json"
        bad.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="must be a JSON object"):
            load_search_config(bad)

    def test_incomplete_scenario_group_set_raises(self, tmp_path):
        data = json.loads(Path("config/search_default.json").read_text())
        del data["scenario"]["debt"]
        bad = tmp_path / "search.json"
        bad.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="scenario_paths"):
            load_search_config(bad)


class TestLoadInitialState:
    def test_default_config_gives_all_zero_state(self):
        state = load_initial_state("config/startstate_default.json")
        assert state == SystemState()

    def test_partial_config_sets_specified_fields(self, tmp_path):
        p = tmp_path / "start.json"
        p.write_text(json.dumps({"F_backlog": 10, "D_tech": 2.5}))
        state = load_initial_state(p)
        assert state.F_backlog == 10
        assert state.D_tech == pytest.approx(2.5)
        assert state.F_completed == 0
        assert state.B_hid_min == 0

    def test_int_fields_are_coerced_to_int(self, tmp_path):
        p = tmp_path / "start.json"
        p.write_text(json.dumps({"F_backlog": 5}))
        state = load_initial_state(p)
        assert isinstance(state.F_backlog, int)

    def test_float_fields_are_coerced_to_float(self, tmp_path):
        p = tmp_path / "start.json"
        p.write_text(json.dumps({"D_tech": 1}))
        state = load_initial_state(p)
        assert isinstance(state.D_tech, float)

    def test_unknown_field_raises(self, tmp_path):
        p = tmp_path / "start.json"
        p.write_text(json.dumps({"nonexistent": 1}))
        with pytest.raises(ValueError, match="Unknown"):
            load_initial_state(p)

    def test_empty_config_gives_default_state(self, tmp_path):
        p = tmp_path / "start.json"
        p.write_text("{}")
        state = load_initial_state(p)
        assert state == SystemState()


# ---------------------------------------------------------------------------
# The scenario grid against the paper
# ---------------------------------------------------------------------------

#: Every value of Tables 6-10 of Section 4.2, transcribed from `paper/main.tex`, as
#: {parameter: (low, medium, high[, fourth tier])}.  The paper is authoritative for these
#: and the config files are only a rendering of it, so this is where the two are held
#: together.  Two rows have been silently swapped in the config before (`d_test`/`r_bugs`
#: and `Gamma_doc`, see the tier-direction tests above), which is what the whole table
#: guards against: a spot check catches the row it happens to name and nothing else.
_PAPER_TABLES: dict[str, dict[str, tuple]] = {
    # Table: Human developer parameters (low = junior, high = senior).
    "human": {
        "H_capacity": (40.0, 40.0, 40.0),
        "c_comm": (1.0, 1.0, 1.0),
        "H_monitoring_overhead": (2.5, 2.5, 2.5),
        "H_escalation_time": (4.0, 2.0, 1.0),
        "H_review_time": (8.0, 4.0, 2.0),
        "h_feature": (4.0, 2.0, 1.0),
        "h_fix": (1.0, 0.5, 0.25),
        "r_bugs_human": (0.20, 0.12, 0.08),
        "f_fix": (0.5, 0.5, 0.5),
        "h_qa": (0.80, 0.62, 0.53),
        "f_qa": (0.09, 0.13, 0.21),
        "d_qa": (0.8, 0.8, 0.8),
        "c_salary": (100_000.0, 150_000.0, 250_000.0),
    },
    # Table: Project dynamics.  The fourth tier is the extreme-velocity stress case.
    "project": {
        "r_arrival": (3.0, 12.0, 75.0, 1000.0),
        "f_arch_chg": (0.13, 0.13, 0.13, 0.13),
        "theta_maj": (0.44, 0.44, 0.44, 0.44),
        "theta_sec": (0.07, 0.07, 0.07, 0.07),
    },
    # Table: Debt dynamics.  delta_min is the throughput *retained* at maximal debt and
    # Gamma_doc a *relief* factor, so both fall as the tier rises.
    "debt": {
        "delta_impl": (0.02, 0.06, 0.13),
        "delta_fix": (0.02, 0.06, 0.13),
        "delta_min": (0.94, 0.77, 0.60),
        "delta_exp": (1.5, 1.5, 1.5),
        "Gamma_doc": (0.40, 0.34, 0.0),
    },
    # Table: Agent cost, capability and effort.  The fourth tier is `pessimistic`: the low
    # tier's capability at ten times the high tier's price.
    "agent": {
        "r_F": (178.0, 178.0, 178.0, 178.0),
        "r_bugs": (0.20, 0.05, 0.03, 0.20),
        "r_arch_chg": (35.0, 35.0, 35.0, 35.0),
        "r_fix": (747.0, 747.0, 747.0, 747.0),
        "r_refactor": (0.02, 0.06, 0.13, 0.02),
        "d_CR": (0.75, 0.82, 0.89, 0.75),
        "alpha_size": (0.10, 0.10, 0.10, 0.10),
        "d_SA": (0.78, 0.83, 0.88, 0.78),
        "r_test": (747.0, 747.0, 747.0, 747.0),
        "d_test": (0.60, 0.75, 0.77, 0.60),
        "c_test_ceil": (0.48, 0.54, 0.60, 0.48),
        "r_doc": (747.0, 747.0, 747.0, 747.0),
        "rho_esc": (0.03, 0.02, 0.01, 0.03),
    },
    # Table: Bug impact.  Absolute dollars per escaped bug, not relative weights.
    "bugimpact": {
        "omega_min": (100.0, 150.0, 250.0),
        "omega_maj": (25_000.0, 300_000.0, 7_000_000.0),
        "omega_sec": (48_000.0, 1_000_000.0, 120_000_000.0),
    },
}

#: `c_token` is a vector, so it does not fit the scalar table above: USD per unit of each
#: role's own throughput rate, k = 1..8, per agent tier.
_PAPER_C_TOKEN: dict[str, tuple[float, ...]] = {
    "low":         (2.28, 11.40, 0.54, 2.28, 0.54, 5.40, 0.54, 0.54),
    "medium":      (3.80, 19.00, 0.90, 3.80, 0.90, 9.00, 0.90, 0.90),
    "high":        (7.60, 38.00, 1.80, 7.60, 1.80, 18.00, 1.80, 1.80),
    "pessimistic": (76.00, 380.00, 18.00, 76.00, 18.00, 180.00, 18.00, 18.00),
}


def _scenario_values(group: str, tier: str) -> dict:
    """The flat {parameter: value} mapping of one scenario file."""
    return _flatten_scenario(
        json.loads(Path(f"config/{group}_{tier}.json").read_text()))


class TestScenarioFilesMatchThePaper:
    """Every number of Section 4.2, held against the config files that render it.

    The paper is authoritative for the scenario grid, so this is the one place where a
    hand-edited config that drifts from the manuscript is caught — including the tiers a
    spot check would not reach.
    """

    @pytest.mark.parametrize("group", list(_PAPER_TABLES))
    def test_every_tabulated_value_is_in_the_config(self, group):
        table = _PAPER_TABLES[group]
        for tier_index, tier in enumerate(scenario_tiers(group)):
            values = _scenario_values(group, tier)
            for name, per_tier in table.items():
                assert name in values, f"{group}/{tier} is missing {name}"
                assert values[name] == pytest.approx(per_tier[tier_index]), \
                    f"{group}/{tier}: {name}"

    def test_the_table_covers_every_scalar_parameter_of_its_group(self):
        """No parameter may sit outside the transcription and drift unnoticed."""
        for group, table in _PAPER_TABLES.items():
            defined = set(_scenario_values(group, scenario_tiers(group)[0]))
            # c_token is a vector and is pinned separately below.
            assert defined - {"c_token"} == set(table), group

    @pytest.mark.parametrize("tier", ["low", "medium", "high", "pessimistic"])
    def test_the_token_prices_match_the_paper(self, tier):
        values = _scenario_values("agent", tier)
        assert tuple(values["c_token"]) == pytest.approx(_PAPER_C_TOKEN[tier])

    def test_the_derived_relations_the_paper_states_in_prose_hold(self):
        """Three cost ratios the paper gives as derivations rather than as numbers.

        An architecture proposal costs five feature implementations, a security audit ten
        code reviews, and refactoring the same as a feature implementation.
        """
        for tier in ("low", "medium", "high", "pessimistic"):
            c = _scenario_values("agent", tier)["c_token"]
            assert c[1] == pytest.approx(5 * c[0]), f"{tier}: architecture = 5 x feature"
            assert c[5] == pytest.approx(10 * c[4]), f"{tier}: security = 10 x review"
            assert c[3] == pytest.approx(c[0]), f"{tier}: refactoring = feature"
