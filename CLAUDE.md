# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Research project simulating and optimizing **AI Agent and Human Developer Team Dynamics**. Mixed teams of AI agents and human developers are modelled as a discrete-time stochastic system; Monte Carlo simulation finds Pareto-optimal team configurations optimizing for development speed (value), software quality (quality), and cost per feature (cost).

The simulation is fully operational. The formal model specification is in **`paper/main.tex`** (Section 3).

---

## Repository Layout

```
paper/
    main.tex            formal model specification (AUTHORITATIVE)

src/                    Python simulation package
    config.py           TeamConfiguration, SimulationHorizon, SearchConfiguration, SCENARIO_GROUPS
    parameters.py       ScenarioParameters (36 fields)
    state.py            SystemState, LatencyBuffer
    rng.py              reproducible PCG-64 seeding (FNV-1a config hash)
    io.py               load_simulation_horizon / load_team_config / load_scenario_params
                        / load_search_config / load_initial_state
    feasibility.py      check_feasibility — human-capacity feasibility condition (§3.3)
    sprint.py           step_sprint — full 10-step sprint transition + SprintRecord
    simulation.py       run_configuration (K replications); SprintSnapshot / ConfigTrace
    pareto.py           aggregate_config_result (bootstrap CI), pareto_frontier (CI-aware)
    results.py          to_dataframe, save_result/load_result, save_traces/load_traces
    analysis.py         mean_trajectory, quantile_band, final_distribution, convergence_check
    results_analysis.py campaign-level §5 analysis: flat-table schema, cost split,
                        value saturation, throughput-filtered quality/cost front
    search_configurations.py random + evolutionary search over agent compositions
    visualization.py    matplotlib plot functions + visualize_single/visualize_multi helpers
    main.py             CLI: `python -m src.main run|analyze|optimize|visualize`

config/
    horizon_default.json      Monte Carlo protocol of §4.2: T=104, K=100, master_seed=42
    team_default.json         default team configuration (N, M, R, F_freeze, A_dev, A_qa)
    human_{low,medium,high}.json      scenario group: human developers (§4.1)
    project_{low,medium,high}.json    scenario group: project dynamics
    debt_{low,medium,high}.json       scenario group: debt dynamics
    agent_{low,medium,high}.json      scenario group: agent costs and throughput
    bugimpact_{low,medium,high}.json  scenario group: bug impact
    search_default.json       search config template: §4.2 range and budget, all groups medium
    search/search_<id>.json   the 243 generated scenario searches (§4.2); id = h-l_p-m_d-h_a-l_b-m
    startstate_default.json   initial SystemState values (all-zero baseline)

scripts/
    generate_search_configs.py  write the 243 config/search/*.json from search_default.json
    collect_results.py          consolidate a campaign into frontiers.csv + scenarios.csv
    analyze_results.py          the §5 results analysis: derived tables + figures
run_experiments.sh      campaign driver: 486 searches in parallel, resumable (§4.2)

tests/                  pytest suite (239 tests, ~12 s)
results/                persisted sweep and campaign results (not committed)
```

---

## Running the Simulation

**Quick smoke test:**
```bash
python -m src.main
```

**Run and save a simulation** (each `--<group>` defaults to that group's `_medium` file):
```bash
python -m src.main run --out results/my_run
python -m src.main run --project config/project_high.json --out results/high_load
python -m src.main run --human config/human_low.json --agent config/agent_high.json \
    --bugimpact config/bugimpact_high.json --out results/junior_team_critical_system
```

**Analyse persisted results:**
```bash
python -m src.main analyze results/my_run/result.json
```

**Run a configuration search** (random, 10,000 candidates — the §4.2 budget; ~36 min):
```bash
python -m src.main optimize config/search_default.json --out results/search_default
```

**Run evolutionary search:**
```bash
python -m src.main optimize config/search_default.json \
    --out results/search_default_evo --strategy evolutionary
```
Useful `optimize` flags: `--samples {all,frontier,none}` controls which candidates get a
`samples/<key>.json` (default `frontier`; `all` is ~12 KB × 10,000 per search), and
`--progress-every N` throttles the progress lines (`0` silences them).

**Run the full §4.2 campaign** (243 scenarios × 2 strategies = 486 searches, ~300 core-hours):
```bash
python scripts/generate_search_configs.py     # once: writes the 243 config/search/*.json
DRY_RUN=1 ./run_experiments.sh                # list the 486 jobs
LIMIT=2 ./run_experiments.sh                  # smoke test
JOBS=14 ./run_experiments.sh                  # full run; re-invoke to resume
python scripts/collect_results.py             # → frontiers.csv, scenarios.csv
```
Results land in `results/experiments/<scenario_id>/<strategy>/`; `index.tsv` records one
line per finished job. Other env knobs: `FILTER` (config glob), `STRATEGIES`, `SAMPLES`,
`OUT_ROOT`, `CONFIG_DIR`, `FORCE=1` (ignore existing results).

**Run the §5 results analysis** (needs a finished or partial campaign):
```bash
python scripts/analyze_results.py                                  # reads results/experiments
python scripts/analyze_results.py --frontiers results/experiments/frontiers.csv  # fast path
python scripts/analyze_results.py --out-dir /tmp/analysis --format png --per-scenario
```
Writes into `results/analysis/`. §5 splits into two objects, and the outputs follow:

- *Bullet 1 — the recorded three-objective frontier*: `frontiers_derived.csv` (frontier
  rows plus `value_ratio`, the human/agent cost split, `quality_cost_ratio`),
  `value_saturation.csv`, `value_deviation_patterns.csv`, `plots/value_saturation.*`.
- *Bullets 2–3 — the throughput-filtered quality/cost front*: `filtered_front.csv`,
  `filtered_front_summary.csv`, `quality_cost_ranking.csv`, `frontier_extremes.csv`,
  `plots/quality_vs_cost.*` and one `quality_vs_cost_by_<group>.*` facet per scenario
  group. `--per-scenario` additionally dumps one figure per search.

Plus `headline_summary.json` with the numbers §5 quotes in prose.

Frontiers are analysed **per (scenario, strategy)**. `--value-threshold` (default 0.99)
classifies bullet-1 deviations; `--throughput-threshold` (default 0.98) is the §5 filter.

**Generate visualizations:**
```bash
python -m src.main visualize results/search_default [--out results/search_default/plots]
python -m src.main visualize results/run1 results/run2 --out results/comparison
```

**Run tests:**
```bash
.venv/bin/python -m pytest tests/ -v
```

---

## Key Design Decisions

- **Configuration has 6 control variables**: `(N, M, A_dev, A_qa, R, F_freeze)`. `N` = human developer count, `M` = max agents per developer (span of control), `R` = sprints per release, `F_freeze` = length of feature-freeze period before each release.
- **Two 8-element composition vectors**: `A_dev` for development sprints and `A_qa` for feature-freeze (QA) sprints. Both must satisfy `sum ≤ N×M`. `A_dev` requires `a_dev,1 ≥ 1` (at least one Feature Implementation Agent). `A_qa` requires `a_qa,1 = a_qa,2 = 0` (no Feature Implementation or Architecture agents during freeze).
- **Release cycle and feature freeze**: A release fires at sprints R, 2R, 3R, …, escaping all unfixed bugs. The last `F_freeze` sprints of each cycle use `A_qa`; remaining sprints use `A_dev`.
- **Exhaustive search over compositions is infeasible**: `src/search_configurations.py` uses uniform composition sampling (stars-and-bars) and/or evolutionary mutation across both vectors.
- **Feasibility pre-filter** runs before any simulation. The human-capacity condition (§3.3, Eq. monitoring) is checked; infeasible configs are recorded and skipped. They still **consume one evaluation of the search budget** (§4.2), in both strategies, so `n_evaluated` is the budget spend and `n_simulated` the realised simulation count — both are written to `manifest.json`.
- **Scenario parameters are fixed constants**, not sampled: every value in a scenario JSON must be a plain number.
- **Five orthogonal scenario groups** (paper §4.1), each with a `low`/`medium`/`high` tier grounded in literature: `human`, `project`, `debt`, `agent`, `bugimpact` (`SCENARIO_GROUPS` in `src/config.py`). The groups partition the 36 `ScenarioParameters` fields exactly, so `load_scenario_params` takes a `{group: path}` mapping with one file per group and rejects a field defined in two files. Tiers combine freely, giving 3^5 = 243 scenarios. Note tier direction is per-group and not always "high = best": e.g. `debt`'s `delta_min` *decreases* with tier, and the `agent` group's `r_bugs`/`d_test` are non-monotonic because tiers follow model price, not benchmark score.
- **Reproducibility**: every random draw is derived from a master seed via `SeedSequence(s0, spawn_key=(FNV1a_hash(config), k))`.
- **Campaign sample retention**: at the §4.2 budget, one `result.json` per candidate would be ~4.9 M files / ~58 GB, so `--samples` defaults to `frontier` — the `ConfigResult`s are buffered in memory and only frontier members are written, *after* `pareto_frontier` has run. Aggregate means and CIs for every simulated candidate always land in `search_summary.json`.
- **21-dimensional state vector**: features (2), bugs by type×stage (12), technical debt and human capacity queues (5), test infrastructure and documentation (2).
- **The §5 filtered front is computed over all candidates, not the recorded frontier**: `search_summary.json` stores objectives *and* config fields for every simulated candidate, not just the frontier ones. `results_analysis.build_campaign` filters that full pool to `value >= 0.98·r_arrival` and runs a fresh two-objective pass. Restricting to the recorded three-objective frontier would be correct under *strict* dominance (a 2-D non-dominated point cannot be 3-D dominated) but loses points under the CI-aware rule of §3.10, because a candidate can be eliminated in 3-D on a *value* significance that no longer applies once value is dropped. The 2-D pass reuses `pareto.pareto_frontier` with the value dimension flattened to a constant — do not write a second dominance implementation. Note the resulting front is only monotone in (quality, cost) after dropping the CI-co-retained points; that is expected.
- **The human/agent cost split is reconstructed, not stored**: `cost` is a single scalar all the way from `ReplicationResult` to `frontiers.csv`. But `C_human` and `C_agents` (`simulation.cost_components`) are deterministic given `(config, params, T)` — only the `max(1, F_completed(T))` denominator is stochastic — so `cost_human = cost · C_human/C_total` is *exact*, for a single replication and for the mean alike. `results_analysis.add_derived_columns` uses this to split any persisted `cost_mean` with no re-simulation. Keep the formula in `cost_components` only; both the simulator and the analysis call it.

---

## Agent Roles (8 AI + 1 human, from paper/main.tex §3.1)

| k | Role | Group |
|---|---|---|
| 1 | Feature Implementation Agent (**required in dev phase: a_dev,1 ≥ 1**) | Production |
| 2 | Architecture & Design Agent (**excluded from QA phase: a_qa,2 = 0**) | Production |
| 3 | Bug Fixing Agent | Production |
| 4 | Refactoring Agent (reduces technical debt D_tech) | Production |
| 5 | Code Review Agent (detection degrades with change volume and debt) | Detection |
| 6 | Security Audit Agent (detects security-class bugs; degrades with change volume and debt) | Detection |
| 7 | Test Development Agent (grows test suite; CI detection) | Detection |
| 8 | Developer Documentation Agent (reduces human monitoring overhead) | Support |
| — | Human Developer (count = N; supervises agents, reviews arch proposals, codes) | Human |

---

## Three Pareto Objectives

| Name | Formula | Direction |
|---|---|---|
| value | F_completed(T) / T | maximize |
| quality | (ω_min·B_esc,min + ω_maj·B_esc,maj + ω_sec·B_esc,sec) / max(1, F_completed(T)) | minimize |
| cost | (C_human + C_agents) / max(1, F_completed(T)) | minimize |

- `C_human = N · c_salary · T/52` (pro-rated annual salary)
- `C_agents = Σ_k a_dev,k · c_token,k · T_dev + Σ_k a_qa,k · c_token,k · T_qa`
- `ω_min ≤ ω_maj ≤ ω_sec` are severity weights for escaped bugs
