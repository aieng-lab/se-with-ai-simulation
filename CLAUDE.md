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
    search_configurations.py random + evolutionary search over agent compositions
    visualization.py    matplotlib plot functions + visualize_single/visualize_multi helpers
    main.py             CLI: `python -m src.main run|analyze|optimize|visualize`

config/
    horizon_default.json      simulation horizon (T, K, master_seed) shared across scenarios
    team_default.json         default team configuration (N, M, R, F_freeze, A_dev, A_qa)
    human_{low,medium,high}.json      scenario group: human developers (§4.1)
    project_{low,medium,high}.json    scenario group: project dynamics
    debt_{low,medium,high}.json       scenario group: debt dynamics
    agent_{low,medium,high}.json      scenario group: agent costs and throughput
    bugimpact_{low,medium,high}.json  scenario group: bug impact
    search_default.json       search config: all five groups at medium tier
    startstate_default.json   initial SystemState values (all-zero baseline)

tests/                  pytest suite (170 tests, ~0.9 s)
results/                persisted sweep results (not committed)
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

**Run a configuration search (random, 500 samples):**
```bash
python -m src.main optimize config/search_default.json --out results/search_default
```

**Run evolutionary search:**
```bash
python -m src.main optimize config/search_default.json \
    --out results/search_default_evo --strategy evolutionary
```

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
- **Feasibility pre-filter** runs before any simulation. The human-capacity condition (§3.3, Eq. monitoring) is checked; infeasible configs are recorded and skipped.
- **Scenario parameters are fixed constants**, not sampled: every value in a scenario JSON must be a plain number.
- **Five orthogonal scenario groups** (paper §4.1), each with a `low`/`medium`/`high` tier grounded in literature: `human`, `project`, `debt`, `agent`, `bugimpact` (`SCENARIO_GROUPS` in `src/config.py`). The groups partition the 36 `ScenarioParameters` fields exactly, so `load_scenario_params` takes a `{group: path}` mapping with one file per group and rejects a field defined in two files. Tiers combine freely, giving 3^5 = 243 scenarios. Note tier direction is per-group and not always "high = best": e.g. `debt`'s `delta_min` *decreases* with tier, and the `agent` group's `r_bugs`/`d_test` are non-monotonic because tiers follow model price, not benchmark score.
- **Reproducibility**: every random draw is derived from a master seed via `SeedSequence(s0, spawn_key=(FNV1a_hash(config), k))`.
- **21-dimensional state vector**: features (2), bugs by type×stage (12), technical debt and human capacity queues (5), test infrastructure and documentation (2).

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
