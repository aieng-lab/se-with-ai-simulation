# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Research project simulating and optimizing **AI Agent and Human Developer Team Dynamics**. Mixed teams of AI agents and human developers are modelled as a discrete-time stochastic system; Monte Carlo simulation finds Pareto-optimal team configurations optimizing for development speed (V), software quality (Q), and cost per feature (C_F).

All four development phases are complete and the simulation is fully operational.

---

## Repository Layout

```
docs/
    CONCEPT.md          original concept
    ROLES.md            15 agent/human roles
    VARIABLES.md        all modelling variables
    FRAMEWORK.md        formal mathematical framework (the authoritative spec)

src/                    Python simulation package
    config.py           TeamConfiguration, SweepSettings
    parameters.py       ScenarioParameters (~74 fields)
    state.py            SystemState, LatencyBuffer
    rng.py              reproducible PCG-64 seeding (FNV-1a config hash)
    sampling.py         ParameterSpec distributions, ScenarioPriors, sample_scenario_parameters
    io.py               load_scenario_config / load_mean_parameters (JSON → priors)
    feasibility.py      check_feasibility — 6-step mean-field stability (§7)
    sprint.py           step_sprint — full §4.1–§4.11 transition + SprintRecord
    simulation.py       run_configuration (K replications); SprintSnapshot / ConfigTrace
    pareto.py           aggregate_config_result (bootstrap CI), pareto_frontier (CI-aware)
    results.py          to_dataframe, save_result/load_result, save_traces/load_traces
    analysis.py         mean_trajectory, quantile_band, final_distribution, convergence_check
    visualization.py    5 matplotlib plot functions (pareto scatter, bar, boxplot, etc.)
    main.py             CLI: `python -m src.main run|analyze`

config/
    default_scenario.json     baseline scenario (λ_F = 2, balanced costs)
    high_load.json            high feature-arrival rate (λ_F = 4)
    low_budget.json           halved AI token + salary costs

experiments/
    search_sweep.py           main search script (random + evolutionary search)
    visualize.py              generates PNG plots from result directories
    configs/
        search_default.json   search config: default scenario
        search_high_load.json search config: high-load scenario
        search_budget.json    search config: low-budget scenario

tests/                  pytest suite (129 tests, ~0.6 s)
results/                persisted sweep results (not committed)
```

---

## Running the Simulation

**Quick smoke test:**
```bash
python -m src.main
```

**Run and save a simulation:**
```bash
python -m src.main run --K 100 --T 52 --out results/my_run
```

**Analyse persisted results:**
```bash
python -m src.main analyze results/my_run/result.json
```

**Run a search sweep (random, 500 samples):**
```bash
python experiments/search_sweep.py experiments/configs/search_default.json \
    --out results/search_default
```

**Run evolutionary search:**
```bash
python experiments/search_sweep.py experiments/configs/search_default.json \
    --out results/search_default_evo --strategy evolutionary
```

**Generate visualizations:**
```bash
python experiments/visualize.py results/search_default [--out results/search_default/plots]
python experiments/visualize.py results/run1 results/run2 --out results/comparison
```

**Run tests:**
```bash
.venv/bin/python -m pytest tests/ -v
```

---

## Key Design Decisions

- **Agent composition A is a 13-element integer tuple** summing to N×M (agents per developer × N developers). Exhaustive search over A is infeasible; `search_sweep.py` uses uniform composition sampling (stars-and-bars) and/or evolutionary mutation.
- **M (span of control) is jointly determined with A**: draw M ∈ [1, M_max], then sample A as a composition of N×M.
- **Feasibility pre-filter** runs before any simulation. Six mean-field stability conditions (§7) are checked using prior means; infeasible configs are recorded and skipped.
- **Scenario parameters** are sampled once per replication from distributions specified in JSON (lognormal, beta, uniform, or fixed). See `src/sampling.py`.
- **Reproducibility**: every random draw is derived from a master seed via `SeedSequence(s0, spawn_key=(FNV1a_hash(config), k))`.
- **τ_CI (CI/CD latency) is not stored in state** — it is recomputed each sprint from `F_completed` and `tests_*`.

---

## Agent Roles (13 AI + 2 human, from docs/ROLES.md)

| k | Role |
|---|---|
| 1 | Requirements Refinement Agent |
| 2 | Architecture & Design Proposal Agent |
| 3 | Feature Implementation Agent (**required: a_3 ≥ 1**) |
| 4 | Bug Fixing Agent |
| 5 | Refactoring Agent |
| 6 | Code Review Agent (throughput unbounded per A8; cost-penalised) |
| 7 | Security Audit Agent |
| 8 | Unit Test Development Agent |
| 9 | Integration Test Development Agent |
| 10 | System & Acceptance Test Development Agent |
| 11 | DevOps/CI-CD Agent |
| 12 | Developer Documentation Agent |
| 13 | User Documentation Agent |
| 14 | Product Owner (fixed singleton, cost via c_PO) |
| 15 | Human Developer (count = N) |

---

## Three Pareto Objectives

| Symbol | Formula | Direction |
|---|---|---|
| V | F_completed(T) / T | maximize |
| Q | B_escaped(T) / max(1, F_completed(T)) | minimize |
| C_F | C_total / max(1, F_completed(T)) | minimize |
