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
    state.py            SystemState
    rng.py              reproducible PCG-64 seeding (FNV-1a config hash)
    io.py               load_simulation_horizon / load_team_config / load_scenario_params
                        / load_search_config / load_initial_state
    feasibility.py      check_feasibility — human-capacity feasibility condition (§3.3)
    sprint.py           step_sprint — full 10-step sprint transition; returns SprintActions
    simulation.py       run_configuration (K replications); SprintSnapshot / ConfigTrace
    pareto.py           aggregate_config_result (bootstrap CI), pareto_frontier (CI-aware)
    results.py          to_dataframe, save_result/load_result, save_traces/load_traces
    analysis.py         mean_trajectory, quantile_band, final_distribution, convergence_check
    composition_analysis.py §5 team compositions on the filtered front: role-presence
                        archetypes, drift along the cost axis, tier sensitivity
    latex_tables.py     LaTeX exports of the §5 tables (longtable snippets to \input;
                        written to <out-dir>/tables/, not typeset by hand)
    results_analysis.py campaign-level §5 analysis: flat-table schema, cost split,
                        value saturation, throughput-filtered quality/cost front
    search_configurations.py random + evolutionary search over agent compositions
    visualization.py    matplotlib plot functions + visualize_single/visualize_multi helpers
    main.py             CLI: `python -m src.main run|analyze|optimize|visualize`

config/
    horizon_default.json      Monte Carlo protocol of §4.2: T=104, K=100, master_seed=42
    team_default.json         default team configuration (N, R, F_freeze, A_dev, A_qa)
    human_{low,medium,high}.json      scenario group: human developers (§4.1)
    project_{low,medium,high}.json    scenario group: project dynamics
    debt_{low,medium,high}.json       scenario group: debt dynamics
    agent_{low,medium,high}.json      scenario group: agent costs and throughput
    bugimpact_{low,medium,high}.json  scenario group: bug impact
    search_default.json       search config template: §4.2 range (N in quarter-FTE steps,
                              R, F_freeze, sigma) and budget, all groups medium
    search/search_<id>.json   the 243 generated scenario searches (§4.2); id = h-l_p-m_d-h_a-l_b-m
    startstate_default.json   initial SystemState values (all-zero baseline)

scripts/
    generate_search_configs.py  write the 243 config/search/*.json from search_default.json
    collect_results.py          consolidate a campaign into frontiers.csv + scenarios.csv
    analyze_results.py          the §5 results analysis: derived tables + figures
run_experiments.sh      campaign driver: 486 searches in parallel, resumable (§4.2)

tests/                  pytest suite (337 tests, ~50 s)
results/                persisted sweep and campaign results (not committed).
                        Everything under results/experiments and results/analysis is
                        stale as of the removal of the span-of-control variable M:
                        the search space, the RNG config hash and the sample-key
                        format (N2_M3_R4_00042 -> N2_R4_00042) all changed, on top of
                        the earlier FTE change. The campaign needs a full re-run.
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
python scripts/analyze_results.py --filtered-front results/analysis/filtered_front.csv
```
The three input sources are mutually exclusive and cost decreasing amounts of work.
`--filtered-front` re-reads a `filtered_front.csv` this script wrote earlier and rebuilds
only what that front alone determines: the figures and `index_optima.{csv,tex}`. The
campaign-level CSVs and `headline_summary.json` are not written, and `value_saturation`
is skipped (every point of that front clears the threshold by construction). It is the
way to iterate on a plot without re-parsing the campaign: ~45 s instead of minutes.
Writes into `results/analysis/`. §5 splits into two objects, and the outputs follow:

- *Bullet 1 — the recorded three-objective frontier*: `frontiers_derived.csv` (frontier
  rows plus `value_ratio`, the human/agent cost split, `quality_cost_ratio` and
  `quality_cost_index`),
  `value_saturation.csv`, `value_deviation_patterns.csv`, `plots/value_saturation.*`.
- *Bullets 2–3 — the throughput-filtered quality/cost front*: `filtered_front.csv`,
  `filtered_front_summary.csv`, `quality_cost_ranking.csv`, `frontier_extremes.csv`,
  `plots/quality_vs_cost.*` and one `quality_vs_cost_by_<group>.*` facet per scenario
  group, plus `plots/quality_cost_matrix_<strategy>.*` and
  `plots/ratio_percentile_matrix_<strategy>.*` (both below).
  `--per-scenario` additionally dumps one figure per search.
- *The per-scenario optimum*: `index_optima.csv` and `tables/index_optima.tex` — the
  lowest-`quality_cost_index` point of each scenario (243 rows; the index is U-shaped
  along a front, so this is the knee, not an end). `results_analysis.index_optimum`
  finds it, defaulting to one row per *front* (486) and taking `by=("scenario_id",)`
  for the table.
- *Team compositions on that front* (below): `composition_prevalence.csv`,
  `composition_by_front_position.csv`, `composition_by_tier.csv`,
  `composition_archetypes.csv`, and the `plots/composition_*` figures.

Plus `headline_summary.json` with the numbers §5 quotes in prose.

Frontiers are analysed **per (scenario, strategy)**. `--value-threshold` (default 0.99)
classifies bullet-1 deviations; `--throughput-threshold` (default 0.98) is the §5 filter.

The two **matrix figures** separate **all five scenario groups at once**: a 3×3 grid
faceting one group's tiers on x and another's on y, with the remaining three as marker
colour, shape and size. Because they draw every front point, they are written one
strategy at a time — random and evolutionary differ by an order of magnitude in point
count. Channel assignment defaults to agent on x, human on y, debt as colour, project as
shape, bugimpact as size, and is set by `--matrix-x`, `--matrix-y`, `--matrix-color`,
`--matrix-marker`, `--matrix-size` (five *different* groups; validated up front).
`--matrix-strategy` restricts output to a single strategy. Both draw through one private
`_draw_matrix` in `src/visualization.py`, differing only in their axis columns:

| figure | x | y |
|---|---|---|
| `quality_cost_matrix_<strategy>.*` | cost (`--matrix-cost-field`: `cost_mean` / `cost_human_mean` / `cost_agent_mean`), log | index, log |
| `ratio_percentile_matrix_<strategy>.*` | front position, linear % | index, log |

Both put the **quality×cost index** on y: `quality_mean * cost_mean`
(`results_analysis.add_quality_cost_index`, `--ratio-cost-field` picks the cost term of
the second figure). Both objectives are *minimised*, so their **product** — not their
ratio — is the scalar that falls whenever either improves; a ratio would rise when cost
improves. Since the throughput filter pins every point to the same feature count, the
per-feature denominators make the index the whole trade-off in one number (weighted
escaped bugs · $ / feature²), and on a log–log quality/cost plane its iso-lines are the
−45° diagonals. The raw two-objective view is unaffected and stays in `quality_vs_cost.*`
and the `quality_vs_cost_by_<group>.*` facets.

`quality_cost_ratio` is **not** a score and is not plotted as one — it is an exchange
rate that fixes the **position along the front** (rank correlation +1.00 with
`quality_mean`, −0.97 with `cost_mean` over the campaign), which is what `rank_frontier`
and `frontier_extremes` need and what x of the second figure is: its percentile within
the point's own (scenario, strategy) front, running 0 % at the quality-dominant end to
100 % at the cost-dominant end (`results_analysis.add_front_position`). Every front spans
the full axis, so a 13-point front is comparable with a 1,190-point one. The index over
that axis is **U-shaped, not monotone** — its minimum is interior on 484 of the 486
campaign fronts, at a median 45 % along the front, with a 6.4× median spread between a
front's best and worst point. That knee is the point of the figure.

Both plot functions derive their columns from the persisted means, so nothing is
re-simulated and no CSV needs regenerating; `add_derived_columns` also persists
`quality_cost_index` for future runs.

The **team composition** figures (`src/composition_analysis.py` derives, `src/visualization.py`
draws) answer what the teams *on* the filtered front look like. Three families, each
written per phase and — for the two that draw one row per archetype/tier — per strategy:

- `composition_archetypes_<phase>_<strategy>.*` — the recurring recipes as a
  quality-ordered ladder. An archetype is a *role-presence signature* (which roles are
  staffed at all), not a cluster: only 43 dev signatures occur on the §4.2 campaign and
  the top 6 cover ~95 %, so this needs no clustering dependency and is deterministic.
  Four panels: supervised mix, background counts, quality/cost median+IQR, prevalence
  and scenario coverage.
- `composition_drift_<phase>.*` — mean composition per within-front cost quantile, one
  row-block per strategy (it facets strategies itself, hence no strategy suffix).
- `composition_tiers_<phase>_<strategy>.*` — 15 scenario tiers × (supervised shares |
  background counts | control variables); colour is the deviation from the group's own
  tier mean, cell text the actual value.

Flags: `--composition-phase {dev,qa,both,all}` (default `all` — writes dev-only, qa-only
and side-by-side), `--composition-strategy`, `--composition-bins` (default 5),
`--composition-archetypes` (default 6), `--no-composition` to skip the ~15 figures.

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

- **Configuration has 5 control variables**: `(N, A_dev, A_qa, R, F_freeze)`. `N` = human staffing level, `R` = sprints per release, `F_freeze` = length of feature-freeze period before each release. There is **no span-of-control variable `M`** — it was removed from the model; how many supervised agents a staffing level can carry follows from the human-capacity condition of §3.6 Eq. (feas_monitoring), not from a separate budget.
- **`N` is a staffing level in FTE, not a headcount** (paper §3.1): `N ∈ ℝ_{>0}` and may be fractional — `N = 0.25` is one developer investing a day a week, the §4.2 range runs `{0.25, 0.50, …, 5.00}`. `net_capacity(H_capacity, c_comm, N) = H_capacity − c_comm·⌈N−1⌉` in `src/config.py` is the per-FTE budget after communication — the peer count follows the *headcount* `⌈N⌉`, since a half-time developer is still one person. `H_capacity` and `c_salary` are both per FTE. The float `N` is folded into `rng._config_hash` at 1/100 FTE resolution.
- **Two 8-element composition vectors**: `A_dev` for development sprints and `A_qa` for feature-freeze (QA) sprints. Neither carries a structural cap on `n_sup` — `TeamConfiguration` deliberately does not check one, because the only collective limit needs the scenario parameters and therefore lives in `src/feasibility.py`. `A_dev` requires `a_dev,1 ≥ 1` (at least one Feature Implementation Agent). `A_qa` requires `a_qa,1 = a_qa,2 = 0` (no Feature Implementation or Architecture agents during freeze).
- **Release cycle and feature freeze**: A release fires at sprints R, 2R, 3R, …, escaping all unfixed bugs. The last `F_freeze` sprints of each cycle use `A_qa`; remaining sprints use `A_dev`.
- **Supervised roles are additionally capped by throughput saturation** (§3.13.1): `search_configurations.saturation_bounds(params, sigma)` returns `u_k = ⌈σ·W_k/ρ_k⌉` per supervised role, where `W_k` is the steady-state work that reaches the role (all of it derived from `r_arrival`) and `ρ_k` the throughput one agent sustains, with the debt amplifier at its floor `delta_min`. `σ` (`SearchConfiguration.sigma`, default and §4.2 value 2) is the headroom for Poisson arrival peaks. Beyond `u_k` an agent simply has no work, so the search does not spend budget there. With the calibrated rates this bites hard: `u_k = 1` for every role in 18 of the 27 project×agent×debt combinations, `u_1 = 2` at the high project tier — so `u_k`, not the feasibility gate, is usually what keeps compositions small.
- **Exhaustive search over compositions is infeasible**: `src/search_configurations.py` draws each supervised role independently and uniformly in `{0, …, u_k}` (`{1, …, u_1}` for the dev implementer) and/or mutates both vectors evolutionarily. Stars-and-bars composition sampling is gone. Every draw is structurally valid by construction, so `_random_draws` returns `TeamConfiguration`s directly — there is no discarded-draw path in the sampler.
- **Feasibility pre-filter** runs before any simulation. One rejection path, recorded and skipped: the human-capacity condition of §3.6 Eq. (feas_monitoring) (`failed_step = 1`). It still **consumes one evaluation of the search budget** (§4.2), in both strategies, so `n_evaluated` is the budget spend and `n_simulated` the realised simulation count — both are written to `manifest.json`. Rejected candidates are reported, never silently redrawn. (The old `failed_step = 0` budget-overrun path went away with `M`.)
- **Scenario parameters are fixed constants**, not sampled: every value in a scenario JSON must be a plain number.
- **Five orthogonal scenario groups** (paper §4.1), each with a `low`/`medium`/`high` tier grounded in literature: `human`, `project`, `debt`, `agent`, `bugimpact` (`SCENARIO_GROUPS` in `src/config.py`). The groups partition the 36 `ScenarioParameters` fields exactly, so `load_scenario_params` takes a `{group: path}` mapping with one file per group and rejects a field defined in two files. Tiers combine freely, giving 3^5 = 243 scenarios. Note tier direction is per-group and not always "high = best": e.g. `debt`'s `delta_min` *decreases* with tier, and the `agent` group's `r_bugs`/`d_test` are non-monotonic because tiers follow model price, not benchmark score.
- **Reproducibility**: every random draw is derived from a master seed via `SeedSequence(s0, spawn_key=(FNV1a_hash(config), k))`.
- **Campaign sample retention**: at the §4.2 budget, one `result.json` per candidate would be ~4.9 M files / ~58 GB, so `--samples` defaults to `frontier` — the `ConfigResult`s are buffered in memory and only frontier members are written, *after* `pareto_frontier` has run. Aggregate means and CIs for every simulated candidate always land in `search_summary.json`.
- **21-dimensional state vector**: features (2), bugs by type×stage (12), technical debt and human capacity queues (5), test infrastructure and documentation (2).
- **The §5 filtered front is computed over all candidates, not the recorded frontier**: `search_summary.json` stores objectives *and* config fields for every simulated candidate, not just the frontier ones. `results_analysis.build_campaign` filters that full pool to `value >= 0.98·r_arrival` and runs a fresh two-objective pass. Restricting to the recorded three-objective frontier would be correct under *strict* dominance (a 2-D non-dominated point cannot be 3-D dominated) but loses points under the CI-aware rule of §3.10, because a candidate can be eliminated in 3-D on a *value* significance that no longer applies once value is dropped. The 2-D pass reuses `pareto.pareto_frontier` with the value dimension flattened to a constant — do not write a second dominance implementation. Note the resulting front is only monotone in (quality, cost) after dropping the CI-co-retained points; that is expected.
- **Composition analysis normalises supervised and background roles differently, and
  weights fronts equally**: the two background services (k=5 Code Review, k=6 Security
  Audit) are unbounded and reach counts of 77 on the front, so a stacked bar over all
  eight roles gives them ~77 % of its width and flattens the six supervised roles into
  nothing. `src/composition_analysis.py` therefore reports supervised roles as a **share
  of `n_sup`** (where the signal is: the Feature Implementation share falls 0.81 → 0.26
  from the cheapest to the priciest front quintile, which the all-eight normalisation
  hides completely) and background roles as **absolute counts**. The per-FTE monitoring
  load is reported separately as `agents_per_fte = n_sup / N`, the quantity
  Eq. (feas_monitoring) bounds (it replaced the M-derived `capacity_util`).
  Separately, every aggregate is averaged per `(scenario_id, strategy)` front *before*
  being averaged across fronts (`_front_mean`): front sizes span 13–1,190 points and
  evolutionary contributes ~10× the points of random, so a flat `groupby.mean()` would
  describe the biggest evolutionary fronts rather than the average front.
- **QA compositions are only valid where `F_freeze > 0`**: with no freeze window `T_qa`
  is zero and `A_qa` is never applied, but the search still writes an arbitrary vector
  into those rows — on the §4.2 campaign 50,680 of the 50,727 `F_freeze == 0` front
  points carry a nonzero unexercised `A_qa`. `composition_analysis.phase_frame` drops
  them and every public function in that module calls it first. Roles 1–2 are also
  structurally zero in QA, so QA panels show six roles, not eight.
- **Agents are billed for the work they performed, not for being deployed**:
  `step_sprint` returns a `SprintActions` with the work each role completed that sprint,
  measured in that role's own throughput unit (features for k=1, change requests for k=2,
  bugs for k=3, debt for k=4, deployed instances for the flat-rate services k=5,6, tests
  for k=7, documents for k=8). `run_configuration` accumulates it over the horizon and
  `simulation.cost_components(config, params, T, actions)` returns
  `C_agents = Σ_k actions_k · c_token,k`. An agent that no work reaches costs nothing.
  Only roles 1–4 changed with this: k=5,6 were already flat per sprint (`r_k = 1`) and
  k=7,8 always produce their full nominal output, so their bills are identical to the
  old nominal-rate formula. Consequence to keep in mind: **cost no longer disciplines
  over-provisioning of roles 1–4** — the saturation bound `u_k` and the human-capacity
  gate are what bound them.
- **The human/agent cost split is recorded, not reconstructed**: because `C_agents` now
  depends on the realised trajectory, no fixed `C_human/C_total` ratio recovers it from
  `cost`. `ReplicationResult` carries `cost_human`, `cost_agent` and the per-role
  `actions`; `AggregatedObjectives` carries their means; both reach `search_summary.json`
  and the flat table. `results_analysis.add_derived_columns` reads them, and falls back to
  `_legacy_cost_components` with a `RuntimeWarning` for campaign files written before this
  change. The two extra means are **reporting only** — the §3.10 dominance rule stays
  three-objective and must not read them.

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

- `C_human = N · c_salary · T/52` (`c_salary` is the annual cost of one FTE, pro-rated)
- `C_agents = Σ_k c_token,k · Σ_t u_k(t)`, where `u_k(t)` is the work role `k` completed in
  sprint `t` (see the cost bullet above)
- `ω_min ≤ ω_maj ≤ ω_sec` are severity weights for escaped bugs
