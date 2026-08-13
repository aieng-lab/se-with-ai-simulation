# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Research project simulating and optimizing **AI Agent and Human Developer Team Dynamics**. Mixed teams of AI agents and human developers are modelled as a discrete-time stochastic system; Monte Carlo simulation searches for the team that delivers the work a project generates at the **lowest total cost per feature**, counting the developers' salaries, the agents' token bills, and the money the defects that reach production cost.

The simulation is fully operational. The formal model specification is in **`paper/main.tex`** (Section 3).

---

## Repository Layout

```
paper/
    main.tex            formal model specification (AUTHORITATIVE)

src/                    Python simulation package
    config.py           TeamConfiguration, SimulationHorizon, SearchConfiguration,
                        SCENARIO_GROUPS, SEARCH_STRATEGIES, SAMPLE_POLICIES
    parameters.py       ScenarioParameters (36 fields)
    state.py            SystemState
    rng.py              reproducible PCG-64 seeding (FNV-1a config hash)
    io.py               load_simulation_horizon / load_team_config / load_scenario_params
                        / load_search_config / load_initial_state
    feasibility.py      check_feasibility — human-capacity feasibility condition (§3.6)
    sprint.py           step_sprint — full 10-step sprint transition; returns SprintActions
    simulation.py       run_configuration (K replications); escaped_bug_cost,
                        cost_components; SprintSnapshot / ConfigTrace
    selection.py        aggregate_config_result (bootstrap CI), the throughput constraint,
                        the `prefers` preference relation, report_optimum (c* and I*)
    results.py          to_dataframe, save_result/load_result, save_traces/load_traces
    analysis.py         mean_trajectory, quantile_band, final_distribution, convergence_check
    campaign.py         minimal campaign consolidation: optima.csv + scenarios.csv schemas
    search_configurations.py evolutionary search + greedy neighbourhood descent
    visualization.py    matplotlib plot functions + visualize_single/visualize_multi helpers
    main.py             CLI: `python -m src.main run|analyze|optimize|visualize`

config/
    horizon_default.json      Monte Carlo protocol of §4.2: T=104, K=100, master_seed=42
    team_default.json         default team configuration (N, R, F_freeze, A_dev, A_qa)
    human_{low,medium,high}.json      scenario group: human developers (§4.1)
    project_{low,medium,high,veryhigh}.json  scenario group: project dynamics
                                      (veryhigh = the extreme-velocity tier, §4.1)
    debt_{low,medium,high}.json       scenario group: debt dynamics
    agent_{low,medium,high}.json      scenario group: agent costs and throughput
    bugimpact_{low,medium,high}.json  scenario group: dollar cost of an escaped bug
    search_default.json       search config template: §4.2 range (N, R, F_freeze, sigma,
                              epsilon) and budget, all groups medium
    search/search_<id>.json   the 324 generated scenario searches (§4.2); id = h-l_p-v_d-h_a-l_b-m
    startstate_default.json   initial SystemState values (all-zero baseline)

scripts/
    generate_search_configs.py  write the 324 config/search/*.json from search_default.json
    collect_results.py          consolidate a campaign into optima.csv + scenarios.csv
run_experiments.sh      campaign driver: 648 searches in parallel, resumable (§4.2)

tests/                  pytest suite (339 tests, ~2 s)
results/                persisted search and campaign results (not committed). **Empty.**
                        Every earlier campaign predates the single-objective redesign
                        (§3.8-3.11) and is unreadable by the current code, so the full
                        648-search campaign has to be re-run before any §5 analysis.
```

**No §5 results analysis exists yet.** The paper's §5 outline was deleted together with the
three-objective Pareto formulation, and the ~2,200 lines of analysis built on it
(`results_analysis.py`, `composition_analysis.py`, `latex_tables.py`,
`scripts/analyze_results.py`) were removed with it. `src/campaign.py` is the thin
replacement that turns a campaign into two CSVs; the real §5 analysis gets rebuilt once the
paper specifies it. Do not resurrect the deleted modules — every concept they were built on
(front position, quality/cost index, filtered front, archetype ladders ordered by quality)
requires a frontier the model no longer has.

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

**Run a configuration search** (evolutionary, the §4.2 budget of E = 10,000 evaluations):
```bash
python -m src.main optimize config/search_default.json --out results/search_default
```

**Run greedy neighbourhood descent:**
```bash
python -m src.main optimize config/search_default.json \
    --out results/search_default_descent --strategy descent
```
Useful `optimize` flags: `--samples {all,best,none}` controls which candidates get a
`samples/<key>.json` (default `best` — `c*` and the indistinguishable set `I*`; `all` is
~12 KB × 10,000 per search), and `--progress-every N` throttles the progress lines
(`0` silences them).

**Run the full §4.2 campaign** (324 scenarios × 2 strategies = 648 searches, ~400 core-hours):
```bash
python scripts/generate_search_configs.py     # once: writes the 324 config/search/*.json
DRY_RUN=1 ./run_experiments.sh                # list the 648 jobs
LIMIT=2 ./run_experiments.sh                  # smoke test
JOBS=14 ./run_experiments.sh                  # full run; re-invoke to resume
python scripts/collect_results.py             # → optima.csv, scenarios.csv
```
Results land in `results/experiments/<scenario_id>/<strategy>/`; `index.tsv` records one
line per finished job. Other env knobs: `FILTER` (config glob), `STRATEGIES`, `SAMPLES`,
`OUT_ROOT`, `CONFIG_DIR`, `FORCE=1` (ignore existing results).

**Generate visualizations:**
```bash
python -m src.main visualize results/search_default [--out results/search_default/plots]
python -m src.main visualize results/run1 results/run2 --out results/comparison
```
`visualize_single` writes `cost_vs_value` (every evaluated candidate against the throughput
floor, with `c*` and `I*` highlighted and `c*`'s cost CI drawn — the search problem of §3.9
in one picture), `cost_split` (the three cost drivers of the reported optima), and
`trajectories` where `samples/<key>_traces.csv` exists. `visualize_multi` adds
`comparison_optima`, one point per scenario.

**Run tests:**
```bash
.venv/bin/python -m pytest tests/ -v
```

---

## The Objective and the Constraint

A configuration is scored by **one** number, the expected cost per feature of Eq. (cost_rep):

| name | formula | direction |
|---|---|---|
| `cost` | `(C_quality + C_human + C_agents) / max(1, F_completed(T))` | minimise |

formed **within** each replication and then averaged over K, so every replication is charged
for the bugs it let through and the agent work it consumed and credited with the features it
actually delivered. The three drivers are recorded separately (`cost_quality`, `cost_human`,
`cost_agent`) and sum to `cost` exactly:

- `C_quality = ω_min·B_esc,min + ω_maj·B_esc,maj + ω_sec·B_esc,sec` — the ω are the
  **absolute dollar cost** of one escaped bug of that severity (§4.1: 100/150/250,
  25k/300k/7M, 48k/1M/120M), *not* relative quality weights. `simulation.escaped_bug_cost`.
- `C_human = N · c_salary · T/52` (`c_salary` is the annual cost of one FTE, pro-rated)
- `C_agents = Σ_k c_token,k · Σ_t u_k(t)`, where `u_k(t)` is the work role `k` completed in
  sprint `t` (see the cost bullets below)

`value = F_completed(T) / T` survives as the **constrained** quantity, not an objective:

| name | formula | role |
|---|---|---|
| `value` | `F_completed(T) / T` | must satisfy `value ≥ (1-ε)·r_arrival` (ε = 0.02) |

Features cannot be implemented faster than they arrive, so `r_arrival` is the expected
maximum attainable value and ε absorbs the Poisson arrival noise.

---

## Key Design Decisions

- **Configuration has 5 control variables**: `(N, A_dev, A_qa, R, F_freeze)`. `N` = human staffing level, `R` = sprints per release, `F_freeze` = length of feature-freeze period before each release. There is **no span-of-control variable `M`** — it was removed from the model; how many supervised agents a staffing level can carry follows from the human-capacity condition of §3.6 Eq. (feas_monitoring), not from a separate budget.
- **`N` is a staffing level in FTE, not a headcount** (paper §3.1): `N ∈ ℝ_{>0}` and may be fractional — `N = 0.25` is one developer investing a day a week, the §4.2 range is `{0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00, 5.00, 7.00, 10.00}` (quarter-FTE up to a two-pizza team; *not* uniform steps). `net_capacity(H_capacity, c_comm, N) = H_capacity − c_comm·⌈N−1⌉` in `src/config.py` is the per-FTE budget after communication — the peer count follows the *headcount* `⌈N⌉`, since a half-time developer is still one person. `H_capacity` and `c_salary` are both per FTE. The float `N` is folded into `rng._config_hash` at 1/100 FTE resolution.
- **Two 8-element composition vectors**: `A_dev` for development sprints and `A_qa` for feature-freeze (QA) sprints. Neither carries a structural cap on `n_sup` — `TeamConfiguration` deliberately does not check one, because the only collective limit needs the scenario parameters and therefore lives in `src/feasibility.py`. `A_dev` requires `a_dev,1 ≥ 1` (at least one Feature Implementation Agent). `A_qa` requires `a_qa,1 = a_qa,2 = 0` (no Feature Implementation or Architecture agents during freeze).
- **Release cycle and feature freeze**: A release fires at sprints R, 2R, 3R, …, escaping all unfixed bugs. The last `F_freeze` sprints of each cycle use `A_qa`; remaining sprints use `A_dev`.
- **Supervised roles are additionally capped by throughput saturation** (§3.11.1): `search_configurations.saturation_bounds(params, sigma)` returns `u_k = ⌈σ·W_k/ρ_k⌉` per supervised role, where `W_k` is the steady-state work that reaches the role (all of it derived from `r_arrival`) and `ρ_k` the throughput one agent sustains, with the debt amplifier at its floor `delta_min`. `σ` (`SearchConfiguration.sigma`, default and §4.2 value 2) is the headroom for Poisson arrival peaks. Beyond `u_k` an agent simply has no work, so the search does not spend budget there. Note `u_4` is now large (`ρ_4 = r_refactor = 0.10`, so ~15 at medium tiers) — see the refactoring bullet.
- **Exhaustive search over compositions is infeasible**: `src/search_configurations.py` draws each supervised role independently and uniformly in `{0, …, u_k}` (`{1, …, u_1}` for the dev implementer) and/or mutates both vectors evolutionarily. Stars-and-bars composition sampling is gone. Every draw is structurally valid by construction, so `_random_draws` returns `TeamConfiguration`s directly — there is no discarded-draw path in the sampler.
- **Feasibility pre-filter** runs before any simulation. One rejection path, recorded and skipped: the human-capacity condition of §3.6 Eq. (feas_monitoring) (`failed_step = 1`). It still **consumes one evaluation of the search budget** (§4.2), in both strategies, so `n_evaluated` is the budget spend and `n_simulated` the realised simulation count — both are written to `manifest.json`. Rejected candidates are reported, never silently redrawn.
- **Scenario parameters are fixed constants**, not sampled: every value in a scenario JSON must be a plain number.
- **Five orthogonal scenario groups** (paper §4.1), each with a `low`/`medium`/`high` tier grounded in literature: `human`, `project`, `debt`, `agent`, `bugimpact` (`SCENARIO_GROUPS` in `src/config.py`). The groups partition the 36 `ScenarioParameters` fields exactly, so `load_scenario_params` takes a `{group: path}` mapping with one file per group and rejects a field defined in two files. Tiers combine freely, but **the grid is not a cube**: `project` carries a fourth `veryhigh` tier (`r_arrival = 1000` issues/week, a deliberately unrealistic extreme-velocity stress case — the high tier at 75 is already managed by two FTE), giving 3^4 · 4 = 324 scenarios. Read the tiers through `config.scenario_tiers(group)` / `config.scenario_tier_combinations()` / `config.scenario_id(tiers)`, never `SCENARIO_TIERS` (the three base tiers) — anything iterating `SCENARIO_TIERS` per group silently drops the `p-v` scenarios. `ALL_SCENARIO_TIERS` is the plotting/ordinal order and `TIER_INITIALS` the `l`/`m`/`h`/`v` codes the scenario ids and the LaTeX scenario codes share. Note tier direction is per-group and not always "high = best": e.g. `debt`'s `delta_min` *decreases* with tier, and the `agent` group's `r_bugs`/`d_test` are non-monotonic because tiers follow model price, not benchmark score.
- **Reproducibility**: every random draw is derived from a master seed via `SeedSequence(s0, spawn_key=(FNV1a_hash(config), k))`.
- **Campaign sample retention**: at the §4.2 budget, one `result.json` per candidate would be ~4.9 M files / ~58 GB, so `--samples` defaults to `best` — the `ConfigResult`s are buffered in memory and only `c*` and `I*` are written, *after* `report_optimum` has run. Aggregate means and CIs for every simulated candidate always land in `search_summary.json`.
- **The selection rule of §3.10 replaces the CI-aware Pareto frontier, and it is a *total preorder*.** `selection.prefers` is Deb's feasibility rule as the lexicographic key `(violation, cost_mean)`: two violators are separated by whose shortfall is smaller, an admissible configuration precedes every violator, and two admissible ones are separated by cost alone. So `rank` is a plain stable sort, `best_index` is a `min`, and **none of the non-transitivity machinery the old CI-aware multi-objective rule needed applies** — there is no O(n²) unpruned dominance loop and no ordering hazard. Do not reintroduce a dominance relation: with one objective there is nothing for it to do.
- **`report_optimum`, not `best_index`, is what a reported `(c*, I*)` pair must come from.** `I*` (Eq. topset) is the admissible candidates whose cost CI overlaps `c*`'s, reduced by the parsimony tie-break — and that tie-break **can remove the raw cost minimiser**, when a smaller team running the same process is statistically tied with it. Reporting the larger one beside a set that excludes it is exactly what §3.10 forbids, so `report_optimum` re-picks `c*` from the surviving members and guarantees `c* ∈ I*`. `main._cmd_optimize` uses it; `search_configurations` deliberately does **not** — parsimony must not steer the search, only the report.
- **The parsimony tie-break (§3.10) applies only inside `I*`**: among configurations that share `(N, R, F_freeze)`, use no more agents of every role in both phases with strictly fewer of at least one, only the smallest is reported. `selection.parsimony_dominates` takes just the two `ParsimonyKey`s — the statistical half of the rule is the caller's precondition, since every member of `I*` is tied with `c*` by construction. Antisymmetric, so `I*` is never emptied and the result is order-independent. It cannot substitute for the `A_qa` canonicalisation below: componentwise `≤` is a partial order and only collapses chains, so two teams like `(…,3,2,…)` and `(…,2,3,…)` both survive.
- **`A_qa` is canonically zero where `F_freeze == 0`**: with no freeze window `T_qa` is zero and `A_qa` is never applied, so `TeamConfiguration.__post_init__` normalises it to the zero vector (after the shape checks, so a malformed vector still raises). This is what keeps a redundant coordinate from multiplying the search space: `rng._config_hash` folds `A_qa` in, so without it two configurations that are the *same model* draw different RNG streams and land at different objective values while being indistinguishable as teams. The sampler, the mutation operator and the neighbourhood all cooperate — `_random_draws` draws `F_freeze` before the compositions and skips the QA draw, `_mutate` drops the `A_qa` option from its four-way choice at `F_freeze == 0`, and `neighbourhood` skips the whole `A_qa` block there (every QA move would canonicalise straight back to the parent). Measured on a smoke evolutionary search before the fix: 23.9 % of evaluated candidates were behavioural duplicates.
- **Agents are billed for the work they performed, not for being deployed — except for three roles that have no countable output**: `step_sprint` returns a `SprintActions` with the work each role completed that sprint, measured in that role's own throughput unit (features for k=1, where an implemented architecture proposal counts as `ARCH_IMPL_COST_UNITS = 5` features; change requests for k=2, bugs for k=3, tests for k=7, documents for k=8). **k=4, 5, 6 are flat-rate** (`r_k = 1`, `u_k = a_k`): the two background review services have no discrete unit of output, and the Refactoring Agent produces a reduction of the normalised `D_tech` rather than an artefact. `run_configuration` accumulates the vector over the horizon and `simulation.cost_components(config, params, T, actions)` returns `C_agents = Σ_k actions_k · c_token,k`. A demand-driven agent that no work reaches costs nothing; a flat-rate one bills whether or not it has anything to do. Consequence to keep in mind: **cost does not discipline over-provisioning of roles 1–3** — the saturation bound `u_k` and the human-capacity gate are what bound them.
- **The Refactoring Agent is no longer a binary control.** `r_refactor` is denominated in `D_tech` units and set to **0.10** (§4.1: one agent clears a tenth of the maximum debt per sprint, so a fully degraded code base occupies one agent for ten sprints), against a debt inflow of `δ_impl·r_arrival + δ_fix·W_3 ≈ 0.75`/sprint at medium tiers. It therefore takes ~8 agents to hold `D_tech` at zero, `δ_impl` and `δ_fix` are **live parameters** on every configuration, and `u_4` rises from 1 to ~15. The role is billed flat at `c_token,4` = one feature implementation (2.28/3.80/7.60 by tier). This replaced `r_refactor = 747` with per-debt-removed billing, under which one agent pinned `D_tech` to zero and the debt-inflow parameters were inert — so **§5 results on the debt tiers will differ qualitatively from anything measured before this change**, and the corresponding limitation was dropped from the paper's Discussion.
- **An implemented architecture proposal costs five features of *capacity*, not just of budget.** `ARCH_IMPL_COST_UNITS = 5.0` enters Eq. (arch_impl) as a divisor (`arch_impl ≤ ⌊r_F·δ_D·a_1/5⌋`) and Eq. (impl_agents) as a multiplier on the residual (`impl_agents ≤ ⌊r_F·δ_D·a_1⌋ − 5·arch_impl`), *as well as* weighting `units[0]`. It is the same code-base-wide analysis that makes producing a proposal five times as expensive as a feature (`c_token,2 = 5·c_token,1` in every tier). A configuration with `a_1 = 1` and the fixture's `r_F = 3` can therefore implement **no** proposals at all — several tests need `a_1` raised for the arch pipeline to move.
- **The test and documentation agents are capped at the coverage they can still buy**: `step_sprint` writes `min(r_test·a_7, c_test_ceil·F_completed − N_test)` tests and `min(r_doc·a_8, F_completed − N_doc)` documents, and bills exactly that increment. The cap sits precisely where the outer `min` of Eq. (coverage)/(doc_cov) already clipped, so `cov_test` and `doc_cov` are **unchanged** wherever the role is staffed in both phases — only the bill moves. `saturation_bounds` is untouched: `u_7 = u_8 = 1` was already the bound, and the transition now realises it — a second test agent writes nothing, bills nothing, and still consumes monitoring.
- **Who did the work is recorded separately from the bill**: the per-role `actions` vector is agent-only by construction, so it cannot answer "who implemented what". `u_1` conflates agent features with implemented proposals, and the human share appears in no entry at all (human cost is the headcount term of Eq. (cost_human), independent of what the humans produced). `SprintActions` therefore carries four extra scalars beside `units` — `impl_human`, `impl_agents`, `fixes_human`, `fixes_agent`, the paper's impl_human'(t) / impl_agents(t) / Σ n^H_i(t) / fixes_agent(t) of §3.6 — which `run_configuration` sums over the horizon into `ReplicationResult`, `selection` averages into `AggregatedObjectives` (means only, **no bootstrap CIs**), and `main._cmd_optimize` writes to `search_summary.json`. Three things to keep in mind: **(a)** `units` must never be lengthened to hold them — its length is hard-checked in `SprintActions.__post_init__` and `simulation.cost_components`; **(b)** `fixes_agent` deliberately duplicates `actions_3`, and `impl_agents` un-conflates `actions_1` (`arch_impl = (actions_1 - impl_agents) / ARCH_IMPL_COST_UNITS`), so no separate `arch_impl` field is needed; **(c)** the invariant worth testing against is `impl_human + impl_agents == F_completed(T) == value·T`, exact by Eq. (impl).
- **Everything except `cost` and `value` is reporting only.** `AggregatedObjectives` bootstraps exactly two quantities — the objective and the constrained one — and carries the three cost drivers, the per-role `actions_mean` and the four contribution counts as plain means. `prefers` and `I*` membership must never read them; `tests/test_selection.py` pins that.
- **21-dimensional state vector**: features (2), bugs by type×stage (12), technical debt and human capacity queues (5), test infrastructure and documentation (2).

---

## Search Strategies (paper §3.11)

Uniform random sampling is **not** a strategy — it is the shared initialiser
(`_random_draws`) both strategies seed from. `SEARCH_STRATEGIES` in `src/config.py` is the
authority; both receive the same budget `E = 10,000` so the comparison measures search
efficiency, not effort.

- **`evolutionary`** (§3.11.4) — `n_samples_initial` random draws, then `n_generations` × `n_offspring` mutations. Parents come 80 % from the **elite** (`rank`'s first `n_elite` entries) and 20 % from the remainder; the `seen` set means no configuration is ever charged twice, and the loop stops once `budget` evaluations are spent. §4.2: 1,000 / 90 / 100 / `n_elite = 100`.
- **`descent`** (§3.11.5) — `n_starts` restarts of `budget // n_starts` evaluations each. Each seeds from `_random_draws` and repeatedly moves to the `prefers`-best member of `neighbourhood(...)`, stopping at a local optimum or when the restart's share runs out mid-neighbourhood. A revisited candidate is free (`scored`), so an incumbent whose whole neighbourhood has been visited is a local optimum at no cost. Manifest entries carry `restart` and `step` where evolutionary's carry `generation`. §4.2: `n_starts = 5`, i.e. 2,000 evaluations per restart.

`neighbourhood` **enumerates exactly what `_mutate` can reach in one step** — the two
adjacent staffing levels in `N_values`, `F_freeze ± 1` within `[0, R-1]`, and, per phase,
every single-agent add / remove / transfer over the supervised roles plus `±1` on each
background service. `R` is never changed. `tests/test_search_configurations.py::TestNeighbourhood`
pins the containment (`{_mutate(cfg) for 600 draws} ⊆ neighbourhood(cfg)`); if you change
one, change the other.

---

## Agent Roles (8 AI + 1 human, from paper/main.tex §3.1)

| k | Role | Group | Billing |
|---|---|---|---|
| 1 | Feature Implementation Agent (**required in dev phase: a_dev,1 ≥ 1**) | Production | per feature (proposals × 5) |
| 2 | Architecture & Design Agent (**excluded from QA phase: a_qa,2 = 0**) | Production | per change request |
| 3 | Bug Fixing Agent | Production | per bug repaired |
| 4 | Refactoring Agent (reduces technical debt D_tech) | Production | **flat per instance/sprint** |
| 5 | Code Review Agent (detection degrades with change volume and debt) | Detection | **flat per instance/sprint** |
| 6 | Security Audit Agent (detects security-class bugs; degrades with change volume and debt) | Detection | **flat per instance/sprint** |
| 7 | Test Development Agent (grows test suite; CI detection) | Detection | per test written |
| 8 | Developer Documentation Agent (reduces human monitoring overhead) | Support | per document written |
| — | Human Developer (count = N; supervises agents, reviews arch proposals, codes) | Human | `c_salary · N · T/52` |
