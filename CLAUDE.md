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
                        SCENARIO_GROUPS, RESULT_MODES,
                        SAMPLE_POLICIES, is_human_only, UNUSED_TIER
    parameters.py       ScenarioParameters (39 fields)
    state.py            SystemState
    rng.py              reproducible PCG-64 seeding (FNV-1a config hash)
    io.py               load_simulation_horizon / load_team_config / load_scenario_params
                        / load_search_config / load_initial_state
    feasibility.py      check_feasibility — human-capacity feasibility condition (§3.6)
    sprint.py           step_sprint — full 10-step sprint transition; returns SprintActions
    simulation.py       run_configuration (K replications, `K=` overridable);
                        escaped_bug_cost, cost_components; SprintSnapshot / ConfigTrace
    selection.py        aggregate_config_result (bootstrap CI), the throughput constraint,
                        the `prefers` preference relation, report_optimum (c* and I*)
    results.py          to_dataframe, save_result/load_result, save_traces/load_traces
    analysis.py         mean_trajectory, quantile_band, final_distribution, convergence_check
    campaign.py         minimal campaign consolidation: optima.csv + scenarios.csv +
                        convergence.csv + elite_turnover.csv schemas, convergence_summary
    search_configurations.py evolutionary search (the only one) +
                        _run_human_only / human_only_grid (the §4.1 baseline) +
                        two_stage_selection (the K'/K selection of §3.10)
    visualization.py    matplotlib plot functions + visualize_single/visualize_multi/
                        visualize_campaign helpers
    main.py             CLI: `python -m src.main run|analyze|optimize|visualize|campaign`

config/
    horizon_default.json      Monte Carlo protocol of §4.2: T=104, K_search=100 (the
                              paper's K'), K_final=1000 (K), master_seed=42
    team_default.json         default team configuration (N, R, F_freeze, A_dev, A_qa)
    human_{low,medium,high}.json      scenario group: human developers (§4.2.1);
                                      carries f_qa / h_qa / d_qa, read only by a
                                      zero-agent team
    {debt,agent}_unused.json  inert placeholders for the two groups the §4.1 baseline
                              does not use; the scenario id says `d-X` / `a-X`
    search_baseline/search_<id>.json  the 36 generated baseline scenarios (§4.1)
    project_{low,medium,high,veryhigh}.json  scenario group: project dynamics
                                      (veryhigh = the extreme-velocity tier, §4.1)
    debt_{low,medium,high}.json       scenario group: debt dynamics
    agent_{low,medium,high,pessimistic}.json  scenario group: agent costs and throughput
                                      (pessimistic = low-tier capability at 10× the high
                                      tier's c_token, §4.1)
    bugimpact_{low,medium,high}.json  scenario group: dollar cost of an escaped bug
    search_default.json       search config template: §4.2 range (N, R, F_freeze, sigma,
                              epsilon) and the evolutionary budget, all groups medium
    search/search_<id>.json   the 432 generated scenario searches (§4.2); id = h-l_p-v_d-h_a-l_b-m
    startstate_default.json   initial SystemState values (all-zero baseline)

scripts/
    generate_search_configs.py  write the 432 config/search/*.json from search_default.json;
                                --baseline writes the 36 config/search_baseline/*.json
    collect_results.py          consolidate a campaign into the four CSVs (no figures)
run_experiments.sh      campaign driver: 432 searches in parallel, resumable (§4.2);
                        MODE=human_only runs the 36 §4.1 baselines instead

tests/                  pytest suite (481 tests, ~90 s — the campaign figures dominate)
results/                persisted search and campaign results (not committed)
    experiments/        STALE.  Holds the pre-2026-08-18 campaign (432 scenario dirs ×
                        `evolutionary` + `descent`), which predates the refactoring-role
                        and scenario-value corrections below and must be regenerated
                        before any §5 number is quoted.  `index.tsv` per job
    analysis/           optima.csv, scenarios.csv, optima_matrix_<mode>.png,
                        human_agent_matrix_<mode>.png — likewise stale
```

**The §4.2 campaign is complete; the §5 write-up is not.** The paper's §5 outline was
deleted together with the three-objective Pareto formulation, and the ~2,200 lines of
analysis built on it (`results_analysis.py`, `composition_analysis.py`, `latex_tables.py`,
`scripts/analyze_results.py`) went with it. What exists now is `src/campaign.py` (two CSVs)
and `visualization.plot_optima_matrix` (the five-dimension optima figure); the rest of §5
gets built once the paper specifies it. Do not resurrect the deleted modules — every concept
they were built on (front position, quality/cost index, filtered front, archetype ladders
ordered by quality) requires a frontier the model no longer has.

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

**Run a configuration search** (evolutionary, the §4.2 budget of E = 10,000 evaluations at
`K_search = 100` replications, followed by the re-simulation of the preliminary `I*` at
`K_final = 1000`, which is outside that budget):
```bash
python -m src.main optimize config/search_default.json --out results/search_default
```

The search itself is **not selectable** — §3.11 describes one — so `optimize` has no
`--strategy` flag.  Useful `optimize` flags: `--samples {all,best,none}` controls which candidates get a
`samples/<key>.json` (default `best` — `c*` and the indistinguishable set `I*`; `all` is
~12 KB × 10,000 per search), and `--progress-every N` throttles the progress lines
(`0` silences them; it also throttles the re-simulation heartbeat).
Each search writes four files: `manifest.json`, `search_summary.json` (the `K_search`
estimates of every candidate in `objectives`, the `K_final` ones of the re-simulated set in
`final_objectives`), `convergence.json` (the §5.1 record) and `samples/`.

**Run the human-only baseline** (§4.1 — a team with no agents at all; 99 configurations,
enumerated exhaustively rather than searched, ~20 s per scenario):
```bash
python scripts/generate_search_configs.py --baseline   # 36 config/search_baseline/*.json
python -m src.main optimize config/search_baseline/search_h-h_p-l_d-X_a-X_b-m.json \
    --human-only --out results/baseline_one
CONFIG_DIR=config/search_baseline MODE=human_only ./run_experiments.sh   # all 36
```
The generated baseline configs already carry `"human_only": true`, so `--human-only` /
`MODE=human_only` only has to be repeated because the result *directory* is named from it.
Results land in `results/experiments/<scenario_id>/human_only/`, beside `evolutionary/`,
and `python -m src.main campaign` picks them up automatically.

**Run the full §4.2 campaign** (432 scenarios, one evolutionary search each; ~215 core-hours
for the search itself, plus the §3.10 re-simulation on top — `|I*_prelim|` × 1,000
replications per scenario, which the tiny runs put at roughly the same order as the search).
The tree on disk predates the 2026-08-18 model corrections *and* the two-stage selection, so
every job in it is stale and the campaign has to be re-run with `FORCE=1` (or after clearing
`results/experiments`) rather than resumed:
```bash
python scripts/generate_search_configs.py --force   # writes the 432 config/search/*.json
                                              # (--force: they already exist)
DRY_RUN=1 ./run_experiments.sh                # list the 432 jobs
FILTER='*_a-p_*' ./run_experiments.sh         # just the pessimistic agent tier
LIMIT=2 ./run_experiments.sh                  # smoke test
JOBS=14 ./run_experiments.sh                  # full run; re-invoke to resume
python -m src.main campaign                   # → the four CSVs and the figures
```
Results land in `results/experiments/<scenario_id>/<mode>/`; `index.tsv` records one
line per finished job. Other env knobs: `FILTER` (config glob), `MODE`, `SAMPLES`,
`OUT_ROOT`, `CONFIG_DIR`, `FORCE=1` (ignore existing results).

**Consolidate a campaign** (`scripts/collect_results.py` is the CSV-only path; both call
`campaign.build_tables`, so the schemas have one owner):
```bash
python -m src.main campaign [results/experiments] [--out results/analysis]
python -m src.main campaign --share human --x debt --color agent   # re-assign channels
python -m src.main campaign --figures-only     # redraw from an existing optima.csv
```
`--figures-only` skips `build_tables` entirely and reads `<--out>/optima.csv` (plus
`elite_turnover.csv` where it exists) instead, so it redraws every figure without re-walking
the 432 result directories or rewriting any CSV.
That is the path to use on the finished campaign whenever only the plotting code changed.
`baseline_uplift.csv` appears whenever the tree holds §4.1 baselines: one row per
campaign scenario that has a baseline at the same `human`/`project`/
`bugimpact` tiers, with `cost_ratio = cost_mean(baseline) / cost_mean(c*)`. The join is on
three groups, not five, because the baseline is inert in `debt` and `agent` — so one
baseline row is the reference for all 12 of their combinations.
`optima.csv` is one row per (scenario, mode) describing `c*` — the `strategy` column
holds `evolutionary` or `human_only` and keeps that name for continuity: the five control
variables, both composition vectors, cost with CI, the three drivers in dollars *and* as
shares of the total (`cost_{quality,human,agent}_share`), value with CI, `value_ratio`,
`|I*|`, the per-role work counts and the contribution split. Every number in it is a
**K_final** estimate: `optimum_row` reads `search_summary.json`'s `final_objectives`, the
re-simulated block of §3.10, and falls back to the K'-replication `objectives` only for a
result written before the two-stage selection existed. `n_indistinguishable` is likewise the
*final* `|I*|`, after the tie-break at `K'`, the re-simulation and the second tie-break.
`scenarios.csv` is the search effort per (scenario, mode), including both replication counts.

`convergence.csv` and `elite_turnover.csv` are the §5.1 reporting, both read from the
per-search `convergence.json` (a small file beside `manifest.json`, deliberately separate so
the campaign does not have to parse the manifest's 10,000-entry `configs` block for four
scalars). `convergence.csv` is one row per (scenario, mode): `tau_first` / `tau_last` /
`tau_mean_last_10` and `incumbent_turnover`, the size of `I*` at all four stages
(`n_prelim` → `n_resimulated` → `n_after_resim` → `n_final`) with `n_discarded_throughput`,
both parsimony counts (`parsimony_removed_prelim` at `K'`, `parsimony_removed` at `K`),
`best_changed_by_resim`, and the relative
half-widths `rho_mean` / `rho_best` of Eq. (rel_halfwidth). `elite_turnover.csv` is long
format — one row per (scenario, mode, generation) — because a search that exhausts its budget
early runs fewer than `G` generations, so a wide table would need padding.
`campaign.convergence_summary` renders the same frame as the text block that
`campaign` prints and writes to `convergence_summary.txt`: order statistics rather than
means, since §5.1 expects those distributions to be strongly skewed.

Four of the five §5 figures per result mode are drawn by `visualization._draw_matrix` — the
five-channel tier grid — so they read identically and differ only in the pair of columns on the axes.
One point per scenario's `c*`; two scenario groups facet the grid (`agent` columns ×
`human` rows → 4 columns × 3 rows) and the other three become marker colour (`debt`), shape (`project`)
and size (`bugimpact`). Two groups have four tiers, and where each sits is deliberate:
**`project` must stay on the shape channel** — shape is the only per-point channel whose
fourth level reads as a peer rather than as "more" — while **`agent` belongs on a facet
axis**, where its fourth tier costs only one extra column. The three per-point channels
(`_TIER_COLORS`/`_TIER_MARKERS`/`_TIER_SIZES` in `src/visualization.py`) give `veryhigh` and
`pessimistic` the *same* fourth slot, which is sound only because no group holds both;
`tests/test_config.py::test_no_group_carries_both_fourth_tiers` pins that. A fifth marker
shape would not survive the smallest size tier, so do not add one. `_check_matrix_groups`
enforces one group per channel, and each mode gets its own figures — overlaying them
would draw every scenario twice.  With one search left the loop runs once, over
`evolutionary`, and the baseline is an overlay rather than a set of its own.

The fifth, `elite_turnover_<mode>.png` (`plot_elite_turnover`), is the §5.1 convergence
figure and deliberately stands outside all of that: convergence is a property of the search,
not of the scenario grid, so it carries **no tier channels and no baseline** — an enumeration
has no generations to turn over, which is also why the `human_only` mode never gets one
(`_turnover_for` returns None for a mode with no rows). It takes the long-format
`elite_turnover.csv`, draws the median `tau_g` with the 5–95 % band over searches, and
overlays the share of searches whose incumbent changed in that generation, because `tau_g`
has a noise floor: candidates that are indistinguishable at `K'` swap ranks inside the elite
without any improvement happening.

**Every measured number in the four bullets below came from the superseded campaign** and
is kept only as a description of what each figure is for and how to read it; re-derive the
ranges from the new `optima.csv` before quoting any of them.

- `optima_matrix_<mode>.png` (`plot_optima_matrix`): y = total cost per feature,
  x = the share of it one driver accounts for (`--share`, default `agent`). Read it knowing
  the x axis is close to mechanical in the three realistic agent columns: total cost spans
  four decades ($75 – $918 k per feature) while the agent bill per feature barely moves
  (median $13.52, range $2.78 – $64.38 there), so the agent share is essentially 1/cost and
  those panels show a near-diagonal. The `a-p` column is the one that breaks it: at 10× the
  token price the agent bill is $40 – $783 per feature (median $277) and its share reaches
  28 %, so the column is the whole diagonal shifted right by roughly a decade.
- `human_agent_matrix_<mode>.png` (`plot_human_agent_matrix`): the two cost drivers the
  team composition actually controls against each other — x = agent cost, y = developer
  cost, both $ per feature, both log. The two are the same unit, so **both axes carry the
  same range** (`_common_log_limits` over both columns, $1 – $10 k for the current campaign)
  at an equal aspect, with the 1:1 diagonal drawn: which side of it a point falls on is
  which driver its optimum spends more on, and almost every optimum sits above it. The
  price is that the agent axis is no longer scaled to its own spread — one decade within a
  column against the four the developer cost spans — so within-column agent differences
  read small, which is their true size. `_draw_matrix`'s `shared_limits` is what does this;
  passing it also makes the panels square, and it is only ever valid for two columns in the
  same unit (never for `plot_optima_matrix`, whose x is a share, and never for the ratio
  figure below, whose x is a rank).
- `quality_development_matrix_<mode>.png` (`plot_quality_development_matrix`): the other
  half of that split — the two team drivers added back together (x = `cost_human_mean +
  cost_agent_mean`, a column derived on a copy inside the plot function, **not** in
  `optima.csv`) against the one the composition only influences (y = `cost_quality_mean`).
  Same unit again, so same treatment as `human_agent_matrix`: shared log range ($7.50 –
  $918 k for the current campaign), equal aspect, 1:1 diagonal. Above the line the escaped
  bugs cost more than the whole team that built the features, where the large majority of
  optima sat on the superseded campaign — the reason the total-cost picture is mostly a picture of the quality term.
- `agent_human_ratio_matrix_<mode>.png` (`plot_agent_human_ratio_matrix`): the
  human/agent trade-off collapsed to one number, y = `cost_agent_mean / cost_human_mean`
  (log, five decades: 0.0008 – 127), against x = **that ratio's percentile rank within the
  frame handed in**, i.e. within the 432 scenarios of one mode, on a linear axis. The panels
  therefore share one empirical distribution and each is a fragment of it: a panel's
  horizontal position says where its scenarios sit campaign-wide, so the reading is which
  part of the axis a panel occupies and how far it spreads vertically — *not* the slope,
  which every ECDF has by construction. Filtering the frame before calling re-ranks what is
  left, so do not pre-slice it. The dashed line at y = 1 is parity; only 23 % of optima are
  above it, and they are concentrated in the `a-p` column (median ratio 0.70 there against
  0.026 / 0.060 / 0.112 at the `low` / `medium` / `high` agent tiers).

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

formed **within** each replication and then averaged over the replications, so every replication is charged
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
- **Roles split into supervised and autonomous** (paper §3.3): `K_sup = {1,2,3,7,8}` (`SUPERVISED_ROLE_INDICES = (0,1,2,6,7)`) and `K_aut = {4,5,6}` (`AUTONOMOUS_ROLE_INDICES = (3,4,5)`). Only `K_sup` enters `n_sup` and therefore Eq. (feas_monitoring); the two detection services and, since the paper made refactoring unsupervised, the **Refactoring Agent** consume no monitoring capacity at all. A team can carry any number of role-4 agents without ever failing the feasibility gate.
- **Supervised roles are additionally capped by throughput saturation** (§3.11.1): `search_configurations.saturation_bounds(params, sigma)` returns `ā_k = ⌈σ·λ_k/η_k⌉` per supervised role, where `λ_k` is the steady-state work that reaches the role (all of it derived from `r_arrival`) and `η_k` the throughput one agent sustains, with the debt amplifier at its floor `delta_min`. `σ` (`SearchConfiguration.sigma`, default and §4.2 value 2) is the headroom for Poisson arrival peaks. Beyond `ā_k` an agent simply has no work, so the search does not spend budget there. Eq. (sat_bound) ranges over `K_sup` only, so **there is no key 3 in the returned dict** — see the refactoring bullet for what bounds role 4 instead.
- **The three autonomous roles are bounded outside `saturation_bounds`, and differently from each other.** `review_sample_max(params) = ⌊r_arrival⌋` is a *sampling range* for k=5,6 only: nothing caps those two, and `_mutate` may take them past it (this is what makes exhaustive search impossible, §3.11). `refactoring_bound(params) = ⌊r_arrival/2⌋` is `ā_4`, a genuine cap honoured by the sampler **and** by the mutation operator (§3.11.2 step 6, §3.11.4). It is *not* scaled by `σ`, and at the four project tiers it is 1 / 6 / 37 / 500.
- **Exhaustive search over compositions is infeasible**: `src/search_configurations.py` draws each supervised role independently and uniformly in `{0, …, ā_k}` (`{1, …, ā_1}` for the dev implementer) and/or mutates both vectors evolutionarily. Stars-and-bars composition sampling is gone. Every draw is structurally valid by construction, so `_random_draws` returns `TeamConfiguration`s directly — there is no discarded-draw path in the sampler.
- **Feasibility pre-filter** runs before any simulation. One rejection path, recorded and skipped: the human-capacity condition of §3.6 Eq. (feas_monitoring) (`failed_step = 1`). It still **counts as one evaluation** (§3.11.2 step 7, §4.3), so `n_evaluated` is the budget spend and `n_simulated` the realised simulation count; both are written to `manifest.json`. Rejected candidates are reported, never silently redrawn — the paper says so explicitly, and a rejection loop would condition every draw on the staffing level drawn with it. `_mutate` likewise tests only the *structural* constraints, so a monitoring-infeasible child is evaluated and rejected rather than retried.
- **Scenario parameters are fixed constants**, not sampled: every value in a scenario JSON must be a plain number.
- **Five orthogonal scenario groups** (paper §4.1), each with a `low`/`medium`/`high` tier grounded in literature: `human`, `project`, `debt`, `agent`, `bugimpact` (`SCENARIO_GROUPS` in `src/config.py`). The groups partition the 39 `ScenarioParameters` fields exactly, so `load_scenario_params` takes a `{group: path}` mapping with one file per group and rejects a field defined in two files. Tiers combine freely, but **the grid is not a cube**: two groups carry a fourth tier, each a deliberate stress case rather than an observed operating point — `project`'s `veryhigh` (`r_arrival = 1000` issues/week, unrealistic extreme velocity; the high tier at 75 is already managed by two FTE) and `agent`'s `pessimistic` (the low tier's capability at 10× the high tier's `c_token`; both halves of the agent grounding, the SWE-bench scores and the token prices, are optimistic). That gives 3^3 · 4 · 4 = 432 scenarios. Read the tiers through `config.scenario_tiers(group)` / `config.scenario_tier_combinations()` / `config.scenario_id(tiers)`, never `SCENARIO_TIERS` (the three base tiers) — anything iterating `SCENARIO_TIERS` per group silently drops the `p-v` and `a-p` scenarios. `ALL_SCENARIO_TIERS` is the plotting/ordinal order and `TIER_INITIALS` the `l`/`m`/`h`/`v`/`p` codes the scenario ids and the LaTeX scenario codes share; a tier initial may coincide with a *group* initial (`a-p`), since the id token format is `<group>-<tier>`. Note tier direction is per-group and not always "high = best": e.g. `debt`'s `delta_min` and `Gamma_doc` both *decrease* as the tier rises (severity up, relief down), while the `agent` group's `r_bugs` falls and its `d_test` rises with price — the paper's footnote resolves the leaderboard's Opus/Fable near-tie in favour of the high tier, so both rows are monotone. `tests/test_io.py` pins all three directions; they have been swapped in the config once already. `pessimistic` is the only tier that changes cost without changing behaviour — `c_token` enters `simulation.cost_components` alone, so it can neither make a candidate infeasible nor move a saturation bound.
- **The human-only baseline is gated structurally, on the team having no agents at all** (paper §4.1). `config.is_human_only(cfg)` is `not any(cfg.A_dev)` — `TeamConfiguration` forces `A_qa` to zero whenever `A_dev` is zero — and it is the single switch for both baseline changes in `step_sprint`: the `f_qa` capacity reservation of Eq. (hafter_qa), and the human review stage of Eqs. (cov_qa)/(qa_human) that **replaces** Step 5's Test → CR → SA pipeline. There is no `_solo` parameter file: `f_qa` / `h_qa` / `d_qa` sit in the ordinary `config/human_*.json`, and `SearchConfiguration.human_only` / `optimize --human-only` only chooses the *enumerator* and are simply never read by an agent-supported team. Consequence worth keeping: an agent-supported run is bit-identical to before the baseline landed — the two paths are mutually exclusive, so no stray RNG draw enters. `tests/test_sprint.py` pins that in both directions.
- **`A_dev` may be the all-zero vector, and only the all-zero vector.** `TeamConfiguration.__post_init__` accepts either an all-zero `A_dev` (with an all-zero `A_qa`) or one with `a_dev,1 ≥ 1`; `(0,1,0,…)` still raises. The samplers and `_mutate_composition` all keep their `a_1 ≥ 1` floor, so **the search can never reach a zero-agent team** — `_run_human_only` is its sole producer.
- **The baseline is provably independent of the `debt` and `agent` groups**, which is why its grid is human(3) × project(4) × bugimpact(3) = **36 scenarios** and not 432. Every `delta_D` consumer multiplies an agent count or `W_agent`; every agent-group parameter multiplies an agent count, or dies in `Φ(·,0) = 0`, or acts on a test suite that never grows. §4.1's "simplified model without debt dynamics" therefore needs **no bypass code** — it is already true, and `tests/test_sprint.py::TestHumanOnlyDebtIsInert` pins it rather than the transition enforcing it. The two inert groups take the `UNUSED_TIER` sentinel, rendered `X`, so a baseline id reads `h-m_p-v_d-X_a-X_b-l`; `config/{debt,agent}_unused.json` supply the placeholder values so the `<group>_<tier>.json` convention holds and `campaign.scenario_params_for_id` needs no special case. `UNUSED_TIER` is deliberately **outside** `TIER_INITIALS` and `ALL_SCENARIO_TIERS` — it must never enter `scenario_tier_combinations()` nor the figure tier channels.
- **`r_bugs_human` is an absolute per-feature rate, not a scale factor** (paper Eq. Binj-human): `B_human(t) ~ Poisson(impl_human'(t) · r_bugs,human)`, with **no** `(2 − δ_D)` amplifier. Human-written code injects at a fixed rate; only the agent stream of Eq. (Binj-agent) is debt-amplified. Values are 0.20 / 0.12 / 0.08 by human tier. This is a change to the **shared** model — it moves mixed-team results too (roughly ×4 on the escaped-bug cost at the medium agent tier) — so **the campaign on disk predates it and must be re-run before any §5 number is quoted**. `saturation_bounds`' `λ_3` deliberately still counts only the agent stream, matching the paper's Eq. (workload); it under-counts, but `ā_3 = 1` regardless because `η_3 = r_fix·δ_min` is in the hundreds.
- **The baseline is drawn as an overlay, never as its own figure set.** `visualize_campaign` splits the `human_only` rows out *before* the per-mode loop and passes them to `_draw_matrix` as a `baseline` frame: one fixed black `*`, outside `_TIER_COLORS`/`_TIER_MARKERS`/`_TIER_SIZES` entirely. Do **not** promote it to a fifth tier shape — the existing note that a fifth shape would not survive the smallest size tier still holds. `_baseline_cell` matches a baseline row on the *facet* groups only and treats `UNUSED_TIER` as matching every tier, which is what repeats the same reference points down all four `agent` columns; say so in the caption, it is a reference level rather than an `a-X` observation. The ratio figure takes no baseline at all — a team with no agents has no agent/developer ratio. Axis floors and shared ranges are computed over `_with_baseline(df, baseline)`, or the baseline's exactly-zero agent cost would drop silently off a log axis.
- **Reproducibility**: every random draw is derived from a master seed via `SeedSequence(s0, spawn_key=(FNV1a_hash(config), k))`. Because the spawn key holds the replication index and nothing about the count, a replication is addressable by index, which is what lets `run_configuration`'s `k_offset` hand the re-simulation of §3.10 a **disjoint** block `[K_search, K_search + K_final)` deterministically.
- **The convergence record of §5.1 is collected in the generation loop, not reconstructed afterwards.** `SearchOutcome` carries `elite_turnover`, `incumbent_changed` and `best_cost`, one entry per generation actually run. The loop keeps the ranking across its body — `rank` is called once before the first generation, for `E_0` over the initial pool, and again at the end of each generation — so the same sort supplies both the parents of the next generation (`P_{g-1}`, as §3.11.4 requires) and `E_g` for `τ_g = |E_g \ E_{g-1}| / n_elite`. The denominator is `n_elite` as Eq. (elite_turnover) writes it, not `|E_g|`. `_run_human_only` leaves all three lists empty; an enumeration has no generations, and `incumbent_turnover` is None there rather than 0.
- **Campaign sample retention**: at the §4.2 budget, one `result.json` per candidate would be ~4.9 M files / ~58 GB, so `--samples` defaults to `best` — the `ConfigResult`s are buffered in memory and only `c*` and `I*` are written, *after* `two_stage_selection` has run. The files written for those keys are the **re-simulated** results (`K_final` replications), because `_cmd_optimize` merges `SelectionReport.final_results` into `outcome.results` and passes `final_aggs` as `write_samples`' `aggs_override` — a sample file must agree with the row `optima.csv` reports. Aggregate means and CIs for every simulated candidate always land in `search_summary.json`, at `K_search`.
- **The selection rule of §3.10 replaces the CI-aware Pareto frontier, and it is a *total preorder*.** `selection.prefers` is Deb's feasibility rule as the lexicographic key `(violation, cost_mean)`: two violators are separated by whose shortfall is smaller, an admissible configuration precedes every violator, and two admissible ones are separated by cost alone. So `rank` is a plain stable sort, `best_index` is a `min`, and **none of the non-transitivity machinery the old CI-aware multi-objective rule needed applies** — there is no O(n²) unpruned dominance loop and no ordering hazard. Do not reintroduce a dominance relation: with one objective there is nothing for it to do.
- **Selection is two-stage, and `two_stage_selection` owns both stages (§3.10).** The search scores every candidate with `K_search = K'` replications; stage 1 is `report_optimum(aggs, floor, keys=…)` **with** parsimony keys, which gives the *preliminary* `c*` and `I*` already reduced by the tie-break (it is called a second time without keys purely to record `n_prelim`, the size before the reduction). Stage 2 re-simulates every member of that reduced set with `K_final = K > K'` replications and repeats the rule on the narrower intervals: a member that no longer satisfies the throughput constraint drops out (`indistinguishable_set` filters on `satisfies_throughput`, so this needs no extra code), `c*` is re-determined and may move, and the parsimony tie-break runs a second time on the better estimates. The re-simulation is **outside** the `E` budget (§4.2) and runs a **disjoint** block of replication indices (`run_configuration(..., K=K_final, k_offset=K_search)`), *not* an extension of the block the search used. That is deliberate and must not be "simplified" away: a candidate is in the preliminary set because its `K'` estimate came out at or near the minimum of thousands of noisy estimates, so that estimate is biased low by the selection itself — reusing those replications would carry a `K'/K` share of the winner's curse into the reported cost, and a fresh block removes it at identical simulation cost. **Applying parsimony at stage 1 is deliberate and is what the cost condition bought**: a member it removes there is both structurally larger and dearer in expectation at `K'`, so it can be dropped before it is re-simulated — every removal is one fewer `K`-replication run. The price, accepted knowingly: the final `c*` can differ from what a stage-2-only tie-break would report if a removed member's noisy `K'` estimate would have come out lowest at `K`. It is statistically tied with its remover and structurally larger, so nothing measurable is lost. Inadmissible searches are the one case with nothing to narrow: the preliminary `I*` is empty, so the single least-violating candidate is re-simulated alone, which is what keeps `optima.csv`'s `value_ratio` a `K` number for every row.
- **`report_optimum`, not `best_index`, is what a reported `(c*, I*)` pair must come from.** `I*` (Eq. topset) is the admissible candidates whose cost CI overlaps `c*`'s, reduced by the parsimony tie-break. **`c*` cannot be removed by that tie-break** and needs no exemption: domination requires a strictly lower mean cost and `c*` is the minimiser, so `c*` stays the raw cost minimiser and is a member of every non-empty `I*`. The other reason to go through `report_optimum` is the inadmissible case: when *no* candidate reaches the throughput floor there is no `c*` in the sense of §3.10, so **`I*` is empty** while the least-violating candidate is still returned and reported (`manifest["best_admissible"] = false`, `n_indistinguishable = 0`). `two_stage_selection` calls it twice (see the bullet above), and `main._cmd_optimize` falls back to `c*` for the `--samples best` file list so an inadmissible search still writes one sample; the *search loop* deliberately does not use it — parsimony must not steer the search, only the report.
- **The parsimony tie-break (§3.10) applies only inside `I*`, and it is cost-aware**: among configurations that share `(N, R, F_freeze)`, one removes another when it uses no more agents of every role in both phases, strictly fewer of at least one, **and** its `cost_mean` is strictly lower. `selection.parsimony_dominates` therefore takes the two `ParsimonyKey`s *and* the two mean costs; only the CI-overlap half of the rule stays the caller's precondition. The cost condition is what makes `c*` untouchable and what makes the tie-break safe to apply at `K'` — do not weaken it to `≤`, which would let an exactly-tied smaller team remove `c*` and bring the exemption back. A strict partial order, so `I*` is never emptied and the dominance filter is order-independent. It cannot substitute for the `A_qa` canonicalisation below: componentwise `≤` is a partial order and only collapses chains, so two teams like `(…,3,2,…)` and `(…,2,3,…)` both survive.
- **`A_qa` is canonically zero where `F_freeze == 0`**: with no freeze window `T_qa` is zero and `A_qa` is never applied, so `TeamConfiguration.__post_init__` normalises it to the zero vector (after the shape checks, so a malformed vector still raises). This is what keeps a redundant coordinate from multiplying the search space: `rng._config_hash` folds `A_qa` in, so without it two configurations that are the *same model* draw different RNG streams and land at different objective values while being indistinguishable as teams. The sampler and the mutation operator cooperate — `_random_draws` draws `F_freeze` before the compositions and skips the QA draw, and `_mutate` drops the `A_qa` option from its four-way choice at `F_freeze == 0` (every QA move would canonicalise straight back to the parent). Measured on a smoke evolutionary search before the fix: 23.9 % of evaluated candidates were behavioural duplicates.
- **Agents are billed for the work they performed, not for being deployed — except for three roles that have no countable output**: `step_sprint` returns a `SprintActions` with the work each role completed that sprint, measured in that role's own throughput unit (features for k=1, where an implemented architecture proposal counts as `ARCH_IMPL_COST_UNITS = 5` features; change requests for k=2, bugs for k=3, tests for k=7, documents for k=8). **k=4, 5, 6 are flat-rate** (`r_k = 1`, `u_k = a_k`): the two review services have no discrete unit of output, and the Refactoring Agent produces a reduction of the normalised `D_tech` rather than an artefact. `run_configuration` accumulates the vector over the horizon and `simulation.cost_components(config, params, T, actions)` returns `C_agents = Σ_k actions_k · c_token,k`. A demand-driven agent that no work reaches costs nothing; a flat-rate one bills whether or not it has anything to do. Consequence to keep in mind: **cost does not discipline over-provisioning of roles 1–3** — the saturation bound `ā_k` and the human-capacity gate are what bound them.
- **The Refactoring Agent is unsupervised and effectively unbounded.** Three things about role 4 together, all recent: (a) it is in `K_aut`, so it consumes no monitoring capacity and cannot make a candidate infeasible; (b) Eq. (sat_bound) no longer covers it, so its only cap is the search-side `refactoring_bound = ⌊r_arrival/2⌋`; (c) Eq. (debt_update) removes `r_refactor·a_·,4` from `D_tech` with **nothing throttling it but the clip at 0** — no queue, no `δ_D` slowdown, and it removes debt even in a sprint that created none. `r_refactor` is denominated in `D_tech` units and calibrated against `δ_impl`/`δ_fix` (§4.2.4: one agent removes the debt one feature or one bug fix adds), so it takes the same **0.02 / 0.06 / 0.13** by *agent* tier — note the tier axis is the agent group's while `δ_impl`/`δ_fix` follow the *debt* group's, and the two are orthogonal. The role is billed flat at `c_token,4` = one feature implementation (2.28/3.80/7.60/76.00 by tier). Both the earlier `r_refactor = 747` (one agent pinned `D_tech` to zero) and the interim tier-invariant `0.10` are gone, so **§5 results on the debt tiers will differ qualitatively from anything measured before this change**.
- **An implemented architecture proposal costs five features of *capacity*, not just of budget.** `ARCH_IMPL_COST_UNITS = 5.0` enters Eq. (arch_impl) as a divisor (`arch_impl ≤ ⌊r_F·δ_D·a_1/5⌋`) and Eq. (impl_agents) as a multiplier on the residual (`impl_agents ≤ ⌊r_F·δ_D·a_1⌋ − 5·arch_impl`), *as well as* weighting `units[0]`. It is the same code-base-wide analysis that makes producing a proposal five times as expensive as a feature (`c_token,2 = 5·c_token,1` in every tier). A configuration with `a_1 = 1` and the fixture's `r_F = 3` can therefore implement **no** proposals at all — several tests need `a_1` raised for the arch pipeline to move.
- **The test and documentation agents are capped at the coverage they can still buy**: `step_sprint` writes `min(r_test·a_7, c_test_ceil·F_completed − N_test)` tests and `min(r_doc·a_8, F_completed − N_doc)` documents, and bills exactly that increment. The cap sits precisely where the outer `min` of Eq. (coverage)/(doc_cov) already clipped, so `cov_test` and `doc_cov` are **unchanged** wherever the role is staffed in both phases — only the bill moves. `saturation_bounds` is untouched: `u_7 = u_8 = 1` was already the bound, and the transition now realises it — a second test agent writes nothing, bills nothing, and still consumes monitoring.
- **Who did the work is recorded separately from the bill**: the per-role `actions` vector is agent-only by construction, so it cannot answer "who implemented what". `u_1` conflates agent features with implemented proposals, and the human share appears in no entry at all (human cost is the headcount term of Eq. (cost_human), independent of what the humans produced). `SprintActions` therefore carries four extra scalars beside `units` — `impl_human`, `impl_agents`, `fixes_human`, `fixes_agent`, the paper's impl_human'(t) / impl_agents(t) / Σ n^H_i(t) / fixes_agent(t) of §3.6 — which `run_configuration` sums over the horizon into `ReplicationResult`, `selection` averages into `AggregatedObjectives` (means only, **no bootstrap CIs**), and `main._cmd_optimize` writes to `search_summary.json`. Three things to keep in mind: **(a)** `units` must never be lengthened to hold them — its length is hard-checked in `SprintActions.__post_init__` and `simulation.cost_components`; **(b)** `fixes_agent` deliberately duplicates `actions_3`, and `impl_agents` un-conflates `actions_1` (`arch_impl = (actions_1 - impl_agents) / ARCH_IMPL_COST_UNITS`), so no separate `arch_impl` field is needed; **(c)** the invariant worth testing against is `impl_human + impl_agents == F_completed(T) == value·T`, exact by Eq. (impl).
- **Everything except `cost` and `value` is reporting only.** `AggregatedObjectives` bootstraps exactly two quantities — the objective and the constrained one — and carries the three cost drivers, the per-role `actions_mean` and the four contribution counts as plain means. `prefers` and `I*` membership must never read them; `tests/test_selection.py` pins that.
- **21-dimensional state vector**: features (2), bugs by type×stage (12), technical debt and human capacity queues (5), test infrastructure and documentation (2).

---

## Search (paper §3.11)

The paper describes **one** search, so the search type is not configurable: there is no
`strategy` field, no `--strategy` flag, and no `SEARCH_STRATEGIES` constant.  Uniform random
sampling is not a strategy either — it is the initialiser (`_random_draws`) the search seeds
from (§3.11.2).  The §4.1 human-only baseline is a **result mode** beside the search, not a
strategy: its configuration space is 99 points, so `_run_human_only` enumerates it outright
and `budget` is never read.  `SearchConfiguration.human_only` (or `optimize --human-only`)
selects it, and `RESULT_MODES = ("evolutionary", "human_only")` is what `campaign.search_dirs`,
the result tree, the `strategy` column of the campaign CSVs and the per-mode figure names
range over.  `main._cmd_optimize` derives the mode from the boolean and still writes it to
`manifest["strategy"]`, so the downstream schemas did not have to change.

- **`evolutionary`** (§3.11.4) — `n_samples_initial` random draws, then `n_generations` × `n_offspring` mutations. Parents come 80 % from the **elite** (`rank`'s first `n_elite` entries) and 20 % from the remainder; the `seen` set means no configuration is ever charged twice, and the loop stops once `budget` evaluations are spent. §4.2: `budget = 10,000`, 1,000 / 90 / 100 / `n_elite = 100`.

`_mutate` perturbs exactly one of `{N, F_freeze, A_dev, A_qa}` (`R` is never mutated), and
`_mutate_composition` applies one of four elementary operations: add / remove / transfer over
the **supervised** roles, or *adjust service* on one of the three **autonomous** roles.  The
last is a separate operation precisely because `K_aut` is not part of the pool the other three
redistribute; within it the two review services are uncapped and the Refactoring Agent is
capped at `ā_4`.

**The service step is `±2^j`, not `±1`** (§3.11.4): `_service_step` draws the sign uniformly
and the exponent `j` uniformly from `{0, …, ⌊log₂ s_k⌋}`, where `s_k` is the role's *sampling*
range of §3.11.2 — `review_sample_max = ⌊r_arrival⌋` for k=5,6 and `refactoring_bound = ā_4`
for k=4 — and clips: floored at 0 for all three, capped at `ā_4` for k=4 only.  The exponential
ladder is what lets the search traverse the service counts at the high project tiers, where
`⌊r_arrival⌋` reaches 1000 and a `±1` walk could not.  Two things to keep straight: `s_k` is a
*step-size* range, so a mutation may still take k=5,6 past `review_max` (nothing caps them);
and `⌊log₂ 0⌋` must never be evaluated — `ā_4` is 0 at `r_arrival < 2`, where the guard leaves
only `j = 0`.  A step that clips back onto the parent value is a no-op, which `_mutate`'s
ten-attempt retry loop then redraws.

The greedy neighbourhood descent and its `neighbourhood` / `_composition_neighbours`
enumeration were **deleted** along with `n_starts` and `_MAX_SEED_DRAWS` when the paper
dropped the second strategy — do not reintroduce them.

---

## Agent Roles (8 AI + 1 human, from paper/main.tex §3.1)

`Oversight` is the `K_sup` / `K_aut` split of §3.3: only the supervised roles consume
monitoring capacity and enter Eq. (feas_monitoring).

| k | Role | Group | Oversight | Billing |
|---|---|---|---|---|
| 1 | Feature Implementation Agent (**required in dev phase: a_dev,1 ≥ 1**) | Production | supervised | per feature (proposals × 5) |
| 2 | Architecture & Design Agent (**excluded from QA phase: a_qa,2 = 0**) | Production | supervised | per change request |
| 3 | Bug Fixing Agent | Production | supervised | per bug repaired |
| 4 | Refactoring Agent (reduces technical debt D_tech) | Production | **autonomous** | **flat per instance/sprint** |
| 5 | Code Review Agent (detection degrades with change volume and debt) | Detection | autonomous | **flat per instance/sprint** |
| 6 | Security Audit Agent (detects security-class bugs; degrades with change volume and debt) | Detection | autonomous | **flat per instance/sprint** |
| 7 | Test Development Agent (grows test suite; CI detection) | Detection | supervised | per test written |
| 8 | Developer Documentation Agent (reduces human monitoring overhead) | Support | supervised | per document written |
| — | Human Developer (count = N; supervises agents, reviews arch proposals, codes) | Human | — | `c_salary · N · T/52` |
