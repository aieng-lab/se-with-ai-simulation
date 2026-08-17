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
                        SCENARIO_GROUPS, SEARCH_STRATEGIES, RESULT_MODES,
                        SAMPLE_POLICIES, is_human_only, UNUSED_TIER
    parameters.py       ScenarioParameters (39 fields)
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
    search_configurations.py evolutionary search + greedy neighbourhood descent +
                        _run_human_only / human_only_grid (the §4.1 baseline)
    visualization.py    matplotlib plot functions + visualize_single/visualize_multi/
                        visualize_campaign helpers
    main.py             CLI: `python -m src.main run|analyze|optimize|visualize|campaign`

config/
    horizon_default.json      Monte Carlo protocol of §4.2: T=104, K=100, master_seed=42
    team_default.json         default team configuration (N, R, F_freeze, A_dev, A_qa)
    human_{low,medium,high}.json      scenario group: human developers (§4.2.1);
                                      carries f_qa / h_review / d_qa, read only by a
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
                              epsilon), the evolutionary budget and the descent restart
                              count, all groups medium
    search/search_<id>.json   the 432 generated scenario searches (§4.2); id = h-l_p-v_d-h_a-l_b-m
    startstate_default.json   initial SystemState values (all-zero baseline)

scripts/
    generate_search_configs.py  write the 432 config/search/*.json from search_default.json;
                                --baseline writes the 36 config/search_baseline/*.json
    collect_results.py          consolidate a campaign into optima.csv + scenarios.csv
run_experiments.sh      campaign driver: 864 searches in parallel, resumable (§4.2)

tests/                  pytest suite (452 tests, ~90 s — the campaign figures dominate)
results/                persisted search and campaign results (not committed)
    experiments/        the §4.2 campaign: 432 scenario dirs × 2 strategies, complete —
                        all 864 jobs, the 108 `a-p` scenarios included, exit 0 with an
                        admissible c*.  `index.tsv` per job
    analysis/           optima.csv, scenarios.csv, optima_matrix_<strategy>.png,
                        human_agent_matrix_<strategy>.png
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

**Run a configuration search** (evolutionary, the §4.2 budget of E = 10,000 evaluations):
```bash
python -m src.main optimize config/search_default.json --out results/search_default
```

**Run greedy neighbourhood descent** (§4.2: 20 restarts, each to a local optimum; no budget):
```bash
python -m src.main optimize config/search_default.json \
    --out results/search_default_descent --strategy descent
```
Useful `optimize` flags: `--samples {all,best,none}` controls which candidates get a
`samples/<key>.json` (default `best` — `c*` and the indistinguishable set `I*`; `all` is
~12 KB × 10,000 per search), and `--progress-every N` throttles the progress lines
(`0` silences them).

**Run the human-only baseline** (§4.1 — a team with no agents at all; 99 configurations,
enumerated exhaustively rather than searched, ~20 s per scenario):
```bash
python scripts/generate_search_configs.py --baseline   # 36 config/search_baseline/*.json
python -m src.main optimize config/search_baseline/search_h-h_p-l_d-X_a-X_b-m.json \
    --out results/baseline_one --strategy human_only
CONFIG_DIR=config/search_baseline STRATEGIES=human_only ./run_experiments.sh   # all 36
```
Results land in `results/experiments/<scenario_id>/human_only/`, beside `evolutionary/`
and `descent/`, and `python -m src.main campaign` picks them up automatically.

**Run the full §4.2 campaign** (432 scenarios × 2 strategies = 864 searches; ~215 core-hours
for the evolutionary half, an order of magnitude less for the descent half). All 864 jobs
are on disk and are skipped, so a re-invocation is a no-op unless `FORCE=1` or a new
scenario appears:
```bash
python scripts/generate_search_configs.py --force   # writes the 432 config/search/*.json
                                              # (--force: they already exist)
DRY_RUN=1 ./run_experiments.sh                # list the 864 jobs
FILTER='*_a-p_*' ./run_experiments.sh         # just the pessimistic agent tier
LIMIT=2 ./run_experiments.sh                  # smoke test
JOBS=14 ./run_experiments.sh                  # full run; re-invoke to resume
python -m src.main campaign                   # → optima.csv, scenarios.csv, the figures
```
Results land in `results/experiments/<scenario_id>/<strategy>/`; `index.tsv` records one
line per finished job. Other env knobs: `FILTER` (config glob), `STRATEGIES`, `SAMPLES`,
`OUT_ROOT`, `CONFIG_DIR`, `FORCE=1` (ignore existing results).

**Consolidate a campaign** (`scripts/collect_results.py` is the CSV-only path; both call
`campaign.build_tables`, so the schemas have one owner):
```bash
python -m src.main campaign [results/experiments] [--out results/analysis]
python -m src.main campaign --share human --x debt --color agent   # re-assign channels
python -m src.main campaign --figures-only     # redraw from an existing optima.csv
```
`--figures-only` skips `build_tables` entirely and reads `<--out>/optima.csv` instead, so it
redraws every figure without re-walking the 864 result directories or rewriting either CSV.
That is the path to use on the finished campaign whenever only the plotting code changed.
`baseline_uplift.csv` appears whenever the tree holds §4.1 baselines: one row per
(campaign scenario, search strategy) that has a baseline at the same `human`/`project`/
`bugimpact` tiers, with `cost_ratio = cost_mean(baseline) / cost_mean(c*)`. The join is on
three groups, not five, because the baseline is inert in `debt` and `agent` — so one
baseline row is the reference for all 12 of their combinations.
`optima.csv` is one row per (scenario, strategy) describing `c*`: the five control
variables, both composition vectors, cost with CI, the three drivers in dollars *and* as
shares of the total (`cost_{quality,human,agent}_share`), value with CI, `value_ratio`,
`|I*|`, the per-role work counts and the contribution split. `scenarios.csv` is the search
effort per (scenario, strategy).

Four §5 figures per strategy, all drawn by `visualization._draw_matrix` — the five-channel
tier grid — so they read identically and differ only in the pair of columns on the axes.
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
enforces one group per channel, and each strategy gets its own figures — overlaying them
would draw every scenario twice.

- `optima_matrix_<strategy>.png` (`plot_optima_matrix`): y = total cost per feature,
  x = the share of it one driver accounts for (`--share`, default `agent`). Read it knowing
  the x axis is close to mechanical in the three realistic agent columns: total cost spans
  four decades ($75 – $918 k per feature) while the agent bill per feature barely moves
  (median $13.52, range $2.78 – $64.38 there), so the agent share is essentially 1/cost and
  those panels show a near-diagonal. The `a-p` column is the one that breaks it: at 10× the
  token price the agent bill is $40 – $783 per feature (median $277) and its share reaches
  28 %, so the column is the whole diagonal shifted right by roughly a decade.
- `human_agent_matrix_<strategy>.png` (`plot_human_agent_matrix`): the two cost drivers the
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
- `quality_development_matrix_<strategy>.png` (`plot_quality_development_matrix`): the other
  half of that split — the two team drivers added back together (x = `cost_human_mean +
  cost_agent_mean`, a column derived on a copy inside the plot function, **not** in
  `optima.csv`) against the one the composition only influences (y = `cost_quality_mean`).
  Same unit again, so same treatment as `human_agent_matrix`: shared log range ($7.50 –
  $918 k for the current campaign), equal aspect, 1:1 diagonal. Above the line the escaped
  bugs cost more than the whole team that built the features, which is where 782 of the 864
  optima sit — the reason the total-cost picture is mostly a picture of the quality term.
- `agent_human_ratio_matrix_<strategy>.png` (`plot_agent_human_ratio_matrix`): the
  human/agent trade-off collapsed to one number, y = `cost_agent_mean / cost_human_mean`
  (log, five decades: 0.0008 – 127), against x = **that ratio's percentile rank within the
  frame handed in**, i.e. within one strategy's 432 scenarios, on a linear axis. The panels
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
- **Feasibility pre-filter** runs before any simulation. One rejection path, recorded and skipped: the human-capacity condition of §3.6 Eq. (feas_monitoring) (`failed_step = 1`). It still **counts as one evaluation** (§4.2), in both strategies, so `n_evaluated` is the evaluation count — the budget spend for the evolutionary search, the reported search effort for the descent — and `n_simulated` the realised simulation count; both are written to `manifest.json`. Rejected candidates are reported, never silently redrawn.
- **Scenario parameters are fixed constants**, not sampled: every value in a scenario JSON must be a plain number.
- **Five orthogonal scenario groups** (paper §4.1), each with a `low`/`medium`/`high` tier grounded in literature: `human`, `project`, `debt`, `agent`, `bugimpact` (`SCENARIO_GROUPS` in `src/config.py`). The groups partition the 39 `ScenarioParameters` fields exactly, so `load_scenario_params` takes a `{group: path}` mapping with one file per group and rejects a field defined in two files. Tiers combine freely, but **the grid is not a cube**: two groups carry a fourth tier, each a deliberate stress case rather than an observed operating point — `project`'s `veryhigh` (`r_arrival = 1000` issues/week, unrealistic extreme velocity; the high tier at 75 is already managed by two FTE) and `agent`'s `pessimistic` (the low tier's capability at 10× the high tier's `c_token`; both halves of the agent grounding, the SWE-bench scores and the token prices, are optimistic). That gives 3^3 · 4 · 4 = 432 scenarios. Read the tiers through `config.scenario_tiers(group)` / `config.scenario_tier_combinations()` / `config.scenario_id(tiers)`, never `SCENARIO_TIERS` (the three base tiers) — anything iterating `SCENARIO_TIERS` per group silently drops the `p-v` and `a-p` scenarios. `ALL_SCENARIO_TIERS` is the plotting/ordinal order and `TIER_INITIALS` the `l`/`m`/`h`/`v`/`p` codes the scenario ids and the LaTeX scenario codes share; a tier initial may coincide with a *group* initial (`a-p`), since the id token format is `<group>-<tier>`. Note tier direction is per-group and not always "high = best": e.g. `debt`'s `delta_min` *decreases* with tier, and the `agent` group's `r_bugs`/`d_test` are non-monotonic because tiers follow model price, not benchmark score. `pessimistic` is the only tier that changes cost without changing behaviour — `c_token` enters `simulation.cost_components` alone, so it can neither make a candidate infeasible nor move a saturation bound.
- **The human-only baseline is gated structurally, on the team having no agents at all** (paper §4.1). `config.is_human_only(cfg)` is `not any(cfg.A_dev)` — `TeamConfiguration` forces `A_qa` to zero whenever `A_dev` is zero — and it is the single switch for both baseline changes in `step_sprint`: the `f_qa` capacity reservation of Eq. (hafter_qa), and the human review stage of Eqs. (cov_qa)/(qa_human) that **replaces** Step 5's Test → CR → SA pipeline. There is no mode flag and no `_solo` parameter file: `f_qa` / `h_review` / `d_qa` sit in the ordinary `config/human_*.json` and are simply never read by an agent-supported team. Consequence worth keeping: an agent-supported run is bit-identical to before the baseline landed — the two paths are mutually exclusive, so no stray RNG draw enters. `tests/test_sprint.py` pins that in both directions.
- **`A_dev` may be the all-zero vector, and only the all-zero vector.** `TeamConfiguration.__post_init__` accepts either an all-zero `A_dev` (with an all-zero `A_qa`) or one with `a_dev,1 ≥ 1`; `(0,1,0,…)` still raises. The samplers, `_mutate_composition` and `_composition_neighbours` all keep their `a_1 ≥ 1` floor, so **neither search strategy can ever reach a zero-agent team** — `_run_human_only` is its sole producer.
- **The baseline is provably independent of the `debt` and `agent` groups**, which is why its grid is human(3) × project(4) × bugimpact(3) = **36 scenarios** and not 432. Every `delta_D` consumer multiplies an agent count or `W_agent`; every agent-group parameter multiplies an agent count, or dies in `Φ(·,0) = 0`, or acts on a test suite that never grows. §4.1's "simplified model without debt dynamics" therefore needs **no bypass code** — it is already true, and `tests/test_sprint.py::TestHumanOnlyDebtIsInert` pins it rather than the transition enforcing it. The two inert groups take the `UNUSED_TIER` sentinel, rendered `X`, so a baseline id reads `h-m_p-v_d-X_a-X_b-l`; `config/{debt,agent}_unused.json` supply the placeholder values so the `<group>_<tier>.json` convention holds and `campaign.scenario_params_for_id` needs no special case. `UNUSED_TIER` is deliberately **outside** `TIER_INITIALS` and `ALL_SCENARIO_TIERS` — it must never enter `scenario_tier_combinations()` nor the figure tier channels.
- **`r_bugs_human` is an absolute per-feature rate, not a scale factor** (paper Eq. Binj-human): `B_human(t) ~ Poisson(impl_human'(t) · r_bugs,human)`, with **no** `(2 − δ_D)` amplifier. Human-written code injects at a fixed rate; only the agent stream of Eq. (Binj-agent) is debt-amplified. Values are 0.20 / 0.12 / 0.08 by human tier. This is a change to the **shared** model — it moves mixed-team results too (roughly ×4 on the escaped-bug cost at the medium agent tier) — so **the 864-job campaign on disk predates it and must be re-run before any §5 number is quoted**. `saturation_bounds`' `W_3` deliberately still counts only the agent stream, matching the paper's Eq. (workload); it under-counts, but `u_3 = 1` regardless because `ρ_3 = r_fix·δ_min` is in the hundreds.
- **The baseline is drawn as an overlay, never as its own figure set.** `visualize_campaign` splits the `human_only` rows out *before* the per-strategy loop and passes them to `_draw_matrix` as a `baseline` frame: one fixed black `*`, outside `_TIER_COLORS`/`_TIER_MARKERS`/`_TIER_SIZES` entirely. Do **not** promote it to a fifth tier shape — the existing note that a fifth shape would not survive the smallest size tier still holds. `_baseline_cell` matches a baseline row on the *facet* groups only and treats `UNUSED_TIER` as matching every tier, which is what repeats the same reference points down all four `agent` columns; say so in the caption, it is a reference level rather than an `a-X` observation. The ratio figure takes no baseline at all — a team with no agents has no agent/developer ratio. Axis floors and shared ranges are computed over `_with_baseline(df, baseline)`, or the baseline's exactly-zero agent cost would drop silently off a log axis.
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
authority — and it holds exactly the paper's two *search* strategies. The §4.1 human-only
baseline is a third **result mode**, not a strategy: its configuration space is 99 points,
so `_run_human_only` enumerates it outright and `budget` / `n_starts` are never read.
`RESULT_MODES = SEARCH_STRATEGIES + ("human_only",)` is what `optimize --strategy`,
`campaign.search_dirs` and the result tree range over; anything comparing search *effort*
must keep using `SEARCH_STRATEGIES`. **`budget` (`E`) is the evolutionary strategy's alone** — the descent is bounded
by convergence, not by money, so the two are not compared at equal effort and the effort
the descent needed is itself a reported result (`n_evaluated`).

- **`evolutionary`** (§3.11.4) — `n_samples_initial` random draws, then `n_generations` × `n_offspring` mutations. Parents come 80 % from the **elite** (`rank`'s first `n_elite` entries) and 20 % from the remainder; the `seen` set means no configuration is ever charged twice, and the loop stops once `budget` evaluations are spent. §4.2: `budget = 10,000`, 1,000 / 90 / 100 / `n_elite = 100`.
- **`descent`** (§3.11.5) — `n_starts` restarts, **each run to a local optimum with no evaluation cap**; `budget` is not read at all. Each seeds from `_random_draws` (redrawing an infeasible seed up to `_MAX_SEED_DRAWS = 100` times, the only thing that terminates the seed loop now that no budget does) and repeatedly moves to the `prefers`-best member of `neighbourhood(...)`, stopping when no neighbour is preferred. A revisited candidate is free (`scored`), so an incumbent whose whole neighbourhood has been visited is a local optimum at no cost. Manifest entries carry `restart` and `step` where evolutionary's carry `generation`. §4.2: `n_starts = 20`; realised spend is a few hundred to a few thousand evaluations, well under the evolutionary budget.

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
