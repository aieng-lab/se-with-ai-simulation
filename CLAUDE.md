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
                        SCENARIO_GROUPS, RESULT_MODES (`search` / `human_only`),
                        SAMPLE_POLICIES, is_human_only, UNUSED_TIER
    parameters.py       ScenarioParameters (37 fields)
    state.py            SystemState
    rng.py              reproducible PCG-64 seeding (FNV-1a config hash)
    io.py               read_json + load_simulation_horizon / load_scenario_params
                        / load_search_config / load_initial_state
    feasibility.py      check_feasibility — human-capacity feasibility condition (§3.6)
    sprint.py           step_sprint — full 10-step sprint transition; returns SprintActions
    simulation.py       run_configuration (K replications, `K=` overridable);
                        escaped_bug_cost, cost_components
    selection.py        aggregate_config_result (bootstrap CI), the throughput constraint,
                        the `preference_key` ranking, report_optimum (the (c*, I*) pair)
    results.py          save_result/load_result (the per-candidate sample JSON)
    campaign.py         the column-name constants (DEV_/QA_/ACTION_PARAMETERS), the
                        shared require_columns / search_rows / baseline_rows helpers, and
                        the campaign consolidation: optima.csv + scenarios.csv +
                        convergence.csv + elite_turnover.csv schemas, convergence_summary,
                        and cost_multipliers — the §5.2 log-cost model (multipliers.csv)
    constellation.py    what the optima are *made of*: the final I* extraction
                        (indistinguishable.csv), the mean/sd of every control
                        variable per cell of a two-group tier grid
                        (constellation_cells.csv + constellation_summary),
                        role_bill — the per-role split of Eq. (cost_agents) — and
                        build_process_summary, the §5.3 role/process shares
                        (process_summary.csv)
    search.py           run_search (the only search) + run_human_only /
                        human_only_grid (the §4.1 baseline) +
                        two_stage_selection (the K'/K selection of §3.10),
                        reporting through the SelectionStats dataclass
    visualization.py    matplotlib plot functions + visualize_campaign /
                        visualize_team_summary helpers
    main.py             CLI: `python -m src.main optimize|campaign|constellation`

config/
    horizon_default.json      Monte Carlo protocol of §4.2: T=104, K_search=100 (the
                              paper's K'), K_final=1000 (K), master_seed=42
    human_{low,medium,high}.json      scenario group: human developers (§4.2.1);
                                      carries h_qa / d_qa, read only by a
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
                              epsilon) and the evaluation budget, all groups medium
    search/search_<id>.json   the 432 generated scenario searches (§4.2); id = h-l_p-v_d-h_a-l_b-m
    startstate_default.json   initial SystemState values (all-zero baseline)

scripts/
    generate_search_configs.py  write the 432 config/search/*.json from search_default.json;
                                --baseline writes the 36 config/search_baseline/*.json.
                                The template is fixed at config/search_default.json;
                                only --out-dir / --baseline / --force are settable.
run_experiments.sh      campaign driver: 432 searches in parallel, resumable (§4.2);
                        MODE=human_only runs the 36 §4.1 baselines instead

tests/                  pytest suite (571 tests, ~75 s — the campaign figures dominate)
results/                persisted search and campaign results (not committed)
    experiments/        the campaign re-run of 2026-08-20: 432 `search` +
                        36 `human_only` scenario dirs, current as of that date.
                        `index.tsv` per job
    analysis/           the consolidation of it — the CSVs and figures below
```

**Stale since 2026-09-22:** the freeze re-inspection (see Key Design Decisions) changed the model after the campaign ran, so `results/`, §5 and every campaign number quoted below predate it until the campaign is rerun. **The §4.2 campaign is complete and §5 is drafted from it in full** (§5.1 convergence,
§5.2 cost trade-offs, §5.3 composition); §6 Discussion and §7 Conclusion are still empty,
as is §2. §5 therefore pins the deliverable exactly: **seven figures and two tables**, and
nothing else. The seven are `elite_turnover`, `human_agent_matrix`,
`quality_development_matrix`, `agent_human_ratio_matrix`, `team_drivers`,
`composition_agent_bill` and `composition_team_bill`; the two tables are the `I*` stage
sizes (from `convergence.csv`) and the cost multipliers (`multipliers.csv`).
`paper/figures/` holds a copy of each under exactly the name the code emits — keep the two
in sync, the four matrix figures once carried a stale `_evolutionary` suffix.

Three rounds of analysis code have been deleted; do not resurrect any of them. **(1)** An earlier
§5 outline went with the three-objective Pareto formulation, taking ~2,200 lines
(`results_analysis.py`, `composition_analysis.py`, `latex_tables.py`,
`scripts/analyze_results.py`) — every concept they were built on (front position,
quality/cost index, filtered front, archetype ladders ordered by quality) requires a
frontier the model no longer has. **(2)** The `visualize` subcommand
(`plot_cost_vs_value` / `plot_cost_split` / `plot_comparison_optima` /
`visualize_single` / `visualize_multi`), the two constellation figures, and the whole
per-sprint trace path (`run --traces`, `analyze traces.csv`, `SprintSnapshot` /
`ReplicationTrace` / `ConfigTrace`, `results.to_dataframe` / `save_traces` / `load_traces`)
were removed once §5 was drafted: the paper reports **nothing** per-sprint — every
objective comes from the terminal state `S(T)` plus an aggregated work record — and none of
those figures appears in it. **(3)** Everything that let a figure or a table be *re-cut*:
the `campaign --x/--y/--color/--marker/--size` and `constellation --rows/--cols` flags,
`CAMPAIGN_CHANNEL_DEFAULTS`, `_matrix_channels`, `_check_matrix_groups`,
`config.require_group_pair`, the `groups=` parameters of `cell_statistics` /
`build_constellation` / `constellation_summary`, the five group parameters of the three
`plot_*_matrix` functions, `visualize_team_summary`'s `block_group` / `bar_group` /
`panel_group` and its channel-collision pooling path, and the `fmt=` parameters of
`save_figure` / `visualize_campaign` / `visualize_team_summary`. §5 reads all seven
figures off one layout and the cell table off one cut, so every one of these validated an
input that only ever took a single value. Dropped with them: `AGENT_ROLE_NAMES` (no
reference anywhere), `SystemState.B_kno` / `B_hid` (tests only — `tests/test_sprint.py`
has module-level helpers instead), `simulation.phase_sprint_counts` with `optima.csv`'s
`T_dev` / `T_qa`, `optima.csv`'s `throughput_floor` (it is `r_arrival · (1-ε)`),
`optimum_row`'s unused `manifest` parameter, and the `agent-sim` console script.
**(4)** The last of the dead flexibility: the `config_dir` parameter of
`scenario_config_path` / `scenario_params_for_id` / `optimum_row` / `build_tables` /
`role_bill` (now the `config.CONFIG_DIR` constant), `admissible_optima`'s `mode=`,
`build_process_summary`'s `by=` (now `constellation.ROLE_SPLIT_GROUP`),
`_padded_log_range`'s `extra_decades_below=`, `plot_team_shape`'s `block_group` /
`bar_group` / `panel_group` **and its `panel_group=None` single-panel branch** — it reads
the `COMPOSITION_*_GROUP` constants directly, so `_ShareWeight.xlabel` and
`config.require_distinct_groups` went with it — and `constellation.search_optima`, a pure
alias for `campaign.search_rows`. **The
figures are byte-identical across all of this** — that is the check to run (below), not a
matter of judgement.

---

## Running the Simulation

The CLI has exactly three subcommands — `optimize`, `campaign`, `constellation` — one per
stage of the paper's pipeline.  There is deliberately **no `run` and no `analyze`**: they
ran and printed a single hand-written configuration, which §5 reports nothing from, and
they were the only readers of `config/team_default.json` and `io.load_team_config` (all
three deleted).  A bare `python -m src.main` prints the usage and exits 2.  To inspect one
persisted configuration, read a search's `samples/<key>.json` with `results.load_result`.

**Run a configuration search** (the §4.2 budget of E = 20,000 evaluations at
`K_search = 100` replications, followed by the re-simulation of the preliminary `I*` at
`K_final = 1000`, which is outside that budget):
```bash
python -m src.main optimize config/search_default.json --out results/search_default
```

There is exactly one search and it is not selectable.  Useful `optimize` flags:
`--samples {best,none}` controls which candidates get a
`samples/<key>.json` (default `best` — `c*` and the indistinguishable set `I*`), and
`--progress-every N` throttles the progress lines
(`0` silences them; it also throttles the re-simulation heartbeat).
Each search writes four files: `manifest.json`, `search_summary.json` (the `K_search`
estimates of every candidate in `objectives`, the `K_final` ones of the re-simulated set in
`final_objectives`), `convergence.json` (the §5.1 record) and `samples/`.

**Run the human-only baseline** (§4.1 — a team with no agents at all; 10,890 configurations,
enumerated exhaustively rather than searched):
```bash
python scripts/generate_search_configs.py --baseline   # 36 config/search_baseline/*.json
python -m src.main optimize config/search_baseline/search_h-h_p-l_d-X_a-X_b-m.json \
    --out results/baseline_one
CONFIG_DIR=config/search_baseline MODE=human_only ./run_experiments.sh   # all 36
```
The enumerator is selected by the search config's `human_only` key alone — the generated
baseline configs carry `"human_only": true` — so there is **no `--human-only` flag**.
`MODE=human_only` still has to be given to the driver, because it names the result
*directory* and does not reach the Python; `CONFIG_DIR` is what picks the 36 baseline
configs, and omitting it would enumerate over the 432 campaign ones.
Results land in `results/experiments/<scenario_id>/human_only/`, beside `search/`,
and `python -m src.main campaign` picks them up automatically.

**Run the full §4.2 campaign** (432 scenarios, one search each; the
2026-08-20 run took 233 core-hours for the searches plus 55 for the §3.10 re-simulation).
The tree on disk is that run; a fresh campaign needs `FORCE=1` (or a cleared
`results/experiments`) rather than a resume:
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

**Consolidate a campaign:**
```bash
python -m src.main campaign [results/experiments] [--out results/analysis]
python -m src.main campaign --figures-only     # redraw from an existing optima.csv
python -m src.main campaign --no-figures       # write the CSVs only
```
`--figures-only` skips `build_tables` entirely and reads `<--out>/optima.csv` and
`elite_turnover.csv` instead, so it redraws every figure without re-walking the 432 result
directories or rewriting any CSV — the path to use on the finished campaign whenever only
the plotting code changed.  It reads `elite_turnover.csv` **only if it exists**: an
all-baseline tree writes none, because an enumeration has no generations, and requiring
it made `--figures-only` fail on a tree the non-`--figures-only` path handles.  There is
no `--format`, no `--config-dir` and **no channel flags** on either subcommand: §5 is
PNG throughout, the scenario JSONs are always read from `config.CONFIG_DIR`, and every figure is
drawn on one fixed layout — `visualization.MATRIX_CHANNELS` for the three optima matrices,
`visualization.COMPOSITION_{BLOCK,BAR,PANEL}_GROUP` for the composition figures and
`constellation.CELL_GROUPS` for the cell grid.  The `--x/--y/--color/--marker/--size` and
`--rows/--cols` flags that used to re-cut them are **gone**: §5 reads every figure off the
one layout, the placements are load-bearing rather than arbitrary (see the matrix bullet
below), and carrying them meant `_check_matrix_groups`, `_matrix_channels`,
`require_group_pair` and a channel-collision pooling path in `visualize_team_summary` all
existed to validate input that only ever took one value.  `--no-figures` is its opposite: rewrite every CSV and the
convergence summary, draw nothing.
`baseline_uplift.csv` appears whenever the tree holds §4.1 baselines: one row per
campaign scenario that has a baseline at the same `human`/`project`/
`bugimpact` tiers, with `cost_ratio = cost_mean(baseline) / cost_mean(c*)` and the baseline's
process and effort split (`baseline_N/R/F_freeze/f_fix/f_qa`). The join is on
three groups, not five, because the baseline is inert in `debt` and `agent` — so one
baseline row is the reference for all 12 of their combinations. It nevertheless carries
**all five** `tier_*` columns and an `admissible` / `baseline_admissible` pair, because
§5.2 reads the comparison *by agent tier* (5.19 / 2.77 / 1.09 / 0.96) and *by debt tier*
(1.98 → 1.06) over the 227 pairs where both sides are admissible — and those two groups
are exactly the ones the join keys cannot supply. Without them the file gave 71.3 % /
1.81 where §5.2 reports 74 % / 1.76 and neither ladder was computable from it at all.
`optima.csv` is one row per (scenario, mode) describing `c*` — the `mode` column holds
`search` or `human_only`: the five control variables, both composition vectors, cost with
CI, the three drivers in dollars *and* as shares of the total
(`cost_{quality,human,agent}_share`), value with CI, `value_ratio`, `|I*|`, the per-role
work counts and the contribution split. Every number in it is a **K_final** estimate:
`optimum_row` reads `search_summary.json`'s `final_objectives`, the re-simulated block of
§3.10. `n_indistinguishable` is likewise the *final* `|I*|`, after the tie-break at `K'`,
the re-simulation and the second tie-break.
`scenarios.csv` is the search effort per (scenario, mode), including both replication counts.

`convergence.csv` and `elite_turnover.csv` are the §5.1 reporting, both read from the
per-search `convergence.json` (a small file beside `manifest.json`, deliberately separate so
the campaign does not have to parse the manifest's 20,000-entry `configs` block for four
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
means, since §5.1 expects those distributions to be strongly skewed. It reports the two
result modes in **separate blocks** and takes the four `I*` stage sizes over the
`best_admissible` rows only, which is what makes the text file Table (results:topset)
rather than an approximation of it: pooled over all 468 runs it printed 990 / 180 / 50 /
34.5 where §5.1 reports 1,068 / 194 / 53 / 37, and 426/468 (91.0 %) where §5.1 reports
410/431 (95.1 %). The reduction-step totals below the stage sizes stay over *all* runs of
the mode — an inadmissible search still spends re-simulation effort — except the `c*`
move rate, whose denominator is the admissible count, since an inadmissible run
re-simulates a single candidate whose `c*` cannot move.

`multipliers.csv` + `multipliers_summary.txt` are §5.2's Table (results:multipliers):
`campaign.cost_multipliers` fits an additive model on `log10(cost_mean)` over the **431
admissible search optima** (`campaign.admissible_optima` — an inadmissible search has an
empty `I*`, so `n_indistinguishable == 0` is the marker) with one dummy per tier and no
interactions. A coefficient exponentiates to a multiplier, so the five groups read as
independent multipliers on the cost per feature, each relative to the **cheapest tier of
its own group** — which is why `multiplier` is 1.0 exactly on the reference row and never
below it. `eta_squared` is the R² of the *single-group* model, i.e. what that group
explains on its own, and is what orders the summary's rows. On the current campaign the fit
is R² = 0.983 at a residual sd of 0.14 decades, and it reproduces the paper's table exactly.

**Analyse the team constellation** (`src/constellation.py`) — what the optimal teams are
*made of*, cut along two scenario groups; `campaign` answers what they cost:
```bash
python -m src.main constellation                          # → the four tables + three figures
python -m src.main constellation --figures-only           # redraw from optima.csv
```
It reads `<--out>/optima.csv` for `c*` and re-walks the result tree for `I*`, so run
`campaign` first; the walk costs about a minute on the 468-directory campaign. The grid is
`constellation.CELL_GROUPS` = `project` × `bugimpact`: 4 × 3 = 12 cells, each holding the
36 scenarios whose `human`, `debt` and `agent` tiers are free. Those two groups because
they shape the optimal team most strongly, and the cut is **fixed** — §5.3 reports the
cell table on it alone, so there is nothing to choose.

**`I*` is `search_summary.json`'s `indistinguishable` key list, *not* its
`final_objectives` block.** `final_objectives` is the **preliminary** set re-simulated at
`K_final`, before the stage-2 throughput filter and the second parsimony tie-break of
§3.10 — in `h-m_p-m_d-m_a-m_b-m` it holds 115 entries against the final 29. The
members' objective dicts are then *looked up* in
`final_objectives`. `campaign.optimum_row` already counts the right one into
`n_indistinguishable`, and `tests/test_constellation.py` pins that the two agree; reading
`final_objectives` wholesale would inflate every cell with configurations the selection
rule rejected.

Four tables and three figures land in `results/analysis/`. **Two populations are reported
separately**: `c_star`, the reported `c*` of each scenario, and `i_star`, the whole
indistinguishable set — `|I*|` has median 36 and reaches 338, so a `c*`-only table partly
tabulates an arbitrary pick inside a statistical tie, and the `i_star` table is what says
whether the pattern survives it (on the current campaign it does, closely). The `i_star`
statistics are **scenario-balanced**: `scenario_means` reduces each scenario's `I*` to one
mean vector *first*, so both populations share a unit of observation and an `n` and their
two `sd` columns are comparable; pooling members instead would let a scenario with
`|I*| = 338` outweigh one with `|I*| = 7` by 48×. `n_members` keeps the pooled size
visible. The human-only baseline is excluded from both — a zero-agent team has no
composition to describe — which is also why these figures take no `baseline` overlay.

`indistinguishable.csv` is one row per `I*` member, using `optima.csv`'s column names so
the two join on `(scenario_id, mode, config_key)`; `is_best` flags `c*`. It is written
for **every** mode in the tree — 21,946 rows on the current campaign, over the 450 of 468
runs with a non-empty `I*`, of which 21,596 are `search` — and
`campaign.search_rows` does the mode filtering the cell tables need, so the file
stays a raw reading of the campaign.
`constellation_cells.csv` is long — one row per (population, cell, parameter) with `mean`,
`sd`, `n_scenarios`, `n_members` — over the 24 `CONSTELLATION_PARAMETERS`: the four
process controls `(N, R, F_freeze, f_fix)`, both 8-element composition vectors, and four derived totals. **`sd` is the spread
across the free scenario groups within the cell, not Monte Carlo error.**
`constellation_summary.txt` renders the same numbers wide as `mean ± sd`, the way
`convergence_summary.txt` renders `convergence.csv`.

Two structural zeros the tables cannot distinguish from a choice, and the summary header
says so: `a_qa_1` and `a_qa_2` are 0 by construction (§3.3), and the **whole `A_qa` vector
is 0 wherever `F_freeze == 0`**, since `TeamConfiguration.__post_init__` canonicalises it
there — so a zero `A_qa` cell means "no feature freeze", not "a freeze with no agents".
On the current campaign `F_freeze == 0` in 68 % of optima and in all but one `p-v` one,
which is what empties that whole row of the `A_qa` block to rounding.

`process_summary.csv` + `process_summary.txt` are the §5.3 statistics that are quoted as
*shares* rather than read off the cell grid: `constellation.build_process_summary` over the
same 431 admissible search optima. Three sections in one long table — `role` (per role: how
often it is staffed, and its share of the head count and of the bill, overall and split by
agent tier), `process` (the share of optima at each `R`, at each `f_fix`, and the share with no
feature freeze), and `istar` (per search, how many distinct `(N, R, F_freeze, f_fix)`
tuples its `I*` spans — the column is still named `n_triples` — and how often `R` / `N` is unique across the tie). **Every share is formed within a
scenario before being averaged** — `role_usage` does this deliberately, since the optimal
team size spans four decades and a ratio of aggregates would report the largest scenario's
composition rather than the typical one; a test pins that a 1000x scenario does not move the
means. The head-count and bill shares disagree sharply and that disagreement *is* the result
(Code Review 69 % of the team, 20 % of the bill; Security 22 % → 46 %; Feature
Implementation 2 % → 21 %), and is the only place that disagreement is reported — the
head-count composition figure that used to draw it has been deleted. The table reproduces every §5.3 percentage
exactly, including the architecture agent's 52/61/22/21 % staffing split by agent tier.

The cell table has **no figures of its own.** `plot_constellation_profile` and
`plot_constellation_heatmap` used to draw it — a per-cell bar chart over the eight roles and
a 12 x 23 heatmap of the whole table — and were **deleted**: §5.3 is written entirely off
`team_drivers` and the two `composition_*` figures below, which state the two facts the 23
parameters collapse onto instead of reporting every parameter. The table itself stays: §5.3's
robustness paragraph ("the mean staffing level of `I*` differs from that of `c*` by a median
of 2 % and at most 14 %") is read off `constellation_cells.csv`. Do not reinstate the two
figures without a place in the paper for them.

**Three figures are drawn from the constellation pass, and they are what §5.3 reports.** The 23
parameters collapse onto two nearly independent facts; `visualize_team_summary`
(called from the constellation subcommand, and redrawn by `--figures-only`) draws one
figure per fact. Both take **`optima.csv`, not `constellation_cells.csv`**: their shares and rates
must be formed *within* a scenario before being averaged, and a ratio of aggregates would
be the largest scenario's ratio rather than the typical one. Do not "simplify" them onto
the cell table. Their three groups are the module constants
`COMPOSITION_BLOCK_GROUP` / `COMPOSITION_BAR_GROUP` / `COMPOSITION_PANEL_GROUP`
(`project` / `bugimpact` / `agent`). `plot_team_shape` reads those three constants
directly rather than taking them as arguments: nothing ever passed anything else, so the
channel-collision validation and the pooled single-panel path it guarded are both gone,
and with them `_ShareWeight.xlabel` and `config.require_distinct_groups`.

- `composition_<what>.png` (`plot_team_shape`): a 100 % stacked horizontal bar per
  (project, bugimpact) cell, eight segments, and **two dollar figures annotated at the
  right end** rather than drawn — they span four decades and any axis showing them would
  flatten the composition. Bars are grouped into blocks by `project` so the same `bugimpact` ladder
  repeats in each: **whether a ladder is reproduced across the blocks is the "which group
  causes this" reading**, and a stacked bar makes an invariance as visible as a change.

  **Three scenario groups are read off the figure, not two**: `COMPOSITION_PANEL_GROUP`
  (`agent`) repeats that whole block structure once per panel, **two panels to a grid
  row**, so the same reading extends to a third group. The agent tier earns a panel
  because `team_drivers.png` puts it among the settings that move the team, and because
  pooling it averages teams the model prices an order of magnitude apart —
  `cost_agent_mean` has median $13.83 at `a-l` and $221.39 at `a-p`. Two to a row rather
  than four abreast: four is a 19-inch strip, and stacking them keeps the left column's
  rows aligned with the right column's, which is what the panels are read against each
  other on. Every panel carries the same rows — that is what makes them comparable — so
  the panels share the row axis, the tier and block labels are drawn once per *grid row*
  (the left column only), the x label once for the whole grid, and only the column of
  absolute totals repeats, since each panel has its own. There is no pooled single-panel
  form any more: the figure has percent ticks and a title ("Share of development costs",
  "Share of agent costs"), which is what the x label used to say.

  **The two annotated columns are the team's bill and the bugs' bill, per feature.** The
  first is the weighting's own total (`_ShareWeight.total_column`); the second is
  `cost_quality_mean`, the third driver of Eq. (cost_rep), which is **annotated but never
  a slice** — the bar describes the team, and what the escaped bugs cost is what that
  team's choices cost elsewhere. Folding it in would turn a team description into a cost
  breakdown; leaving it out entirely, as the figure did at first, leaves the reader
  without the one number that says whether the bill beside it matters, and usually it does
  not: the bug column runs **$83 to $517 k per feature** across the cells while the team
  column stays inside $12 – $3,777. It is therefore rendered by `_compact_dollars`
  (`$83`, `$851`, `$26.2k`, `$517k`) — five characters, three decades, since the
  annotation says which decade the bugs cost in and the bar carries the composition. The
  column appears only where `optima.csv` has `cost_quality_mean`; nothing else needs it.
  `_ANNOTATION_OFFSETS` places both columns in **points** off the bar's end, and the
  inter-panel gap is then *solved for* rather than guessed — `wspace` is a fraction of the
  axes width `tight_layout` has just decided, so the code reads the realised positions
  back and sets it to give every gap exactly `margin` inches. The title and legend strips
  are reserved in inches for the same reason, and `suptitle` is given an explicit `y`
  rather than its default fraction, which would push the title further off a taller grid.
  One layout detail worth keeping: within a grid row the panels
  are drawn **right to left** (`ax.set_zorder`), because a panel's totals column overhangs
  into its neighbour's gridspec cell and would otherwise be painted over by that
  neighbour's background. The margins are reserved in **inches**, not in figure fractions
  — the figure grows with the grid, and a fraction would grow the margins with it.

  **`weight` decides what a slice measures, and both weightings are a bill, not a head
  count.** Each is a `_ShareWeight` carrying its own columns, labels, palette, annotation
  and **file stem** — the stem names what is split, not the key, so `weight="agent"` writes
  `composition_agent_bill.png`. `visualize_team_summary` writes every entry of
  `_SHARE_WEIGHTS`, which is exactly `agent` and `team`.
  A share of the head count systematically overstates the three *autonomous* roles: they
  are flat-rate (`r_k = 1`, so `u_k = a_k`), they consume no monitoring capacity, and
  nothing but their own saturation bounds them, so the search stacks them freely. The two
  views disagree sharply — Code Review is **69 % of the mean optimal team but 20 % of its
  bill**, Security Audit **22 % → 46 %** (its `c_token` is 10× Review's), and Feature
  Implementation **2 % → 21 %**, invisible by head count and a fifth of the budget. The
  bill is what Eq. (cost_rep) actually charges for, so it is the honest unit for "what is
  this team mostly doing", and it is the only one drawn.
  Two head-count renderings are **gone**, and neither should come back. The companion
  dumbbell panel that carried the campaign-wide head-count → bill move per role went with
  `_weight_divergence` / `_draw_divergence` and the `comparable` field of `_ShareWeight`:
  it was a single campaign-wide statement with no honest place beside a 2 × 2 grid of
  panels. The `count` weighting itself (`composition_head_count.png`) followed it: no
  subcommand could reach it, it appears in no §5 figure, and the disagreement it drew is
  already **stated numerically** by `constellation.role_usage` in `process_summary.csv`,
  whose `head count` and `bill` columns are where §5.3's 69/20, 22/46 and 2/21 come from.

  What the cost view then states: **Security Audit takes the bill and `bugimpact` is what
  sets how much of it** — 48 → 69 → 75 % across the `bugimpact` ladder at `a-l`, and
  12 → 37 → 52 % at `a-h` — paid for out of Feature Implementation and Code Review, with
  the ladder repeating near-identically in all four project blocks of every panel. The
  panels are what separate the two effects: the *ladder* is `bugimpact`'s, its *level* is
  the agent tier's, and pooling them (the figure before the panels) reported only their
  product. Feature Implementation moves the other way, 9 % of the agent bill at `a-l` to
  33 % at `a-h`: cheaper, better implementers are worth pointing at the work. The
  right-hand annotation is `cost_agent_mean` — the reported dollars per feature — not the
  summed bill, so it is in the unit §5 quotes.

  **`weight="team"` is the second view and the one that says whether the agents matter at
  all**: the same eight roles plus the developers as a ninth slice, off the agent palette
  in grey so they cannot read as a ninth agent. The agent side is `cost_agent_mean`
  apportioned by the `agent` shares (`AGENT_COST_PARAMETERS`), so the nine slices add to
  `cost_team_mean = cost_human_mean + cost_agent_mean` exactly and the figure is adding
  like for like in dollars per feature. It deliberately **excludes `cost_quality_mean`**
  from the stack, the third driver of Eq. (cost_rep): escaped bugs are what the team's
  choices cost elsewhere, not a line item of the team, and folding them in would make the
  bar a cost breakdown rather than a team description. They are annotated beside it
  instead — see the two annotated columns above.

  What it states, now that the agent tier is a panel rather than something averaged over:
  the developer share of the team bill falls **down** the project tiers and **across** the
  agent tiers, monotonically in both, from 97 % at (`p-l`, `a-l`) to 9 % at (`p-v`, `a-p`)
  — 97/93/89/79 at `p-l`, 96/81/73/70 at `p-m`, 87/62/48/53 at `p-h`, 56/19/13/9 at `p-v`.
  At realistic velocities the optimal team's bill is overwhelmingly salary and only the
  extreme-velocity stress tier flips it to the agents, but *how* overwhelmingly is the
  agent tier's doing: better agents buy headcount down. `a-p` is not a return to `a-l`,
  and the pooled figure made it look like one — those agents have the `a-l` capability at
  10× the price, so they take a *larger* share of the bill than the `a-h` ones do at every
  project tier but `p-h`. Team cost per feature falls the other way across the project
  tiers, since `N` is an overhead spread over more features.

  The bill columns come from **`constellation.role_bill`**, which the *caller* must apply
  (`main._cmd_constellation` does): `bill_k = actions_k · c_token,k` is the exact per-role
  split of Eq. (cost_agents) — `actions_k` is already the work role `k` completed in its
  own throughput unit — and it needs the scenario configs, which `visualization` does not
  read. Summing the eight `bill_k` reproduces `cost_agent_mean · F_completed` to within
  0.3 % (the residual is mean-of-ratios versus ratio-of-means); a **share** has no such gap
  at all, being formed inside a single row where the horizon and the feature count cancel.
  `AGENT_COST_PARAMETERS` sidesteps the gap entirely by rescaling the split onto the
  reported `cost_agent_mean`, which is why the `team` weighting uses those and not
  `bill_k / F_completed`.
- `team_drivers.png` (`plot_team_drivers`, titled *Impact of scenario on team
  composition*): one row per (scenario group, tier) on a shared y axis, **seven panels**, one
  per structural quantity in `_TEAM_QUANTITIES` — the four process controls (`N`, `R`,
  `F_freeze`, `f_fix`), the supervised and autonomous agent counts per arriving feature, and
  Security's share of the detection pair. Each row is a median with an interquartile bar.
  A group whose rows sit on top of each other does not move that quantity; one whose rows
  march across it does. The marginal medians are honest **because the campaign is a full
  factorial**: the scenarios at any one tier are perfectly balanced over the other four
  groups, which is what lets five groups share one axis instead of a facet grid apiece.
  Order statistics rather than mean/sd because several of these quantities pile on a search
  boundary — a third of the optima sit at `N`'s floor and a fifth at its ceiling, and
  `F_freeze` is 0 in 68 % of them. There is **no campaign-median rule**: across seven panels
  it was six more lines of ink for a reference the axis already carries.
  **The agent count is split into supervised and autonomous on purpose** (§3.3): the two
  halves are bounded by different things — the supervised roles by Eq. (feas_monitoring)
  and the saturation bounds, an autonomous one by nothing but its own saturation — and they
  differ by ~40×, so a single "agents per feature" panel is the autonomous count with a
  rounding error and says nothing about the constraint. The two panels keep **separate x
  ranges** even though they share a unit: a shared range was tried and puts every
  autonomous row on one tick, destroying the only thing that panel is for. Reading the
  split off the *positions* is what `composition_*.png` is for. **Every panel is linear**,
  including `N` and the supervised count, which spread over two decades and were on log
  axes at one point.
  This is the one place `veryhigh` and `pessimistic` appear on the same axis, so its
  legend merges them into a single fourth-tier entry; the shared `_TIER_STYLES` fourth slot is
  otherwise safe only because no *group* holds both.

Three of the four §5 figures are drawn by `visualization._draw_matrix` — the five-channel
tier grid — so they read identically and differ only in the pair of columns on the axes.
The two whose axes are both dollars per feature additionally share `_draw_cost_plane`,
which owns the shared log range, the equal aspect and the 1:1 diagonal.
One point per scenario's `c*`; two scenario groups facet the grid (`agent` columns ×
`human` rows → 4 columns × 3 rows) and the other three become marker colour (`debt`), shape (`project`)
and size (`bugimpact`). Two groups have four tiers, and where each sits is deliberate:
**`project` must stay on the shape channel** — shape is the only per-point channel whose
fourth level reads as a peer rather than as "more" — while **`agent` belongs on a facet
axis**, where its fourth tier costs only one extra column. The three per-point channels
(`_TIER_STYLES` in `src/visualization.py`) give `veryhigh` and
`pessimistic` the *same* fourth slot, which is sound only because no group holds both;
`tests/test_config.py::test_no_group_carries_both_fourth_tiers` pins that. A fifth marker
shape would not survive the smallest size tier, so do not add one. The assignment lives in
`visualization.MATRIX_CHANNELS` and is not an argument; that it names five distinct groups
is pinned once by `tests/test_visualization.py::TestMatrixChannels` rather than re-checked
at every call. `visualize_campaign` draws one figure set, for the search
optima; the human-only baseline is an overlay on them rather than a set of its own, so the
figure names carry no mode suffix.

**Every font size in the grid is set explicitly, in `_MATRIX_FONT`, and they are well
above matplotlib's defaults.** The grid is 17–20 inches wide — one panel per tier of two
groups — and is read at `\linewidth` in the paper, a reduction of roughly a third, so what
the sizes are chosen against is the printed size, not the canvas. Enlarging the figure
instead achieves nothing: the same reduction cancels it. Keep the panel geometry fixed when
changing them. Two consequences of the larger type that are already paid for: the legend
column (`_LEGEND_WIDTH_IN`) is wider and `_BASELINE_LABEL` is wrapped, or the longest
colour-legend entry overruns the figure edge; and the per-panel `n = …` is drawn **inside**
the panel, at the bottom right, because on the top row a title would share its line with
the centred facet heading and `agent = pessimistic` reaches both ends of a narrow panel.
Stacking two titles is not an option — matplotlib repositions every title on each draw, and
`pad` is an axes-level offset that moves all of them together. That corner is empty for the
same reason the 1:1 diagonal is worth drawing: nearly every optimum sits above it. The
ratio figure's x label is kept short (`percentile rank of the ratio`) for the same reason —
at `_MATRIX_FONT["axis"]` the long form ran into the next panel's.

The fourth, `elite_turnover.png` (`plot_elite_turnover`), is the §5.1 convergence
figure and deliberately stands outside all of that: convergence is a property of the search,
not of the scenario grid, so it carries **no tier channels and no baseline** — an enumeration
has no generations to turn over, so the `human_only` rows are filtered out of the turnover
table before it is drawn. It takes the long-format
`elite_turnover.csv` and draws **exactly three curves**: `tau_g` in blue on a linear left
axis as a mean (solid) and a median (dotted) over searches, and the **mean** of `c*`'s
**relative, unsigned drop since the previous generation** in green (`_incumbent_cost_drop`,
from the `best_cost` column) on a **logarithmic right-hand axis** labelled in per cent.
Colour separates the quantities, line style the statistics.  The drop is taken in that form
deliberately: relative, since the optima span four decades and a pooled dollar drop would
picture the scenario grid instead of the search; unsigned and per-generation, so it is a
rate like `tau_g`.  It still needs **its own log axis** — on the current campaign it runs
1.0 % → 0.03 %, one to two decades below `tau_g`, and on a shared linear axis it is a flat
line on zero (that layout was tried and rejected, as was a CI band).  Reading the curves
together is the point: `tau_g` has a noise floor — candidates indistinguishable at `K'` swap
ranks inside the elite without any improvement happening — so turnover that persists after
the drop curve has decayed is exactly that churn.  **The drop has no median curve**, and the
log axis is why: that median is exactly 0 in every generation of the current campaign, since
an incumbent moves in a handful of generations and stands still in the rest (44 % of
searches move at `g = 2`, 7 % at `g = 90`), and a zero has no place on a log axis.  The
`_drop_label` guard remains for the degenerate search whose incumbent never moves at all: it
appends "zero throughout — not drawable on a log axis" to the legend entry rather than
letting matplotlib omit the curve silently.  The title is deliberately plain (`Elite turnover and c* drop per generation`) — the search count is
annotated inside the axes.  The `incumbent_changed` column is still written to the CSV; the
figure no longer draws it.

- `human_agent_matrix.png` (`plot_human_agent_matrix`): the two cost drivers the
  team composition actually controls against each other — x = agent cost, y = developer
  cost, both $ per feature, both log. The two are the same unit, so **both axes carry the
  same range** (`_common_log_limits` over both columns, $0.62 – $10 k for the current campaign)
  at an equal aspect, with the 1:1 diagonal drawn: which side of it a point falls on is
  which driver its optimum spends more on, and almost every optimum sits above it. The
  price is that the agent axis is no longer scaled to its own spread — one decade within a
  column against the four the developer cost spans — so within-column agent differences
  read small, which is their true size. `_draw_matrix`'s `shared_limits` is what does this;
  passing it also makes the panels square, and it is only ever valid for two columns in the
  same unit (never for the ratio figure below, whose x is a rank, and never for any figure
  putting a share or a rank on an axis).  A cost-share figure was tried — total cost against
  the share of it one driver accounts for — and **deliberately dropped**: with the agent bill
  per feature nearly constant across the three realistic agent columns, its x axis was close
  to `1/cost` and the panels were a near-diagonal restating the y axis.  Do not reinstate it;
  the `cost_{quality,human,agent}_share` columns stay in `optima.csv`, where they are read as
  numbers rather than plotted.
- `quality_development_matrix.png` (`plot_quality_development_matrix`): the other
  half of that split — the two team drivers added back together (x = `cost_human_mean +
  cost_agent_mean`, derived as `cost_team_mean` on a copy inside the plot function —
  the same name and quantity `constellation.role_bill` writes, but **not** in a raw
  `optima.csv`) against the one the composition only influences (y = `cost_quality_mean`).
  Same unit again, so same treatment as `human_agent_matrix`: shared log range ($5.90 –
  $1.5 M for the current campaign), equal aspect, 1:1 diagonal. Above the line the escaped
  bugs cost more than the whole team that built the features, where 93 % of optima sit —
  the reason the total-cost picture is mostly a picture of the quality term.
- `agent_human_ratio_matrix.png` (`plot_agent_human_ratio_matrix`): the
  human/agent trade-off collapsed to one number, y = `cost_agent_mean / cost_human_mean`
  (log, five decades: 0.003 – 193), against x = **that ratio's percentile rank within the
  frame handed in**, i.e. within the 432 scenarios of one mode, on a linear axis. The panels
  therefore share one empirical distribution and each is a fragment of it: a panel's
  horizontal position says where its scenarios sit campaign-wide, so the reading is which
  part of the axis a panel occupies and how far it spreads vertically — *not* the slope,
  which every ECDF has by construction. Filtering the frame before calling re-ranks what is
  left, so do not pre-slice it. The dashed line at y = 1 is parity; 31 % of optima are
  above it, and they are concentrated in the two dearest agent columns (median ratio 0.52 at
  `a-p` and 0.50 at `a-h`, against 0.035 / 0.228 at the `low` / `medium` agent tiers).

**Run tests:**
```bash
.venv/bin/python -m pytest tests/ -q
```

Two of them are worth knowing about because they are what the paper is held to:

`tests/test_io.py::TestScenarioFilesMatchThePaper` transcribes **every value of §4.2's
Tables 6–10** and asserts the `config/*.json` grid against it, including the `c_token`
vector, the fourth tiers, and the three cost relations the paper states in prose (an
architecture proposal = 5 feature implementations, a security audit = 10 code reviews,
refactoring = one feature implementation). The scenario files are only a rendering of the
manuscript and rows have been silently swapped in them twice before, so a spot check
catches the row it happens to name and nothing else.

`tests/test_simulation.py::TestAgainstThePersistedCampaign` re-simulates the reported `c*`
of four scenarios and compares against `results/analysis/optima.csv` **bit for bit**. It
works because every stream is addressed by `(config_hash, replication index)`:
`run_configuration(cfg, K=K_final, k_offset=K_search)` reproduces exactly the block §3.10
already ran, so the persisted campaign is an *exact* golden record of the transition, the
cost accounting and the bootstrap. Four scenarios spanning both result modes and both
stress tiers cost 8 s, against the ~30 minutes a search would. It catches a change in
*behaviour* — dropping the `(2 - δ_D)` amplifier from Eq. (r_bugs_eff) fails all three
agent-supported scenarios and correctly leaves the human-only one green — but not a
perturbation below the discretisation of the Poisson and binomial draws, which is
unobservable in the model anyway. It skips where `results/analysis/` is absent.

**After any change to the analysis or plotting code, check the figures are unchanged.**
`--figures-only` overwrites `results/analysis/*.png`, so redraw into a scratch copy:
```bash
cp -r results/analysis /tmp/figcheck && rm -f /tmp/figcheck/*.png
.venv/bin/python -m src.main campaign --figures-only --out /tmp/figcheck
.venv/bin/python -m src.main constellation --figures-only --out /tmp/figcheck
for f in paper/figures/*.png; do cmp "$f" "/tmp/figcheck/$(basename $f)"; done
```
All seven are byte-identical today and must stay that way unless §5 changes.

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

- **Configuration has 6 control variables**: `(N, A_dev, A_qa, R, F_freeze, f_fix)`, plus `f_qa` for the human-only baseline. `N` = human staffing level, `R` = sprints per release, `F_freeze` = length of feature-freeze period before each release, `f_fix` = share of the developers' coding time spent on bug fixing (the rest implements features). `f_fix` and `f_qa` used to be scenario constants; both are now searched in 10 % steps (`f_fix_values` / `f_qa_values` in the search config, `_mutate` moves `f_fix` to an adjacent value). `f_qa` is read only by a zero-agent team, so `TeamConfiguration` normalises it to 0 whenever `A_dev` is non-zero — the same reasoning as the `A_qa` canonicalisation. There is **no span-of-control variable `M`** — it was removed from the model; how many supervised agents a staffing level can carry follows from the human-capacity condition of §3.6 Eq. (feas_monitoring), not from a separate budget.
- **`N` is a staffing level in FTE, not a headcount** (paper §3.1): `N ∈ ℝ_{>0}` and may be fractional — `N = 0.25` is one developer investing a day a week, the §4.2 range is `{0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00, 5.00, 7.00, 10.00}` (quarter-FTE up to a two-pizza team; *not* uniform steps). `net_capacity(H_capacity, c_comm, N) = H_capacity − c_comm·⌈N−1⌉` in `src/config.py` is the per-FTE budget after communication — the peer count follows the *headcount* `⌈N⌉`, since a half-time developer is still one person. `H_capacity` and `c_salary` are both per FTE. The float `N` is folded into `rng.config_hash` at 1/100 FTE resolution.
- **Two 8-element composition vectors**: `A_dev` for development sprints and `A_qa` for feature-freeze (QA) sprints. Neither carries a structural cap on `n_sup` — `TeamConfiguration` deliberately does not check one, because the only collective limit needs the scenario parameters and therefore lives in `src/feasibility.py`. `A_dev` requires `a_dev,1 ≥ 1` (at least one Feature Implementation Agent). `A_qa` requires `a_qa,1 = a_qa,2 = 0` (no Feature Implementation or Architecture agents during freeze).
- **Release cycle and feature freeze**: A release fires at sprints R, 2R, 3R, …, escaping all unfixed bugs. The last `F_freeze` sprints of each cycle use `A_qa`; remaining sprints use `A_dev`. A freeze sprint injects nothing; Code Review and Security Audit instead re-inspect the standing hidden queues with the volume `Φ(F_cycle, a)`, the test suite is not re-run, and the re-inspection is assumed independent of the first one (a §3.12 limitation marked REVIEW REQUIRED). The human-only baseline reserves `f_qa` in every sprint and matches `cov_qa` against `F_cycle` in a freeze; mixed-team humans do no QA. Development sprints draw exactly as before, so `F_freeze = 0` configurations are bit-identical to the pre-freeze-review model.
- **Roles split into supervised and autonomous** (paper §3.3): `K_sup = {1,2,3,7,8}` (`SUPERVISED_ROLE_INDICES = (0,1,2,6,7)`) and `K_aut = {4,5,6}` (`AUTONOMOUS_ROLE_INDICES = (3,4,5)`). Only `K_sup` enters `n_sup` and therefore Eq. (feas_monitoring); the two detection services and, since the paper made refactoring unsupervised, the **Refactoring Agent** consume no monitoring capacity at all. A team can carry any number of role-4 agents without ever failing the feasibility gate.
- **Supervised roles are additionally capped by throughput saturation** (§3.11.1): `search.saturation_bounds(params, sigma)` returns `ā_k = ⌈σ·λ_k/η_k⌉` per supervised role, where `λ_k` is the steady-state work that reaches the role (all of it derived from `r_arrival`) and `η_k` the throughput one agent sustains, with the debt amplifier at its floor `delta_min`. `σ` (`SearchConfiguration.sigma`, default and §4.2 value 2) is the headroom for Poisson arrival peaks. Beyond `ā_k` an agent simply has no work, so the search does not spend budget there. Eq. (sat_bound) ranges over `K_sup` only, so **there is no key 3 in the returned dict** — see the refactoring bullet for what bounds role 4 instead.
- **The three autonomous roles are bounded outside `saturation_bounds`, and differently from each other.** `review_sample_max(params) = ⌊r_arrival⌋` is a *sampling range* for k=5,6 only: nothing caps those two, and `_mutate` may take them past it (this is what makes exhaustive search impossible, §3.11). `refactoring_bound(params) = ⌊r_arrival/2⌋` is `ā_4`, a genuine cap honoured by the sampler **and** by the mutation operator (§3.11.2 step 6, §3.11.4). It is *not* scaled by `σ`, and at the four project tiers it is 1 / 6 / 37 / 500.
- **Exhaustive search over compositions is infeasible**: `src/search.py` draws each supervised role independently and uniformly in `{0, …, ā_k}` (`{1, …, ā_1}` for the dev implementer) and/or mutates both vectors. Stars-and-bars composition sampling is gone. Every draw is structurally valid by construction, so `_random_draws` returns `TeamConfiguration`s directly — there is no discarded-draw path in the sampler.
- **Feasibility pre-filter** runs before any simulation. One rejection path, recorded and skipped: the human-capacity condition of §3.6 Eq. (feas_monitoring) (the manifest entry carries `feasible: false` and a `reason`). It still **counts as one evaluation** (§3.11.2 step 7, §4.3), so `n_evaluated` is the budget spend and `n_simulated` the realised simulation count; both are written to `manifest.json`. Rejected candidates are reported, never silently redrawn — the paper says so explicitly, and a rejection loop would condition every draw on the staffing level drawn with it. `_mutate` likewise tests only the *structural* constraints, so a monitoring-infeasible child is evaluated and rejected rather than retried.
- **Scenario parameters are fixed constants**, not sampled: every value in a scenario JSON must be a plain number.
- **Five orthogonal scenario groups** (paper §4.1), each with a `low`/`medium`/`high` tier grounded in literature: `human`, `project`, `debt`, `agent`, `bugimpact` (`SCENARIO_GROUPS` in `src/config.py`). The groups partition the 37 `ScenarioParameters` fields exactly, so `load_scenario_params` takes a `{group: path}` mapping with one file per group and rejects a field defined in two files. Tiers combine freely, but **the grid is not a cube**: two groups carry a fourth tier, each a deliberate stress case rather than an observed operating point — `project`'s `veryhigh` (`r_arrival = 1000` issues/week, unrealistic extreme velocity; the high tier at 75 is already managed by two FTE) and `agent`'s `pessimistic` (the low tier's capability at 10× the high tier's `c_token`; both halves of the agent grounding, the SWE-bench scores and the token prices, are optimistic). That gives 3^3 · 4 · 4 = 432 scenarios. Read the tiers through `config.scenario_tiers(group)` / `config.scenario_tier_combinations()` / `config.scenario_id(tiers)`, never `SCENARIO_TIERS` (the three base tiers) — anything iterating `SCENARIO_TIERS` per group silently drops the `p-v` and `a-p` scenarios. `ALL_SCENARIO_TIERS` is the plotting/ordinal order and `TIER_INITIALS` the `l`/`m`/`h`/`v`/`p` codes the scenario ids and the LaTeX scenario codes share; a tier initial may coincide with a *group* initial (`a-p`), since the id token format is `<group>-<tier>`. Note tier direction is per-group and not always "high = best": e.g. `debt`'s `delta_min` and `Gamma_doc` both *decrease* as the tier rises (severity up, relief down), while the `agent` group's `r_bugs` falls and its `d_test` rises with price — the paper's footnote resolves the leaderboard's Opus/Fable near-tie in favour of the high tier, so both rows are monotone. `tests/test_io.py` pins all three directions; they have been swapped in the config once already. `pessimistic` is the only tier that changes cost without changing behaviour — `c_token` enters `simulation.cost_components` alone, so it can neither make a candidate infeasible nor move a saturation bound.
- **The human-only baseline is gated structurally, on the team having no agents at all** (paper §4.1). `config.is_human_only(cfg)` is `not any(cfg.A_dev)` — `TeamConfiguration` forces `A_qa` to zero whenever `A_dev` is zero — and it is the single switch for both baseline changes in `step_sprint`: the `f_qa` capacity reservation of Eq. (hafter_qa), and the human review stage of Eqs. (cov_qa)/(qa_human) that **replaces** Step 5's Test → CR → SA pipeline. There is no `_solo` parameter file: `h_qa` / `d_qa` sit in the ordinary `config/human_*.json` (`f_qa` is a control variable), and `SearchConfiguration.human_only` only chooses the *enumerator* and is simply never read by an agent-supported team. Consequence worth keeping: an agent-supported run is bit-identical to before the baseline landed — the two paths are mutually exclusive, so no stray RNG draw enters. `tests/test_sprint.py` pins that in both directions.
- **`A_dev` may be the all-zero vector, and only the all-zero vector.** `TeamConfiguration.__post_init__` accepts either an all-zero `A_dev` (with an all-zero `A_qa`) or one with `a_dev,1 ≥ 1`; `(0,1,0,…)` still raises. The samplers and `_mutate_composition` all keep their `a_1 ≥ 1` floor, so **the search can never reach a zero-agent team** — `_run_human_only` is its sole producer.
- **The baseline is provably independent of the `debt` and `agent` groups**, which is why its grid is human(3) × project(4) × bugimpact(3) = **36 scenarios** and not 432. Every `delta_D` consumer multiplies an agent count or `W_agent`; every agent-group parameter multiplies an agent count, or dies in `Φ(·,0) = 0`, or acts on a test suite that never grows. §4.1's "simplified model without debt dynamics" therefore needs **no bypass code** — it is already true, and `tests/test_sprint.py::TestHumanOnlyDebtIsInert` pins it rather than the transition enforcing it. The two inert groups take the `UNUSED_TIER` sentinel, rendered `X`, so a baseline id reads `h-m_p-v_d-X_a-X_b-l`; `config/{debt,agent}_unused.json` supply the placeholder values so the `<group>_<tier>.json` convention holds and `campaign.scenario_params_for_id` needs no special case. `UNUSED_TIER` is deliberately **outside** `TIER_INITIALS` and `ALL_SCENARIO_TIERS` — it must never enter `scenario_tier_combinations()` nor the figure tier channels.
- **`r_bugs_human` is an absolute per-feature rate, not a scale factor** (paper Eq. Binj-human): `B_human(t) ~ Poisson(impl_human'(t) · r_bugs,human)`, with **no** `(2 − δ_D)` amplifier. Human-written code injects at a fixed rate; only the agent stream of Eq. (Binj-agent) is debt-amplified. Values are 0.20 / 0.12 / 0.08 by human tier. It is part of the **shared** model, so it moves mixed-team results too (roughly ×4 on the escaped-bug cost at the medium agent tier). `saturation_bounds`' `λ_3` deliberately still counts only the agent stream, matching the paper's Eq. (workload); it under-counts, but `ā_3 = 1` regardless because `η_3 = r_fix·δ_min` is in the hundreds.
- **The baseline is drawn as an overlay, never as its own figure set.** `visualize_campaign` splits the `human_only` rows out *before* the per-mode loop and passes them to `_draw_matrix` as a `baseline` frame. It carries the **shape and size channels** — `project` and `bugimpact` on the default layout are two of the three groups the baseline varies, so its points are read against the search optima on exactly those scales (`_baseline_style_groups` maps the tiers, `_baseline_marker_kwargs` draws them). **Colour stays fixed at black on every layout**: with shape and size shared it is the only channel left that says "no agents", and on the default layout `debt` sits there and the baseline has no debt tier to encode. A channel whose group the baseline holds at `UNUSED_TIER` has no tier and falls back to the fixed `*` at `_BASELINE_SIZE`. The baseline therefore appears as a fourth entry in the *colour* legend, not as a legend block of its own. Do **not** promote it to a fifth tier shape — it reuses the four, and the note that a fifth shape would not survive the smallest size tier still holds. `_baseline_cell` matches a baseline row on the *facet* groups only and treats `UNUSED_TIER` as matching every tier, which is what repeats the same reference points down all four `agent` columns; say so in the caption, it is a reference level rather than an `a-X` observation. The ratio figure takes no baseline at all — a team with no agents has no agent/developer ratio. Axis floors and shared ranges are computed over `_with_baseline(df, baseline)`, or the baseline's exactly-zero agent cost would drop silently off a log axis.
- **Reproducibility**: every random draw is derived from a master seed via `SeedSequence(s0, spawn_key=(FNV1a_hash(config), k))`. Because the spawn key holds the replication index and nothing about the count, a replication is addressable by index, which is what lets `run_configuration`'s `k_offset` hand the re-simulation of §3.10 a **disjoint** block `[K_search, K_search + K_final)` deterministically.
- **The convergence record of §5.1 is collected in the generation loop, not reconstructed afterwards.** `SearchOutcome` carries `elite_turnover`, `incumbent_changed` and `best_cost`, one entry per generation actually run. The loop keeps the ranking across its body — `rank` is called once before the first generation, for `E_0` over the initial pool, and again at the end of each generation — so the same sort supplies both the parents of the next generation (`P_{g-1}`, as §3.11.4 requires) and `E_g` for `τ_g = |E_g \ E_{g-1}| / n_elite`. The denominator is `n_elite` as Eq. (elite_turnover) writes it, not `|E_g|`. `_run_human_only` leaves all three lists empty; an enumeration has no generations, and `incumbent_turnover` is None there rather than 0.
- **Campaign sample retention**: at the §4.2 budget, one `result.json` per candidate would be ~4.9 M files / ~58 GB, which is why `SAMPLE_POLICIES` is only `("best", "none")` — the retain-everything policy was removed rather than left as a foot-gun — and why `--samples` defaults to `best` — the `ConfigResult`s are buffered in memory and only `c*` and `I*` are written, *after* `two_stage_selection` has run. The files written for those keys are the **re-simulated** results (`K_final` replications), because `_cmd_optimize` merges `SelectionReport.final_results` into `outcome.results` and passes `final_aggs` as `write_samples`' `aggs_override` — a sample file must agree with the row `optima.csv` reports. Aggregate means and CIs for every simulated candidate always land in `search_summary.json`, at `K_search`.
- **The selection rule of §3.10 replaces the CI-aware Pareto frontier, and it is a *total preorder*.** `selection.preference_key` is Deb's feasibility rule as the lexicographic key `(violation, cost_mean)`: two violators are separated by whose shortfall is smaller, an admissible configuration precedes every violator, and two admissible ones are separated by cost alone. So `rank` is a plain stable sort, `best_index` is a `min`, and **none of the non-transitivity machinery the old CI-aware multi-objective rule needed applies** — there is no O(n²) unpruned dominance loop and no ordering hazard. Do not reintroduce a dominance relation: with one objective there is nothing for it to do.
- **Selection is two-stage, and `two_stage_selection` owns both stages (§3.10).** The search scores every candidate with `K_search = K'` replications; stage 1 is `report_optimum(aggs, floor, keys=…)` **with** parsimony keys, which gives the *preliminary* `c*` and `I*` already reduced by the tie-break (it is called a second time without keys purely to record `n_prelim`, the size before the reduction). Stage 2 re-simulates every member of that reduced set with `K_final = K > K'` replications and repeats the rule on the narrower intervals: a member that no longer satisfies the throughput constraint drops out (`report_optimum` filters on `satisfies_throughput`, so this needs no extra code), `c*` is re-determined and may move, and the parsimony tie-break runs a second time on the better estimates. The re-simulation is **outside** the `E` budget (§4.2) and runs a **disjoint** block of replication indices (`run_configuration(..., K=K_final, k_offset=K_search)`), *not* an extension of the block the search used. That is deliberate and must not be "simplified" away: a candidate is in the preliminary set because its `K'` estimate came out at or near the minimum of thousands of noisy estimates, so that estimate is biased low by the selection itself — reusing those replications would carry a `K'/K` share of the winner's curse into the reported cost, and a fresh block removes it at identical simulation cost. **Applying parsimony at stage 1 is deliberate and is what the cost condition bought**: a member it removes there is both structurally larger and dearer in expectation at `K'`, so it can be dropped before it is re-simulated — every removal is one fewer `K`-replication run. The price, accepted knowingly: the final `c*` can differ from what a stage-2-only tie-break would report if a removed member's noisy `K'` estimate would have come out lowest at `K`. It is statistically tied with its remover and structurally larger, so nothing measurable is lost. Inadmissible searches are the one case with nothing to narrow: the preliminary `I*` is empty, so the single least-violating candidate is re-simulated alone, which is what keeps `optima.csv`'s `value_ratio` a `K` number for every row.
- **`report_optimum`, not `best_index`, is what a reported `(c*, I*)` pair must come from.** `I*` (Eq. topset) is the admissible candidates whose cost CI overlaps `c*`'s, reduced by the parsimony tie-break. **`c*` cannot be removed by that tie-break** and needs no exemption: domination requires a strictly lower mean cost and `c*` is the minimiser, so `c*` stays the raw cost minimiser and is a member of every non-empty `I*`. The other reason to go through `report_optimum` is the inadmissible case: when *no* candidate reaches the throughput floor there is no `c*` in the sense of §3.10, so **`I*` is empty** while the least-violating candidate is still returned and reported (`manifest["best_admissible"] = false`, and `optima.csv`'s `n_indistinguishable = 0`). `two_stage_selection` calls it twice (see the bullet above), and `main._cmd_optimize` falls back to `c*` for the `--samples best` file list so an inadmissible search still writes one sample; the *search loop* deliberately does not use it — parsimony must not steer the search, only the report.
- **The parsimony tie-break (§3.10) applies only inside `I*`, and it is cost-aware**: among configurations that share `(N, R, F_freeze, f_fix, f_qa)`, one removes another when it uses no more agents of every role in both phases, strictly fewer of at least one, **and** its `cost_mean` is strictly lower. `selection.parsimony_dominates` therefore takes the two `ParsimonyKey`s *and* the two mean costs; only the CI-overlap half of the rule stays the caller's precondition. The cost condition is what makes `c*` untouchable and what makes the tie-break safe to apply at `K'` — do not weaken it to `≤`, which would let an exactly-tied smaller team remove `c*` and bring the exemption back. A strict partial order, so `I*` is never emptied and the dominance filter is order-independent. It cannot substitute for the `A_qa` canonicalisation below: componentwise `≤` is a partial order and only collapses chains, so two teams like `(…,3,2,…)` and `(…,2,3,…)` both survive.
- **Both composition vectors are required, and the field order is the paper's.**
  `TeamConfiguration` is `(N, A_dev, A_qa, R, F_freeze, f_fix, f_qa)`, matching §3.3's tuple
  plus the baseline-only `f_qa`; `F_freeze`, `f_fix` (0.5) and `f_qa` (0.0) carry defaults.  `A_qa` used to default to the empty tuple
  meaning "same as `A_dev`" — a shorthand the model has no notion of, reachable only
  through the deleted `io.load_team_config`.  Passing it explicitly is now mandatory; the
  tests spell out an all-zero `_NO_QA` where they run `F_freeze = 0`, which the
  canonicalisation below makes exact anyway.
- **`A_qa` is canonically zero where `F_freeze == 0`**: with no freeze window `T_qa` is zero and `A_qa` is never applied, so `TeamConfiguration.__post_init__` normalises it to the zero vector (after the shape checks, so a malformed vector still raises). This is what keeps a redundant coordinate from multiplying the search space: `rng.config_hash` folds `A_qa` in, so without it two configurations that are the *same model* draw different RNG streams and land at different objective values while being indistinguishable as teams. The sampler and the mutation operator cooperate — `_random_draws` draws `F_freeze` before the compositions and skips the QA draw, and `_mutate` drops the `A_qa` option from its four-way choice at `F_freeze == 0` (every QA move would canonicalise straight back to the parent). Measured on a smoke search before the fix: 23.9 % of evaluated candidates were behavioural duplicates.
- **Agents are billed for the work they performed, not for being deployed — except for three roles that have no countable output**: `step_sprint` returns a `SprintActions` with the work each role completed that sprint, measured in that role's own throughput unit (features for k=1, where an implemented architecture proposal counts as `ARCH_IMPL_COST_UNITS = 5` features; change requests for k=2, bugs for k=3, tests for k=7, documents for k=8). **k=4, 5, 6 are flat-rate** (`r_k = 1`, `u_k = a_k`): the two review services have no discrete unit of output, and the Refactoring Agent produces a reduction of the normalised `D_tech` rather than an artefact. `run_configuration` accumulates the vector over the horizon and `simulation.cost_components(config, params, T, actions)` returns `C_agents = Σ_k actions_k · c_token,k`. A demand-driven agent that no work reaches costs nothing; a flat-rate one bills whether or not it has anything to do. Consequence to keep in mind: **cost does not discipline over-provisioning of roles 1–3** — the saturation bound `ā_k` and the human-capacity gate are what bound them.
- **The Refactoring Agent is unsupervised and effectively unbounded.** Three things about role 4 together: (a) it is in `K_aut`, so it consumes no monitoring capacity and cannot make a candidate infeasible; (b) Eq. (sat_bound) no longer covers it, so its only cap is the search-side `refactoring_bound = ⌊r_arrival/2⌋`; (c) Eq. (debt_update) removes `r_refactor·a_·,4` from `D_tech` with **nothing throttling it but the clip at 0** — no queue, no `δ_D` slowdown, and it removes debt even in a sprint that created none. `r_refactor` is denominated in `D_tech` units and calibrated against `δ_impl`/`δ_fix` (§4.2.4: one agent removes the debt one feature or one bug fix adds), so it takes the same **0.02 / 0.06 / 0.13** by *agent* tier — note the tier axis is the agent group's while `δ_impl`/`δ_fix` follow the *debt* group's, and the two are orthogonal. The role is billed flat at `c_token,4` = one feature implementation (2.28/3.80/7.60/76.00 by tier).
- **An implemented architecture proposal costs five features of *capacity*, not just of budget.** `ARCH_IMPL_COST_UNITS = 5.0` enters Eq. (arch_impl) as a divisor (`arch_impl ≤ ⌊r_F·δ_D·a_1/5⌋`) and Eq. (impl_agents) as a multiplier on the residual (`impl_agents ≤ ⌊r_F·δ_D·a_1⌋ − 5·arch_impl`), *as well as* weighting `units[0]`. It is the same code-base-wide analysis that makes producing a proposal five times as expensive as a feature (`c_token,2 = 5·c_token,1` in every tier). A configuration with `a_1 = 1` and the fixture's `r_F = 3` can therefore implement **no** proposals at all — several tests need `a_1` raised for the arch pipeline to move.
- **The test and documentation agents are capped at the coverage they can still buy**: `step_sprint` writes `min(r_test·a_7, c_test_ceil·F_completed − N_test)` tests and `min(r_doc·a_8, F_completed − N_doc)` documents, and bills exactly that increment. The cap sits precisely where the outer `min` of Eq. (coverage)/(doc_cov) already clipped, so `cov_test` and `doc_cov` are **unchanged** wherever the role is staffed in both phases — only the bill moves. `saturation_bounds` is untouched: `u_7 = u_8 = 1` was already the bound, and the transition now realises it — a second test agent writes nothing, bills nothing, and still consumes monitoring.
- **Who did the work is recorded separately from the bill**: the per-role `actions` vector is agent-only by construction, so it cannot answer "who implemented what". `u_1` conflates agent features with implemented proposals, and the human share appears in no entry at all (human cost is the headcount term of Eq. (cost_human), independent of what the humans produced). `SprintActions` therefore carries four extra scalars beside `units` — `impl_human`, `impl_agents`, `fixes_human`, `fixes_agent`, the paper's impl_human'(t) / impl_agents(t) / Σ n^H_i(t) / fixes_agent(t) of §3.6 — which `run_configuration` sums over the horizon into `ReplicationResult`, `selection` averages into `AggregatedObjectives` (means only, **no bootstrap CIs**), and `main._cmd_optimize` writes to `search_summary.json`. Three things to keep in mind: **(a)** `units` must never be lengthened to hold them — its length is hard-checked in `SprintActions.__post_init__` and `simulation.cost_components`; **(b)** `fixes_agent` deliberately duplicates `actions_3`, and `impl_agents` un-conflates `actions_1` (`arch_impl = (actions_1 - impl_agents) / ARCH_IMPL_COST_UNITS`), so no separate `arch_impl` field is needed; **(c)** the invariant worth testing against is `impl_human + impl_agents == F_completed(T) == value·T`, exact by Eq. (impl).
- **Everything except `cost` and `value` is reporting only.** `AggregatedObjectives` bootstraps exactly two quantities — the objective and the constrained one — and carries the three cost drivers, the per-role `actions_mean` and the four contribution counts as plain means. `preference_key` and `I*` membership must never read them; `tests/test_selection.py` pins that.
- **22-dimensional state vector**: features (3, incl. `F_cycle`, the features implemented since the last release), bugs by type×stage (12), technical debt and human capacity queues (5), test infrastructure and documentation (2).

---

## Search (paper §3.11)

There is exactly **one** search (`search.run_search`) and it is not configurable — no
strategy name anywhere.  Uniform random sampling is not an alternative to it; it is the
initialiser (`_random_draws`) the search seeds from (§3.11.2).

The §4.1 human-only baseline is a **result mode** beside the search, not a second search:
its configuration space is 10,890 points (99 `(N, R, F_freeze)` × 11 `f_fix` × 10 `f_qa`), so `search.run_human_only` enumerates it outright and
`budget` is never read.  `SearchConfiguration.human_only` selects it.  `RESULT_MODES = ("search", "human_only")` is what `campaign.search_dirs`, the
result-tree directory names, the `mode` column of the campaign CSVs and
`manifest["mode"]` all range over.

- **The search** (§3.11.4) — `n_samples_initial` random draws, then `n_generations` × `n_offspring` mutations. Parents come 80 % from the **elite** (`rank`'s first `n_elite` entries) and 20 % from the remainder; the `seen` set means no configuration is ever charged twice, and the loop stops once `budget` evaluations are spent. §4.2: `budget = 20,000`, 2,000 / 180 / 100 / `n_elite = 100`.

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
dropped the second search — do not reintroduce them.

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
