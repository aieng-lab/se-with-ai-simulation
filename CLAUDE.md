# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Simulation and cost optimisation of mixed AI-agent / human-developer teams. A team is a
discrete-time stochastic system; an evolutionary search over Monte Carlo estimates finds the
team that delivers the project's features at the lowest **cost per feature** (salaries +
agent token bills + the dollar cost of escaped bugs). Paper target: EMSE (Springer, svjour3).

- **`paper/main.tex` is authoritative** (§3 is the formal model). The code is a rendering of it.
- **Status:** the §4.2 campaign (E = 20,000, finished 2026-09-25) is complete and the paper
  is drafted in full. Results archive: Zenodo DOI 10.5281/zenodo.23358924 (cited in the
  data/code availability statement and `README.md`). Code:
  `https://github.com/aieng-lab/se-with-ai-simulation`, Apache-2.0.
- `README.md` is the user-facing reproduction guide; keep its commands in sync with the CLI.

## Layout

```
paper/        main.tex, literature.bib (URLs live in the bib, not inline),
              figures/ (the six §5 PNGs, copies of what the code emits, plus the
              hand-written TikZ overview.tex)
src/          config, parameters (37 fields), state (22-dim), rng, io, feasibility (§3.6),
              sprint (the transition), simulation (replications + cost accounting),
              selection (§3.10), results, search (§3.11 + human-only enumeration),
              campaign + constellation (consolidation, §5 tables), visualization, main (CLI)
config/       scenario tiers per group, horizon_default.json (T=104, K'=100, K=1000, seed 42),
              search_default.json (template), search/ (432), search_baseline/ (36)
scripts/      generate_search_configs.py, check_bib_dblp.py
mining-data/  repository-mining data grounding the project velocity tiers
run_experiments.sh   parallel, resumable campaign driver
results/      experiments/ (468 runs) and analysis/ (CSVs + figures); not committed
```

## Commands

```bash
.venv/bin/python -m pytest tests/ -q                     # 574 tests, ~2 min
python -m src.main optimize <search_config.json> --out <dir>
JOBS=14 ./run_experiments.sh                              # 432 searches (FORCE=1 to rerun)
CONFIG_DIR=config/search_baseline MODE=human_only ./run_experiments.sh   # 36 baselines
python -m src.main campaign [--figures-only | --no-figures]
python -m src.main constellation [--figures-only]         # run campaign first
```

The CLI has exactly three subcommands (`optimize`, `campaign`, `constellation`). The
human-only enumerator is chosen by the search config's `"human_only": true`, not a flag;
`MODE=human_only` only names the result directory.

**After any change to analysis or plotting code, check the figures are byte-identical:**
```bash
cp -r results/analysis /tmp/figcheck && rm -f /tmp/figcheck/*.png
.venv/bin/python -m src.main campaign --figures-only --out /tmp/figcheck
.venv/bin/python -m src.main constellation --figures-only --out /tmp/figcheck
for f in paper/figures/*.png; do cmp "$f" "/tmp/figcheck/$(basename $f)"; done
```

Two tests the paper is held to:
- `tests/test_io.py::TestScenarioFilesMatchThePaper` — every value of §4.2's tier tables
  against `config/*.json` (rows have been silently swapped there before).
- `tests/test_simulation.py::TestAgainstThePersistedCampaign` — re-simulates four `c*` and
  compares with `results/analysis/optima.csv` bit for bit (skips without `results/`).

## §5 deliverable

Exactly **six figures and three tables**: `elite_turnover`, `human_agent_matrix`,
`quality_development_matrix`, `team_drivers_violin`, `composition_agent_bill`,
`composition_team_bill`; tables from `convergence.csv` (I* stage sizes),
`multipliers.csv` (cost multipliers) and `baseline_uplift.csv` (baseline ratios, read by
agent tier over pairs where both sides are admissible). `paper/figures/` holds copies under
exactly the emitted names. `agent_human_ratio_matrix` is still drawn but unused.

Figure layouts are fixed (`visualization.MATRIX_CHANNELS`, `COMPOSITION_{BLOCK,BAR,PANEL}_GROUP`,
`constellation.CELL_GROUPS`) — there are no re-cut flags. Points to keep:
- Matrix figures: `agent` × `human` facets, colour = `debt`, shape = `project`, size =
  `bugimpact`. `project` must stay on shape; `veryhigh` and `pessimistic` share the fourth
  style slot because no group holds both; no fifth marker shape. The human-only baseline is
  an overlay (black, shape/size channels), never its own figure set.
- Composition figures and `team_drivers` read `optima.csv`, **not** `constellation_cells.csv`:
  shares are formed within a scenario before averaging. Both composition weightings are a
  **bill**, never a head count; escaped-bug cost is annotated, never a slice.
  `constellation.role_bill` must be applied by the caller (`main._cmd_constellation`).
- `elite_turnover` carries no tier channels and no baseline.

Analysis facts that are easy to get wrong:
- Every number in `optima.csv` is a **K_final** estimate (`search_summary.json`'s
  `final_objectives`).
- **`I*` is `search_summary.json`'s `indistinguishable` list**, not `final_objectives`
  (that is the preliminary set before the stage-2 filter and tie-break).
- Admissible search optima: `campaign.admissible_optima` (`n_indistinguishable == 0` marks
  an inadmissible search). Multipliers and `process_summary` are fitted over the 432 of them.
- `i_star` statistics are scenario-balanced (one mean vector per scenario first).

Deleted on purpose — **do not resurrect**: the three-objective Pareto analysis, any
per-sprint trace path, the `run`/`analyze`/`visualize` subcommands, the figure re-cut flags
and their validation helpers, the constellation profile/heatmap figures, the head-count
composition figure, the cost-share figure, the greedy neighbourhood search, and dead
parameters whose only value was the default. Keep code changes minimal.

## Model invariants

Objective: `cost = (C_quality + C_human + C_agents) / max(1, F_completed(T))`, formed per
replication then averaged; drivers sum to it exactly. `C_quality` uses absolute dollar costs
per escaped bug (ω). Constraint: `value = F_completed/T ≥ (1-ε)·r_arrival`, ε = 0.02.
Everything else (drivers, per-role actions, contribution counts) is reporting only and must
never enter `preference_key` or `I*` membership.

- **Controls** `(N, A_dev, A_qa, R, F_freeze, f_fix)` + `f_qa` (baseline only). `N` is FTE,
  may be fractional; communication overhead uses headcount `⌈N⌉`. No span-of-control `M`.
- **Canonicalisation in `TeamConfiguration`:** `A_qa` = 0 when `F_freeze == 0`; `f_qa` = 0
  when `A_dev` is non-zero; `A_dev` is all-zero (human-only) or has `a_dev,1 ≥ 1`;
  `a_qa,1 = a_qa,2 = 0`. Both vectors must be passed explicitly.
- **Roles:** supervised `K_sup = {1,2,3,7,8}` (indices 0,1,2,6,7) enter the monitoring
  feasibility of §3.6; autonomous `K_aut = {4,5,6}` do not. Roles 4–6 bill flat
  (`u_k = a_k`); others bill per unit of completed work (architecture implementation = 5
  feature units, also as a capacity divisor). Test/doc agents are capped at the coverage
  they can still buy.
- **Bounds:** `saturation_bounds` covers `K_sup` only (`⌈σ·λ_k/η_k⌉`, σ = 2);
  `refactoring_bound = ⌊r_arrival/2⌋` caps role 4; `review_sample_max = ⌊r_arrival⌋` is only
  a sampling range for roles 5, 6.
- **`r_bugs_human`** is an absolute per-feature rate with no `(2 − δ_D)` amplifier.
- **Human-only baseline** (`config.is_human_only`) replaces the agent detection pipeline
  with the human QA stage; it is provably independent of `debt` and `agent` (36 scenarios,
  `UNUSED_TIER` rendered `X`, outside `ALL_SCENARIO_TIERS`). Agent-supported runs are
  bit-identical with or without that code path.
- **Scenarios:** five groups, tiers combine freely; `project` has `veryhigh`, `agent` has
  `pessimistic` → 432. Iterate via `config.scenario_tiers` / `scenario_tier_combinations`,
  never `SCENARIO_TIERS`. Tier direction is per group (see memory).
- **RNG:** streams keyed by `(FNV1a(config), replication index)`, so the §3.10
  re-simulation uses the disjoint block `k_offset = K_search` (deliberate: removes the
  winner's curse).
- **Selection (§3.10):** `preference_key = (violation, cost_mean)`, a total preorder — no
  dominance relation. `report_optimum` gives `(c*, I*)`; `two_stage_selection` applies it at
  K' with parsimony, re-simulates at K, applies it again. Parsimony is cost-aware (strictly
  lower mean cost required), so `c*` can never be removed. Inadmissible search → empty `I*`,
  least-violating candidate still reported.
- **Search (§3.11):** one search, not configurable: 2,000 initial draws, then generations of
  100 offspring, parents 80 % from the elite (100); no candidate charged twice; infeasible
  candidates count as evaluations and are never redrawn. `_mutate` changes one of
  `{N, F_freeze, A_dev, A_qa, f_fix}` (never `R`; `f_fix` to an adjacent value);
  autonomous-role service steps are `±2^j`.
- **Samples:** `--samples best` writes only `c*` and `I*`, with the re-simulated (K) results.
