# What Should a Software Team Spend on Agents?

This repository contains the simulation and the analysis code for the paper *What Should a Software Team Spend on Agents? Simulation-Based Cost Optimization of Mixed AI-Human Teams*.

We model mixed teams of human developers and AI agents in eight roles as a discrete-time stochastic process that covers bug injection and detection, technical debt, human supervision of agents, and release cycles with feature freezes. Using Monte Carlo simulation and evolutionary search, we determine the team that delivers the features a project demands at the lowest total cost per feature, i.e., the developers' salaries, the agents' token costs, and the costs of the bugs that escape into production. We do this for 432 scenarios that vary developer capability, project velocity, technical debt dynamics, agent capability and price, and bug costs, and compare the results with a human-only baseline.


## Repository structure

| Path | Content |
|---|---|
| `src/` | The simulation package: the model (`sprint.py`, `simulation.py`), the search and selection (`search.py`, `selection.py`), the analysis of the results (`campaign.py`, `constellation.py`, `visualization.py`), and the command line interface (`main.py`). |
| `config/` | The scenario parameters (one file per tier of each of the five scenario groups), the Monte Carlo protocol (`horizon_default.json`), and the search configurations of all 432 scenarios (`search/`) and the 36 human-only baselines (`search_baseline/`). |
| `scripts/` | `generate_search_configs.py` creates the search configurations in `config/search/` and `config/search_baseline/`. |
| `run_experiments.sh` | Runs all searches of the experiments in parallel. |
| `mining-data/` | The repository mining data and script we used to ground the project velocity tiers. |
| `tests/` | The test suite. |
| `paper/` | The manuscript and the figures. |

## Reproducing the results

In the following we describe the steps to reproduce our results. Everything we describe works as is on Ubuntu 24.04 LTS with Python 3.10 or newer.

### Setting up the environment

First, checkout the repository from GitHub:

```bash
git clone git@github.com:aieng-lab/se-with-ai-simulation.git
cd se-with-ai-simulation
```

We recommend using a virtual environment. To create a virtual environment, run the following command:

```bash
python3 -m venv .venv
```

We can now activate the virtual environment and install the package with all dependencies for the analysis and the tests:

```bash
source .venv/bin/activate
pip install -e ".[analysis,dev]"
```

To check that everything works, you can run the tests (about two minutes):

```bash
python -m pytest tests/ -q
```

### Running a single search

To get a feeling for the simulation, you can run the search for a single scenario, e.g., all scenario groups at the medium tier:

```bash
python -m src.main optimize config/search/search_h-m_p-m_d-m_a-m_b-m.json --out results/single
```

The scenario is encoded in the file name: `h` (human developers), `p` (project dynamics), `d` (technical debt), `a` (agents) and `b` (bug impact), each with the tier `l` (low), `m` (medium), `h` (high), `v` (very high, project only) or `p` (pessimistic, agents only). A search evaluates 20,000 team configurations with 100 replications each and afterwards re-simulates the best configurations with 1,000 replications. This takes about one core-hour. The result directory contains the `manifest.json` with all evaluated configurations, `search_summary.json` with their estimated costs, `convergence.json` with the convergence of the search, and `samples/` with the detailed results for the cost-optimal team and all teams that are statistically indistinguishable from it.

### Running the experiments

The search configurations for all 432 scenarios are part of the repository. They can be re-created from the template `config/search_default.json` as follows:

```bash
python scripts/generate_search_configs.py --force             # 432 scenarios
python scripts/generate_search_configs.py --baseline --force  # 36 human-only baselines
```

All searches are run with `run_experiments.sh`, which executes them in parallel and can be resumed if it is interrupted. `JOBS` sets the number of parallel searches:

```bash
JOBS=14 ./run_experiments.sh                                                  # mixed teams
CONFIG_DIR=config/search_baseline MODE=human_only JOBS=14 ./run_experiments.sh  # human-only baseline
```

Our run took about 520 core-hours. You can test the setup with `LIMIT=2 ./run_experiments.sh` (only the first two scenarios) or `DRY_RUN=1 ./run_experiments.sh` (only list the scenarios). The results are written to `results/experiments/<scenario>/<search|human_only>/`.

### Our results

The raw results of our experiments and the consolidated results from which all tables and figures of the paper are derived are available on Zenodo: **[DOI will be added]**. To use them instead of running the experiments yourself, extract the archive in the root of the repository, which creates the `results/` folder.

### Analysis of the results

The analysis is done in two steps. First, the results of all searches are consolidated:

```bash
python -m src.main campaign
```

This reads `results/experiments/` and writes the following to `results/analysis/`:

- `optima.csv`: the cost-optimal team of each scenario, including the cost per feature and its split into the costs for developers, agents, and escaped bugs;
- `convergence.csv`, `elite_turnover.csv`, and `convergence_summary.txt`: the convergence of the searches (Section 5.1);
- `multipliers.csv` and `multipliers_summary.txt`: the cost multipliers of the scenario groups (Section 5.2);
- `baseline_uplift.csv`: the comparison with the human-only baseline (Section 5.4);
- `scenarios.csv`: the search effort per scenario;
- the figures `elite_turnover.png`, `human_agent_matrix.png`, and `quality_development_matrix.png`.

Second, the composition of the cost-optimal teams is analyzed:

```bash
python -m src.main constellation
```

This writes `indistinguishable.csv` (all teams that are statistically indistinguishable from the optimum), `constellation_cells.csv` and `process_summary.csv` (the shares reported in Section 5.3), and the figures `team_drivers_violin.png`, `composition_agent_bill.png`, and `composition_team_bill.png`.

Both commands accept `--figures-only` to only redraw the figures from existing CSV files, and `campaign` accepts `--no-figures` to only write the CSV files.

### Grounding of the project velocity

The data for the project velocity tiers were collected with SEART's GitHub Search tool and are available in `mining-data/results.csv.gz`. The statistics are computed with:

```bash
cd mining-data
python project_dynamics_stats.py --top-stars 1000 --output-dir stats_top1000
```

The bug severity data we used is available in `mining-data/Data_Ohira_Bug_Distribution.xlsx`.

## License

The code is licensed under the Apache License 2.0, see `LICENSE`.
