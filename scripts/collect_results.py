#!/usr/bin/env python3
"""Consolidate the 486 searches of paper §4.2 into two flat tables.

Reads every ``<results-dir>/<scenario_id>/<strategy>/{manifest,search_summary}.json``
written by ``python -m src.main optimize`` and emits:

    frontiers.csv   one row per Pareto-optimal configuration, with its scenario tiers,
                    search strategy, the five control variables, both 8-element
                    composition vectors, and the three objectives with 95 % CIs
    scenarios.csv   one row per (scenario, strategy): realised evaluation counts,
                    frontier size, and wall-clock time — the search-effort figures
                    §4.2 promises to report

The table schemas live in ``src/results_analysis.py``, which
``scripts/analyze_results.py`` builds the §5 analysis on top of.

Usage
-----
    python scripts/collect_results.py
    python scripts/collect_results.py --results-dir results/experiments --out-dir /tmp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.results_analysis import build_tables, search_dirs   # noqa: E402

DEFAULT_RESULTS_DIR = "results/experiments"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR, metavar="DIR",
                        help=f"Campaign output root (default: {DEFAULT_RESULTS_DIR})")
    parser.add_argument("--out-dir", default=None, metavar="DIR",
                        help="Where to write the CSVs (default: the results dir)")
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        parser.error(f"{results_dir} is not a directory")
    out_dir = Path(args.out_dir) if args.out_dir else results_dir

    searches = search_dirs(results_dir)
    if not searches:
        print(f"error: no finished searches found under {results_dir}", file=sys.stderr)
        return 1

    frontiers, scenarios = build_tables(results_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    frontiers_path = out_dir / "frontiers.csv"
    scenarios_path = out_dir / "scenarios.csv"
    frontiers.to_csv(frontiers_path, index=False)
    scenarios.to_csv(scenarios_path, index=False)

    n_scenarios = len({sid for sid, _, _ in searches})
    print(f"Collected {len(searches)} search(es) over {n_scenarios} scenario(s)")
    print(f"  {frontiers_path}   {len(frontiers)} frontier configurations")
    print(f"  {scenarios_path}   {len(scenarios)} (scenario, strategy) pairs")
    missing = 2 * n_scenarios - len(searches)
    if missing:
        print(f"  note: {missing} (scenario, strategy) pair(s) have no results yet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
