#!/usr/bin/env python3
"""Consolidate the 864 searches of paper §4.2 into two flat tables.

Reads every ``<results-dir>/<scenario_id>/<strategy>/{manifest,search_summary}.json``
written by ``python -m src.main optimize`` and emits:

    optima.csv      one row per (scenario, strategy): the reported optimum c* — its
                    scenario tiers, the five control variables, both 8-element
                    composition vectors, the cost objective with its 95 % CI, the three
                    cost drivers in dollars and as shares of the total, the achieved
                    value, and the size of the indistinguishable set I* it is reported
                    with
    scenarios.csv   one row per (scenario, strategy): realised evaluation counts,
                    feasibility rejections, and wall-clock time — the search-effort
                    figures §4.2 promises to report

The table schemas live in ``src/campaign.py``.

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

from src.campaign import build_tables, search_dirs   # noqa: E402
from src.config import SEARCH_STRATEGIES             # noqa: E402

DEFAULT_RESULTS_DIR = "results/experiments"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR, metavar="DIR",
                        help=f"Campaign output root (default: {DEFAULT_RESULTS_DIR})")
    parser.add_argument("--out-dir", default=None, metavar="DIR",
                        help="Where to write the CSVs (default: the results dir)")
    parser.add_argument("--config-dir", default="config", metavar="DIR",
                        help="Where the scenario JSONs live, for r_arrival "
                             "(default: config)")
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        parser.error(f"{results_dir} is not a directory")
    out_dir = Path(args.out_dir) if args.out_dir else results_dir

    searches = search_dirs(results_dir)
    if not searches:
        print(f"error: no finished searches found under {results_dir}", file=sys.stderr)
        return 1

    optima, scenarios = build_tables(results_dir, args.config_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    optima_path = out_dir / "optima.csv"
    scenarios_path = out_dir / "scenarios.csv"
    optima.to_csv(optima_path, index=False)
    scenarios.to_csv(scenarios_path, index=False)

    n_scenarios = len({sid for sid, _, _ in searches})
    print(f"Collected {len(searches)} search(es) over {n_scenarios} scenario(s)")
    print(f"  {optima_path}   {len(optima)} reported optima")
    print(f"  {scenarios_path}   {len(scenarios)} (scenario, strategy) pairs")
    missing = len(SEARCH_STRATEGIES) * n_scenarios - len(searches)
    if missing:
        print(f"  note: {missing} (scenario, strategy) pair(s) have no results yet")
    if len(optima) < len(searches):
        print(f"  note: {len(searches) - len(optima)} search(es) simulated no candidate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
