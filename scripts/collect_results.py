#!/usr/bin/env python3
"""Consolidate the 486 searches of paper §4.2 into two flat tables.

Reads every ``<results-dir>/<scenario_id>/<strategy>/{manifest,search_summary}.json``
written by ``python -m src.main optimize`` and emits:

    frontiers.csv   one row per Pareto-optimal configuration, with its scenario tiers,
                    search strategy, the six control variables, both 8-element
                    composition vectors, and the three objectives with 95 % CIs
    scenarios.csv   one row per (scenario, strategy): realised evaluation counts,
                    frontier size, and wall-clock time — the search-effort figures
                    §4.2 promises to report

Usage
-----
    python scripts/collect_results.py
    python scripts/collect_results.py --results-dir results/experiments --out-dir /tmp
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.config import AGENT_ROLE_COUNT, SCENARIO_GROUPS   # noqa: E402

DEFAULT_RESULTS_DIR = "results/experiments"
STRATEGIES = ("random", "evolutionary")


def _tiers_from_scenario_id(scenario_id: str) -> dict[str, str]:
    """Recover ``{group: tier}`` from a scenario id like ``h-l_p-m_d-h_a-l_b-m``."""
    initial_to_tier = {"l": "low", "m": "medium", "h": "high"}
    tokens = scenario_id.split("_")
    if len(tokens) != len(SCENARIO_GROUPS):
        raise ValueError(f"scenario id {scenario_id!r} does not have "
                         f"{len(SCENARIO_GROUPS)} tokens")
    tiers = {}
    for group, token in zip(SCENARIO_GROUPS, tokens):
        _, _, tier_initial = token.partition("-")
        if tier_initial not in initial_to_tier:
            raise ValueError(f"unknown tier {token!r} in scenario id {scenario_id!r}")
        tiers[group] = initial_to_tier[tier_initial]
    return tiers


def _search_dirs(results_dir: Path) -> list[tuple[str, str, Path]]:
    """Find every (scenario_id, strategy, dir) that holds a finished search."""
    found = []
    for scenario_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        for strategy in STRATEGIES:
            d = scenario_dir / strategy
            if (d / "search_summary.json").is_file():
                found.append((scenario_dir.name, strategy, d))
    return found


def _frontier_rows(scenario_id: str, strategy: str, manifest: dict, summary: dict) -> list[dict]:
    tiers = _tiers_from_scenario_id(scenario_id)
    rows = []
    for key in summary["pareto_frontier"]:
        obj = summary["objectives"][key]
        row: dict = {"scenario_id": scenario_id}
        row.update({f"tier_{group}": tiers[group] for group in SCENARIO_GROUPS})
        row.update({
            "strategy": strategy,
            "config_key": key,
            "N": obj["N"], "M": obj["M"], "R": obj["R"],
            # F_freeze and A_qa were added to search_summary.json alongside the campaign;
            # fall back to the manifest for searches written before that.
            "F_freeze": obj.get("F_freeze", manifest["configs"][key]["F_freeze"]),
        })
        A_dev = obj["A_dev"]
        A_qa = obj.get("A_qa", manifest["configs"][key]["A_qa"])
        for k in range(AGENT_ROLE_COUNT):
            row[f"a_dev_{k + 1}"] = A_dev[k]
        for k in range(AGENT_ROLE_COUNT):
            row[f"a_qa_{k + 1}"] = A_qa[k]
        for name in ("value", "quality", "cost"):
            row[f"{name}_mean"] = obj[f"{name}_mean"]
            row[f"{name}_ci_lo"] = obj[f"{name}_ci"][0]
            row[f"{name}_ci_hi"] = obj[f"{name}_ci"][1]
        rows.append(row)
    return rows


def _scenario_row(scenario_id: str, strategy: str, manifest: dict, summary: dict) -> dict:
    tiers = _tiers_from_scenario_id(scenario_id)
    n_evaluated = manifest["n_evaluated"]
    n_simulated = manifest.get("n_simulated", manifest.get("n_feasible", 0))
    row: dict = {"scenario_id": scenario_id}
    row.update({f"tier_{group}": tiers[group] for group in SCENARIO_GROUPS})
    row.update({
        "strategy": strategy,
        "n_evaluated": n_evaluated,
        "n_simulated": n_simulated,
        "n_infeasible": n_evaluated - n_simulated,
        "n_frontier": len(summary["pareto_frontier"]),
        "wall_clock_s": manifest.get("wall_clock_s"),
        "T": manifest["horizon"]["T"],
        "K": manifest["horizon"]["K"],
        "master_seed": manifest["horizon"]["master_seed"],
    })
    return row


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
    out_dir.mkdir(parents=True, exist_ok=True)

    searches = _search_dirs(results_dir)
    if not searches:
        print(f"error: no finished searches found under {results_dir}", file=sys.stderr)
        return 1

    frontier_rows: list[dict] = []
    scenario_rows: list[dict] = []
    for scenario_id, strategy, d in searches:
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        summary = json.loads((d / "search_summary.json").read_text(encoding="utf-8"))
        frontier_rows.extend(_frontier_rows(scenario_id, strategy, manifest, summary))
        scenario_rows.append(_scenario_row(scenario_id, strategy, manifest, summary))

    frontiers_path = out_dir / "frontiers.csv"
    scenarios_path = out_dir / "scenarios.csv"
    pd.DataFrame(frontier_rows).to_csv(frontiers_path, index=False)
    pd.DataFrame(scenario_rows).to_csv(scenarios_path, index=False)

    n_scenarios = len({sid for sid, _, _ in searches})
    print(f"Collected {len(searches)} search(es) over {n_scenarios} scenario(s)")
    print(f"  {frontiers_path}   {len(frontier_rows)} frontier configurations")
    print(f"  {scenarios_path}   {len(scenario_rows)} (scenario, strategy) pairs")
    missing = 2 * n_scenarios - len(searches)
    if missing:
        print(f"  note: {missing} (scenario, strategy) pair(s) have no results yet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
