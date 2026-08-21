#!/usr/bin/env python3
"""Generate one search configuration per campaign scenario.

The five scenario groups each come in a low / medium / high tier; `project` carries a
fourth "very high" tier and `agent` a fourth "pessimistic" one, so combining them freely
yields 3^3 * 4 * 4 = 432 scenarios.  Every generated file is the template with its five
`scenario` entries repointed at one tier combination; the search range, budgets, horizon
and seed are inherited unchanged, so all 432 searches share the experimental setup.

Usage
-----
    python scripts/generate_search_configs.py                 # write config/search/
    python scripts/generate_search_configs.py --force         # overwrite existing files
    python scripts/generate_search_configs.py --out-dir /tmp/x --template t.json
    python scripts/generate_search_configs.py --baseline   # config/search_baseline/
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.config import (   # noqa: E402
    SCENARIO_GROUPS,
    UNUSED_TIER,
    scenario_id,
    scenario_tier_combinations,
    scenario_tiers,
)

DEFAULT_TEMPLATE = "config/search_default.json"
DEFAULT_OUT_DIR = "config/search"
BASELINE_OUT_DIR = "config/search_baseline"

#: Groups the human-only baseline varies.  `debt` and `agent` are inert for a zero-agent
#: team (see config/{debt,agent}_unused.json), so they take the UNUSED_TIER sentinel and
#: the grid is 3 x 4 x 3 = 36 scenarios rather than 432.
BASELINE_VARYING_GROUPS: tuple[str, ...] = ("human", "project", "bugimpact")


def build_config(template: dict, tiers: tuple[str, ...]) -> dict:
    """Return the template with its scenario paths repointed at `tiers`."""
    sid = scenario_id(tiers)
    tier_list = ", ".join(f"{g}={t}" for g, t in zip(SCENARIO_GROUPS, tiers))
    config = dict(template)
    config["description"] = f"Campaign scenario {sid} ({tier_list})"
    config["scenario_id"] = sid
    config["scenario"] = {
        group: f"config/{group}_{tier}.json"
        for group, tier in zip(SCENARIO_GROUPS, tiers)
    }
    # Keep the key order stable: description, scenario_id, scenario, horizon, search.
    return {key: config[key] for key in ("description", "scenario_id", "scenario",
                                         "horizon", "search")}


def baseline_tier_combinations() -> list[tuple[str, ...]]:
    """Tier tuples of the human-only baseline grid, in SCENARIO_GROUPS order.

    `debt` and `agent` are held at UNUSED_TIER, so the product runs over the other three
    groups only: 3 human x 4 project x 3 bugimpact = 36 scenarios.
    """
    per_group = [
        scenario_tiers(g) if g in BASELINE_VARYING_GROUPS else (UNUSED_TIER,)
        for g in SCENARIO_GROUPS
    ]
    return list(itertools.product(*per_group))


def build_baseline_config(template: dict, tiers: tuple[str, ...]) -> dict:
    """Like `build_config`, but for a human-only baseline scenario.

    Points `debt` and `agent` at their `_unused.json` placeholders and sets `human_only`,
    which enumerates the 99-point zero-agent grid instead of searching.
    """
    config = build_config(template, tiers)
    tier_list = ", ".join(
        f"{g}={t}" for g, t in zip(SCENARIO_GROUPS, tiers) if t != UNUSED_TIER
    )
    config["description"] = (
        f"Human-only baseline {config['scenario_id']} ({tier_list}); "
        f"debt and agent are inert for a zero-agent team and are marked X"
    )
    config["search"] = {**config["search"], "human_only": True}
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--template", default=DEFAULT_TEMPLATE, metavar="PATH",
                        help=f"Search config to derive from (default: {DEFAULT_TEMPLATE})")
    parser.add_argument("--out-dir", default=None, metavar="DIR",
                        help=f"Output directory (default: {DEFAULT_OUT_DIR}, "
                             f"or {BASELINE_OUT_DIR} with --baseline)")
    parser.add_argument("--baseline", action="store_true",
                        help="Write the 36 human-only baseline scenarios instead of the "
                             "432 campaign scenarios")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing files instead of failing")
    args = parser.parse_args(argv)

    template = json.loads(Path(args.template).read_text(encoding="utf-8"))
    for key in ("horizon", "search"):
        if key not in template:
            parser.error(f"template {args.template} is missing the {key!r} key")

    default_out = BASELINE_OUT_DIR if args.baseline else DEFAULT_OUT_DIR
    out_dir = Path(args.out_dir or default_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    combinations = (baseline_tier_combinations() if args.baseline
                    else scenario_tier_combinations())
    build = build_baseline_config if args.baseline else build_config

    written = 0
    for tiers in combinations:
        config = build(template, tiers)
        path = out_dir / f"search_{config['scenario_id']}.json"
        if path.exists() and not args.force:
            print(f"error: {path} exists; pass --force to overwrite", file=sys.stderr)
            return 1
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        written += 1

    expected = len(combinations)
    if args.baseline:
        shape = " x ".join(
            str(len(scenario_tiers(g))) for g in SCENARIO_GROUPS
            if g in BASELINE_VARYING_GROUPS
        )
        print(f"Wrote {written} human-only baseline configurations to {out_dir}/ "
              f"({shape} = {expected} scenarios, one exhaustive enumeration each)")
    else:
        shape = " x ".join(str(len(scenario_tiers(g))) for g in SCENARIO_GROUPS)
        print(f"Wrote {written} search configurations to {out_dir}/ "
              f"({shape} = {expected} scenarios, one search each)")
    print(f"Template: {args.template}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
