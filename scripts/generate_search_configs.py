#!/usr/bin/env python3
"""Generate the 243 scenario search configurations of paper §4.2.

The five orthogonal scenario groups (paper §4.1) each come in a low / medium / high
tier, so combining them freely yields 3^5 = 243 scenarios.  Every generated file is
`config/search_default.json` with its five `scenario` entries repointed at one tier
combination; the search range, budget, horizon, and seed are inherited unchanged so
that all 243 searches share the experimental setup.

Each scenario is solved twice — once with random and once with evolutionary search —
via the `optimize --strategy` override, so 243 files cover all 486 searches.

Usage
-----
    python scripts/generate_search_configs.py                 # write config/search/
    python scripts/generate_search_configs.py --force         # overwrite existing files
    python scripts/generate_search_configs.py --out-dir /tmp/x --template t.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.config import SCENARIO_GROUPS, SCENARIO_TIERS   # noqa: E402

DEFAULT_TEMPLATE = "config/search_default.json"
DEFAULT_OUT_DIR = "config/search"


def scenario_id(tiers: tuple[str, ...]) -> str:
    """Compact label of a tier combination, e.g. ``h-l_p-m_d-h_a-l_b-m``.

    One ``<group initial>-<tier initial>`` token per group, in SCENARIO_GROUPS order.
    The group initials are distinct (human, project, debt, agent, bugimpact), so the
    label is unambiguous and round-trips to the tier tuple.
    """
    return "_".join(f"{group[0]}-{tier[0]}" for group, tier in zip(SCENARIO_GROUPS, tiers))


def build_config(template: dict, tiers: tuple[str, ...]) -> dict:
    """Return the template with its scenario paths repointed at `tiers`."""
    sid = scenario_id(tiers)
    tier_list = ", ".join(f"{g}={t}" for g, t in zip(SCENARIO_GROUPS, tiers))
    config = dict(template)
    config["description"] = (
        f"Paper Section 4.2 scenario {sid} ({tier_list}); "
        f"solve with --strategy random and --strategy evolutionary"
    )
    config["scenario_id"] = sid
    config["scenario"] = {
        group: f"config/{group}_{tier}.json"
        for group, tier in zip(SCENARIO_GROUPS, tiers)
    }
    # Keep the key order stable: description, scenario_id, scenario, horizon, search.
    return {key: config[key] for key in ("description", "scenario_id", "scenario",
                                         "horizon", "search")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--template", default=DEFAULT_TEMPLATE, metavar="PATH",
                        help=f"Search config to derive from (default: {DEFAULT_TEMPLATE})")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, metavar="DIR",
                        help=f"Output directory (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing files instead of failing")
    args = parser.parse_args(argv)

    template = json.loads(Path(args.template).read_text(encoding="utf-8"))
    for key in ("horizon", "search"):
        if key not in template:
            parser.error(f"template {args.template} is missing the {key!r} key")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for tiers in itertools.product(SCENARIO_TIERS, repeat=len(SCENARIO_GROUPS)):
        config = build_config(template, tiers)
        path = out_dir / f"search_{config['scenario_id']}.json"
        if path.exists() and not args.force:
            print(f"error: {path} exists; pass --force to overwrite", file=sys.stderr)
            return 1
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        written += 1

    expected = len(SCENARIO_TIERS) ** len(SCENARIO_GROUPS)
    print(f"Wrote {written} search configurations to {out_dir}/ "
          f"({len(SCENARIO_TIERS)}^{len(SCENARIO_GROUPS)} = {expected} scenarios, "
          f"{2 * expected} searches with both strategies)")
    print(f"Template: {args.template}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
