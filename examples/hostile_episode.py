"""One simulated 90-day episode, shield off then shield on. SYNTHETIC: no real plant is involved.

The careless fitter of arm H-C proposes the same commands in both runs, on the same seed. H-C0 lets
them through and H-C1 puts the shield between them and the plant. One seed shows the mechanism and
proves nothing: the rates are in the README, on 500 held-out seeds. This script uses a development
seed and writes nothing, so it does not touch the ledger.

    python examples/hostile_episode.py [seed]
"""

from __future__ import annotations

import sys

from closed_loop.eval.arms import ARMS
from closed_loop.eval.context import default_context
from closed_loop.eval.episode import run_episode, scenario_for

SAFETY_PROPERTIES = ("P1", "P2", "P3")


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1100
    context = default_context()
    print(f"Seed {seed}, 90 simulated days, SYNTHETIC. Violations of P1 to P3, the plant's count:")
    for name, label in (("H-C0", "shield off"), ("H-C1", "shield on ")):
        result = run_episode(ARMS[name], seed, scenario_for("nominal"), context=context)
        found = sum(result.plant_violations.get(p, 0) for p in SAFETY_PROPERTIES)
        print(
            f"  {name} ({label}): violations {found}, throughput {result.throughput:.4f}, "
            f"overridden {result.shield_overrides} of {result.commands_proposed} commands, "
            f"{result.wall_s:.1f} s"
        )


if __name__ == "__main__":
    main()
