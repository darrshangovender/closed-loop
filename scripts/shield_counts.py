"""Solver counts of the three production games of the shield (packet P5d-a; ADR 0003 A12).

    python scripts/shield_counts.py [--out docs/adr/progress/P5d-solver-counts.json]

For each game (the crusher job at ``retest_q`` 2 and 8, the pump pair, the conveyor): raw states,
commands, winning states, observable states, mean safe commands per winning state, iterations,
the relation digest and, at table widths 1 to 4, entries, empty entries, the table hash, the mean
number of safe commands at exact width and the reachable product nodes (``shield.checker.check``
walked over the production model and the production table, which is a count of the product and
does not touch the independent reference). Nothing is typed by hand; the numbers in the progress
note are this file's output.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Hashable, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from closed_loop.shield.abstraction import (  # noqa: E402
    ABSTRACTION_VERSION,
    AbstractModel,
    ConveyorModel,
    CrusherJobModel,
    PumpPairModel,
)
from closed_loop.shield.checker import check, empty_entries  # noqa: E402
from closed_loop.shield.game import solve  # noqa: E402
from closed_loop.shield.table import build_shield  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WIDTHS = (1, 2, 3, 4)


class Walk:
    """Adapter: the checker walks anything with ``outcomes``; a production model has ``step``."""

    def __init__(self, model: AbstractModel) -> None:
        self.m = model
        self.name = model.name

    def commands(self) -> Sequence[str]:
        return self.m.commands()

    def initial(self) -> Hashable:
        return self.m.initial()

    def observable(self, s: Hashable) -> Hashable:
        return self.m.observable(s)

    def health(self, s: Hashable) -> int | tuple[int, ...]:
        return self.m.health(s)

    def outcomes(self, s: Hashable, c: str) -> frozenset[Hashable | str]:
        return self.m.step(s, c)


def count(label: str, model: AbstractModel, *, walk: bool) -> dict[str, Any]:
    t0 = time.perf_counter()
    win = solve(model)
    solve_s = time.perf_counter() - t0
    states = list(model.states())
    row: dict[str, Any] = {
        "raw_states": len(states),
        "commands": len(model.commands()),
        "winning_states": len(win.states),
        "observable_states_total": len({model.observable(s) for s in states}),
        "observable_states_in_W": len({model.observable(s) for s in win.states}),
        "mean_safe_commands_per_winning_state": round(
            sum(len(v) for v in win.safe.values()) / len(win.safe), 4
        ),
        "iterations": win.iterations,
        "solve_seconds": round(solve_s, 2),
        "relation_sha256": win.relation_sha256,
        "widths": {},
    }
    for w in WIDTHS:
        table = build_shield(model, win, w)
        exact = [
            len(c)
            for (_o, hs), c in table.entries.items()
            if all(hi - lo + 1 == w for lo, hi in zip(hs[0::2], hs[1::2], strict=True))
        ]
        entry: dict[str, Any] = {
            "entries": len(table.entries),
            "empty_entries": len(empty_entries(table)),
            "table_sha256_16": table.sha256()[:16],
            "mean_safe_at_exact_width": round(sum(exact) / len(exact), 4) if exact else None,
        }
        if walk:
            res = check(Walk(model), table, model.initial(), w)  # type: ignore[arg-type]
            entry["reachable_product_nodes"] = res.reachable
            entry["check_findings"] = len(res.findings)
        row["widths"][str(w)] = entry
    print(label, {k: v for k, v in row.items() if k != "widths"}, flush=True)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "docs/adr/progress/P5d-solver-counts.json")
    args = ap.parse_args()
    games = {
        "crusher (retest_q=2)": (CrusherJobModel(2), True),
        "pump pair": (PumpPairModel(), True),
        "conveyor": (ConveyorModel(), True),
        "crusher, stale spec (retest_q=8)": (CrusherJobModel(8), False),
    }
    out = {
        "abstraction_version": ABSTRACTION_VERSION,
        "games": {label: count(label, m, walk=w) for label, (m, w) in games.items()},
    }
    args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
