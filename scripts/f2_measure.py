"""Measure the F2 stack on development seeds and write the numbers to a JSON file (packet P11c).

    python scripts/f2_measure.py episodes [--seed 1023] [--small] [--out FILE]
    python scripts/f2_measure.py mismatch [--lives 20] [--out FILE]

``episodes`` runs the three F2 arms (F2-B3, F2-P0, F2-P1) on one development seed, in this process,
one after the other, and records for each: wall and CPU seconds, the split of the time (the
pipeline, the rule layer, the agent, the loop), ``run_asset`` calls against the caching rule, the
drafts the agent made (how many named a mode, how many were inspections) and the plant's own
summary. ``mismatch`` computes the F1-F2 estimator mismatch of 20 development lives and writes the
``ResultRecord``. Only ``dev`` (1000 to 1999) and ``calib`` (2000 to 2999) seeds are accepted. All
numbers are SYNTHETIC development numbers, not results; the machine is shared, so seconds are
reported as measured and the CPU seconds beside them.

``--small`` uses the small fitted context of the tests (a few seconds to fit); without it the full
default context is fitted first (about 30 to 60 s), as a real run does.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_OUT = ROOT / "docs" / "adr" / "progress" / "P11c-f2-measure.json"


def lower_priority() -> None:
    """Run at below-normal priority on Windows: the laptop is shared with the owner's work."""
    if os.name == "nt":
        import ctypes

        kernel = ctypes.windll.kernel32  # type: ignore[attr-defined]
        kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000)


def context(small: bool) -> Any:
    from closed_loop.eval.context import ContextSpec, default_context, prepare_context

    if not small:
        return default_context()
    return prepare_context(
        ContextSpec(
            conformal_seeds=(2000, 2040),
            proxy_seeds=(2000, 2040),
            diagnoser_seeds=(2300, 2314),
            world_model_seeds=(1000, 1020),
        )
    )


def episodes(seed: int, small: bool) -> dict[str, Any]:
    if not 1000 <= seed <= 2999:
        raise SystemExit(
            "only development (1000-1999) and calibration (2000-2999) seeds are allowed"
        )
    from closed_loop.eval.episode import scenario_for
    from closed_loop.f2.episode import F2_ARMS, simulate_f2_episode

    t = time.perf_counter()
    ctx = context(small)
    out: dict[str, Any] = {
        "seed": seed,
        "context": "small" if small else "default",
        "context_s": round(time.perf_counter() - t, 1),
        "budget_s_per_episode": 30.0,
        "composed_estimate_s": 14.0,
        "episodes": [],
    }
    for name, arm in F2_ARMS.items():
        wall, cpu = time.perf_counter(), time.process_time()
        record = simulate_f2_episode(arm, seed, scenario_for("nominal"), context=ctx)
        wall, cpu = time.perf_counter() - wall, time.process_time() - cpu
        r, log = record.result, record.log
        out["episodes"].append(
            {
                "arm": name,
                "wall_s": round(wall, 2),
                "cpu_s": round(cpu, 2),
                "seconds_by_phase": {k: round(v, 2) for k, v in log.seconds.items()},
                "run_asset_calls": log.run_asset_calls,
                "run_asset_calls_by_the_caching_rule": log.expected_run_asset_calls,
                "analyse_run_calls": log.analyse_calls,
                "requests": log.requests,
                "alerts_raised": log.alerts_raised,
                "drafts": len(log.drafts),
                "drafts_naming_a_mode": log.repair_drafts,
                "drafts_inspection_first": log.inspection_drafts,
                "alerts_without_a_draft": log.runs_without_a_draft,
                "runs_completed": log.runs_completed,
                "inspections_issued": log.inspections_issued,
                "repairs_issued": log.repairs_issued,
                "wrong_compliant_presented": log.wrong_compliant_presented,
                "wrong_compliant_signed": log.wrong_compliant_signed,
                "throughput": round(r.throughput, 4),
                "cost": round(r.cost, 3),
                "unplanned_failures": r.unplanned_failures,
                "planned_jobs": r.planned_jobs,
                "violations": dict(r.violations),
                "shield_overrides": r.shield_overrides,
                "shield_admitted": r.shield_admitted,
                "draft_confidence": sorted(
                    round(d.confidence, 2) for d in log.drafts if d.confidence is not None
                ),
            }
        )
        print(
            f"{name}: {wall:.1f} s wall, {cpu:.1f} s cpu, run_asset {log.run_asset_calls} "
            f"(rule {log.expected_run_asset_calls}), {len(log.drafts)} drafts",
            flush=True,
        )
    return out


def mismatch(lives: int) -> dict[str, Any]:
    from closed_loop.f2.mismatch import dev_lives, estimator_mismatch, mismatch_record

    t = time.perf_counter()
    rows = estimator_mismatch(dev_lives(lives))
    record = mismatch_record(rows)
    return {
        "seconds": round(time.perf_counter() - t, 1),
        "record": dataclasses.asdict(record) | {"fidelity": record.fidelity.value},
        "lives": [dataclasses.asdict(r) | {"diff": r.diff} for r in rows],
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("what", choices=["episodes", "mismatch"])
    parser.add_argument("--seed", type=int, default=1023)
    parser.add_argument("--lives", type=int, default=20)
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    lower_priority()
    result = episodes(args.seed, args.small) if args.what == "episodes" else mismatch(args.lives)
    out = args.out or DEFAULT_OUT.with_name(f"P11c-f2-measure-{args.what}.json")
    out.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
