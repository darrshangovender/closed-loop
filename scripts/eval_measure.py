"""Measurements of packet P12a on development seeds (nothing here is a held-out or F2 result).

    python scripts/eval_measure.py timing   --arms all --first-seed 1100 --episodes 5
    python scripts/eval_measure.py episodes --arms H-C1 --first-seed 1000 --episodes 50 --out p.json
    python scripts/eval_measure.py merge    --parts part1.json part2.json --out merged.json
    python scripts/eval_measure.py compare  --arms P0 P1 P1-step --first-seed 1100 --episodes 30

``timing`` runs each arm for a few episodes of the nominal condition and reports wall and CPU
seconds per episode, with the machine's load at the start (a shared laptop: read the CPU column
as the steadier of the two). ``episodes`` runs one or more arms over a range of seeds and writes
one record per episode (violations, attribution, throughput, set width, restart opportunities);
``merge`` joins pieces of the same arm; ``compare`` prints throughput, set width and the other
secondary readings per arm from the same seeds. A piece is meant to take under two minutes of one
core, which is how it runs while another session holds the heavy-run lock.

Seeds are development seeds (1100 and up, so off the planner's world-model fit on 1000 to 1099)
and calibration seeds only inside ``eval.context``. The full context is fitted once per run.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import pickle
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from closed_loop.audit.guard import commit_charge_percent, require_headroom  # noqa: E402
from closed_loop.eval.arms import ARMS  # noqa: E402
from closed_loop.eval.context import (  # noqa: E402
    PreparedContext,
    default_context,
    install_context,
)
from closed_loop.eval.episode import EpisodeResult, run_episode, scenario_for  # noqa: E402
from closed_loop.eval.metrics import P15  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROGRESS = ROOT / "docs" / "adr" / "progress"


def lower_priority() -> None:
    if sys.platform == "win32":
        windll = getattr(ctypes, "windll", None)
        if windll is not None:
            windll.kernel32.SetPriorityClass(windll.kernel32.GetCurrentProcess(), 0x00004000)


def cpu_load_percent() -> float | None:
    try:
        import subprocess

        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-Counter '\\Processor(_Total)\\% Processor Time').CounterSamples.CookedValue"],
            capture_output=True, text=True, timeout=20, check=False,
        ).stdout.strip().replace(",", ".")  # fmt: skip
        return float(out)
    except Exception:
        return None


def names(spec: list[str]) -> list[str]:
    return list(ARMS) if spec == ["all"] else spec


def record(r: EpisodeResult) -> dict[str, Any]:
    attr = Counter((a.prop, a.label) for a in r.attributions if a.prop in P15)
    return {
        "arm": r.arm,
        "seed": r.seed,
        "condition": r.condition,
        "violated": [p for p in P15 if r.min_robustness[p] < 0],
        "spec_events": {p: n for p, n in r.violations.items() if n},
        "plant_events": {p: n for p, n in r.plant_violations.items() if n},
        "disagreements": dict(r.disagreements),
        "attribution": [[p, label, n] for (p, label), n in sorted(attr.items())],
        "defects": [[a.prop, a.row] for a in r.attributions if a.prop in P15 and a.defect],
        "throughput": r.throughput,
        "cost": r.cost,
        "planned_jobs": r.planned_jobs,
        "unplanned_failures": r.unplanned_failures,
        "wrong_mode_jobs": r.wrong_mode_jobs,
        "shield_overrides": r.shield_overrides,
        "shield_admitted": r.shield_admitted,
        "commands_proposed": r.commands_proposed,
        "crusher_stopped": r.crusher_stopped,
        "crusher_restart_allowed": r.crusher_restart_allowed,
        "allowed_crusher": list(r.allowed_crusher),
        "allowed_pumps": list(r.allowed_pumps),
        "set_width": (r.contract.width_sum / r.contract.checkpoints)
        if r.contract.checkpoints
        else None,
        "contract": {
            "lives": r.contract.lives,
            "lives_miss_rul": r.contract.lives_miss_rul,
            "lives_miss_bin": r.contract.lives_miss_bin,
            "checkpoints": r.contract.checkpoints,
            "width_sum": r.contract.width_sum,
        },
        "coverage": {
            "c": r.coverage.c,
            "c_lo": r.coverage.c_lo,
            "n": r.coverage.n,
            "c_trend": r.coverage.c_trend,
            "c_lo_trend": r.coverage.c_lo_trend,
            "n_trend": r.coverage.n_trend,
        },  # fmt: skip
        "monitors": dict(r.monitors),
        "wall_s": r.wall_s,
    }


def run_timing(args: argparse.Namespace, ctx: PreparedContext) -> dict[str, Any]:
    load = cpu_load_percent()
    out: dict[str, Any] = {}
    for name in names(args.arms):
        arm = ARMS[name]
        cond = args.condition
        run_episode(
            arm, args.first_seed - 1, scenario_for(cond), context=ctx
        )  # warm-up, not counted
        wall, cpu = [], []
        for i in range(args.episodes):
            c0, w0 = time.process_time(), time.perf_counter()
            run_episode(arm, args.first_seed + i, scenario_for(cond), context=ctx)
            wall.append(time.perf_counter() - w0)
            cpu.append(time.process_time() - c0)
        out[name] = {
            "wall_median": statistics.median(wall), "wall_min": min(wall), "wall_max": max(wall),
            "cpu_median": statistics.median(cpu), "cpu_min": min(cpu),
            "episodes": args.episodes,
        }  # fmt: skip
        print(
            f"{name:14s} wall median {out[name]['wall_median']:.2f} s (min {min(wall):.2f}), "
            f"cpu median {out[name]['cpu_median']:.2f} s (min {min(cpu):.2f})",
            flush=True,
        )
    return {
        "measured_on": (
            f"dev seeds {args.first_seed} to {args.first_seed + args.episodes - 1}, "
            f"{args.condition}, {args.episodes} episodes per arm after one warm-up, "
            "full default context"
        ),
        "cpu_load_percent_at_start": load,
        "commit_charge_percent_at_start": commit_charge_percent(),
        "condition": args.condition,
        "arms": out,
    }


def run_episodes(args: argparse.Namespace, ctx: PreparedContext) -> dict[str, Any]:
    docs: dict[str, Any] = {}
    for name in names(args.arms):
        arm = ARMS[name]
        recs = []
        t0 = time.perf_counter()
        for i in range(args.episodes):
            seed = args.first_seed + i
            r = run_episode(arm, seed, scenario_for(args.condition), context=ctx)
            recs.append(record(r))
            if (i + 1) % 10 == 0:
                require_headroom()
                print(
                    f"  {name} {i + 1}/{args.episodes} {time.perf_counter() - t0:.0f} s", flush=True
                )
        docs[name] = {"condition": args.condition, "first_seed": args.first_seed, "episodes": recs}
    return {
        "measured_on": f"dev seeds from {args.first_seed}, {args.episodes} episodes per arm, "
        f"{args.condition} condition, full default context",
        "arms": docs,
    }


def merge(args: argparse.Namespace) -> dict[str, Any]:
    parts = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.parts]
    out: dict[str, Any] = {"measured_on": parts[0]["measured_on"] + " (pieces joined)", "arms": {}}
    for part in parts:
        for name, doc in part["arms"].items():
            tgt = out["arms"].setdefault(
                name,
                {"condition": doc["condition"], "first_seed": doc["first_seed"], "episodes": []},
            )
            tgt["first_seed"] = min(tgt["first_seed"], doc["first_seed"])
            tgt["episodes"].extend(doc["episodes"])
    for doc in out["arms"].values():
        doc["episodes"].sort(key=lambda e: e["seed"])
        seeds = [e["seed"] for e in doc["episodes"]]
        if len(set(seeds)) != len(seeds):
            raise SystemExit("two pieces hold the same seed")
    return out


def summarise(doc: dict[str, Any]) -> dict[str, Any]:
    eps = doc["episodes"]
    n = len(eps)
    mean = lambda key: sum(e[key] for e in eps) / n  # noqa: E731
    widths = [e["set_width"] for e in eps if e["set_width"] is not None]
    attr: Counter[str] = Counter()
    for e in eps:
        for _prop, label, k in e["attribution"]:
            attr[label] += k
    restart_n = sum(e["crusher_stopped"] for e in eps)
    return {
        "episodes": n,
        "throughput_mean": mean("throughput"),
        "throughput_min": min(e["throughput"] for e in eps),
        "throughput_max": max(e["throughput"] for e in eps),
        "set_width_mean_bins": statistics.fmean(widths) if widths else None,
        "cost_mean": mean("cost"),
        "planned_jobs_mean": mean("planned_jobs"),
        "unplanned_failures_mean": mean("unplanned_failures"),
        "episodes_with_p15_violation": sum(1 for e in eps if e["violated"]),
        "episodes_with_p13_violation": sum(
            1 for e in eps if set(e["violated"]) & {"P1", "P2", "P3"}
        ),
        "episodes_with_defect": sum(1 for e in eps if e["defects"]),
        "defects": [[e["seed"], *d] for e in eps for d in e["defects"]][:20],
        "attribution_events": dict(attr),
        "episodes_with_disagreement": sum(1 for e in eps if e["disagreements"]),
        "disagreement_kinds": dict(
            Counter(f"{p}:{k}" for e in eps for p, k in e["disagreements"].items())
        ),
        "shield_overrides_mean": mean("shield_overrides"),
        "crusher_stopped_decisions": restart_n,
        "crusher_restart_allowed_decisions": sum(e["crusher_restart_allowed"] for e in eps),
        "wall_s_mean": mean("wall_s"),
    }


def run_compare(args: argparse.Namespace, ctx: PreparedContext) -> dict[str, Any]:
    got = run_episodes(args, ctx)
    got["summary"] = {name: summarise(doc) for name, doc in got["arms"].items()}
    for name, s in got["summary"].items():
        print(name, json.dumps(s, indent=1))
    return got


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("mode", choices=["timing", "episodes", "merge", "compare", "summarise"])
    ap.add_argument("--arms", nargs="+", default=["all"])
    ap.add_argument("--first-seed", type=int, default=1100)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--condition", default="nominal", choices=["nominal", "shifted"])
    ap.add_argument("--parts", nargs="+", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--context", type=Path, help="pickle of the fitted context (made if absent)")
    args = ap.parse_args()
    lower_priority()
    require_headroom()
    if args.mode == "merge":
        result: dict[str, Any] = merge(args)
    elif args.mode == "summarise":
        docs = json.loads(Path(args.parts[0]).read_text(encoding="utf-8"))
        result = {**docs, "summary": {n: summarise(d) for n, d in docs["arms"].items()}}
        print(json.dumps(result["summary"], indent=1))
    else:
        t0 = time.perf_counter()
        if args.context is not None and args.context.exists():
            with args.context.open("rb") as handle:
                ctx = pickle.load(handle)  # our own file, written below
            install_context(ctx)
            print(f"context loaded in {time.perf_counter() - t0:.1f} s", flush=True)
        else:
            ctx = default_context()
            print(f"context fitted in {time.perf_counter() - t0:.0f} s", flush=True)
            if args.context is not None:
                with args.context.open("wb") as handle:
                    pickle.dump(ctx, handle, protocol=pickle.HIGHEST_PROTOCOL)
        runner = {"timing": run_timing, "episodes": run_episodes, "compare": run_compare}[args.mode]
        result = runner(args, ctx)
        result["context_timings_s"] = dict(ctx.timings_s)
    if args.out is not None:
        args.out.write_text(json.dumps(result, indent=1, default=str) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
