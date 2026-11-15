"""Measurements of Batch 8a: the abstraction conformance gap of the shield, and hostile counts.

Three modes, each writing a JSON file that the progress notes quote (nothing is typed by hand):

    python scripts/hostile_and_conformance.py conformance [--episodes 200] [--first-seed 1000]
    python scripts/hostile_and_conformance.py hostile [--episodes 100] [--first-seed 1000]
                                                      [--real-screener]
    python scripts/hostile_and_conformance.py interlock [--episodes 100] [--first-seed 1000]

``conformance`` runs the Careless proposer (``p_skip`` 0.05) around the bound rule B3 with the
shield OFF on F1 episodes of the default plant, and measures how far the traces follow the
abstraction (``closed_loop.shield.conformance``). ``hostile`` counts the violations of B3 alone,
of Careless and of Gullible with the screener off and on, and lists the screener's misses.
``interlock`` runs the same inner proposers behind the hand-written interlock (ADR 0003 A4) and
reports the violations that get through, what the six rules did, and how many quarters passed
from each violation to the first negative online robustness of its property.

The episode loop here is the bare one of the proposer tests (plant, estimator, a V3 conformal
provider at 90%, one proposer); it is replaced by ``eval.episode`` when P12a lands. The estimator is
told the renewal of a component from the plant's life id: that is the simulation side acting as the
maintenance system, not a proposer reading truth.

Seeds are development seeds (1000-1999) for the episodes and calibration seeds (2000-2019) for the
conformal provider's scores. Nothing here is a held-out result and nothing was tuned on it.

Each mode can be run in pieces, each of them under two minutes of one core, which is how it is
run while another session holds the heavy-run lock: ``conformance --save-traces PART`` simulates
a range of seeds and pickles the traces, ``conformance-merge --parts PART ...`` evaluates all
of them together; ``hostile`` and ``interlock`` take ``--arm K`` (one arm per run, ``--out`` a part)
and ``merge-arms --parts FILE ...`` joins the parts. The merged numbers are exactly those of one
long run: the pieces only split it.

The run takes minutes of one core. It lowers its own priority, refuses to start at or above 88%
commit charge, and takes the shared heavy-run lock if ``--lock`` is given (waiting up to
``--wait-min`` minutes, then running anyway only with ``--small``, which cuts the episode count).
"""

from __future__ import annotations

import argparse
import ctypes
import dataclasses
import json
import pickle
import re
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from closed_loop.audit.guard import GuardError, HeavyRunLock, require_headroom  # noqa: E402
from closed_loop.core.trace import Trace  # noqa: E402
from closed_loop.eval.interlock import RULES, InterlockedProposer  # noqa: E402
from closed_loop.perception import TheilSenEstimator, development_data, health_set  # noqa: E402
from closed_loop.plant import PlantParams, Scenario, make_plant  # noqa: E402
from closed_loop.plant.notes import NOTE_BANK, cached_scan  # noqa: E402
from closed_loop.proposers import Careless, Gullible, make_baseline, screener_misses  # noqa: E402
from closed_loop.proposers.base import Proposer  # noqa: E402
from closed_loop.shield.abstraction import (  # noqa: E402
    ConveyorModel,
    CrusherJobModel,
    PumpPairModel,
)
from closed_loop.shield.conformance import F1AbstractionMap, conformance, result_records  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROGRESS = ROOT / "docs" / "adr" / "progress"
CALIB_SEEDS = tuple(range(2000, 2020))
PROPERTIES = ("P1", "P2", "P3", "P4", "P5", "P6")


def lower_priority() -> None:
    if sys.platform == "win32":
        below_normal = 0x00004000
        windll = getattr(ctypes, "windll", None)
        if windll is not None:
            windll.kernel32.SetPriorityClass(windll.kernel32.GetCurrentProcess(), below_normal)


class Runner:
    """One plant, one provider factory, many episodes."""

    def __init__(self) -> None:
        self.params = PlantParams()
        self.dev = development_data(self.params, CALIB_SEEDS)

    def episode(self, proposer: Proposer, seed: int) -> tuple[Trace, list[tuple[str, str, int]]]:
        plant = make_plant(self.params)
        obs = plant.reset(seed, Scenario())
        proposer.reset(seed)
        estimator = TheilSenEstimator(self.params)
        provider = self.dev.provider("V3", alpha=0.10, seed=seed)
        violations: list[tuple[str, str, int]] = []
        last_day = -1
        for _ in range(self.params.horizon_q):
            if obs.day != last_day:
                last_day = obs.day
                life = plant.truth.life_id
                for aid, view in obs.assets.items():
                    out = estimator.update(aid, obs.day, view.hi_last, life[aid])
                    iv = provider.predict(out, 0.90)
                    lo, hi = health_set(iv)
                    plant.set_perception(
                        aid, set_lo=lo, set_hi=hi, rul_lo=iv.lo, rul_hi=iv.hi, prior_only=iv.prior
                    )
            res = plant.step(proposer.propose(obs))
            violations.extend(res.violations)
            obs = res.obs
            if res.done:
                break
        return plant.trace(), violations


def record_dict(r: object) -> dict[str, object]:
    d = dataclasses.asdict(r)  # type: ignore[call-overload]
    d["fidelity"] = str(d["fidelity"])
    return d  # type: ignore[no-any-return]


def simulate_conformance(first_seed: int, episodes: int) -> dict[str, object]:
    """Simulate Careless(B3) episodes; the traces and the counters, nothing evaluated yet."""
    runner = Runner()
    traces: list[Trace] = []
    plant_violations: Counter[str] = Counter()
    skips = 0
    started = time.perf_counter()
    for i in range(episodes):
        proposer = Careless(make_baseline("B3", diagnoser=None), p_skip=0.05)
        trace, violations = runner.episode(proposer, first_seed + i)
        traces.append(trace)
        plant_violations.update(v[0] for v in violations)
        skips += len(proposer.skips)
        if (i + 1) % 10 == 0:
            require_headroom()
            print(
                f"  episode {i + 1}/{episodes}, {time.perf_counter() - started:.0f} s", flush=True
            )
    return {
        "first_seed": first_seed,
        "episodes": episodes,
        "traces": traces,
        "plant_violations": dict(plant_violations),
        "skips": skips,
        "seconds": time.perf_counter() - started,
    }


def evaluate_conformance(parts: list[dict[str, Any]]) -> dict[str, object]:
    """Join contiguous parts (in seed order) and run the conformance test on all of them."""
    parts = sorted(parts, key=lambda q: q["first_seed"])
    first_seed = int(parts[0]["first_seed"])
    episodes = 0
    for q in parts:
        if int(q["first_seed"]) != first_seed + episodes:
            raise SystemExit("the parts are not contiguous in seed")
        episodes += int(q["episodes"])
    traces = [t for q in parts for t in q["traces"]]
    plant_violations: Counter[str] = Counter()
    for q in parts:
        plant_violations.update(q["plant_violations"])
    skips = sum(int(q["skips"]) for q in parts)
    sim_s = sum(float(q["seconds"]) for q in parts)
    t0 = time.perf_counter()
    models = {"crusher": CrusherJobModel(), "pumps": PumpPairModel(), "conveyor": ConveyorModel()}
    report = conformance(traces, F1AbstractionMap(), models)
    conf_s = time.perf_counter() - t0
    measured_on = (
        f"{episodes} F1 episodes (90 days, default PlantParams), dev seeds "
        f"{first_seed}-{first_seed + episodes - 1}, Careless(B3, p_skip=0.05), "
        "shield off, V3 conformal provider at 90% calibrated on seeds 2000-2019, retest_q = 2"
    )
    records = result_records(
        report, measured_on=measured_on, script="scripts/hostile_and_conformance.py"
    )
    return {
        "measured_on": measured_on,
        "episodes": episodes,
        "first_seed": first_seed,
        "pieces": len(parts),
        "seconds": {"simulate": round(sim_s, 1), "conformance": round(conf_s, 1)},
        "careless_skips": skips,
        "plant_violations_by_property": {p: plant_violations.get(p, 0) for p in PROPERTIES},
        "report": {
            f.name: (
                [list(map(str, e)) for e in getattr(report, f.name)]
                if f.name == "unmatched_examples"
                else getattr(report, f.name)
            )
            for f in dataclasses.fields(report)
        },
        "records": [record_dict(r) for r in records],
    }


def run_conformance(args: argparse.Namespace) -> dict[str, object]:
    part = simulate_conformance(args.first_seed, args.episodes)
    if args.save_traces is not None:
        with args.save_traces.open("wb") as handle:
            pickle.dump(part, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return {
            "saved": str(args.save_traces),
            "first_seed": args.first_seed,
            "episodes": args.episodes,
        }
    return evaluate_conformance([part])


def merge_conformance(args: argparse.Namespace) -> dict[str, object]:
    parts = []
    for path in args.parts:
        with Path(path).open("rb") as handle:
            parts.append(pickle.load(handle))  # our own files, written by this script
    return evaluate_conformance(parts)


def merge_arms(args: argparse.Namespace) -> dict[str, object]:
    docs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.parts]
    merged = dict(docs[0])
    merged["arms"] = {label: arm for d in docs for label, arm in d["arms"].items()}
    counts = {arm["episodes"] for arm in merged["arms"].values()}
    if len(counts) > 1:  # the parts ran different numbers of episodes: say so in the header
        merged["measured_on"] = re.sub(
            r"^\d+ F1 episodes (each|per arm)",
            r"F1 episodes per arm (the number is under each arm)",
            str(merged["measured_on"]),
        )
    return merged


def tally(
    runner: Runner, make: Callable[[], Proposer], args: argparse.Namespace
) -> dict[str, object]:
    total: Counter[str] = Counter()
    with_violation: Counter[str] = Counter()
    notes_seen = notes_dropped = 0
    obeyed: Counter[str] = Counter()
    skips = p1_or_p3 = 0
    for i in range(args.episodes):
        proposer = make()
        _trace, violations = runner.episode(proposer, args.first_seed + i)
        props = Counter(v[0] for v in violations)
        total.update(props)
        with_violation.update(props.keys())
        p1_or_p3 += bool(props.get("P1") or props.get("P3"))
        notes_seen += getattr(proposer, "notes_seen", 0)
        notes_dropped += getattr(proposer, "notes_dropped", 0)
        obeyed.update(o.note_id for o in getattr(proposer, "obeyed", ()))
        skips += len(getattr(proposer, "skips", ()))
        if (i + 1) % 10 == 0:
            require_headroom()
    return {
        "episodes": args.episodes,
        "violations": {p: total.get(p, 0) for p in PROPERTIES},
        "episodes_with_violation": {p: with_violation.get(p, 0) for p in PROPERTIES},
        "episodes_with_p1_or_p3": p1_or_p3,
        "careless_skips": skips,
        "notes_seen": notes_seen,
        "notes_dropped_by_screener": notes_dropped,
        "malicious_notes_obeyed": sum(obeyed.values()),
        "obeyed_by_note": dict(sorted(obeyed.items())),
    }


def run_hostile(args: argparse.Namespace) -> dict[str, object]:
    runner = Runner()

    def b3() -> Proposer:
        return make_baseline("B3", diagnoser=None)

    makers: dict[str, Callable[[], Proposer]] = {
        "B3 alone": b3,
        "Careless(B3), p_skip 0.05": lambda: Careless(b3(), 0.05),
        "Gullible(B3), screener off": lambda: Gullible(b3(), False, cached_scan),
        "Gullible(B3), screener on (cached verdicts)": lambda: Gullible(b3(), True, cached_scan),
    }
    if args.real_screener:
        from plant_copilot.injection import scan_text

        makers["Gullible(B3), screener on (real scan_text)"] = lambda: Gullible(
            b3(), True, scan_text
        )
    chosen = list(makers) if args.arm is None else [list(makers)[args.arm]]
    arms: dict[str, object] = {label: tally(runner, makers[label], args) for label in chosen}
    bank = {
        "notes": len(NOTE_BANK),
        "malicious": sum(n.malicious for n in NOTE_BANK),
        "screener_misses_cached": [n.note_id for n in screener_misses(cached_scan)],
    }
    return {
        "measured_on": (
            f"{args.episodes} F1 episodes each (90 days, default PlantParams), dev seeds "
            f"{args.first_seed}-{args.first_seed + args.episodes - 1}, inner proposer B3 without a "
            "diagnoser, shield off, V3 conformal provider at 90% calibrated on seeds 2000-2019"
        ),
        "p_malicious_note": runner.params.p_malicious_note,
        "note_bank": bank,
        "arms": arms,
    }


LAG_BINS = ("0", "1", "2-4", "5+", "never")


def lag_bin(lag: int | None) -> str:
    if lag is None:
        return "never"
    return "0" if lag == 0 else "1" if lag == 1 else "2-4" if lag <= 4 else "5+"


def run_interlock(args: argparse.Namespace) -> dict[str, object]:
    runner = Runner()

    def b3() -> Proposer:
        return make_baseline("B3", diagnoser=None)

    arms: dict[str, Callable[[], Proposer]] = {
        "B3 + interlock": b3,
        "H-C-I: Careless(B3) + interlock": lambda: Careless(b3(), 0.05),
        "H-G-I: Gullible(B3, screener off) + interlock": lambda: Gullible(b3(), False, cached_scan),
    }
    out: dict[str, object] = {}
    chosen = list(arms) if args.arm is None else [list(arms)[args.arm]]
    for label in chosen:
        make = arms[label]
        violations: Counter[str] = Counter()
        episodes_with: Counter[str] = Counter()
        rules: Counter[str] = Counter()
        lags: dict[str, Counter[str]] = {p: Counter() for p in PROPERTIES}
        refused = forced = admitted = 0
        for i in range(args.episodes):
            interlock = InterlockedProposer(make())
            _trace, plant_violations = runner.episode(interlock, args.first_seed + i)
            record = interlock.episode_record(plant_violations)
            props = Counter(v.prop for v in record.violations)
            violations.update(props)
            episodes_with.update(props.keys())
            rules.update(record.by_rule)
            refused += record.refused
            forced += record.forced
            admitted += record.admitted
            for v in record.violations:
                lags[v.prop][lag_bin(v.lag_q)] += 1
            if (i + 1) % 10 == 0:
                require_headroom()
        out[label] = {
            "episodes": args.episodes,
            "violations": {p: violations.get(p, 0) for p in PROPERTIES},
            "episodes_with_violation": {p: episodes_with.get(p, 0) for p in PROPERTIES},
            "refused": refused,
            "forced": forced,
            "admitted": admitted,
            "by_rule": {r: rules.get(r, 0) for r in RULES},
            "quarters_to_first_negative_online_robustness": {
                p: {b: lags[p].get(b, 0) for b in LAG_BINS} for p in PROPERTIES
            },
        }
    return {
        "measured_on": (
            f"{args.episodes} F1 episodes per arm (90 days, default PlantParams), dev seeds "
            f"{args.first_seed}-{args.first_seed + args.episodes - 1}, inner proposer B3 without a "
            "diagnoser, hand-written interlock (0003 A4) with the interval midpoint as the point "
            "RUL, shield off, V3 conformal provider at 90% calibrated on seeds 2000-2019"
        ),
        "arms": out,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "mode",
        choices=["conformance", "hostile", "interlock", "conformance-merge", "merge-arms"],
    )
    ap.add_argument(
        "--save-traces", type=Path, help="conformance: pickle the traces, evaluate later"
    )
    ap.add_argument(
        "--parts", nargs="+", type=Path, help="the pieces of conformance-merge / merge-arms"
    )
    ap.add_argument("--arm", type=int, help="hostile / interlock: run only the arm with this index")
    ap.add_argument("--episodes", type=int)
    ap.add_argument("--first-seed", type=int, default=1000)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--real-screener", action="store_true")
    ap.add_argument("--lock", type=Path, help="the shared heavy-run lock file")
    ap.add_argument("--wait-min", type=float, default=0.0)
    ap.add_argument("--small", action="store_true", help="run a tenth of the episodes if no lock")
    args = ap.parse_args()
    if args.episodes is None:
        args.episodes = 200 if args.mode.startswith("conformance") else 100
    out = (
        args.out
        or PROGRESS
        / {
            "conformance": "P5c-conformance-200.json",
            "conformance-merge": "P5c-conformance-200.json",
            "hostile": "P6b-hostile-counts.json",
            "interlock": "A4-interlock-counts.json",
            "merge-arms": "merged.json",
        }[args.mode]
    )
    lower_priority()
    require_headroom()
    lock = (
        HeavyRunLock(args.lock, "mnesarchus-batch8a", f"closed-loop {args.mode}")
        if args.lock
        else None
    )
    deadline = time.time() + 60.0 * args.wait_min
    held = False
    while lock is not None and not held:
        try:
            lock.__enter__()
            held = True
        except GuardError as exc:
            if time.time() >= deadline:
                if not args.small:
                    print(f"{exc}; not running (use --small for a tenth of the work)")
                    return 2
                args.episodes = max(args.episodes // 10, 4)
                lock = None
                print(f"{exc}; running {args.episodes} episodes only")
                break
            time.sleep(2.0)
    try:
        runs = {
            "conformance": run_conformance,
            "conformance-merge": merge_conformance,
            "hostile": run_hostile,
            "interlock": run_interlock,
            "merge-arms": merge_arms,
        }
        result = runs[args.mode](args)
    finally:
        if lock is not None and held:
            lock.__exit__(None, None, None)
    out.write_text(json.dumps(result, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
