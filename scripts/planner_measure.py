"""Measure the belief planner (packet P8b): exact-DP validation, seconds per episode, dev seeds.

Everything here is on SYNTHETIC lives and on **development** seeds only (1000-1999): the world
model is fitted on 1000-1099, the diagnoser on 1000-1089, and the comparison episodes are 1500-1539.
No held-out seed is touched; nothing here is a pre-registered result.

``dp``
    the planner against exact dynamic programming on the tiny instance of 0001 section 8
    (``closed_loop.planning.validation``), for a list of particle counts and planner seeds.
``episodes``
    paired F1 episodes of the planner (one run per particle count) and of the bound rule B3 on the
    same seeds, with the same estimator, conformal provider and diagnoser. Records per episode the
    plant's cost counters, the violations the plant itself recorded, mean throughput, wall seconds,
    the process CPU seconds (which do not depend on what else the machine is doing), and the
    seconds spent inside the proposer and inside the planner's ``assess``.

The machine is shared; wall seconds are an upper bound on a quiet machine. Run under the heavy-run
lock or at below-normal priority.

    python scripts/planner_measure.py dp --particles 128 256 1024 --seeds 0 1 2
        --lookahead 1 2 3 --out PATH
    python scripts/planner_measure.py episodes --first-seed 1500 --n-seeds 40 \
        --particles 128 256 --b3-k 10 --out PATH
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from closed_loop.core.timegrid import DAY_Q  # noqa: E402
from closed_loop.planning.proposer import PlannerProposer, default_models  # noqa: E402
from closed_loop.planning.validation import run_validation  # noqa: E402
from closed_loop.proposers.baselines import BoundRule  # noqa: E402

CAL_EDGES = np.array([0.0, 0.001, 0.01, 0.05, 0.2, 0.5, 1.0 + 1e-9])
DEV_FIT = range(1000, 1100)
DIAG_FIT = range(1000, 1090)
PERCEPTION_SEEDS = tuple(range(1000, 1008))


def cpu_load_percent() -> float | None:
    try:
        import ctypes

        class FileTime(ctypes.Structure):
            _fields_ = [("lo", ctypes.c_uint32), ("hi", ctypes.c_uint32)]

        def read() -> tuple[int, int, int]:
            idle, kern, user = FileTime(), FileTime(), FileTime()
            ctypes.windll.kernel32.GetSystemTimes(  # type: ignore[attr-defined]
                ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user)
            )

            def f(t: FileTime) -> int:
                return (t.hi << 32) | t.lo

            return f(idle), f(kern), f(user)

        a = read()
        time.sleep(0.5)
        b = read()
        idle, total = b[0] - a[0], (b[1] - a[1]) + (b[2] - a[2])
        return 100.0 * (1.0 - idle / total) if total else None
    except Exception:  # not Windows, or no access: the number is only context
        return None


# ------------------------------------------------------------------------------------------- dp
def one_validation(n: int, seed: int, depth: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    r = run_validation(n, seed=seed, lookahead_readings=depth)
    row: dict[str, Any] = dict(r.summary())
    row["seconds"] = round(time.perf_counter() - t0, 1)
    row["value_target_met"] = r.value_target_met
    row["agreement_target_met"] = r.agreement_target_met
    print(json.dumps(row), flush=True)
    return row


def run_dp(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {"cpu_load_percent_at_start": cpu_load_percent(), "runs": []}
    for depth in args.lookahead:
        for n in args.particles:
            for seed in args.seeds:
                out["runs"].append(one_validation(n, seed, depth))
    return out


# ------------------------------------------------------------------------------------- episodes
def fitted_diagnoser() -> Any:
    from closed_loop.causal import mdiag
    from closed_loop.causal.diagnoser import make_diagnoser

    lives = mdiag.collect_lives(DIAG_FIT, mdiag.diagnosis_params(mdiag.DEFAULT_TILT))
    diag = make_diagnoser("scm")
    diag.fit(lives)
    return diag


def run_episode(proposer: Any, seed: int, dev: Any, plant_params: Any) -> dict[str, Any]:
    from closed_loop.perception import TheilSenEstimator, health_set
    from closed_loop.plant import Scenario, make_plant

    plant = make_plant(plant_params)
    obs = plant.reset(seed, Scenario())
    proposer.reset(seed)
    estimator = TheilSenEstimator(plant_params)
    provider = dev.provider("V3", alpha=0.10, seed=seed)
    assess_s = 0.0
    planner = getattr(proposer, "planner", None)
    if planner is not None:
        real_assess = planner.assess

        def timed(*a: Any, **k: Any) -> Any:
            nonlocal assess_s
            t = time.perf_counter()
            try:
                return real_assess(*a, **k)
            finally:
                assess_s += time.perf_counter() - t

        planner.assess = timed
    violations: Counter[str] = Counter()
    cal = np.zeros((len(CAL_EDGES) - 1, 3))  # per risk bin: count, sum of predicted, sum realised
    n_full = n_assessed = 0
    window = getattr(proposer, "horizon_days", None)
    propose_s = 0.0
    last_day = -1
    w0, c0 = time.perf_counter(), time.process_time()
    for _ in range(plant_params.horizon_q):
        if obs.day != last_day:
            last_day = obs.day
            # The simulation side tells the estimator when a part was fitted; a proposer never
            # sees the plant's truth.
            life = plant.truth.life_id
            for aid, view in obs.assets.items():
                out = estimator.update(aid, obs.day, view.hi_last, life[aid])
                iv = provider.predict(out, 0.90)
                lo, hi = health_set(iv)
                plant.set_perception(
                    aid, set_lo=lo, set_hi=hi, rul_lo=iv.lo, rul_hi=iv.hi, prior_only=iv.prior
                )
        t = time.perf_counter()
        commands = proposer.propose(obs)
        propose_s += time.perf_counter() - t
        log = getattr(proposer, "decisions", None)
        if log and log[-1][0] == obs.quarter and window is not None:
            # The planner's chance of failing inside its window if left alone, against whether the
            # asset really would have (the realised remaining life of 0003 A10). The harness reads
            # the plant's truth here; the proposer never does.
            truth = plant.truth.true_rul_days
            for aid, a in log[-1][1].assessments.items():
                p_never = float(a.open_loop_p[-1])
                b = int(np.searchsorted(CAL_EDGES, p_never, side="right")) - 1
                cal[min(b, len(CAL_EDGES) - 2)] += (1.0, p_never, float(truth[aid] <= window))
                n_assessed += 1
                n_full += bool(a.full)
        res = plant.step(commands)
        for prop, _aid, _q in res.violations:
            violations[prop] += 1
        obs = res.obs
        if res.done:
            break
    wall, cpu = time.perf_counter() - w0, time.process_time() - c0
    st = plant.stats()
    return {
        "seed": seed,
        "cost": st.cost,
        "planned_jobs": st.planned_jobs,
        "unplanned_failures": st.unplanned_failures,
        "wrong_mode_jobs": st.wrong_mode_jobs,
        "wo_rejected": st.wo_rejected,
        "inspections": st.inspections,
        "orders": st.orders,
        "violations": dict(violations),
        "throughput": float(np.mean(plant.trace().throughput)),
        "wall_s": wall,
        "cpu_s": cpu,
        "propose_s": propose_s,
        "assess_s": assess_s,
        "days": plant_params.horizon_q // DAY_Q,
        "assessed_asset_days": n_assessed,
        "full_assessments": n_full,
        "calibration": cal.tolist(),
    }


def summarise(rows: list[dict[str, Any]]) -> dict[str, float]:
    def m(key: str) -> float:
        return statistics.fmean(r[key] for r in rows)

    return {
        "episodes": len(rows),
        "cost_mean": m("cost"),
        "planned_jobs_mean": m("planned_jobs"),
        "unplanned_failures_mean": m("unplanned_failures"),
        "unplanned_failures_total": sum(r["unplanned_failures"] for r in rows),
        "wrong_mode_jobs_mean": m("wrong_mode_jobs"),
        "inspections_mean": m("inspections"),
        "throughput_mean": m("throughput"),
        "violations_total": sum(sum(r["violations"].values()) for r in rows),
        "wall_s_median": statistics.median(r["wall_s"] for r in rows),
        "cpu_s_median": statistics.median(r["cpu_s"] for r in rows),
        "cpu_s_mean": m("cpu_s"),
        "propose_s_mean": m("propose_s"),
        "assess_s_mean": m("assess_s"),
        "full_assessments_mean": m("full_assessments"),
        "assessed_asset_days_mean": m("assessed_asset_days"),
    }


def reliability(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Predicted against realised chance of failing inside the window, pooled over episodes."""
    cal = np.sum([r["calibration"] for r in rows], axis=0)
    out = []
    for i in range(len(CAL_EDGES) - 1):
        n, sp, sy = cal[i]
        out.append(
            {
                "bin_lo": float(CAL_EDGES[i]),
                "bin_hi": float(min(CAL_EDGES[i + 1], 1.0)),
                "asset_days": int(n),
                "mean_predicted": float(sp / n) if n else float("nan"),
                "realised": float(sy / n) if n else float("nan"),
                "failures": int(sy),
            }
        )
    return out


def paired(a: list[dict[str, Any]], b: list[dict[str, Any]], key: str) -> dict[str, float]:
    d = np.array([x[key] - y[key] for x, y in zip(a, b, strict=True)], dtype=float)
    se = float(d.std(ddof=1) / np.sqrt(d.size)) if d.size > 1 else float("nan")
    return {"mean_diff": float(d.mean()), "se": se, "n": int(d.size)}


def run_episodes(args: argparse.Namespace) -> dict[str, Any]:
    from closed_loop.perception import development_data
    from closed_loop.plant import PlantParams

    params = PlantParams()
    seeds = list(range(args.first_seed, args.first_seed + args.n_seeds))
    if any(not 1000 <= s < 2000 for s in seeds):
        raise SystemExit("this script uses development seeds only (1000-1999)")
    out: dict[str, Any] = {
        "seeds": f"{seeds[0]}..{seeds[-1]} (dev)",
        "world_model_fit": f"{DEV_FIT[0]}..{DEV_FIT[-1]} (dev)",
        "diagnoser_fit": f"{DIAG_FIT[0]}..{DIAG_FIT[-1]} (dev)",
        "cpu_load_percent_at_start": cpu_load_percent(),
        "arms": {},
    }
    dev = development_data(params, PERCEPTION_SEEDS)
    diag = None if args.no_diagnoser else fitted_diagnoser()
    models = default_models(args.mismatch, DEV_FIT)
    arms: dict[str, Any] = {}
    for k in args.b3_k:
        arms[f"B3-k{k:g}"] = lambda k=k: BoundRule(k=k, diagnoser=diag)
    for n in args.particles:
        arms[f"planner-N{n}"] = lambda n=n: PlannerProposer(
            models,
            n_particles=n,
            diagnoser=diag,
            mismatch=args.mismatch,
            **({} if args.lookahead is None else {"lookahead_readings": args.lookahead}),
        )
    rows: dict[str, list[dict[str, Any]]] = {}
    for name, factory in arms.items():
        proposer = factory()
        rows[name] = []
        for s in seeds:
            row = run_episode(proposer, s, dev, params)
            rows[name].append(row)
            print(
                f"{name} seed {s}: cost {row['cost']:.2f} jobs {row['planned_jobs']} "
                f"fail {row['unplanned_failures']} insp {row['inspections']} "
                f"wall {row['wall_s']:.2f}s cpu {row['cpu_s']:.2f}s",
                flush=True,
            )
        out["arms"][name] = {"summary": summarise(rows[name]), "episodes": rows[name]}
        if name.startswith("planner"):
            out["arms"][name]["reliability_of_p_fail_in_window"] = reliability(rows[name])
    out["paired_vs_B3"] = {}
    ref = f"B3-k{args.b3_k[0]:g}"
    for name in rows:
        if name == ref:
            continue
        out["paired_vs_B3"][f"{name} - {ref}"] = {
            key: paired(rows[name], rows[ref], key)
            for key in ("cost", "planned_jobs", "unplanned_failures", "inspections", "throughput")
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    dp = sub.add_parser("dp")
    dp.add_argument("--particles", type=int, nargs="+", default=[128])
    dp.add_argument("--seeds", type=int, nargs="+", default=[0])
    dp.add_argument("--lookahead", type=int, nargs="+", default=[2])
    ep = sub.add_parser("episodes")
    ep.add_argument("--first-seed", type=int, default=1500)
    ep.add_argument("--n-seeds", type=int, default=40)
    ep.add_argument("--particles", type=int, nargs="+", default=[128, 256])
    ep.add_argument("--b3-k", type=float, nargs="+", default=[10.0])
    ep.add_argument("--mismatch", type=float, default=0.0)
    ep.add_argument("--no-diagnoser", action="store_true")
    ep.add_argument("--lookahead", type=int, default=None)
    for p in (dp, ep):
        p.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    result = run_dp(args) if args.cmd == "dp" else run_episodes(args)
    if args.out is not None:
        args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
