"""Spike: what does an F2 episode cost, and where does the time go?

Answers ADR 0001 section 13 ("F2 episode cost, about 30 s") with a measurement, for packet P11a/P11b.
Run with the venv python from the repository root; prints a report. SYNTHETIC data only.

    python docs/adr/spikes/f2_pieces_timing.py [seed ...]

There is no F2 episode runner yet (packet P11c). What exists are the pieces, so this script composes
an episode from them and times each piece with the real sibling code:

  * the plant's life table (``simulate_lives``) gives each of the four assets a first life;
  * every life is rendered (``render_life``) and the real ``run_asset`` runs on each history;
  * each day the script looks at the pipeline's fused score and asks the real ``analyse_run`` (via
    ``AssetHealthAlertSource.raise_alert``) only on days it has crossed the alert threshold (the
    caching rule of 0001 section 3);
  * an alert with an interval goes through the real orchestrator, the simulated human, the real
    gate and ``PlantCMMS``; the asset is replaced three days after the release (or when it fails,
    if no alert came first), the new life is rendered and the history re-run.

What it does not include: the plant's own step loop, the shield and the F1 estimator (0.2 s per
episode, docs/adr/progress/P3b), and plant-copilot beyond the cached clause read and ``scan_text``
(measured separately, section "copilot"). The replacement policy is the simple one above, not an
arm of the benchmark; the counts of lives, alerts and runs are what they are for this policy.
"""

from __future__ import annotations

import dataclasses
import statistics
import sys
import time
from datetime import timedelta

T0 = time.perf_counter()
import numpy as np  # noqa: E402

from closed_loop.core.rng import rng_for  # noqa: E402
from closed_loop.f2.agent import (  # noqa: E402
    build_orchestrator,
    new_mock_cmms,
    new_registry,
    procedure_store,
    register_approver,
    run_id_for,
)
from closed_loop.f2.alert_source import AssetHealthAlertSource, RunCache, aware  # noqa: E402
from closed_loop.f2.cmms import PlantCMMS, inventory_of  # noqa: E402
from closed_loop.f2.human import SimHumanApprover  # noqa: E402
from closed_loop.f2.render import (  # noqa: E402
    LifeLatent,
    asset_health_id,
    render_life,
    sim_day,
    sim_time,
)
from closed_loop.plant.observe import simulate_lives  # noqa: E402
from closed_loop.plant.params import (  # noqa: E402
    ASSET_CLASS,
    ASSET_IDS,
    MODES_BY_CLASS,
    PARTS,
    PlantParams,
)
from closed_loop.plant.scenario import Scenario  # noqa: E402

T_IMPORT_CLOSED_LOOP = time.perf_counter() - T0

P = PlantParams()
HORIZON = P.horizon_days
JOB_HOURS = 6


def timed(fn, *a, **k):
    s = time.perf_counter()
    out = fn(*a, **k)
    return time.perf_counter() - s, out


def next_life_latent(aid: str, life_id: int, start_day: float, seed: int) -> LifeLatent:
    """A later life drawn the way the plant draws one (catalogue ranges, onset 24 to 60 d)."""
    rng = rng_for(seed, "timing.life", aid, life_id)
    modes = MODES_BY_CLASS[ASSET_CLASS[aid]]
    w = np.array([m.weight for m in modes])
    spec = modes[int(np.searchsorted(np.cumsum(w / w.sum()), rng.random(), side="right"))]
    onset = start_day + 24.0 + 36.0 * rng.random()
    dur = spec.duration_days[0] + (spec.duration_days[1] - spec.duration_days[0]) * rng.random()
    shape = spec.shape[0] + (spec.shape[1] - spec.shape[0]) * rng.random()
    return LifeLatent(aid, life_id, spec.mode, onset, onset + dur, shape, 60.0, start_day, None, True)


def episode(seed: int) -> dict[str, float]:
    """One composed F2 episode. Returns seconds by piece and the counts."""
    t = dict.fromkeys(("render", "pipeline", "analysis", "agent", "human", "cmms"), 0.0)
    n = dict.fromkeys(("lives", "alerts", "no_interval", "releases", "rejects", "blocked"), 0)
    table = simulate_lives(P, seed, Scenario())
    cache = RunCache()
    clock_now = [sim_time(0.0)]
    clock = lambda: aware(clock_now[0])  # noqa: E731
    asset_map = {a: asset_health_id(a) for a in ASSET_IDS}
    source = AssetHealthAlertSource(cache, clock, asset_map)
    registry = new_registry()
    key = register_approver(registry, "planner1", "Planner One", ["PLANNER"])
    key_so = register_approver(registry, "officer1", "Officer One", ["SAFETY_OFFICER"])
    mock = new_mock_cmms(registry, clock)
    stock = {p: s.stock for p, s in PARTS.items()}
    queued: list[tuple[str, object]] = []
    cmms = PlantCMMS(mock, inventory_of(lambda: stock), lambda d, wo: queued.append((wo, d)))
    orch = build_orchestrator(source, cmms, registry, clock)
    rng = rng_for(seed, "timing.human")
    humans = [
        SimHumanApprover("planner1", key, clock, P, rng, roles=("PLANNER",)),
        SimHumanApprover("officer1", key_so, clock, P, rng, roles=("SAFETY_OFFICER",)),
    ]

    histories: dict[str, list] = {}
    latents: dict[str, LifeLatent] = {}
    replace_at: dict[str, float] = {}

    def refresh(aid: str) -> None:
        dt, _ = timed(cache.set_history, aid, histories[aid])
        t["pipeline"] += dt

    for aid in ASSET_IDS:
        life = table.lives[aid]
        lat = LifeLatent(aid, 0, life.mode, table.onset_day[aid], table.failure_day[aid],
                         life.shape, life.downtime_hours)
        latents[aid] = lat
        dt, seg = timed(render_life, lat, P, seed, 0)
        t["render"] += dt
        histories[aid] = [seg]
        n["lives"] += 1
        refresh(aid)

    stop_at: dict[str, float] = {}
    for day in range(HORIZON):
        clock_now[0] = sim_time(day + 0.99)  # end of the plant day
        for aid in ASSET_IDS:
            lat = latents[aid]
            segs = histories[aid]
            if aid not in replace_at and lat.failure_day <= day + 1:
                # an unplanned failure: the asset stops at the failure and restarts after repair
                stop_at[aid] = lat.failure_day
                replace_at[aid] = lat.failure_day + lat.downtime_hours / 24.0 + 1.0
            end = replace_at.get(aid)
            if end is not None:
                if end <= day + 1 and end < HORIZON:
                    cut_latent = dataclasses.replace(lat, end_day=stop_at[aid])
                    dt, cut = timed(render_life, cut_latent, P, seed, len(segs) - 1)
                    t["render"] += dt
                    new = next_life_latent(aid, lat.life_id + 1, end, seed)
                    dt, seg = timed(render_life, new, P, seed, len(segs))
                    t["render"] += dt
                    histories[aid] = [*segs[:-1], cut, seg]
                    latents[aid] = new
                    del replace_at[aid]
                    n["lives"] += 1
                    refresh(aid)
                continue
            if day < 24:
                continue
            run = cache.run(aid)
            # the pipeline's own score gates the (costly) analysis: look only once it has crossed
            frame = run.frame.loc[: pd_ts(clock_now[0])]
            if not (frame["fused"].tail(24) >= 1.0).any():
                continue
            outcome = source.raise_alert(aid)
            if outcome.kind == "no_interval":
                n["no_interval"] += 1
            if outcome.kind != "raised":
                continue
            n["alerts"] += 1
            rid = run_id_for(outcome.alert_id)
            dt, state = timed(orch.run, outcome.alert_id, run_id=rid)
            t["agent"] += dt
            for _ in range(96):  # up to a day, a quarter at a time after the latency
                if state.status.value not in ("AWAITING_APPROVAL", "BLOCKED"):
                    break
                for human in humans:
                    dt, state = timed(human.act, orch, rid)
                    t["human"] += dt
                clock_now[0] = clock_now[0] + timedelta(minutes=15)
            if state.status.value == "COMPLETED":
                n["releases"] += 1
                replace_at[aid] = sim_day(clock_now[0]) + 3.0
                stop_at[aid] = replace_at[aid] - 0.25
            elif state.status.value == "REJECTED":
                n["rejects"] += 1
            else:
                n["blocked"] += 1
    t["pipeline"] = cache.pipeline_seconds
    t["analysis"] = cache.analyse_seconds
    t["total"] = sum(t.values())
    return {**t, **{f"n_{k}": float(v) for k, v in n.items()},
            "n_run_asset": float(cache.run_asset_calls), "n_analyse": float(cache.analyse_calls)}


def pd_ts(instant):
    import pandas as pd

    return pd.Timestamp(aware(instant).replace(tzinfo=None))


def main() -> None:
    seeds = [int(a) for a in sys.argv[1:]] or [1, 2, 3]
    print(f"import closed_loop pieces: {T_IMPORT_CLOSED_LOOP:.1f} s (siblings are imported lazily)")
    s = time.perf_counter()
    import asset_health.pipeline  # noqa: F401

    print(f"import asset_health.pipeline (first sibling call): {time.perf_counter() - s:.1f} s")
    s = time.perf_counter()
    import maintenance_agent.orchestrator  # noqa: F401

    print(f"import maintenance_agent.orchestrator: {time.perf_counter() - s:.1f} s")
    dt, _ = timed(procedure_store)
    print(f"procedure store (signed library, once per process): {dt:.2f} s")

    print("\ncopilot (plant-copilot): clause specification and injection screening")
    s = time.perf_counter()
    import plant_copilot.injection  # noqa: F401

    print(f"  import plant_copilot.injection: {time.perf_counter() - s:.1f} s")
    from datetime import date

    from closed_loop.plant.notes import NOTE_BANK
    from closed_loop.spec.clauses import spec_from_cache, spec_from_copilot

    dt, _ = timed(spec_from_cache, date(2026, 10, 1))
    print(f"  spec_from_cache (the F0/F1 path): {dt * 1000:.2f} ms")
    s = time.perf_counter()
    spec_from_copilot(date(2026, 10, 1))
    first = time.perf_counter() - s
    s = time.perf_counter()
    spec_from_copilot(date(2026, 10, 1))
    again = time.perf_counter() - s
    print(f"  spec_from_copilot live, first ask incl. corpus load: {first:.2f} s; again: {again:.2f} s")
    from plant_copilot.injection import scan_text

    texts = [n.text for n in NOTE_BANK]
    dt, _ = timed(lambda: [scan_text(x) for x in texts])
    print(f"  scan_text over the {len(texts)}-note bank: {dt * 1000:.1f} ms "
          f"({dt / len(texts) * 1000:.2f} ms per note)")
    # per episode: the clause specification from the cache (about 1 ms, an f2 test compares it with
    # the live read) and, for the screened Gullible arms only, one scan_text per handover note:
    # 3 handovers a day for 90 days
    copilot_episode = 3 * HORIZON * dt / len(texts)

    print("\ncomposed episodes (real asset-health, maintenance-agent and plant code)")
    rows = []
    wall0, cpu0 = time.perf_counter(), time.process_time()
    for seed in seeds:
        r = episode(seed)
        rows.append(r)
        print(f"  seed {seed}: total {r['total']:.1f} s = pipeline {r['pipeline']:.1f} "
              f"({r['n_run_asset']:.0f} run_asset) + analysis {r['analysis']:.1f} "
              f"({r['n_analyse']:.0f} analyse_run) + agent {r['agent']:.2f} + human {r['human']:.2f}"
              f" + render {r['render']:.2f}; lives {r['n_lives']:.0f}, alerts {r['n_alerts']:.0f},"
              f" releases {r['n_releases']:.0f}, rejects {r['n_rejects']:.0f},"
              f" no-interval {r['n_no_interval']:.0f}")
    wall, cpu = time.perf_counter() - wall0, time.process_time() - cpu0
    print(
        f"\nprocess CPU seconds per wall second over the episodes: {cpu / wall:.2f} "
        "(near 1.0: the process had a core to itself; lower: the machine was shared)"
    )
    print("summary over seeds (seconds per episode)")
    for key in ("total", "pipeline", "analysis", "agent", "human", "render"):
        vals = [r[key] for r in rows]
        print(f"  {key:9s} mean {statistics.mean(vals):6.2f}  min {min(vals):6.2f}  "
              f"max {max(vals):6.2f}")
    print(f"  copilot   about {copilot_episode * 1000:.0f} ms ({3 * HORIZON} notes screened live at "
          f"the rate above; the clause specification comes from the cache)")
    per_run = statistics.mean(r["pipeline"] / max(r["n_run_asset"], 1) for r in rows)
    calls = statistics.mean(r["n_run_asset"] for r in rows)
    analyses = statistics.mean(r["n_analyse"] for r in rows)
    print(f"  run_asset per call, mean: {per_run:.2f} s, {calls:.1f} calls per episode")
    print(f"  analyse_run calls per episode, mean: {analyses:.1f}")


if __name__ == "__main__":
    main()
