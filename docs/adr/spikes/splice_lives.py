"""Spike: can two rendered component lives be spliced so that asset-health sees one asset history?

Answers ADR 0001 section 15 item 2 and 0002 packet P11a. Run with the venv python; prints a report.
SYNTHETIC data only. Not part of the package; not imported by tests.

    python splice_lives.py [seed ...]

Question. When the plant replaces a component, F2 renders the new life as a fresh asset-health
record and concatenates it onto the old one, with ``MaintenanceEvent(kind="repair")`` at the join.
Does the real pipeline then behave? Four checks, taken from the packet text:

  1. segments and baselines: ``run_asset`` cuts at the repair, and the new segment re-learns its
     baseline (21 days plus 48 hours settle, so about 23 days blind);
  2. blind window: no decision (fused score NaN, decision_valid False) in those 23 days;
  3. no phantom alarm at the join: ``analyse_run`` opens nothing in the blind window of life 2, and
     the alarm of life 1 closes at the repair;
  4. scores before the join are those of a single-life run, and scores after it are those of a
     run on life 2 alone (the pipeline is causal and segment-local).

Both lives are rendered as full-horizon records from the same epoch (midnight, so the diurnal,
weekly and seasonal phases of the generator line up with the calendar) and then cut by time, which
is also what makes the telemetry independent of when the policy chooses to stop. Two renderings are
tried: ``clock off`` (no logger clock drift or NTP steps, the public ``DataFaultConfig`` knob) and
``clock on`` (the generator default, drift drawn separately for each record).
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta

t0 = time.perf_counter()
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from asset_health.analysis import AnalysisConfig, analyse_run, load_default_rul_model  # noqa: E402
from asset_health.catalog import PROFILES, spec_from_profile  # noqa: E402
from asset_health.config import DataFaultConfig, PipelineConfig, SimConfig  # noqa: E402
from asset_health.models import (  # noqa: E402
    AssetType,
    FailureEvent,
    FailureMode,
    MaintenanceEvent,
)
from asset_health.pipeline import AssetRun, run_asset  # noqa: E402
from asset_health.simulate.faults import apply_data_faults  # noqa: E402
from asset_health.simulate.fleet import simulate_asset  # noqa: E402

T_IMPORT = time.perf_counter() - t0

EPOCH = datetime(2026, 10, 1, 0, 0, 0)  # midnight: generator phases line up with the calendar
HORIZON_DAYS = 90
STOP_DAY = 40.0  # life 1 is replaced here (it would fail at day 52)
JOB_HOURS = 6  # the asset is stopped, and silent, for the job
SCORES = ["cusum", "iforest", "rule", "fused"]


def render(seed: int, tag: int, failure: FailureEvent, *, clock: bool, spec):
    """One full-horizon record for one life: (clean, dirty)."""
    faults = DataFaultConfig() if clock else DataFaultConfig(
        clock_drift_asset_fraction=0.0, clock_steps_max=0
    )
    cfg = SimConfig(
        seed=seed,
        start=EPOCH,
        days=HORIZON_DAYS,
        fleet={AssetType.SLURRY_PUMP: 1},
        faults=faults,
    )
    rng = np.random.default_rng(np.random.SeedSequence([seed, tag]))
    clean, _latent, _repair = simulate_asset(rng, cfg, spec, failure, [])
    dirty, events = apply_data_faults(rng, clean, spec.asset_id, cfg.faults, cfg.sample_interval_s)
    return clean, dirty, events


def same(a: pd.Series, b: pd.Series) -> tuple[bool, float]:
    """Equal including NaN positions; the largest absolute difference where both are finite."""
    a, b = a.align(b, join="inner")
    both = a.notna() & b.notna()
    nan_ok = bool((a.isna() == b.isna()).all())
    diff = float((a[both] - b[both]).abs().max()) if both.any() else 0.0
    return nan_ok and diff < 1e-9, diff


def describe(run: AssetRun, join: pd.Timestamp) -> None:
    print(f"  blocked={run.blocked}  rows on grid={len(run.frame)}  repairs={run.repairs}")
    for k, seg in enumerate(run.segments):
        f = seg.features
        print(
            f"  segment {k}: {seg.start} .. {seg.end}  ready={f.ready}  "
            f"baseline {f.baseline_start} .. {f.baseline_end}"
        )
    after = run.frame.loc[join:]
    valid = after.index[after["decision_valid"].to_numpy()]
    first_valid = valid[0] if len(valid) else None
    blind_h = (first_valid - join) / pd.Timedelta(hours=1) if first_valid is not None else None
    print(f"  first valid decision after the join: {first_valid}  ({blind_h} h after the join)")
    in_blind = after.loc[: join + pd.Timedelta(days=21, hours=48) - pd.Timedelta(hours=1)]
    print(
        f"  inside the 21 d + 48 h window: fused finite rows={int(in_blind['fused'].notna().sum())},"
        f" decision_valid rows={int(in_blind['decision_valid'].sum())}"
    )


def alert_history(run: AssetRun, first: pd.Timestamp, last: pd.Timestamp, join: pd.Timestamp):
    """Daily analysis from ``first`` to ``last``: when a work order is open, and its hypothesis."""
    cfg = AnalysisConfig()
    model = load_default_rul_model()
    rows = []
    t = first
    while t <= last:
        a = analyse_run(run, cfg, model, as_of=t)
        wo = a.work_orders[0] if a.work_orders else None
        rows.append((t, wo.failure_mode_hypothesis.value if wo else None,
                     wo.severity.value if wo else None))
        t += pd.Timedelta(days=1)
    return rows


def summarise_alerts(rows, join: pd.Timestamp) -> None:
    opened = [r for r in rows if r[1] is not None]
    before = [r for r in opened if r[0] < join]
    after = [r for r in opened if r[0] >= join]
    print(f"  daily analysis {rows[0][0].date()} .. {rows[-1][0].date()}: "
          f"open on {len(before)} days before the join, {len(after)} days after it")
    if before:
        print(f"    first open before the join: {before[0][0]}  {before[0][1]} {before[0][2]}")
    if after:
        print(f"    first open after the join:  {after[0][0]}  {after[0][1]} {after[0][2]}")
        gap = (after[0][0] - join) / pd.Timedelta(days=1)
        print(f"    which is {gap:.1f} days after the join")
    right_after = [r for r in after if (r[0] - join) < pd.Timedelta(days=23)]
    print(f"    open inside the 23-day blind window of life 2 (phantom if > 0): {len(right_after)}")


def one_case(seed: int, clock: bool) -> dict[str, object]:
    print(f"\n=== seed {seed}, logger clock {'ON' if clock else 'OFF'} ===")
    profile = PROFILES[AssetType.SLURRY_PUMP]
    spec = spec_from_profile(profile, "PMP-01", "SYNTHETIC pump", baseline_scale=1.1)
    stop = EPOCH + timedelta(days=STOP_DAY)
    restart = stop + timedelta(hours=JOB_HOURS)
    f1 = FailureEvent(
        asset_id="PMP-01", mode=FailureMode.IMPELLER_WEAR, race=None,
        onset=EPOCH + timedelta(days=24), failure=EPOCH + timedelta(days=52),
        shape_k=2.6, amplitude=1.1, downtime_hours=48.0,
    )
    f2 = FailureEvent(
        asset_id="PMP-01", mode=FailureMode.BEARING_SPALL, race="outer",
        onset=restart + timedelta(days=25), failure=restart + timedelta(days=55),
        shape_k=3.2, amplitude=1.0, downtime_hours=60.0,
    )
    s = time.perf_counter()
    _c1, d1, ev1 = render(seed, 1, f1, clock=clock, spec=spec)
    _c2, d2, ev2 = render(seed, 2, f2, clock=clock, spec=spec)
    print(f"  rendered two {HORIZON_DAYS}-day records: {time.perf_counter() - s:.2f} s")
    drift = [e.detail for e in (*ev1, *ev2) if e.kind == "clock_drift"]
    print(f"  logger clock drift drawn: {drift if drift else 'none'}")

    half = pd.Timedelta(minutes=30)
    life1 = d1.loc[d1["timestamp"] < stop - half]
    life2 = d2.loc[d2["timestamp"] >= restart - half]
    life2 = life2.loc[life2["timestamp"] < EPOCH + timedelta(days=HORIZON_DAYS) - half]
    spliced = pd.concat([life1, life2], ignore_index=True)
    join_ev = MaintenanceEvent(
        asset_id="PMP-01", timestamp=restart, kind="repair", note="component replaced (planned)"
    )
    pcfg = PipelineConfig()
    print(f"  life 1 rows {len(life1)}, life 2 rows {len(life2)}, "
          f"gap rows (stopped, silent) {int(JOB_HOURS)}")

    s = time.perf_counter()
    run = run_asset(spliced, spec, [join_ev], pcfg)
    t_splice = time.perf_counter() - s
    print(f"  run_asset on the spliced record: {t_splice:.2f} s")
    describe(run, pd.Timestamp(restart))

    # references: life 1 alone up to the stop, life 2 alone from the restart
    ref1 = run_asset(life1.reset_index(drop=True), spec, [], pcfg)
    ref2 = run_asset(life2.reset_index(drop=True), spec, [], pcfg)
    res: dict[str, object] = {}
    pre = run.frame.loc[: pd.Timestamp(stop) - pd.Timedelta(hours=1)]
    post = run.frame.loc[pd.Timestamp(restart):]
    ok_all = True
    for col in SCORES:
        ok, diff = same(pre[col], ref1.frame[col])
        ok_all &= ok
        print(f"  before the join, {col:8s} equals the life-1-only run: {ok}  (max diff {diff:.2e})")
    res["pre_equal"] = ok_all
    ok_all = True
    for col in SCORES:
        ok, diff = same(post[col], ref2.frame[col])
        ok_all &= ok
        print(f"  after the join,  {col:8s} equals the life-2-only run: {ok}  (max diff {diff:.2e})")
    res["post_equal"] = ok_all
    # why does life 2 differ from a life-2-only run? the data-quality masks look back across the join
    mv = run.clean.valid.loc[pd.Timestamp(restart):]
    rv = ref2.clean.valid
    mv, rv = mv.align(rv, join="inner")
    changed = (mv != rv).any(axis=1)
    print(f"  masks (valid flags) differing after the join: {int(changed.sum())} rows"
          + (f", first {changed[changed].index[0]}, last {changed[changed].index[-1]}"
             if changed.any() else ""))
    z_changed = (run.z24.loc[pd.Timestamp(restart):].sub(ref2.z24, fill_value=0.0).abs() > 1e-9
                 ).any(axis=1)
    print(f"  z24 rows differing after the join: {int(z_changed.sum())}"
          + (f", first {z_changed[z_changed].index[0]}" if z_changed.any() else ""))
    hi_ok, hi_diff = same(run.z24.loc[: pd.Timestamp(stop)].mean(axis=1),
                          ref1.z24.mean(axis=1))
    print(f"  before the join, mean z24 equals the life-1-only run: {hi_ok} (max diff {hi_diff:.2e})")

    s = time.perf_counter()
    rows = alert_history(
        run, pd.Timestamp(EPOCH + timedelta(days=21)),
        pd.Timestamp(EPOCH + timedelta(days=HORIZON_DAYS - 1)), pd.Timestamp(restart),
    )
    print(f"  {len(rows)} daily analyse_run calls: {time.perf_counter() - s:.1f} s")
    summarise_alerts(rows, pd.Timestamp(restart))
    res["rows"] = rows
    return res


def main() -> None:
    seeds = [int(a) for a in sys.argv[1:]] or [3]
    print(f"import asset_health pieces (this process): {T_IMPORT:.1f} s")
    for seed in seeds:
        for clock in (False, True):
            one_case(seed, clock)


if __name__ == "__main__":
    main()
