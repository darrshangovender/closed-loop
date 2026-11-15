"""Spike: where does asset_health.pipeline.run_asset spend its time, and is there a cheap config?

Answers ADR 0001 section "Runtime budget". Run with the venv python; prints a table.
SYNTHETIC data only. Not part of the package; not imported by tests.

    python profile_run_asset.py [days]

Measures (single process, wall-clock, perf_counter):
  - import time of asset_health pieces
  - generate_fleet for one slurry pump with a failure
  - run_asset total, then each stage separately (ingest, features, detect)
  - analyse_run (alerts + RUL + work order)
  - cheap variants: fewer trees, coarser cadence
  - cost of re-running run_asset on a truncated history (as_of) at several cut points
  - cProfile top entries for run_asset
"""

from __future__ import annotations

import cProfile
import io
import pstats
import sys
import time

t0 = time.perf_counter()
import pandas as pd  # noqa: E402

from asset_health.analysis import AnalysisConfig, analyse_run, load_default_rul_model  # noqa: E402
from asset_health.config import IngestConfig, PipelineConfig, SimConfig  # noqa: E402
from asset_health.detect.ensemble import detect_asset  # noqa: E402
from asset_health.features.build import build_features  # noqa: E402
from asset_health.ingest.checks import ingest_asset  # noqa: E402
from asset_health.models import AssetType  # noqa: E402
from asset_health.pipeline import run_asset  # noqa: E402
from asset_health.simulate.fleet import generate_fleet  # noqa: E402

T_IMPORT = time.perf_counter() - t0


def timed(fn, *a, repeat=3, **k):
    best, out = 1e9, None
    for _ in range(repeat):
        s = time.perf_counter()
        out = fn(*a, **k)
        best = min(best, time.perf_counter() - s)
    return best, out


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    print(f"import asset_health pieces (this process, warm disk cache): {T_IMPORT:.1f} s")

    cfg = SimConfig(
        seed=3, days=days, fleet={AssetType.SLURRY_PUMP: 1}, failure_fraction=1.0
    )
    s = time.perf_counter()
    fleet = generate_fleet(cfg)
    print(f"generate_fleet 1 pump {days} d: {time.perf_counter() - s:.2f} s")
    spec = fleet.specs[0]
    raw = fleet.raw[spec.asset_id]
    print(f"asset {spec.asset_id} rows {len(raw)} cols {list(raw.columns)[:6]}...")
    truth = fleet.truth
    print("failure events:", [(f.asset_id, f.mode.value) for f in getattr(truth, 'failures', [])])

    pcfg = PipelineConfig()
    t_total, run = timed(run_asset, raw, spec, fleet.maintenance, pcfg)
    print(f"run_asset total (best of 3): {t_total:.2f} s")

    # stage split
    maint = [m for m in fleet.maintenance if m.asset_id == spec.asset_id]
    t_ing, clean = timed(ingest_asset, raw, spec, pcfg.ingest, maint)
    print(f"  stage ingest_asset:      {t_ing:.2f} s")
    idx = clean.frame.index
    t_feat, feats = timed(build_features, clean, idx[0], idx[-1] + pd.Timedelta(seconds=1), pcfg.features)
    print(f"  stage build_features:    {t_feat:.2f} s")
    t_det, _ = timed(detect_asset, feats, [], pcfg.detect, pcfg.features)
    print(f"  stage detect_asset:      {t_det:.2f} s")

    # analysis
    acfg = AnalysisConfig()
    model = load_default_rul_model()
    t_an, an = timed(analyse_run, run, acfg, model)
    print(f"analyse_run (alerts + RUL + WO): {t_an:.2f} s")

    # cheap variants
    cheap = pcfg.model_copy(deep=True)
    cheap.detect.if_trees = 10
    t_c, _ = timed(run_asset, raw, spec, fleet.maintenance, cheap)
    print(f"run_asset if_trees=10: {t_c:.2f} s")

    # causal truncation cost at several cut points
    for frac in (0.25, 0.5, 0.75, 1.0):
        cut = idx[0] + (idx[-1] - idx[0]) * frac
        t_cut, _ = timed(run_asset, raw, spec, fleet.maintenance, pcfg, cut, repeat=2)
        print(f"run_asset as_of={frac:.2f} of record ({(cut - idx[0]).days} d): {t_cut:.2f} s")

    # coarser cadence (needs matching sim + ingest cadence); fidelity effect NOT assessed here
    for step in (7200, 14400):
        try:
            c2 = SimConfig(
                seed=3, days=days, fleet={AssetType.SLURRY_PUMP: 1}, failure_fraction=1.0,
                sample_interval_s=step,
            )
            f2 = generate_fleet(c2)
            sp2 = f2.specs[0]
            p2 = PipelineConfig(ingest=IngestConfig(sample_interval_s=step))
            t2, r2 = timed(run_asset, f2.raw[sp2.asset_id], sp2, f2.maintenance, p2, repeat=2)
            print(f"cadence {step // 3600} h: run_asset {t2:.2f} s, blocked={r2.blocked}, "
                  f"fused valid={int(r2.frame['fused'].notna().sum()) if not r2.blocked else 0}")
        except Exception as exc:  # report, do not hide
            print(f"cadence {step // 3600} h: FAILED {type(exc).__name__}: {str(exc)[:160]}")

    # profile
    pr = cProfile.Profile()
    pr.enable()
    run_asset(raw, spec, fleet.maintenance, pcfg)
    pr.disable()
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(14)
    print(buf.getvalue()[:3500])


if __name__ == "__main__":
    main()
