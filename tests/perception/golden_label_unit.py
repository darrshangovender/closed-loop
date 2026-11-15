"""A fixed run of V2, V3 and V7 (the per-label providers), recorded bit for bit.

``golden_label_unit.json`` was written from the code as it stood BEFORE the life-unit variants were
added (docs follow-up to P4b), by ``dump_golden``. ``test_conformal_life_unit.py`` replays the same
run and compares every float by its hexadecimal form, so any change in what V2, V3 or V7 do with
``unit="label"`` shows up as a difference. Nothing here may use the new provider names.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from closed_loop.perception.conformal import (
    development_data,
    make_provider,
    replay_open_loop,
)
from closed_loop.perception.coverage import open_loop_lives
from closed_loop.perception.estimator import EstimatorOutput
from closed_loop.perception.labels import Label, label_status
from closed_loop.plant.params import PlantParams

GOLDEN = Path(__file__).with_name("golden_label_unit.json")
NAMES = ("V2", "V3", "V7")
ESTIMATES = (
    EstimatorOutput("PU-201", 30, "estimated", 25.0, 4.0, 3.0),
    EstimatorOutput("PU-201", 31, "estimated", 6.0, 1.5, 3.0),
    EstimatorOutput("PU-201", 32, "not_degrading", 60.0, 18.0, None),
)


def _hex(x: float) -> str:
    return float(x).hex()


def label_stream() -> list[list[Label]]:
    """Twelve lives of mixed labels: censored and observed, audit and not, prior and not."""
    rng = np.random.default_rng(2024)
    lives: list[list[Label]] = []
    day = 20
    for k in range(12):
        asset = ("PU-201", "PU-202", "CR-501")[k % 3]
        n = int(rng.integers(5, 16))
        censored = bool(rng.random() < 0.6)
        end = float(day + n + rng.uniform(0.0, 12.0))
        audit = bool(rng.random() < 0.3)
        labels = []
        for i in range(n):
            lo = float(rng.uniform(-2.0, 10.0))
            hi = lo + float(rng.uniform(1.0, 25.0))
            cd = day + i
            m, sigma = float(rng.uniform(1.0, 40.0)), float(rng.uniform(0.5, 6.0))
            truth = float(rng.uniform(0.0, 50.0))
            prior = bool(rng.random() < 0.35)
            labels.append(
                Label(
                    life_id=k // 3,
                    asset_id=asset,
                    checkpoint_day=cd,
                    lo=lo,
                    hi=hi,
                    level=0.9,
                    resolved_day=100 + 3 * k,
                    status=label_status(lo, hi, cd, end, censored),
                    end_day=end,
                    censored=censored,
                    audit=audit,
                    prior=prior,
                    true_rul=truth,
                    m=m,
                    sigma=sigma,
                    blind=False,
                )
            )
        lives.append(labels)
        day += 2
    return lives


def run_stream(name: str, **kw: Any) -> dict[str, Any]:
    rng = np.random.default_rng(99)
    dev = rng.standard_normal(1500)
    prior = 1.5 * rng.standard_normal(800)
    provider = make_provider(
        name, alpha=0.10, development_scores=dev, prior_scores=prior, seed=11, **kw
    )
    out: dict[str, Any] = {"intervals": []}
    for labels in label_stream():
        provider.observe(labels)
        for est in ESTIMATES:
            iv = provider.predict(est, 0.90)
            out["intervals"].append([_hex(iv.lo), _hex(iv.m), _hex(iv.hi), iv.prior])
    out["state"] = {k: _hex(v) for k, v in sorted(provider.state().items())}
    return out


def run_replay(name: str) -> dict[str, Any]:
    params = PlantParams()
    dev = development_data(params, range(2000, 2006))
    lives = open_loop_lives(params, 1000)
    provider = dev.provider(name, alpha=0.10, seed=1000)
    cps, labels = replay_open_loop(lives, provider, 0.90)
    digest = hashlib.sha256()
    for cp in cps:
        digest.update(
            f"{cp.asset_id}|{cp.day}|{_hex(cp.lo)}|{_hex(cp.hi)}|{cp.est_status}\n".encode()
        )
    for lab in labels:
        digest.update(f"{lab.asset_id}|{lab.checkpoint_day}|{lab.status.value}\n".encode())
    return {
        "n_checkpoints": len(cps),
        "n_labels": len(labels),
        "sha256": digest.hexdigest(),
        "state": {k: _hex(v) for k, v in sorted(provider.state().items())},
    }


def golden_run() -> dict[str, Any]:
    return {name: {"stream": run_stream(name), "replay": run_replay(name)} for name in NAMES}


def dump_golden() -> None:  # run once, on the code before the life-unit variants existed
    GOLDEN.write_text(json.dumps(golden_run(), indent=1) + "\n", encoding="utf-8", newline="\n")
