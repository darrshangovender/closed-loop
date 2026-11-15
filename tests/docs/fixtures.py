"""A small, made-up results file and a README built around it, for the tests that prove the README
checks fail on what they are meant to catch. Every number here is invented for the test and is
labelled so; nothing is a measurement."""

from __future__ import annotations

from typing import Any, Final

from closed_loop.core.results import ResultRecord
from closed_loop.core.types import Fidelity
from closed_loop.eval.claims import Verdict, verdict_records

FIXTURE_NOTE: Final[str] = "made-up numbers for a test of the README checks"


def rec(
    name: str,
    value: float,
    n: int,
    *,
    lo: float | None = None,
    hi: float | None = None,
    k: int | None = None,
    on: str = FIXTURE_NOTE,
    **extra: Any,
) -> ResultRecord:
    ex = dict(extra)
    if k is not None:
        ex["k"] = k
    return ResultRecord(
        name=name, value=value, n=n, fidelity=Fidelity.F1, measured_on=on,
        script="tests.docs.fixtures", ci_lo=lo, ci_hi=hi,
        ci_kind="confseq-0.05" if lo is not None else "", extra=ex,
    )  # fmt: skip


def small_records() -> list[ResultRecord]:
    """Enough records for every table of the Results block to have a row, none of them real."""
    verdicts = verdict_records(
        [
            Verdict(
                "C-SHIELD", "win", "", "H-C1: 0 of 640, upper bound 0.0071", "the rule", 0.00556,
                {"n_max": 640},
            ),
            Verdict(
                "C-PLAN.P1", "inconclusive", "", "P1 against B3-S: difference 0.0431", "the rule",
                0.00556, {"n_max": 640},
            ),
            Verdict(
                "C-PERC.pooled", "loss", "", "Gap_B inside the band: 0.0117", "the rule",
                0.00278, {"n_max": 640},
            ),
        ]
    )  # fmt: skip
    records = [
        rec(
            "run/params",
            1.0,
            1,
            on="parameters of this run",
            mode="full",
            seed_role="heldout",
            n_seeds=640,
            primary_variant="V2L",
            monitor_mode="pess",
        ),
        *verdicts,
        rec(
            "claim/C-SHIELD/viol_p13/H-C1",
            0.0,
            640,
            lo=0.0,
            hi=0.0071,
            k=0,
            on="640 invented H-C1 episodes",
        ),
        rec(
            "claim/C-SHIELD/viol_p13/H-C0",
            0.4125,
            640,
            lo=0.37,
            hi=0.46,
            k=264,
            on="640 invented H-C0 episodes",
        ),
        rec("H-C1.nominal.throughput", 0.9345, 640),
        rec("H-C1.nominal.override_rate", 208.9, 640),
        rec("claim/C-SHIELD/throughput_ratio/H-C", 0.944, 640),
        rec(
            "claim/C-SHIELD/interlock/H-C",
            0.0625,
            640,
            lo=-0.01,
            hi=0.13,
            throughput_shield=0.9345,
            throughput_interlock=0.9832,
            override_shield=208.9,
            override_interlock=131.3,
        ),
        rec("claim/C-PLAN/cost_diff/P1/nominal", 0.0431, 300, lo=0.02, hi=0.07),
        rec("claim/C-PLAN/cost_diff/P1/shifted", -0.0094, 300, lo=-0.06, hi=0.03),
        rec("claim/C-PLAN/failure_diff/P1/nominal", 0.225, 300, lo=0.1, hi=0.35),
        rec("claim/C-PLAN/failure_diff/P1/shifted", 0.15, 300, lo=0.0, hi=0.3),
        rec("claim/C-PERC/gap_b/pooled/V1", 0.0117, 500, lo=-0.01, hi=0.03),
        rec("claim/C-PERC/gap_b/pooled/V2L", 0.0082, 500, lo=-0.02, hi=0.04),
        rec("claim/C-PERC/gap_b/trend/V1", 0.0611, 500, lo=0.02, hi=0.1),
        rec("claim/C-PERC/gap_b/trend/V2L", 0.0507, 500, lo=0.01, hi=0.09),
    ]
    return records


GOOD_README: Final[str] = """# closed-loop

[![tests](https://example.invalid/actions/workflows/tests.yml/badge.svg)](https://example.invalid/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB)](https://python.org)
[![Synthetic](https://img.shields.io/badge/data-synthetic-blue)](docs/)

> A closed-loop benchmark on a simulated plant, with a runtime shield and a pre-registered run.

The shield held 0 of 640 episodes in the hostile arm, and the planner's cost difference was
0.0431 against the rule, both on the numbers in the Results block.

## Why this exists

Because a claim about a safety layer needs a count and a denominator.

## Quick start

```bash
pip install -e ".[dev]"
python benchmarks/run.py --quick
```

## How it works

```mermaid
flowchart LR
  proposer --> shield --> plant
```

## Results

{block}

## Design decisions

The level is 0.05, as registered.

## Limitations

The data are synthetic.

## Project layout

`closed_loop/` holds the code.

## Tests

`pytest tests/`.

Built by [Darrshan Govender](https://github.com/darrshangovender) · [Agulhas Code](https://agulhascode.co.za)
"""
