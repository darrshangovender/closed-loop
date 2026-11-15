"""The stages of the benchmark, one per row of the 0001 section 13 budget (packet P12b).

A stage is a named piece of the run that produces ``ResultRecord``s. ``STAGES`` is the registry, in
the order they run:

====================  =====  ================================================================
key                   row    what it does
====================  =====  ================================================================
``reach``             -      what each verdict can show at its registered size (made-up data)
``f0``                1      proof counts, independent checker, differential, sabotage suite
``perception_open``   2      open-loop coverage of the ten interval variants, per seed
``perception_closed`` 3      the bound rule B3 under each variant, nominal and shifted
``policy``            4      B0, B1, B2, B3 behind the shield, the planner arms and ablations
``hostile``           5      Careless, Gullible, the interlock arms, specification arms
``causal``            6      X-corr, X-scm, X-mis, and M-DIAG reversed and unreversed
``search``            7      CMA-ES against random search (packet P9b; built)
``f2``                8, 9   F2 arms and the replay of F1 finds (needs packet P11c)
``conformance_dp``    10     abstraction conformance on Careless traces, planner against exact DP
====================  =====  ================================================================

**Plug-in point for a stage whose packet is not built.** A ``Stage`` lists the modules it
``needs``; if one cannot be imported the stage reports ``not built`` (loudly, in the report, in the
run's completeness record and on the console) and the run does not call itself complete.

* ``search`` (P9b) runs ``closed_loop.search.falsify``: one search run is one task of the pool, with
  ``falsify_detailed``, ``classify`` and ``target_arm() -> Arm`` (the R-cma / R-rand target,
  "B3 + Careless + Gullible with shield"). It was a plug-in point until the module landed.
* ``f2`` needs ``closed_loop.f2.episode`` and ``closed_loop.f2.replay`` (P11c) **and**
  ``closed_loop.eval.stage_f2``, a module of P11c that defines ``run(sc: StageContext) ->
  StageOutput``; it is called as is. Nothing else in the runner changes.
  What C-TOGETHER reads from it (``claims.verdicts_from_records``): one record
  ``claim_name("C-TOGETHER", "arm", <arm>)`` per F2 arm with ``extra["completed"]`` and
  ``extra["tried"]`` (episodes run end to end, episodes tried) and, for an arm that failed,
  ``extra["cause"]``; and one record ``claim_name("C-TOGETHER", "confirmed")`` with ``extra["k"]``
  replays that confirmed the violation and ``n`` replays. ``sc.counts.f2_episodes`` and
  ``f2_replays`` say how many to run; ``sc.runner`` and ``sc.context`` are the pool and the
  fitted state; F2 seeds are ``sc.published_f2_seeds``, drawn once from the held-out guard
  (``HeldoutGuard.f2_seeds``, role ``f2``, so the ledger counts them); None in a ``--quick`` run,
  which has no F2 seeds: use development seeds there.

Every number that leaves a stage is a ``ResultRecord`` with its fidelity and what it was measured
on. The records that claims read (``claims.claim_name``) are produced here, at the level each claim
is judged at, from per-seed values paired on the seed.
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
import importlib.util
import json
import re
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

from closed_loop.core.results import ResultRecord
from closed_loop.core.types import Fidelity
from closed_loop.eval.arms import ARMS, PLANNER_PARTICLES, arm
from closed_loop.eval.claims import (
    SearchPairs,
    Verdict,
    claim_name,
    decide_pair,
    verdict_records,
    verdicts_from_records,
)
from closed_loop.eval.context import PreparedContext, current_context, shield_kit
from closed_loop.eval.counts import (
    ALPHA_DESCRIPTIVE,
    ALPHA_PRIMARY,
    FULL,
    QUICK,
    QUICK_FIRST_DEV_SEED,
    Counts,
)
from closed_loop.eval.episode import (
    SCHEDULING_ALPHA,
    EpisodeResult,
    scenario_for,
    simulate_episode,
)
from closed_loop.eval.metrics import (
    METRICS,
    EpisodeCoverage,
    aggregate,
    latency_report,
    pooled_contract,
)
from closed_loop.eval.paths import scrub
from closed_loop.eval.runner import (
    WORST_COVERAGE,
    CellRun,
    Failed,
    Runner,
    arm_fingerprint,
    simulation_code_sha256,
)
from closed_loop.eval.tuned import TunedValues, use_tuned
from closed_loop.perception.conformal import ALL_VARIANTS, ALPHA_LIFE
from closed_loop.perception.coverage import open_loop_lives
from closed_loop.stats import (
    betting_cs,
    paired_betting_cs,
    paired_cs,
    wilson,
    zero_event_upper,
)

__all__ = [
    "BenchmarkResult",
    "CAUSAL_TRAIN_START",
    "HOSTILE_ARMS",
    "NotBuiltError",
    "OpenLoopTask",
    "PERCEPTION_ARMS",
    "POLICY_ARMS",
    "STAGES",
    "CellStore",
    "Stage",
    "StageContext",
    "StageOutput",
    "finish",
    "run_benchmark",
    "paired_record",
    "run_pipeline",
    "value_record",
]

SCRIPT: Final[str] = "closed_loop.eval.stages"
CAUSAL_TRAIN_START: Final[int] = 2400  # calib seeds beyond those the fitted context uses (< 2375)
SEARCH_FIRST_SEED: Final[int] = 3000  # the first seed of the role search-dev, used by --quick
OPEN_LOOP_VARIANTS: Final[tuple[str, ...]] = ALL_VARIANTS


def _group(name: str) -> str:
    if name.startswith("B3-V"):
        return "perception"
    if name.startswith(("H-", "S-")):
        return "hostile"
    if name.startswith("P"):  # P0, P1, P2 and their variants and ablations: the planner arms
        return "planner"
    return "policy"


PERCEPTION_ARMS: Final[tuple[str, ...]] = tuple(n for n in ARMS if _group(n) == "perception")
POLICY_ARMS: Final[tuple[str, ...]] = tuple(
    n for n in ARMS if _group(n) in ("policy", "planner")
)  # the rule arms and the planner arms
HOSTILE_ARMS: Final[tuple[str, ...]] = tuple(n for n in ARMS if _group(n) == "hostile")


class NotBuiltError(Exception):
    """A stage needs a module that does not exist yet; the message says which and whose packet."""


# ----------------------------------------------------------------------------- records


def _span_text(seeds: Sequence[int]) -> str:
    return f"seeds {min(seeds)} to {max(seeds)}" if seeds else "no seeds"


def value_record(
    name: str,
    values: Sequence[float],
    *,
    lo: float = 0.0,
    hi: float = 1.0,
    alpha: float = ALPHA_DESCRIPTIVE,
    fidelity: Fidelity,
    measured_on: str,
    extra: Mapping[str, float | int | str] | None = None,
    proportion: bool = False,
) -> ResultRecord:
    """The mean of ``values`` (clipped to ``[lo, hi]``, the clips counted) with a betting
    confidence sequence at level ``alpha``; for a proportion the count and, with no event, the
    zero-event upper bound."""
    if not values:
        raise ValueError(f"{name}: no values")
    arr = np.asarray(values, dtype=np.float64)
    clipped = int(np.sum((arr < lo) | (arr > hi)))
    unit = (np.clip(arr, lo, hi) - lo) / (hi - lo)
    low, up = betting_cs(unit, alpha)
    ci_lo, ci_hi = float(low[-1]), float(up[-1])
    if ci_lo > ci_hi:  # an empty running intersection (probability at most alpha): last raw set
        raw_lo, raw_hi = betting_cs(unit, alpha, intersect=False)
        ci_lo, ci_hi = float(raw_lo[-1]), float(raw_hi[-1])
        if ci_lo > ci_hi:
            ci_lo = ci_hi = float(unit.mean())
    ex: dict[str, float | int | str] = {"clipped": clipped, **(extra or {})}
    if proportion:
        k = int(round(float(arr.sum())))
        ex["k"] = k
        if k == 0:
            ex["zero_event_upper"] = zero_event_upper(len(values), alpha)
    return ResultRecord(
        name=name,
        value=float(lo + (hi - lo) * unit.mean()),
        n=len(values),
        fidelity=fidelity,
        measured_on=measured_on,
        script=SCRIPT,
        ci_lo=lo + (hi - lo) * ci_lo,
        ci_hi=lo + (hi - lo) * ci_hi,
        ci_kind=f"confseq-{alpha:g}",
        extra=ex,
    )


def paired_record(
    name: str,
    a: Mapping[int, float],
    b: Mapping[int, float],
    *,
    lo: float,
    hi: float,
    alpha: float,
    fidelity: Fidelity,
    measured_on: str,
    extra: Mapping[str, float | int | str] | None = None,
    betting_only: bool = False,
) -> ResultRecord | None:
    """The paired difference ``mean(a - b)`` over the seeds both cells have, with a confidence
    sequence: ``stats.paired_cs`` (the spread-scaled sequence when it applies, the betting
    sequence otherwise, 0003 A16), or with ``betting_only`` the betting sequence alone, which is
    valid at every size and is what the safety leg of C-PLAN is read on. ``None`` if no seed is
    shared."""
    common = [s for s in a if s in b]  # the order of the run: a confidence sequence reads it
    if not common:
        return None
    va = np.clip(np.array([a[s] for s in common]), lo, hi)
    vb = np.clip(np.array([b[s] for s in common]), lo, hi)
    cs = paired_betting_cs if betting_only else paired_cs
    c_lo, c_hi = cs(va, vb, lo, hi, alpha)
    c_lo, c_hi = min(c_lo, c_hi), max(c_lo, c_hi)
    ex: dict[str, float | int | str] = {
        "seeds_a": len(a),
        "seeds_b": len(b),
        "range_lo": lo,
        "range_hi": hi,
        **(extra or {}),
    }
    return ResultRecord(
        name=name,
        value=float(np.mean(va - vb)),
        n=len(common),
        fidelity=fidelity,
        measured_on=f"{len(common)} paired seeds ({_span_text(common)}): {measured_on}",
        script=SCRIPT,
        ci_lo=c_lo,
        ci_hi=c_hi,
        ci_kind=f"confseq-{alpha:g}",
        extra=ex,
    )


def per_seed(results: Sequence[EpisodeResult], metric: str) -> dict[int, float]:
    """The per-episode value of a metric by seed (episodes where it is undefined are left out)."""
    spec = METRICS[metric]
    out: dict[int, float] = {}
    for r in results:
        v = spec.value(r)
        if v is not None:
            out[r.seed] = float(v)
    return out


def _range(metric: str) -> tuple[float, float]:
    spec = METRICS[metric]
    return spec.lo, spec.hi


def cell_records(arm_name: str, condition: str, run: CellRun) -> list[ResultRecord]:
    """Every applicable metric of one cell at the descriptive level, and a summary of the cell."""
    results: list[EpisodeResult] = run.values
    out: list[ResultRecord] = []
    failed = len(run.failures)
    first = results[0] if results else None
    flags: dict[str, float | int | str] = {"failed": failed}
    if first is not None:
        if first.provisional:
            flags["provisional"] = ",".join(first.provisional)
        flags["tuning"] = first.tuning
    for metric in METRICS:
        try:
            rec = aggregate(results, metric, ALPHA_DESCRIPTIVE)
        except ValueError:
            continue  # the metric does not apply to this arm
        out.append(dataclasses.replace(rec, extra={**rec.extra, **flags}))
    hw = run.halfwidth
    summary_extra: dict[str, float | int | str] = {
        "arm": arm_name,
        "condition": condition,
        "n_max": run.n_max,
        "stopped_by": run.stopped_by,
        "failed": failed,
    }  # how many were computed and how many came from the cache is the run log's, not a result
    if hw is not None:
        summary_extra["half_width"] = hw
    if failed:
        summary_extra["failures"] = "; ".join(f"seed {f.seed}: {f.kind}" for f in run.failures[:10])
    out.extend(contract_records(arm_name, condition, results))
    out.extend(latency_records(arm_name, condition, results))
    out.append(
        ResultRecord(
            name=f"{arm_name}.{condition}.cell",
            value=float(run.n),
            n=run.n,
            fidelity=Fidelity.F1,
            measured_on=(
                f"cell of arm {arm_name}, {condition} condition, {_span_text(run.seeds)}: "
                f"{run.n} episodes run, stopped by {run.stopped_by}"
            ),
            script=SCRIPT,
            extra=summary_extra,
        )
    )
    return out


def contract_records(
    arm_name: str, condition: str, results: Sequence[EpisodeResult]
) -> list[ResultRecord]:
    """The perception contract of a cell, both readings of 0003 A10 side by side: the share of
    lives with at least one checkpoint outside the set, on the realised RUL and on the health bin
    (the abstraction's ``h``), and the share of checkpoints outside it, with Wilson intervals."""
    pooled = pooled_contract(results)
    lives, cps = int(pooled["lives"]), int(pooled["checkpoints"])
    if not lives or not cps:
        return []
    source = next((r.contract_source for r in results if r.contract_source), "")
    where = (
        f"{len(results)} episodes of arm {arm_name}, {condition} condition, shield-set source "
        f"{source}, scored on both truths (0003 A10)"
    )
    sums = {
        "any_miss_rul": (sum(r.contract.lives_miss_rul for r in results), lives, "lives"),
        "any_miss_bin": (sum(r.contract.lives_miss_bin for r in results), lives, "lives"),
        "step_miss_rul": (sum(r.contract.miss_rul for r in results), cps, "checkpoints"),
        "step_miss_bin": (sum(r.contract.miss_bin for r in results), cps, "checkpoints"),
    }
    out: list[ResultRecord] = []
    for key, (k, n, unit) in sums.items():
        lo, hi = wilson(k, n)
        out.append(
            ResultRecord(
                name=f"contract.{arm_name}.{condition}.{key}",
                value=k / n,
                n=n,
                fidelity=Fidelity.F1,
                measured_on=f"{where}: share of {unit} outside the set",
                script=SCRIPT,
                ci_lo=lo,
                ci_hi=hi,
                ci_kind="wilson-0.95",
                extra={"k": k, "source": source},
            )
        )
    out.append(
        ResultRecord(
            name=f"contract.{arm_name}.{condition}.mean_width_bins",
            value=pooled["mean_width_bins"],
            n=cps,
            fidelity=Fidelity.F1,
            measured_on=f"{where}: mean width of the set in health bins",
            script=SCRIPT,
            extra={"source": source},
        )
    )
    return out


def latency_records(
    arm_name: str, condition: str, results: Sequence[EpisodeResult]
) -> list[ResultRecord]:
    """Quarters from a violation to the first negative online robustness of its property, for the
    arms that carry the hand-written interlock (0003 A4: the monitors run online in the loop)."""
    hist = latency_report(results)
    out: list[ResultRecord] = []
    for prop, bins in hist.items():
        total = sum(bins.values())
        if not total:
            continue
        for label, count in bins.items():
            out.append(
                ResultRecord(
                    name=f"latency.{arm_name}.{condition}.{prop}.{label}",
                    value=float(count),
                    n=total,
                    fidelity=Fidelity.F1,
                    measured_on=(
                        f"{len(results)} episodes of arm {arm_name}, {condition} condition: "
                        f"violations of {prop} "
                        + (
                            "that the online monitor did not notice"
                            if label == "never"
                            else f"noticed {label} quarters after they occurred"
                        )
                    ),
                    script=SCRIPT,
                    extra={"k": count, "prop": prop, "bin": label},
                )
            )
    return out


def claim_cell(
    results: Sequence[EpisodeResult], metric: str, name: str, alpha: float = ALPHA_PRIMARY
) -> ResultRecord:
    """A cell value at the claim level, renamed to the name a claim reads."""
    rec = aggregate(results, metric, alpha)
    failed = sum(1 for r in results if r.failure)
    return dataclasses.replace(rec, name=name, extra={**rec.extra, "failed": failed})


# ----------------------------------------------------------------------------- the context


class CellStore:
    """The cells run so far, by ``(arm, condition)``."""

    def __init__(self) -> None:
        self._runs: dict[tuple[str, str], CellRun] = {}

    def put(self, arm_name: str, condition: str, run: CellRun) -> None:
        self._runs[(arm_name, condition)] = run

    def run(self, arm_name: str, condition: str) -> CellRun | None:
        return self._runs.get((arm_name, condition))

    def results(self, arm_name: str, condition: str) -> list[EpisodeResult]:
        run = self.run(arm_name, condition)
        return [] if run is None else list(run.values)

    def has(self, arm_name: str, condition: str) -> bool:
        return (arm_name, condition) in self._runs

    def cells(self) -> list[tuple[str, str]]:
        return list(self._runs)


@dataclass
class StageContext:
    """Everything a stage reads. ``seeds`` are the F1 seeds, one list shared by every cell so that
    arms are paired; ``seed_role`` says whether they are held-out or (``--quick``) development."""

    mode: str
    counts: Counts
    runner: Runner
    context: PreparedContext
    seeds: tuple[int, ...]
    seed_role: str
    tuned: TunedValues
    primary_variant: str
    monitor_mode: str
    root: Path
    log: Callable[[str], None] = lambda _msg: None
    store: CellStore = field(default_factory=CellStore)
    shared: dict[str, Any] = field(default_factory=dict)
    prereg_sha: str | None = None  # the frozen pre-registration's hash; None for --quick
    # the seeds of the published search runs, drawn from the held-out guard in a full run
    # (``search_seed(prereg_sha, i)``, 0003 A14); None in a ``--quick`` run
    published_search_seeds: tuple[int, ...] | None = None
    # the seeds of the F2 arms, drawn from the guard in the same way (``f2_seed(prereg_sha, i)``,
    # role "f2", 0003 A16); None in a ``--quick`` run, which uses development seeds
    published_f2_seeds: tuple[int, ...] | None = None

    @property
    def search_seeds(self) -> tuple[int, ...]:
        """One list for the four cells of the search, so that the optimisers pair on the seed."""
        if self.published_search_seeds is not None:
            return self.published_search_seeds
        if self.mode == "full":
            raise ValueError("a full run's search seeds come from the held-out guard (A14)")
        return tuple(range(SEARCH_FIRST_SEED, SEARCH_FIRST_SEED + self.counts.search_runs))


@dataclass(frozen=True)
class StageOutput:
    key: str
    status: str  # "done" | "not built" | "partial"
    records: tuple[ResultRecord, ...] = ()
    notes: tuple[str, ...] = ()
    seconds: float = 0.0


@dataclass(frozen=True)
class Stage:
    key: str
    title: str
    rows: tuple[int, ...]  # the row(s) of the 0001 section 13 budget
    body: Callable[[StageContext], StageOutput]
    needs: tuple[str, ...] = ()
    plug_in: str = ""  # whom the stage waits for, and what they add


def missing_modules(stage: Stage) -> list[str]:
    """The modules the stage needs that cannot be found (without importing them)."""
    out: list[str] = []
    for name in stage.needs:
        try:
            if importlib.util.find_spec(name) is None:
                out.append(name)
        except (ImportError, ValueError):
            out.append(name)
    return out


# ----------------------------------------------------------------------------- cell stages


def _run_cells(sc: StageContext, names: Sequence[str]) -> list[ResultRecord]:
    records: list[ResultRecord] = []
    for name in names:
        a = ARMS[name]
        for condition in a.conditions:
            run = sc.runner.run_cell(
                a,
                sc.seeds,
                scenario_for(condition),
                batch=sc.counts.batch,
                n_max=sc.counts.cell_limit(name, condition, _group(name)),
            )
            sc.store.put(name, condition, run)
            records.extend(cell_records(name, condition, run))
            sc.log(
                f"[{name} {condition}] {run.n} episodes, stopped by {run.stopped_by}"
                + (f", {len(run.failures)} FAILED" if run.failures else "")
                + (f", {run.retried} retried" if run.retried else "")
                + f" ({run.computed} run, {run.cached} cached, {run.seconds:.0f} s)"
            )
    return records


def run_perception_closed(sc: StageContext) -> StageOutput:
    records = _run_cells(sc, PERCEPTION_ARMS)
    return StageOutput("perception_closed", "done", tuple(records))


def run_policy(sc: StageContext) -> StageOutput:
    records = _run_cells(sc, POLICY_ARMS)
    return StageOutput("policy", "done", tuple(records))


def run_hostile(sc: StageContext) -> StageOutput:
    records = _run_cells(sc, HOSTILE_ARMS)
    return StageOutput("hostile", "done", tuple(records))


# ----------------------------------------------------------------------------- open loop


@dataclass(frozen=True)
class OpenLoopTask:
    """The first lives of the four assets of one seed, nobody acting, replayed through every
    interval variant: ``{variant: EpisodeCoverage}`` (the oracle coverage of the 90% interval)."""

    condition: str
    variants: tuple[str, ...] = OPEN_LOOP_VARIANTS

    def label(self) -> str:
        return f"open-loop/{self.condition}"

    def fingerprint(self) -> dict[str, Any]:
        return {"kind": "open-loop", "condition": self.condition, "variants": list(self.variants)}

    def __call__(self, seed: int) -> dict[str, EpisodeCoverage]:
        from closed_loop.eval.metrics import coverage_of
        from closed_loop.perception.conformal import replay_open_loop

        ctx = current_context()
        lives = open_loop_lives(ctx.params, seed, scenario_for(self.condition))
        out: dict[str, EpisodeCoverage] = {}
        for variant in self.variants:
            alpha = ALPHA_LIFE if variant == "V6" else SCHEDULING_ALPHA
            provider = ctx.dev.provider(variant, alpha=alpha, seed=seed)
            checkpoints, _labels = replay_open_loop(lives, provider, 1.0 - alpha)
            out[variant] = coverage_of(checkpoints)
        return out


def run_perception_open(sc: StageContext) -> StageOutput:
    records: list[ResultRecord] = []
    coverage: dict[tuple[str, str], dict[int, EpisodeCoverage]] = {}
    notes: list[str] = []
    for condition in ("nominal", "shifted"):
        run = sc.runner.run_seeds(
            OpenLoopTask(condition),
            sc.seeds,
            batch=sc.counts.batch,
            n_max=sc.counts.open_loop_seeds,
            standin=lambda _failure: dict.fromkeys(OPEN_LOOP_VARIANTS, WORST_COVERAGE),
        )
        if run.failures:
            notes.append(
                f"open loop {condition}: {len(run.failures)} seeds failed and count at the worst "
                "coverage (zero), as a failed closed-loop episode does"
            )
        done = list(run.seeds)
        by_seed = dict(zip(done, run.values, strict=True))
        for variant in OPEN_LOOP_VARIANTS:
            cov = {s: v[variant] for s, v in by_seed.items()}
            coverage[(condition, variant)] = cov
            where = (
                f"open-loop first lives of {len(cov)} seeds ({_span_text(done)}), {condition} "
                f"condition, nobody acting, variant {variant}, scheduling interval"
            )
            for field_name, metric in (
                ("c", "m_cov"),
                ("c_lo", "m_cov_lo"),
                ("c_trend", "m_cov_trend"),
                ("c_lo_trend", "m_cov_lo_trend"),
            ):
                vals = [
                    getattr(c, field_name)
                    for c in cov.values()
                    if getattr(c, field_name) is not None
                ]
                if vals:
                    records.append(
                        value_record(
                            f"open.{variant}.{condition}.{metric}",
                            vals,
                            fidelity=Fidelity.F1,
                            measured_on=f"{where}: {METRICS[metric].describe}",
                            extra={
                                "excluded": len(cov) - len(vals),
                                "failed": len(run.failures),
                            },
                        )
                    )
    sc.shared["open_loop"] = coverage
    return StageOutput("perception_open", "done", tuple(records), tuple(notes))


# ----------------------------------------------------------------------------- F0


def _games(retest_q: int) -> list[tuple[str, Any, Any]]:
    """(key, model, reference) for every game that exists in the abstraction and the reference."""
    from closed_loop.shield import abstraction, reference

    wanted = (
        ("crusher", "CrusherJobModel", "CrusherReference", (retest_q,)),
        ("pumps", "PumpPairModel", "PumpPairReference", ()),
        ("conveyor", "ConveyorModel", "ConveyorReference", ()),
    )
    games: list[tuple[str, Any, Any]] = []
    for key, model_name, ref_name, args in wanted:
        m_cls, r_cls = getattr(abstraction, model_name, None), getattr(reference, ref_name, None)
        if m_cls is not None and r_cls is not None:
            games.append((key, m_cls(*args), r_cls(*args)))
    return games


def _run_sabotage(root: Path) -> tuple[int, int, str]:
    """The sabotage matrix of P5b/P5a as the test suite that holds it: (passed, total, note)."""
    tests = [
        root / "tests" / "shield" / "test_p5b_sabotage.py",
        root / "tests" / "shield" / "test_sabotage.py",
    ]
    if not all(t.is_file() for t in tests):
        return 0, 0, "the sabotage test files are not in this checkout"
    if importlib.util.find_spec("pytest") is None:
        return 0, 0, "pytest is not installed"
    # ``-o addopts=`` drops the project's own "-q": with two of them pytest prints no summary line
    cmd = [sys.executable, "-m", "pytest", "-o", "addopts=", "-q", "-p", "no:cacheprovider",
           "--no-header", *map(str, tests)]  # fmt: skip
    proc = subprocess.run(  # noqa: S603 - our own interpreter and our own test files
        cmd, capture_output=True, text=True, cwd=root, timeout=3600, check=False
    )
    lines = [x for x in proc.stdout.strip().splitlines() if x.strip()]
    tail = next((x for x in reversed(lines) if re.search(r"\d+ (passed|failed|error)", x)), "")
    counts = {w: int(m.group(1)) for w in ("passed", "failed", "error")
              if (m := re.search(rf"(\d+) {w}", tail))}  # fmt: skip
    passed = counts.get("passed", 0)
    total = passed + counts.get("failed", 0) + counts.get("error", 0)
    # the summary ends with the time the suite took ("in 118.30s (0:01:58)"): not a result
    stable = re.sub(r"\s+in\s+[\d.]+s(\s+\([\d:]+\))?", "", tail).strip(" =")
    return passed, total, stable or "pytest printed no summary line"


def run_f0(sc: StageContext) -> StageOutput:
    from closed_loop.shield.checker import (
        check,
        differential,
        empty_entries,
        structural_differences,
    )
    from closed_loop.shield.game import solve
    from closed_loop.shield.table import build_shield

    records: list[ResultRecord] = []
    notes: list[str] = []

    def rec(name: str, value: float, n: int, what: str, **extra: float | int | str) -> ResultRecord:
        return ResultRecord(
            name=f"F0.{name}",
            value=float(value),
            n=int(n),
            fidelity=Fidelity.F0,
            measured_on=f"abstract games of the shield (finite, exhaustive): {what}",
            script=SCRIPT,
            extra=extra,
        )

    kinds = {"clause": sc.context.spec_clause.retest_q}
    if sc.counts.f0_stale:
        kinds["stale"] = sc.context.spec_stale.retest_q
    seen: set[str] = set()
    for kind, retest_q in kinds.items():
        for key, model, ref in _games(retest_q):
            label = key if kind == "clause" else f"{key}_stale"
            if kind == "stale" and key != "crusher":
                continue
            if label in seen:
                continue
            seen.add(label)
            win = solve(model)
            states = list(model.states())
            observables = {model.observable(s) for s in states}
            observable_win = {model.observable(s) for s in win.states}
            safe_counts = [len(win.safe[s]) for s in win.states]
            n_cmd = len(tuple(model.commands()))
            records += [
                rec(f"{label}.states_raw", len(states), len(states), f"{label} raw states",
                    commands=n_cmd, retest_q=retest_q),
                rec(f"{label}.states_winning", len(win.states), len(states),
                    f"{label} states in the winning region", k=len(win.states),
                    iterations=win.iterations),
                rec(f"{label}.states_observable", len(observable_win), len(observables),
                    f"{label} observable states in the winning region", k=len(observable_win)),
                rec(f"{label}.mean_safe_commands",
                    float(np.mean(safe_counts)) if safe_counts else 0.0, len(safe_counts),
                    f"{label} mean safe commands per winning state, of {n_cmd}", commands=n_cmd),
            ]  # fmt: skip
            diff = differential(model, ref)
            struct = structural_differences(model, ref)
            pairs = len(states) * n_cmd
            records += [
                rec(f"{label}.differential_mismatches", len(diff), pairs,
                    f"{label} (state, command) pairs where the model differs from the "
                    "reference written from the English text", k=len(diff)),
                rec(f"{label}.structural_differences", len(struct), 1,
                    f"{label} differences in states, commands, initial state, observables",
                    k=len(struct)),
            ]  # fmt: skip
            if diff:
                notes.append(f"{label}: the differential found {len(diff)} mismatches")
            for width in sc.counts.f0_widths:
                table = build_shield(model, win, width)
                res = check(ref, table, model.initial(), width)
                game_states = {s for s, _box in res.visited}
                kinds_found = sorted({f[0] for f in res.findings})
                records += [
                    rec(f"{label}.w{width}.reachable_product", res.reachable, res.reachable,
                        f"{label} reachable (state, perception set) pairs of the shielded product, "
                        f"set width at most {width}", table_sha256=table.sha256()),
                    rec(f"{label}.w{width}.reachable_states", len(game_states), len(states),
                        f"{label} distinct game states reached, set width at most {width}",
                        k=len(game_states)),
                    rec(f"{label}.w{width}.findings", len(res.findings), res.reachable,
                        f"{label} violations or deadlocks found by the checker, set width at most "
                        f"{width}", k=len(res.findings), kinds=",".join(kinds_found)),
                    rec(f"{label}.w{width}.empty_entries", len(empty_entries(table)),
                        len(table.entries), f"{label} table entries that admit no command, width "
                        f"at most {width}", k=len(empty_entries(table))),
                ]  # fmt: skip
                if res.findings:
                    notes.append(f"{label} width {width}: {len(res.findings)} checker findings")
    if not sc.counts.f0_sabotage:
        notes.append(
            "cut for --quick: the sabotage suite, the stale-specification game, widths 1-3"
        )
        return StageOutput("f0", "done", tuple(records), tuple(notes))
    passed, total, tail = _run_sabotage(sc.root)
    records.append(
        rec("sabotage_suite", passed, total, "the sabotage tests of the shield and checker (each "
            "mutation must turn the checker or the differential red)", k=passed, summary=tail)
    )  # fmt: skip
    if total == 0:
        notes.append(f"sabotage suite not run: {tail}")
    elif passed < total:
        notes.append(f"sabotage suite: {total - passed} of {total} tests did not pass: {tail}")
    return StageOutput("f0", "done" if total else "partial", tuple(records), tuple(notes))


# ----------------------------------------------------------------------------- causal


def run_causal(sc: StageContext) -> StageOutput:
    from closed_loop.causal import mdiag
    from closed_loop.causal.diagnoser import make_diagnoser
    from closed_loop.causal.queries import (
        build_cases,
        make_effect_model,
        paired_difference,
        per_life_scores,
        records_from_scores,
    )
    from closed_loop.causal.twin import causal_params, generate_observational

    c = sc.counts
    records: list[ResultRecord] = []
    notes: list[str] = []
    role = sc.seed_role
    test_effect = list(sc.seeds[: c.causal_test_seeds])
    test_mdiag = list(sc.seeds[: c.mdiag_test_seeds])

    # ---- X-corr, X-scm, X-mis (F1+D): trained on calib lives, scored on held-out twin worlds
    params = causal_params()
    train = generate_observational(
        c.causal_train_lives, CAUSAL_TRAIN_START, "pi_obs", params=params
    )
    cases, skipped = build_cases(test_effect, params)
    where = (
        f"{len(cases)} test lives from {role} seeds ({_span_text(test_effect)}), F1+D plant, "
        f"operator policy pi_obs, twin-world truth; trained on {len(train)} observational lives "
        f"from calib seeds {CAUSAL_TRAIN_START}.."
    )
    scores = {}
    for kind, x_name in (
        ("correlational", "X-corr"),
        ("scm", "X-scm"),
        ("scm_misspecified", "X-mis"),
    ):
        model = make_effect_model(kind)
        model.fit(train)
        scores[x_name] = per_life_scores(model, cases)
        records.extend(records_from_scores(scores[x_name], x_name, where).values())
    notes.append(f"effect study: {len(cases)} test lives, {skipped} lives without an alert")
    for metric in ("q1", "q2_brier", "q4"):
        for a, b in (("X-scm", "X-corr"), ("X-scm", "X-mis"), ("X-mis", "X-corr")):
            records.append(
                paired_difference(
                    scores[a],
                    scores[b],
                    metric,
                    a_name=a,
                    b_name=b,
                    alpha=ALPHA_DESCRIPTIVE,
                    measured_on=where,
                )  # fmt: skip
            )
    q1 = paired_difference(
        scores["X-scm"], scores["X-corr"], "q1", a_name="X-scm", b_name="X-corr",
        alpha=ALPHA_PRIMARY, measured_on=where,
    )  # fmt: skip
    records.append(dataclasses.replace(q1, name=claim_name("C-CAUSE", "q1", "scm-corr")))

    # ---- M-DIAG: reversed and unreversed load-to-mode correlation
    tilt = mdiag.DEFAULT_TILT
    train_seeds = range(CAUSAL_TRAIN_START, CAUSAL_TRAIN_START + c.mdiag_train_seeds)
    lives = mdiag.collect_lives(train_seeds, mdiag.diagnosis_params(+tilt), role="calib")
    diagnosers = {k: make_diagnoser(k) for k in ("scm", "corr", "oracle")}
    for k in ("scm", "corr"):
        diagnosers[k].fit(lives)
    for setting, test_tilt in (("reversed", -tilt), ("unreversed", +tilt)):
        test_lives = mdiag.collect_lives(test_mdiag, mdiag.diagnosis_params(test_tilt), role=role)
        alerts = mdiag.alert_cases(test_lives)
        outcomes = {k: mdiag.evaluate(d, alerts) for k, d in diagnosers.items()}
        n_with_alert = len({a.life for a in alerts})
        where_d = (
            f"{n_with_alert} lives with an alert (of {len(test_lives)}; {len(alerts)} alerts, "
            f"scored per life) on first lives of {role} seeds "
            f"({_span_text(test_mdiag)}), F1 plant, mode_load_tilt train {tilt:+g} test "
            f"{test_tilt:+g} ({setting}), nobody acting; trained on {len(lives)} calib lives"
        )
        records.extend(
            mdiag.records(
                setting,
                alerts,
                outcomes,
                alpha=ALPHA_DESCRIPTIVE,
                measured_on=where_d,
                n_lives=len(test_lives),
            ).values()
        )
        primary = mdiag.records(
            setting,
            alerts,
            outcomes,
            alpha=ALPHA_PRIMARY,
            measured_on=where_d,
            n_lives=len(test_lives),
        )
        records.append(
            dataclasses.replace(primary["scm-corr"], name=claim_name("C-DIAG", setting, "scm-corr"))
        )
        notes.append(
            f"M-DIAG {setting}: {len(alerts)} alerts in {n_with_alert} of {len(test_lives)} lives "
            f"({len(test_lives) - n_with_alert} lives had no alert and are excluded)"
        )
    return StageOutput("causal", "done", tuple(records), tuple(notes))


# ----------------------------------------------------------------------------- search (P9b)

SEARCH_MODULE: Final[str] = "closed_loop.search.falsify"
SEARCH_EVAL_ALLOWANCE_S: Final[float] = 30.0  # per evaluation; a run is cut at budget x this
# the cells of the search, in the order a run is numbered within one seed (see ``search_hand_off``)
SEARCH_CELLS: Final[tuple[tuple[str, str], ...]] = (
    ("in", "cma"),
    ("in", "random"),
    ("out", "cma"),
    ("out", "random"),
)
# the order in which a run's kinds are said (most serious first)
SEARCH_KINDS: Final[tuple[str, ...]] = (
    "defect",
    "unexplained_beyond_assumptions",
    "contract_failure",
)


@dataclass(frozen=True)
class SearchOutcome:
    """One search run reduced to what the claims read (the evaluations themselves stay behind).

    ``kinds`` counts the violating evaluations by the A5 kind of ``search.falsify.classify``; a
    violation of P6 alone has no attribution and is not in it.
    """

    first_violation_eval: int | None
    evals: int
    violating: int
    kinds: tuple[tuple[str, int], ...] = ()
    # the first violating evaluation of the run, ``(theta, properties it violates)``: what the F2
    # replay of F1 finds starts from (prereg M-CONF); None if the run found nothing
    first_find: tuple[tuple[float, ...], tuple[str, ...]] | None = None
    failed: bool = False

    def of_kind(self, kind: str) -> int:
        return dict(self.kinds).get(kind, 0)

    def found(self, budget: int) -> bool:
        first = self.first_violation_eval
        return first is not None and first <= budget

    def evals_used(self, budget: int) -> int:
        """Evaluations to the first violation, censored at the budget when there was none."""
        first = self.first_violation_eval
        return first if first is not None and first <= budget else budget


def failed_search(budget: int) -> SearchOutcome:
    """The stand-in for a run that did not finish: nothing found, the whole budget spent."""
    return SearchOutcome(None, budget, 0, (), failed=True)


@dataclass(frozen=True)
class SearchTask:
    """One search run (the seed is the run's seed) of one optimiser in one regime against the
    shielded target of packet P9b. The tuned values travel with the task, as for an episode."""

    regime: str
    algo: str
    budget: int
    tuned: TunedValues

    def label(self) -> str:
        return f"search/R_{self.regime}/{self.algo}"

    def fingerprint(self) -> dict[str, Any]:
        module = importlib.import_module(SEARCH_MODULE)
        return {
            "kind": "search",
            "regime": self.regime,
            "algo": self.algo,
            "budget": self.budget,
            "target": arm_fingerprint(module.target_arm()),
            "tuned": dataclasses.asdict(self.tuned),
        }

    def __call__(self, seed: int) -> SearchOutcome:
        module = importlib.import_module(SEARCH_MODULE)
        with use_tuned(self.tuned):
            result = module.falsify_detailed(
                module.target_arm, self.regime, self.algo, self.budget, seed
            )
        kinds: Counter[str] = Counter()
        for ev in result.violations:
            # the (property, label) events, so that classify applies the rule of 0003 A15 per
            # property (a P6 event is attributed but is not a defect); bare labels would be read
            # as violations of P1 to P5
            kind = module.classify(result.shielded, result.regime, ev.events)
            if kind != "none":
                kinds[kind] += 1
        first = result.violations[0] if result.violations else None  # evaluation order
        return SearchOutcome(
            result.run.first_violation_eval,
            int(result.run.evals),
            len(result.violations),
            tuple(sorted(kinds.items())),
            None
            if first is None
            else (tuple(float(x) for x in first.theta), tuple(str(p) for p in first.violated)),
        )


@contextlib.contextmanager
def _run_deadline(runner: Runner, seconds: float) -> Iterator[None]:
    """A search run is hundreds of episodes: give the runner's per-task limit room for that."""
    saved = runner.timeout_s
    if saved is not None:
        runner.timeout_s = max(saved, seconds)
    try:
        yield
    finally:
        runner.timeout_s = saved


@dataclass(frozen=True)
class SearchFind:
    """One find of the search that the F2 replay starts from: the first violating evaluation of a
    run (``seed``, ``regime`` and ``algo`` name the run; ``theta`` is the scenario vector and
    ``props`` the properties it violated at F1)."""

    seed: int
    theta: tuple[float, ...]
    regime: str
    algo: str
    props: tuple[str, ...]


def search_hand_off(
    runs: Mapping[tuple[str, str], Mapping[int, SearchOutcome]],
    seeds: Sequence[int],
    limit: int,
) -> list[SearchFind]:
    """The replay set of M-CONF (prereg 5.1): the first violating evaluation of each search run, in
    run order across regimes and algorithms, capped at ``limit``.

    The rule, which is deterministic and does not favour a cell: the runs are numbered by the seed
    list (run ``i`` is the ``i``-th seed) and, within a seed, by ``SEARCH_CELLS`` (R_in CMA-ES,
    R_in random, R_out CMA-ES, R_out random); the finds are taken in that order, one per run that
    found anything, until ``limit`` is reached. So 25 replays are the finds of the first runs of
    all four cells, and not the first 25 of one cell, and no find is chosen for its severity.
    """
    chosen: list[SearchFind] = []
    for seed in seeds:
        for regime, algo in SEARCH_CELLS:
            outcome = runs.get((regime, algo), {}).get(seed)
            if outcome is None or outcome.first_find is None:
                continue
            theta, props = outcome.first_find
            chosen.append(SearchFind(seed, theta, regime, algo, props))
            if len(chosen) >= limit:
                return chosen
    return chosen


def search_pairs(
    cma: Mapping[int, SearchOutcome],
    rand: Mapping[int, SearchOutcome],
    seeds: Sequence[int],
    budget: int,
) -> SearchPairs:
    """The run pairs of C-SEARCH in ``R_out`` (0003 A16): run ``i`` of CMA-ES against run ``i`` of
    random search, which share a seed. A pair goes to the algorithm that found a violation when
    the other did not, or that needed fewer evaluations when both did; the rest are dropped and
    counted (``decide_pair``). A run that failed found nothing."""
    took_cma = took_rand = neither = tied = 0
    for seed in seeds:
        a, b = cma[seed], rand[seed]
        first_a = a.first_violation_eval if a.found(budget) else None
        first_b = b.first_violation_eval if b.found(budget) else None
        by_cma, by_rand = decide_pair(first_a, first_b, budget)
        took_cma += by_cma
        took_rand += by_rand
        if not (by_cma or by_rand):
            if first_a is None and first_b is None:
                neither += 1
            else:
                tied += 1
    return SearchPairs(took_cma, took_rand, neither, tied)


def search_output(
    runs: Mapping[tuple[str, str], Mapping[int, SearchOutcome]],
    failed: Mapping[tuple[str, str], Sequence[int]],
    seeds: Sequence[int],
    budget: int,
    f2_replays: int,
    published: bool,
) -> tuple[list[ResultRecord], list[str], list[SearchFind]]:
    """Everything the search stage writes, from the outcomes of its runs: the records C-SEARCH and
    C-SHIELD read, the notes, and the finds handed to the F2 replay. ``runs`` is keyed by ``(regime,
    algorithm)`` and then by seed; ``published`` says whether the seeds are the derived ones.

    A function of the outcomes alone, so that ``eval.reachability`` can pass made-up runs through
    the same code (0003 A16).
    """
    records: list[ResultRecord] = []
    seed_text = (
        f"seeds search_seed(pre-registration hash, i) for i = 0 to {len(seeds) - 1}"
        if published
        else f"development seeds (search-dev) {seeds[0]} to {seeds[-1]}"
    )
    where = (
        f"{len(seeds)} search runs per cell, {seed_text}, budget {budget} "
        "evaluations, target arm B3 + Careless + Gullible with shield"
    )
    find: dict[tuple[str, str], dict[int, float]] = {}
    evals: dict[tuple[str, str], dict[int, float]] = {}
    for key, outcomes in runs.items():
        regime, algo = key
        n_failed = len(failed[key])
        find[key] = {s: float(o.found(budget)) for s, o in outcomes.items()}
        evals[key] = {s: float(o.evals_used(budget)) for s, o in outcomes.items()}
        on = f"{where}, regime R_{regime}, {algo}"
        flag = {"failed": n_failed}
        for alpha, rname in (
            (ALPHA_DESCRIPTIVE, f"search.R_{regime}.{algo}.find_rate"),
            (ALPHA_PRIMARY, claim_name("C-SEARCH", "find_rate", f"R_{regime}", algo)),
        ):
            records.append(
                value_record(
                    rname,
                    list(find[key].values()),
                    alpha=alpha,
                    fidelity=Fidelity.F1,
                    proportion=True,
                    measured_on=f"{on}: a violation found within the budget"
                    + (
                        f"; {n_failed} of the runs failed and count as nothing found"
                        if n_failed
                        else ""
                    ),
                    extra=flag,
                )  # fmt: skip
            )
        records.append(
            value_record(
                f"search.R_{regime}.{algo}.evals_to_first",
                list(evals[key].values()),
                hi=float(budget),
                fidelity=Fidelity.F1,
                measured_on=f"{on}: evaluations to the first violation, censored at the budget",
                extra=flag,
            )  # fmt: skip
        )
        for kind in SEARCH_KINDS:
            if regime == "in" and kind == "unexplained_beyond_assumptions":
                continue  # a kind of regime out only
            if regime == "out" and kind == "defect":
                continue  # a kind of regime in only
            records.append(
                value_record(
                    f"search.R_{regime}.{algo}.{kind}_runs",
                    [float(o.of_kind(kind) > 0) for o in outcomes.values()],
                    fidelity=Fidelity.F1,
                    proportion=True,
                    measured_on=f"{on}: runs with at least one violating evaluation of kind {kind}",
                )  # fmt: skip
            )
    # the verdict is decided on run pairs with an exact interval (0003 A16); the two differences
    # below are descriptive numbers in its headline, at the descriptive level
    pairs = search_pairs(runs[("out", "cma")], runs[("out", "random")], seeds, budget)
    share = pairs.share()
    records.append(
        ResultRecord(
            name=claim_name("C-SEARCH", "pairs", "R_out"),
            value=share.value,
            n=pairs.decided,
            fidelity=Fidelity.F1,
            measured_on=(
                "CMA-ES's share of the decided run pairs in R_out (a pair goes to the algorithm "
                "that found a violation when the other did not, or that needed fewer evaluations "
                f"when both did; the others are dropped and counted); {where}"
            ),
            script=SCRIPT,
            ci_lo=share.lo,
            ci_hi=share.hi,
            ci_kind=f"clopper-pearson-{ALPHA_PRIMARY:g}",
            extra={
                "k": pairs.cma,
                "cma": pairs.cma,
                "random": pairs.rand,
                "neither": pairs.neither,
                "tied": pairs.tied,
                "pairs": pairs.total,
            },
        )
    )
    d = paired_record(
        claim_name("C-SEARCH", "find_diff", "R_out"),
        find[("out", "cma")], find[("out", "random")], lo=0.0, hi=1.0, alpha=ALPHA_DESCRIPTIVE,
        fidelity=Fidelity.F1,
        measured_on=f"P(find | CMA-ES) minus P(find | random), R_out, descriptive; {where}",
    )  # fmt: skip
    e = paired_record(
        claim_name("C-SEARCH", "evals_diff", "R_out"),
        evals[("out", "cma")], evals[("out", "random")], lo=0.0, hi=float(budget),
        alpha=ALPHA_DESCRIPTIVE, fidelity=Fidelity.F1,
        measured_on=(
            f"evaluations to first violation, CMA-ES minus random, R_out, descriptive; {where}"
        ),
    )  # fmt: skip
    records.extend(x for x in (d, e) if x is not None)
    r_in = _runs_with(runs, "in", "defect")
    r_in_contract = _runs_with(runs, "in", "contract_failure")
    n_in = sum(len(runs[("in", a)]) for a in ("cma", "random"))
    for name in (claim_name("C-SEARCH", "r_in_violations"), claim_name("C-SHIELD", "search_r_in")):
        records.append(
            ResultRecord(
                name=name,
                value=float(r_in),
                n=n_in,
                fidelity=Fidelity.F1,
                measured_on=(
                    "R_in runs with a defect: a violation of the shielded target that the "
                    f"attribution labels assumption_held; {where}"
                ),
                script=SCRIPT,
                extra={"k": r_in, "contract_failure_runs": r_in_contract},
            )  # fmt: skip
        )
    notes = [
        f"search R_in: {r_in} of {n_in} runs with a defect, {r_in_contract} with a contract "
        "failure (a violation with a broken assumption)"
    ]
    out_unexplained = _runs_with(runs, "out", "unexplained_beyond_assumptions")
    if out_unexplained:
        notes.append(
            f"search R_out: {out_unexplained} of the runs had a violation that none of A_exec, "
            "A_phys and C1 explains (a gap the three do not name)"
        )
    n_failed_runs = sum(len(v) for v in failed.values())
    if n_failed_runs:
        notes.append(f"search: {n_failed_runs} of the runs failed and count as nothing found")
    finds = search_hand_off(runs, seeds, f2_replays)
    notes.append(
        f"search: {len(finds)} finds (the first of a run, in run order across cells; at most "
        f"{f2_replays}) handed to the F2 replay"
    )
    return records, notes, finds


def run_search(sc: StageContext) -> StageOutput:
    """CMA-ES against random search over the scenario vector, per regime (0002 section 3.11).

    Ten runs per (regime, algorithm), the same seed list for both algorithms so that runs pair by
    seed. Each run is one task of the pool (cached, resumable, reaped on failure). The module is
    ``closed_loop.search.falsify`` of packet P9b: ``falsify_detailed``, ``classify`` and
    ``target_arm``.
    """
    module = importlib.import_module(SEARCH_MODULE)
    for needed in ("falsify_detailed", "classify", "target_arm"):
        if not hasattr(module, needed):
            raise NotBuiltError(f"{SEARCH_MODULE} must export {needed} (packet P9b)")
    budget = sc.counts.search_budget
    seeds = sc.search_seeds
    runs: dict[tuple[str, str], dict[int, SearchOutcome]] = {}
    failed: dict[tuple[str, str], list[int]] = {}
    with _run_deadline(sc.runner, budget * SEARCH_EVAL_ALLOWANCE_S):
        for regime in ("in", "out"):
            for algo in ("cma", "random"):
                run = sc.runner.run_seeds(
                    SearchTask(regime, algo, budget, sc.tuned),
                    seeds,
                    batch=sc.counts.batch,
                    n_max=len(seeds),
                    standin=lambda _f: failed_search(budget),
                )
                runs[(regime, algo)] = dict(zip(run.seeds, run.values, strict=True))
                failed[(regime, algo)] = [f.seed for f in run.failures]
                sc.log(
                    f"[search R_{regime} {algo}] {run.n} runs of {budget} evaluations "
                    f"({run.computed} run, {run.cached} cached, {run.seconds:.0f} s)"
                    + (f", {len(run.failures)} FAILED" if run.failures else "")
                )
    records, notes, finds = search_output(
        runs, failed, seeds, budget, sc.counts.f2_replays, sc.published_search_seeds is not None
    )
    sc.shared["search_violations"] = finds  # an empty list is an answer: the search ran, found none
    return StageOutput("search", "done", tuple(records), tuple(notes))


def _runs_with(
    runs: Mapping[tuple[str, str], Mapping[int, SearchOutcome]], regime: str, kind: str
) -> int:
    """Runs of either optimiser in ``regime`` with at least one violating evaluation of ``kind``."""
    return sum(
        1
        for algo in ("cma", "random")
        for outcome in runs[(regime, algo)].values()
        if outcome.of_kind(kind) > 0
    )


# ----------------------------------------------------------------------------- F2 (P11c)


def run_f2(sc: StageContext) -> StageOutput:
    module = importlib.import_module("closed_loop.eval.stage_f2")
    out: StageOutput = module.run(sc)
    return out


# ----------------------------------------------------------------------------- reachability


def run_reachability(sc: StageContext) -> StageOutput:
    """The floors of prereg section 6, the reachability table and the coverage simulation of
    ``eval.reachability`` (0003 A16), as ``reach.*`` records. Every one of them is made-up data:
    it says what the statistics can show at the registered sizes, nothing about the plant.

    The status is ``partial`` (so the run does not call itself complete) when a case that should
    reach its label does not, or one marked as unable to reach its label does."""
    from closed_loop.eval import reachability

    started = time.perf_counter()
    records = reachability.reach_records(quick=sc.mode == "quick")
    verdicts = [r for r in records if r.name.startswith("reach.verdict.")]
    wrong = [
        r.name.removeprefix("reach.verdict.")
        for r in verdicts
        if (r.value == 1.0) != bool(r.extra["reachable"])
    ]
    cannot = [r.name.removeprefix("reach.verdict.") for r in verdicts if not r.extra["reachable"]]
    notes = [
        f"reachability: {sum(1 for r in verdicts if r.value == 1.0)} of {len(verdicts)} cases give "
        "the label the registered rule names (made-up data)",
    ]
    if cannot:
        notes.append("cases the registered rules cannot satisfy: " + ", ".join(cannot))
    if wrong:
        notes.append("CASES THAT DO NOT BEHAVE AS RECORDED: " + ", ".join(wrong))
    sc.log(f"[reach] {len(verdicts)} cases, {len(wrong)} not as recorded")
    return StageOutput(
        "reach", "partial" if wrong else "done", tuple(records), tuple(notes),
        time.perf_counter() - started,
    )  # fmt: skip


# ----------------------------------------------------------------------------- conformance, DP


@dataclass(frozen=True)
class ConformanceTask:
    """A Careless-around-B3 episode with the shield off, returned as its trace.

    Not cached: a trace is a megabyte, and re-simulating 200 of them costs about a minute and a
    half of one core."""

    def label(self) -> str:
        return "conformance/H-C0"

    def fingerprint(self) -> dict[str, Any]:
        return {"kind": "conformance-trace", "arm": "H-C0"}

    def __call__(self, seed: int) -> Any:
        return simulate_episode(arm("H-C0"), seed, scenario_for("nominal")).trace


def run_conformance_dp(sc: StageContext) -> StageOutput:
    from closed_loop.planning.validation import run_validation
    from closed_loop.shield.conformance import conformance, result_records

    c = sc.counts
    seeds = sc.seeds[: c.conformance_episodes]
    run = sc.runner.run_seeds(
        ConformanceTask(), seeds, batch=c.batch, n_max=c.conformance_episodes, use_cache=False
    )
    kit = shield_kit(sc.context, "clause")
    report = conformance(
        run.values, kit.mapper, {"crusher": kit.crusher_model, "pumps": kit.pumps_model}
    )
    where = (
        f"{len(run.values)} F1 episodes of Careless around B3 with the shield off, nominal "
        f"condition, {sc.seed_role} seeds ({_span_text(run.seeds)}), games of the clause "
        f"specification"
    )
    records = result_records(report, measured_on=where, script="closed_loop.shield.conformance")
    notes = []
    if run.failures:
        notes.append(f"conformance: {len(run.failures)} episodes failed and are left out")

    t0 = time.perf_counter()
    val = run_validation(PLANNER_PARTICLES, n_beliefs=c.exact_dp_beliefs)
    where_dp = (
        f"belief planner with {PLANNER_PARTICLES} particles against exact dynamic programming on "
        f"the tabular instance, {val.n} random beliefs, horizon {val.horizon}"
    )
    summary = val.summary()
    for name, value, n, extra in (
        ("exact_dp.value_within_5pct", val.n_policy_within, val.n, {"k": val.n_policy_within}),
        ("exact_dp.action_agreement", val.n_agree, val.n_decisive, {"k": val.n_agree}),
        ("exact_dp.policy_error_mean", summary["policy_error_mean"], val.n, {}),
        ("exact_dp.policy_error_max", summary["policy_error_max"], val.n, {}),
        (
            "exact_dp.estimate_within_5pct",
            val.n_estimate_within,
            val.n,
            {"k": val.n_estimate_within},
        ),
    ):
        records.append(
            ResultRecord(
                name=name,
                value=float(value),
                n=int(n),
                fidelity=Fidelity.F0,
                measured_on=where_dp,
                script="closed_loop.planning.validation",
                extra={**extra, "lookahead_readings": val.lookahead_readings},
            )  # fmt: skip
        )
    sc.log(f"[conformance_dp] exact DP: {time.perf_counter() - t0:.0f} s")  # a run log line
    return StageOutput("conformance_dp", "done", tuple(records), tuple(notes))


# ----------------------------------------------------------------------------- the registry

STAGES: Final[tuple[Stage, ...]] = (
    # no row of the 0001 section 13 budget: made-up data, about a minute of CPU (0003 A16)
    Stage(
        "reach",
        "Reachability of the verdicts at their registered sizes (made-up data)",
        (),
        run_reachability,
    ),
    Stage("f0", "F0: proof counts, checker, differential, sabotage", (1,), run_f0),
    Stage("perception_open", "Open-loop perception", (2,), run_perception_open),
    Stage("perception_closed", "Closed-loop perception under B3", (3,), run_perception_closed),
    Stage("policy", "Policy arms and ablations", (4,), run_policy),
    Stage("hostile", "Hostile proposers, interlock and specification arms", (5,), run_hostile),
    Stage("causal", "Causal study: X-corr, X-scm, X-mis, M-DIAG", (6,), run_causal),
    Stage(
        "search",
        "Search: CMA-ES against random",
        (7,),
        run_search,
        needs=("closed_loop.search.falsify",),
        plug_in="packet P9b: closed_loop/search/falsify.py exporting falsify_detailed, classify "
        "and target_arm",
    ),
    Stage(
        "f2",
        "F2 arms and replay",
        (8, 9),
        run_f2,
        needs=("closed_loop.f2.episode", "closed_loop.f2.replay", "closed_loop.eval.stage_f2"),
        plug_in="packet P11c: closed_loop/f2/episode.py, replay.py and eval/stage_f2.py (run(sc))",
    ),
    Stage("conformance_dp", "Conformance and exact DP", (10,), run_conformance_dp),
)
STAGE_KEYS: Final[tuple[str, ...]] = tuple(s.key for s in STAGES)


@dataclass(frozen=True)
class PipelineResult:
    records: tuple[ResultRecord, ...]
    outputs: tuple[StageOutput, ...]
    complete: bool
    not_built: tuple[str, ...]


def run_pipeline(sc: StageContext, only: Sequence[str] | None = None) -> PipelineResult:
    """Run the stages in order (all of them, or ``only`` those named), then build the claim inputs.

    A stage whose ``needs`` are missing, or that raises ``NotBuiltError``, is recorded as ``not
    built`` and the run goes on. The result is *complete* only if every stage in the registry ran
    in full.
    """
    wanted = list(only) if only is not None else list(STAGE_KEYS)
    unknown = [k for k in wanted if k not in STAGE_KEYS]
    if unknown:
        raise ValueError(f"unknown stage {unknown}; the registry has {list(STAGE_KEYS)}")
    outputs: list[StageOutput] = []
    records: list[ResultRecord] = []
    for stage in STAGES:
        if stage.key not in wanted:
            continue
        started = time.perf_counter()
        sc.runner.check_headroom(f"stage {stage.key}")
        missing = missing_modules(stage)
        if missing:
            note = f"NOT BUILT: needs {', '.join(missing)} ({stage.plug_in})"
            sc.log(f"[{stage.key}] {note}")
            outputs.append(StageOutput(stage.key, "not built", (), (note,)))
            continue
        sc.log(f"[{stage.key}] {stage.title}")
        try:
            out = stage.body(sc)
        except NotBuiltError as exc:
            note = f"NOT BUILT: {exc}"
            sc.log(f"[{stage.key}] {note}")
            outputs.append(StageOutput(stage.key, "not built", (), (note,)))
            continue
        out = dataclasses.replace(out, seconds=time.perf_counter() - started)
        sc.log(f"[{stage.key}] done in {out.seconds:.0f} s, {len(out.records)} records")
        outputs.append(out)
        records.extend(out.records)
    from closed_loop.eval.assemble import claim_records  # assembly is not simulation (F3)

    records.extend(claim_records(sc))
    not_built = tuple(o.key for o in outputs if o.status == "not built")
    skipped = tuple(k for k in STAGE_KEYS if k not in wanted)
    complete = not not_built and not skipped and all(o.status == "done" for o in outputs)
    return PipelineResult(tuple(records), tuple(outputs), complete, not_built)


def failed_cells(store: CellStore) -> list[tuple[str, str, list[Failed]]]:
    """Cells that have failed episodes, for the report."""
    out = []
    for arm_name, condition in store.cells():
        run = store.run(arm_name, condition)
        if run is not None and run.failures:
            out.append((arm_name, condition, run.failures))
    return out


# ----------------------------------------------------------------------------- the whole run


@dataclass(frozen=True)
class BenchmarkResult:
    """Every record of a run (stages, claim inputs, run metadata, verdicts) and whether the run is
    complete."""

    records: tuple[ResultRecord, ...]
    pipeline: PipelineResult
    verdicts: tuple[Verdict, ...]

    @property
    def complete(self) -> bool:
        return self.pipeline.complete


def run_records(
    sc: StageContext,
    result: PipelineResult,
    *,
    prereg_sha: str | None,
    previous_headlines: Sequence[Mapping[str, Any]] = (),
    ledger_facts: Mapping[str, Any] | None = None,
) -> list[ResultRecord]:
    """The ``run/*`` records: parameters, completeness, stage status, run history.

    ``run/params`` carries what the verdicts need besides the claim records (the primary variant,
    the monitor mode, ``n_max`` of the hostile cells), so ``--check`` can rebuild them.
    """
    by_key = {s.key: s for s in STAGES}
    c = sc.counts
    out: list[ResultRecord] = [
        ResultRecord(
            name="run/params",
            value=float(len(sc.seeds)),
            n=len(sc.seeds),
            fidelity=Fidelity.F1,
            measured_on=f"parameters of this run: {sc.mode} mode, {sc.seed_role} seeds",
            script=SCRIPT,
            extra={
                "mode": sc.mode,
                "seed_role": sc.seed_role,
                "n_seeds": len(sc.seeds),
                "prereg_sha256": prereg_sha or "",
                "primary_variant": sc.primary_variant,
                "monitor_mode": sc.monitor_mode,
                "n_max_hostile": c.n_max_hostile,
                "n_max_planner": c.n_max_planner,
                "counts": json.dumps(dataclasses.asdict(c), sort_keys=True),
                "tuned": json.dumps(dataclasses.asdict(sc.tuned), sort_keys=True),
                "planner_particles": PLANNER_PARTICLES,
                "code_sha256": simulation_code_sha256(),
            },
        )
    ]
    for o in result.outputs:
        stage = by_key[o.key]
        out.append(
            ResultRecord(
                name=f"run/stage/{o.key}",
                value=0.0,
                n=len(o.records),
                fidelity=Fidelity.F1,
                measured_on=f"status of stage {o.key}: {stage.title}",
                script=SCRIPT,
                extra={
                    "rows": ",".join(map(str, stage.rows)),
                    "status": o.status,
                    "notes": "; ".join(o.notes),
                },
            )
        )
    partial = tuple(o.key for o in result.outputs if o.status == "partial")
    skipped = tuple(k for k in STAGE_KEYS if k not in {o.key for o in result.outputs})
    out.append(
        ResultRecord(
            name="run/completeness",
            value=1.0 if result.complete else 0.0,
            n=len(result.outputs),
            fidelity=Fidelity.F1,
            measured_on="whether every stage of the registry ran in full",
            script=SCRIPT,
            extra={
                "not_built": ",".join(result.not_built),
                "partial": ",".join(partial),
                "skipped": ",".join(skipped),
            },
        )
    )
    out.append(
        ResultRecord(
            name="run/history",
            value=float(len(previous_headlines)),
            n=len(previous_headlines),
            fidelity=Fidelity.F1,
            measured_on="full runs recorded in the ledger before this one",
            script=SCRIPT,
        )
    )
    if ledger_facts is not None:
        facts = {k: v for k, v in ledger_facts.items() if isinstance(v, str | int | float)}
        facts.update({k: int(v) for k, v in ledger_facts.items() if isinstance(v, bool)})
        out.append(
            ResultRecord(
                name="run/ledger",
                value=float(ledger_facts.get("heldout_evaluations", 0)),
                n=int(ledger_facts.get("run_number", 0)),
                fidelity=Fidelity.F1,
                measured_on=(
                    "what the ledger says about this run: held-out seeds handed out so far, the "
                    "number of this full run, and whether the running code is the frozen code"
                ),
                script=SCRIPT,
                extra=facts,
            )
        )
    for i, head in enumerate(previous_headlines):
        for key, val in sorted(head.items()):
            if key.startswith(("verdict/", "numbers/")):
                out.append(_headline_record(i, key, val))
    return out


def _headline_record(index: int, key: str, text: Any) -> ResultRecord:
    return ResultRecord(
        name=f"run/headline/{index}/{key}",
        value=0.0,
        n=0,
        fidelity=Fidelity.F1,
        measured_on=f"headline value {key} of full run {index + 1}",
        script=SCRIPT,
        extra={"text": scrub(str(text))},  # the ledger keeps the original (0003 A20)
    )


def finish(
    sc: StageContext,
    result: PipelineResult,
    *,
    prereg_sha: str | None,
    previous_headlines: Sequence[Mapping[str, Any]] = (),
    ledger_facts: Mapping[str, Any] | None = None,
) -> BenchmarkResult:
    """Add the run records and the verdicts to the stage records."""
    records = [
        *result.records,
        *run_records(
            sc,
            result,
            prereg_sha=prereg_sha,
            previous_headlines=previous_headlines,
            ledger_facts=ledger_facts,
        ),
    ]
    verdicts = verdicts_from_records(records)
    records.extend(verdict_records(verdicts))
    current = {f"verdict/{v.claim}": v.text for v in verdicts}
    current.update({f"numbers/{v.claim}": v.headline for v in verdicts})  # the numbers behind
    records.extend(
        _headline_record(len(previous_headlines), key, text)
        for key, text in sorted(current.items())
    )
    return BenchmarkResult(tuple(records), result, tuple(verdicts))


def f1_seed_plan(mode: str, counts: Counts, guard: Any | None) -> tuple[tuple[int, ...], str]:
    """The F1 seeds of a run: ``dev`` seeds for ``--quick``, else the guard's held-out ones."""
    n = counts.largest_cell
    if mode == "quick":
        return tuple(range(QUICK_FIRST_DEV_SEED, QUICK_FIRST_DEV_SEED + n)), "dev"
    if guard is None:
        raise ValueError("a full run needs the held-out guard")
    return tuple(guard.seeds(n)), "heldout"


def run_benchmark(
    *,
    mode: str,
    root: Path,
    workers: int,
    guard: Any | None,
    prereg_sha: str | None,
    only: Sequence[str] | None = None,
    previous_headlines: Sequence[Mapping[str, Any]] = (),
    ledger_facts: Mapping[str, Any] | None = None,
    headroom: Callable[[], float | None] | None = None,
    limit: float | None = None,
    log: Callable[[str], None] = print,
    context: PreparedContext | None = None,
    counts: Counts | None = None,
    cache_dir: Path | None = None,
) -> BenchmarkResult:
    """Fit the context, run the stages, assemble the records. Writes nothing but the cache."""
    from closed_loop.audit.guard import COMMIT_LIMIT_PCT, commit_charge_percent
    from closed_loop.eval.arms import MONITOR_MODE, PRIMARY_VARIANT
    from closed_loop.eval.context import prepare_context
    from closed_loop.eval.tuned import current as current_tuned

    if mode not in ("quick", "full"):
        raise ValueError(f"mode must be quick or full, got {mode!r}")
    if mode == "full" and only is not None:
        raise ValueError("a full run runs every stage; --stages is for --quick")
    counts = counts if counts is not None else (QUICK if mode == "quick" else FULL)
    tuned = current_tuned()
    seeds, role = f1_seed_plan(mode, counts, guard)
    published = (
        None if mode == "quick" or guard is None else tuple(guard.search_seeds(counts.search_runs))
    )
    published_f2 = (
        None if mode == "quick" or guard is None else tuple(guard.f2_seeds(counts.f2_episodes))
    )
    if ledger_facts is not None and guard is not None:  # the count after this run drew its seeds
        ledger_facts = {**ledger_facts, "heldout_evaluations": guard.ledger.heldout_evaluations()}
    started = time.perf_counter()
    ctx = context if context is not None else prepare_context()
    log(f"context ready in {time.perf_counter() - started:.0f} s; {len(seeds)} {role} seeds")
    cache = cache_dir or (root / "benchmarks" / ".cache" / ("quick" if mode == "quick" else "full"))
    with Runner(
        workers=workers,
        context=ctx,
        cache_dir=cache,
        tuned=tuned,
        headroom=headroom if headroom is not None else commit_charge_percent,
        limit=limit if limit is not None else COMMIT_LIMIT_PCT,
        timeout_s=counts.episode_timeout_s,
        log=log,
    ) as runner:
        sc = StageContext(
            mode=mode,
            counts=counts,
            runner=runner,
            context=ctx,
            seeds=seeds,
            seed_role=role,
            tuned=tuned,
            primary_variant=PRIMARY_VARIANT,
            monitor_mode=MONITOR_MODE,
            root=root,
            log=log,
            prereg_sha=prereg_sha,
            published_search_seeds=published,
            published_f2_seeds=published_f2,
        )
        result = run_pipeline(sc, only)
        return finish(
            sc,
            result,
            prereg_sha=prereg_sha,
            previous_headlines=previous_headlines,
            ledger_facts=ledger_facts,
        )
