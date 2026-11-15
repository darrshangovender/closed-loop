"""Can each verdict be reached at its registered size? (packet P12c, ADR 0003 A16)

A15 tested every verdict function on a perfect input, but it fed the function finished intervals,
so it could not see that a betting sequence has a floor set by the range and not by the spread.
This module closes that gap. For each claim it builds per-seed values at the registered count
(``Counts``) from the spreads stated in ``SPREADS``, passes them through the code the benchmark
uses (``assemble.claim_records`` for the cell claims, ``queries.paired_difference`` and
``mdiag.records`` for the causal ones, ``stages.search_output`` for C-SEARCH) and reads the verdict
off ``claims.verdicts_from_records``. Nothing here comes from the plant: **every number is made-up
data** that tests the statistics and the rules, and the records it writes say so.

A row of the table is a *case*: a claim, a leg, an input and the label the registered rule should
give it.

* ``perfect``: the best the component could honestly do, not an unreachable ideal. A perception
  variant is on its 0.95 target against a split-conformal V1 at 0.88; a monitor never false-alarms
  and always detects; a planner costs what the cheapest honest plan costs. It should **win**.
* ``useless``: the component is no better than its baseline, or worse by the margin the loss rule
  names. Where the rule's loss condition is "the baseline is better" (C-CAUSE, C-DIAG, C-PLAN,
  C-SEARCH) the input is a component that is clearly worse, since a bound that merely fails to
  show a benefit is "inconclusive" by design (A15). It should **lose**.

A case the registered rules cannot satisfy is marked ``reachable=False`` with the arithmetic in
``why``; the tests mark it ``xfail(strict=True)`` so that the day the numbers change it turns up.
Rules are never moved to make a case pass.

Spreads. ``SPREADS`` are per-episode values read on the 40 development episodes of the tuning
cache written by ``scripts/tune.py`` on 2 November 2026, after the baselines stopped stalling
and the plant stopped stopping (0003 A17 and A18; seeds 1100 to 1139, ``benchmarks/.cache/tune``;
read-only here). Where
nothing was measured the value is an assumption and says so. Each case takes a ``scale`` that
multiplies every spread, so that ``break_even`` can say how much noisier the plant may be than
the stated spreads before the registered label is lost.

The coverage simulation (``coverage_sim``) is a port of the throw-away script that chose the
kurtosis guard of ``stats.confseq.paired_cs``: a time-uniform check of the spread-scaled sequence on
lumpy and rare-event shapes.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
from numpy.typing import NDArray

from closed_loop.causal import mdiag
from closed_loop.causal.queries import PerLifeScores, QueryConfig, paired_difference
from closed_loop.core.results import ResultRecord
from closed_loop.core.types import Fidelity
from closed_loop.eval.arms import MONITOR_MODE, PRIMARY_VARIANT
from closed_loop.eval.assemble import claim_records
from closed_loop.eval.claims import (
    INCONCLUSIVE,
    LOSS,
    WIN,
    Verdict,
    c_cause,
    c_diag,
    est_of,
    verdicts_from_records,
)
from closed_loop.eval.counts import ALPHA_PERC, ALPHA_PRIMARY, FULL, Counts
from closed_loop.eval.episode import EpisodeResult
from closed_loop.eval.metrics import Attribution, EpisodeCoverage
from closed_loop.eval.runner import CellRun
from closed_loop.eval.stages import (
    CellStore,
    SearchOutcome,
    StageContext,
    search_output,
)
from closed_loop.eval.tuned import PROVISIONAL
from closed_loop.stats import wilson
from closed_loop.stats.confseq import (
    SPREAD_KURTOSIS_DIVISOR,
    SPREAD_MIN_PAIRS,
    SPREAD_R,
    _spread_halfwidth,
    paired_betting_cs,
)

__all__ = [
    "CASES",
    "COVERAGE_CASES",
    "FLOOR_SIZES",
    "SPREADS",
    "Case",
    "Floor",
    "Row",
    "Spreads",
    "SimRow",
    "break_even",
    "label_at",
    "clipped_normal_latent",
    "coverage_sim",
    "floors",
    "reach_records",
    "reachability_rows",
    "spread_halfwidths",
]

SCRIPT: Final[str] = "closed_loop.eval.reachability"
MADE_UP: Final[str] = "MADE-UP DATA, not a plant result"
SEED: Final[int] = 20261101
LIVES_PER_SEED: Final[int] = 4  # causal.twin: four assets, one life each, for every seed
ALERT_SHARE: Final[float] = 0.6  # counts.QUICK_FLOORS: "about 60% of lives" raise an alert
FLOOR_SIZES: Final[tuple[int, ...]] = (100, 300, 500, 700)


# ----------------------------------------------------------------------------- spreads


@dataclass(frozen=True)
class Spreads:
    """The spreads the made-up data are built from. See the module docstring for where they come
    from; ``scaled`` multiplies every standard deviation, leaving the correlations alone."""

    # c_e^lo of one episode, B3 under a perception variant: sd 0.0625 nominal and 0.074 to 0.085
    # shifted on the 40 dev episodes (kurtosis 2.7 to 5.2, 40% of episodes at exactly 1)
    cov_lo_sd: float = 0.07
    # the same seed under two variants: dev correlation 0.86 between P0 and B3 on c_e^lo
    cov_rho: float = 0.8
    # c_e (the pooled coverage Gap_B is read on): dev sd 0.081 to 0.089. The open-loop coverage of
    # a seed is not the closed-loop coverage of that seed, so the two are taken as independent and
    # their difference has sd gap_b_sd (an assumption: only the closed-loop side was measured)
    gap_b_sd: float = 0.11
    # the same for the trend-estimated stratum, which has few checkpoints per episode: the
    # closed-loop side was measured at sd 0.196 for B3 on 39 dev episodes (0.237 before the
    # baselines stopped stalling), the open-loop side is assumed at 0.10, so the difference has sd
    # sqrt(0.196^2 + 0.10^2) = 0.22
    gap_b_trend_sd: float = 0.22
    # M-COST of B3 with k = 0.25 (dev, after 0003 A18): 5.33 cost units, sd 0.89, max 6.14, sample
    # kurtosis 4.1 (before A17: 8.8, sd 17.8, from one episode that stalled for 59 days). Jobs 3.78,
    # sd 0.48; the lost throughput is 78 quarters an episode (1.56 units at 0.02 a quarter, sd
    # 0.47). More jobs go with more lost quarters, so the total sd is 0.89 and not the 0.67 of two
    # independent parts: the model gives the lost part the sd (0.75) that makes the total right.
    # The planner's own cost is 4.16 units, sd 3.12, kurtosis 14.5 (2 of 40 episodes fail) and the
    # paired P0 minus B3 difference has mean -1.18 units, sd 3.08, kurtosis 14.9
    b3_jobs: float = 3.775
    b3_jobs_sd: float = 0.48
    b3_lost: float = 1.555
    b3_lost_sd: float = 0.75
    # M-CF Q1 per life, scaled: the SCM near the one-day floor, the correlational model near the cap
    q1_scm_sd: float = 0.10
    q1_corr_sd: float = 0.15
    # M-DIAG per life: a life has one or two alerts (1,895 on 1,200 dev lives)
    diag_alerts_two: float = 0.6

    def scaled(self, k: float) -> Spreads:
        return replace(
            self,
            cov_lo_sd=self.cov_lo_sd * k,
            gap_b_sd=self.gap_b_sd * k,
            gap_b_trend_sd=self.gap_b_trend_sd * k,
            b3_jobs_sd=self.b3_jobs_sd * k,
            b3_lost_sd=self.b3_lost_sd * k,
            q1_scm_sd=self.q1_scm_sd * k,
            q1_corr_sd=self.q1_corr_sd * k,
        )


SPREADS: Final[Spreads] = Spreads()


@functools.lru_cache(maxsize=256)
def clipped_normal_latent(mean: float, sd: float) -> tuple[float, float]:
    """``(mu, sigma)`` of a normal that, clipped to [0, 1], has the given mean and sd.

    Per-episode coverage sits near 1 and is clipped there (40% of dev episodes are exactly 1), so
    a plain normal with the same mean and sd would put mass above 1.
    """
    from scipy.optimize import fsolve  # type: ignore[import-untyped]  # heavy import
    from scipy.special import ndtri  # type: ignore[import-untyped]  # heavy import

    z = ndtri((np.arange(4000) + 0.5) / 4000.0)

    def residual(p: NDArray[np.float64]) -> list[float]:
        x = np.clip(p[0] + p[1] * z, 0.0, 1.0)
        return [float(x.mean()) - mean, float(x.std()) - sd]

    sol, _info, ok, _msg = fsolve(residual, [mean, sd], full_output=True)
    if ok != 1 or max(abs(r) for r in residual(sol)) > 1e-6:
        raise ValueError(f"no clipped normal has mean {mean} and sd {sd}")
    return float(sol[0]), float(sol[1])


def _bounded(rng: np.random.Generator, n: int, mean: float, sd: float) -> NDArray[np.float64]:
    mu, sigma = clipped_normal_latent(round(mean, 6), round(sd, 6))
    return np.clip(mu + sigma * rng.standard_normal(n), 0.0, 1.0)


def _bounded_pair(
    rng: np.random.Generator, n: int, mean_a: float, mean_b: float, sd: float, rho: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Two [0, 1] values per seed with the given means and sd and (latent) correlation ``rho``."""
    mu_a, s_a = clipped_normal_latent(round(mean_a, 6), round(sd, 6))
    mu_b, s_b = clipped_normal_latent(round(mean_b, 6), round(sd, 6))
    z1 = rng.standard_normal(n)
    z2 = rho * z1 + math.sqrt(1.0 - rho * rho) * rng.standard_normal(n)
    return np.clip(mu_a + s_a * z1, 0.0, 1.0), np.clip(mu_b + s_b * z2, 0.0, 1.0)


# ----------------------------------------------------------------------------- the floors


@dataclass(frozen=True)
class Floor:
    """Half-widths at ``n`` pairs at the level ``alpha``, for a paired difference in [-1, 1]."""

    n: int
    alpha: float
    betting: float  # whatever the data (every difference exactly 0)
    spread_factor: float  # the spread-scaled half-width is this times the sample sd


def spread_halfwidths(
    sd: NDArray[np.float64] | float, n: int | NDArray[np.int64], alpha: float
) -> NDArray[np.float64]:
    """``sd * sqrt(2 (n r + 1) / (n^2 r) * log(sqrt(n r + 1) / alpha))``, vectorised over ``sd``.

    The array form of ``stats.confseq._spread_halfwidth`` (a test holds them together)."""
    r = SPREAD_R
    t = np.asarray(n, dtype=np.float64)
    return np.asarray(sd) * np.sqrt(
        2.0 * (t * r + 1.0) / (t * t * r) * np.log(np.sqrt(t * r + 1.0) / alpha)
    )


def floors(alpha: float = ALPHA_PRIMARY, sizes: Sequence[int] = FLOOR_SIZES) -> list[Floor]:
    """The numbers of prereg section 6: the betting floor and the spread factor at each size."""
    out = []
    for n in sizes:
        zero = np.zeros(n)
        lo, hi = paired_betting_cs(zero, zero, 0.0, 1.0, alpha)
        out.append(Floor(n, alpha, (hi - lo) / 2.0, _spread_halfwidth(n, 1.0, alpha)))
    return out


# ----------------------------------------------------------------------------- made-up episodes

_PROPS: Final[tuple[str, ...]] = ("P1", "P2", "P3", "P4", "P5", "P6")
_MODES: Final[tuple[str, ...]] = ("opt", "pess", "inf", "proxy")


def _episode(
    arm: str,
    seed: int,
    condition: str,
    *,
    cost: float = 20.0,
    failures: int = 0,
    throughput: float = 0.9,
    coverage: EpisodeCoverage | None = None,
    violates: bool = False,
    defect: bool = False,
    monitors: Mapping[str, float | None] | None = None,
    shift_day: float | None = None,
) -> EpisodeResult:
    mins = dict.fromkeys(_PROPS, 1.0)
    attributions: tuple[Attribution, ...] = ()
    if violates or defect:
        mins["P2"] = -1.0
    if defect:
        attributions = (Attribution("P2", 10, "assumption_held"),)
    return EpisodeResult(
        arm=arm,
        seed=seed,
        condition=condition,
        fidelity=Fidelity.F1,
        violations={p: int(mins[p] < 0) for p in _PROPS},
        min_robustness=mins,
        throughput=throughput,
        cost=cost,
        unplanned_failures=failures,
        planned_jobs=2,
        shield_overrides=0,
        shield_admitted=0,
        monitor_fired_day=None,
        labels=(),
        wall_s=0.0,
        attributions=attributions,
        commands_proposed=100,
        coverage=coverage if coverage is not None else _coverage(0.95, 0.95),
        monitors=dict(monitors or {}),
        shift_day=shift_day,
    )


def _cell(arm: str, condition: str, results: Sequence[EpisodeResult]) -> CellRun:
    return CellRun(
        label=f"{arm}/{condition}",
        seeds=tuple(r.seed for r in results),
        values=list(results),
        failures=[],
        n_max=len(results),
        stopped_by="n_max",
        halfwidth=None,
        computed=len(results),
        cached=0,
    )


def _coverage(c: float, c_lo: float, c_trend: float | None = None) -> EpisodeCoverage:
    t = c if c_trend is None else c_trend
    return EpisodeCoverage(c, c_lo, 50, t, t, 20, 5, 55)


def _context(
    store: CellStore, shared: dict[str, Any] | None = None, *, primary: str = PRIMARY_VARIANT
) -> StageContext:
    return StageContext(
        mode="quick",
        counts=FULL,
        runner=cast(Any, None),  # the claim records read neither the pool nor the context
        context=cast(Any, None),
        seeds=(),
        seed_role="made-up",
        tuned=PROVISIONAL,
        primary_variant=primary,
        monitor_mode=MONITOR_MODE,
        root=Path("."),
        store=store,
        shared=shared or {},
    )


def _params(n_max_hostile: int, primary: str = PRIMARY_VARIANT) -> ResultRecord:
    return ResultRecord(
        name="run/params",
        value=0.0,
        n=0,
        fidelity=Fidelity.F1,
        measured_on=MADE_UP,
        script=SCRIPT,
        extra={
            "primary_variant": primary,
            "monitor_mode": MONITOR_MODE,
            "n_max_hostile": n_max_hostile,
        },
    )


def _verdicts(
    sc: StageContext, extra: Sequence[ResultRecord] = (), *, n_max_hostile: int = 500
) -> dict[str, Verdict]:
    """The benchmark's own path: the claim-level records of the cells, then the verdicts."""
    records = [*claim_records(sc), *extra, _params(n_max_hostile, sc.primary_variant)]
    return {v.claim: v for v in verdicts_from_records(records)}


def _seeds(n: int) -> range:
    return range(1100, 1100 + n)


# ----------------------------------------------------------------------------- the scenarios
# Each returns the verdicts of one pass through the benchmark's code; ``scale`` multiplies every
# spread of ``SPREADS``.


def _perc(gap_v1: float, gap_primary: float, scale: float, counts: Counts) -> dict[str, Verdict]:
    """C-PERC: Gap_B per seed, open loop minus closed loop, for V1 and the primary variant."""
    sp = SPREADS.scaled(scale)
    n = counts.open_loop_seeds
    rng = np.random.default_rng([SEED, 1])
    store = CellStore()
    open_loop: dict[tuple[str, str], dict[int, EpisodeCoverage]] = {}
    for variant, gap in (("V1", gap_v1), (PRIMARY_VARIANT, gap_primary)):
        # independent open- and closed-loop values (closed loop at 0.85, so that a gap of 0.10
        # leaves the open loop below 1), so that the difference has sd gap_b_sd
        sd_p, sd_t = sp.gap_b_sd / math.sqrt(2.0), sp.gap_b_trend_sd / math.sqrt(2.0)
        closed_p, closed_t = _bounded(rng, n, 0.85, sd_p), _bounded(rng, n, 0.85, sd_t)
        open_p, open_t = _bounded(rng, n, 0.85 + gap, sd_p), _bounded(rng, n, 0.85 + gap, sd_t)
        results = [
            _episode(f"B3-{variant}", s, "nominal", coverage=_coverage(cp, cp, ct))
            for s, cp, ct in zip(_seeds(n), closed_p, closed_t, strict=True)
        ]
        store.put(f"B3-{variant}", "nominal", _cell(f"B3-{variant}", "nominal", results))
        open_loop[("nominal", variant)] = {
            s: _coverage(op, op, ot) for s, op, ot in zip(_seeds(n), open_p, open_t, strict=True)
        }
    return _verdicts(_context(store, {"open_loop": open_loop}))


def _adapt(
    shifted: float, v1_shifted: float, nominal: float, scale: float, counts: Counts
) -> dict[str, Verdict]:
    """C-ADAPT: the primary variant against V1 under ``shifted``, and its coverage in both."""
    sp = SPREADS.scaled(scale)
    n = counts.n_max_perception
    rng = np.random.default_rng([SEED, 2])
    var_s, v1_s = _bounded_pair(rng, n, shifted, v1_shifted, sp.cov_lo_sd, SPREADS.cov_rho)
    var_n = _bounded(rng, n, nominal, sp.cov_lo_sd)
    store = CellStore()
    name = f"B3-{PRIMARY_VARIANT}"
    for arm, cond, vals in (
        ("B3-V1", "shifted", v1_s),
        (name, "shifted", var_s),
        (name, "nominal", var_n),
    ):
        results = [
            _episode(arm, s, cond, coverage=_coverage(v, v))
            for s, v in zip(_seeds(n), vals, strict=True)
        ]
        store.put(arm, cond, _cell(arm, cond, results))
    return _verdicts(_context(store))


def _b3_cost(rng: np.random.Generator, n: int, sp: Spreads) -> NDArray[np.float64]:
    """B3's cost per episode in cost units: planned jobs plus the cost of the throughput lost."""
    jobs = np.maximum(rng.normal(sp.b3_jobs, sp.b3_jobs_sd, n), 1.0)  # a cost, not a job count
    lost = np.maximum(rng.normal(sp.b3_lost, sp.b3_lost_sd, n), 0.0)
    return np.asarray(jobs + lost, dtype=np.float64)


def _plan(
    p_cost: float | None,
    p_fail: float,
    b3_fail: float,
    scale: float,
    counts: Counts,
    *,
    p_cost_sd: float = 0.0,
) -> dict[str, Verdict]:
    """C-PLAN, P0 against B3 (nominal): ``p_cost`` units flat (``None``: far below B3), episodes
    with a failure in the share ``p_fail`` for the planner and ``b3_fail`` for B3."""
    sp = SPREADS.scaled(scale)
    n = counts.n_max_planner
    rng = np.random.default_rng([SEED, 3])
    b3_cost = _b3_cost(rng, n, sp)
    if p_cost is None:
        p_costs = np.full(n, 0.5)
        b3_cost = (
            b3_cost + 100.0
        )  # a cost leg that is cleared by a mile: this case reads the safety leg
    else:
        p_costs = np.maximum(p_cost + p_cost_sd * rng.standard_normal(n), 0.5)
    p_f = (rng.random(n) < p_fail).astype(int)
    b_f = (rng.random(n) < b3_fail).astype(int)
    store = CellStore()
    for arm, costs, fails in ((f"B3-{PRIMARY_VARIANT}", b3_cost, b_f), ("P0", p_costs, p_f)):
        results = [
            _episode(arm, s, "nominal", cost=float(c), failures=int(f))
            for s, c, f in zip(_seeds(n), costs, fails, strict=True)
        ]
        store.put(arm, "nominal", _cell(arm, "nominal", results))
    return _verdicts(_context(store))


def _mon(false_alarms: int, detected: int, scale: float, counts: Counts) -> dict[str, Verdict]:
    """C-MON for the chosen mode on P1: ``false_alarms`` of the unshifted episodes fire,
    ``detected`` of the shifted ones fire within 30 days of the shift."""
    nominal_n, shifted_n = counts.mon_nominal, counts.mon_shifted
    store = CellStore()
    nominal = [
        _episode(
            "P1", s, "nominal", monitors=dict.fromkeys(_MODES, 50.0 if i < false_alarms else None)
        )
        for i, s in enumerate(_seeds(nominal_n))
    ]
    shifted = [
        _episode(
            "P1",
            s,
            "shifted",
            shift_day=45.0,
            monitors=dict.fromkeys(_MODES, 55.0 if i < detected else None),
        )
        for i, s in enumerate(_seeds(shifted_n))
    ]
    store.put("P1", "nominal", _cell("P1", "nominal", nominal))
    store.put("P1", "shifted", _cell("P1", "shifted", shifted))
    return _verdicts(_context(store))


def _shield(
    unshielded_rate: float, defects: int, throughput_ratio: float, scale: float, counts: Counts
) -> dict[str, Verdict]:
    """C-SHIELD: the hostile arms at ``n_max_hostile``; the shielded arms have no violation of
    P1-P3 (and ``defects`` episodes with a defect in each), the unshielded ones violate at
    ``unshielded_rate``."""
    n = counts.n_max_hostile
    rng = np.random.default_rng([SEED, 4])
    store = CellStore()
    for pair in ("H-C", "H-G"):
        for arm, shielded in ((f"{pair}0", False), (f"{pair}1", True)):
            viol = rng.random(n) < unshielded_rate if not shielded else np.zeros(n, dtype=bool)
            results = [
                _episode(
                    arm,
                    s,
                    "nominal",
                    violates=bool(v),
                    defect=shielded and i < defects,
                    throughput=0.99 if not shielded else 0.99 * throughput_ratio,
                )
                for i, (s, v) in enumerate(zip(_seeds(n), viol, strict=True))
            ]
            store.put(arm, "nominal", _cell(arm, "nominal", results))
    return _verdicts(_context(store), n_max_hostile=n)


def _lives(
    rng: np.random.Generator, n_lives: int, p_scm: float, p_corr: float, two: float
) -> tuple[list[mdiag.AlertCase], dict[str, mdiag.Outcome]]:
    """``n_lives`` lives with one or two alerts each; the alert is diagnosed right by the SCM with
    probability ``p_scm`` and by the correlational model with probability ``p_corr``."""
    cases: list[mdiag.AlertCase] = []
    for life in range(n_lives):
        for k in range(1 + int(rng.random() < two)):
            cases.append(
                mdiag.AlertCase(
                    seed=1000 + life,
                    asset_id="crusher",
                    day=30 + k,
                    level=1 + k,
                    truth_cause="degradation",
                    truth_mode="bearing",
                    history=cast(Any, None),  # records() reads the life and nothing else
                    life=life,
                )
            )
    outcomes = {}
    for name, p in (("scm", p_scm), ("corr", p_corr)):
        ok = rng.random(len(cases)) < p
        text = np.array(["x"] * len(cases))
        outcomes[name] = mdiag.Outcome(name, text, text, ok, ok, ok)
    return cases, outcomes


def _diag(p_scm: float, p_corr: float, scale: float, counts: Counts) -> Verdict:
    n_lives = round(counts.mdiag_test_seeds * LIVES_PER_SEED * ALERT_SHARE)
    rng = np.random.default_rng([SEED, 5])
    cases, outcomes = _lives(rng, n_lives, p_scm, p_corr, SPREADS.diag_alerts_two)
    rec = mdiag.records("reversed", cases, outcomes, alpha=ALPHA_PRIMARY, measured_on=MADE_UP)
    return c_diag(est_of(rec["scm-corr"]))


def _cause(scm: float, corr: float, scale: float, counts: Counts) -> Verdict:
    sp = SPREADS.scaled(scale)
    n = round(counts.causal_test_seeds * LIVES_PER_SEED * ALERT_SHARE)
    rng = np.random.default_rng([SEED, 6])
    a = _bounded(rng, n, scm, sp.q1_scm_sd)
    b = _bounded(rng, n, corr, sp.q1_corr_sd)

    def scores(q1: NDArray[np.float64]) -> PerLifeScores:
        z = np.zeros(n)
        return PerLifeScores(
            z,
            q1,
            np.zeros((n, 2)),
            np.zeros((n, 2)),
            z,
            z,
            z,
            np.zeros(n, dtype=bool),
            n,
            QueryConfig(),
        )

    rec = paired_difference(
        scores(a),
        scores(b),
        "q1",
        a_name="X-scm",
        b_name="X-corr",
        alpha=ALPHA_PRIMARY,
        measured_on=MADE_UP,
    )
    return c_cause(est_of(rec))


def _search(
    cma_first: Callable[[int], int | None],
    rand_first: Callable[[int], int | None],
    scale: float,
    counts: Counts,
) -> Verdict:
    """C-SEARCH: ``cma_first(i)`` and ``rand_first(i)`` are the evaluations to the first violation
    of run ``i`` in R_out (None: none found); R_in finds nothing."""
    budget = counts.search_budget
    seeds = list(range(3000, 3000 + counts.search_runs))

    def run(first: Callable[[int], int | None]) -> dict[int, SearchOutcome]:
        out = {}
        for i, s in enumerate(seeds):
            f = first(i)
            out[s] = SearchOutcome(f, budget, int(f is not None))
        return out

    blank = {s: SearchOutcome(None, budget, 0) for s in seeds}
    runs = {
        ("in", "cma"): blank,
        ("in", "random"): blank,
        ("out", "cma"): run(cma_first),
        ("out", "random"): run(rand_first),
    }
    records, _notes, _finds = search_output(
        runs, {k: () for k in runs}, seeds, budget, counts.f2_replays, True
    )
    return {v.claim: v for v in verdicts_from_records([*records, _params(counts.n_max_hostile)])}[
        "C-SEARCH"
    ]


# ----------------------------------------------------------------------------- the table


@dataclass(frozen=True)
class Case:
    """One row of the table: an input, the label the registered rule should give it, and the
    function that gives the verdict (``scale`` multiplies every spread)."""

    claim: str  # the verdict's name, "C-PLAN.P0"
    leg: str  # which leg or stratum the case is about; "" if the claim has one
    kind: str  # "perfect" | "useless" | "null"
    what: str  # the input, in words
    want: str  # the label the registered rule gives it, if it can
    n: int  # pairs, episodes or runs behind the verdict
    run: Callable[[float], Verdict]
    bound: Callable[[Verdict], str]
    reachable: bool = True
    why: str = ""  # for a case that cannot reach ``want``: the arithmetic

    @property
    def id(self) -> str:
        return f"{self.claim}{'/' + self.leg if self.leg else ''}/{self.kind}"


@dataclass(frozen=True)
class Row:
    case: Case
    label: str
    bound: str

    @property
    def reached(self) -> bool:
        return self.label == self.case.want


def _pick(
    builder: Callable[[float], Mapping[str, Verdict]], key: str
) -> Callable[[float], Verdict]:
    return lambda scale: builder(scale)[key]


def _interval(prefix: str) -> Callable[[Verdict], str]:
    def fmt(v: Verdict) -> str:
        n = v.numbers
        return f"{prefix} [{float(n[f'{prefix}.lo']):+.4f}, {float(n[f'{prefix}.hi']):+.4f}]"

    return fmt


def _two(a: str, b: str) -> Callable[[Verdict], str]:
    return lambda v: f"{_interval(a)(v)}; {_interval(b)(v)}"


def _search_bound(v: Verdict) -> str:
    n = v.numbers
    return (
        f"CMA-ES takes {n['pairs.cma']}, random {n['pairs.random']}; "
        f"share [{float(n['share.lo']):.3f}, {float(n['share.hi']):.3f}]"
    )


CHEAPEST_PLAN_UNITS: Final[float] = 2.0  # two planned jobs, no failure, no lost throughput


def _b3_cost_moments(sp: Spreads) -> tuple[float, float]:
    """Mean and sd, in cost units, of the B3 cost model of ``_b3_cost``."""
    mean = sp.b3_jobs + sp.b3_lost
    return mean, math.sqrt(sp.b3_jobs_sd**2 + sp.b3_lost_sd**2)


def _perc_trend_why(counts: Counts) -> str:
    n = counts.open_loop_seeds
    factor = _spread_halfwidth(n, 1.0, ALPHA_PERC)
    half = factor * SPREADS.gap_b_trend_sd
    return (
        f"a loss needs the Gap_B intervals of V1 and of the primary variant inside +-0.03. The "
        f"trend stratum has few checkpoints an episode and the closed-loop coverage of B3 has sd "
        f"0.196 on dev, so the per-seed difference has sd {SPREADS.gap_b_trend_sd:.2f}; at n = {n} "
        f"and level 0.05 / 18 the spread-scaled half-width is {factor:.4f} x s = {half:.3f}, "
        f"beyond 0.03 even for a gap of exactly 0 (it would need s at most {0.03 / factor:.3f}). "
        f"The pooled stratum (s = {SPREADS.gap_b_sd:.2f}, half-width "
        f"{factor * SPREADS.gap_b_sd:.3f}) does reach it"
    )


def _plan_cost_why(counts: Counts) -> str:
    n = counts.n_max_planner
    mean, sd = _b3_cost_moments(SPREADS)
    betting = floors(sizes=(n,))[0].betting * 200.0
    factor = _spread_halfwidth(n, 1.0, ALPHA_PRIMARY)
    need_spread = 4.0 + factor * sd
    return (
        f"the cost margin is 0.02 of the cap (4 units). B3, now that it no longer stalls (0003 "
        f"A17, A18), costs {mean:.2f} units on dev, sd {sd:.2f}, so a planner can save at most "
        f"{mean - CHEAPEST_PLAN_UNITS:.2f} units even if it costs only {CHEAPEST_PLAN_UNITS:g} "
        f"(two planned jobs, nothing else), while a win needs more than 4 units plus the "
        f"half-width: "
        f"{need_spread:.2f} on the spread-scaled bound ({factor:.3f} x {sd:.2f} = "
        f"{factor * sd:.2f} units). The planner's own cost on dev is 4.16 units, so its saving is "
        f"{mean - 4.16:.2f}, and its cost has a tail (2 of 40 episodes fail; the paired difference "
        f"has sample kurtosis 14.9 against the guard's limit of "
        f"{n / SPREAD_KURTOSIS_DIVISOR:.0f} at n = {n}), which sends the bound to the betting "
        f"sequence: {betting:.0f} units whatever the data, so {4 + betting:.0f} units are needed"
    )


def _build_cases(counts: Counts = FULL) -> tuple[Case, ...]:
    c = counts
    n_cause = round(c.causal_test_seeds * LIVES_PER_SEED * ALERT_SHARE)
    n_diag = round(c.mdiag_test_seeds * LIVES_PER_SEED * ALERT_SHARE)
    cases: list[Case] = []

    def add(*args: Any, **kw: Any) -> None:
        cases.append(Case(*args, **kw))

    # ---- C-PERC: two verdicts at ALPHA_PERC; the thesis is that open-loop calibration does not
    # carry over, so "perfect" is a gap well beyond the band and "useless" is no gap
    for stratum in ("pooled", "trend"):
        key = f"C-PERC.{stratum}"
        win = _pick(lambda s: _perc(0.10, 0.10, s, c), key)
        none = _pick(lambda s: _perc(0.0, 0.0, s, c), key)
        gap = _two("gap_v1", "gap_primary")
        add(
            key,
            stratum,
            "perfect",
            "Gap_B = +0.10 for V1 and the primary variant",
            WIN,
            c.open_loop_seeds,
            win,
            gap,
        )
        add(
            key,
            stratum,
            "useless",
            "Gap_B = 0 for V1 and the primary variant",
            LOSS,
            c.open_loop_seeds,
            none,
            gap,
            reachable=stratum != "trend",
            why=_perc_trend_why(c) if stratum == "trend" else "",
        )

    # ---- C-ADAPT
    a_bound = _two("diff_shifted", "cov_shifted")
    add(
        "C-ADAPT",
        "",
        "perfect",
        "variant at its 0.95 target in both conditions, V1 at 0.88 under shifted",
        WIN,
        c.n_max_perception,
        _pick(lambda s: _adapt(0.95, 0.88, 0.95, s, c), "C-ADAPT"),
        a_bound,
    )
    add(
        "C-ADAPT",
        "",
        "useless",
        "variant no better than V1 (both 0.88 under shifted)",
        LOSS,
        c.n_max_perception,
        _pick(lambda s: _adapt(0.88, 0.88, 0.95, s, c), "C-ADAPT"),
        a_bound,
    )
    add(
        "C-ADAPT",
        "nominal",
        "useless",
        "variant at 0.95 against V1 at 0.88, but 0.85 under nominal",
        LOSS,
        c.n_max_perception,
        _pick(lambda s: _adapt(0.95, 0.88, 0.85, s, c), "C-ADAPT"),
        _interval("cov_nominal"),
    )

    # ---- C-CAUSE and C-DIAG: per-life values
    add(
        "C-CAUSE",
        "",
        "perfect",
        "SCM at 0.03 of the cap (about a day), correlational at 0.93",
        WIN,
        n_cause,
        lambda s: _cause(0.03, 0.93, s, c),
        _interval("q1_diff"),
    )
    add(
        "C-CAUSE",
        "",
        "useless",
        "SCM at 0.93, correlational at 0.30",
        LOSS,
        n_cause,
        lambda s: _cause(0.93, 0.30, s, c),
        _interval("q1_diff"),
    )
    add(
        "C-DIAG",
        "",
        "perfect",
        "SCM right on every alert, correlational on 92% of them",
        WIN,
        n_diag,
        lambda s: _diag(1.0, 0.92, s, c),
        _interval("reversed_diff"),
    )
    add(
        "C-DIAG",
        "",
        "useless",
        "SCM right on 85% of alerts, correlational on 92%",
        LOSS,
        n_diag,
        lambda s: _diag(0.85, 0.92, s, c),
        _interval("reversed_diff"),
    )

    # ---- C-PLAN: the cost leg and the safety leg, each read with the other cleared
    cost_bound = _two("cost_diff", "failure_diff")
    add(
        "C-PLAN.P0",
        "cost",
        "perfect",
        "P0 at the cheapest honest plan (2 units), no failures either side",
        WIN,
        c.n_max_planner,
        _pick(lambda s: _plan(2.0, 0.0, 0.0, s, c), "C-PLAN.P0"),
        cost_bound,
        reachable=False,
        why=_plan_cost_why(c),
    )
    add(
        "C-PLAN.P0",
        "cost",
        "useless",
        "P0 costs 160 units (the shielded planner of dev: 0.79 of the cap)",
        LOSS,
        c.n_max_planner,
        _pick(lambda s: _plan(160.0, 0.0, 0.0, s, c), "C-PLAN.P0"),
        cost_bound,
    )
    add(
        "C-PLAN.P0",
        "safety",
        "perfect",
        "failure shares identical (0 and 0), cost far below B3",
        WIN,
        c.n_max_planner,
        _pick(lambda s: _plan(None, 0.0, 0.0, s, c), "C-PLAN.P0"),
        cost_bound,
    )
    add(
        "C-PLAN.P0",
        "safety",
        "useless",
        "P0 fails in every episode, B3 in none, cost far below B3",
        LOSS,
        c.n_max_planner,
        _pick(lambda s: _plan(None, 1.0, 0.0, s, c), "C-PLAN.P0"),
        cost_bound,
    )

    # ---- C-MON: the chosen mode
    mon_bound = _two("false_alarm", "detection")
    add(
        "C-MON",
        "",
        "perfect",
        "no false alarm in 700, every one of 300 shifted episodes detected",
        WIN,
        c.mon_nominal,
        _pick(lambda s: _mon(0, c.mon_shifted, s, c), "C-MON"),
        mon_bound,
    )
    add(
        "C-MON",
        "",
        "useless",
        "no false alarm, but 0 of 300 shifted episodes detected",
        LOSS,
        c.mon_shifted,
        _pick(lambda s: _mon(0, 0, s, c), "C-MON"),
        mon_bound,
    )

    # ---- C-SHIELD
    sh_bound = _two("viol_p13.H-C1", "viol_p13.H-C0")
    add(
        "C-SHIELD",
        "",
        "perfect",
        "no P1-P3 violation behind the shield, 40% and 30% without it",
        WIN,
        c.n_max_hostile,
        _pick(lambda s: _shield(0.35, 0, 0.9, s, c), "C-SHIELD"),
        sh_bound,
    )
    add(
        "C-SHIELD",
        "",
        "useless",
        "one defect (an assumption_held violation) in each shielded arm",
        LOSS,
        c.n_max_hostile,
        _pick(lambda s: _shield(0.35, 1, 0.9, s, c), "C-SHIELD"),
        sh_bound,
    )

    # ---- C-SEARCH: ten run pairs
    def sweep(i: int) -> int | None:
        return 20 + i

    def late(i: int) -> int | None:
        return 120 + i if i < 3 else None

    add(
        "C-SEARCH",
        "",
        "perfect",
        "CMA-ES finds first in all ten pairs (random finds in 3 of 10, later)",
        WIN,
        c.search_runs,
        lambda s: _search(sweep, late, s, c),
        _search_bound,
    )
    add(
        "C-SEARCH",
        "",
        "useless",
        "random search finds first in all ten pairs",
        LOSS,
        c.search_runs,
        lambda s: _search(late, sweep, s, c),
        _search_bound,
    )
    add(
        "C-SEARCH",
        "",
        "null",
        "CMA-ES first in 5 pairs, random first in 5",
        INCONCLUSIVE,
        c.search_runs,
        lambda s: _search(
            lambda i: 20 + i if i % 2 == 0 else 120, lambda i: 120 if i % 2 == 0 else 20 + i, s, c
        ),
        _search_bound,
    )
    return tuple(cases)


CASES: Final[tuple[Case, ...]] = _build_cases()


def reachability_rows(cases: Sequence[Case] = CASES) -> list[Row]:
    """Run every case at the stated spreads."""
    return [Row(k, (v := k.run(1.0)).label, k.bound(v)) for k in cases]


def label_at(case: Case, scale: float) -> str | None:
    """The label ``case`` gives when every spread is ``scale`` times the stated one, or ``None`` if
    no data have such spreads (a bounded value cannot have a standard deviation that large)."""
    try:
        return case.run(scale).label
    except ValueError:
        return None


def break_even(case: Case, *, lo: float = 0.05, hi: float = 8.0, steps: int = 9) -> float | None:
    """The largest multiple of the stated spreads at which ``case`` still gives its label, found by
    bisection (the label is assumed to be lost monotonically as the spreads grow), or ``None`` if
    it is not reached even at ``lo`` times them. A scale at which no data exist counts as lost."""
    if label_at(case, lo) != case.want:
        return None
    if label_at(case, hi) == case.want:
        return hi
    good, bad = lo, hi
    for _ in range(steps):
        mid = math.sqrt(good * bad)
        good, bad = (mid, bad) if label_at(case, mid) == case.want else (good, mid)
    return good


# ----------------------------------------------------------------------------- coverage simulation

COVERAGE_LOOKS: Final[tuple[int, ...]] = tuple(range(SPREAD_MIN_PAIRS, 501, 50))
COVERAGE_GUARDS: Final[tuple[int, ...]] = (25, 50, 100)
_CHUNK: Final[int] = 1000


def _binary(p: float) -> Callable[[np.random.Generator, tuple[int, int]], NDArray[np.float64]]:
    return lambda r, s: r.binomial(1, p, s).astype(np.float64)


def _lump(
    p: float, size: float, base: float
) -> Callable[[np.random.Generator, tuple[int, int]], NDArray[np.float64]]:
    return lambda r, s: r.normal(0.0, base, s) + size * r.binomial(1, p, s)


def _cost_shape(r: np.random.Generator, s: tuple[int, int]) -> NDArray[np.float64]:
    return (
        r.poisson(2.6, s)
        - r.poisson(4.1, s)
        + 8.0 * r.binomial(1, 0.075, s)
        + 0.02 * (r.normal(60, 25, s) - r.normal(138, 40, s))
    )


# name -> (draw(rng, (reps, 500)), true mean): the shapes of the throw-away script, a rate from
# 0.005 to 0.5, a lump on a noisy base, and a cost difference like the dev one
COVERAGE_CASES: Final[Mapping[str, tuple[Callable[..., NDArray[np.float64]], float]]] = {
    **{f"binary {p}": (_binary(p), p) for p in (0.005, 0.02, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5)},
    "lump 8 p.02 base .5": (_lump(0.02, 8.0, 0.5), 8.0 * 0.02),
    "lump 8 p.075 base .5": (_lump(0.075, 8.0, 0.5), 8.0 * 0.075),
    "lump 8 p.02 base 2": (_lump(0.02, 8.0, 2.0), 8.0 * 0.02),
    "lump 8 p.075 base 2": (_lump(0.075, 8.0, 2.0), 8.0 * 0.075),
    "dev-like cost": (_cost_shape, 2.6 - 4.1 + 8 * 0.075 + 0.02 * (60 - 138)),
}


@dataclass(frozen=True)
class SimRow:
    """For one shape and one guard ``G``: the share of runs in which the true mean left the
    spread-scaled sequence at some look where the guard (sample kurtosis at most n / G) let it
    be used, and the share of runs in which the guard let it be used at n = 300."""

    case: str
    guard: int
    miss: float
    used_at_300: float
    reps: int
    alpha: float

    @property
    def miss_over_alpha(self) -> float:
        return self.miss / self.alpha


def coverage_sim(
    reps: int,
    seed: int = 2,
    *,
    cases: Mapping[str, tuple[Callable[..., NDArray[np.float64]], float]] | None = None,
    guards: Sequence[int] = COVERAGE_GUARDS,
    looks: Sequence[int] = COVERAGE_LOOKS,
    alpha: float = ALPHA_PRIMARY,
) -> list[SimRow]:
    """Time-uniform coverage of the spread-scaled sequence, with a fixed seed.

    A run is looked at after ``looks`` pairs (100 to 500, the registered sizes). At each look the
    sequence of ``paired_cs`` is used only if the sample kurtosis is at most ``n / G``; otherwise it
    falls back to the betting sequence, which is valid by itself, so the miss of the whole rule is
    at most the miss counted here plus ``alpha``. ``reps`` is rounded down to a multiple of 1,000.
    """
    chosen = cases if cases is not None else COVERAGE_CASES
    rng = np.random.default_rng(seed)
    t_look = np.asarray(looks)
    out: list[SimRow] = []
    for name, (draw, mu) in chosen.items():
        chunks = max(1, reps // _CHUNK)
        miss = dict.fromkeys(guards, 0)
        used = dict.fromkeys(guards, 0.0)
        for _ in range(chunks):
            x = draw(rng, (_CHUNK, int(t_look.max()))).astype(np.float64)
            s1, s2, s3, s4 = (np.cumsum(x**k, axis=1) for k in (1, 2, 3, 4))
            bad = {g: np.zeros(_CHUNK, dtype=bool) for g in guards}
            for t in t_look:
                m1, m2, m3, m4 = (s[:, t - 1] / t for s in (s1, s2, s3, s4))
                var = np.maximum(m2 - m1**2, 0.0)
                c4 = m4 - 4 * m1 * m3 + 6 * m1**2 * m2 - 3 * m1**4
                with np.errstate(divide="ignore", invalid="ignore"):
                    kurt = np.where(var > 0, c4 / var**2, np.inf)
                half = spread_halfwidths(np.sqrt(var * t / (t - 1)), int(t), alpha)
                outside = np.abs(m1 - mu) > half
                for g in guards:
                    ok = kurt <= t / g
                    bad[g] |= ok & outside
                    if t == 300:
                        used[g] += float(ok.mean())
            for g in guards:
                miss[g] += int(bad[g].sum())
        n = chunks * _CHUNK
        out.extend(SimRow(name, g, miss[g] / n, used[g] / chunks, n, alpha) for g in guards)
    return out


# ----------------------------------------------------------------------------- records


def _record(name: str, value: float, n: int, what: str, **extra: float | int | str) -> ResultRecord:
    return ResultRecord(
        name=name,
        value=value,
        n=n,
        fidelity=Fidelity.F1,
        measured_on=f"{MADE_UP}: {what}",
        script=SCRIPT,
        extra={"made_up": 1, **extra},
    )


def reach_records(quick: bool = False, sim_reps: int | None = None) -> list[ResultRecord]:
    """The records of the ``reach`` stage, all labelled made-up data (names ``reach.*``).

    The floors and half-width factors of prereg section 6, the verdict of every case of the
    table at the registered sizes, and the coverage simulation with a fixed seed (``sim_reps``
    runs a shape; default 40,000, or 2,000 when ``quick``)."""
    reps = sim_reps if sim_reps is not None else (2000 if quick else 40000)
    out: list[ResultRecord] = []
    for f in floors():
        out.append(
            _record(
                f"reach.floor.n{f.n}",
                f.betting,
                f.n,
                f"half-width of the betting sequence for a paired difference in [-1, 1] at n = "
                f"{f.n}, level {f.alpha:.5f}, whatever the data",
                alpha=f.alpha,
            )
        )
        out.append(
            _record(
                f"reach.spread_factor.n{f.n}",
                f.spread_factor,
                f.n,
                f"the spread-scaled half-width is this times the sample sd, n = {f.n}, level "
                f"{f.alpha:.5f}",
                alpha=f.alpha,
            )
        )
    for row in reachability_rows():
        k = row.case
        out.append(
            _record(
                f"reach.verdict.{k.id}",
                1.0 if row.reached else 0.0,
                k.n,
                f"{k.claim} {k.leg or 'verdict'}, {k.kind} input ({k.what}): the rule should give "
                f"{k.want}",
                label=row.label,
                want=k.want,
                reachable=int(k.reachable),
                bound=row.bound,
                why=k.why,
            )
        )
    for sim in coverage_sim(reps):
        lo, hi = wilson(round(sim.miss * sim.reps), sim.reps)
        rec = _record(
            f"reach.cover.{sim.case}.G{sim.guard}",
            sim.miss,
            sim.reps,
            f"share of runs in which the true mean left the spread-scaled sequence at some look "
            f"of {COVERAGE_LOOKS[0]} to {COVERAGE_LOOKS[-1]} pairs where the guard (kurtosis at "
            f"most n / {sim.guard}) allowed it, level {sim.alpha:.5f}",
            miss_over_alpha=sim.miss_over_alpha,
            used_at_300=sim.used_at_300,
            guard=sim.guard,
            registered_guard=int(sim.guard == SPREAD_KURTOSIS_DIVISOR),
        )
        out.append(replace(rec, ci_lo=lo, ci_hi=hi, ci_kind="wilson-0.95"))
    return out
