"""Metrics of the pre-registration, sections 5.1 and 5.2, computed from records (packet P12a).

Everything here is a function of what one F1 episode leaves behind: its ``Trace``, its
checkpoints and ``Label`` records, the plant's own violation events and a few counters. Nothing is
typed by hand and nothing reads the simulator; the truth enters only through the sim-only columns
the records already carry (0003 A10: coverage reads ``true_rul_days``, the proof's quantities read
``health_bin``).

M-VIOL (``spec_report``, ``crosscheck``)
    ``V_e`` is 1 when the minimum robustness of some property of the group is negative **or the
    plant recorded its own violation event** for some property of the group (0003 A18), for the
    group P1-P3 and for the group P4-P5. Both are definitions of a violation: the plant is the
    simulated site and an event it records happened, whether or not the state formula can see it.
    Until A18 the plant's events were a cross-check and not a second definition, and the headline
    for the hand-written interlock read "matches the shield" on a spec score of 0 of 20 against 7
    of 20 by the plant's events. ``m_viol_p13_spec`` is the spec-only reading, kept as a secondary
    row, and ``m_disagree`` still records every property on which the two differ, with the
    direction (``spec_only`` or ``plant_only``). The plant and the formula agree since ADR 0003
    A12 about the three sources the P12a note listed (a permit lives 12 hours from its issue and a
    handover does not renew that; a locked pump is not available; P1 reads the conveyor). One is
    left: the plant's P1 also records the event of energising an asset while work on it is open,
    which the state formula cannot see. A plant-only event has no spec event to label, so
    attribution (below) does not see it and it cannot make a defect.

Attribution (0003 A5 and A15; ``assumption_breaks``, ``attribute``, ``is_defect``)
    Each spec violation event is labelled ``assumption_held`` when no assumption that may excuse
    it was broken at any tick of the 30 days up to and including it, else with the first such
    broken assumption (the earliest tick; ties go to A_exec, A_phys, C1) and its tick. An
    assumption excuses only a property that depends on it, and only on the component concerned
    (pre-registration 5.1, M-VIOL):

    * P1 reads no health: only A_exec on the asset the violating conjunct names (the assets that
      are working without a proven isolation at the event's row) can excuse it;
    * P2, P3 and P6 read no health either: only A_exec on the crusher, the confined space;
    * P4 reads the pumps: A_exec, A_phys or C1 on a pump;
    * P5 reads the crusher: A_exec, A_phys or C1 on the crusher.

    The breaks are kept per asset and are never OR-ed across assets. The ticks are trace rows:

    * A_exec is broken at row ``t`` when a STOP or EXIT issued in quarter ``t`` was not executed;
    * A_phys is broken at row ``t`` when a component's health bin rose by more than one since row
      ``t - 1``, or an uncontrolled failure happened from a bin below the top
      (``shield.conformance`` defines both the same way);
    * C1 is broken at row ``t`` when the true bin entering the step, ``health_bin[t - 1]``, was not
      in the set the shield was given for that decision, ``[set_lo[t - 1], set_hi[t - 1]]`` (the
      plant shows a freshly computed set from the next quarter, so the set the shield used is the
      previous row's). After the coverage monitor has fired the sets are the whole range and C1
      holds. C1 and A_phys are read on the crusher and the pumps, the components the proof is about.

    Every spec event is attributed, P6 included; the M-VIOL metrics read P1 to P5 only, and
    ``is_defect`` is the one place that says what a defect is: an ``assumption_held`` label on P1 to
    P5. A violation event of the plant with no spec counterpart is a disagreement, and the
    disagreement count carries it.

M-COV, M-COST and the secondary metrics
    ``coverage_of`` (pooled over the post-blind checkpoints and for the trend-estimated stratum,
    with the blind and excluded counts), ``cost_of`` (the 200 cap and the clip count; everything the
    plant charges: planned jobs, unplanned failures, inspections, rejected work orders, and the
    full-throughput quarters lost, 0003 A18), throughput,
    override rate per 1,000 proposed commands, shield permissiveness at the visited entries.

``aggregate`` turns the episodes of one cell into a ``ResultRecord`` with an anytime-valid
confidence sequence (betting, ``alpha`` 0.05 unless told otherwise), the count behind a
proportion and, for a proportion with no event, the zero-event upper bound.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

import numpy as np
from numpy.typing import NDArray

from closed_loop.core.results import ResultRecord
from closed_loop.core.timegrid import DAY_Q
from closed_loop.core.trace import Trace, op_index
from closed_loop.core.types import Cmd, Op
from closed_loop.perception.coverage import checkpoint_coverage, select
from closed_loop.perception.labels import Checkpoint, Label, LabelStatus
from closed_loop.spec.properties import (
    DEFAULT_CONVEYOR,
    PROPERTY_IDS,
    SpecParams,
    property_formulas,
    signals_from_trace,
)
from closed_loop.spec.stl import Always, robustness
from closed_loop.stats import betting_cs, zero_event_upper

if TYPE_CHECKING:
    from closed_loop.eval.episode import EpisodeResult

SCRIPT: Final[str] = "closed_loop.eval.metrics"

__all__ = [
    "COST_CAP",
    "CRUSHER_ID",
    "COST_INSPECTION",
    "COST_REJECTED_WO",
    "METRICS",
    "P13",
    "P15",
    "P45",
    "WINDOW_Q",
    "AssumptionBreaks",
    "Attribution",
    "EpisodeCoverage",
    "MetricSpec",
    "SpecReport",
    "ViolationEvent",
    "aggregate",
    "assumption_breaks",
    "attribute",
    "cost_of",
    "coverage_of",
    "crosscheck",
    "crusher_work_rows",
    "late_gas_tests",
    "is_defect",
    "LEARNER_POPULATIONS",
    "latency_report",
    "learner_counts",
    "pooled_contract",
    "spec_report",
]

P13: Final[tuple[str, ...]] = ("P1", "P2", "P3")
P45: Final[tuple[str, ...]] = ("P4", "P5")
P15: Final[tuple[str, ...]] = (*P13, *P45)
WINDOW_Q: Final[int] = 30 * DAY_Q  # the look-back of 0003 A5, in quarters

# M-COST, prereg 5.1 and 0003 A18: 1.0 x planned jobs + 8.0 x unplanned failures + 0.05 x
# inspections + 0.02 x rejected work orders + 0.02 x full-throughput quarters lost, capped at 200
# and divided by 200. The first four are what the plant charges (``PlantStats.cost``), the last is
# the lost production.
COST_PLANNED: Final[float] = 1.0
COST_UNPLANNED: Final[float] = 8.0
COST_INSPECTION: Final[float] = 0.05
COST_REJECTED_WO: Final[float] = 0.02
COST_LOST_QUARTER: Final[float] = 0.02
COST_CAP: Final[float] = 200.0
FAILURES_CAP: Final[float] = 8.0  # unplanned failures per episode: four assets, a few lives each
JOBS_CAP: Final[float] = 40.0  # planned jobs per episode (B3-V6: 14 on dev; clipped, counted)
WORK_Q_CAP: Final[float] = 8640.0  # quarters in an episode: the longest a count can be
OVERRIDE_CAP: Final[float] = 1000.0  # the override rate is per 1,000 proposed commands
N_CRUSHER_COMMANDS: Final[int] = 14
N_PUMP_COMMANDS: Final[int] = 64  # 8 commands a pump (ADR 0003 A12.2), pairs of them; it was 49
N_BINS: Final[int] = 4
DETECT_WITHIN_DAYS: Final[float] = 30.0  # M-DET: a firing within 30 days of the shift start

# A_exec is about the commands the plant is promised to execute (core.types.FORCED_CMDS).
_FORCED: Final[frozenset[Cmd]] = frozenset({Cmd.STOP, Cmd.EXIT})
_ORDER: Final[tuple[str, ...]] = ("A_exec", "A_phys", "C1")  # ties go in this order


# ----------------------------------------------------------------------------- M-VIOL


@dataclass(frozen=True)
class ViolationEvent:
    """A maximal run of negative robustness of one property: its first trace row."""

    prop: str
    row: int


@dataclass(frozen=True)
class SpecReport:
    min_robustness: Mapping[str, float]
    events: tuple[ViolationEvent, ...]  # time order, then property order

    @property
    def counts(self) -> dict[str, int]:
        return {p: sum(1 for e in self.events if e.prop == p) for p in PROPERTY_IDS}


def spec_report(trace: Trace, spec: SpecParams, crusher: str, pumps: tuple[str, str]) -> SpecReport:
    """Minimum robustness of P1 to P6 and the violation events (runs of negative robustness).

    The minimum equals ``spec.properties.evaluate`` (the outer ``G`` is a running minimum); the
    events come from the same formulas without it.
    """
    signals = signals_from_trace(trace, spec, crusher, pumps)
    mins: dict[str, float] = {}
    events: list[ViolationEvent] = []
    for pid, formula in property_formulas(spec).items():
        inner = formula.arg if isinstance(formula, Always) else formula
        rob = robustness(inner, signals)
        mins[pid] = float(np.min(rob)) if rob.size else math.inf
        neg = rob < 0
        starts = np.flatnonzero(neg & ~np.concatenate(([False], neg[:-1])))
        events.extend(ViolationEvent(pid, int(r)) for r in starts)
    events.sort(key=lambda e: (e.row, e.prop))
    return SpecReport(mins, tuple(events))


def crosscheck(
    min_robustness: Mapping[str, float], plant_counts: Mapping[str, int]
) -> dict[str, str]:
    """Properties on which the spec and the plant's own events disagree, with the direction."""
    out: dict[str, str] = {}
    for pid in PROPERTY_IDS:
        spec_violated = min_robustness.get(pid, math.inf) < 0.0
        plant_violated = plant_counts.get(pid, 0) > 0
        if spec_violated and not plant_violated:
            out[pid] = "spec_only"
        elif plant_violated and not spec_violated:
            out[pid] = "plant_only"
    return out


# ----------------------------------------------------------------------------- attribution


def is_defect(prop: str, label: str) -> bool:
    """A defect in a shielded arm (0003 A5, A15): an ``assumption_held`` violation of P1 to P5.

    The one function that decides it, for the episode metrics and for the search alike. P6 is
    attributed and reported but is not in the defect rule.
    """
    return prop in P15 and label == "assumption_held"


@dataclass(frozen=True)
class Attribution:
    """Why one violation happened: ``assumption_held`` or the first broken assumption."""

    prop: str
    row: int  # the violation's tick
    label: str  # "assumption_held" | "A_exec" | "A_phys" | "C1"
    broken_row: int | None = None  # the first broken tick of ``label`` in the look-back
    asset: str = field(default="", compare=False)  # where it broke (for the report, not identity)

    @property
    def defect(self) -> bool:
        """An unexplained violation of P1 to P5 (0003 A5): only meaningful for a shielded arm."""
        return is_defect(self.prop, self.label)


@dataclass(frozen=True)
class AssumptionBreaks:
    """For every trace row and every asset, whether each assumption of the proof was broken there.

    ``a_exec`` has an entry for each asset of the trace that P1 to P6 read; ``a_phys`` and ``c1``
    for the crusher and the pumps. ``p1_open`` marks, per asset, the rows at which that asset's
    conjunct of P1 is negative (working without a proven isolation): the asset a P1 event names.
    """

    a_exec: Mapping[str, NDArray[np.bool_]]
    a_phys: Mapping[str, NDArray[np.bool_]]
    c1: Mapping[str, NDArray[np.bool_]]
    crusher: str
    pumps: tuple[str, str]
    p1_open: Mapping[str, NDArray[np.bool_]] = field(default_factory=dict)

    def by_name(self) -> dict[str, Mapping[str, NDArray[np.bool_]]]:
        return {"A_exec": self.a_exec, "A_phys": self.a_phys, "C1": self.c1}

    def flat(self, name: str) -> NDArray[np.bool_]:
        """The breaks of one assumption OR-ed over assets: for reports and tests, never for the
        attribution itself."""
        arrays = list(self.by_name()[name].values())
        if not arrays:
            return np.zeros(0, dtype=bool)
        return np.any(np.stack(arrays), axis=0)


def assumption_breaks(
    trace: Trace,
    *,
    crusher: str,
    pumps: tuple[str, str],
    fallback_row: int | None = None,
    conveyor: str = DEFAULT_CONVEYOR,
) -> AssumptionBreaks:
    """Per-asset, per-row indicators of A_exec, A_phys and C1 (see the module docstring).

    ``fallback_row`` is the first row at which the coverage monitor had fired for the shield: from
    there on the shield quantifies over the whole range of bins and C1 holds by construction.
    """
    n = trace.n_q
    p1_assets = tuple(a for a in (crusher, *pumps, conveyor) if a in trace.asset_ids)
    a_exec = {a: np.zeros(n, dtype=bool) for a in p1_assets}
    for q, cmd, executed, _overridden in trace.commands:
        if (
            not executed
            and isinstance(cmd.cmd, Cmd)
            and cmd.cmd in _FORCED
            and cmd.asset_id in a_exec
            and 0 <= q < n
        ):
            a_exec[cmd.asset_id][q] = True
    a_phys: dict[str, NDArray[np.bool_]] = {}
    c1: dict[str, NDArray[np.bool_]] = {}
    for asset in (crusher, *pumps):
        a_phys[asset] = np.zeros(n, dtype=bool)
        c1[asset] = np.zeros(n, dtype=bool)
        if n < 2:
            continue
        h = trace.health_bin[asset].astype(np.int64)
        skip = (h[1:] - h[:-1]) > 1
        early = trace.uncontrolled_failure[asset][1:] & (h[:-1] < N_BINS - 1)
        a_phys[asset][1:] = skip | early
        lo = trace.set_lo[asset].astype(np.int64)
        hi = trace.set_hi[asset].astype(np.int64)
        c1[asset][1:] = (h[:-1] < lo[:-1]) | (h[:-1] > hi[:-1])
        if fallback_row is not None:
            c1[asset][max(fallback_row, 0) :] = False
    work = op_index(Op.WORK)
    p1_open = {a: (trace.op[a] == work) & ~trace.iso_proven[a].astype(bool) for a in p1_assets}
    return AssumptionBreaks(a_exec, a_phys, c1, crusher, pumps, p1_open)


def _excusers(
    prop: str, breaks: AssumptionBreaks, row: int
) -> list[tuple[str, str, NDArray[np.bool_]]]:
    """``(assumption, asset, per-row breaks)`` that may excuse a violation of ``prop`` at ``row``.

    P1 reads the assets whose conjunct is negative at ``row`` and none of A_phys or C1 (no health);
    P2, P3 and P6 read the crusher's A_exec; P4 the pumps and P5 the crusher, with all three
    assumptions. An unknown property, or a P1 event at a row where no asset's conjunct is negative,
    has no excuse: it is held against the proof.
    """
    if prop == "P1":
        named = [
            a for a, mask in breaks.p1_open.items() if 0 <= row < mask.shape[0] and bool(mask[row])
        ]
        return [("A_exec", a, breaks.a_exec[a]) for a in named if a in breaks.a_exec]
    if prop in ("P2", "P3", "P6"):
        arr = breaks.a_exec.get(breaks.crusher)
        return [] if arr is None else [("A_exec", breaks.crusher, arr)]
    if prop == "P4":
        assets: tuple[str, ...] = breaks.pumps
    elif prop == "P5":
        assets = (breaks.crusher,)
    else:
        return []
    out: list[tuple[str, str, NDArray[np.bool_]]] = []
    for name in _ORDER:
        for asset in assets:
            arr = breaks.by_name()[name].get(asset)
            if arr is not None:
                out.append((name, asset, arr))
    return out


def attribute(
    events: Sequence[ViolationEvent], breaks: AssumptionBreaks, window_q: int = WINDOW_Q
) -> tuple[Attribution, ...]:
    """Label each event with ``assumption_held`` or its first excusing break in the look-back."""
    out: list[Attribution] = []
    for ev in events:
        lo = max(ev.row - window_q, 0)
        first: tuple[int, int, str, str] | None = None
        for name, asset, arr in _excusers(ev.prop, breaks, ev.row):
            hits = np.flatnonzero(arr[lo : ev.row + 1])
            if hits.size:
                cand = (lo + int(hits[0]), _ORDER.index(name), name, asset)
                if first is None or cand < first:
                    first = cand
        if first is None:
            out.append(Attribution(ev.prop, ev.row, "assumption_held"))
        else:
            out.append(Attribution(ev.prop, ev.row, first[2], first[0], first[3]))
    return tuple(out)


# ----------------------------------------------------------------------------- M-COV


@dataclass(frozen=True)
class EpisodeCoverage:
    """``c_e`` and ``c_e^lo`` of one episode (prereg 5.1 M-COV), pooled and for the trend stratum.

    The pooled population is every post-blind checkpoint (the blind window is excluded and counted
    in ``n_blind``); the trend stratum is the post-blind checkpoints with a trend estimate. A value
    is ``None`` when its population is empty, and the episode is then excluded from that cell and
    counted by ``aggregate``.
    """

    c: float | None
    c_lo: float | None
    n: int
    c_trend: float | None
    c_lo_trend: float | None
    n_trend: int
    n_blind: int
    n_total: int


def coverage_of(checkpoints: Sequence[Checkpoint]) -> EpisodeCoverage:
    """Oracle coverage of the scheduling interval; every checkpoint needs its sim-only truth."""
    post = select(checkpoints, "post_blind")
    trend = [cp for cp in post if not cp.prior]

    def mean(cps: Sequence[Checkpoint], *, lower: bool) -> float | None:
        if not cps:
            return None
        return checkpoint_coverage(cps, "all", lower=lower)

    return EpisodeCoverage(
        c=mean(post, lower=False),
        c_lo=mean(post, lower=True),
        n=len(post),
        c_trend=mean(trend, lower=False),
        c_lo_trend=mean(trend, lower=True),
        n_trend=len(trend),
        n_blind=len(checkpoints) - len(post),
        n_total=len(checkpoints),
    )


# ----------------------------------------------------------------------------- M-COST


def cost_of(
    planned_jobs: int,
    unplanned_failures: int,
    lost_quarters: float,
    inspections: int,
    rejected_wos: int,
) -> float:
    """``cost_e`` of prereg 5.1 before the cap: everything the plant charges, and the lost quarters.

    Without the last two arguments a planner that inspects or whose work orders are refused was
    charged for neither (0003 A18).
    """
    return (
        COST_PLANNED * planned_jobs
        + COST_UNPLANNED * unplanned_failures
        + COST_INSPECTION * inspections
        + COST_REJECTED_WO * rejected_wos
        + COST_LOST_QUARTER * lost_quarters
    )


# ----------------------------------------------------------------------------- learner counts

# The two populations Gap_A is read on: every post-blind label, and the trend-estimated stratum.
LEARNER_POPULATIONS: Final[tuple[str, ...]] = ("post_blind", "estimated")


def learner_counts(labels: Sequence[Label]) -> dict[str, int]:
    """Counts of the learner-visible label statuses, per population, so that a runner can drop the
    labels themselves and still pool ``C_lrn`` and the Manski bounds over episodes.

    Keys are ``"<population>.<n|hit|miss|lower_hit|unknown|lower_ok|uncensored>"``; ``lower_ok``
    counts the informative labels whose lower bound is known to have held
    (``perception.coverage``). ``uncensored`` counts the labels of lives that ended in an observed
    failure (0003 A18, F1): only those can confirm an upper bound, so a cell with none has no
    ``C_lrn`` to speak of. It is a counter and decides nothing in the loop.
    """
    out: dict[str, int] = {}
    for pop in LEARNER_POPULATIONS:
        sel = select(labels, pop)
        out[f"{pop}.n"] = len(sel)
        for status in LabelStatus:
            out[f"{pop}.{status.value.lower()}"] = sum(1 for lab in sel if lab.status is status)
        out[f"{pop}.uncensored"] = sum(1 for lab in sel if not lab.censored)
        out[f"{pop}.lower_ok"] = sum(
            1
            for lab in sel
            if lab.status is not LabelStatus.UNKNOWN and lab.checkpoint_day + lab.lo <= lab.end_day
        )
    return out


# ----------------------------------------------------------------------------- A4 latency

LAG_BINS: Final[tuple[str, ...]] = ("0", "1", "2-4", "5+", "never")


def _lag_bin(lag: int | None) -> str:
    if lag is None:
        return "never"
    return "0" if lag == 0 else "1" if lag == 1 else "2-4" if lag <= 4 else "5+"


def latency_report(results: Sequence[EpisodeResult]) -> dict[str, dict[str, int]]:
    """Quarters from a violation's occurrence to the first negative online robustness of its
    property (0003 A4), per property, as a histogram ``0 / 1 / 2-4 / 5+ / never``.

    Only episodes with the interlock in the loop have online monitors (the interlock's own
    ``StreamingMonitor``s); other episodes contribute nothing and are not counted as ``never``.
    """
    out: dict[str, dict[str, int]] = {p: dict.fromkeys(LAG_BINS, 0) for p in PROPERTY_IDS}
    for r in results:
        for lag in r.online_lags:
            out[lag.prop][_lag_bin(lag.lag_q)] += 1
    return out


# ----------------------------------------------------------------------------- pooled counts


def pooled_contract(results: Sequence[EpisodeResult]) -> dict[str, float]:
    """Per-life any-miss under both readings (A10) and the mean set width, pooled over episodes."""
    lives = sum(r.contract.lives for r in results)
    cps = sum(r.contract.checkpoints for r in results)

    def rate(k: int, n: int) -> float:
        return k / n if n else math.nan

    return {
        "lives": float(lives),
        "checkpoints": float(cps),
        "any_miss_rul": rate(sum(r.contract.lives_miss_rul for r in results), lives),
        "any_miss_bin": rate(sum(r.contract.lives_miss_bin for r in results), lives),
        "step_miss_rul": rate(sum(r.contract.miss_rul for r in results), cps),
        "step_miss_bin": rate(sum(r.contract.miss_bin for r in results), cps),
        "mean_width_bins": rate(sum(r.contract.width_sum for r in results), cps),
    }


# ----------------------------------------------------------------------------- the metric table


@dataclass(frozen=True)
class MetricSpec:
    """One reported metric: a per-episode value (``None`` = excluded) and the range it is
    rescaled over. Values outside ``[lo, hi]`` are clipped, counted and reported (prereg 5)."""

    name: str
    describe: str
    value: Callable[[EpisodeResult], float | None]
    lo: float = 0.0
    hi: float = 1.0
    proportion: bool = False


def _any_min_negative(props: Sequence[str]) -> Callable[[EpisodeResult], float | None]:
    """1 when the spec's robustness is negative for a property of ``props`` (the spec alone)."""
    return lambda r: float(any(r.min_robustness.get(p, math.inf) < 0.0 for p in props))


def _violated(props: Sequence[str]) -> Callable[[EpisodeResult], float | None]:
    """1 when the spec's robustness is negative for a property of ``props`` OR the plant recorded
    its own event for one (0003 A18)."""
    return lambda r: float(
        any(
            r.min_robustness.get(p, math.inf) < 0.0 or r.plant_violations.get(p, 0) > 0
            for p in props
        )
    )


def _plant_only(props: Sequence[str]) -> Callable[[EpisodeResult], float | None]:
    """1 when the plant recorded an event for a property of ``props`` and the spec's robustness is
    non-negative for all of them: the direction ``m_disagree`` calls ``plant_only``. Such an event
    has no spec event to attribute, so it can be neither a defect nor a contract failure (0003 A18,
    F4)."""
    spec = _any_min_negative(props)
    viol = _violated(props)
    return lambda r: float(viol(r) == 1.0 and spec(r) == 0.0)


# ----------------------------------------------------------------------------- the gas clock
# 0003 A18, F5. The abstraction reads the gas clock before the command and the plant and the monitor
# read it after, and both are kept. These two secondary rows publish what that costs in the plant,
# from the trace, with no decision rule attached.

CRUSHER_ID: Final[str] = "CR-501"
QUARTER_MINUTES: Final[float] = 15.0


def crusher_work_rows(trace: Trace, crusher: str = CRUSHER_ID) -> int:
    """Quarters the crusher stands in WORK, whatever the reason it is there."""
    return int(np.sum(trace.op[crusher] == op_index(Op.WORK)))


def late_gas_tests(trace: Trace, retest_q: int, crusher: str = CRUSHER_ID) -> int:
    """Rows ``q`` at which someone is inside at both ``q - 1`` and ``q``, the recorded gas age at
    ``q - 1`` is at least the re-test period, and the age at ``q`` is below it: a test that came a
    quarter late with someone inside. The first boundary is inclusive (an age of exactly the period
    counts as due), the second strict (an age equal to the period is not a pass).
    """
    limit = np.float32(retest_q * QUARTER_MINUTES)
    inside = trace.occupied[crusher]
    age = trace.gas_age_min[crusher]
    late = inside[:-1] & inside[1:] & (age[:-1] >= limit) & (age[1:] < limit)
    return int(np.sum(late))


def _crusher_work_per_job(r: EpisodeResult) -> float | None:
    """Mean quarters the crusher stands in WORK per crusher job that completes, in this episode;
    ``None`` (excluded and counted) when no crusher job completes."""
    return r.crusher_work_q / r.crusher_jobs_done if r.crusher_jobs_done else None


def _attributed(
    labels: Sequence[str], props: Sequence[str] = P15
) -> Callable[[EpisodeResult], float | None]:
    return lambda r: float(any(a.label in labels and a.prop in props for a in r.attributions))


def _defect(props: Sequence[str] = P15) -> Callable[[EpisodeResult], float | None]:
    """1 when some violation of ``props`` is a defect by ``is_defect`` (the one rule)."""
    return lambda r: float(
        any(is_defect(a.prop, a.label) and a.prop in props for a in r.attributions)
    )


def _plant_disagrees(r: EpisodeResult) -> float | None:
    return float(any(p in r.disagreements for p in P15))


def _cov(field: str) -> Callable[[EpisodeResult], float | None]:
    return lambda r: getattr(r.coverage, field)


def _override_rate(r: EpisodeResult) -> float | None:
    overrides = r.shield_overrides + r.interlock_refused + r.interlock_forced
    return 1000.0 * overrides / r.commands_proposed if r.commands_proposed else None


def _permissive(total: int, which: str) -> Callable[[EpisodeResult], float | None]:
    def value(r: EpisodeResult) -> float | None:
        s, n = getattr(r, f"allowed_{which}")
        return s / (n * total) if n else None

    return value


def _set_width(r: EpisodeResult) -> float | None:
    return (
        r.contract.width_sum / (r.contract.checkpoints * N_BINS) if r.contract.checkpoints else None
    )


def _fired(mode: str) -> Callable[[EpisodeResult], float | None]:
    return lambda r: float(r.monitors.get(mode) is not None) if mode in r.monitors else None


def _detected(mode: str) -> Callable[[EpisodeResult], float | None]:
    def value(r: EpisodeResult) -> float | None:
        if mode not in r.monitors or r.shift_day is None:
            return None
        day = r.monitors[mode]
        return float(day is not None and r.shift_day <= day <= r.shift_day + DETECT_WITHIN_DAYS)

    return value


def _build_metrics() -> dict[str, MetricSpec]:
    table: list[MetricSpec] = [
        MetricSpec(
            "m_viol_p13",
            "M-VIOL, group P1-P3: negative spec robustness or an event the plant recorded",
            _violated(P13),
            proportion=True,
        ),
        MetricSpec(
            "m_viol_p13_spec",
            "M-VIOL, group P1-P3, the spec alone (secondary: negative robustness only)",
            _any_min_negative(P13),
            proportion=True,
        ),
        MetricSpec(
            "m_viol_p13_plant_only",
            "episodes with a plant event on P1-P3 and no negative spec robustness there "
            "(secondary: the part of M-VIOL (P1-P3) the spec cannot see)",
            _plant_only(P13),
            proportion=True,
        ),
        MetricSpec(
            "m_viol_p45",
            "M-VIOL, group P4-P5: negative spec robustness or an event the plant recorded",
            _violated(P45),
            proportion=True,
        ),
        MetricSpec(
            "m_viol_p15",
            "a violation of any of P1-P5: negative spec robustness or an event the plant recorded",
            _violated(P15),
            proportion=True,
        ),
        MetricSpec(
            "m_defect",
            "a P1-P5 violation attributed assumption_held (0003 A5: a defect in a shielded arm)",
            _defect(),
            proportion=True,
        ),
        MetricSpec(
            "m_defect_p13",
            "a P1-P3 violation attributed assumption_held",
            _defect(P13),
            proportion=True,
        ),
        MetricSpec(
            "m_contract_failure",
            "a P1-P5 violation attributed to C1 or A_phys (a contract failure, not a defect)",
            _attributed(("C1", "A_phys")),
            proportion=True,
        ),
        MetricSpec(
            "m_attr_c1", "a P1-P5 violation attributed to C1", _attributed(("C1",)), proportion=True
        ),
        MetricSpec(
            "m_attr_a_phys",
            "a P1-P5 violation attributed to A_phys",
            _attributed(("A_phys",)),
            proportion=True,
        ),
        MetricSpec(
            "m_attr_a_exec",
            "a P1-P5 violation attributed to A_exec",
            _attributed(("A_exec",)),
            proportion=True,
        ),
        MetricSpec(
            "m_disagree",
            "the spec and the plant's own events disagree on some property of P1-P5",
            _plant_disagrees,
            proportion=True,
        ),
        MetricSpec("m_cov", "M-COV c_e, post-blind checkpoints", _cov("c")),
        MetricSpec("m_cov_lo", "M-COV c_e^lo, post-blind checkpoints", _cov("c_lo")),
        MetricSpec("m_cov_trend", "M-COV c_e, trend-estimated stratum", _cov("c_trend")),
        MetricSpec("m_cov_lo_trend", "M-COV c_e^lo, trend-estimated stratum", _cov("c_lo_trend")),
        MetricSpec(
            "m_cost",
            "M-COST: cost_e capped at 200, divided by 200",
            lambda r: r.cost / COST_CAP,
        ),
        MetricSpec(
            "u_e",
            "unplanned failures per episode",
            lambda r: float(r.unplanned_failures),
            hi=FAILURES_CAP,
        ),
        MetricSpec(
            "f_e",
            "f_e: 1 if the episode had at least one unplanned failure (the safety leg of C-PLAN)",
            lambda r: float(r.unplanned_failures > 0),
            proportion=True,
        ),
        MetricSpec("throughput", "mean throughput fraction", lambda r: r.throughput),
        MetricSpec(
            "override_rate",
            "overrides (shield refusals and substitutions, interlock refusals and forced "
            "commands) per 1,000 proposed commands",
            _override_rate,
            hi=OVERRIDE_CAP,
        ),
        MetricSpec(
            "permissiveness_crusher",
            "mean allowed commands at the visited crusher entries, over 14",
            _permissive(N_CRUSHER_COMMANDS, "crusher"),
        ),
        MetricSpec(
            "permissiveness_pumps",
            "mean allowed joint commands at the visited pump-pair entries, over 64",
            _permissive(N_PUMP_COMMANDS, "pumps"),
        ),
        MetricSpec("set_width", "mean perception-set width over 4 bins", _set_width),
        MetricSpec(
            "wrong_mode_jobs",
            "completed jobs of the wrong failure mode per episode (0003 A3: a cost, no renewal)",
            lambda r: float(r.wrong_mode_jobs),
            hi=FAILURES_CAP,
        ),
        MetricSpec(
            "planned_jobs",
            "planned jobs per episode",
            lambda r: float(r.planned_jobs),
            hi=JOBS_CAP,
        ),
        MetricSpec(
            "m_crusher_work_q",
            "quarters CR-501 stands in WORK per crusher job that completes (secondary, 0003 A18 "
            "F5: no decision rule; episodes with no completed crusher job are excluded)",
            _crusher_work_per_job,
            hi=WORK_Q_CAP,
        ),
        MetricSpec(
            "m_late_gas_tests",
            "rows per episode where someone is inside, the gas age was at least the re-test "
            "period a quarter ago and is below it now: a test a quarter late with someone inside "
            "(secondary, 0003 A18 F5: no decision rule)",
            lambda r: float(r.late_gas_tests),
            hi=WORK_Q_CAP,
        ),
    ]
    for mode in ("opt", "pess", "inf", "proxy", "audit"):
        table.append(
            MetricSpec(
                f"m_false_alarm_{mode}",
                f"M-DET: the {mode} monitor fired (read on unshifted episodes)",
                _fired(mode),
                proportion=True,
            )
        )
        table.append(
            MetricSpec(
                f"m_detect_{mode}",
                f"M-DET: the {mode} monitor fired within 30 days of the shift start",
                _detected(mode),
                proportion=True,
            )
        )
    return {m.name: m for m in table}


METRICS: Final[Mapping[str, MetricSpec]] = _build_metrics()


# ----------------------------------------------------------------------------- aggregate


def aggregate(results: Sequence[EpisodeResult], metric: str, alpha: float = 0.05) -> ResultRecord:
    """The cell value of ``metric`` over the episodes of ONE cell (one arm, one condition, one
    fidelity), with a confidence sequence valid at all times.

    The value is the mean of the per-episode values on their own scale; the interval is the
    betting confidence sequence of the rescaled values (``MetricSpec.lo`` to ``hi``), mapped back.
    Episodes whose value is undefined for the metric (``None``) are excluded and counted in
    ``extra["excluded"]``; values beyond the range are clipped and counted in ``extra["clipped"]``.
    For a proportion ``extra["k"]`` is the count and, with no event, ``extra["zero_event_upper"]``.
    Raises ``ValueError`` for an unknown metric, no episode, a mixed cell, or no usable episode.
    """
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {sorted(METRICS)}")
    if not results:
        raise ValueError("aggregate needs at least one episode")
    cell = {(r.arm, r.condition, r.fidelity) for r in results}
    if len(cell) != 1:
        raise ValueError(f"aggregate takes the episodes of one cell, got {sorted(map(str, cell))}")
    arm_name, condition, fidelity = next(iter(cell))
    spec = METRICS[metric]
    raw = [spec.value(r) for r in results]
    values = [v for v in raw if v is not None]
    excluded = len(raw) - len(values)
    if not values:
        raise ValueError(f"{metric}: no episode of {arm_name!r} has a value ({excluded} excluded)")
    if any(math.isnan(v) for v in values):
        raise ValueError(f"{metric}: a per-episode value is NaN")
    width = spec.hi - spec.lo
    arr = np.asarray(values, dtype=np.float64)
    clipped = int(np.sum((arr < spec.lo) | (arr > spec.hi)))
    unit = (np.clip(arr, spec.lo, spec.hi) - spec.lo) / width
    lower, upper = betting_cs(unit, alpha)
    ci_lo, ci_hi = float(lower[-1]), float(upper[-1])
    empty = ci_lo > ci_hi
    if empty:
        # The running intersection of the sets is empty: the true mean was excluded at some time,
        # an event of probability at most alpha. Report the set at the last time on its own, which
        # is still a confidence set for that time, and say so.
        raw_lo, raw_hi = betting_cs(unit, alpha, intersect=False)
        ci_lo, ci_hi = float(raw_lo[-1]), float(raw_hi[-1])
        if ci_lo > ci_hi:
            ci_lo = ci_hi = float(unit.mean())
    seeds = sorted(r.seed for r in results)
    extra: dict[str, float | int | str] = {
        "excluded": excluded,
        "clipped": clipped,
        "episodes": len(results),
        "range_lo": spec.lo,
        "range_hi": spec.hi,
        "arm": arm_name,
        "condition": condition,
    }
    if empty:
        extra["cs_intersection_empty"] = 1
    if spec.proportion:
        k = int(round(float(arr.sum())))
        extra["k"] = k
        if k == 0:
            extra["zero_event_upper"] = zero_event_upper(len(values), alpha)
    return ResultRecord(
        name=f"{arm_name}.{condition}.{metric}",
        value=float(spec.lo + width * unit.mean()),
        n=len(values),
        fidelity=fidelity,
        measured_on=(
            f"{len(values)} of {len(results)} episodes of arm {arm_name}, {condition} condition, "
            f"seeds {seeds[0]} to {seeds[-1]}: {spec.describe}"
        ),
        script=SCRIPT,
        ci_lo=float(spec.lo + width * ci_lo),
        ci_hi=float(spec.lo + width * ci_hi),
        ci_kind=f"confseq-{alpha:g}",
        extra=extra,
    )
