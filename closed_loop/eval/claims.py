"""Claim verdicts: one function per claim of the pre-registration, section 7 (packet P12b).

A verdict is **win**, **loss** or **inconclusive**, with the numbers it used. A loss and an
inconclusive result carry the same weight as a win (prereg section 7): the same record, the same
place in the report. A claim whose inputs were not produced (a stage that is not built, a run cut
short) is **not run**, which is a fourth state and not a quiet inconclusive: the report prints it
and the run does not call itself complete.

**A loss is a bound that shows the miss** (prereg section 7, 0003 A15). An interval too wide to
decide is inconclusive, never a loss, so that no rule can turn a lack of data into a verdict
against a component; every verdict function is tested on a perfect input (a win) and on a sample
too small to decide (inconclusive).

Each function takes the numbers it needs as ``Est`` values (an estimate with its confidence-sequence
interval at the level the claim is judged at, ``counts.ALPHA_PRIMARY`` = 0.05 / 9, or 0.05 / 18 for
the two C-PERC verdicts) and returns a ``Verdict``. They are pure: ``verdicts_from_records``
rebuilds every verdict from the ``ResultRecord`` list of ``results.json`` alone, which is what
``run.py --check`` does to see that no number or verdict was edited by hand.

Readings that the pre-registration leaves open, taken here and stated in each verdict's ``notes``:

* A loss rule is a veto. Where a win and a loss condition both hold (C-ADAPT: the interval of the
  paired difference clears 0.02 while the upper bound of coverage in ``nominal`` is below 0.92), the
  verdict is a loss.
* C-PERC is decided on ``Gap_B`` in the ``nominal`` condition. A win needs the interval of V1 (the
  split method, calibrated on open-loop lives) to exclude [-0.03, +0.03]; a loss needs the
  intervals of V1 **and** of the primary variant to lie inside it; anything else is inconclusive.
* C-PLAN is judged in ``nominal``; the ``shifted`` rows are secondary and use the same rule.
* "Coverage of Vx at least 0.92" (C-ADAPT) is read on the lower bound of the cell mean for a win
  and on its upper bound for a loss, like every other rule of section 7.
* C-SHIELD's vacuity label (0003 A13) applies when any shielded arm of a pair (H-C1 against H-C0,
  H-G1 against H-G0, P1 against P0) has less than half the mean throughput of its unshielded twin.
  The interlock "matches" the shield on a proposer when both bounds of the paired difference in
  M-VIOL (P1-P3), interlock minus shield, lie inside +-0.05 (the registered decision margin for
  proportions); it is "separated" when a bound is beyond 0.05 and then the verdict says which one
  is better; otherwise it says that they cannot be separated and which way the estimate leans.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

from closed_loop.core.results import ResultRecord
from closed_loop.core.types import Fidelity
from closed_loop.eval.counts import ALPHA_DESCRIPTIVE, ALPHA_PERC, ALPHA_PRIMARY
from closed_loop.stats.intervals import clopper_pearson

__all__ = [
    "ADAPT_VARIANTS",
    "COMPLETED",
    "INCONCLUSIVE",
    "INTERLOCK_MARGIN",
    "LOSS",
    "NOT_RUN",
    "SHIELD_PAIRS",
    "WIN",
    "Est",
    "SearchPairs",
    "Verdict",
    "c_adapt",
    "c_cause",
    "c_diag",
    "c_mon",
    "c_perc",
    "c_plan",
    "c_search",
    "c_shield",
    "c_together",
    "claim_name",
    "decide_pair",
    "est_of",
    "verdict_records",
    "verdicts_from_records",
]

WIN: Final[str] = "win"
LOSS: Final[str] = "loss"
INCONCLUSIVE: Final[str] = "inconclusive"
NOT_RUN: Final[str] = "not run"
COMPLETED: Final[str] = "completed"  # C-TOGETHER has no win: it completes or it does not
VACUOUS: Final[str] = "vacuous: the plant is mostly stopped"
SCRIPT: Final[str] = "closed_loop.eval.claims"

PERC_BAND: Final[float] = 0.03  # Gap_B inside +-0.03 is "no gap"
ADAPT_MARGIN: Final[float] = 0.02
ADAPT_COVERAGE: Final[float] = 0.92  # nominal 0.95 minus the tolerance 0.03
CAUSE_MARGIN: Final[float] = 1.0 / 30.0  # one day on the 30-day scale of M-CF
DIAG_MARGIN: Final[float] = 0.02
PLAN_MARGIN: Final[float] = 0.02
PLAN_FAILURES: Final[float] = 0.1  # on f_e, the share of episodes with an unplanned failure
MON_FALSE_ALARM: Final[float] = 0.02
MON_DETECT_WIN: Final[float] = 0.5
MON_DETECT_LOSS: Final[float] = 0.1
SHIELD_VIOL_FLOOR: Final[float] = 0.05  # unshielded P1-P3 violation rate must clear this
VACUITY_RATIO: Final[float] = 0.5  # 0003 A13
INTERLOCK_MARGIN: Final[float] = 0.05
SEARCH_SHARE: Final[float] = 0.5  # CMA-ES's share of the decided run pairs: above 0.5 a win

# The variants C-ADAPT reports (the verdict is the primary variant's, the rest are secondary rows).
ADAPT_VARIANTS: Final[tuple[str, ...]] = ("V2", "V3", "V4", "V5", "V6", "V2L", "V3L")
# (shielded, unshielded, interlock) arms of the three pairs C-SHIELD compares.
SHIELD_PAIRS: Final[Mapping[str, tuple[str, str, str]]] = {
    "H-C": ("H-C1", "H-C0", "H-C-I"),
    "H-G": ("H-G1", "H-G0", "H-G-I"),
    "P": ("P1", "P0", "P1-I"),
}
WIN_ARMS: Final[tuple[str, ...]] = ("H-C1", "H-G1")  # the arms whose zero count defines the win
UNSHIELDED_ARMS: Final[tuple[str, ...]] = ("H-C0", "H-G0")


# 0003 A18: printed beside the throughput ratios of C-SHIELD.
H_G_RATIO_NOTE: Final[str] = (
    "H-G ratio: the unshielded H-G arm obeys a note that begins work on a running asset with no "
    "work order, and an asset in WORK with no work order has no legal way out; that is what the "
    "hostile arm does, not a stall of a safety layer. It keeps the H-G pair's unshielded "
    "throughput low and the ratio high, so a ratio above 1 for H-G does not say the shield helped"
)


def fmt(x: float) -> str:
    return f"{x:.4f}"


def pct_pair(est: Est) -> str:
    return f"{fmt(est.value)} (interval {fmt(est.lo)} to {fmt(est.hi)})"


@dataclass(frozen=True)
class Est:
    """An estimate and its confidence-sequence interval at the level the claim is judged at."""

    value: float
    lo: float
    hi: float
    n: int
    k: int | None = None  # the count behind a proportion
    failed: int = 0  # episodes that failed and were counted at their worst value

    def __post_init__(self) -> None:
        if math.isnan(self.value) or math.isnan(self.lo) or math.isnan(self.hi):
            raise ValueError("an estimate cannot be NaN")
        if self.lo > self.hi:
            raise ValueError(f"interval is empty: {self.lo} > {self.hi}")


@dataclass(frozen=True)
class Verdict:
    claim: str
    label: str  # win | loss | inconclusive | not run
    qualifier: str  # "" or the A13 vacuity text
    headline: str
    rule: str  # the registered rule, in words
    alpha: float
    numbers: Mapping[str, float | int | str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        return self.label + (f", {self.qualifier}" if self.qualifier else "")


def _est_numbers(prefix: str, est: Est) -> dict[str, float | int | str]:
    out: dict[str, float | int | str] = {
        f"{prefix}.value": est.value,
        f"{prefix}.lo": est.lo,
        f"{prefix}.hi": est.hi,
        f"{prefix}.n": est.n,
    }
    if est.k is not None:
        out[f"{prefix}.k"] = est.k
    if est.failed:
        out[f"{prefix}.failed"] = est.failed
    return out


def _failed_note(*ests: Est) -> tuple[str, ...]:
    n = sum(e.failed for e in ests)
    return (
        (
            f"FAILED EPISODES: {n} failed or timed-out episodes are counted at their worst value "
            "in the cells behind this verdict (tried twice if they timed out or lost a worker), "
            "so the verdict reads a run that did not complete every episode",
        )
        if n
        else ()
    )


# ----------------------------------------------------------------------------- C-PERC


def c_perc(stratum: str, gap_v1: Est, gap_primary: Est, primary: str) -> Verdict:
    """C-PERC for one stratum (``pooled`` or ``trend``), judged at ``ALPHA_PERC``.

    ``Gap_B = C_orc(open loop) - C_orc(closed loop)`` per seed, ``nominal`` condition.
    """
    rule = (
        "win: the Gap_B interval of V1 excludes [-0.03, +0.03]; loss: the intervals of V1 and of "
        "the primary variant both lie inside it; otherwise inconclusive"
    )

    def outside(e: Est) -> bool:
        return e.lo > PERC_BAND or e.hi < -PERC_BAND

    def inside(e: Est) -> bool:
        return e.lo >= -PERC_BAND and e.hi <= PERC_BAND

    numbers = {**_est_numbers("gap_v1", gap_v1), **_est_numbers("gap_primary", gap_primary)}
    numbers["primary"] = primary
    where = f"{stratum} coverage, nominal condition"
    if outside(gap_v1):
        direction = "above" if gap_v1.lo > PERC_BAND else "below"
        label = WIN
        headline = (
            f"Gap_B for V1 on {where} is {pct_pair(gap_v1)}, {direction} the band of 0.03: "
            "open-loop calibration did not carry over to the closed loop"
        )
    elif inside(gap_v1) and inside(gap_primary):
        label = LOSS
        headline = (
            f"Gap_B on {where} is inside the band of 0.03 for V1 {pct_pair(gap_v1)} and for "
            f"{primary} {pct_pair(gap_primary)}: the thesis is not supported on this plant"
        )
    else:
        label = INCONCLUSIVE
        headline = (
            f"Gap_B on {where}: V1 {pct_pair(gap_v1)}, {primary} {pct_pair(gap_primary)}; "
            "neither outside the band for V1 nor inside it for both"
        )
    return Verdict(
        f"C-PERC.{stratum}", label, "", headline, rule, ALPHA_PERC, numbers,
        _failed_note(gap_v1, gap_primary),
    )  # fmt: skip


# ----------------------------------------------------------------------------- C-ADAPT


def c_adapt(
    variant: str, diff_shifted: Est, cov_shifted: Est, cov_nominal: Est, *, primary: bool = True
) -> Verdict:
    """C-ADAPT for one variant: paired ``c_e^lo(Vx) - c_e^lo(V1)`` under ``shifted``, coverage."""
    rule = (
        "win: lower bound of the paired difference at least 0.02 and lower bound of shifted "
        "c_e^lo at least 0.92; loss: upper bound of the paired difference below 0.02, or upper "
        "bound of nominal c_e^lo below 0.92"
    )
    no_better = diff_shifted.hi < ADAPT_MARGIN
    nominal_low = cov_nominal.hi < ADAPT_COVERAGE
    win_cond = diff_shifted.lo >= ADAPT_MARGIN and cov_shifted.lo >= ADAPT_COVERAGE
    numbers = {
        **_est_numbers("diff_shifted", diff_shifted),
        **_est_numbers("cov_shifted", cov_shifted),
        **_est_numbers("cov_nominal", cov_nominal),
    }
    notes = list(_failed_note(diff_shifted, cov_shifted, cov_nominal))
    if no_better or nominal_low:
        label = LOSS
        why = []
        if no_better:
            why.append(
                f"the upper bound of the paired difference is {fmt(diff_shifted.hi)}, below 0.02"
            )
        if nominal_low:
            why.append(
                f"the upper bound of nominal coverage c_e^lo is {fmt(cov_nominal.hi)}, below 0.92"
            )
        headline = f"{variant} against V1 under shifted: " + " and ".join(why)
        if win_cond:
            notes.append("the win conditions also held; the loss rule is a veto")
    elif win_cond:
        label = WIN
        headline = (
            f"{variant} against V1, shifted: paired c_e^lo difference {pct_pair(diff_shifted)}, "
            f"shifted c_e^lo {pct_pair(cov_shifted)}"
        )
    else:
        label = INCONCLUSIVE
        headline = (
            f"{variant} against V1, shifted: paired c_e^lo difference {pct_pair(diff_shifted)}, "
            f"shifted c_e^lo {pct_pair(cov_shifted)}, nominal c_e^lo {pct_pair(cov_nominal)}"
        )
    claim = "C-ADAPT" if primary else f"C-ADAPT.{variant}"
    # a secondary row is shown at the descriptive level (prereg section 6)
    alpha = ALPHA_PRIMARY if primary else ALPHA_DESCRIPTIVE
    return Verdict(claim, label, "", headline, rule, alpha, numbers, tuple(notes))


# ----------------------------------------------------------------------------- C-CAUSE, C-DIAG


def c_cause(q1_diff: Est) -> Verdict:
    """C-CAUSE: paired ``M-CF(scm) - M-CF(corr)`` (negative: the SCM is better). Favoured by
    construction, because the SCM is handed the structure of the plant."""
    rule = "win: upper bound of the paired difference below -1/30; loss: lower bound above 0"
    if q1_diff.hi < -CAUSE_MARGIN:
        label = WIN
    elif q1_diff.lo > 0.0:
        label = LOSS
    else:
        label = INCONCLUSIVE
    headline = (
        f"paired M-CF(scm) minus M-CF(corr) under confounded training is {pct_pair(q1_diff)} on "
        f"{q1_diff.n} test lives"
    )
    notes = (
        "favoured by construction: the structural model is handed the structure of the plant",
        *_failed_note(q1_diff),
    )
    return Verdict(
        "C-CAUSE", label, "", headline, rule, ALPHA_PRIMARY, _est_numbers("q1_diff", q1_diff), notes
    )


def c_diag(reversed_diff: Est, unreversed_diff: Est | None = None) -> Verdict:
    """C-DIAG: paired ``M-DIAG(scm) - M-DIAG(corr)`` on lives where the load-to-mode correlation of
    the training data is reversed. The unreversed result is reported with no decision rule."""
    rule = "win: lower bound of the paired difference above 0.02; loss: upper bound below 0"
    if reversed_diff.lo > DIAG_MARGIN:
        label = WIN
    elif reversed_diff.hi < 0.0:
        label = LOSS
    else:
        label = INCONCLUSIVE
    headline = (
        f"paired M-DIAG(scm) minus M-DIAG(corr) on reversed-correlation lives is "
        f"{pct_pair(reversed_diff)} over {reversed_diff.n} lives with an alert"
    )
    numbers = _est_numbers("reversed_diff", reversed_diff)
    notes = list(_failed_note(reversed_diff))
    if unreversed_diff is not None:
        numbers.update(_est_numbers("unreversed_diff", unreversed_diff))
        notes.append(
            f"unreversed, no decision rule: {pct_pair(unreversed_diff)} over "
            f"{unreversed_diff.n} lives (the correlational model may win there)"
        )
    return Verdict("C-DIAG", label, "", headline, rule, ALPHA_PRIMARY, numbers, tuple(notes))


# ----------------------------------------------------------------------------- C-PLAN


def c_plan(
    arm: str,
    baseline: str,
    cost_diff: Est,
    failure_diff: Est,
    condition: str,
    shifted: tuple[Est, Est] | None = None,
) -> Verdict:
    """C-PLAN for one planner arm against its B3 comparator.

    ``cost_diff`` is the paired ``M-COST(P) - M-COST(B3)`` and ``failure_diff`` the paired
    ``f_e(P) - f_e(B3)``, where ``f_e`` is 1 if the episode had at least one unplanned failure
    (0003 A16: the count of failures per episode has a range of 8 and a floor four times as wide,
    so the safety leg reads the incidence, on the betting sequence over [-1, 1]). The safety leg
    is read on the upper bound for a win (0003 A15: the lower bound let an interval of [0.09, 0.9]
    pass, which is backwards for a safety co-condition) and on the lower bound for a loss.
    ``u_e``, the count, is still a reported row.

    ``shifted`` is the paired cost and failure differences under the ``shifted`` condition. The
    verdict is decided on ``nominal`` and the shifted numbers decide nothing, but the headline
    prints them beside the nominal ones (0003 A18: the second review saw the sign of the planner's
    paired mean change under shift on 20 seeds, and a reader of the nominal verdict alone would not
    know).
    """
    rule = (
        "win: upper bound of the cost difference below -0.02 and upper bound of the paired "
        "difference in f_e (the share of episodes with an unplanned failure) at most 0.1; loss: "
        "lower bound of the cost difference above 0, or lower bound of the f_e difference above 0.1"
    )
    worse_failures = failure_diff.lo > PLAN_FAILURES
    safe = failure_diff.hi <= PLAN_FAILURES
    if cost_diff.lo > 0.0 or worse_failures:
        label = LOSS
    elif cost_diff.hi < -PLAN_MARGIN and safe:
        label = WIN
    else:
        label = INCONCLUSIVE
    headline = (
        f"{arm} against {baseline}, {condition}: paired M-COST difference {pct_pair(cost_diff)}, "
        f"paired difference in the share of episodes with an unplanned failure "
        f"{pct_pair(failure_diff)}"
    )
    if shifted is not None:
        headline += (
            f"; under shift: paired M-COST difference {pct_pair(shifted[0])}, paired difference "
            f"in the share of episodes with an unplanned failure {pct_pair(shifted[1])}"
        )
    claim = f"C-PLAN.{arm}" if condition == "nominal" else f"C-PLAN.{arm}.{condition}"
    numbers = {
        **_est_numbers("cost_diff", cost_diff),
        **_est_numbers("failure_diff", failure_diff),
    }
    if shifted is not None:
        numbers.update(_est_numbers("shifted.cost_diff", shifted[0]))
        numbers.update(_est_numbers("shifted.failure_diff", shifted[1]))
    alpha = ALPHA_PRIMARY if condition == "nominal" else ALPHA_DESCRIPTIVE  # shifted: secondary
    return Verdict(
        claim, label, "", headline, rule, alpha, numbers,
        _failed_note(cost_diff, failure_diff),
    )  # fmt: skip


# ----------------------------------------------------------------------------- C-MON


def c_mon(mode: str, false_alarm: Est, detection: Est, *, chosen: bool = True) -> Verdict:
    """C-MON for one monitor mode: false alarms on unshifted episodes, detection within 30 days."""
    rule = (
        "win: false-alarm upper bound at most 0.02 and detection lower bound at least 0.5; loss: "
        "false-alarm lower bound above 0.02, or detection upper bound below 0.1"
    )
    if false_alarm.lo > MON_FALSE_ALARM or detection.hi < MON_DETECT_LOSS:
        label = LOSS
    elif false_alarm.hi <= MON_FALSE_ALARM and detection.lo >= MON_DETECT_WIN:
        label = WIN
    else:
        label = INCONCLUSIVE
    k_fa = (
        f"{false_alarm.k} of {false_alarm.n}" if false_alarm.k is not None else str(false_alarm.n)
    )
    k_det = f"{detection.k} of {detection.n}" if detection.k is not None else str(detection.n)
    headline = (
        f"mode {mode}: false alarms {k_fa} unshifted episodes, upper bound {fmt(false_alarm.hi)}; "
        f"detected within 30 days {k_det} shifted episodes, lower bound {fmt(detection.lo)}"
    )
    numbers = {**_est_numbers("false_alarm", false_alarm), **_est_numbers("detection", detection)}
    claim = "C-MON" if chosen else f"C-MON.{mode}"
    alpha = ALPHA_PRIMARY if chosen else ALPHA_DESCRIPTIVE  # the other modes are secondary rows
    return Verdict(
        claim, label, "", headline, rule, alpha, numbers,
        _failed_note(false_alarm, detection),
    )  # fmt: skip


# ----------------------------------------------------------------------------- C-SEARCH


@dataclass(frozen=True)
class SearchPairs:
    """The run pairs of C-SEARCH in ``R_out`` (prereg section 7, 0003 A16).

    Run ``i`` of CMA-ES is paired with run ``i`` of random search (the same seed). An algorithm
    takes a pair if it found a violation and the other did not, or if both did and it needed fewer
    evaluations. A pair in which neither found anything, or both needed the same number of
    evaluations, is dropped and counted (``neither``, ``tied``).
    """

    cma: int  # pairs CMA-ES takes
    rand: int  # pairs random search takes
    neither: int = 0  # dropped: no violation found by either
    tied: int = 0  # dropped: both found, the same number of evaluations

    def __post_init__(self) -> None:
        if min(self.cma, self.rand, self.neither, self.tied) < 0:
            raise ValueError(f"a pair count cannot be negative: {self}")

    @property
    def decided(self) -> int:
        return self.cma + self.rand

    @property
    def total(self) -> int:
        return self.decided + self.neither + self.tied

    def share(self, alpha: float = ALPHA_PRIMARY) -> Est:
        """CMA-ES's share of the decided pairs with its exact (Clopper-Pearson) interval."""
        lo, hi = clopper_pearson(self.cma, self.decided, alpha)
        value = self.cma / self.decided if self.decided else 0.5
        return Est(value, lo, hi, self.decided, self.cma)


def decide_pair(cma_first: int | None, rand_first: int | None, budget: int) -> tuple[bool, bool]:
    """Who takes one pair: ``(cma, random)``, at most one True; ``(False, False)`` drops it.

    ``*_first`` is the number of the first violating evaluation, or None if there was none; a
    value beyond ``budget`` counts as none.
    """
    if cma_first is not None and cma_first <= budget:
        if rand_first is not None and rand_first <= budget:
            return cma_first < rand_first, rand_first < cma_first
        return True, False
    return False, rand_first is not None and rand_first <= budget


def c_search(
    pairs: SearchPairs,
    find_cma: Est,
    find_rand: Est,
    find_diff: Est,
    evals_diff: Est | None,
    r_in_violations: int | None,
) -> Verdict:
    """C-SEARCH in ``R_out``, decided on run pairs with an exact interval (0003 A16).

    With ten pairs a sequence over the find-rate difference is the whole range whatever the data,
    so the verdict reads the share of the decided pairs that CMA-ES takes, at the claim level
    ``ALPHA_PRIMARY``: with ten decided pairs only a clean sweep decides, either way. The find-rate
    difference (``find_diff``, ``find_cma``, ``find_rand``) and the evaluations difference
    (``evals_diff``, CMA-ES minus random, scaled to the budget, negative: CMA-ES faster) are
    descriptive numbers in the headline and decide nothing. ``r_in_violations`` is the number of
    ``R_in`` runs that found a defect, a violation of the shielded target that the attribution
    labels ``assumption_held`` (a violation with a broken assumption is a contract failure and is
    not counted); it is reported for C-SHIELD.
    """
    rule = (
        "win: the exact (Clopper-Pearson) lower bound of CMA-ES's share of the decided run pairs "
        "in R_out is above 0.5; loss: the exact upper bound of that share is below 0.5 (random "
        "search better); anything between, and no decided pair, is inconclusive. A pair is decided "
        "for the algorithm that found a violation when the other did not, or that needed fewer "
        "evaluations when both did; other pairs are dropped and counted"
    )
    share = pairs.share()
    if pairs.decided == 0:
        label = INCONCLUSIVE
    elif share.lo > SEARCH_SHARE:
        label = WIN
    elif share.hi < SEARCH_SHARE:
        label = LOSS
    else:
        label = INCONCLUSIVE
    notes: list[str] = []
    basis = (
        f"{pairs.total} run pairs: CMA-ES takes {pairs.cma}, random search takes {pairs.rand}, "
        f"{pairs.neither + pairs.tied} dropped (neither found a violation: {pairs.neither}; same "
        f"number of evaluations: {pairs.tied}); "
        + (
            f"CMA-ES's share of the {pairs.decided} decided pairs {pct_pair(share)}"
            if pairs.decided
            else "no pair is decided"
        )
    )
    headline = (
        f"R_out: CMA-ES found a violation in {find_cma.k} of {find_cma.n} runs, random search in "
        f"{find_rand.k} of {find_rand.n}; {basis}; descriptive, find-rate difference "
        f"{pct_pair(find_diff)}"
        + (
            f", evaluations to first violation, CMA-ES minus random {pct_pair(evals_diff)}"
            if evals_diff is not None
            else ""
        )
    )
    numbers = {
        **_est_numbers("find_cma", find_cma),
        **_est_numbers("find_rand", find_rand),
        **_est_numbers("find_diff", find_diff),
        **_est_numbers("share", share),
        "pairs.cma": pairs.cma,
        "pairs.random": pairs.rand,
        "pairs.neither": pairs.neither,
        "pairs.tied": pairs.tied,
    }
    if evals_diff is not None:
        numbers.update(_est_numbers("evals_diff", evals_diff))
    if r_in_violations is not None:
        numbers["r_in_violations"] = r_in_violations
        notes.append(
            f"R_in: runs with a defect: {r_in_violations}"
            + (
                " (an assumption_held violation of the shielded target, see C-SHIELD)"
                if r_in_violations
                else ""
            )
        )
    return Verdict("C-SEARCH", label, "", headline, rule, ALPHA_PRIMARY, numbers, tuple(notes))


# ----------------------------------------------------------------------------- C-TOGETHER


def c_together(
    completed: Mapping[str, tuple[int, int]],
    confirmed: tuple[int, int] | None,
    causes: Mapping[str, str] | None = None,
) -> Verdict:
    """C-TOGETHER: every F2 arm completes an episode, and the confirmed fraction is reported.

    ``completed`` maps an F2 arm to ``(episodes completed, episodes tried)``; ``confirmed`` is
    ``(replays showing the same violation, replays)``. There is no numerical threshold: a low
    fraction is a finding about F1, so the label is ``completed`` and never ``win``. Any arm that
    cannot complete an episode is an integration failure, reported with its cause, and the verdict
    is a loss.
    """
    rule = (
        "completed: the F2 episodes run end to end for every F2 arm and the confirmed fraction is "
        "reported with its interval, with no numerical threshold; loss: an F2 arm that cannot "
        "complete an episode (an integration failure)"
    )
    broken = sorted(a for a, (done, tried) in completed.items() if done == 0 or done < tried)
    numbers: dict[str, float | int | str] = {}
    for arm, (done, tried) in sorted(completed.items()):
        numbers[f"completed.{arm}"] = done
        numbers[f"tried.{arm}"] = tried
    notes: list[str] = []
    if confirmed is not None:
        numbers["confirmed.k"], numbers["confirmed.n"] = confirmed
    if not completed:
        label, headline = INCONCLUSIVE, "no F2 episode was tried"
    elif broken:
        label = LOSS
        headline = "F2 arms that did not complete every episode: " + ", ".join(
            f"{a} ({causes.get(a, 'cause not recorded') if causes else 'cause not recorded'})"
            for a in broken
        )
    elif confirmed is None or confirmed[1] == 0:
        label = COMPLETED
        headline = (
            "every F2 arm completed its episodes; the search left no find to replay, so the "
            "confirmed fraction is 0 of 0"
        )
    else:
        label = COMPLETED
        headline = (
            f"every F2 arm completed its episodes; {confirmed[0]} of {confirmed[1]} replays "
            "confirmed the F1 violation"
        )
    return Verdict("C-TOGETHER", label, "", headline, rule, ALPHA_PRIMARY, numbers, tuple(notes))


# ----------------------------------------------------------------------------- C-SHIELD


@dataclass(frozen=True)
class ShieldInputs:
    """What C-SHIELD reads. ``shielded`` and ``unshielded`` hold M-VIOL (P1-P3) of the hostile arms
    at the claim level; ``defects`` the number of episodes with an ``assumption_held`` violation of
    P1-P5 in each shielded arm (nominal); ``ratios`` the mean-throughput ratio of each pair."""

    n_max: int
    shielded: Mapping[str, Est]
    unshielded: Mapping[str, Est]
    defects: Mapping[str, int]
    ratios: Mapping[str, float]
    interlock: Mapping[str, tuple[Est, float, float, float, float]] = field(default_factory=dict)
    p45: Mapping[str, tuple[int, int]] = field(
        default_factory=dict
    )  # arm -> (contract failures, k)
    search_defects: int | None = None
    # arm -> episodes whose M-VIOL (P1-P3) is a plant event the spec did not see (0003 A18, F4)
    plant_only: Mapping[str, int] = field(default_factory=dict)


def c_shield(inp: ShieldInputs) -> Verdict:
    """C-SHIELD (prereg section 7, 0003 A5, A13 and A4)."""
    rule = (
        "win: M-VIOL (P1-P3) is 0 in all n_max episodes of H-C1 and H-G1, no violation of P1-P5 "
        "in a shielded arm is labelled assumption_held, and the lower bound of M-VIOL (P1-P3) of "
        "H-C0 and H-G0 exceeds 0.05; loss: any assumption_held violation of P1-P5 in a shielded "
        "arm; the verdict prints the shielded throughput as a ratio of the unshielded"
    )
    defect_arms = sorted(a for a, k in inp.defects.items() if k > 0)
    search_defect = bool(inp.search_defects)
    numbers: dict[str, float | int | str] = {"n_max": inp.n_max}
    for arm, est in {**inp.shielded, **inp.unshielded}.items():
        numbers.update(_est_numbers(f"viol_p13.{arm}", est))
    for arm, k in inp.defects.items():
        numbers[f"defects.{arm}"] = k
    for arm, k in inp.plant_only.items():
        numbers[f"plant_only.{arm}"] = k
    for pair, ratio in inp.ratios.items():
        numbers[f"throughput_ratio.{pair}"] = ratio
    if inp.search_defects is not None:
        numbers["search_defects"] = inp.search_defects

    notes: list[str] = []
    zero = {a: (e.k == 0 and e.n == inp.n_max and e.failed == 0) for a, e in inp.shielded.items()}
    teeth = {a: e.lo > SHIELD_VIOL_FLOOR for a, e in inp.unshielded.items()}
    vacuous = sorted(p for p, r in inp.ratios.items() if r < VACUITY_RATIO)
    missing_win = [a for a in WIN_ARMS if a not in inp.shielded]
    missing_unshielded = [a for a in UNSHIELDED_ARMS if a not in inp.unshielded]
    if defect_arms or search_defect:
        label = LOSS
        where = ", ".join(defect_arms) if defect_arms else "the R_in search"
        headline = (
            f"unexplained violation (an assumption_held label) of P1-P5 in a shielded arm: "
            f"{where}. "
            "A defect in the abstraction or the plant: the shield is unproven until it is fixed "
            "and the shielded set is re-run in a new registered version"
        )
    elif missing_win or missing_unshielded:
        label = NOT_RUN
        headline = f"cells missing: {', '.join(missing_win + missing_unshielded)}"
    elif all(zero.values()) and all(teeth.values()):
        label = WIN
        parts = [
            f"{a}: 0 of {inp.shielded[a].n}, upper bound {fmt(inp.shielded[a].hi)}"
            for a in WIN_ARMS
        ]
        headline = (
            "P1-P3 violations by the shielded hostile arms: "
            + "; ".join(parts)
            + "; unshielded: "
            + "; ".join(
                f"{a} {inp.unshielded[a].k} of {inp.unshielded[a].n}, lower bound "
                f"{fmt(inp.unshielded[a].lo)}"
                for a in UNSHIELDED_ARMS
            )
            + ". Zero events in a finite number of episodes bound the rate from above; they do "
            "not show that it is zero"
        )
    else:
        label = INCONCLUSIVE
        reasons = []
        for a, ok in zero.items():
            if not ok:
                e = inp.shielded[a]
                reasons.append(
                    f"{a} has {e.k} violating episodes of {e.n} (n_max {inp.n_max}, "
                    f"{e.failed} failed)" + _violation_kinds(a, e.k or 0, inp.plant_only)
                )
        for a, ok in teeth.items():
            if not ok:
                reasons.append(
                    f"{a} (shield off) has a lower bound of {fmt(inp.unshielded[a].lo)}, "
                    "not above 0.05: the comparison has no teeth"
                )
        headline = "; ".join(reasons)
    qualifier = ""
    if label == WIN and vacuous:
        qualifier = VACUOUS
    ratio_text = ", ".join(f"{p} {fmt(r)}" for p, r in sorted(inp.ratios.items()))
    notes.append(
        f"shielded mean throughput as a ratio of the same proposer unshielded: {ratio_text}"
        + (f"; below 0.5 for {', '.join(vacuous)}" if vacuous else "")
    )
    if "H-G" in inp.ratios:
        notes.append(H_G_RATIO_NOTE)
    notes.extend(_interlock_notes(inp, numbers))
    for arm, (contract, k) in sorted(inp.p45.items()):
        notes.append(
            f"{arm}: P4-P5 violations in {k} episodes, {contract} of them attributed to the "
            "contract (C1 or A_phys), reported as what the proof does not cover"
        )
    if inp.search_defects is None:
        notes.append(
            "the R_in search stage did not run; defects it would find are not counted here"
        )
    notes.append(
        "an episode that failed counts at its worst value; the stand-in carries no attribution, "
        "so a failure cannot make a defect"
    )
    return Verdict(
        "C-SHIELD", label, qualifier, headline, rule, ALPHA_PRIMARY, numbers, tuple(notes)
    )


def _violation_kinds(arm: str, k: int, plant_only: Mapping[str, int]) -> str:
    """What the violating episodes of a shielded arm are (0003 A18, F4). A violation with a spec
    event has a label; one that is not a defect is a broken assumption of the contract. A plant-only
    event has no spec event to label, so calling it a contract failure would claim an attribution
    nobody made."""
    if not k:
        return ""
    if arm not in plant_only:
        return "; these are contract failures, not defects"
    only = min(plant_only[arm], k)
    seen = k - only
    parts = []
    if seen:
        parts.append(
            f"{seen} attributed to a broken assumption of the contract "
            "(contract failures, not defects)"
        )
    if only:
        parts.append(
            f"{only} plant-only events (the plant recorded the event, the specification did not "
            "see it, so there is nothing to attribute: neither a defect label nor a contract "
            "failure)"
        )
    return "; of these " + " and ".join(parts)


def _interlock_notes(inp: ShieldInputs, numbers: dict[str, float | int | str]) -> list[str]:
    """The comparison row of 0003 A4: shield against the hand-written interlock."""
    if not inp.interlock:
        return ["the interlock arms were not run"]
    notes: list[str] = []
    verdicts: list[str] = []
    for pair, (diff, thr_s, thr_i, over_s, over_i) in sorted(inp.interlock.items()):
        numbers.update(_est_numbers(f"interlock_diff.{pair}", diff))
        numbers[f"throughput_shield.{pair}"] = thr_s
        numbers[f"throughput_interlock.{pair}"] = thr_i
        numbers[f"override_shield.{pair}"] = over_s
        numbers[f"override_interlock.{pair}"] = over_i
        if diff.lo > -INTERLOCK_MARGIN and diff.hi < INTERLOCK_MARGIN:
            word, better = "matches", ""
        elif diff.lo >= INTERLOCK_MARGIN:
            word, better = "is separated from", ": the shield is better"
        elif diff.hi <= -INTERLOCK_MARGIN:
            word, better = "is separated from", ": the interlock is better"
        else:
            lean = "the shield" if diff.value >= 0.0 else "the interlock"
            word, better = "cannot be separated from", f"; the estimate leans to {lean}"
        verdicts.append(word)
        notes.append(
            f"interlock against shield, {pair}: paired M-VIOL (P1-P3) difference interlock minus "
            f"shield {pct_pair(diff)}, so the interlock {word} the shield on violations{better}; "
            f"throughput {fmt(thr_i)} against {fmt(thr_s)}; overrides per 1,000 proposed commands "
            f"{fmt(over_i)} against {fmt(over_s)}"
        )
    if verdicts and all(v == "matches" for v in verdicts):
        notes.insert(0, "headline: the hand-written interlock matches the shield on M-VIOL (P1-P3)")
    return notes


# ----------------------------------------------------------------------------- records <-> inputs

_PREFIX: Final[str] = "claim"


def claim_name(claim: str, *parts: str) -> str:
    """The record name stages use for an input of ``claim``: ``claim/C-SHIELD/viol_p13/H-C1``."""
    return "/".join((_PREFIX, claim, *parts))


def est_of(rec: ResultRecord) -> Est:
    """An ``Est`` from a record at the claim level (its confidence sequence must be present)."""
    if rec.ci_lo is None or rec.ci_hi is None:
        raise ValueError(f"{rec.name}: a claim input needs a confidence interval")
    k = rec.extra.get("k")
    return Est(
        rec.value, rec.ci_lo, rec.ci_hi, rec.n,
        None if k is None else int(k), int(rec.extra.get("failed", 0)),
    )  # fmt: skip


def verdict_records(verdicts: Sequence[Verdict]) -> list[ResultRecord]:
    """One ``ResultRecord`` per verdict (value +1 win, -1 loss, 0 otherwise)."""
    out: list[ResultRecord] = []
    for v in verdicts:
        value = {WIN: 1.0, LOSS: -1.0}.get(v.label, 0.0)
        extra: dict[str, float | int | str] = {
            "label": v.label,
            "qualifier": v.qualifier,
            "headline": v.headline,
            "rule": v.rule,
            "alpha": v.alpha,
        }
        for i, note in enumerate(v.notes):
            extra[f"note.{i}"] = note
        for key, number in v.numbers.items():
            extra[f"in.{key}"] = number
        out.append(
            ResultRecord(
                name=f"verdict/{v.claim}",
                value=value,
                n=int(v.numbers.get("n_max", 0)) if isinstance(v.numbers.get("n_max"), int) else 0,
                fidelity=Fidelity.F1,
                measured_on=f"verdict of claim {v.claim} from the claim-level records of this run",
                script=SCRIPT,
                extra=extra,
            )
        )
    return out


def verdicts_from_stored(records: Sequence[ResultRecord]) -> list[Verdict]:
    """The verdicts as stored in ``results.json`` (to compare with ``verdicts_from_records``)."""
    out: list[Verdict] = []
    for rec in records:
        if not rec.name.startswith("verdict/"):
            continue
        ex = rec.extra
        notes = tuple(str(ex[k]) for k in sorted(ex, key=_note_order) if str(k).startswith("note."))
        numbers = {k[3:]: v for k, v in ex.items() if str(k).startswith("in.")}
        out.append(
            Verdict(
                rec.name.removeprefix("verdict/"),
                str(ex["label"]),
                str(ex["qualifier"]),
                str(ex["headline"]),
                str(ex["rule"]),
                float(ex["alpha"]),
                numbers,
                notes,
            )
        )
    return out


def _note_order(key: str) -> tuple[int, str]:
    return (int(key.split(".")[1]), key) if key.startswith("note.") else (-1, key)


Lookup = Callable[..., ResultRecord | None]


def _get(by_name: Mapping[str, ResultRecord], claim: str, *parts: str) -> ResultRecord | None:
    return by_name.get(claim_name(claim, *parts))


def _not_run(claim: str, why: str, alpha: float = ALPHA_PRIMARY) -> Verdict:
    return Verdict(claim, NOT_RUN, "", why, "inputs of this claim were not produced", alpha)


def verdicts_from_records(records: Sequence[ResultRecord]) -> list[Verdict]:
    """Every verdict, rebuilt from the claim-level records alone.

    Run parameters (the primary variant, the monitor mode chosen, ``n_max`` of the hostile cells)
    come from the record ``run/params``. A claim whose inputs are absent is ``not run``.
    """
    by_name = {r.name: r for r in records}
    params = by_name.get("run/params")
    if params is None:
        return [_not_run(c, "no run/params record") for c in _CLAIMS]
    primary = str(params.extra["primary_variant"])
    mode = str(params.extra["monitor_mode"])
    n_max_hostile = int(params.extra["n_max_hostile"])
    out: list[Verdict] = []

    # ---- C-PERC (two verdicts)
    for stratum in ("pooled", "trend"):
        v1 = _get(by_name, "C-PERC", "gap_b", stratum, "V1")
        pv = _get(by_name, "C-PERC", "gap_b", stratum, primary)
        if v1 is None or pv is None:
            out.append(_not_run(f"C-PERC.{stratum}", "Gap_B records missing", ALPHA_PERC))
        else:
            out.append(c_perc(stratum, est_of(v1), est_of(pv), primary))

    # ---- C-ADAPT: the primary variant is the verdict, the others are secondary rows
    def adapt(variant: str, is_primary: bool) -> Verdict:
        d = _get(by_name, "C-ADAPT", "diff_c_lo", variant)
        s = _get(by_name, "C-ADAPT", "cov_lo_shifted", variant)
        n = _get(by_name, "C-ADAPT", "cov_lo_nominal", variant)
        if d is None or s is None or n is None:
            return _not_run("C-ADAPT" if is_primary else f"C-ADAPT.{variant}", "records missing")
        return c_adapt(variant, est_of(d), est_of(s), est_of(n), primary=is_primary)

    if primary in ADAPT_VARIANTS:
        out.append(adapt(primary, True))
    else:
        out.append(_not_run("C-ADAPT", f"the primary variant {primary} is not a C-ADAPT candidate"))
    out.extend(adapt(v, False) for v in ADAPT_VARIANTS if v != primary)

    # ---- C-CAUSE, C-DIAG
    q1 = _get(by_name, "C-CAUSE", "q1", "scm-corr")
    out.append(c_cause(est_of(q1)) if q1 else _not_run("C-CAUSE", "records missing"))
    rev = _get(by_name, "C-DIAG", "reversed", "scm-corr")
    unrev = _get(by_name, "C-DIAG", "unreversed", "scm-corr")
    out.append(
        c_diag(est_of(rev), est_of(unrev) if unrev else None)
        if rev
        else _not_run("C-DIAG", "records missing")
    )

    # ---- C-PLAN: P0 against B3 (no shield), P1 against B3 with shield
    for arm, base in (("P0", f"B3-{primary}"), ("P1", "B3-S")):
        for condition in ("nominal", "shifted"):
            cost = _get(by_name, "C-PLAN", "cost_diff", arm, condition)
            fail = _get(by_name, "C-PLAN", "failure_diff", arm, condition)
            claim = f"C-PLAN.{arm}" if condition == "nominal" else f"C-PLAN.{arm}.{condition}"
            if cost is None or fail is None:
                out.append(_not_run(claim, "records missing"))
            else:
                shift = None
                if condition == "nominal":
                    sc = _get(by_name, "C-PLAN", "cost_diff", arm, "shifted")
                    sf = _get(by_name, "C-PLAN", "failure_diff", arm, "shifted")
                    shift = (est_of(sc), est_of(sf)) if sc is not None and sf is not None else None
                out.append(c_plan(arm, base, est_of(cost), est_of(fail), condition, shift))

    # ---- C-MON: the chosen mode is the verdict, all four are reported
    for m in ("opt", "pess", "inf", "proxy"):
        fa = _get(by_name, "C-MON", "false_alarm", m)
        det = _get(by_name, "C-MON", "detection", m)
        if fa is None or det is None:
            out.append(_not_run("C-MON" if m == mode else f"C-MON.{m}", "records missing"))
        else:
            out.append(c_mon(m, est_of(fa), est_of(det), chosen=(m == mode)))

    # ---- C-SEARCH
    fc = _get(by_name, "C-SEARCH", "find_rate", "R_out", "cma")
    fr = _get(by_name, "C-SEARCH", "find_rate", "R_out", "random")
    fd = _get(by_name, "C-SEARCH", "find_diff", "R_out")
    ed = _get(by_name, "C-SEARCH", "evals_diff", "R_out")
    pr = _get(by_name, "C-SEARCH", "pairs", "R_out")
    rin = _get(by_name, "C-SEARCH", "r_in_violations")
    if fc and fr and fd and pr:
        out.append(
            c_search(
                SearchPairs(
                    int(pr.extra["cma"]),
                    int(pr.extra["random"]),
                    int(pr.extra["neither"]),
                    int(pr.extra["tied"]),
                ),
                est_of(fc),
                est_of(fr),
                est_of(fd),
                est_of(ed) if ed else None,
                int(rin.value) if rin else None,
            )  # fmt: skip
        )
    else:
        out.append(_not_run("C-SEARCH", "the search stage was not built or did not run"))

    # ---- C-TOGETHER
    done = {
        r.name.rsplit("/", 1)[1]: (int(r.extra["completed"]), int(r.extra["tried"]))
        for r in records
        if r.name.startswith(claim_name("C-TOGETHER", "arm") + "/")
    }
    conf = by_name.get(claim_name("C-TOGETHER", "confirmed"))
    causes = {
        r.name.rsplit("/", 1)[1]: str(r.extra.get("cause", ""))
        for r in records
        if r.name.startswith(claim_name("C-TOGETHER", "arm") + "/")
    }
    if done:
        out.append(c_together(done, (int(conf.extra["k"]), conf.n) if conf else None, causes))
    else:
        out.append(_not_run("C-TOGETHER", "the F2 stage was not built or did not run"))

    # ---- C-SHIELD
    out.append(_shield_from_records(by_name, n_max_hostile))
    return out


_CLAIMS: Final[tuple[str, ...]] = (
    "C-PERC", "C-ADAPT", "C-CAUSE", "C-DIAG", "C-PLAN", "C-SHIELD", "C-MON", "C-SEARCH",
    "C-TOGETHER",
)  # fmt: skip


def _shield_from_records(by_name: Mapping[str, ResultRecord], n_max: int) -> Verdict:
    shielded = {a: by_name.get(claim_name("C-SHIELD", "viol_p13", a)) for a in WIN_ARMS}
    unshielded = {a: by_name.get(claim_name("C-SHIELD", "viol_p13", a)) for a in UNSHIELDED_ARMS}
    if any(v is None for v in (*shielded.values(), *unshielded.values())):
        return _not_run("C-SHIELD", "hostile-arm records missing")
    prefix = claim_name("C-SHIELD", "defects") + "/"
    defects = {
        r.name.removeprefix(prefix): int(r.extra["k"])
        for r in by_name.values()
        if r.name.startswith(prefix)
    }
    ratios = {
        r.name.rsplit("/", 1)[1]: r.value
        for r in by_name.values()
        if r.name.startswith(claim_name("C-SHIELD", "throughput_ratio") + "/")
    }
    interlock: dict[str, tuple[Est, float, float, float, float]] = {}
    for r in by_name.values():
        if r.name.startswith(claim_name("C-SHIELD", "interlock") + "/"):
            ex = r.extra
            interlock[r.name.rsplit("/", 1)[1]] = (
                est_of(r),
                float(ex["throughput_shield"]), float(ex["throughput_interlock"]),
                float(ex["override_shield"]), float(ex["override_interlock"]),
            )  # fmt: skip
    p45 = {}
    prefix45 = claim_name("C-SHIELD", "p45") + "/"
    for r in by_name.values():
        if r.name.startswith(prefix45):
            p45[r.name.removeprefix(prefix45)] = (int(r.extra["contract"]), int(r.extra["k"]))
    plant_only = {}
    prefix_po = claim_name("C-SHIELD", "plant_only") + "/"
    for r in by_name.values():
        if r.name.startswith(prefix_po):
            plant_only[r.name.removeprefix(prefix_po)] = int(r.extra.get("k", 0))
    search = by_name.get(claim_name("C-SHIELD", "search_r_in"))
    inp = ShieldInputs(
        n_max=n_max,
        shielded={a: est_of(r) for a, r in shielded.items() if r is not None},
        unshielded={a: est_of(r) for a, r in unshielded.items() if r is not None},
        defects=defects,
        ratios=ratios,
        interlock=interlock,
        p45=p45,
        search_defects=int(search.value) if search is not None else None,
        plant_only=plant_only,
    )
    return c_shield(inp)
