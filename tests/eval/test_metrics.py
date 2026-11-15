"""``eval.metrics``: M-VIOL, the A5 attribution, M-COV, M-COST and ``aggregate`` on hand-built data.

Every trace here is built by hand from ``Trace.empty`` so that each test names the one thing it
breaks. Rows follow the plant's convention: row ``r`` is the state at the end of step ``r`` and its
truth columns are read at the start of step ``r + 1``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from closed_loop.core.results import ResultRecord
from closed_loop.core.trace import Trace, op_index
from closed_loop.core.types import Cmd, Command, Fidelity, Op
from closed_loop.eval.episode import EpisodeResult
from closed_loop.eval.interlock import ViolationLag
from closed_loop.eval.metrics import (
    COST_CAP,
    METRICS,
    WINDOW_Q,
    AssumptionBreaks,
    Attribution,
    EpisodeCoverage,
    ViolationEvent,
    aggregate,
    assumption_breaks,
    attribute,
    cost_of,
    coverage_of,
    crosscheck,
    is_defect,
    latency_report,
    pooled_contract,
    spec_report,
)
from closed_loop.perception.contract import ContractCounts
from closed_loop.perception.labels import Checkpoint
from closed_loop.plant.params import ASSET_IDS, CRUSHER, PUMPS
from closed_loop.spec.properties import SpecParams, evaluate
from closed_loop.stats import betting_cs, zero_event_upper

SPEC = SpecParams()
N = 400


def quiet(n: int = N) -> Trace:
    return Trace.empty(n, ASSET_IDS)


def result(**kw: Any) -> EpisodeResult:
    base: dict[str, Any] = {
        "arm": "A",
        "seed": 1,
        "condition": "nominal",
        "fidelity": Fidelity.F1,
        "violations": {},
        "min_robustness": {p: 1.0 for p in ("P1", "P2", "P3", "P4", "P5", "P6")},
        "throughput": 1.0,
        "cost": 0.0,
        "unplanned_failures": 0,
        "planned_jobs": 0,
        "shield_overrides": 0,
        "shield_admitted": 0,
        "monitor_fired_day": None,
        "labels": (),
        "wall_s": 0.0,
    }
    return EpisodeResult(**{**base, **kw})


# ----------------------------------------------------------------------------- M-VIOL


class TestSpecReport:
    def test_a_quiet_trace_violates_nothing(self) -> None:
        rep = spec_report(quiet(), SPEC, CRUSHER, PUMPS)
        assert rep.events == ()
        assert all(v > 0 for v in rep.min_robustness.values())
        assert rep.counts == dict.fromkeys(("P1", "P2", "P3", "P4", "P5", "P6"), 0)

    def test_the_minimum_is_the_evaluate_of_the_specification(self) -> None:
        tr = quiet()
        for a in PUMPS:
            tr.op[a][100:110] = op_index(Op.ISO)
        tr.uncontrolled_failure[CRUSHER][50] = True
        tr.op[CRUSHER][200:210] = op_index(Op.WORK)  # working on an unproven asset
        rep = spec_report(tr, SPEC, CRUSHER, PUMPS)
        want = evaluate(tr, SPEC, CRUSHER, PUMPS)
        assert rep.min_robustness == pytest.approx(want)
        assert {p for p, v in want.items() if v < 0} == {"P1", "P4", "P5"}

    def test_events_are_runs_of_negative_robustness_and_each_has_its_first_row(self) -> None:
        tr = quiet()
        for a in PUMPS:
            tr.op[a][100:110] = op_index(Op.ISO)
            tr.op[a][300:305] = op_index(Op.ISO)
        rep = spec_report(tr, SPEC, CRUSHER, PUMPS)
        p4 = [e for e in rep.events if e.prop == "P4"]
        assert p4 == [ViolationEvent("P4", 100), ViolationEvent("P4", 300)]
        assert rep.counts["P4"] == 2

    def test_events_come_in_time_order(self) -> None:
        tr = quiet()
        tr.uncontrolled_failure[CRUSHER][250] = True
        tr.op[CRUSHER][100:102] = op_index(Op.WORK)
        rep = spec_report(tr, SPEC, CRUSHER, PUMPS)
        assert [e.row for e in rep.events] == sorted(e.row for e in rep.events)
        assert [(e.prop, e.row) for e in rep.events][:2] == [("P1", 100), ("P5", 250)]

    def test_an_uncontrolled_failure_is_one_event_at_its_row(self) -> None:
        tr = quiet()
        tr.uncontrolled_failure[CRUSHER][77] = True
        rep = spec_report(tr, SPEC, CRUSHER, PUMPS)
        assert rep.events == (ViolationEvent("P5", 77),)


class TestCrosscheck:
    def test_agreement_is_silent(self) -> None:
        ok = {"P1": 1.0, "P2": -2.0, "P3": 1.0, "P4": 1.0, "P5": 1.0, "P6": 1.0}
        assert crosscheck(ok, {"P2": 3}) == {}
        assert crosscheck({p: 1.0 for p in ok}, {}) == {}

    def test_each_disagreement_is_named_with_its_direction(self) -> None:
        spec = {"P1": 1.0, "P2": -2.0, "P3": 1.0, "P4": 1.0, "P5": 1.0, "P6": 1.0}
        assert crosscheck(spec, {"P1": 2}) == {"P1": "plant_only", "P2": "spec_only"}

    def test_a_property_missing_from_the_spec_report_counts_as_satisfied(self) -> None:
        assert crosscheck({}, {"P3": 1}) == {"P3": "plant_only"}


# ----------------------------------------------------------------------------- attribution


class TestAssumptionBreaks:
    def test_a_quiet_trace_breaks_nothing(self) -> None:
        b = assumption_breaks(quiet(), crusher=CRUSHER, pumps=PUMPS)
        assert not any(b.flat(name).any() for name in ("A_exec", "A_phys", "C1"))

    def test_a_refused_stop_or_exit_breaks_a_exec_at_its_quarter(self) -> None:
        tr = quiet()
        tr.commands.append((5, Command(CRUSHER, Cmd.STOP), False, False))
        tr.commands.append((9, Command(CRUSHER, Cmd.EXIT), False, False))
        tr.commands.append((12, Command(CRUSHER, Cmd.STOP), True, False))  # executed: fine
        tr.commands.append((14, Command(CRUSHER, Cmd.START), False, False))  # refusable: fine
        b = assumption_breaks(tr, crusher=CRUSHER, pumps=PUMPS)
        assert np.flatnonzero(b.a_exec[CRUSHER]).tolist() == [5, 9]
        assert not b.a_exec[PUMPS[0]].any()  # the breaks are kept per asset

    def test_a_bin_that_skips_breaks_a_phys_at_the_row_of_the_jump(self) -> None:
        tr = quiet()
        tr.health_bin[PUMPS[0]][10:] = 2  # 0 -> 2 between rows 9 and 10
        b = assumption_breaks(tr, crusher=CRUSHER, pumps=PUMPS)
        assert np.flatnonzero(b.a_phys[PUMPS[0]]).tolist() == [10]
        assert not b.a_phys[CRUSHER].any() and not b.a_phys[PUMPS[1]].any()

    def test_a_rise_of_one_is_allowed(self) -> None:
        tr = quiet()
        tr.health_bin[CRUSHER][10:] = 1
        tr.health_bin[CRUSHER][20:] = 2
        assert not assumption_breaks(tr, crusher=CRUSHER, pumps=PUMPS).flat("A_phys").any()

    def test_a_failure_from_below_the_top_bin_breaks_a_phys_and_from_the_top_does_not(self) -> None:
        tr = quiet()
        tr.health_bin[CRUSHER][:] = 1
        tr.uncontrolled_failure[CRUSHER][30] = True
        tr.health_bin[PUMPS[1]][:] = 3
        tr.uncontrolled_failure[PUMPS[1]][40] = True
        b = assumption_breaks(tr, crusher=CRUSHER, pumps=PUMPS)
        assert np.flatnonzero(b.a_phys[CRUSHER]).tolist() == [30]
        assert not b.a_phys[PUMPS[1]].any()

    def test_the_conveyor_is_outside_the_proof(self) -> None:
        tr = quiet()
        tr.health_bin["CV-101"][10:] = 3
        tr.set_hi["CV-101"][:] = 0
        b = assumption_breaks(tr, crusher=CRUSHER, pumps=PUMPS)
        assert not b.flat("A_phys").any() and not b.flat("C1").any()
        assert "CV-101" not in b.a_phys and "CV-101" not in b.c1  # only A_exec reads the conveyor
        assert "CV-101" in b.a_exec

    def test_c1_compares_the_bin_entering_the_step_with_the_set_the_shield_was_given(self) -> None:
        tr = quiet()
        tr.health_bin[CRUSHER][10:] = 2  # row 10 onwards: truth read at the start of step 11
        tr.set_hi[CRUSHER][:] = 1  # the set says bins 0..1 throughout
        b = assumption_breaks(tr, crusher=CRUSHER, pumps=PUMPS)
        # step t starts in the bin of row t - 1, so the first step to start in bin 2 is t = 11
        assert np.flatnonzero(b.c1[CRUSHER]).tolist() == list(range(11, N))

    def test_c1_holds_when_the_set_contains_the_bin(self) -> None:
        tr = quiet()
        tr.health_bin[CRUSHER][:] = 2
        tr.set_lo[CRUSHER][:] = 2
        tr.set_hi[CRUSHER][:] = 3
        assert not assumption_breaks(tr, crusher=CRUSHER, pumps=PUMPS).flat("C1").any()

    def test_c1_is_read_on_each_pump_too(self) -> None:
        tr = quiet()
        tr.health_bin[PUMPS[1]][5:8] = 3
        tr.set_hi[PUMPS[1]][:] = 2
        b = assumption_breaks(tr, crusher=CRUSHER, pumps=PUMPS)
        assert np.flatnonzero(b.c1[PUMPS[1]]).tolist() == [6, 7, 8]
        assert not b.c1[PUMPS[0]].any() and not b.c1[CRUSHER].any()

    def test_the_first_step_has_no_predecessor_and_breaks_nothing(self) -> None:
        tr = quiet()
        tr.health_bin[CRUSHER][0] = 3
        tr.set_hi[CRUSHER][:] = 0
        assert not assumption_breaks(tr, crusher=CRUSHER, pumps=PUMPS).c1[CRUSHER][0]

    def test_after_the_monitor_has_fired_the_shield_had_the_whole_range(self) -> None:
        tr = quiet()
        tr.health_bin[CRUSHER][10:] = 2
        tr.set_hi[CRUSHER][:] = 1
        b = assumption_breaks(tr, crusher=CRUSHER, pumps=PUMPS, fallback_row=100)
        assert np.flatnonzero(b.c1[CRUSHER]).tolist() == list(range(11, 100))


def breaks_at(
    n: int,
    *,
    a_exec: list[int] = (),  # type: ignore[assignment]
    a_phys: list[int] = (),  # type: ignore[assignment]
    c1: list[int] = (),  # type: ignore[assignment]
    pump_a_exec: list[int] = (),  # type: ignore[assignment]
    pump_a_phys: list[int] = (),  # type: ignore[assignment]
    pump_c1: list[int] = (),  # type: ignore[assignment]
    conveyor_a_exec: list[int] = (),  # type: ignore[assignment]
    p1: list[int] = (),  # type: ignore[assignment]
    p1_pump: list[int] = (),  # type: ignore[assignment]
    p1_conveyor: list[int] = (),  # type: ignore[assignment]
) -> AssumptionBreaks:
    """Breaks built by hand, per asset: the unprefixed names are the crusher's, ``pump_`` is the
    first pump, ``p1*`` mark the rows at which that asset's conjunct of P1 is negative."""

    def row(rows: list[int]) -> np.ndarray:
        arr = np.zeros(n, dtype=bool)
        arr[list(rows)] = True
        return arr

    conveyor = "CV-101"
    return AssumptionBreaks(
        a_exec={CRUSHER: row(a_exec), PUMPS[0]: row(pump_a_exec), PUMPS[1]: row([]),
                conveyor: row(conveyor_a_exec)},
        a_phys={CRUSHER: row(a_phys), PUMPS[0]: row(pump_a_phys), PUMPS[1]: row([])},
        c1={CRUSHER: row(c1), PUMPS[0]: row(pump_c1), PUMPS[1]: row([])},
        crusher=CRUSHER,
        pumps=PUMPS,
        p1_open={CRUSHER: row(p1), PUMPS[0]: row(p1_pump), PUMPS[1]: row([]),
                 conveyor: row(p1_conveyor)},
    )  # fmt: skip


class TestAttribute:
    def test_no_break_in_the_look_back_is_assumption_held(self) -> None:
        out = attribute([ViolationEvent("P3", 500)], breaks_at(1000))
        assert out == (Attribution("P3", 500, "assumption_held"),)
        assert out[0].defect and out[0].broken_row is None

    def test_a_break_names_the_assumption_and_its_tick(self) -> None:
        out = attribute([ViolationEvent("P5", 500)], breaks_at(1000, c1=[480]))
        assert out == (Attribution("P5", 500, "C1", 480),)
        assert not out[0].defect and out[0].asset == CRUSHER

    def test_the_first_break_in_time_wins_not_the_last(self) -> None:
        out = attribute([ViolationEvent("P5", 500)], breaks_at(1000, c1=[300], a_phys=[450]))
        assert out[0].label == "C1" and out[0].broken_row == 300
        out = attribute([ViolationEvent("P5", 500)], breaks_at(1000, c1=[450], a_exec=[300]))
        assert out[0].label == "A_exec" and out[0].broken_row == 300

    def test_a_tie_in_time_goes_to_a_exec_then_a_phys_then_c1(self) -> None:
        assert (
            attribute([ViolationEvent("P5", 50)], breaks_at(100, c1=[40], a_phys=[40]))[0].label
            == "A_phys"
        )
        assert (
            attribute(
                [ViolationEvent("P5", 50)], breaks_at(100, c1=[40], a_phys=[40], a_exec=[40])
            )[0].label
            == "A_exec"
        )

    def test_the_look_back_is_30_days_inclusive_of_both_ends(self) -> None:
        assert WINDOW_Q == 30 * 96
        row = 4000
        inside = attribute([ViolationEvent("P5", row)], breaks_at(5000, c1=[row - WINDOW_Q]))
        outside = attribute([ViolationEvent("P5", row)], breaks_at(5000, c1=[row - WINDOW_Q - 1]))
        at_tick = attribute([ViolationEvent("P5", row)], breaks_at(5000, a_phys=[row]))
        after = attribute([ViolationEvent("P5", row)], breaks_at(5000, a_phys=[row + 1]))
        assert inside[0].label == "C1"
        assert outside[0].label == "assumption_held"
        assert at_tick[0].label == "A_phys"
        assert after[0].label == "assumption_held"

    def test_an_early_violation_looks_back_to_the_start_only(self) -> None:
        out = attribute([ViolationEvent("P5", 10)], breaks_at(100, c1=[0]))
        assert out[0].label == "C1" and out[0].broken_row == 0

    def test_every_event_is_attributed_separately(self) -> None:
        events = [ViolationEvent("P1", 100), ViolationEvent("P4", 5000)]
        out = attribute(events, breaks_at(6000, a_exec=[90], p1=[100]))
        assert [a.label for a in out] == ["A_exec", "assumption_held"]

    def test_end_to_end_a_shielded_style_trace_with_a_missed_bin(self) -> None:
        tr = quiet()
        for row, h in ((100, 1), (101, 2), (102, 3)):  # one bin at a time: A_phys holds
            tr.health_bin[CRUSHER][row:] = h
        tr.set_hi[CRUSHER][:] = 2  # the set misses bin 3 from the step that starts in it
        tr.uncontrolled_failure[CRUSHER][150] = True  # from the top bin: not an A_phys failure
        rep = spec_report(tr, SPEC, CRUSHER, PUMPS)
        brk = assumption_breaks(tr, crusher=CRUSHER, pumps=PUMPS)
        got = attribute(rep.events, brk)
        assert [(a.prop, a.label) for a in got] == [("P5", "C1")]
        assert got[0].broken_row == 103


class TestOnlyAComponentsOwnAssumptionsExcuseIt:
    """0003 A15 and prereg 5.1: an assumption excuses only a property that depends on it, and only
    on the component concerned."""

    @pytest.mark.parametrize("prop", ["P1", "P2", "P3", "P6"])
    def test_a_c1_break_and_an_a_phys_break_cannot_excuse_a_property_that_reads_no_health(
        self, prop: str
    ) -> None:
        brk = breaks_at(1000, c1=[480], a_phys=[480], pump_c1=[480], pump_a_phys=[480], p1=[500])
        out = attribute([ViolationEvent(prop, 500)], brk)
        assert out[0].label == "assumption_held" and is_defect(prop, out[0].label) == (prop != "P6")

    def test_a_c1_break_with_p1_p2_and_p3_events_stays_assumption_held(self) -> None:
        # the regression of the review: a synthetic call labelled all three as C1 failures
        events = [ViolationEvent(p, 500) for p in ("P1", "P2", "P3")]
        out = attribute(events, breaks_at(1000, c1=[480], p1=[500]))
        assert [a.label for a in out] == ["assumption_held"] * 3
        assert all(a.defect for a in out)

    @pytest.mark.parametrize("prop", ["P2", "P3", "P6"])
    def test_a_refused_stop_or_exit_on_the_crusher_does_excuse_the_crusher_properties(
        self, prop: str
    ) -> None:
        out = attribute([ViolationEvent(prop, 500)], breaks_at(1000, a_exec=[480]))
        assert out[0].label == "A_exec" and out[0].asset == CRUSHER

    def test_a_refused_command_on_a_pump_does_not_excuse_a_crusher_property(self) -> None:
        for prop in ("P2", "P3", "P6"):
            out = attribute([ViolationEvent(prop, 500)], breaks_at(1000, pump_a_exec=[480]))
            assert out[0].label == "assumption_held", prop

    def test_a_pumps_c1_miss_does_not_excuse_a_crusher_event(self) -> None:
        out = attribute([ViolationEvent("P5", 500)], breaks_at(1000, pump_c1=[480]))
        assert out[0].label == "assumption_held"
        out = attribute([ViolationEvent("P5", 500)], breaks_at(1000, pump_c1=[480], c1=[490]))
        assert out[0].label == "C1" and out[0].broken_row == 490  # the crusher's own, later

    def test_the_crushers_c1_miss_does_not_excuse_a_pump_event(self) -> None:
        assert (
            attribute([ViolationEvent("P4", 500)], breaks_at(1000, c1=[480], a_phys=[480]))[0].label
            == "assumption_held"
        )
        out = attribute([ViolationEvent("P4", 500)], breaks_at(1000, pump_c1=[480]))
        assert out[0].label == "C1" and out[0].asset == PUMPS[0]
        out = attribute(
            [ViolationEvent("P4", 500)], breaks_at(1000, pump_a_phys=[470], pump_c1=[480])
        )
        assert out[0].label == "A_phys" and out[0].broken_row == 470

    def test_p1_is_excused_only_on_the_asset_the_violating_conjunct_names(self) -> None:
        # the crusher's conjunct is the negative one; a refused STOP on a pump does not excuse it
        brk = breaks_at(1000, pump_a_exec=[480], p1=[500])
        assert attribute([ViolationEvent("P1", 500)], brk)[0].label == "assumption_held"
        # the pump's conjunct is the negative one; the same refused STOP now does
        brk = breaks_at(1000, pump_a_exec=[480], p1_pump=[500])
        out = attribute([ViolationEvent("P1", 500)], brk)
        assert out[0].label == "A_exec" and out[0].asset == PUMPS[0]
        # the conveyor is a P1 asset too
        brk = breaks_at(1000, conveyor_a_exec=[480], p1_conveyor=[500])
        out = attribute([ViolationEvent("P1", 500)], brk)
        assert out[0].label == "A_exec" and out[0].asset == "CV-101"

    def test_a_p1_event_where_no_conjunct_is_negative_has_no_excuse(self) -> None:
        out = attribute([ViolationEvent("P1", 500)], breaks_at(1000, a_exec=[480]))
        assert out[0].label == "assumption_held"

    def test_end_to_end_a_c1_break_with_p1_p2_p3_events_is_still_a_set_of_defects(self) -> None:
        tr = quiet(600)
        tr.health_bin[CRUSHER][100:] = 2
        tr.set_hi[CRUSHER][:] = 1  # C1 broken on the crusher from row 101
        tr.op[CRUSHER][300:305] = op_index(Op.WORK)  # P1: working, isolation not proven
        tr.occupied[CRUSHER][300:305] = True  # P2: no permit; P3: no gas test
        tr.gas_age_min[CRUSHER][:] = 600.0
        rep = spec_report(tr, SPEC, CRUSHER, PUMPS)
        brk = assumption_breaks(tr, crusher=CRUSHER, pumps=PUMPS)
        got = {a.prop: a for a in attribute(rep.events, brk)}
        assert brk.c1[CRUSHER].any() and {"P1", "P2", "P3"} <= set(got)
        assert all(got[p].label == "assumption_held" for p in ("P1", "P2", "P3"))

    def test_end_to_end_a_pumps_c1_miss_does_not_excuse_a_crusher_failure(self) -> None:
        tr = quiet(600)
        tr.health_bin[PUMPS[0]][100:] = 2
        tr.set_hi[PUMPS[0]][:] = 1  # C1 broken on a pump
        tr.health_bin[CRUSHER][:] = 3  # already critical, so the failure is not an early one
        tr.uncontrolled_failure[CRUSHER][300] = True  # P5, the crushers own
        rep = spec_report(tr, SPEC, CRUSHER, PUMPS)
        brk = assumption_breaks(tr, crusher=CRUSHER, pumps=PUMPS)
        got = attribute(rep.events, brk)
        assert brk.c1[PUMPS[0]].any() and [(a.prop, a.label) for a in got] == [
            ("P5", "assumption_held")
        ]


class TestIsDefect:
    @pytest.mark.parametrize("prop", ["P1", "P2", "P3", "P4", "P5"])
    def test_an_unexplained_violation_of_p1_to_p5_is_a_defect(self, prop: str) -> None:
        assert is_defect(prop, "assumption_held")
        for label in ("A_exec", "A_phys", "C1"):
            assert not is_defect(prop, label)

    def test_p6_is_attributed_but_is_not_in_the_defect_rule(self) -> None:
        assert not is_defect("P6", "assumption_held")

    def test_the_metric_and_the_attribution_use_the_same_rule(self) -> None:
        held_p6 = result(attributions=(Attribution("P6", 10, "assumption_held"),))
        held_p2 = result(attributions=(Attribution("P2", 10, "assumption_held"),))
        assert METRICS["m_defect"].value(held_p6) == 0.0
        assert METRICS["m_defect"].value(held_p2) == 1.0
        assert METRICS["m_defect_p13"].value(held_p2) == 1.0
        assert not Attribution("P6", 10, "assumption_held").defect


# ----------------------------------------------------------------------------- M-COV, M-COST


def cp(day: int, lo: float, hi: float, truth: float, status: str = "estimated") -> Checkpoint:
    return Checkpoint(
        0, CRUSHER, day, lo, 0.5 * (lo + hi), hi, 1.0, 0.9, status, true_rul=truth, health_bin=0
    )


class TestCoverage:
    def test_pooled_and_trend_strata_exclude_the_blind_window(self) -> None:
        cps = [
            cp(0, 0.0, 10.0, 50.0, "prior"),  # blind: outside every population
            cp(30, 5.0, 15.0, 10.0, "not_degrading"),  # post-blind, prior interval, covered
            cp(31, 5.0, 15.0, 40.0, "insufficient"),  # post-blind, prior, missed above
            cp(32, 5.0, 15.0, 10.0, "estimated"),  # trend, covered
            cp(33, 5.0, 15.0, 2.0, "estimated"),  # trend, missed below (lower bound missed too)
        ]
        c = coverage_of(cps)
        assert (c.n, c.n_trend, c.n_blind, c.n_total) == (4, 2, 1, 5)
        assert c.c == pytest.approx(2 / 4)
        assert c.c_trend == pytest.approx(1 / 2)
        # the lower-bound version only asks L >= lo: 3 of 4 pooled, 1 of 2 in the stratum
        assert c.c_lo == pytest.approx(3 / 4)
        assert c.c_lo_trend == pytest.approx(1 / 2)

    def test_the_interval_edges_are_inside(self) -> None:
        c = coverage_of([cp(30, 5.0, 15.0, 5.0, "estimated"), cp(31, 5.0, 15.0, 15.0, "estimated")])
        assert c.c == 1.0 and c.c_lo == 1.0

    def test_an_episode_with_no_post_blind_checkpoint_has_no_value_and_is_counted(self) -> None:
        c = coverage_of([cp(0, 0.0, 10.0, 5.0, "prior"), cp(1, 0.0, 10.0, 5.0, "prior")])
        assert (c.c, c.c_lo, c.c_trend, c.c_lo_trend) == (None, None, None, None)
        assert (c.n, c.n_blind, c.n_total) == (0, 2, 2)

    def test_an_episode_with_no_trend_checkpoint_keeps_its_pooled_value(self) -> None:
        c = coverage_of([cp(30, 5.0, 15.0, 10.0, "not_degrading")])
        assert c.c == 1.0 and c.c_trend is None and c.n_trend == 0

    def test_no_checkpoint_at_all(self) -> None:
        c = coverage_of([])
        assert c == EpisodeCoverage(None, None, 0, None, None, 0, 0, 0)

    def test_the_truth_is_required(self) -> None:
        bare = Checkpoint(0, CRUSHER, 30, 5.0, 10.0, 15.0, 1.0, 0.9, "estimated")
        with pytest.raises(ValueError, match="no sim-only truth"):
            coverage_of([bare])


class TestCost:
    def test_the_formula_of_the_preregistration(self) -> None:
        assert cost_of(3, 1, 100.0, 4, 5) == pytest.approx(
            1.0 * 3 + 8.0 * 1 + 0.05 * 4 + 0.02 * 5 + 0.02 * 100.0
        )
        assert cost_of(0, 0, 0.0, 0, 0) == 0.0

    def test_a_planned_job_is_one_unit_and_a_failure_eight(self) -> None:
        assert cost_of(1, 0, 0.0, 0, 0) == 1.0 and cost_of(0, 1, 0.0, 0, 0) == 8.0
        assert cost_of(0, 0, 50.0, 0, 0) == pytest.approx(1.0)

    def test_an_inspection_and_a_rejected_order_are_priced(self) -> None:
        """0003 A18 R7: M-COST used to leave out the two things the plant charges for besides jobs
        and failures, so a controller could inspect and re-submit for nothing."""
        assert cost_of(0, 0, 0.0, 20, 0) == pytest.approx(1.0)
        assert cost_of(0, 0, 0.0, 0, 50) == pytest.approx(1.0)

    def test_the_constants_are_the_plants_own(self) -> None:
        from closed_loop.eval.metrics import COST_INSPECTION, COST_REJECTED_WO
        from closed_loop.plant.params import PlantParams

        params = PlantParams()
        assert COST_INSPECTION == params.inspect_cost
        assert COST_REJECTED_WO == params.rejected_wo_cost

    @pytest.mark.parametrize(
        ("seed", "inspected", "rejected"),
        [(1103, True, False), (1105, False, True)],
        ids=["an inspection", "a rejected order"],
    )
    def test_the_metric_equals_the_plants_own_counter_on_a_real_episode(
        self, small_context: Any, seed: int, inspected: bool, rejected: bool
    ) -> None:
        """The plant adds up its own cost as it goes (jobs, failures, inspections, rejections); the
        metric is that plus the lost quarters. A term missing or swapped in ``cost_of`` shows here:
        the two episodes are chosen so that each of the two new terms is the only one present."""
        from closed_loop.eval.arms import ARMS
        from closed_loop.eval.episode import scenario_for, simulate_episode
        from closed_loop.eval.metrics import COST_LOST_QUARTER

        rec = simulate_episode(ARMS["P1"], seed, scenario_for("nominal"), context=small_context)
        stats = rec.plant.stats()
        assert (stats.inspections > 0, stats.wo_rejected > 0) == (inspected, rejected)
        assert rec.result.cost == pytest.approx(
            stats.cost + COST_LOST_QUARTER * rec.result.lost_quarters, abs=1e-9
        )


# ----------------------------------------------------------------------------- aggregate


class TestAggregate:
    def cell(self, flags: list[bool], **common: Any) -> list[EpisodeResult]:
        out = []
        for i, bad in enumerate(flags):
            mr = {p: 1.0 for p in ("P1", "P2", "P3", "P4", "P5", "P6")}
            if bad:
                mr["P2"] = -1.0
            out.append(result(seed=1000 + i, min_robustness=mr, **common))
        return out

    def test_a_proportion_has_its_count_and_a_confidence_sequence(self) -> None:
        flags = [True] * 5 + [False] * 15
        rec = aggregate(self.cell(flags), "m_viol_p13")
        assert isinstance(rec, ResultRecord)
        assert rec.value == pytest.approx(0.25) and rec.n == 20
        assert rec.extra["k"] == 5 and rec.extra["excluded"] == 0 and rec.extra["clipped"] == 0
        lo, hi = betting_cs(np.array([1.0] * 5 + [0.0] * 15), 0.05)
        assert (rec.ci_lo, rec.ci_hi) == pytest.approx((lo[-1], hi[-1]))
        assert rec.ci_kind == "confseq-0.05" and rec.fidelity is Fidelity.F1
        assert "20 of 20 episodes of arm A" in rec.measured_on and rec.script

    def test_group_p45_does_not_see_p2(self) -> None:
        rec = aggregate(self.cell([True] * 4 + [False] * 6), "m_viol_p45")
        assert rec.value == 0.0 and rec.extra["k"] == 0

    def test_zero_events_carry_the_zero_event_upper_bound(self) -> None:
        rec = aggregate(self.cell([False] * 50), "m_viol_p13")
        assert rec.value == 0.0
        assert rec.extra["zero_event_upper"] == pytest.approx(zero_event_upper(50, 0.05))

    def test_one_event_is_not_a_zero_event_cell(self) -> None:
        rec = aggregate(self.cell([True] + [False] * 49), "m_viol_p13")
        assert rec.extra["k"] == 1 and "zero_event_upper" not in rec.extra

    def test_alpha_is_a_parameter(self) -> None:
        cell = self.cell([True] * 10 + [False] * 30)
        wide, narrow = aggregate(cell, "m_viol_p13", 0.001), aggregate(cell, "m_viol_p13", 0.05)
        assert wide.ci_hi - wide.ci_lo > narrow.ci_hi - narrow.ci_lo  # type: ignore[operator]
        assert wide.ci_kind == "confseq-0.001"

    def test_an_empty_running_intersection_is_reported_and_flagged_not_hidden(self) -> None:
        # at alpha 0.2 this stream makes the intersection of the sets empty (probability <= alpha)
        cell = self.cell([True] * 10 + [False] * 30)
        rec = aggregate(cell, "m_viol_p13", 0.2)
        assert rec.extra["cs_intersection_empty"] == 1
        assert rec.ci_lo <= rec.ci_hi  # type: ignore[operator]
        raw_lo, raw_hi = betting_cs(np.array([1.0] * 10 + [0.0] * 30), 0.2, intersect=False)
        assert (rec.ci_lo, rec.ci_hi) == pytest.approx((raw_lo[-1], raw_hi[-1]))
        assert "cs_intersection_empty" not in aggregate(cell, "m_viol_p13", 0.05).extra

    def test_episodes_with_no_value_are_excluded_and_counted(self) -> None:
        cell = [
            result(seed=1, coverage=EpisodeCoverage(0.9, 0.95, 10, None, None, 0, 5, 15)),
            result(seed=2, coverage=EpisodeCoverage(None, None, 0, None, None, 0, 15, 15)),
            result(seed=3, coverage=EpisodeCoverage(0.7, 0.8, 10, 0.5, 0.6, 2, 5, 15)),
        ]
        pooled = aggregate(cell, "m_cov")
        assert pooled.value == pytest.approx(0.8) and pooled.n == 2
        assert pooled.extra["excluded"] == 1 and pooled.extra["episodes"] == 3
        trend = aggregate(cell, "m_cov_trend")
        assert trend.value == pytest.approx(0.5) and trend.n == 1 and trend.extra["excluded"] == 2

    def test_m_cost_is_capped_at_200_divided_by_200_and_the_clip_is_counted(self) -> None:
        cell = [result(seed=i, cost=c) for i, c in enumerate([50.0, 100.0, 250.0, 400.0])]
        rec = aggregate(cell, "m_cost")
        assert rec.value == pytest.approx((0.25 + 0.5 + 1.0 + 1.0) / 4)
        assert rec.extra["clipped"] == 2
        assert COST_CAP == 200.0

    def test_u_e_is_on_its_own_scale(self) -> None:
        rec = aggregate(
            [result(seed=i, unplanned_failures=k) for i, k in enumerate([0, 1, 2, 1])], "u_e"
        )
        assert rec.value == pytest.approx(1.0)
        assert 0.0 <= rec.ci_lo <= rec.value <= rec.ci_hi <= 8.0  # type: ignore[operator]

    def test_f_e_is_one_when_the_episode_had_a_failure_whatever_the_count(self) -> None:
        # 0003 A16: the safety leg of C-PLAN reads the share of episodes with an unplanned
        # failure, a proportion over [0, 1], not the count over [0, 8]
        counts = [0, 1, 2, 0, 5]
        rec = aggregate([result(seed=i, unplanned_failures=k) for i, k in enumerate(counts)], "f_e")
        assert rec.value == pytest.approx(3 / 5) and rec.extra["k"] == 3
        assert (METRICS["f_e"].lo, METRICS["f_e"].hi) == (0.0, 1.0) and METRICS["f_e"].proportion
        assert [METRICS["f_e"].value(result(unplanned_failures=k)) for k in (0, 1, 8)] == [0, 1, 1]
        assert "u_e" in METRICS  # the count is still a reported row

    def test_f_e_with_no_failure_has_its_zero_event_bound(self) -> None:
        rec = aggregate([result(seed=i, unplanned_failures=0) for i in range(50)], "f_e", 0.05 / 9)
        assert rec.value == 0.0 and rec.extra["k"] == 0 and "zero_event_upper" in rec.extra

    def test_throughput(self) -> None:
        rec = aggregate(
            [result(seed=i, throughput=t) for i, t in enumerate([0.5, 1.0])], "throughput"
        )
        assert rec.value == pytest.approx(0.75)

    def test_the_override_rate_is_per_thousand_proposed_commands(self) -> None:
        r = result(shield_overrides=30, commands_proposed=1500)
        assert METRICS["override_rate"].value(r) == pytest.approx(20.0)
        assert METRICS["override_rate"].value(result()) is None
        both = result(interlock_refused=10, interlock_forced=5, commands_proposed=500)
        assert METRICS["override_rate"].value(both) == pytest.approx(30.0)

    def test_permissiveness_is_allowed_commands_over_the_alphabet(self) -> None:
        r = result(allowed_crusher=(600, 100), allowed_pumps=(3000, 100))
        assert METRICS["permissiveness_crusher"].value(r) == pytest.approx(6.0 / 14)
        assert METRICS["permissiveness_pumps"].value(r) == pytest.approx(30.0 / 64)
        assert METRICS["permissiveness_crusher"].value(result()) is None

    def test_set_width_is_over_four_bins(self) -> None:
        r = result(contract=ContractCounts(lives=1, checkpoints=10, width_sum=30))
        assert METRICS["set_width"].value(r) == pytest.approx(0.75)

    def test_the_attribution_metrics_split_defects_from_contract_failures(self) -> None:
        held = result(attributions=(Attribution("P2", 10, "assumption_held"),))
        c1 = result(attributions=(Attribution("P5", 10, "C1", 5),))
        phys = result(attributions=(Attribution("P5", 10, "A_phys", 5),))
        none = result()
        cell = lambda *rs: [  # noqa: E731
            dataclass_replace(r, seed=i) for i, r in enumerate(rs)
        ]
        recs = cell(held, c1, phys, none)
        assert aggregate(recs, "m_defect").extra["k"] == 1
        assert aggregate(recs, "m_contract_failure").extra["k"] == 2
        assert aggregate(recs, "m_attr_c1").extra["k"] == 1
        assert aggregate(recs, "m_attr_a_phys").extra["k"] == 1
        assert aggregate(recs, "m_attr_a_exec").extra["k"] == 0

    # ---- 0003 A18 R6: M-VIOL counts the events the plant recorded, not the spec alone
    def test_a_plant_event_with_a_positive_spec_robustness_is_a_violation(self) -> None:
        """The second review found P2 events the plant recorded that the spec scored positive
        (the spec reads the trace at its own resolution); they now count."""
        r = result(plant_violations={"P2": 1})
        assert METRICS["m_viol_p13"].value(r) == 1.0
        assert METRICS["m_viol_p15"].value(r) == 1.0
        assert METRICS["m_viol_p45"].value(r) == 0.0  # the other group is not touched

    def test_a_plant_only_p1_counts_in_m_viol_p13_and_not_in_the_spec_only_row(self) -> None:
        """The review: H-C-I scored 0 of 20 by spec and 7 of 20 by the plant's P1 events."""
        r = result(plant_violations={"P1": 1})
        assert r.min_robustness["P1"] > 0.0
        assert METRICS["m_viol_p13"].value(r) == 1.0
        assert METRICS["m_viol_p13_spec"].value(r) == 0.0
        assert METRICS["m_disagree"].value(result(disagreements={"P1": "plant_only"})) == 1.0

    def test_a_plant_only_p13_event_is_the_part_the_spec_cannot_see(self) -> None:
        """0003 A18, F4: the part of M-VIOL (P1-P3) with no spec event to attribute."""
        plant_only = result(plant_violations={"P1": 1})
        assert METRICS["m_viol_p13_plant_only"].value(plant_only) == 1.0
        mr = {p: 1.0 for p in ("P1", "P2", "P3", "P4", "P5", "P6")} | {"P2": -1.0}
        seen = result(min_robustness=mr, plant_violations={"P2": 1})  # both see it
        assert METRICS["m_viol_p13"].value(seen) == 1.0
        assert METRICS["m_viol_p13_plant_only"].value(seen) == 0.0
        assert METRICS["m_viol_p13_plant_only"].value(result(min_robustness=mr)) == 0.0
        assert METRICS["m_viol_p13_plant_only"].value(result()) == 0.0
        assert METRICS["m_viol_p13_plant_only"].value(result(plant_violations={"P4": 1})) == 0.0

    def test_a_plant_event_in_group_p45_counts_in_that_group_only(self) -> None:
        r = result(plant_violations={"P4": 2})
        assert METRICS["m_viol_p45"].value(r) == 1.0
        assert METRICS["m_viol_p13"].value(r) == 0.0
        assert METRICS["m_viol_p15"].value(r) == 1.0

    def test_negative_robustness_still_counts_without_a_plant_event(self) -> None:
        mr = {p: 1.0 for p in ("P1", "P2", "P3", "P4", "P5", "P6")} | {"P3": -0.5}
        r = result(min_robustness=mr)
        assert METRICS["m_viol_p13"].value(r) == 1.0
        assert METRICS["m_viol_p13_spec"].value(r) == 1.0

    def test_the_spec_only_secondary_ignores_the_plant_events(self) -> None:
        r = result(plant_violations={"P2": 3})
        assert METRICS["m_viol_p13_spec"].value(r) == 0.0
        assert METRICS["m_viol_p13"].value(r) == 1.0

    def test_p6_and_a_zero_count_are_not_violations(self) -> None:
        assert METRICS["m_viol_p15"].value(result(plant_violations={"P6": 4})) == 0.0
        assert METRICS["m_viol_p13"].value(result(plant_violations={"P2": 0})) == 0.0

    def test_a_plant_only_event_is_not_a_defect(self) -> None:
        """Attribution is unchanged: only a spec event gets a label, so a plant-only event is a
        violation of M-VIOL and cannot be a defect."""
        r = result(plant_violations={"P2": 1})
        assert METRICS["m_defect"].value(r) == 0.0

    def test_the_attribution_of_p6_is_not_a_defect_of_m_viol(self) -> None:
        r = result(attributions=(Attribution("P6", 10, "assumption_held"),))
        assert METRICS["m_defect"].value(r) == 0.0

    def test_a_disagreement_is_reported_as_one(self) -> None:
        r = result(disagreements={"P2": "spec_only"})
        assert METRICS["m_disagree"].value(r) == 1.0
        assert METRICS["m_disagree"].value(result(disagreements={"P6": "plant_only"})) == 0.0

    def test_monitor_firing_and_detection_within_30_days_of_the_shift(self) -> None:
        shifted = result(monitors={"proxy": 52.0, "opt": None}, shift_day=40.0)
        assert METRICS["m_false_alarm_proxy"].value(shifted) == 1.0
        assert METRICS["m_false_alarm_opt"].value(shifted) == 0.0
        assert METRICS["m_detect_proxy"].value(shifted) == 1.0
        late = result(monitors={"proxy": 71.0}, shift_day=40.0)
        assert METRICS["m_detect_proxy"].value(late) == 0.0
        early = result(monitors={"proxy": 35.0}, shift_day=40.0)
        assert METRICS["m_detect_proxy"].value(early) == 0.0  # before the shift: a false alarm
        assert METRICS["m_detect_proxy"].value(result(monitors={"proxy": 5.0})) is None  # no shift
        assert METRICS["m_false_alarm_audit"].value(result()) is None  # no such monitor

    def test_the_detection_window_is_closed_at_both_ends(self) -> None:
        at_start = result(monitors={"proxy": 40.0}, shift_day=40.0)
        at_end = result(monitors={"proxy": 70.0}, shift_day=40.0)
        just_after = result(monitors={"proxy": 70.5}, shift_day=40.0)
        assert METRICS["m_detect_proxy"].value(at_start) == 1.0
        assert METRICS["m_detect_proxy"].value(at_end) == 1.0
        assert METRICS["m_detect_proxy"].value(just_after) == 0.0

    def test_a_mixed_cell_is_refused(self) -> None:
        with pytest.raises(ValueError, match="one cell"):
            aggregate([result(arm="A"), result(arm="B")], "throughput")
        with pytest.raises(ValueError, match="one cell"):
            aggregate([result(condition="nominal"), result(condition="shifted")], "throughput")

    def test_nonsense_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown metric"):
            aggregate([result()], "m_nothing")
        with pytest.raises(ValueError, match="at least one"):
            aggregate([], "throughput")
        with pytest.raises(ValueError, match="no episode"):
            aggregate([result()], "override_rate")

    def test_every_metric_evaluates_on_a_default_episode(self) -> None:
        r = result()
        for name, spec in METRICS.items():
            v = spec.value(r)
            assert v is None or math.isfinite(v), name

    def test_the_metric_names_the_brief_lists_exist(self) -> None:
        wanted = {
            "m_viol_p13",
            "m_viol_p45",
            "m_cov",
            "m_cov_lo",
            "m_cov_trend",
            "m_cov_lo_trend",
            "m_cost",
            "u_e",
            "f_e",
            "throughput",
            "override_rate",
            "permissiveness_crusher",
            "permissiveness_pumps",
            "set_width",
            "m_defect",
            "m_contract_failure",
        }
        assert wanted <= set(METRICS)


def dataclass_replace(r: EpisodeResult, **kw: Any) -> EpisodeResult:
    import dataclasses

    return dataclasses.replace(r, **kw)


# ----------------------------------------------------------------------------- latency, contract


class TestLatencyAndContract:
    def test_the_histogram_of_quarters_to_the_first_negative_online_robustness(self) -> None:
        lags = (
            ViolationLag("P4", "PU-201|PU-202", 10, 10),
            ViolationLag("P1", "CR-501", 20, 21),
            ViolationLag("P1", "CR-501", 30, 33),
            ViolationLag("P1", "CR-501", 40, 60),
            ViolationLag("P3", "CR-501", 50, None),
        )
        rep = latency_report([result(online_lags=lags), result()])
        assert rep["P4"]["0"] == 1
        assert rep["P1"] == {"0": 0, "1": 1, "2-4": 1, "5+": 1, "never": 0}
        assert rep["P3"]["never"] == 1
        assert sum(sum(h.values()) for h in rep.values()) == 5

    def test_the_edges_of_the_lag_bins(self) -> None:
        lags = tuple(ViolationLag("P1", "CR-501", 100, 100 + k) for k in (0, 1, 2, 4, 5))
        rep = latency_report([result(online_lags=lags)])["P1"]
        assert rep == {"0": 1, "1": 1, "2-4": 2, "5+": 1, "never": 0}

    def test_episodes_without_the_interlock_contribute_nothing(self) -> None:
        rep = latency_report([result(), result()])
        assert all(sum(h.values()) == 0 for h in rep.values())

    def test_pooled_contract_sums_counts_before_dividing(self) -> None:
        a = result(
            contract=ContractCounts(
                lives=2, lives_miss_rul=1, lives_miss_bin=0, checkpoints=10, miss_rul=1,
                miss_bin=0, width_sum=30,
            )
        )  # fmt: skip
        b = result(
            contract=ContractCounts(
                lives=6, lives_miss_rul=0, lives_miss_bin=3, checkpoints=30, miss_rul=0,
                miss_bin=2, width_sum=60,
            )
        )  # fmt: skip
        rep = pooled_contract([a, b])
        assert rep["lives"] == 8 and rep["checkpoints"] == 40
        assert rep["any_miss_rul"] == pytest.approx(1 / 8)
        assert rep["any_miss_bin"] == pytest.approx(3 / 8)
        assert rep["step_miss_bin"] == pytest.approx(2 / 40)
        assert rep["mean_width_bins"] == pytest.approx(90 / 40)

    def test_pooled_contract_of_nothing_is_nan_not_zero(self) -> None:
        rep = pooled_contract([result()])
        assert math.isnan(rep["any_miss_rul"]) and math.isnan(rep["mean_width_bins"])


# ----------------------------------------------------------------------------- P12b additions


def label(status: str, *, prior: bool = False, blind: bool = False, lo: float = 5.0,
          day: int = 30, end: float = 60.0) -> Any:  # fmt: skip
    from closed_loop.perception.labels import Label, LabelStatus

    return Label(
        life_id=0, asset_id="PU-201", checkpoint_day=day, lo=lo, hi=lo + 10.0, level=0.9,
        resolved_day=61, status=LabelStatus(status), end_day=end, censored=False, audit=False,
        prior=prior, blind=blind,
    )  # fmt: skip


class TestLearnerCounts:
    def test_counts_by_status_for_the_pooled_and_trend_populations(self) -> None:
        from closed_loop.eval.metrics import learner_counts

        labels = [
            label("HIT"), label("HIT"), label("MISS"), label("LOWER_HIT"), label("UNKNOWN"),
            label("HIT", prior=True),  # a prior-status label: pooled, not the trend stratum
            label("HIT", prior=True, blind=True),  # inside the blind window: in neither
        ]  # fmt: skip
        c = learner_counts(labels)
        assert c["post_blind.n"] == 6 and c["estimated.n"] == 5
        assert (c["post_blind.hit"], c["post_blind.miss"]) == (3, 1)
        assert (c["post_blind.lower_hit"], c["post_blind.unknown"]) == (1, 1)
        assert (c["estimated.hit"], c["estimated.miss"]) == (2, 1)

    def test_lower_ok_counts_the_informative_labels_whose_lower_bound_held(self) -> None:
        from closed_loop.eval.metrics import learner_counts

        held = label("HIT", lo=5.0, day=30, end=60.0)  # 30 + 5 <= 60
        broke = label("MISS", lo=40.0, day=30, end=60.0)  # 30 + 40 > 60
        unknown = label("UNKNOWN", lo=5.0, day=30, end=60.0)  # not informative
        assert learner_counts([held, broke, unknown])["post_blind.lower_ok"] == 1

    def test_the_counts_reproduce_the_learner_coverage_of_the_labels(self) -> None:
        from closed_loop.eval.metrics import learner_counts
        from closed_loop.perception.coverage import learner_coverage

        labels = [label(s) for s in ("HIT", "HIT", "MISS", "LOWER_HIT", "UNKNOWN", "HIT")]
        c = learner_counts(labels)
        lc = learner_coverage(labels, "post_blind")
        assert c["post_blind.hit"] / (
            c["post_blind.hit"] + c["post_blind.miss"] + c["post_blind.lower_hit"]
        ) == pytest.approx(lc["c_lrn"])

    def test_no_labels_is_all_zeros(self) -> None:
        from closed_loop.eval.metrics import learner_counts

        assert set(learner_counts([]).values()) == {0}

    def test_an_episode_result_carries_the_counts_and_a_default_of_none(self) -> None:
        assert result().learner == {} and result().failure == ""


class TestUncensoredCounter:
    """0003 A18, F1: a counter of the labels of lives that ended in an observed failure."""

    def test_counts_the_labels_whose_life_ended_in_a_failure_per_population(self) -> None:
        import dataclasses

        from closed_loop.eval.metrics import learner_counts

        cens = dataclasses.replace(label("LOWER_HIT"), censored=True)
        cens_miss = dataclasses.replace(
            label("MISS"), censored=True
        )  # censored past the upper bound
        labels = [
            label("HIT"), label("MISS"), cens, cens_miss,
            dataclasses.replace(label("UNKNOWN"), censored=True),
            label("HIT", prior=True),  # pooled, not the trend stratum
            label("HIT", prior=True, blind=True),  # in neither
        ]  # fmt: skip
        c = learner_counts(labels)
        # the uncensored ones are the HIT and the MISS that saw the failure; a MISS may be censored
        assert c["post_blind.uncensored"] == 3 and c["estimated.uncensored"] == 2
        assert c["post_blind.miss"] == 2  # one of the two MISS labels is censored: the status alone
        # does not say it, which is why the counter exists

    def test_a_policy_that_prevents_every_failure_has_none(self) -> None:
        import dataclasses

        from closed_loop.eval.metrics import learner_counts

        labels = [dataclasses.replace(label(s), censored=True) for s in ("LOWER_HIT", "UNKNOWN")]
        assert learner_counts(labels)["post_blind.uncensored"] == 0
        assert learner_counts([])["post_blind.uncensored"] == 0


class TestTheGasClockRows:
    """0003 A18, F5, on hand-built traces: the job stand and the late tests."""

    LIMIT = 30.0  # retest_q 2 x 15 minutes

    def trace(self, inside: list[tuple[int, int]], ages: dict[int, float]) -> Trace:
        tr = Trace.empty(20, ASSET_IDS)
        for a, b in inside:
            tr.occupied[CRUSHER][a:b] = True
        for q, age in ages.items():
            tr.gas_age_min[CRUSHER][q] = age
        return tr

    def test_the_crusher_work_rows_count_work_and_nothing_else(self) -> None:
        from closed_loop.eval.metrics import crusher_work_rows

        tr = Trace.empty(20, ASSET_IDS)
        tr.op[CRUSHER][3:9] = op_index(Op.WORK)
        tr.op[CRUSHER][9:12] = op_index(Op.ISO)
        tr.op["PU-201"][0:10] = op_index(Op.WORK)  # another asset: not counted
        assert crusher_work_rows(tr) == 6
        assert crusher_work_rows(Trace.empty(20, ASSET_IDS)) == 0

    def test_per_job_it_is_the_rows_over_the_jobs_and_excluded_without_one(self) -> None:
        assert (
            METRICS["m_crusher_work_q"].value(result(crusher_work_q=46, crusher_jobs_done=2))
            == 23.0
        )
        assert (
            METRICS["m_crusher_work_q"].value(result(crusher_work_q=160, crusher_jobs_done=0))
            is None
        )
        assert METRICS["m_crusher_work_q"].value(result()) is None  # the default: no job

    def test_a_test_a_quarter_late_with_someone_inside_is_one_late_test(self) -> None:
        from closed_loop.eval.metrics import late_gas_tests

        # inside rows 4..8; age 30 at row 6 (due), 15 at row 7 (the test passed a quarter late)
        tr = self.trace([(4, 9)], {4: 0.0, 5: 15.0, 6: 30.0, 7: 15.0, 8: 30.0})
        assert late_gas_tests(tr, 2) == 1

    def test_the_first_boundary_is_inclusive_an_age_equal_to_the_period_is_due(self) -> None:
        from closed_loop.eval.metrics import late_gas_tests

        due = self.trace([(4, 8)], {4: 29.999, 5: 15.0})
        assert late_gas_tests(due, 2) == 0  # a hair under the period a quarter ago: not due
        at = self.trace([(4, 8)], {4: self.LIMIT, 5: 15.0})
        assert late_gas_tests(at, 2) == 1

    def test_the_second_boundary_is_strict_an_age_equal_to_the_period_now_is_not_a_pass(
        self,
    ) -> None:
        from closed_loop.eval.metrics import late_gas_tests

        still = self.trace([(4, 8)], {4: 45.0, 5: self.LIMIT})
        assert late_gas_tests(still, 2) == 0  # the age did not fall below the period
        passed = self.trace([(4, 8)], {4: 45.0, 5: 29.999})
        assert late_gas_tests(passed, 2) == 1

    def test_someone_must_be_inside_at_both_rows(self) -> None:
        from closed_loop.eval.metrics import late_gas_tests

        ages = {4: 45.0, 5: 0.0}
        assert late_gas_tests(self.trace([(5, 8)], ages), 2) == 0  # inside only now
        assert late_gas_tests(self.trace([(3, 5)], ages), 2) == 0  # inside only a quarter ago
        assert late_gas_tests(self.trace([], ages), 2) == 0
        assert late_gas_tests(self.trace([(3, 8)], ages), 2) == 1

    def test_the_period_is_the_one_given(self) -> None:
        from closed_loop.eval.metrics import late_gas_tests

        tr = self.trace([(4, 8)], {4: 30.0, 5: 0.0})
        assert late_gas_tests(tr, 2) == 1
        assert late_gas_tests(tr, 3) == 0  # 45 minutes: 30 is not yet due
        assert late_gas_tests(tr, 1) == 1

    def test_each_late_row_counts_once(self) -> None:
        from closed_loop.eval.metrics import late_gas_tests

        ages = {4: 30.0, 5: 0.0, 6: 15.0, 7: 30.0, 8: 15.0}
        assert late_gas_tests(self.trace([(3, 10)], ages), 2) == 2  # rows 5 and 8

    def test_the_other_assets_are_not_read(self) -> None:
        from closed_loop.eval.metrics import late_gas_tests

        tr = Trace.empty(20, ASSET_IDS)
        tr.occupied["PU-201"][4:8] = True
        tr.gas_age_min["PU-201"][4] = 60.0
        tr.gas_age_min["PU-201"][5] = 0.0
        assert late_gas_tests(tr, 2) == 0

    def test_the_metric_reads_the_episode_count(self) -> None:
        assert METRICS["m_late_gas_tests"].value(result(late_gas_tests=3)) == 3.0
        assert METRICS["m_late_gas_tests"].value(result()) == 0.0


class TestJobMetrics:
    def test_wrong_mode_jobs_and_planned_jobs_are_per_episode_counts(self) -> None:
        r = result(wrong_mode_jobs=2, planned_jobs=5)
        assert METRICS["wrong_mode_jobs"].value(r) == 2.0
        assert METRICS["planned_jobs"].value(r) == 5.0
        agg = aggregate([r, result(wrong_mode_jobs=0, planned_jobs=1)], "wrong_mode_jobs")
        assert agg.value == 1.0 and agg.extra["clipped"] == 0
