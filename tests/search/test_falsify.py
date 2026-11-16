"""P9b: the falsification loop, its objective, the R_in decoder and the findings it produces.

Most tests run the loop on a stand-in evaluation (a function of theta that builds the same
``Evaluation`` a plant episode would), so they cost nothing and can be exact about budgets, seeds
and labels. The tests marked by the ``small_context`` fixture run real F1 episodes on a small fitted
context; they are the ones that say the stand-in is not hiding anything.
"""

from __future__ import annotations

import dataclasses
import json
import math
import pickle
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from closed_loop.core.types import BIN_EDGES_DAYS
from closed_loop.eval.arms import CARELESS_P_SKIP, Arm
from closed_loop.eval.arms import arm as registry_arm
from closed_loop.eval.context import PreparedContext, use_context
from closed_loop.eval.episode import simulate_episode
from closed_loop.eval.metrics import Attribution, assumption_breaks
from closed_loop.plant.params import CRUSHER, PUMPS, PlantParams
from closed_loop.plant.scenario import (
    FAULT_DRIFT,
    FAULT_NONE,
    FAULT_SPIKE,
    FAULT_STUCK,
    THETA_DIM,
    Scenario,
)
from closed_loop.proposers import Careless, Gullible
from closed_loop.proposers.baselines import BoundRule
from closed_loop.search import falsify as F
from closed_loop.search.falsify import (
    Evaluation,
    Finding,
    Objective,
    build_finding,
    classify,
    collect_findings,
    dimensions_of,
    evaluate_theta,
    falsify,
    falsify_detailed,
    first_events_of,
    merge_documents,
    normalise,
    objective_of,
    property_scales,
    r_in_exceedances,
    sanity_arm,
    target_arm,
    violated_of,
    what_gets_through,
)
from closed_loop.search.random_search import SearchRun
from closed_loop.search.shrink import nominal_theta
from closed_loop.spec.properties import PROPERTY_IDS, SpecParams, evaluate
from tests.spec._helpers import CRUSHER as SPEC_CRUSHER
from tests.spec._helpers import PUMPS as SPEC_PUMPS
from tests.spec._helpers import clean_trace, random_trace, violating_trace

SPEC = SpecParams()
BIG = 1.0e9  # what an idle property reads (spec.properties.HOLDS_MARGIN)

thetas = arrays(
    np.float64,
    THETA_DIM,
    elements=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
)
corners = arrays(np.float64, THETA_DIM, elements=st.sampled_from([0.0, 1.0]))


def raw_robustness(**override: float) -> dict[str, float]:
    base = {"P1": 1.0, "P2": BIG, "P3": BIG, "P4": 1.5, "P5": 500.0, "P6": 1.0}
    base.update(override)
    return base


# ----------------------------------------------------------------------------- the objective


class TestObjective:
    def test_each_property_is_scaled_by_its_own_units(self) -> None:
        s = property_scales(SPEC)
        assert s == {
            "P1": 1.0,
            "P2": 720.0,  # 12 h of permit, in minutes
            "P3": 30.0,  # two quarters of re-test period, in minutes
            "P4": 0.5,
            "P5": BIN_EDGES_DAYS[-1] * 24.0,  # the edge of the critical bin, in hours
            "P6": 1.0,
        }

    def test_the_scales_follow_the_specification(self) -> None:
        stale = SpecParams(retest_q=8, permit_max_h=6.0)
        s = property_scales(stale)
        assert s["P3"] == 120.0 and s["P2"] == 360.0

    @pytest.mark.parametrize(
        ("pid", "raw", "expected"),
        [
            ("P1", -1.0, -1.0),
            ("P2", 360.0, 0.5),
            ("P2", -1.0, -1.0 / 720.0),
            ("P3", 15.0, 0.5),
            ("P3", -15.0, -0.5),
            ("P3", -300.0, -1.0),  # clipped
            ("P4", 0.5, 1.0),
            ("P4", -0.5, -1.0),
            ("P5", 36.0, 0.5),
            ("P5", -1.0, -1.0 / 72.0),
            ("P6", -1.0, -1.0),
        ],
    )
    def test_the_objective_is_the_smallest_normalised_property(
        self, pid: str, raw: float, expected: float
    ) -> None:
        assert objective_of(raw_robustness(**{pid: raw}), SPEC) == pytest.approx(expected)

    def test_an_idle_episode_reads_one(self) -> None:
        assert objective_of(raw_robustness(), SPEC) == 1.0

    @given(
        st.fixed_dictionaries(
            {
                p: st.floats(-1.0e3, 1.0e9, allow_nan=False, allow_infinity=False)
                for p in PROPERTY_IDS
            }
        )
    )
    def test_the_sign_is_the_sign_of_the_worst_property_and_the_value_is_in_the_box(
        self, raw: dict[str, float]
    ) -> None:
        value = objective_of(raw, SPEC)
        assert -1.0 <= value <= 1.0
        assert (value < 0.0) == any(v < 0.0 for v in raw.values())
        assert value != 0.0 or min(raw.values()) >= 0.0  # a zero is never a hidden violation

    def test_a_missing_property_and_a_nan_are_errors(self) -> None:
        raw = raw_robustness()
        del raw["P6"]
        with pytest.raises(ValueError, match="missing"):
            objective_of(raw, SPEC)
        with pytest.raises(ValueError, match="NaN"):
            objective_of(raw_robustness(P3=math.nan), SPEC)

    def test_normalise_keeps_every_property_and_never_moves_the_sign(self) -> None:
        raw = raw_robustness(P3=-5.0, P2=12.0)
        out = normalise(raw, SPEC)
        assert set(out) == set(PROPERTY_IDS)
        assert all((out[p] < 0) == (raw[p] < 0) for p in PROPERTY_IDS)

    def test_a_violation_too_small_to_scale_is_still_negative(self) -> None:
        tiny = -5e-324  # the smallest negative float: divided by a scale it would round to -0.0
        for pid in PROPERTY_IDS:
            assert objective_of(raw_robustness(**{pid: tiny}), SPEC) < 0.0
        assert (
            objective_of(raw_robustness(P3=5e-324), SPEC) == 0.0
        )  # a positive one is no violation


class TestWhatAnEvaluationRecords:
    def test_the_violated_properties_are_those_below_zero_in_property_order(self) -> None:
        raw = raw_robustness(P5=-1.0, P2=-3.0, P3=0.0)
        assert violated_of(raw) == ("P2", "P5")
        assert violated_of(raw_robustness()) == ()

    def test_the_first_event_of_each_property_is_its_earliest_and_the_order_is_by_property(
        self,
    ) -> None:
        events = [
            Attribution("P3", 10, "C1", 4),
            Attribution("P1", 20, "assumption_held"),
            Attribution("P3", 30, "A_phys", 25),
            Attribution("P1", 40, "A_exec", 39),
        ]
        first = first_events_of(events)
        assert [(a.prop, a.row, a.label) for a in first] == [
            ("P1", 20, "assumption_held"),
            ("P3", 10, "C1"),
        ]
        assert first_events_of([]) == ()


class TestTheObjectiveDropsWhenAViolationIsPresent:
    """Monotonicity, checked on traces built by hand rather than on a plant."""

    @staticmethod
    def value(trace: Any) -> float:
        return objective_of(evaluate(trace, SPEC, SPEC_CRUSHER, SPEC_PUMPS), SPEC)

    def test_the_clean_trace_is_positive_and_reads_its_tightest_margin(self) -> None:
        # the clean job re-tests every second quarter: 15 of the 30 minutes are always spare
        assert self.value(clean_trace()) == pytest.approx(0.5)

    @pytest.mark.parametrize("pid", PROPERTY_IDS)
    def test_a_violation_of_any_one_property_takes_the_objective_below_zero(self, pid: str) -> None:
        clean = self.value(clean_trace())
        broken = self.value(violating_trace(pid))
        assert clean > 0.0 > broken
        # and it is that property that is negative, nothing else
        raw = evaluate(violating_trace(pid), SPEC, SPEC_CRUSHER, SPEC_PUMPS)
        assert [p for p in PROPERTY_IDS if raw[p] < 0.0] == [pid]

    def test_a_staler_gas_test_gives_a_lower_objective_all_the_way_through_zero(self) -> None:
        ages = [0.0, 15.0, 20.0, 25.0, 29.0, 30.0, 31.0, 40.0, 55.0]
        values = []
        for age in ages:
            tr = clean_trace()
            tr.gas_age_min[SPEC_CRUSHER][16] = age
            values.append(self.value(tr))
        # flat while 15 minutes of margin remain elsewhere in the job, then strictly down
        assert values[0] == values[1] == pytest.approx(0.5)
        assert all(a > b for a, b in zip(values[1:], values[2:], strict=False))
        assert values[ages.index(30.0)] == pytest.approx(0.0)
        assert all(v < 0.0 for a, v in zip(ages, values, strict=True) if a > 30.0)

    def test_a_violation_added_to_a_violation_does_not_raise_the_objective(self) -> None:
        one = violating_trace("P3")
        both = violating_trace("P3")
        both.op[SPEC_CRUSHER][2:4] = 4  # WORK on an energised asset as well (P1)
        assert self.value(both) <= self.value(one) < 0.0

    @given(st.integers(0, 2**31))
    def test_on_arbitrary_traces_the_sign_is_the_monitors(self, seed: int) -> None:
        tr = random_trace(np.random.default_rng(seed), 40)
        raw = evaluate(tr, SPEC, SPEC_CRUSHER, SPEC_PUMPS)
        value = objective_of(raw, SPEC)
        assert (value < 0.0) == any(raw[p] < 0.0 for p in PROPERTY_IDS)


# ----------------------------------------------------------------------------- A5 classification


class TestClassify:
    def test_no_violation_is_none(self) -> None:
        assert classify(True, "in", []) == "none"
        assert classify(False, "out", []) == "none"

    def test_an_unexplained_violation_behind_the_shield_in_r_in_is_a_defect(self) -> None:
        assert classify(True, "in", ["assumption_held"]) == "defect"

    def test_one_unexplained_violation_among_explained_ones_is_still_a_defect(self) -> None:
        assert classify(True, "in", ["C1", "assumption_held", "A_phys"]) == "defect"

    @pytest.mark.parametrize("label", ["C1", "A_phys", "A_exec"])
    def test_a_violation_explained_by_a_broken_assumption_is_a_contract_failure(
        self, label: str
    ) -> None:
        assert classify(True, "in", [label]) == "contract_failure"
        assert classify(True, "out", [label, "C1"]) == "contract_failure"

    def test_an_unexplained_violation_outside_the_assumption_set_is_not_a_defect(self) -> None:
        assert classify(True, "out", ["assumption_held"]) == "unexplained_beyond_assumptions"

    def test_an_unshielded_arm_has_no_defects_only_what_the_shield_is_for(self) -> None:
        for regime in ("in", "out"):
            assert classify(False, regime, ["assumption_held"]) == "unshielded"
            assert classify(False, regime, ["C1"]) == "unshielded"

    @pytest.mark.parametrize("prop", ["P1", "P2", "P3", "P4", "P5"])
    def test_an_unexplained_violation_of_p1_to_p5_is_a_defect(self, prop: str) -> None:
        assert classify(True, "in", [(prop, "assumption_held")]) == "defect"
        assert (
            classify(True, "out", [(prop, "assumption_held")]) == "unexplained_beyond_assumptions"
        )

    def test_an_unexplained_violation_of_p6_alone_is_reported_and_is_not_a_defect(self) -> None:
        # 0003 A15: the defect rule reads P1 to P5; P6 is attributed and reported, no more
        for regime in ("in", "out"):
            kind = classify(True, regime, [("P6", "assumption_held")])
            assert kind == "p6_unexplained" and kind not in ("defect", "contract_failure")
        assert "p6_unexplained" in F.KINDS

    def test_a_p6_event_does_not_hide_a_defect_or_make_a_contract_failure_of_nothing(self) -> None:
        both = [("P6", "assumption_held"), ("P2", "assumption_held")]
        assert classify(True, "in", both) == "defect"
        explained = [("P6", "assumption_held"), ("P1", "A_exec")]
        assert classify(True, "in", explained) == "contract_failure"
        assert classify(True, "in", [("P6", "A_exec")]) == "contract_failure"

    def test_a_bare_label_is_read_as_a_violation_of_p1_to_p5_the_stricter_reading(self) -> None:
        assert classify(True, "in", ["assumption_held"]) == classify(
            True, "in", [("P1", "assumption_held")]
        )

    @given(
        st.sampled_from(PROPERTY_IDS),
        st.sampled_from(["assumption_held", "A_exec", "A_phys", "C1"]),
    )
    def test_a_shielded_r_in_event_is_a_defect_exactly_when_metrics_says_so(
        self, prop: str, label: str
    ) -> None:
        from closed_loop.eval.metrics import is_defect

        assert (classify(True, "in", [(prop, label)]) == "defect") == is_defect(prop, label)


# ----------------------------------------------------------------------------- the loop, stubbed


def stub_evaluation(
    theta: np.ndarray,
    regime: str,
    seed: int,
    value: float,
    *,
    violated: tuple[str, ...] = ("P3",),
    labels: Mapping[str, int] | None = None,
    events: tuple[Attribution, ...] = (),
    pairs: Mapping[tuple[str, str], int] | None = None,
) -> Evaluation:
    bad = value < 0.0
    by_event = (
        dict(pairs)
        if pairs is not None
        else {(prop, label): n for prop in violated for label, n in (labels or {}).items()}
    )
    return Evaluation(
        theta=np.array(theta, dtype=np.float64),
        regime=regime,
        seed=seed,
        arm="stub",
        objective=value,
        robustness={p: (-1.0 if bad and p in violated else 1.0) for p in PROPERTY_IDS},
        violated=violated if bad else (),
        first_events=events if bad else (),
        label_counts=dict(labels or {}) if bad else {},
        wall_s=0.001,
        cpu_s=0.001,
        event_counts=by_event if bad else {},
    )


def bowl(radius: float = 1.2) -> Callable[..., Evaluation]:
    """Violates inside a ball around the middle of the box; the value is how far inside it."""

    def runner(arm: Arm, theta: np.ndarray, regime: str, seed: int, ctx: Any) -> Evaluation:
        value = float(np.linalg.norm(theta - 0.5)) - radius
        return stub_evaluation(theta, regime, seed, value)

    return runner


class Spy:
    """Wraps a runner and keeps what the loop passed it, in call order."""

    def __init__(self, inner: Callable[..., Evaluation]) -> None:
        self.inner = inner
        self.calls: list[tuple[str, str, int, np.ndarray, Any]] = []

    def __call__(self, arm: Arm, theta: np.ndarray, regime: str, seed: int, ctx: Any) -> Evaluation:
        self.calls.append((arm.name, regime, seed, theta.copy(), ctx))
        return self.inner(arm, theta, regime, seed, ctx)


class TestTheLoop:
    @pytest.mark.parametrize("algo", ["cma", "random"])
    @pytest.mark.parametrize("budget", [1, 2, 13, 14, 15, 29, 100])
    def test_the_budget_is_respected_to_the_evaluation(self, algo: str, budget: int) -> None:
        spy = Spy(bowl())
        res = falsify_detailed(target_arm, "in", algo, budget, 3000, runner=spy)
        assert len(spy.calls) == budget == res.run.evals == len(res.run.history)

    @pytest.mark.parametrize("algo", ["cma", "random"])
    def test_every_point_is_in_the_box_and_every_call_sees_the_same_plant_seed(
        self, algo: str
    ) -> None:
        spy = Spy(bowl())
        falsify_detailed(target_arm, "out", algo, 60, 3004, runner=spy)
        assert {c[2] for c in spy.calls} == {3004}  # the plant seed is the search seed
        assert {c[1] for c in spy.calls} == {"out"}
        assert {c[0] for c in spy.calls} == {"R-target"}
        pts = np.array([c[3] for c in spy.calls])
        assert pts.shape == (60, THETA_DIM) and pts.min() >= 0.0 and pts.max() <= 1.0

    def test_both_optimisers_are_charged_through_the_same_function(self) -> None:
        a, b = Spy(bowl()), Spy(bowl())
        falsify_detailed(target_arm, "in", "cma", 30, 3001, runner=a)
        falsify_detailed(target_arm, "in", "random", 30, 3001, runner=b)
        assert a.calls[0][:3] == b.calls[0][:3]  # same arm, regime, plant seed
        assert not np.array_equal(a.calls[5][3], b.calls[5][3])  # different points

    def test_the_run_record_says_what_it_is(self) -> None:
        res = falsify_detailed(target_arm, "out", "random", 25, 3002, runner=bowl())
        assert isinstance(res.run, SearchRun)
        assert (res.run.algo, res.run.regime, res.run.seed) == ("random", "R_out", 3002)
        res = falsify_detailed(target_arm, "in", "cma", 25, 3002, runner=bowl())
        assert (res.run.algo, res.run.regime) == ("cmaes", "R_in")

    def test_cmaes_is_another_name_for_cma(self) -> None:
        a = falsify_detailed(target_arm, "in", "cma", 40, 3003, runner=bowl())
        b = falsify_detailed(target_arm, "in", "cmaes", 40, 3003, runner=bowl())
        assert np.array_equal(a.run.history, b.run.history)

    def test_the_first_violation_is_the_first_call_with_a_negative_value(self) -> None:
        spy = Spy(bowl(radius=1.5))
        res = falsify_detailed(target_arm, "in", "random", 200, 3005, runner=spy)
        values = [float(np.linalg.norm(c[3] - 0.5) - 1.5) for c in spy.calls]
        first = next((i + 1 for i, v in enumerate(values) if v < 0.0), None)
        assert res.run.first_violation_eval == first
        assert [e.index for e in res.violations] == [i + 1 for i, v in enumerate(values) if v < 0.0]
        assert res.run.best_f == pytest.approx(min(values))

    def test_a_search_that_finds_nothing_says_so(self) -> None:
        res = falsify_detailed(target_arm, "in", "random", 50, 3006, runner=bowl(radius=-1.0))
        assert res.run.first_violation_eval is None and res.violations == ()

    def test_stop_on_first_ends_the_run_at_the_first_violation(self) -> None:
        spy = Spy(bowl(radius=1.5))
        res = falsify_detailed(
            target_arm, "in", "random", 300, 3007, runner=spy, stop_on_first=True
        )
        assert res.run.first_violation_eval == res.run.evals == len(spy.calls) < 300
        assert len(res.violations) == 1 and res.violations[0].index == res.run.evals
        assert (res.run.algo, res.run.regime, res.run.seed) == ("random", "R_in", 3007)

    def test_stop_on_first_labels_a_cma_run_as_one(self) -> None:
        res = falsify_detailed(
            target_arm, "out", "cma", 300, 3013, runner=bowl(radius=1.5), stop_on_first=True
        )
        assert (res.run.algo, res.run.regime) == ("cmaes", "R_out")
        assert res.run.first_violation_eval == res.run.evals < 300

    def test_stop_on_first_still_runs_to_the_budget_when_nothing_violates(self) -> None:
        spy = Spy(bowl(radius=-1.0))
        res = falsify_detailed(target_arm, "in", "cma", 20, 3008, runner=spy, stop_on_first=True)
        assert res.run.evals == 20 == len(spy.calls)

    def test_the_same_run_twice_is_the_same_run(self) -> None:
        a = falsify_detailed(target_arm, "out", "cma", 45, 3009, runner=bowl())
        b = falsify_detailed(target_arm, "out", "cma", 45, 3009, runner=bowl())
        assert np.array_equal(a.run.history, b.run.history)
        assert np.array_equal(a.run.theta_best, b.run.theta_best)

    def test_the_objective_can_not_be_corrupted_through_the_argument(self) -> None:
        def vandal(arm: Arm, theta: np.ndarray, regime: str, seed: int, ctx: Any) -> Evaluation:
            theta[:] = 99.0
            return stub_evaluation(theta, regime, seed, 0.5)

        res = falsify_detailed(target_arm, "in", "random", 10, 3010, runner=vandal)
        assert res.run.theta_best.max() < 1.0

    def test_the_context_is_passed_through(self) -> None:
        spy = Spy(bowl())
        sentinel = object()
        falsify_detailed(target_arm, "in", "random", 3, 3011, context=sentinel, runner=spy)  # type: ignore[arg-type]
        assert all(c[4] is sentinel for c in spy.calls)

    def test_bad_arguments_are_refused(self) -> None:
        with pytest.raises(ValueError, match="algo"):
            falsify_detailed(target_arm, "in", "annealing", 10, 1, runner=bowl())
        with pytest.raises(ValueError, match="regime"):
            falsify_detailed(target_arm, "beyond", "cma", 10, 1, runner=bowl())
        for budget in (0, -3):
            with pytest.raises(ValueError, match="budget"):
                falsify_detailed(target_arm, "in", "random", budget, 1, runner=bowl())

    def test_the_frozen_function_returns_a_search_run_and_reads_the_module_evaluation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spy = Spy(bowl())
        monkeypatch.setattr(F, "evaluate_theta", spy)
        run = falsify(target_arm, "in", "random", 7, 3012)
        assert isinstance(run, SearchRun) and run.evals == 7 == len(spy.calls)

    def test_the_runners_search_task_reduces_a_real_run_of_this_module_to_its_kinds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from closed_loop.eval import stages
        from closed_loop.eval.tuned import PROVISIONAL

        def runner(arm: Arm, theta: np.ndarray, regime: str, seed: int, ctx: Any) -> Evaluation:
            value = float(np.linalg.norm(theta - 0.5)) - 1.8
            return stub_evaluation(theta, regime, seed, value, labels={"C1": 2})

        spy = Spy(runner)
        monkeypatch.setattr(F, "evaluate_theta", spy)
        task = stages.SearchTask("out", "cma", 40, PROVISIONAL)
        outcome = task(3000)
        direct = falsify_detailed(target_arm, "out", "cma", 40, 3000, runner=runner)
        assert len(spy.calls) == 40
        assert outcome.evals == 40 == direct.run.evals
        assert outcome.first_violation_eval == direct.run.first_violation_eval is not None
        assert outcome.violating == len(direct.violations) > 0
        assert dict(outcome.kinds) == {"contract_failure": outcome.violating}
        assert json.dumps(task.fingerprint())  # the stage can cache it

    def test_what_the_runners_stage_reads_from_a_result_is_there(self) -> None:
        res = falsify_detailed(target_arm, "in", "random", 5, 3001, runner=bowl())
        assert res.shielded is True and res.regime == "in"  # plain "in"/"out", as classify takes it
        assert isinstance(res.run.evals, int) and res.run.first_violation_eval is None
        assert all(hasattr(ev, "label_counts") for ev in res.violations)
        assert set(F.KINDS) >= {"defect", "contract_failure", "unexplained_beyond_assumptions"}


# ----------------------------------------------------------------------------- a target to reach

SLOPE = (3, 8, 13, 18, 20, 21, 22, 30)  # eight coordinates whose mean the stand-in minimises
SLOPE_LEVEL = 0.85


def slope_value(theta: np.ndarray) -> float:
    return float(SLOPE_LEVEL - np.mean(theta[list(SLOPE)]))


def slope_runner(arm: Arm, theta: np.ndarray, regime: str, seed: int, ctx: Any) -> Evaluation:
    return stub_evaluation(theta, regime, seed, slope_value(theta), violated=("P4",))


class TestTheOptimiserReachesWhatTheNominalScenarioDoesNot:
    """The sanity arm violates at nominal, so it says nothing about the optimiser. This target
    does: the nominal scenario is clean, a violation exists (every one of eight coordinates near
    the top of its range), and it is a tiny corner of the box (a uniform draw lands there with
    probability 3e-4). The stand-in is a smooth slope, which is the kindest landscape there is;
    the same check on the real plant is `scripts/search_measure.py batch --arm B3-V3`."""

    def test_the_nominal_scenario_is_clean_and_a_violation_exists(self) -> None:
        assert slope_value(nominal_theta("in")) > 0.0
        corner = nominal_theta("in").copy()
        corner[list(SLOPE)] = 1.0
        assert slope_value(corner) < 0.0

    def test_cma_es_reaches_it_within_200_evaluations_on_some_seeds_and_random_search_on_none(
        self,
    ) -> None:
        # Which seeds succeed depends on the machine: CMA-ES takes eigenvectors of its covariance,
        # and different LAPACK builds and CPUs return different (equally valid) bases, so the
        # sampled points differ. Over these 100 seeds CMA-ES found the corner on 25 to 38 on the
        # machines checked (Windows and Linux, numpy 2.4 and 2.5); random search on none. So the
        # test checks the rate over many seeds, not which seeds.
        seeds = range(3000, 3100)
        cma = [
            falsify_detailed(target_arm, "in", "cma", 200, s, runner=slope_runner).run
            for s in seeds
        ]
        rnd = [
            falsify_detailed(target_arm, "in", "random", 200, s, runner=slope_runner).run
            for s in seeds
        ]
        found_cma = [r.first_violation_eval for r in cma if r.first_violation_eval is not None]
        assert len(found_cma) >= 15, len(found_cma)  # slow, not fast
        assert all(f <= 200 for f in found_cma)
        assert all(r.first_violation_eval is None for r in rnd)
        # and what it found is a point that really violates
        best = min(cma, key=lambda r: r.best_f)
        assert slope_value(best.theta_best) == pytest.approx(best.best_f) and best.best_f < 0.0


# ----------------------------------------------------------------------------- the R_in decoder


class TestRInDecoder:
    """A decoded R_in scenario breaks no assumption by construction (0001 section 9).

    A_exec: forced commands are never refused. A_phys: the load is clipped so that the wear rate
    cannot move by a factor of 3, so no health bin can be skipped. The gate: only sensor faults it
    masks. C1 is a statistical property of the perception layer and is not something a scenario
    can be built to hold; it is attributed from the trace, not decoded away.
    """

    @given(st.one_of(thetas, corners))
    def test_no_theta_decodes_to_a_scenario_outside_the_assumption_set(
        self, theta: np.ndarray
    ) -> None:
        params = PlantParams()
        r = Scenario.decode(theta, "in").resolve(0, params)
        assert r.forced_refusal_p == 0.0  # A_exec
        assert set(r.fault_type) <= {FAULT_NONE, FAULT_STUCK, FAULT_SPIKE}  # masked faults only
        lo, hi = r.load_clip
        assert (lo, hi) == params.load_clip_in
        assert math.exp(params.beta_load * (hi - lo)) < 3.0  # A_phys: the wear rate cannot triple
        assert 0.0 <= r.load_shift_frac <= 0.6
        assert all(0.7 - 1e-9 <= m <= 1.4 + 1e-9 for m in r.rate_mult)
        assert 0.5 - 1e-9 <= r.approver_delay_mult <= 3.0 + 1e-9
        assert 0.5 - 1e-9 <= r.spare_lead_mult <= 2.0 + 1e-9
        assert r_in_exceedances(theta, "in") == ()

    def test_the_out_regime_leaves_it_in_every_way_the_in_regime_does_not(self) -> None:
        params = PlantParams()
        out = Scenario.decode(np.ones(THETA_DIM), "out").resolve(0, params)
        assert out.forced_refusal_p > 0.0
        assert FAULT_DRIFT in out.fault_type
        assert math.exp(params.beta_load * (out.load_clip[1] - out.load_clip[0])) > 3.0
        assert out.load_shift_frac > 0.6
        found = r_in_exceedances(np.ones(THETA_DIM), "out")
        for needle in (
            "A_exec",
            "gate does not mask",
            "load shift",
            "approver delay",
            "spare lead-time",
            "wear-rate",
        ):
            assert any(needle in f for f in found), (needle, found)

    def test_nothing_is_beyond_r_in_at_nominal_in_either_regime(self) -> None:
        for regime in ("in", "out"):
            assert r_in_exceedances(nominal_theta(regime), regime) == ()

    @settings(max_examples=6)
    @given(theta=st.one_of(thetas, corners))
    def test_an_episode_under_an_r_in_scenario_breaks_neither_a_exec_nor_a_phys(
        self, theta: np.ndarray, small_context: PreparedContext
    ) -> None:
        # H-C0 is the arm that issues the most STOP and EXIT commands and the most trouble
        rec = simulate_episode(
            registry_arm("H-C0"), 3000, Scenario.decode(theta, "in"), context=small_context
        )
        breaks = assumption_breaks(rec.trace, crusher=CRUSHER, pumps=PUMPS)
        assert not breaks.flat("A_exec").any()
        assert not breaks.flat("A_phys").any()

    def test_the_same_check_has_teeth_an_out_scenario_does_break_a_exec(
        self, small_context: PreparedContext
    ) -> None:
        theta = nominal_theta("out").copy()
        theta[20:23] = 1.0  # holders absent: forced commands refused with probability 0.3
        rec = simulate_episode(
            registry_arm("H-C0"), 3000, Scenario.decode(theta, "out"), context=small_context
        )
        assert assumption_breaks(rec.trace, crusher=CRUSHER, pumps=PUMPS).flat("A_exec").any()


# ----------------------------------------------------------------------------- the real plant


class TestTheTargets:
    def test_the_target_is_careless_gullible_b3_behind_the_shield(
        self, small_context: PreparedContext
    ) -> None:
        a = target_arm()
        assert a.shield and not a.interlock and not a.screen and not a.monitor
        assert a.perception == "V3" and a.shield_set == "life95" and a.spec == "clause"
        with use_context(small_context):
            proposer = a.proposer()
        assert isinstance(proposer, Careless) and proposer.p_skip == CARELESS_P_SKIP
        assert isinstance(proposer.inner, Gullible) and proposer.inner.screen is False
        assert isinstance(proposer.inner.inner, BoundRule)

    def test_the_sanity_arm_is_the_registrys_careless_with_the_shield_off(self) -> None:
        a = sanity_arm()
        assert a.name == "H-C0" and not a.shield and not a.interlock

    def test_an_arm_pickles(self) -> None:
        for a in (target_arm(), sanity_arm()):
            assert pickle.loads(pickle.dumps(a)).name == a.name


class TestOnThePlant:
    def test_an_evaluation_is_deterministic_for_a_fixed_theta_and_seed(
        self, small_context: PreparedContext
    ) -> None:
        theta = np.random.default_rng(5).random(THETA_DIM)
        a = evaluate_theta(sanity_arm(), theta, "in", 3000, small_context)
        b = evaluate_theta(sanity_arm(), theta, "in", 3000, small_context)
        assert a.objective == b.objective
        assert dict(a.robustness) == dict(b.robustness)
        assert a.violated == b.violated and a.first_events == b.first_events
        assert dict(a.label_counts) == dict(b.label_counts)

    def test_the_seed_and_the_theta_each_matter(self, small_context: PreparedContext) -> None:
        rng = np.random.default_rng(6)
        theta, other = rng.random(THETA_DIM), rng.random(THETA_DIM)
        base = evaluate_theta(sanity_arm(), theta, "in", 3000, small_context)
        moved_seed = evaluate_theta(sanity_arm(), theta, "in", 3001, small_context)
        moved_theta = evaluate_theta(sanity_arm(), other, "in", 3000, small_context)
        assert dict(base.robustness) != dict(moved_seed.robustness)
        assert dict(base.robustness) != dict(moved_theta.robustness)

    def test_the_objective_is_the_monitors_robustness_scaled(
        self, small_context: PreparedContext
    ) -> None:
        theta = nominal_theta("in")
        ev = evaluate_theta(sanity_arm(), theta, "in", 3000, small_context)
        rec = simulate_episode(
            sanity_arm(), 3000, Scenario.decode(theta, "in"), context=small_context
        )
        assert dict(ev.robustness) == dict(rec.result.min_robustness)
        assert ev.objective == objective_of(ev.robustness, small_context.spec_clause)
        assert ev.violated == tuple(p for p in PROPERTY_IDS if rec.result.min_robustness[p] < 0.0)
        assert ev.cpu_s > 0.0 and ev.wall_s > 0.0

    @pytest.mark.usefixtures("provisional_defaults")  # the seed was chosen under the provisional B3
    def test_the_objective_is_scaled_by_the_clause_the_plant_ran_under_not_by_the_stale_one(
        self, small_context: PreparedContext
    ) -> None:
        # the shielded target never violates; its tightest margin is the permit time left (P2), so
        # a stale specification with a shorter permit would scale it differently
        ctx = dataclasses.replace(
            small_context, spec_stale=SpecParams(retest_q=8, permit_max_h=6.0)
        )
        ev = evaluate_theta(target_arm(), nominal_theta("in"), "in", 3000, ctx)
        assert ev.violated == () and ev.robustness["P2"] < 1.0e6  # someone went in
        assert ev.objective == objective_of(ev.robustness, ctx.spec_clause)
        assert ev.objective != objective_of(ev.robustness, ctx.spec_stale)

    @pytest.mark.parametrize("regime", ["in", "out"])
    def test_the_regime_reaches_the_decoder(
        self, regime: str, small_context: PreparedContext
    ) -> None:
        theta = nominal_theta("out").copy()
        theta[0] = theta[5] = theta[10] = theta[15] = 1.0  # fastest wear: x1.4 in R_in, x2 in R_out
        ev = evaluate_theta(sanity_arm(), theta, regime, 3000, small_context)
        rec = simulate_episode(
            sanity_arm(), 3000, Scenario.decode(theta, regime), context=small_context
        )
        assert dict(ev.robustness) == dict(rec.result.min_robustness)
        other = simulate_episode(
            sanity_arm(), 3000, Scenario.decode(theta, "out" if regime == "in" else "in"),
            context=small_context,
        )  # fmt: skip
        assert dict(ev.robustness) != dict(other.result.min_robustness)

    def test_the_attribution_is_the_episodes_own(self, small_context: PreparedContext) -> None:
        theta = nominal_theta("in")
        ev = evaluate_theta(sanity_arm(), theta, "in", 3000, small_context)
        rec = simulate_episode(
            sanity_arm(), 3000, Scenario.decode(theta, "in"), context=small_context
        )
        assert sum(ev.label_counts.values()) == len(rec.result.attributions)
        assert sum(ev.event_counts.values()) == len(rec.result.attributions)
        assert dict(ev.event_counts) == {
            (a.prop, a.label): sum(
                1 for b in rec.result.attributions if (b.prop, b.label) == (a.prop, a.label)
            )
            for a in rec.result.attributions
        }
        assert ev.events == tuple(sorted(ev.event_counts))
        for a in ev.first_events:
            earliest = min(x.row for x in rec.result.attributions if x.prop == a.prop)
            assert a.row == earliest

    def test_careless_with_the_shield_off_violates_within_a_few_evaluations(
        self, small_context: PreparedContext
    ) -> None:
        # a sanity check that the objective can be minimised, not a result (full-size version:
        # docs/adr/progress/P9b-falsify.md)
        res = falsify_detailed(
            sanity_arm, "in", "random", 12, 3000, context=small_context, stop_on_first=True
        )
        assert res.run.first_violation_eval is not None
        assert res.violations[0].objective < 0.0 and res.violations[0].index == res.run.evals
        assert res.shielded is False

    def test_a_find_of_the_unshielded_arm_shrinks_and_is_labelled_unshielded(
        self, small_context: PreparedContext
    ) -> None:
        res = falsify_detailed(
            sanity_arm, "in", "random", 12, 3000, context=small_context, stop_on_first=True
        )
        found = res.violations[0]
        f = build_finding(res, found, arm=sanity_arm(), context=small_context, max_calls=6)
        assert f.kind == f.shrunk_kind == "unshielded"
        assert f.shrunk_objective < 0.0 and set(f.shrunk_props) & set(f.props)
        assert f.non_nominal_after <= f.non_nominal_before and f.shrink_calls <= 6
        again = evaluate_theta(sanity_arm(), np.array(f.shrunk_theta), "in", 3000, small_context)
        assert again.objective == f.shrunk_objective  # the shrunk scenario reproduces


# ----------------------------------------------------------------------------- findings (stubbed)


def drift_runner(prop: str = "P2", label: str = "C1") -> Callable[..., Evaluation]:
    """Violates iff asset 0's sensor fault is the unmasked drift (R_out only decodes it)."""

    def runner(arm: Arm, theta: np.ndarray, regime: str, seed: int, ctx: Any) -> Evaluation:
        hit = regime == "out" and theta[3] >= 0.75
        value = -0.4 if hit else 0.6
        events = (Attribution(prop, 4000, label, 3500),)
        return stub_evaluation(
            theta, regime, seed, value, violated=(prop,), labels={label: 3}, events=events
        )

    return runner


def a_find(theta: np.ndarray, runner: Callable[..., Evaluation], regime: str = "out") -> Any:
    ev = runner(target_arm(), theta, regime, 3000, None)
    ev = dataclasses.replace(ev, index=17)
    res = F.FalsifyResult(
        run=SearchRun("cmaes", f"R_{regime}", 3000, ev.objective, 17, 20, theta, np.array([-0.4])),
        arm="R-target",
        shielded=True,
        regime=regime,
        violations=(ev,),
        seconds=0.1,
    )
    return res, ev


class TestFindings:
    def test_a_find_is_shrunk_to_the_dimension_it_needs_and_labelled(self) -> None:
        theta = np.full(THETA_DIM, 0.9)
        theta[3] = 0.97
        res, ev = a_find(theta, drift_runner())
        f = build_finding(res, ev, arm=target_arm(), runner=drift_runner())
        assert isinstance(f, Finding)
        assert [d.index for d in f.dimensions] == [3]
        d = f.dimensions[0]
        assert d.decoded == "unmasked gain drift" and d.name == "CR-501 sensor-fault type"
        assert d.beyond_r_in is not None and "gate does not mask" in d.beyond_r_in
        assert 0.75 <= f.shrunk_theta[3] <= 0.77
        assert f.beyond_r_in == ("a sensor fault the data-quality gate does not mask",)
        assert f.label == "C1" and f.kind == f.shrunk_kind == "contract_failure"
        assert f.non_nominal_before == THETA_DIM and f.non_nominal_after == 1
        assert f.eval_index == 17 and f.seed == 3000 and f.regime == "out" and f.shielded

    def test_the_shrunk_theta_reproduces_the_violation(self) -> None:
        theta = np.full(THETA_DIM, 0.9)
        theta[3] = 0.97
        res, ev = a_find(theta, drift_runner())
        f = build_finding(res, ev, arm=target_arm(), runner=drift_runner())
        assert (
            drift_runner()(target_arm(), np.array(f.shrunk_theta), "out", 3000, None).objective < 0
        )

    def test_a_find_whose_only_unexplained_event_is_p6_is_not_a_defect(self) -> None:
        def runner(arm: Arm, theta: np.ndarray, regime: str, seed: int, ctx: Any) -> Evaluation:
            hit = theta[9] >= 0.5
            return stub_evaluation(
                theta, regime, seed, -0.2 if hit else 0.5, violated=("P6",),
                labels={"assumption_held": 1},
                events=(Attribution("P6", 100, "assumption_held"),),
                pairs={("P6", "assumption_held"): 1},
            )  # fmt: skip

        theta = nominal_theta("in").copy()
        theta[9] = 0.8
        res, ev = a_find(theta, runner, regime="in")
        f = build_finding(res, ev, arm=target_arm(), runner=runner)
        assert f.kind == f.shrunk_kind == "p6_unexplained" and f.label == "assumption_held"
        assert f.props == f.shrunk_props == ("P6",)

    def test_an_unexplained_violation_in_r_in_is_a_defect_and_stays_one_when_shrunk(self) -> None:
        def runner(arm: Arm, theta: np.ndarray, regime: str, seed: int, ctx: Any) -> Evaluation:
            hit = theta[9] >= 0.5
            return stub_evaluation(
                theta, regime, seed, -0.2 if hit else 0.5, violated=("P2",),
                labels={"assumption_held": 1},
                events=(Attribution("P2", 100, "assumption_held"),),
            )  # fmt: skip

        theta = np.full(THETA_DIM, 0.8)
        res, ev = a_find(theta, runner, regime="in")
        f = build_finding(res, ev, arm=target_arm(), runner=runner)
        assert f.kind == f.shrunk_kind == "defect" and f.label == "assumption_held"
        assert [d.index for d in f.dimensions] == [9] and f.beyond_r_in == ()

    def test_shrinking_keeps_a_violation_of_a_property_the_find_had(self) -> None:
        # at nominal the stand-in violates P1; the find violated P3 and needs coordinate 8
        def runner(arm: Arm, theta: np.ndarray, regime: str, seed: int, ctx: Any) -> Evaluation:
            prop = "P3" if theta[8] >= 0.5 else "P1"
            return stub_evaluation(theta, regime, seed, -0.3, violated=(prop,))

        theta = nominal_theta("in").copy()
        theta[8] = 0.9
        res, ev = a_find(theta, runner, regime="in")
        assert ev.violated == ("P3",)
        f = build_finding(res, ev, arm=target_arm(), runner=runner)
        assert [d.index for d in f.dimensions] == [8]
        assert f.shrunk_props == ("P3",)

    def test_each_surviving_dimension_is_ablated_once_and_is_necessary_when_the_violation_goes(
        self,
    ) -> None:
        def runner(arm: Arm, theta: np.ndarray, regime: str, seed: int, ctx: Any) -> Evaluation:
            hit = theta[8] >= 0.5 and theta[21] >= 0.4
            return stub_evaluation(theta, regime, seed, -0.3 if hit else 0.5, violated=("P3",))

        theta = nominal_theta("in").copy()
        theta[[8, 21, 12]] = 0.9
        res, ev = a_find(theta, runner, regime="in")
        calls: list[np.ndarray] = []

        def counting(*a: Any) -> Evaluation:
            calls.append(a[1].copy())
            return runner(*a)

        f = build_finding(res, ev, arm=target_arm(), runner=counting)
        assert [d.index for d in f.dimensions] == [8, 21]
        assert [d.necessary for d in f.dimensions] == [True, True]
        # the ablations are the shrunk scenario with one dimension put back, each run once
        shrunk = np.array(f.shrunk_theta)
        for d in f.dimensions:
            ablated = shrunk.copy()
            ablated[d.index] = nominal_theta("in")[d.index]
            assert sum(np.array_equal(c, ablated) for c in calls) == 1

    def test_a_shrink_cut_short_leaves_the_ablation_unasked(self) -> None:
        theta = np.full(THETA_DIM, 0.9)
        theta[3] = 0.97
        res, ev = a_find(theta, drift_runner())
        f = build_finding(res, ev, arm=target_arm(), runner=drift_runner(), max_calls=2)
        assert f.shrink_truncated and all(d.necessary is None for d in f.dimensions)

    def test_the_label_is_that_of_the_earliest_violation_whatever_the_property(self) -> None:
        def runner(arm: Arm, theta: np.ndarray, regime: str, seed: int, ctx: Any) -> Evaluation:
            events = (Attribution("P2", 4000, "C1", 3500), Attribution("P3", 3000, "A_phys", 2900))
            return stub_evaluation(
                theta, regime, seed, -0.4, violated=("P2", "P3"),
                labels={"C1": 1, "A_phys": 1}, events=events,
            )  # fmt: skip

        theta = nominal_theta("out").copy()
        theta[3] = 0.9
        res, ev = a_find(theta, runner)
        f = build_finding(res, ev, arm=target_arm(), runner=runner)
        assert f.label == "A_phys"  # P3's event is at row 3000, before P2's at 4000
        assert [a.prop for a in f.shrunk_events] == ["P3", "P2"]

    def test_a_budget_on_the_shrink_is_respected_and_reported(self) -> None:
        theta = np.full(THETA_DIM, 0.9)
        theta[3] = 0.97
        res, ev = a_find(theta, drift_runner())
        calls = []

        def counting(*a: Any) -> Evaluation:
            calls.append(1)
            return drift_runner()(*a)

        f = build_finding(res, ev, arm=target_arm(), runner=counting, max_calls=3)
        assert len(calls) <= 3 and f.shrink_truncated and f.shrunk_objective < 0.0

    def test_an_evaluation_that_does_not_violate_cannot_be_shrunk(self) -> None:
        res, _ = a_find(np.full(THETA_DIM, 0.9), drift_runner())
        ok = stub_evaluation(np.full(THETA_DIM, 0.9), "out", 3000, 0.4)
        with pytest.raises(ValueError, match="does not violate"):
            build_finding(res, ok, arm=target_arm(), runner=drift_runner())

    def test_collect_takes_the_deepest_violation_of_each_run_that_has_one(self) -> None:
        def runner(arm: Arm, theta: np.ndarray, regime: str, seed: int, ctx: Any) -> Evaluation:
            return stub_evaluation(theta, regime, seed, -float(theta[3]), violated=("P2",))

        shallow = np.full(THETA_DIM, 0.4)
        deep = np.full(THETA_DIM, 0.4)
        deep[3] = 0.9
        evs = [
            dataclasses.replace(runner(target_arm(), t, "out", 3000, None), index=i + 1)
            for i, t in enumerate((shallow, deep))
        ]
        res = F.FalsifyResult(
            run=SearchRun("random", "R_out", 3000, -0.9, 1, 2, deep, np.array([-0.4, -0.9])),
            arm="R-target", shielded=True, regime="out", violations=tuple(evs), seconds=0.0,
        )  # fmt: skip
        empty = F.FalsifyResult(
            run=SearchRun("random", "R_out", 3001, 0.2, None, 2, deep, np.array([0.3, 0.2])),
            arm="R-target", shielded=True, regime="out", violations=(), seconds=0.0,
        )  # fmt: skip
        found = collect_findings([res, empty], arm_factory=target_arm, runner=runner)
        assert len(found) == 1 and found[0].eval_index == 2 and found[0].seed == 3000

    def test_dimensions_are_decoded_for_a_reader(self) -> None:
        theta = nominal_theta("out").copy()
        theta[0] = 1.0  # wear rate of the crusher: 2.0 in R_out
        theta[30], theta[31] = 0.5, 0.25  # a load shift of 1.0 of the load, a quarter in
        theta[23] = 0.5
        dims = {d.index: d for d in dimensions_of(theta, "out")}
        assert set(dims) == {0, 23, 30, 31}
        assert dims[0].decoded == "x2.000" and "wear-rate" in (dims[0].beyond_r_in or "")
        assert dims[23].decoded == "quarter 4320" and dims[23].beyond_r_in is None
        assert dims[30].decoded.startswith("+1.000") and "R_in cap" in (dims[30].beyond_r_in or "")
        assert dims[31].decoded.startswith("day 22.")


class TestWhatGetsThrough:
    def finding(self, seed: int, algo: str = "cmaes", regime: str = "out") -> Finding:
        theta = np.full(THETA_DIM, 0.9)
        theta[3] = 0.97
        res, ev = a_find(theta, drift_runner(), regime=regime)
        ev = dataclasses.replace(ev, seed=seed)
        f = build_finding(res, ev, arm=target_arm(), runner=drift_runner())
        return dataclasses.replace(f, algo=algo, seed=seed)

    def test_finds_that_shrank_to_the_same_scenario_are_one_entry_with_every_run_that_found_it(
        self,
    ) -> None:
        doc = what_gets_through([self.finding(3000), self.finding(3001, "random")])
        assert len(doc["entries"]) == 1
        entry = doc["entries"][0]
        assert [(r["algo"], r["seed"]) for r in entry["found_by"]] == [
            ("cmaes", 3000),
            ("random", 3001),
        ]
        assert entry["regime"] == "R_out" and entry["kind"] == "contract_failure"
        assert entry["broken_assumption"] == "C1" and entry["properties"] == ["P2"]
        assert entry["dimensions"][0]["decoded"] == "unmasked gain drift"
        assert entry["beyond_r_in"] == ["a sensor fault the data-quality gate does not mask"]
        assert entry["shielded"] is True and entry["non_nominal_after"] == 1
        assert (
            entry["plant_seed"] == 3000
        )  # the representative: fewest dimensions, then lowest seed

    def test_an_unshielded_find_is_listed_as_one_and_not_as_a_contract_failure(self) -> None:
        f = dataclasses.replace(self.finding(3000), shielded=False, shrunk_kind="unshielded")
        entry = what_gets_through([f])["entries"][0]
        assert entry["shielded"] is False and entry["kind"] == "unshielded"

    def test_different_minimal_scenarios_are_different_entries(self) -> None:
        other = dataclasses.replace(self.finding(3002), regime="in")
        doc = what_gets_through([self.finding(3000), other])
        assert len(doc["entries"]) == 2
        assert [e["id"] for e in doc["entries"]] == ["WGT-01", "WGT-02"]

    def test_the_document_says_what_it_is_and_survives_json(self) -> None:
        doc = what_gets_through([self.finding(3000)])
        assert doc["synthetic"] is True and doc["fidelity"] == "F1" and doc["seed_role"] == "search"
        assert "development" in doc["note"]
        assert json.loads(json.dumps(doc)) == doc

    def test_the_pieces_of_a_long_run_join_into_the_document_of_the_whole(self) -> None:
        a, b, c = self.finding(3000), self.finding(3001, "random"), self.finding(3002)
        other = dataclasses.replace(c, regime="in")
        whole = what_gets_through([a, b, other])
        joined = merge_documents(
            [what_gets_through([a]), what_gets_through([b]), what_gets_through([other])]
        )
        assert joined == whole
        assert [e["id"] for e in joined["entries"]] == ["WGT-01", "WGT-02"]
        twin = next(e for e in joined["entries"] if e["regime"] == "R_out")
        assert [(r["algo"], r["seed"]) for r in twin["found_by"]] == [
            ("cmaes", 3000),
            ("random", 3001),
        ]

    def test_joining_keeps_the_runs_of_every_piece(self) -> None:
        theta = np.full(THETA_DIM, 0.9)

        def piece(seed: int, first: int | None) -> dict[str, Any]:
            res = F.FalsifyResult(
                run=SearchRun("cmaes", "R_in", seed, 0.3, first, 20, theta, np.array([0.3])),
                arm="R-target", shielded=True, regime="in", violations=(), seconds=1.0,
            )  # fmt: skip
            return what_gets_through([], [res])

        joined = merge_documents([piece(3000, None), piece(3001, 4)])
        assert [(r["seed"], r["found"]) for r in joined["runs"]] == [(3000, False), (3001, True)]
        assert joined["entries"] == []

    def test_every_run_is_listed_found_or_not(self) -> None:
        theta = np.full(THETA_DIM, 0.9)
        _, ev = a_find(theta, drift_runner())
        found = F.FalsifyResult(
            run=SearchRun("cmaes", "R_out", 3000, -0.4, 17, 20, theta, np.array([-0.4])),
            arm="R-target", shielded=True, regime="out", violations=(ev,), seconds=2.0,
        )  # fmt: skip
        missed = F.FalsifyResult(
            run=SearchRun("random", "R_out", 3001, 0.3, None, 20, theta, np.array([0.3])),
            arm="R-target", shielded=True, regime="out", violations=(), seconds=2.0,
        )  # fmt: skip
        doc = what_gets_through([], [found, missed])
        assert [r["found"] for r in doc["runs"]] == [True, False]
        assert doc["runs"][0]["first_violation_eval"] == 17 and doc["runs"][1]["evals"] == 20
        assert doc["entries"] == []


# ----------------------------------------------------------------------------- the objective object


class TestObjectiveObject:
    def test_it_counts_keeps_violations_in_order_and_sums_time(self) -> None:
        obj = Objective(target_arm(), "in", 3000, runner=bowl(radius=0.0))
        assert obj(np.full(THETA_DIM, 0.5)) == pytest.approx(0.0)  # on the edge: not a violation
        assert obj.calls == 1 and obj.violations == []
        assert obj(np.full(THETA_DIM, 0.9)) > 0.0
        obj.runner = bowl(radius=5.0)
        assert obj(np.full(THETA_DIM, 0.5)) < 0.0
        assert obj.calls == 3 and [e.index for e in obj.violations] == [3]
        assert obj.seconds == pytest.approx(0.003) and obj.cpu_seconds == pytest.approx(0.003)
        assert len(obj.points) == len(obj.values) == 3

    def test_a_value_of_exactly_zero_does_not_stop_a_search_that_stops_on_violations(self) -> None:
        obj = Objective(target_arm(), "in", 3000, runner=bowl(radius=0.0), stop_on_first=True)
        assert obj(np.full(THETA_DIM, 0.5)) == pytest.approx(0.0)  # no StopSearch
        obj.runner = bowl(radius=5.0)
        with pytest.raises(F.StopSearch):
            obj(np.full(THETA_DIM, 0.5))
        assert obj.calls == 2 and len(obj.violations) == 1

    def test_an_unknown_regime_is_refused(self) -> None:
        with pytest.raises(ValueError, match="regime"):
            Objective(target_arm(), "beyond", 1)
