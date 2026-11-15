"""P9b: scenario shrinking and the nominal vector it shrinks towards."""

from __future__ import annotations

import zlib

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from closed_loop.core.timegrid import HORIZON_Q, SHIFT_Q
from closed_loop.core.trace import Trace
from closed_loop.eval.arms import arm
from closed_loop.eval.context import PreparedContext
from closed_loop.eval.episode import simulate_episode
from closed_loop.plant.params import PlantParams
from closed_loop.plant.scenario import (
    FAULT_NONE,
    REGIMES,
    THETA_DIM,
    Scenario,
)
from closed_loop.search.shrink import (
    COORDINATES,
    ShrinkResult,
    _reset,  # a phase whose guarantee is tested alone
    nominal_theta,
    non_nominal,
    shrink,
    shrink_detailed,
)

Theta = np.ndarray
NOM = nominal_theta("in")

thetas = arrays(
    np.float64,
    THETA_DIM,
    elements=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
)


def dev(theta: Theta, i: int, regime: str = "in") -> float:
    return float(abs(theta[i] - nominal_theta(regime)[i]))


def far_from_nominal() -> Theta:
    """A vector with every coordinate well away from nominal, so a test has something to shrink."""
    return np.clip(NOM + np.where(NOM > 0.5, -0.45, 0.45), 0.0, 1.0)


class TestNominalTheta:
    @pytest.mark.parametrize("regime", REGIMES)
    def test_it_decodes_to_a_plant_with_no_adversary(self, regime: str) -> None:
        params = PlantParams()
        r = Scenario.decode(nominal_theta(regime), regime).resolve(7, params)
        assert r.rate_mult == pytest.approx((1.0,) * 4)
        assert r.onset_shift_days == pytest.approx((0.0,) * 4)
        assert r.residual_shift_days == pytest.approx((0.0,) * 4)
        assert r.fault_type == (FAULT_NONE,) * 4
        assert r.holder_p == pytest.approx((params.holder_present_p,) * 3)
        assert r.approver_delay_mult == pytest.approx(1.0)
        assert r.approver_catch_p == pytest.approx(params.approver_catch_p)
        assert r.spare_lead_mult == pytest.approx(1.0)
        assert r.load_shift_frac == 0.0
        assert r.forced_refusal_p == 0.0

    @pytest.mark.parametrize("regime", REGIMES)
    def test_the_gas_excursions_and_the_note_are_parked_where_nothing_can_reach_them(
        self, regime: str
    ) -> None:
        r = Scenario.decode(nominal_theta(regime), regime).resolve(7, PlantParams())
        assert r.gas_adv_q == (HORIZON_Q - 1, HORIZON_Q - 1)
        assert r.note_q == HORIZON_Q - 1
        # the plant posts a note at handover k = q // SHIFT_Q for q < HORIZON_Q, and puts the
        # adversary's note at handover max(round(note_q / SHIFT_Q), 1): past the last one
        last_handover = (HORIZON_Q - 1) // SHIFT_Q
        assert max(round(r.note_q / SHIFT_Q), 1) > last_handover

    def test_the_regimes_differ_only_where_their_ranges_do(self) -> None:
        a, b = nominal_theta("in"), nominal_theta("out")
        differing = {int(i) for i in np.flatnonzero(np.abs(a - b) > 1e-12)}
        expected = {5 * k for k in range(4)} | {27, 29}  # rate, delay, lead
        assert differing == expected

    def test_it_is_in_the_box_read_only_and_one_value_per_coordinate(self) -> None:
        for regime in REGIMES:
            v = nominal_theta(regime)
            assert v.shape == (THETA_DIM,) and v.min() >= 0.0 and v.max() <= 1.0
            assert not v.flags.writeable
        assert len(COORDINATES) == THETA_DIM and len(set(COORDINATES)) == THETA_DIM

    def test_an_unknown_regime_is_refused(self) -> None:
        with pytest.raises(ValueError, match="regime"):
            nominal_theta("beyond")

    def test_the_plant_runs_it_exactly_as_it_runs_the_nominal_condition(
        self, small_context: PreparedContext
    ) -> None:
        # B0 raises work orders only after a failure, and the failure times depend on the scenario
        # only through the fields above: identical traces say the vector changes nothing
        b0 = arm("B0")
        plain = simulate_episode(b0, 3001, Scenario(theta=None), context=small_context).trace
        parked = simulate_episode(
            b0, 3001, Scenario.decode(nominal_theta("in"), "in"), context=small_context
        ).trace
        assert_same_trace(plain, parked)


def assert_same_trace(a: Trace, b: Trace) -> None:
    assert a.n_q == b.n_q and a.asset_ids == b.asset_ids
    for asset in a.asset_ids:
        for field in ("op", "permit", "occupied", "alarm", "iso_proven", "health_bin"):
            assert np.array_equal(getattr(a, field)[asset], getattr(b, field)[asset]), (
                asset,
                field,
            )
    assert np.array_equal(a.throughput, b.throughput)
    assert a.commands == b.commands


class TestNonNominal:
    def test_the_nominal_vector_has_none(self) -> None:
        assert non_nominal(NOM) == ()

    def test_a_deviation_counts_when_it_is_larger_than_tol(self) -> None:
        x = NOM.copy()
        x[3] = NOM[3] + 0.02  # exactly tol: not larger
        x[8] = NOM[8] + 0.021
        assert non_nominal(x, tol=0.02) == (8,)

    def test_it_clips_to_the_box_first(self) -> None:
        x = NOM.copy()
        x[26] = 5.0  # clipped to 1.0, which is nominal for the note time
        assert non_nominal(x) == ()


def conjunction(needs: dict[int, float], regime: str = "in"):
    """Violates iff every listed coordinate is at least ``need`` away from nominal."""
    nom = nominal_theta(regime)

    def violates(x: Theta) -> bool:
        return all(abs(x[i] - nom[i]) >= need for i, need in needs.items())

    return violates


class TestShrink:
    def test_a_violation_that_needs_nothing_shrinks_to_nominal_in_two_calls(self) -> None:
        calls = []
        res = shrink_detailed(far_from_nominal(), lambda x: calls.append(1) or True)
        assert np.array_equal(res.theta, NOM)
        assert res.calls == 2 and not res.truncated
        assert res.non_nominal_before == THETA_DIM
        assert res.non_nominal_after == 0

    def test_a_threshold_on_one_coordinate_leaves_that_coordinate_just_past_it(self) -> None:
        pred = conjunction({8: 0.3})
        start = far_from_nominal()
        out = shrink(start, pred, tol=0.02)
        assert pred(out)
        assert non_nominal(out) == (8,)
        assert 0.3 <= dev(out, 8) <= 0.3 + 0.02

    def test_two_coordinates_that_are_needed_together_both_survive(self) -> None:
        pred = conjunction({8: 0.25, 21: 0.2})
        out = shrink(far_from_nominal(), pred, tol=0.02)
        assert pred(out) and set(non_nominal(out)) == {8, 21}
        assert 0.25 <= dev(out, 8) <= 0.27 and 0.2 <= dev(out, 21) <= 0.22

    def test_a_coordinate_that_has_no_effect_is_reset_for_free(self) -> None:
        # coordinate 4 is the sensor-fault day: the predicate does not look at it
        pred = conjunction({12: 0.3})
        out = shrink(far_from_nominal(), pred)
        assert out[4] == NOM[4]

    def test_the_input_is_not_edited(self) -> None:
        start = far_from_nominal()
        before = start.copy()
        shrink(start, conjunction({8: 0.3}))
        assert np.array_equal(start, before)

    def test_an_input_that_does_not_violate_is_refused(self) -> None:
        with pytest.raises(ValueError, match="does not violate"):
            shrink(far_from_nominal(), lambda x: False)

    def test_the_nominal_vector_given_a_violation_comes_back_after_one_call(self) -> None:
        res = shrink_detailed(NOM.copy(), lambda x: True)
        assert np.array_equal(res.theta, NOM) and res.calls == 1

    def test_tol_must_be_a_proper_fraction(self) -> None:
        for bad in (0.0, 1.0, -0.1, 2.0):
            with pytest.raises(ValueError, match="tol"):
                shrink(far_from_nominal(), lambda x: True, tol=bad)

    def test_the_regime_sets_the_nominal_vector(self) -> None:
        out = shrink(far_from_nominal(), lambda x: True, regime="out")
        assert np.array_equal(out, nominal_theta("out"))
        assert not np.array_equal(out, nominal_theta("in"))

    def test_an_explicit_nominal_vector_overrides_the_regime(self) -> None:
        mine = np.full(THETA_DIM, 0.25)
        out = shrink(far_from_nominal(), lambda x: True, nominal=mine)
        assert np.array_equal(out, mine)

    def test_the_predicate_gets_private_copies(self) -> None:
        inner = conjunction({8: 0.3})
        seen: list[Theta] = []

        def vandal(x: Theta) -> bool:
            verdict = inner(x)
            seen.append(x.copy())
            x[:] = 0.123  # whatever the predicate does to its argument must not reach the search
            return verdict

        out = shrink(far_from_nominal(), vandal)
        assert inner(out) and non_nominal(out) == (8,)
        assert all(not np.all(s == 0.123) for s in seen)

    def test_the_same_inputs_give_the_same_calls_and_the_same_answer(self) -> None:
        def run() -> tuple[Theta, list[bytes]]:
            log: list[bytes] = []

            def pred(x: Theta) -> bool:
                log.append(x.tobytes())
                return conjunction({8: 0.25, 21: 0.2})(x)

            return shrink(far_from_nominal(), pred), log

        (a, log_a), (b, log_b) = run(), run()
        assert np.array_equal(a, b) and log_a == log_b

    def test_the_number_of_calls_is_logarithmic_in_tol_not_linear_in_the_dimension(self) -> None:
        res = shrink_detailed(far_from_nominal(), conjunction({8: 0.3}), tol=0.02)
        # one check, the delta-debugging resets, and 6 bisection steps for the one survivor
        assert res.calls <= 1 + 2 * THETA_DIM + 8

    def test_a_finer_tol_gets_nearer_the_threshold(self) -> None:
        pred = conjunction({8: 0.3})
        coarse = dev(shrink(far_from_nominal(), pred, tol=0.1), 8)
        fine = dev(shrink(far_from_nominal(), pred, tol=0.001), 8)
        assert 0.3 <= fine <= 0.301 < coarse <= 0.4


class TestBudget:
    def test_the_budget_is_respected_and_the_result_still_violates(self) -> None:
        inner = conjunction({8: 0.25, 21: 0.2})
        for max_calls in (1, 2, 3, 5, 8, 13):
            calls: list[int] = []

            def pred(x: Theta, calls: list[int] = calls) -> bool:
                calls.append(1)
                return inner(x)

            res = shrink_detailed(far_from_nominal(), pred, max_calls=max_calls)
            assert len(calls) == res.calls <= max_calls
            assert inner(res.theta)

    def test_a_spent_budget_is_reported(self) -> None:
        res = shrink_detailed(far_from_nominal(), conjunction({8: 0.25}), max_calls=3)
        assert res.truncated and res.calls == 3
        full = shrink_detailed(far_from_nominal(), conjunction({8: 0.25}))
        assert not full.truncated and full.non_nominal_after <= res.non_nominal_after

    def test_one_call_is_only_the_check(self) -> None:
        start = far_from_nominal()
        res = shrink_detailed(start, lambda x: True, max_calls=1)
        assert np.array_equal(res.theta, start) and res.truncated

    def test_a_budget_of_nothing_is_refused(self) -> None:
        with pytest.raises(ValueError, match="max_calls"):
            shrink(far_from_nominal(), lambda x: True, max_calls=0)

    def test_progress_made_before_the_budget_ended_is_kept(self) -> None:
        # the first reset of everything succeeds (call 2); the budget then ends the search
        res = shrink_detailed(far_from_nominal(), lambda x: True, max_calls=2)
        assert np.array_equal(res.theta, NOM) and res.calls == 2 and not res.truncated


def scrambled(x: Theta) -> bool:
    """A violation set with no structure at all: a hash of the vector decides."""
    return zlib.crc32(np.round(x, 9).tobytes()) % 3 != 0


class TestInvariants:
    """What holds for any predicate, however badly behaved."""

    @given(thetas)
    def test_the_result_violates_has_no_more_non_nominal_coordinates_and_stays_between(
        self, theta: Theta
    ) -> None:
        assume(scrambled(theta))
        res = shrink_detailed(theta, scrambled)
        assert scrambled(res.theta)
        assert res.non_nominal_after <= res.non_nominal_before
        assert len(non_nominal(res.theta)) <= len(non_nominal(theta))
        lo = np.minimum(NOM, theta) - 1e-12
        hi = np.maximum(NOM, theta) + 1e-12
        assert np.all(res.theta >= lo) and np.all(res.theta <= hi)

    @given(thetas, st.lists(st.integers(0, THETA_DIM - 1), min_size=1, max_size=4, unique=True))
    def test_a_conjunction_keeps_no_coordinate_it_does_not_read(
        self, theta: Theta, needed: list[int]
    ) -> None:
        theta = theta.copy()
        for i in needed:  # make the coordinates the predicate reads far from nominal
            theta[i] = 0.2 if NOM[i] > 0.5 else 0.8
        pred = conjunction({i: 0.15 for i in needed})
        assert pred(theta)
        out = shrink(theta, pred, tol=0.02)
        assert pred(out)
        assert set(non_nominal(out)) <= set(needed)

    @given(thetas, st.integers(1, 30))
    def test_a_budget_never_costs_the_violation(self, theta: Theta, max_calls: int) -> None:
        assume(scrambled(theta))
        res = shrink_detailed(theta, scrambled, max_calls=max_calls)
        assert res.calls <= max_calls and scrambled(res.theta)

    @given(thetas)
    def test_the_result_type_carries_the_counts(self, theta: Theta) -> None:
        res = shrink_detailed(theta, lambda x: True)
        assert isinstance(res, ShrinkResult)
        assert res.non_nominal_before == len(non_nominal(theta))
        assert res.non_nominal_after == 0


class TestTheResetPhaseOnItsOwn:
    """Delta debugging promises a 1-minimal answer: no one coordinate left can be put back."""

    @given(thetas)
    def test_after_the_reset_phase_no_single_remaining_coordinate_can_be_reset(
        self, theta: Theta
    ) -> None:
        assume(scrambled(theta))
        x = theta.copy()
        active = list(non_nominal(x, NOM, 0.02))
        _reset(x, active, NOM, scrambled)
        assert scrambled(x)
        for i in active:
            trial = x.copy()
            trial[i] = NOM[i]
            assert not scrambled(trial), i
        assert set(active) == set(non_nominal(x, NOM, 0.02))

    def test_a_coordinate_that_could_not_be_reset_at_first_is_tried_again_once_another_was(
        self,
    ) -> None:
        a, b, c = 8, 12, 21
        # far = at least 0.3 from nominal; resetting a alone fails at first, resetting b works
        # and then a can go too, but never everything at once
        table = {
            (1, 1, 1): True,
            (0, 1, 1): False,
            (1, 0, 1): True,
            (1, 0, 0): False,
            (0, 0, 1): True,
        }

        def pred(x: Theta) -> bool:
            return table.get(tuple(int(abs(x[i] - NOM[i]) >= 0.3) for i in (a, b, c)), False)  # type: ignore[arg-type]

        x = NOM.copy()
        x[[a, b, c]] = 0.9
        assert pred(x)
        active = [a, b, c]
        _reset(x, active, NOM, pred)
        assert active == [c] and pred(x)
        assert x[a] == NOM[a] and x[b] == NOM[b]
