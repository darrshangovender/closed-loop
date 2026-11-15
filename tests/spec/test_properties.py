"""P2: the six properties, their margins and the signals they read."""

import numpy as np
import pytest

from closed_loop.core.trace import Trace, op_index, permit_index
from closed_loop.core.types import Op, Permit
from closed_loop.spec.properties import (
    PROPERTY_IDS,
    SpecParams,
    evaluate,
    property_formulas,
    signals_from_trace,
)
from closed_loop.spec.stl import robustness
from tests.spec._helpers import (
    ASSETS,
    CONVEYOR,
    CRUSHER,
    N_Q,
    PUMPS,
    clean_trace,
    violating_trace,
)

PARAMS = SpecParams()
TRACE_FIELDS = (
    "op",
    "permit",
    "occupied",
    "gas_age_min",
    "alarm",
    "iso_proven",
    "permit_margin_min",
    "true_rul_days",
    "uncontrolled_failure",
)


def margins(tr: Trace, p: SpecParams = PARAMS) -> dict[str, float]:
    return evaluate(tr, p, CRUSHER, PUMPS)


class TestHandWrittenTraces:
    def test_clean_trace_is_positive_for_every_property(self) -> None:
        got = margins(clean_trace())
        assert set(got) == set(PROPERTY_IDS)
        assert all(v > 0.0 for v in got.values()), got

    @pytest.mark.parametrize("pid", PROPERTY_IDS)
    def test_each_defect_breaks_exactly_its_own_property(self, pid: str) -> None:
        got = margins(violating_trace(pid))
        negative = sorted(k for k, v in got.items() if v < 0.0)
        assert negative == [pid], got

    def test_the_formulas_are_keyed_p1_to_p6(self) -> None:
        assert list(property_formulas(PARAMS)) == list(PROPERTY_IDS)

    def test_defect_sizes(self) -> None:
        assert margins(violating_trace("P1"))["P1"] == -1.0
        assert margins(violating_trace("P4"))["P4"] == -0.5
        assert margins(violating_trace("P6"))["P6"] == -1.0
        assert margins(violating_trace("P3"))["P3"] == -30.0  # 30 minutes allowed, 60 elapsed


class TestMarginUnits:
    def test_p3_margin_is_the_period_minus_minutes_since_the_test(self) -> None:
        for age in (0.0, 15.0, 30.0, 45.0, 60.0):
            tr = clean_trace()
            tr.gas_age_min[CRUSHER][16] = age
            sig = signals_from_trace(tr, PARAMS, CRUSHER, PUMPS)
            assert sig["crusher.gas_fresh"][16] == 30.0 - age

    def test_exactly_at_the_period_is_compliant_and_one_quarter_later_is_not(self) -> None:
        ok, bad = clean_trace(), clean_trace()
        ok.gas_age_min[CRUSHER][16] = 30.0
        bad.gas_age_min[CRUSHER][16] = 45.0
        assert margins(ok)["P3"] == 0.0 and margins(bad)["P3"] == -15.0

    def test_p3_period_follows_the_clause(self) -> None:
        tr = clean_trace()
        tr.gas_age_min[CRUSHER][16] = 90.0
        assert margins(tr, SpecParams(retest_q=2, break_q=1))["P3"] == -60.0
        stale = SpecParams(retest_q=8, break_q=8)  # the 2-hour rule: 120 minutes
        assert margins(tr, stale)["P3"] > 0.0

    def test_p4_margin_is_available_pumps_minus_half(self) -> None:
        tr = clean_trace()
        sig = signals_from_trace(tr, PARAMS, CRUSHER, PUMPS)["pumps.available"]
        assert sig[0] == 1.5 and sig[13] == 0.5  # both RUN, then one pump locked off
        tr.op[PUMPS[0]][:] = op_index(Op.STOP)  # a stopped pump counts as available
        assert signals_from_trace(tr, PARAMS, CRUSHER, PUMPS)["pumps.available"][0] == 1.5

    def test_p2_margin_is_minutes_to_expiry_and_minus_one_while_revalidating(self) -> None:
        tr = clean_trace()
        sig = signals_from_trace(tr, PARAMS, CRUSHER, PUMPS)["crusher.permit_ok"]
        assert sig[10] == 600.0 - 15.0 * 3  # the plant's own margin, below the 12 h cap
        assert sig[3] == -1.0  # no permit yet
        tr.permit[CRUSHER][12] = permit_index(Permit.REVAL)
        assert signals_from_trace(tr, PARAMS, CRUSHER, PUMPS)["crusher.permit_ok"][12] == -1.0

    def test_p2_uses_the_clause_cap_on_permit_age(self) -> None:
        tr = Trace.empty(60, ASSETS)
        tr.permit[CRUSHER][:] = permit_index(Permit.VALID)
        tr.permit_margin_min[CRUSHER][:] = 1.0e6  # the plant claims it never expires
        tr.occupied[CRUSHER][:] = True
        tr.gas_age_min[CRUSHER][:] = 0.0
        short = SpecParams(permit_max_h=4.0)  # 240 minutes = 16 quarters
        got = signals_from_trace(tr, short, CRUSHER, PUMPS)["crusher.permit_ok"]
        assert got[0] == 240.0 and got[16] == 0.0 and got[17] == -15.0
        assert evaluate(tr, short, CRUSHER, PUMPS)["P2"] < 0.0
        assert evaluate(tr, SpecParams(permit_max_h=24.0), CRUSHER, PUMPS)["P2"] > 0.0

    def test_permit_age_counts_from_its_own_first_quarter_not_from_the_trace_start(self) -> None:
        """Mutant found by the pass: age measured from quarter 0 instead of the permit's issue."""
        tr = Trace.empty(60, ASSETS)
        tr.permit[CRUSHER][30:50] = permit_index(Permit.VALID)
        tr.permit[CRUSHER][50:] = permit_index(Permit.NONE)
        tr.permit_margin_min[CRUSHER][:] = 1.0e6
        short = SpecParams(permit_max_h=4.0)
        got = signals_from_trace(tr, short, CRUSHER, PUMPS)["crusher.permit_ok"]
        assert got[30] == 240.0 and got[31] == 225.0 and got[46] == 0.0 and got[47] == -15.0
        # a second permit later starts its own clock
        tr.permit[CRUSHER][55:] = permit_index(Permit.VALID)
        again = signals_from_trace(tr, short, CRUSHER, PUMPS)["crusher.permit_ok"]
        assert again[55] == 240.0 and again[50] == -1.0

    def test_p5_margin_is_remaining_life_in_hours(self) -> None:
        tr = clean_trace()
        sig = signals_from_trace(tr, PARAMS, CRUSHER, PUMPS)["crusher.no_failure"]
        assert sig[0] == 20.0 * 24.0
        tr.true_rul_days[CRUSHER][:] = np.nan  # truth not recorded
        assert signals_from_trace(tr, PARAMS, CRUSHER, PUMPS)["crusher.no_failure"][0] == 1.0

    def test_signals_are_float32_and_trace_length(self) -> None:
        sig = signals_from_trace(clean_trace(), PARAMS, CRUSHER, PUMPS)
        assert all(v.dtype == np.float32 and v.shape == (N_Q,) for v in sig.values())
        for f in property_formulas(PARAMS).values():
            assert f.atoms() <= set(sig)


class TestBreakRule:
    def _reentry(self, gap: int, age_at_reentry: float, p: SpecParams) -> float:
        """Occupied 0..4, empty for ``gap`` quarters, then occupied again with the given age."""
        n = 5 + gap + 4
        tr = Trace.empty(n, ASSETS)
        tr.permit[CRUSHER][:] = permit_index(Permit.VALID)
        tr.permit_margin_min[CRUSHER][:] = 500.0
        tr.occupied[CRUSHER][:5] = True
        tr.occupied[CRUSHER][5 + gap :] = True
        tr.gas_age_min[CRUSHER][:] = 0.0
        tr.gas_age_min[CRUSHER][5 + gap] = age_at_reentry
        return evaluate(tr, p, CRUSHER, PUMPS)["P3"]

    def test_a_break_longer_than_break_q_needs_a_test_taken_after_it_began(self) -> None:
        p = SpecParams(
            retest_q=6, break_q=1
        )  # 90 min re-test, but a break over 15 min needs a retest
        # empty for 3 quarters (45 min): a 60-minute-old test pre-dates the break
        assert self._reentry(3, 60.0, p) == -15.0
        assert self._reentry(3, 45.0, p) == 0.0  # taken the instant the space emptied
        assert self._reentry(3, 15.0, p) == 30.0

    def test_a_break_of_exactly_break_q_is_not_a_break(self) -> None:
        p = SpecParams(retest_q=6, break_q=1)
        assert self._reentry(1, 60.0, p) > 0.0  # one empty quarter = 15 min, not longer than 15

    def test_a_revision_without_a_break_clause_adds_no_constraint(self) -> None:
        p = SpecParams(retest_q=6, break_q=6)
        assert self._reentry(3, 60.0, p) > 0.0

    def test_first_entry_is_judged_by_the_period_alone(self) -> None:
        tr = Trace.empty(8, ASSETS)
        tr.permit[CRUSHER][:] = permit_index(Permit.VALID)
        tr.permit_margin_min[CRUSHER][:] = 500.0
        tr.gas_age_min[CRUSHER][:] = 0.0
        tr.occupied[CRUSHER][5:] = True
        assert evaluate(tr, PARAMS, CRUSHER, PUMPS)["P3"] > 0.0


class TestP6:
    def _trace(self, alarm_at: int, occupied_until: int) -> Trace:
        tr = Trace.empty(12, ASSETS)
        tr.permit[CRUSHER][:] = permit_index(Permit.VALID)
        tr.permit_margin_min[CRUSHER][:] = 500.0
        tr.gas_age_min[CRUSHER][:] = 0.0
        tr.occupied[CRUSHER][2:occupied_until] = True
        tr.alarm[CRUSHER][alarm_at] = True
        return tr

    def test_leaving_in_the_quarter_after_the_alarm_is_compliant(self) -> None:
        assert margins(self._trace(alarm_at=4, occupied_until=5))["P6"] > 0.0

    def test_staying_a_quarter_too_long_violates(self) -> None:
        assert margins(self._trace(alarm_at=4, occupied_until=6))["P6"] < 0.0

    def test_the_allowance_follows_the_clause(self) -> None:
        tr = self._trace(alarm_at=4, occupied_until=6)  # two quarters inside from the alarm
        assert evaluate(tr, SpecParams(alarm_exit_q=1), CRUSHER, PUMPS)["P6"] < 0.0
        assert evaluate(tr, SpecParams(alarm_exit_q=2), CRUSHER, PUMPS)["P6"] > 0.0

    def test_an_alarm_in_an_empty_space_is_harmless(self) -> None:
        tr = self._trace(alarm_at=9, occupied_until=5)
        assert margins(tr)["P6"] > 0.0


class TestDecomposition:
    """A_dec (0001 section 4.2): no atom reads two components, except the pump pair."""

    def test_atoms_are_namespaced_by_one_role(self) -> None:
        roles = {"crusher", "pump_a", "pump_b", "conveyor", "pumps"}
        for f in property_formulas(PARAMS).values():
            for atom in f.atoms():
                assert atom.split(".")[0] in roles, atom

    def test_each_atom_changes_with_one_asset_only_except_the_pump_pair(self) -> None:
        rng = np.random.default_rng(31)
        from tests.spec._helpers import random_trace

        reads: dict[str, set[str]] = {}
        for _ in range(40):
            base = random_trace(rng, 30)
            ref = signals_from_trace(base, PARAMS, CRUSHER, PUMPS)
            for asset in ASSETS:
                other = random_trace(rng, 30)
                mixed = random_trace(rng, 30)
                for a in ASSETS:  # take everything from base, but one asset from `other`
                    src = other if a == asset else base
                    for name in TRACE_FIELDS:
                        getattr(mixed, name)[a][:] = getattr(src, name)[a]
                new = signals_from_trace(mixed, PARAMS, CRUSHER, PUMPS)
                for atom in ref:
                    if not np.array_equal(ref[atom], new[atom]):
                        reads.setdefault(atom, set()).add(asset)
        assert {a for a, s in reads.items() if len(s) > 1} == {"pumps.available"}
        assert reads["pumps.available"] == set(PUMPS)
        assert reads["crusher.permit_ok"] == {CRUSHER}
        assert reads["conveyor.working"] == {CONVEYOR} == reads["conveyor.iso_proven"]


class TestP1CoversTheConveyor:
    """ADR 0003 A12.3: P1 gains a fourth conjunct, ``conveyor.working -> conveyor.iso_proven``.

    The plant recorded P1 events on CV-101 (a careless BEGIN on a running belt) that the
    specification could not see; the closed loop counted 41 of them in 200 episodes.
    """

    def test_work_on_an_unproven_conveyor_is_a_p1_violation_and_only_that(self) -> None:
        got = margins(violating_trace("P1-conveyor"))
        assert sorted(k for k, v in got.items() if v < 0.0) == ["P1"], got
        assert got["P1"] == -1.0

    def test_the_formula_names_the_conveyor_beside_the_crusher_and_the_pumps(self) -> None:
        atoms = property_formulas(PARAMS)["P1"].atoms()
        assert {"conveyor.working", "conveyor.iso_proven"} <= set(atoms)
        assert {a.split(".")[0] for a in atoms} == {"crusher", "pump_a", "pump_b", "conveyor"}

    def test_work_on_a_proven_conveyor_is_fine(self) -> None:
        tr = clean_trace()
        tr.op[CONVEYOR][5:12] = op_index(Op.WORK)
        tr.iso_proven[CONVEYOR][5:12] = True
        assert margins(tr)["P1"] > 0.0

    def test_the_signal_is_the_trace_of_the_conveyor_and_not_of_another_asset(self) -> None:
        tr = clean_trace()
        tr.op[CONVEYOR][8:10] = op_index(Op.WORK)  # unproven
        tr.op[CRUSHER][20:22] = op_index(Op.WORK)  # the crusher's own work, proven below
        tr.iso_proven[CRUSHER][20:22] = True
        sig = signals_from_trace(tr, PARAMS, CRUSHER, PUMPS)
        assert list(sig["conveyor.working"][7:11]) == [-1.0, 1.0, 1.0, -1.0]
        assert (sig["conveyor.iso_proven"] == -1.0).all()
        assert margins(tr)["P1"] == -1.0

    def test_a_trace_without_the_conveyor_is_refused(self) -> None:
        tr = Trace.empty(N_Q, (*PUMPS, CRUSHER, "CV-999"))
        with pytest.raises(ValueError, match="conveyor 'CV-101' is not in the trace"):
            signals_from_trace(tr, PARAMS, CRUSHER, PUMPS)
        assert "conveyor.working" in signals_from_trace(tr, PARAMS, CRUSHER, PUMPS, "CV-999")

    def test_the_conveyor_cannot_be_the_crusher_or_a_pump(self) -> None:
        tr = clean_trace()
        for bad in (CRUSHER, PUMPS[1]):
            with pytest.raises(ValueError, match="conveyor cannot"):
                signals_from_trace(tr, PARAMS, CRUSHER, PUMPS, bad)


class TestRefusals:
    def test_assets_must_exist_and_differ(self) -> None:
        tr = clean_trace()
        with pytest.raises(ValueError, match="not in the trace"):
            signals_from_trace(tr, PARAMS, "NOPE", PUMPS)
        with pytest.raises(ValueError, match="not in the trace"):
            signals_from_trace(tr, PARAMS, CRUSHER, (PUMPS[0], "NOPE"))
        with pytest.raises(ValueError, match="two different"):
            signals_from_trace(tr, PARAMS, CRUSHER, (PUMPS[0], PUMPS[0]))
        with pytest.raises(ValueError, match="crusher cannot"):
            signals_from_trace(tr, PARAMS, PUMPS[0], (PUMPS[0], PUMPS[1]))

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"retest_q": 0},
            {"break_q": 0},
            {"alarm_exit_q": 0},
            {"retest_q": True},
            {"retest_q": 1.5},
            {"permit_max_h": 0.0},
            {"permit_max_h": float("inf")},
        ],
    )
    def test_bad_parameters(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            SpecParams(**kwargs)

    def test_params_are_frozen_and_copy_their_provenance(self) -> None:
        source = {"retest_q": "clause"}
        p = SpecParams(provenance=source)
        source["retest_q"] = "changed"
        assert p.provenance["retest_q"] == "clause"
        with pytest.raises(AttributeError):
            p.retest_q = 5  # type: ignore[misc]

    def test_empty_trace(self) -> None:
        tr = Trace.empty(0, ASSETS)
        assert all(v == float("inf") for v in margins(tr).values())

    def test_robustness_matches_evaluate(self) -> None:
        tr = violating_trace("P3")
        sig = signals_from_trace(tr, PARAMS, CRUSHER, PUMPS)
        f = property_formulas(PARAMS)["P3"]
        assert float(robustness(f, sig).min()) == margins(tr)["P3"]
