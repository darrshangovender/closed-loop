"""P3a: Scenario.decode, the 32-dimensional layout of 0001 section 9."""

import dataclasses

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from closed_loop.plant.params import PlantParams
from closed_loop.plant.scenario import FAULT_DRIFT, THETA_DIM, ResolvedScenario, Scenario

from ._helpers import theta_of

P = PlantParams()

theta_strategy = st.lists(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=THETA_DIM, max_size=THETA_DIM
).map(np.array)


def resolved(theta: np.ndarray, regime: str = "in", seed: int = 1) -> ResolvedScenario:
    return Scenario.decode(theta, regime).resolve(seed, P)


def changed_fields(a: ResolvedScenario, b: ResolvedScenario) -> set[str]:
    return {f.name for f in dataclasses.fields(ResolvedScenario)
            if getattr(a, f.name) != getattr(b, f.name)}  # fmt: skip


# index -> the one resolved field it drives (regime "in"; forced_refusal_p is 0 there)
def expected_field(idx: int) -> set[str]:
    if idx < 20:
        a, k = divmod(idx, 5)
        return {
            0: {"rate_mult"},
            1: {"onset_shift_days"},
            2: {"residual_shift_days"},
            3: {"fault_type"},
            4: {"fault_day"},
        }[k]
    return {
        20: {"holder_p"}, 21: {"holder_p"}, 22: {"holder_p"},
        23: {"gas_adv_q"}, 24: {"gas_adv_q"},
        25: {"note_u"}, 26: {"note_q"},
        27: {"approver_delay_mult"}, 28: {"approver_catch_p"}, 29: {"spare_lead_mult"},
        30: {"load_shift_frac"}, 31: {"load_shift_day"},
    }[idx]  # fmt: skip


class TestLayout:
    @pytest.mark.parametrize("idx", range(THETA_DIM))
    def test_each_coordinate_drives_exactly_its_documented_field(self, idx):
        base = resolved(np.full(THETA_DIM, 0.5))
        moved = np.full(THETA_DIM, 0.5)
        moved[idx] = 0.9
        assert changed_fields(base, resolved(moved)) == expected_field(idx)

    def test_per_asset_coordinates_hit_the_right_asset(self):
        base = resolved(np.full(THETA_DIM, 0.5))
        for a in range(4):
            moved = np.full(THETA_DIM, 0.5)
            moved[5 * a] = 1.0
            got = resolved(moved).rate_mult
            assert [i for i in range(4) if got[i] != base.rate_mult[i]] == [a]

    def test_documented_ranges_regime_in(self):
        lo, hi = resolved(np.zeros(THETA_DIM)), resolved(np.ones(THETA_DIM))
        assert (lo.rate_mult[0], hi.rate_mult[0]) == (0.7, 1.4)
        assert (lo.onset_shift_days[0], hi.onset_shift_days[0]) == (-10.0, 10.0)
        assert (lo.residual_shift_days[0], hi.residual_shift_days[0]) == (-15.0, 15.0)
        assert (lo.approver_delay_mult, hi.approver_delay_mult) == (0.5, 3.0)
        assert (lo.spare_lead_mult, hi.spare_lead_mult) == (0.5, 2.0)
        assert (lo.load_shift_frac, hi.load_shift_frac) == (0.0, 0.6)
        assert hi.holder_p == pytest.approx((0.3, 0.3, 0.3))
        assert lo.holder_p == pytest.approx((0.9, 0.9, 0.9))
        assert hi.gas_adv_q == (P.horizon_q - 1, P.horizon_q - 1)
        assert hi.load_shift_day == P.horizon_days - 1
        assert hi.approver_catch_p == 1.0 and lo.approver_catch_p == 0.0


class TestTotality:
    @pytest.mark.parametrize("fill", [0.0, 1.0, 0.5])
    def test_corners_and_centre(self, fill):
        for regime in ("in", "out"):
            r = resolved(np.full(THETA_DIM, fill), regime)
            assert all(np.isfinite(r.rate_mult))
            assert r.fault_type and all(0 <= t <= 3 for t in r.fault_type)

    def test_every_corner_of_the_per_asset_block(self):
        rng = np.random.default_rng(3)
        for _ in range(64):
            theta = rng.integers(0, 2, THETA_DIM).astype(float)
            for regime in ("in", "out"):
                resolved(theta, regime)

    @given(theta_strategy, st.sampled_from(["in", "out"]))
    @settings(max_examples=150)
    def test_total_and_deterministic_on_the_cube(self, theta, regime):
        a = resolved(theta, regime)
        b = resolved(theta.copy(), regime)
        assert a == b
        assert a.fault_type[0] in (0, 1, 2, 3)

    def test_values_outside_the_cube_are_clipped(self):
        wild = np.full(THETA_DIM, 7.0)
        assert resolved(wild) == resolved(np.ones(THETA_DIM))
        assert resolved(-wild) == resolved(np.zeros(THETA_DIM))

    def test_theta_is_copied_and_read_only(self):
        theta = np.full(THETA_DIM, 0.5)
        sc = Scenario.decode(theta, "in")
        theta[:] = 1.0
        assert sc.theta is not None and np.all(sc.theta == 0.5)
        with pytest.raises(ValueError):
            sc.theta[0] = 0.0

    def test_refuses_bad_input(self):
        with pytest.raises(ValueError):
            Scenario.decode(np.full(THETA_DIM, np.nan), "in")
        with pytest.raises(ValueError):
            Scenario.decode(np.zeros(31), "in")
        with pytest.raises(ValueError):
            Scenario(regime="sideways")
        with pytest.raises(ValueError):
            Scenario(condition="chaotic")


class TestRegimes:
    @given(theta_strategy)
    @settings(max_examples=150)
    def test_regime_in_stays_inside_the_assumptions(self, theta):
        r = resolved(theta, "in")
        assert r.forced_refusal_p == 0.0  # A_exec
        assert FAULT_DRIFT not in r.fault_type  # only faults the gate masks
        assert r.load_shift_frac <= 0.6 + 1e-12
        assert all(0.7 - 1e-12 <= m <= 1.4 + 1e-12 for m in r.rate_mult)
        assert r.load_clip == P.load_clip_in

    def test_regime_out_lifts_each_limit(self):
        r = resolved(np.ones(THETA_DIM), "out")
        assert r.forced_refusal_p == pytest.approx(0.3)
        assert r.load_shift_frac == 2.0
        assert r.rate_mult[0] == 2.0
        assert r.load_clip == P.load_clip_out
        drift = resolved(theta_of(c3=0.99), "out")
        assert drift.fault_type[0] == FAULT_DRIFT

    def test_forced_refusal_scales_with_holder_absence_only_in_out(self):
        for bias in (0.0, 0.5, 1.0):
            th = theta_of(c20=bias, c21=bias, c22=bias)
            assert resolved(th, "out").forced_refusal_p == pytest.approx(0.3 * bias)
            assert resolved(th, "in").forced_refusal_p == 0.0

    def test_the_a_phys_clip_keeps_the_rate_ratio_below_three(self):
        lo, hi = P.load_clip_in
        assert np.exp(P.beta_load * (hi - lo)) < 3.0
        lo, hi = P.load_clip_out
        assert np.exp(P.beta_load * (hi - lo)) > 3.0


class TestConditions:
    def test_nominal_has_no_shift(self):
        r = Scenario().resolve(5, P)
        assert (r.load_shift_frac, r.drift_all_per_day, r.forced_refusal_p) == (0.0, 0.0, 0.0)
        assert r.rate_mult == (1.0,) * 4 and not r.from_theta

    def test_shifted_matches_the_preregistered_magnitudes(self):
        days = []
        for seed in range(200):
            r = Scenario.shifted().resolve(seed, P)
            assert r.load_shift_frac == 0.25 and r.drift_all_per_day == 0.005
            assert r.drift_all_day == r.load_shift_day
            days.append(r.load_shift_day)
        assert 30.0 <= min(days) and max(days) <= 60.0
        assert 40.0 < np.mean(days) < 50.0  # uniform on 30-60
        assert len(set(days)) > 150

    def test_shift_day_is_keyed_by_the_seed(self):
        a = Scenario.shifted().resolve(11, P).load_shift_day
        assert a == Scenario.shifted().resolve(11, P).load_shift_day
        assert a != Scenario.shifted().resolve(12, P).load_shift_day

    def test_with_theta_the_vector_governs(self):
        r = Scenario(theta=np.zeros(THETA_DIM), condition="shifted").resolve(3, P)
        assert r.load_shift_frac == 0.0 and r.from_theta
