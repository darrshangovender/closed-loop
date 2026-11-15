"""P3a: the degradation SCM, life draws, the wear clock and the no-intervention life table."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from closed_loop.core.timegrid import DAY_Q
from closed_loop.plant.degradation import (
    AssetClock,
    build_physical,
    draw_life,
    first_life,
    next_life,
    severity,
)
from closed_loop.plant.observe import simulate_lives
from closed_loop.plant.params import (
    ASSET_CLASS,
    ASSET_IDS,
    MODES_BY_CLASS,
    PlantParams,
)
from closed_loop.plant.scenario import THETA_DIM, Scenario

from ._helpers import quiet_params, theta_of

P = PlantParams()
NOMINAL = Scenario()

theta_strategy = st.lists(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=THETA_DIM, max_size=THETA_DIM
).map(np.array)


def physical(seed: int, scenario: Scenario = NOMINAL, params: PlantParams = P):
    return build_physical(params, seed, scenario.resolve(seed, params))


class TestLifeStatistics:
    """0001 section 2 and P3a done criteria, over 2,000 seeds of the four assets."""

    N_SEEDS = 2000

    @pytest.fixture(scope="class")
    @classmethod
    def lives(cls):
        rows = []
        for seed in range(cls.N_SEEDS):
            phys = physical(seed)
            for aid in ASSET_IDS:
                life = first_life(phys, aid)
                cum = phys.cum[aid]
                onset_day = float(np.searchsorted(cum, life.tau_onset, side="left")) / DAY_Q
                fail_day = float(np.searchsorted(cum, life.tau_fail, side="left")) / DAY_Q
                rows.append((aid, life, onset_day, fail_day))
        return rows

    def test_sixty_percent_fail_inside_the_horizon(self, lives, record_property):
        k = sum(1 for _, _, _, f in lives if f < P.horizon_days)
        n = len(lives)
        record_property("failing_inside_horizon", f"{k}/{n} = {k / n:.4f}")
        # target 0.6; 8,000 lives give a standard error of 0.0055, so 0.03 is more than five of them
        assert abs(k / n - P.p_fail_in_episode) < 0.03, f"{k}/{n}"

    def test_the_designed_flag_agrees_with_the_realised_failure_in_the_nominal_condition(
        self, lives
    ):
        # no shift and wear rate 1: the design clock IS the clock, so the flag is exact
        wrong = [1 for _, life, _, f in lives if (f < P.horizon_days) != life.will_fail]
        assert len(wrong) == 0

    def test_durations_and_shapes_stay_in_the_catalogue_ranges(self, lives):
        for aid, life, _, _ in lives:
            spec = next(m for m in MODES_BY_CLASS[ASSET_CLASS[aid]] if m.mode == life.mode)
            lo, hi = spec.duration_days
            # the design clock is within a few percent of 1.0, and a clamp can shorten a life
            # slightly
            assert lo * 0.85 <= life.duration <= hi * 1.15, (aid, life.mode, life.duration)
            assert spec.shape[0] <= life.shape <= spec.shape[1]
            assert life.signature == spec.signature

    def test_first_onsets_are_never_before_day_24(self, lives):
        assert min(o for _, _, o, _ in lives) >= 24.0

    def test_mode_shares_follow_the_catalogue_weights(self, lives):
        for cls, modes in MODES_BY_CLASS.items():
            mine = [life.mode for aid, life, _, _ in lives if ASSET_CLASS[aid] == cls]
            for m in modes:
                share = mine.count(m.mode) / len(mine)
                assert abs(share - m.weight) < 0.03, (cls, m.mode, share)

    def test_calendar_duration_is_the_nominal_duration_scaled_by_the_load_clock(self, lives):
        # the clock runs between exp(beta (lo - 1)) and exp(beta (hi - 1)) of real time
        lo_c, hi_c = P.load_clip_in
        fastest, slowest = np.exp(P.beta_load * (hi_c - 1)), np.exp(P.beta_load * (lo_c - 1))
        for aid, life, onset_day, fail_day in lives:
            if fail_day < P.horizon_days:
                cal = fail_day - onset_day
                assert life.duration / fastest - 0.1 <= cal <= life.duration / slowest + 0.1, aid


class TestFirstOnsetProperty:
    @given(theta_strategy, st.sampled_from(["in", "out"]), st.integers(0, 10_000))
    @settings(max_examples=60)
    def test_first_onset_never_before_day_24_for_any_scenario(self, theta, regime, seed):
        table = simulate_lives(P, seed, Scenario.decode(theta, regime))
        assert min(table.onset_day.values()) >= 24.0

    @given(theta_strategy, st.integers(0, 10_000))
    @settings(max_examples=40)
    def test_failure_is_after_onset(self, theta, seed):
        table = simulate_lives(P, seed, Scenario.decode(theta, "out"))
        for aid in ASSET_IDS:
            assert table.failure_day[aid] > table.onset_day[aid]


class TestTwinProperty:
    """The life table is bit-identical when only an unrelated stream or coordinate changes."""

    @staticmethod
    def same(a, b, assets=ASSET_IDS) -> bool:
        for aid in assets:
            for name in ("hi", "sig", "load", "load_true", "severity", "true_rul_days"):
                x, y = getattr(a, name)[aid], getattr(b, name)[aid]
                if x.tobytes() != y.tobytes():
                    return False
            if a.lives[aid] != b.lives[aid]:
                return False
        return True

    def test_changing_one_assets_coordinates_leaves_the_other_assets_identical(self):
        base = np.full(THETA_DIM, 0.4)
        a = simulate_lives(P, 17, Scenario.decode(base, "in"))
        for victim in range(4):
            moved = base.copy()
            moved[5 * victim : 5 * victim + 5] = [0.9, 0.1, 0.8, 0.9, 0.3]
            b = simulate_lives(P, 17, Scenario.decode(moved, "in"))
            others = [x for i, x in enumerate(ASSET_IDS) if i != victim]
            assert self.same(a, b, others), f"asset {victim} leaked into the others"
            assert not self.same(a, b, [ASSET_IDS[victim]]), "the victim should have changed"

    def test_coordinates_that_do_not_touch_physics_leave_the_whole_table_identical(self):
        base = np.full(THETA_DIM, 0.4)
        a = simulate_lives(P, 9, Scenario.decode(base, "in"))
        for idx in (20, 21, 22, 23, 24, 25, 26, 27, 28, 29):
            moved = base.copy()
            moved[idx] = 0.95
            b = simulate_lives(P, 9, Scenario.decode(moved, "in"))
            assert self.same(a, b), f"coordinate {idx} should not move the life table"

    def test_the_load_shift_moves_every_asset_but_keeps_the_exogenous_noise(self):
        a = simulate_lives(P, 4, Scenario.decode(theta_of(c30=0.0), "in"))
        b = simulate_lives(P, 4, Scenario.decode(theta_of(c30=1.0, c31=0.0), "in"))
        assert not self.same(a, b)
        # readings before the shift day differ only through the stream-free wear clock, never the
        # noise: the observation noise arrays are keyed by (stream, asset), not by the scenario
        for aid in ASSET_IDS:
            assert a.lives[aid].mode == b.lives[aid].mode
            assert a.lives[aid].shape == b.lives[aid].shape

    def test_a_different_seed_changes_everything(self):
        a = simulate_lives(P, 1, NOMINAL)
        b = simulate_lives(P, 2, NOMINAL)
        assert not self.same(a, b)

    def test_same_seed_same_table_bit_for_bit(self):
        assert self.same(simulate_lives(P, 5, NOMINAL), simulate_lives(P, 5, NOMINAL))

    def test_a_life_draw_does_not_depend_on_which_other_lives_were_drawn(self):
        phys = physical(3)
        alone = next_life(phys, "PU-201", 2, 10.0)
        for other in range(1, 6):
            next_life(phys, "CR-501", other, 5.0)
        again = next_life(phys, "PU-201", 2, 10.0)
        assert alone == again
        assert alone != next_life(phys, "PU-201", 3, 10.0)


class TestLoadAndAPhys:
    @given(theta_strategy, st.integers(0, 5000))
    @settings(max_examples=80)
    def test_regime_in_never_lets_the_wear_rate_jump_by_a_factor_of_three(self, theta, seed):
        phys = physical(seed, Scenario.decode(theta, "in"))
        lo, hi = P.load_clip_in
        for aid in ASSET_IDS:
            rho = phys.rho_day[aid]
            assert phys.load_day[aid].min() >= lo - 1e-12 and phys.load_day[aid].max() <= hi + 1e-12
            ratio = rho[1:] / rho[:-1]
            assert ratio.max() < 3.0 and ratio.min() > 1.0 / 3.0
            # the same bound for the derated clock
            rd = phys.derated(aid)[1]
            assert (rd[1:] / rd[:-1]).max() < 3.0

    def test_regime_in_load_shift_stays_under_the_cap_at_the_largest_setting(self):
        sc = Scenario.decode(theta_of(c30=1.0, c31=0.3), "in")
        phys = physical(1, sc)
        assert phys.resolved.load_shift_frac == pytest.approx(0.6)
        for aid in ASSET_IDS:
            rho = phys.rho_day[aid]
            assert (rho[1:] / rho[:-1]).max() < 3.0

    def test_regime_out_can_exceed_the_cap_so_the_test_above_has_teeth(self):
        worst = 0.0
        for seed in range(40):
            sc = Scenario.decode(theta_of(c30=1.0, c31=0.4), "out")
            phys = physical(seed, sc)
            for aid in ASSET_IDS:
                rho = phys.rho_day[aid]
                worst = max(worst, float((rho[1:] / rho[:-1]).max()))
        assert worst > 3.0

    def test_the_shifted_condition_raises_load_from_the_shift_day(self):
        phys = physical(2, Scenario.shifted())
        day = phys.shift_day
        assert 30 <= day <= 60
        for aid in ASSET_IDS:
            load = phys.load_day[aid]
            assert load[day:].mean() > load[:day].mean() + 0.15

    def test_the_load_is_shared_inside_a_circuit_and_not_across(self):
        phys = physical(8)
        d_pump = phys.load_day["PU-201"] - phys.load_day["PU-202"]
        d_cross = phys.load_day["PU-201"] - phys.load_day["CR-501"]
        # same circuit: the difference is the constant per-asset offset; across circuits it moves
        assert d_pump.std() < 1e-12
        assert d_cross.std() > 1e-3

    def test_a_load_rise_speeds_the_clock_by_exp_beta_dl(self):
        quiet = quiet_params()
        phys = physical(1, Scenario.shifted(), quiet)
        day = phys.shift_day
        rho = phys.rho_day["CV-101"]
        assert rho[0] == pytest.approx(1.0)
        assert rho[day + 1] == pytest.approx(np.exp(quiet.beta_load * 0.25))


class TestWearClock:
    def test_cumulative_clock_is_one_day_per_day_at_load_one(self):
        phys = physical(1, NOMINAL, quiet_params())
        cum = phys.cum["CV-101"]
        assert cum[DAY_Q * 10] == pytest.approx(10.0)
        assert cum[0] == 0.0 and np.all(np.diff(cum) > 0)

    def test_realised_rul_falls_by_exactly_one_day_per_day_while_running(self):
        table = simulate_lives(P, 6, NOMINAL)
        for aid in ASSET_IDS:
            r = table.true_rul_days[aid]
            alive = r[:-1] > 1.0
            assert np.allclose(r[:-1][alive] - r[1:][alive], 1.0, atol=1e-9)

    def test_failure_is_where_the_clock_reaches_tau_fail(self):
        phys = physical(6)
        life = first_life(phys, "PU-201")
        clock = AssetClock(phys, "PU-201", life)
        q = clock.fail_q()
        assert clock.tau(q) >= life.tau_fail > clock.tau(q - 1)

    def test_a_stopped_asset_does_not_age_and_failure_moves_later(self):
        phys = physical(6)
        life = first_life(phys, "PU-201")
        clock = AssetClock(phys, "PU-201", life)
        q_fail = clock.fail_q()
        clock.set_running(1000, False)
        assert clock.fail_q() > 10**9  # not running: no failure scheduled
        tau_stop = clock.tau(1000)
        assert clock.tau(2000) == tau_stop
        clock.set_running(3000, True)
        assert clock.tau(3000) == tau_stop
        assert clock.fail_q() == pytest.approx(q_fail + 2000, abs=60)  # 2000 quarters of idleness

    def test_severity_is_zero_before_onset_one_at_failure_and_convex_for_shape_above_one(self):
        phys = physical(1)
        life = first_life(phys, "CV-101")
        assert severity(life, life.tau_onset - 1.0) == 0.0
        assert severity(life, life.tau_fail) == 1.0
        assert severity(life, life.tau_fail + 5.0) == 1.0
        mid = severity(life, life.tau_onset + 0.5 * life.duration)
        assert mid == pytest.approx(0.5**life.shape)
        assert mid < 0.5

    def test_dense_truth_arrays_agree_with_the_scalar_queries(self):
        from closed_loop.core.types import bin_of_rul

        phys = physical(12)
        life = first_life(phys, "CR-501")
        ops = [
            (500, "run", False),
            (900, "run", True),
            (1500, "derate", True),
            (2500, "run", False),
        ]

        def clock_at(q: int) -> AssetClock:
            """A clock that has seen only the changes up to quarter ``q`` (the scalar queries are
            valid for the present state only)."""
            c = AssetClock(phys, "CR-501", life)
            for t, kind, value in ops:
                if t <= q:
                    (c.set_running if kind == "run" else c.set_derated)(t, value)
            return c

        n_q = 4000
        full = clock_at(n_q)
        rul, bins, life_ids = full.truth_arrays(n_q)
        rng = np.random.default_rng(0)
        for q in sorted(
            {0, 499, 500, 899, 900, 1499, 1500, 2499, 2500, 3999, *rng.integers(0, n_q, 80)}
        ):
            c = clock_at(int(q))
            # row q is the end of step q, so the truth is the clock's reading at the start of q + 1
            assert rul[q] == pytest.approx(c.realised_rul_days(int(q) + 1), abs=1e-4), q
            assert bins[q] == bin_of_rul(c.rate_rul_days(int(q) + 1)), q
        assert set(life_ids.tolist()) == {0}

    def test_renewal_starts_a_fresh_life_and_clears_the_derate(self):
        phys = physical(4)
        clock = AssetClock(phys, "PU-202", first_life(phys, "PU-202"))
        clock.set_derated(100, True)
        old = clock.life
        clock.start_life(2000, next_life(phys, "PU-202", 1, clock.tau(2000)))
        assert clock.life.life_id == 1 and not clock.derated
        assert clock.remaining_clock(2000) == pytest.approx(
            clock.life.tau_fail - clock.life.tau_start
        )
        assert clock.life.tau_fail > old.tau_start
        _, _, life_ids = clock.truth_arrays(3000)
        assert life_ids[1999] == 0 and life_ids[2000] == 1


class TestModeTilt:
    def test_zero_tilt_ignores_the_load_level(self):
        for lvl in (-0.3, 0.3):
            modes = [draw_life(P, s, "PU-201", 0, tau_start=0.0, first=False, load_level=lvl).mode
                     for s in range(1500)]  # fmt: skip
            assert abs(modes.count("IMPELLER_WEAR") / 1500 - 0.7) < 0.04

    def test_tilt_makes_wear_likelier_under_heavy_load_and_reverses_with_its_sign(self):
        def wear_share(tilt: float, level: float) -> float:
            p = PlantParams(mode_load_tilt=tilt)
            modes = [draw_life(p, s, "PU-201", 0, tau_start=0.0, first=False, load_level=level).mode
                     for s in range(2000)]  # fmt: skip
            return modes.count("IMPELLER_WEAR") / 2000

        heavy, light = wear_share(6.0, 0.2), wear_share(6.0, -0.2)
        assert heavy > light + 0.2
        assert wear_share(-6.0, 0.2) < wear_share(-6.0, -0.2) - 0.2

    def test_a_single_mode_class_is_unaffected_by_the_tilt(self):
        p = PlantParams(mode_load_tilt=9.0)
        assert {draw_life(p, s, "CV-101", 0, tau_start=0.0, first=False, load_level=0.5).mode
                for s in range(50)} == {"BEARING_DEFECT"}  # fmt: skip


class TestLifeTableShape:
    def test_arrays_cover_the_horizon_and_are_read_consistent(self):
        t = simulate_lives(P, 3, NOMINAL)
        assert t.n_days == 90
        for aid in ASSET_IDS:
            assert t.hi[aid].shape == (90,) and t.sig[aid].shape == (3, 90)
            assert t.load[aid].shape == (90,) and t.true_rul_days[aid].shape == (90,)
            assert t.fails_in_horizon(aid) == (t.failure_day[aid] < 90)

    def test_failure_day_is_where_the_true_rul_reaches_zero(self):
        t = simulate_lives(P, 6, NOMINAL)
        for aid in ASSET_IDS:
            if t.fails_in_horizon(aid):
                day = int(t.failure_day[aid])  # failure falls inside this day
                assert t.true_rul_days[aid][day] > 0.0  # read at its start: still alive
                assert np.all(t.true_rul_days[aid][day + 1 :] == 0.0)
            else:
                assert t.true_rul_days[aid][-1] > 0.0
