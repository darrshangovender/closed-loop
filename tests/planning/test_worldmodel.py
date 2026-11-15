"""P8a: particle sets, the two world models, the failure rollout and the development fit."""

import math

import numpy as np
import pytest

from closed_loop.planning.worldmodel import (
    ACTION_CONTINUE,
    ACTION_INSPECT,
    ACTION_NAMES,
    ACTION_REPLACE,
    Particles,
    PlantWorldModel,
    RenewableWorldModel,
    TabularWorldModel,
    WorldModel,
    failure_days,
    weighted_quantile,
)

from ._helpers import COST, FITTED, plant_model, quiet_model, small_tabular


def uniform(n: int, **cols: np.ndarray) -> Particles:
    return Particles({k: np.asarray(v, dtype=float) for k, v in cols.items()}, np.full(n, 1.0 / n))


class TestParticles:
    def test_weights_must_be_a_distribution(self):
        with pytest.raises(ValueError, match="sum to 1"):
            Particles({"a": np.zeros(3)}, np.array([0.5, 0.2, 0.2]))
        with pytest.raises(ValueError, match="non-negative"):
            Particles({"a": np.zeros(2)}, np.array([1.5, -0.5]))
        with pytest.raises(ValueError, match="finite"):
            Particles({"a": np.zeros(2)}, np.array([np.nan, 1.0]))
        with pytest.raises(ValueError, match="non-empty"):
            Particles({"a": np.zeros(0)}, np.zeros(0))

    def test_state_arrays_must_match_the_weights(self):
        with pytest.raises(ValueError, match="shape"):
            Particles({"a": np.zeros(4)}, np.full(3, 1 / 3))
        with pytest.raises(ValueError, match="at least one"):
            Particles({}, np.full(3, 1 / 3))

    def test_arrays_are_read_only(self):
        p = uniform(4, a=np.arange(4.0))
        with pytest.raises(ValueError):
            p.state["a"][0] = 9.0
        with pytest.raises(ValueError):
            p.weights[0] = 0.9
        with pytest.raises(TypeError):
            p.state["b"] = np.zeros(4)  # type: ignore[index]

    def test_ess_is_n_for_uniform_weights_and_one_for_a_point_mass(self):
        assert uniform(10, a=np.zeros(10)).ess() == pytest.approx(10.0)
        w = np.zeros(10)
        w[3] = 1.0
        assert Particles({"a": np.zeros(10)}, w).ess() == pytest.approx(1.0)

    def test_take_gathers_with_repeats_and_resets_the_weights(self):
        p = Particles({"a": np.array([10.0, 20.0, 30.0])}, np.array([0.1, 0.2, 0.7]))
        q = p.take(np.array([2, 2, 0, 2]))
        assert list(q.state["a"]) == [30.0, 30.0, 10.0, 30.0]
        assert np.allclose(q.weights, 0.25)

    def test_with_weights_checks_unless_trusted(self):
        p = uniform(3, a=np.zeros(3))
        with pytest.raises(ValueError):
            p.with_weights(np.array([0.5, 0.5, 0.5]))
        assert p.with_weights(np.array([0.2, 0.3, 0.5])).weights[2] == 0.5

    def test_mean_uses_the_weights(self):
        p = Particles({"a": np.array([0.0, 10.0])}, np.array([0.25, 0.75]))
        assert p.mean("a") == pytest.approx(7.5)

    def test_weighted_quantile_matches_the_unweighted_one_for_equal_weights(self):
        rng = np.random.default_rng(0)
        v = rng.normal(size=2001)
        got = weighted_quantile(v, np.full(v.size, 1 / v.size), np.array([0.1, 0.5, 0.9]))
        assert np.allclose(got, np.quantile(v, [0.1, 0.5, 0.9]), atol=0.02)

    def test_weighted_quantile_follows_the_weights(self):
        v = np.array([1.0, 2.0, 3.0])
        heavy_top = weighted_quantile(v, np.array([0.01, 0.01, 0.98]), 0.5)[0]
        heavy_bottom = weighted_quantile(v, np.array([0.98, 0.01, 0.01]), 0.5)[0]
        assert heavy_top > 2.9 and heavy_bottom < 1.1


class TestProtocols:
    def test_both_models_fill_the_protocol(self):
        for m in (plant_model(), small_tabular()):
            assert isinstance(m, WorldModel)
            assert isinstance(m, RenewableWorldModel)

    def test_a_third_party_model_can_fill_the_protocol(self):
        class Adapter:  # what a real model adapter would implement
            def prior(self, rng, n):
                return uniform(n, z=rng.random(n))

            def step(self, p, action, rng):
                return Particles({"z": p.state["z"] + 0.1}, p.weights)

            def loglik(self, p, obs, noise_scale=1.0):
                return np.zeros(p.n) if obs is None else -((obs - p.state["z"]) ** 2)

            def failed(self, p):
                return p.state["z"] > 1.0

        assert isinstance(Adapter(), WorldModel)
        assert not isinstance(Adapter(), RenewableWorldModel)
        assert not isinstance(object(), WorldModel)

    def test_action_names_and_numbers_are_fixed(self):
        assert ACTION_NAMES == ("continue", "inspect", "replace")
        assert (ACTION_CONTINUE, ACTION_INSPECT, ACTION_REPLACE) == (0, 1, 2)


class TestPlantModelConstruction:
    def test_missing_keys_are_all_named(self):
        bad = {k: v for k, v in FITTED.items() if k not in ("beta_load", "first_p_in")}
        with pytest.raises(KeyError, match="beta_load.*first_p_in|first_p_in.*beta_load"):
            PlantWorldModel(bad, asset_class="SLURRY_PUMP")

    def test_a_class_with_no_modes_is_refused(self):
        with pytest.raises(KeyError, match="CRUSHER"):
            PlantWorldModel(FITTED, asset_class="CRUSHER")

    def test_a_missing_mode_field_is_named(self):
        bad = {k: v for k, v in FITTED.items() if k != "SLURRY_PUMP.BEARING_DEFECT.k_hi"}
        with pytest.raises(KeyError, match="BEARING_DEFECT.k_hi"):
            PlantWorldModel(bad, asset_class="SLURRY_PUMP")

    @pytest.mark.parametrize("mismatch", [-1.0, 1.0, 1.5])
    def test_mismatch_must_leave_the_clock_running_forward(self, mismatch):
        with pytest.raises(ValueError, match="mismatch"):
            plant_model(mismatch)

    def test_inconsistent_ranges_are_refused(self):
        with pytest.raises(ValueError, match="durations"):
            plant_model(**{"SLURRY_PUMP.IMPELLER_WEAR.dur_hi": 10.0})
        with pytest.raises(ValueError, match="shapes"):
            plant_model(**{"SLURRY_PUMP.IMPELLER_WEAR.k_lo": 5.0})
        with pytest.raises(ValueError, match="hi_fail"):
            plant_model(hi_fail=0.5)
        with pytest.raises(ValueError, match="hi_noise_sd"):
            plant_model(hi_noise_sd=0.0)

    def test_modes_come_from_the_keys_not_from_the_plant(self):
        assert plant_model().modes == ("BEARING_DEFECT", "IMPELLER_WEAR")


class TestPlantModelPrior:
    def test_a_prior_is_deterministic_given_the_generator(self):
        a = plant_model().prior(np.random.default_rng(5), 64)
        b = plant_model().prior(np.random.default_rng(5), 64)
        c = plant_model().prior(np.random.default_rng(6), 64)
        for k in a.state:
            assert np.array_equal(a.state[k], b.state[k])
        assert not np.array_equal(a.state["x"], c.state["x"])

    def test_a_first_component_starts_before_its_onset_floor(self):
        p = plant_model().prior(np.random.default_rng(1), 4000)
        onset = -p.state["x"] * p.state["dur"]
        assert onset.min() >= 24.0 - 1e-9
        assert np.all(p.state["x"] < 0.0)

    def test_the_share_failing_inside_the_horizon_follows_the_fitted_share(self):
        m = plant_model()
        p = m.prior(np.random.default_rng(2), 20000)
        onset = -p.state["x"] * p.state["dur"]
        fail_day = onset + p.state["dur"]
        assert np.mean(fail_day < 90.0) == pytest.approx(0.6, abs=0.02)

    def test_modes_follow_their_weights_and_ranges(self):
        m = plant_model()
        p = m.prior(np.random.default_rng(3), 20000)
        wear = m.modes.index("IMPELLER_WEAR")
        assert np.mean(p.state["mode"] == wear) == pytest.approx(0.7, abs=0.02)
        for i, name in enumerate(m.modes):
            sel = p.state["mode"] == i
            lo, hi = FITTED[f"SLURRY_PUMP.{name}.dur_lo"], FITTED[f"SLURRY_PUMP.{name}.dur_hi"]
            assert p.state["dur"][sel].min() >= lo and p.state["dur"][sel].max() <= hi
            klo, khi = FITTED[f"SLURRY_PUMP.{name}.k_lo"], FITTED[f"SLURRY_PUMP.{name}.k_hi"]
            assert p.state["k"][sel].min() >= klo and p.state["k"][sel].max() <= khi

    def test_a_renewal_has_its_onset_in_the_renewal_range(self):
        p = plant_model().renewal(np.random.default_rng(4), 4000)
        onset = -p.state["x"] * p.state["dur"]
        assert onset.min() >= 24.0 - 1e-9 and onset.max() <= 60.0 + 1e-9
        assert np.all(p.state["x"] < 0.0)

    def test_weights_start_uniform(self):
        p = plant_model().prior(np.random.default_rng(0), 128)
        assert np.allclose(p.weights, 1 / 128) and p.n == 128


class TestPlantModelDynamics:
    def one(self, x=-0.5, dur=40.0, k=2.0, rate=1.0, mode=1):
        return uniform(1, mode=[mode], x=[x], dur=[dur], k=[k], rate=[rate])

    def test_progress_advances_by_rho_over_duration_with_no_noise(self):
        m = quiet_model()
        p = m.step(self.one(), ACTION_CONTINUE, np.random.default_rng(0), load=1.0)
        assert p.state["x"][0] == pytest.approx(-0.5 + 1.0 / 40.0)

    def test_the_load_sets_the_speed(self):
        m = quiet_model()
        p = m.step(self.one(), ACTION_CONTINUE, np.random.default_rng(0), load=1.4)
        assert p.state["x"][0] == pytest.approx(-0.5 + math.exp(0.8 * 0.4) / 40.0)

    def test_without_a_load_the_forecast_mean_sets_the_speed(self):
        m = quiet_model()
        p = m.step(self.one(), ACTION_CONTINUE, np.random.default_rng(0), load_mean=0.7)
        assert p.state["x"][0] == pytest.approx(-0.5 + math.exp(0.8 * -0.3) / 40.0)

    def test_the_rate_multiplier_scales_the_speed(self):
        m = quiet_model()
        p = m.step(self.one(rate=1.25), ACTION_CONTINUE, np.random.default_rng(0), load=1.0)
        assert p.state["x"][0] == pytest.approx(-0.5 + 1.25 / 40.0)

    def test_inspect_moves_the_state_like_continue(self):
        m = plant_model()
        a = m.step(self.one(), ACTION_CONTINUE, np.random.default_rng(7), load=1.1)
        b = m.step(self.one(), ACTION_INSPECT, np.random.default_rng(7), load=1.1)
        assert np.array_equal(a.state["x"], b.state["x"])

    def test_mismatch_slows_the_model_clock(self):
        slow = quiet_model(0.25).step(self.one(), 0, np.random.default_rng(0), load=1.0)
        true = quiet_model(0.0).step(self.one(), 0, np.random.default_rng(0), load=1.0)
        assert slow.state["x"][0] + 0.5 == pytest.approx(0.75 * (true.state["x"][0] + 0.5))
        fast = quiet_model(-0.25).step(self.one(), 0, np.random.default_rng(0), load=1.0)
        assert fast.state["x"][0] + 0.5 == pytest.approx(1.25 * (true.state["x"][0] + 0.5))

    def test_failure_is_absorbing(self):
        m = quiet_model()
        p = m.step(self.one(x=0.99), ACTION_CONTINUE, np.random.default_rng(0), load=1.0)
        assert m.failed(p)[0] and p.state["x"][0] == 1.0
        again = m.step(p, ACTION_CONTINUE, np.random.default_rng(1), load=1.0)
        assert m.failed(again)[0] and again.state["x"][0] == 1.0

    def test_replace_draws_a_fresh_component(self):
        m = plant_model()
        old = self.one(x=1.0)
        new = m.step(old, ACTION_REPLACE, np.random.default_rng(3))
        assert not m.failed(new)[0] and new.state["x"][0] < 0.0 and new.n == 1

    def test_an_unknown_action_is_refused(self):
        with pytest.raises(ValueError, match="action"):
            plant_model().step(self.one(), 7, np.random.default_rng(0))

    def test_a_step_does_not_touch_its_input(self):
        m = plant_model()
        p = self.one()
        before = p.state["x"].copy()
        m.step(p, 0, np.random.default_rng(0), load=1.0)
        assert np.array_equal(p.state["x"], before)

    def test_stepping_is_deterministic_given_the_generator(self):
        m = plant_model()
        p = m.prior(np.random.default_rng(0), 32)
        a = m.step(p, 0, np.random.default_rng(9))
        b = m.step(p, 0, np.random.default_rng(9))
        assert np.array_equal(a.state["x"], b.state["x"])


class TestPlantModelObservation:
    def test_severity_is_zero_before_onset_and_one_at_failure(self):
        m = quiet_model()
        p = uniform(
            3, mode=[0, 0, 0], x=[-0.3, 0.5, 1.0], dur=[40.0] * 3, k=[2.0] * 3, rate=[1.0] * 3
        )
        assert list(m.severity(p)) == pytest.approx([0.0, 0.25, 1.0])

    def test_the_expected_reading_follows_the_observation_model(self):
        m = quiet_model()
        p = uniform(1, mode=[0], x=[0.5], dur=[40.0], k=[2.0], rate=[1.0])
        # 1 + 8 * 0.25 + 2 * (1.1 - 1)
        assert m.hi_mean(p, 1.1)[0] == pytest.approx(3.2)

    def test_loglik_peaks_at_the_expected_reading(self):
        m = quiet_model()
        p = uniform(1, mode=[0], x=[0.5], dur=[40.0], k=[2.0], rate=[1.0])
        mean = m.hi_mean(p)[0]
        at = m.loglik(p, float(mean))[0]
        assert at == pytest.approx(-math.log(0.3) - 0.5 * math.log(2 * math.pi))
        assert m.loglik(p, float(mean + 0.3))[0] < at
        assert m.loglik(p, float(mean - 0.3))[0] == pytest.approx(m.loglik(p, float(mean + 0.3))[0])

    def test_an_inspection_reading_is_scored_with_a_quarter_of_the_noise(self):
        m = quiet_model()
        p = uniform(1, mode=[0], x=[0.5], dur=[40.0], k=[2.0], rate=[1.0])
        off = float(m.hi_mean(p)[0] + 0.3)  # one daily sd away
        daily = m.loglik(p, off)[0]
        inspected = m.loglik(p, off, noise_scale=0.25)[0]
        assert inspected < daily  # four sd away under the tighter noise
        assert inspected == pytest.approx(-0.5 * 16 - math.log(0.075) - 0.5 * math.log(2 * math.pi))

    def test_no_observation_scores_zero(self):
        m = plant_model()
        p = m.prior(np.random.default_rng(0), 8)
        assert np.array_equal(m.loglik(p, None), np.zeros(8))

    def test_a_non_positive_noise_scale_is_refused(self):
        m = plant_model()
        p = m.prior(np.random.default_rng(0), 4)
        with pytest.raises(ValueError, match="noise_scale"):
            m.loglik(p, 2.0, noise_scale=0.0)

    def test_the_load_enters_the_expected_reading(self):
        m = quiet_model()
        p = m.prior(np.random.default_rng(0), 4)
        assert m.hi_mean(p, 1.3)[0] - m.hi_mean(p, 1.0)[0] == pytest.approx(0.6)
        assert m.loglik(p, 1.6, load=1.3)[0] > m.loglik(p, 1.6, load=1.0)[0]

    def test_a_failed_particle_reads_the_failure_level(self):
        m = quiet_model()
        p = uniform(1, mode=[0], x=[1.0], dur=[40.0], k=[2.0], rate=[1.0])
        assert m.hi_mean(p)[0] == pytest.approx(9.0)
        assert m.failed(p)[0]


class TestRemainingLife:
    def test_remaining_life_matches_the_dynamics_with_no_noise(self):
        m = quiet_model()
        p = uniform(1, mode=[0], x=[0.2], dur=[30.0], k=[2.0], rate=[1.25])
        days = m.remaining_life_days(p)[0]
        assert days == pytest.approx(0.8 * 30.0 / 1.25)
        cur, n = p, 0
        while not m.failed(cur)[0]:
            cur = m.step(cur, 0, np.random.default_rng(0), load=1.0)
            n += 1
        assert n == math.ceil(days)

    def test_a_failed_particle_has_no_remaining_life(self):
        m = quiet_model()
        p = uniform(1, mode=[0], x=[1.0], dur=[30.0], k=[2.0], rate=[1.0])
        assert m.remaining_life_days(p)[0] == 0.0

    def test_the_load_forecast_and_the_mismatch_both_scale_it(self):
        p = uniform(1, mode=[0], x=[0.0], dur=[30.0], k=[2.0], rate=[1.0])
        base = quiet_model().remaining_life_days(p)[0]
        assert quiet_model().remaining_life_days(p, load=1.2)[0] == pytest.approx(
            base / math.exp(0.8 * 0.2)
        )
        assert quiet_model(0.25).remaining_life_days(p)[0] == pytest.approx(base / 0.75)

    def test_failure_days_rollout_stamps_the_middle_of_the_day(self):
        m = quiet_model()
        p = uniform(2, mode=[0, 0], x=[0.5, 1.0], dur=[40.0, 40.0], k=[2.0, 2.0], rate=[1.0, 1.0])
        got = failure_days(m, p, np.random.default_rng(0), 100, load_mean=1.0)
        assert got[1] == 0.0  # already failed
        assert got[0] == pytest.approx(20 - 0.5)  # x = 0.5 + d / 40 reaches 1 on day 20

    def test_failure_days_marks_survivors_as_infinite(self):
        m = quiet_model()
        p = uniform(1, mode=[0], x=[-1.0], dur=[40.0], k=[2.0], rate=[1.0])
        assert np.isinf(failure_days(m, p, np.random.default_rng(0), 10)[0])

    def test_failure_days_rollout_is_deterministic_and_spreads_with_noise(self):
        m = plant_model()
        p = m.prior(np.random.default_rng(0), 128)
        a = failure_days(m, p, np.random.default_rng(1), 300)
        b = failure_days(m, p, np.random.default_rng(1), 300)
        assert np.array_equal(a, b)
        assert np.std(a[np.isfinite(a)]) > 5.0


class TestResample:
    def cloud(self, n=400):
        m = plant_model()
        rng = np.random.default_rng(0)
        p = m.prior(rng, n)
        # A cloud that already knows something: half the weight on a narrow band.
        w = np.where(p.state["dur"] > 55.0, 0.0, 1.0)
        return m, p.with_weights(w / w.sum())

    def test_mode_is_never_changed_and_failed_stays_failed(self):
        m, p = self.cloud()
        x = p.state["x"].copy()
        x[::7] = 1.0
        p = Particles({**p.state, "x": x}, p.weights)
        idx = np.random.default_rng(1).integers(0, p.n, size=p.n)
        q = m.resample(p, idx, np.random.default_rng(2))
        assert np.array_equal(q.state["mode"], p.state["mode"][idx])
        assert np.all(q.state["x"][p.state["x"][idx] >= 1.0] == 1.0)
        assert np.all(q.state["x"] <= 1.0)

    def test_the_cloud_keeps_its_mean_and_variance(self):
        m, p = self.cloud(4000)
        from closed_loop.planning.belief import systematic_resample

        idx = systematic_resample(p.weights, np.random.default_rng(1))
        q = m.resample(p, idx, np.random.default_rng(2), 0.95)
        for name in ("dur", "k"):
            before = np.log(p.state[name])
            after = np.log(q.state[name])
            mb = np.dot(p.weights, before)
            vb = np.dot(p.weights, (before - mb) ** 2)
            assert after.mean() == pytest.approx(mb, abs=0.05 * math.sqrt(vb) + 1e-3)
            assert after.var() == pytest.approx(vb, rel=0.1)

    def test_the_jitter_is_small_but_not_zero(self):
        m, p = self.cloud(200)
        idx = np.zeros(200, dtype=int)
        q = m.resample(p, idx, np.random.default_rng(2))
        assert len(np.unique(q.state["dur"])) == 200  # copies of one particle are pulled apart
        assert np.ptp(np.log(q.state["dur"])) < 0.5

    def test_the_shrink_parameter_is_bounded(self):
        m, p = self.cloud(50)
        with pytest.raises(ValueError, match="shrink_delta"):
            m.resample(p, np.arange(50), np.random.default_rng(0), 0.4)
        with pytest.raises(ValueError, match="shrink_delta"):
            m.resample(p, np.arange(50), np.random.default_rng(0), 1.0)


class TestTabularConstruction:
    def test_rows_must_sum_to_one(self):
        m = small_tabular()
        bad_T = m.T.copy()
        bad_T[1, 0] = [0.5, 0.2, 0.0, 0.0]
        with pytest.raises(ValueError, match="sum to 1"):
            TabularWorldModel(bad_T, m.O, COST)
        bad_O = m.O.copy()
        bad_O[2] = [0.3, 0.3, 0.3]
        with pytest.raises(ValueError, match="sum to 1"):
            TabularWorldModel(m.T, bad_O, COST)

    def test_every_row_of_every_matrix_sums_to_one(self):
        m = small_tabular()
        assert np.allclose(m.T.sum(axis=2), 1.0)
        assert np.allclose(m.O.sum(axis=1), 1.0)
        assert m.O_inspect is not None and np.allclose(m.O_inspect.sum(axis=1), 1.0)

    def test_negative_probabilities_and_bad_shapes_are_refused(self):
        m = small_tabular()
        neg = m.T.copy()
        neg[0, 0] = [1.2, -0.2, 0.0, 0.0]
        with pytest.raises(ValueError, match="non-negative"):
            TabularWorldModel(neg, m.O, COST)
        with pytest.raises(ValueError, match="shape"):
            TabularWorldModel(m.T[:2], m.O, COST)
        with pytest.raises(ValueError, match="shape"):
            TabularWorldModel(m.T, m.O[:3], COST)
        with pytest.raises(ValueError, match="O_inspect"):
            TabularWorldModel(m.T, m.O, COST, O_inspect=m.O[:, :2])

    def test_the_cost_table_needs_every_action_and_the_failure_cost(self):
        m = small_tabular()
        with pytest.raises(KeyError, match="failure"):
            TabularWorldModel(m.T, m.O, {"continue": 0.0, "inspect": 0.1, "replace": 1.0})
        with pytest.raises(ValueError, match="finite"):
            TabularWorldModel(m.T, m.O, {**COST, "failure": float("inf")})

    def test_the_initial_distribution_and_failure_state_are_checked(self):
        m = small_tabular()
        with pytest.raises(ValueError, match="initial"):
            TabularWorldModel(m.T, m.O, COST, initial=np.array([0.5, 0.5]))
        with pytest.raises(ValueError, match="sum to 1"):
            TabularWorldModel(m.T, m.O, COST, initial=np.array([0.5, 0.2, 0.0, 0.0]))
        with pytest.raises(ValueError, match="failure_state"):
            TabularWorldModel(m.T, m.O, COST, failure_state=9)

    def test_the_arrays_are_copied(self):
        m = small_tabular()
        t = m.T.copy()
        model = TabularWorldModel(t, m.O, COST)
        t[0, 0] = [0.0, 0.0, 0.0, 1.0]
        assert model.T[0, 0, 0] == 0.8


class TestTabularBehaviour:
    def test_step_frequencies_match_the_transition_row(self):
        m = small_tabular()
        p = Particles({"s": np.full(40000, 1, dtype=np.int64)}, np.full(40000, 1 / 40000))
        q = m.step(p, ACTION_CONTINUE, np.random.default_rng(0))
        freq = np.bincount(q.state["s"], minlength=4) / 40000
        assert np.allclose(freq, m.T[0, 1], atol=0.01)

    def test_replace_sends_every_particle_to_the_first_level(self):
        m = small_tabular()
        p = Particles({"s": np.array([0, 1, 2, 3], dtype=np.int64)}, np.full(4, 0.25))
        q = m.step(p, ACTION_REPLACE, np.random.default_rng(0))
        assert list(q.state["s"]) == [0, 0, 0, 0]

    def test_prior_follows_the_initial_distribution(self):
        m = TabularWorldModel(
            small_tabular().T, small_tabular().O, COST, initial=np.array([0.5, 0.5, 0, 0])
        )
        p = m.prior(np.random.default_rng(0), 20000)
        assert np.mean(p.state["s"] == 1) == pytest.approx(0.5, abs=0.02)
        assert set(np.unique(p.state["s"])) <= {0, 1}

    def test_loglik_is_the_log_of_the_confusion_entry(self):
        m = small_tabular()
        p = Particles({"s": np.array([0, 1, 2, 3], dtype=np.int64)}, np.full(4, 0.25))
        assert np.allclose(m.loglik(p, 2.0), np.log(m.O[:, 2]))

    def test_an_inspection_reading_uses_the_sharper_matrix(self):
        m = small_tabular()
        p = Particles({"s": np.array([0, 1, 2, 3], dtype=np.int64)}, np.full(4, 0.25))
        assert np.allclose(m.loglik(p, 1.0, noise_scale=0.25), np.log(m.O_inspect[:, 1]))

    def test_without_an_inspection_matrix_the_confusion_is_tempered_and_renormalised(self):
        m = small_tabular(inspect_sharper=False)
        sharp = m.observation_matrix(0.25)
        assert np.allclose(sharp.sum(axis=1), 1.0)
        assert np.all(sharp.max(axis=1) >= m.O.max(axis=1) - 1e-12)  # modes get more mass
        assert np.allclose(m.observation_matrix(1.0), m.O)

    def test_an_impossible_symbol_has_log_probability_minus_infinity(self):
        m = TabularWorldModel(small_tabular().T, np.array([[1, 0], [1, 0], [0, 1], [0, 1.0]]), COST)
        p = Particles({"s": np.array([0, 2], dtype=np.int64)}, np.array([0.5, 0.5]))
        ll = m.loglik(p, 1.0)
        assert ll[0] == -np.inf and ll[1] == 0.0

    def test_symbols_must_be_valid_indices(self):
        m = small_tabular()
        p = m.prior(np.random.default_rng(0), 4)
        for bad in (3.0, -1.0, 0.5):
            with pytest.raises(ValueError, match="symbol"):
                m.loglik(p, bad)

    def test_none_scores_zero_and_failed_marks_the_last_level(self):
        m = small_tabular()
        p = Particles({"s": np.array([0, 3], dtype=np.int64)}, np.array([0.5, 0.5]))
        assert np.array_equal(m.loglik(p, None), np.zeros(2))
        assert list(m.failed(p)) == [False, True]

    def test_the_exact_filter_is_bayes_rule(self):
        m = small_tabular()
        b = np.array([0.6, 0.3, 0.1, 0.0])
        nxt = m.predict(b, ACTION_CONTINUE)
        assert nxt == pytest.approx([0.48, 0.36, 0.14, 0.02])
        post, z = m.posterior(nxt, 2)
        assert z == pytest.approx(float(nxt @ m.O[:, 2]))
        assert post.sum() == pytest.approx(1.0)
        assert post[2] == pytest.approx(nxt[2] * m.O[2, 2] / z)
        post_i, z_i = m.posterior(nxt, 2, inspected=True)
        assert z_i == pytest.approx(float(nxt @ m.O_inspect[:, 2]))
        assert post_i[2] > post[2]  # the sharper reading believes a level-2 symbol more

    def test_the_symbol_probabilities_sum_to_one(self):
        m = small_tabular()
        b = np.array([0.1, 0.4, 0.4, 0.1])
        for inspected in (False, True):
            total = sum(m.posterior(b, y, inspected)[1] for y in range(m.n_symbols))
            assert total == pytest.approx(1.0)


@pytest.fixture(scope="module")
def fitted():
    from closed_loop.planning.fitting import fit_world_model

    return fit_world_model(range(1000, 1030))


class TestFitting:
    def test_the_fit_builds_a_model_for_every_class(self, fitted):
        for cls in ("CRUSHER", "BELT_CONVEYOR", "SLURRY_PUMP"):
            m = PlantWorldModel(fitted, asset_class=cls)
            assert len(m.modes) >= 1
        assert PlantWorldModel(fitted, asset_class="BELT_CONVEYOR").modes == ("BEARING_DEFECT",)

    def test_the_fit_recovers_the_observation_relation(self, fitted):
        assert fitted["hi_base"] == pytest.approx(1.0, abs=0.1)
        assert fitted["hi_fail"] == pytest.approx(9.0, abs=0.2)
        assert fitted["hi_noise_sd"] == pytest.approx(0.3, abs=0.05)
        assert fitted["hi_load_coef"] == pytest.approx(2.0, abs=0.3)
        assert fitted["beta_load"] == pytest.approx(0.8, abs=0.1)
        assert fitted["first_onset_min"] >= 24.0 - 1e-9
        assert 0.4 < fitted["first_p_in"] < 0.8

    def test_ranges_sit_inside_the_catalogue_not_on_it(self, fitted):
        # sample minima and maxima of a finite set of lives never reach beyond the catalogue
        assert (
            20.0
            <= fitted["SLURRY_PUMP.BEARING_DEFECT.dur_lo"]
            < fitted["SLURRY_PUMP.BEARING_DEFECT.dur_hi"]
            <= 45.0
        )
        assert 1.2 <= fitted["CRUSHER.LINER_WEAR.k_lo"] and fitted["CRUSHER.LINER_WEAR.k_hi"] <= 3.0
        weights = [fitted[f"CRUSHER.{m}.weight"] for m in ("LINER_WEAR", "BEARING_DEFECT")]
        assert sum(weights) == pytest.approx(1.0)

    def test_only_development_seeds_may_be_used(self):
        from closed_loop.planning.fitting import fit_world_model

        for bad in ([5], [2000], [999], [1000, 3500]):
            with pytest.raises(ValueError, match="development"):
                fit_world_model(bad)
        with pytest.raises(ValueError, match="at least one"):
            fit_world_model([])
