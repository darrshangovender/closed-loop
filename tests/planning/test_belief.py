"""P8a: the belief tracker. Weights, resampling, blind window, determinism, exactness."""

import numpy as np
import pytest

from closed_loop.core.rng import rng_for
from closed_loop.planning.belief import BeliefTracker, systematic_resample
from closed_loop.planning.worldmodel import (
    ACTION_CONTINUE,
    PlantWorldModel,
    TabularWorldModel,
)

from ._helpers import COST, FITTED, plant_model, small_tabular


def readings(model: PlantWorldModel, seed: int, days: int = 90, load: float = 1.0):
    """A synthetic health-index series drawn from the model itself (the truth is one particle)."""
    rng = np.random.default_rng(seed)
    cur = model.prior(rng, 1)
    out = []
    for d in range(days):
        hi = float(model.hi_mean(cur, load)[0] + 0.3 * rng.standard_normal())
        out.append((d, hi, cur))
        cur = model.step(cur, ACTION_CONTINUE, rng, load=load)
    return out


def failing_series(model: PlantWorldModel, lo: int, hi: int, days: int = 100):
    """A series from the model whose truth fails between day ``lo`` and ``hi``, and that day."""
    for seed in range(100, 400):
        series = readings(model, seed, days)
        fail = next((d for d, _, st in series if model.failed(st)[0]), None)
        if fail is not None and lo <= fail <= hi:
            return series, fail
    raise AssertionError("no series found")


def run(
    tracker: BeliefTracker, asset: str, series, *, load: float | None = 1.0, stop: int | None = None
):
    for d, hi, _ in series[:stop]:
        tracker.update(asset, d, hi, False, load=load)


class TestSystematicResample:
    def test_counts_follow_the_weights(self):
        w = np.array([0.5, 0.25, 0.125, 0.125])
        idx = systematic_resample(w, np.random.default_rng(0))
        counts = np.bincount(idx, minlength=4)
        assert list(counts) == [2, 1, 0, 1] or list(counts) == [2, 1, 1, 0]

    def test_systematic_counts_never_stray_more_than_one_from_n_times_w(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            w = rng.random(40)
            w /= w.sum()
            counts = np.bincount(systematic_resample(w, rng), minlength=40)
            assert np.all(np.abs(counts - 40 * w) < 1.0 + 1e-9)

    def test_a_point_mass_is_copied_everywhere(self):
        w = np.zeros(6)
        w[4] = 1.0
        assert set(systematic_resample(w, np.random.default_rng(0))) == {4}

    def test_indices_stay_in_range_even_with_rounding_at_the_top(self):
        w = np.full(7, 1 / 7)
        for s in range(20):
            assert systematic_resample(w, np.random.default_rng(s)).max() <= 6


class TestWeights:
    def test_weights_are_normalised_after_every_update(self):
        m = plant_model()
        tr = BeliefTracker(m, seed=1)
        for d, hi, _ in readings(m, 1):
            tr.update("A", d, hi, False, load=1.0)
            w = tr.particles("A").weights
            assert w.sum() == pytest.approx(1.0) and np.all(w >= 0.0)

    def test_the_default_is_128_particles(self):
        tr = BeliefTracker(plant_model(), seed=0)
        assert tr.particles("A").n == 128
        assert BeliefTracker(plant_model(), n_particles=32).particles("A").n == 32

    def test_a_reading_moves_weight_toward_the_particles_that_explain_it(self):
        m = plant_model()
        tr = BeliefTracker(
            m, seed=2, ess_fraction=1e-9
        )  # never resample, so weights tell the story
        tr.update("A", 60, 4.0, False, load=1.0)  # well above baseline: degradation is under way
        p = tr.particles("A")
        mean_hi = m.hi_mean(p)
        near = np.abs(mean_hi - 4.0) < 1.0
        assert near.any() and p.weights[near].sum() > 0.8

    def test_the_effective_sample_size_drops_on_an_informative_reading(self):
        m = plant_model()
        tr = BeliefTracker(m, seed=2, ess_fraction=1e-9)
        before = tr.particles("A").ess()
        tr.update("A", 60, 4.0, False, load=1.0)
        assert tr.ess("A") < before


class TestResamplingTrigger:
    def test_resampling_happens_when_the_effective_sample_size_is_low(self):
        m = plant_model()
        tr = BeliefTracker(m, seed=3)
        tr.update("A", 60, 4.0, False, load=1.0)
        assert tr.n_resamples("A") == 1
        assert np.allclose(tr.particles("A").weights, 1 / 128)  # uniform again after resampling
        assert tr.ess("A") == pytest.approx(128.0)

    def test_it_does_not_resample_when_the_reading_is_uninformative(self):
        flat = TabularWorldModel(
            small_tabular().T, np.full((4, 3), 1 / 3), COST, initial=np.array([0.4, 0.3, 0.2, 0.1])
        )
        tr = BeliefTracker(flat, seed=0, n_particles=64)
        for d in range(5):
            tr.update("A", d, float(d % 3), False)
        assert tr.n_resamples("A") == 0

    def test_the_trigger_level_is_the_fraction_given(self):
        m = plant_model()
        never = BeliefTracker(m, seed=3, ess_fraction=1e-9)
        always = BeliefTracker(m, seed=3, ess_fraction=1.0)
        for tr in (never, always):
            tr.update("A", 60, 4.0, False, load=1.0)
        assert never.n_resamples("A") == 0 and always.n_resamples("A") == 1

    def test_resampling_keeps_the_particle_count_and_the_modes_it_had(self):
        m = plant_model()
        tr = BeliefTracker(m, seed=4)
        before = set(np.unique(tr.particles("A").state["mode"]))
        tr.update("A", 60, 4.0, False, load=1.0)
        after = set(np.unique(tr.particles("A").state["mode"]))
        assert tr.particles("A").n == 128 and after <= before


class TestMissingReadings:
    def test_a_masked_day_propagates_and_does_not_reweight(self):
        m = plant_model()
        a = BeliefTracker(m, seed=5)
        series = readings(m, 5, 40)
        run(a, "A", series, stop=30)
        w_before = a.particles("A").weights.copy()
        x_before = a.particles("A").state["x"].copy()
        a.update("A", 30, None, False, load=1.0)
        p = a.particles("A")
        assert np.array_equal(p.weights, w_before)  # no reweighting, no resampling
        assert not np.array_equal(p.state["x"], x_before)  # the day still passed
        assert a.day("A") == 30

    def test_a_gap_is_bridged_by_stepping_every_missing_day(self):
        m = plant_model()
        a, b = BeliefTracker(m, seed=6), BeliefTracker(m, seed=6)
        series = readings(m, 6, 60)
        run(a, "A", series, stop=30)
        run(b, "A", series, stop=30)
        for d in range(30, 35):  # b sees every day masked
            b.update("A", d, None, False, load=1.0)
        a.update("A", 34, None, False, load=1.0)  # a jumps in one call
        assert a.day("A") == b.day("A") == 34
        assert np.allclose(a.particles("A").state["x"], b.particles("A").state["x"], atol=0.05)

    def test_a_missing_reading_does_not_count_as_a_degenerate_event(self):
        m = plant_model()
        tr = BeliefTracker(m, seed=1)
        for d in range(25, 40):
            tr.update("A", d, None, False)
        assert tr.degenerate_events == 0 and tr.n_resamples("A") == 0


class TestBlindWindow:
    def test_blind_days_leave_the_weights_uniform_whatever_is_read(self):
        m = plant_model()
        tr = BeliefTracker(m, seed=7)
        for d in range(23):
            tr.update("A", d, 8.0 if d % 2 else 0.0, False, load=1.0)  # wild readings
        assert np.allclose(tr.particles("A").weights, 1 / 128)
        assert tr.n_resamples("A") == 0

    def test_the_belief_at_the_end_of_the_window_is_the_prior_carried_by_the_dynamics(self):
        m = plant_model()
        tr = BeliefTracker(m, seed=7)
        for d in range(23):
            tr.update("A", d, 8.0, False, load=1.0)
        # replay the same keyed generators by hand: the first reading (day 0) carried a load of 1,
        # so every later step ran under it
        p = m.prior(rng_for(7, "belief_prior", "A", 0), 128)
        for d in range(1, 23):
            p = m.step(p, ACTION_CONTINUE, rng_for(7, "belief", "A", 0, d, 0), load=1.0)
        assert np.array_equal(tr.particles("A").state["x"], p.state["x"])

    def test_the_first_reading_after_the_window_is_used_and_the_last_inside_it_is_not(self):
        # a longer window than the plant's 23 days, so that the particles already differ at its edge
        m = plant_model()
        tr = BeliefTracker(m, seed=7, blind_days=60.0, ess_fraction=1e-9)
        tr.update("A", 58, 4.0, False, load=1.0)
        tr.update("A", 59, 4.0, False, load=1.0)  # day 59 is still blind (< 60)
        assert np.allclose(tr.particles("A").weights, 1 / 128)
        tr.update("A", 60, 4.0, False, load=1.0)  # day 60 is not
        assert not np.allclose(tr.particles("A").weights, 1 / 128)

    def test_an_inspection_reading_is_used_inside_the_window(self):
        m = plant_model()
        tr = BeliefTracker(m, seed=7, blind_days=100.0, ess_fraction=1e-9)
        tr.update("A", 70, 4.0, False, load=1.0)
        assert np.allclose(tr.particles("A").weights, 1 / 128)  # a daily reading is ignored
        tr.update("A", 70, 4.0, True, load=1.0)
        assert not np.allclose(tr.particles("A").weights, 1 / 128)  # an inspection is not

    def test_a_new_life_restarts_the_window(self):
        m = plant_model()
        tr = BeliefTracker(m, seed=7, blind_days=60.0, ess_fraction=1e-9)
        for d in range(70, 75):  # life 0 is past its window: readings count
            tr.update("A", d, 4.0, False, load=1.0)
        assert not np.allclose(tr.particles("A").weights, 1 / 128)
        tr.reset_life("A", 75)
        assert tr.life("A") == 1 and tr.day("A") == 75
        assert np.allclose(tr.particles("A").weights, 1 / 128)
        for d in range(76, 135):
            tr.update("A", d, 4.0, False, load=1.0)
        assert np.allclose(tr.particles("A").weights, 1 / 128)  # blind until day 135
        tr.update("A", 135, 4.0, False, load=1.0)
        assert not np.allclose(tr.particles("A").weights, 1 / 128)

    def test_a_new_life_draws_a_renewal_not_a_first_component(self):
        m = plant_model()
        tr = BeliefTracker(m, seed=7)
        tr.update("A", 0, 1.0, False)
        tr.reset_life("A", 50)
        onset = -tr.particles("A").state["x"] * tr.particles("A").state["dur"]
        assert onset.min() >= 24.0 - 1e-9 and onset.max() <= 60.0 + 1e-9

    def test_the_length_of_the_window_is_a_parameter(self):
        tr = BeliefTracker(plant_model(), seed=0, blind_days=0.0, ess_fraction=1e-9)
        tr.update("A", 60, 4.0, False, load=1.0)
        assert not np.allclose(tr.particles("A").weights, 1 / 128)


class TestLoadAlignment:
    """Found at 23:41: a reading is taken at the start of its day, so the clock between two readings
    ran under the load seen at the *previous* reading."""

    def test_the_step_into_a_day_uses_the_previous_days_load(self):
        m = plant_model()
        tr = BeliefTracker(m, seed=11, blind_days=1000.0)  # propagation only: no reweighting
        loads = [1.0, 1.4, 0.6, 1.2]
        for d, load in enumerate(loads):
            tr.update("A", d, 2.0, False, load=load)
        p = m.prior(rng_for(11, "belief_prior", "A", 0), 128)
        for d in range(1, len(loads)):
            p = m.step(p, ACTION_CONTINUE, rng_for(11, "belief", "A", 0, d, 0), load=loads[d - 1])
        assert np.array_equal(tr.particles("A").state["x"], p.state["x"])

    def test_the_first_step_after_a_gap_uses_the_last_load_seen(self):
        m = plant_model()
        tr = BeliefTracker(m, seed=12, blind_days=1000.0)
        tr.update("A", 0, 2.0, False, load=1.3)
        tr.update("A", 3, 2.0, False, load=0.5)
        p = m.prior(rng_for(12, "belief_prior", "A", 0), 128)
        for d in (1, 2, 3):
            p = m.step(p, ACTION_CONTINUE, rng_for(12, "belief", "A", 0, d, 0), load=1.3)
        assert np.array_equal(tr.particles("A").state["x"], p.state["x"])

    def test_a_second_reading_the_same_day_does_not_change_the_load_for_the_next_step(self):
        m = plant_model()
        a = BeliefTracker(m, seed=13, blind_days=1000.0)
        b = BeliefTracker(m, seed=13, blind_days=1000.0)
        for t in (a, b):
            t.update("A", 0, 2.0, False, load=1.0)
            t.update("A", 1, 2.0, False, load=1.2)
        b.update("A", 1, 2.0, True, load=0.4)  # an inspection with a different load reading
        a.update("A", 2, 2.0, False, load=1.0)
        b.update("A", 2, 2.0, False, load=1.0)
        assert np.array_equal(a.particles("A").state["x"], b.particles("A").state["x"])


class TestInspection:
    def test_an_inspection_reading_is_weighted_more_sharply(self):
        m = plant_model()
        daily = BeliefTracker(m, seed=8, ess_fraction=1e-9, blind_days=0.0)
        inspect = BeliefTracker(m, seed=8, ess_fraction=1e-9, blind_days=0.0)
        daily.update("A", 30, 4.0, False, load=1.0)
        inspect.update("A", 30, 4.0, True, load=1.0)
        assert inspect.ess("A") < daily.ess("A")

    def test_a_second_reading_on_the_same_day_is_folded_in(self):
        m = plant_model()
        tr = BeliefTracker(m, seed=8, ess_fraction=1e-9)
        tr.update("A", 30, 4.0, False, load=1.0)
        x_after_daily = tr.particles("A").state["x"].copy()
        w1 = tr.particles("A").weights.copy()
        tr.update("A", 30, 4.1, True, load=1.0)
        assert np.array_equal(tr.particles("A").state["x"], x_after_daily)  # no extra day passed
        assert not np.array_equal(tr.particles("A").weights, w1)
        assert tr.day("A") == 30

    def test_the_noise_scale_of_an_inspection_is_set_at_construction(self):
        m = plant_model()
        a = BeliefTracker(m, seed=8, ess_fraction=1e-9, blind_days=0.0, inspect_noise_scale=0.25)
        b = BeliefTracker(m, seed=8, ess_fraction=1e-9, blind_days=0.0, inspect_noise_scale=1.0)
        a.update("A", 30, 4.0, True, load=1.0)
        b.update("A", 30, 4.0, False, load=1.0)
        c = BeliefTracker(m, seed=8, ess_fraction=1e-9, blind_days=0.0, inspect_noise_scale=1.0)
        c.update("A", 30, 4.0, True, load=1.0)
        assert np.allclose(c.particles("A").weights, b.particles("A").weights)
        assert not np.allclose(a.particles("A").weights, b.particles("A").weights)


class TestDays:
    def test_going_back_in_time_is_refused(self):
        tr = BeliefTracker(plant_model(), seed=0)
        tr.update("A", 30, 2.0, False)
        with pytest.raises(ValueError, match="before"):
            tr.update("A", 29, 2.0, False)

    def test_a_first_update_after_day_zero_carries_the_prior_forward(self):
        m = plant_model()
        tr = BeliefTracker(m, seed=0, ess_fraction=1e-9)
        tr.update("A", 10, None, False)
        x0 = m.prior(rng_for(0, "belief_prior", "A", 0), 128).state["x"]
        assert np.all(tr.particles("A").state["x"] > x0)


class TestDeterminism:
    def test_the_same_seed_and_readings_give_the_same_particles(self):
        m = plant_model()
        series = readings(m, 11)
        a, b = BeliefTracker(m, seed=3), BeliefTracker(m, seed=3)
        run(a, "A", series)
        run(b, "A", series)
        for k, v in a.particles("A").state.items():
            assert np.array_equal(v, b.particles("A").state[k])
        assert np.array_equal(a.particles("A").weights, b.particles("A").weights)

    def test_the_seed_matters(self):
        m = plant_model()
        series = readings(m, 11)
        a, b = BeliefTracker(m, seed=3), BeliefTracker(m, seed=4)
        run(a, "A", series, stop=40)
        run(b, "A", series, stop=40)
        assert not np.array_equal(a.particles("A").state["x"], b.particles("A").state["x"])

    def test_one_asset_does_not_disturb_another(self):
        m = plant_model()
        s1, s2 = readings(m, 21), readings(m, 22)
        alone = BeliefTracker(m, seed=9)
        run(alone, "A", s1)
        both = BeliefTracker(m, seed=9)
        for (d, h1, _), (_, h2, _) in zip(s1, s2, strict=True):  # interleaved updates
            both.update("B", d, h2, False, load=1.0)
            both.update("A", d, h1, False, load=1.0)
        assert np.array_equal(alone.particles("A").state["x"], both.particles("A").state["x"])

    def test_two_assets_get_different_draws(self):
        tr = BeliefTracker(plant_model(), seed=9)
        assert not np.array_equal(tr.particles("A").state["x"], tr.particles("B").state["x"])

    def test_a_repeated_run_in_a_fresh_process_gives_the_same_digest(self):
        import subprocess
        import sys
        import textwrap

        code = textwrap.dedent(
            """
            import hashlib, sys
            sys.path.insert(0, ".")
            from tests.planning._helpers import plant_model
            from tests.planning.test_belief import readings
            from closed_loop.planning.belief import BeliefTracker
            m = plant_model()
            tr = BeliefTracker(m, seed=5)
            for d, hi, _ in readings(m, 1, 60):
                tr.update("A", d, hi, False, load=1.0)
            p = tr.particles("A")
            print(hashlib.sha256(p.state["x"].tobytes() + p.weights.tobytes()).hexdigest())
            """
        )
        out = {
            subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True, check=True
            ).stdout.strip()
            for _ in range(2)
        }
        assert len(out) == 1 and len(next(iter(out))) == 64


class TestModelSources:
    def test_a_mapping_gives_each_asset_its_own_model(self):
        crusher = PlantWorldModel(
            {
                **{k: v for k, v in FITTED.items() if not k.startswith("SLURRY_PUMP")},
                "CRUSHER.LINER_WEAR.weight": 1.0,
                "CRUSHER.LINER_WEAR.dur_lo": 45.0,
                "CRUSHER.LINER_WEAR.dur_hi": 85.0,
                "CRUSHER.LINER_WEAR.k_lo": 1.2,
                "CRUSHER.LINER_WEAR.k_hi": 3.0,
            },
            asset_class="CRUSHER",
        )
        tr = BeliefTracker({"CR": crusher, "PU": plant_model()}, seed=0)
        assert tr.model_for("CR") is crusher
        assert tr.particles("CR").state["mode"].max() == 0
        assert tr.particles("PU").state["mode"].max() == 1

    def test_a_callable_is_asked_once_per_asset(self):
        calls = []

        def factory(asset: str):
            calls.append(asset)
            return plant_model()

        tr = BeliefTracker(factory, seed=0)
        for _ in range(3):
            tr.update("A", 0, 1.0, False, load=1.0)
        tr.update("B", 0, 1.0, False, load=1.0)
        assert calls == ["A", "B"]

    def test_a_missing_asset_in_a_mapping_is_an_error(self):
        tr = BeliefTracker({"A": plant_model()}, seed=0)
        with pytest.raises(KeyError):
            tr.update("Z", 0, 1.0, False)

    def test_a_model_without_a_resample_method_still_works(self):
        class Plain:
            def __init__(self):
                self.inner = small_tabular()

            def prior(self, rng, n):
                return self.inner.prior(rng, n)

            def step(self, p, action, rng):
                return self.inner.step(p, action, rng)

            def loglik(self, p, obs, noise_scale=1.0):
                return self.inner.loglik(p, obs, noise_scale)

            def failed(self, p):
                return self.inner.failed(p)

        tr = BeliefTracker(Plain(), seed=0, n_particles=64)
        for d, y in enumerate([0, 1, 1, 2, 2, 2]):
            tr.update("A", d, float(y), False)
        assert tr.particles("A").n == 64

    def test_invalid_settings_are_refused(self):
        m = plant_model()
        for kwargs in (
            {"n_particles": 1},
            {"ess_fraction": 0.0},
            {"ess_fraction": 1.5},
            {"blind_days": -1.0},
            {"inspect_noise_scale": 0.0},
            {"load_ema_alpha": 0.0},
        ):
            with pytest.raises(ValueError):
                BeliefTracker(m, **kwargs)


class TestAgainstTheExactFilter:
    """The particle filter must converge to Bayes' rule where Bayes' rule can be computed."""

    SEQUENCE = [(0, False), (2, False), (2, True), (1, False), (2, False), (2, True), (2, False)]

    def exact(self, model: TabularWorldModel):
        b = model.initial.copy()
        out = [b]
        for i, (y, inspected) in enumerate(self.SEQUENCE):
            if i > 0:
                b = model.predict(b, ACTION_CONTINUE)
            b, _ = model.posterior(b, y, inspected)
            out.append(b)
        return out

    def test_the_weighted_state_frequencies_match_the_exact_belief(self):
        model = small_tabular()
        exact = self.exact(model)
        tr = BeliefTracker(model, n_particles=40000, seed=1, blind_days=0.0)
        for i, (y, inspected) in enumerate(self.SEQUENCE):
            tr.update("A", i, float(y), inspected)
            p = tr.particles("A")
            freq = np.bincount(p.state["s"], weights=p.weights, minlength=4)
            assert np.abs(freq - exact[i + 1]).sum() / 2 < 0.03, (i, freq, exact[i + 1])

    def test_failure_probability_matches_the_exact_belief(self):
        model = small_tabular()
        exact = self.exact(model)
        tr = BeliefTracker(model, n_particles=40000, seed=2, blind_days=0.0)
        for i, (y, inspected) in enumerate(self.SEQUENCE):
            tr.update("A", i, float(y), inspected)
        assert tr.p_failed("A") == pytest.approx(exact[-1][3], abs=0.01)

    def test_the_inspection_matrix_is_what_the_tracker_uses_for_an_inspection(self):
        model = small_tabular()
        exact_ins, _ = model.posterior(model.initial, 2, inspected=True)
        exact_day, _ = model.posterior(model.initial, 2, inspected=False)
        a = BeliefTracker(model, n_particles=40000, seed=3, blind_days=0.0)
        a.update("A", 0, 2.0, True)
        b = BeliefTracker(model, n_particles=40000, seed=3, blind_days=0.0)
        b.update("A", 0, 2.0, False)
        pa = np.bincount(a.particles("A").state["s"], weights=a.particles("A").weights, minlength=4)
        pb = np.bincount(b.particles("A").state["s"], weights=b.particles("A").weights, minlength=4)
        assert np.abs(pa - exact_ins).sum() / 2 < 0.02
        assert np.abs(pb - exact_day).sum() / 2 < 0.02


class TestDegeneracy:
    def test_an_impossible_reading_redraws_from_the_prior_and_is_counted(self):
        sharp = TabularWorldModel(
            small_tabular().T,
            np.array([[1.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]),
            COST,
            initial=np.array([0.5, 0.5, 0.0, 0.0]),
        )
        tr = BeliefTracker(sharp, n_particles=64, seed=0, blind_days=0.0)
        tr.update("A", 0, 2.0, False)  # symbol 2 only comes from level 3, which nobody is in
        assert tr.degenerate_events == 1
        p = tr.particles("A")
        assert p.weights.sum() == pytest.approx(1.0)

    def test_a_reading_nobody_can_produce_after_a_redraw_leaves_uniform_weights(self):
        dead = TabularWorldModel(
            small_tabular().T,
            np.array([[1.0, 0, 0], [1.0, 0, 0], [1.0, 0, 0], [1.0, 0, 0]]),
            COST,
        )
        tr = BeliefTracker(dead, n_particles=16, seed=0, blind_days=0.0)
        tr.update("A", 0, 2.0, False)
        assert tr.degenerate_events == 1
        assert np.allclose(tr.particles("A").weights, 1 / 16)

    def test_a_silent_replacement_is_recovered(self):
        m = plant_model()
        tr = BeliefTracker(m, seed=12)
        # an asset that looks nearly dead, then reads like a new one
        for d in range(23, 50):
            tr.update("A", d, 8.5, False, load=1.0)
        tr.update("A", 50, 1.0, False, load=1.0)
        assert tr.p_failed("A") < 0.5


class TestQueries:
    def test_a_degrading_asset_shortens_the_remaining_life(self):
        m = plant_model()
        series, fail = failing_series(m, 70, 88)
        tr = BeliefTracker(m, seed=14)
        run(tr, "A", series, stop=fail - 25)
        early = tr.remaining_life_quantiles("A")
        run(tr, "A", series[fail - 25 : fail - 5], stop=None)
        late = tr.remaining_life_quantiles("A")
        assert early.shape == (3,) and np.all(np.diff(early) >= 0)
        assert late[1] < early[1]
        assert late[1] < 25.0  # five days before the failure the belief has come near

    def test_the_load_forecast_is_a_moving_average_of_what_was_seen(self):
        tr = BeliefTracker(plant_model(), seed=0, load_ema_alpha=0.5)
        assert tr.load_forecast("A") is None
        tr.update("A", 0, 1.0, False, load=1.0)
        tr.update("A", 1, 1.0, False, load=1.2)
        assert tr.load_forecast("A") == pytest.approx(1.1)

    def test_the_rollout_quantiles_are_deterministic_and_ordered(self):
        m = plant_model()
        series = readings(m, 15, 50)
        tr = BeliefTracker(m, seed=15)
        run(tr, "A", series)
        a = tr.failure_day_quantiles("A")
        b = tr.failure_day_quantiles("A")
        assert np.array_equal(a, b) and np.all(np.diff(a) >= 0)

    def test_a_model_without_remaining_life_says_so(self):
        tr = BeliefTracker(small_tabular(), n_particles=16, seed=0)
        with pytest.raises(NotImplementedError):
            tr.remaining_life_days("A")


class TestCalibrationAgainstItsOwnModel:
    """A regression guard on the filter, not the pre-registered figure.

    The truth is drawn from the model, so a correct filter's 90% interval should cover about 90%.
    With 128 particles it does not (it is overconfident: see the progress note), with enough it
    does. The recorded plant coverage at N = 128 is a separate measurement.
    """

    def coverage(self, n_particles: int, trials: int) -> tuple[int, int]:
        m = plant_model()
        inside = total = 0
        for t in range(trials):
            series = readings(m, 3000 + t, 90)
            fail_day = next((d for d, _, st in series if m.failed(st)[0]), None)
            if fail_day is None:
                continue
            tr = BeliefTracker(m, seed=t, n_particles=n_particles)
            for d, hi, _ in series:
                tr.update("A", d, hi, False, load=1.0)
                rul = fail_day - d
                if d >= 23 and rul > 0 and hi >= 2.0 and d % 4 == 0:
                    lo, _, up = tr.failure_day_quantiles("A")
                    total += 1
                    inside += int(lo <= rul <= up)
        return inside, total

    def test_with_many_particles_the_interval_is_about_calibrated(self):
        inside, total = self.coverage(1024, 14)
        assert total >= 40
        assert inside / total >= 0.78

    def test_more_particles_help(self):
        low = self.coverage(32, 14)
        high = self.coverage(1024, 14)
        assert high[0] / high[1] > low[0] / low[1]
