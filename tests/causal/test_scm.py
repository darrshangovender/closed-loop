"""P7: the structural causal model. Abduction, action, prediction, against the plant's own twin.

Most checks run on the quiet configuration (no noise, no data-quality events, no direct load term in
the health index), where the model's equations are exactly the plant's and the answers can be
exact up to the plant's 15-minute grid.
"""

import dataclasses

import numpy as np
import pytest

from closed_loop.causal.records import LifeRecord
from closed_loop.causal.scm import DAY_Q, SCMEffectModel
from closed_loop.causal.twin import Query, twin_truths
from closed_loop.plant import Scenario
from closed_loop.plant.observe import simulate_lives

GRID = 3.0 / DAY_Q  # three quarters: the beta estimate is good to a few parts in a thousand


def true_life(rec: LifeRecord, params):
    return simulate_lives(params, rec.seed, Scenario()).lives[rec.asset_id]


class TestFitting:
    def test_the_load_sensitivity_is_recovered(self, quiet_scm):
        assert quiet_scm.beta == pytest.approx(0.8, abs=0.01)
        assert quiet_scm.beta_se < 0.02

    def test_the_derate_factor_is_read_off_the_observed_load(self, quiet_scm):
        assert quiet_scm.derate_factor == pytest.approx(0.7, abs=1e-6)
        assert "derate steps" in quiet_scm.derate_factor_source

    def test_the_baseline_is_estimated_not_copied(self, quiet_scm, noisy_lives):
        m = SCMEffectModel(n_beta_lives=4)
        m.fit(noisy_lives[:24])
        assert m.h0 != 1.0 and m.h0 == pytest.approx(1.0, abs=0.1)
        assert m.c_h != 2.0 and m.c_h == pytest.approx(2.0, abs=0.5)
        assert m.sigma_h == pytest.approx(0.3, abs=0.1)

    def test_the_failure_level_is_the_handed_in_threshold(self):
        assert SCMEffectModel().hi_fail == 9.0
        assert SCMEffectModel(hi_fail=8.0).hi_fail == 8.0

    def test_without_a_derated_life_the_default_is_used_and_says_so(self, quiet_params):
        from closed_loop.causal.twin import generate_observational

        passive = generate_observational(8, 3100, "none", params=quiet_params)
        assert all(lf.derate_start is None for lf in passive)
        m = SCMEffectModel(n_beta_lives=3)
        m._fit_derate_factor(passive)
        assert m.derate_factor == 0.7 and "default" in m.derate_factor_source

    def test_a_model_must_be_fitted_before_it_answers(self, quiet_lives):
        m = SCMEffectModel()
        with pytest.raises(RuntimeError, match="fit"):
            m.counterfactual_failure_day(quiet_lives[0], 30.0, 0.7)
        with pytest.raises(RuntimeError, match="fit"):
            m.p_fail_before(quiet_lives[0], None, 100.0)

    def test_fit_needs_lives(self):
        with pytest.raises(ValueError, match="at least one"):
            SCMEffectModel().fit([])

    def test_a_record_without_daily_arrays_cannot_be_used(self, quiet_scm, quiet_lives):
        lf = quiet_lives[0]
        bare = LifeRecord(
            lf.asset_id, lf.hi_days, lf.hi, lf.derate_days, lf.fail_day, lf.censored_day
        )
        with pytest.raises(ValueError, match="daily"):
            quiet_scm.abduct(bare)

    def test_readings_must_be_consecutive(self, quiet_scm, quiet_lives):
        lf = quiet_lives[0]
        keep = np.arange(lf.days.size) % 2 == 0
        gappy = dataclasses.replace(
            lf,
            days=lf.days[keep],
            sig=lf.sig[:, keep],
            load=lf.load[keep],
            hi_days=lf.hi_days[lf.hi_days % 2 == 0],
            hi=lf.hi[lf.hi_days % 2 == 0],
        )
        with pytest.raises(ValueError, match="consecutive"):
            quiet_scm.abduct(gappy)


class TestRobustness:
    """Regressions found by probing the noisy plant."""

    def test_the_baseline_fit_survives_residuals_that_are_numerically_zero(self, quiet_lives):
        """Found at 00:55: a health index that is exactly linear in the load made the trimmed refit
        throw every point away (the residual spread was 1e-16, so 3 sd kept nothing) and the fit
        came back as zeros and NaN."""
        rng = np.random.default_rng(0)
        lives = []
        for lf in quiet_lives[:8]:
            load = lf.load + 0.05 * rng.standard_normal(lf.load.size)
            hi_all = np.interp(lf.hi_days, lf.days, 1.0 + 2.0 * (load - 1.0))
            lives.append(
                dataclasses.replace(lf, load=load, hi=np.where(lf.hi_days < 20, hi_all, lf.hi))
            )
        m = SCMEffectModel(n_beta_lives=3)
        m._fit_observation(lives)
        assert np.isfinite([m.h0, m.c_h, m.sigma_h]).all()
        assert m.h0 == pytest.approx(1.0, abs=1e-6) and m.c_h == pytest.approx(2.0, abs=1e-6)

    def test_the_baseline_fit_with_only_a_noisy_load_reading_is_finite(self):
        """The configuration of the 00:55 probe: a health index that is exactly 1.0 before any wear
        and a load that is 1.0 plus reading noise. Two guards (a spread floor and a minimum number
        of kept points) each keep the trimmed refit from throwing every point away."""
        from closed_loop.causal.twin import causal_params, generate_observational

        p = causal_params(
            hi_noise_sd=0.0,
            sig_noise_sd=0.0,
            load_noise_sd=0.0,
            load_level_sd=0.0,
            load_obs_sd=0.01,
            upset_p_per_circuit=0.0,
            p_gain_fault=0.0,
            p_stuck_day=0.0,
            p_spike_day=0.0,
            hi_load_coef=0.0,
        )
        lives = generate_observational(8, 3000, "pi_obs", params=p)
        m = SCMEffectModel(n_beta_lives=3)
        m._fit_observation(lives)
        assert m.h0 == pytest.approx(1.0, abs=1e-9) and abs(m.c_h) < 1e-9
        assert np.isfinite(m.sigma_h)

    def test_the_derate_factor_is_learned_from_the_data_not_the_default(self, quiet_params):
        """A plant whose derate keeps 60% of the load, not the default 70%."""
        from closed_loop.causal.twin import causal_params, generate_observational

        p = dataclasses.replace(quiet_params, derate_fraction=0.4)
        lives = generate_observational(12, 3000, "pi_obs", params=causal_params(p))
        m = SCMEffectModel(n_beta_lives=3)
        m._fit_derate_factor(lives)
        assert m.derate_factor == pytest.approx(0.6, abs=1e-6)
        assert "derate steps" in m.derate_factor_source

    def test_the_failure_clock_is_the_middle_of_the_failure_quarter(self, quiet_lives):
        """The plant fails at the first quarter whose start has the clock at the target, so the
        clock at that quarter's start overshoots the target by up to a quarter: half of it is
        taken back."""
        cum = np.arange(0.0, 201.0)  # a clock that runs at one day per day
        at_a_day = SCMEffectModel._clock_at(quiet_lives[0], cum, 50.0)
        five_quarters = SCMEffectModel._clock_at(quiet_lives[0], cum, 50.0 + 5.0 / DAY_Q)
        assert at_a_day == pytest.approx(50.0 - 0.5 / DAY_Q)
        assert five_quarters == pytest.approx(50.0 + 4.5 / DAY_Q)

    def test_lives_with_an_unmodelled_sensor_drift_cannot_drag_the_load_sensitivity(
        self, quiet_lives
    ):
        """A life whose health index drifts for reasons the structure does not contain would, in an
        untrimmed profile, bend the load sensitivity to fit it. Trimming the worst fifth of the
        lives leaves the estimate where the clean lives put it."""
        drifting = []
        for i, lf in enumerate(quiet_lives):
            if i % 4 == 0 and lf.failed:
                ramp = 0.08 * np.maximum(lf.hi_days - 25.0, 0.0)
                drifting.append(dataclasses.replace(lf, hi=lf.hi + ramp))
            else:
                drifting.append(lf)
        assert sum(a is not b for a, b in zip(drifting, quiet_lives, strict=True)) >= 4
        trimmed = SCMEffectModel(n_beta_lives=24, trim=0.75, beta_tol=0.01)
        raw = SCMEffectModel(n_beta_lives=24, trim=1.0, beta_tol=0.01)
        trimmed.fit(drifting)
        raw.fit(drifting)
        assert trimmed.beta == pytest.approx(0.8, abs=0.06)
        assert abs(raw.beta - 0.8) > abs(trimmed.beta - 0.8) + 0.05

    def test_the_trim_fraction_is_checked(self):
        with pytest.raises(ValueError, match="trim"):
            SCMEffectModel(trim=0.3)

    def test_the_default_trims_the_worst_fifth(self):
        assert SCMEffectModel().trim == 0.8


class TestAbduction:
    def test_the_exogenous_terms_of_a_noiseless_life_are_recovered(
        self, quiet_scm, quiet_lives, quiet_params
    ):
        checked = 0
        for lf in quiet_lives:
            if not lf.failed:
                continue
            truth = true_life(lf, quiet_params)
            ab = quiet_scm.abduct(lf)
            assert ab.tau_on == pytest.approx(truth.tau_onset, abs=0.5)
            assert ab.dur == pytest.approx(truth.duration, abs=0.6)
            assert ab.shape == pytest.approx(truth.shape, abs=0.05)
            assert ab.tau_fail == pytest.approx(truth.tau_fail, abs=0.1)
            assert ab.sse < 1e-3
            checked += 1
        assert checked >= 15

    def test_a_life_not_seen_to_fail_is_still_abduced_from_its_path(
        self, quiet_scm, quiet_lives, quiet_params
    ):
        """Cut a failed life short, well inside the degradation, and ask where it was headed."""
        lf = next(x for x in quiet_lives if x.failed and x.fail_day > 90)
        truth = true_life(lf, quiet_params)
        cut = int(lf.fail_day - 8)
        keep = lf.days <= cut
        short = dataclasses.replace(
            lf,
            days=lf.days[keep],
            sig=lf.sig[:, keep],
            load=lf.load[keep],
            hi_days=lf.hi_days[lf.hi_days <= cut],
            hi=lf.hi[lf.hi_days <= cut],
            derate_days=lf.derate_days[lf.derate_days <= cut],
            fail_day=None,
            censored_day=float(cut),
        )
        ab = quiet_scm.abduct(short)
        assert ab.tau_fail > cut * 0.6  # the failure is ahead of the record
        assert ab.tau_fail == pytest.approx(truth.tau_fail, rel=0.03)
        assert ab.shape == pytest.approx(truth.shape, abs=0.3)

    def test_abduction_is_cached_per_life_and_load_sensitivity(self, quiet_scm, quiet_lives):
        a = quiet_scm.abduct(quiet_lives[0])
        assert quiet_scm.abduct(quiet_lives[0]) is a
        assert quiet_scm.abduct(quiet_lives[0], quiet_scm.beta + 0.1) is not a

    def test_a_healthy_life_has_its_failure_beyond_the_record(self, quiet_scm, quiet_lives):
        """Cut a life before any wear shows: the failure must lie after the last reading."""
        lf = quiet_lives[0]
        cut = 18
        keep = lf.days <= cut
        young = dataclasses.replace(
            lf,
            days=lf.days[keep],
            sig=lf.sig[:, keep],
            load=lf.load[keep],
            hi_days=lf.hi_days[lf.hi_days <= cut],
            hi=lf.hi[lf.hi_days <= cut],
            derate_days=np.zeros(0),
            derate_start=None,
            fail_day=None,
            censored_day=float(cut),
        )
        ab = quiet_scm.abduct(young)
        assert ab.tau_fail > cut


class TestCounterfactualAgainstTheTwin:
    """The SCM's answer equals the plant's own, up to the plant's 15-minute grid."""

    def test_derate_from_the_first_alert(self, quiet_scm, quiet_lives, quiet_params):
        errs = []
        for seed in sorted({lf.seed for lf in quiet_lives})[:4]:
            lives = [lf for lf in quiet_lives if lf.seed == seed]
            picked = [(lf, lf.first_alert_day()) for lf in lives if lf.first_alert_day()]
            qs = [Query("derate", lf.asset_id, float(t)) for lf, t in picked]
            for (lf, t), ans in zip(
                picked, twin_truths(seed, qs, params=quiet_params), strict=True
            ):
                errs.append(
                    abs(quiet_scm.counterfactual_failure_day(lf, float(t), 0.7) - ans.failure_day)
                )
        assert len(errs) >= 12
        assert max(errs) < GRID * 2.0
        assert np.mean(errs) < GRID

    def test_the_world_where_nobody_acts(self, quiet_scm, quiet_lives, quiet_params):
        """Undo the operator's derate and the failure is earlier: the abduction has to see it."""
        errs, earlier = [], 0
        for seed in sorted({lf.seed for lf in quiet_lives})[:4]:
            lives = [lf for lf in quiet_lives if lf.seed == seed]
            qs = [Query("nothing", lf.asset_id) for lf in lives]
            for lf, ans in zip(lives, twin_truths(seed, qs, params=quiet_params), strict=True):
                got = quiet_scm.counterfactual_failure_day(lf, 0.0, 1.0)
                errs.append(abs(got - ans.failure_day))
                earlier += int(lf.derate_start is not None and got < lf.end_day - 0.5)
        assert max(errs) < GRID * 2.0
        assert earlier >= 5

    def test_a_derate_postpones_the_failure_by_the_amount_the_clock_says(
        self, quiet_scm, quiet_lives
    ):
        lf = next(x for x in quiet_lives if x.failed and x.first_alert_day())
        t0 = float(lf.first_alert_day())
        n = quiet_scm.counterfactual_failure_day(lf, t0, 1.0)
        d = quiet_scm.counterfactual_failure_day(lf, t0, 0.7)
        # the clock runs exp(0.8 * -0.3) = 0.787 as fast: the remaining time grows by 1 / 0.787
        assert (d - t0) / (n - t0) == pytest.approx(1.0 / np.exp(0.8 * -0.3), rel=0.01)

    def test_a_later_derate_postpones_it_less(self, quiet_scm, quiet_lives):
        lf = next(x for x in quiet_lives if x.failed)
        a = quiet_scm.counterfactual_failure_day(lf, 30.0, 0.7)
        b = quiet_scm.counterfactual_failure_day(lf, 60.0, 0.7)
        assert a > b > quiet_scm.counterfactual_failure_day(lf, 0.0, 1.0) - 1e-9

    def test_a_retained_load_of_one_ignores_the_derate_day(self, quiet_scm, quiet_lives):
        lf = quiet_lives[1]
        assert quiet_scm.counterfactual_failure_day(
            lf, 10.0, 1.0
        ) == quiet_scm.counterfactual_failure_day(lf, 90.0, 1.0)

    def test_a_stronger_derate_postpones_it_further(self, quiet_scm, quiet_lives):
        lf = next(x for x in quiet_lives if x.failed)
        assert quiet_scm.counterfactual_failure_day(
            lf, 30.0, 0.6
        ) > quiet_scm.counterfactual_failure_day(lf, 30.0, 0.8)

    def test_the_answer_is_on_the_plants_quarter_grid(self, quiet_scm, quiet_lives):
        for lf in quiet_lives[:6]:
            t = quiet_scm.counterfactual_failure_day(lf, 30.0, 0.7)
            assert t * DAY_Q == pytest.approx(round(t * DAY_Q), abs=1e-6)


class TestTheGrid:
    def test_the_crossing_is_rounded_up_to_the_next_quarter_as_the_plant_does(
        self, quiet_scm, quiet_lives, monkeypatch
    ):
        """With the load at its undisturbed value the clock is the calendar: a clock total of
        50.001 days is reached during the 97th quarter after day 50, so the failure is stamped at
        the start of the quarter after it, as the plant stamps its own (first quarter whose start
        has the clock at the target)."""
        from closed_loop.causal.scm import Abduction

        lf = quiet_lives[0]
        monkeypatch.setattr(
            SCMEffectModel,
            "abduct",
            lambda self, life, beta=None: Abduction(0.0, 50.001, 2.0, 50.001, 0.0, 0),
        )
        got = quiet_scm.counterfactual_failure_day(lf, 0.0, 1.0)
        assert got == pytest.approx(50.0 + 1.0 / DAY_Q)
        monkeypatch.setattr(
            SCMEffectModel,
            "abduct",
            lambda self, life, beta=None: Abduction(0.0, 50.0, 2.0, 50.0, 0.0, 0),
        )
        assert quiet_scm.counterfactual_failure_day(lf, 0.0, 1.0) == pytest.approx(50.0)


class TestAbductionIsWhatMakesItExact:
    def test_removing_the_abduction_step_breaks_the_exactness(
        self, quiet_scm, quiet_lives, quiet_params, monkeypatch
    ):
        """The mutation of the packet: without abduction the model can only guess the clock total
        from the population, and the noise-free test fails."""
        from closed_loop.causal.scm import Abduction

        guess = float(np.median(quiet_scm.fresh_clock))

        def no_abduction(self, life, beta=None):
            return Abduction(0.0, guess, 2.0, guess, 0.0, 0)

        seed = quiet_lives[0].seed
        lives = [lf for lf in quiet_lives if lf.seed == seed]
        picked = [(lf, lf.first_alert_day()) for lf in lives if lf.first_alert_day()]
        qs = [Query("derate", lf.asset_id, float(t)) for lf, t in picked]
        answers = twin_truths(seed, qs, params=quiet_params)
        monkeypatch.setattr(SCMEffectModel, "abduct", no_abduction)
        errs = [
            abs(quiet_scm.counterfactual_failure_day(lf, float(t), 0.7) - a.failure_day)
            for (lf, t), a in zip(picked, answers, strict=True)
        ]
        assert max(errs) > 5.0  # against under 0.05 with the abduction


@pytest.fixture(scope="module")
def model(quiet_lives):
    m = SCMEffectModel(misspecified=True, n_beta_lives=4)
    m.fit(quiet_lives)
    return m


class TestMisspecified:
    def test_it_has_no_load_sensitivity(self, model):
        assert model.name == "scm_misspecified" and model.beta == 0.0 and model.beta_se == 0.0

    def test_a_derate_cannot_postpone_anything(self, model, quiet_lives):
        for lf in quiet_lives[:8]:
            n = model.counterfactual_failure_day(lf, 30.0, 1.0)
            assert model.counterfactual_failure_day(lf, 30.0, 0.7) == n

    def test_a_seen_failure_stays_where_it_was(self, model, quiet_lives):
        lf = next(x for x in quiet_lives if x.failed)
        assert model.counterfactual_failure_day(lf, 30.0, 0.7) == pytest.approx(
            lf.fail_day, abs=0.02
        )


class TestFailureProbability:
    def test_it_is_a_probability_and_grows_with_the_horizon(self, quiet_scm, quiet_lives):
        lf = next(x for x in quiet_lives if x.failed)
        t = quiet_scm.counterfactual_failure_day(lf, 0.0, 1.0)
        ps = [quiet_scm.p_fail_before(lf, None, d) for d in (t - 30, t - 1, t + 1, t + 30)]
        assert all(0.0 <= p <= 1.0 for p in ps)
        assert ps == sorted(ps) and ps[0] < 0.01 and ps[-1] > 0.99

    def test_it_is_half_at_the_predicted_day(self, quiet_scm, quiet_lives):
        lf = next(x for x in quiet_lives if x.failed)
        t = quiet_scm.counterfactual_failure_day(lf, 0.0, 1.0)
        assert quiet_scm.p_fail_before(lf, None, t) == pytest.approx(0.5, abs=0.01)

    def test_replacing_early_removes_the_old_failure(self, quiet_scm, quiet_lives):
        lf = next(x for x in quiet_lives if x.failed)
        t = quiet_scm.counterfactual_failure_day(lf, 0.0, 1.0)
        horizon = t + 5.0
        assert quiet_scm.p_fail_before(lf, None, horizon) > 0.99
        assert (
            quiet_scm.p_fail_before(lf, t - 20.0, horizon) < 0.05
        )  # a new part does not fail so soon

    def test_replacing_after_the_failure_changes_nothing(self, quiet_scm, quiet_lives):
        lf = next(x for x in quiet_lives if x.failed)
        t = quiet_scm.counterfactual_failure_day(lf, 0.0, 1.0)
        assert quiet_scm.p_fail_before(lf, t + 3.0, t + 10.0) == pytest.approx(
            quiet_scm.p_fail_before(lf, None, t + 10.0), abs=0.05
        )

    def test_a_replacement_then_a_long_horizon_lets_the_new_part_fail(self, quiet_scm, quiet_lives):
        lf = quiet_lives[0]
        p = quiet_scm.p_fail_before(lf, 30.0, 30.0 + 2.0 * float(quiet_scm.fresh_clock.max()))
        assert p > 0.9

    def test_the_spread_is_at_least_a_quarter(self, quiet_scm, quiet_lives):
        assert quiet_scm.failure_sd(quiet_lives[0], 0.0, 1.0) >= 1.0 / DAY_Q

    def test_the_spread_grows_with_the_unseen_future(self, quiet_scm, quiet_lives):
        """A load the record cannot show matters more the further past the record the failure is."""
        lf = quiet_lives[2]
        near = quiet_scm.failure_sd(lf, 0.0, 1.0)
        far = quiet_scm.failure_sd(lf, 0.0, 0.5)  # a heavy derate pushes the failure far out
        assert far >= near
