"""P4a: the Theil-Sen streaming estimator, its blind window and its class prior. SYNTHETIC."""

import math
import time

import numpy as np
import pytest

from closed_loop.perception.estimator import (
    STATUS_ESTIMATED,
    STATUS_INSUFFICIENT,
    STATUS_NOT_DEGRADING,
    STATUS_PRIOR,
    ClassPrior,
    EstimatorOutput,
    TheilSenEstimator,
    class_prior,
    theil_sen,
)
from closed_loop.plant.observe import simulate_lives
from closed_loop.plant.params import MODES_BY_CLASS, PlantParams
from closed_loop.plant.scenario import Scenario

AID = "PU-201"
P = PlantParams()


def feed(est, hi_by_day, *, asset=AID, life_id=0, first_day=0):
    outs = []
    for i, h in enumerate(hi_by_day):
        outs.append(est.update(asset, first_day + i, h, life_id))
    return outs


def ramp(slope, level0, n, start_day=0):
    return [level0 + slope * (d - start_day) for d in range(start_day, start_day + n)]


class TestTheilSen:
    def test_exact_on_a_line(self):
        days = [3, 4, 5, 7, 8]
        values = [1.0 + 0.5 * d for d in days]
        slope, level = theil_sen(days, values)
        assert slope == pytest.approx(0.5)
        assert level == pytest.approx(1.0 + 0.5 * 8)

    def test_one_outlier_does_not_move_it_but_moves_least_squares(self):
        days = list(range(8))
        values = [2.0 + 0.2 * d for d in days]
        values[3] += 40.0
        slope, _ = theil_sen(days, values)
        ols = float(np.polyfit(days, values, 1)[0])
        assert slope == pytest.approx(0.2, abs=1e-9)
        assert abs(ols - 0.2) > 0.4  # the outlier drags least squares far from the true slope

    def test_needs_two_increasing_points(self):
        with pytest.raises(ValueError):
            theil_sen([1], [1.0])
        with pytest.raises(ValueError, match="increasing"):
            theil_sen([1, 1, 2], [1.0, 2.0, 3.0])

    def test_the_median_of_an_even_count_averages_the_middle_pair(self):
        # three points, three slopes (1, 1.5, 2): the odd case
        slope, _ = theil_sen([0, 1, 2], [0.0, 1.0, 3.0])
        assert slope == pytest.approx(1.5)
        # four points, six slopes 1, 1, 4/3, 3/2, 3/2, 2: the middle pair is 4/3 and 3/2
        slope, _ = theil_sen([0, 1, 2, 3], [0.0, 1.0, 3.0, 4.0])
        assert slope == pytest.approx(0.5 * (4 / 3 + 3 / 2))

    def test_the_level_is_robust_to_an_outlier_on_the_last_day(self):
        days = list(range(8))
        values = [2.0 + 0.2 * d for d in days]
        values[-1] += 30.0
        slope, level = theil_sen(days, values)
        assert slope == pytest.approx(0.2, abs=0.05)
        assert level == pytest.approx(2.0 + 0.2 * 7, abs=0.5)  # not 31.4


class TestStatuses:
    def test_blind_window_lasts_blind_days_after_a_reset(self):
        est = TheilSenEstimator(P)
        feed(est, ramp(0.3, 3.0, 5))  # some history
        est.reset_life(AID, 5)
        outs = feed(est, ramp(0.3, 3.0, 40), first_day=5, life_id=1)
        status = [o.status for o in outs]
        n_blind = int(P.blind_days)
        assert status[:n_blind] == [STATUS_PRIOR] * n_blind  # days 5 .. 27
        assert status[n_blind] != STATUS_PRIOR  # day 28
        assert outs[n_blind - 1].day == 5 + 22 and outs[n_blind].day == 5 + 23

    def test_first_life_is_blind_for_the_first_23_days_too(self):
        est = TheilSenEstimator(P)
        outs = feed(est, ramp(0.3, 3.0, 30))
        assert [o.status for o in outs[:23]] == [STATUS_PRIOR] * 23
        assert outs[23].status != STATUS_PRIOR

    def test_blind_readings_are_not_stored_so_the_first_estimate_waits_for_fresh_points(self):
        est = TheilSenEstimator(P)
        outs = feed(est, ramp(0.3, 3.0, 40))
        # day 23 is the first non-blind day: one point in the window, four more needed
        assert [o.status for o in outs[23:27]] == [STATUS_INSUFFICIENT] * 4
        assert outs[27].status == STATUS_ESTIMATED  # five readings: days 23 to 27

    def test_insufficient_below_min_points_in_the_window(self):
        est = TheilSenEstimator(P, window_days=8, min_points=5)
        feed(est, ramp(0.3, 3.0, 30))
        # mask four of the next eight days: the window then holds fewer than five readings
        outs = [
            est.update(AID, 30 + i, None if i in (1, 2, 3, 4) else 3.0 + 0.3 * (30 + i), 0)
            for i in range(8)
        ]
        assert outs[-1].status == STATUS_INSUFFICIENT
        assert outs[0].status == STATUS_ESTIMATED

    def test_a_reading_leaves_the_window_after_exactly_window_days(self):
        est = TheilSenEstimator(P, window_days=8, min_points=5)
        feed(est, [None] * 30)
        statuses = {}
        for i in range(10):
            day = 30 + i
            # five readings on days 30 to 34, then nothing
            value = 3.0 + 0.3 * day if i < 5 else None
            statuses[day] = est.update(AID, day, value, 0).status
        assert statuses[34] == STATUS_ESTIMATED
        assert statuses[37] == STATUS_ESTIMATED  # window = days 30 to 37: all five still in
        assert statuses[38] == STATUS_INSUFFICIENT  # window = days 31 to 38: four left

    def test_five_points_are_enough_and_four_are_not(self):
        est = TheilSenEstimator(P)
        feed(est, [None] * 30)
        statuses = [est.update(AID, 30 + i, 3.0 + 0.3 * (30 + i), 0).status for i in range(6)]
        assert statuses[:4] == [STATUS_INSUFFICIENT] * 4
        assert statuses[4] == STATUS_ESTIMATED

    def test_nan_counts_as_masked(self):
        est = TheilSenEstimator(P)
        feed(est, [None] * 30)
        a = [est.update(AID, 30 + i, math.nan if i % 2 else 5.0 + 0.3 * i, 0) for i in range(8)]
        assert a[-1].hi is None  # the echo of a masked day is None
        assert a[-1].status == STATUS_INSUFFICIENT  # only four readings

    def test_slope_below_the_minimum_is_not_degrading(self):
        est = TheilSenEstimator(P, min_slope=0.0625)
        outs = feed(est, ramp(0.0625 - 1 / 1024, 4.0, 40))
        assert outs[-1].status == STATUS_NOT_DEGRADING
        est = TheilSenEstimator(P, min_slope=0.0625)
        outs = feed(est, ramp(0.0625, 4.0, 40))
        assert outs[-1].status == STATUS_ESTIMATED  # exactly the minimum is allowed

    def test_level_below_the_alert_threshold_is_not_degrading(self):
        """Climbing steeply enough (0.06 per day) but still under the threshold of 2.0."""
        outs = feed(TheilSenEstimator(P), ramp(0.06, 0.0, 40))
        by_day = {o.day: o for o in outs}
        assert 0.06 * 32 < P.hi_alert < 0.06 * 34
        assert by_day[32].status == STATUS_NOT_DEGRADING  # level 1.92
        assert by_day[34].status == STATUS_ESTIMATED  # level 2.04

    def test_a_level_exactly_at_the_threshold_is_degrading(self):
        """The comparison is strict: at exactly hi_alert the index counts as degrading. Binary
        fractions keep the arithmetic exact."""
        est = TheilSenEstimator(P, min_slope=0.0625)
        series = [P.hi_alert - 0.0625 * (39 - d) for d in range(40)]
        out = feed(est, series)[-1]
        assert est.trend(AID).level == P.hi_alert  # type: ignore[union-attr]
        assert out.status == STATUS_ESTIMATED

    def test_a_flat_noisy_healthy_series_is_not_degrading(self):
        rng = np.random.default_rng(5)
        est = TheilSenEstimator(P)
        outs = feed(est, list(1.0 + 0.01 * rng.standard_normal(60)))
        assert {o.status for o in outs[30:]} == {STATUS_NOT_DEGRADING}


class TestEstimate:
    def test_a_noiseless_ramp_gives_the_exact_extrapolation(self):
        slope, level0 = 0.1, 1.0
        est = TheilSenEstimator(P)
        outs = feed(est, ramp(slope, level0, 50))
        o = outs[40]
        level = level0 + slope * 40
        assert o.status == STATUS_ESTIMATED
        assert o.m_days == pytest.approx((P.hi_fail - level) / slope)
        assert o.hi == pytest.approx(level)

    def test_the_point_is_capped_at_the_horizon_and_floored_at_zero(self):
        est = TheilSenEstimator(P)
        outs = feed(est, ramp(0.06, 2.5, 50))  # (9 - ~5.4) / 0.06 = 60: below the cap
        assert 0.0 < outs[-1].m_days < 90.0
        est = TheilSenEstimator(P, horizon_cap_days=20.0)
        outs = feed(est, ramp(0.06, 2.5, 50))
        assert outs[-1].m_days == 20.0
        est = TheilSenEstimator(P)
        outs = feed(est, ramp(0.5, 3.0, 50))  # far past the failure level by day 49
        assert outs[-1].status == STATUS_ESTIMATED and outs[-1].m_days == 0.0

    def test_the_scale_is_positive_has_a_floor_and_grows_with_the_point(self):
        est = TheilSenEstimator(P)
        near = feed(est, ramp(0.5, 3.0, 40))[-1]
        far = feed(TheilSenEstimator(P), ramp(0.07, 2.5, 60))[-1]
        assert near.sigma_days >= 1.0 and far.sigma_days >= 1.0
        assert far.m_days > near.m_days and far.sigma_days > near.sigma_days
        wide = feed(TheilSenEstimator(P, sigma_floor_days=7.0), ramp(0.5, 3.0, 40))[-1]
        assert wide.sigma_days == 7.0

    def test_the_scale_is_the_delta_method_value(self):
        """Noiseless ramp 1.0 + 0.1 d: day 39, level 4.9, point 41 days; window days 32 to 39."""
        out = feed(TheilSenEstimator(P), ramp(0.1, 1.0, 40))[-1]
        sxx = sum((d - 35.5) ** 2 for d in range(32, 40))
        assert sxx == pytest.approx(42.0)
        se_slope = 0.3 / math.sqrt(sxx)
        se_level = 0.3 * math.sqrt(1 / 8 + (39 - 35.5) ** 2 / sxx)
        m = (9.0 - 4.9) / 0.1
        assert out.m_days == pytest.approx(m)
        assert out.sigma_days == pytest.approx(math.hypot(m * se_slope / 0.1, se_level / 0.1))

    def test_output_echoes_asset_day_and_reading(self):
        est = TheilSenEstimator(P)
        (o,) = feed(est, [2.5], asset="CR-501", first_day=0)
        assert isinstance(o, EstimatorOutput)
        assert (o.asset_id, o.day, o.hi) == ("CR-501", 0, 2.5)

    def test_assets_are_independent(self):
        est = TheilSenEstimator(P)
        a = feed(est, ramp(0.3, 3.0, 40), asset="PU-201")
        b = feed(est, ramp(0.1, 3.0, 40), asset="PU-202")
        solo = feed(TheilSenEstimator(P), ramp(0.1, 3.0, 40), asset="PU-202")
        assert [o.m_days for o in b] == [o.m_days for o in solo]
        assert a[-1].m_days != b[-1].m_days


class TestCausality:
    def test_estimates_do_not_change_when_the_future_is_truncated(self):
        table = simulate_lives(P, 1000, Scenario())
        hi = table.hi[AID]
        full = TheilSenEstimator(P)
        full_out = [
            full.update(AID, d, None if np.isnan(h) else float(h), 0) for d, h in enumerate(hi)
        ]
        for cut in (24, 41, 60, 77):
            part = TheilSenEstimator(P)
            part_out = [
                part.update(AID, d, None if np.isnan(h) else float(h), 0)
                for d, h in enumerate(hi[:cut])
            ]
            assert part_out == full_out[:cut]

    def test_changing_a_future_reading_changes_nothing_before_it(self):
        base = ramp(0.3, 3.0, 50)
        alt = list(base)
        alt[40] += 25.0
        a = feed(TheilSenEstimator(P), base)
        b = feed(TheilSenEstimator(P), alt)
        assert a[:40] == b[:40]
        assert a[40] != b[40]

    def test_days_must_increase(self):
        est = TheilSenEstimator(P)
        feed(est, ramp(0.3, 3.0, 10))
        with pytest.raises(ValueError, match="causal"):
            est.update(AID, 9, 3.0, 0)
        with pytest.raises(ValueError, match="causal"):
            est.update(AID, 5, 3.0, 0)

    def test_bad_inputs_are_refused(self):
        est = TheilSenEstimator(P)
        with pytest.raises(ValueError, match="day"):
            est.update(AID, -1, 3.0, 0)
        with pytest.raises(ValueError, match="day"):
            est.update(AID, 1.5, 3.0, 0)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="finite"):
            est.update(AID, 0, math.inf, 0)
        with pytest.raises(ValueError, match="unknown asset"):
            est.update("XX-1", 0, 3.0, 0)
        with pytest.raises(ValueError):
            TheilSenEstimator(P, min_points=1)
        with pytest.raises(ValueError):
            TheilSenEstimator(P, window_days=4, min_points=5)
        with pytest.raises(ValueError):
            TheilSenEstimator(P, min_slope=0.0)


class TestLives:
    def test_a_new_life_id_is_an_implicit_reset(self):
        est = TheilSenEstimator(P)
        feed(est, ramp(0.3, 3.0, 40), life_id=0)
        o = est.update(AID, 40, 9.5, 1)
        assert o.status == STATUS_PRIOR
        assert est.life_start_day(AID) == 40.0
        later = [est.update(AID, 41 + i, 3.0, 1) for i in range(30)]
        assert later[21].status == STATUS_PRIOR and later[22].status != STATUS_PRIOR  # day 62, 63

    def test_reset_life_then_update_with_the_next_life_id_resets_once(self):
        est = TheilSenEstimator(P)
        feed(est, ramp(0.3, 3.0, 40), life_id=0)
        est.reset_life(AID, 40)
        est.update(AID, 40, 3.0, 1)
        assert est.life_start_day(AID) == 40.0
        est.update(AID, 41, 3.0, 1)
        assert est.life_start_day(AID) == 40.0  # no second reset

    def test_after_a_reset_the_trend_of_the_old_life_is_gone(self):
        est = TheilSenEstimator(P)
        feed(est, ramp(0.3, 3.0, 40))
        assert est.trend(AID) is not None
        est.reset_life(AID, 40)
        assert est.trend(AID) is None

    def test_a_reset_forgets_the_old_readings_even_when_the_blind_window_is_short(self):
        """With a blind window shorter than the trend window the old life's readings would still
        be inside it at the first post-blind day if the reset did not clear them."""
        short = PlantParams(blind_days=2.0)
        est = TheilSenEstimator(short)
        feed(est, ramp(0.3, 3.0, 40), life_id=0)
        est.reset_life(AID, 40)
        outs = [est.update(AID, 40 + i, 3.0, 1) for i in range(4)]
        assert [o.status for o in outs[:2]] == [STATUS_PRIOR, STATUS_PRIOR]
        assert outs[2].status == STATUS_INSUFFICIENT  # one fresh reading, not eight

    def test_the_age_of_the_prior_counts_from_the_reset(self):
        est = TheilSenEstimator(P)
        feed(est, ramp(0.3, 3.0, 5))
        est.reset_life(AID, 5)
        a = est.update(AID, 5, 1.0, 1)
        b = est.update(AID, 15, 1.0, 1)
        prior = class_prior("SLURRY_PUMP", P.renewal_onset_days)
        assert a.m_days == pytest.approx(prior.point_and_scale(0.0)[0])
        assert b.m_days == pytest.approx(prior.point_and_scale(10.0)[0])


class TestTrend:
    def test_trend_is_available_after_the_window_fills_and_forecasts_a_ramp(self):
        est = TheilSenEstimator(P)
        feed(est, ramp(0.4, 3.0, 40))
        tr = est.trend(AID)
        assert tr is not None and tr.day == 39 and tr.n == 8
        assert tr.slope == pytest.approx(0.4)
        assert tr.forecast(1) == pytest.approx(3.0 + 0.4 * 40)
        assert tr.forecast(3) == pytest.approx(3.0 + 0.4 * 42)

    def test_predictive_sd_grows_with_the_horizon_and_shrinks_with_the_window(self):
        est = TheilSenEstimator(P)
        feed(est, ramp(0.4, 3.0, 40))
        tr = est.trend(AID)
        assert tr is not None
        assert tr.predictive_sd(3) > tr.predictive_sd(1) > P.hi_noise_sd
        # day 39, window days 32 to 39 (mean 35.5, Sxx 42): one day ahead is 4.5 from the mean
        expect = 0.3 * math.sqrt(1.0 + 1.0 / 8 + 4.5**2 / 42.0)
        assert tr.predictive_sd(1) == pytest.approx(expect)
        assert tr.noise_sd == P.hi_noise_sd

    def test_trend_is_none_while_blind_or_short(self):
        est = TheilSenEstimator(P)
        assert est.trend(AID) is None
        feed(est, ramp(0.4, 3.0, 10))
        assert est.trend(AID) is None
        feed(est, ramp(0.4, 3.0, 30)[10:], first_day=10)
        assert est.trend(AID) is not None


class TestClassPrior:
    @pytest.mark.parametrize("asset_class", sorted(MODES_BY_CLASS))
    def test_the_fitted_weibull_matches_the_catalogue_life_law(self, asset_class):
        """KS distance to a plain Monte Carlo of onset offset + duration, mode-weighted."""
        prior = class_prior(asset_class, P.renewal_onset_days)
        rng = np.random.default_rng(17)
        modes = MODES_BY_CLASS[asset_class]
        w = np.array([m.weight for m in modes])
        n = 200_000
        pick = rng.choice(len(modes), size=n, p=w / w.sum())
        lo = np.array([m.duration_days[0] for m in modes])[pick]
        hi = np.array([m.duration_days[1] for m in modes])[pick]
        total = rng.uniform(*P.renewal_onset_days, size=n) + rng.uniform(lo, hi)
        total.sort()
        cdf_w = 1.0 - np.exp(-((total / prior.scale) ** prior.shape))
        ecdf = (np.arange(n) + 0.5) / n
        assert float(np.max(np.abs(cdf_w - ecdf))) < 0.06

    def test_conditional_quantiles_follow_the_weibull_survival_law(self):
        prior = ClassPrior(shape=3.0, scale=50.0)
        for age in (0.0, 10.0, 30.0):
            for p in (0.1, 0.5, 0.9):
                r = prior.remaining_quantile(age, p)
                surv = math.exp(-(((age + r) / 50.0) ** 3) + (age / 50.0) ** 3)
                assert surv == pytest.approx(1.0 - p)

    def test_remaining_life_shrinks_with_age_and_quantiles_increase_with_p(self):
        prior = class_prior("SLURRY_PUMP", P.renewal_onset_days)
        meds = [prior.point_and_scale(a)[0] for a in (0.0, 20.0, 40.0, 60.0)]
        assert meds == sorted(meds, reverse=True) and meds[-1] > 0
        qs = [prior.remaining_quantile(10.0, p) for p in (0.1, 0.3, 0.5, 0.7, 0.9)]
        assert qs == sorted(qs)

    def test_bad_arguments(self):
        prior = ClassPrior(shape=3.0, scale=50.0)
        with pytest.raises(ValueError):
            prior.remaining_quantile(-1.0, 0.5)
        with pytest.raises(ValueError):
            prior.remaining_quantile(1.0, 1.0)
        with pytest.raises(ValueError, match="unknown asset class"):
            class_prior("NOPE", (24.0, 60.0))

    def test_the_fit_uses_no_randomness_and_is_cached(self):
        a = class_prior("CRUSHER", P.renewal_onset_days)
        b = class_prior("CRUSHER", P.renewal_onset_days)
        assert a is b


class TestSanityOnTheSimulatedPlant:
    """Not a tuning target: a guard against an estimator that is plainly broken. (A healthy life can
    still be estimated as degrading when a process upset or a gain drift raises the index; that is
    the plant's confounding and is measured, not tested away.)"""

    def test_estimates_near_failure_are_in_the_right_region(self):
        errs: list[float] = []
        n_seen = 0
        for seed in range(1000, 1030):
            table = simulate_lives(P, seed, Scenario())
            for aid in table.lives:
                fail = table.failure_day[aid]
                if fail >= table.n_days:
                    continue
                est = TheilSenEstimator(P)
                for d in range(int(fail)):
                    h = table.hi[aid][d]
                    o = est.update(aid, d, None if np.isnan(h) else float(h), 0)
                    if o.status == STATUS_ESTIMATED and table.true_rul_days[aid][d] <= 10.0:
                        errs.append(abs(o.m_days - table.true_rul_days[aid][d]))
                        n_seen += 1
        assert n_seen > 100
        assert float(np.median(errs)) < 6.0  # a 10-day horizon, median error under 6 days


def test_updates_are_fast():
    """A generous bound (the estimator costs tens of microseconds per update)."""
    est = TheilSenEstimator(P)
    series = ramp(0.2, 3.0, 90)
    feed(est, series)  # warm the prior cache
    start = time.perf_counter()
    for rep in range(10):
        e = TheilSenEstimator(P)
        for d, h in enumerate(series):
            e.update(AID, d, h, rep)
    per_update = (time.perf_counter() - start) / (10 * len(series))
    assert per_update < 5e-4
