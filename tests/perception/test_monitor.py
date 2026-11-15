"""P4c: the coverage monitor (label modes and the label-free proxy). SYNTHETIC data only."""

import math
from functools import cache

import numpy as np
import pytest

from closed_loop.perception.estimator import TheilSenEstimator, Trend
from closed_loop.perception.labels import Label, LabelStatus
from closed_loop.perception.monitor import (
    LABEL_MODES,
    MODES,
    PROXY_ALPHA,
    CoverageMonitor,
    ProxyBand,
    life_fraction,
    proxy_development_scores,
)
from closed_loop.plant.params import PlantParams
from closed_loop.stats.intervals import wilson

H, M, LH, U = LabelStatus.HIT, LabelStatus.MISS, LabelStatus.LOWER_HIT, LabelStatus.UNKNOWN
P = PlantParams()


def life_labels(statuses, *, resolved=50, asset="PU-201", life=0, prior=False):
    return [
        Label(
            life, asset, 10 + i, 5.0, 20.0, 0.99, resolved, st, float(resolved), True, False, prior
        )
        for i, st in enumerate(statuses)
    ]


# ----------------------------------------------------------------------------- the three rules


class TestLifeFraction:
    LIFE = [M, U, LH, H]

    def test_opt_counts_only_misses(self):
        assert life_fraction(life_labels(self.LIFE), "opt") == pytest.approx(1 / 4)

    def test_pess_also_counts_unknown(self):
        assert life_fraction(life_labels(self.LIFE), "pess") == pytest.approx(2 / 4)

    def test_inf_drops_unknown_from_the_fraction(self):
        assert life_fraction(life_labels(self.LIFE), "inf") == pytest.approx(1 / 3)

    def test_all_unknown(self):
        labs = life_labels([U, U, U])
        assert life_fraction(labs, "opt") == 0.0  # opt never counts UNKNOWN as a miss
        assert life_fraction(labs, "pess") == 1.0  # pess always does
        assert life_fraction(labs, "inf") is None  # nothing informative: nothing to push

    def test_lower_hit_is_covered_in_every_mode(self):
        labs = life_labels([LH, LH])
        assert [life_fraction(labs, m) for m in LABEL_MODES] == [0.0, 0.0, 0.0]

    def test_empty_and_bad_mode(self):
        assert life_fraction([], "opt") is None
        with pytest.raises(ValueError, match="mode"):
            life_fraction(life_labels([M]), "proxy")


class TestLabelModes:
    def test_one_value_per_life_not_per_label(self):
        mon = CoverageMonitor("opt", p0=0.05)
        mon.observe(life_labels([M] * 100, resolved=60))
        assert mon.n_units == 1 and mon.n_lives == 1 and mon.n_labels == 100

    def test_lives_are_grouped_by_asset_and_life_id(self):
        mon = CoverageMonitor("opt", p0=0.05)
        mon.observe(
            life_labels([M, H], resolved=60, asset="PU-201", life=0)
            + life_labels([H, H], resolved=60, asset="PU-202", life=0)
            + life_labels([M, M], resolved=60, asset="PU-201", life=1)
        )
        assert mon.n_units == 3

    def test_the_value_pushed_is_the_fraction_of_that_life(self):
        a = CoverageMonitor("opt", p0=0.05)
        a.observe(life_labels([M, H, H, H], resolved=60))
        b = CoverageMonitor("opt", p0=0.05)
        b._process.update(0.25)  # noqa: SLF001
        assert a.log_e_value == pytest.approx(b.log_e_value)

    def test_pess_fires_on_all_unknown_lives_and_opt_never_does(self):
        mon = {m: CoverageMonitor(m, p0=0.01) for m in LABEL_MODES}
        for k in range(40):
            labs = life_labels([U] * 10, resolved=60 + k, life=k)
            for m in mon.values():
                m.observe(labs)
        assert mon["pess"].fired
        assert not mon["opt"].fired and mon["opt"].e_value < 1.0
        assert not mon["inf"].fired
        assert mon["inf"].n_units == 0 and mon["inf"].n_skipped_lives == 40

    def test_inf_uses_the_informative_labels_only(self):
        mon = CoverageMonitor("inf", p0=0.01)
        for k in range(30):
            mon.observe(
                life_labels([M, U, U, U], resolved=60 + k, life=k)
            )  # one informative: a miss
        assert mon.fired

    def test_fired_fields(self):
        mon = CoverageMonitor("opt", p0=0.01)
        k = 0
        while not mon.fired:
            mon.observe(life_labels([M] * 5, resolved=70 + k, life=k))
            k += 1
        assert mon.fired_unit == mon.n_units == k
        assert mon.fired_at_label == 5 * k
        assert mon.fired_day == 70 + k - 1
        for j in range(3):  # latched, the first firing is kept
            mon.observe(life_labels([M] * 5, resolved=200 + j, life=100 + j))
        assert mon.fired and mon.fired_day == 70 + k - 1 and mon.fired_unit == k

    def test_a_life_cannot_be_sent_twice(self):
        mon = CoverageMonitor("opt", p0=0.05)
        mon.observe(life_labels([M], resolved=60))
        with pytest.raises(ValueError, match="already observed"):
            mon.observe(life_labels([H], resolved=61))

    def test_labels_must_arrive_in_time_order(self):
        mon = CoverageMonitor("opt", p0=0.05)
        mon.observe(life_labels([H], resolved=60, life=0))
        with pytest.raises(ValueError, match="time order"):
            mon.observe(life_labels([H], resolved=59, life=1))

    def test_prior_labels_can_be_left_out(self):
        a = CoverageMonitor("opt", p0=0.05, include_prior=True)
        b = CoverageMonitor("opt", p0=0.05, include_prior=False)
        labs = life_labels([M, M, M], resolved=60, prior=True) + life_labels(
            [H, H], resolved=60, prior=False, life=1
        )
        a.observe(labs)
        b.observe(labs)
        assert a.n_units == 2 and b.n_units == 1 and b.n_labels == 2

    def test_bad_construction(self):
        with pytest.raises(ValueError, match="mode"):
            CoverageMonitor("sometimes")
        with pytest.raises(ValueError, match="p0"):
            CoverageMonitor("opt", p0=0.0)
        with pytest.raises(ValueError, match="only proxy"):
            CoverageMonitor("opt", band=ProxyBand(2.0))
        with pytest.raises(ValueError, match="needs a ProxyBand"):
            CoverageMonitor("proxy")
        with pytest.raises(ValueError, match="nominal miss rate"):
            CoverageMonitor("proxy", p0=0.01, band=ProxyBand(2.0, 0.05))
        assert MODES == ("opt", "pess", "inf", "proxy")

    def test_labels_are_refused_in_proxy_mode_and_days_in_label_modes(self):
        proxy = CoverageMonitor.proxy(ProxyBand(2.0))
        with pytest.raises(ValueError, match="observe_day"):
            proxy.observe(life_labels([M]))
        with pytest.raises(ValueError, match="proxy"):
            CoverageMonitor("opt").observe_day("PU-201", 3, 1.0, None)


# ----------------------------------------------------------------------------- power and size


def synthetic_stream(mode, rates, rng, *, n_cp=20, p0=0.01, delta=0.01):
    """Per-checkpoint misses with probability ``rates[k]`` in life ``k``; the monitor stops when it
    fires. Returns the monitor."""
    mon = CoverageMonitor(mode, p0=p0, delta=delta)
    for k, rate in enumerate(rates):
        statuses = [M if x < rate else H for x in rng.random(n_cp)]
        mon.observe(life_labels(statuses, resolved=50 + k, life=k))
        if mon.fired:
            break
    return mon


class TestDetectionDelay:
    """Done-when of P4c: the detection delay on an injected shift (miss rate 0.01 to 0.10) is
    measured here; the figures are written to the progress note."""

    def delays(self, mode, n_streams=120, seed=7):
        rng = np.random.default_rng(seed)
        pre, post = 40, 120
        rates = [0.01] * pre + [0.10] * post
        out = []
        for _ in range(n_streams):
            mon = synthetic_stream(mode, rates, rng)
            if mon.fired and mon.fired_unit > pre:
                out.append(mon.fired_unit - pre)  # lives after the shift
        return out

    def test_a_tenfold_rise_in_the_miss_rate_is_detected_within_a_few_dozen_lives(self):
        delays = self.delays("opt")
        assert len(delays) >= 110  # nearly every stream fires after the shift
        assert float(np.median(delays)) < 40
        assert max(delays) < 120

    def test_pess_and_inf_detect_it_too(self):
        for mode in ("pess", "inf"):
            delays = self.delays(mode, n_streams=60)
            assert len(delays) >= 55, mode
            assert float(np.median(delays)) < 40, mode

    def test_no_shift_no_systematic_firing(self):
        rng = np.random.default_rng(8)
        fired = sum(synthetic_stream("opt", [0.01] * 100, rng).fired for _ in range(100))
        assert fired <= 4


class TestFalseAlarms:
    """Done-when of P4c: at most ``delta`` on 500 unshifted streams, with a Wilson interval."""

    def test_per_checkpoint_misses_at_the_nominal_rate(self):
        rng = np.random.default_rng(9)
        n = 500
        k = sum(synthetic_stream("opt", [0.01] * 100, rng).fired for _ in range(n))
        lo, _ = wilson(k, n)
        assert lo <= 0.01, f"{k} of {n} streams fired"

    def test_whole_life_misses_at_the_nominal_rate(self):
        """Misses come in lives that are all missed or all covered (the dependence of a life's
        checkpoints): the per-life unit keeps the false-alarm rate where the null puts it."""
        rng = np.random.default_rng(10)
        n, fired = 500, 0
        for _ in range(n):
            mon = CoverageMonitor("opt", p0=0.01)
            for k in range(100):
                bad = rng.random() < 0.01
                mon.observe(life_labels([M if bad else H] * 20, resolved=50 + k, life=k))
            fired += mon.fired
        assert wilson(fired, n)[0] <= 0.01, f"{fired} of {n} streams fired"


# ----------------------------------------------------------------------------- the proxy


def trend(day=30, level=4.0, slope=0.2, n=8, noise_sd=0.3):
    dbar = day - 3.5
    return Trend(day, level, slope, n, 42.0, dbar, noise_sd)


class TestProxyBand:
    def test_fit_takes_the_conformal_rank(self):
        z = np.arange(1.0, 100.0)  # n = 99, alpha 0.05: rank ceil(100 * 0.95) = 95
        assert ProxyBand.fit(z, 0.05).quantile == 95.0
        assert ProxyBand.fit(z[::-1].copy(), 0.10).quantile == 90.0
        assert ProxyBand.fit(z, 0.05).alpha == 0.05

    def test_too_few_residuals_and_bad_input(self):
        with pytest.raises(ValueError, match="too few"):
            ProxyBand.fit(np.arange(1.0, 10.0), 0.05)  # ceil(10 * 0.95) = 10 > 9
        with pytest.raises(ValueError):
            ProxyBand.fit(np.array([]))
        with pytest.raises(ValueError):
            ProxyBand.fit(np.array([1.0, -1.0]))
        with pytest.raises(ValueError):
            ProxyBand.fit(np.arange(1.0, 100.0), 1.0)

    def test_interval_is_the_forecast_plus_minus_the_scaled_quantile(self):
        tr = trend()
        lo, hi = ProxyBand(2.0).interval(tr)
        assert 0.5 * (lo + hi) == pytest.approx(tr.forecast(1))
        assert hi - lo == pytest.approx(4.0 * tr.predictive_sd(1))

    def test_the_default_alpha(self):
        assert ProxyBand(2.0).alpha == PROXY_ALPHA == 0.05


class TestProxyDays:
    BAND = ProxyBand(2.0, 0.05)

    def feed(self, mon, asset, day, hi, tr, life=0):
        mon.observe_day(asset, day, hi, tr, life)

    def test_todays_reading_is_scored_against_yesterdays_band(self):
        mon = CoverageMonitor.proxy(self.BAND)
        tr = trend(day=30, level=4.0, slope=0.0)
        lo, hi = self.BAND.interval(tr)
        self.feed(mon, "PU-201", 30, 4.0, tr)
        assert mon.n_days == 0  # nothing to score on the first day
        self.feed(mon, "PU-201", 31, hi + 0.01, trend(day=31, slope=0.0))
        assert mon.n_days == 1 and mon.n_units == 1
        before = mon.log_e_value
        assert before > 0.0  # a miss was pushed: the e-value rose above 1
        self.feed(mon, "PU-201", 32, 4.0, trend(day=32, slope=0.0))
        assert mon.n_days == 2 and mon.log_e_value < before  # a day inside the band lowers it

    def test_the_edges_of_the_band_are_inside(self):
        tr = trend(day=30, level=4.0, slope=0.0)
        lo, hi = self.BAND.interval(tr)
        for edge in (lo, hi):
            m = CoverageMonitor.proxy(self.BAND)
            self.feed(m, "PU-201", 30, 4.0, tr)
            self.feed(m, "PU-201", 31, edge, None)
            ref = CoverageMonitor.proxy(self.BAND)
            ref._process.update(0.0)  # noqa: SLF001
            assert m.log_e_value == pytest.approx(ref.log_e_value)

    def test_a_masked_day_scores_nothing_but_tomorrows_band_is_still_set(self):
        mon = CoverageMonitor.proxy(self.BAND)
        self.feed(mon, "PU-201", 30, 4.0, trend(day=30))
        self.feed(mon, "PU-201", 31, None, trend(day=31))
        assert mon.n_days == 0
        self.feed(mon, "PU-201", 32, 9.0, trend(day=32))  # far outside yesterday's band
        assert mon.n_days == 1
        mon2 = CoverageMonitor.proxy(self.BAND)
        self.feed(mon2, "PU-201", 30, 4.0, trend(day=30))
        self.feed(mon2, "PU-201", 31, math.nan, trend(day=31))
        assert mon2.n_days == 0

    def test_no_trend_means_no_band_for_tomorrow(self):
        mon = CoverageMonitor.proxy(self.BAND)
        self.feed(mon, "PU-201", 30, 4.0, trend(day=30))
        assert mon.n_days == 0
        self.feed(mon, "PU-201", 31, 4.0, None)  # scored against day 30's band; no trend today
        assert mon.n_days == 1
        self.feed(mon, "PU-201", 32, 9.0, trend(day=32))  # no band for day 32
        assert mon.n_days == 1

    def test_a_day_that_skips_a_day_is_not_scored(self):
        mon = CoverageMonitor.proxy(self.BAND)
        self.feed(mon, "PU-201", 30, 4.0, trend(day=30))
        self.feed(mon, "PU-201", 32, 9.0, trend(day=32))
        assert mon.n_days == 0

    def test_a_new_life_drops_the_stored_band(self):
        mon = CoverageMonitor.proxy(self.BAND)
        self.feed(mon, "PU-201", 30, 4.0, trend(day=30), life=0)
        self.feed(mon, "PU-201", 31, 9.0, trend(day=31), life=1)
        assert mon.n_days == 0

    def test_assets_have_their_own_bands(self):
        mon = CoverageMonitor.proxy(self.BAND)
        self.feed(mon, "PU-201", 30, 4.0, trend(day=30, level=4.0, slope=0.0))
        self.feed(mon, "PU-202", 30, 8.0, trend(day=30, level=8.0, slope=0.0))
        self.feed(mon, "PU-201", 31, 4.0, trend(day=31, level=4.0, slope=0.0))
        self.feed(mon, "PU-202", 31, 8.0, trend(day=31, level=8.0, slope=0.0))
        assert mon.n_days == 2 and not mon.fired
        assert mon.log_e_value < 0.0  # two inside-band days

    def test_a_stale_trend_is_not_used(self):
        mon = CoverageMonitor.proxy(self.BAND)
        self.feed(mon, "PU-201", 30, 4.0, trend(day=29))  # the trend is from another day
        self.feed(mon, "PU-201", 31, 9.0, trend(day=31))
        assert mon.n_days == 0

    def test_a_run_of_misses_fires_and_reports_day_and_count(self):
        mon = CoverageMonitor.proxy(self.BAND)
        day = 30
        self.feed(mon, "PU-201", day, 4.0, trend(day=day, slope=0.0))
        while not mon.fired and day < 60:
            day += 1
            self.feed(mon, "PU-201", day, 9.0, trend(day=day, slope=0.0))
        assert mon.fired and mon.fired_day == day and mon.fired_at_label == mon.n_days
        assert mon.n_days <= 4

    def test_observe_day_is_for_proxy_mode_only(self):
        with pytest.raises(ValueError, match="proxy"):
            CoverageMonitor("opt").observe_day("PU-201", 3, 1.0, None)


@cache
def _band():
    return ProxyBand.fit(proxy_development_scores(P, range(2000, 2012)), 0.05)


class TestProxyWithTheRealEstimator:
    def run(self, hi_series, band):
        est = TheilSenEstimator(P)
        mon = CoverageMonitor.proxy(band)
        for d, h in enumerate(hi_series):
            est.update("PU-201", d, h, 0)
            mon.observe_day("PU-201", d, h, est.trend("PU-201"), 0)
        return mon

    def band(self):
        return _band()

    def test_a_smooth_ramp_with_calibrated_noise_does_not_fire(self):
        band = self.band()
        rng = np.random.default_rng(1)
        fired = 0
        for _ in range(30):
            ramp = 2.5 + 0.15 * np.arange(70) + P.hi_noise_sd * rng.standard_normal(70)
            fired += self.run(list(ramp), band).fired
        assert fired <= 2

    def test_a_path_that_gets_noisier_than_the_calibrated_model_fires_it(self):
        band = self.band()
        rng = np.random.default_rng(2)
        fired = 0
        for _ in range(30):
            noise = P.hi_noise_sd * rng.standard_normal(70)
            noise[45:] *= 3.0  # the daily noise triples from day 45
            fired += self.run(list(2.5 + 0.15 * np.arange(70) + noise), band).fired
        assert fired >= 22

    def test_a_sustained_step_is_absorbed_by_the_trend_and_does_not_fire_it(self):
        """A process upset moves the index for days; Theil-Sen follows it within three days, so the
        band is missed at the start and the end of the step and not in between. Silence here does
        not mean the contract held."""
        band = self.band()
        ramp = 2.5 + 0.15 * np.arange(70)
        ramp[45:52] += 2.0
        est = TheilSenEstimator(P)
        mon = CoverageMonitor.proxy(band)
        stored = None
        misses = []
        for d, h in enumerate(ramp):
            est.update("PU-201", d, float(h), 0)
            if stored is not None and not (stored[0] <= h <= stored[1]):
                misses.append(d)
            tr = est.trend("PU-201")
            stored = band.interval(tr) if tr is not None else None
            mon.observe_day("PU-201", d, float(h), tr, 0)
        assert [d for d in misses if 45 <= d <= 51] == [45, 46, 47]
        assert [d for d in misses if d >= 52][0] == 52  # and again when the upset ends
        assert not mon.fired

    def test_a_trend_the_extrapolator_follows_is_not_seen(self):
        """Silence does not mean the contract held: a slow drift is forecast one day ahead."""
        band = self.band()
        rng = np.random.default_rng(3)
        fired = 0
        for _ in range(30):
            drift = 0.045 * np.maximum(np.arange(70) - 35, 0)
            ramp = 2.5 + 0.15 * np.arange(70) + drift + P.hi_noise_sd * rng.standard_normal(70)
            fired += self.run(list(ramp), band).fired
        assert fired <= 3


class TestProxyDevelopmentScores:
    def test_scores_are_positive_and_the_band_misses_about_alpha_on_other_seeds(self):
        z = proxy_development_scores(P, range(2000, 2030))
        assert z.ndim == 1 and len(z) > 3000 and np.all(z >= 0.0)
        band = ProxyBand.fit(z, 0.05)
        other = proxy_development_scores(P, range(1000, 1030))
        miss = float(np.mean(other > band.quantile))
        assert 0.03 < miss < 0.07

    def test_only_dev_and_calib_seeds(self):
        for bad in ([3000], [2**41], [5], [1000, 3500]):
            with pytest.raises(ValueError, match="dev .* and calib"):
                proxy_development_scores(P, bad)
        with pytest.raises(ValueError, match="no one-day-ahead"):
            proxy_development_scores(P, [])
