"""P4b: the seven conformal providers. SYNTHETIC data only.

Each rule of 0001 section 6.3 has a unit test on hand-built labels (so a mutated rule is caught
there); the statistical claims of the packet run the real classes on synthetic score streams.
"""

import dataclasses
import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from closed_loop.audit.seeds import DEV_SEEDS
from closed_loop.perception.conformal import (
    ALPHA_LIFE,
    AUDIT_RHO,
    GAMMA,
    VARIANTS,
    WINDOW,
    conformal_quantiles,
    development_data,
    is_audit_life,
    make_provider,
    replay_open_loop,
    score,
)
from closed_loop.perception.coverage import checkpoint_coverage, open_loop_lives
from closed_loop.perception.estimator import EstimatorOutput
from closed_loop.perception.labels import Label, LabelStatus, RulInterval, label_status
from closed_loop.plant.params import PlantParams
from closed_loop.stats.intervals import wilson

H, M, LH, U = LabelStatus.HIT, LabelStatus.MISS, LabelStatus.LOWER_HIT, LabelStatus.UNKNOWN
P = PlantParams()
ALPHA = 0.10
LEVEL = 0.90


def est(m=0.0, sigma=1.0, status="estimated"):
    return EstimatorOutput("PU-201", 10, status, m, sigma, 5.0)


def lab(
    status,
    *,
    lo=-1.0,
    hi=1.0,
    day=10,
    resolved=None,
    end=None,
    censored=False,
    audit=False,
    prior=False,
    truth=None,
    m=0.0,
    sigma=1.0,
    life=0,
):
    end = day + 3.0 if end is None else end
    return Label(
        life,
        "PU-201",
        day,
        lo,
        hi,
        LEVEL,
        int(math.ceil(end)) if resolved is None else resolved,
        status,
        end,
        censored,
        audit,
        prior,
        true_rul=truth,
        m=m,
        sigma=sigma,
    )


def exact_label(lo, hi, rul, day):
    """A label for a checkpoint whose life failed ``rul`` days later (m = 0, sigma = 1)."""
    end = day + rul
    status = label_status(lo, hi, day, end, False)
    return Label(
        0,
        "PU-201",
        day,
        lo,
        hi,
        LEVEL,
        day,
        status,
        end,
        False,
        False,
        False,
        true_rul=rul,
        m=0.0,
        sigma=1.0,
    )


def dev(n=1000, seed=0):
    return np.random.default_rng(seed).standard_normal(n)


def life_maxima(rng, n_lives, n_est, n_prior, prior_shift=0.0):
    """The ``(n_lives, 4)`` scores of V6 for lives of ``n_est`` estimated and ``n_prior`` prior
    checkpoints of standard normal scores (the prior ones shifted by ``prior_shift``)."""
    r = rng.standard_normal((n_lives, n_est))
    p = rng.standard_normal((n_lives, n_prior)) + prior_shift
    return np.column_stack([r.max(1), (-r).max(1), p.max(1), (-p).max(1)])


def provider(name, *, alpha=None, scores=None, seed=7, **kw):
    alpha = (ALPHA_LIFE if name == "V6" else ALPHA) if alpha is None else alpha
    scores = dev() if scores is None else scores
    if name == "V6":
        scores = life_maxima(np.random.default_rng(1), 200, 12, 12)
    return make_provider(name, alpha=alpha, development_scores=scores, seed=seed, **kw)


# ----------------------------------------------------------------------------- quantiles


class TestQuantiles:
    def test_hand_ranks(self):
        pool = np.arange(1.0, 100.0)  # n = 99
        assert conformal_quantiles(pool, 0.05) == (5.0, 95.0)  # ranks floor(5.0) and ceil(95.0)
        assert conformal_quantiles(pool, 0.10) == (10.0, 90.0)
        lo, hi = conformal_quantiles(pool, 0.005)
        assert (lo, hi) == (-math.inf, math.inf)  # ranks 0 and 100 lie outside the pool
        assert conformal_quantiles(pool, 0.0) == (-math.inf, math.inf)
        assert conformal_quantiles(pool, -0.3) == (-math.inf, math.inf)

    def test_the_extreme_ranks_that_still_fit_in_the_pool_are_finite(self):
        pool = np.arange(1.0, 100.0)  # n = 99
        # a = 0.01: ranks floor(1.0) = 1 and ceil(99.0) = 99, the smallest and largest scores
        assert conformal_quantiles(pool, 0.01) == (1.0, 99.0)
        # one step further in and the pool is too small for the level
        lo, hi = conformal_quantiles(pool, 0.0099)
        assert (lo, hi) == (-math.inf, math.inf)

    def test_a_rounding_error_in_the_rank_does_not_tip_it(self):
        pool = np.arange(1.0, 100.0)
        for a in (0.05, 0.1, 0.2, 0.25, 0.01 * 3):
            lo, hi = conformal_quantiles(pool, a)
            assert lo == math.floor(100 * a + 1e-9) and hi == math.ceil(100 * (1 - a) - 1e-9)

    def test_above_one_half_is_treated_as_one_half(self):
        pool = np.arange(1.0, 100.0)
        assert conformal_quantiles(pool, 0.9) == conformal_quantiles(pool, 0.5) == (50.0, 50.0)

    def test_empty_pool_is_refused(self):
        with pytest.raises(ValueError, match="empty"):
            conformal_quantiles(np.array([]), 0.1)

    @given(
        st.lists(st.floats(-50, 50, allow_nan=False), min_size=1, max_size=200),
        st.floats(0.001, 0.499),
    )
    def test_lower_never_exceeds_upper_and_values_come_from_the_pool(self, xs, a):
        pool = np.sort(np.array(xs))
        lo, hi = conformal_quantiles(pool, a)
        assert lo <= hi
        for q in (lo, hi):
            assert math.isinf(q) or q in pool

    def test_each_tail_misses_with_probability_at_most_a(self):
        rng = np.random.default_rng(3)
        n, a, reps = 49, 0.1, 20000
        pools = np.sort(rng.standard_normal((reps, n)), axis=1)
        k_hi, k_lo = math.ceil((n + 1) * (1 - a)), math.floor((n + 1) * a)
        new = rng.standard_normal(reps)
        up = float(np.mean(new > pools[:, k_hi - 1]))
        low = float(np.mean(new < pools[:, k_lo - 1]))
        assert up <= a + 0.006 and low <= a + 0.006
        assert up >= a - 0.025 and low >= a - 0.025  # and not wastefully conservative


# ----------------------------------------------------------------------------- the factory


class TestFactory:
    def test_all_seven_exist_with_their_names(self):
        assert VARIANTS == ("V1", "V2", "V3", "V4", "V5", "V6", "V7")
        for name in VARIANTS:
            assert provider(name).name == name

    def test_bad_arguments(self):
        with pytest.raises(ValueError, match="unknown variant"):
            make_provider("V8", alpha=0.1, development_scores=dev(), seed=0)
        for alpha in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(ValueError, match="alpha"):
                make_provider("V1", alpha=alpha, development_scores=dev(), seed=0)
        with pytest.raises(ValueError, match="non-empty"):
            make_provider("V1", alpha=0.1, development_scores=np.array([]), seed=0)
        with pytest.raises(ValueError, match="non-finite"):
            make_provider("V3", alpha=0.1, development_scores=np.array([0.0, np.nan]), seed=0)
        with pytest.raises(ValueError, match="shape"):
            make_provider("V6", alpha=0.05, development_scores=dev(), seed=0)
        with pytest.raises(ValueError, match="shape"):  # the old (n_lives, 2) layout
            make_provider("V6", alpha=0.05, development_scores=np.zeros((50, 2)), seed=0)
        bad = life_maxima(np.random.default_rng(0), 50, 5, 5)
        bad[3, 1] = np.nan
        with pytest.raises(ValueError, match=r"NaN or a \+inf"):
            make_provider("V6", alpha=0.05, development_scores=bad, seed=0)
        bad[3, 1] = np.inf
        with pytest.raises(ValueError, match=r"NaN or a \+inf"):
            make_provider("V6", alpha=0.05, development_scores=bad, seed=0)
        with pytest.raises(ValueError):
            provider("V2", gamma=0.0)

    def test_a_provider_serves_only_the_level_it_was_built_for(self):
        p = provider("V1")
        p.predict(est(), LEVEL)
        with pytest.raises(ValueError, match="one provider per level"):
            p.predict(est(), 0.99)

    def test_the_interval_is_the_quantiles_scaled_by_sigma_and_shifted_by_m(self):
        scores = np.arange(-49.0, 51.0)  # n = 100: a = 0.05 -> ranks 5 and 96 (values -45 and 46)
        p = make_provider("V1", alpha=0.10, development_scores=scores, seed=0)
        iv = p.predict(est(m=40.0, sigma=2.0), 0.90)
        assert (iv.lo, iv.hi) == (40.0 + 2.0 * -45.0, 40.0 + 2.0 * 46.0)
        assert iv.m == 40.0 and iv.level == 0.90 and iv.prior is False

    def test_prior_status_gets_the_prior_interval_and_pool(self):
        scores = np.arange(-49.0, 51.0)
        prior = (
            np.arange(-9.0, 11.0) * 10.0
        )  # n = 20: ranks floor(21 * 0.05) = 1, ceil(21 * 0.95) = 20
        p = make_provider("V1", alpha=0.10, development_scores=scores, seed=0, prior_scores=prior)
        a = p.predict(est(status="estimated"), 0.90)
        b = p.predict(est(status="not_degrading"), 0.90)
        assert b.prior and not a.prior
        assert (b.lo, b.hi) == (-90.0, 100.0)
        assert (a.lo, a.hi) == (-45.0, 46.0)

    def test_without_prior_scores_the_prior_status_reuses_the_trend_pool(self):
        p = make_provider("V1", alpha=0.10, development_scores=np.arange(-49.0, 51.0), seed=0)
        a, b = p.predict(est(), 0.90), p.predict(est(status="insufficient"), 0.90)
        assert (a.lo, a.hi) == (b.lo, b.hi) and b.prior


# ----------------------------------------------------------------------------- V1 and V6


def _wilson_contains(k, n, lo, hi):
    w_lo, w_hi = wilson(k, n)
    return w_lo <= hi and w_hi >= lo


class TestNominalCoverage:
    """Done-when of P4b: V1 and V6 hit their nominal coverage on an exchangeable score stream."""

    def test_v1_split_covers_at_its_level(self):
        rng = np.random.default_rng(11)
        n_dev, reps, hits = 199, 6000, 0
        for _ in range(reps):
            p = make_provider(
                "V1", alpha=0.10, development_scores=rng.standard_normal(n_dev), seed=0
            )
            iv = p.predict(est(), 0.90)
            hits += iv.lo <= rng.standard_normal() <= iv.hi
        # conformal coverage lies in [1 - alpha, 1 - alpha + 1 / (n + 1)] = [0.90, 0.905]
        assert _wilson_contains(hits, reps, 0.90, 0.905)
        assert hits / reps >= 0.89

    def test_v6_band_holds_for_a_whole_life_at_alpha_life(self):
        rng = np.random.default_rng(12)
        n_dev, n_cp, reps, hits = 400, 20, 3000, 0
        for _ in range(reps):
            r = rng.standard_normal((n_dev, n_cp))
            # a life of estimated checkpoints only: the prior columns hold -inf (no checkpoint)
            none = np.full(n_dev, -np.inf)
            scores = np.column_stack([r.max(axis=1), (-r).max(axis=1), none, none])
            p = make_provider("V6", alpha=ALPHA_LIFE, development_scores=scores, seed=0)
            iv = p.predict(est(), 1 - ALPHA_LIFE)
            new = rng.standard_normal(n_cp)
            hits += bool(np.all((iv.lo <= new) & (new <= iv.hi)))
        # two one-sided tails of 0.0125 each on this class: 0.975 or better; never below 0.95
        assert hits / reps >= 0.95
        assert _wilson_contains(hits, reps, 0.95, 0.99)

    def test_v6_is_wider_than_the_step_band_at_the_same_level(self):
        rng = np.random.default_rng(13)
        r = rng.standard_normal((400, 20))
        life = make_provider(
            "V6",
            alpha=0.05,
            development_scores=life_maxima(np.random.default_rng(14), 400, 20, 1),
            seed=0,
        )
        step = make_provider("V1", alpha=0.05, development_scores=r.ravel(), seed=0)
        a, b = life.predict(est(), 0.95), step.predict(est(), 0.95)
        assert a.hi > b.hi and a.lo < b.lo

    def test_v6_never_updates(self):
        p = provider("V6", alpha=0.05)
        before = p.predict(est(), 0.95)
        p.observe([lab(M, resolved=20), lab(H, resolved=21)])
        assert p.predict(est(), 0.95) == before
        assert p.state()["n_labels"] == 2.0

    def test_v1_never_updates(self):
        p = provider("V1")
        before = p.predict(est(), LEVEL)
        for d in range(50):
            p.observe([lab(M, resolved=d, end=d)])
        assert p.predict(est(), LEVEL) == before


# ----------------------------------------------------------------------------- the update rules


def a_est(p):
    return p.state()["alpha_t_est"]


class TestAciRules:
    def test_v2_counts_a_censored_label_as_a_hit_whatever_its_status(self):
        for status in (M, LH, U):
            p = provider("V2")
            p.observe([lab(status, censored=True, end=20.0, resolved=20)])
            assert a_est(p) == pytest.approx(ALPHA + GAMMA * ALPHA), status

    def test_v2_counts_an_observed_miss_as_a_miss(self):
        p = provider("V2")
        p.observe([lab(M, censored=False, end=13.0, resolved=13, hi=0.5, lo=-0.5)])
        assert a_est(p) == pytest.approx(ALPHA + GAMMA * (ALPHA - 1.0))

    def test_v2_observed_hit(self):
        p = provider("V2")
        p.observe([lab(H, censored=False, resolved=13)])
        assert a_est(p) == pytest.approx(ALPHA + GAMMA * ALPHA)

    def test_v3_skips_unknown_and_counts_the_other_censored_statuses_correctly(self):
        p = provider("V3")
        p.observe([lab(U, censored=True, resolved=20, end=20.0)])
        assert a_est(p) == ALPHA and p.state()["skipped_est"] == 1.0
        assert p.state()["updates_est"] == 0.0
        p.observe([lab(LH, censored=True, resolved=21, end=21.0)])
        assert a_est(p) == pytest.approx(ALPHA + GAMMA * ALPHA)  # a hit
        p.observe([lab(M, censored=True, resolved=22, end=22.0)])
        assert a_est(p) == pytest.approx(ALPHA + GAMMA * ALPHA + GAMMA * (ALPHA - 1.0))
        assert p.state()["updates_est"] == 2.0

    def test_v3_exact_labels(self):
        p = provider("V3")
        p.observe([lab(H, resolved=11), lab(M, resolved=12, hi=0.5, lo=-0.5)])
        assert a_est(p) == pytest.approx(ALPHA + GAMMA * ALPHA + GAMMA * (ALPHA - 1.0))

    def test_v4_updates_only_from_audit_lives(self):
        p = provider("V4")
        p.observe([lab(M, audit=False, resolved=12, end=12.0)] * 5)
        assert a_est(p) == ALPHA and p.state()["skipped_est"] == 5.0
        assert p.state()["scores_added_est"] == 0.0  # a non-audit failure adds no score either
        p.observe([lab(M, audit=True, resolved=13, end=13.0, hi=0.5, lo=-0.5)])
        assert a_est(p) == pytest.approx(ALPHA + GAMMA * (ALPHA - 1.0))
        assert p.state()["scores_added_est"] == 1.0

    def test_v4_uses_the_v3_rule_on_audit_labels(self):
        p = provider("V4")
        p.observe([lab(U, audit=True, censored=True, end=14.0, resolved=14)])
        assert a_est(p) == ALPHA  # unknown is skipped
        p.observe([lab(LH, audit=True, censored=True, end=15.0, resolved=15)])
        assert a_est(p) == pytest.approx(ALPHA + GAMMA * ALPHA)

    def test_v7_reads_the_truth_and_ignores_the_status(self):
        p = provider("V7")
        # the status says LOWER_HIT, the truth is inside [-1, 1]: a hit
        p.observe([lab(LH, censored=True, truth=0.4, end=20.0, resolved=20)])
        assert a_est(p) == pytest.approx(ALPHA + GAMMA * ALPHA)
        # the status says UNKNOWN, the truth is outside: a miss
        p.observe([lab(U, censored=True, truth=7.0, end=21.0, resolved=21)])
        assert a_est(p) == pytest.approx(ALPHA + GAMMA * ALPHA + GAMMA * (ALPHA - 1.0))
        assert p.state()["updates_est"] == 2.0

    def test_v7_needs_the_truth(self):
        with pytest.raises(ValueError, match="true_rul"):
            provider("V7").observe([lab(H, truth=None)])

    def test_v7_adds_the_true_score_even_for_a_censored_life(self):
        p = provider("V7")
        p.observe([lab(U, censored=True, truth=3.0, m=1.0, sigma=2.0, end=20.0, resolved=20)])
        assert p.state()["scores_added_est"] == 1.0

    def test_only_an_observed_failure_adds_a_score(self):
        for name in ("V2", "V3", "V5"):
            p = provider(name)
            p.observe([lab(LH, censored=True, end=12.0, resolved=12)])
            assert p.state()["scores_added_est"] == 0.0, name
            p.observe([lab(H, censored=False, resolved=13, end=13.0)])
            assert p.state()["scores_added_est"] == 1.0, name

    def test_the_window_is_a_rolling_pool_of_at_most_500(self):
        p = provider("V3", scores=dev(5000))
        assert p.state()["window_est"] == WINDOW
        p.observe([exact_label(-1.0, 1.0, 0.1 * k, 10) for k in range(60)])
        assert p.state()["window_est"] == WINDOW
        small = provider("V3", scores=dev(40))
        assert small.state()["window_est"] == 40
        small.observe([exact_label(-1.0, 1.0, 0.0, 10)])
        assert small.state()["window_est"] == 41

    def test_a_new_score_moves_the_interval(self):
        p = provider("V3", scores=dev(300), window=300)
        before = p.predict(est(), LEVEL)
        p.observe([exact_label(-1.0, 1.0, 50.0, 10)] * 12)  # twelve huge scores
        after = p.predict(est(), LEVEL)
        assert after.hi > before.hi

    def test_a_new_score_moves_the_interval_through_the_pool_alone(self):
        """With the tracker frozen (gamma ~ 0) only the rolling pool can move the interval, so a
        sorted copy that went stale after a new score would show here."""
        p = provider("V3", scores=dev(300), window=300, gamma=1e-12)
        before = p.predict(est(), LEVEL)
        # hits (the interval is wide) with scores far in the upper tail
        p.observe([exact_label(-1e3, 1e3, 500.0, 10)] * 12)
        after = p.predict(est(), LEVEL)
        assert p.state()["scores_added_est"] == 12.0
        assert after.hi > before.hi and after.lo == pytest.approx(before.lo, abs=1e-6)

    def test_a_label_without_point_and_scale_updates_the_tracker_but_adds_no_score(self):
        p = provider("V3")
        bare = lab(M, resolved=12, end=13.0, hi=0.5, lo=-0.5, m=None, sigma=None)
        p.observe([bare])
        assert a_est(p) == pytest.approx(ALPHA + GAMMA * (ALPHA - 1.0))
        assert p.state()["scores_added_est"] == 0.0

    def test_prior_labels_move_the_prior_tracker_only(self):
        p = provider("V3")
        p.observe([lab(M, prior=True, resolved=12, end=12.0, hi=0.5, lo=-0.5)])
        s = p.state()
        assert s["alpha_t_est"] == ALPHA
        assert s["alpha_t_prior"] == pytest.approx(ALPHA + GAMMA * (ALPHA - 1.0))
        assert s["updates_prior"] == 1.0 and s["updates_est"] == 0.0

    def test_the_miss_indicator_moves_the_interval_the_right_way(self):
        """A miss lowers alpha_t, which widens the interval; a hit raises it and narrows it."""
        p = provider("V3")
        w0 = p.predict(est(), LEVEL)
        for d in range(40):
            p.observe([exact_label(-1.0, 1.0, 9.0, 10 + d)])
        w1 = p.predict(est(), LEVEL)
        assert (w1.hi - w1.lo) > (w0.hi - w0.lo)
        q = provider("V3")
        for d in range(40):
            q.observe([exact_label(-1.0, 1.0, 0.0, 10 + d)])
        w2 = q.predict(est(), LEVEL)
        assert (w2.hi - w2.lo) < (w0.hi - w0.lo)


class TestPidRules:
    def test_a_miss_raises_the_offset_and_a_hit_lowers_it(self):
        p = provider("V5")
        width = p._streams[False].width  # noqa: SLF001
        p.observe([lab(M, resolved=11, end=11.0, hi=0.5, lo=-0.5)])
        s = p.state()
        assert s["offset_est"] == pytest.approx(0.05 * width * (1.0 - ALPHA))
        assert s["integral_est"] == pytest.approx(1.0 - ALPHA)
        p.observe([lab(H, resolved=12, end=12.0)])
        s = p.state()
        assert s["offset_est"] == pytest.approx(0.05 * width * (1.0 - ALPHA - ALPHA))
        assert s["integral_est"] == pytest.approx(1.0 - 2 * ALPHA)

    def test_v5_uses_the_v3_label_rule(self):
        p = provider("V5")
        p.observe([lab(U, censored=True, end=20.0, resolved=20)])
        assert p.state()["offset_est"] == 0.0 and p.state()["skipped_est"] == 1.0
        p.observe([lab(LH, censored=True, end=21.0, resolved=21)])
        assert p.state()["integral_est"] == pytest.approx(-ALPHA)  # a hit
        p.observe([lab(M, censored=True, end=22.0, resolved=22)])
        assert p.state()["integral_est"] == pytest.approx(-ALPHA + 1.0 - ALPHA)

    def test_offset_widens_both_sides_and_a_negative_one_narrows_them(self):
        p = provider("V5")
        base = p.predict(est(), LEVEL)
        for d in range(30):  # censored misses: a miss indicator, and no score added
            p.observe([lab(M, censored=True, resolved=11 + d, end=11.0 + d, hi=0.5, lo=-0.5)])
        wide = p.predict(est(), LEVEL)
        assert wide.lo < base.lo and wide.hi > base.hi
        q = provider("V5")
        for d in range(60):
            q.observe([lab(LH, censored=True, resolved=11 + d, end=11.0 + d)])
        narrow = q.predict(est(), LEVEL)
        assert narrow.lo > base.lo and narrow.hi < base.hi and narrow.lo <= narrow.hi

    def test_a_narrowing_that_crosses_collapses_onto_the_middle(self):
        p = provider("V5", eta_rel=20.0)
        for d in range(60):  # a long run of hits drives the offset far below zero
            p.observe([lab(LH, censored=True, resolved=11 + d, end=11.0 + d)])
        iv = p.predict(est(m=30.0, sigma=2.0), LEVEL)
        assert iv.lo == iv.hi  # an empty interval would be refused; it sits on one point instead

    def test_the_integrator_saturates(self):
        p = provider("V5", k_i=0.5, eta_rel=1e-9)
        width = p._streams[False].width  # noqa: SLF001
        base = p.predict(est(), LEVEL)
        for d in range(2000):
            p.observe([lab(M, censored=True, resolved=11 + d, end=11.0 + d, hi=0.5, lo=-0.5)])
        top = p.predict(est(), LEVEL)
        assert top.hi - base.hi == pytest.approx(0.5 * width, rel=1e-3)  # K_I * width * tanh(inf)


# ----------------------------------------------------------------------------- guarantees


def _stream_run(name, scores_by_step, *, alpha=ALPHA, dev_scores=None, **kw):
    """Immediate exact feedback at every step; returns the miss indicators."""
    p = make_provider(
        name,
        alpha=alpha,
        development_scores=dev(2000) if dev_scores is None else dev_scores,
        seed=3,
        **kw,
    )
    misses = np.empty(len(scores_by_step), dtype=int)
    for t, r in enumerate(scores_by_step):
        iv = p.predict(est(), 1 - alpha)
        lab_ = exact_label(iv.lo, iv.hi, float(r), t)
        misses[t] = lab_.status is M
        p.observe([lab_])
    return misses, p


class TestAciGuarantee:
    def test_long_run_miss_frequency_is_near_alpha_on_a_shifted_stream(self):
        """Gibbs and Candes: |mean(err) - alpha| <= (max(alpha_1, 1-alpha_1) + gamma)/(gamma T)."""
        rng = np.random.default_rng(21)
        T = 12000
        stream = np.concatenate(
            [rng.standard_normal(T // 2), 1.5 + 2.0 * rng.standard_normal(T // 2)]
        )
        misses, _ = _stream_run("V7", stream, alpha=ALPHA)
        bound = (max(ALPHA, 1 - ALPHA) + GAMMA) / (GAMMA * T)
        assert abs(misses.mean() - ALPHA) <= bound
        assert abs(misses.mean() - ALPHA) < 0.01

    def test_split_does_not_hold_its_level_on_the_same_stream(self):
        rng = np.random.default_rng(21)
        T = 12000
        stream = np.concatenate(
            [rng.standard_normal(T // 2), 1.5 + 2.0 * rng.standard_normal(T // 2)]
        )
        misses, _ = _stream_run("V1", stream, alpha=ALPHA)
        assert misses[T // 2 :].mean() > 0.3  # the shift is real, and V1 never adapts to it

    def test_v3_and_v2_hold_their_level_when_every_label_is_exact(self):
        rng = np.random.default_rng(22)
        stream = np.concatenate([rng.standard_normal(3000), 2.0 * rng.standard_normal(3000)])
        for name in ("V2", "V3", "V4"):
            # exact labels, all audit: the three rules coincide with plain ACI
            p = make_provider(name, alpha=ALPHA, development_scores=dev(2000), seed=3)
            miss = 0
            for t, r in enumerate(stream):
                iv = p.predict(est(), LEVEL)
                lb = exact_label(iv.lo, iv.hi, float(r), t)
                lb = dataclasses.replace(lb, audit=True)
                miss += lb.status is M
                p.observe([lb])
            bound = (max(ALPHA, 1 - ALPHA) + GAMMA) / (GAMMA * len(stream))
            assert abs(miss / len(stream) - ALPHA) <= bound + 1e-12, name


class TestPidTracksAStep:
    def test_pid_returns_to_its_level_after_a_scale_step_and_split_never_does(self):
        rng = np.random.default_rng(31)
        pre, post = 2000, 4000
        stream = np.concatenate([rng.standard_normal(pre), 2.0 * rng.standard_normal(post)])

        def steps_to_track(misses):
            """First step after the shift from which the next 200 steps miss at most alpha+0.04."""
            for t in range(pre, len(misses) - 200):
                if misses[t : t + 200].mean() <= ALPHA + 0.04:
                    return t - pre
            return None

        pid, _ = _stream_run("V5", stream)
        split, _ = _stream_run("V1", stream)
        assert steps_to_track(split) is None
        t_pid = steps_to_track(pid)
        assert t_pid is not None and t_pid < 1500
        assert pid[:pre].mean() < ALPHA + 0.03


# ----------------------------------------------------------------------------- order, truth, seeds


class TestOrderAndHonesty:
    def test_labels_must_arrive_in_time_order(self):
        for name in VARIANTS:
            p = provider(name)
            truth = 0.0 if name == "V7" else None
            p.observe([lab(H, resolved=20, truth=truth)])
            p.observe([lab(H, resolved=20, truth=truth)])  # equal is fine
            with pytest.raises(ValueError, match="time order"):
                p.observe([lab(H, resolved=19, truth=truth)])

    def test_updates_happen_only_in_observe(self):
        p = provider("V3")
        s0 = dict(p.state())
        for _ in range(5):
            p.predict(est(), LEVEL)
        assert p.state() == s0

    def test_predict_is_pure(self):
        p = provider("V5")
        assert p.predict(est(m=30.0, sigma=3.0), LEVEL) == p.predict(est(m=30.0, sigma=3.0), LEVEL)

    @pytest.mark.parametrize("name", ["V1", "V2", "V3", "V4", "V5", "V6"])
    def test_the_sim_only_truth_is_never_read_except_by_the_oracle(self, name):
        rng = np.random.default_rng(5)
        with_truth, without = [], []
        for d in range(80):
            lo, hi = -1.0 - rng.random(), 1.0 + rng.random()
            end = 10.0 + d + rng.uniform(-3, 5)
            censored = bool(rng.random() < 0.5)
            status = label_status(lo, hi, 10 + d, end, censored)
            kw = dict(
                day=10 + d,
                lo=lo,
                hi=hi,
                end=end,
                censored=censored,
                audit=bool(d % 2),
                prior=bool(d % 3 == 0),
                resolved=10 + d,
            )
            with_truth.append(lab(status, truth=float(rng.uniform(-9, 9)), **kw))
            without.append(lab(status, truth=None, **kw))
        a, b = provider(name), provider(name)
        a.observe(with_truth)
        b.observe(without)
        assert a.state() == b.state()
        for status in ("estimated", "insufficient"):
            assert a.predict(est(m=20.0, sigma=3.0, status=status), a_level(name)) == b.predict(
                est(m=20.0, sigma=3.0, status=status), a_level(name)
            )

    def test_the_oracle_does_use_the_truth(self):
        a, b = provider("V7"), provider("V7")
        a.observe([lab(H, truth=0.0, resolved=11)])
        b.observe([lab(H, truth=99.0, resolved=11)])
        assert a.state() != b.state()


def a_level(name):
    return 1 - (ALPHA_LIFE if name == "V6" else ALPHA)


class TestDeterminism:
    @pytest.mark.parametrize("name", VARIANTS)
    def test_same_seed_same_providers(self, name):
        stream = [
            lab(
                M if k % 3 else H,
                resolved=11 + k,
                end=11.0 + k,
                audit=bool(k % 2),
                truth=0.2 * k - 3.0,
                hi=0.5,
                lo=-0.5,
            )
            for k in range(30)
        ]
        a, b = provider(name, scores=dev(2000)), provider(name, scores=dev(2000))
        a.observe(stream)
        b.observe(stream)
        assert a.state() == b.state()
        level = a_level(name)
        assert a.predict(est(m=5.0), level) == b.predict(est(m=5.0), level)

    def test_the_initial_pool_depends_on_the_seed_when_the_development_set_is_large(self):
        a = provider("V3", scores=dev(3000), seed=1).predict(est(), LEVEL)
        b = provider("V3", scores=dev(3000), seed=2).predict(est(), LEVEL)
        c = provider("V3", scores=dev(3000), seed=1).predict(est(), LEVEL)
        assert a == c and a != b

    def test_state_is_logged_for_every_variant(self):
        for name in VARIANTS:
            s = provider(name).state()
            assert s["alpha"] > 0.0 and s["n_labels"] == 0.0
        assert "alpha_t_est" in provider("V2").state()
        assert "offset_est" in provider("V5").state()
        s6 = provider("V6").state()
        for key in ("q_plus_estimated", "q_minus_estimated", "q_plus_prior", "q_minus_prior"):
            assert key in s6
        assert "q_plus" not in s6 and "q_minus" not in s6  # the pooled pair is gone (0003 A18)


class TestAuditAssignment:
    def test_deterministic_and_independent_of_the_asset_and_the_life(self):
        assert is_audit_life(5, "PU-201", 0) == is_audit_life(5, "PU-201", 0)
        draws = {(a, k): is_audit_life(5, a, k) for a in ("PU-201", "PU-202") for k in range(200)}
        assert draws[("PU-201", 3)] == is_audit_life(5, "PU-201", 3)
        same = sum(draws[("PU-201", k)] == draws[("PU-202", k)] for k in range(200))
        assert same < 195  # the two assets do not share draws

    def test_the_rate_is_rho(self):
        n = 6000
        k = sum(is_audit_life(s, "CR-501", 0) for s in range(n))
        lo, hi = wilson(k, n)
        assert lo <= AUDIT_RHO <= hi

    def test_rho_zero_and_one(self):
        assert not any(is_audit_life(s, "CR-501", 0, 0.0) for s in range(200))
        assert all(is_audit_life(s, "CR-501", 0, 1.0) for s in range(200))
        with pytest.raises(ValueError):
            is_audit_life(0, "CR-501", 0, 1.5)


# ----------------------------------------------------------------------------- development data


@pytest.fixture(scope="module")
def calib():
    return development_data(P, range(2000, 2030))


@pytest.fixture(scope="module")
def lives_by_seed():
    return {s: open_loop_lives(P, s) for s in range(1000, 1020)}


class TestDevelopmentData:
    def test_shapes_and_counts(self, calib):
        assert calib.scores.ndim == 1 and calib.prior_scores.ndim == 1
        assert calib.life_scores.shape == (4 * 30, 4)  # estimated pair, then prior-class pair (A18)
        assert np.all(np.isfinite(calib.scores)) and np.all(np.isfinite(calib.prior_scores))
        n_cp = sum(len(x.days) for s in range(2000, 2030) for x in open_loop_lives(P, s))
        assert len(calib.scores) + len(calib.prior_scores) == n_cp
        assert calib.seeds == tuple(range(2000, 2030))

    def test_life_maxima_bound_the_scores_of_their_life_class_by_class(self, calib):
        ls = calib.life_scores
        assert not np.any(np.isposinf(ls)) and not np.any(np.isnan(ls))
        assert ls[:, 0].max() == pytest.approx(calib.scores.max())
        assert ls[:, 1].max() == pytest.approx((-calib.scores).max())
        assert ls[:, 2].max() == pytest.approx(calib.prior_scores.max())
        assert ls[:, 3].max() == pytest.approx((-calib.prior_scores).max())

    def test_a_life_without_a_class_holds_minus_inf_there_and_every_life_has_one_class(self, calib):
        ls = calib.life_scores
        no_est, no_prior = np.isneginf(ls[:, 0]), np.isneginf(ls[:, 2])
        assert np.array_equal(no_est, np.isneginf(ls[:, 1]))  # r and -r are -inf together
        assert np.array_equal(no_prior, np.isneginf(ls[:, 3]))
        assert not np.any(no_est & no_prior)  # a life with checkpoints has at least one class

    def test_the_columns_are_the_maxima_of_the_lives_own_scores(self):
        data = development_data(P, [2000])
        lives = [x for x in open_loop_lives(P, 2000) if len(x.outputs)]
        assert data.life_scores.shape == (len(lives), 4)
        for row, x in zip(data.life_scores, lives, strict=True):
            r = np.array(
                [
                    score(float(t), o.m_days, o.sigma_days)
                    for o, t in zip(x.outputs, x.true_rul, strict=True)
                ]
            )
            is_est = np.array([o.status == "estimated" for o in x.outputs])
            for col, (sel, sign) in enumerate(
                [(is_est, 1), (is_est, -1), (~is_est, 1), (~is_est, -1)]
            ):
                want = (sign * r[sel]).max() if sel.any() else -np.inf
                assert row[col] == pytest.approx(want), col

    def test_scores_follow_the_definition(self):
        data = development_data(P, [2000])
        lives = open_loop_lives(P, 2000)
        r = [
            score(float(t), o.m_days, o.sigma_days)
            for x in lives
            for o, t in zip(x.outputs, x.true_rul, strict=True)
        ]
        assert sorted(r) == pytest.approx(sorted(np.concatenate([data.scores, data.prior_scores])))

    def test_only_dev_and_calib_seeds_are_accepted(self):
        for bad in ([999], [3000], [3500], [2**40], [0], [1000, 4000]):
            with pytest.raises(ValueError, match="dev .* and calib"):
                development_data(P, bad)
        development_data(P, [1000])
        development_data(P, [2999])

    def test_provider_helper_passes_the_right_arrays(self, calib):
        v1 = calib.provider("V1", alpha=0.10, seed=0)
        v6 = calib.provider("V6", alpha=0.05, seed=0)
        assert v1.state()["pool_estimated"] == len(calib.scores)
        assert v1.state()["pool_prior"] == len(calib.prior_scores)
        assert v6.state()["n_dev_lives"] == 120

    def test_no_dev_data_means_an_error(self):
        with pytest.raises(ValueError, match="no estimated|prior"):
            development_data(P, [])


# ----------------------------------------------------------------------------- the open-loop replay


class _Spy:
    """Records the order of calls and what the labels carried."""

    name = "V3"

    def __init__(self, name="V3"):
        self.name = name
        self.events: list[tuple] = []

    def predict(self, est_out, level):
        self.events.append(("predict", est_out.day))
        return RulInterval(est_out.m_days - 5.0, est_out.m_days, est_out.m_days + 5.0, level, False)

    def observe(self, labels):
        self.events.append(
            (
                "observe",
                labels[0].resolved_day,
                len(labels),
                any(lab.true_rul is not None for lab in labels),
            )
        )

    def state(self):
        return {}


class TestReplay:
    def test_labels_are_delivered_when_they_are_visible_and_before_that_days_predictions(
        self, lives_by_seed
    ):
        lives = lives_by_seed[1000]
        spy = _Spy()
        cps, labels = replay_open_loop(lives, spy, 0.9)
        observes = [e for e in spy.events if e[0] == "observe"]
        assert sum(e[2] for e in observes) == len(labels) == len(cps)
        days = [e[1] for e in observes]
        assert days == sorted(days)
        # the observe of day d comes before any predict of day d
        last_seen = -1
        for e in spy.events:
            if e[0] == "observe":
                last_seen = e[1]
            else:
                assert e[1] >= last_seen

    def test_non_oracle_providers_get_learner_views(self, lives_by_seed):
        lives = lives_by_seed[1001]
        spy = _Spy("V3")
        cps, labels = replay_open_loop(lives, spy, 0.9)
        assert all(not e[3] for e in spy.events if e[0] == "observe")
        assert all(lab.true_rul is not None for lab in labels)  # the caller still has the truth
        spy7 = _Spy("V7")
        replay_open_loop(lives, spy7, 0.9)
        assert all(e[3] for e in spy7.events if e[0] == "observe")

    def test_each_checkpoint_gets_exactly_one_label_of_the_right_kind(self, lives_by_seed):
        lives = lives_by_seed[1002]
        cps, labels = replay_open_loop(lives, _Spy(), 0.9)
        assert sorted((c.asset_id, c.day) for c in cps) == sorted(
            (x.asset_id, x.checkpoint_day) for x in labels
        )
        by_asset = {x.asset_id: x for x in lives}
        for lab_ in labels:
            x = by_asset[lab_.asset_id]
            assert lab_.censored == x.censored and lab_.end_day == x.end_day

    def test_one_plant_seed_only(self, lives_by_seed):
        mixed = lives_by_seed[1000][:2] + lives_by_seed[1001][2:]
        with pytest.raises(ValueError, match="one plant seed"):
            replay_open_loop(mixed, _Spy(), 0.9)

    def test_empty(self):
        assert replay_open_loop([], _Spy(), 0.9) == ([], [])

    def test_open_loop_coverage_of_split_is_near_nominal_on_other_seeds(self, calib, lives_by_seed):
        """Open-loop calibration carries over to other open-loop seeds: the trend-estimated
        checkpoints cover near 0.90 (a wide tolerance: 20 seeds, correlated days)."""
        cps = []
        for lives in lives_by_seed.values():
            p = calib.provider("V1", alpha=0.10, seed=1)  # a provider serves one episode
            c, _ = replay_open_loop(lives, p, 0.90)
            cps += c
        est_cov = checkpoint_coverage(cps, "estimated")
        assert 0.82 < est_cov < 0.97
        assert 0.0 < checkpoint_coverage(cps, "prior") <= 1.0

    def test_the_audit_flag_follows_is_audit_life(self, lives_by_seed):
        lives = lives_by_seed[1003]
        _, labels = replay_open_loop(lives, _Spy(), 0.9)
        for lab_ in labels:
            assert lab_.audit == is_audit_life(1003, lab_.asset_id, 0)


def test_the_documented_constants():
    assert GAMMA == 0.02 and WINDOW == 500 and AUDIT_RHO == 0.10 and ALPHA_LIFE == 0.05
    assert DEV_SEEDS.start == 1000


# ----------------------------------------------------------------------------- V6, two classes


class TestLifeBandTakesTheTwoClasses:
    """0003 A18: V6 splits on ``prior`` like every other variant, four maxima at tail alpha / 4."""

    def band(self, scores, alpha=ALPHA_LIFE):
        return make_provider("V6", alpha=alpha, development_scores=scores, seed=0)

    def test_the_prior_pair_for_a_prior_output_and_the_estimated_pair_otherwise(self):
        # estimated scores near 0 and prior scores shifted by +5: the classes cannot be confused
        scores = life_maxima(np.random.default_rng(3), 300, 10, 10, prior_shift=5.0)
        p = self.band(scores)
        s = p.state()
        est_iv = p.predict(est(status="estimated"), 0.95)
        assert (est_iv.lo, est_iv.hi) == (-s["q_minus_estimated"], s["q_plus_estimated"])
        for status in ("prior", "insufficient", "not_degrading"):
            iv = p.predict(est(status=status), 0.95)
            assert (iv.lo, iv.hi) == (-s["q_minus_prior"], s["q_plus_prior"]), status
            assert iv.prior is True
        assert s["q_plus_prior"] > s["q_plus_estimated"] + 3.0  # the shift is in the prior band
        assert -s["q_minus_prior"] > -s["q_minus_estimated"] + 3.0

    def test_every_column_uses_all_the_lives_and_a_life_without_the_class_holds_minus_inf(self):
        n = 200
        scores = np.zeros((n, 4))
        scores[:, 0] = np.arange(1.0, n + 1)  # estimated max r: 1..200
        scores[:, 1] = np.arange(1.0, n + 1)
        scores[:100, 2:] = -np.inf  # half the lives have no prior checkpoint
        scores[100:, 2] = np.arange(1.0, 101)
        scores[100:, 3] = np.arange(1.0, 101)
        s = self.band(scores).state()
        tail = ALPHA_LIFE / 4.0
        assert tail == 0.0125
        # rank ceil((n + 1) (1 - tail)) = 199 of all 200 lives, -inf ones included
        assert s["q_plus_estimated"] == 199.0 and s["q_minus_estimated"] == 199.0
        # the prior column: the 199th smallest of 100 x -inf then 1..100 is 99, not the 100th
        # of the 100 finite ones (which a column using only its own lives would give)
        assert s["q_plus_prior"] == 99.0 and s["q_minus_prior"] == 99.0
        assert s["n_dev_lives"] == 200.0

    def test_the_tail_is_alpha_over_four_not_alpha_over_two(self):
        n = 400
        col = np.arange(1.0, n + 1)
        scores = np.column_stack([col, col, col, col])
        s = self.band(scores, alpha=0.08).state()
        # tail 0.02: rank ceil(401 x 0.98) = 393; alpha / 2 = 0.04 would give 385
        assert s["q_plus_estimated"] == 393.0

    def test_minus_inf_is_allowed_and_plus_inf_and_nan_are_not(self):
        ok = life_maxima(np.random.default_rng(0), 60, 4, 4)
        ok[7, 2:] = -np.inf
        self.band(ok)
        for bad in (np.inf, np.nan):
            arr = ok.copy()
            arr[0, 3] = bad
            with pytest.raises(ValueError, match=r"NaN or a \+inf"):
                self.band(arr)

    def test_a_new_life_leaves_its_band_anywhere_with_probability_at_most_alpha_life(self):
        """The union bound on exchangeable lives, at the registered level, with lives of random
        length, some without any estimated or any prior checkpoint."""
        rng = np.random.default_rng(2026)
        n_dev, reps, out = 300, 2500, 0

        def life():
            n_e, n_p = int(rng.integers(0, 15)), int(rng.integers(0, 15))
            # prior-class scores run wider than the estimated ones, as they do in the estimator
            return rng.standard_normal(n_e), 1.8 * rng.standard_normal(n_p) + 0.5

        def maxima(e, p):
            row = []
            for x in (e, p):
                row += [x.max(), (-x).max()] if x.size else [-np.inf, -np.inf]
            return row

        for _ in range(reps):
            scores = np.array([maxima(*life()) for _ in range(n_dev)])
            s = self.band(scores).state()
            e, p = life()
            hit = e.size and (e.max() > s["q_plus_estimated"] or -e.min() > s["q_minus_estimated"])
            hit = hit or (p.size and (p.max() > s["q_plus_prior"] or -p.min() > s["q_minus_prior"]))
            out += bool(hit)
        assert out / reps <= ALPHA_LIFE, f"{out} of {reps} new lives left the band"
        assert out > 0  # and the experiment can fail: the band is not infinite

    def test_the_pooled_band_it_replaces_could_not_vouch_for_a_prior_output(self):
        # prior scores near -3, estimated scores near +2: one pooled pair would carry the estimated
        # tail into the prior band; the two classes keep them apart
        rng = np.random.default_rng(5)
        scores = life_maxima(rng, 300, 10, 10, prior_shift=-3.0)
        s = self.band(scores).state()
        pooled_plus = max(s["q_plus_estimated"], s["q_plus_prior"])
        assert s["q_plus_prior"] < pooled_plus  # the prior band is narrower than a pooled one
