"""P1: betting confidence sequences. Coverage is checked by simulation, not by argument."""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from closed_loop.stats.confseq import (
    BET_CAP,
    SPREAD_KURTOSIS_DIVISOR,
    SPREAD_MIN_PAIRS,
    SPREAD_R,
    CSAccumulator,
    _predictable_bets,
    _spread_halfwidth,
    betting_cs,
    paired_betting_cs,
    paired_cs,
    zero_event_upper,
)
from closed_loop.stats.intervals import wilson

ALPHA = 0.05


def _ever_excludes(streams: np.ndarray, mean: float, alpha: float = ALPHA) -> np.ndarray:
    """For each stream, does the hedged capital at the true mean ever reach 1/alpha?

    Uses the same bets as ``betting_cs`` but evaluates the capital only at ``mean``, which makes
    10,000 streams affordable. ``betting_cs`` itself is exercised on fewer streams below.
    """
    lam = _predictable_bets(streams, alpha, BET_CAP)
    step = lam * (streams - mean)
    log_up = np.cumsum(np.log1p(step), axis=-1)
    log_down = np.cumsum(np.log1p(-step), axis=-1)
    log_k = np.logaddexp(log_up, log_down) - math.log(2.0)
    return (log_k >= -math.log(alpha)).any(axis=-1)


class TestCoverage:
    @pytest.mark.parametrize("mean", [0.01, 0.1, 0.5])
    def test_capital_at_true_mean_rarely_reaches_threshold(self, mean: float) -> None:
        """10,000 Bernoulli streams of length 500: at most 5% ever exclude the true mean."""
        rng = np.random.default_rng(20261031)
        bad = 0
        total = 10_000
        for _ in range(10):
            streams = (rng.random((total // 10, 500)) < mean).astype(np.float64)
            bad += int(_ever_excludes(streams, mean).sum())
        lo, hi = wilson(bad, total)
        assert bad / total <= 0.05, f"miscoverage {bad}/{total} (Wilson {lo:.4f}-{hi:.4f})"

    @pytest.mark.parametrize("mean", [0.01, 0.1, 0.5])
    def test_public_sequence_contains_true_mean_at_every_prefix(self, mean: float) -> None:
        rng = np.random.default_rng(7)
        bad = 0
        total = 80
        for _ in range(total):
            x = (rng.random(300) < mean).astype(np.float64)
            lower, upper = betting_cs(x, ALPHA)
            if np.any(lower > mean) or np.any(upper < mean):
                bad += 1
        assert bad / total <= 0.05

    def test_the_sequence_is_not_vacuous(self) -> None:
        """A stream of ones pins the lower bound near 1; a stream of zeros pins the upper to 0."""
        lower, upper = betting_cs(np.ones(400))
        assert lower[-1] > 0.9 and upper[-1] == 1.0
        lower, upper = betting_cs(np.zeros(400))
        assert lower[-1] == 0.0 and upper[-1] < 0.05

    def test_a_wrong_mean_is_excluded(self) -> None:
        rng = np.random.default_rng(3)
        x = (rng.random(2000) < 0.3).astype(np.float64)
        lower, upper = betting_cs(x)
        assert lower[-1] > 0.25 and upper[-1] < 0.35
        assert not lower[-1] <= 0.5 <= upper[-1]

    def test_smaller_alpha_gives_a_wider_set(self) -> None:
        x = (np.random.default_rng(5).random(500) < 0.4).astype(np.float64)
        lo5, hi5 = betting_cs(x, 0.05)
        lo1, hi1 = betting_cs(x, 0.01)
        assert hi1[-1] - lo1[-1] > hi5[-1] - lo5[-1]


class TestWidth:
    def test_width_shrinks_like_inverse_sqrt_n(self) -> None:
        """The width times sqrt(n) stays within a factor of two over a 32-fold range of n.

        A betting sequence pays an iterated-logarithm factor, so exact constancy is not expected.
        """
        rng = np.random.default_rng(11)
        sizes = (100, 400, 1600, 3200)
        widths_by_n = []
        for n in sizes:
            widths = []
            for _ in range(4):
                x = (rng.random(n) < 0.5).astype(np.float64)
                lo, hi = betting_cs(x)
                widths.append(hi[-1] - lo[-1])
            widths_by_n.append(float(np.mean(widths)))
        scaled = [w * math.sqrt(n) for w, n in zip(widths_by_n, sizes, strict=True)]
        assert max(scaled) / min(scaled) < 2.0, scaled
        # and it really does shrink: 32 times the data gives well under a third of the width
        assert widths_by_n[-1] < widths_by_n[0] / 3.0


class TestZeroEvents:
    def test_compared_with_the_rule_of_three(self) -> None:
        """Anytime validity costs a factor of about 2.5 over the fixed-n rule of three at n = 2000.

        The value is reported in docs/adr/progress/P1-statistics.md; the bounds here only keep it
        honest in both directions: never below the exact binomial bound, never wildly above.
        """
        n = 2000
        value = zero_event_upper(n)
        exact_binomial = 1.0 - ALPHA ** (1.0 / n)
        rule_of_three = 3.0 / n
        assert value > exact_binomial > 0.0
        assert abs(exact_binomial - rule_of_three) / rule_of_three < 0.01
        assert value < 3.0 * rule_of_three

    def test_equals_the_sequence_on_a_run_of_zeros(self) -> None:
        for n in (50, 200):
            _, upper = betting_cs(np.zeros(n), ALPHA, 100_001)
            assert zero_event_upper(n) == pytest.approx(upper[-1], abs=2e-4)

    def test_decreases_with_n_and_edge_cases(self) -> None:
        values = [zero_event_upper(n) for n in (0, 1, 10, 100, 1000)]
        assert values[0] == 1.0
        assert all(a >= b for a, b in zip(values, values[1:], strict=False))
        assert values[-1] < 0.01

    def test_refuses_bad_arguments(self) -> None:
        for bad in (-1, 2.5, True):
            with pytest.raises(ValueError):
                zero_event_upper(bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            zero_event_upper(10, alpha=1.5)


class TestRunningIntersection:
    def test_accumulator_is_the_running_intersection_of_the_raw_sets(self) -> None:
        rng = np.random.default_rng(13)
        x = (rng.random(400) < 0.2).astype(np.float64)
        raw_lo, raw_hi = betting_cs(x, ALPHA, 1001, intersect=False)
        acc = CSAccumulator(ALPHA, 1001)
        assert acc.bounds() == (0.0, 1.0) and acc.n == 0
        for i, v in enumerate(x):
            acc.update(float(v))
            lo, hi = acc.bounds()
            assert lo == pytest.approx(raw_lo[: i + 1].max(), abs=1e-12)
            assert hi == pytest.approx(raw_hi[: i + 1].min(), abs=1e-12)
        assert acc.n == 400

    def test_accumulator_matches_betting_cs_at_every_prefix(self) -> None:
        x = np.random.default_rng(17).random(250)
        lo, hi = betting_cs(x)
        acc = CSAccumulator()
        for i, v in enumerate(x):
            acc.update(float(v))
            assert acc.bounds() == pytest.approx((lo[i], hi[i]), abs=1e-12)

    def test_the_raw_sequence_is_not_always_nested_but_the_intersection_is(self) -> None:
        # a burst after a long quiet run widens the raw upper bound again at some prefix
        x = np.concatenate([np.zeros(300), np.ones(30), np.zeros(300)])
        raw_lo, raw_hi = betting_cs(x, ALPHA, 1001, intersect=False)
        lo, hi = betting_cs(x)
        assert np.any(np.diff(raw_hi) > 0) or np.any(np.diff(raw_lo) < 0)
        assert np.all(np.diff(lo) >= 0) and np.all(np.diff(hi) <= 0)

    @given(st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=40))
    def test_bounds_are_ordered_inside_the_unit_interval(self, xs: list[float]) -> None:
        lo, hi = betting_cs(xs, ALPHA, 101)
        assert np.all((0.0 <= lo) & (hi <= 1.0))
        assert np.all(lo <= hi)

    @given(st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=40))
    def test_intersection_never_widens(self, xs: list[float]) -> None:
        lo, hi = betting_cs(xs, ALPHA, 101)
        raw_lo, raw_hi = betting_cs(xs, ALPHA, 101, intersect=False)
        assert np.all(lo >= raw_lo) and np.all(hi <= raw_hi)


class TestBets:
    def test_bets_are_capped_and_positive(self) -> None:
        x = np.random.default_rng(1).random(300)
        lam = _predictable_bets(x, ALPHA, BET_CAP)
        assert np.all(lam > 0) and np.all(lam <= BET_CAP)
        assert lam.max() == BET_CAP  # the cap binds early, on short noisy history

    def test_bets_are_predictable(self) -> None:
        """The bet at step t uses only x[:t-1] (0-based: x[: t-1] for the t-th bet)."""
        rng = np.random.default_rng(2)
        x = rng.random(60)
        lam = _predictable_bets(x, ALPHA, 0.9)
        for t in (1, 5, 30, 59):
            changed = x.copy()
            changed[t:] = rng.random(60 - t)
            assert _predictable_bets(changed, ALPHA, 0.9)[: t + 1] == pytest.approx(lam[: t + 1])

    def test_bets_shrink_on_noisy_streams_and_grow_on_quiet_ones(self) -> None:
        quiet = _predictable_bets(np.full(500, 0.3), 0.05, 0.9)
        noisy = _predictable_bets((np.arange(500) % 2).astype(np.float64), 0.05, 0.9)
        assert quiet[-1] > noisy[-1]


class TestRefusals:
    @pytest.mark.parametrize("bad", [-0.001, 1.001, float("nan"), float("inf"), -float("inf")])
    def test_values_outside_the_unit_interval_are_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError, match="in \\[0, 1\\]"):
            betting_cs([0.5, bad, 0.5])
        with pytest.raises(ValueError, match="in \\[0, 1\\]"):
            CSAccumulator().update(bad)

    def test_a_rejected_update_leaves_the_state_alone(self) -> None:
        acc = CSAccumulator()
        acc.update(0.5)
        before = (acc.n, acc.bounds())
        with pytest.raises(ValueError):
            acc.update(2.0)
        assert (acc.n, acc.bounds()) == before

    def test_bad_parameters(self) -> None:
        for alpha in (0.0, 1.0, -0.1, 2.0):
            with pytest.raises(ValueError, match="alpha"):
                betting_cs([0.5], alpha)
            with pytest.raises(ValueError, match="alpha"):
                CSAccumulator(alpha)
        for grid in (2, 0, True):
            with pytest.raises(ValueError, match="grid"):
                betting_cs([0.5], grid=grid)
        with pytest.raises(ValueError, match="cap"):
            betting_cs([0.5], c=1.0)
        with pytest.raises(ValueError, match="one-dimensional"):
            betting_cs(np.zeros((2, 2)))

    def test_empty_input_gives_empty_bounds(self) -> None:
        lo, hi = betting_cs([])
        assert lo.shape == hi.shape == (0,)

    def test_boolean_values_are_accepted_as_zero_and_one(self) -> None:
        acc = CSAccumulator()
        acc.update(True)
        acc.update(False)
        assert acc.n == 2


BOTH = pytest.mark.parametrize("fn", [paired_cs, paired_betting_cs], ids=["spread", "betting"])


class TestPaired:
    """Written for the betting sequence, which is ``paired_betting_cs`` since 0003 A16. Every
    behaviour is checked for it and for ``paired_cs``, which falls back to it."""

    @BOTH
    def test_known_difference_is_bracketed_on_the_original_scale(self, fn) -> None:
        rng = np.random.default_rng(23)
        a = np.clip(rng.normal(60.0, 5.0, 1500), 0.0, 100.0)
        b = np.clip(a - 10.0 + rng.normal(0.0, 3.0, 1500), 0.0, 100.0)
        lo, hi = fn(a, b, 0.0, 100.0)
        truth = float(np.mean(a - b))
        assert lo <= truth <= hi
        assert -100.0 <= lo < hi <= 100.0
        assert hi - lo < 10.0  # 1,500 pairs pin a 10-unit gap to well under that width

    @BOTH
    def test_swapping_the_arms_negates_and_reverses_the_bounds(self, fn) -> None:
        rng = np.random.default_rng(29)
        a = rng.random(200)
        b = rng.random(200) * 0.5
        lo, hi = fn(a, b, 0.0, 1.0)
        lo2, hi2 = fn(b, a, 0.0, 1.0)
        assert (lo2, hi2) == pytest.approx((-hi, -lo), abs=2.1e-3)  # one grid cell of the rescale

    @BOTH
    def test_affine_change_of_units_scales_the_bounds_exactly(self, fn) -> None:
        rng = np.random.default_rng(31)
        a = rng.integers(0, 9, 100) / 8.0
        b = rng.integers(0, 9, 100) / 8.0
        lo, hi = fn(a, b, 0.0, 1.0)
        lo2, hi2 = fn(a * 4.0 + 3.0, b * 4.0 + 3.0, 3.0, 7.0)
        assert (lo2, hi2) == pytest.approx((4.0 * lo, 4.0 * hi), abs=1e-9)

    @BOTH
    def test_paired_coverage_of_a_true_difference(self, fn) -> None:
        rng = np.random.default_rng(37)
        bad = 0
        for _ in range(60):
            a = (rng.random(200) < 0.30).astype(np.float64)
            b = (rng.random(200) < 0.10).astype(np.float64)
            lo, hi = fn(a, b, 0.0, 1.0)
            bad += not (lo <= 0.20 <= hi)
        assert bad / 60 <= 0.05

    @BOTH
    def test_empty_gives_the_whole_range_and_bad_input_is_refused(self, fn) -> None:
        assert fn(np.array([]), np.array([]), 0.0, 10.0) == (-10.0, 10.0)
        with pytest.raises(ValueError, match="lo < hi"):
            fn(np.array([1.0]), np.array([1.0]), 2.0, 2.0)
        with pytest.raises(ValueError, match="equal in shape"):
            fn(np.array([1.0, 2.0]), np.array([1.0]), 0.0, 5.0)
        with pytest.raises(ValueError, match="within"):
            fn(np.array([6.0]), np.array([1.0]), 0.0, 5.0)
        with pytest.raises(ValueError, match="within"):
            fn(np.array([1.0]), np.array([float("nan")]), 0.0, 5.0)

    @BOTH
    def test_a_bad_level_is_refused_at_any_size(self, fn) -> None:
        # 300 pairs take the spread route for paired_cs: the level is checked there as well
        a = np.random.default_rng(41).random(300)
        for alpha in (0.0, 1.0, -0.1):
            with pytest.raises(ValueError, match="alpha"):
                fn(a, a[::-1].copy(), 0.0, 1.0, alpha)

    @BOTH
    @given(
        st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=0, max_size=150),
        st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=150, max_size=150),
    )
    def test_the_bounds_are_ordered_and_inside_the_range_of_the_difference(
        self, fn, xs: list[float], ys: list[float]
    ) -> None:
        a = np.array(xs + ys)
        b = np.array(ys + xs)
        lo, hi = fn(a, b, 0.0, 1.0, 0.05)
        assert -1.0 <= lo <= hi <= 1.0


# ----------------------------------------------------------------------------- 0003 A16


def reference_halfwidth(n: int, s: float, alpha: float) -> float:
    """Theorem 2.2 written out again, with ``r`` computed from its definition (``a = 0.05 / 9``)."""
    a = 0.05 / 9
    la = -2.0 * math.log(a)
    r = (la + math.log(la + 1.0)) / 300.0
    return s * math.sqrt(
        2.0 * (n * r + 1.0) / (n * n * r) * math.log(math.sqrt(n * r + 1.0) / alpha)
    )


def lumpy(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Two arms on [0, 1] whose difference is uniform on [-0.3, 0.3]: light tails (kurtosis 1.8, so
    the guard lets the spread sequence through from 100 pairs, whose limit is 2) and a spread."""
    d = rng.uniform(-0.3, 0.3, n)
    a = 0.5 + d / 2.0 + 0.04
    b = 0.5 - d / 2.0
    return a, b


class TestSpreadSequence:
    def test_the_registered_constants(self) -> None:
        assert SPREAD_MIN_PAIRS == 100 and SPREAD_KURTOSIS_DIVISOR == 50
        la = -2.0 * math.log(0.05 / 9)
        assert SPREAD_R == pytest.approx((la + math.log(la + 1.0)) / 300.0, rel=1e-13)
        assert SPREAD_R == pytest.approx(0.0427276355, abs=1e-10)

    def test_it_is_the_formula_of_the_theorem_with_the_sample_sd_and_the_mean(self) -> None:
        a, b = lumpy(np.random.default_rng(1), 300)
        d = a - b
        lo, hi = paired_cs(a, b, 0.0, 1.0, 0.05 / 9)
        half = reference_halfwidth(300, float(np.std(d, ddof=1)), 0.05 / 9)
        assert lo == pytest.approx(float(d.mean()) - half, abs=1e-12)
        assert hi == pytest.approx(float(d.mean()) + half, abs=1e-12)
        assert (hi - lo) / 2.0 == pytest.approx(0.216233 * float(np.std(d, ddof=1)), rel=1e-5)

    def test_the_sample_sd_has_n_minus_one_in_its_denominator(self) -> None:
        # 100 pairs: ddof=0 would shrink the half-width by sqrt(99 / 100), half a percent
        a, b = lumpy(np.random.default_rng(2), 100)
        d = a - b
        lo, hi = paired_cs(a, b, 0.0, 1.0, 0.05)
        half = reference_halfwidth(100, math.sqrt(float(np.sum((d - d.mean()) ** 2)) / 99), 0.05)
        assert (hi - lo) / 2.0 == pytest.approx(half, rel=1e-12)
        assert (hi - lo) / 2.0 > reference_halfwidth(100, float(np.std(d)), 0.05)

    def test_the_level_enters_only_through_the_logarithm(self) -> None:
        # r is one constant (tuned to 0.05 / 9) whatever the level of the call
        a, b = lumpy(np.random.default_rng(3), 250)
        sd = float(np.std(a - b, ddof=1))
        for alpha in (0.05, 0.05 / 9, 0.001):
            lo, hi = paired_cs(a, b, 0.0, 1.0, alpha)
            assert (hi - lo) / 2.0 == pytest.approx(reference_halfwidth(250, sd, alpha), rel=1e-12)

    def test_the_array_form_and_the_scalar_form_agree(self) -> None:
        from closed_loop.eval.reachability import spread_halfwidths

        for n in (100, 300, 700):
            assert spread_halfwidths(0.37, n, 0.05 / 9) == pytest.approx(
                _spread_halfwidth(n, 0.37, 0.05 / 9), rel=1e-13
            )

    def test_the_prereg_factors_at_300_and_500(self) -> None:
        # docs/PREREGISTRATION.draft.md section 6: 0.216 s at n = 300 and 0.168 s at n = 500
        assert _spread_halfwidth(300, 1.0, 0.05 / 9) == pytest.approx(0.216, abs=5e-4)
        assert _spread_halfwidth(500, 1.0, 0.05 / 9) == pytest.approx(0.168, abs=5e-4)

    def test_swapping_the_arms_negates_the_bounds_exactly(self) -> None:
        a, b = lumpy(np.random.default_rng(4), 200)
        lo, hi = paired_cs(a, b, 0.0, 1.0)
        lo2, hi2 = paired_cs(b, a, 0.0, 1.0)
        assert (lo2, hi2) == pytest.approx((-hi, -lo), abs=1e-12)

    def test_the_result_does_not_depend_on_the_order_of_the_pairs(self) -> None:
        a, b = lumpy(np.random.default_rng(5), 200)
        order = np.random.default_rng(6).permutation(200)
        assert paired_cs(a, b, 0.0, 1.0) == pytest.approx(paired_cs(a[order], b[order], 0.0, 1.0))

    def test_it_is_clipped_to_the_range_of_a_difference(self) -> None:
        # a level of 1e-200 makes the half-width several times the sd; the bound stays inside
        # [-w, w]. 100 uniform values on [0.8, 1.0] against zero: mean 0.9, sd 0.058, half-width
        # about 0.2, which passes 1
        d = 0.8 + 0.2 * (np.arange(100) + 0.5) / 100
        lo, hi = paired_cs(d, np.zeros(100), 0.0, 1.0, 1e-200)
        assert hi == 1.0 and 0.0 < lo < 0.9
        lo, hi = paired_cs(np.zeros(100), d, 0.0, 1.0, 1e-200)
        assert lo == -1.0 and -0.9 < hi < 0.0

    # ---- the guard
    def test_under_100_pairs_it_is_the_betting_sequence(self) -> None:
        a, b = lumpy(np.random.default_rng(7), 99)
        assert paired_cs(a, b, 0.0, 1.0, 0.05 / 9) == paired_betting_cs(a, b, 0.0, 1.0, 0.05 / 9)

    def test_from_100_pairs_it_is_the_spread_sequence(self) -> None:
        a, b = lumpy(np.random.default_rng(7), 100)
        got = paired_cs(a, b, 0.0, 1.0, 0.05 / 9)
        assert got != paired_betting_cs(a, b, 0.0, 1.0, 0.05 / 9)
        assert got[1] - got[0] < 0.2  # the spread-scaled width, not the betting floor of 0.49

    @staticmethod
    def two_point(n: int, ones: int) -> tuple[np.ndarray, np.ndarray]:
        """Differences of ``ones`` ones and zeros: kurtosis ``n^2 / (ones (n - ones)) - 3``."""
        d = np.zeros(n)
        d[:ones] = 1.0
        return d, np.zeros(n)

    def test_kurtosis_at_or_under_n_over_50_uses_the_spread_sequence_and_over_it_does_not(
        self,
    ) -> None:
        # n = 100: the limit is 2. With 28 ones the kurtosis is 1.96, with 27 it is 2.07
        under = self.two_point(100, 28)
        over = self.two_point(100, 27)
        assert paired_cs(*under, 0.0, 1.0) != paired_betting_cs(*under, 0.0, 1.0)
        assert paired_cs(*over, 0.0, 1.0) == paired_betting_cs(*over, 0.0, 1.0)

    def test_kurtosis_exactly_at_the_limit_is_allowed(self) -> None:
        # n = 100 with 25 values at +1, 25 at -1 and 50 at 0: mean 0, m2 = m4 = 0.5, kurtosis
        # 0.5 / 0.25 = 2 exactly, which is n / 50
        d = np.concatenate([np.ones(25), -np.ones(25), np.zeros(50)])
        a, b = np.maximum(d, 0.0), np.maximum(-d, 0.0)
        lo, hi = paired_cs(a, b, 0.0, 1.0, 0.05)
        half = reference_halfwidth(100, math.sqrt(50.0 / 99.0), 0.05)
        assert (lo, hi) == pytest.approx((-half, half), abs=1e-12)
        assert (lo, hi) != paired_betting_cs(a, b, 0.0, 1.0, 0.05)

    def test_the_limit_grows_with_n(self) -> None:
        # 2% of the pairs are 8: kurtosis about 1/0.02 - 3 = 47, over the limit of 6 at n = 300
        # and 10 at n = 500, and under the limit of 100 only at n = 5,000
        rng = np.random.default_rng(8)
        for n, spread in ((300, False), (500, False), (5000, True)):
            d = 8.0 * (rng.random(n) < 0.02)
            d[:3] = 8.0  # at least a few events whatever the draw
            got = paired_cs(d / 8.0, np.zeros(n), 0.0, 1.0, 0.05 / 9)
            assert (got != paired_betting_cs(d / 8.0, np.zeros(n), 0.0, 1.0, 0.05 / 9)) is spread

    def test_the_rare_events_that_broke_the_spread_sequence_go_to_the_betting_sequence(
        self,
    ) -> None:
        # rate 0.02: kurtosis about 47 > 6 at n = 300 on almost every draw
        rng = np.random.default_rng(9)
        on_betting = 0
        for _ in range(40):
            d = (rng.random(300) < 0.02).astype(np.float64)
            if d.sum() == 0:
                continue
            on_betting += paired_cs(d, np.zeros(300), 0.0, 1.0, 0.05 / 9) == paired_betting_cs(
                d, np.zeros(300), 0.0, 1.0, 0.05 / 9
            )
        assert on_betting >= 35

    # ---- zero spread
    def test_identical_arms_have_the_floor_of_the_betting_sequence_not_a_point(self) -> None:
        for n, floor in ((300, 0.080), (500, 0.048), (700, 0.034)):
            z = np.zeros(n)
            lo, hi = paired_cs(z, z, 0.0, 1.0, 0.05 / 9)
            assert (lo, hi) == pytest.approx((-floor, floor), abs=5e-4)
            assert (lo, hi) == paired_betting_cs(z, z, 0.0, 1.0, 0.05 / 9)

    def test_all_ones_is_the_betting_sequence_too(self) -> None:
        # a always 1, b always 0: every difference is the top of the range, no spread
        o, z = np.ones(300), np.zeros(300)
        lo, hi = paired_cs(o, z, 0.0, 1.0, 0.05 / 9)
        assert (lo, hi) == pytest.approx((0.92, 1.0), abs=5e-4)
        assert (lo, hi) == paired_betting_cs(o, z, 0.0, 1.0, 0.05 / 9)

    def test_a_constant_difference_that_the_mean_cannot_reproduce_is_still_no_spread(self) -> None:
        # the mean of 300 copies of 0.1 - 0.3 may differ from it by an ulp, and the variance of
        # that is not exactly 0 (kurtosis 1): the rule must not read it as a spread
        a, b = np.full(300, 0.1), np.full(300, 0.3)
        got = paired_cs(a, b, 0.0, 1.0, 0.05 / 9)
        assert got == paired_betting_cs(a, b, 0.0, 1.0, 0.05 / 9)
        assert got[1] - got[0] > 0.1
        for value in (0.1, 0.7, 1.0 / 3.0, 0.123456789):
            c = np.full(300, value)
            z = np.zeros(300)
            assert paired_cs(c, z, 0.0, 1.0) == paired_betting_cs(c, z, 0.0, 1.0)

    def test_its_documentation_says_what_it_is(self) -> None:
        doc = paired_cs.__doc__ or ""
        for text in ("Theorem 2.2", "52(6)", "2613-2640", "2024", "asymptotic", "fixed constant"):
            assert text in doc, text
        assert "the paper" in doc and "betting" in doc  # the choice of r is not the paper's


class TestScaleOfTheDifferences:
    """Found by hypothesis in the P12d full run: one difference of 1.55e-91 among zeros had a
    fourth power of 0 and a squared variance of 0, and the kurtosis divided by it."""

    def test_a_difference_too_small_to_square_is_not_a_division_by_zero(self) -> None:
        a = np.zeros(150)
        a[0] = 1.55e-91
        got = paired_cs(a, np.zeros(150), 0.0, 1.0, 0.05)
        assert -1.0 <= got[0] <= got[1] <= 1.0
        assert got == paired_betting_cs(a, np.zeros(150), 0.0, 1.0, 0.05)  # kurtosis 150: the guard

    @pytest.mark.parametrize("scale", [1e-200, 1e-91, 1e-8, 1.0, 1e8, 1e150])
    def test_the_interval_scales_with_the_units(self, scale: float) -> None:
        rng = np.random.default_rng(12)
        d = rng.uniform(-0.3, 0.3, 300)
        a, b = 0.5 + d / 2.0, 0.5 - d / 2.0
        lo, hi = paired_cs(a, b, 0.0, 1.0, 0.05 / 9)
        lo2, hi2 = paired_cs(a * scale, b * scale, 0.0, scale, 0.05 / 9)
        assert (lo2, hi2) == pytest.approx((lo * scale, hi * scale), rel=1e-9, abs=1e-300)


class TestSpreadCoverage:
    """The spread-scaled sequence is asymptotic, so its coverage is checked by simulation, on the
    shapes of the throw-away script that chose the guard (a rate, a lump, a cost difference), with
    the real function at the looks of 100 to 500 pairs."""

    LOOKS = (100, 200, 300, 400, 500)
    SHAPES = {
        "binary 0.3": (lambda r, s: r.binomial(1, 0.3, s).astype(float), 0.3),
        "binary 0.5": (lambda r, s: r.binomial(1, 0.5, s).astype(float), 0.5),
        "lump 8 p.075 base 2": (
            lambda r, s: r.normal(0, 2.0, s) + 8.0 * r.binomial(1, 0.075, s),
            0.6,
        ),
        "normal": (lambda r, s: r.normal(-2.3, 3.0, s), -2.3),
    }

    @pytest.mark.slow
    @pytest.mark.parametrize("name", list(SHAPES))
    def test_the_real_function_keeps_the_true_mean_at_every_look(self, name: str) -> None:
        draw, mu = self.SHAPES[name]
        rng = np.random.default_rng(20261101)
        alpha = 0.05 / 9
        runs, bad = 300, 0
        for _ in range(runs):
            x = draw(rng, (500,))
            hit = False
            for t in self.LOOKS:
                # the lumpy shapes lie in [-20, 20] with the second arm at 0
                lo, hi = paired_cs(x[:t], np.zeros(t), -20.0, 20.0, alpha)
                hit |= not (lo <= mu <= hi)
            bad += hit
        # the miss of the whole rule is at most (the miss of the spread route) + alpha, about 0.006
        # a run at the level of the claim; 300 runs expect under two, and a seeded draw gives what
        # it gives. A route that was wrong by a constant factor would miss in tens of runs
        assert bad <= 3, f"{bad} of {runs} runs left the sequence at some look"

    def test_the_vectorised_simulation_is_the_real_function(self) -> None:
        # the simulation computes the same half-width and the same guard as paired_cs: on single
        # paths the two give the same answer, and where the guard fails it is the betting one
        from closed_loop.eval.reachability import spread_halfwidths

        rng = np.random.default_rng(10)
        routes = {"spread": 0, "betting": 0}
        for n in (100, 150, 300, 500):
            for p in (0.0, 0.02, 0.3, 0.5):  # a lump of 8 on a noisy base, from none to common
                x = rng.uniform(-2.0, 2.0, n) + 8.0 * (rng.random(n) < p)
                m1, m2 = x.mean(), (x**2).mean()
                m3, m4 = (x**3).mean(), (x**4).mean()
                var = max(m2 - m1**2, 0.0)
                kurt = (m4 - 4 * m1 * m3 + 6 * m1**2 * m2 - 3 * m1**4) / var**2
                zeros = np.zeros(n)
                got = paired_cs(x, zeros, -20.0, 20.0, 0.05 / 9)
                if kurt <= n / 50:
                    half = float(spread_halfwidths(np.sqrt(var * n / (n - 1)), n, 0.05 / 9))
                    assert (got[1] - got[0]) / 2.0 == pytest.approx(half, rel=1e-9)
                    routes["spread"] += 1
                else:
                    assert got == paired_betting_cs(x, zeros, -20.0, 20.0, 0.05 / 9)
                    routes["betting"] += 1
        assert routes["spread"] >= 3 and routes["betting"] >= 2, routes
