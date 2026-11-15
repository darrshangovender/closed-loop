"""P1: Wilson interval; P12c: the exact (Clopper-Pearson) interval."""

import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy import stats

from closed_loop.stats.intervals import clopper_pearson, wilson


class TestKnownValues:
    @pytest.mark.parametrize(
        ("k", "n", "lo", "hi"),
        [
            (0, 10, 0.0, 0.2775),
            (10, 10, 0.7225, 1.0),
            (5, 10, 0.2366, 0.7634),
            (1, 100, 0.0018, 0.0545),
            (50, 100, 0.4038, 0.5962),
        ],
    )
    def test_textbook_values(self, k: int, n: int, lo: float, hi: float) -> None:
        got = wilson(k, n)
        assert got[0] == pytest.approx(lo, abs=5e-4) and got[1] == pytest.approx(hi, abs=5e-4)

    def test_zero_trials_is_uninformative(self) -> None:
        assert wilson(0, 0) == (0.0, 1.0)

    def test_zero_successes_has_a_lower_bound_of_exactly_zero(self) -> None:
        assert wilson(0, 500)[0] == 0.0 and wilson(500, 500)[1] == 1.0

    def test_z_widens_the_interval(self) -> None:
        narrow, wide = wilson(30, 100, z=1.0), wilson(30, 100, z=2.576)
        assert wide[0] < narrow[0] < narrow[1] < wide[1]


class TestAgainstScipy:
    @given(
        st.integers(min_value=1, max_value=2000).flatmap(
            lambda n: st.tuples(st.integers(min_value=0, max_value=n), st.just(n))
        )
    )
    def test_matches_scipy_wilson(self, kn: tuple[int, int]) -> None:
        k, n = kn
        level = 2.0 * stats.norm.cdf(1.96) - 1.0
        ref = stats.binomtest(k, n).proportion_ci(confidence_level=level, method="wilson")
        lo, hi = wilson(k, n)
        assert lo == pytest.approx(ref.low, abs=1e-9) and hi == pytest.approx(ref.high, abs=1e-9)

    @given(
        st.integers(min_value=1, max_value=500).flatmap(
            lambda n: st.tuples(st.integers(min_value=0, max_value=n), st.just(n))
        )
    )
    def test_contains_the_point_estimate_and_stays_in_range(self, kn: tuple[int, int]) -> None:
        k, n = kn
        lo, hi = wilson(k, n)
        assert 0.0 <= lo <= k / n <= hi <= 1.0


class TestRefusals:
    @pytest.mark.parametrize("k, n", [(-1, 5), (6, 5), (1, -2)])
    def test_bad_counts(self, k: int, n: int) -> None:
        with pytest.raises(ValueError, match="0 <= k <= n"):
            wilson(k, n)

    @pytest.mark.parametrize("z", [0.0, -1.0, float("inf"), float("nan")])
    def test_bad_z(self, z: float) -> None:
        with pytest.raises(ValueError, match="z must"):
            wilson(1, 5, z=z)


class TestClopperPearson:
    """The exact interval C-SEARCH is decided on (0003 A16)."""

    @pytest.mark.parametrize(
        ("k", "n", "alpha", "lo", "hi"),
        [
            (0, 10, 0.05, 0.0, 0.3085),  # 1 - 0.025 ** (1 / 10)
            (10, 10, 0.05, 0.6915, 1.0),  # 0.025 ** (1 / 10)
            (5, 10, 0.05, 0.1871, 0.8129),
            (1, 100, 0.05, 0.0003, 0.0545),
            (10, 10, 0.05 / 9, 0.5551, 1.0),  # the clean sweep of ten pairs at the claim level
            (9, 10, 0.05 / 9, 0.4247, 0.9997),
        ],
    )
    def test_textbook_values(self, k: int, n: int, alpha: float, lo: float, hi: float) -> None:
        got = clopper_pearson(k, n, alpha)
        assert got[0] == pytest.approx(lo, abs=5e-4) and got[1] == pytest.approx(hi, abs=5e-4)

    def test_the_ends_are_exactly_zero_and_one(self) -> None:
        assert clopper_pearson(0, 500)[0] == 0.0 and clopper_pearson(500, 500)[1] == 1.0
        assert clopper_pearson(0, 1) == (0.0, pytest.approx(0.975))
        assert clopper_pearson(1, 1) == (pytest.approx(0.025), 1.0)

    def test_no_trials_is_uninformative(self) -> None:
        assert clopper_pearson(0, 0) == (0.0, 1.0)

    def test_the_default_level_is_five_percent_with_a_tail_of_2_5_percent_each_side(self) -> None:
        assert clopper_pearson(3, 20) == pytest.approx(clopper_pearson(3, 20, 0.05))
        lo, hi = clopper_pearson(3, 20, 0.05)
        assert lo == pytest.approx(stats.beta.ppf(0.025, 3, 18))
        assert hi == pytest.approx(stats.beta.ppf(0.975, 4, 17))

    def test_a_smaller_alpha_widens_it(self) -> None:
        narrow, wide = clopper_pearson(7, 20, 0.1), clopper_pearson(7, 20, 0.001)
        assert wide[0] < narrow[0] < narrow[1] < wide[1]

    def test_numpy_integers_are_accepted(self) -> None:
        import numpy as np

        assert clopper_pearson(np.int64(4), np.int64(10)) == clopper_pearson(4, 10)

    @given(
        st.integers(min_value=1, max_value=1500).flatmap(
            lambda n: st.tuples(st.integers(min_value=0, max_value=n), st.just(n))
        ),
        st.sampled_from([0.1, 0.05, 0.05 / 9, 0.01]),
    )
    def test_matches_scipys_exact_interval(self, kn: tuple[int, int], alpha: float) -> None:
        k, n = kn
        ref = stats.binomtest(k, n).proportion_ci(confidence_level=1.0 - alpha, method="exact")
        lo, hi = clopper_pearson(k, n, alpha)
        assert lo == pytest.approx(ref.low, abs=1e-9) and hi == pytest.approx(ref.high, abs=1e-9)

    @given(
        st.integers(min_value=1, max_value=400).flatmap(
            lambda n: st.tuples(st.integers(min_value=0, max_value=n), st.just(n))
        )
    )
    def test_contains_the_point_estimate_and_stays_in_range(self, kn: tuple[int, int]) -> None:
        k, n = kn
        lo, hi = clopper_pearson(k, n)
        assert 0.0 <= lo <= k / n <= hi <= 1.0

    @pytest.mark.parametrize("k, n", [(-1, 5), (6, 5), (1, -2)])
    def test_bad_counts(self, k: int, n: int) -> None:
        with pytest.raises(ValueError, match="0 <= k <= n"):
            clopper_pearson(k, n)

    @pytest.mark.parametrize("k, n", [(1.5, 5), (1, 5.0), (True, 5), ("1", 5)])
    def test_counts_must_be_integers(self, k: object, n: object) -> None:
        with pytest.raises(ValueError, match="must be an int"):
            clopper_pearson(k, n)  # type: ignore[arg-type]

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.05, 1.5, float("nan")])
    def test_bad_alpha(self, alpha: float) -> None:
        with pytest.raises(ValueError, match="alpha"):
            clopper_pearson(1, 5, alpha)
