"""P9a: uniform random search and the SearchRun record."""

import math

import numpy as np
import pytest
from scipy import stats

from closed_loop.search.random_search import SearchRun, random_search, summarise_run


def sphere(x: np.ndarray) -> float:
    return float(np.sum((x - 0.5) ** 2))


class TestSampling:
    def test_points_are_uniform_on_the_unit_box(self) -> None:
        seen = []
        random_search(lambda x: seen.append(x.copy()) or 0.0, 8, 4000, seed=1)
        pts = np.array(seen)
        assert pts.shape == (4000, 8)
        assert pts.min() >= 0.0 and pts.max() < 1.0
        # one KS test per coordinate, Bonferroni-corrected to a 1% family-wise level
        pvalues = [stats.kstest(pts[:, j], "uniform").pvalue for j in range(8)]
        assert min(pvalues) > 0.01 / 8, pvalues
        assert abs(pts.mean() - 0.5) < 0.01

    def test_coordinates_are_independent(self) -> None:
        seen = []
        random_search(lambda x: seen.append(x.copy()) or 0.0, 6, 3000, seed=2)
        corr = np.corrcoef(np.array(seen).T)
        off_diagonal = corr[~np.eye(6, dtype=bool)]
        assert np.abs(off_diagonal).max() < 0.08

    def test_same_seed_same_points_different_seed_different_points(self) -> None:
        def points(seed: int) -> np.ndarray:
            seen: list[np.ndarray] = []
            random_search(lambda x: seen.append(x.copy()) or 0.0, 4, 50, seed=seed)
            return np.array(seen)

        assert np.array_equal(points(3), points(3))
        assert not np.array_equal(points(3), points(4))

    def test_points_do_not_depend_on_the_objective(self) -> None:
        a: list[np.ndarray] = []
        b: list[np.ndarray] = []
        random_search(lambda x: a.append(x.copy()) or float(x[0]), 4, 30, seed=5)
        random_search(lambda x: b.append(x.copy()) or -float(x.sum()), 4, 30, seed=5)
        assert np.array_equal(np.array(a), np.array(b))

    def test_the_objective_cannot_corrupt_the_sampler_by_mutating_its_argument(self) -> None:
        def vandal(x: np.ndarray) -> float:
            x[:] = 99.0
            return 0.0

        run = random_search(vandal, 3, 20, seed=6)
        assert run.theta_best.max() < 1.0


class TestRunRecord:
    def test_best_is_the_minimum_and_theta_best_attains_it(self) -> None:
        run = random_search(sphere, 5, 300, seed=7)
        assert run.algo == "random" and run.evals == 300 and run.seed == 7
        assert run.theta_best.shape == (5,)
        assert sphere(run.theta_best) == run.best_f == run.history.min()

    def test_history_is_the_running_minimum(self) -> None:
        run = random_search(sphere, 5, 200, seed=8)
        assert run.history.shape == (200,)
        assert np.all(np.diff(run.history) <= 0.0)
        assert run.history[-1] == run.best_f

    def test_first_violation_is_the_one_based_index_of_the_first_negative(self) -> None:
        values = iter([3.0, 2.0, 0.0, -0.5, -9.0, 1.0])
        run = random_search(lambda x: next(values), 2, 6, seed=9)
        assert run.first_violation_eval == 4  # zero is not a violation; -0.5 is the 4th value
        assert run.best_f == -9.0

    def test_no_violation_is_none(self) -> None:
        run = random_search(lambda x: 1.0 + float(x[0]), 2, 50, seed=10)
        assert run.first_violation_eval is None

    def test_regime_label_is_carried(self) -> None:
        assert random_search(sphere, 2, 5, seed=1, regime="R_in").regime == "R_in"
        assert random_search(sphere, 2, 5, seed=1).regime == ""

    def test_ties_keep_the_first_point(self) -> None:
        run = random_search(lambda x: 1.0, 3, 10, seed=11)
        first = []
        random_search(lambda x: first.append(x.copy()) or 1.0, 3, 10, seed=11)
        assert np.array_equal(run.theta_best, first[0])

    def test_infinite_values_are_allowed(self) -> None:
        values = iter([math.inf, 2.0, -math.inf])
        run = random_search(lambda x: next(values), 2, 3, seed=1)
        assert run.best_f == -math.inf and run.first_violation_eval == 3

    def test_summarise_run_matches_the_arguments(self) -> None:
        pts = np.array([[0.1], [0.2], [0.3]])
        run = summarise_run("x", "R", 4, np.array([2.0, -1.0, 0.5]), pts)
        assert isinstance(run, SearchRun)
        assert (run.algo, run.regime, run.seed, run.evals) == ("x", "R", 4, 3)
        assert run.best_f == -1.0 and run.theta_best.tolist() == [0.2]
        assert run.first_violation_eval == 2

    def test_run_is_frozen(self) -> None:
        run = random_search(sphere, 2, 5, seed=1)
        with pytest.raises(AttributeError):
            run.best_f = 0.0  # type: ignore[misc]


class TestRefusals:
    def test_nan_from_the_objective(self) -> None:
        values = iter([1.0, float("nan")])
        with pytest.raises(ValueError, match="NaN at evaluation 2"):
            random_search(lambda x: next(values), 2, 5, seed=1)

    @pytest.mark.parametrize("dim, budget", [(0, 5), (3, 0), (-1, 5), (3, True), (2.5, 5)])
    def test_bad_dimension_or_budget(self, dim, budget) -> None:
        with pytest.raises(ValueError):
            random_search(sphere, dim, budget, seed=1)
