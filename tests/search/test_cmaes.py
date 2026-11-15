"""P9a: CMA-ES. Convergence budgets are the ones stated in docs/adr/0002-interfaces.md, P9a."""

import math

import numpy as np
import pytest

from closed_loop.search.cmaes import CMAES, cmaes_search


def sphere(x: np.ndarray) -> float:
    """Optimum 0 at 0.3 in every coordinate, inside the box."""
    return float(np.sum((x - 0.3) ** 2))


def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock on z = 4x - 2 in [-2, 2]^d: optimum 0 at x = 0.75, inside the box."""
    z = 4.0 * x - 2.0
    return float(np.sum(100.0 * (z[1:] - z[:-1] ** 2) ** 2 + (1.0 - z[:-1]) ** 2))


def drive(f, dim: int, seed: int, target: float, budget: int, **kw) -> tuple[CMAES, bool]:
    opt = CMAES(dim, seed, **kw)
    while opt.evals < budget and opt.best_f >= target:
        x = opt.ask()
        opt.tell(x, np.array([f(row) for row in x]))
    return opt, opt.best_f < target


class TestConvergence:
    def test_sphere_d10_reaches_1e_minus_8_within_6000_evaluations_on_all_10_seeds(self) -> None:
        results = [drive(sphere, 10, seed, 1e-8, 6000) for seed in range(10)]
        assert all(ok for _, ok in results), [o.evals for o, _ in results]

    def test_rosenbrock_d5_reaches_1e_minus_6_within_20000_on_at_least_8_of_10_seeds(self) -> None:
        results = [drive(rosenbrock, 5, seed, 1e-6, 20000) for seed in range(10)]
        reached = sum(ok for _, ok in results)
        assert reached >= 8, [(o.evals, o.best_f) for o, _ in results]

    def test_an_optimum_on_the_boundary_is_found_without_leaving_the_box(self) -> None:
        # f falls towards x0 = 0 and x1 = 1, the faces of the box
        def f(x: np.ndarray) -> float:
            return float(x[0] + (1.0 - x[1]) + np.sum((x[2:] - 0.5) ** 2))

        opt, _ = drive(f, 6, 3, -1.0, 4000)
        assert opt.best_f < 1e-4
        assert opt.best_x[0] < 1e-3 and opt.best_x[1] > 1 - 1e-3

    def test_an_optimum_outside_the_box_is_found_on_its_corner(self) -> None:
        def f(x: np.ndarray) -> float:
            return float(np.sum((x - 1.4) ** 2))

        opt, _ = drive(f, 4, 4, -1.0, 3000)
        assert np.all(opt.best_x > 0.999)
        assert opt.best_f == pytest.approx(4 * 0.4**2, abs=1e-3)

    def test_ill_conditioned_ellipsoid(self) -> None:
        weights = 10.0 ** np.linspace(0, 4, 6)

        def f(x: np.ndarray) -> float:
            return float(np.sum(weights * (x - 0.4) ** 2))

        _, ok = drive(f, 6, 7, 1e-8, 20000)
        assert ok

    def test_cmaes_search_finds_a_violation_of_a_shifted_objective(self) -> None:
        run = cmaes_search(lambda x: float(np.sum((x - 0.7) ** 2)) - 0.01, 8, 300, seed=1)
        assert run.first_violation_eval is not None and run.first_violation_eval <= 300


class TestBox:
    def test_every_asked_point_is_inside_the_box_even_when_the_objective_pulls_outwards(
        self,
    ) -> None:
        opt = CMAES(5, seed=1)
        for _ in range(150):
            x = opt.ask()
            assert x.min() >= 0.0 and x.max() <= 1.0
            opt.tell(x, -x.sum(axis=1) * 100.0 + x[:, 0] ** 2)  # downhill is out of the box

    def test_a_large_initial_step_still_stays_inside(self) -> None:
        opt = CMAES(4, seed=2, sigma0=5.0)
        for _ in range(30):
            x = opt.ask()
            assert x.min() >= 0.0 and x.max() <= 1.0
            opt.tell(x, np.array([sphere(r) for r in x]))

    def test_samples_outside_the_box_pay_a_penalty_in_the_ranking(self) -> None:
        opt = CMAES(2, seed=3)
        x = opt.ask()
        raw = opt._pending[2]  # type: ignore[index]
        assert (raw < 0).any() or (raw > 1).any(), "seed chosen so some samples fall outside"
        f = np.zeros(opt.popsize)  # a flat objective: only the penalty can separate the points
        ranked = opt._ranking_values(f, raw, x)
        outside = np.any((raw < 0) | (raw > 1), axis=1)
        assert ranked[outside].min() > ranked[~outside].max()


class TestContracts:
    def test_default_population_is_4_plus_floor_3_ln_dim(self) -> None:
        assert CMAES(32, 0).popsize == 14  # the 32-dimensional scenario vector of 0001, section 9
        assert CMAES(10, 0).popsize == 10 and CMAES(5, 0).popsize == 8
        assert CMAES(2, 0, popsize=6).popsize == 6

    def test_ask_and_tell_shapes(self) -> None:
        opt = CMAES(7, 0, popsize=12)
        x = opt.ask()
        assert x.shape == (12, 7) and x.dtype == np.float64
        opt.tell(x, np.arange(12.0))
        assert opt.evals == 12
        assert opt.best_x.shape == (7,)

    def test_best_before_any_tell(self) -> None:
        opt = CMAES(3, 0)
        assert opt.best_f == math.inf and opt.evals == 0 and opt.best_x.shape == (3,)

    def test_best_is_the_best_point_told_not_the_last(self) -> None:
        opt = CMAES(3, 0)
        x = opt.ask()
        f = np.arange(opt.popsize, dtype=float)[::-1].copy()  # the last row is the best
        opt.tell(x, f)
        assert opt.best_f == 0.0 and np.array_equal(opt.best_x, x[-1])
        x2 = opt.ask()
        opt.tell(x2, np.full(opt.popsize, 5.0))
        assert opt.best_f == 0.0 and np.array_equal(opt.best_x, x[-1])

    def test_ask_twice_and_tell_without_ask_are_refused(self) -> None:
        opt = CMAES(3, 0)
        with pytest.raises(RuntimeError, match="ask"):
            opt.tell(np.zeros((opt.popsize, 3)), np.zeros(opt.popsize))
        opt.ask()
        with pytest.raises(RuntimeError, match="tell"):
            opt.ask()

    def test_tell_checks_what_it_is_given(self) -> None:
        opt = CMAES(3, 0)
        x = opt.ask()
        with pytest.raises(ValueError, match="returned by the last ask"):
            opt.tell(x + 0.001, np.zeros(opt.popsize))
        with pytest.raises(ValueError, match="shape"):
            opt.tell(x, np.zeros(opt.popsize - 1))
        with pytest.raises(ValueError, match="NaN"):
            opt.tell(x, np.full(opt.popsize, np.nan))
        opt.tell(x, np.zeros(opt.popsize))  # the batch is still pending after refusals

    def test_infinite_values_rank_without_breaking_the_update(self) -> None:
        opt = CMAES(3, 0)
        x = opt.ask()
        f = np.arange(opt.popsize, dtype=float)
        f[0], f[1] = np.inf, -np.inf
        opt.tell(x, f)
        assert opt.best_f == -np.inf
        assert np.isfinite(opt.mean).all() and np.isfinite(opt.sigma)
        assert np.array_equal(opt.best_x, x[1])

    def test_bad_constructor_arguments(self) -> None:
        for kwargs in (
            {"dim": 0}, {"dim": 2.5}, {"dim": True}, {"popsize": 3}, {"popsize": True},
            {"sigma0": 0.0}, {"sigma0": float("inf")}, {"sigma0": -1.0}, {"restarts": -1},
        ):  # fmt: skip
            args = {"dim": 3, "seed": 0, **kwargs}
            with pytest.raises(ValueError):
                CMAES(**args)


class TestDeterminism:
    def test_same_seed_same_trajectory_different_seed_different(self) -> None:
        def trajectory(seed: int) -> np.ndarray:
            opt = CMAES(6, seed)
            out = []
            for _ in range(15):
                x = opt.ask()
                out.append(x)
                opt.tell(x, np.array([sphere(r) for r in x]))
            return np.concatenate(out)

        assert np.array_equal(trajectory(11), trajectory(11))
        assert not np.array_equal(trajectory(11), trajectory(12))

    def test_streams_are_keyed_so_the_seed_is_what_matters(self) -> None:
        a, b = CMAES(4, 5), CMAES(4, 5)
        np.random.seed(0)  # global state must not matter
        assert np.array_equal(a.ask(), b.ask())

    def test_cmaes_search_is_deterministic(self) -> None:
        r1 = cmaes_search(sphere, 5, 100, seed=3)
        r2 = cmaes_search(sphere, 5, 100, seed=3)
        assert r1.best_f == r2.best_f and np.array_equal(r1.history, r2.history)
        assert np.array_equal(r1.theta_best, r2.theta_best)


class TestRestarts:
    def test_a_flat_objective_triggers_restarts_that_double_the_population(self) -> None:
        opt = CMAES(4, 0, restarts=3)
        base = opt.popsize
        sizes = [base]
        for _ in range(4000):
            x = opt.ask()
            opt.tell(x, np.zeros(opt.popsize))
            if opt.popsize != sizes[-1]:
                sizes.append(opt.popsize)
            if opt.restarts_used == 3 and len(opt.stop_log) >= 4:
                break
        assert sizes == [base, 2 * base, 4 * base, 8 * base]
        assert opt.restarts_used == 3
        assert opt.popsize == base * 2**opt.restarts_used

    def test_restart_starts_from_a_new_mean_and_a_fresh_step_size(self) -> None:
        opt = CMAES(4, 0, restarts=1)
        first_start = opt.mean.copy()
        for _ in range(4000):
            x = opt.ask()
            opt.tell(x, np.zeros(opt.popsize))
            if opt.restarts_used == 1:
                break
        assert opt.restarts_used == 1
        assert opt.sigma == opt.sigma0 and opt.generation == 0
        # a restart draws from its own keyed stream, so it cannot start where the first run did
        assert not np.array_equal(opt.mean, first_start)
        assert np.all((opt.mean >= 0.0) & (opt.mean < 1.0))

    def test_zero_restarts_never_restarts(self) -> None:
        opt = CMAES(3, 0, restarts=0)
        base = opt.popsize
        for _ in range(1500):
            x = opt.ask()
            opt.tell(x, np.zeros(opt.popsize))
        assert opt.restarts_used == 0 and opt.popsize == base
        assert opt.stop_log  # the criterion fired and was logged, but ignored

    def test_the_best_point_survives_a_restart(self) -> None:
        opt = CMAES(3, 0, restarts=2)
        x = opt.ask()
        f = np.arange(opt.popsize, dtype=float) + 10.0
        f[2] = -7.0
        opt.tell(x, f)
        best_x = opt.best_x.copy()
        for _ in range(4000):
            if opt.restarts_used >= 1:
                break
            x = opt.ask()
            opt.tell(x, np.full(opt.popsize, 100.0))
        assert opt.restarts_used >= 1
        assert opt.best_f == -7.0 and np.array_equal(opt.best_x, best_x)

    def test_ipop_helps_on_a_multimodal_function(self) -> None:
        # Rastrigin on [-5.12, 5.12]^4 mapped to the unit box, optimum 0 at x = 0.5
        def rastrigin(x: np.ndarray) -> float:
            z = (x - 0.5) * 10.24
            return float(10 * len(z) + np.sum(z**2 - 10 * np.cos(2 * np.pi * z)))

        opt, ok = drive(rastrigin, 4, 1, 1e-6, 30000, restarts=6, sigma0=0.3)
        assert ok, (opt.best_f, opt.restarts_used)


class TestInternals:
    @pytest.mark.parametrize("big_path", [False, True])
    def test_one_update_matches_the_equations_written_out_again(self, big_path: bool) -> None:
        """Paths, h_sigma, rank-one and rank-mu terms and the step size, from the 2016 tutorial.

        With ``big_path`` the evolution path of the step size is huge, so h_sigma is 0 and the
        cumulation of the covariance path is switched off. (Two mutants survived the convergence
        tests before this one existed: no rank-one term, and h_sigma stuck at 1.)
        """
        n = 5
        opt = CMAES(n, 3, popsize=10)
        rng = np.random.default_rng(0)
        root = rng.normal(size=(n, n))
        opt._cov = root @ root.T + np.eye(n)
        opt._decompose()
        cov = opt._cov.copy()
        opt._ps = rng.normal(size=n) * (20.0 if big_path else 0.3)
        opt._pc = rng.normal(size=n)
        ps, pc, sigma = opt._ps.copy(), opt._pc.copy(), opt.sigma
        opt.generation = 4
        mu = opt._weights.shape[0]
        y_sel = rng.normal(size=(mu, n))
        w = opt._weights
        y_w = w @ y_sel

        values, vectors = np.linalg.eigh(cov)
        inv_sqrt = vectors @ np.diag(values**-0.5) @ vectors.T
        cs, cc, c1, cmu = opt._cs, opt._cc, opt._c1, opt._cmu
        ps2 = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * opt._mu_eff) * (inv_sqrt @ y_w)
        chi_n = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        hs = float(
            np.linalg.norm(ps2) / math.sqrt(1 - (1 - cs) ** (2 * 5)) < (1.4 + 2 / (n + 1)) * chi_n
        )
        pc2 = (1 - cc) * pc + hs * math.sqrt(cc * (2 - cc) * opt._mu_eff) * y_w
        rank_mu = sum(w[i] * np.outer(y_sel[i], y_sel[i]) for i in range(mu))
        cov2 = (
            (1 - c1 - cmu) * cov
            + c1 * (np.outer(pc2, pc2) + (1 - hs) * cc * (2 - cc) * cov)
            + cmu * rank_mu
        )
        sigma2 = sigma * math.exp(min(1.0, (cs / opt._ds) * (np.linalg.norm(ps2) / chi_n - 1)))

        opt._update_paths_and_covariance(y_sel, y_w)
        assert hs == (0.0 if big_path else 1.0)
        assert np.allclose(opt._ps, ps2) and np.allclose(opt._pc, pc2)
        assert np.allclose(opt._cov, cov2)
        assert opt.sigma == pytest.approx(min(max(sigma2, 1e-16), 2.0))

    def test_ties_are_broken_by_a_stable_sort(self) -> None:
        """60 candidates in 3 exact score groups: the selected parents are the first of each rank.

        (A small tied batch cannot tell a stable sort from quicksort, because numpy falls back to
        insertion sort below 17 items; 60 mixed ties can.)
        """
        opt = CMAES(4, 5, popsize=60, sigma0=0.01)
        opt.mean = np.full(4, 0.5)  # every sample lands inside the box: no repair penalty
        x = opt.ask()
        _, y, _ = opt._pending  # type: ignore[misc]
        mean, sigma = opt.mean.copy(), opt.sigma
        scores = (np.arange(60) % 3).astype(float)
        order = np.argsort(scores, kind="stable")
        chosen = order[: opt._weights.shape[0]]
        assert chosen.tolist() == list(range(0, 60, 3)) + list(range(1, 30, 3))
        expected = mean + sigma * (opt._weights @ y[chosen])
        opt.tell(x, scores)
        assert np.allclose(opt.mean, expected, atol=0, rtol=1e-12)

    def test_infinite_values_are_still_separated_by_how_far_outside_the_box_they_are(self) -> None:
        opt = CMAES(2, 0)
        x_box = np.array([[0.5, 0.5], [1.0, 0.5], [0.5, 0.5]])
        x_raw = np.array([[0.5, 0.5], [1.4, 0.5], [0.5, 0.5]])
        ranked = opt._ranking_values(np.array([np.inf, np.inf, 0.0]), x_raw, x_box)
        assert ranked[1] > ranked[0] > ranked[2]
        low = opt._ranking_values(np.array([-np.inf, -np.inf, 0.0]), x_raw, x_box)
        assert low[1] > low[0] and low[0] < low[2]

    def test_strategy_parameters_follow_the_published_formulas(self) -> None:
        n, lam = 10, 10
        opt = CMAES(n, 0)
        mu = lam // 2
        w = math.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / float((w**2).sum())
        assert np.allclose(opt._weights, w) and opt._weights.sum() == pytest.approx(1.0)
        assert opt._mu_eff == pytest.approx(mu_eff)
        assert opt._cs == pytest.approx((mu_eff + 2) / (n + mu_eff + 5))
        assert opt._cc == pytest.approx((4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n))
        assert opt._c1 == pytest.approx(2 / ((n + 1.3) ** 2 + mu_eff))
        assert opt._cmu == pytest.approx(
            min(1 - opt._c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        )
        assert opt._ds == pytest.approx(
            1 + 2 * max(0, math.sqrt((mu_eff - 1) / (n + 1)) - 1) + opt._cs
        )
        assert np.all(np.diff(opt._weights) < 0)  # best candidates weigh more

    def test_covariance_stays_symmetric_positive_definite_and_sigma_bounded(self) -> None:
        opt = CMAES(8, 2)
        for _ in range(200):
            x = opt.ask()
            opt.tell(x, np.array([rosenbrock(r) for r in x]))
            assert np.allclose(opt._cov, opt._cov.T)
            assert np.linalg.eigvalsh(opt._cov).min() > 0.0
            assert 1e-16 <= opt.sigma <= 2.0

    def test_the_covariance_adapts_to_an_elongated_valley(self) -> None:
        def f(x: np.ndarray) -> float:  # long along the diagonal, narrow across it
            u = (x[0] + x[1]) / math.sqrt(2)
            v = (x[0] - x[1]) / math.sqrt(2)
            return float(0.01 * (u - 0.7) ** 2 + 100 * v**2)

        opt = CMAES(2, 1, restarts=0)
        for _ in range(120):
            x = opt.ask()
            opt.tell(x, np.array([f(r) for r in x]))
        values, vectors = np.linalg.eigh(opt._cov)
        assert values[1] / values[0] > 50.0
        assert abs(vectors[0, 1]) == pytest.approx(
            abs(vectors[1, 1]), abs=0.15
        )  # long axis ~ diagonal


class TestCmaesSearch:
    def test_budget_is_respected_exactly(self) -> None:
        calls = []
        run = cmaes_search(lambda x: calls.append(1) or sphere(x), 6, 77, seed=1)
        assert len(calls) == 77 and run.evals == 77 and run.history.shape == (77,)
        assert run.algo == "cmaes"

    def test_history_is_monotone_and_best_attained(self) -> None:
        run = cmaes_search(sphere, 6, 200, seed=2, regime="R_out")
        assert np.all(np.diff(run.history) <= 0) and run.regime == "R_out"
        assert sphere(run.theta_best) == run.best_f == run.history[-1]

    def test_options_are_passed_through(self) -> None:
        calls = []
        cmaes_search(lambda x: calls.append(1) or 0.0, 3, 50, seed=1, popsize=7)
        assert len(calls) == 50

    def test_nan_is_refused(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            cmaes_search(lambda x: float("nan"), 3, 20, seed=1)

    def test_bad_budget(self) -> None:
        for budget in (0, -3, True):
            with pytest.raises(ValueError):
                cmaes_search(sphere, 3, budget, seed=1)
