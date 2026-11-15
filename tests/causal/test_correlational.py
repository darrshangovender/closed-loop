"""P7: the correlation-only models. Ridge effect model, multinomial logistic regression."""

import numpy as np
import pytest

from closed_loop.causal.correlational import (
    N_FEATURES,
    CorrelationalEffectModel,
    SoftmaxRegression,
    design_row,
    reference_point,
)
from closed_loop.causal.records import LifeRecord
from closed_loop.causal.twin import causal_params, generate_observational

# --------------------------------------------------------------------------- synthetic lives


def synthetic_life(
    i: int, rng: np.random.Generator, *, effect: float, treat: bool | None = None
) -> LifeRecord:
    """A hand-made life: the health index reaches 2.0 on ``t_ref`` with slope ``s``; the remaining
    life is exactly ``20 + 30 s + effect * flag`` days, so a ridge fit has a known answer."""
    t_ref = 40 + (i % 7)
    s = float(rng.uniform(0.1, 0.5))
    n = t_ref + 1
    days = np.arange(n, dtype=float)
    hi = 2.0 - s * (t_ref - days)  # a straight line: its slope over any week is exactly s
    flag = bool(rng.random() < 0.5) if treat is None else treat
    remaining = 20.0 + 30.0 * s + (effect if flag else 0.0)
    return LifeRecord(
        asset_id="PU-201",
        hi_days=days,
        hi=hi,
        derate_days=np.arange(t_ref + 1, t_ref + 1 + 3, dtype=float) if flag else np.zeros(0),
        fail_day=t_ref + remaining,
        censored_day=None,
        days=days,
        sig=np.zeros((3, n)),
        load=np.ones(n),
        derate_start=float(t_ref + 0.0) if flag else None,
        mode="IMPELLER_WEAR",
        seed=i,
    )


@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(0)
    return [synthetic_life(i, rng, effect=7.0) for i in range(160)]


class TestReferencePoint:
    def test_features_at_the_first_alert(self):
        rng = np.random.default_rng(1)
        lf = synthetic_life(3, rng, effect=7.0, treat=True)
        pt = reference_point(lf, 2.0, 23.0)
        assert pt is not None and pt.t_ref == 43 and pt.hi == pytest.approx(2.0)
        assert pt.slope > 0.05 and pt.flag == 1.0 and pt.delay == 0.0
        assert pt.outcome == pytest.approx(lf.fail_day - 43)

    def test_an_untreated_life_has_no_flag_and_no_delay(self):
        lf = synthetic_life(0, np.random.default_rng(1), effect=7.0, treat=False)
        pt = reference_point(lf, 2.0, 23.0)
        assert pt.flag == 0.0 and pt.delay == 0.0

    def test_a_life_that_never_alerts_has_no_reference(self):
        n = 30
        days = np.arange(n, dtype=float)
        lf = LifeRecord(
            "PU-201", days, np.ones(n), np.zeros(0), None, 30.0, days, np.zeros((3, n)), np.ones(n)
        )
        assert reference_point(lf, 2.0, 23.0) is None

    def test_the_delay_is_counted_from_the_reference_day(self):
        lf = synthetic_life(2, np.random.default_rng(1), effect=7.0, treat=True)
        import dataclasses

        late = dataclasses.replace(lf, derate_start=float(42 + 6))
        assert reference_point(late, 2.0, 23.0).delay == pytest.approx(6.0)

    def test_the_design_row_layout(self):
        lf = synthetic_life(0, np.random.default_rng(1), effect=7.0, treat=True)
        pt = reference_point(lf, 2.0, 23.0)
        row = design_row(pt, 1.0, 3.0)
        assert row.shape == (N_FEATURES,)
        assert row[3] == 1.0 and row[4] == 3.0 and row[5] == pytest.approx(pt.hi)
        assert design_row(pt, 0.0, 3.0)[3:].tolist() == [0.0, 0.0, 0.0, 0.0]


class TestFit:
    def test_too_few_lives_are_refused(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="at least"):
            CorrelationalEffectModel().fit([synthetic_life(i, rng, effect=1.0) for i in range(5)])

    def test_a_negative_ridge_is_refused(self):
        with pytest.raises(ValueError):
            CorrelationalEffectModel(ridge=-1.0)

    def test_it_must_be_fitted_before_use(self, synthetic):
        with pytest.raises(RuntimeError, match="fit"):
            CorrelationalEffectModel().counterfactual_failure_day(synthetic[0], 40.0, 0.7)

    def test_the_effect_of_the_flag_is_recovered_when_the_data_are_unconfounded(self, synthetic):
        m = CorrelationalEffectModel(ridge=1e-6)
        m.fit(synthetic)
        shifts = []
        for lf in synthetic[:40]:
            pt = reference_point(lf, 2.0, 23.0)
            shifts.append(
                m.counterfactual_failure_day(lf, float(pt.t_ref), 0.7)
                - m.counterfactual_failure_day(lf, float(pt.t_ref), 1.0)
            )
        assert np.mean(shifts) == pytest.approx(7.0, abs=0.05)
        assert np.std(shifts) < 0.05

    def test_a_larger_ridge_shrinks_the_effect(self, synthetic):
        def effect(lam):
            m = CorrelationalEffectModel(ridge=lam)
            m.fit(synthetic)
            lf = synthetic[3]
            t = float(reference_point(lf, 2.0, 23.0).t_ref)
            return m.counterfactual_failure_day(lf, t, 0.7) - m.counterfactual_failure_day(
                lf, t, 1.0
            )

        assert abs(effect(1e5)) < 0.5 * abs(effect(1e-6))

    def test_residual_spread_is_estimated(self, synthetic):
        m = CorrelationalEffectModel(ridge=1e-6)
        m.fit(synthetic)
        assert m.sigma < 0.05  # the synthetic outcome is an exact function of the features


@pytest.fixture(scope="module")
def model(synthetic):
    m = CorrelationalEffectModel(ridge=1e-6)
    m.fit(synthetic)
    return m


class TestCounterfactual:
    def test_no_change_of_treatment_keeps_the_individuals_outcome(self, model, synthetic):
        untreated = next(lf for lf in synthetic if lf.derate_start is None)
        assert model.counterfactual_failure_day(untreated, 0.0, 1.0) == pytest.approx(
            untreated.end_day
        )

    def test_treating_an_untreated_life_adds_the_flag_effect(self, model, synthetic):
        lf = next(x for x in synthetic if x.derate_start is None)
        t = float(reference_point(lf, 2.0, 23.0).t_ref)
        assert model.counterfactual_failure_day(lf, t, 0.7) == pytest.approx(
            lf.end_day + 7.0, abs=0.05
        )

    def test_untreating_a_treated_life_removes_it(self, model, synthetic):
        lf = next(x for x in synthetic if x.derate_start is not None)
        assert model.counterfactual_failure_day(lf, 0.0, 1.0) == pytest.approx(
            lf.end_day - 7.0, abs=0.05
        )

    def test_any_retained_load_below_one_is_the_same_flag(self, model, synthetic):
        lf = next(x for x in synthetic if x.derate_start is None)
        a = model.counterfactual_failure_day(lf, 45.0, 0.9)
        assert a == model.counterfactual_failure_day(lf, 45.0, 0.5)

    def test_a_life_with_no_alert_has_no_counterfactual(self, model):
        n = 30
        days = np.arange(n, dtype=float)
        lf = LifeRecord(
            "PU-201", days, np.ones(n), np.zeros(0), None, 30.0, days, np.zeros((3, n)), np.ones(n)
        )
        with pytest.raises(ValueError, match="never alerted"):
            model.counterfactual_failure_day(lf, 10.0, 0.7)

    def test_the_failure_probability_is_half_at_the_predicted_day(self, model, synthetic):
        lf = synthetic[5]
        t = model.counterfactual_failure_day(lf, 0.0, 1.0)
        assert model.p_fail_before(lf, None, t) == pytest.approx(0.5)
        assert model.p_fail_before(lf, None, t + 20) > 0.99
        assert model.p_fail_before(lf, None, t - 20) < 0.01

    def test_a_replacement_hands_the_horizon_to_a_new_component(self, model, synthetic):
        lf = synthetic[5]
        t = model.counterfactual_failure_day(lf, 0.0, 1.0)
        assert model.p_fail_before(lf, t - 10.0, t + 3.0) < 0.05
        assert model.p_fail_before(lf, t - 10.0, t + 400.0) > 0.95


# -------------------------------------------------------------- confounding in the plant


@pytest.fixture(scope="module")
def confounded():
    return generate_observational(96, 1000, "pi_obs", params=causal_params())


@pytest.fixture(scope="module")
def randomised():
    return generate_observational(96, 1000, "randomised", params=causal_params())


def balance(lives):
    pts = [(reference_point(lf, 2.0, 23.0), lf) for lf in lives]
    pts = [(p, lf) for p, lf in pts if p is not None]
    a = np.array([p.flag for p, _ in pts]) > 0
    return pts, a


def smd(x, a):
    t, u = x[a], x[~a]
    return float((t.mean() - u.mean()) / np.sqrt(0.5 * (t.var() + u.var())))


class TestTheConfoundingIsReal:
    """The operator's policy ties the treatment to the thing that drives the outcome."""

    def test_the_confounded_operator_derates_nearly_everyone_who_lives_long_enough(
        self, confounded
    ):
        pts, a = balance(confounded)
        assert a.mean() > 0.85

    def test_those_left_alone_are_the_ones_that_failed_quickly(self, confounded):
        pts, a = balance(confounded)
        outcome = np.array([p.outcome for p, _ in pts])
        assert (~a).sum() >= 1
        assert outcome[~a].mean() < outcome[a].mean() - 10.0  # immortal-time bias, large

    def test_a_randomised_operator_balances_the_covariates(self, randomised):
        pts, a = balance(randomised)
        assert 0.3 < a.mean() < 0.7
        for name in ("slope", "hi", "outcome"):
            v = np.array([getattr(p, name) for p, _ in pts])
            assert abs(smd(v, a)) < 0.4, name

    def test_the_confounded_data_are_far_less_balanced_than_the_randomised(
        self, confounded, randomised
    ):
        pts_c, a_c = balance(confounded)
        pts_r, a_r = balance(randomised)
        v_c = np.array([p.outcome for p, _ in pts_c])
        v_r = np.array([p.outcome for p, _ in pts_r])
        assert abs(smd(v_c, a_c)) > 3.0 * abs(smd(v_r, a_r))

    def test_the_model_trains_on_either_dataset(self, confounded, randomised):
        for lives in (confounded, randomised):
            m = CorrelationalEffectModel()
            m.fit(lives)
            assert m.n_train >= 60 and m.sigma > 0.0


# --------------------------------------------------------------------------- softmax
class TestSoftmaxRegression:
    def blobs(self, n=300, seed=0):
        rng = np.random.default_rng(seed)
        centres = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]])
        y = rng.integers(0, 3, n)
        return centres[y] + rng.standard_normal((n, 2)), y

    def test_it_separates_blobs(self):
        X, y = self.blobs()
        m = SoftmaxRegression(3).fit(X, y)
        assert (m.predict_proba(X).argmax(axis=1) == y).mean() > 0.9
        assert m.converged

    def test_probabilities_are_distributions(self):
        X, y = self.blobs()
        p = SoftmaxRegression(3).fit(X, y).predict_proba(X[:20])
        assert p.shape == (20, 3) and np.allclose(p.sum(axis=1), 1.0) and p.min() >= 0.0

    def test_it_is_calibrated_on_the_data_it_was_drawn_from(self):
        """Labels drawn from a known softmax: the fitted probabilities approach the true ones."""
        rng = np.random.default_rng(3)
        X = rng.standard_normal((4000, 2))
        W = np.array([[1.5, -1.0, 0.0], [-0.5, 1.0, 0.5]])
        logits = X @ W
        P = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        y = np.array([rng.choice(3, p=row) for row in P])
        m = SoftmaxRegression(3, l2=1e-3).fit(X, y)
        assert np.abs(m.predict_proba(X) - P).mean() < 0.03

    def test_a_stronger_penalty_shrinks_the_weights(self):
        X, y = self.blobs()
        weak = SoftmaxRegression(3, l2=0.01).fit(X, y)
        strong = SoftmaxRegression(3, l2=1000.0).fit(X, y)
        assert np.abs(strong.W).sum() < 0.5 * np.abs(weak.W).sum()

    def test_a_class_never_seen_gets_a_low_probability(self):
        X, y = self.blobs()
        y = np.where(y == 2, 1, y)
        p = SoftmaxRegression(3).fit(X, y).predict_proba(X)
        assert p[:, 2].max() < 0.05

    def test_fitting_is_deterministic(self):
        X, y = self.blobs()
        a = SoftmaxRegression(3).fit(X, y).predict_proba(X)
        b = SoftmaxRegression(3).fit(X, y).predict_proba(X)
        assert np.array_equal(a, b)

    def test_features_on_very_different_scales_are_standardised_before_the_penalty(self):
        X, y = self.blobs()
        X = X * np.array([1.0, 1e-4])  # the second feature carries a class but looks negligible
        acc = (SoftmaxRegression(3).fit(X, y).predict_proba(X).argmax(axis=1) == y).mean()
        assert acc > 0.9

    def test_constant_features_are_harmless(self):
        X, y = self.blobs()
        X = np.column_stack([X, np.ones(len(X))])
        assert (SoftmaxRegression(3).fit(X, y).predict_proba(X).argmax(axis=1) == y).mean() > 0.9

    def test_input_checks(self):
        X, y = self.blobs(50)
        with pytest.raises(ValueError, match="at least two"):
            SoftmaxRegression(1)
        with pytest.raises(ValueError, match="out of range"):
            SoftmaxRegression(3).fit(X, y + 5)
        with pytest.raises(ValueError, match="X must be"):
            SoftmaxRegression(3).fit(X, y[:10])
        with pytest.raises(RuntimeError, match="fit"):
            SoftmaxRegression(3).predict_proba(X)
