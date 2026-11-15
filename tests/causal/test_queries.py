"""P7: queries Q1, Q2 and Q4, their scoring arithmetic, and the three models on real twin truth."""

import numpy as np
import pytest

from closed_loop.causal.queries import (
    Case,
    QueryConfig,
    build_cases,
    calibration_slope,
    make_effect_model,
    paired_difference,
    per_life_scores,
    records_from_scores,
    score,
)
from closed_loop.causal.records import LifeRecord
from closed_loop.core.types import Fidelity

# --------------------------------------------------------------------------------- the model zoo


def test_the_three_kinds_and_nothing_else():
    assert make_effect_model("correlational").name == "correlational"
    assert make_effect_model("scm").name == "scm"
    assert make_effect_model("scm_misspecified").name == "scm_misspecified"
    with pytest.raises(ValueError, match="unknown effect model"):
        make_effect_model("neural")


# ------------------------------------------------------------------------ hand-built arithmetic


def tiny_life() -> LifeRecord:
    n = 5
    days = np.arange(n, dtype=float)
    return LifeRecord(
        "PU-201", days, np.ones(n), np.zeros(0), None, 5.0, days, np.zeros((3, n)), np.ones(n)
    )


def case(*, truth_nothing, truth_derate, truth_replace, t0=80, horizon=110.0, replace_day=85.0):
    return Case(
        tiny_life(), t0, horizon, replace_day, truth_nothing, truth_derate, truth_replace, float(t0)
    )


class Stub:
    """A model that answers with fixed numbers."""

    name = "stub"

    def __init__(self, t_derate, t_nothing, p_nothing, p_replace):
        self.t = {0.7: t_derate, 1.0: t_nothing}
        self.p = (p_nothing, p_replace)

    def fit(self, lives):
        return None

    def counterfactual_failure_day(self, life, derate_from_day, load_cap):
        return self.t[round(load_cap, 3)]

    def p_fail_before(self, life, replace_day, horizon_day):
        return self.p[0] if replace_day is None else self.p[1]


GAIN = case(truth_nothing=100.0, truth_derate=115.0, truth_replace=200.0)  # derating gains 15 days
SMALL = case(truth_nothing=100.0, truth_derate=105.0, truth_replace=200.0)  # derating gains 5 days


class TestArithmetic:
    def test_a_perfect_model_has_no_error_no_brier_and_no_regret(self):
        sc = per_life_scores(Stub(115.0, 100.0, 1.0, 0.0), [GAIN])
        assert sc.q1_err_days[0] == 0.0 and sc.q1_scaled[0] == 0.0
        assert sc.q2_brier[0] == 0.0
        assert sc.q4_regret_days[0] == 0.0 and sc.q4_derated[0]

    def test_q1_is_the_absolute_error_capped_and_scaled(self):
        sc = per_life_scores(Stub(100.0, 100.0, 0.5, 0.5), [GAIN])
        assert sc.q1_err_days[0] == 15.0 and sc.q1_scaled[0] == pytest.approx(0.5)
        far = per_life_scores(Stub(300.0, 100.0, 0.5, 0.5), [GAIN])
        assert far.q1_err_days[0] == 185.0 and far.q1_scaled[0] == 1.0  # capped at 30 days

    def test_q2_outcomes_are_failure_before_the_horizon_in_each_world(self):
        sc = per_life_scores(Stub(115.0, 100.0, 0.5, 0.5), [GAIN])
        assert list(sc.q2_out[0]) == [1.0, 0.0]  # 100 < 110 yes; 200 < 110 no
        assert list(sc.q2_pred[0]) == [0.5, 0.5]
        assert sc.q2_brier[0] == pytest.approx(0.25)

    def test_the_brier_score_averages_the_two_arms(self):
        sc = per_life_scores(Stub(115.0, 100.0, 0.8, 0.1), [GAIN])
        assert sc.q2_brier[0] == pytest.approx(((0.8 - 1) ** 2 + (0.1 - 0) ** 2) / 2)

    def test_q4_a_model_that_never_derates_loses_the_gain_over_the_cost(self):
        sc = per_life_scores(Stub(100.0, 100.0, 0.5, 0.5), [GAIN])
        # derating is worth 115 - 10 = 105 against 100 for doing nothing; the model said no
        assert not sc.q4_derated[0] and sc.q4_regret_days[0] == pytest.approx(5.0)
        assert sc.q4_scaled[0] == pytest.approx(5.0 / 30.0)

    def test_q4_a_model_that_always_derates_overpays_when_the_gain_is_small(self):
        sc = per_life_scores(Stub(200.0, 100.0, 0.5, 0.5), [SMALL])
        assert sc.q4_derated[0]
        assert sc.q4_regret_days[0] == pytest.approx(100.0 - (105.0 - 10.0))  # 5 days

    def test_q4_a_perfect_model_leaves_a_small_gain_alone(self):
        sc = per_life_scores(Stub(105.0, 100.0, 1.0, 0.0), [SMALL])
        assert not sc.q4_derated[0] and sc.q4_regret_days[0] == 0.0

    def test_the_cost_of_a_derate_is_a_parameter(self):
        cheap = per_life_scores(
            Stub(105.0, 100.0, 1.0, 0.0), [SMALL], QueryConfig(derate_cost_days=1.0)
        )
        assert cheap.q4_derated[0]

    def test_scores_are_per_case_and_in_range(self):
        sc = per_life_scores(Stub(110.0, 100.0, 0.3, 0.2), [GAIN, SMALL, GAIN])
        assert sc.n == 3 and sc.q1_scaled.shape == (3,)
        for v in (sc.q1_scaled, sc.q2_brier, sc.q4_scaled):
            assert np.all((v >= 0.0) & (v <= 1.0))


class TestCalibrationSlope:
    def test_a_calibrated_forecast_has_slope_one(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0.05, 0.95, 20000)
        y = (rng.random(20000) < p).astype(float)
        assert calibration_slope(p, y) == pytest.approx(1.0, abs=0.1)

    def test_an_overconfident_forecast_has_slope_below_one(self):
        rng = np.random.default_rng(1)
        p = rng.uniform(0.05, 0.95, 20000)
        y = (rng.random(20000) < p).astype(float)
        sharp = 1.0 / (1.0 + np.exp(-3.0 * np.log(p / (1 - p))))
        assert calibration_slope(sharp, y) < 0.6

    def test_an_inverted_forecast_has_a_negative_slope(self):
        rng = np.random.default_rng(2)
        p = rng.uniform(0.05, 0.95, 5000)
        y = (rng.random(5000) < p).astype(float)
        assert calibration_slope(1.0 - p, y) < -0.5

    def test_a_constant_forecast_has_no_slope(self):
        assert calibration_slope(np.full(50, 0.3), np.arange(50) % 2) == 0.0

    def test_perfect_separation_gives_a_large_finite_slope(self):
        p = np.array([0.001] * 20 + [0.999] * 20)
        y = np.array([0.0] * 20 + [1.0] * 20)
        s = calibration_slope(p, y)
        assert np.isfinite(s) and s > 1.0


class TestPairedDifference:
    def test_equal_models_differ_by_nothing(self):
        a = per_life_scores(Stub(110.0, 100.0, 0.4, 0.4), [GAIN] * 30)
        r = paired_difference(a, a, "q1", a_name="a", b_name="b")
        assert r.value == 0.0 and r.ci_lo <= 0.0 <= r.ci_hi and r.n == 30

    def test_the_difference_is_antisymmetric(self):
        a = per_life_scores(Stub(115.0, 100.0, 1.0, 0.0), [GAIN] * 40)
        b = per_life_scores(Stub(100.0, 100.0, 0.5, 0.5), [GAIN] * 40)
        ab = paired_difference(a, b, "q1", a_name="a", b_name="b")
        ba = paired_difference(b, a, "q1", a_name="b", b_name="a")
        assert ab.value == pytest.approx(-ba.value) and ab.value < 0
        assert ab.ci_lo == pytest.approx(-ba.ci_hi) and ab.ci_hi == pytest.approx(-ba.ci_lo)

    def test_a_clear_difference_excludes_zero_with_enough_cases(self):
        a = per_life_scores(Stub(115.0, 100.0, 1.0, 0.0), [GAIN] * 400)
        b = per_life_scores(Stub(100.0, 100.0, 0.5, 0.5), [GAIN] * 400)
        r = paired_difference(a, b, "q2_brier", a_name="a", b_name="b")
        assert r.ci_hi < 0.0 and r.fidelity is Fidelity.F1D

    def test_input_checks(self):
        a = per_life_scores(Stub(115.0, 100.0, 1.0, 0.0), [GAIN] * 3)
        b = per_life_scores(Stub(115.0, 100.0, 1.0, 0.0), [GAIN] * 4)
        with pytest.raises(ValueError, match="different numbers"):
            paired_difference(a, b, "q1", a_name="a", b_name="b")
        with pytest.raises(ValueError, match="metric"):
            paired_difference(a, a, "q9", a_name="a", b_name="b")


class TestRecords:
    def test_a_model_gets_four_records_labelled_f1d(self):
        sc = per_life_scores(Stub(115.0, 100.0, 0.9, 0.1), [GAIN, SMALL] * 10)
        recs = records_from_scores(sc, "stub", "20 hand-built cases")
        assert set(recs) == {"Q1_mcf", "Q2_brier", "Q2_calibration_slope", "Q4_regret"}
        for r in recs.values():
            assert r.fidelity is Fidelity.F1D and r.script == "closed_loop.causal.queries"
            assert r.name.startswith("stub.")
        for k in ("Q1_mcf", "Q2_brier", "Q4_regret"):
            assert recs[k].ci_lo is not None and recs[k].ci_lo <= recs[k].value <= recs[k].ci_hi
            assert recs[k].n == 20

    def test_the_extra_fields_carry_the_unscaled_numbers(self):
        sc = per_life_scores(Stub(100.0, 100.0, 0.5, 0.5), [GAIN] * 5)
        recs = records_from_scores(sc, "stub", "5 cases")
        assert recs["Q1_mcf"].extra["mean_abs_error_days"] == 15.0
        assert recs["Q4_regret"].extra["mean_regret_days"] == 5.0
        assert recs["Q2_brier"].extra["base_rate_nothing"] == 1.0


# ----------------------------------------------------------------------------- real twin truth


class TestBuildCases:
    def test_every_case_has_a_reference_day_and_consistent_truth(self, study_cases):
        cases, skipped = study_cases
        assert len(cases) + skipped == 24 and len(cases) >= 18
        for k in cases:
            assert k.t0 == k.life.first_alert_day()
            assert k.horizon == k.t0 + 30.0 and k.replace_day == k.t0 + 5.0
            assert k.truth_derate > k.truth_nothing  # derating never hurts
            assert k.truth_replace >= min(k.truth_nothing, k.replace_day)
            assert k.derate_applied == k.t0  # no refusals in the study plant

    def test_building_is_deterministic(self, study_cases, noisy_params):
        again, _ = build_cases(range(1200, 1202), noisy_params)
        first = study_cases[0][: len(again)]
        assert [k.truth_derate for k in again] == [k.truth_derate for k in first]

    def test_the_factual_world_is_the_confounded_operators(self, study_cases):
        cases, _ = study_cases
        assert sum(k.life.derate_start is not None for k in cases) >= 0.7 * len(cases)

    def test_the_truth_has_real_variation_in_the_failure_questions(self, study_cases):
        cases, _ = study_cases
        out = np.array([[k.truth_nothing < k.horizon, k.truth_replace < k.horizon] for k in cases])
        assert 0.0 < out[:, 0].mean() < 1.0  # the question is not trivial for do(nothing)


@pytest.fixture(scope="module")
def scored(study_cases, noisy_scm, noisy_misspecified, noisy_corr):
    cases, _ = study_cases
    return {
        "scm": per_life_scores(noisy_scm, cases),
        "scm_misspecified": per_life_scores(noisy_misspecified, cases),
        "correlational": per_life_scores(noisy_corr, cases),
    }


class TestTheThreeModelsOnTwinTruth:
    def test_the_scm_with_abduction_is_close_to_the_truth(self, scored):
        # favoured by construction: it was handed the structure
        assert scored["scm"].q1_err_days.mean() < 3.0

    def test_it_beats_both_other_models_on_q1(self, scored):
        s = scored["scm"].q1_err_days.mean()
        assert s < 0.5 * scored["scm_misspecified"].q1_err_days.mean()
        assert s < 0.5 * scored["correlational"].q1_err_days.mean()

    def test_it_forecasts_the_failure_questions_better_than_the_correlational_model(self, scored):
        assert scored["scm"].q2_brier.mean() < scored["correlational"].q2_brier.mean()

    def test_all_scores_are_finite_and_in_range(self, scored):
        for sc in scored.values():
            for v in (sc.q1_scaled, sc.q2_brier, sc.q4_scaled):
                assert np.all(np.isfinite(v)) and np.all((v >= 0) & (v <= 1))

    def test_score_returns_records_for_a_fitted_model(self, study_cases, noisy_scm):
        cases, _ = study_cases
        recs = score(noisy_scm, [1200], cases=cases)
        assert recs["Q1_mcf"].n == len(cases)
        assert recs["Q1_mcf"].name == "scm.Q1_mcf"
        assert "pi_obs" in recs["Q1_mcf"].measured_on

    def test_score_needs_cases(self, noisy_scm):
        with pytest.raises(ValueError, match="no test case"):
            score(noisy_scm, [1200], cases=[])
