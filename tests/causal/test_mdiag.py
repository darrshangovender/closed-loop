"""P7: M-DIAG, the per-life metric of C-DIAG, and the way the shift is built."""

import numpy as np
import pytest

from closed_loop.causal import mdiag
from closed_loop.causal.diagnoser import ScmDiagnoser, make_diagnoser
from closed_loop.core.types import Fidelity
from closed_loop.plant import PlantParams


@pytest.fixture(scope="module")
def lives():
    return mdiag.collect_lives(range(1800, 1830), mdiag.diagnosis_params(8.0))


@pytest.fixture(scope="module")
def cases(lives):
    return mdiag.alert_cases(lives)


@pytest.fixture(scope="module")
def outcomes(fitted_diagnosers, cases):
    out = {k: mdiag.evaluate(d, cases) for k, d in fitted_diagnosers.items()}
    out["oracle"] = mdiag.evaluate(make_diagnoser("oracle"), cases)
    return out


class TestSeedsAndParameters:
    def test_only_development_seeds_are_accepted(self):
        for bad in ([999], [2000], [1000, 3500], [5]):
            with pytest.raises(ValueError, match="development seeds"):
                mdiag.collect_lives(bad, PlantParams())

    def test_the_tilt_is_the_only_thing_the_helper_changes(self):
        p = mdiag.diagnosis_params(-8.0)
        base = PlantParams()
        assert p.mode_load_tilt == -8.0 and p.horizon_q == base.horizon_q
        assert p.p_fail_in_episode == base.p_fail_in_episode and not p.derate_enabled

    def test_the_default_tilt_is_fixed_in_advance(self):
        assert mdiag.DEFAULT_TILT == 8.0

    def test_collecting_is_deterministic(self):
        a = mdiag.collect_lives([1850], mdiag.diagnosis_params(8.0))
        b = mdiag.collect_lives([1850], mdiag.diagnosis_params(8.0))
        assert all(
            np.array_equal(x.hi, y.hi) and x.alerts == y.alerts for x, y in zip(a, b, strict=True)
        )

    def test_the_two_signs_change_the_modes_and_nothing_else(self):
        plus = mdiag.collect_lives([1851, 1852], mdiag.diagnosis_params(+8.0))
        minus = mdiag.collect_lives([1851, 1852], mdiag.diagnosis_params(-8.0))
        for x, y in zip(plus, minus, strict=True):
            if x.mode == y.mode:
                assert np.array_equal(x.load, y.load) and x.fail_day == y.fail_day
            assert np.array_equal(x.load[:5], y.load[:5])  # the load streams are the same


class TestCases:
    def test_there_is_one_case_per_labelled_alert(self, lives, cases):
        assert len(cases) == sum(len(lf.alerts) for lf in lives) and len(cases) > 100

    def test_each_history_ends_on_the_alert_day(self, cases):
        for k in cases[:60]:
            assert k.history.days[-1] == k.day and k.history.asset_id == k.asset_id
            assert k.level in (1, 2)

    def test_the_alert_reading_is_present_and_above_the_alert_level(self, cases):
        for k in cases[:60]:
            assert not np.isnan(k.history.y[-1, 0]) and k.history.y[-1, 0] >= 2.0 - 1e-9

    def test_the_truth_is_one_of_the_vocabulary(self, cases):
        assert {k.truth_cause for k in cases} <= {"degradation", "sensor_fault", "process_upset"}
        assert {k.truth_mode for k in cases} <= {"IMPELLER_WEAR", "LINER_WEAR", "BEARING_DEFECT"}

    def test_asset_class_comes_from_the_register(self, cases):
        assert {k.asset_class for k in cases} == {"CRUSHER", "BELT_CONVEYOR", "SLURRY_PUMP"}


class TestEvaluate:
    def test_both_means_cause_and_mode(self, outcomes, cases):
        for o in outcomes.values():
            assert np.array_equal(o.correct, o.cause_ok & o.mode_ok)
            truth_c = np.array([k.truth_cause for k in cases])
            assert np.array_equal(o.cause_ok, o.cause == truth_c)

    def test_the_oracle_scores_one(self, outcomes):
        assert outcomes["oracle"].correct.all()

    def test_a_diagnoser_that_always_says_the_same_scores_its_frequency(self, cases):
        class Always:
            name = "always"

            def fit(self, lives):
                return None

            def diagnose(self, asset_id, day, history):
                from closed_loop.causal.diagnoser import Diagnosis, labels_for

                lab = labels_for(asset_id)
                post = {k: 0.0 for k in lab}
                post["degradation:BEARING_DEFECT"] = 1.0
                return Diagnosis(asset_id, day, "BEARING_DEFECT", "degradation", post)

        o = mdiag.evaluate(Always(), cases)
        want = np.mean(
            [k.truth_cause == "degradation" and k.truth_mode == "BEARING_DEFECT" for k in cases]
        )
        assert o.correct.mean() == pytest.approx(want)

    def test_outcome_arrays_match_the_cases(self, outcomes, cases):
        for o in outcomes.values():
            assert o.correct.shape == o.cause.shape == o.mode.shape == (len(cases),)


class TestRecords:
    def test_accuracy_records_carry_the_setting_and_a_confidence_sequence(self, outcomes, cases):
        recs = mdiag.records("reversed", cases, outcomes, measured_on="30 seeds")
        for k in ("scm", "corr", "oracle"):
            r = recs[k]
            assert r.name == f"M-DIAG.{k}.reversed" and r.n == len({c.life for c in cases})
            assert r.fidelity is Fidelity.F1 and r.script == "closed_loop.causal.mdiag"
            assert r.ci_lo <= r.value <= r.ci_hi
            assert r.value == pytest.approx(mdiag.life_means(cases, outcomes[k].correct).mean())
            assert r.extra["alerts"] == len(cases)  # the alerts are counted, the lives are the unit
        assert recs["oracle"].value == 1.0

    def test_the_paired_difference_is_scm_minus_corr_per_life(self, outcomes, cases):
        r = mdiag.records("unshifted", cases, outcomes, measured_on="x")["scm-corr"]
        a = mdiag.life_means(cases, outcomes["scm"].correct)
        b = mdiag.life_means(cases, outcomes["corr"].correct)
        assert r.n == a.size == b.size and a.size < len(cases)  # lives, not alerts
        assert r.value == pytest.approx(np.mean(a - b))
        assert r.extra["scm_only_right"] == int(np.sum(a > b))
        assert r.extra["corr_only_right"] == int(np.sum(a < b))
        assert r.ci_lo <= r.value <= r.ci_hi

    def test_a_stricter_level_gives_a_wider_interval(self, outcomes, cases):
        loose = mdiag.records("x", cases, outcomes, alpha=0.05, measured_on="x")["scm-corr"]
        strict = mdiag.records("x", cases, outcomes, alpha=0.05 / 9, measured_on="x")["scm-corr"]
        assert strict.ci_hi - strict.ci_lo > loose.ci_hi - loose.ci_lo

    def test_a_clear_winner_gets_a_confidence_sequence_above_zero(self, cases):
        """scm always right, corr always wrong: the interval for scm - corr must sit above zero."""
        n = len(cases)
        right = mdiag.Outcome(
            "scm",
            np.array([""] * n),
            np.array([""] * n),
            np.ones(n, bool),
            np.ones(n, bool),
            np.ones(n, bool),
        )
        wrong = mdiag.Outcome(
            "corr",
            np.array([""] * n),
            np.array([""] * n),
            np.zeros(n, bool),
            np.zeros(n, bool),
            np.zeros(n, bool),
        )
        r = mdiag.records("x", cases, {"scm": right, "corr": wrong}, measured_on="x")["scm-corr"]
        assert r.value == 1.0 and r.ci_lo > 0.5 and r.ci_hi == pytest.approx(1.0)

    def test_the_level_reaches_the_accuracy_intervals_too(self, outcomes, cases):
        loose = mdiag.records("x", cases, outcomes, alpha=0.05, measured_on="x")["scm"]
        strict = mdiag.records("x", cases, outcomes, alpha=0.05 / 9, measured_on="x")["scm"]
        assert strict.ci_hi - strict.ci_lo > loose.ci_hi - loose.ci_lo

    def test_the_comparison_is_absent_without_both_diagnosers(self, outcomes, cases):
        recs = mdiag.records("x", cases, {"scm": outcomes["scm"]}, measured_on="x")
        assert "scm-corr" not in recs

    def test_no_decision_is_attached_to_a_record(self, outcomes, cases):
        recs = mdiag.records("x", cases, outcomes, measured_on="x")
        for r in recs.values():
            assert not any(
                "win" in str(v).lower() or "loss" in str(v).lower() for v in r.extra.values()
            )


class TestPerLife:
    """Prereg 5.1: alerts in one life share a mode and a cause, so the unit is the life."""

    def case(self, life: int, seed: int = 1) -> mdiag.AlertCase:
        return mdiag.AlertCase(seed, "PU-201", 10, 1, "degradation", "LINER_WEAR", None, life)  # type: ignore[arg-type]

    def test_a_life_scores_the_mean_of_its_alerts_and_counts_once(self):
        cases = [self.case(0), self.case(0), self.case(0), self.case(1)]
        right = np.array([True, True, False, True])
        assert mdiag.life_means(cases, right).tolist() == pytest.approx([2 / 3, 1.0])

    def test_lives_are_in_the_order_they_first_appear(self):
        cases = [self.case(5), self.case(2), self.case(5), self.case(9)]
        values = np.array([True, False, False, True])
        assert mdiag.life_means(cases, values).tolist() == pytest.approx([0.5, 0.0, 1.0])

    def test_an_unnumbered_case_is_its_own_life(self):
        cases = [self.case(-1), self.case(-1), self.case(3)]
        assert mdiag.life_means(cases, np.array([True, False, True])).tolist() == [1.0, 0.0, 1.0]

    def test_many_alerts_in_one_life_do_not_narrow_the_interval(self):
        """The reason for the unit: 200 alerts of 2 lives are 2 data points, not 200."""
        wrong_life = [self.case(0)] * 100
        right_life = [self.case(1)] * 100
        cases = wrong_life + right_life
        blank = np.array([""] * 200)
        scm_correct = np.array([False] * 100 + [True] * 100)
        scm = mdiag.Outcome(
            "scm", blank, blank, scm_correct, np.ones(200, bool), np.ones(200, bool)
        )
        corr = mdiag.Outcome(
            "corr", blank, blank, np.zeros(200, bool), np.zeros(200, bool), np.zeros(200, bool)
        )
        r = mdiag.records("x", cases, {"scm": scm, "corr": corr}, measured_on="x")["scm-corr"]
        assert r.n == 2 and r.extra["alerts"] == 200
        assert r.ci_lo < 0.0  # two lives cannot show a difference of 0.5 at this level

    def test_lives_with_no_alert_are_excluded_and_counted(self, outcomes, cases):
        n_with = len({c.life for c in cases})
        recs = mdiag.records("x", cases, outcomes, measured_on="x", n_lives=n_with + 7)
        assert recs["scm"].extra["lives_without_alert"] == 7
        assert recs["scm-corr"].extra["lives_without_alert"] == 7
        assert (
            "lives_without_alert"
            not in mdiag.records("x", cases, outcomes, measured_on="x")["scm"].extra
        )

    def test_the_cases_know_the_life_they_came_from(self, lives, cases):
        index = {id(lf): i for i, lf in enumerate(lives)}
        assert {c.life for c in cases} <= set(index.values())
        assert len({c.life for c in cases}) == sum(1 for lf in lives if lf.alerts)


class TestBreakdown:
    def test_it_separates_cause_class_and_level(self, outcomes, cases):
        bd = mdiag.breakdown(cases, outcomes)
        assert bd["n_alerts"] == len(cases)
        assert sum(bd["n_by_cause"].values()) == len(cases)
        for k in ("scm", "corr", "oracle"):
            b = bd[k]
            assert 0.0 <= b["both"] <= 1.0 and b["both"] <= b["cause_only"]
            assert set(b["by_class"]) == {"CRUSHER", "BELT_CONVEYOR", "SLURRY_PUMP"}
            assert set(b["by_level"]) == {1, 2}
            assert b["mode_on_degradation"] >= b["mode_only"] - 0.2
        assert bd["oracle"]["both"] == 1.0

    def test_both_on_degradation_equals_mode_on_degradation_for_a_right_cause(
        self, outcomes, cases
    ):
        bd = mdiag.breakdown(cases, outcomes)
        assert bd["scm"]["both_on_degradation"] <= bd["scm"]["mode_on_degradation"] + 1e-12


class TestEndToEnd:
    def test_the_two_settings_run_and_give_sensible_accuracy(self, fitted_diagnosers):
        for tilt in (+8.0, -8.0):
            cs = mdiag.alert_cases(
                mdiag.collect_lives(range(1900, 1920), mdiag.diagnosis_params(tilt))
            )
            for kind in ("scm", "corr"):
                acc = mdiag.evaluate(fitted_diagnosers[kind], cs).correct.mean()
                assert 0.7 < acc <= 1.0, (tilt, kind, acc)

    def test_a_diagnoser_fitted_on_one_sign_is_not_refitted_by_testing_on_the_other(
        self, diag_train
    ):
        d = ScmDiagnoser()
        d.fit(diag_train)
        before = (d.h0, d.pi.copy(), {k: v.copy() for k, v in d.gains.items()})
        cs = mdiag.alert_cases(mdiag.collect_lives(range(1910, 1914), mdiag.diagnosis_params(-8.0)))
        mdiag.evaluate(d, cs)
        assert d.h0 == before[0] and np.array_equal(d.pi, before[1])
        assert all(np.array_equal(d.gains[k], before[2][k]) for k in d.gains)


class TestSeedRoles:
    """P12b: the final run trains on calib seeds and tests on derived held-out seeds."""

    def test_each_role_accepts_its_own_seeds_only(self):
        params = PlantParams()
        with pytest.raises(ValueError, match="belong to another role"):
            mdiag.collect_lives([1500], params, role="calib")
        with pytest.raises(ValueError, match="belong to another role"):
            mdiag.collect_lives([2400], params, role="heldout")  # a calib seed is not held out
        with pytest.raises(ValueError, match="belong to another role"):
            mdiag.collect_lives([3001], params, role="heldout")  # nor is a search seed
        with pytest.raises(ValueError, match="development seeds"):
            mdiag.collect_lives([2400], params, role="dev")
        with pytest.raises(ValueError, match="role must be one of"):
            mdiag.collect_lives([1500], params, role="search")

    def test_calib_and_heldout_seeds_are_accepted_and_give_lives(self):
        params = mdiag.diagnosis_params(8.0)
        assert len(mdiag.collect_lives([2400], params, role="calib")) == 4
        big = 2**63 + 12345  # a derived held-out seed is a 64-bit integer
        assert len(mdiag.collect_lives([big], params, role="heldout")) == 4

    def test_the_default_role_is_dev(self):
        assert len(mdiag.collect_lives([1500], mdiag.diagnosis_params(8.0))) == 4
