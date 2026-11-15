"""P7: diagnosis in the loop (0003 A3). Interface, the fitted structure, behaviour, invariance."""

import dataclasses

import numpy as np
import pytest

from closed_loop.causal import mdiag
from closed_loop.causal.diagnoser import (
    AlertTruthTable,
    CorrDiagnoser,
    Diagnosis,
    OracleDiagnoser,
    ScmDiagnoser,
    class_modes,
    labels_for,
    make_diagnoser,
)
from closed_loop.causal.records import CAUSES, ObservationHistory
from closed_loop.plant import PlantParams

PLANT = PlantParams()  # the generator, used here only to build crafted readings


def craft(
    asset: str,
    z: list[float],
    family: str,
    *,
    g: float = 0.0,
    dl: float = 0.0,
    mask_last_hi: bool = False,
    old_load: float = 0.0,
    n_old: int = 10,
) -> ObservationHistory:
    """A history that ends in the given severities ``z`` (health-index units), noise-free.

    ``n_old`` older days with load deviation ``old_load`` and no wear come first.
    """
    gains = PLANT.sig_loading_bearing if family == "bearing" else PLANT.sig_loading_wear
    c = PLANT.sig_load_coef
    zs = [0.0] * n_old + list(z)
    dls = [old_load] * n_old + [dl] * len(z)
    rows = []
    for i, (zz, d) in enumerate(zip(zs, dls, strict=True)):
        gg = g if i >= n_old else 0.0
        hi = PLANT.hi_base + zz + gg + PLANT.hi_load_coef * d
        sig = [gains[j] * zz / (PLANT.hi_fail - PLANT.hi_base) + c[j] * d for j in range(3)]
        rows.append([hi, *sig])
    y = np.array(rows)
    if mask_last_hi:
        y[-1, 0] = np.nan
    n = len(zs)
    return ObservationHistory(asset, np.arange(n), y, 1.0 + np.array(dls))


@pytest.fixture(scope="module")
def scm(fitted_diagnosers):
    return fitted_diagnosers["scm"]


@pytest.fixture(scope="module")
def corr(fitted_diagnosers):
    return fitted_diagnosers["corr"]


@pytest.fixture(scope="module")
def held_out():
    lives = mdiag.collect_lives(range(1700, 1740), mdiag.diagnosis_params(mdiag.DEFAULT_TILT))
    return mdiag.alert_cases(lives)


class TestInterface:
    def test_the_kinds_and_their_names(self):
        assert [make_diagnoser(k).name for k in ("scm", "corr", "oracle")] == [
            "scm",
            "corr",
            "oracle",
        ]
        assert isinstance(make_diagnoser("scm"), ScmDiagnoser)
        assert isinstance(make_diagnoser("corr"), CorrDiagnoser)
        assert isinstance(make_diagnoser("oracle"), OracleDiagnoser)
        with pytest.raises(ValueError, match="unknown diagnoser"):
            make_diagnoser("llm")

    def test_a_diagnosis_is_a_frozen_value(self):
        d = Diagnosis(
            "PU-201", 40, "IMPELLER_WEAR", "degradation", {"degradation:IMPELLER_WEAR": 1.0}
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            d.mode = "x"  # type: ignore[misc]
        assert [f.name for f in dataclasses.fields(Diagnosis)] == [
            "asset_id",
            "day",
            "mode",
            "cause",
            "posterior",
        ]

    def test_the_class_of_an_asset_fixes_its_modes(self):
        assert [m for m, _ in class_modes("CV-101")] == ["BEARING_DEFECT"]
        assert {m for m, _ in class_modes("CR-501")} == {"LINER_WEAR", "BEARING_DEFECT"}
        assert {m for m, _ in class_modes("PU-202")} == {"IMPELLER_WEAR", "BEARING_DEFECT"}
        assert len(labels_for("CV-101")) == 3 and len(labels_for("PU-201")) == 6

    def test_unfitted_diagnosers_refuse_to_answer(self):
        h = craft("PU-201", [1.0], "wear")
        for kind in ("scm", "corr"):
            with pytest.raises(RuntimeError, match="fit"):
                make_diagnoser(kind).diagnose("PU-201", h.days[-1], h)

    def test_fitting_needs_labelled_lives(self):
        with pytest.raises(ValueError):
            ScmDiagnoser().fit([])
        with pytest.raises(ValueError, match="too few"):
            CorrDiagnoser().fit([])


class TestPosterior:
    @pytest.mark.parametrize("kind", ["scm", "corr"])
    def test_it_is_a_distribution_over_the_classes_labels(self, fitted_diagnosers, kind):
        for asset, fam in (("PU-201", "wear"), ("CR-501", "bearing"), ("CV-101", "bearing")):
            h = craft(asset, [0.6, 0.8, 1.0, 1.2], fam)
            d = fitted_diagnosers[kind].diagnose(asset, int(h.days[-1]), h)
            assert set(d.posterior) == set(labels_for(asset))
            assert sum(d.posterior.values()) == pytest.approx(1.0)
            assert all(v >= 0.0 for v in d.posterior.values())
            assert f"{d.cause}:{d.mode}" == max(d.posterior, key=d.posterior.__getitem__)
            assert d.cause in CAUSES and d.asset_id == asset and d.day == h.days[-1]

    @pytest.mark.parametrize("kind", ["scm", "corr"])
    def test_a_conveyor_is_never_diagnosed_with_a_wear_mode(self, fitted_diagnosers, kind):
        h = craft("CV-101", [2.0, 3.0], "wear")  # wear signatures on an asset that cannot wear
        d = fitted_diagnosers[kind].diagnose("CV-101", int(h.days[-1]), h)
        assert d.mode == "BEARING_DEFECT"

    @pytest.mark.parametrize("kind", ["scm", "corr"])
    def test_a_masked_health_index_still_gives_a_diagnosis(self, fitted_diagnosers, kind):
        h = craft("PU-201", [1.0, 1.2, 1.5], "wear", mask_last_hi=True)
        d = fitted_diagnosers[kind].diagnose("PU-201", int(h.days[-1]), h)
        assert sum(d.posterior.values()) == pytest.approx(1.0)

    @pytest.mark.parametrize("kind", ["scm", "corr"])
    def test_a_short_history_is_enough(self, fitted_diagnosers, kind):
        h = craft("PU-201", [1.5], "bearing", n_old=0)
        d = fitted_diagnosers[kind].diagnose("PU-201", 0, h)
        assert sum(d.posterior.values()) == pytest.approx(1.0)

    @pytest.mark.parametrize("kind", ["scm", "corr"])
    def test_readings_after_the_day_asked_about_are_ignored(self, fitted_diagnosers, kind):
        h = craft("PU-201", [0.8, 1.0, 1.2, 4.0, 5.0], "wear")
        day = int(h.days[-3])
        a = fitted_diagnosers[kind].diagnose("PU-201", day, h)
        cut = ObservationHistory(h.asset_id, h.days[:-2], h.y[:-2], h.load[:-2])
        b = fitted_diagnosers[kind].diagnose("PU-201", day, cut)
        assert a.posterior == b.posterior

    @pytest.mark.parametrize("kind", ["scm", "corr"])
    def test_a_day_before_the_first_reading_is_an_error(self, fitted_diagnosers, kind):
        h = craft("PU-201", [1.0], "wear")
        with pytest.raises(ValueError, match="on or before"):
            fitted_diagnosers[kind].diagnose("PU-201", -5, h)


class TestWhatTheStructureSays:
    """Crafted, noise-free readings: the SCM must read them as the equations do."""

    def test_a_bearing_signature_means_a_bearing_mode_and_a_wear_signature_a_wear_mode(self, scm):
        b = scm.diagnose("PU-201", 14, craft("PU-201", [0.8, 1.0, 1.2, 1.5, 2.0], "bearing"))
        w = scm.diagnose("PU-201", 14, craft("PU-201", [0.8, 1.0, 1.2, 1.5, 2.0], "wear"))
        assert (b.mode, b.cause) == ("BEARING_DEFECT", "degradation")
        assert (w.mode, w.cause) == ("IMPELLER_WEAR", "degradation")

    def test_a_health_index_rise_with_no_signature_is_a_sensor_fault(self, scm):
        h = craft("PU-201", [0.0, 0.0, 0.0, 0.0, 0.0], "wear", g=1.5)
        d = scm.diagnose("PU-201", int(h.days[-1]), h)
        assert d.cause == "sensor_fault"

    def test_a_rise_that_the_load_explains_is_a_process_upset(self, scm):
        h = craft("PU-201", [0.0, 0.0, 0.0, 0.0, 0.0], "wear", dl=0.5)
        d = scm.diagnose("PU-201", int(h.days[-1]), h)
        assert d.cause == "process_upset"

    def test_the_same_rise_with_matching_signatures_is_degradation(self, scm):
        d = scm.diagnose("PU-201", 14, craft("PU-201", [1.0, 1.0, 1.0, 1.0, 1.0], "wear"))
        assert d.cause == "degradation"

    def test_more_evidence_for_a_sensor_fault_raises_its_probability(self, scm):
        small = scm.diagnose("PU-201", 14, craft("PU-201", [0.6] * 5, "wear", g=0.4))
        large = scm.diagnose("PU-201", 14, craft("PU-201", [0.0] * 5, "wear", g=1.0))
        assert (
            large.posterior["sensor_fault:IMPELLER_WEAR"]
            + large.posterior["sensor_fault:BEARING_DEFECT"]
            > small.posterior["sensor_fault:IMPELLER_WEAR"]
            + small.posterior["sensor_fault:BEARING_DEFECT"]
        )

    def test_the_mode_evidence_accumulates_over_the_window(self, scm):
        one = scm.diagnose("PU-201", 10, craft("PU-201", [1.0], "wear", n_old=10))
        five = scm.diagnose("PU-201", 14, craft("PU-201", [1.0] * 5, "wear", n_old=10))

        def p_wear(d):
            return sum(v for k, v in d.posterior.items() if k.endswith("IMPELLER_WEAR"))

        assert p_wear(five) > p_wear(one)


class TestPriorsAreUsed:
    def test_the_mode_prior_moves_the_posterior_toward_the_common_mode(self, scm):
        """Same readings, opposite priors: the posterior follows the prior (weak evidence: a sensor
        reading that fits neither family well)."""
        import copy

        h = craft("PU-201", [0.6] * 5, "wear")
        y = h.y.copy()
        y[-5:, 1:] = scm.sig_b  # signatures at their baseline value: neither family shows
        flat = ObservationHistory(h.asset_id, h.days, y, h.load)

        def wear_mass(prior_wear: float) -> float:
            other = copy.deepcopy(scm)
            other.mode_prior["SLURRY_PUMP"] = {
                "IMPELLER_WEAR": prior_wear,
                "BEARING_DEFECT": 1.0 - prior_wear,
            }
            d = other.diagnose("PU-201", int(flat.days[-1]), flat)
            return sum(v for k, v in d.posterior.items() if k.endswith("IMPELLER_WEAR"))

        assert wear_mass(0.99) > wear_mass(0.01) + 0.3

    def test_the_posterior_is_the_rule_of_the_docstring_written_out_again(self, scm):
        """Prior of the mode, times the cause-averaged likelihood of each earlier day in the window,
        times the cause frequency and the likelihood of today for each cause."""
        # weak evidence, so that the posterior is not saturated and a wrong factor shows
        h = craft("PU-201", [0.3, 0.4, 0.3, 0.5, 0.4, 0.5], "bearing", n_old=3)
        modes = class_modes("PU-201")
        k = h.n
        first = max(k - scm.window, 0)
        joint = np.zeros((len(CAUSES), len(modes)))
        for jm, (m, fam) in enumerate(modes):
            ev = scm.mode_prior["SLURRY_PUMP"][m]
            for j in range(first, k - 1):
                like = scm._day_likelihood(h.y[j], float(h.load[j]), fam)
                ev *= sum(scm.pi[i] * like[c] for i, c in enumerate(CAUSES))
            today = scm._day_likelihood(h.y[k - 1], float(h.load[k - 1]), fam)
            for ic, c in enumerate(CAUSES):
                joint[ic, jm] = scm.pi[ic] * ev * today[c]
        joint /= joint.sum()
        d = scm.diagnose("PU-201", int(h.days[-1]), h)
        for ic, c in enumerate(CAUSES):
            for jm, (m, _) in enumerate(modes):
                assert d.posterior[f"{c}:{m}"] == pytest.approx(joint[ic, jm], rel=1e-9, abs=1e-12)

    def test_the_window_is_what_it_says(self, scm):
        """Readings older than the window change nothing; the window's first reading does."""
        base = craft("PU-201", [0.7, 0.9, 1.2, 1.4, 1.6, 1.8], "bearing", n_old=3)
        y = base.y.copy()
        y[: base.n - scm.window] += 5.0  # wreck everything older than the window
        old_changed = ObservationHistory(base.asset_id, base.days, y, base.load)
        assert (
            scm.diagnose("PU-201", 8, base).posterior
            == scm.diagnose("PU-201", 8, old_changed).posterior
        )
        y2 = base.y.copy()
        y2[base.n - scm.window, 1:] += 1.0  # the oldest reading inside the window
        in_window = ObservationHistory(base.asset_id, base.days, y2, base.load)
        assert (
            scm.diagnose("PU-201", 8, base).posterior
            != scm.diagnose("PU-201", 8, in_window).posterior
        )

    def test_a_large_load_excursion_is_evidence_for_an_upset_and_against_the_rest(self, scm):
        """The same health-index excess and signatures, once with the load at its usual value and
        once with a large excursion: the load is an effect of the cause, so it shifts the odds."""

        def odds(dl: float) -> float:
            h = craft("PU-201", [0.5], "wear", dl=dl, n_old=0)
            like = scm._day_likelihood(h.y[-1], float(h.load[-1]), "wear")
            return like["process_upset"] / like["degradation"]

        assert odds(0.5) > 100.0 * odds(0.0)

    def test_the_conveyor_has_one_mode_whatever_the_signatures_say(self, scm):
        d = scm.diagnose("CV-101", 14, craft("CV-101", [1.0] * 5, "wear"))
        assert {k.split(":")[1] for k in d.posterior} == {"BEARING_DEFECT"}


class TestTheSCMEstimatesItsCoefficients:
    def test_baseline_and_load_coefficients_are_estimates_near_the_plants(self, scm):
        assert scm.h0 != PLANT.hi_base and scm.h0 == pytest.approx(PLANT.hi_base, abs=0.1)
        assert scm.c_h != PLANT.hi_load_coef and scm.c_h == pytest.approx(
            PLANT.hi_load_coef, abs=0.4
        )
        assert scm.sd_h == pytest.approx(PLANT.hi_noise_sd, abs=0.1)
        assert np.allclose(scm.sig_sd, PLANT.sig_noise_sd, atol=0.1)

    def test_the_signature_gains_are_estimates_near_the_plants_per_unit_of_index(self, scm):
        span = PLANT.hi_fail - PLANT.hi_base
        want_b = np.array(PLANT.sig_loading_bearing) / span
        want_w = np.array(PLANT.sig_loading_wear) / span
        assert np.allclose(scm.gains["bearing"], want_b, atol=0.05)
        assert np.allclose(scm.gains["wear"], want_w, atol=0.05)
        assert not np.array_equal(scm.gains["bearing"], want_b)

    def test_the_mode_priors_are_the_label_frequencies_of_each_class(self, scm):
        assert scm.mode_prior["SLURRY_PUMP"]["IMPELLER_WEAR"] == pytest.approx(0.7, abs=0.1)
        assert scm.mode_prior["CRUSHER"]["LINER_WEAR"] == pytest.approx(0.7, abs=0.15)
        assert scm.mode_prior["BELT_CONVEYOR"]["BEARING_DEFECT"] == 1.0
        for p in scm.mode_prior.values():
            assert sum(p.values()) == pytest.approx(1.0)

    def test_the_cause_frequencies_come_from_the_labelled_alerts(self, scm):
        assert scm.pi.sum() == pytest.approx(1.0)
        assert scm.pi[CAUSES.index("degradation")] > 0.6
        assert scm.n_alerts["degradation"] > scm.n_alerts["sensor_fault"]

    def test_the_load_of_an_upset_is_larger_than_that_of_a_degrading_asset(self, scm):
        assert scm.dl_mu["process_upset"] > scm.dl_mu["degradation"] + 0.1

    def test_the_priors_are_distributions(self, scm):
        for p in (*scm.pz.values(), scm.pg):
            assert p.sum() == pytest.approx(1.0) and p.min() > 0.0

    def test_lives_with_a_sensor_fault_do_not_enter_the_gain_regression(self, diag_train):
        """A life whose health index sits 5 units too high would, taken at its word, say that wear
        produces almost no signature. The labelled sensor-fault lives are left out of the gains."""
        from closed_loop.causal.records import AlertLabel

        clean = ScmDiagnoser()
        clean.fit(diag_train)
        wear_lives = [lf for lf in diag_train if lf.mode in ("IMPELLER_WEAR", "LINER_WEAR")][:25]
        faulty = [
            dataclasses.replace(
                lf,
                hi=lf.hi + 5.0,
                alerts=(AlertLabel(f"{lf.asset_id}-L0-A1", 30, 1, "sensor_fault", lf.mode),),
            )
            for lf in wear_lives
        ]
        mixed = ScmDiagnoser()
        mixed.fit([*diag_train, *faulty])
        assert np.allclose(
            mixed.gains["wear"], clean.gains["wear"], atol=0.01
        )  # baseline moves a little
        assert np.allclose(mixed.gains["bearing"], clean.gains["bearing"], atol=0.01)

    def test_the_fit_reads_labels_not_the_generators_parameters(self):
        """Lives from a plant with different coefficients give different estimates."""
        other = dataclasses.replace(PlantParams(), hi_noise_sd=0.5, mode_load_tilt=8.0)
        lives = mdiag.collect_lives(range(1000, 1060), other)
        d = ScmDiagnoser()
        d.fit(lives)
        assert d.sd_h == pytest.approx(0.5, abs=0.12)


class TestInvarianceToTheLoadModeAssociation:
    """The point of the structure: the mode does not depend on the load, so neither does the SCM."""

    def test_the_scm_ignores_the_load_history_older_than_its_window(self, scm):
        light = craft("PU-201", [0.8, 1.0, 1.2, 1.5, 2.0], "wear", old_load=-0.3)
        heavy = craft("PU-201", [0.8, 1.0, 1.2, 1.5, 2.0], "wear", old_load=+0.3)
        a = scm.diagnose("PU-201", 14, light)
        b = scm.diagnose("PU-201", 14, heavy)
        assert a.posterior == b.posterior

    def test_the_correlational_model_does_not(self, corr):
        # the last five readings (the window) are identical; only the older load history differs
        light = craft("PU-201", [0.8, 1.0, 1.2, 1.4, 1.6], "wear", old_load=-0.3, n_old=40)
        heavy = craft("PU-201", [0.8, 1.0, 1.2, 1.4, 1.6], "wear", old_load=+0.3, n_old=40)
        a = corr.diagnose("PU-201", int(light.days[-1]), light)
        b = corr.diagnose("PU-201", int(heavy.days[-1]), heavy)
        assert a.posterior != b.posterior

    def test_trained_where_heavy_means_wear_it_leans_toward_wear_on_a_heavy_asset(self, corr):
        def p_wear(old_load):
            h = craft("PU-201", [0.9, 1.0, 1.1, 1.2, 1.3], "wear", old_load=old_load, n_old=40)
            d = corr.diagnose("PU-201", int(h.days[-1]), h)
            return sum(v for k, v in d.posterior.items() if k.endswith("IMPELLER_WEAR"))

        assert p_wear(+0.3) > p_wear(-0.3)

    def test_the_lever_really_builds_the_association_and_reversing_it_reverses_it(self):
        def corr_load_wear(tilt):
            lives = mdiag.collect_lives(range(1000, 1100), mdiag.diagnosis_params(tilt))
            pumps = [lf for lf in lives if lf.asset_id in ("PU-201", "PU-202", "CR-501")]
            load = np.array([lf.load[:20].mean() for lf in pumps])
            wear = np.array([float("WEAR" in lf.mode) for lf in pumps])
            return float(np.corrcoef(load, wear)[0, 1])

        pos, neg = corr_load_wear(+mdiag.DEFAULT_TILT), corr_load_wear(-mdiag.DEFAULT_TILT)
        assert pos > 0.15 and neg < -0.15


class TestOracle:
    def test_it_needs_a_lookup(self):
        h = craft("PU-201", [1.0], "wear")
        with pytest.raises(RuntimeError, match="bind"):
            OracleDiagnoser().diagnose("PU-201", 10, h)

    def test_it_returns_the_truth_with_a_one_hot_posterior(self):
        o = OracleDiagnoser()
        o.bind(lambda a, d: ("IMPELLER_WEAR", "degradation"))
        d = o.diagnose("PU-201", 10, craft("PU-201", [1.0], "wear"))
        assert (d.mode, d.cause) == ("IMPELLER_WEAR", "degradation")
        assert d.posterior["degradation:IMPELLER_WEAR"] == 1.0 and sum(d.posterior.values()) == 1.0

    def test_fitting_it_does_nothing(self):
        assert OracleDiagnoser().fit([]) is None

    def test_the_table_returns_the_latest_alert_on_or_before_the_day(self):
        t = AlertTruthTable({"PU-201": [(40, "A", "degradation"), (60, "B", "degradation")]})
        assert t("PU-201", 40) == ("A", "degradation")
        assert t("PU-201", 59) == ("A", "degradation")
        assert t("PU-201", 70) == ("B", "degradation")
        with pytest.raises(KeyError):
            t("PU-201", 39)
        with pytest.raises(KeyError):
            t("CR-501", 70)

    def test_the_table_can_grow(self):
        t = AlertTruthTable()
        t.add("PU-201", 50, "M", "sensor_fault")
        t.add("PU-201", 30, "N", "degradation")
        assert t("PU-201", 45) == ("N", "degradation") and t("PU-201", 51) == ("M", "sensor_fault")


class TestOnHeldOutAlerts:
    @pytest.mark.parametrize("kind", ["scm", "corr"])
    def test_both_diagnosers_are_mostly_right_in_distribution(
        self, fitted_diagnosers, held_out, kind
    ):
        out = mdiag.evaluate(fitted_diagnosers[kind], held_out)
        assert len(held_out) > 150
        assert out.correct.mean() > 0.8
        assert (
            out.mode_ok[np.array([k.truth_cause == "degradation" for k in held_out])].mean() > 0.95
        )

    def test_the_oracle_is_always_right(self, held_out):
        out = mdiag.evaluate(OracleDiagnoser(), held_out)
        assert out.correct.all()

    def test_the_diagnosis_day_is_the_alert_day(self, fitted_diagnosers, held_out):
        k = held_out[0]
        d = fitted_diagnosers["scm"].diagnose(k.asset_id, k.day, k.history)
        assert d.day == k.day == int(k.history.days[-1])
