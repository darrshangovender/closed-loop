"""P7: observational lives under an operator, and twin-world ground truth."""

import dataclasses

import numpy as np
import pytest

from closed_loop.causal.twin import (
    CAUSAL_HORIZON_DAYS,
    MAX_TRUTH_DAY,
    Query,
    causal_params,
    generate_observational,
    passive_world,
    run_world,
    twin_truth,
    twin_truths,
)
from closed_loop.core.types import Fidelity
from closed_loop.plant import PlantParams, Scenario, make_plant
from closed_loop.plant.params import ASSET_IDS


class TestCausalParams:
    def test_it_is_the_f1d_study_plant(self):
        p = causal_params()
        assert p.derate_enabled and p.p_fail_in_episode == 1.0
        assert p.horizon_days == CAUSAL_HORIZON_DAYS == 150
        assert p.p_refuse == 0.0

    def test_the_tilt_and_other_fields_can_be_set(self):
        p = causal_params(tilt=-5.0, hi_noise_sd=0.1)
        assert p.mode_load_tilt == -5.0 and p.hi_noise_sd == 0.1

    def test_the_base_is_not_modified(self):
        base = PlantParams()
        causal_params(base, tilt=3.0)
        assert base.mode_load_tilt == 0.0 and not base.derate_enabled

    def test_an_invalid_override_is_refused(self):
        with pytest.raises(ValueError):
            causal_params(p_fail_in_episode=2.0)


@pytest.fixture(scope="module")
def params():
    return causal_params()


class TestRunWorld:
    def test_four_lives_one_per_asset(self, params):
        lives = run_world(1000, params, "pi_obs")
        assert [lf.asset_id for lf in lives] == list(ASSET_IDS)
        assert all(lf.seed == 1000 and lf.life_id == 0 for lf in lives)

    def test_the_run_is_deterministic(self, params):
        a = run_world(1001, params, "pi_obs")
        b = run_world(1001, params, "pi_obs")
        for x, y in zip(a, b, strict=True):
            assert x.fail_day == y.fail_day and x.derate_start == y.derate_start
            assert np.array_equal(x.hi, y.hi) and np.array_equal(x.load, y.load)

    def test_unknown_policy_is_refused(self, params):
        with pytest.raises(ValueError, match="policy"):
            run_world(1000, params, "magic")

    def test_a_derating_operator_needs_the_derate_command(self):
        with pytest.raises(ValueError, match="derate_enabled"):
            run_world(1000, PlantParams(), "pi_obs")
        run_world(1000, PlantParams(), "none")  # nobody acting needs no command

    def test_nobody_acting_derates_nobody(self, params):
        for lf in run_world(1002, params, "none"):
            assert lf.derate_start is None and lf.derate_days.size == 0

    def test_the_operators_choice_does_not_disturb_the_exogenous_streams(self, params):
        """Until the first derate a derating and a passive world have the same readings."""
        a = run_world(1003, params, "none")
        b = run_world(1003, params, "pi_obs")
        for x, y in zip(a, b, strict=True):
            first = y.derate_start if y.derate_start is not None else 10**6
            n = int(min(first, x.days.size, y.days.size))
            assert n > 20
            assert np.array_equal(x.hi_days[x.hi_days < n], y.hi_days[y.hi_days < n])
            assert np.array_equal(x.sig[:, :n], y.sig[:, :n])
            assert np.array_equal(x.load[:n], y.load[:n])

    def test_the_confounded_operator_derates_when_the_smoothed_index_is_high(self, params):
        n_derated = 0
        for seed in (1000, 1001, 1002):
            for lf in run_world(seed, params, "pi_obs"):
                if lf.derate_start is None:
                    continue
                n_derated += 1
                d = int(lf.derate_start)
                before = lf.hi[lf.hi_days <= d][-3:]
                assert before.mean() > 3.0 - 1e-9  # the rule: smoothed index above theta
        assert n_derated >= 6

    def test_the_operator_looks_only_now_and_then_so_the_derate_lags_the_crossing(self, params):
        """Each day the operator looks with probability p_check = 0.2: the derate comes a random
        number of days after the smoothed index first passes the threshold, not on the day."""
        lags = []
        for seed in (1000, 1001, 1002, 1003):
            for lf in run_world(seed, params, "pi_obs"):
                if lf.derate_start is None:
                    continue
                smooth = np.convolve(lf.hi, np.ones(3) / 3.0, mode="valid")
                days = lf.hi_days[2:]
                crossed = days[smooth > 3.0]
                lags.append(int(lf.derate_start) - int(crossed[0]))
        assert len(lags) >= 10 and min(lags) >= 0
        assert max(lags) >= 3 and np.mean(lags) > 1.0

    def test_a_derate_shows_in_the_observed_load_and_in_the_derate_days(self, params):
        lf = next(x for x in run_world(1000, params, "pi_obs") if x.derate_start is not None)
        d1 = int(lf.derate_days[0])
        assert (
            lf.derate_days[0] == np.floor(lf.derate_start) + 1
            or lf.derate_days[0] >= lf.derate_start
        )
        i = int(np.searchsorted(lf.days, d1))
        assert lf.load[i] / lf.load[i - 1] == pytest.approx(0.7, abs=0.08)
        assert np.all(np.diff(lf.derate_days) == 1.0)  # a derate is for the rest of the life

    def test_the_randomised_policy_derates_at_the_first_alert(self, params):
        seen = 0
        for seed in range(1000, 1006):
            for lf in run_world(seed, params, "randomised"):
                if lf.derate_start is None:
                    continue
                seen += 1
                assert lf.derate_start == pytest.approx(lf.first_alert_day(), abs=0.05)
        assert seen >= 4

    def test_the_randomised_coin_is_independent_of_the_asset_state(self, params):
        shares = [
            np.mean(
                [
                    lf.derate_start is not None
                    for seed in range(1000, 1015)
                    for lf in run_world(seed, params, "randomised")
                    if lf.first_alert_day() is not None
                ]
            )
        ]
        assert 0.3 < shares[0] < 0.7

    def test_failure_day_is_the_plants_and_the_record_stops_there(self, params):
        for lf in run_world(1004, params, "pi_obs"):
            if lf.fail_day is None:
                assert lf.censored_day == float(CAUSAL_HORIZON_DAYS)
            else:
                assert lf.days[-1] <= lf.fail_day < lf.days[-1] + 1.0
                assert lf.censored_day is None

    def test_alerts_carry_the_post_mortem_labels(self, params):
        labelled = [a for lf in run_world(1005, params, "none") for a in lf.alerts]
        assert labelled and all(a.level in (1, 2) for a in labelled)
        assert {a.cause for a in labelled} <= {"degradation", "sensor_fault", "process_upset"}
        assert all(a.alert_id.endswith(f"-A{a.level}") for a in labelled)


class TestPassiveWorld:
    @pytest.mark.parametrize("seed", [1000, 1007, 1019])
    def test_it_equals_the_stepped_plant(self, seed):
        p = causal_params(derate_enabled=False, p_fail_in_episode=0.6, horizon_q=8640, tilt=5.0)
        for x, y in zip(run_world(seed, p, "none"), passive_world(seed, p), strict=True):
            assert x.asset_id == y.asset_id and x.mode == y.mode and x.alerts == y.alerts
            assert x.fail_day == y.fail_day and x.censored_day == y.censored_day
            assert np.array_equal(x.days, y.days) and np.array_equal(x.hi_days, y.hi_days)
            assert np.array_equal(x.hi, y.hi) and np.array_equal(x.sig, y.sig)
            assert np.array_equal(x.load, y.load)

    def test_it_equals_the_stepped_plant_with_a_shifted_scenario(self):
        p = causal_params(derate_enabled=False)
        sc = Scenario.shifted()
        for x, y in zip(run_world(1010, p, "none", sc), passive_world(1010, p, sc), strict=True):
            assert x.fail_day == y.fail_day and x.alerts == y.alerts
            assert np.array_equal(x.hi, y.hi)


class TestGenerateObservational:
    def test_n_lives_from_consecutive_seeds(self):
        lives = generate_observational(10, 1100, "none", params=causal_params())
        assert len(lives) == 10
        assert [lf.seed for lf in lives] == [1100] * 4 + [1101] * 4 + [1102] * 2

    def test_it_is_deterministic(self):
        a = generate_observational(4, 1100, "pi_obs")
        b = generate_observational(4, 1100, "pi_obs")
        assert all(np.array_equal(x.hi, y.hi) for x, y in zip(a, b, strict=True))

    def test_n_must_be_positive(self):
        with pytest.raises(ValueError):
            generate_observational(0, 1)


class TestQuery:
    def test_validation(self):
        with pytest.raises(ValueError, match="kind"):
            Query("jump", "PU-201")
        with pytest.raises(ValueError, match="asset"):
            Query("nothing", "XX-1")
        with pytest.raises(ValueError, match="day"):
            Query("derate", "PU-201", -1.0)
        with pytest.raises(ValueError, match="load_cap"):
            Query("derate", "PU-201", 1.0, 0.0)


class TestTwinTruth:
    def test_nothing_is_the_plants_no_intervention_failure(self, params):
        plant = make_plant(params, fidelity=Fidelity.F1D)
        plant.reset(1200, Scenario())
        truth = plant.truth.failure_time_q
        for aid in ASSET_IDS:
            got = twin_truth(1200, Query("nothing", aid), params=params)
            assert got.failure_day == pytest.approx(truth[aid] / 96.0)
            assert got.applied_day is None

    def test_a_derate_postpones_the_failure(self, params):
        for aid in ASSET_IDS:
            n = twin_truth(1201, Query("nothing", aid), params=params).failure_day
            d = twin_truth(1201, Query("derate", aid, 30.0), params=params)
            assert d.failure_day > n
            assert d.applied_day == 30.0

    def test_an_earlier_derate_postpones_it_further(self, params):
        t20 = twin_truth(1202, Query("derate", "PU-201", 20.0), params=params).failure_day
        t40 = twin_truth(1202, Query("derate", "PU-201", 40.0), params=params).failure_day
        assert t20 > t40

    def test_the_gain_is_about_a_fifth_of_the_remaining_clock(self, params):
        """exp(0.8 (0.7 - 1)) = 0.79: the clock runs a fifth slower, so the gain is about 27% of
        the remaining time at the moment of the derate."""
        n = twin_truth(1203, Query("nothing", "CR-501"), params=params).failure_day
        t0 = 30.0
        d = twin_truth(1203, Query("derate", "CR-501", t0), params=params).failure_day
        remaining = n - t0
        assert (d - n) / remaining == pytest.approx(1 / np.exp(0.8 * -0.3) - 1, rel=0.2)

    def test_a_replacement_before_the_failure_moves_it_to_the_new_component(self, params):
        n = twin_truth(1204, Query("nothing", "PU-202"), params=params).failure_day
        r = twin_truth(1204, Query("replace", "PU-202", 40.0), params=params)
        assert n > 40.0 and r.failure_day > n + 20.0  # a new part: onset 24+ days away

    def test_a_replacement_after_the_failure_leaves_the_old_failure(self, params):
        n = twin_truth(1204, Query("nothing", "PU-202"), params=params).failure_day
        r = twin_truth(1204, Query("replace", "PU-202", min(n + 5.0, 149.0)), params=params)
        assert r.failure_day == pytest.approx(n)

    def test_other_assets_are_unmoved_by_an_intervention_on_one(self, params):
        alone = twin_truths(
            1205, [Query("nothing", "PU-201"), Query("nothing", "CV-101")], params=params
        )
        both = twin_truths(
            1205,
            [
                Query("derate", "CR-501", 30.0),
                Query("nothing", "PU-201"),
                Query("nothing", "CV-101"),
            ],
            params=params,
        )
        assert both[1].failure_day == alone[0].failure_day
        assert both[2].failure_day == alone[1].failure_day

    def test_a_batch_equals_the_queries_one_by_one(self, params):
        qs = [
            Query("derate", "CR-501", 35.0),
            Query("derate", "PU-201", 50.0),
            Query("replace", "CV-101", 60.0),
            Query("nothing", "PU-202"),
        ]
        batch = twin_truths(1206, qs, params=params)
        for q, b in zip(qs, batch, strict=True):
            assert twin_truth(1206, q, params=params).failure_day == b.failure_day

    def test_two_derates_of_one_asset_are_two_separate_worlds(self, params):
        qs = [Query("derate", "PU-201", 60.0), Query("derate", "PU-201", 20.0)]
        a = twin_truths(1207, qs[:1], params=params)[0].failure_day
        b = twin_truths(1207, qs[1:], params=params)[0].failure_day
        both = twin_truths(1207, qs, params=params)
        assert [x.query for x in both] == qs
        assert both[0].failure_day == a and both[1].failure_day == b

    def test_a_derate_with_another_retained_load_is_refused(self, params):
        with pytest.raises(ValueError, match="derates to"):
            twin_truth(1200, Query("derate", "PU-201", 30.0, 0.5), params=params)

    def test_an_intervention_past_the_episode_is_refused(self, params):
        with pytest.raises(ValueError, match="past the end"):
            twin_truth(1200, Query("derate", "PU-201", 151.0), params=params)

    def test_the_truth_is_capped(self):
        assert MAX_TRUTH_DAY == 400.0

    def test_a_refused_command_is_repeated_until_it_executes(self):
        busy = causal_params(p_refuse=0.9)
        free = causal_params()
        got = twin_truth(1208, Query("derate", "PU-201", 40.0), params=busy)
        assert got.applied_day is not None and got.applied_day >= 40.0
        # the same intervention at the day it really took effect, in a world with no refusals
        ref = twin_truth(1208, Query("derate", "PU-201", got.applied_day), params=free)
        assert got.failure_day == pytest.approx(ref.failure_day, abs=1e-9)

    def test_the_truth_does_not_depend_on_which_other_queries_share_the_batch(self, params):
        a = twin_truths(1209, [Query("derate", "PU-201", 45.0)], params=params)[0]
        b = twin_truths(
            1209,
            [
                Query("derate", "PU-201", 45.0),
                Query("derate", "CR-501", 10.0),
                Query("replace", "CV-101", 80.0),
            ],
            params=params,
        )[0]
        assert a.failure_day == b.failure_day


def test_the_query_dataclass_is_frozen():
    q = Query("nothing", "PU-201")
    with pytest.raises(dataclasses.FrozenInstanceError):
        q.day = 3.0  # type: ignore[misc]
