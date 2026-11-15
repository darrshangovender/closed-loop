"""P3a: the observation chain y_d = (hi, sig_bearing, sig_proc, sig_hyd) and its causes (A3)."""

import math

import numpy as np
import pytest

from closed_loop.core.timegrid import DAY_Q
from closed_loop.plant.degradation import build_physical
from closed_loop.plant.observe import (
    CAUSES,
    ObsStreams,
    alert_cause,
    blind_until_day,
    build_obs_streams,
    is_blind,
    observe_day,
    simulate_lives,
)
from closed_loop.plant.params import ASSET_IDS, PlantParams
from closed_loop.plant.scenario import Scenario

from ._helpers import quiet_params, theta_of

P = PlantParams()
Q = quiet_params()
N = 90


def zero_streams(drift: float = 0.0, masked: bool = False) -> ObsStreams:
    return ObsStreams(
        hi_noise=np.zeros(N),
        sig_noise=np.zeros((3, N)),
        load_noise=np.zeros(N),
        masked=np.full(N, masked),
        drift_frac=np.full(N, drift),
        baseline_fault_day=None,
    )


def read(sev: float, signature: str, load: float = 1.0, drift: float = 0.0):
    return observe_day(Q, zero_streams(drift), 5, sev, signature, load)


class TestSignatureChannels:
    """The five statements of 0003 A3, one test each."""

    def test_a_bearing_fault_moves_sig_bearing_not_the_process_channels(self):
        healthy, faulty = read(0.0, "bearing"), read(1.0, "bearing")
        d_b = faulty.sig_bearing - healthy.sig_bearing
        d_p = faulty.sig_proc - healthy.sig_proc
        d_h = faulty.sig_hyd - healthy.sig_hyd
        assert d_b == pytest.approx(Q.sig_loading_bearing[0])
        assert d_b > 5 * abs(d_p)  # only the small stated leak reaches the process channel
        assert d_h == 0.0

    def test_wear_moves_sig_proc_and_sig_hyd_not_sig_bearing(self):
        healthy, worn = read(0.0, "wear"), read(1.0, "wear")
        d_b = worn.sig_bearing - healthy.sig_bearing
        assert worn.sig_proc - healthy.sig_proc == pytest.approx(Q.sig_loading_wear[1])
        assert worn.sig_hyd - healthy.sig_hyd == pytest.approx(Q.sig_loading_wear[2])
        assert d_b < 0.25 * (worn.sig_proc - healthy.sig_proc)

    def test_a_sensor_gain_fault_moves_hi_without_any_signature_movement(self):
        clean, drifting = read(0.3, "bearing"), read(0.3, "bearing", drift=0.1)
        assert drifting.hi is not None and clean.hi is not None
        assert drifting.hi - clean.hi == pytest.approx(0.1 * Q.hi_fail)
        assert (drifting.sig_bearing, drifting.sig_proc, drifting.sig_hyd) == (
            clean.sig_bearing,
            clean.sig_proc,
            clean.sig_hyd,
        )
        assert drifting.load == clean.load

    def test_a_process_upset_moves_every_channel_through_load(self):
        calm, upset = read(0.2, "wear", load=1.0), read(0.2, "wear", load=1.5)
        assert calm.hi is not None and upset.hi is not None
        assert upset.hi - calm.hi == pytest.approx(Q.hi_load_coef * 0.5)
        assert upset.sig_bearing - calm.sig_bearing == pytest.approx(Q.sig_load_coef[0] * 0.5)
        assert upset.sig_proc - calm.sig_proc == pytest.approx(Q.sig_load_coef[1] * 0.5)
        assert upset.sig_hyd - calm.sig_hyd == pytest.approx(Q.sig_load_coef[2] * 0.5)
        assert upset.load - calm.load == pytest.approx(0.5)  # the load itself is observed

    def test_the_three_causes_leave_distinct_fingerprints(self):
        base = read(0.2, "bearing")
        deg = read(0.8, "bearing")
        sens = read(0.2, "bearing", drift=0.2)
        proc = read(0.2, "bearing", load=1.5)

        def moves(o):
            assert o.hi is not None and base.hi is not None
            return (
                o.hi - base.hi > 0.5,
                o.sig_bearing - base.sig_bearing > 0.5,
                o.sig_proc - base.sig_proc > 0.5,
                o.load - base.load > 0.2,
            )

        assert moves(deg) == (True, True, False, False)
        assert moves(sens) == (True, False, False, False)
        assert moves(proc) == (True, True, True, True)

    def test_hi_is_monotone_in_severity_before_noise(self):
        for sig in ("bearing", "wear"):
            his = [read(s, sig).hi for s in np.linspace(0, 1, 21)]
            assert all(b > a for a, b in zip(his, his[1:], strict=False))  # type: ignore[operator]
            assert his[0] == pytest.approx(Q.hi_base) and his[-1] == pytest.approx(Q.hi_fail)

    def test_noise_scales_with_its_sd(self):
        streams = ObsStreams(
            hi_noise=np.full(N, 2.0),
            sig_noise=np.full((3, N), -1.0),
            load_noise=np.full(N, 3.0),
            masked=np.zeros(N, dtype=bool),
            drift_frac=np.zeros(N),
            baseline_fault_day=None,
        )
        p = PlantParams()
        o = observe_day(p, streams, 0, 0.0, "bearing", 1.0)
        assert o.hi == pytest.approx(p.hi_base + 2.0 * p.hi_noise_sd)
        assert o.sig_bearing == pytest.approx(-p.sig_noise_sd)
        assert o.load == pytest.approx(1.0 + 3.0 * p.load_obs_sd)


class TestMaskingAndTheTuple:
    def test_a_masked_day_has_no_hi_but_keeps_the_signatures(self):
        o = observe_day(Q, zero_streams(masked=True), 3, 0.5, "wear", 1.0)
        assert o.hi is None
        assert o.sig_proc > 0
        assert math.isnan(o.y[0]) and o.y[1:] == (o.sig_bearing, o.sig_proc, o.sig_hyd)

    def test_y_is_the_documented_four_tuple(self):
        o = read(0.5, "bearing")
        assert o.y == (o.hi, o.sig_bearing, o.sig_proc, o.sig_hyd)
        assert len(o.y) == 4


class TestAlertCause:
    def test_the_largest_contribution_names_the_cause(self):
        s_clean = zero_streams()
        assert alert_cause(Q, s_clean, 5, sev=0.6, load=1.0) == "degradation"
        assert alert_cause(Q, s_clean, 5, sev=0.0, load=1.6) == "process_upset"
        assert alert_cause(Q, zero_streams(drift=0.3), 5, sev=0.05, load=1.0) == "sensor_fault"

    def test_every_label_is_a_known_cause(self):
        for sev, load, drift in [
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.5),
            (0.0, 0.5, 0.0),
            (0.3, 1.3, 0.1),
        ]:
            assert alert_cause(Q, zero_streams(drift), 1, sev, load) in CAUSES


class TestDataQualityStreams:
    def phys(self, seed: int, scenario: Scenario | None = None, params: PlantParams = P):
        scenario = scenario if scenario is not None else Scenario()
        return build_physical(params, seed, scenario.resolve(seed, params))

    def test_stuck_and_spike_days_occur_at_about_their_rates(self):
        masked = 0
        total = 0
        for seed in range(150):
            phys = self.phys(seed)
            for aid in ASSET_IDS:
                s = build_obs_streams(P, seed, aid, phys)
                masked += int(s.masked.sum())
                total += N
        rate = masked / total
        expected = 1 - (1 - P.p_stuck_day) * (1 - P.p_spike_day)
        assert abs(rate - expected) < 0.005, (rate, expected)

    def test_a_baseline_gain_fault_appears_at_its_rate_and_is_not_masked(self):
        k, total = 0, 0
        for seed in range(400):
            phys = self.phys(seed)
            for aid in ASSET_IDS:
                s = build_obs_streams(P, seed, aid, phys)
                total += 1
                if s.baseline_fault_day is not None:
                    k += 1
                    assert s.drift_frac[-1] > 0
                    assert s.drift_frac[int(s.baseline_fault_day)] == 0.0
                else:
                    assert not s.drift_frac.any()
        assert abs(k / total - P.p_gain_fault) < 0.02, (k, total)

    def test_adversary_stuck_masks_the_documented_run_of_days(self):
        sc = Scenario.decode(theta_of(c3=0.3, c4=0.4), "in")  # asset 0, type 1 (stuck), day 0.4*89
        phys = self.phys(1, sc, Q)
        s = build_obs_streams(Q, 1, "CR-501", phys)
        fday = phys.resolved.fault_day[0]
        days = np.flatnonzero(s.masked)
        assert list(days) == [d for d in range(N) if fday <= d < fday + Q.stuck_days]
        assert not build_obs_streams(Q, 1, "CV-101", phys).masked.any()

    def test_adversary_spike_masks_one_day(self):
        sc = Scenario.decode(theta_of(c3=0.55, c4=0.7), "in")  # type 2
        phys = self.phys(1, sc, Q)
        assert build_obs_streams(Q, 1, "CR-501", phys).masked.sum() == 1

    def test_drift_fault_is_a_beyond_assumption_fault(self):
        th = theta_of(c3=0.9, c4=0.2)
        phys_in = self.phys(2, Scenario.decode(th, "in"), Q)
        phys_out = self.phys(2, Scenario.decode(th, "out"), Q)
        assert not build_obs_streams(Q, 2, "CR-501", phys_in).drift_frac.any()
        d = build_obs_streams(Q, 2, "CR-501", phys_out).drift_frac
        assert d[-1] == pytest.approx(Q.shifted_drift_per_day * (N - 1 - 0.2 * (N - 1)))
        assert not build_obs_streams(Q, 2, "CV-101", phys_out).drift_frac.any()

    def test_the_shifted_condition_drifts_every_channel_after_the_shift(self):
        phys = self.phys(3, Scenario.shifted(), Q)
        day0 = phys.resolved.drift_all_day
        for aid in ASSET_IDS:
            d = build_obs_streams(Q, 3, aid, phys).drift_frac
            assert d[: int(day0)].max() == 0.0
            assert d[-1] == pytest.approx(0.005 * (N - 1 - day0))

    def test_noise_streams_are_keyed_per_asset_and_seed(self):
        phys = self.phys(5)
        a = build_obs_streams(P, 5, "PU-201", phys)
        b = build_obs_streams(P, 5, "PU-202", phys)
        a2 = build_obs_streams(P, 5, "PU-201", phys)
        assert np.array_equal(a.hi_noise, a2.hi_noise)
        assert not np.array_equal(a.hi_noise, b.hi_noise)
        assert abs(np.corrcoef(a.hi_noise, b.hi_noise)[0, 1]) < 0.4
        assert not np.array_equal(
            a.hi_noise, build_obs_streams(P, 6, "PU-201", self.phys(6)).hi_noise
        )

    def test_noise_arrays_are_read_only(self):
        s = build_obs_streams(P, 1, "CR-501", self.phys(1))
        with pytest.raises(ValueError):
            s.hi_noise[0] = 0.0


class TestBlindWindow:
    def test_blind_for_23_days_after_a_life_starts(self):
        assert blind_until_day(0.0, P) == 23.0
        assert blind_until_day(40.0, P) == 63.0
        assert is_blind(22.99, 0.0, P) and not is_blind(23.0, 0.0, P)
        assert is_blind(62.0, 40.0, P) and not is_blind(63.5, 40.0, P)

    def test_follows_blind_days(self):
        assert blind_until_day(10.0, PlantParams(blind_days=5.0)) == 15.0


class TestLifeTableReadings:
    def test_the_table_reading_equals_a_direct_observe_day_call(self):
        # the plant and the table share one code path
        seed = 4
        t = simulate_lives(P, seed, Scenario())
        phys = build_physical(P, seed, Scenario().resolve(seed, P))
        for aid in ASSET_IDS:
            life = t.lives[aid]
            s = build_obs_streams(P, seed, aid, phys)
            for d in (0, 17, 45, 89):
                o = observe_day(P, s, d, float(t.severity[aid][d]), life.signature,
                                float(phys.load_day[aid][d]))  # fmt: skip
                want = np.nan if o.hi is None else o.hi
                got = t.hi[aid][d]
                assert (math.isnan(want) and math.isnan(got)) or got == want
                assert tuple(t.sig[aid][:, d]) == (o.sig_bearing, o.sig_proc, o.sig_hyd)

    def test_hi_reaches_the_failure_level_where_a_noise_free_life_fails(self):
        quiet = quiet_params()
        t = simulate_lives(quiet, 3, Scenario())
        for aid in ASSET_IDS:
            if t.fails_in_horizon(aid):
                day = int(t.failure_day[aid])
                # the reading just before failure is close to hi_fail
                assert t.hi[aid][day] > quiet.hi_fail * 0.6

    def test_healthy_assets_read_near_the_baseline(self):
        # pooled over seeds: one asset can sit on a low load for its first weeks (hi moves 2 per
        # unit load)
        pooled = []
        for seed in range(30):
            t = simulate_lives(P, seed, Scenario())
            for aid in ASSET_IDS:
                pre_onset = t.hi[aid][: int(t.onset_day[aid])]
                pooled.extend(pre_onset[~np.isnan(pre_onset)].tolist())
        assert abs(np.mean(pooled) - P.hi_base) < 0.05
        assert abs(np.std(pooled) - P.hi_noise_sd) < 0.15  # noise plus load wander

    def test_masked_days_are_nan_in_the_table_and_only_hi_is_masked(self):
        seed = next(
            s for s in range(200) if np.isnan(simulate_lives(P, s, Scenario()).hi["CR-501"]).any()
        )
        t = simulate_lives(P, seed, Scenario())
        assert not np.isnan(t.sig["CR-501"]).any()
        assert not np.isnan(t.load["CR-501"]).any()

    def test_day_count_matches_the_time_grid(self):
        assert simulate_lives(P, 1, Scenario()).n_days * DAY_Q == P.horizon_q
