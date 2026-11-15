"""P7: lives, histories and the recorder."""

import dataclasses

import numpy as np
import pytest

from closed_loop.causal.records import (
    CAUSES,
    AlertLabel,
    HistoryRecorder,
    LifeRecord,
    ObservationHistory,
)
from closed_loop.causal.twin import passive_world
from closed_loop.core.types import Fidelity
from closed_loop.plant import PlantParams, Scenario, make_plant


def make_record(**kw) -> LifeRecord:
    n = 40
    days = np.arange(n, dtype=float)
    hi = 1.0 + np.maximum(days - 24.0, 0.0) * 0.2
    masked = np.zeros(n, dtype=bool)
    masked[[5, 30]] = True
    base = dict(
        asset_id="PU-201",
        hi_days=days[~masked],
        hi=hi[~masked],
        derate_days=np.zeros(0),
        fail_day=None,
        censored_day=40.0,
        days=days,
        sig=np.vstack([0.1 * np.ones(n), 0.2 * np.ones(n), 0.3 * np.ones(n)]),
        load=np.ones(n),
        mode="BEARING_DEFECT",
        seed=7,
    )
    base.update(kw)
    return LifeRecord(**base)


class TestLifeRecord:
    def test_exactly_one_of_failure_and_censoring(self):
        with pytest.raises(ValueError, match="exactly one"):
            make_record(fail_day=30.0)
        with pytest.raises(ValueError, match="exactly one"):
            make_record(censored_day=None)
        assert make_record(fail_day=39.5, censored_day=None).failed

    def test_end_day_is_whichever_is_set(self):
        assert make_record().end_day == 40.0
        assert make_record(fail_day=39.5, censored_day=None).end_day == 39.5

    def test_arrays_are_read_only_copies(self):
        hi = np.arange(3.0)
        rec = make_record(
            hi_days=np.arange(3.0), hi=hi, days=np.zeros(0), sig=np.zeros((3, 0)), load=np.zeros(0)
        )
        hi[0] = 99.0
        assert rec.hi[0] == 0.0
        with pytest.raises(ValueError):
            rec.hi[0] = 1.0
        with pytest.raises(ValueError):
            rec.sig[0, 0] = 1.0

    def test_shapes_must_agree(self):
        with pytest.raises(ValueError, match="same length"):
            make_record(hi=np.zeros(3))
        with pytest.raises(ValueError, match="aligned"):
            make_record(load=np.ones(7))
        with pytest.raises(ValueError, match="increasing"):
            make_record(hi_days=np.array([3.0, 2.0]), hi=np.array([1.0, 1.0]))
        with pytest.raises(ValueError, match="need days"):
            make_record(days=np.zeros(0))

    def test_masked_days_are_absent_not_nan(self):
        bad = make_record()
        hi = bad.hi.copy()
        hi[2] = np.nan
        with pytest.raises(ValueError, match="absent"):
            make_record(hi=hi)

    def test_it_is_frozen_and_compares_by_identity(self):
        rec = make_record()
        with pytest.raises(dataclasses.FrozenInstanceError):
            rec.mode = "x"  # type: ignore[misc]
        assert rec != make_record()

    def test_the_first_alert_is_the_first_crossing_after_the_blind_window(self):
        rec = make_record()
        # hi = 1 + 0.2 (d - 24): reaches 2.0 at day 29
        assert rec.first_alert_day(2.0, 23.0) == 29
        assert rec.first_alert_day(2.0, 36.0) == 36  # the window moves the start of the search
        assert rec.first_alert_day(50.0, 23.0) is None

    def test_a_crossing_inside_the_blind_window_does_not_count(self):
        n = 30
        days = np.arange(n, dtype=float)
        hi = np.where(days == 10, 3.0, 1.0)
        rec = make_record(
            hi_days=days, hi=hi, days=days, sig=np.zeros((3, n)), load=np.ones(n), censored_day=30.0
        )
        assert rec.first_alert_day(2.0, 23.0) is None


class TestHistoryAt:
    def test_it_holds_the_readings_up_to_the_day(self):
        rec = make_record()
        h = rec.history_at(20)
        assert h.n == 21 and h.days[-1] == 20 and h.asset_id == "PU-201"
        assert h.y.shape == (21, 4) and h.load.shape == (21,)

    def test_a_masked_day_is_nan_in_the_health_index_and_present_in_the_signatures(self):
        h = make_record().history_at(30)
        assert np.isnan(h.y[5, 0]) and np.isnan(h.y[30, 0])
        assert not np.isnan(h.y[5, 1:]).any()
        assert np.isfinite(h.y[[0, 1, 29], 0]).all()

    def test_channels_are_hi_then_the_three_signatures(self):
        h = make_record().history_at(10)
        assert np.allclose(h.y[3, 1:], [0.1, 0.2, 0.3])

    def test_a_record_without_daily_arrays_has_no_history(self):
        rec = make_record(days=np.zeros(0), sig=np.zeros((3, 0)), load=np.zeros(0))
        with pytest.raises(ValueError, match="no daily"):
            rec.history_at(5)

    def test_the_history_does_not_see_the_future(self):
        rec = make_record()
        a = rec.history_at(15)
        assert a.days.max() == 15 and np.isfinite(a.y[:, 1:]).all()


class TestObservationHistory:
    def test_shape_checks(self):
        with pytest.raises(ValueError, match="y must be"):
            ObservationHistory("A", np.arange(3), np.zeros((3, 3)), np.zeros(3))
        with pytest.raises(ValueError, match="increasing"):
            ObservationHistory("A", np.array([1, 1]), np.zeros((2, 4)), np.zeros(2))

    def test_window_takes_the_last_readings(self):
        h = make_record().history_at(20).window(5)
        assert h.n == 5 and list(h.days) == [16, 17, 18, 19, 20]

    def test_arrays_are_read_only(self):
        h = make_record().history_at(5)
        with pytest.raises(ValueError):
            h.y[0, 0] = 5.0

    def test_it_carries_no_truth_field(self):
        names = {f.name for f in dataclasses.fields(ObservationHistory)}
        assert names == {"asset_id", "days", "y", "load", "life_start_day"}


class TestRecorder:
    def test_a_stream_of_observations_gives_the_same_history_as_the_record(self):
        params = PlantParams()
        seed = 1003
        plant = make_plant(params, fidelity=Fidelity.F1)
        obs = plant.reset(seed, Scenario())
        rec = HistoryRecorder()
        rec.record(obs)
        while obs.day < 40:
            obs = plant.step(()).obs
            rec.record(obs)
        reference = {lf.asset_id: lf for lf in passive_world(seed, params)}
        for aid in ("CR-501", "PU-201"):
            got = rec.history(aid)
            want = reference[aid].history_at(got.days[-1])
            assert np.array_equal(got.days, want.days)
            assert np.array_equal(got.y, want.y, equal_nan=True)
            assert np.allclose(got.load, want.load)

    def test_the_same_day_twice_is_recorded_once(self):
        plant = make_plant(PlantParams(), fidelity=Fidelity.F1)
        obs = plant.reset(1, Scenario())
        rec = HistoryRecorder()
        rec.record(obs)
        rec.record(obs)
        assert rec.history("PU-201").n == 1

    def test_reset_life_starts_a_fresh_history(self):
        plant = make_plant(PlantParams(), fidelity=Fidelity.F1)
        obs = plant.reset(1, Scenario())
        rec = HistoryRecorder()
        rec.record(obs)
        rec.reset_life("PU-201", 12)
        h = rec.history("PU-201")
        assert h.n == 0 and h.life_start_day == 12

    def test_an_unseen_asset_has_an_empty_history(self):
        assert HistoryRecorder().history("PU-201").n == 0


def test_alert_label_and_the_cause_vocabulary():
    assert CAUSES == ("degradation", "sensor_fault", "process_upset")
    lab = AlertLabel("PU-201-L0-A1", 41, 1, "degradation", "IMPELLER_WEAR")
    with pytest.raises(dataclasses.FrozenInstanceError):
        lab.day = 3  # type: ignore[misc]
