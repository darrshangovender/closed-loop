"""P11a: splicing lives into one record the real asset-health pipeline can run.

These are the checks of the spike docs/adr/spikes/splice_lives.py as tests, on the real
``run_asset`` and ``analyse_run``. The expensive runs are made once per session (conftest.py).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from closed_loop.f2.render import LifeLatent, RenderedTelemetry, render_life, sim_time
from closed_loop.f2.splice import SpliceError, maintenance_events, run_segments, splice
from closed_loop.plant.params import PlantParams

pd = pytest.importorskip("pandas", reason="f2 test needs the sibling repositories (pandas)")
pytestmark = pytest.mark.f2

BLIND = pd.Timedelta(days=21, hours=48)  # FeatureConfig: baseline_days + settle_hours


def piece(params: PlantParams, **kw) -> RenderedTelemetry:
    base = dict(
        asset_id="PU-201",
        life_id=0,
        mode="IMPELLER_WEAR",
        onset_day=24.0,
        failure_day=52.0,
        shape=2.0,
        downtime_hours=48.0,
        start_day=0.0,
        end_day=40.0,
    )
    segment = kw.pop("segment", 0)
    base.update(kw)
    return render_life(LifeLatent(**base), params, 3, segment)


class TestOneSegmentPerPiece:
    def test_the_pipeline_cuts_at_the_repair_record(self, lives, spliced_run) -> None:
        join = pd.Timestamp(lives[1].start)
        assert spliced_run.repairs == [join]
        assert [pd.Timestamp(s.start) for s in spliced_run.segments][1] == join
        assert len(spliced_run.segments) == 2
        assert not spliced_run.blocked

    def test_the_new_life_relearns_its_baseline(self, lives, spliced_run) -> None:
        join = pd.Timestamp(lives[1].start)
        second = spliced_run.segments[1].features
        assert second.ready
        assert second.baseline_start == join + pd.Timedelta(hours=48)
        assert second.baseline_end == second.baseline_start + pd.Timedelta(days=21)

    def test_the_join_is_the_first_row_of_the_new_life(self, lives, spliced_run) -> None:
        join = pd.Timestamp(lives[1].start)
        grid = spliced_run.frame.index
        assert join in grid
        # the stop and the restart are six hours apart: a gap of missing rows, not a bridge
        gap = spliced_run.clean.present.loc[
            pd.Timestamp(lives[0].stop) : join - pd.Timedelta(hours=1)
        ]
        assert len(gap) == 6 and not gap.any()


class TestBlindWindow:
    def test_no_decision_for_23_days_after_the_join(self, lives, spliced_run) -> None:
        join = pd.Timestamp(lives[1].start)
        window = spliced_run.frame.loc[join : join + BLIND - pd.Timedelta(hours=1)]
        assert len(window) == 23 * 24
        assert not window["decision_valid"].any()
        assert window["fused"].isna().all()

    def test_decisions_resume_afterwards(self, lives, spliced_run) -> None:
        join = pd.Timestamp(lives[1].start)
        after = spliced_run.frame.loc[join + BLIND :]
        assert after["decision_valid"].any()
        first = after.index[after["decision_valid"].to_numpy()][0]
        assert first - join >= BLIND

    def test_the_plants_blind_days_equal_the_pipelines(self, params: PlantParams) -> None:
        assert params.blind_days == pytest.approx(BLIND / pd.Timedelta(days=1))


class TestScoresBeforeTheJoin:
    @pytest.mark.parametrize("column", ["cusum", "iforest", "rule", "fused"])
    def test_equal_a_single_life_run_bit_for_bit(
        self, column: str, lives, spliced_run, life1_run
    ) -> None:
        before = spliced_run.frame.loc[: pd.Timestamp(lives[0].stop) - pd.Timedelta(hours=1)]
        reference = life1_run.frame[column]
        got = before[column].reindex(reference.index)
        np.testing.assert_array_equal(got.to_numpy(), reference.to_numpy())

    def test_the_health_index_input_is_equal_too(self, lives, spliced_run, life1_run) -> None:
        before = spliced_run.z24.loc[: pd.Timestamp(lives[0].stop) - pd.Timedelta(hours=1)]
        np.testing.assert_array_equal(
            before.reindex(life1_run.z24.index).to_numpy(), life1_run.z24.to_numpy()
        )

    def test_the_life_1_run_has_something_to_compare(self, life1_run) -> None:
        assert life1_run.frame["fused"].notna().sum() > 100


class TestNoPhantomAlarm:
    @staticmethod
    def open_orders(run, instant):
        from asset_health.analysis import AnalysisConfig, analyse_run, load_default_rul_model

        return analyse_run(
            run, AnalysisConfig(), load_default_rul_model(), as_of=pd.Timestamp(instant)
        ).work_orders

    def test_the_alarm_of_life_1_is_open_before_the_replacement(self, lives, spliced_run) -> None:
        orders = self.open_orders(spliced_run, lives[0].stop - pd.Timedelta(hours=2))
        assert orders and orders[0].failure_mode_hypothesis.value == "impeller_wear"

    @pytest.mark.parametrize("hours", [0, 6, 24 * 5, 24 * 22])
    def test_nothing_opens_at_the_join_or_in_the_blind_window(
        self, hours: int, lives, spliced_run
    ) -> None:
        assert self.open_orders(spliced_run, lives[1].start + pd.Timedelta(hours=hours)) == []

    def test_the_alarm_of_life_2_names_the_bearing_family(self, lives, spliced_run) -> None:
        from asset_health.models import HYPOTHESIS_FAMILY, FailureMode

        orders = self.open_orders(spliced_run, sim_time(89.0))
        assert orders
        assert HYPOTHESIS_FAMILY[orders[0].failure_mode_hypothesis] == FailureMode.BEARING_SPALL


class TestValidation:
    def test_nothing_to_splice(self) -> None:
        with pytest.raises(SpliceError, match="nothing"):
            splice([])

    def test_pieces_of_two_assets_are_refused(self, params: PlantParams) -> None:
        a = piece(params)
        b = piece(
            params,
            asset_id="PU-202",
            life_id=1,
            start_day=40.25,
            end_day=None,
            repair_at_start=True,
            segment=1,
        )
        with pytest.raises(SpliceError, match="not PU-201"):
            splice([a, b])

    def test_labels_must_run_from_zero(self, params: PlantParams) -> None:
        with pytest.raises(SpliceError, match="labelled segment 1"):
            splice([piece(params, segment=1)])

    def test_a_piece_may_not_start_before_the_previous_one_stops(self, params: PlantParams) -> None:
        a = piece(params)
        b = piece(params, life_id=1, start_day=39.0, end_day=None, repair_at_start=True, segment=1)
        with pytest.raises(SpliceError, match="before piece 0 stops"):
            splice([a, b])

    def test_a_new_life_without_a_repair_record_is_refused(self, params: PlantParams) -> None:
        a = piece(params)
        b = piece(params, life_id=1, start_day=40.25, end_day=None, segment=1)
        with pytest.raises(SpliceError, match="no repair record"):
            splice([a, b])

    def test_lives_may_not_go_backwards(self, params: PlantParams) -> None:
        a = piece(params, life_id=2)
        b = piece(params, life_id=1, start_day=40.25, end_day=None, repair_at_start=True, segment=1)
        with pytest.raises(SpliceError, match="goes back"):
            splice([a, b])

    def test_a_restart_in_the_same_hour_as_the_stop_is_allowed(self, params: PlantParams) -> None:
        a = piece(params, end_day=40.0)
        b = piece(params, life_id=1, start_day=40.0, end_day=None, repair_at_start=True, segment=1)
        assert b.start == a.stop
        assert len(splice([a, b])) == a.n_rows + b.n_rows

    def test_an_empty_piece_is_refused(self, params: PlantParams) -> None:
        whole = piece(params)
        empty = dataclasses.replace(whole, frame=whole.frame.iloc[:0])
        with pytest.raises(SpliceError, match="no rows"):
            splice([empty])


class TestRecords:
    def test_the_spliced_frame_is_the_pieces_in_order(self, lives) -> None:
        frame = splice(lives)
        assert len(frame) == lives[0].n_rows + lives[1].n_rows
        assert frame["timestamp"].is_monotonic_increasing
        assert list(frame.columns) == list(lives[0].frame.columns)

    def test_one_repair_record_at_the_join_with_the_asset_health_id(self, lives) -> None:
        events = maintenance_events(lives)
        assert len(events) == 1
        assert events[0].kind == "repair" and events[0].asset_id == "PMP-01"
        assert events[0].timestamp == lives[1].start

    def test_a_single_piece_has_no_records(self, lives) -> None:
        assert maintenance_events(lives[:1]) == []

    def test_a_job_on_the_same_component_is_logged_as_a_repair(self, params: PlantParams) -> None:
        """A wrong-mode job renews nothing, but the CMMS cannot tell: the record is a repair."""
        a = piece(params)
        b = piece(params, start_day=40.25, end_day=None, repair_at_start=True, segment=1)
        assert a.latent.life_id == b.latent.life_id
        assert len(maintenance_events([a, b])) == 1
        c = piece(params, start_day=40.25, end_day=None, repair_at_start=False, segment=1)
        assert maintenance_events([a, c]) == []

    def test_a_repair_record_on_a_worn_component_hides_the_fault_from_the_pipeline(
        self, params: PlantParams
    ) -> None:
        """The pipeline re-learns its baseline after every repair record, on whatever it then sees.

        After a job that did not renew the worn impeller the new baseline contains the wear, so the
        wear is no longer an anomaly. The same life with no record opens an urgent impeller-wear
        alarm by day 66; with the record nothing opens through day 89. This is how a wrong-mode job
        blinds the detector at F2 (finding recorded in docs/adr/progress/P11a-f2-spike-render-
        splice.md; the F1 plant blinds the estimator after a renewal only).
        """
        life = dict(onset_day=20.0, failure_day=110.0)
        a = piece(params, end_day=38.0, **life)
        logged = piece(
            params, start_day=38.25, end_day=None, repair_at_start=True, segment=1, **life
        )
        unlogged = piece(params, start_day=38.25, end_day=None, segment=1, **life)
        hidden = run_segments([a, logged])
        shown = run_segments([a, unlogged])
        assert hidden.repairs and not shown.repairs
        window = slice(sim_time(62.0), sim_time(80.0))
        assert (
            hidden.frame.loc[window, "fused"].mean() < 0.4 * shown.frame.loc[window, "fused"].mean()
        )
        assert TestNoPhantomAlarm.open_orders(shown, sim_time(72.0))
        assert TestNoPhantomAlarm.open_orders(hidden, sim_time(72.0)) == []


class TestRunSegments:
    def test_as_of_replays_the_moment(self, lives) -> None:
        cut = lives[0].stop - pd.Timedelta(days=3)
        run = run_segments(lives[:1], as_of=cut)
        assert run.frame.index[-1] <= pd.Timestamp(cut)

    def test_the_segment_record_is_frozen(self, lives) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            lives[0].segment = 5  # type: ignore[misc]
