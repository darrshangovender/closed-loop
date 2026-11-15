"""P11c: the telemetry side of an F2 episode against the real pipeline.

Pieces of an asset's history follow its op sequence; the pipeline runs when the history changes and
not otherwise; a replacement starts a new life with a repair record; the alert precheck never hides
an alert that the real analysis would find; scenario faults are written into the rendered rows.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from closed_loop.core.timegrid import DAY_Q
from closed_loop.core.types import Command, PlanCmd
from closed_loop.f2.alert_source import RunCache, aware
from closed_loop.f2.episode import f2_params
from closed_loop.f2.render import sim_time
from closed_loop.f2.telemetry import F2Telemetry, FaultPlan, fault_plan, inject_faults, wall_day
from closed_loop.plant import make_plant
from closed_loop.plant.degradation import build_physical
from closed_loop.plant.jobs import JobRunner
from closed_loop.plant.params import ASSET_IDS, PlantParams
from closed_loop.plant.scenario import FAULT_SPIKE, FAULT_STUCK, Scenario

pytestmark = pytest.mark.f2

SEED = 1019  # PU-201 has a bearing life that fails on day 64


def world(seed: int = SEED, scenario: Scenario | None = None) -> tuple[Any, Any, Any, Any, Any]:
    params = f2_params(PlantParams())
    scenario = scenario if scenario is not None else Scenario()
    plant = make_plant(params)
    obs = plant.reset(seed, scenario)
    phys = build_physical(params, seed, scenario.resolve(seed, params))
    cache = RunCache()
    telemetry = F2Telemetry(plant, phys, seed, params, cache)
    return params, plant, obs, telemetry, cache


def daily(telemetry: F2Telemetry, plant: Any, obs: Any) -> None:
    for aid in ASSET_IDS:
        telemetry.estimate(aid, obs.day, obs.assets[aid], obs.day - plant.life_start_day(aid))


@pytest.fixture(scope="module")
def replaced_pump() -> dict[str, Any]:
    """PU-201 gets a correct bearing job on day 31 and is replaced; 45 days are stepped."""
    params, plant, obs, telemetry, cache = world()
    runner = JobRunner(source="test")
    last_day = -1
    stop_quarter = None
    for _ in range(45 * DAY_Q):
        telemetry.observe(obs.assets, obs.quarter)
        if obs.day != last_day:
            last_day = obs.day
            daily(telemetry, plant, obs)
        commands = []
        if obs.quarter == 31 * DAY_Q:
            commands.append(Command("PU-201", PlanCmd.RAISE_WO, "BEARING_DEFECT", "test"))
        for view in obs.assets.values():
            if view.job is not None:
                cmd = runner.next_command(view, view.job, obs.quarter)
                if cmd is not None:
                    commands.append(cmd)
        if stop_quarter is None and obs.assets["PU-201"].op.value != "RUN":
            stop_quarter = obs.quarter
        obs = plant.step(commands).obs
    return {
        "plant": plant,
        "telemetry": telemetry,
        "cache": cache,
        "obs": obs,
        "stop_quarter": stop_quarter,
    }


class TestAReplacementStartsANewLife:
    def test_the_job_renewed_the_component(self, replaced_pump: dict[str, Any]) -> None:
        plant = replaced_pump["plant"]
        assert plant.des.by_id["PU-201"].life_id == 1
        assert plant.stats().planned_jobs == 1 and plant.stats().wrong_mode_jobs == 0

    def test_the_history_is_two_pieces_the_second_with_a_repair_record(
        self, replaced_pump: dict[str, Any]
    ) -> None:
        pieces = replaced_pump["telemetry"].assets["PU-201"].history()
        assert [p.segment for p in pieces] == [0, 1]
        assert [p.latent.life_id for p in pieces] == [0, 1]
        assert pieces[1].latent.repair_at_start and not pieces[0].latent.repair_at_start
        assert pieces[0].latent.end_day == pytest.approx(replaced_pump["stop_quarter"] / DAY_Q)
        assert pieces[1].start >= pieces[0].stop

    def test_each_life_has_its_own_latent_from_the_plants_clock(
        self, replaced_pump: dict[str, Any]
    ) -> None:
        latents = replaced_pump["telemetry"].assets["PU-201"].latents
        assert set(latents) == {0, 1}
        restart_day = latents[1].failure_day
        assert latents[1].failure_day > latents[1].onset_day > 31.0
        assert latents[0].mode == "BEARING_DEFECT" and restart_day > 45.0

    def test_run_asset_ran_four_times_and_once_more_for_the_restart(
        self, replaced_pump: dict[str, Any]
    ) -> None:
        assert replaced_pump["cache"].run_asset_calls == 5
        assert replaced_pump["telemetry"].stats.pipeline_runs == 5

    def test_the_pipeline_cut_its_run_at_the_repair_record(
        self, replaced_pump: dict[str, Any]
    ) -> None:
        run = replaced_pump["cache"].run("PU-201")
        assert len(run.repairs) == 1 and len(run.segments) == 2

    def test_the_other_assets_were_not_touched_by_the_replacement(
        self, replaced_pump: dict[str, Any]
    ) -> None:
        for aid in ("CR-501", "CV-101", "PU-202"):
            assert len(replaced_pump["telemetry"].assets[aid].history()) == 1


class TestTheAlertPrecheck:
    def test_it_never_hides_an_alert_the_real_analysis_finds(self, pump_cache: RunCache) -> None:
        params, plant, obs, telemetry, _ = world()
        telemetry.cache = pump_cache
        seen_true = seen_false = False
        for day in range(20, 52):
            instant = aware(sim_time(float(day)))
            has_order = bool(pump_cache.analysis("PU-201", instant).work_orders)
            may = telemetry.may_alert("PU-201", instant)
            assert may or not has_order, day
            seen_true |= may
            seen_false |= not may
        assert seen_true and seen_false  # it filters early days and passes the days of the alert

    def test_an_asset_the_pipeline_has_not_seen_cannot_have_an_alert(self) -> None:
        _, _, _, telemetry, _ = world()
        assert telemetry.may_alert("PU-201", aware(sim_time(10.0))) is False


class TestTheEstimate:
    def test_a_stopped_asset_holds_its_last_estimate_and_runs_nothing(self) -> None:
        params, plant, obs, telemetry, cache = world()
        telemetry.observe(obs.assets, obs.quarter)
        first = telemetry.estimate("PU-201", 0, obs.assets["PU-201"], 0.0)
        runs = cache.run_asset_calls
        telemetry.assets["PU-201"]._open = None  # the asset has stopped
        held = telemetry.estimate("PU-201", 1, obs.assets["PU-201"], 1.0)
        assert held.day == 1 and held.m_days == first.m_days
        assert cache.run_asset_calls == runs and telemetry.stats.held_estimates == 1

    def test_the_first_days_are_a_class_prior_because_the_pipeline_is_learning_its_baseline(
        self,
    ) -> None:
        params, plant, obs, telemetry, cache = world()
        telemetry.observe(obs.assets, obs.quarter)
        out = telemetry.estimate("PU-202", 0, obs.assets["PU-202"], 0.0)
        assert out.status in ("insufficient", "not_degrading", "prior")

    def test_a_degrading_life_gets_a_trend_estimate_close_to_the_truth(self) -> None:
        """PU-202 of seed 1100 fails on day 82.2; asset-health's p50 follows it (day 70: 11.2 d)."""
        params, plant, obs, telemetry, cache = world(1100)
        telemetry.observe(obs.assets, obs.quarter)
        telemetry.estimate("PU-202", 0, obs.assets["PU-202"], 0.0)
        out = telemetry.estimate("PU-202", 70, obs.assets["PU-202"], 70.0)
        assert out.status == "estimated" and 6.0 < out.m_days < 16.0


class TestWallDay:
    def test_it_is_the_rule_of_the_plants_life_table(self) -> None:
        params = f2_params(PlantParams())
        phys = build_physical(params, SEED, Scenario().resolve(SEED, params))
        plant = make_plant(params)
        plant.reset(SEED, Scenario())
        s = plant.des.by_id["PU-201"]
        cum = phys.cum["PU-201"]
        assert wall_day(cum, 0, s.life.tau_fail - s.life.tau_start) == pytest.approx(
            plant.truth.failure_time_q["PU-201"] / DAY_Q
        )


# ----------------------------------------------------------------------------- scenario faults


@pytest.fixture(scope="module")
def frame() -> Any:
    pd = pytest.importorskip("pandas")
    origin = sim_time(0).replace(tzinfo=None)
    stamps = pd.date_range(origin, periods=24 * 20, freq="h")
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "timestamp": stamps,
            "vib_rms": 1.0 + 0.1 * rng.standard_normal(len(stamps)),
            "temp_c": 40.0 + rng.standard_normal(len(stamps)),
            "load_pct": 60.0 + rng.standard_normal(len(stamps)),
        }
    )


class TestInjectFaults:
    def test_no_fault_returns_the_frame_itself(self, frame: Any) -> None:
        assert inject_faults(frame, FaultPlan()) is frame

    def test_a_stuck_sensor_holds_the_last_good_value_for_its_days(self, frame: Any) -> None:
        out = inject_faults(frame, FaultPlan(FAULT_STUCK, 5.0, 3))
        window = (frame["timestamp"] >= frame["timestamp"].iloc[5 * 24]) & (
            frame["timestamp"] < frame["timestamp"].iloc[8 * 24]
        )
        held = out.loc[window, "vib_rms"]
        assert held.nunique() == 1 and held.iloc[0] == frame["vib_rms"].iloc[5 * 24 - 1]
        assert (out.loc[~window, "vib_rms"] == frame.loc[~window, "vib_rms"]).all()
        assert (out["load_pct"] == frame["load_pct"]).all()  # not a measured channel

    def test_a_spike_multiplies_three_hourly_rows_by_six(self, frame: Any) -> None:
        out = inject_faults(frame, FaultPlan(FAULT_SPIKE, 7.0))
        changed = np.flatnonzero(out["vib_rms"].to_numpy() != frame["vib_rms"].to_numpy())
        assert list(changed) == [7 * 24, 7 * 24 + 1, 7 * 24 + 2]
        assert np.allclose(
            out["vib_rms"].to_numpy()[changed], 6.0 * frame["vib_rms"].to_numpy()[changed]
        )

    def test_a_gain_error_scales_the_measured_channels_by_one_plus_the_drift_of_the_day(
        self, frame: Any
    ) -> None:
        drift = np.zeros(90)
        drift[10:] = 0.2
        out = inject_faults(frame, FaultPlan(drift=drift))
        before, after = slice(0, 10 * 24), slice(10 * 24, 20 * 24)
        assert np.allclose(out["temp_c"].to_numpy()[before], frame["temp_c"].to_numpy()[before])
        assert np.allclose(out["temp_c"].to_numpy()[after], 1.2 * frame["temp_c"].to_numpy()[after])
        assert (out["load_pct"] == frame["load_pct"]).all()

    def test_the_original_frame_is_not_changed(self, frame: Any) -> None:
        before = frame.copy()
        inject_faults(frame, FaultPlan(FAULT_STUCK, 2.0, 2, np.full(90, 0.1)))
        assert frame.equals(before)


class TestFaultPlan:
    def test_the_shifted_condition_puts_the_plants_drift_into_every_asset(self) -> None:
        params = PlantParams()
        scenario = Scenario.shifted()
        phys = build_physical(params, 1100, scenario.resolve(1100, params))
        day = scenario.resolve(1100, params).drift_all_day
        for aid in ASSET_IDS:
            plan = fault_plan(phys, aid, params, 1100)
            assert plan.active and plan.drift is not None
            assert plan.drift[int(day) + 20] > plan.drift[int(day) - 1] >= 0.0

    def test_the_nominal_condition_has_no_scenario_fault(self) -> None:
        params = PlantParams()
        phys = build_physical(params, 1100, Scenario().resolve(1100, params))
        plans = [fault_plan(phys, aid, params, 1100) for aid in ASSET_IDS]
        assert all(p.kind == 0 for p in plans)


# ------------------------------------------------------------------ a restart past the failure


class TestARestartPastTheFailureDay:
    """P12g, G1. A life's failure day is fixed when its first piece opens, on the assumption that
    the asset does not stop again, so a long stop can put a restart past it. The piece is dropped,
    its repair record is carried to the next piece and the drop is counted. Before the fix the
    guard that was meant to do this sat after ``dataclasses.replace``, whose ``__post_init__``
    raises ``RenderError`` on the same condition, and the first full run lost an F2 episode to
    it."""

    def build(self) -> tuple[Any, Any, int]:
        from closed_loop.f2.telemetry import _Piece

        _params, _plant, obs, telemetry, _cache = world()
        telemetry.observe(obs.assets, obs.quarter)  # PU-201 runs: its first piece opens at 0
        tel = telemetry.assets["PU-201"]
        latent = tel.latents[0]
        first = tel._open
        first.end_q = 10 * DAY_Q
        tel._pieces.append(first)
        past_q = int((latent.failure_day + 3.0) * DAY_Q)
        assert past_q < 85 * DAY_Q, (
            "the seed's failure day moved: the test needs room in the horizon"
        )
        tel._pieces.append(_Piece(0, past_q, past_q + 5, True))  # restarts past the failure day
        tel._records[1] = dataclasses.replace(
            latent,
            life_id=1,
            onset_day=latent.failure_day + 8.0,
            failure_day=latent.failure_day + 70.0,
        )
        tel._open = _Piece(1, past_q + DAY_Q, None, False)
        return telemetry, tel, past_q

    def test_the_piece_is_dropped_not_raised_and_its_repair_goes_to_the_next(self) -> None:
        _telemetry, tel, _ = self.build()
        pieces = tel.history()
        assert [p.latent.life_id for p in pieces] == [0, 1]
        assert pieces[1].latent.repair_at_start and not pieces[0].latent.repair_at_start

    def test_the_drop_is_counted_once_however_often_the_history_is_asked_for(self) -> None:
        _telemetry, tel, _ = self.build()
        assert tel.dropped_pieces == 0
        tel.history()
        tel.history()
        assert tel.dropped_pieces == 1

    def test_the_count_reaches_the_stats_of_the_episode(self) -> None:
        telemetry, _tel, _ = self.build()
        telemetry.estimate("PU-201", 80, world()[2].assets["PU-201"], 80.0)
        assert telemetry.stats.dropped_pieces == 1

    def test_a_history_with_no_such_piece_drops_nothing(
        self, replaced_pump: dict[str, Any]
    ) -> None:
        tel = replaced_pump["telemetry"].assets["PU-201"]
        assert len(tel.history()) == 2 and tel.dropped_pieces == 0

    def test_the_latent_still_refuses_a_piece_that_starts_after_the_life_has_failed(self) -> None:
        from closed_loop.f2.render import RenderError

        _telemetry, tel, _ = self.build()
        latent = tel.latents[0]
        with pytest.raises(RenderError, match="start after the life has failed"):
            dataclasses.replace(latent, start_day=latent.failure_day + 1.0)

    def test_a_piece_that_ends_before_it_starts_is_dropped_too(self) -> None:
        from closed_loop.f2.telemetry import _Piece

        _telemetry, tel, _past_q = self.build()
        tel._pieces[-1] = _Piece(0, 20 * DAY_Q, 20 * DAY_Q, True)  # zero length, before the failure
        assert [p.latent.life_id for p in tel.history()] == [0, 1]
        assert tel.dropped_pieces == 1
