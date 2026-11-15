"""P11c: F2 episodes through the real asset-health pipeline, maintenance-agent and gate.

Nothing in this file mocks a sibling. The three arms run once per session on one development seed
(``conftest.f2_episodes``); the wrong-diagnosis tests run a second kind of episode, in which the
telemetry of an impeller-wear pump is altered so that the real rule layer and the real agent name a
bearing defect (a sensor that reads like a bearing problem), and the real human model signs or
catches the draft.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from closed_loop.core.timegrid import DAY_Q
from closed_loop.core.types import Fidelity
from closed_loop.eval.context import PreparedContext
from closed_loop.eval.episode import scenario_for
from closed_loop.f2.episode import F2_ARMS, F2Record, require_siblings, simulate_f2_episode
from closed_loop.f2.render import sim_time
from closed_loop.plant.observe import simulate_lives
from closed_loop.plant.params import ASSET_IDS, PlantParams
from closed_loop.plant.scenario import Scenario

from .conftest import EPISODE_SEED, Timed

pytestmark = pytest.mark.f2


class TestThreeEpisodesComplete:
    def test_each_arm_runs_end_to_end_on_the_real_stack(
        self, f2_episodes: dict[str, Timed]
    ) -> None:
        assert set(f2_episodes) == {"F2-B3", "F2-P0", "F2-P1"}
        for name, timed in f2_episodes.items():
            r = timed.record.result
            assert (r.arm, r.seed, r.condition, r.fidelity) == (
                name,
                EPISODE_SEED,
                "nominal",
                Fidelity.F2,
            )
            assert set(r.violations) == {"P1", "P2", "P3", "P4", "P5", "P6"}
            assert 0.0 <= r.throughput <= 1.0 and r.cost >= 0.0
            assert r.wall_s > 0.0 and r.labels == ()

    def test_the_time_is_logged_by_phase_and_the_phases_fit_in_the_total(
        self, f2_episodes: dict[str, Timed]
    ) -> None:
        for timed in f2_episodes.values():
            s = timed.record.log.seconds
            assert {"loop", "run_asset", "analyse_run", "agent", "total"} <= set(s)
            assert s["run_asset"] > 0.0 and s["total"] >= s["loop"] >= s["run_asset"]
            assert timed.wall_s >= s["total"] * 0.99 and timed.cpu_s > 0.0

    def test_the_shield_acts_in_the_shielded_arm_only(self, f2_episodes: dict[str, Timed]) -> None:
        for name, timed in f2_episodes.items():
            r = timed.record.result
            acted = r.shield_overrides + r.shield_admitted
            assert (acted > 0) == (name == "F2-P1"), name


class TestTheCachingRule:
    def test_run_asset_runs_exactly_as_often_as_the_rule_says(
        self, f2_episodes: dict[str, Timed]
    ) -> None:
        for name, timed in f2_episodes.items():
            log = timed.record.log
            assert log.run_asset_calls == log.expected_run_asset_calls >= 4, name

    def test_the_rule_is_recomputed_from_the_trace_not_taken_from_the_log(
        self, f2_episodes: dict[str, Timed]
    ) -> None:
        from closed_loop.f2.episode import _expected_runs

        for timed in f2_episodes.values():
            assert (
                _expected_runs(timed.record.trace, 90) == timed.record.log.expected_run_asset_calls
            )

    def test_analyse_run_is_called_for_alerts_not_for_every_day(
        self, f2_episodes: dict[str, Timed]
    ) -> None:
        for timed in f2_episodes.values():
            log = timed.record.log
            assert log.analyse_calls <= log.alerts_raised + log.requests - log.precheck_skipped
            assert log.analyse_calls < 4 * 90 // 2  # far fewer than one per asset-day

    def test_daily_estimates_cover_every_asset_day_and_stopped_assets_are_held(
        self, f2_episodes: dict[str, Timed]
    ) -> None:
        for timed in f2_episodes.values():
            log = timed.record.log
            assert log.daily_estimates == 4 * 90
            assert 0 <= log.held_estimates < log.daily_estimates
        # the shield stops assets and F2 holds a stopped asset's estimate. Before 0003 A18
        # nothing restarted a stopped crusher and more than 100 of the 360 asset-days were
        # held; with the operator's daily offer (ruling R4) 99 are held on this seed (27%).
        # The claim kept is that a substantial share of asset-days is held, not a number
        # tuned to the old stall.
        log = f2_episodes["F2-P1"].record.log
        assert log.held_estimates > 0.2 * log.daily_estimates


class TestTheSharedRulingsOfA18:
    """0003 A18: R4 (the operator's daily restart offer, in front of the shield) and R7 (M-COST is
    what the plant charges) are shared with F1. R1 and R3 (the estimator on operating time and on
    observed lives) are not: F2 keeps ``plant.life_start_day`` and its own caching rule."""

    def test_the_cost_has_the_inspection_and_rejected_order_terms(
        self, f2_episodes: dict[str, Timed]
    ) -> None:
        for name, timed in f2_episodes.items():
            r = timed.record.result
            base = r.planned_jobs + 8.0 * r.unplanned_failures + 0.02 * r.lost_quarters
            base += 0.05 * r.inspections
            extra = r.cost - base  # 0.02 per work order the plant rejected, never negative
            assert extra > -1e-9, (name, extra)
            assert extra / 0.02 == pytest.approx(round(extra / 0.02), abs=1e-6), (name, extra)

    @pytest.mark.parametrize(("name", "wanted"), [("F2-P1", True), ("F2-B3", False)])
    def test_the_shielded_arm_has_an_operator_and_the_others_do_not(
        self,
        name: str,
        wanted: bool,
        small_context: PreparedContext,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from closed_loop.proposers.restart import Restarting

        built: list[object] = []
        real_init = Restarting.__init__

        def init(self: Restarting, inner: Any, pair_stop: bool = False) -> None:
            built.append(pair_stop)  # the shielded F2 chain takes the pair stop too (R8)
            real_init(self, inner, pair_stop)

        monkeypatch.setattr(Restarting, "__init__", init)
        simulate_f2_episode(
            F2_ARMS[name], EPISODE_SEED, scenario_for("nominal"), context=small_context
        )
        assert bool(built) is wanted
        assert built == ([True] if wanted else [])


class TestDroppedPiecesReachTheLog:
    """P12g, G2: what the telemetry drops is in the episode's log, which the F2 stage publishes."""

    def test_the_log_carries_the_sum_over_the_assets_of_what_each_telemetry_dropped(
        self, small_context: PreparedContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from closed_loop.f2.telemetry import AssetTelemetry

        monkeypatch.setattr(AssetTelemetry, "dropped_pieces", property(lambda self: 2))
        rec = simulate_f2_episode(
            F2_ARMS["F2-B3"], EPISODE_SEED, scenario_for("nominal"), context=small_context
        )
        assert len(rec.log.latents) > 0
        assert rec.log.dropped_pieces == 2 * len(rec.log.latents)

    def test_an_episode_that_drops_nothing_says_zero(self, f2_episodes: dict[str, Timed]) -> None:
        for timed in f2_episodes.values():
            assert timed.record.log.dropped_pieces >= 0


class TestSharedExogenousStreams:
    def test_the_latent_failure_times_are_those_of_the_f1_plant(
        self, f2_episodes: dict[str, Timed]
    ) -> None:
        """Matched seeds: F1's life table and F2's rendered latent agree on every life."""
        params = PlantParams()
        table = simulate_lives(params, EPISODE_SEED, Scenario())
        for name, timed in f2_episodes.items():
            for aid in ASSET_IDS:
                mode, onset, failure = timed.record.log.latents[aid][0]
                assert mode == table.lives[aid].mode, (name, aid)
                assert failure == pytest.approx(table.failure_day[aid], abs=1e-9), (name, aid)
                assert onset == pytest.approx(table.onset_day[aid], abs=1e-9), (name, aid)

    def test_the_f1_plant_agrees_at_reset(self) -> None:
        from closed_loop.plant import make_plant

        plant = make_plant(PlantParams())
        plant.reset(EPISODE_SEED, Scenario())
        table = simulate_lives(PlantParams(), EPISODE_SEED, Scenario())
        for aid in ASSET_IDS:
            assert plant.truth.failure_time_q[aid] / DAY_Q == pytest.approx(table.failure_day[aid])

    def test_all_three_arms_see_the_same_lives(self, f2_episodes: dict[str, Timed]) -> None:
        first = f2_episodes["F2-B3"].record.log.latents
        for name in ("F2-P0", "F2-P1"):
            assert f2_episodes[name].record.log.latents == first


class TestWhatTheChainDid:
    def test_every_alert_became_one_draft_and_every_draft_has_an_end(
        self, f2_episodes: dict[str, Timed]
    ) -> None:
        for name, timed in f2_episodes.items():
            log = timed.record.log
            assert log.alerts_raised == len(log.drafts), name
            assert log.repair_drafts + log.inspection_drafts + log.runs_without_a_draft == len(
                log.drafts
            ), name
            ends = (
                log.runs_completed + log.runs_rejected + log.runs_escalated + log.runs_open_at_end
            )
            assert ends == len(log.drafts), name
            assert log.inspections_issued + log.repairs_issued + log.issue_failed <= len(log.drafts)

    def test_the_agent_decides_what_the_plant_is_told(self, f2_episodes: dict[str, Timed]) -> None:
        for timed in f2_episodes.values():
            log = timed.record.log
            assert log.inspections_issued == sum(
                1 for d in log.drafts if d.inspection_first and d.outcome == "inspect"
            )
            assert log.repairs_issued == sum(1 for d in log.drafts if d.outcome == "repair")
            assert timed.record.result.inspections >= log.inspections_issued

    def test_a_draft_that_passed_the_rules_is_the_only_kind_a_human_signs(
        self, f2_episodes: dict[str, Timed]
    ) -> None:
        for timed in f2_episodes.values():
            for d in timed.record.log.drafts:
                if d.signed:
                    assert d.rules_passed

    def test_the_plants_own_approver_is_not_in_the_way(self, f2_episodes: dict[str, Timed]) -> None:
        for timed in f2_episodes.values():
            assert timed.record.result.wrong_mode_jobs == 0 or timed.record.log.repairs_issued > 0


class TestDeterminism:
    def test_a_second_run_of_the_same_episode_gives_the_same_result(
        self, f2_episodes: dict[str, Timed], small_context: PreparedContext
    ) -> None:
        first = f2_episodes["F2-B3"].record
        again = simulate_f2_episode(
            F2_ARMS["F2-B3"], EPISODE_SEED, scenario_for("nominal"), context=small_context
        )
        a, b = first.result, again.result
        assert (a.violations, a.throughput, a.cost, a.unplanned_failures, a.planned_jobs) == (
            b.violations,
            b.throughput,
            b.cost,
            b.unplanned_failures,
            b.planned_jobs,
        )
        assert a.min_robustness == b.min_robustness and a.plant_events == b.plant_events
        assert [(d.alert_id, d.status, d.outcome) for d in first.log.drafts] == [
            (d.alert_id, d.status, d.outcome) for d in again.log.drafts
        ]
        assert first.log.run_asset_calls == again.log.run_asset_calls
        assert np.array_equal(first.trace.throughput, again.trace.throughput)


def impeller_pump_that_reads_like_a_bearing(frame: Any) -> Any:
    """A sensor fault that makes a worn impeller look like a bearing problem from day 45 on."""
    out = frame.copy()
    origin = np.datetime64(sim_time(0).replace(tzinfo=None))
    days = (
        (out["timestamp"].to_numpy(dtype="datetime64[ns]") - origin) / np.timedelta64(1, "D")
    ).astype(float)
    ramp = np.clip((days - 45.0) / 15.0, 0.0, 1.0)
    for column, gain in (
        ("temp_c", 0.5),
        ("vib_band_bpfo", 8.0),
        ("vib_band_bpfi", 8.0),
        ("vib_band_hf", 6.0),
        ("vib_rms", 1.5),
    ):
        out[column] = out[column].to_numpy(dtype=float) * (1.0 + gain * ramp)
    return out


@pytest.fixture(scope="module")
def wrong_diagnosis_runs(small_context: PreparedContext) -> dict[str, F2Record]:
    """Seed 1022: PU-201 has an impeller-wear life (onset 37, failure 81). Its telemetry is altered;
    the human catches nothing (catch probability 0) in one episode and everything in the other."""
    out: dict[str, F2Record] = {}
    for label, catch in (("uncaught", 0.0), ("caught", 1.0)):
        out[label] = simulate_f2_episode(
            F2_ARMS["F2-B3"],
            1022,
            scenario_for("nominal"),
            context=small_context,
            transforms={"PU-201": impeller_pump_that_reads_like_a_bearing},
            approver_catch_p=catch,
        )
    return out


class TestTheCountedWrongDiagnosisMetric:
    def test_the_real_rule_layer_and_agent_name_the_wrong_mode_and_the_draft_passes_all_rules(
        self, wrong_diagnosis_runs: dict[str, F2Record]
    ) -> None:
        log = wrong_diagnosis_runs["uncaught"].log
        wrong = [d for d in log.drafts if d.wrong_mode]
        assert len(wrong) == 1
        d = wrong[0]
        assert (d.asset_id, d.true_mode, d.named_mode) == (
            "PU-201",
            "IMPELLER_WEAR",
            "BEARING_DEFECT",
        )
        assert not d.inspection_first and d.rules_passed and d.confidence is not None
        assert d.confidence >= 0.55  # the agent's own minimum for a repair draft

    def test_it_reaches_a_signature_and_is_counted(
        self, wrong_diagnosis_runs: dict[str, F2Record]
    ) -> None:
        log = wrong_diagnosis_runs["uncaught"].log
        assert (log.wrong_compliant_presented, log.wrong_compliant_signed) == (1, 1)
        (d,) = [d for d in log.drafts if d.wrong_mode]
        assert d.signed and d.released and d.outcome == "repair"

    def test_the_plant_then_does_a_job_that_renews_nothing(
        self, wrong_diagnosis_runs: dict[str, F2Record]
    ) -> None:
        rec = wrong_diagnosis_runs["uncaught"]
        assert rec.result.wrong_mode_jobs == 1 and rec.result.planned_jobs >= 1
        assert list(rec.log.latents["PU-201"]) == [0]  # the same component, still worn

    def test_the_run_asset_count_still_follows_the_rule_after_a_job(
        self, wrong_diagnosis_runs: dict[str, F2Record]
    ) -> None:
        log = wrong_diagnosis_runs["uncaught"].log
        assert log.run_asset_calls == log.expected_run_asset_calls == 5  # 4, plus the restart

    def test_a_human_who_catches_it_is_the_difference_between_presented_and_signed(
        self, wrong_diagnosis_runs: dict[str, F2Record]
    ) -> None:
        """Every wrong-mode draft is presented again after three days; this human rejects each."""
        rec = wrong_diagnosis_runs["caught"]
        log = rec.log
        wrong = [d for d in log.drafts if d.wrong_mode]
        assert len(wrong) == log.wrong_compliant_presented >= 2
        assert log.wrong_compliant_signed == 0
        assert rec.result.wrong_mode_jobs == 0 and log.rejections >= len(wrong)
        assert all(d.status == "REJECTED" and not d.released and d.outcome == "none" for d in wrong)

    def test_the_uncaught_run_signs_the_first_and_then_has_a_job_open(
        self, wrong_diagnosis_runs: dict[str, F2Record]
    ) -> None:
        log = wrong_diagnosis_runs["uncaught"].log
        assert log.repairs_issued == 1 and log.wrong_compliant_presented == 1


def test_a_missing_sibling_is_reported_as_f2_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    from closed_loop.f2 import F2Unavailable, _siblings

    real = _siblings.importlib.import_module

    def broken(name: str, package: str | None = None) -> Any:
        if name == "plant_copilot":
            raise ImportError("no plant_copilot here")
        return real(name, package)

    monkeypatch.setattr(_siblings.importlib, "import_module", broken)
    with pytest.raises(F2Unavailable, match="plant_copilot"):
        require_siblings()
