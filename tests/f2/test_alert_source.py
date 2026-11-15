"""P11b: asset-health as an alert source for maintenance-agent. Real pipeline, real agent models."""

from __future__ import annotations

import ast
from datetime import timedelta
from pathlib import Path

import pytest

from closed_loop.f2.agent import run_id_for
from closed_loop.f2.alert_source import (
    FEATURE_SIGNAL,
    HYPOTHESIS_MODE,
    SEVERITY,
    Z_ALARM,
    AssetHealthAlertSource,
    DuplicateAlert,
    RunCache,
    agent_assets,
)
from closed_loop.f2.render import sim_time
from closed_loop.plant.params import ASSET_CLASS, ASSET_IDS, MODES_BY_CLASS

from .conftest import ASSET_MAP, make_world, pump_lives

pytestmark = pytest.mark.f2


def raised(source, **kw) -> str:
    outcome = source.raise_alert("PU-201", **kw)
    assert outcome.kind == "raised", outcome
    assert outcome.alert_id is not None
    return outcome.alert_id


class TestProtocol:
    def test_it_is_an_alert_source(self, pump_world) -> None:
        from maintenance_agent.ports import AlertSource

        assert isinstance(pump_world.source, AlertSource)

    def test_nothing_is_listed_before_an_alert_is_raised(self, pump_world) -> None:
        assert pump_world.source.list_alert_ids() == []

    def test_an_unknown_alert_raises_the_siblings_not_found_error(self, pump_world) -> None:
        from maintenance_agent.ports import NotFoundError

        with pytest.raises(NotFoundError):
            pump_world.source.get_alert("nope")
        with pytest.raises(NotFoundError):
            pump_world.source.get_evidence("nope")

    def test_the_alert_is_a_snapshot_and_a_copy(self, pump_world) -> None:
        world = pump_world
        alert_id = raised(world.source)
        before = world.source.get_alert(alert_id)
        world.advance(days=1)
        assert world.source.get_alert(alert_id) == before  # not re-read from the pipeline
        before.rul.lower_days = 99.0
        assert world.source.get_alert(alert_id).rul.lower_days != 99.0
        ev = world.source.get_evidence(alert_id)
        ev.clear()
        assert world.source.get_evidence(alert_id)


class TestMapping:
    def test_the_interval_is_asset_healths_own_p10_p50_p90(self, pump_world, pump_cache) -> None:
        alert_id = raised(pump_world.source)
        order = pump_cache.analysis("PU-201", pump_world.clock()).work_orders[0]
        rul = pump_world.source.get_alert(alert_id).rul
        assert (rul.lower_days, rul.median_days, rul.upper_days) == (
            order.rul.p10_days,
            order.rul.p50_days,
            order.rul.p90_days,
        )

    def test_confidence_is_asset_healths_target_coverage(self, pump_world, pump_cache) -> None:
        """Mismatch 5: p10 to p90 is an 80% interval; the field says so."""
        rul = pump_world.source.get_alert(raised(pump_world.source)).rul
        assert rul.confidence == pump_cache.analysis_cfg.rul.target_coverage == 0.8

    def test_as_of_is_the_simulation_date_not_the_wall_clock(self, pump_world) -> None:
        alert = pump_world.source.get_alert(raised(pump_world.source))
        assert alert.rul.as_of == sim_time(39.5).date()
        assert alert.raised_at == pump_world.clock()

    def test_the_severity_map_covers_every_asset_health_severity(self) -> None:
        from asset_health.models import Severity

        assert set(SEVERITY) == {s.value for s in Severity}
        assert {SEVERITY[s] for s in ("watch", "warning")} == {"WARN"}
        assert {SEVERITY[s] for s in ("urgent", "critical")} == {"ALARM"}

    def test_the_hypothesis_map_covers_every_fault_hypothesis(self) -> None:
        from asset_health.models import FaultHypothesis

        assert set(HYPOTHESIS_MODE) == {h.value for h in FaultHypothesis}
        from maintenance_agent.models import FailureMode

        agent_modes = {m.value for m in FailureMode}
        assert {m for m in HYPOTHESIS_MODE.values() if m} <= agent_modes
        assert HYPOTHESIS_MODE["overheating"] is None  # mismatch 3: the agent has no such mode

    def test_a_fused_score_above_one_is_clamped_and_counted(self, pump_world, pump_cache) -> None:
        """Mismatch 6: the score is not bounded to [0, 1]; the agent's field is."""
        alert_id = raised(pump_world.source)
        peak = pump_cache.analysis("PU-201", pump_world.clock()).traces[-1].episode.peak_score
        assert peak > 1.0
        assert pump_world.source.get_alert(alert_id).anomaly_score == 1.0
        assert pump_world.source.stats.score_clamped == 1

    def test_asset_healths_latest_safe_date_is_ignored(self, pump_world, pump_cache) -> None:
        """Mismatch 8: the agent plans and the verifier enforces its own policy date."""
        alert_id = raised(pump_world.source)
        order = pump_cache.analysis("PU-201", pump_world.clock()).work_orders[0]
        state = pump_world.orch.run(alert_id, run_id=run_id_for(alert_id))
        agent_date = state.draft["latest_safe_date"]
        alert = pump_world.source.get_alert(alert_id)
        assert agent_date == alert.rul.policy_latest_date().isoformat()
        assert order.latest_safe_date is not None
        assert agent_date != order.latest_safe_date.date().isoformat()


class TestEvidence:
    def test_each_item_is_the_rule_layers_reading_in_robust_sigma(
        self, pump_world, pump_cache
    ) -> None:
        """Mismatch 7: alarm is baseline plus the detection threshold in sigma units."""
        alert_id = raised(pump_world.source)
        order = pump_cache.analysis("PU-201", pump_world.clock()).work_orders[0]
        items = pump_world.source.get_evidence(alert_id)
        assert [e.id for e in items] == [f"{alert_id}-E{i}" for i in range(1, len(items) + 1)]
        assert len(items) == len(order.evidence) > 0
        for item, ev in zip(items, order.evidence, strict=True):
            assert item.signal == FEATURE_SIGNAL[ev.feature]
            assert item.value == pytest.approx(abs(ev.z_score))
            assert (item.baseline, item.alarm, item.unit) == (0.0, Z_ALARM, "robust sigma")
            assert item.exceedance() == pytest.approx(abs(ev.z_score) / Z_ALARM)
            assert item.observed_at == pump_world.clock()
            assert item.comment.startswith("SYNTHETIC") and ev.feature in item.comment

    def test_the_alarm_level_is_the_rule_layers_own_gate(self) -> None:
        """The first condition of a rule is a 3.5-sigma gate; the alarm level is that gate."""
        from asset_health.detect.rules import DEFAULT_RULES

        gates = {c.threshold for r in DEFAULT_RULES for c in r.any_of[:1]}
        assert Z_ALARM == 3.5 and Z_ALARM in gates

    def test_every_asset_health_channel_has_a_counterpart(self) -> None:
        from asset_health.catalog import CHANNEL_DESCRIPTIONS

        assert set(CHANNEL_DESCRIPTIONS) <= set(FEATURE_SIGNAL)

    def test_the_signals_are_in_the_agents_playbook(self) -> None:
        from maintenance_agent.planners.rules import Playbook

        signals = set(Playbook.from_fixtures().signals)
        assert set(FEATURE_SIGNAL.values()) <= signals

    def test_a_falling_channel_has_a_positive_exceedance(self, pump_world, pump_cache) -> None:
        """Pressure and flow are worn when they fall: the item carries the size of the deviation."""
        from asset_health.models import Evidence

        order = pump_cache.analysis("PU-201", pump_world.clock()).work_orders[0]
        fall = Evidence(
            feature="pressure_kpa", description="x", observed=300.0, expected=400.0, z_score=-4.2
        )
        patched = order.model_copy(update={"evidence": [fall]})
        items, _, _ = pump_world.source._evidence_items("a-1", patched, pump_world.clock())
        assert items[0].signal == "disch_pressure_drop_pct"
        assert items[0].value == pytest.approx(4.2)
        assert items[0].exceedance() == pytest.approx(4.2 / Z_ALARM)

    def test_an_unmapped_channel_is_dropped_and_counted(self, pump_world, pump_cache) -> None:
        from asset_health.models import Evidence

        order = pump_cache.analysis("PU-201", pump_world.clock()).work_orders[0]
        odd = Evidence(
            feature="mystery_channel", description="x", observed=1.0, expected=0.5, z_score=4.0
        )
        patched = order.model_copy(update={"evidence": [odd, *order.evidence]})
        items, kept, dropped = pump_world.source._evidence_items("a-1", patched, pump_world.clock())
        assert dropped == ["mystery_channel"] and len(items) == len(order.evidence)
        assert kept == [e.feature for e in order.evidence]
        assert pump_world.source.stats.dropped_features == 1

    def test_the_agents_own_scoring_of_these_readings(self, pump_world) -> None:
        """What the real planner makes of the mapped evidence at this reading (a finding)."""
        from maintenance_agent.planners.rules import Playbook, score_modes

        alert_id = raised(pump_world.source)
        playbook = Playbook.from_fixtures()
        scores = score_modes(pump_world.source.get_evidence(alert_id), playbook, "SLURRY_PUMP")
        assert scores[0].mode == "IMPELLER_WEAR"
        assert scores[0].score < playbook.min_confidence


class TestAssetRegister:
    def test_the_four_plant_assets_come_from_the_agents_own_register(self) -> None:
        register = agent_assets()
        assert set(register) == set(ASSET_IDS)
        for asset_id, info in register.items():
            assert info.asset_class == ASSET_CLASS[asset_id]

    def test_the_bill_of_materials_uses_the_plants_part_numbers(self) -> None:
        """Mismatch 15: the part each plant mode consumes is the part the agent will draft."""
        from maintenance_agent.adapters.synthetic import load_fixture

        register = agent_assets()
        for info in register.values():
            for spec in MODES_BY_CLASS[info.asset_class]:
                component = {
                    "BEARING_DEFECT": "bearing_de",
                    "IMPELLER_WEAR": "impeller",
                    "LINER_WEAR": "liner_set",
                }[spec.mode]
                assert info.bom[component] == spec.part_no
        fixture = {
            a["asset"]["asset_id"]: a["asset"]["bom"] for a in load_fixture("alerts.json")["alerts"]
        }
        # the one difference from the agent's own register, recorded as a finding
        assert fixture["PU-202"]["impeller"] == "SYN-IMP-2002"
        assert register["PU-202"].bom["impeller"] == "SYN-IMP-2001"
        assert fixture["PU-201"] == register["PU-201"].bom


class TestRaise:
    def test_a_new_alert_has_the_plants_id_form(self, pump_world) -> None:
        assert raised(pump_world.source) == "PU-201-L0-A1"

    def test_asking_again_for_the_same_work_order_returns_the_same_alert(self, pump_world) -> None:
        first = raised(pump_world.source)
        again = pump_world.source.raise_alert("PU-201")
        assert (again.kind, again.alert_id) == ("existing", first)
        assert pump_world.source.list_alert_ids() == [first]

    def test_refresh_raises_a_fresh_reading_under_a_fresh_id(self, pump_world) -> None:
        first = raised(pump_world.source)
        second = raised(pump_world.source, refresh=True)
        assert (first, second) == ("PU-201-L0-A1", "PU-201-L0-A2")
        assert pump_world.source.list_alert_ids() == [first, second]

    def test_a_duplicate_alert_id_is_refused(self, pump_world) -> None:
        first = raised(pump_world.source)
        with pytest.raises(DuplicateAlert):
            pump_world.source.raise_alert("PU-201", first, refresh=True)

    def test_an_explicit_id_is_used(self, pump_world) -> None:
        assert raised(pump_world.source, alert_id="PU-201-L0-A7") == "PU-201-L0-A7"

    def test_a_quiet_asset_raises_nothing(self, pump_world) -> None:
        pump_world.now = sim_time(10.0)
        outcome = pump_world.source.raise_alert("PU-201")
        assert outcome.kind == "none" and pump_world.source.list_alert_ids() == []

    def test_an_asset_that_is_not_served_is_refused(self, pump_world) -> None:
        with pytest.raises(KeyError):
            pump_world.source.raise_alert("CR-501")

    def test_construction_needs_a_register_entry_for_every_mapped_asset(self, pump_cache) -> None:
        with pytest.raises(KeyError, match="CR-999"):
            AssetHealthAlertSource(pump_cache, lambda: sim_time(1.0), {"CR-999": "CRU-01"})

    def test_a_reading_with_no_interval_is_not_an_alert(self, params) -> None:
        """Mismatch 4: not-degrading and insufficient-trend readings carry no p10, p50, p90.

        Seed 5 opens a work order on day 37 whose RUL status is not-degrading.
        """
        cache = RunCache()
        cache.set_history("PU-201", pump_lives(params, seed=5)[:1])
        world = make_world(
            params,
            lambda clock: AssetHealthAlertSource(cache, clock, ASSET_MAP),
            now=sim_time(37.0),
        )
        outcome = world.source.raise_alert("PU-201")
        assert outcome.kind == "no_interval" and "not_degrading" in outcome.reason
        assert world.source.list_alert_ids() == []
        assert world.source.stats.no_interval == 1
        assert world.source.stats.missed_status == {"not_degrading": 1}
        # the same episode becomes an alert once asset-health has an interval
        for hours in range(24, 24 * 8, 24):
            world.now = sim_time(37.0) + timedelta(hours=hours)
            if world.source.raise_alert("PU-201").kind == "raised":
                break
        else:
            pytest.fail("no interval was ever published in the eight days that followed")
        assert world.source.stats.raised == 1


class TestIntervalHook:
    def test_a_conformal_interval_replaces_asset_healths(self, params, pump_cache) -> None:
        def conformal(asset: str, estimate) -> tuple[float, float, float, float]:
            assert asset == "PU-201" and estimate.status.value == "estimated"
            return 3.0, 5.0, 9.0, 0.9

        world = make_world(
            params,
            lambda clock: AssetHealthAlertSource(pump_cache, clock, ASSET_MAP, interval=conformal),
            now=sim_time(39.5),
        )
        rul = world.source.get_alert(raised(world.source)).rul
        assert (rul.lower_days, rul.median_days, rul.upper_days, rul.confidence) == (3, 5, 9, 0.9)
        state = world.orch.run("PU-201-L0-A1", run_id="conformal")
        assert state.draft["latest_safe_date"] == "2026-11-11"  # as_of + 3 days - 1 margin day

    def test_none_keeps_asset_healths_interval(self, params, pump_cache) -> None:
        world = make_world(
            params,
            lambda clock: AssetHealthAlertSource(
                pump_cache, clock, ASSET_MAP, interval=lambda a, e: None
            ),
            now=sim_time(39.5),
        )
        assert world.source.get_alert(raised(world.source)).rul.confidence == 0.8


class TestRunCache:
    def test_the_pipeline_runs_once_per_history(self, lives) -> None:
        cache = RunCache()
        assert not cache.has("PU-201")
        assert cache.set_history("PU-201", lives[:1]) is True
        assert cache.run_asset_calls == 1 and cache.has("PU-201") and cache.pipeline_seconds > 0
        assert cache.set_history("PU-201", lives[:1]) is False
        assert cache.run_asset_calls == 1
        assert cache.set_history("PU-201", lives) is True  # a join changes the history
        assert cache.run_asset_calls == 2
        assert len(cache.segments("PU-201")) == 2

    def test_an_analysis_is_made_once_per_instant(self, pump_cache) -> None:
        before = pump_cache.analyse_calls
        a = pump_cache.analysis("PU-201", sim_time(30.0))
        assert pump_cache.analysis("PU-201", sim_time(30.0)) is a
        assert pump_cache.analyse_calls == before + 1 and pump_cache.analyse_seconds > 0

    def test_a_naive_and_an_aware_instant_are_the_same_instant(self, pump_cache) -> None:
        from closed_loop.f2.alert_source import aware

        a = pump_cache.analysis("PU-201", sim_time(31.0))
        assert pump_cache.analysis("PU-201", aware(sim_time(31.0))) is a

    def test_a_new_history_drops_the_old_analyses(self, lives) -> None:
        cache = RunCache()
        cache.set_history("PU-201", lives[:1])
        old = cache.analysis("PU-201", sim_time(30.0))
        cache.set_history("PU-201", lives)
        assert cache.analysis("PU-201", sim_time(30.0)) is not old

    def test_an_unknown_asset_is_refused(self, lives) -> None:
        with pytest.raises(KeyError):
            RunCache().set_history("XX-1", lives[:1])


class TestConfidenceIsNotRead:
    """0001 section 15 item 3: does any agent rule read ``RULInterval.confidence``? No."""

    def test_no_source_line_in_the_agent_reads_it(self) -> None:
        import maintenance_agent

        root = Path(maintenance_agent.__file__).parent
        readers = []
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr == "confidence":
                    if "rul" in ast.unparse(node.value):
                        readers.append(f"{path.name}:{node.lineno}")
        assert readers == []

    def test_the_draft_and_the_verdict_do_not_depend_on_it(self, params, pump_cache) -> None:
        def run(level: float):
            world = make_world(
                params,
                lambda clock: AssetHealthAlertSource(
                    pump_cache,
                    clock,
                    ASSET_MAP,
                    interval=lambda a, e: (e.p10_days, e.p50_days, e.p90_days, level),
                ),
                now=sim_time(39.5),
            )
            return world.orch.run(raised(world.source), run_id="r")

        low, high = run(0.5), run(0.99)
        assert low.draft_hash == high.draft_hash
        assert low.verification.passed and high.verification.passed
        assert low.verification.rule_ids() == high.verification.rule_ids() == set()
        # it is data all the same: the alert fingerprint covers it, so the digest differs
        assert low.verification.digest != high.verification.digest

    def test_a_change_of_confidence_alone_voids_the_draft(self, pump_world) -> None:
        """The field is unread by the rules but not inert: V14 sees a changed alert."""
        world = pump_world
        alert_id = raised(world.source)
        world.orch.run(alert_id, run_id="r")
        world.source._alerts[alert_id].rul.confidence = 0.5
        state = world.orch.refresh_verification("r")
        assert state.status.value == "BLOCKED" and state.verification.rule_ids() == {"V14"}
