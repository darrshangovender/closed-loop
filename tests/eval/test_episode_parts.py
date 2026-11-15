"""The parts of the episode loop on their own: operating clock, audit hold, label delivery."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest

from closed_loop.core.types import Command, Op, Permit, PlanCmd
from closed_loop.eval.arms import ARMS
from closed_loop.eval.context import PreparedContext
from closed_loop.eval.episode import (
    AUDIT_LOWER_K,
    _AuditHold,
    _Counting,
    _OperatingClock,
    _Perception,
    scenario_for,
    simulate_episode,
)
from closed_loop.perception import conformal
from closed_loop.perception.conformal import is_audit_life
from closed_loop.perception.estimator import TheilSenEstimator
from closed_loop.perception.labels import Label
from closed_loop.perception.monitor import CoverageMonitor
from closed_loop.plant.params import ASSET_IDS
from closed_loop.plant.plant import AssetView, Observation

# written before benchmarks/tuned.json existed: they run on the provisional defaults (conftest)
pytestmark = pytest.mark.usefixtures("provisional_defaults_module")

CR = "CR-501"


def view(aid: str, op: Op, **kw: Any) -> AssetView:
    base: dict[str, Any] = {
        "asset_id": aid,
        "op": op,
        "permit": Permit.NONE,
        "occupied": False,
        "gas_age_q": 99,
        "alarm": False,
        "soon": False,
        "set_lo": 0,
        "set_hi": 3,
        "rul_lo": None,
        "rul_hi": None,
        "prior_only": False,
        "hi_last": None,
        "job_open": False,
        "spares": {},
    }
    return AssetView(**{**base, **kw})


def assets(ops: Mapping[str, Op], **kw: Any) -> Mapping[str, AssetView]:
    return MappingProxyType({a: view(a, ops.get(a, Op.RUN), **kw) for a in ASSET_IDS})


def obs(q: int, mapping: Mapping[str, AssetView]) -> Observation:
    return Observation(q, q // 96, 0, mapping, None, (), 2, True, False)


class TestOperatingClock:
    def test_a_running_asset_accumulates_one_quarter_per_view_after_the_first(self) -> None:
        clock = _OperatingClock()
        m = assets({})
        for q in range(10):
            clock.observe(obs(q, m))
        assert clock.u[CR] == 9  # the view at quarter 0 has no step before it

    def test_a_stop_freezes_the_clock_and_a_restart_resumes_it(self) -> None:
        clock = _OperatingClock()
        run, stopped = assets({}), assets({CR: Op.STOP})
        for q in range(10):
            clock.observe(obs(q, run))
        assert clock.u[CR] == 9 and clock.stop_u[CR] is None
        for q in range(10, 15):
            clock.observe(obs(q, stopped))
        assert clock.u[CR] == 9 and clock.stop_u[CR] == 9
        clock.observe(obs(15, assets({})))  # START executed in step 14: the part aged in it
        assert clock.u[CR] == 10
        assert clock.u["CV-101"] == 15  # the others ran throughout

    def test_a_quiet_quarter_reuses_the_mapping_and_still_counts(self) -> None:
        clock = _OperatingClock()
        m = assets({CR: Op.WORK})
        for q in range(1, 6):
            clock.observe(obs(q, m))  # the same mapping object each time
        assert clock.u[CR] == 0 and clock.u["PU-201"] == 5

    def test_a_failure_ends_the_life_with_the_failing_quarter_counted_as_run(self) -> None:
        clock = _OperatingClock()
        run = assets({})
        for q in range(30):
            clock.observe(obs(q, run))
        before = clock.u[CR]
        clock.observe(obs(30, assets({CR: Op.FAIL})))
        assert clock.failure_pending[CR] is True
        assert clock.fail_q[CR] == 30 and clock.fail_u[CR] == before + 1
        assert clock.stop_u[CR] == before

    def test_only_a_running_asset_can_fail_so_the_repair_states_are_not_failures(self) -> None:
        clock = _OperatingClock()
        clock.observe(obs(1, assets({CR: Op.FAIL})))
        clock.failure_pending[CR] = False
        for q, op in enumerate((Op.LOCKED, Op.ISO, Op.WORK, Op.ISO, Op.STOP), start=2):
            clock.observe(obs(q, assets({CR: op})))
        assert clock.failure_pending[CR] is False
        clock.observe(obs(7, assets({})))
        clock.observe(obs(8, assets({CR: Op.FAIL})))  # fails again after a wrong-mode start
        assert clock.failure_pending[CR] is True

    def test_each_asset_is_counted_on_its_own(self) -> None:
        clock = _OperatingClock()
        for q in range(8):
            clock.observe(obs(q, assets({"PU-201": Op.STOP} if q >= 4 else {})))
        assert clock.u["PU-201"] == 3 and clock.u["PU-202"] == 7


class _FakeEstimator:
    """Records the calls of ``update``; returns a fresh object each time."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, float | None, int]] = []

    def update(self, aid: str, day: int, hi: float | None, life: int) -> object:
        self.calls.append((aid, day, hi, life))
        return object()


def perception_stub(clock: _OperatingClock | None = None) -> Any:
    """Just the attributes ``_estimate``, ``operating_day`` and ``renewed`` read; the methods are
    the real ones, bound with ``__get__``."""
    ns = SimpleNamespace(
        clock=clock or _OperatingClock(),
        estimator=_FakeEstimator(),
        _last={},
        _life_seen=dict.fromkeys(ASSET_IDS, 0),
        estimator_updates=0,
        handed=[],
    )
    ns.operating_day = _Perception.operating_day.__get__(ns)
    ns._estimate = _Perception._estimate.__get__(ns)
    ns._hand_to_plant = lambda aid, out: ns.handed.append(aid)
    ns.renewed = _Perception.renewed.__get__(ns)
    return ns


class TestTheEstimatorRunsOnOperatingTime:
    """0003 A18 R1: the estimator day is the quarters the asset has run over DAY_Q, not the
    calendar day, so a stopped asset neither ages nor is given a reading it did not take."""

    def test_an_asset_that_never_stops_has_the_calendar_day(self) -> None:
        clock = _OperatingClock()
        ns = perception_stub(clock)
        for q in range(0, 5 * 96 + 1):
            clock.observe(obs(q, assets({})))
            if q % 96 == 0:
                assert ns.operating_day(CR) == q // 96, q

    def test_a_stop_of_three_days_makes_the_operating_day_three_days_behind(self) -> None:
        clock = _OperatingClock()
        ns = perception_stub(clock)
        for q in range(0, 2 * 96):
            clock.observe(obs(q, assets({})))
        for q in range(2 * 96, 5 * 96):
            clock.observe(obs(q, assets({CR: Op.STOP})))
        for q in range(5 * 96, 6 * 96 + 1):
            clock.observe(obs(q, assets({})))
        assert ns.operating_day(CR) == 6 - 3
        assert ns.operating_day("CV-101") == 6  # the others ran

    def test_a_stopped_asset_is_not_updated_again_and_keeps_its_output(self) -> None:
        ns = perception_stub()
        a, fresh_a = ns._estimate(CR, 1.0, 0, 7)
        b, fresh_b = ns._estimate(CR, 2.0, 0, 7)  # the same operating day: it did not run
        assert (fresh_a, fresh_b) == (True, False) and a is b
        assert ns.estimator.calls == [(CR, 7, 1.0, 0)] and ns.estimator_updates == 1

    def test_the_next_operating_day_and_a_new_life_both_update(self) -> None:
        ns = perception_stub()
        ns._estimate(CR, 1.0, 0, 7)
        _, fresh = ns._estimate(CR, 1.5, 0, 8)
        assert fresh
        _, fresh = ns._estimate(CR, None, 1, 8)
        assert fresh
        assert [c[1:] for c in ns.estimator.calls] == [(7, 1.0, 0), (8, 1.5, 0), (8, None, 1)]

    def test_in_a_whole_episode_the_day_the_estimator_is_given_is_the_operating_day(
        self, small_context: PreparedContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """B3-S on seed 1101 is stopped by its shield (it stalled before A18): the estimator day
        is the asset own clock at every call, falls behind the calendar once it has stopped, and
        never repeats within a life (the estimator would raise)."""
        held: dict[str, Any] = {}
        seen: list[tuple[str, int, int, int, int]] = []
        real_daily, real_renewed = _Perception.daily, _Perception.renewed
        real_update = TheilSenEstimator.update
        real_observe_day = CoverageMonitor.observe_day
        proxy_days: list[tuple[str, int, int]] = []

        def observe_day(
            self: Any, aid: str, day: int, hi: Any, trend: Any, life_id: int = 0
        ) -> None:
            proxy_days.append((aid, life_id, day))
            real_observe_day(self, aid, day, hi, trend, life_id)

        def daily(self: Any, observation: Observation) -> None:
            held["p"], held["obs"] = self, observation
            real_daily(self, observation)

        def renewed(self: Any, observation: Observation) -> None:
            held["p"], held["obs"] = self, observation
            real_renewed(self, observation)

        def update(self: Any, aid: str, day: int, hi: Any, life: int) -> Any:
            p = held["p"]
            seen.append((aid, day, life, p.clock.u[aid] // 96, held["obs"].day))
            return real_update(self, aid, day, hi, life)

        monkeypatch.setattr(_Perception, "daily", daily)
        monkeypatch.setattr(_Perception, "renewed", renewed)
        monkeypatch.setattr(TheilSenEstimator, "update", update)
        monkeypatch.setattr(CoverageMonitor, "observe_day", observe_day)
        simulate_episode(ARMS["B3-S"], 1101, scenario_for("nominal"), context=small_context)
        assert seen and proxy_days
        # the proxy monitor scores one reading per operating day: a stopped asset gives none
        proxy_by_life: dict[tuple[str, int], list[int]] = {}
        for aid, life, day in proxy_days:
            proxy_by_life.setdefault((aid, life), []).append(day)
        for days in proxy_by_life.values():
            assert all(b > a for a, b in zip(days, days[1:], strict=False)), days
        assert all(day == op_day for _a, day, _l, op_day, _c in seen)
        assert any(day < cal for _a, day, _l, _o, cal in seen), "no asset ever fell behind"
        per_life: dict[tuple[str, int], list[int]] = {}
        for aid, day, life, _o, _c in seen:
            per_life.setdefault((aid, life), []).append(day)
        for days in per_life.values():
            assert all(b > a for a, b in zip(days, days[1:], strict=False))


class _FiredMonitor:
    """Stands in for a monitor that has fired, with the day it stamped."""

    def __init__(self, fired_day: float | None, fired: bool = True) -> None:
        self.fired = fired
        self.fired_day = fired_day


class TestMonitorFiringDaysAreOnTheCalendar:
    """0003 A18, F2. The proxy monitor is fed operating days and stamps one; ``m_detect`` compares
    the stamp with the calendar ``shift_day``. The recorded day is the calendar day."""

    def stub(self, **monitors: _FiredMonitor) -> Any:
        ns = SimpleNamespace(monitors=monitors, fired_day=dict.fromkeys(monitors))
        ns.note = _Perception._note_firings.__get__(ns)
        return ns

    def test_a_proxy_stamped_on_the_operating_clock_is_recorded_on_the_calendar(self) -> None:
        # an asset stood stopped for 10 days: its operating day is 42 when the calendar says 52
        ns = self.stub(proxy=_FiredMonitor(42.0))
        ns.note(52.0)
        assert ns.fired_day["proxy"] == 52.0

    def test_a_label_mode_keeps_the_calendar_day_of_the_label_that_completed_the_evidence(
        self,
    ) -> None:
        ns = self.stub(opt=_FiredMonitor(47.0), pess=_FiredMonitor(None))
        ns.note(52.0)
        assert ns.fired_day["opt"] == 47.0  # resolved_day, calendar
        assert ns.fired_day["pess"] == 52.0  # no stamp: the day it was noticed

    def test_a_monitor_that_has_not_fired_is_left_alone_and_the_first_firing_stands(self) -> None:
        ns = self.stub(proxy=_FiredMonitor(None, fired=False), opt=_FiredMonitor(30.0))
        ns.note(40.0)
        assert ns.fired_day == {"proxy": None, "opt": 30.0}
        ns.monitors["proxy"].fired = True
        ns.note(41.0)
        ns.note(60.0)
        assert ns.fired_day["proxy"] == 41.0

    def test_detection_reads_the_calendar_day_against_the_calendar_shift(self) -> None:
        from closed_loop.eval.metrics import METRICS

        ns = self.stub(proxy=_FiredMonitor(65.0))  # on the operating clock this would read as late
        ns.note(72.0)
        assert 72.0 - 40.0 > 30.0 >= 65.0 - 40.0  # the two clocks give different answers here
        from tests.eval.test_metrics import result

        r = result(monitors={"proxy": ns.fired_day["proxy"]}, shift_day=40.0)
        assert METRICS["m_detect_proxy"].value(r) == 0.0  # 32 days after the shift: outside 30
        r = result(monitors={"proxy": 65.0}, shift_day=40.0)
        assert METRICS["m_detect_proxy"].value(r) == 1.0  # what the operating stamp would claim


class TestACompletedJobIsARenewalAtOnce:
    """0003 A18 R3: in a quarter that is not the first of the day, a moved ``jobs_done`` gives the
    estimator a new life and hands the plant the new part interval and set, so that a shielded
    arm is not held to the old part set for the rest of the day."""

    def with_done(self, done: int) -> Observation:
        m = MappingProxyType(
            {a: view(a, Op.RUN, jobs_done=(done if a == CR else 0)) for a in ASSET_IDS}
        )
        return obs(50, m)

    def test_a_moved_count_starts_a_new_life_with_no_reading_and_tells_the_plant(self) -> None:
        ns = perception_stub()
        ns.renewed(self.with_done(1))
        assert ns.estimator.calls == [(CR, ns.operating_day(CR), None, 1)]
        assert ns.handed == [CR] and ns._life_seen[CR] == 1 and ns.estimator_updates == 1
        assert CR in ns._last and ns._last[CR][1] == 1

    def test_an_unchanged_count_does_nothing(self) -> None:
        ns = perception_stub()
        ns.renewed(self.with_done(0))
        assert ns.estimator.calls == [] and ns.handed == [] and ns._last == {}

    def test_the_same_count_twice_renews_once(self) -> None:
        ns = perception_stub()
        ns.renewed(self.with_done(1))
        ns.renewed(self.with_done(1))
        assert len(ns.estimator.calls) == 1 and ns.handed == [CR]


class TestAStoppedAssetKeepsItsOutputWhileItStands:
    """0003 A18 R1, the observable form: while an asset stands stopped its estimate is the same
    object, so its prior does not fall with the calendar and its status cannot change."""

    def test_the_estimator_output_handed_to_the_plant_is_the_same_while_the_asset_does_not_run(
        self, small_context: PreparedContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[str, int, int, object]] = []
        real = _Perception._hand_to_plant

        def spy(self: Any, aid: str, out: Any) -> Any:
            calls.append((aid, self.clock.u[aid] // 96, self._life_seen[aid], out))
            return real(self, aid, out)

        monkeypatch.setattr(_Perception, "_hand_to_plant", spy)
        simulate_episode(ARMS["B3-S"], 1101, scenario_for("nominal"), context=small_context)
        by_asset: dict[str, list[tuple[int, int, object]]] = {}
        for aid, op_day, life, out in calls:
            by_asset.setdefault(aid, []).append((op_day, life, out))
        repeats = 0
        for seq in by_asset.values():
            for (d1, l1, o1), (d2, l2, o2) in zip(seq, seq[1:], strict=False):
                if (d1, l1) == (d2, l2):
                    repeats += 1
                    assert o1 is o2, "the asset did not run, yet the estimate was recomputed"
                else:
                    assert o1 is not o2
        assert repeats > 0, "no asset ever stood stopped across a day: the check is vacuous"


class TestAWrongRepairUnderAShieldDoesNotLeaveTheAssetStuck:
    """0003 A18 R3. Calendar maintenance (B1) repairs the crusher with the wrong mode on seed 1103
    (it fits nothing, 0003 A3). Behind a shield the old arrangement left the estimator and the
    shield on the old part, a day at a time. Now a completed job renews the life in the quarter it
    completes and the plant is told, so the asset is running again at once."""

    def test_every_completed_job_is_followed_by_a_restart_within_a_few_quarters(
        self, small_context: PreparedContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dataclasses

        from closed_loop.core.trace import op_index

        renewals: list[tuple[int, str]] = []
        real = _Perception.renewed

        def spy(self: Any, observation: Observation) -> None:
            before = dict(self._life_seen)
            real(self, observation)
            renewals.extend(
                (observation.quarter, a) for a in ASSET_IDS if self._life_seen[a] != before[a]
            )

        monkeypatch.setattr(_Perception, "renewed", spy)
        arm = dataclasses.replace(
            ARMS["B1"], name="B1-S", shield=True, perception=ARMS["B3-S"].perception
        )
        rec = simulate_episode(arm, 1103, scenario_for("nominal"), context=small_context)
        assert rec.plant.stats().wrong_mode_jobs >= 1, "the seed no longer gives a wrong repair"
        assert renewals, "no job completed in a quarter that is not a day's first"
        running = op_index(Op.RUN)
        for q, aid in renewals:
            window = rec.trace.op[aid][q : q + 8]
            assert (window == running).any(), (aid, q, "stuck stopped after the job")


class StubProposer:
    name = "stub"

    def __init__(self, commands: list[Command]) -> None:
        self.commands = commands
        self.calls = 0
        self.seed: int | None = None

    def reset(self, seed: int) -> None:
        self.seed = seed

    def propose(self, observation: Observation) -> list[Command]:
        self.calls += 1
        return list(self.commands)


WO = Command(CR, PlanCmd.RAISE_WO, "LINER_WEAR:0", "stub")
OTHER = Command("CV-101", PlanCmd.RAISE_WO, "BEARING_DEFECT:0", "stub")


class TestAuditHold:
    def hold(self, audit: bool) -> tuple[_AuditHold, StubProposer]:
        inner = StubProposer([WO, OTHER])
        return _AuditHold(inner, lambda aid: audit and aid == CR), inner

    def test_an_audit_life_is_held_until_its_lower_bound_is_under_a_day(self) -> None:
        hold, _ = self.hold(True)
        assert hold.propose(obs(10, assets({}, rul_lo=5.0))) == [OTHER]
        assert hold.propose(obs(11, assets({}, rul_lo=AUDIT_LOWER_K))) == [
            OTHER
        ]  # not yet: "under"
        assert hold.propose(obs(12, assets({}, rul_lo=AUDIT_LOWER_K - 0.01))) == [WO, OTHER]
        assert hold.held == 2

    def test_an_ordinary_life_is_never_held(self) -> None:
        hold, _ = self.hold(False)
        assert hold.propose(obs(10, assets({}, rul_lo=50.0))) == [WO, OTHER]
        assert hold.held == 0

    def test_a_failed_asset_is_always_repaired(self) -> None:
        hold, _ = self.hold(True)
        assert WO in hold.propose(obs(10, assets({CR: Op.FAIL}, rul_lo=50.0)))

    def test_without_an_interval_there_is_nothing_to_wait_for(self) -> None:
        hold, _ = self.hold(True)
        assert WO in hold.propose(obs(10, assets({})))

    def test_only_work_orders_are_held_and_the_job_commands_pass(self) -> None:
        inner = StubProposer(
            [Command(CR, PlanCmd.INSPECT), Command(CR, PlanCmd.ORDER, "SYN-LNR-5001")]
        )
        hold = _AuditHold(inner, lambda aid: True)
        assert len(hold.propose(obs(1, assets({}, rul_lo=9.0)))) == 2

    def test_reset_is_forwarded_and_clears_the_count(self) -> None:
        hold, inner = self.hold(True)
        hold.propose(obs(1, assets({}, rul_lo=9.0)))
        hold.reset(7)
        assert inner.seed == 7 and hold.held == 0


class TestCounting:
    def test_it_counts_what_the_inner_proposer_issues_and_changes_nothing(self) -> None:
        inner = StubProposer([WO, OTHER])
        counted = _Counting(inner)
        assert counted.propose(obs(1, assets({}))) == [WO, OTHER]
        counted.propose(obs(2, assets({})))
        assert counted.proposed == 4 and counted.name == "stub"
        counted.reset(3)
        assert counted.proposed == 0 and inner.seed == 3


class TestDeliveryOfLabels:
    """Providers get the labels of one life at a time, in time order, without the truth."""

    @pytest.fixture
    def delivered(
        self, small_context: PreparedContext, monkeypatch: pytest.MonkeyPatch
    ) -> list[list[Label]]:
        calls: list[list[Label]] = []
        real = conformal._Online.observe

        def spy(self: Any, labels: Any) -> None:
            calls.append(list(labels))
            real(self, labels)

        monkeypatch.setattr(conformal._Online, "observe", spy)
        return calls

    def test_one_life_per_call_in_non_decreasing_resolution_order(
        self, small_context: PreparedContext, delivered: list[list[Label]]
    ) -> None:
        for seed in (1100, 1101, 1103):
            delivered.clear()
            simulate_episode(ARMS["B3-V3"], seed, scenario_for("nominal"), context=small_context)
            assert delivered, "an online provider should have been told about some life"
            days = [c[0].resolved_day for c in delivered]
            assert days == sorted(days)
            for labels in delivered:
                assert len({(lab.asset_id, lab.life_id) for lab in labels}) == 1
                assert len({lab.resolved_day for lab in labels}) == 1

    def test_a_learner_never_sees_the_truth_and_the_oracle_does(
        self, small_context: PreparedContext, delivered: list[list[Label]]
    ) -> None:
        simulate_episode(ARMS["B3-V3"], 1100, scenario_for("nominal"), context=small_context)
        assert delivered and all(lab.true_rul is None for c in delivered for lab in c)
        delivered.clear()
        simulate_episode(ARMS["B3-V7"], 1100, scenario_for("nominal"), context=small_context)
        assert delivered and all(lab.true_rul is not None for c in delivered for lab in c)

    def test_a_label_reaches_the_provider_no_earlier_than_its_resolution_day(
        self, small_context: PreparedContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[tuple[int, int]] = []
        real = conformal._Online.observe

        def spy(self: Any, labels: Any) -> None:
            seen.append((labels[0].resolved_day, self._n_labels))
            real(self, labels)

        monkeypatch.setattr(conformal._Online, "observe", spy)
        from closed_loop.eval import episode as episode_module

        days_at_delivery: list[int] = []
        real_daily = episode_module._Perception.daily

        def daily(self: Any, observation: Observation) -> None:
            before = len(seen)
            real_daily(self, observation)
            days_at_delivery.extend([observation.day] * (len(seen) - before))

        monkeypatch.setattr(episode_module._Perception, "daily", daily)
        simulate_episode(ARMS["B3-V3"], 1100, scenario_for("nominal"), context=small_context)
        assert seen and len(seen) == len(days_at_delivery)
        for (resolved_day, _n), day in zip(seen, days_at_delivery, strict=True):
            assert resolved_day <= day


class TestAuditAssignment:
    def test_the_loop_draws_audit_lives_with_the_providers_rule(self) -> None:
        # is_audit_life is the rule both the V4 provider and the audit hold use
        draws = [is_audit_life(1100, a, life) for a in ASSET_IDS for life in range(50)]
        assert 0 < sum(draws) < len(draws) / 3  # rho = 0.10
        assert is_audit_life(1100, CR, 3) == is_audit_life(1100, CR, 3)

    def test_the_v4_arm_holds_work_orders_of_audit_lives_and_the_others_have_no_hold(
        self, small_context: PreparedContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from closed_loop.eval import episode as episode_module

        made: list[_AuditHold] = []

        class Recording(_AuditHold):
            def __init__(self, *a: Any, **k: Any) -> None:
                super().__init__(*a, **k)
                made.append(self)

        monkeypatch.setattr(episode_module, "_AuditHold", Recording)
        for name in ("B3-V3", "B0", "H-C0"):
            simulate_episode(ARMS[name], 1100, scenario_for("nominal"), context=small_context)
        assert made == []
        held = 0
        for seed in range(1100, 1112):
            made.clear()
            rec = simulate_episode(
                ARMS["B3-V4"], seed, scenario_for("nominal"), context=small_context
            )
            assert len(made) == 1
            held += made[0].held
            assert rec.result.unplanned_failures >= 0
        assert held > 0

    def test_the_audit_monitor_exists_for_v4_only(self, small_context: PreparedContext) -> None:
        v4 = simulate_episode(
            ARMS["B3-V4-audit"], 1100, scenario_for("nominal"), context=small_context
        )
        v3 = simulate_episode(ARMS["B3-V3"], 1100, scenario_for("nominal"), context=small_context)
        assert "audit" in v4.result.monitors and "audit" not in v3.result.monitors
        assert v4.result.monitor_mode == "audit"
