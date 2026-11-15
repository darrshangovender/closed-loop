"""P8b: ``PlannerProposer`` as a controller: job choice, bookkeeping, diagnosis, a real episode.

Most tests drive ``propose`` with hand-built observations and a planner whose verdict is fixed, so
that they are about the plumbing (what the proposer does with a verdict) and not about the maths.
The last class runs short episodes on the SYNTHETIC plant.
"""

from __future__ import annotations

import dataclasses
from types import MappingProxyType

import numpy as np
import pytest

from closed_loop.causal.diagnoser import Diagnosis
from closed_loop.core.timegrid import DAY_Q
from closed_loop.core.types import Cmd, Command, Op, Permit, PlanCmd
from closed_loop.planning.planner import AssetAssessment
from closed_loop.planning.proposer import PlannerProposer, default_models
from closed_loop.planning.worldmodel import ACTION_INSPECT, ACTION_REPLACE
from closed_loop.plant import PlantParams, Scenario, make_plant
from closed_loop.plant.des import APPROVED, JobView
from closed_loop.plant.jobs import ALPHABET
from closed_loop.plant.observe import DailyObs
from closed_loop.plant.plant import InspectResult
from tests.planning._obs import ASSETS, observation, view

PUMP_PART = "SYN-IMP-2001"
SPARES = MappingProxyType({"SYN-BRG-1001": 3, "SYN-BRG-2001": 3, PUMP_PART: 2, "SYN-LNR-5001": 1})


@pytest.fixture(scope="module")
def models():
    return default_models(fit_seeds=range(1000, 1030))


class StubDiagnoser:
    """Always says the same thing, so that a test can say what the diagnosis was."""

    name = "stub"

    def __init__(self, cause: str = "degradation", mode: str = "BEARING_DEFECT") -> None:
        self.cause, self.mode = cause, mode
        self.calls = 0

    def fit(self, lives) -> None:  # type: ignore[no-untyped-def]
        pass

    def diagnose(self, asset_id, day, history):  # type: ignore[no-untyped-def]
        self.calls += 1
        return Diagnosis(asset_id, day, self.mode, self.cause, {f"{self.cause}:{self.mode}": 1.0})


def verdicts(proposer: PlannerProposer, wanting=(), inspecting=()) -> None:
    """Fix the planner's assessment of every asset."""
    q = np.array([1.0, 1.0, 0.0])
    zero = np.zeros(15)

    def assessment(aid: str, tracker) -> AssetAssessment:  # type: ignore[no-untyped-def]
        action = ACTION_REPLACE if aid in wanting else ACTION_INSPECT if aid in inspecting else 0
        return AssetAssessment(aid, action, q, q, np.zeros(3), 0.0, True, 0.0, True, 0, zero, zero)

    proposer.planner._assess_cached = assessment  # type: ignore[method-assign]


def obs_for(day: int = 30, quarter_in_day: int = 0, **overrides):
    views = {a: view(a, spares=SPARES, day=day, hi=2.0) for a in ASSETS}
    views.update(overrides)
    return observation(views, day=day, quarter=day * DAY_Q + quarter_in_day)


def make(models, diagnoser=None, **kw) -> PlannerProposer:
    return PlannerProposer(models, n_particles=32, diagnoser=diagnoser, seed=1, **kw)


def raise_wos(cmds: list[Command]) -> list[Command]:
    return [c for c in cmds if c.cmd is PlanCmd.RAISE_WO]


def a_job(aid: str, *, status: str = APPROVED, ready: bool = True, mode: str = "BEARING_DEFECT"):
    return JobView(aid, f"WO-{aid}-1", mode, status, ready, False, 12, 0, "SYN-BRG-1001", True)


# --------------------------------------------------------------------------------- job choice
class TestAPlanIsAJobChoice:
    def test_the_proposer_has_the_shape_of_the_proposer_protocol(self, models) -> None:
        p = make(models)
        assert p.name == "planner" and callable(p.reset) and callable(p.propose)

    def test_a_replace_verdict_becomes_a_work_order_with_the_diagnosed_mode(self, models) -> None:
        p = make(models, StubDiagnoser("degradation", "BEARING_DEFECT"))
        verdicts(p, wanting=("CV-101",))
        out = p.propose(obs_for())
        (wo,) = raise_wos(out)
        assert wo.asset_id == "CV-101"
        assert wo.arg == f"BEARING_DEFECT:{(30 * DAY_Q) // 32}"
        assert wo.source == "planner"

    def test_an_open_work_order_is_carried_out_by_the_job_runner(self, models) -> None:
        p = make(models)
        verdicts(p)
        job = a_job("CV-101")
        out = p.propose(obs_for(**{"CV-101": view("CV-101", job=job, spares=SPARES)}))
        # an approved, released job on a running asset: the first command is STOP
        assert out == [Command("CV-101", Cmd.STOP, None, "planner")]

    def test_every_micro_command_comes_from_the_runner_alphabet(self, models) -> None:
        p = make(models)
        verdicts(p)
        for op, permit in ((Op.STOP, Permit.NONE), (Op.LOCKED, Permit.NONE), (Op.ISO, Permit.NONE)):
            job = a_job("CV-101")
            v = dataclasses.replace(view("CV-101", job=job, spares=SPARES), op=op, permit=permit)
            out = p.propose(obs_for(**{"CV-101": v}))
            assert out and all(c.cmd in ALPHABET for c in out)

    def test_a_planning_command_is_never_a_micro_command(self, models) -> None:
        p = make(models, StubDiagnoser())
        verdicts(p, wanting=("CV-101",), inspecting=("PU-201",))
        out = p.propose(obs_for())
        assert {c.cmd for c in out} <= {PlanCmd.RAISE_WO, PlanCmd.INSPECT, PlanCmd.ORDER}


class TestRetryAndCooldown:
    def test_a_refused_work_order_is_raised_again_for_two_hours(self, models) -> None:
        p = make(models, StubDiagnoser())
        verdicts(p, wanting=("CV-101",))
        assert len(raise_wos(p.propose(obs_for(quarter_in_day=0)))) == 1
        for q in range(1, 9):  # not accepted: no job on the next observations
            assert len(raise_wos(p.propose(obs_for(quarter_in_day=q)))) == 1
        assert raise_wos(p.propose(obs_for(quarter_in_day=9))) == []

    def test_once_the_job_is_open_the_work_order_is_not_raised_again(self, models) -> None:
        p = make(models, StubDiagnoser())
        verdicts(p, wanting=("CV-101",))
        p.propose(obs_for(quarter_in_day=0))
        opened = obs_for(
            quarter_in_day=1,
            **{"CV-101": view("CV-101", job=a_job("CV-101", ready=False), spares=SPARES)},
        )
        assert raise_wos(p.propose(opened)) == []

    def test_a_withdrawn_work_order_is_not_raised_again_within_the_day(self, models) -> None:
        p = make(models, StubDiagnoser())
        verdicts(p, wanting=("CV-101",))
        p.propose(obs_for(quarter_in_day=0))
        later = obs_for(day=30, quarter_in_day=40)  # the retry window is over, same day
        assert raise_wos(p.propose(later)) == []

    def test_planning_happens_at_the_start_of_each_day_not_every_quarter(self, models) -> None:
        p = make(models, StubDiagnoser())
        verdicts(p, inspecting=("PU-201",))
        n_inspect = [
            sum(c.cmd is PlanCmd.INSPECT for c in p.propose(obs_for(quarter_in_day=q)))
            for q in range(4)
        ]
        assert n_inspect == [1, 0, 0, 0]


class TestPumpsAreNeverIsolatedTogether:
    def pumps(self, op_a: Op, op_b: Op):
        va = dataclasses.replace(
            view("PU-201", job=a_job("PU-201", mode="IMPELLER_WEAR"), spares=SPARES), op=op_a
        )
        vb = dataclasses.replace(
            view("PU-202", job=a_job("PU-202", mode="IMPELLER_WEAR"), spares=SPARES), op=op_b
        )
        return {"PU-201": va, "PU-202": vb}

    def test_prove_is_dropped_while_the_other_pump_is_isolated_in_work_or_failed(self, models):
        for other in (Op.ISO, Op.WORK, Op.FAIL):
            p = make(models)
            verdicts(p)
            out = p.propose(obs_for(**self.pumps(Op.LOCKED, other)))
            assert not [c for c in out if c.cmd is Cmd.PROVE and c.asset_id == "PU-201"], other

    def test_prove_goes_ahead_when_the_other_pump_is_running_stopped_or_locked_alone(self, models):
        for other in (Op.RUN, Op.STOP):
            p = make(models)
            verdicts(p)
            out = p.propose(obs_for(**self.pumps(Op.LOCKED, other)))
            assert [c for c in out if c.cmd is Cmd.PROVE and c.asset_id == "PU-201"]

    def test_two_pumps_locked_in_the_same_quarter_are_not_both_proved(self, models) -> None:
        p = make(models)
        verdicts(p)
        out = p.propose(obs_for(**self.pumps(Op.LOCKED, Op.LOCKED)))
        assert len([c for c in out if c.cmd is Cmd.PROVE]) == 1

    def test_the_crusher_and_the_conveyor_are_not_held_back(self, models) -> None:
        p = make(models)
        verdicts(p)
        locked = dataclasses.replace(
            view("CV-101", job=a_job("CV-101"), spares=SPARES), op=Op.LOCKED
        )
        out = p.propose(obs_for(**{"CV-101": locked, **self.pumps(Op.ISO, Op.ISO)}))
        assert [c for c in out if c.cmd is Cmd.PROVE and c.asset_id == "CV-101"]


# --------------------------------------------------------------------------------- bookkeeping
class TestRenewal:
    def test_a_completed_work_order_starts_a_new_life_in_the_belief(self, models) -> None:
        p = make(models)
        verdicts(p)
        p.propose(obs_for(day=30))
        assert p.tracker.life("PU-201") == 0
        done = obs_for(
            day=31, quarter_in_day=10, **{"PU-201": view("PU-201", day=30, spares=SPARES)}
        )
        done = dataclasses.replace(
            done,
            assets=MappingProxyType(
                {**done.assets, "PU-201": dataclasses.replace(done.assets["PU-201"], jobs_done=1)}
            ),
        )
        p.propose(done)
        assert p.tracker.life("PU-201") == 1 and p.tracker.day("PU-201") == 31

    def test_the_old_parts_reading_is_not_fed_to_the_new_belief(self, models) -> None:
        p = make(models)
        verdicts(p)
        calls: list[tuple] = []
        real = p.tracker.update
        p.tracker.update = lambda *a, **k: (calls.append(a), real(*a, **k))[1]  # type: ignore[method-assign]
        p.propose(obs_for(day=30))
        done = obs_for(day=31, quarter_in_day=5)
        done = dataclasses.replace(
            done,
            assets=MappingProxyType(
                {
                    **done.assets,
                    "PU-201": dataclasses.replace(
                        view("PU-201", day=30, spares=SPARES), jobs_done=1
                    ),
                }
            ),
        )
        calls.clear()
        p.propose(
            done
        )  # a new day, but PU-201's reading is the old part's (day 30 <= renewal day 31)
        assert "PU-201" not in [c[0] for c in calls]

    def test_the_old_parts_inspection_is_not_fed_to_the_new_belief(self, models) -> None:
        p = make(models)
        verdicts(p)
        old = dataclasses.replace(
            view("PU-201", day=30, spares=SPARES), inspection=InspectResult(30, 4.0, False)
        )
        p.propose(obs_for(day=30, **{"PU-201": old}))
        calls: list[tuple] = []
        real = p.tracker.update
        p.tracker.update = lambda *a, **k: (calls.append(a), real(*a, **k))[1]  # type: ignore[method-assign]
        renewed = dataclasses.replace(
            view("PU-201", day=30, spares=SPARES),
            jobs_done=1,
            inspection=InspectResult(31, 4.1, False),
        )
        p.propose(obs_for(day=31, quarter_in_day=5, **{"PU-201": renewed}))
        assert not [c for c in calls if c[0] == "PU-201" and len(c) > 3 and c[3] is True]

    def test_the_asset_is_left_alone_on_the_day_it_is_renewed(self, models) -> None:
        p = make(models, StubDiagnoser())
        verdicts(p, wanting=("PU-201",))
        done = obs_for(day=31, quarter_in_day=5)
        p.propose(obs_for(day=30))
        done = dataclasses.replace(
            done,
            assets=MappingProxyType(
                {**done.assets, "PU-201": dataclasses.replace(done.assets["PU-201"], jobs_done=1)}
            ),
        )
        assert [c for c in raise_wos(p.propose(done)) if c.asset_id == "PU-201"] == []


class TestInspectionsAndOrders:
    def test_an_inspection_result_enters_the_belief_as_a_sharper_reading(self, models) -> None:
        p = make(models)
        verdicts(p)
        calls: list[tuple] = []
        real = p.tracker.update
        p.tracker.update = lambda *a, **k: (calls.append((a, k)), real(*a, **k))[1]  # type: ignore[method-assign]
        v = dataclasses.replace(
            view("PU-201", day=30, spares=SPARES), inspection=InspectResult(30, 3.1, False)
        )
        p.propose(obs_for(day=30, **{"PU-201": v}))
        inspected = [c for c in calls if len(c[0]) > 3 and c[0][3] is True]
        assert len(inspected) == 1 and inspected[0][0][:3] == ("PU-201", 30, 3.1)
        calls.clear()
        p.propose(obs_for(day=30, quarter_in_day=1, **{"PU-201": v}))  # the same result again
        assert not [c for c in calls if len(c[0]) > 3 and c[0][3] is True]

    def test_a_part_out_of_stock_is_ordered_once_and_not_again_for_a_month(self, models) -> None:
        p = make(models)
        verdicts(p)
        empty = {"SYN-BRG-1001": 0}
        views = {"CV-101": view("CV-101", spares=empty, day=30, hi=2.0)}
        out = p.propose(obs_for(day=30, **views))
        assert [c.arg for c in out if c.cmd is PlanCmd.ORDER] == ["SYN-BRG-1001"]
        assert not [c for c in p.propose(obs_for(day=45, **views)) if c.cmd is PlanCmd.ORDER]
        assert [c for c in p.propose(obs_for(day=61, **views)) if c.cmd is PlanCmd.ORDER]

    def test_a_recently_inspected_asset_is_not_inspected_again_at_once(self, models) -> None:
        p = make(models)
        verdicts(p, inspecting=("PU-201",))
        first = [c for c in p.propose(obs_for(day=30)) if c.cmd is PlanCmd.INSPECT]
        v = dataclasses.replace(
            view("PU-201", day=31, spares=SPARES), inspection=InspectResult(31, 3.0, False)
        )
        second = [
            c for c in p.propose(obs_for(day=31, **{"PU-201": v})) if c.cmd is PlanCmd.INSPECT
        ]
        assert len(first) == 1 and second == []


class TestDiagnosisIsInTheLoop:
    def test_a_sensor_fault_raises_no_work_order(self, models) -> None:
        p = make(models, StubDiagnoser("sensor_fault", "BEARING_DEFECT"))
        verdicts(p, wanting=("CV-101",))
        assert raise_wos(p.propose(obs_for())) == []

    def test_a_process_upset_raises_no_work_order(self, models) -> None:
        p = make(models, StubDiagnoser("process_upset", "BEARING_DEFECT"))
        verdicts(p, wanting=("CV-101",))
        assert raise_wos(p.propose(obs_for())) == []

    def test_the_work_order_carries_the_diagnosed_mode_not_the_belief_mode(self, models) -> None:
        p = make(models, StubDiagnoser("degradation", "BEARING_DEFECT"))
        verdicts(p, wanting=("PU-201",))
        (wo,) = raise_wos(p.propose(obs_for()))
        assert wo.arg is not None and wo.arg.startswith("BEARING_DEFECT:")

    def test_an_asset_that_has_failed_is_repaired_whatever_the_diagnosis(self, models) -> None:
        p = make(models, StubDiagnoser("sensor_fault", "BEARING_DEFECT"))
        verdicts(p)
        failed = dataclasses.replace(view("CV-101", spares=SPARES, day=30), op=Op.FAIL)
        (wo,) = raise_wos(p.propose(obs_for(**{"CV-101": failed})))
        assert wo.asset_id == "CV-101"

    def test_without_a_diagnoser_the_most_probable_mode_under_the_belief_is_used(self, models):
        p = make(models, None)
        verdicts(p, wanting=("PU-201",))
        (wo,) = raise_wos(p.propose(obs_for()))
        assert wo.arg is not None and wo.arg.split(":")[0] in ("IMPELLER_WEAR", "BEARING_DEFECT")

    def test_a_diagnoser_named_by_string_is_built_but_not_fitted(self, models) -> None:
        p = PlannerProposer(models, n_particles=32, diagnoser="corr", seed=1)
        assert p.diagnoser is not None and p.diagnoser.name == "corr"
        verdicts(p, wanting=("CV-101",))
        with pytest.raises((RuntimeError, ValueError)):
            p.propose(obs_for())  # asking an unfitted diagnoser is an error, not a guess

    def test_fitting_without_a_diagnoser_is_an_error(self, models) -> None:
        with pytest.raises(ValueError):
            make(models, None).fit_diagnoser([])

    def test_the_diagnosis_is_asked_once_per_asset_per_day(self, models) -> None:
        stub = StubDiagnoser()
        p = make(models, stub)
        verdicts(p, wanting=("CV-101",))
        p.propose(obs_for(quarter_in_day=0))
        p.propose(obs_for(quarter_in_day=1))
        assert stub.calls == 1


class TestAStalledAssetIsRepairedLikeAFailedOne:
    """0003 A18 R5: an asset the shield left stopped, with no work order, for a day and two
    quarters is given a job whatever the planner's verdict and the diagnosis say."""

    def stopped(self, day: int, quarter_in_day: int) -> object:
        v = dataclasses.replace(view("CV-101", spares=SPARES, day=day), op=Op.STOP)
        return obs_for(day=day, quarter_in_day=quarter_in_day, **{"CV-101": v})

    def test_a_job_once_the_stop_has_stood_long_enough_and_not_before(self, models) -> None:
        from closed_loop.proposers.stall import STALL_Q

        p = make(models, StubDiagnoser("degradation", "BEARING_DEFECT"))
        verdicts(p)  # the planner wants nothing
        t0 = 30 * DAY_Q
        for q in (0, 1, 50, DAY_Q - 1):
            assert raise_wos(p.propose(self.stopped(30, q))) == [], q
        before = t0 + STALL_Q - 1
        assert raise_wos(p.propose(self.stopped(before // DAY_Q, before % DAY_Q))) == []
        at = t0 + STALL_Q
        (wo,) = raise_wos(p.propose(self.stopped(at // DAY_Q, at % DAY_Q)))
        assert wo.asset_id == "CV-101" and wo.source == "planner"

    def test_the_quarter_the_stop_becomes_a_stall_replans_though_the_views_are_the_same_object(
        self, models
    ) -> None:
        """On the plant a quiet quarter hands the proposer the same mapping of views; the stalled
        set changing is what must trigger the plan then, not a new day or a new mapping."""
        from closed_loop.proposers.stall import STALL_Q

        p = make(models, StubDiagnoser("degradation", "BEARING_DEFECT"))
        verdicts(p)
        t0 = 30 * DAY_Q
        first = self.stopped(30, 0)
        assert raise_wos(p.propose(first)) == []

        def later(q: int) -> object:
            return dataclasses.replace(first, quarter=q, day=q // DAY_Q, shift=(q % DAY_Q) // 32)

        assert raise_wos(p.propose(later(t0 + STALL_Q - 1))) == []
        (wo,) = raise_wos(p.propose(later(t0 + STALL_Q)))
        assert wo.asset_id == "CV-101"

    def test_the_diagnosed_mode_is_used_whatever_the_cause(self, models) -> None:
        from closed_loop.proposers.stall import STALL_Q

        p = make(models, StubDiagnoser("sensor_fault", "BEARING_DEFECT"))
        verdicts(p)
        t0 = 30 * DAY_Q
        p.propose(self.stopped(30, 0))
        at = t0 + STALL_Q
        (wo,) = raise_wos(p.propose(self.stopped(at // DAY_Q, at % DAY_Q)))
        assert wo.arg is not None and wo.arg.startswith("BEARING_DEFECT:")

    def test_a_stopped_asset_with_a_job_open_is_not_a_stalled_one(self, models) -> None:
        from closed_loop.proposers.stall import STALL_Q

        p = make(models, StubDiagnoser())
        verdicts(p)
        busy = dataclasses.replace(
            view("CV-101", spares=SPARES, job=a_job("CV-101", ready=False)), op=Op.STOP
        )
        t0 = 30 * DAY_Q
        for t in (t0, t0 + STALL_Q + 3):
            out = p.propose(obs_for(day=t // DAY_Q, quarter_in_day=t % DAY_Q, **{"CV-101": busy}))
            assert raise_wos(out) == []

    def test_a_running_asset_is_never_stalled(self, models) -> None:
        from closed_loop.proposers.stall import STALL_Q

        p = make(models, StubDiagnoser())
        verdicts(p)
        t0 = 30 * DAY_Q
        for t in (t0, t0 + STALL_Q + 3, t0 + 3 * DAY_Q):
            assert raise_wos(p.propose(obs_for(day=t // DAY_Q, quarter_in_day=t % DAY_Q))) == []


class TestReset:
    def test_reset_clears_the_beliefs_and_the_bookkeeping(self, models) -> None:
        p = make(models, StubDiagnoser())
        verdicts(p, wanting=("CV-101",))
        p.propose(obs_for())
        assert p.decisions
        p.reset(5)
        assert p.decisions == [] and p.tracker.seed == 5 and p.planner.seed == 5
        verdicts(p, wanting=("CV-101",))
        assert len(raise_wos(p.propose(obs_for()))) == 1  # not held back by the old cooldown


# ------------------------------------------------------------------------------- a real episode
def run_episode(proposer: PlannerProposer, seed: int, days: int = 60):
    plant = make_plant(PlantParams())
    obs = plant.reset(seed, Scenario())
    proposer.reset(seed)
    issued: list[tuple[int, Command]] = []
    violations = []
    for _ in range(days * DAY_Q):
        cmds = proposer.propose(obs)
        issued.extend((obs.quarter, c) for c in cmds)
        res = plant.step(cmds)
        violations.extend(res.violations)
        obs = res.obs
        if res.done:
            break
    return plant, issued, violations


@pytest.fixture(scope="module")
def episode(models):
    p = PlannerProposer(models, diagnoser=None)
    return run_episode(p, 1503, days=60) + (p,)


class TestOnThePlant:
    def test_every_command_is_a_planning_command_or_in_the_runner_alphabet(self, episode) -> None:
        _, issued, _, _ = episode
        ok = {PlanCmd.RAISE_WO, PlanCmd.INSPECT, PlanCmd.ORDER} | set(ALPHABET)
        assert issued and {c.cmd for _, c in issued} <= ok

    def test_it_raises_a_work_order_only_for_a_job_it_means_to_do(self, episode) -> None:
        plant, issued, _, _ = episode
        stats = plant.stats()
        wos = [c for _, c in issued if c.cmd is PlanCmd.RAISE_WO]
        assert len(wos) >= stats.planned_jobs  # retries mean at least as many, never fewer

    def test_the_same_seed_gives_the_same_commands(self, models, episode) -> None:
        _, first, _, _ = episode
        second = run_episode(PlannerProposer(models, diagnoser=None), 1503, days=60)[1]
        assert first == second

    def test_a_different_seed_gives_a_different_episode(self, models, episode) -> None:
        _, first, _, _ = episode
        other = run_episode(PlannerProposer(models, diagnoser=None), 1504, days=60)[1]
        assert first != other

    def test_the_plan_is_visible_in_the_decision_log(self, episode) -> None:
        *_, p = episode
        assert p.decisions
        days = [q // DAY_Q for q, _ in p.decisions]
        assert days == sorted(days) and len(set(days)) > 30

    def test_there_is_no_pump_pair_violation(self, episode) -> None:
        _, _, violations, _ = episode
        assert not [v for v in violations if v[0] == "P4"]

    def test_the_reading_in_a_daily_observation_is_a_daily_obs(self, episode) -> None:
        # the proposer reads last_obs; check the type it relies on has the fields it uses
        assert {"day", "hi", "load"} <= {f.name for f in dataclasses.fields(DailyObs)}


class TestQuietQuarters:
    """When the plant hands back the very same assets mapping, nothing has changed."""

    def test_the_same_mapping_again_costs_nothing_and_returns_no_commands(self, models) -> None:
        p = make(models)
        verdicts(p)
        obs = obs_for()
        p.propose(obs)

        def boom(*_a, **_k):
            raise AssertionError("a quiet quarter was processed")

        p._look = boom  # type: ignore[method-assign]
        again = dataclasses.replace(obs, quarter=obs.quarter + 1)
        assert again.assets is obs.assets and p.propose(again) == []

    def test_it_is_not_taken_while_a_work_order_is_open(self, models) -> None:
        p = make(models)
        verdicts(p)
        views = {"CV-101": view("CV-101", job=a_job("CV-101"), spares=SPARES)}
        obs = obs_for(**views)
        first = p.propose(obs)
        second = p.propose(dataclasses.replace(obs, quarter=obs.quarter + 1))
        assert first and second  # the runner is asked every quarter

    def test_a_down_asset_is_planned_again_when_the_plants_picture_changes(self, models) -> None:
        p = make(models, StubDiagnoser())
        verdicts(p)
        failed = dataclasses.replace(view("CV-101", spares=SPARES, day=30), op=Op.FAIL)
        obs = obs_for(**{"CV-101": failed})
        assert raise_wos(p.propose(obs))
        p._last_raise.clear()  # a refused order, a long time ago: ready to be raised again
        same = dataclasses.replace(obs, quarter=obs.quarter + 20)
        assert raise_wos(p.propose(same)) == []  # the plant's picture has not changed: no new plan
        changed = dataclasses.replace(same, assets=MappingProxyType(dict(obs.assets)))
        assert raise_wos(p.propose(changed))

    def test_it_is_not_taken_when_the_picture_changes_within_the_day(self, models) -> None:
        p = make(models)
        verdicts(p)
        p.propose(obs_for(day=30))
        done = obs_for(day=30, quarter_in_day=10)
        done = dataclasses.replace(
            done,
            assets=MappingProxyType(
                {**done.assets, "PU-201": dataclasses.replace(done.assets["PU-201"], jobs_done=1)}
            ),
        )
        p.propose(done)  # same day, a new mapping: the completed work order must be seen
        assert p.tracker.life("PU-201") == 1

    def test_it_is_not_taken_on_a_new_day(self, models) -> None:
        p = make(models, StubDiagnoser())
        verdicts(p, inspecting=("PU-201",))
        obs = obs_for(day=30)
        p.propose(obs)
        nxt = dataclasses.replace(obs, day=31, quarter=31 * DAY_Q)
        assert any(c.cmd is PlanCmd.INSPECT for c in p.propose(nxt))

    def test_switching_the_fast_path_off_changes_no_command_on_a_real_episode(self, models) -> None:
        def run(fast: bool):
            p = PlannerProposer(models, diagnoser=None)
            p.fast_path = fast
            return run_episode(p, 1505, days=50)[1]

        assert run(True) == run(False)
