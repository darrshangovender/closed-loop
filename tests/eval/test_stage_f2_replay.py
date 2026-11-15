"""The F2 stage replays the search stage's finds and nothing else, and survives what goes wrong.

Everything here is run with the siblings and the plant replaced by stand-ins: what is tested is the
plumbing of ``eval.stage_f2`` (which finds are replayed, what stays in the denominator, what is
cached and what is cut off by a time limit), not an F2 episode (``tests/f2`` does that).
"""

from __future__ import annotations

import time
import types
from pathlib import Path
from typing import Any

import pytest

from closed_loop.audit.guard import GuardError
from closed_loop.core.results import ResultRecord
from closed_loop.core.types import Fidelity
from closed_loop.eval import stage_f2
from closed_loop.eval.arms import ARMS
from closed_loop.eval.claims import claim_name
from closed_loop.eval.counts import Counts
from closed_loop.eval.runner import Runner
from closed_loop.eval.stage_f2 import F2_ARMS, F2Slim
from closed_loop.eval.stages import SearchFind, StageContext, run_pipeline
from closed_loop.f2.replay import ReplayOutcome
from tests.eval.test_stages import (
    FakeEval,
    FakeResult,
    FakeRun,
    context_for,
    install_fake_search,
    invented,
    record_named,
)

ARM_NAMES = tuple(F2_ARMS)


def fake_log(**kw: Any) -> Any:
    base: dict[str, Any] = {
        "seconds": {"total": 1.5},
        "run_asset_calls": 4,
        "expected_run_asset_calls": 4,
        "repair_drafts": 1,
        "inspection_drafts": 0,
        "runs_without_a_draft": 0,
        "wrong_compliant_signed": 0,
        "wrong_compliant_presented": 0,
        "dropped_pieces": 0,
    }
    return types.SimpleNamespace(**{**base, **kw})


def outcome(find: SearchFind, confirmed: bool, reason: str = "plant_logic") -> ReplayOutcome:
    return ReplayOutcome(
        arm="R-target", seed=find.seed, regime=find.regime, confirmed=confirmed,
        reason="confirmed" if confirmed else reason, f1_props=find.props, f2_props=(),
        common=find.props if confirmed else (), f1_labels=(), f1_planned_jobs=0, f2_repairs=0,
        f2_inspections=0, f2_drafts=0, f2_runs_rejected=0, f2_runs_escalated=0, evidence="e",
        f1_wall_s=0.0, f2_wall_s=0.0,
    )  # fmt: skip


class World:
    """The stand-ins, and a record of what the stage asked of them."""

    def __init__(self, mp: pytest.MonkeyPatch) -> None:
        self.episodes: list[tuple[str, int]] = []
        self.replays: list[tuple[float, str, int]] = []
        self.episode_behaviour: Any = None
        self.replay_behaviour: Any = None
        mp.setattr(stage_f2, "require_siblings", lambda: None)
        mp.setattr(stage_f2, "simulate_f2_episode", self.simulate)
        mp.setattr(stage_f2, "replay_theta", self.replay)
        mp.setattr(stage_f2, "_target_arm", lambda: ARMS["H-G1"])
        mp.setattr(stage_f2, "dev_lives", lambda n: [])
        mp.setattr(stage_f2, "estimator_mismatch", lambda lives: [])
        mp.setattr(
            stage_f2,
            "mismatch_record",
            lambda rows: ResultRecord(
                name="f2.estimator_mismatch.mae_diff_days", value=0.0, n=1,
                fidelity=Fidelity.F2, measured_on="a stand-in", script="test",
            ),
        )  # fmt: skip

    def simulate(self, arm: Any, seed: int, scenario: Any, **kw: Any) -> Any:
        self.episodes.append((arm.name, seed))
        if self.episode_behaviour is not None:
            self.episode_behaviour(arm.name, seed)
        return types.SimpleNamespace(result=invented(arm.name, seed, "nominal", {}), log=fake_log())

    def replay(self, arm: Any, theta: Any, regime: str, seed: int, **kw: Any) -> ReplayOutcome:
        self.replays.append((float(theta[0]), regime, seed))
        if self.replay_behaviour is not None:
            self.replay_behaviour(seed)
        find = SearchFind(seed, tuple(float(x) for x in theta), regime, "?", ("P2",))
        return outcome(find, confirmed=True)


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> World:
    return World(monkeypatch)


def f2_context(
    tmp_path: Path | None = None, *, episodes: int = 3, replays: int = 25, timeout: float = 900.0
) -> StageContext:
    runner = Runner(
        workers=1, headroom=lambda: 40.0, timeout_s=None,
        cache_dir=None if tmp_path is None else tmp_path / "cache",
    )  # fmt: skip
    counts = Counts(f2_episodes=episodes, f2_replays=replays, episode_timeout_s=timeout)
    return context_for(runner, counts)


def find(i: int, regime: str = "out", algo: str = "cma") -> SearchFind:
    return SearchFind(3000 + i, (0.01 * i, 0.5), regime, algo, ("P2",))


# ----------------------------------------------------------------------------- the replay set


class TestTheReplaySetIsTheSearchStagesFinds:
    def test_the_stage_replays_exactly_the_finds_the_search_left_in_the_order_it_left_them(
        self, world: World
    ) -> None:
        sc = f2_context()
        finds = [find(0, "in", "cma"), find(0, "out", "random"), find(1, "out", "cma")]
        sc.shared["search_violations"] = finds
        out = stage_f2.run(sc)
        assert world.replays == [(f.theta[0], f.regime, f.seed) for f in finds]
        confirmed = record_named(out.records, claim_name("C-TOGETHER", "confirmed"))
        assert (confirmed.n, confirmed.extra["k"]) == (3, 3)

    def test_the_search_stage_and_the_f2_stage_agree_end_to_end(
        self, world: World, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # the first violating evaluation of each search run, in run order across the four cells
        def behave(regime: str, algo: str, budget: int, seed: int) -> FakeResult:
            if seed == 3002:  # this seed finds nothing anywhere
                return FakeResult(FakeRun(None, budget), (), True, regime)
            first = FakeEval({"A_exec": 1}, objective=-0.1, theta=(seed / 1e4, 0.1 * len(algo)))
            later = FakeEval({"A_exec": 1}, objective=-0.9, theta=(0.99, 0.99))  # more severe
            return FakeResult(FakeRun(4, budget), (first, later), True, regime)

        install_fake_search(monkeypatch, behave)
        counts = Counts(search_runs=4, search_budget=200, f2_episodes=1, f2_replays=25)
        sc = context_for(Runner(workers=1, headroom=lambda: 40.0, timeout_s=None), counts)
        run_pipeline(sc, ("search",))
        stage_f2.run(sc)
        replayed = world.replays
        # 3 seeds x 4 cells = 12 finds, severity ignored: each is the run's FIRST violation
        assert len(replayed) == 12 and all(theta != 0.99 for theta, _r, _s in replayed)
        want = [(s / 1e4, r, s) for s in (3000, 3001, 3003) for r in ("in", "in", "out", "out")]
        assert replayed == want  # run order: by seed, then R_in/R_out and CMA-ES/random

    def test_the_set_is_capped_at_the_number_of_replays_asked_for(self, world: World) -> None:
        sc = f2_context(replays=2)
        sc.shared["search_violations"] = [find(i) for i in range(6)]
        stage_f2.run(sc)
        assert [s for _t, _r, s in world.replays] == [3000, 3001]

    def test_the_stage_has_no_search_of_its_own(self) -> None:
        text = Path(stage_f2.__file__).read_text(encoding="utf-8")
        assert "falsify_detailed" not in text and "SEARCH_SEED" not in text
        assert not hasattr(stage_f2, "_candidates")


class TestNoFinds:
    def test_a_search_that_found_nothing_is_zero_of_zero_and_replays_nothing(
        self, world: World, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def never(*a: Any, **k: Any) -> Any:
            raise AssertionError("a search was started although the search stage found nothing")

        monkeypatch.setattr(stage_f2, "_target_arm", never)
        sc = f2_context()
        sc.shared["search_violations"] = []
        out = stage_f2.run(sc)
        confirmed = record_named(out.records, claim_name("C-TOGETHER", "confirmed"))
        assert (confirmed.n, confirmed.extra["k"]) == (0, 0) and world.replays == []
        assert any("found no violation to replay" in n for n in out.notes)

    def test_fewer_finds_than_the_cap_replays_what_exists_and_says_the_count(
        self, world: World
    ) -> None:
        sc = f2_context(replays=25)
        sc.shared["search_violations"] = [find(0), find(1)]
        out = stage_f2.run(sc)
        confirmed = record_named(out.records, claim_name("C-TOGETHER", "confirmed"))
        assert confirmed.n == 2 and confirmed.extra["requested"] == 25
        assert any("2 replayed (25 at most)" in n for n in out.notes)

    def test_a_quick_run_that_did_not_search_says_so_and_does_not_search(
        self, world: World
    ) -> None:
        out = stage_f2.run(f2_context())  # nothing in sc.shared
        confirmed = record_named(out.records, claim_name("C-TOGETHER", "confirmed"))
        assert confirmed.n == 0 and world.replays == []
        assert any("did not run in this invocation" in n for n in out.notes)

    def test_a_full_run_without_the_hand_off_is_an_error_not_a_fallback(self, world: World) -> None:
        sc = f2_context()
        sc = StageContext(**{**sc.__dict__, "mode": "full"})
        with pytest.raises(RuntimeError, match="needs the finds of the search stage"):
            stage_f2.run(sc)
        assert world.replays == []

    def test_an_empty_list_is_an_answer_and_none_is_not(self, world: World) -> None:
        sc = f2_context()
        sc = StageContext(**{**sc.__dict__, "mode": "full"})
        sc.shared["search_violations"] = []  # the search ran and found nothing: fine in a full run
        out = stage_f2.run(sc)
        assert record_named(out.records, claim_name("C-TOGETHER", "confirmed")).n == 0


# ----------------------------------------------------------------------------- what goes wrong


class TestAReplayThatCannotRunStaysInTheDenominator:
    def test_an_error_is_an_unconfirmed_replay_with_a_reason_code(self, world: World) -> None:
        def behave(seed_of_the_find: int) -> None:
            if seed_of_the_find == 3001:
                raise RuntimeError("the siblings fell over")

        world.replay_behaviour = behave
        sc = f2_context()
        sc.shared["search_violations"] = [find(0), find(1), find(2)]
        out = stage_f2.run(sc)  # no dead run
        confirmed = record_named(out.records, claim_name("C-TOGETHER", "confirmed"))
        assert (confirmed.n, confirmed.extra["k"]) == (3, 2)
        assert record_named(out.records, "f2.replay.reason.replay_error").extra["k"] == 1
        assert record_named(out.records, "f2.replay.reason.replay_timeout").extra["k"] == 0

    def test_a_replay_over_its_time_limit_is_cut_off_and_counted(self, world: World) -> None:
        def behave(seed_of_the_find: int) -> None:
            if seed_of_the_find == 3000:
                while True:  # a replay that never ends; Python-level, so the limit can land
                    time.sleep(0.005)

        world.replay_behaviour = behave
        sc = f2_context(timeout=0.1)  # a replay is allowed three times that
        sc.shared["search_violations"] = [find(0), find(1)]
        started = time.perf_counter()
        out = stage_f2.run(sc)
        assert time.perf_counter() - started < 30.0
        confirmed = record_named(out.records, claim_name("C-TOGETHER", "confirmed"))
        assert (confirmed.n, confirmed.extra["k"]) == (2, 1)
        assert record_named(out.records, "f2.replay.reason.replay_timeout").extra["k"] == 1

    def test_every_reason_code_has_a_record_even_at_zero(self, world: World) -> None:
        sc = f2_context()
        sc.shared["search_violations"] = [find(0)]
        out = stage_f2.run(sc)
        names = {r.name for r in out.records}
        for code in (*stage_f2.REASON_CODES, *stage_f2.FAILURE_CODES):
            assert f"f2.replay.reason.{code}" in names


class TestEpisodes:
    def test_an_episode_that_raises_is_a_counted_failure_not_a_dead_run(self, world: World) -> None:
        def behave(arm: str, seed: int) -> None:
            if arm == ARM_NAMES[0] and seed == 1101:
                raise KeyError("boom")

        world.episode_behaviour = behave
        out = stage_f2.run(f2_context(episodes=3))
        arm_rec = record_named(out.records, claim_name("C-TOGETHER", "arm", ARM_NAMES[0]))
        assert (arm_rec.extra["completed"], arm_rec.extra["tried"]) == (2, 3)
        assert "1101" in str(arm_rec.extra["cause"]) and "KeyError" in str(arm_rec.extra["cause"])
        other = record_named(out.records, claim_name("C-TOGETHER", "arm", ARM_NAMES[1]))
        assert other.extra["completed"] == 3
        assert out.status == "partial"

    def test_an_episode_over_its_time_limit_is_cut_off(self, world: World) -> None:
        def behave(arm: str, seed: int) -> None:
            if arm == ARM_NAMES[1] and seed == 1100:
                while True:
                    time.sleep(0.005)

        world.episode_behaviour = behave
        out = stage_f2.run(f2_context(episodes=2, timeout=0.2))
        rec = record_named(out.records, claim_name("C-TOGETHER", "arm", ARM_NAMES[1]))
        assert rec.extra["completed"] == 1 and "timeout" in str(rec.extra["cause"])

    def test_finished_episodes_are_cached_so_a_second_run_does_not_redo_them(
        self, world: World, tmp_path: Path
    ) -> None:
        stage_f2.run(f2_context(tmp_path, episodes=3))
        assert len(world.episodes) == 3 * len(ARM_NAMES)
        world.episodes.clear()
        out = stage_f2.run(f2_context(tmp_path, episodes=3))  # a new runner, the same cache
        assert world.episodes == []
        assert (
            record_named(out.records, claim_name("C-TOGETHER", "arm", ARM_NAMES[0])).extra[
                "completed"
            ]
            == 3
        )

    def test_a_pause_between_batches_resumes_from_the_cache(
        self, world: World, tmp_path: Path
    ) -> None:
        charges = iter([40.0, 40.0, 99.0])  # the gate trips before the second batch of an arm
        sc = f2_context(tmp_path, episodes=stage_f2.F2_BATCH + 2)
        sc.runner._headroom = lambda: next(charges, 40.0)
        with pytest.raises(GuardError):
            stage_f2.run(sc)
        done = len(world.episodes)
        assert 0 < done <= stage_f2.F2_BATCH * len(ARM_NAMES)
        world.episodes.clear()
        stage_f2.run(f2_context(tmp_path, episodes=stage_f2.F2_BATCH + 2))
        assert len(world.episodes) == (stage_f2.F2_BATCH + 2) * len(ARM_NAMES) - done

    def test_the_cache_key_names_the_arm_and_the_sibling_lock(self) -> None:
        a = stage_f2.F2EpisodeTask(ARM_NAMES[0], "aa").fingerprint()
        assert a != stage_f2.F2EpisodeTask(ARM_NAMES[1], "aa").fingerprint()
        assert a != stage_f2.F2EpisodeTask(ARM_NAMES[0], "bb").fingerprint()

    def test_the_cached_episode_keeps_what_the_records_read_and_drops_the_trace(self) -> None:
        import pickle

        slim = F2Slim(invented(ARM_NAMES[0], 1100, "nominal", {}), fake_log())
        assert pickle.loads(pickle.dumps(slim)).result.arm == ARM_NAMES[0]


def test_a_replay_is_cached_by_the_finds_it_replays_and_their_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stage_f2, "_target_arm", lambda: ARMS["H-G1"])
    a = stage_f2.ReplayTask((find(0), find(1)), None).fingerprint()
    assert a == stage_f2.ReplayTask((find(0), find(1)), None).fingerprint()
    assert a != stage_f2.ReplayTask((find(0), find(2)), None).fingerprint()
    assert a != stage_f2.ReplayTask((find(1), find(0)), None).fingerprint()  # the index is the key
    assert a != stage_f2.ReplayTask((find(0), find(1)), "lock").fingerprint()


def test_the_clock_goes_to_the_run_log_and_not_to_the_records(world: World) -> None:
    sc = f2_context(episodes=2)
    lines: list[str] = []
    sc.log = lines.append  # type: ignore[method-assign]
    out = stage_f2.run(sc)
    assert not any("seconds" in r.name for r in out.records)
    assert not any(" s" in n.split(":")[-1] and "estimator" in n for n in out.notes)
    assert any("seconds per episode" in line for line in lines)
    assert any("the stage took" in line for line in lines)
