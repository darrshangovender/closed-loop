"""``eval.runner``: sequential stopping, workers, failures, the cache and the commit-charge gate.

Most tests replace ``EpisodeTask.__call__`` with a cheap function that builds an ``EpisodeResult``
from the seed, so that the logic of the runner (batches, stopping, the cache, failures) is tested
without simulating a plant. The tests of the worker pool and of the slimming use real episodes of
the cheapest arm on the small context.
"""

from __future__ import annotations

import dataclasses
import os
import pickle
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from closed_loop.core.types import Fidelity
from closed_loop.eval import runner as runner_module
from closed_loop.eval.arms import ARMS
from closed_loop.eval.context import PreparedContext
from closed_loop.eval.counts import StopRule, stop_rule_for
from closed_loop.eval.episode import EpisodeResult, run_episode, scenario_for
from closed_loop.eval.metrics import aggregate, learner_counts
from closed_loop.eval.runner import (
    MAX_WORKERS,
    CellCache,
    CommitChargeStop,
    Done,
    EpisodeTask,
    EpisodeTimeout,
    Failed,
    Runner,
    effective_workers,
    fingerprint,
    run_cell,
    simulation_code_sha256,
    slim_result,
    worst_episode,
)
from closed_loop.eval.tuned import PROVISIONAL, use_tuned
from closed_loop.stats import betting_cs

PROPS = ("P1", "P2", "P3", "P4", "P5", "P6")
SEEDS = list(range(1100, 1100 + 200))


def fake_result(task: EpisodeTask, seed: int, **kw: Any) -> EpisodeResult:
    base: dict[str, Any] = {
        "arm": task.arm.name,
        "seed": seed,
        "condition": task.scenario.condition,
        "fidelity": task.fidelity,
        "violations": dict.fromkeys(PROPS, 0),
        "min_robustness": dict.fromkeys(PROPS, 1.0),
        "throughput": 1.0,
        "cost": 20.0 + (seed % 7),  # M-COST between 0.10 and 0.13
        "unplanned_failures": 0,
        "planned_jobs": 2,
        "shield_overrides": 0,
        "shield_admitted": 0,
        "monitor_fired_day": None,
        "labels": (),
        "wall_s": float(seed % 5),  # a stopwatch: allowed to differ, and does
    }
    return EpisodeResult(**{**base, **kw})


CALLS: list[int] = []


@pytest.fixture
def cheap(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """Replace the episode by a function of the seed; ``cheap(fn)`` installs ``fn(task, seed)``."""
    CALLS.clear()

    def install(fn: Callable[[EpisodeTask, int], EpisodeResult] | None = None) -> None:
        def call(self: EpisodeTask, seed: int) -> EpisodeResult:
            CALLS.append(seed)
            return fn(self, seed) if fn is not None else fake_result(self, seed)

        monkeypatch.setattr(EpisodeTask, "__call__", call)

    return install


def runner_for(tmp_path: Path | None, **kw: Any) -> Runner:
    kw.setdefault("workers", 1)
    kw.setdefault("headroom", lambda: 40.0)
    return Runner(cache_dir=tmp_path, **kw)


def same(a: list[EpisodeResult], b: list[EpisodeResult]) -> bool:
    return [fingerprint(x) for x in a] == [fingerprint(x) for x in b]


# ----------------------------------------------------------------------------- workers


class TestWorkers:
    def test_the_cap_is_three(self) -> None:
        assert MAX_WORKERS == 3
        assert effective_workers(3) == 3
        assert effective_workers(8) == 3
        assert effective_workers(1000) == 3

    def test_at_least_one(self) -> None:
        assert effective_workers(0) == 1
        assert effective_workers(-4) == 1

    def test_a_runner_applies_the_cap(self) -> None:
        assert Runner(workers=12).workers == 3
        assert Runner(workers=2).workers == 2

    def test_run_cell_without_a_runner_makes_a_capped_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[int] = []

        def fake(self: Runner, *args: Any, **kw: Any) -> Any:
            seen.append(self.workers)
            return SimpleNamespace(results=[])

        monkeypatch.setattr(Runner, "run_cell", fake)
        run_cell(ARMS["B0"], [1], scenario_for("nominal"), workers=99)
        run_cell(ARMS["B0"], [1], scenario_for("nominal"), workers=2)
        assert seen == [3, 2]


# ----------------------------------------------------------------------------- stopping


def halfwidth_after(values: list[float], n: int, alpha: float) -> float:
    lo, hi = betting_cs(np.array(values[:n]), alpha)
    return float(hi[-1] - lo[-1]) / 2.0


class TestSequentialStopping:
    def test_it_stops_at_the_first_batch_that_meets_the_target_and_not_before(
        self, cheap: Callable[..., None]
    ) -> None:
        cheap()
        arm = ARMS["B0"]  # primary metric M-COST
        costs = [(20.0 + (s % 7)) / 200.0 for s in SEEDS]
        alpha, batch, target = 0.05 / 9, 10, 0.1
        expected = next(
            (
                n
                for n in range(batch, len(SEEDS) + 1, batch)
                if halfwidth_after(costs, n, alpha) <= target
            ),
            None,
        )
        assert expected is not None and expected < len(SEEDS)  # the test would be vacuous otherwise
        assert halfwidth_after(costs, expected - batch, alpha) > target  # not before
        run = runner_for(None).run_cell(
            arm,
            SEEDS,
            scenario_for("nominal"),
            batch=batch,
            n_max=len(SEEDS),
            target_halfwidth=target,
        )
        assert run.n == expected and run.stopped_by == "target"
        assert run.halfwidth is not None and run.halfwidth <= target
        assert CALLS == SEEDS[:expected]  # nothing run past the stopping batch

    def test_a_tighter_target_needs_more_episodes(self, cheap: Callable[..., None]) -> None:
        cheap()
        loose = runner_for(None).run_cell(
            ARMS["B0"], SEEDS, scenario_for("nominal"), batch=10, n_max=200, target_halfwidth=0.15
        )
        tight = runner_for(None).run_cell(
            ARMS["B0"], SEEDS, scenario_for("nominal"), batch=10, n_max=200, target_halfwidth=0.08
        )
        assert loose.n < tight.n

    def test_it_never_goes_past_n_max(self, cheap: Callable[..., None]) -> None:
        cheap()
        run = runner_for(None).run_cell(
            ARMS["B0"], SEEDS, scenario_for("nominal"), batch=4, n_max=10, target_halfwidth=1e-9
        )
        assert run.n == 10 and run.stopped_by == "n_max"
        assert CALLS == SEEDS[:10]

    def test_the_last_partial_batch_is_run_and_no_seed_is_dropped(
        self, cheap: Callable[..., None]
    ) -> None:
        cheap()
        run = runner_for(None).run_cell(
            ARMS["B0"], SEEDS, scenario_for("nominal"), batch=4, n_max=10, target_halfwidth=1e-9
        )
        assert [r.seed for r in run.results] == SEEDS[:10]

    def test_a_cell_with_no_event_runs_to_n_max(self, cheap: Callable[..., None]) -> None:
        cheap()  # H-C0 is judged on M-VIOL (P1-P3): no event, and a narrow sequence soon
        run = runner_for(None).run_cell(
            ARMS["H-C0"], SEEDS, scenario_for("nominal"), batch=5, n_max=40, target_halfwidth=0.2
        )
        assert run.n == 40 and run.stopped_by == "n_max"

    def test_a_cell_with_events_stops_when_the_target_is_met(
        self, cheap: Callable[..., None]
    ) -> None:
        def half_violate(task: EpisodeTask, seed: int) -> EpisodeResult:
            bad = seed % 2 == 0
            mins = dict.fromkeys(PROPS, 1.0)
            if bad:
                mins["P2"] = -1.0
            return fake_result(task, seed, min_robustness=mins)

        cheap(half_violate)
        run = runner_for(None).run_cell(
            ARMS["H-C0"], SEEDS, scenario_for("nominal"), batch=5, n_max=40, target_halfwidth=0.45
        )
        assert run.stopped_by == "target" and run.n < 40
        rec = aggregate(run.results, "m_viol_p13", 0.05 / 9)
        assert rec.extra["k"] > 0

    def test_the_rule_can_be_given(self, cheap: Callable[..., None]) -> None:
        cheap()
        rule = StopRule("throughput", 0.5)  # throughput is constant 1: met at the first batch
        run = runner_for(None).run_cell(
            ARMS["B0"], SEEDS, scenario_for("nominal"), batch=5, n_max=40, rule=rule
        )
        assert run.n == 5

    def test_the_registered_target_is_used_when_none_is_given(
        self, cheap: Callable[..., None]
    ) -> None:
        # a regression of the review: ``run_cell`` had a default of 0.05 that hid the 0.01 the
        # pre-registration gives M-COST; a run that names no target reads the arm's own rule
        cheap()
        run = runner_for(None).run_cell(
            ARMS["B0"], SEEDS, scenario_for("nominal"), batch=10, n_max=len(SEEDS)
        )
        registered = stop_rule_for("B0")
        assert registered.target == 0.01 and run.halfwidth is not None
        assert (run.stopped_by == "target") == (run.halfwidth <= registered.target)
        looser = runner_for(None).run_cell(
            ARMS["B0"], SEEDS, scenario_for("nominal"), batch=10, n_max=len(SEEDS),
            target_halfwidth=0.2,
        )  # fmt: skip
        assert looser.n < run.n or looser.stopped_by == "target"

    def test_a_fixed_cell_runs_every_episode_whatever_the_half_width(
        self, cheap: Callable[..., None]
    ) -> None:
        cheap()  # the monitor arm's cells do not stop early (0003 A15)
        run = runner_for(None).run_cell(
            ARMS["P1"], SEEDS, scenario_for("nominal"), batch=10, n_max=40
        )
        assert run.n == 40 and run.stopped_by == "n_max"
        assert run.halfwidth is not None and run.halfwidth <= 0.5  # it was measured, and not used

    def test_a_fixed_rule_does_not_stop_even_when_its_target_is_met(
        self, cheap: Callable[..., None]
    ) -> None:
        cheap()
        # throughput is constant 1, so the half-width is under any target at the first batch; a
        # fixed rule runs on (the monitor arm's registered rule has a target of 0.0 as well, which
        # nothing meets, but the flag is what says "no early stop")
        stops = runner_for(None).run_cell(
            ARMS["B0"], SEEDS, scenario_for("nominal"), batch=5, n_max=20,
            rule=StopRule("throughput", 0.5),
        )  # fmt: skip
        fixed = runner_for(None).run_cell(
            ARMS["B0"], SEEDS, scenario_for("nominal"), batch=5, n_max=20,
            rule=StopRule("throughput", 0.5, fixed=True),
        )  # fmt: skip
        assert stops.n == 5 and stops.stopped_by == "target"
        assert fixed.n == 20 and fixed.stopped_by == "n_max"

    def test_seeds_must_be_distinct(self, cheap: Callable[..., None]) -> None:
        cheap()
        with pytest.raises(ValueError, match="distinct"):
            runner_for(None).run_cell(ARMS["B0"], [1, 2, 2], scenario_for("nominal"), batch=2)

    def test_fewer_seeds_than_n_max_is_reported(self, cheap: Callable[..., None]) -> None:
        cheap()
        run = runner_for(None).run_cell(
            ARMS["B0"], SEEDS[:6], scenario_for("nominal"), batch=3, n_max=50, target_halfwidth=1e-9
        )
        assert run.n == 6 and run.stopped_by == "seeds"


# ----------------------------------------------------------------------------- failures


class TestFailures:
    def test_a_failed_episode_is_an_episode_with_the_worst_value_and_is_reported(
        self, cheap: Callable[..., None]
    ) -> None:
        def sometimes_fails(task: EpisodeTask, seed: int) -> EpisodeResult:
            if seed == SEEDS[3]:
                raise ValueError("a bug in this seed")
            return fake_result(task, seed)

        cheap(sometimes_fails)
        run = runner_for(None).run_cell(
            ARMS["H-C0"], SEEDS, scenario_for("nominal"), batch=5, n_max=10, target_halfwidth=1e-9
        )
        assert run.n == 10 and len(run.results) == 10  # counted as an episode, no seed dropped
        assert [f.seed for f in run.failures] == [SEEDS[3]]
        assert run.failures[0].kind == "error" and "a bug in this seed" in run.failures[0].detail
        stand_in = run.results[3]
        assert stand_in.failure.startswith("error") and stand_in.seed == SEEDS[3]
        viol = aggregate(run.results, "m_viol_p13", 0.05)
        assert viol.extra["k"] == 1  # the failure is the one violation: worst value
        cost = aggregate(run.results, "m_cost", 0.05)
        assert cost.value > aggregate([r for r in run.results if not r.failure], "m_cost").value

    def test_a_timeout_is_a_failure_of_kind_timeout(self, cheap: Callable[..., None]) -> None:
        def slow(task: EpisodeTask, seed: int) -> EpisodeResult:
            if seed == SEEDS[1]:
                end = time.perf_counter() + 30.0
                while time.perf_counter() < end:  # a busy loop, as a simulation is
                    pass
            return fake_result(task, seed)

        cheap(slow)
        started = time.perf_counter()
        run = runner_for(None, timeout_s=0.4).run_cell(
            ARMS["B0"], SEEDS, scenario_for("nominal"), batch=3, n_max=3, target_halfwidth=1e-9
        )
        assert time.perf_counter() - started < 10.0
        assert [(f.seed, f.kind) for f in run.failures] == [(SEEDS[1], "timeout")]
        assert run.results[1].failure.startswith("timeout")
        assert run.n == 3

    def test_a_timeout_that_lands_after_the_function_returned_does_not_surface_later(self) -> None:
        # the deadline fires while the function is finishing; nothing may be raised afterwards
        for _ in range(20):
            runner_module._with_deadline(lambda: None, 0.0005)
            time.sleep(0.002)

    def test_the_deadline_raises_in_the_calling_thread(self) -> None:
        with pytest.raises(EpisodeTimeout):
            runner_module._with_deadline(lambda: [None for _ in iter(int, 1)], 0.2)

    def test_the_stand_in_is_the_worst_value_of_every_primary_metric(self) -> None:
        arm = ARMS["B3-V3"]
        nominal = worst_episode(arm, 5, "nominal", Fidelity.F1, "timeout: x")
        shifted = worst_episode(arm, 5, "shifted", Fidelity.F1, "timeout: x")
        for metric, want in (
            ("m_viol_p13", 1.0), ("m_viol_p45", 1.0), ("m_viol_p15", 1.0), ("m_cost", 1.0),
            ("u_e", 8.0), ("f_e", 1.0), ("throughput", 0.0), ("m_cov", 0.0), ("m_cov_lo", 0.0),
            ("m_cov_trend", 0.0), ("m_false_alarm_proxy", 1.0),
        ):  # fmt: skip
            assert aggregate([nominal], metric).value == pytest.approx(want), metric
        assert aggregate([shifted], "m_detect_proxy").value == 0.0  # not detected: the worst
        assert nominal.failure == "timeout: x"

    def test_a_stand_in_for_an_arm_with_no_perception_has_no_coverage(self) -> None:
        r = worst_episode(ARMS["B0"], 5, "nominal", Fidelity.F1, "error: x")
        with pytest.raises(ValueError, match="no episode"):
            aggregate([r], "m_cov")
        assert r.monitors == {}


class TestTransientFailures:
    """0003 A15: a timeout or a dead worker is tried once more and never written to the cache; an
    episode that raised is a deterministic failure, cached and not retried."""

    @staticmethod
    def hang() -> None:
        end = time.perf_counter() + 30.0
        while time.perf_counter() < end:  # a busy loop, as a simulation is
            pass

    def run(self, tmp_path: Path | None, **kw: Any) -> Any:
        return runner_for(tmp_path, timeout_s=0.3).run_cell(
            ARMS["B0"],
            SEEDS,
            scenario_for("nominal"),
            batch=3,
            n_max=3,
            target_halfwidth=1e-9,
            **kw,
        )

    def test_a_timeout_that_does_not_come_back_is_a_success_on_the_second_try(
        self, cheap: Callable[..., None]
    ) -> None:
        def once_slow(task: EpisodeTask, seed: int) -> EpisodeResult:
            if seed == SEEDS[1] and CALLS.count(seed) == 1:  # the first try (the call is logged)
                self.hang()
            return fake_result(task, seed)

        cheap(once_slow)
        run = self.run(None)
        assert run.failures == [] and run.retried == 1 and run.n == 3
        assert CALLS.count(SEEDS[1]) == 2
        assert not run.results[1].failure

    def test_a_timeout_that_repeats_is_a_failure_after_exactly_one_retry(
        self, cheap: Callable[..., None]
    ) -> None:
        def always_slow(task: EpisodeTask, seed: int) -> EpisodeResult:
            if seed == SEEDS[1]:
                self.hang()
            return fake_result(task, seed)

        cheap(always_slow)
        run = self.run(None)
        assert [(f.seed, f.kind) for f in run.failures] == [(SEEDS[1], "timeout")]
        assert CALLS.count(SEEDS[1]) == 2 and run.retried == 1  # not three

    def test_a_transient_failure_is_not_written_to_the_cache_so_a_resume_tries_again(
        self, cheap: Callable[..., None], tmp_path: Path
    ) -> None:
        def always_slow(task: EpisodeTask, seed: int) -> EpisodeResult:
            if seed == SEEDS[1]:
                self.hang()
            return fake_result(task, seed)

        cheap(always_slow)
        first = self.run(tmp_path)
        assert [f.kind for f in first.failures] == ["timeout"]
        (cell,) = tmp_path.glob("*.cell")
        assert b"timeout" not in cell.read_bytes()  # the failure itself is not in the file
        CALLS.clear()
        cheap()  # the machine is quieter now: the same seed runs to the end
        second = self.run(tmp_path)
        assert second.failures == [] and CALLS == [SEEDS[1]]  # only the failed seed was run again
        assert second.cached == 2
        third = self.run(tmp_path)
        assert third.cached == 3 and third.computed == 0  # the success is cached like any other

    def test_a_transient_failure_that_an_older_run_left_in_the_cache_is_not_an_answer(
        self, cheap: Callable[..., None], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def always_slow(task: EpisodeTask, seed: int) -> EpisodeResult:
            if seed == SEEDS[2]:
                self.hang()
            return fake_result(task, seed)

        cheap(always_slow)
        with (
            monkeypatch.context() as old_rules
        ):  # what a run did before the rule: cache the timeout
            old_rules.setattr(runner_module, "_transient", lambda outcome: False)
            old = self.run(tmp_path)
        assert [f.kind for f in old.failures] == ["timeout"] and CALLS.count(SEEDS[2]) == 1
        cheap()
        CALLS.clear()
        run = self.run(tmp_path)  # the rule is back: the cached timeout is not taken as an answer
        assert run.failures == [] and CALLS == [SEEDS[2]]

    def test_an_error_is_cached_as_the_failure_it_is_and_not_retried(
        self, cheap: Callable[..., None], tmp_path: Path
    ) -> None:
        def bug(task: EpisodeTask, seed: int) -> EpisodeResult:
            if seed == SEEDS[1]:
                raise ValueError("a bug in this seed")
            return fake_result(task, seed)

        cheap(bug)
        run = self.run(tmp_path)
        assert [f.kind for f in run.failures] == ["error"] and run.retried == 0
        assert CALLS.count(SEEDS[1]) == 1
        CALLS.clear()
        again = self.run(tmp_path)
        assert CALLS == [] and [f.kind for f in again.failures] == ["error"]


# ----------------------------------------------------------------------------- the cache


class TestCache:
    def run(self, tmp_path: Path, n: int = 12, **kw: Any) -> Any:
        return runner_for(tmp_path, **kw).run_cell(
            ARMS["B0"], SEEDS, scenario_for("nominal"), batch=4, n_max=n, target_halfwidth=1e-9
        )

    def test_a_second_run_reads_everything_from_the_cache(
        self, cheap: Callable[..., None], tmp_path: Path
    ) -> None:
        cheap()
        first = self.run(tmp_path)
        assert (first.computed, first.cached) == (12, 0)
        CALLS.clear()
        second = self.run(tmp_path)
        assert (second.computed, second.cached) == (0, 12) and CALLS == []
        assert same(first.results, second.results)

    def test_an_interrupted_run_resumes_and_gives_identical_results(
        self, cheap: Callable[..., None], tmp_path: Path
    ) -> None:
        cheap()
        whole = self.run(tmp_path / "whole")
        CALLS.clear()
        charges = iter([50.0, 50.0, 95.0])  # the third batch boundary is over the limit
        with pytest.raises(CommitChargeStop):
            runner_for(tmp_path / "cut", headroom=lambda: next(charges)).run_cell(
                ARMS["B0"], SEEDS, scenario_for("nominal"), batch=4, n_max=12, target_halfwidth=1e-9
            )
        done = len(CALLS)
        assert done == 8  # two batches finished before the stop
        resumed = self.run(tmp_path / "cut")
        assert (resumed.computed, resumed.cached) == (4, 8)
        assert CALLS[done:] == SEEDS[8:12]  # only the remaining seeds were simulated
        assert same(whole.results, resumed.results)

    def test_a_torn_tail_is_ignored_and_the_lost_episode_is_run_again(
        self, cheap: Callable[..., None], tmp_path: Path
    ) -> None:
        cheap()
        whole = self.run(tmp_path)
        (path,) = list(tmp_path.glob("*.cell"))
        data = path.read_bytes()
        path.write_bytes(data[:-7])  # the app stopped in the middle of the last record
        CALLS.clear()
        again = self.run(tmp_path)
        assert CALLS == [SEEDS[11]]  # exactly the torn record
        assert same(whole.results, again.results)
        # and the file is whole again: a third run finds all twelve
        CALLS.clear()
        assert self.run(tmp_path).cached == 12 and CALLS == []

    def test_a_corrupt_record_and_everything_after_it_are_not_trusted(
        self, cheap: Callable[..., None], tmp_path: Path
    ) -> None:
        cheap()
        self.run(tmp_path)
        (path,) = list(tmp_path.glob("*.cell"))
        raw = bytearray(path.read_bytes())
        raw[len(raw) // 2] ^= 0xFF
        path.write_bytes(bytes(raw))
        CALLS.clear()
        again = self.run(tmp_path)
        assert 0 < len(CALLS) < 12 and again.n == 12
        assert [r.seed for r in again.results] == SEEDS[:12]

    def test_a_file_with_a_foreign_header_is_not_used(
        self, cheap: Callable[..., None], tmp_path: Path
    ) -> None:
        cheap()
        self.run(tmp_path)
        (path,) = list(tmp_path.glob("*.cell"))
        other = CellCache(tmp_path, path.stem, {"task": "something else"})
        assert other.load() == {}

    def test_the_key_depends_on_everything_that_can_change_an_episode(self) -> None:
        env = {"code": "c1"}
        tuned = PROVISIONAL
        base = EpisodeTask(ARMS["B0"], scenario_for("nominal"), Fidelity.F1, tuned)
        key = runner_module.cell_key(base, env)[0]
        variants = {
            "scenario": EpisodeTask(ARMS["B0"], scenario_for("shifted"), Fidelity.F1, tuned),
            "arm": EpisodeTask(ARMS["B1"], scenario_for("nominal"), Fidelity.F1, tuned),
            "fidelity": EpisodeTask(ARMS["B0"], scenario_for("nominal"), Fidelity.F1D, tuned),
            "tuned": EpisodeTask(
                ARMS["B0"],
                scenario_for("nominal"),
                Fidelity.F1,
                dataclasses.replace(tuned, k_b3=9.0),
            ),
            "slim": dataclasses.replace(base, slim=False),
        }
        keys = {name: runner_module.cell_key(t, env)[0] for name, t in variants.items()}
        assert all(k != key for k in keys.values()), keys
        assert len(set(keys.values())) == len(keys)
        assert runner_module.cell_key(base, {"code": "c2"})[0] != key
        assert runner_module.cell_key(base, env)[0] == key  # and it is stable

    def test_arms_that_differ_only_in_their_proposer_arguments_have_different_keys(self) -> None:
        env = {"code": "c"}
        a = EpisodeTask(ARMS["P0-mis10"], scenario_for("nominal"), Fidelity.F1, PROVISIONAL)
        b = EpisodeTask(ARMS["P0-mis25"], scenario_for("nominal"), Fidelity.F1, PROVISIONAL)
        assert runner_module.cell_key(a, env)[0] != runner_module.cell_key(b, env)[0]
        # the arm fingerprint names no memory address, so it is the same in another process
        assert "0x" not in str(a.fingerprint())

    def test_no_cache_directory_means_nothing_is_written(
        self, cheap: Callable[..., None], tmp_path: Path
    ) -> None:
        cheap()
        run = runner_for(None).run_cell(
            ARMS["B0"], SEEDS, scenario_for("nominal"), batch=4, n_max=8, target_halfwidth=1e-9
        )
        assert run.computed == 8 and not list(tmp_path.iterdir())

    def test_failures_are_cached_too_so_that_a_resume_does_not_retry_them(
        self, cheap: Callable[..., None], tmp_path: Path
    ) -> None:
        def broken(task: EpisodeTask, seed: int) -> EpisodeResult:
            if seed == SEEDS[2]:
                raise RuntimeError("deterministic failure")
            return fake_result(task, seed)

        cheap(broken)
        self.run(tmp_path, n=4)
        CALLS.clear()
        again = self.run(tmp_path, n=4)
        assert CALLS == [] and [f.seed for f in again.failures] == [SEEDS[2]]

    def test_the_cache_holds_the_slim_result(self, tmp_path: Path) -> None:
        cache = CellCache(tmp_path, "k", {"x": 1})
        r = fake_result(
            EpisodeTask(ARMS["B0"], scenario_for("nominal"), Fidelity.F1, PROVISIONAL), 1
        )
        cache.append(Done(1, r))
        cache.append(Failed(2, "error", "boom"))
        loaded = CellCache(tmp_path, "k", {"x": 1}).load()
        assert loaded[1] == Done(1, r) and loaded[2] == Failed(2, "error", "boom")


# ----------------------------------------------------------------------------- the gate


class TestCommitCharge:
    def test_a_run_over_the_limit_does_not_start_a_batch(self, cheap: Callable[..., None]) -> None:
        cheap()
        with pytest.raises(CommitChargeStop) as info:
            runner_for(None, headroom=lambda: 91.0).run_cell(
                ARMS["B0"], SEEDS, scenario_for("nominal"), batch=4, n_max=8
            )
        assert CALLS == [] and info.value.charge == 91.0 and "88" in str(info.value)

    def test_the_limit_is_inclusive_and_88_point_0_stops(self, cheap: Callable[..., None]) -> None:
        cheap()
        with pytest.raises(CommitChargeStop):
            runner_for(None, headroom=lambda: 88.0).run_cell(
                ARMS["B0"], SEEDS, scenario_for("nominal"), batch=4, n_max=8
            )
        run = runner_for(None, headroom=lambda: 87.9).run_cell(
            ARMS["B0"], SEEDS, scenario_for("nominal"), batch=4, n_max=8, target_halfwidth=1e-9
        )
        assert run.n == 8

    def test_it_stops_at_the_batch_boundary_not_inside_a_batch(
        self, cheap: Callable[..., None]
    ) -> None:
        cheap()
        charges = iter([50.0, 99.0])
        with pytest.raises(CommitChargeStop):
            runner_for(None, headroom=lambda: next(charges)).run_cell(
                ARMS["B0"], SEEDS, scenario_for("nominal"), batch=4, n_max=12, target_halfwidth=1e-9
            )
        assert CALLS == SEEDS[:4]  # the whole first batch ran, the second never started

    def test_a_batch_that_is_all_in_the_cache_is_not_held_up_by_the_charge(
        self, cheap: Callable[..., None], tmp_path: Path
    ) -> None:
        cheap()
        done = runner_for(tmp_path).run_cell(
            ARMS["B0"], SEEDS, scenario_for("nominal"), batch=4, n_max=8, target_halfwidth=1e-9
        )
        CALLS.clear()
        again = runner_for(tmp_path, headroom=lambda: 99.0).run_cell(
            ARMS["B0"], SEEDS, scenario_for("nominal"), batch=4, n_max=8, target_halfwidth=1e-9
        )
        assert CALLS == [] and again.cached == 8 and same(done.results, again.results)
        # but the first batch with work to do is held up
        with pytest.raises(CommitChargeStop):
            runner_for(tmp_path, headroom=lambda: 99.0).run_cell(
                ARMS["B0"], SEEDS, scenario_for("nominal"), batch=4, n_max=12,
                target_halfwidth=1e-9,
            )  # fmt: skip
        assert CALLS == []

    def test_an_unreadable_charge_does_not_stop_a_run(self, cheap: Callable[..., None]) -> None:
        cheap()
        run = runner_for(None, headroom=lambda: None).run_cell(
            ARMS["B0"], SEEDS, scenario_for("nominal"), batch=4, n_max=4, target_halfwidth=1e-9
        )
        assert run.n == 4


# ----------------------------------------------------------------------------- code hash


class TestSimulationCodeHash:
    def make(self, root: Path) -> None:
        for rel in (
            "closed_loop/plant/plant.py",
            "closed_loop/eval/episode.py",
            "closed_loop/eval/report.py",
            "closed_loop/eval/claims.py",
            "closed_loop/eval/runner.py",
            "closed_loop/eval/counts.py",
            "closed_loop/eval/stages.py",
            "closed_loop/audit/ledger.py",
        ):
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_bytes(f"# {rel}\n".encode())

    def test_it_changes_with_a_module_that_takes_part_in_a_simulation(self, tmp_path: Path) -> None:
        self.make(tmp_path)
        before = simulation_code_sha256.__wrapped__(str(tmp_path))  # type: ignore[attr-defined]
        (tmp_path / "closed_loop/plant/plant.py").write_bytes(b"# changed\n")
        assert simulation_code_sha256.__wrapped__(str(tmp_path)) != before  # type: ignore[attr-defined]

    @pytest.mark.parametrize(
        "rel",
        [
            "closed_loop/eval/runner.py",
            "closed_loop/eval/stages.py",
            "closed_loop/f2/episode.py",
            "closed_loop/search/falsify.py",
        ],
    )
    def test_it_changes_with_the_modules_that_define_what_a_worker_runs(
        self, tmp_path: Path, rel: str
    ) -> None:
        # the tasks of the pool (an episode, an open-loop replay, a search run) and the F2 episode
        # live in these: a cache keyed without them would serve results of the old code
        self.make(tmp_path)
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b"# before\n")
        before = simulation_code_sha256.__wrapped__(str(tmp_path))  # type: ignore[attr-defined]
        (tmp_path / rel).write_bytes(b"# after\n")
        assert simulation_code_sha256.__wrapped__(str(tmp_path)) != before  # type: ignore[attr-defined]

    @pytest.mark.parametrize(
        "rel",
        [
            "closed_loop/eval/report.py",
            "closed_loop/eval/claims.py",
            "closed_loop/eval/assemble.py",
            "closed_loop/eval/counts.py",
            "closed_loop/audit/ledger.py",
        ],
    )
    def test_it_ignores_modules_that_only_read_results(self, tmp_path: Path, rel: str) -> None:
        self.make(tmp_path)
        before = simulation_code_sha256.__wrapped__(str(tmp_path))  # type: ignore[attr-defined]
        (tmp_path / rel).write_bytes(b"# edited: a table, not a simulation\n")
        assert simulation_code_sha256.__wrapped__(str(tmp_path)) == before  # type: ignore[attr-defined]

    @pytest.mark.parametrize("name", ["clause_cache.json", "note_verdicts.json"])
    def test_it_changes_with_the_data_files_the_simulation_reads(
        self, tmp_path: Path, name: str
    ) -> None:
        # 0003 A15: the clause cache and the note verdicts decide a result as a module does
        self.make(tmp_path)
        data = tmp_path / "closed_loop" / "spec" / name
        data.parent.mkdir(parents=True, exist_ok=True)
        data.write_bytes(b'{"v": 1}\n')
        before = simulation_code_sha256.__wrapped__(str(tmp_path))  # type: ignore[attr-defined]
        data.write_bytes(b'{"v": 2}\n')
        assert simulation_code_sha256.__wrapped__(str(tmp_path)) != before  # type: ignore[attr-defined]

    def test_other_data_files_are_not_part_of_it(self, tmp_path: Path) -> None:
        self.make(tmp_path)
        other = tmp_path / "closed_loop" / "spec" / "other.json"
        other.parent.mkdir(parents=True, exist_ok=True)
        other.write_bytes(b"{}\n")
        before = simulation_code_sha256.__wrapped__(str(tmp_path))  # type: ignore[attr-defined]
        other.write_bytes(b'{"changed": true}\n')
        assert simulation_code_sha256.__wrapped__(str(tmp_path)) == before  # type: ignore[attr-defined]

    def test_it_does_not_depend_on_line_endings(self, tmp_path: Path) -> None:
        self.make(tmp_path)
        before = simulation_code_sha256.__wrapped__(str(tmp_path))  # type: ignore[attr-defined]
        p = tmp_path / "closed_loop/eval/episode.py"
        p.write_bytes(p.read_bytes().replace(b"\n", b"\r\n"))
        assert simulation_code_sha256.__wrapped__(str(tmp_path)) == before  # type: ignore[attr-defined]


# ----------------------------------------------------------------------------- real episodes

SEED = 1100


class TestSlimming:
    def test_slimming_keeps_what_the_metrics_read(self, small_context: PreparedContext) -> None:
        full = run_episode(ARMS["B3-V1"], SEED, scenario_for("nominal"), context=small_context)
        slim = slim_result(full)
        assert full.labels and full.checkpoints and not slim.labels and not slim.checkpoints
        for metric in ("m_cov", "m_cov_lo", "m_cov_trend", "m_cost", "throughput", "u_e",
                       "m_viol_p15", "set_width", "wrong_mode_jobs"):  # fmt: skip
            try:
                a = aggregate([full], metric).value
            except ValueError:
                with pytest.raises(ValueError):
                    aggregate([slim], metric)
                continue
            assert aggregate([slim], metric).value == a, metric
        assert slim.learner == learner_counts(full.labels) and slim.learner["post_blind.n"] > 0
        assert len(pickle.dumps(slim)) * 3 < len(pickle.dumps(full))

    def test_the_learner_counts_agree_with_the_labels(self, small_context: PreparedContext) -> None:
        full = run_episode(ARMS["B3-V1"], SEED, scenario_for("nominal"), context=small_context)
        counts = full.learner
        total = sum(counts[f"post_blind.{s}"] for s in ("hit", "miss", "lower_hit", "unknown"))
        assert total == counts["post_blind.n"]
        assert counts["estimated.n"] <= counts["post_blind.n"]


@pytest.mark.slow
class TestPoolOfWorkers:
    def cell(self, small_context: PreparedContext, workers: int) -> Any:
        with Runner(
            workers=workers, context=small_context, headroom=lambda: 40.0, timeout_s=300.0
        ) as r:
            return r.run_cell(
                ARMS["B0"], SEEDS[:6], scenario_for("nominal"), batch=3, n_max=6,
                target_halfwidth=1e-9,
            )  # fmt: skip

    def test_results_are_identical_for_one_and_three_workers(
        self, small_context: PreparedContext
    ) -> None:
        one = self.cell(small_context, 1)
        three = self.cell(small_context, 3)
        assert one.n == three.n == 6
        assert same(one.results, three.results)
        assert [r.seed for r in three.results] == SEEDS[:6]  # in seed order, however they finished
        assert not one.failures and not three.failures

    def test_the_pool_survives_a_cell_and_serves_the_next(
        self, small_context: PreparedContext
    ) -> None:
        with Runner(workers=2, context=small_context, headroom=lambda: 40.0, timeout_s=300.0) as r:
            a = r.run_cell(ARMS["B0"], SEEDS[:4], scenario_for("nominal"), batch=2, n_max=4)
            pool = r._pool
            b = r.run_cell(ARMS["B1"], SEEDS[:4], scenario_for("nominal"), batch=2, n_max=4)
            assert r._pool is pool and a.n == b.n == 4

    def test_a_worker_that_dies_is_replaced_and_the_episode_is_counted_failed_if_it_repeats(
        self, small_context: PreparedContext, tmp_path: Path
    ) -> None:
        # a task whose seed kills the worker process outright (os._exit), module level so that a
        # spawned worker can import it
        from tests.eval import _workers_support as support

        with Runner(workers=2, context=small_context, headroom=lambda: 40.0, timeout_s=300.0) as r:
            run = r.run_seeds(support.DieOnSeed(1102), [1100, 1101, 1102, 1103], batch=4, n_max=4)
        assert [f.seed for f in run.failures] == [1102] and run.failures[0].kind == "worker-died"
        assert sorted(run.values) == [1100, 1101, 1103]


def test_the_environment_fingerprint_names_both_specifications(
    small_context: PreparedContext,
) -> None:
    env = runner_module.environment_fingerprint(small_context)
    assert env["spec_clause"] == repr(small_context.spec_clause)
    assert env["spec_stale"] == repr(small_context.spec_stale)
    assert env["spec_clause"] != env["spec_stale"]  # two clauses, two specifications
    none = runner_module.environment_fingerprint(None)
    assert none["spec_clause"] is None and none["spec_stale"] is None


def test_the_environment_fingerprint_names_the_code_and_the_libraries(
    small_context: PreparedContext,
) -> None:
    env = runner_module.environment_fingerprint(small_context)
    assert env["code"] == simulation_code_sha256() and env["python"] and env["numpy"]
    assert env["scipy"] and env["context"] == repr(small_context.spec)
    assert os.sep not in str(env["code"])


def test_a_use_tuned_override_travels_with_the_task() -> None:
    with use_tuned(k_b3=11.0):
        from closed_loop.eval.tuned import current

        task = EpisodeTask(ARMS["B3-V3"], scenario_for("nominal"), Fidelity.F1, current())
    assert task.tuned.k_b3 == 11.0 and task.fingerprint()["tuned"]["k_b3"] == 11.0
    again = pickle.loads(pickle.dumps(task))
    assert again.fingerprint() == task.fingerprint() and again.arm.name == task.arm.name


class TestReadCellHeader:
    """P12g: the public reader of a cell file's header (it used to be two private constants read by
    a script)."""

    def cell(self, tmp_path: Path) -> CellCache:
        cache = CellCache(
            tmp_path, "k" * 40, {"task": {"kind": "demo", "n": 3}, "env": {"code": "c"}}
        )
        cache.append(Done(1, "one"))
        cache.append(Done(2, "two"))
        return cache

    def test_it_returns_the_header_the_cell_was_made_with_and_its_key(self, tmp_path: Path) -> None:
        from closed_loop.eval.runner import read_cell_header

        cache = self.cell(tmp_path)
        header = read_cell_header(cache.path)
        assert header == {"key": "k" * 40, "task": {"kind": "demo", "n": 3}, "env": {"code": "c"}}

    def test_it_agrees_with_what_the_cache_loads_with_that_header(self, tmp_path: Path) -> None:
        from closed_loop.eval.runner import read_cell_header

        cache = self.cell(tmp_path)
        header = read_cell_header(cache.path)
        again = CellCache(tmp_path, "k" * 40, {k: v for k, v in header.items() if k != "key"})
        assert {s: o.value for s, o in again.load().items()} == {1: "one", 2: "two"}

    def test_a_file_that_is_not_a_cell_is_refused(self, tmp_path: Path) -> None:
        from closed_loop.eval.runner import read_cell_header

        junk = tmp_path / "junk.cell"
        junk.write_bytes(b"not a cell at all, just bytes")
        with pytest.raises(ValueError, match="not a cell file"):
            read_cell_header(junk)

    def test_a_header_that_was_altered_or_cut_short_is_refused(self, tmp_path: Path) -> None:
        from closed_loop.eval.runner import _HEAD, _MAGIC, read_cell_header

        cache = self.cell(tmp_path)
        raw = bytearray(cache.path.read_bytes())
        raw[len(_MAGIC) + _HEAD.size + 32 + 3] ^= 0x01
        altered = tmp_path / "altered.cell"
        altered.write_bytes(bytes(raw))
        with pytest.raises(ValueError, match="torn or altered"):
            read_cell_header(altered)
        short = tmp_path / "short.cell"
        short.write_bytes(cache.path.read_bytes()[: len(_MAGIC) + _HEAD.size + 40])
        with pytest.raises(ValueError, match="torn or altered"):
            read_cell_header(short)
        only_magic = tmp_path / "magic.cell"
        only_magic.write_bytes(cache.path.read_bytes()[: len(_MAGIC)])
        with pytest.raises(ValueError, match="no header"):
            read_cell_header(only_magic)
