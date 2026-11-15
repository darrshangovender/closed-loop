"""Workers do not outlive a failed or interrupted run (a regression test of P12b).

A spawned worker in the middle of a long episode keeps the interpreter from exiting (it joins its
workers at exit) and, if the parent was killed, stays on the machine for as long as the episode
lasts. The coordinator's report: an orphaned worker hangs ``benchmarks/run.py`` on the owner's
machine. Each test starts real ``spawn`` workers that "run an episode" of ten minutes, ends the
run in one of the ways a run ends badly, and requires that every worker is gone.
"""

from __future__ import annotations

import _thread
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from closed_loop.eval import runner as runner_module
from closed_loop.eval.runner import CellCache, Runner, _execute
from tests.eval._workers_support import Echo, Hang, all_gone, pid_alive, wait_for_pids

pytestmark = pytest.mark.slow
ROOT = Path(__file__).resolve().parents[2]


def runner(**kw: Any) -> Runner:
    return Runner(workers=2, headroom=lambda: 40.0, timeout_s=None, **kw)


def start_busy_workers(r: Runner, pid_dir: Path) -> list[int]:
    pool = r._ensure_pool()
    for seed in (1, 2):
        pool.submit(_execute, Hang(str(pid_dir)), seed, None)
    return wait_for_pids(pid_dir, 2)


class TestReap:
    def test_close_with_kill_ends_workers_that_are_in_the_middle_of_an_episode(
        self, tmp_path: Path
    ) -> None:
        r = runner()
        pids = start_busy_workers(r, tmp_path)
        assert all(pid_alive(p) for p in pids)
        started = time.time()
        r.close(kill=True)
        assert all_gone(pids, 20.0) and time.time() - started < 30.0

    def test_a_polite_close_gives_idle_workers_a_moment_then_ends_the_rest(
        self, tmp_path: Path
    ) -> None:
        r = runner()
        pids = start_busy_workers(r, tmp_path)
        pool = r._pool
        assert pool is not None
        r._pool = None
        started = time.time()
        runner_module.reap(pool, grace_s=0.5)  # busy workers do not leave by themselves
        assert all_gone(pids, 20.0) and time.time() - started < 30.0

    def test_leaving_a_with_block_on_an_exception_kills_at_once(self, tmp_path: Path) -> None:
        pids: list[int] = []
        with pytest.raises(RuntimeError, match="the run failed"):
            with runner() as r:
                pids += start_busy_workers(r, tmp_path)
                raise RuntimeError("the run failed")
        assert all_gone(pids, 20.0)

    def test_a_pool_is_made_again_after_a_reap(self, tmp_path: Path) -> None:
        r = runner()
        start_busy_workers(r, tmp_path)
        r.close(kill=True)
        run = r.run_seeds(Echo(), [1, 2, 3], batch=3, n_max=3)
        r.close()
        assert sorted(run.values) == [1, 2, 3] and not run.failures


class TestEndedBadly:
    def test_an_interrupt_in_the_middle_of_a_cell_reaps_the_workers(self, tmp_path: Path) -> None:
        pids: list[int] = []

        def interrupt_when_busy() -> None:
            pids.extend(wait_for_pids(tmp_path, 2))
            _thread.interrupt_main()  # what Ctrl+C does

        threading.Thread(target=interrupt_when_busy, daemon=True).start()
        with pytest.raises(KeyboardInterrupt):  # noqa: SIM117
            with runner() as r:
                r.run_seeds(Hang(str(tmp_path)), [1, 2], batch=2, n_max=2)
        assert len(pids) == 2 and all_gone(pids, 20.0)

    def test_a_failure_while_collecting_results_reaps_the_workers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken(self: CellCache, outcome: Any) -> None:
            raise RuntimeError("the disk is full")

        monkeypatch.setattr(CellCache, "append", broken)
        r = runner(cache_dir=tmp_path / "cache")
        # seed 1 returns as soon as one worker is busy; seed 2 stays in its episode
        with pytest.raises(RuntimeError, match="disk is full"):
            r.run_seeds(Hang(str(tmp_path), quick_seed=1), [1, 2], batch=2, n_max=2)
        pids = wait_for_pids(tmp_path, 1)
        assert all_gone(pids, 20.0) and r._pool is None

    def test_a_failure_between_batches_reaps_the_idle_workers(self) -> None:
        # no episode is in flight and no generator is suspended: only the handler of ``run_seeds``
        # stands between a failing stopping rule and two idle workers kept for a run that is over
        r = runner()
        pids: list[int] = []

        def stop(_values: list[Any]) -> tuple[bool, float | None]:
            assert r._pool is not None
            pids.extend(p.pid for p in r._pool._processes.values() if p.pid is not None)
            raise RuntimeError("the stopping rule failed")

        with pytest.raises(RuntimeError, match="stopping rule failed"):
            r.run_seeds(Echo(), [1, 2, 3, 4], batch=2, n_max=4, stop=stop)
        assert r._pool is None
        assert pids and all_gone(pids, 20.0)

    def test_a_pause_at_a_batch_boundary_keeps_the_idle_pool(self, tmp_path: Path) -> None:
        # a commit-charge pause is not a failure: the next run of the same runner reuses its pool
        charges = iter([50.0, 99.0])
        r = Runner(workers=2, headroom=lambda: next(charges), timeout_s=None)
        from closed_loop.audit.guard import GuardError

        with pytest.raises(GuardError):
            r.run_seeds(Echo(), [1, 2, 3, 4], batch=2, n_max=4)
        assert r._pool is not None
        pids = [p.pid for p in r._pool._processes.values() if p.pid is not None]
        r.close()
        assert pids and all_gone(pids, 20.0)

    def script(self, tmp_path: Path, ending: str) -> Path:
        path = tmp_path / "parent.py"
        path.write_text(
            textwrap.dedent(
                f"""
                import os, sys
                sys.path.insert(0, {str(ROOT)!r})
                from closed_loop.eval.runner import Runner, _execute
                from tests.eval._workers_support import Hang, wait_for_pids

                if __name__ == "__main__":  # a spawned worker imports this file too
                    r = Runner(workers=2, headroom=lambda: 40.0, timeout_s=None)
                    pool = r._ensure_pool()
                    for s in (1, 2):
                        pool.submit(_execute, Hang({str(tmp_path / "pids")!r}), s, None)
                    wait_for_pids({str(tmp_path / "pids")!r}, 2)
                    print("ready", flush=True)
                    {ending}
                """
            ),
            encoding="utf-8",
        )
        (tmp_path / "pids").mkdir(exist_ok=True)
        return path

    def test_a_parent_that_dies_of_an_unhandled_error_does_not_hang_and_leaves_no_worker(
        self, tmp_path: Path
    ) -> None:
        script = self.script(tmp_path, 'raise RuntimeError("boom, and nobody called close")')
        done = subprocess.run(
            [sys.executable, str(script)], capture_output=True, text=True, timeout=240,
            check=False, cwd=ROOT,
        )  # fmt: skip
        assert done.returncode != 0 and "ready" in done.stdout and "boom" in done.stderr
        pids = wait_for_pids(tmp_path / "pids", 2, timeout_s=1.0)
        assert all_gone(pids, 30.0)

    def test_a_parent_that_is_killed_outright_leaves_no_worker(self, tmp_path: Path) -> None:
        script = self.script(tmp_path, "os._exit(0)")  # no exit hook, no close: a killed process
        done = subprocess.run(
            [sys.executable, str(script)], capture_output=True, text=True, timeout=240,
            check=False, cwd=ROOT,
        )  # fmt: skip
        assert done.returncode == 0 and "ready" in done.stdout
        pids = wait_for_pids(tmp_path / "pids", 2, timeout_s=1.0)
        assert all_gone(pids, 60.0)  # each worker saw its parent go and left


def test_the_exit_hook_exists_on_this_interpreter() -> None:
    # it is private API (threading._register_atexit): if a Python release drops it, the tests above
    # of a parent that dies of an unhandled error would hang, and this says why
    assert getattr(runner_module.threading, "_register_atexit", None) is not None
    assert callable(runner_module._reap_all_pools)
