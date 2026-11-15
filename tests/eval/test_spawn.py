"""An arm and a prepared context cross a ``spawn`` boundary and give the same episode."""

from __future__ import annotations

import dataclasses
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import pytest

from closed_loop.eval.arms import ARMS, Arm
from closed_loop.eval.context import PreparedContext, install_context
from closed_loop.eval.episode import EpisodeResult, run_episode, scenario_for, summary

SEED = 1100


def run_in_worker(arm: Arm, seed: int, condition: str) -> EpisodeResult:
    """Module level, so a spawned worker can import it. The arm arrives pickled; the context was
    installed by the initializer, and ``run_episode`` finds it as the process default."""
    return run_episode(arm, seed, scenario_for(condition))


@pytest.mark.slow
def test_a_spawned_worker_rebuilds_the_arm_and_gives_the_same_episode(
    small_context: PreparedContext,
) -> None:
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=1, mp_context=ctx, initializer=_install, initargs=(small_context,)
    ) as pool:
        remote = pool.submit(run_in_worker, ARMS["H-G0"], SEED, "nominal").result(timeout=600)
    local = run_episode(ARMS["H-G0"], SEED, scenario_for("nominal"), context=small_context)
    assert summary(remote) == summary(local)
    assert dataclasses.replace(remote, wall_s=0.0) == dataclasses.replace(local, wall_s=0.0)


def _install(context: PreparedContext) -> None:
    # The worker's default context is the one handed over: no half-minute refit per worker.
    from closed_loop.eval import context as module

    module._PROCESS.clear()
    install_context(context)
