"""The F2 stage of the benchmark: F2 arms, the counted wrong-diagnosis metric, the replay of F1
finds and the estimator mismatch (packet P11c). Everything is SYNTHETIC.

``stages.run_f2`` imports this module and calls ``run(sc)``; nothing else in the runner knows about
F2. It is a stage of the main process: an F2 episode needs the three siblings (a 15 to 45 s import)
and about 15 to 45 s of one core, and the stage is 180 episodes at most, so the episodes run one
after another here rather than on the pool of F1 workers (each of which would pay the import). They
still go through a one-worker ``Runner`` (``Runner.in_process``), so each episode has a time limit,
a failure is a counted outcome and not a dead run, and a finished episode is in the cache: a run
that is paused or interrupted does not redo it.

What leaves the stage, as ``ResultRecord``s with fidelity ``F2``:

* ``claim/C-TOGETHER/arm/<arm>``: episodes completed against episodes tried, and the cause of the
  first failure (``extra["completed"]``, ``extra["tried"]``, ``extra["cause"]``);
* ``claim/C-TOGETHER/confirmed``: replays that confirmed the F1 violation against replays
  (``extra["k"]``), and ``f2.replay.reason.<code>`` for every unconfirmed one;
* ``f2.<arm>.wrong_compliant_signed`` and ``.presented``: the counted metric of 0003 A3, wrong-mode
  drafts that passed all sixteen verifier rules and reached a signature (or were presented to a
  human), with the drafts that named a mode as the denominator in ``extra``;
* per arm: throughput, cost, unplanned failures, seconds per episode, ``run_asset`` calls against
  the caching rule;
* ``f2.estimator_mismatch.mae_diff_days``: F2 minus F1 error of the point RUL on 20 development
  lives.

Seeds: ``sc.published_f2_seeds``, drawn once from the held-out guard (``HeldoutGuard.f2_seeds``,
which derives them with ``audit.seeds.f2_seed`` and counts them in the ledger); development seeds
in a ``--quick`` run, which has no pre-registration hash.

The replay set is the search stage's finds and nothing else (M-CONF, prereg 5.1): the list of
``stages.SearchFind`` in ``sc.shared["search_violations"]``, the first violating evaluation of each
search run, taken in run order across regimes and algorithms and capped at ``f2_replays``. There is
no search of this stage's own: a private search would measure something other than what is
registered. If the search ran and found nothing the list is empty and the stage says "0 of 0"; if
no search ran in this invocation (``--quick --stages f2``) it says that; in a full run, where the
search precedes this stage, a missing list is an error. A replay that cannot run (an error, a
timeout) is kept in the denominator with a reason code.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

import numpy as np

from closed_loop.audit.prereg import file_sha256
from closed_loop.core.results import ResultRecord
from closed_loop.core.types import Fidelity
from closed_loop.eval.claims import claim_name
from closed_loop.eval.counts import QUICK_FIRST_DEV_SEED
from closed_loop.eval.episode import EpisodeResult, scenario_for
from closed_loop.eval.runner import Failed, arm_fingerprint, slim_result
from closed_loop.eval.stages import (
    NotBuiltError,
    SearchFind,
    StageContext,
    StageOutput,
    value_record,
)
from closed_loop.f2._siblings import F2Unavailable
from closed_loop.f2.episode import F2_ARMS, F2Log, require_siblings, simulate_f2_episode
from closed_loop.f2.mismatch import dev_lives, estimator_mismatch, mismatch_record
from closed_loop.f2.replay import REASON_CODES, ReplayOutcome, replay_theta

__all__ = ["run", "seeds_for"]

SCRIPT: Final[str] = "closed_loop.eval.stage_f2"
DETERMINISM_CHECKS: Final[int] = 2  # replays whose F2 episode is run twice
ESTIMATOR_LIVES: Final[int] = 20
EPISODE_BUDGET_S: Final[float] = 30.0  # the budget of an F2 episode (a line of the run log)
F2_BATCH: Final[int] = 4  # episodes between two looks at the commit charge (about two minutes)
REPLAY_TIME_FACTOR: Final[float] = 3.0  # a replay is an F1 episode and one or two F2 episodes
# the stage's own reason codes, for a replay that did not finish (the codes of ``f2.replay`` are for
# one that did)
FAILURE_CODES: Final[tuple[str, ...]] = ("replay_error", "replay_timeout")


class _Episode(Protocol):
    """What the records read of an F2 episode (an ``F2Record``, or the slim one the cache holds)."""

    @property
    def result(self) -> EpisodeResult: ...

    @property
    def log(self) -> F2Log: ...


@dataclass(frozen=True)
class F2Slim:
    """An F2 episode without its trace: the result and the sibling-side log, to be cached."""

    result: EpisodeResult
    log: F2Log


def _siblings_hash(root: Path) -> str | None:
    lock = root / "integration" / "siblings.lock"
    return file_sha256(lock) if lock.is_file() else None


@dataclass(frozen=True)
class F2EpisodeTask:
    """One F2 arm, nominal condition; the seed is the episode's. ``siblings`` is the hash of the
    lock file of the three repositories, so a changed sibling is a changed cache key."""

    arm_name: str
    siblings: str | None

    def label(self) -> str:
        return f"f2/{self.arm_name}"

    def fingerprint(self) -> dict[str, object]:
        return {
            "kind": "f2-episode",
            "arm": arm_fingerprint(F2_ARMS[self.arm_name]),
            "condition": "nominal",
            "siblings_lock": self.siblings,
        }

    def __call__(self, seed: int) -> F2Slim:
        rec = simulate_f2_episode(F2_ARMS[self.arm_name], seed, scenario_for("nominal"))
        return F2Slim(slim_result(rec.result), rec.log)


def _target_arm() -> object:
    module = importlib.import_module("closed_loop.search.falsify")
    return module.target_arm()


@dataclass(frozen=True)
class ReplayTask:
    """The replay of one find; the "seed" a runner hands it is the index into ``finds``."""

    finds: tuple[SearchFind, ...]
    siblings: str | None
    determinism_checks: int = DETERMINISM_CHECKS

    def label(self) -> str:
        return "f2/replay"

    def fingerprint(self) -> dict[str, object]:
        listed = [[f.seed, f.regime, f.algo, list(f.theta), list(f.props)] for f in self.finds]
        return {
            "kind": "f2-replay",
            "target": arm_fingerprint(_target_arm()),  # type: ignore[arg-type]
            "finds": hashlib.sha256(json.dumps(listed).encode("utf-8")).hexdigest(),
            "determinism_checks": self.determinism_checks,
            "siblings_lock": self.siblings,
        }

    def __call__(self, seed: int) -> ReplayOutcome:  # the runner calls the index a seed
        index = seed
        find = self.finds[index]
        outcome = replay_theta(
            _target_arm(),  # type: ignore[arg-type]
            np.asarray(find.theta, dtype=np.float64),
            find.regime,
            find.seed,
            verify_determinism=index < self.determinism_checks,
        )
        return dataclasses.replace(outcome, f2_log=None)  # the log is not needed, and is big


def failed_replay(find: SearchFind, failure: Failed) -> ReplayOutcome:
    """A replay that did not finish stays in the denominator, unconfirmed, with its reason code."""
    code = "replay_timeout" if failure.kind == "timeout" else "replay_error"
    return ReplayOutcome(
        arm="R-target",
        seed=find.seed,
        regime=find.regime,
        confirmed=False,
        reason=code,
        f1_props=find.props,
        f2_props=(),
        common=(),
        f1_labels=(),
        f1_planned_jobs=0,
        f2_repairs=0,
        f2_inspections=0,
        f2_drafts=0,
        f2_runs_rejected=0,
        f2_runs_escalated=0,
        evidence=f"{failure.kind}: {failure.detail}"[:300],
        f1_wall_s=0.0,
        f2_wall_s=0.0,
    )


def replay_set(sc: StageContext) -> tuple[list[SearchFind], str]:
    """The finds to replay, capped at ``f2_replays``, and a sentence about where they came from."""
    shared = sc.shared.get("search_violations")
    if shared is None:
        if sc.mode == "full":
            raise RuntimeError(
                "the F2 replay needs the finds of the search stage, which runs before this one in "
                "a full run, and it left none behind (not even an empty list)"
            )
        return [], "the search stage did not run in this invocation, so there is nothing to replay"
    finds = list(shared)[: sc.counts.f2_replays]
    if not finds:
        return [], "the search stage ran and found no violation to replay"
    return finds, f"{len(finds)} finds of the search stage"


def seeds_for(sc: StageContext, n: int) -> list[int]:
    """The F2 seeds: the ones a full run drew from the held-out guard (``HeldoutGuard.f2_seeds``,
    so the ledger counts them, 0003 A16) and carried on the context; development seeds in a quick
    run. A full run that carries none is an error, never a quiet derivation of its own."""
    if sc.published_f2_seeds is not None:
        if len(sc.published_f2_seeds) < n:
            raise ValueError(
                f"{n} F2 seeds are needed and the run drew {len(sc.published_f2_seeds)}"
            )
        return list(sc.published_f2_seeds[:n])
    if sc.prereg_sha is None:
        return list(range(QUICK_FIRST_DEV_SEED, QUICK_FIRST_DEV_SEED + n))
    raise ValueError("a full run's F2 seeds come from the held-out guard, not from the hash (A16)")


def _where(sc: StageContext, n: int) -> str:
    role = "development" if sc.prereg_sha is None else "f2"
    return f"{n} F2 episodes per arm, nominal condition, {role} seeds"


def _arm_records(
    arm: str, runs: Sequence[_Episode], tried: int, cause: str, where: str
) -> list[ResultRecord]:
    records: list[ResultRecord] = []
    done = len(runs)
    records.append(
        ResultRecord(
            name=claim_name("C-TOGETHER", "arm", arm),
            value=float(done),
            n=tried,
            fidelity=Fidelity.F2,
            measured_on=f"{arm}: episodes that ran end to end, {where}",
            script=SCRIPT,
            extra={"completed": done, "tried": tried, "cause": cause},
        )
    )
    if not runs:
        return records
    res = [r.result for r in runs]
    logs = [r.log for r in runs]
    on = f"{arm}, {where}"
    for name, values, hi in (
        (f"f2.{arm}.throughput", [r.throughput for r in res], 1.0),
        (f"f2.{arm}.cost_e", [min(r.cost, 200.0) / 200.0 for r in res], 1.0),
    ):
        records.append(value_record(name, values, hi=hi, fidelity=Fidelity.F2, measured_on=on))
    records.append(
        value_record(
            f"f2.{arm}.unplanned_failures",
            [float(r.unplanned_failures) for r in res],
            hi=4.0,
            fidelity=Fidelity.F2,
            measured_on=on,
        )
    )
    calls = [float(log.run_asset_calls) for log in logs]
    records.append(
        value_record(
            f"f2.{arm}.run_asset_calls",
            calls,
            hi=max(20.0, max(calls)),
            fidelity=Fidelity.F2,
            measured_on=on,
            extra={
                "equal_to_caching_rule": sum(
                    1 for log in logs if log.run_asset_calls == log.expected_run_asset_calls
                ),
            },
        )
    )
    named = sum(log.repair_drafts for log in logs)
    inspections = sum(log.inspection_drafts for log in logs)
    undrafted = sum(log.runs_without_a_draft for log in logs)
    for kind, counts in (
        ("signed", [log.wrong_compliant_signed for log in logs]),
        ("presented", [log.wrong_compliant_presented for log in logs]),
    ):
        records.append(
            ResultRecord(
                name=f"f2.{arm}.wrong_compliant_{kind}",
                value=float(sum(counts)),
                n=len(logs),
                fidelity=Fidelity.F2,
                measured_on=(
                    f"{on}: wrong-mode drafts that passed all sixteen rules and "
                    + ("reached a signature" if kind == "signed" else "were put before a human")
                    + "; the drafts that named a mode are the denominator in extra"
                ),
                script=SCRIPT,
                extra={
                    "k": sum(counts),
                    "drafts": named + inspections + undrafted,
                    "repair_drafts": named,
                    "inspection_first_drafts": inspections,
                    "alerts_without_a_draft": undrafted,
                },
            )
        )
    dropped = [log.dropped_pieces for log in logs]
    records.append(
        ResultRecord(
            name=f"f2.dropped_pieces.{arm}",
            value=float(sum(dropped)),
            n=len(logs),
            fidelity=Fidelity.F2,
            measured_on=(
                f"{on}: pieces of an asset's telemetry dropped (value: in all episodes; k: the "
                "episodes with any). The asset restarted past the failure day fixed at its "
                "life's first start, which assumes it never stops again, so the pipeline sees no "
                "more rows of that life and the repair record goes to the next piece "
                "(f2/telemetry.py, history)"
            ),
            script=SCRIPT,
            extra={"k": sum(1 for d in dropped if d), "total": sum(dropped)},
        )
    )
    return records


def run(sc: StageContext) -> StageOutput:
    """Run the F2 arms, the replays and the estimator mismatch; see the module docstring."""
    started = time.perf_counter()
    try:
        require_siblings()
    except F2Unavailable as exc:
        raise NotBuiltError(f"F2 needs the three sibling repositories: {exc}") from exc
    n = sc.counts.f2_episodes
    seeds = seeds_for(sc, n)
    where = _where(sc, n)
    records: list[ResultRecord] = []
    notes: list[str] = []
    siblings = _siblings_hash(sc.root)
    # one worker in this process: the siblings are imported once, and the runner gives each episode
    # a time limit, counts a failure instead of dying of it, and caches what finished
    local = sc.runner.in_process(timeout_s=sc.counts.episode_timeout_s)
    for name in F2_ARMS:
        run = local.run_seeds(
            F2EpisodeTask(name, siblings), seeds, batch=F2_BATCH, n_max=len(seeds)
        )
        episodes: list[F2Slim] = list(run.values)
        cause = ""
        if run.failures:  # an integration failure is a result: record where it broke
            first = run.failures[0]
            cause = f"seed {first.seed}: {first.kind}: {first.detail}"[:300]
            sc.log(f"[f2 {name}] {len(run.failures)} episodes FAILED, the first: {cause}")
        sc.log(
            f"[f2 {name}] {len(episodes)} of {len(seeds)} episodes "
            f"({run.computed} run, {run.cached} cached, {run.seconds:.0f} s)"
        )
        records.extend(_arm_records(name, episodes, len(seeds), cause, where))
        notes.append(f"{name}: {len(episodes)} of {len(seeds)} episodes completed")
        if episodes:  # the clock is the run log's: two runs of the same code do not share it
            wall = [e.log.seconds["total"] for e in episodes]
            sc.log(
                f"[f2 {name}] seconds per episode: mean {sum(wall) / len(wall):.1f}, "
                f"max {max(wall):.1f} (budget {EPISODE_BUDGET_S:g})"
            )

    # ---- replay of F1 finds: the search stage's finds, and nothing found by this stage
    want = sc.counts.f2_replays
    finds, source = replay_set(sc)
    outcomes: list[ReplayOutcome] = []
    if finds:
        replay_runner = sc.runner.in_process(
            timeout_s=REPLAY_TIME_FACTOR * sc.counts.episode_timeout_s
        )
        run = replay_runner.run_seeds(
            ReplayTask(tuple(finds), siblings),
            list(range(len(finds))),
            batch=F2_BATCH,
            n_max=len(finds),
            standin=lambda failure: failed_replay(finds[failure.seed], failure),
        )
        outcomes = list(run.values)
        for i, outcome in zip(run.seeds, outcomes, strict=True):
            sc.log(f"[f2 replay {i}] {outcome.reason}: {outcome.evidence}")
    notes.append(f"replay: {source}; {len(outcomes)} replayed ({want} at most)")
    k = sum(1 for o in outcomes if o.confirmed)
    where_r = (
        f"{len(outcomes)} first violating evaluations of the search runs (in run order across "
        "regimes and algorithms) replayed at F2 on the same plant seed; " + source
    )
    records.append(
        ResultRecord(
            name=claim_name("C-TOGETHER", "confirmed"),
            value=float(k),
            n=len(outcomes),
            fidelity=Fidelity.F2,
            measured_on=where_r,
            script=SCRIPT,
            extra={"k": k, "requested": want},
        )
    )
    reasons = Counter(o.reason for o in outcomes if not o.confirmed)
    for code in (*REASON_CODES, *FAILURE_CODES):
        records.append(
            ResultRecord(
                name=f"f2.replay.reason.{code}",
                value=float(reasons.get(code, 0)),
                n=len(outcomes),
                fidelity=Fidelity.F2,
                measured_on=where_r,
                script=SCRIPT,
                extra={"k": reasons.get(code, 0)},
            )
        )

    # ---- estimator mismatch, F1 against F2, 20 development lives
    sc.runner.check_headroom("f2 estimator mismatch")
    rows = estimator_mismatch(dev_lives(ESTIMATOR_LIVES))
    records.append(mismatch_record(rows))
    notes.append(f"estimator mismatch: {len(rows)} lives")
    sc.log(f"[f2] the stage took {time.perf_counter() - started:.0f} s")
    status = (
        "done"
        if all(
            r.extra.get("completed") == r.extra.get("tried")
            for r in records
            if r.name.startswith(claim_name("C-TOGETHER", "arm") + "/")
        )
        else "partial"
    )
    return StageOutput("f2", status, tuple(records), tuple(notes), time.perf_counter() - started)
