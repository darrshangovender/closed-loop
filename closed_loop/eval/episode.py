"""The F1 episode loop: one arm, one seed, one scenario, wired end to end (packet P12a).

``run_episode`` builds the SYNTHETIC plant, the streaming estimator, two interval providers (the
scheduling interval and the shield's set), the diagnoser, the coverage monitors, the arm's proposer
and, if the arm has one, the shield or the hand-written interlock; it steps the plant for 90 days,
then scores the trace against the Rev 3 specification and returns an ``EpisodeResult``.

The tiers are kept apart
------------------------
The controller side (the proposer, the shield, the interlock, the estimator, the providers, the
monitors) receives ``Observation`` objects and nothing else. Truth is read in two kinds of place
only:

* **after the episode**, from the ``Trace`` and the plant's counters, to score it;
* **by the oracle arms**, which are handed the truth by definition: the true cause and mode of
  the oracle diagnoser (``_OracleLookup``) and the realised RUL of the oracle interval providers
  (``_oracle_truth``). ``tests/eval/test_episode_tiers.py`` pins that nothing else in this module
  reads ``plant.truth``.

The life id is what the maintenance system sees: ``view.jobs_done``, the count of completed work
orders (0003 A18). It used to be the plant's own life id, which a repair of the wrong mode does not
move (0003 A3), so perception and the proposers disagreed after a wrong repair and a shield-stopped
asset could be left stuck. Now they agree: a completed job is a renewal, the estimator starts a new
life (``_Perception.renewed``, in the quarter the count moves, so that the shield is not given an
old life's set for a day), and a wrong repair resets the estimator on a part that is still
failing. That is a real way for C1 to break and it is measured, not hidden.

The estimator runs on the operating clock (0003 A18): its day for an asset is the quarters the asset
has run, over ``DAY_Q``. A stopped asset does not age and takes no new reading, so its output is
reused until it runs again. Checkpoints keep the calendar day.

Labels. A life ends in an observed failure or in a replacement; lives still running at the end of
the episode are censored at the horizon. A checkpoint's label is resolved on the **operating clock**
(quarters the component actually ran): a component that stopped for a job and went on, as after a
wrong-mode repair, does not age while it is stopped, so calendar time between the checkpoint and the
end would overstate how long the component survived. For a replaced life the end is the last time it
ran, not the completion of the job (``resolve_labels`` documents why); the label reaches the
providers and the monitors on the day the loop learns of it. Providers get ``learner_view`` copies,
except the oracle variants V7 and V7L, which are handed the truth by definition.

Sim-only truth is put on the checkpoints after the episode, from the trace, for scoring. The row
convention (``plant.F1Plant``): row ``r`` of the trace is the state at the end of step ``r`` and its
truth columns are read at the start of step ``r + 1``; a checkpoint made at the start of decision
quarter ``q`` therefore reads row ``q - 1`` (row 0 plus one quarter for ``q = 0``).
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from closed_loop.causal.diagnoser import OracleDiagnoser
from closed_loop.core.timegrid import DAY_Q
from closed_loop.core.trace import Trace
from closed_loop.core.types import Command, Fidelity, Op, PlanCmd
from closed_loop.eval.arms import Arm
from closed_loop.eval.context import PreparedContext, current_context, shield_kit, use_context
from closed_loop.eval.interlock import InterlockedProposer, InterlockEpisode, ViolationLag
from closed_loop.eval.metrics import (
    Attribution,
    EpisodeCoverage,
    assumption_breaks,
    attribute,
    cost_of,
    coverage_of,
    crosscheck,
    crusher_work_rows,
    late_gas_tests,
    learner_counts,
    spec_report,
)
from closed_loop.eval.tuned import current as current_tuned
from closed_loop.perception.conformal import ALPHA_LIFE, IntervalProvider, is_audit_life
from closed_loop.perception.contract import (
    ContractCounts,
    contract_counts,
    health_set,
    make_shield_provider,
    nominal_miss_level,
    shield_set_alpha,
)
from closed_loop.perception.estimator import EstimatorOutput, TheilSenEstimator
from closed_loop.perception.labels import Checkpoint, Label, learner_view, resolve_labels
from closed_loop.perception.monitor import PROXY_ALPHA, CoverageMonitor
from closed_loop.plant import make_plant
from closed_loop.plant.params import ASSET_IDS, CONVEYOR, CRUSHER, PUMPS
from closed_loop.plant.plant import AlertTruth, F1Plant, Observation
from closed_loop.plant.scenario import Scenario
from closed_loop.proposers.base import Proposer
from closed_loop.proposers.restart import Restarting
from closed_loop.shield.runtime import ShieldedProposer
from closed_loop.shield.table import ShieldTable

__all__ = [
    "AUDIT_LOWER_K",
    "SCHEDULING_ALPHA",
    "EpisodeRecord",
    "EpisodeResult",
    "run_episode",
    "scenario_for",
    "simulate_episode",
    "summary",
]

SCHEDULING_ALPHA: Final[float] = 0.10  # scheduling interval: two-sided 0.90 (prereg section 2)
AUDIT_LOWER_K: Final[float] = 1.0  # days: an audit life is run until its lower bound is below this
ORACLE_VARIANTS: Final[frozenset[str]] = frozenset({"V7", "V7L"})
LABEL_MODES: Final[tuple[str, ...]] = ("opt", "pess", "inf")


def scenario_for(condition: str, regime: str = "in") -> Scenario:
    """The nominal or shifted scenario without an adversary vector."""
    if condition not in ("nominal", "shifted"):
        raise ValueError(f"condition must be 'nominal' or 'shifted', got {condition!r}")
    return Scenario(theta=None, condition=condition, regime=regime)


# ----------------------------------------------------------------------------- the result


@dataclass(frozen=True)
class EpisodeResult:
    """One episode, as frozen in 0002 3.13, plus what 0003 A4/A5 and the metrics need.

    ``violations`` counts the *spec's* violation events (maximal runs of negative robustness of the
    inner formula) for P1 to P6; ``plant_violations`` counts the plant's own edge-triggered events,
    the cross-check of M-VIOL, and ``disagreements`` names the properties on which the two differ.
    ``cost`` is ``cost_e`` of prereg 5.1 before the 200 cap. ``labels`` and ``checkpoints`` are
    those of the scheduling interval, with the sim-only truth filled in for scoring.
    """

    arm: str
    seed: int
    condition: str
    fidelity: Fidelity
    violations: Mapping[str, int]
    min_robustness: Mapping[str, float]
    throughput: float
    cost: float
    unplanned_failures: int
    planned_jobs: int
    shield_overrides: int
    shield_admitted: int
    monitor_fired_day: float | None
    labels: tuple[Label, ...]
    wall_s: float
    # ---- added by P12a; every one has a default
    plant_violations: Mapping[str, int] = field(default_factory=dict)
    plant_events: tuple[tuple[str, str, int], ...] = ()  # (property, asset, row), the plant's own
    disagreements: Mapping[str, str] = field(default_factory=dict)
    attributions: tuple[Attribution, ...] = ()
    checkpoints: tuple[Checkpoint, ...] = ()
    coverage: EpisodeCoverage = EpisodeCoverage(None, None, 0, None, None, 0, 0, 0)
    contract: ContractCounts = ContractCounts()
    contract_source: str = ""
    monitors: Mapping[str, float | None] = field(default_factory=dict)  # mode -> day it fired
    monitor_mode: str = ""
    fallback_row: int | None = None  # first trace row the shield was given the whole range
    shift_day: float | None = None
    interlock_refused: int = 0
    interlock_forced: int = 0
    interlock_admitted: int = 0
    interlock_by_rule: Mapping[str, int] = field(default_factory=dict)
    online_lags: tuple[ViolationLag, ...] = ()
    commands_proposed: int = 0
    allowed_crusher: tuple[int, int] = (0, 0)  # (sum of allowed counts, decisions)
    allowed_pumps: tuple[int, int] = (0, 0)
    uncovered: int = 0
    crusher_stopped: int = 0  # shield decisions taken with the crusher stopped
    crusher_restart_allowed: int = 0  # ... at which the table would have allowed a START
    lost_quarters: float = 0.0
    wrong_mode_jobs: int = 0
    # ---- added by P12f (0003 A18, F5): secondary rows from the trace, no decision rule
    crusher_work_q: int = 0  # quarters CR-501 stands in WORK
    crusher_jobs_done: int = 0  # crusher work orders completed (the maintenance system's count)
    late_gas_tests: int = 0  # see ``metrics.late_gas_tests``
    inspections: int = 0
    orders: int = 0
    provisional: tuple[str, ...] = ()
    tuning: str = ""
    # ---- added by P12b; both survive the runner's slimming (it drops ``labels``, ``checkpoints``
    # and ``plant_events``, the bulky records, and keeps what the metrics need)
    learner: Mapping[str, int] = field(default_factory=dict)  # see metrics.learner_counts
    failure: str = ""  # non-empty: the episode failed or timed out; this is its stand-in


@dataclass
class EpisodeRecord:
    """The result with what produced it, for tests and for metrics that need the trace."""

    result: EpisodeResult
    trace: Trace
    plant: F1Plant
    shield_checkpoints: tuple[Checkpoint, ...]
    shield_labels: tuple[Label, ...]


def summary(r: EpisodeResult) -> dict[str, Any]:
    """The pinned part of a result: everything but the wall time and the bulky records."""
    return {
        "arm": r.arm,
        "seed": r.seed,
        "condition": r.condition,
        "fidelity": r.fidelity.value,
        "violations": dict(r.violations),
        "plant_violations": dict(r.plant_violations),
        "disagreements": dict(r.disagreements),
        "min_robustness": {k: float(v) for k, v in r.min_robustness.items()},
        "attributions": [[a.prop, a.row, a.label, a.broken_row] for a in r.attributions],
        "throughput": float(r.throughput),
        "cost": float(r.cost),
        "planned_jobs": r.planned_jobs,
        "unplanned_failures": r.unplanned_failures,
        "shield_overrides": r.shield_overrides,
        "shield_admitted": r.shield_admitted,
        "monitors": dict(r.monitors),
        "n_labels": len(r.labels),
        "n_checkpoints": len(r.checkpoints),
        "coverage": dataclasses.asdict(r.coverage),
        "contract": dataclasses.asdict(r.contract),
        "commands_proposed": r.commands_proposed,
        "allowed_crusher": list(r.allowed_crusher),
        "allowed_pumps": list(r.allowed_pumps),
        "crusher_stopped": r.crusher_stopped,
        "crusher_restart_allowed": r.crusher_restart_allowed,
    }


# ----------------------------------------------------------------------------- the simulation side


def _oracle_truth(plant: F1Plant, aid: str) -> float:
    """Realised RUL of one asset now, for the oracle providers V7 and V7L (sim only, by design)."""
    return float(plant.truth.true_rul_days[aid])


class _OracleLookup:
    """Truth for the oracle diagnoser (arm P1-oracle only): the life's true mode and the cause of
    the latest alert of the current life, else ``degradation``. A perfect diagnoser knows both."""

    def __init__(self, plant: F1Plant) -> None:
        self._plant = plant
        self._alerts: dict[str, list[AlertTruth]] = {aid: [] for aid in ASSET_IDS}

    def add(self, alert_id: str) -> None:
        a = self._plant.alert_truth(alert_id)
        self._alerts[a.asset_id].append(a)

    def __call__(self, asset_id: str, day: int) -> tuple[str, str]:
        truth = self._plant.truth
        life = truth.life_id[asset_id]
        rows = [a for a in self._alerts[asset_id] if a.life_id == life and a.day <= day]
        return str(truth.life_mode[asset_id]), (rows[-1].cause if rows else "degradation")


def _bind_oracle(proposer: Any, lookup: _OracleLookup) -> bool:
    """Bind the lookup to the oracle diagnoser inside a (wrapped) proposer, if there is one."""
    p = proposer
    while p is not None:
        d = getattr(p, "diagnoser", None)
        if isinstance(d, OracleDiagnoser):
            d.bind(lookup)
            return True
        p = getattr(p, "inner", None)
    return False


# ------------------------------------------------------------------ controller-side wrappers


class _Counting:
    """Counts the commands the inner proposer issues (the denominator of the override rate)."""

    def __init__(self, inner: Proposer) -> None:
        self.inner = inner
        self.name = inner.name
        self.proposed = 0

    def reset(self, seed: int) -> None:
        self.inner.reset(seed)
        self.proposed = 0

    def propose(self, obs: Observation) -> list[Command]:
        out = self.inner.propose(obs)
        self.proposed += len(out)
        return out


class _AuditHold:
    """V4's audit lives are run to their lower bound (prereg section 4, 0001 section 6.3).

    A work order the inner proposer raises for an asset whose current life is an audit life is held
    back until the perception layer's lower bound is below ``lower_k`` days, or the asset has
    failed. Which lives are audit lives is the maintenance system's randomised schedule
    (``is_audit_life``), told to this wrapper by the loop; the inner proposer does not know.
    """

    def __init__(
        self, inner: Proposer, is_audit: Callable[[str], bool], lower_k: float = AUDIT_LOWER_K
    ) -> None:
        self.inner = inner
        self.name = inner.name
        self._is_audit = is_audit
        self._lower_k = lower_k
        self.held = 0

    def reset(self, seed: int) -> None:
        self.inner.reset(seed)
        self.held = 0

    def propose(self, obs: Observation) -> list[Command]:
        out: list[Command] = []
        for c in self.inner.propose(obs):
            if c.cmd is PlanCmd.RAISE_WO and self._is_audit(c.asset_id):
                view = obs.assets[c.asset_id]
                if (
                    view.op is not Op.FAIL
                    and view.rul_lo is not None
                    and view.rul_lo >= self._lower_k
                ):
                    self.held += 1
                    continue
            out.append(c)
        return out


class _RecordingTable:
    """A shield table that counts how many commands it allows at each entry it is asked about."""

    def __init__(self, table: ShieldTable) -> None:
        self._table = table
        self.allowed_sum = 0
        self.decisions = 0
        self.stopped = 0  # decisions with the crusher stopped (crusher table only)
        self.restart_allowed = 0  # ... of which the table would have allowed a START

    def allowed(self, o: Hashable, health_set: Sequence[int]) -> frozenset[str]:
        out = self._table.allowed(o, health_set)
        self.allowed_sum += len(out)
        self.decisions += 1
        if isinstance(o, tuple) and o and o[0] == "STOP":  # a stopped crusher (the pump key is not)
            self.stopped += 1
            self.restart_allowed += "START" in out
        return out

    def __getattr__(self, name: str) -> Any:
        return getattr(self._table, name)


# ----------------------------------------------------------------------------- operating clock


class _OperatingClock:
    """Quarters each asset has run, and when it last stopped or failed, from the views alone.

    ``u[aid]`` is the number of decision quarters ``j`` in ``1..q`` whose view shows the asset
    running: the asset aged during the step before. A stop is a running view followed by another
    state; ``stop_u`` is ``u`` then. A failure (running, then ``FAIL``) is a life that ended at
    ``fail_u = u + 1``: the quarter it failed in ran.
    """

    def __init__(self) -> None:
        self.u: dict[str, int] = dict.fromkeys(ASSET_IDS, 0)
        self.stop_u: dict[str, int | None] = dict.fromkeys(ASSET_IDS)
        self.fail_u: dict[str, int | None] = dict.fromkeys(ASSET_IDS)
        self.fail_q: dict[str, int | None] = dict.fromkeys(ASSET_IDS)
        self.failure_pending: dict[str, bool] = dict.fromkeys(ASSET_IDS, False)
        self._assets: object = None
        self._ops: dict[str, Op] = {}
        self._running: tuple[str, ...] = ()

    def observe(self, obs: Observation) -> None:
        if obs.assets is not self._assets:
            self._assets = obs.assets
            new = {aid: v.op for aid, v in obs.assets.items()}
            for aid, op in new.items():
                before = self._ops.get(aid, Op.RUN)
                if before is Op.RUN and op is not Op.RUN:
                    self.stop_u[aid] = self.u[aid]
                    if op is Op.FAIL:
                        self.fail_u[aid] = self.u[aid] + 1
                        self.fail_q[aid] = obs.quarter
                        self.failure_pending[aid] = True
            self._ops = new
            self._running = tuple(aid for aid, op in new.items() if op is Op.RUN)
        if obs.quarter > 0:
            for aid in self._running:
                self.u[aid] += 1


# ----------------------------------------------------------------------------- perception


@dataclass
class _Stream:
    """One interval stream: a provider, its level, the checkpoints of the open lives, the labels."""

    provider: IntervalProvider
    level: float
    open: dict[str, list[tuple[Checkpoint, int]]]
    checkpoints: list[Checkpoint] = field(default_factory=list)
    labels: list[Label] = field(default_factory=list)


class _Perception:
    """The daily perception layer of one episode: estimator, providers, labels, monitors.

    Once a day (the first decision quarter of the day) it updates the estimator, asks the
    scheduling provider for the 90% interval and the shield provider for the set, hands both to
    the plant, records the checkpoints and feeds the passive monitors. When a life ends it resolves
    the labels and delivers them, in time order, to the provider and the monitors of each stream.
    """

    def __init__(self, arm: Arm, ctx: PreparedContext, seed: int, plant: F1Plant) -> None:
        self.arm = arm
        self.seed = seed
        self.plant = plant
        self.estimator = TheilSenEstimator(ctx.params)
        self.clock = _OperatingClock()
        variant = arm.perception
        if variant == "V6":
            alpha_s = ALPHA_LIFE
        else:
            alpha_s = SCHEDULING_ALPHA
        self.oracle = variant in ORACLE_VARIANTS
        self.sched = _Stream(
            ctx.dev.provider(variant, alpha=alpha_s, seed=seed),
            1.0 - alpha_s,
            {aid: [] for aid in ASSET_IDS},
        )
        if arm.shield_set == "step99" and variant == "V6":
            raise ValueError(
                f"{arm.name}: a per-step 99% set needs a per-step variant, not the per-life band V6"
            )
        self.shield = _Stream(
            make_shield_provider(arm.shield_set, variant=variant, development=ctx.dev, seed=seed),
            1.0 - shield_set_alpha(arm.shield_set),
            {aid: [] for aid in ASSET_IDS},
        )
        p0 = nominal_miss_level(arm.shield_set)
        self.monitors: dict[str, CoverageMonitor] = {m: CoverageMonitor(m, p0) for m in LABEL_MODES}
        self.monitors["proxy"] = CoverageMonitor.proxy(ctx.proxy_band)
        assert abs(PROXY_ALPHA - ctx.proxy_band.alpha) < 1e-12
        if variant == "V4":
            self.monitors["audit"] = CoverageMonitor("opt", p0)  # audit lives' labels only
        self.cur: dict[str, int | None] = dict.fromkeys(ASSET_IDS)
        self.ended: dict[str, bool] = dict.fromkeys(ASSET_IDS, False)
        self.point: dict[str, float] = {}
        # the last estimator output of each asset with the operating day and life it was made for
        self._last: dict[str, tuple[int, int, EstimatorOutput]] = {}
        self._life_seen: dict[str, int] = dict.fromkeys(ASSET_IDS, 0)  # jobs_done last looked at
        self.estimator_updates = 0  # calls to ``estimator.update`` (a test reads it)
        self.fired_day: dict[str, float | None] = dict.fromkeys(self.monitors)

    # ------------------------------------------------------------------ queries
    def point_rul(self, view: Any) -> float | None:
        return self.point.get(view.asset_id)

    def is_audit_now(self, aid: str) -> bool:
        life = self.cur[aid]
        return life is not None and is_audit_life(self.seed, aid, life)

    # ------------------------------------------------------------------ the daily update
    def operating_day(self, aid: str) -> int:
        """The estimator's day for ``aid``: whole days the asset has run (0003 A18). For an asset
        that never stops this is the calendar day at the first quarter of each day."""
        return self.clock.u[aid] // DAY_Q

    def _estimate(
        self, aid: str, hi: float | None, life: int, op_day: int
    ) -> tuple[EstimatorOutput, bool]:
        """The estimator's output for ``aid`` and whether it is new. If the operating day has not
        moved and the life has not changed, the last output is reused: a stopped asset does not age
        and gives no new reading, and the estimator takes one update a day."""
        last = self._last.get(aid)
        if last is not None and last[0] == op_day and last[1] == life:
            return last[2], False
        out = self.estimator.update(aid, op_day, hi, life)
        self.estimator_updates += 1
        self._last[aid] = (op_day, life, out)
        return out, True

    def _hand_to_plant(self, aid: str, out: EstimatorOutput) -> tuple[Any, Any]:
        iv_s = self.sched.provider.predict(out, self.sched.level)
        iv_h = self.shield.provider.predict(out, self.shield.level)
        lo, hi = health_set(iv_h)
        self.plant.set_perception(
            aid, set_lo=lo, set_hi=hi, rul_lo=iv_s.lo, rul_hi=iv_s.hi, prior_only=iv_s.prior
        )
        self.point[aid] = out.m_days
        return iv_s, iv_h

    def daily(self, obs: Observation) -> None:
        day = obs.day
        for aid in ASSET_IDS:
            view = obs.assets[aid]
            life = view.jobs_done
            self._life_seen[aid] = life
            self._advance(aid, life, day)
            op_day = self.operating_day(aid)
            out, fresh = self._estimate(aid, view.hi_last, life, op_day)
            iv_s, iv_h = self._hand_to_plant(aid, out)
            if not self.ended[aid]:
                self._checkpoint(self.sched, aid, life, day, out, iv_s.lo, iv_s.hi)
                self._checkpoint(self.shield, aid, life, day, out, iv_h.lo, iv_h.hi)
                if fresh:
                    self.monitors["proxy"].observe_day(
                        aid, op_day, view.hi_last, self.estimator.trend(aid), life
                    )
        self._note_firings(float(day))

    def renewed(self, obs: Observation) -> None:
        """In the quarters when ``daily`` does not run: an asset whose ``jobs_done`` moved gets a
        new life at once, as a proposer takes a completed job for a renewal. The estimator starts
        the new part (no reading yet), both providers predict and the plant is told, so that a
        shielded arm is not given the old part's set for the rest of the day. Labels and
        checkpoints are not touched: the next ``daily`` closes the old life."""
        for aid in ASSET_IDS:
            life = obs.assets[aid].jobs_done
            if life == self._life_seen[aid]:
                continue
            self._life_seen[aid] = life
            op_day = self.operating_day(aid)
            out = self.estimator.update(aid, op_day, None, life)
            self.estimator_updates += 1
            self._last[aid] = (op_day, life, out)
            self._hand_to_plant(aid, out)

    def _checkpoint(
        self,
        stream: _Stream,
        aid: str,
        life: int,
        day: int,
        out: EstimatorOutput,
        lo: float,
        hi: float,
    ) -> None:
        truth = _oracle_truth(self.plant, aid) if self.oracle else None
        cp = Checkpoint(
            life, aid, day, lo, out.m_days, hi, out.sigma_days, stream.level, out.status,
            true_rul=truth,
        )  # fmt: skip
        stream.open[aid].append((cp, self.clock.u[aid]))

    # ------------------------------------------------------------------ lives end
    def _advance(self, aid: str, life: int, day: int) -> None:
        clock = self.clock
        if self.cur[aid] is None:
            self.cur[aid] = life
            return
        if clock.failure_pending[aid]:
            clock.failure_pending[aid] = False
            if not self.ended[aid]:
                fq = clock.fail_q[aid]
                fu = clock.fail_u[aid]
                assert fq is not None and fu is not None
                self._close(aid, censored=False, end_u=fu, resolved_at=fq / DAY_Q, deliver=True)
                self.ended[aid] = True
        if life != self.cur[aid]:
            if not self.ended[aid]:
                su = clock.stop_u[aid]
                self._close(
                    aid,
                    censored=True,
                    end_u=clock.u[aid] if su is None else su,
                    resolved_at=float(day),
                    deliver=True,
                )
            self.cur[aid] = life
            self.ended[aid] = False

    def finish(self, horizon_days: float) -> None:
        """Censor the lives still running at the end of the episode (not delivered: it is over)."""
        for aid in ASSET_IDS:
            if self.cur[aid] is not None and not self.ended[aid]:
                self._close(
                    aid,
                    censored=True,
                    end_u=self.clock.u[aid],
                    resolved_at=horizon_days,
                    deliver=False,
                )

    def _close(
        self, aid: str, *, censored: bool, end_u: int, resolved_at: float, deliver: bool
    ) -> None:
        life = self.cur[aid]
        assert life is not None
        audit = is_audit_life(self.seed, aid, life)
        for stream in (self.sched, self.shield):
            held = stream.open[aid]
            stream.open[aid] = []
            labels: list[Label] = []
            for cp, u_i in held:
                # the operating clock, in days; integer quarters over one divisor keep the order
                # of ``end_day <= resolved_at`` exact
                end_day = (cp.day * DAY_Q + max(end_u - u_i, 0)) / DAY_Q
                labels.extend(resolve_labels([cp], end_day, censored, audit, resolved_at))
                stream.checkpoints.append(cp)
            stream.labels.extend(labels)
            if deliver and labels:
                visible = labels if self.oracle else [learner_view(lab) for lab in labels]
                stream.provider.observe(visible)
                if stream is self.shield:
                    for mode in LABEL_MODES:
                        self.monitors[mode].observe(labels)
                    if "audit" in self.monitors and audit:
                        self.monitors["audit"].observe(labels)

    # ------------------------------------------------------------------ monitors
    def _note_firings(self, day: float) -> None:
        """Record the day each monitor first fired, on the calendar (0003 A18, F2).

        The label modes stamp ``resolved_day``, which is calendar. The proxy monitor is fed the
        operating day of each asset (R1) and stamps that, which falls behind the calendar by the
        days the asset stood stopped and is not comparable with the calendar ``shift_day`` that
        ``m_detect`` and ``m_false_alarm`` read. Its firing is therefore recorded as the calendar
        ``day`` of the update that fired it; the monitor itself keeps reading operating days.
        """
        for mode, mon in self.monitors.items():
            if mon.fired and self.fired_day[mode] is None:
                stamp = mon.fired_day
                self.fired_day[mode] = (
                    float(stamp) if mode != "proxy" and stamp is not None else day
                )

    def acting_fired(self) -> bool:
        mode = self.arm.monitor_mode
        return self.arm.monitor and self.fired_day.get(mode) is not None


# ----------------------------------------------------------------------------- the episode


def _wrap(
    arm: Arm,
    ctx: PreparedContext,
    base: Proposer,
    perception: _Perception | None,
    plant: F1Plant,
) -> tuple[Proposer, dict[str, Any]]:
    """The chain proposer -> restart -> counter -> audit hold -> shield | interlock, and handles.

    The restart wrapper (0003 A18) is in arms with a shield or an interlock only: nothing else stops
    an asset except for a job. It sits under the counter, so an offer counts as a proposed command,
    and the shield or the interlock above it decides.
    """
    handles: dict[str, Any] = {}
    inner: Proposer = base
    if arm.shield or arm.interlock:
        inner = Restarting(base, pair_stop=arm.shield and arm.operator_pair)
        handles["restarting"] = inner
    counting = _Counting(inner)
    handles["counting"] = counting
    chain: Any = counting
    if perception is not None and arm.perception == "V4":
        chain = _AuditHold(chain, perception.is_audit_now)
    if arm.shield:
        kit = shield_kit(ctx, arm.spec)
        crusher, pumps = _RecordingTable(kit.crusher), _RecordingTable(kit.pumps)
        handles["crusher_table"], handles["pump_table"] = crusher, pumps
        chain = ShieldedProposer(
            chain,
            crusher,  # type: ignore[arg-type]
            pumps,  # type: ignore[arg-type]
            kit.mapper,
            conveyor=kit.conveyor,  # CV-101 is inside the proof since ADR 0003 A12.3
            crusher_id=CRUSHER,
            pump_ids=PUMPS,
            conveyor_id=CONVEYOR,
            crusher_model=kit.crusher_model,
            pumps_model=kit.pumps_model,
            conveyor_model=kit.conveyor_model,
        )
        handles["shield"] = chain
    elif arm.interlock:
        point = perception.point_rul if perception is not None else None
        chain = InterlockedProposer(chain, ctx.spec_clause, point_rul=point)
        handles["interlock"] = chain
    return chain, handles


def simulate_episode(
    arm: Arm,
    seed: int,
    scenario: Scenario,
    fidelity: Fidelity = Fidelity.F1,
    *,
    context: PreparedContext | None = None,
) -> EpisodeRecord:
    """Run one episode and score it; ``run_episode`` is this without the trace and the plant."""
    if fidelity not in (Fidelity.F1, Fidelity.F1D):
        raise ValueError(f"the F1 loop runs F1 and F1+D, not {fidelity}; F2 has its own runner")
    ctx = context if context is not None else current_context()
    if arm.perception == "none" and arm.monitor_mode == "audit":
        raise ValueError(f"{arm.name}: an audit monitor needs perception V4")
    started = time.perf_counter()
    tuned = current_tuned()
    with use_context(ctx):
        # SpecParams has the four fields SpecLike names; mypy objects to read-only attributes
        plant = make_plant(ctx.params, fidelity=fidelity, spec=ctx.spec_clause)  # type: ignore[arg-type]
        obs = plant.reset(seed, scenario)
        resolved = scenario.resolve(seed, ctx.params)
        perception = None if arm.perception == "none" else _Perception(arm, ctx, seed, plant)
        base = arm.proposer()
        active, handles = _wrap(arm, ctx, base, perception, plant)
        oracle: _OracleLookup | None = None
        if arm.diagnoser == "oracle":
            oracle = _OracleLookup(plant)
            if not _bind_oracle(active, oracle):
                raise ValueError(
                    f"{arm.name}: diagnoser 'oracle' but the proposer has none to bind"
                )
        active.reset(seed)
        override_sources = {"shield"}
        if "interlock" in handles:
            override_sources.add(handles["interlock"].name)

        plant_events: list[tuple[str, str, int]] = []
        fallback_q: int | None = None
        last_day = -1
        for step_no in range(ctx.params.horizon_q):
            if perception is not None:
                perception.clock.observe(obs)
                if obs.day != last_day:
                    last_day = obs.day
                    perception.daily(obs)
                    if perception.acting_fired():
                        plant.set_monitor_fired(True)
                else:
                    perception.renewed(obs)
            # Quarter 0 is a warm-up quarter: the first daily update lands in the plant's views
            # only from the next quarter on, and a controller that decided now would be shown the
            # default whole-range set and stop the crusher on an artefact of the start.
            commands = [] if step_no == 0 else active.propose(obs)
            overridden = [c for c in commands if c.source in override_sources]
            res = plant.step(commands, overridden)
            plant_events.extend(res.violations)
            if oracle is not None:
                for alert_id in res.obs.alerts:
                    oracle.add(alert_id)
            obs = res.obs
            if fallback_q is None and obs.monitor_fired:
                fallback_q = obs.quarter
            if res.done:
                break
        if perception is not None:
            perception.finish(float(ctx.params.horizon_days))
        trace = plant.trace()

    return _summarise(
        arm, seed, scenario, fidelity, ctx, plant, trace, perception, handles, plant_events,
        fallback_q, resolved.load_shift_day if scenario.condition == "shifted" else None,
        tuned, started, obs.assets[CRUSHER].jobs_done,
    )  # fmt: skip


def _truth_at_checkpoints(cps: Sequence[Checkpoint], trace: Trace) -> tuple[Checkpoint, ...]:
    """Fill the sim-only truth of each checkpoint from the trace (row ``q - 1``; see the module
    docstring)."""
    out: list[Checkpoint] = []
    for cp in cps:
        row = max(cp.day * DAY_Q - 1, 0)
        rul = float(trace.true_rul_days[cp.asset_id][row])
        if cp.day == 0:
            rul += 1.0 / DAY_Q  # row 0 is one quarter after the start of the episode
        out.append(
            dataclasses.replace(
                cp, true_rul=rul, health_bin=int(trace.health_bin[cp.asset_id][row])
            )
        )
    return tuple(out)


def _summarise(
    arm: Arm,
    seed: int,
    scenario: Scenario,
    fidelity: Fidelity,
    ctx: PreparedContext,
    plant: F1Plant,
    trace: Trace,
    perception: _Perception | None,
    handles: Mapping[str, Any],
    plant_events: Sequence[tuple[str, str, int]],
    fallback_q: int | None,
    shift_day: float | None,
    tuned: Any,
    started: float,
    crusher_jobs_done: int = 0,
) -> EpisodeRecord:
    report = spec_report(trace, ctx.spec_clause, CRUSHER, PUMPS)
    plant_counts: dict[str, int] = {}
    for prop, _asset, _q in plant_events:
        plant_counts[prop] = plant_counts.get(prop, 0) + 1
    breaks = assumption_breaks(trace, crusher=CRUSHER, pumps=PUMPS, fallback_row=fallback_q)
    attributions = attribute(report.events, breaks)

    stats = plant.stats()
    lost = float(np.sum(1.0 - trace.throughput.astype(np.float64)))
    cost = cost_of(
        stats.planned_jobs, stats.unplanned_failures, lost, stats.inspections, stats.wo_rejected
    )

    sched_cps: tuple[Checkpoint, ...] = ()
    shield_cps: tuple[Checkpoint, ...] = ()
    sched_labels: tuple[Label, ...] = ()
    shield_labels: tuple[Label, ...] = ()
    monitors: dict[str, float | None] = {}
    if perception is not None:
        sched_cps = _truth_at_checkpoints(perception.sched.checkpoints, trace)
        shield_cps = _truth_at_checkpoints(perception.shield.checkpoints, trace)
        sched_labels = _labels_with_truth(perception.sched.labels, sched_cps)
        shield_labels = _labels_with_truth(perception.shield.labels, shield_cps)
        monitors = dict(perception.fired_day)

    shield: ShieldedProposer | None = handles.get("shield")
    interlock: InterlockedProposer | None = handles.get("interlock")
    record: InterlockEpisode | None = interlock.episode_record(plant_events) if interlock else None
    crusher_t: _RecordingTable | None = handles.get("crusher_table")
    pump_t: _RecordingTable | None = handles.get("pump_table")
    counting: _Counting = handles["counting"]
    day_fired = monitors.get(arm.monitor_mode) if arm.monitor else None

    result = EpisodeResult(
        arm=arm.name,
        seed=seed,
        condition=scenario.condition,
        fidelity=fidelity,
        violations=report.counts,
        min_robustness=dict(report.min_robustness),
        throughput=float(np.mean(trace.throughput.astype(np.float64))),
        cost=cost,
        unplanned_failures=stats.unplanned_failures,
        planned_jobs=stats.planned_jobs,
        shield_overrides=shield.overrides if shield else 0,
        shield_admitted=shield.admitted if shield else 0,
        monitor_fired_day=day_fired,
        labels=sched_labels,
        wall_s=0.0,
        plant_violations=plant_counts,
        plant_events=tuple(plant_events),
        disagreements=crosscheck(report.min_robustness, plant_counts),
        attributions=attributions,
        checkpoints=sched_cps,
        coverage=coverage_of(sched_cps),
        contract=contract_counts(shield_cps, "all") if shield_cps else ContractCounts(),
        contract_source=arm.shield_set if perception is not None else "",
        monitors=monitors,
        monitor_mode=arm.monitor_mode if arm.monitor else "",
        fallback_row=fallback_q,
        shift_day=shift_day,
        interlock_refused=record.refused if record else 0,
        interlock_forced=record.forced if record else 0,
        interlock_admitted=record.admitted if record else 0,
        interlock_by_rule=dict(record.by_rule) if record else {},
        online_lags=record.violations if record else (),
        commands_proposed=counting.proposed,
        allowed_crusher=(crusher_t.allowed_sum, crusher_t.decisions) if crusher_t else (0, 0),
        allowed_pumps=(pump_t.allowed_sum, pump_t.decisions) if pump_t else (0, 0),
        uncovered=shield.uncovered if shield else 0,
        crusher_stopped=crusher_t.stopped if crusher_t else 0,
        crusher_restart_allowed=crusher_t.restart_allowed if crusher_t else 0,
        lost_quarters=lost,
        wrong_mode_jobs=stats.wrong_mode_jobs,
        crusher_work_q=crusher_work_rows(trace),
        crusher_jobs_done=crusher_jobs_done,
        late_gas_tests=late_gas_tests(trace, ctx.spec_clause.retest_q),
        inspections=stats.inspections,
        orders=stats.orders,
        provisional=tuned.provisional_names(arm.tuned),
        learner=learner_counts(sched_labels),
        tuning=tuned.source
        if tuned.sha256 is None
        else f"{tuned.source} sha256:{tuned.sha256[:12]}",
    )
    result = dataclasses.replace(result, wall_s=time.perf_counter() - started)
    return EpisodeRecord(result, trace, plant, shield_cps, shield_labels)


def _labels_with_truth(labels: Sequence[Label], cps: Sequence[Checkpoint]) -> tuple[Label, ...]:
    truth = {(cp.asset_id, cp.life_id, cp.day): cp.true_rul for cp in cps}
    return tuple(
        dataclasses.replace(
            lab, true_rul=truth.get((lab.asset_id, lab.life_id, lab.checkpoint_day))
        )
        for lab in labels
    )


def run_episode(
    arm: Arm,
    seed: int,
    scenario: Scenario,
    fidelity: Fidelity = Fidelity.F1,
    *,
    context: PreparedContext | None = None,
) -> EpisodeResult:
    """One episode of ``arm`` on plant seed ``seed`` (0002 3.13). ``context`` is the prepared,
    picklable fitted state; without it the process default is used, built on first use."""
    return simulate_episode(arm, seed, scenario, fidelity, context=context).result
