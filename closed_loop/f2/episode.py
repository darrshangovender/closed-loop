"""The F2 episode: the F1 plant under the real asset-health pipeline, maintenance-agent and gate
(packet P11c). Everything is SYNTHETIC.

What is real here, and what is not
----------------------------------
Real sibling code: ``asset_health`` (ingest, features, detection, the rule layer and the RUL model,
on telemetry its own generator writes from the plant's latent failure events), ``maintenance_agent``
(the ``Orchestrator``, the sixteen verifier rules, the Ed25519 ``ApprovalGate``, ``MockCMMS``) and
``plant_copilot`` (the clause specification, through the checked-in cache that an ``f2`` test
recomputes). Ours: the plant (shared with F1, so the exogenous streams are the same), the adapters
of ``closed_loop.f2``, the simulated human, and this loop.

How an F2 episode runs
----------------------
The proposer of the arm is unchanged. A ``RAISE_WO`` it issues is read as a *request*: the
controller wants a work order. The request goes to asset-health; if the rule layer has an open work
order for the asset, an alert is raised and the real orchestrator drafts a work order from it, the
sixteen rules check the draft, a simulated human signs or rejects it through the real gate, and
``PlantCMMS`` releases it. Only then does the plant get a job, made from the *draft*: an inspection
when the draft is inspection-first, a repair with the draft's top failure mode otherwise. The mode
on the job is the agent's, not the proposer's: at F2 the diagnosis is asset-health's rule layer
(0003 A3), and the proposer's own diagnoser plays no part. Everything else the proposer issues (the
micro-commands of an open job, ``INSPECT``, ``ORDER``) goes to the plant, behind the shield for a
shielded arm.

The plant's own approver is turned into a pass-through (a negligible delay and no catch): the real
gate has approved by then, and counting an approval twice would be a modelling error.

Telemetry and the estimate come from ``closed_loop.f2.telemetry`` (the caching rule of 0001
section 3 is built there and checked in ``tests/f2/test_episode.py``). The interval providers of F1
are applied to asset-health's estimate without label feedback, so what changes between F1 and F2 is
the estimator, and the estimator mismatch is the measured price (``closed_loop.f2.mismatch``).
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import partial
from typing import Any, Final

import numpy as np

from closed_loop.core.rng import rng_for
from closed_loop.core.timegrid import DAY_Q
from closed_loop.core.trace import Trace
from closed_loop.core.types import Command, Fidelity, Op, PlanCmd
from closed_loop.eval.arms import PRIMARY_VARIANT, Arm, make_bound_rule, make_planner
from closed_loop.eval.context import PreparedContext, current_context, shield_kit, use_context
from closed_loop.eval.episode import EpisodeResult
from closed_loop.eval.metrics import (
    assumption_breaks,
    attribute,
    cost_of,
    crosscheck,
    spec_report,
)
from closed_loop.f2._siblings import SIBLING_MODULES, load
from closed_loop.f2.agent import (
    build_orchestrator,
    new_mock_cmms,
    new_registry,
    register_approver,
    run_id_for,
)
from closed_loop.f2.alert_source import AssetHealthAlertSource, RunCache, aware
from closed_loop.f2.cmms import PlantCMMS, inventory_of
from closed_loop.f2.human import SimHumanApprover, on_day_shift
from closed_loop.f2.render import PLANT_START, asset_health_id
from closed_loop.f2.telemetry import F2Telemetry, FrameTransform
from closed_loop.perception.conformal import ALPHA_LIFE
from closed_loop.perception.contract import health_set, make_shield_provider, shield_set_alpha
from closed_loop.plant import make_plant
from closed_loop.plant.degradation import Physical, build_physical, mode_spec
from closed_loop.plant.params import (
    ASSET_CLASS,
    ASSET_IDS,
    CONVEYOR,
    CRUSHER,
    PUMPS,
    PlantParams,
)
from closed_loop.plant.plant import F1Plant, Observation, StepResult
from closed_loop.plant.scenario import Scenario
from closed_loop.proposers.base import Proposer
from closed_loop.proposers.restart import Restarting
from closed_loop.shield.runtime import ShieldedProposer

__all__ = [
    "F2_ARMS",
    "F2Log",
    "F2Record",
    "DraftRecord",
    "PASS_THROUGH_H",
    "SCHEDULING_ALPHA",
    "f2_params",
    "require_siblings",
    "run_f2_episode",
    "simulate_f2_episode",
]

SCHEDULING_ALPHA: Final[float] = 0.10  # as F1: a two-sided 0.90 scheduling interval
PASS_THROUGH_H: Final[float] = (
    1e-6  # the plant's own approval delay once the real gate has approved
)
# A work order asset-health already has open gets a new alert (a fresh reading, a fresh draft) when
# the last alert on it is older than the agent's own staleness limit (verifier V10: 3 days).
RE_ALERT_DAYS: Final[float] = 3.0
ISSUE_WINDOW_Q: Final[int] = DAY_Q  # how long a released order is retried against the plant
APPROVER_IDS: Final[tuple[str, str]] = ("planner1", "officer1")


def f2_params(params: PlantParams) -> PlantParams:
    """The plant of an F2 episode: ``params`` with the plant's own approver made a pass-through.

    Nothing exogenous changes (the approver streams are drawn and unused), so an F2 plant and an F1
    plant of the same seed have the same load, wear, refusals, gas and notes.
    """
    return dataclasses.replace(params, approver_median_h=PASS_THROUGH_H, approver_catch_p=0.0)


# ----------------------------------------------------------------------------- arms


def _f2_arms() -> dict[str, Arm]:
    common: dict[str, Any] = {
        "monitor": False,
        "perception": PRIMARY_VARIANT,
        "conditions": ("nominal",),
    }
    return {
        "F2-B3": Arm(
            "F2-B3", partial(make_bound_rule, None), shield=False, tuned=("k_b3",), **common
        ),
        "F2-P0": Arm(
            "F2-P0", partial(make_planner, None), shield=False, tuned=("delta_plan",), **common
        ),
        "F2-P1": Arm(
            "F2-P1", partial(make_planner, None), shield=True, tuned=("delta_plan",), **common
        ),
    }


# The three F2 arms of the pre-registration (section 4): the F1 rule arm B3, the planner without and
# with the shield, each with no diagnoser of its own, because the diagnosis is asset-health's.
F2_ARMS: Final[Mapping[str, Arm]] = _f2_arms()


# ----------------------------------------------------------------------------- the clock


class SimClock:
    """The simulation clock of the agent side: aware UTC, advanced by the loop."""

    def __init__(self) -> None:
        self.quarter = 0

    def __call__(self) -> datetime:
        return aware(PLANT_START + timedelta(minutes=15 * self.quarter))


# ----------------------------------------------------------------------------- the log


@dataclass(frozen=True)
class DraftRecord:
    """One work-order draft the agent made, with what the simulation knows of the truth."""

    asset_id: str
    alert_id: str
    day: float
    asset_health_hypothesis: str
    asset_health_severity: str
    status: str  # AWAITING_APPROVAL at the draft, then the final status of the run
    has_draft: bool  # False when the agent escalated without producing a draft
    inspection_first: bool
    named_mode: str | None  # None for an inspection-first draft
    confidence: float | None
    true_mode: str
    rules_passed: bool
    signed: bool
    released: bool
    wrong_mode: bool  # a repair draft whose mode is not the component's
    outcome: str  # the plant's side: "inspect", "repair", "refused", "none"


@dataclass(frozen=True)
class F2Log:
    """What happened on the sibling side of one episode (counters, drafts, timings)."""

    arm: str
    seed: int
    run_asset_calls: int
    expected_run_asset_calls: int
    analyse_calls: int
    requests: int
    alerts_raised: int
    no_alert: int
    no_interval: int
    repeat_ignored: int
    busy_ignored: int
    precheck_skipped: int
    drafts: tuple[DraftRecord, ...]
    runs_completed: int
    runs_rejected: int
    runs_escalated: int
    runs_open_at_end: int
    inspections_issued: int
    repairs_issued: int
    issue_failed: int
    plant_refused_mode: int
    approvals: int
    rejections: int
    wrong_compliant_presented: int
    wrong_compliant_signed: int
    held_estimates: int
    daily_estimates: int
    estimate_status: Mapping[str, int]
    latents: Mapping[
        str, Mapping[int, tuple[str, float, float]]
    ]  # asset -> life -> (mode, onset, failure)
    seconds: Mapping[str, float]
    pipeline_cache_runs: Mapping[str, int] = field(default_factory=dict)
    dropped_pieces: int = 0  # see ``telemetry.AssetTelemetry.history`` (P12g)

    @property
    def repair_drafts(self) -> int:
        """Drafts that name a failure mode and ask for a repair."""
        return sum(1 for d in self.drafts if d.has_draft and not d.inspection_first)

    @property
    def inspection_drafts(self) -> int:
        """Drafts that ask for a diagnostic inspection and name no mode."""
        return sum(1 for d in self.drafts if d.has_draft and d.inspection_first)

    @property
    def runs_without_a_draft(self) -> int:
        """Alerts on which the agent escalated without writing a draft."""
        return sum(1 for d in self.drafts if not d.has_draft)


@dataclass(frozen=True)
class F2Record:
    """An F2 episode: the ``EpisodeResult`` of 0002 3.13, the sibling-side log and the trace."""

    result: EpisodeResult
    log: F2Log
    trace: Trace


# ----------------------------------------------------------------------------- the agent chain


@dataclass
class _Run:
    asset_id: str
    alert_id: str
    run_id: str
    index: int  # position in the draft list
    shift: str | None
    status: str = "AWAITING_APPROVAL"


@dataclass
class _Due:
    draft: Any
    wo_number: str
    run_index: int
    shift: str | None
    deadline_q: int


class AgentChain:
    """The request-to-release path of one episode: alert source, orchestrator, humans, CMMS.

    ``request`` turns a proposer's ``RAISE_WO`` into an alert and a run; ``tick`` lets the humans
    look at the runs that wait for them; ``commands`` hands the plant the job a released draft
    describes. Nothing here reads truth except ``_truth_mode`` (the human's catch model and the
    wrong-mode count), which is the simulation standing in for the real state of the part.
    """

    def __init__(
        self,
        plant: F1Plant,
        cache: RunCache,
        telemetry: F2Telemetry,
        clock: SimClock,
        seed: int,
        human_params: PlantParams,
    ) -> None:
        self.plant = plant
        self.cache = cache
        self.telemetry = telemetry
        self.clock = clock
        asset_map = {aid: asset_health_id(aid) for aid in ASSET_IDS}
        self.registry = new_registry()
        keys = {
            "planner1": register_approver(self.registry, "planner1", "Planner One", ["PLANNER"]),
            "officer1": register_approver(
                self.registry, "officer1", "Officer One", ["SAFETY_OFFICER"]
            ),
        }
        self.source = AssetHealthAlertSource(cache, clock, asset_map)
        self.mock = new_mock_cmms(self.registry, clock)
        self.cmms = PlantCMMS(self.mock, inventory_of(lambda: plant.des.stock), self._queue)
        self.orch = build_orchestrator(self.source, self.cmms, self.registry, clock)
        self.humans = [
            SimHumanApprover(
                "planner1",
                keys["planner1"],
                clock,
                human_params,
                rng_for(seed, "f2.human", "planner1"),
                roles=("PLANNER",),
                truth_mode=self._truth_mode,
            ),
            SimHumanApprover(
                "officer1",
                keys["officer1"],
                clock,
                human_params,
                rng_for(seed, "f2.human", "officer1"),
                roles=("SAFETY_OFFICER",),
                truth_mode=self._truth_mode,
                on_shift=on_day_shift,
            ),
        ]
        self.inflight: dict[str, _Run] = {}
        self.due: dict[str, _Due] = {}
        self.drafts: list[DraftRecord] = []
        self._last_alert: dict[str, float] = {}
        self._pending_shift: dict[str, str | None] = {}
        self.requests = 0
        self.no_alert = 0
        self.no_interval = 0
        self.repeat_ignored = 0
        self.busy_ignored = 0
        self.precheck_skipped = 0
        self.completed = 0
        self.rejected = 0
        self.escalated = 0
        self.inspections = 0
        self.repairs = 0
        self.issue_failed = 0
        self.plant_refused_mode = 0
        self.agent_seconds = 0.0
        self._released: dict[str, _Due] = {}

    # ------------------------------------------------------------------ truth, for the human only
    def _truth_mode(self, asset_id: str) -> str:
        return str(self.plant.des.by_id[asset_id].life.mode)

    # ------------------------------------------------------------------ the CMMS callback
    def _queue(self, draft: Any, wo_number: str) -> None:
        """A work order was released: the plant gets its job (mismatch 14)."""
        aid = draft.asset_id
        run = self.inflight.get(aid)
        shift = run.shift if run is not None else None
        index = run.index if run is not None else -1
        self._released[wo_number] = _Due(
            draft, wo_number, index, shift, self.clock.quarter + ISSUE_WINDOW_Q
        )
        self.due[aid] = self._released[wo_number]

    # ------------------------------------------------------------------ requests
    def request(self, cmd: Command, obs: Observation) -> None:
        """The proposer wants a work order for ``cmd.asset_id`` (its ``RAISE_WO``)."""
        aid = cmd.asset_id
        self.requests += 1
        if aid in self.inflight or aid in self.due:
            self.busy_ignored += 1
            return
        started = time.perf_counter()
        try:
            self._request(cmd, obs)
        finally:
            self.agent_seconds += time.perf_counter() - started

    def _request(self, cmd: Command, obs: Observation) -> None:
        aid = cmd.asset_id
        now_day = obs.quarter / DAY_Q
        # an asset-health work order that already has an alert gets a fresh one only when the
        # agent's reading has gone stale (verifier V10), not on every request of the proposer
        if now_day - self._last_alert.get(aid, -1e9) < RE_ALERT_DAYS:
            self.repeat_ignored += 1
            return
        if not self.telemetry.may_alert(aid, self.clock()):
            self.no_alert += 1
            self.precheck_skipped += 1
            return
        outcome = self.source.raise_alert(aid, refresh=True)
        if outcome.kind == "none":
            self.no_alert += 1
            return
        if outcome.kind == "no_interval":
            self.no_interval += 1
            return
        assert outcome.alert_id is not None
        alert_id = outcome.alert_id
        self._last_alert[aid] = now_day
        state = self.orch.run(alert_id, run_id=run_id_for(alert_id))
        shift = cmd.arg.partition(":")[2].strip() if cmd.arg and ":" in cmd.arg else None
        record = self.source.record(alert_id)
        draft = state.draft or {}
        hyps = draft.get("hypotheses") or []
        inspection_first = bool(draft.get("inspection_first", False))
        named = None if inspection_first or not hyps else str(hyps[0]["mode"])
        index = len(self.drafts)
        self.drafts.append(
            DraftRecord(
                aid,
                alert_id,
                now_day,
                record.hypothesis,
                record.severity,
                state.status.value,
                bool(state.draft),
                inspection_first,
                named,
                None if not hyps else float(hyps[0]["confidence"]),
                self._truth_mode(aid),
                bool(state.verification is not None and state.verification.passed),
                False,
                False,
                bool(named is not None and named != self._truth_mode(aid)),
                "none",
            )
        )
        self.inflight[aid] = _Run(aid, alert_id, run_id_for(alert_id), index, shift or None)
        self._close_if_done(self.inflight[aid], state)

    # ------------------------------------------------------------------ the humans
    def tick(self, quarter: int) -> None:
        """Advance the agent-side clock and let the humans look at the runs waiting for them."""
        self.clock.quarter = quarter
        if not self.inflight:
            return
        started = time.perf_counter()
        for run in list(self.inflight.values()):
            state = None
            for human in self.humans:
                state = human.act(self.orch, run.run_id)
                if state.status.value not in ("AWAITING_APPROVAL", "BLOCKED"):
                    break
            if state is not None:
                self._close_if_done(run, state)
        self.agent_seconds += time.perf_counter() - started

    def _close_if_done(self, run: _Run, state: Any) -> None:
        status = state.status.value
        run.status = status
        if status in ("AWAITING_APPROVAL", "BLOCKED", "RUNNING"):
            return
        self.inflight.pop(run.asset_id, None)
        draft = self.drafts[run.index]
        signed = status == "COMPLETED" or state.approval is not None
        self.drafts[run.index] = dataclasses.replace(
            draft, status=status, signed=bool(signed), released=status == "COMPLETED"
        )
        if status == "COMPLETED":
            self.completed += 1
            due = self.due.get(run.asset_id)
            if due is not None:
                due.run_index = run.index
        elif status == "REJECTED":
            self.rejected += 1
        else:
            self.escalated += 1

    # ------------------------------------------------------------------ the plant's side
    def commands(self, quarter: int) -> list[Command]:
        """The job commands for released drafts, retried until the plant takes them or the window
        closes. An inspection-first draft is an ``INSPECT``; a repair draft is a ``RAISE_WO``."""
        out: list[Command] = []
        for aid, due in list(self.due.items()):
            if quarter > due.deadline_q:
                self.issue_failed += 1
                self._note_outcome(due, "refused")
                del self.due[aid]
                continue
            draft = due.draft
            if bool(draft.inspection_first):
                out.append(Command(aid, PlanCmd.INSPECT, None, "f2.agent"))
                continue
            top = draft.top_mode()
            if top is None:
                self.issue_failed += 1
                self._note_outcome(due, "refused")
                del self.due[aid]
                continue
            mode = top.value
            arg = mode if not due.shift else f"{mode}:{due.shift}"
            out.append(Command(aid, PlanCmd.RAISE_WO, arg, "f2.agent"))
        return out

    def after_step(self, result: StepResult) -> None:
        """A released order is done once the plant executed its command."""
        if not self.due:
            return
        executed = {(c.asset_id, c.cmd) for c in result.executed if c.source == "f2.agent"}
        for aid, due in list(self.due.items()):
            kind = PlanCmd.INSPECT if bool(due.draft.inspection_first) else PlanCmd.RAISE_WO
            if (aid, kind) in executed:
                if kind is PlanCmd.INSPECT:
                    self.inspections += 1
                    self._note_outcome(due, "inspect")
                else:
                    self.repairs += 1
                    self._note_outcome(due, "repair")
                del self.due[aid]
        refused = [c for c in result.refused if c.source == "f2.agent"]
        for c in refused:
            pending = self.due.get(c.asset_id)
            if pending is not None and c.cmd is PlanCmd.RAISE_WO and not self._mode_known(c):
                self.plant_refused_mode += 1
                self.issue_failed += 1
                self._note_outcome(pending, "refused")
                del self.due[c.asset_id]

    def _mode_known(self, cmd: Command) -> bool:
        mode = (cmd.arg or "").partition(":")[0]
        return mode_spec(ASSET_CLASS[cmd.asset_id], mode) is not None

    def _note_outcome(self, due: _Due, outcome: str) -> None:
        if 0 <= due.run_index < len(self.drafts):
            self.drafts[due.run_index] = dataclasses.replace(
                self.drafts[due.run_index], outcome=outcome
            )

    # ------------------------------------------------------------------ the counted metric
    def review_counts(self) -> tuple[int, int, int, int]:
        """(approvals, rejections, wrong-mode drafts that passed every rule and were presented,
        those among them that reached a signature): the counted metric of 0003 A3."""
        approvals = sum(1 for h in self.humans for r in h.log if r.kind == "approve")
        rejections = sum(1 for h in self.humans for r in h.log if r.kind == "reject")
        presented = sum(1 for d in self.drafts if d.wrong_mode and d.rules_passed)
        signed = sum(1 for h in self.humans for r in h.log if r.kind == "approve" and r.wrong_mode)
        return approvals, rejections, presented, signed


class _Interceptor:
    """Sits between the proposer and the shield: a ``RAISE_WO`` becomes a request to the agent, and
    the jobs the agent has released are added to what goes to the plant."""

    def __init__(self, inner: Proposer, chain: AgentChain) -> None:
        self.inner = inner
        self.name = inner.name
        self._chain = chain
        self.proposed = 0

    def reset(self, seed: int) -> None:
        self.inner.reset(seed)
        self.proposed = 0

    def propose(self, obs: Observation) -> list[Command]:
        out: list[Command] = []
        for cmd in self.inner.propose(obs):
            self.proposed += 1
            if cmd.cmd is PlanCmd.RAISE_WO and cmd.source != "f2.agent":
                self._chain.request(cmd, obs)
            else:
                out.append(cmd)
        out.extend(self._chain.commands(obs.quarter))
        return out


# ----------------------------------------------------------------------------- perception


class _Perception:
    """The daily perception layer of an F2 episode: asset-health's estimate through F1's
    interval providers."""

    def __init__(
        self,
        arm: Arm,
        ctx: PreparedContext,
        seed: int,
        plant: F1Plant,
        telemetry: F2Telemetry,
    ) -> None:
        variant = arm.perception
        alpha_s = ALPHA_LIFE if variant == "V6" else SCHEDULING_ALPHA
        self.plant = plant
        self.telemetry = telemetry
        self.sched = ctx.dev.provider(variant, alpha=alpha_s, seed=seed)
        self.level_s = 1.0 - alpha_s
        self.shield = make_shield_provider(
            arm.shield_set, variant=variant, development=ctx.dev, seed=seed
        )
        self.level_h = 1.0 - shield_set_alpha(arm.shield_set)

    def daily(self, obs: Observation) -> None:
        day = obs.day
        for aid in ASSET_IDS:
            view = obs.assets[aid]
            age = day - self.plant.life_start_day(aid)
            out = self.telemetry.estimate(aid, day, view, age)
            iv_s = self.sched.predict(out, self.level_s)
            iv_h = self.shield.predict(out, self.level_h)
            lo, hi = health_set(iv_h)
            self.plant.set_perception(
                aid, set_lo=lo, set_hi=hi, rul_lo=iv_s.lo, rul_hi=iv_s.hi, prior_only=iv_s.prior
            )


# ----------------------------------------------------------------------------- the episode


def require_siblings() -> None:
    """Raise ``F2Unavailable`` unless the three siblings and pandas can be imported."""
    for module in (*SIBLING_MODULES, "pandas"):
        load(module)


def _expected_runs(trace: Trace, horizon_days: int) -> int:
    """How many times the caching rule says ``run_asset`` runs, from the plant's trace alone.

    One run for each asset that is running at some daily update, plus one for each pair of
    consecutive running updates with a stop between them (the asset started again, so its history
    changed). The count does not look at the telemetry bookkeeping it is checked against.
    """
    run_idx = list(Op).index(Op.RUN)
    total = 0
    for aid in ASSET_IDS:
        not_running = (trace.op[aid] != run_idx).astype(np.int64)
        csum = np.concatenate(([0], np.cumsum(not_running)))
        updates = [d for d in range(horizon_days) if d == 0 or not_running[d * DAY_Q - 1] == 0]
        if not updates:
            continue
        total += 1
        for d1, d2 in zip(updates, updates[1:], strict=False):
            lo, hi = max(d1 * DAY_Q - 1, 0), d2 * DAY_Q  # rows d1*96-1 .. d2*96-1 inclusive
            if csum[hi] - csum[lo] > 0:
                total += 1
    return total


def simulate_f2_episode(
    arm: Arm,
    seed: int,
    scenario: Scenario,
    *,
    context: PreparedContext | None = None,
    faults: bool = True,
    transforms: Mapping[str, FrameTransform] | None = None,
    approver_catch_p: float | None = None,
) -> F2Record:
    """Run one F2 episode and return its result, the sibling-side log and the trace.

    ``faults`` writes the plant's exogenous sensor faults into the telemetry; ``transforms`` alters
    the rendered rows of an asset (a test or a replay of a sensor scenario); ``approver_catch_p``
    overrides the human's chance of catching a wrong-mode draft (otherwise the plant's, or the
    scenario's).
    """
    require_siblings()
    ctx = context if context is not None else current_context()
    started = time.perf_counter()
    seconds: dict[str, float] = {}
    base_params = ctx.params
    params = f2_params(base_params)
    resolved = scenario.resolve(seed, base_params)
    with use_context(ctx):
        plant = make_plant(params, fidelity=Fidelity.F1, spec=ctx.spec_clause)  # type: ignore[arg-type]
        obs = plant.reset(seed, scenario)
        phys: Physical = build_physical(params, seed, resolved)
        cache = RunCache()
        telemetry = F2Telemetry(
            plant, phys, seed, params, cache, faults=faults, transforms=transforms
        )
        clock = SimClock()
        human_params = dataclasses.replace(
            params,
            approver_median_h=base_params.approver_median_h * resolved.approver_delay_mult,
            approver_catch_p=(
                approver_catch_p
                if approver_catch_p is not None
                else base_params.approver_catch_p
                if resolved.approver_catch_p is None
                else resolved.approver_catch_p
            ),
        )
        chain = AgentChain(plant, cache, telemetry, clock, seed, human_params)
        perception = (
            None if arm.perception == "none" else _Perception(arm, ctx, seed, plant, telemetry)
        )
        base = arm.proposer()
        if arm.shield:  # a stopped asset is offered a restart once a day (0003 A18, R4)
            base = Restarting(base, pair_stop=arm.operator_pair)
        interceptor = _Interceptor(base, chain)
        active: Any = interceptor
        shield: ShieldedProposer | None = None
        if arm.shield:
            kit = shield_kit(ctx, arm.spec)
            shield = ShieldedProposer(
                interceptor,
                kit.crusher,
                kit.pumps,
                kit.mapper,
                conveyor=kit.conveyor,
                crusher_id=CRUSHER,
                pump_ids=PUMPS,
                conveyor_id=CONVEYOR,
                crusher_model=kit.crusher_model,
                pumps_model=kit.pumps_model,
                conveyor_model=kit.conveyor_model,
            )
            active = shield
        active.reset(seed)

        plant_events: list[tuple[str, str, int]] = []
        last_day = -1
        t_loop = time.perf_counter()
        for step_no in range(ctx.params.horizon_q):
            telemetry.observe(obs.assets, obs.quarter)
            chain.tick(obs.quarter)
            if perception is not None and obs.day != last_day:
                last_day = obs.day
                perception.daily(obs)
            # quarter 0 is the warm-up quarter of F1 as well: nobody is commanded
            commands = [] if step_no == 0 else active.propose(obs)
            overridden = [c for c in commands if c.source == "shield"]
            res = plant.step(commands, overridden)
            chain.after_step(res)
            plant_events.extend(res.violations)
            obs = res.obs
            if res.done:
                break
        seconds["loop"] = time.perf_counter() - t_loop
        trace = plant.trace()

    seconds["run_asset"] = cache.pipeline_seconds
    seconds["analyse_run"] = cache.analyse_seconds
    seconds["agent"] = chain.agent_seconds
    seconds["total"] = time.perf_counter() - started
    approvals, rejections, presented, signed = chain.review_counts()
    stats = plant.stats()
    expected = _expected_runs(trace, ctx.params.horizon_days)
    latents: dict[str, dict[int, tuple[str, float, float]]] = {
        aid: {lid: (lat.mode, lat.onset_day, lat.failure_day) for lid, lat in tel.latents.items()}
        for aid, tel in telemetry.assets.items()
    }
    log = F2Log(
        arm=arm.name,
        seed=seed,
        run_asset_calls=cache.run_asset_calls,
        expected_run_asset_calls=expected,
        analyse_calls=cache.analyse_calls,
        requests=chain.requests,
        alerts_raised=chain.source.stats.raised,
        no_alert=chain.no_alert,
        no_interval=chain.no_interval,
        repeat_ignored=chain.repeat_ignored,
        busy_ignored=chain.busy_ignored,
        precheck_skipped=chain.precheck_skipped,
        drafts=tuple(chain.drafts),
        runs_completed=chain.completed,
        runs_rejected=chain.rejected,
        runs_escalated=chain.escalated,
        runs_open_at_end=len(chain.inflight),
        inspections_issued=chain.inspections,
        repairs_issued=chain.repairs,
        issue_failed=chain.issue_failed,
        plant_refused_mode=chain.plant_refused_mode,
        approvals=approvals,
        rejections=rejections,
        wrong_compliant_presented=presented,
        wrong_compliant_signed=signed,
        held_estimates=telemetry.stats.held_estimates,
        daily_estimates=telemetry.stats.daily_estimates,
        estimate_status=dict(telemetry.stats.estimated_status),
        dropped_pieces=telemetry.stats.dropped_pieces,
        latents=latents,
        seconds=seconds,
    )
    result = _summarise(
        arm, seed, scenario, ctx, plant, trace, plant_events, shield, interceptor, stats, started
    )
    return F2Record(result, log, trace)


def _summarise(
    arm: Arm,
    seed: int,
    scenario: Scenario,
    ctx: PreparedContext,
    plant: F1Plant,
    trace: Trace,
    plant_events: Sequence[tuple[str, str, int]],
    shield: ShieldedProposer | None,
    interceptor: _Interceptor,
    stats: Any,
    started: float,
) -> EpisodeResult:
    report = spec_report(trace, ctx.spec_clause, CRUSHER, PUMPS)
    plant_counts: dict[str, int] = {}
    for prop, _asset, _q in plant_events:
        plant_counts[prop] = plant_counts.get(prop, 0) + 1
    breaks = assumption_breaks(trace, crusher=CRUSHER, pumps=PUMPS, fallback_row=None)
    lost = float(np.sum(1.0 - trace.throughput.astype(np.float64)))
    resolved = scenario.resolve(seed, ctx.params)
    return EpisodeResult(
        arm=arm.name,
        seed=seed,
        condition=scenario.condition,
        fidelity=Fidelity.F2,
        violations=report.counts,
        min_robustness=dict(report.min_robustness),
        throughput=float(np.mean(trace.throughput.astype(np.float64))),
        cost=cost_of(
            stats.planned_jobs, stats.unplanned_failures, lost, stats.inspections, stats.wo_rejected
        ),
        unplanned_failures=stats.unplanned_failures,
        planned_jobs=stats.planned_jobs,
        shield_overrides=shield.overrides if shield else 0,
        shield_admitted=shield.admitted if shield else 0,
        monitor_fired_day=None,
        labels=(),
        wall_s=time.perf_counter() - started,
        plant_violations=plant_counts,
        plant_events=tuple(plant_events),
        disagreements=crosscheck(report.min_robustness, plant_counts),
        attributions=attribute(report.events, breaks),
        shift_day=resolved.load_shift_day if scenario.condition == "shifted" else None,
        commands_proposed=interceptor.proposed,
        uncovered=shield.uncovered if shield else 0,
        lost_quarters=lost,
        wrong_mode_jobs=stats.wrong_mode_jobs,
        inspections=stats.inspections,
        orders=stats.orders,
    )


def run_f2_episode(arm: Arm, seed: int, scenario: Scenario) -> EpisodeResult:
    """One F2 episode of ``arm`` on plant seed ``seed`` (0002 3.13).

    Raises ``F2Unavailable`` if a sibling or pandas is missing. The log of the sibling side (call
    counts, drafts, the counted wrong-diagnosis metric, the split of the time) is
    ``simulate_f2_episode(...).log``.
    """
    return simulate_f2_episode(arm, seed, scenario).result
