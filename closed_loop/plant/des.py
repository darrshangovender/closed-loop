"""Discrete-event core of the SYNTHETIC F1 plant: shifts, permits, gas, crew, spares, failures.

``Des`` advances one quarter per ``step``. Within a step, commands are applied first, then the
environment moves, then the plant's own checks of the English properties of 0001 section 5 run.
The state recorded for quarter ``q`` is the state at the **end** of step ``q`` (after the commands
issued at ``q`` and the environment's move); the observation returned from that step is the one the
proposer sees when it decides at quarter ``q + 1``. The sim-only truth columns of the trace are read
at the start of quarter ``q + 1`` so that they line up with that end-of-step state.

Command semantics (the abstraction of 0001 section 4.1 is an over-approximation of these):

* ``STOP`` and ``EXIT`` are forced: never refused in regime ``in`` (assumption A_exec).
* every other command may be turned into a no-op: with probability ``p_refuse`` (crew or holder
  absent, one keyed draw per asset and quarter), when no fitter is free (``BEGIN``), when the work
  order is not released or its spare is out of stock (``BEGIN``), or because it is physically
  impossible (``START`` under a lock, ``COMPLETE`` with a person inside, ``CLOSE`` during work,
  ``UNISO`` with a permit open, ``ISSUE`` outside ``ISO`` and outside ``WORK`` with nobody inside,
  ``ENTER`` on anything but the crusher).
* commands that are possible but unsafe **execute** and the violation is recorded at that quarter,
  which is what a careless proposer produces: ``BEGIN`` before isolation is proven, ``ENTER`` with a
  stale gas test, ``START`` while work is in progress.

A permit lives at most ``permit_max_h`` from its issue (CORP-HSE-004 clause 4.1; clause 5.1 records
a handover on the permit and does not renew the limit, ADR 0003 A12.1). It records ``limit_q`` at
issue. It is ``VALID`` until the earlier of ``limit_q`` and the end of the shift that issued or
revalidated it, and then becomes ``REVAL``, which the incoming holder revalidates (the new expiry is
again the earlier of ``limit_q`` and the end of that shift, and ``limit_q`` does not move). At the
end of the tick in which ``limit_q`` is reached a permit that is not ``NONE`` becomes ``NONE`` from
``VALID`` or ``REVAL`` alike: it is spent and a new ``ISSUE`` is needed, which works in ``ISO`` and
in ``WORK`` with nobody inside. ``view.soon`` is true when the next quarter is a handover or when a
permit reaches ``limit_q`` at the next quarter. A gas test older than ``gas_retest_q`` quarters, or
a person inside under a permit that is not ``VALID``, is a violation (P3, P2). A pump pair with
neither pump in ``RUN`` or ``STOP`` is a violation (P4): a locked pump is not available (A12.2).
Properties P1-P6 are written here directly from the English text, without importing
``closed_loop.spec``.

A job (``RAISE_WO``) is approved after a log-normal delay, only on the day shift for repairs that
need a safety officer, and a wrong-mode work order that the approver does not catch runs to
completion without renewing the failing component (docs/adr/0003 A3): the cost is real and the life
continues. Everything is SYNTHETIC.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from closed_loop.core.rng import rng_for
from closed_loop.core.timegrid import DAY_Q, QUARTER_MIN, SHIFT_Q, is_handover_next, shift_of
from closed_loop.core.trace import GAS_AGE_NEVER_MIN, PERMIT_MARGIN_NONE_MIN, Trace
from closed_loop.core.types import FORCED_CMDS, N_BINS, Cmd, Command, Op, Permit, PlanCmd
from closed_loop.plant.degradation import (
    AssetClock,
    LifeDraw,
    Physical,
    first_life,
    mode_spec,
    next_life,
)
from closed_loop.plant.params import (
    ASSET_CLASS,
    ASSET_IDS,
    CONFINED_JOBS,
    CRUSHER,
    ENTERABLE_CLASSES,
    JOB_WORK_Q,
    PARTS,
    PUMPS,
    SAFETY_OFFICER_JOBS,
    PlantParams,
)
from closed_loop.plant.scenario import ResolvedScenario

NEVER_Q = -(10**9)
PUMP_PAIR_ID = "PU-201|PU-202"  # the asset id P4 is reported against
DO_REPLACE = "do:replace"  # Command.source of a do(replace) intervention (twin worlds)
_N_APPROVALS = 48
_N_INSPECTIONS = 64

_OP_IDX = {op: i for i, op in enumerate(Op)}
_PERMIT_IDX = {p: i for i, p in enumerate(Permit)}
_RUN, _STOP, _LOCKED, _ISO, _WORK, _FAIL = (
    Op.RUN,
    Op.STOP,
    Op.LOCKED,
    Op.ISO,
    Op.WORK,
    Op.FAIL,
)
_PUMP_UNAVAILABLE = frozenset({_LOCKED, _ISO, _WORK, _FAIL})  # P4: only RUN and STOP are available

# Job status names.
AWAITING, APPROVED, COMPLETED = "AWAITING_APPROVAL", "APPROVED", "COMPLETED"


class SpecLike(Protocol):
    """The four ``spec.SpecParams`` fields the plant's own checks read (spec is not imported)."""

    retest_q: int
    break_q: int
    permit_max_h: float
    alarm_exit_q: int


@dataclass(frozen=True, slots=True)
class JobView:
    """What a proposer may see of an open job. No truth fields."""

    asset_id: str
    wo_id: str
    mode: str  # the mode the work order was raised with
    status: str  # AWAITING_APPROVAL | APPROVED | COMPLETED
    ready: bool  # approved and the target shift has begun
    confined: bool  # the repair needs a confined-space entry
    work_total_q: int
    work_done_q: int
    part_no: str | None
    part_available: bool  # reserved for this job or in stock


class _Job:
    __slots__ = (
        "asset_id", "wo_id", "mode", "confined", "work_total_q", "progress_q", "part_no",
        "part_taken", "raised_q", "approve_q", "target_q", "status", "needs_officer", "completed",
        "k_draw",
    )  # fmt: skip

    def __init__(self, asset_id: str, wo_id: str, mode: str, raised_q: int):
        self.asset_id = asset_id
        self.wo_id = wo_id
        self.mode = mode
        self.confined = False
        self.work_total_q = 0
        self.progress_q = 0
        self.part_no: str | None = None
        self.part_taken = False
        self.raised_q = raised_q
        self.approve_q = 0
        self.target_q = 0
        self.status = AWAITING
        self.needs_officer = False
        self.completed = False
        self.k_draw = 0

    @property
    def work_done(self) -> bool:
        return self.progress_q >= self.work_total_q


class AssetState:
    __slots__ = (
        "idx", "aid", "cls", "enterable", "op", "permit", "occ", "last_pass_q", "exc_until",
        "alarm", "iso", "expiry_q", "job", "clock", "life", "life_id", "jobs_done", "n_wo",
        "fail_extra_q", "inspect_due_q", "inspect_count", "f1", "f2", "f3", "p6_start",
        "p6_reported", "hist", "unc_fail_q", "sets", "life_start_q", "renewals",
        "adv_gas_pending", "limit_q",
    )  # fmt: skip

    def __init__(self, idx: int, aid: str, clock: AssetClock, life: LifeDraw):
        self.idx = idx
        self.aid = aid
        self.cls = ASSET_CLASS[aid]
        self.enterable = self.cls in ENTERABLE_CLASSES
        self.op = _RUN
        self.permit = Permit.NONE
        self.occ = False
        self.last_pass_q = NEVER_Q
        self.exc_until = NEVER_Q  # a gas excursion lasts until this quarter (exclusive)
        self.alarm = False
        self.iso = False
        self.expiry_q = NEVER_Q
        self.limit_q = NEVER_Q  # the quarter at which the permit is spent (12 h from its issue)
        self.job: _Job | None = None
        self.clock = clock
        self.life = life
        self.life_id = 0
        self.jobs_done = 0
        self.n_wo = 0
        self.fail_extra_q = 0
        self.inspect_due_q = -1
        self.inspect_count = 0
        self.f1 = self.f2 = self.f3 = False
        self.p6_start = -1
        self.p6_reported = False
        # rows (q_start, op, permit, occ, alarm, iso, last_pass_q, expiry_q): state at the end of
        # step q
        self.hist: list[tuple[int, int, int, bool, bool, bool, int, int]] = []
        self.unc_fail_q: list[int] = []
        # perception rows (q_start, set_lo, set_hi): given to the shield from this row on
        self.sets: list[tuple[int, int, int]] = [(0, 0, N_BINS - 1)]
        self.life_start_q = 0
        self.renewals = 0
        self.adv_gas_pending: list[int] = []


@dataclass(frozen=True)
class DesStreams:
    """Pre-drawn keyed streams of the DES: indexed by quarter, by work order or by handover."""

    u_refuse: Mapping[str, np.ndarray]
    u_holder: Mapping[str, np.ndarray]
    u_gas: Mapping[str, np.ndarray]
    appr_z: Mapping[str, np.ndarray]
    appr_c: Mapping[str, np.ndarray]
    insp_z: Mapping[str, np.ndarray]
    notes: np.ndarray  # (n_shifts, 2): kind, pick


def build_des_streams(params: PlantParams, seed: int) -> DesStreams:
    n_q = params.horizon_q
    n_shifts = n_q // SHIFT_Q + 2
    refuse, holder, gas, az, ac, iz = {}, {}, {}, {}, {}, {}
    for aid in ASSET_IDS:
        refuse[aid] = rng_for(seed, "refuse", aid).random(n_q)
        holder[aid] = rng_for(seed, "holder", aid).random(n_q)
        gas[aid] = rng_for(seed, "gas", aid).random(n_q)
        r = rng_for(seed, "approver", aid)
        az[aid] = r.standard_normal(_N_APPROVALS)
        ac[aid] = r.random(_N_APPROVALS)
        iz[aid] = rng_for(seed, "inspect", aid).standard_normal(_N_INSPECTIONS)
    notes = rng_for(seed, "note").random((n_shifts, 2))
    return DesStreams(refuse, holder, gas, az, ac, iz, notes)


@dataclass
class DesStats:
    planned_jobs: int = 0
    unplanned_failures: int = 0
    wrong_mode_jobs: int = 0
    wo_rejected: int = 0
    inspections: int = 0
    orders: int = 0
    cost: float = 0.0


@dataclass(frozen=True)
class StepOutcome:
    executed: tuple[Command, ...]
    refused: tuple[Command, ...]
    violations: tuple[tuple[str, str, int], ...]
    touched: frozenset[int]  # asset indices whose state changed this step
    handover: bool  # the quarter just simulated was the last of a shift


_QUIET = StepOutcome((), (), (), frozenset(), False)
_QUIET_HANDOVER = StepOutcome((), (), (), frozenset(), True)


class Des:
    """The mutable state and rules of one episode. One instance per ``reset``."""

    def __init__(
        self,
        params: PlantParams,
        seed: int,
        resolved: ResolvedScenario,
        phys: Physical,
        streams: DesStreams,
        spec: SpecLike | None = None,
    ):
        self.params = params
        self.seed = seed
        self.r = resolved
        self.phys = phys
        self.st = streams
        self.retest_q = spec.retest_q if spec is not None else params.gas_retest_q
        self.break_q = spec.break_q if spec is not None else params.gas_break_q
        self.alarm_exit_q = spec.alarm_exit_q if spec is not None else params.alarm_exit_q
        permit_h = spec.permit_max_h if spec is not None else params.permit_max_h
        self.permit_max_q = max(int(permit_h * 60 / QUARTER_MIN), 1)
        self.q = 0
        self.states: list[AssetState] = []
        for i, aid in enumerate(ASSET_IDS):
            life = first_life(phys, aid)
            s = AssetState(i, aid, AssetClock(phys, aid, life), life)
            s.adv_gas_pending = list(resolved.gas_adv_q)
            self.states.append(s)
        self.by_id = {s.aid: s for s in self.states}
        self.stock: dict[str, int] = {p: spec_.stock for p, spec_ in PARTS.items()}
        self.orders: list[tuple[int, str]] = []
        self.pending: list[_Job] = []
        self.stats = DesStats()
        self.violations: list[tuple[str, str, int]] = []
        self.commands_log: list[tuple[int, Command, bool, bool]] = []
        self.throughput: list[float] = []
        self.f4 = False
        self._busy = 0  # assets in WORK plus inspections under way: fitters in use
        self._inspecting = 0
        self._tp = 1.0
        self._watch: list[AssetState] = []
        self.stock_version = 0
        self.inspection_hook: Callable[[AssetState, int, float], None] | None = None
        self._fail_q = [s.clock.fail_q() for s in self.states]
        self._next_fail = min(self._fail_q)
        self._catch_p = (
            resolved.approver_catch_p
            if resolved.approver_catch_p is not None
            else params.approver_catch_p
        )
        for s in self.states:
            self._record(s, -1)

    # ------------------------------------------------------------------ small helpers
    def _refresh_fail(self, s: AssetState) -> None:
        self._fail_q[s.idx] = s.clock.fail_q()
        self._next_fail = min(self._fail_q)

    def _set_op(self, s: AssetState, new: Op, q: int) -> None:
        old = s.op
        if old is new:
            return
        s.op = new
        if old is _WORK:
            self._busy -= 1
        elif new is _WORK:
            self._busy += 1
        if (old is _RUN) != (new is _RUN):
            s.clock.set_running(q, new is _RUN)
            self._refresh_fail(s)

    def crew_free(self) -> int:
        return max(self.params.crew_per_shift - self._busy, 0)

    def safety_officer_on_shift(self, q: int) -> bool:
        return shift_of(q) == 0

    def stock_of(self, part_no: str | None) -> int:
        return 0 if part_no is None else self.stock.get(part_no, 0)

    def _expiry(self, s: AssetState, q: int) -> int:
        """The permit is valid until the earlier of its 12 hour limit and the end of this shift."""
        return min(s.limit_q, (q // SHIFT_Q + 1) * SHIFT_Q)

    def soon(self, s: AssetState, q: int) -> bool:
        """At decision quarter ``q``: this step ends a shift, or a permit reaches its limit."""
        return is_handover_next(q) or (s.permit is not Permit.NONE and q + 1 >= s.limit_q)

    def permit_limit_left_q(self, s: AssetState, q: int) -> int | None:
        """Quarters from decision quarter ``q`` until the permit is spent; None without a permit."""
        return None if s.permit is Permit.NONE else max(s.limit_q - q, 0)

    def gas_age_q(self, s: AssetState, q: int) -> int:
        return q - s.last_pass_q if s.last_pass_q > NEVER_Q // 2 else 10**6

    def job_view(self, s: AssetState, q_next: int) -> JobView | None:
        j = s.job
        if j is None:
            return None
        return JobView(
            asset_id=s.aid,
            wo_id=j.wo_id,
            mode=j.mode,
            status=j.status,
            ready=j.status == APPROVED and q_next >= j.target_q,
            confined=j.confined,
            work_total_q=j.work_total_q,
            work_done_q=min(j.progress_q, j.work_total_q),
            part_no=j.part_no,
            part_available=j.part_taken or self.stock_of(j.part_no) > 0,
        )

    # ------------------------------------------------------------------ the step
    def step(
        self,
        commands: Sequence[Command],
        overridden: Sequence[Command] = (),
    ) -> StepOutcome:
        q = self.q
        if (
            not commands
            and not self._watch
            and not self.pending
            and not self.orders
            and not self._inspecting
            and q + 1 < self._next_fail
        ):  # a quiet quarter: nothing can change, so nothing is recomputed
            self.throughput.append(self._tp)
            self.q = q + 1
            return _QUIET_HANDOVER if (q + 1) % SHIFT_Q == 0 else _QUIET
        n_viol0 = len(self.violations)
        touched: set[int] = set()
        executed: list[Command] = []
        refused: list[Command] = []
        over_ids = {id(c) for c in overridden}
        for cmd in commands:
            ok = self._apply(cmd, q, touched)
            (executed if ok else refused).append(cmd)
            self.commands_log.append((q, cmd, ok, id(cmd) in over_ids))
        if touched:
            self._refresh_watch()
        n_cmd_touched = len(touched)
        self._environment(q, touched)
        self._check(q, touched)
        for i in touched:
            self._record(self.states[i], q)
        if touched:
            self._tp = self._throughput()
            if len(touched) != n_cmd_touched:
                self._refresh_watch()
        self.throughput.append(self._tp)
        self.q = q + 1
        return StepOutcome(
            tuple(executed),
            tuple(refused),
            tuple(self.violations[n_viol0:]),
            frozenset(touched),
            is_handover_next(q),
        )

    def _refresh_watch(self) -> None:
        """Assets whose environment must be advanced every quarter (a permit of any kind ends at
        its limit, so a REVAL permit is watched as well as a VALID one)."""
        self._watch = [
            s
            for s in self.states
            if s.occ or s.alarm or s.op is _WORK or s.permit is not Permit.NONE
        ]

    # ------------------------------------------------------------------ commands
    def _apply(self, cmd: Command, q: int, touched: set[int]) -> bool:
        s = self.by_id.get(cmd.asset_id)
        if s is None:
            return False
        c = cmd.cmd
        if cmd.source == DO_REPLACE and c is PlanCmd.RAISE_WO:
            return self._do_replace(s, q, touched)
        if c is Cmd.NOOP:
            return True
        forced = isinstance(c, Cmd) and c in FORCED_CMDS
        u = float(self.st.u_refuse[s.aid][q])
        if forced:
            if u < self.r.forced_refusal_p:
                return False
        elif u < self.params.p_refuse:
            return False
        touched.add(s.idx)
        if isinstance(c, PlanCmd):
            if c is PlanCmd.INSPECT:
                return self._inspect(s, q)
            if c is PlanCmd.ORDER:
                return self._order(cmd, q)
            return self._raise_wo(s, cmd, q)
        handler = self._HANDLERS.get(c)
        return handler(self, s, q) if handler is not None else False

    def _h_start(self, s: AssetState, q: int) -> bool:
        if s.op is _STOP:
            self._set_op(s, _RUN, q)
            if s.job is not None and s.job.completed:
                s.job = None  # the work order closes when the asset is back in service
            return True
        if s.op is _RUN:
            return True
        if s.op is _WORK:  # energising while work is in progress: P1, whoever is inside
            self._violate("P1", s.aid, q)
            s.iso = False
            self._set_op(s, _RUN, q)
            return True
        return False  # LOCKED or ISO: the lock prevents it; FAIL: nothing to start

    def _h_stop(self, s: AssetState, q: int) -> bool:
        if s.op is _RUN:
            self._set_op(s, _STOP, q)
        return True

    def _h_lock(self, s: AssetState, q: int) -> bool:
        if s.op in (_STOP, _FAIL):
            self._set_op(s, _LOCKED, q)
            s.iso = False
            return True
        return s.op is _LOCKED

    def _h_prove(self, s: AssetState, q: int) -> bool:
        if s.op is _LOCKED:
            self._set_op(s, _ISO, q)
            s.iso = True
            return True
        return False

    def _h_issue(self, s: AssetState, q: int) -> bool:
        # ISO, or WORK with nobody inside (a spent permit is replaced mid-job, ADR 0003 A12.1)
        if s.op is not _ISO and not (s.op is _WORK and not s.occ):
            return False
        if s.permit is Permit.NONE:
            s.permit = Permit.VALID
            s.limit_q = q + self.permit_max_q
            s.expiry_q = self._expiry(s, q)
        return True

    def _h_reval(self, s: AssetState, q: int) -> bool:
        if s.permit is not Permit.REVAL:
            return s.permit is Permit.VALID
        if float(self.st.u_holder[s.aid][q]) < self.r.holder_p[shift_of(q)]:
            s.permit = Permit.VALID
            s.expiry_q = self._expiry(s, q)  # limit_q does not move: a handover does not renew it
        return True  # executed; if the holder was absent the permit stays REVAL

    def _h_begin(self, s: AssetState, q: int) -> bool:
        if s.op is _WORK:
            return True
        job = s.job
        if job is not None and job.status != APPROVED:
            return False  # the work order is not released
        if job is not None and not job.part_taken:
            if self.stock_of(job.part_no) <= 0 and job.part_no is not None:
                return False
        if self.crew_free() <= 0:
            return False
        if s.op is _ISO and s.permit is not Permit.VALID:
            return False  # isolated but no permit: the permit office refuses
        if job is not None and not job.part_taken and job.part_no is not None:
            self.stock[job.part_no] -= 1
            self.stock_version += 1
            job.part_taken = True
        self._set_op(
            s, _WORK, q
        )  # from RUN, STOP, LOCKED or FAIL this is work on an unproven asset
        return True

    def _h_test(self, s: AssetState, q: int) -> bool:
        if s.op is not _WORK:
            return False
        if q < s.exc_until:
            s.alarm = True  # the test fails
        else:
            s.last_pass_q = q
        return True

    def _h_enter(self, s: AssetState, q: int) -> bool:
        if not s.enterable:
            return False
        if s.occ:
            return True
        # Re-entry after a break needs a fresh test. A test older than the re-test period is caught
        # by the state check below (P3) at this same quarter, so only the window in between is an
        # event.
        if self.break_q < self.gas_age_q(s, q) <= self.retest_q:
            self._violate("P3", s.aid, q)
        s.occ = True
        return True

    def _h_exit(self, s: AssetState, q: int) -> bool:
        if s.occ:
            s.occ = False
            s.last_pass_q = NEVER_Q  # a break: a fresh test before anyone goes back in
        return True

    def _h_complete(self, s: AssetState, q: int) -> bool:
        if s.op is not _WORK or s.occ:
            return False
        self._set_op(s, _ISO, q)
        s.iso = True
        job = s.job
        if job is not None and not job.completed and job.work_done:
            self._finish_job(s, job, q)
        return True

    def _h_close(self, s: AssetState, q: int) -> bool:
        if s.op is _WORK:
            return False
        if s.op is _ISO and s.permit is not Permit.NONE:
            s.permit = Permit.NONE
        return s.op is _ISO or s.permit is Permit.NONE

    def _h_uniso(self, s: AssetState, q: int) -> bool:
        if s.op is not _ISO or s.permit is not Permit.NONE:
            return False
        self._set_op(s, _STOP, q)
        s.iso = False
        return True

    def _h_derate(self, s: AssetState, q: int) -> bool:
        if not self.params.derate_enabled or s.clock.derated:
            return False
        s.clock.set_derated(q, True)
        self._refresh_fail(s)
        return True

    _HANDLERS = {
        Cmd.START: _h_start,
        Cmd.STOP: _h_stop,
        Cmd.LOCK: _h_lock,
        Cmd.PROVE: _h_prove,
        Cmd.ISSUE: _h_issue,
        Cmd.REVAL: _h_reval,
        Cmd.BEGIN: _h_begin,
        Cmd.TEST: _h_test,
        Cmd.ENTER: _h_enter,
        Cmd.EXIT: _h_exit,
        Cmd.COMPLETE: _h_complete,
        Cmd.CLOSE: _h_close,
        Cmd.UNISO: _h_uniso,
        Cmd.DERATE: _h_derate,
    }

    # ------------------------------------------------------------------ planning commands
    def _inspect(self, s: AssetState, q: int) -> bool:
        if s.inspect_due_q >= 0 or self.crew_free() <= 0:
            return False
        s.inspect_due_q = q + self.params.inspect_q
        self._busy += 1
        self._inspecting += 1
        self.stats.inspections += 1
        self.stats.cost += self.params.inspect_cost
        return True

    def _order(self, cmd: Command, q: int) -> bool:
        part = cmd.arg
        if part is None or part not in PARTS:
            return False
        if any(p == part for _, p in self.orders):
            return False
        lead = PARTS[part].lead_days * self.r.spare_lead_mult
        self.orders.append((q + max(int(math.ceil(lead * DAY_Q)), 1), part))
        self.stats.orders += 1
        return True

    def _raise_wo(self, s: AssetState, cmd: Command, q: int) -> bool:
        if s.job is not None or cmd.arg is None:
            return False
        mode, _, shift_txt = cmd.arg.partition(":")
        spec = mode_spec(s.cls, mode)
        if spec is None:
            return False  # a mode the matrix does not know for this class fails closed
        s.n_wo += 1
        job = _Job(s.aid, f"WO-{s.aid}-{s.n_wo}", mode, q)
        job.confined = (s.cls, mode) in CONFINED_JOBS
        job.needs_officer = (s.cls, mode) in SAFETY_OFFICER_JOBS
        job.part_no = spec.part_no
        job.work_total_q = JOB_WORK_Q[(s.cls, mode)] + (s.fail_extra_q if s.op is _FAIL else 0)
        if shift_txt.strip().isdigit():
            job.target_q = int(shift_txt) * SHIFT_Q
        k = job.k_draw = (s.n_wo - 1) % _N_APPROVALS
        hours = (
            self.params.approver_median_h
            * self.r.approver_delay_mult
            * math.exp(self.params.approver_sigma * float(self.st.appr_z[s.aid][k]))
        )
        approve = q + max(int(math.ceil(hours * 60 / QUARTER_MIN)), 1)
        if job.needs_officer:
            while not self.safety_officer_on_shift(approve):
                approve += 1
        job.approve_q = approve
        s.job = job
        self.pending.append(job)
        return True

    def _do_replace(self, s: AssetState, q: int, touched: set[int]) -> bool:
        """do(replace): the causal intervention of the twin worlds, no job and no cost."""
        touched.add(s.idx)
        self._renew(s, q)
        if s.op is _FAIL:
            self._set_op(s, _RUN, q)
        return True

    # ------------------------------------------------------------------ jobs and renewal
    def _renew(self, s: AssetState, q: int) -> None:
        s.life_id += 1
        s.renewals += 1
        life = next_life(self.phys, s.aid, s.life_id, s.clock.tau(q))
        s.life = life
        s.clock.start_life(q, life)
        s.life_start_q = q
        s.fail_extra_q = 0
        self._refresh_fail(s)

    def _finish_job(self, s: AssetState, job: _Job, q: int) -> None:
        job.completed = True
        job.status = COMPLETED
        s.jobs_done += 1
        self.stats.planned_jobs += 1
        self.stats.cost += self.params.planned_job_cost
        if job.mode == s.life.mode:
            self._renew(s, q)  # the right repair renews the component
        else:
            self.stats.wrong_mode_jobs += 1  # the work was done, the failing part is still there

    # ------------------------------------------------------------------ environment
    def _environment(self, q: int, touched: set[int]) -> None:
        params = self.params
        nxt = q + 1
        for s in self._watch:
            j = s.job
            if j is not None and s.op is _WORK and not j.completed:
                if s.occ or not j.confined:
                    j.progress_q += 1
            # gas excursions: only while someone is inside
            if s.occ:
                started = float(self.st.u_gas[s.aid][q]) < params.gas_excursion_rate
                if not started and s.adv_gas_pending and q >= s.adv_gas_pending[0]:
                    started = True
                    s.adv_gas_pending.pop(0)
                if started and q >= s.exc_until:
                    s.exc_until = nxt + params.excursion_q
            alarm = q < s.exc_until
            if alarm != s.alarm:
                s.alarm = alarm
                touched.add(s.idx)
            # A permit is spent at its 12 hour limit, from VALID or REVAL alike; otherwise a VALID
            # one lapses into REVAL at the end of the shift that issued or revalidated it.
            if s.permit is not Permit.NONE:
                if nxt >= s.limit_q:
                    s.permit = Permit.NONE
                    touched.add(s.idx)
                elif s.permit is Permit.VALID and nxt >= s.expiry_q:
                    s.permit = Permit.REVAL
                    touched.add(s.idx)
        if nxt >= self._next_fail:
            for s in self.states:
                if s.op is _RUN and nxt >= self._fail_q[s.idx]:
                    self._fail(s, q, touched)
        self._due_events(nxt, touched)

    def _fail(self, s: AssetState, q: int, touched: set[int]) -> None:
        s.op = _FAIL
        s.iso = False
        s.clock.set_running(q + 1, False)
        s.unc_fail_q.append(q)
        self._refresh_fail(s)
        self.stats.unplanned_failures += 1
        self.stats.cost += self.params.unplanned_cost_ratio * self.params.planned_job_cost
        s.fail_extra_q = int(math.ceil(s.life.downtime_hours * 60 / QUARTER_MIN))
        if s.job is not None and not s.job.completed:
            s.job.work_total_q += s.fail_extra_q
        if s.cls == "CRUSHER":
            self._violate("P5", s.aid, q)
        touched.add(s.idx)

    def _due_events(self, nxt: int, touched: set[int]) -> None:
        """Approvals, deliveries and inspections that fall due by the start of quarter ``nxt``."""
        if self.pending:
            still = []
            for job in self.pending:
                if job.approve_q <= nxt:
                    self._decide(job, nxt, touched)
                else:
                    still.append(job)
            self.pending = still
        if self.orders:
            keep = []
            for due, part in self.orders:
                if due <= nxt:
                    self.stock[part] += 1
                    self.stock_version += 1
                    for s in self.states:
                        touched.add(s.idx)
                else:
                    keep.append((due, part))
            self.orders = keep
        for s in self.states:
            if 0 <= s.inspect_due_q <= nxt:
                self._finish_inspection(s, nxt)
                touched.add(s.idx)

    def _decide(self, job: _Job, nxt: int, touched: set[int]) -> None:
        s = self.by_id[job.asset_id]
        touched.add(s.idx)
        wrong = job.mode != s.life.mode
        if wrong and float(self.st.appr_c[s.aid][job.k_draw]) < self._catch_p:
            s.job = None  # the approver caught the wrong diagnosis; the work order is withdrawn
            self.stats.wo_rejected += 1
            self.stats.cost += self.params.rejected_wo_cost
            return
        job.status = APPROVED

    def _finish_inspection(self, s: AssetState, nxt: int) -> None:
        s.inspect_due_q = -1
        self._busy -= 1
        self._inspecting -= 1
        k = s.inspect_count % _N_INSPECTIONS
        s.inspect_count += 1
        if self.inspection_hook is not None:  # the plant layer owns the observation streams
            self.inspection_hook(s, nxt, float(self.st.insp_z[s.aid][k]))

    # ------------------------------------------------------------------ the plant's own checks
    def _violate(self, prop: str, asset_id: str, q: int) -> None:
        self.violations.append((prop, asset_id, q))

    def _check(self, q: int, touched: set[int]) -> None:
        """P1-P6 from the English properties (0001 section 5), on the end-of-step state of ``q``."""
        if not touched and not self._watch:
            return
        for s in self.states:
            if s.idx not in touched and not s.occ:
                continue
            p1 = s.op is _WORK and not s.iso
            p2 = s.occ and s.permit is not Permit.VALID
            p3 = s.occ and self.gas_age_q(s, q) > self.retest_q
            if p1 and not s.f1:
                self._violate("P1", s.aid, q)
            if p2 and not s.f2:
                self._violate("P2", s.aid, q)
            if p3 and not s.f3:
                self._violate("P3", s.aid, q)
            s.f1, s.f2, s.f3 = p1, p2, p3
            if s.enterable:
                # P6: an alarm with someone inside needs an exit within alarm_exit_q quarters
                if s.occ and s.alarm and s.p6_start < 0:
                    s.p6_start = q
                elif not s.occ:
                    s.p6_start = -1
                    s.p6_reported = False
                if (
                    s.p6_start >= 0
                    and s.occ
                    and not s.p6_reported
                    and q - s.p6_start >= self.alarm_exit_q
                ):
                    self._violate("P6", s.aid, s.p6_start)
                    s.p6_reported = True
        a, b = (self.by_id[p] for p in PUMPS)
        # a locked pump cannot be started until the lock is removed: it is not available (A12.2)
        lost = a.op in _PUMP_UNAVAILABLE and b.op in _PUMP_UNAVAILABLE
        if lost and not self.f4:
            self._violate("P4", PUMP_PAIR_ID, q)
        self.f4 = lost

    def _throughput(self) -> float:
        by = self.by_id
        if by[CRUSHER].op is not _RUN or by["CV-101"].op is not _RUN:
            return 0.0
        running = [by[p] for p in PUMPS if by[p].op is _RUN]
        if not running:
            return 0.0
        base = 1.0 if len(running) == 2 else self.params.single_pump_throughput
        loss = self.params.derate_throughput_loss * sum(1 for s in self.states if s.clock.derated)
        return max(base - loss, 0.0)

    # ------------------------------------------------------------------ recording and the Trace
    def _record(self, s: AssetState, q: int) -> None:
        """Append a history row if the traced state of ``s`` differs from its last row."""
        row = (
            max(q, 0),
            _OP_IDX[s.op],
            _PERMIT_IDX[s.permit],
            s.occ,
            s.alarm,
            s.iso,
            s.last_pass_q,
            s.expiry_q if s.permit is Permit.VALID else NEVER_Q,
        )
        if s.hist:
            last = s.hist[-1]
            if last[1:] == row[1:]:
                return
            if last[0] == row[0]:
                s.hist[-1] = row
                return
        s.hist.append(row)

    def set_sets(self, s: AssetState, q: int, lo: int, hi: int) -> None:
        """Record the perception set given to the shield from quarter ``q`` on."""
        last = s.sets[-1]
        if (last[1], last[2]) == (lo, hi):
            return
        if last[0] == q:
            s.sets[-1] = (q, lo, hi)
        else:
            s.sets.append((q, lo, hi))

    def build_trace(self) -> Trace:
        n = self.q
        ids = ASSET_IDS
        tr = Trace.empty(n, ids)
        for s in self.states:
            a = s.aid
            rows = s.hist
            for i, (q0, op, permit, occ, alarm, iso, last_pass, expiry) in enumerate(rows):
                q1 = rows[i + 1][0] if i + 1 < len(rows) else n
                q1 = min(q1, n)
                if q0 >= q1:
                    continue
                tr.op[a][q0:q1] = op
                tr.permit[a][q0:q1] = permit
                tr.occupied[a][q0:q1] = occ
                tr.alarm[a][q0:q1] = alarm
                tr.iso_proven[a][q0:q1] = iso
                qs = np.arange(q0, q1)
                if last_pass > NEVER_Q // 2:
                    age = np.maximum(qs - last_pass, 0) * float(QUARTER_MIN)
                    tr.gas_age_min[a][q0:q1] = np.minimum(age, GAS_AGE_NEVER_MIN)
                else:
                    tr.gas_age_min[a][q0:q1] = GAS_AGE_NEVER_MIN
                if permit == _PERMIT_IDX[Permit.VALID]:
                    tr.permit_margin_min[a][q0:q1] = (expiry - qs) * float(QUARTER_MIN)
                elif permit == _PERMIT_IDX[Permit.REVAL]:
                    tr.permit_margin_min[a][q0:q1] = -1.0
                else:
                    tr.permit_margin_min[a][q0:q1] = PERMIT_MARGIN_NONE_MIN
            rul, bins, _ = s.clock.truth_arrays(n)
            tr.true_rul_days[a][:] = rul
            tr.health_bin[a][:] = bins
            for i, (q0, lo, hi) in enumerate(s.sets):
                q1 = s.sets[i + 1][0] if i + 1 < len(s.sets) else n
                q1 = min(q1, n)
                if q0 < q1:
                    tr.set_lo[a][q0:q1] = lo
                    tr.set_hi[a][q0:q1] = hi
            for fq in s.unc_fail_q:
                if fq < n:
                    tr.uncontrolled_failure[a][fq] = True
        tr.throughput[:] = np.asarray(self.throughput[:n], dtype=np.float32)
        tr.commands.extend(self.commands_log)
        tr.validate()
        return tr


__all__ = [
    "AWAITING",
    "APPROVED",
    "COMPLETED",
    "DO_REPLACE",
    "NEVER_Q",
    "PUMP_PAIR_ID",
    "AssetState",
    "Des",
    "DesStats",
    "DesStreams",
    "JobView",
    "SpecLike",
    "StepOutcome",
    "build_des_streams",
]
