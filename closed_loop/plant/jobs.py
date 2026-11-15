"""Job requirements and the job runner of the SYNTHETIC plant.

``requirement`` answers "which permits does this job need, which work procedure governs it and who
may sign it" from the asset class, the failure mode and the task type. When ``maintenance_agent`` is
installed it wraps ``maintenance_agent.requirements.requirement_for`` (docs/adr/0002 section 3.5:
the one place outside ``f2`` and ``spec.clauses`` that may import that sibling, lazily and only
inside the function); otherwise it answers from a vendored copy of the matrix.
``tests/plant/test_jobs.py`` compares the two for every class and mode when both exist, and says why
it skipped when they do not.

``JobRunner`` is the part of a maintenance team that carries a released work order out. It looks at
one ``AssetView`` and its open job and says what to do next, one command per quarter: stop, lock,
prove the isolation, issue the permit, begin, (for a confined space: test the atmosphere, enter,
re-test on a schedule, leave before the handover and before the permit's 12 hour limit, revalidate
the permit after a handover, issue a new one when the old one is spent, go back in), complete,
close the permit, remove the isolation, start. It keeps no state of its own, so after a refused or
overridden command it simply asks again (the re-plan of 0002), and it works from what a proposer
can see. A permit lives 12 hours from its issue and a handover does not renew that (0003 A12.1):
the time it will go in for is the shorter of the shift and what is left on the permit.

Everything here is SYNTHETIC.
"""

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from closed_loop.core.timegrid import SHIFT_Q
from closed_loop.core.types import Cmd, Command, Op, Permit, PlanCmd
from closed_loop.plant.des import APPROVED, AWAITING, COMPLETED, JobView
from closed_loop.plant.params import ASSET_CLASS
from closed_loop.plant.plant import AssetView

# -------------------------------------------------------------------------------------- the matrix
PERMIT_TYPES: tuple[str, ...] = (
    "LOTO",
    "CONFINED_SPACE",
    "HOT_WORK",
    "WORK_AT_HEIGHT",
    "HV_ACCESS",
)
HIGH_RISK: frozenset[str] = frozenset({"CONFINED_SPACE", "HOT_WORK", "HV_ACCESS"})

# Vendored from maintenance_agent.requirements (S: version 0.1.1, MATRIX_VERSION "1"). The
# comparison test is the guard against drift; do not edit one side alone.
_BASE_PERMITS: frozenset[str] = frozenset({"LOTO"})
_CLASS_PERMITS: dict[str, frozenset[str]] = {
    "BELT_CONVEYOR": frozenset(),
    "SLURRY_PUMP": frozenset(),
    "VENT_FAN": frozenset(),
    "BALL_MILL": frozenset(),
    "CRUSHER": frozenset(),
    "HV_MOTOR": frozenset({"HV_ACCESS"}),
}
_MODE_PERMITS: dict[tuple[str, str], frozenset[str]] = {
    ("BALL_MILL", "LINER_WEAR"): frozenset({"CONFINED_SPACE"}),
    ("CRUSHER", "LINER_WEAR"): frozenset({"CONFINED_SPACE", "HOT_WORK", "WORK_AT_HEIGHT"}),
    ("CRUSHER", "BEARING_DEFECT"): frozenset({"WORK_AT_HEIGHT"}),
    ("VENT_FAN", "IMBALANCE"): frozenset({"WORK_AT_HEIGHT"}),
}
_REPAIR_PROCEDURE: dict[tuple[str, str], str] = {
    ("BELT_CONVEYOR", "BEARING_DEFECT"): "PRC-MECH-010",
    ("BELT_CONVEYOR", "GEAR_WEAR"): "PRC-MECH-011",
    ("BELT_CONVEYOR", "MISALIGNMENT"): "PRC-MECH-012",
    ("SLURRY_PUMP", "BEARING_DEFECT"): "PRC-MECH-010",
    ("SLURRY_PUMP", "MISALIGNMENT"): "PRC-MECH-012",
    ("SLURRY_PUMP", "SEAL_LEAK"): "PRC-PUMP-020",
    ("SLURRY_PUMP", "IMPELLER_WEAR"): "PRC-PUMP-021",
    ("VENT_FAN", "BEARING_DEFECT"): "PRC-MECH-010",
    ("VENT_FAN", "MISALIGNMENT"): "PRC-MECH-012",
    ("VENT_FAN", "IMBALANCE"): "PRC-MECH-013",
    ("BALL_MILL", "GEAR_WEAR"): "PRC-MECH-011",
    ("BALL_MILL", "BEARING_DEFECT"): "PRC-MILL-032",
    ("BALL_MILL", "LINER_WEAR"): "PRC-MILL-030",
    ("CRUSHER", "LINER_WEAR"): "PRC-CRSH-031",
    ("CRUSHER", "BEARING_DEFECT"): "PRC-CRSH-032",
    ("HV_MOTOR", "WINDING_INSULATION"): "PRC-ELEC-040",
    ("HV_MOTOR", "BEARING_DEFECT"): "PRC-ELEC-041",
    **{
        (cls, "LUBE_DEGRADATION"): "PRC-LUBE-050"
        for cls in ("BELT_CONVEYOR", "SLURRY_PUMP", "VENT_FAN", "BALL_MILL", "CRUSHER")
    },
}
_INSPECTION_PROCEDURE: dict[str, str] = {
    "BELT_CONVEYOR": "PRC-INSP-060",
    "SLURRY_PUMP": "PRC-INSP-060",
    "VENT_FAN": "PRC-INSP-060",
    "BALL_MILL": "PRC-INSP-060",
    "CRUSHER": "PRC-INSP-060",
    "HV_MOTOR": "PRC-INSP-061",
}


@dataclass(frozen=True)
class JobRequirement:
    permits: frozenset[str]
    work_procedure: str | None
    role: str
    known: bool


def _role(permits: frozenset[str]) -> str:
    return "SAFETY_OFFICER" if permits & HIGH_RISK else "PLANNER"


def vendored_requirement(
    asset_class: str, mode: str | None, inspection_first: bool
) -> JobRequirement:
    """The vendored copy of the matrix. An unknown class or repair fails closed (every permit)."""
    everything = frozenset(PERMIT_TYPES)
    if asset_class not in _CLASS_PERMITS:
        return JobRequirement(everything, None, _role(everything), False)
    permits = set(_BASE_PERMITS) | set(_CLASS_PERMITS[asset_class])
    if inspection_first:
        fixed = frozenset(permits)
        return JobRequirement(fixed, _INSPECTION_PROCEDURE.get(asset_class), _role(fixed), True)
    if mode is None or (asset_class, mode) not in _REPAIR_PROCEDURE:
        return JobRequirement(everything, None, _role(everything), False)
    permits |= _MODE_PERMITS.get((asset_class, mode), frozenset())
    fixed = frozenset(permits)
    return JobRequirement(fixed, _REPAIR_PROCEDURE[(asset_class, mode)], _role(fixed), True)


def _agent_requirement_for() -> Callable[..., Any] | None:
    """``maintenance_agent.requirements.requirement_for`` if the sibling is installed, else None."""
    try:
        module = importlib.import_module("maintenance_agent.requirements")
    except ImportError:
        return None
    fn: Callable[..., Any] = module.requirement_for
    return fn


def agent_requirement(
    asset_class: str, mode: str | None, inspection_first: bool
) -> JobRequirement | None:
    """The sibling's answer in our type, or None when ``maintenance_agent`` is not installed."""
    fn = _agent_requirement_for()
    if fn is None:
        return None
    req = fn(asset_class, mode, inspection_first)
    return JobRequirement(
        frozenset(str(getattr(p, "value", p)) for p in req.permits),
        req.work_procedure,
        str(req.role),
        bool(req.known),
    )


def requirement(asset_class: str, mode: str, inspection_first: bool = False) -> JobRequirement:
    """The matrix answer: the real one when installed, the vendored copy otherwise."""
    real = agent_requirement(asset_class, mode, inspection_first)
    return real if real is not None else vendored_requirement(asset_class, mode, inspection_first)


def matrix_source() -> str:
    """Which table ``requirement`` is answering from: ``"maintenance_agent"`` or ``"vendored"``."""
    return "vendored" if _agent_requirement_for() is None else "maintenance_agent"


def work_order_command(
    asset_id: str, mode: str, target_shift: int | None = None, source: str = ""
) -> Command:
    """``RAISE_WO`` for ``asset_id`` with failure mode ``mode``, optionally from a shift on."""
    arg = mode if target_shift is None else f"{mode}:{target_shift}"
    return Command(asset_id, PlanCmd.RAISE_WO, arg, source)


# ------------------------------------------------------------------------------------- the runner
ALPHABET: frozenset[Cmd] = frozenset(
    {
        Cmd.STOP, Cmd.LOCK, Cmd.PROVE, Cmd.ISSUE, Cmd.REVAL, Cmd.BEGIN, Cmd.TEST, Cmd.ENTER,
        Cmd.EXIT, Cmd.COMPLETE, Cmd.CLOSE, Cmd.UNISO, Cmd.START,
    }
)  # fmt: skip


class JobRunner:
    """Carries out one released work order, one command per quarter. Stateless.

    ``retest_q`` is the gas re-test period the team believes in (Rev 3: 2 quarters; a team on the
    superseded Rev 2 procedure would pass 8). It re-tests when the last pass is at least ``retest_q
    - test_margin_q`` quarters old, so that one refused test does not by itself breach the period.
    ``break_q`` is the longest gap after which a fresh test is needed before re-entry.
    ``min_entry_window_q`` is how many quarters must remain in the shift before it starts an entry
    (test, enter, and some time inside).
    """

    def __init__(
        self,
        retest_q: int = 2,
        break_q: int = 1,
        test_margin_q: int = 1,
        min_entry_window_q: int = 4,
        source: str = "job-runner",
    ):
        if retest_q < 1 or break_q < 0 or test_margin_q < 0 or min_entry_window_q < 2:
            raise ValueError(
                "retest_q >= 1, break_q >= 0, test_margin_q >= 0, min_entry_window_q >= 2"
            )
        self.retest_q = retest_q
        self.break_q = break_q
        self.test_margin_q = test_margin_q
        self.min_entry_window_q = min_entry_window_q
        self.source = source

    def _cmd(self, asset_id: str, cmd: Cmd) -> Command:
        return Command(asset_id, cmd, None, self.source)

    def next_command(
        self,
        view: AssetView,
        job: JobView | None = None,
        quarter: int | None = None,
    ) -> Command | None:
        """The next command of the job on ``view``'s asset, or None to wait.

        ``job`` defaults to ``view.job``. ``quarter`` (the decision quarter) lets the runner see how
        long is left in the shift; without it only ``view.soon`` is known.
        """
        job = job if job is not None else view.job
        if job is None or job.status == AWAITING:
            return None
        a = view.asset_id
        op, permit = view.op, view.permit
        if job.status == COMPLETED:  # close out: permit, isolation, back into service
            if op is Op.ISO:
                return self._cmd(a, Cmd.CLOSE if permit is not Permit.NONE else Cmd.UNISO)
            if op is Op.STOP:
                return self._cmd(a, Cmd.START)
            return None
        if job.status != APPROVED or not job.ready:
            return None
        if op is Op.RUN:
            # A running asset is taken out of service only once the spare is here (0003 A17). The
            # runner used to stop, lock, prove and permit it first and look at the part at BEGIN;
            # with none in stock the asset then sat isolated to the end of the episode while the
            # permit was renewed for nobody. Waiting costs nothing the asset is not already
            # costing. An asset that is already down (STOP or FAIL, below) is made safe whether
            # or not the part is here.
            return self._cmd(a, Cmd.STOP) if job.part_available else None
        if op in (Op.STOP, Op.FAIL):
            return self._cmd(a, Cmd.LOCK)
        if op is Op.LOCKED:
            return self._cmd(a, Cmd.PROVE)
        if op is Op.ISO:
            if permit is Permit.NONE:
                return self._cmd(a, Cmd.ISSUE)
            if permit is Permit.REVAL:
                return self._cmd(a, Cmd.REVAL)
            return self._cmd(a, Cmd.BEGIN) if job.part_available else None
        if op is not Op.WORK:
            return None
        return self._in_work(view, job, quarter)

    def _in_work(self, view: AssetView, job: JobView, quarter: int | None) -> Command | None:
        a = view.asset_id
        done = job.work_done_q >= job.work_total_q
        entry = self._needs_entry(view, job)
        if done:
            return self._cmd(a, Cmd.EXIT if view.occupied else Cmd.COMPLETE)
        if not entry:
            return None  # the fitters are working; nothing to command
        if view.occupied:
            # leave on an alarm, before the handover or the permit's 12 hour limit, and when the
            # permit is no longer valid (it is a violation already, and the way out is the same)
            if view.alarm or view.soon or view.permit is not Permit.VALID:
                return self._cmd(a, Cmd.EXIT)
            if view.gas_age_q >= self.retest_q - self.test_margin_q:
                return self._cmd(a, Cmd.TEST)
            return None  # working inside
        if view.permit is Permit.REVAL:
            return self._cmd(a, Cmd.REVAL)
        if view.permit is Permit.NONE:
            return self._cmd(a, Cmd.ISSUE)  # spent at its limit (ADR 0003 A12.1): a new one
        if view.alarm:
            return None
        left = self._time_left_q(view, quarter)
        if view.soon or (left is not None and left < self.min_entry_window_q):
            return None  # too close to the handover or the permit's limit to go in
        if view.gas_age_q <= self.break_q:
            return self._cmd(a, Cmd.ENTER)
        return self._cmd(a, Cmd.TEST)

    @staticmethod
    def _time_left_q(view: AssetView, quarter: int | None) -> int | None:
        """Quarters an entry could last: the shorter of the shift and the permit's 12 hours."""
        left = SHIFT_Q - (quarter % SHIFT_Q) if quarter is not None else None
        on_permit = view.permit_limit_left_q
        if on_permit is None:
            return left
        return on_permit if left is None else min(left, on_permit)

    @staticmethod
    def _needs_entry(view: AssetView, job: JobView) -> bool:
        cls = ASSET_CLASS.get(view.asset_id)
        if cls is None:
            return job.confined
        return "CONFINED_SPACE" in requirement(cls, job.mode, False).permits
