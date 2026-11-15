"""The hand-written interlock: the shield's simple baseline (ADR 0003 A4). SYNTHETIC.

The brief says a sophisticated component that does not beat its simple baseline gets published. The
shield's baseline is the kind of interlock a controls engineer writes by hand, exactly these six
rules and no others:

1. refuse ``BEGIN`` unless the isolation is proven;
2. refuse ``ENTER`` unless the permit is valid;
3. refuse ``ENTER`` when the gas test is older than the re-test period;
4. force ``EXIT`` when the gas-test margin is at or below one quarter, or the alarm is on;
5. refuse ``LOCK`` of a pump when the other pump is not running;
6. force ``STOP`` when the crusher's point RUL estimate is under 3 days, and refuse a ``START``
   of the stopped crusher while it still is (0003 A18: without that refusal a restart offered once
   a day by the operator (``proposers.restart``) and rule 6 would flap once a day).

It uses no game, no perception set and no model checking. A refused command is simply dropped (the
asset waits); a forced command takes the place of whatever the inner proposer wanted for that
asset in that quarter. Anything else passes untouched.

Runtime verification in the loop. The triggers of rules 4 and 6 read **margins from
``StreamingMonitor``s**, fed one quarter at a time from the observation: the re-test margin (the
``crusher.gas_fresh`` atom of the specification: minutes left in the re-test period), the alarm, and
the RUL margin (point estimate minus 3 days). The six property formulas of the specification run
online too, on the same signals. For each violation the plant records, ``episode_record`` gives the
quarters from its occurrence to the first negative online robustness of that property (``None``
when it never went negative). The formulas are monitored without their outer ``G``: the running
minimum of ``G`` stays negative for ever after the first violation and could not time the second.

What the interlock can see. It sees the ``Observation``, so some atoms of the specification are
rebuilt from less than the plant's trace holds, in a way that keeps their sign (the sign is what a
verdict needs): ``iso_proven`` is tracked from the asset's operating state (ISO proves it; WORK
keeps what it was when work began); the permit margin is the time left in the shift while the
permit is valid; the gas-test age is ``(gas_age_q - 1)`` quarters, because the view is one quarter
ahead of the trace row it describes. ``tests/eval/test_interlock.py`` compares the online verdicts
with the offline ``spec.evaluate`` on whole episodes.

Rule 6 holds off for the rest of the day on which a work order completes on the crusher: until
the next daily update the interval on the view is the old component's, and stopping the part that
has just been renewed would be the interlock fighting the repair. The estimate is otherwise taken
as it comes.

The point RUL estimate. ``AssetView`` carries the perception layer's interval (``rul_lo``,
``rul_hi``), not the estimator's point. The default ``default_point_rul`` is the midpoint of the
interval; it is a stand-in, and ``point_rul=`` is the injection point for a better one when the
episode loop carries the estimator's ``m_days`` (never truth). With no interval there is no estimate
and rule 6 does not fire.

The seam is that of ``shield.runtime.ShieldedProposer``: ``propose(obs)``, ``reset(seed=None)``,
``name``, ``inner`` and the counters ``overrides`` and ``admitted``, so an arm can swap one for the
other.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from closed_loop.core.timegrid import QUARTER_MIN, SHIFT_Q
from closed_loop.core.trace import GAS_AGE_NEVER_MIN
from closed_loop.core.types import BIN_EDGES_DAYS, Cmd, Command, Op, Permit, PlanCmd
from closed_loop.plant.params import CONVEYOR, CRUSHER, PUMPS
from closed_loop.plant.plant import AssetView, Observation
from closed_loop.proposers.base import Proposer
from closed_loop.spec.monitor import StreamingMonitor
from closed_loop.spec.properties import (
    HOLDS_MARGIN,
    SpecParams,
    property_formulas,
)
from closed_loop.spec.stl import Always, Atom, Formula

__all__ = [
    "RULES",
    "STOP_RUL_DAYS",
    "InterlockEpisode",
    "InterlockedProposer",
    "PointRul",
    "ViolationLag",
    "default_point_rul",
]

PointRul = Callable[[AssetView], float | None]

STOP_RUL_DAYS: Final[float] = BIN_EDGES_DAYS[-1]  # "under 3 days" is the top health bin
_GAS_CAP_Q = 99  # AssetView.gas_age_q is capped here; treat it as "never tested"
_NEVER_EMPTY_MIN = 1.0e7  # spec: minutes empty when the space has not been occupied yet
_NO_ESTIMATE = 1.0e9  # RUL margin when there is no estimate: nothing to act on

R_BEGIN = "refuse BEGIN without proven isolation"
R_ENTER_PERMIT = "refuse ENTER without a valid permit"
R_ENTER_GAS = "refuse ENTER on a gas test older than the re-test period"
R_EXIT_GAS = "force EXIT: gas-test margin at or below one quarter"
R_EXIT_ALARM = "force EXIT: gas alarm on"
R_PUMP_LOCK = "refuse pump LOCK while the other pump is not running"
R_STOP = "force STOP: crusher point RUL under 3 days"
RULES: Final[tuple[str, ...]] = (
    R_BEGIN,
    R_ENTER_PERMIT,
    R_ENTER_GAS,
    R_EXIT_GAS,
    R_EXIT_ALARM,
    R_PUMP_LOCK,
    R_STOP,
)

_GAS_MARGIN = "crusher.gas_fresh"  # the specification's atom (minutes)
_ALARM_CLEAR = "crusher.alarm_clear"  # +1 no alarm, -1 alarm
_RUL_MARGIN = "crusher.rul_margin"  # point RUL minus 3 days


def default_point_rul(view: AssetView) -> float | None:
    """Midpoint of the perception interval, or None without one (see the module docstring)."""
    if view.rul_lo is None or view.rul_hi is None:
        return None
    return 0.5 * (view.rul_lo + view.rul_hi)


@dataclass(frozen=True)
class ViolationLag:
    """One violation of the plant and when the online monitor of its property noticed.

    ``lag_q`` is ``first_negative - quarter`` in quarters (0: the monitor was negative at the
    quarter of the violation); ``None`` if the robustness never went negative from then on.
    """

    prop: str
    asset_id: str
    quarter: int
    first_negative: int | None

    @property
    def lag_q(self) -> int | None:
        return None if self.first_negative is None else self.first_negative - self.quarter


@dataclass(frozen=True)
class InterlockEpisode:
    """What the interlock did and saw in one episode."""

    refused: int
    forced: int
    admitted: int
    by_rule: Mapping[str, int]
    violations: tuple[ViolationLag, ...]

    @property
    def undetected(self) -> int:
        return sum(1 for v in self.violations if v.lag_q is None)


def _online_formula(f: Formula) -> Formula:
    return f.arg if isinstance(f, Always) else f


class InterlockedProposer:
    """Wraps ``inner`` in the six rules; see the module docstring."""

    def __init__(
        self,
        inner: Proposer,
        spec: SpecParams | None = None,
        *,
        point_rul: PointRul | None = None,
        crusher_id: str = CRUSHER,
        pump_ids: tuple[str, str] = PUMPS,
        conveyor_id: str = CONVEYOR,
    ) -> None:
        self.inner = inner
        self.spec = spec if spec is not None else SpecParams()
        self.point_rul: PointRul = point_rul if point_rul is not None else default_point_rul
        self.crusher_id = crusher_id
        self.pump_ids = pump_ids
        self.conveyor_id = conveyor_id
        self.name = f"interlock({inner.name})"
        self._properties = {
            pid: _online_formula(f) for pid, f in property_formulas(self.spec).items()
        }
        self.reset()

    # ------------------------------------------------------------------ the Proposer seam
    def reset(self, seed: int | None = None) -> None:
        """Clear the counters and the monitors; with a seed, reset the inner proposer too."""
        self.overrides = 0
        self.admitted = 0
        self.refused = 0
        self.forced = 0
        self.by_rule: dict[str, int] = dict.fromkeys(RULES, 0)
        self._monitors = {pid: StreamingMonitor(f) for pid, f in self._properties.items()}
        self._gas = StreamingMonitor(Atom(_GAS_MARGIN))
        self._alarm = StreamingMonitor(Atom(_ALARM_CLEAR))
        self._rul = StreamingMonitor(Atom(_RUL_MARGIN))
        self._robustness: dict[str, list[float]] = {pid: [] for pid in self._properties}
        self._first_row = 0
        self._prev_op: dict[str, Op] = {}
        self._proven: dict[str, bool] = {}
        self._last_occupied_row: int | None = None
        self._permit_run_start: int | None = None
        self._jobs_done: int | None = None
        self._renewed_day = -1
        self._estimate_is_stale = False
        if seed is not None:
            self.inner.reset(seed)

    def propose(self, obs: Observation) -> list[Command]:
        gas_margin, alarm_margin, rul_margin = self._observe(obs)
        crusher = obs.assets[self.crusher_id]
        forced = self._forced(crusher, gas_margin, alarm_margin, rul_margin)
        out: list[Command] = []
        for c in self.inner.propose(obs):
            rule = self._refusal(c, obs, gas_margin, rul_margin)
            if rule is not None:
                self._count(rule, refused=True)
                continue
            if forced is not None and c.asset_id == self.crusher_id and isinstance(c.cmd, Cmd):
                if c.cmd is forced[0].cmd:
                    self.admitted += 1
                    forced = None  # the inner proposer already issued it
                    out.append(c)
                continue  # replaced by the forced command below
            self.admitted += 1
            out.append(c)
        if forced is not None:
            command, rule = forced
            self._count(rule, refused=False)
            out.append(command)
        return out

    # ------------------------------------------------------------------ record
    def episode_record(self, violations: Sequence[tuple[str, str, int]]) -> InterlockEpisode:
        """The episode's counts and, for each violation the plant recorded, its detection lag.

        ``violations`` is what ``StepResult.violations`` held, concatenated over the episode:
        ``(property id, asset id, quarter)``, the quarter being the trace row of the occurrence.
        """
        lags = tuple(
            ViolationLag(prop, asset, q, self._first_negative(prop, q))
            for prop, asset, q in sorted(violations, key=lambda v: (v[2], v[0], v[1]))
        )
        return InterlockEpisode(
            refused=self.refused,
            forced=self.forced,
            admitted=self.admitted,
            by_rule=dict(self.by_rule),
            violations=lags,
        )

    def robustness(self, prop: str) -> list[float]:
        """The online robustness of ``prop`` (without its outer G) at rows ``-1, 0, 1, ...``."""
        return list(self._robustness[prop])

    def _first_negative(self, prop: str, row: int) -> int | None:
        series = self._robustness.get(prop)
        if series is None:
            return None  # a property the interlock does not monitor
        for r in range(max(row, -1), len(series) - 1):
            if series[r + 1] < 0.0:
                return r
        return None

    # ------------------------------------------------------------------ rules
    def _count(self, rule: str, *, refused: bool) -> None:
        self.by_rule[rule] += 1
        self.overrides += 1
        if refused:
            self.refused += 1
        else:
            self.forced += 1

    def _forced(
        self, crusher: AssetView, gas_margin: float, alarm_margin: float, rul_margin: float
    ) -> tuple[Command, str] | None:
        if crusher.occupied and alarm_margin < 0.0:
            return Command(self.crusher_id, Cmd.EXIT, None, self.name), R_EXIT_ALARM
        if crusher.occupied and gas_margin <= QUARTER_MIN:
            return Command(self.crusher_id, Cmd.EXIT, None, self.name), R_EXIT_GAS
        if crusher.op is Op.RUN and rul_margin < 0.0 and not self._estimate_is_stale:
            return Command(self.crusher_id, Cmd.STOP, None, self.name), R_STOP
        return None

    def _refusal(
        self, c: Command, obs: Observation, gas_margin: float, rul_margin: float
    ) -> str | None:
        if isinstance(c.cmd, PlanCmd):
            return None  # maintenance planning is outside the interlock
        if not isinstance(c.cmd, Cmd):
            # neither kind of command: a plain string would reach the plant's handler table with no
            # rule applied. It cannot be refused under a rule (that would add one), so it stops the
            # episode (P12g, G6; ``Command`` already refuses to be built so)
            raise TypeError(
                f"the interlock was handed a command that is neither a Cmd nor a PlanCmd: {c.cmd!r}"
            )
        view = obs.assets.get(c.asset_id)
        if view is None:
            return None
        if (
            c.cmd is Cmd.START
            and c.asset_id == self.crusher_id
            and rul_margin < 0.0
            and not self._estimate_is_stale
        ):
            return R_STOP  # rule 6 would stop it again at once
        if c.cmd is Cmd.BEGIN and not self._is_proven(view):
            return R_BEGIN
        if c.cmd is Cmd.ENTER:
            if view.permit is not Permit.VALID:
                return R_ENTER_PERMIT
            # the entry is judged at the quarter it happens, one quarter after the row the margin
            # describes: the test must still be inside the period then
            if c.asset_id == self.crusher_id and gas_margin - QUARTER_MIN < 0.0:
                return R_ENTER_GAS
        if c.cmd is Cmd.LOCK and c.asset_id in self.pump_ids:
            other = self.pump_ids[1] if c.asset_id == self.pump_ids[0] else self.pump_ids[0]
            other_view = obs.assets.get(other)
            if other_view is None or other_view.op is not Op.RUN:
                return R_PUMP_LOCK
        return None

    def _is_proven(self, view: AssetView) -> bool:
        if view.op is Op.ISO:
            return True
        return view.op is Op.WORK and self._proven.get(view.asset_id, False)

    # ------------------------------------------------------------------ online monitors
    def _observe(self, obs: Observation) -> tuple[float, float, float]:
        """Push this quarter's signals through every monitor; return the three trigger margins."""
        for aid, view in obs.assets.items():
            prev = self._prev_op.get(aid)
            if view.op is Op.WORK and prev is not Op.WORK:
                self._proven[aid] = prev is Op.ISO
            elif view.op is not Op.WORK:
                self._proven.pop(aid, None)
        # A work order just completed: the estimate on the view is the old component's until the
        # next daily update, so rule 6 holds off for the rest of the day.
        done = obs.assets[self.crusher_id].jobs_done
        if self._jobs_done is not None and done != self._jobs_done:
            self._renewed_day = obs.day
        self._jobs_done = done
        self._estimate_is_stale = obs.day <= self._renewed_day
        crusher_permit = obs.assets[self.crusher_id].permit
        if crusher_permit is not Permit.NONE and self._permit_run_start is None:
            self._permit_run_start = obs.quarter - 1
        signals = self._signals(obs)
        for pid, monitor in self._monitors.items():
            self._robustness[pid].append(monitor.push(signals))
        gas = self._gas.push(signals)
        alarm = self._alarm.push(signals)
        rul = self._rul.push(signals)
        for aid, view in obs.assets.items():
            self._prev_op[aid] = view.op
        row = obs.quarter - 1
        crusher = obs.assets[self.crusher_id]
        if crusher.occupied:
            self._last_occupied_row = row
        if crusher.permit is Permit.NONE:
            self._permit_run_start = None
        return gas, alarm, rul

    def _signals(self, obs: Observation) -> dict[str, float]:
        """The atoms of the specification (``spec.properties.signals_from_trace``), one quarter.

        The observation at decision quarter ``q`` describes the trace row ``q - 1``.
        """
        retest_min = float(self.spec.retest_q * QUARTER_MIN)
        row = obs.quarter - 1
        crusher = obs.assets[self.crusher_id]
        pump_a, pump_b = (obs.assets[p] for p in self.pump_ids)
        conveyor = obs.assets[self.conveyor_id]

        def sign(flag: bool) -> float:
            return 1.0 if flag else -1.0

        out: dict[str, float] = {}
        roles = (
            ("crusher", crusher),
            ("pump_a", pump_a),
            ("pump_b", pump_b),
            ("conveyor", conveyor),
        )
        for role, view in roles:
            out[f"{role}.working"] = sign(view.op is Op.WORK)
            out[f"{role}.iso_proven"] = sign(self._is_proven(view))
        occupied = crusher.occupied
        out["crusher.occupied"] = sign(occupied)
        out["crusher.in_space"] = HOLDS_MARGIN if occupied else -HOLDS_MARGIN
        out["crusher.alarm_occupied"] = sign(crusher.alarm and occupied)
        age_min = (
            GAS_AGE_NEVER_MIN
            if crusher.gas_age_q >= _GAS_CAP_Q
            else float(crusher.gas_age_q - 1) * QUARTER_MIN
        )
        out[_GAS_MARGIN] = retest_min - age_min
        if self._last_occupied_row is None:
            empty_min = _NEVER_EMPTY_MIN
        else:
            empty_min = float(row - 1 - self._last_occupied_row) * QUARTER_MIN
        out["crusher.gas_since_break"] = empty_min - age_min
        if crusher.permit is Permit.VALID:
            # time left in the issuing shift, and the specification's cap of ``permit_max_h`` from
            # the first quarter of the unbroken run of permit values other than NONE
            left = float(SHIFT_Q - (row % SHIFT_Q)) * QUARTER_MIN
            age = row - (self._permit_run_start if self._permit_run_start is not None else row)
            cap = self.spec.permit_max_h * 60.0 - age * QUARTER_MIN
            out["crusher.permit_ok"] = min(left, cap)
        else:
            out["crusher.permit_ok"] = -1.0
        available = sum(1 for v in (pump_a, pump_b) if v.op in (Op.RUN, Op.STOP))
        out["pumps.available"] = available - 0.5
        failed = crusher.op is Op.FAIL and self._prev_op.get(self.crusher_id) is not Op.FAIL
        point = self.point_rul(crusher)
        out["crusher.no_failure"] = (
            -1.0 if failed else (1.0 if point is None else max(point * 24.0, 1e-3))
        )
        out[_ALARM_CLEAR] = sign(not crusher.alarm)
        out[_RUL_MARGIN] = _NO_ESTIMATE if point is None else point - STOP_RUL_DAYS
        return out
