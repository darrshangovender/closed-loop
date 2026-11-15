"""Independent reference relations for the three safety games (packets P5b and P5d-b).

Written from the English of ADR 0001 sections 4.1 and 5 and ADR 0003 A9, A11 and A12, and (for the
first two games) the prose specification in the P5a note, without reading ``abstraction.py`` or
``game.py``. It imports nothing from them, so a slip copied between the two would have to be made
twice. The conveyor game and the A12 changes to the other two were written without reading any
production code or any note about it: the production model was being changed at the same time from
the same English.

The shape differs on purpose from a table of rules. Every nondeterministic fact is a named *choice*:
whether the plant refuses the command, whether the holder is present, and five environment bits
(test result, alarm onset, health step, handover announcement, permit reaching its limit). A step is
the image, over the whole choice space, of deterministic functions: the command stage, then the
environment stage, then hazard predicates named after the properties they protect. ``outcomes`` is
the set of results, so "every combination is an outcome" holds by construction and cannot be
forgotten for one branch. Each rule is its own small method so a test can break exactly one of them
by subclassing.

Stated readings (the places the English leaves room; the P5b and P5d notes say which side is right):

* A step is the controller's command, then the environment's move. Facts about "after the command"
  (is the component running, is anyone inside) are read from the state between the two.
* ``NOOP``, ``STOP`` and ``EXIT`` are never refused. Any other command is also an outcome as
  ``NOOP``.
* A command whose guard fails is a no-op, except where the specification names it a violation.
* Renewal is the environment's choice (ADR 0003 A11): completing the work on a component renews
  it (its health bin becomes 0) or does not (a job raised for the wrong failure mode finishes and
  leaves the worn part in place). Both are outcomes, in both games. (This rule was added by P5c,
  from the wording of the amendment, in this file and in ``abstraction.py`` together; every other
  rule here was written without reading that file.)
* A permit lives at most 12 hours from issue (ADR 0003 A12.1). The game has no clock, so the
  environment may end a live permit at any announced handover: a permit that is not ``NONE``
  becomes ``REVAL`` or ``NONE``, both outcomes, and the next quarter is announced or not, both
  outcomes (a handover and a permit limit can fall on consecutive quarters). ``ISSUE`` works with
  the permit ``NONE`` and the asset in ``ISO``, or in ``WORK`` with nobody inside (the ADR's text
  after this file's reading was confirmed; the first wording left out "nobody inside").
* A locked pump is not available (A12.2). The pump has six ops, eight commands and no ``ISOLATE``;
  isolation is ``LOCK`` then ``PROVE``. The conveyor (A12.3) has the same per-asset rules and no
  pair rule, so nothing but the two controller rules can be violated in it.
* A violation is the sink ``"VIOL"``. In the crusher game an uncontrolled failure is a violation;
  in the pump game and the conveyor game it is the state ``("FAIL", 3)``, and only the availability
  rule (pump pair) and the controller rules can be violated.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from itertools import product
from typing import NamedTuple, Protocol

VIOL = "VIOL"
TOP_BIN = 3
BINS = (0, 1, 2, 3)


class ReferenceModel(Protocol):
    """The relation alone: every outcome of command ``c`` in state ``s`` (may contain ``VIOL``)."""

    def outcomes(self, s: Hashable, c: str) -> frozenset[Hashable | str]: ...


class ReferenceGame(ReferenceModel, Protocol):
    """A reference relation plus what the checker needs to walk it.

    The extra members match ``AbstractModel`` on purpose: a reference (or a sabotaged subclass of
    one) can be handed to a solver written against that interface, which is how the sabotage suite
    builds a shield from a wrong model.
    """

    name: str

    def states(self) -> Sequence[Hashable]: ...
    def commands(self) -> Sequence[str]: ...
    def step(self, s: Hashable, c: str) -> frozenset[Hashable | str]: ...
    def initial(self) -> Hashable: ...
    def observable(self, s: Hashable) -> Hashable: ...
    def health(self, s: Hashable) -> int | tuple[int, ...]: ...


# --------------------------------------------------------------------------------------------
# Crusher job game
# --------------------------------------------------------------------------------------------

CRUSHER_OPS = ("RUN", "STOP", "LOCKED", "ISO", "WORK", "FAIL")
PERMITS = ("NONE", "VALID", "REVAL")
CRUSHER_COMMANDS = (
    "NOOP", "START", "STOP", "LOCK", "PROVE", "ISSUE", "REVAL",
    "BEGIN", "TEST", "ENTER", "EXIT", "COMPLETE", "CLOSE", "UNISO",
)  # fmt: skip
_NEVER_REFUSED = frozenset({"NOOP", "STOP", "EXIT"})  # STOP and EXIT are A_exec; NOOP is a no-op

# (executed command, op before) -> op after. Anything not listed leaves the op alone.
_CRUSHER_OP_MOVES = {
    ("START", "STOP"): "RUN",
    ("STOP", "RUN"): "STOP",
    ("LOCK", "STOP"): "LOCKED",
    ("LOCK", "FAIL"): "LOCKED",
    ("PROVE", "LOCKED"): "ISO",
    ("BEGIN", "ISO"): "WORK",
    ("COMPLETE", "WORK"): "ISO",
    ("UNISO", "ISO"): "STOP",
}


class _C(NamedTuple):
    op: str
    h: int
    permit: str
    occ: int
    gas: int
    alarm: int
    soon: int


class _Mid(NamedTuple):
    """The crusher between the controller's move and the environment's."""

    op: str
    h: int
    permit: str
    occ: int
    gas: int
    tested: bool


class _Env(NamedTuple):
    """What the environment decides after the controller's move (every combination happens)."""

    test_alarm: bool  # an atmosphere test comes back with the alarm on
    alarm_onset: bool  # the alarm comes on while someone is inside, without a test
    health_up: bool  # a running component moves up one bin
    soon_next: bool  # a handover is announced for the end of the next tick
    permit_ends: bool  # at an announced step the permit has reached its 12-hour limit (A12.1)


_BOTH = (False, True)
_ENV_CHOICES = tuple(_Env(*f) for f in product(_BOTH, repeat=5))


class CrusherReference:
    """The crusher job game of ADR 0001 section 4.1, with ``retest_q`` quarters between gas tests.

    State ``(op, h, permit, occ, gas, alarm, soon)``; ``gas`` runs 1..retest_q+1, where 1 is "a
    test came out of the last tick" and retest_q+1 is stale or never tested.
    """

    name = "crusher-reference"

    def __init__(self, retest_q: int = 2) -> None:
        if retest_q < 1:
            raise ValueError(f"retest_q must be at least 1, got {retest_q}")
        self.retest_q = retest_q
        self._cache: dict[tuple[Hashable, str], frozenset[Hashable | str]] = {}

    # -- the model interface -------------------------------------------------------------

    @property
    def stale(self) -> int:
        return self.retest_q + 1

    def commands(self) -> tuple[str, ...]:
        return CRUSHER_COMMANDS

    def states(self) -> list[Hashable]:
        gases = range(1, self.stale + 1)
        return list(product(CRUSHER_OPS, BINS, PERMITS, (0, 1), gases, (0, 1), (0, 1)))

    def initial(self) -> Hashable:
        return ("RUN", 0, "NONE", 0, self.stale, 0, 0)

    def observable(self, s: Hashable) -> Hashable:
        c = self._parse(s)
        return (c.op, c.permit, c.occ, c.gas, c.alarm, c.soon)

    def health(self, s: Hashable) -> int:
        return self._parse(s).h

    def step(self, s: Hashable, c: str) -> frozenset[Hashable | str]:
        return self.outcomes(s, c)

    def outcomes(self, s: Hashable, c: str) -> frozenset[Hashable | str]:
        if c not in CRUSHER_COMMANDS:
            raise ValueError(f"unknown crusher command {c!r}")
        key = (s, c)
        hit = self._cache.get(key)
        if hit is None:
            st = self._parse(s)
            moves = {
                renewed
                for e in self._executions(c)
                for holder in _BOTH
                for renewed in self._renewal_choices(st, e, self._command_stage(st, e, holder))
            }
            hit = frozenset(
                VIOL if mid is None else self._environment_and_hazards(st, mid, env)
                for mid in moves
                for env in self._env_choices()
            )
            self._cache[key] = hit
        return hit

    def _executions(self, c: str) -> tuple[str, ...]:
        """What the plant may do with command ``c``: carry it out or, if refusable, do nothing."""
        return (c,) if c in _NEVER_REFUSED else (c, "NOOP")

    def _env_choices(self) -> tuple[_Env, ...]:
        return _ENV_CHOICES

    def _renewal_choices(self, st: _C, e: str, mid: _Mid | None) -> tuple[_Mid | None, ...]:
        """Completing the work renews the component or leaves the worn part in place (A11)."""
        if mid is not None and e == "COMPLETE" and st.op == "WORK":
            return (mid._replace(h=0), mid)
        return (mid,)

    def _environment_and_hazards(self, st: _C, mid: _Mid, env: _Env) -> Hashable | str:
        after = self._environment_stage(st, mid, env)
        if after is None:
            return VIOL
        if after.occ == 1 and (
            self._p2_violated(after) or self._p3_violated(after) or self._p6_violated(st)
        ):
            return VIOL
        return tuple(after)

    # -- command stage ---------------------------------------------------------------------

    def _command_stage(self, st: _C, e: str, holder_present: bool) -> _Mid | None:
        if self._controller_violation(st, e):
            return None
        permit = st.permit
        if e == "ISSUE" and self._issue_applies(st):
            permit = "VALID"
        elif e == "REVAL" and st.permit == "REVAL" and holder_present:
            permit = "VALID"
        elif e == "CLOSE" and st.op == "ISO" and st.permit != "NONE":
            permit = "NONE"
        occ = 1 if e == "ENTER" else 0 if e == "EXIT" else st.occ
        gas = self.stale if e == "EXIT" else st.gas
        return _Mid(
            op=_CRUSHER_OP_MOVES.get((e, st.op), st.op),
            h=st.h,
            permit=permit,
            occ=occ,
            gas=gas,
            tested=e == "TEST" and st.op == "WORK",
        )

    def _issue_applies(self, st: _C) -> bool:
        """None on record, and the asset isolated or its work open with nobody inside (A12.1)."""
        return st.permit == "NONE" and (st.op == "ISO" or (st.op == "WORK" and st.occ == 0))

    def _controller_violation(self, st: _C, e: str) -> bool:
        """The executed command is itself a violation (P1 and the job procedure)."""
        if e == "START":
            return st.op in ("LOCKED", "ISO", "WORK")  # breaks isolation
        if e == "BEGIN":
            return st.op in ("RUN", "STOP", "LOCKED", "FAIL") or (
                st.op == "ISO" and st.permit != "VALID"
            )  # P1: work on an asset whose zero energy is not proven, or with no valid permit
        if e == "ENTER":
            return st.occ == 0 and not self._may_enter(st)
        if e == "COMPLETE":
            return st.occ == 1  # finishing with a person inside
        if e == "CLOSE":
            return st.op == "WORK"  # closing the permit while the work is open
        if e == "UNISO":
            return st.op == "ISO" and st.permit != "NONE"  # lifting isolation under a live permit
        return False

    def _may_enter(self, st: _C) -> bool:
        return st.op == "WORK" and st.permit == "VALID" and st.gas == 1 and st.alarm == 0

    # -- environment stage -----------------------------------------------------------------

    def _environment_stage(self, before: _C, mid: _Mid, env: _Env) -> _C | None:
        if mid.tested:
            gas, alarm = 1, int(env.test_alarm)
        else:
            gas = min(mid.gas + 1, self.stale)
            alarm = before.alarm | int(mid.occ == 1 and env.alarm_onset)

        h = mid.h
        if mid.op == "RUN" and env.health_up:
            if h == TOP_BIN:
                return None  # uncontrolled failure of the critical asset: P5
            h += 1

        if before.soon == 1:  # the announced handover happens at the end of this tick
            permit = self._permit_after_handover(mid.permit, env.permit_ends)
        else:
            permit = mid.permit
        return _C(mid.op, h, permit, mid.occ, gas, alarm, self._announcement(before, env))

    def _announcement(self, before: _C, env: _Env) -> int:
        """Whether the next quarter is announced. A handover and a permit limit can fall on
        consecutive quarters, so an announced step may be followed by another (A12.1)."""
        return int(env.soon_next)

    def _permit_after_handover(self, permit: str, ends: bool) -> str:
        """A live permit is revalidated by the incoming holder, or it has reached its limit and is
        spent (A12.1). There is no clock, so the environment may end it at any announced step."""
        if permit == "NONE":
            return "NONE"
        return "NONE" if ends else "REVAL"

    # -- hazards on an occupied state (checked last) -------------------------------------

    def _p2_violated(self, after: _C) -> bool:
        return after.permit != "VALID"

    def _p3_violated(self, after: _C) -> bool:
        return after.gas == self.stale

    def _p6_violated(self, before: _C) -> bool:
        return before.alarm == 1  # already on before this tick and someone is still inside

    # -- helpers -----------------------------------------------------------------------

    def _parse(self, s: Hashable) -> _C:
        if not isinstance(s, tuple) or len(s) != 7:
            raise ValueError(f"not a crusher state: {s!r}")
        c = _C(*s)
        if (
            c.op not in CRUSHER_OPS
            or c.h not in BINS
            or c.permit not in PERMITS
            or c.occ not in (0, 1)
            or c.gas not in range(1, self.stale + 1)
            or c.alarm not in (0, 1)
            or c.soon not in (0, 1)
        ):
            raise ValueError(f"not a crusher state: {s!r}")
        return c


# --------------------------------------------------------------------------------------------
# One pump's rules (shared by the pump pair and the conveyor)
# --------------------------------------------------------------------------------------------

PUMP_OPS = ("RUN", "STOP", "LOCKED", "ISO", "WORK", "FAIL")
PUMP_COMMANDS_ONE = ("NOOP", "START", "STOP", "LOCK", "PROVE", "BEGIN", "COMPLETE", "UNISO")
_PUMP_NEVER_REFUSED = frozenset({"NOOP", "STOP"})  # A_exec; every other command may be refused
_AVAILABLE = frozenset({"RUN", "STOP"})  # a stopped pump counts as available; a locked one does not
_NOT_PROVEN = ("RUN", "STOP", "LOCKED", "FAIL")  # ops in which zero energy is not proven

_PUMP_OP_MOVES = {
    ("START", "STOP"): "RUN",
    ("STOP", "RUN"): "STOP",
    ("LOCK", "STOP"): "LOCKED",
    ("LOCK", "FAIL"): "LOCKED",
    ("PROVE", "LOCKED"): "ISO",
    ("BEGIN", "ISO"): "WORK",
    ("COMPLETE", "WORK"): "ISO",
    ("UNISO", "ISO"): "STOP",
}

_PumpResult = tuple[str, int] | None  # None: the pump's own move is a violation


class _PumpRules:
    """The per-asset rules of ADR 0003 A12.2, shared with the conveyor (A12.3).

    A pump is ``(op, h)``. A command is carried out or, if it may be refused, becomes ``NOOP``; the
    pump then steps up a bin if it is running (an environment choice) and a completed job renews it
    or not (A11). ``_pump`` is the set of results, ``None`` standing for a violation of a controller
    rule.
    """

    def __init__(self) -> None:
        self._one: dict[tuple[str, int, str], frozenset[_PumpResult]] = {}

    def _pump(self, op: str, h: int, cmd: str) -> frozenset[_PumpResult]:
        key = (op, h, cmd)
        hit = self._one.get(key)
        if hit is None:
            hit = frozenset(
                self._pump_resolve(op, h, e, up, renewed)
                for e in self._pump_executions(cmd)
                for up in _BOTH
                for renewed in _BOTH
            )
            self._one[key] = hit
        return hit

    def _pump_executions(self, cmd: str) -> tuple[str, ...]:
        return (cmd,) if cmd in _PUMP_NEVER_REFUSED else (cmd, "NOOP")

    def _pump_resolve(self, op: str, h: int, e: str, up: bool, renewed: bool) -> _PumpResult:
        if self._pump_violation(op, e):
            return None
        new_op = _PUMP_OP_MOVES.get((e, op), op)
        new_h = 0 if e == "COMPLETE" and op == "WORK" and renewed else h
        if new_op == "RUN" and up:
            if new_h == TOP_BIN:
                return ("FAIL", TOP_BIN)  # a failed pump is unavailable, not itself a violation
            new_h += 1
        return (new_op, new_h)

    def _pump_violation(self, op: str, e: str) -> bool:
        if e == "START":
            return op in ("LOCKED", "ISO", "WORK")  # energising a locked or isolated asset
        if e == "BEGIN":
            return op in _NOT_PROVEN  # P1: work on an asset whose zero energy is not proven
        return False


# --------------------------------------------------------------------------------------------
# Pump pair game
# --------------------------------------------------------------------------------------------


class PumpPairReference(_PumpRules):
    """The duty pair of ADR 0001 section 4.1: state ``(op1, h1, op2, h2)``, command ``"a1|a2"``.

    The pumps move independently; a pair of outcomes is every combination of the two pumps'
    outcomes, and it is a violation if either pump's own move was one or if neither pump is
    available afterwards (P4).
    """

    name = "pump-pair-reference"

    def __init__(self) -> None:
        super().__init__()
        self._cache: dict[tuple[Hashable, str], frozenset[Hashable | str]] = {}

    def commands(self) -> tuple[str, ...]:
        return tuple(f"{a}|{b}" for a in PUMP_COMMANDS_ONE for b in PUMP_COMMANDS_ONE)

    def states(self) -> list[Hashable]:
        return list(product(PUMP_OPS, BINS, PUMP_OPS, BINS))

    def initial(self) -> Hashable:
        return ("RUN", 0, "RUN", 0)

    def observable(self, s: Hashable) -> Hashable:
        op1, _h1, op2, _h2 = self._parse(s)
        return (op1, op2)

    def health(self, s: Hashable) -> tuple[int, int]:
        _op1, h1, _op2, h2 = self._parse(s)
        return (h1, h2)

    def step(self, s: Hashable, c: str) -> frozenset[Hashable | str]:
        return self.outcomes(s, c)

    def outcomes(self, s: Hashable, c: str) -> frozenset[Hashable | str]:
        a1, _, a2 = c.partition("|")
        if a1 not in PUMP_COMMANDS_ONE or a2 not in PUMP_COMMANDS_ONE or c != f"{a1}|{a2}":
            raise ValueError(f"unknown pump-pair command {c!r}")
        key = (s, c)
        hit = self._cache.get(key)
        if hit is None:
            op1, h1, op2, h2 = self._parse(s)
            hit = frozenset(
                self._pair(x, y) for x in self._pump(op1, h1, a1) for y in self._pump(op2, h2, a2)
            )
            self._cache[key] = hit
        return hit

    # -- the pair -----------------------------------------------------------------------

    def _pair(self, x: _PumpResult, y: _PumpResult) -> Hashable | str:
        if x is None or y is None or self._p4_violated(x[0], y[0]):
            return VIOL
        return (x[0], x[1], y[0], y[1])

    def _p4_violated(self, op1: str, op2: str) -> bool:
        return not (self._available(op1) or self._available(op2))

    def _available(self, op: str) -> bool:
        return op in _AVAILABLE

    def _parse(self, s: Hashable) -> tuple[str, int, str, int]:
        if (
            not isinstance(s, tuple)
            or len(s) != 4
            or s[0] not in PUMP_OPS
            or s[2] not in PUMP_OPS
            or s[1] not in BINS
            or s[3] not in BINS
        ):
            raise ValueError(f"not a pump-pair state: {s!r}")
        return (s[0], s[1], s[2], s[3])


# --------------------------------------------------------------------------------------------
# Conveyor game (ADR 0003 A12.3)
# --------------------------------------------------------------------------------------------


class ConveyorReference(_PumpRules):
    """The conveyor CV-101: state ``(op, h)``, the pump's eight commands and the pump's rules.

    There is no pair rule and a failure is not a violation, so the only violations are the two
    controller rules: ``START`` in ``LOCKED``, ``ISO`` or ``WORK``, and ``BEGIN`` in ``RUN``,
    ``STOP``, ``LOCKED`` or ``FAIL`` (work on an asset whose zero energy is not proven, P1).
    """

    name = "conveyor-reference"

    def __init__(self) -> None:
        super().__init__()
        self._cache: dict[tuple[Hashable, str], frozenset[Hashable | str]] = {}

    def commands(self) -> tuple[str, ...]:
        return PUMP_COMMANDS_ONE

    def states(self) -> list[Hashable]:
        return list(product(PUMP_OPS, BINS))

    def initial(self) -> Hashable:
        return ("RUN", 0)

    def observable(self, s: Hashable) -> Hashable:
        op, _h = self._parse(s)
        return (op,)

    def health(self, s: Hashable) -> int:
        return self._parse(s)[1]

    def step(self, s: Hashable, c: str) -> frozenset[Hashable | str]:
        return self.outcomes(s, c)

    def outcomes(self, s: Hashable, c: str) -> frozenset[Hashable | str]:
        if c not in PUMP_COMMANDS_ONE:
            raise ValueError(f"unknown conveyor command {c!r}")
        key = (s, c)
        hit = self._cache.get(key)
        if hit is None:
            op, h = self._parse(s)
            hit = frozenset(VIOL if r is None else r for r in self._pump(op, h, c))
            self._cache[key] = hit
        return hit

    def _parse(self, s: Hashable) -> tuple[str, int]:
        if not isinstance(s, tuple) or len(s) != 2 or s[0] not in PUMP_OPS or s[1] not in BINS:
            raise ValueError(f"not a conveyor state: {s!r}")
        return (s[0], s[1])
