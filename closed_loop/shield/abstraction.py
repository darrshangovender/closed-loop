"""The three finite safety games the shield is synthesised on (SYNTHETIC toy plant).

This docstring is the specification of the abstraction. It is written so that an independent
engineer can build a reference model from it and from ADR 0001 sections 4.1 and 5 and ADR 0003
A9 to A12 without opening the code below. Any rule that the code applies and this text does not
state is a defect here.

Time and order. One step is one quarter hour. Inside a step the controller issues one command and
then the environment moves. The environment is adversarial: where a rule says "any of", every
choice is an outcome of ``step``. ``step(s, c)`` returns the set of all outcomes, which may contain
the sink ``VIOL`` (a covered property was violated; nothing follows from it).

What the abstraction takes as given (modelling assumptions, not results):

* A_exec. ``STOP`` and ``EXIT`` are always executed. Every other command except ``NOOP`` may be
  refused by the plant, and a refused command behaves exactly as ``NOOP``. Both cases are
  outcomes. "Stop is always safe" is this assumption together with the ordering above: a stopped
  component cannot fail in the tick it is stopped, because the environment moves after the command.
  For a pump or the conveyor only ``STOP`` and ``NOOP`` are never refused (they have no ``EXIT``).
* A_phys. The hidden health bin of a running component rises by at most one per tick, and an
  uncontrolled failure happens only from bin 3 while running. A component that is not running does
  not degrade. The abstraction cannot see a load shift that skips a bin; that is measured by the
  conformance test, not assumed away.
* A_iso. Isolation holds once proven: no energy returns while the state is ISO or WORK.
* A_soon. The plant announces the end of a permit one tick ahead through ``soon``: a permit ends
  only at the end of a tick for which ``soon`` was 1 (a shift handover, or the permit reaching its
  12 hours from issue). The abstraction has no clock; it lets the environment announce at any tick
  and end the permit at the announced one, which over-approximates both causes. The two can fall
  on consecutive quarters, so the plant can announce twice in a row and the game contains that.
* A_dec. The crusher job, the pump pair and the conveyor do not interact (ADR 0001, section 4.5;
  the conveyor's atoms read only the conveyor, ADR 0003 A12.3), so each is solved on its own.

Crusher job game (``CrusherJobModel``)
--------------------------------------
A state is the tuple ``(op, h, permit, occ, gas, alarm, soon)``.

* ``op`` in RUN, STOP, LOCKED, ISO, WORK, FAIL. LOCKED: lock applied, zero energy not yet
  proven. ISO: zero energy proven. FAIL: failed, not running.
* ``h`` in 0..3, the hidden health bin (3 is critical). It is the only hidden variable.
* ``permit`` in NONE, VALID, REVAL. REVAL: a handover happened since issue, so the incoming
  holder must revalidate. NONE: no permit yet, or it was closed, or it was spent at its limit.
* ``occ`` in 0, 1: a person is inside the confined space.
* ``gas`` in 1..R+1 where R is ``retest_q`` (2 quarters, 30 minutes, by default). It counts quarters
  since the last atmosphere test; 1 means a test came out of the previous tick and R+1 means
  stale or never.
* ``alarm`` in 0, 1: the gas alarm is on. It stays on until a later test comes out clean.
* ``soon`` in 0, 1: this tick ends a shift (a handover) or the permit's 12 hours from its issue.
  It is the plant's ``view.soon``.

The observable part is ``(op, permit, occ, gas, alarm, soon)`` (``h`` dropped). The initial state
is ``(RUN, 0, NONE, 0, R+1, 0, 0)``. The commands are, in this order: NOOP, START, STOP, LOCK,
PROVE, ISSUE, REVAL, BEGIN, TEST, ENTER, EXIT, COMPLETE, CLOSE, UNISO. Raw size is 1,728 states
for R = 2.

Effect of the command that is executed (``e`` is the command, or NOOP if it was refused). A rule
whose condition does not hold leaves the state unchanged, except where it says VIOL.

* START: STOP becomes RUN. In LOCKED, ISO or WORK it is VIOL (energising under lock).
* STOP: RUN becomes STOP.
* LOCK: STOP or FAIL becomes LOCKED. PROVE: LOCKED becomes ISO.
* ISSUE: with permit NONE, the permit becomes VALID when ``op`` is ISO, or when ``op`` is WORK and
  nobody is inside. Otherwise nothing changes (ADR 0003 A12.1: a permit spent at its limit is
  replaced in WORK, and the plant refuses an ISSUE with someone inside; NONE with someone inside is
  the sink).
* REVAL: with permit REVAL, the incoming holder is present or not (both outcomes); present makes
  the permit VALID. (Absent leaves the state as it was, which is also what a refused REVAL does,
  so this choice adds no outcome of its own.)
* BEGIN: ISO with permit VALID becomes WORK. In RUN, STOP, LOCKED or FAIL, or in ISO without a
  VALID permit, it is VIOL (work on an energised or unpermitted asset, P1 and P2).
* TEST: in WORK, an atmosphere test happens (below). Elsewhere nothing.
* ENTER: when nobody is inside, it is VIOL unless ``op`` is WORK, the permit is VALID, ``gas`` is 1
  (a test in the previous tick) and ``alarm`` is 0; then ``occ`` becomes 1. With someone already
  inside it changes nothing. The permit and alarm conditions (and the test condition when R = 2)
  are also implied by the last rule below; they are stated here so the entry condition is in one
  place, and removing one of them alone does not change ``step``.
* EXIT: ``occ`` becomes 0 and ``gas`` becomes R+1 (a break; a fresh test is needed to re-enter).
* COMPLETE: VIOL while someone is inside; otherwise WORK becomes ISO and ``h`` becomes 0 or stays
  as it was (both are outcomes). Renewal is the environment's choice, not the controller's: a job
  raised for the wrong failure mode completes and does not renew the component (ADR 0003 A3,
  A11), and the controller cannot tell which happened. (The feasibility spike always renewed.)
* CLOSE: VIOL in WORK; ISO with a permit that is not NONE makes the permit NONE.
* UNISO: ISO with permit NONE becomes STOP; ISO with any other permit is VIOL.

Then the environment moves, in all combinations:

* Test. If a test happened, ``gas`` becomes 1 and ``alarm`` becomes 0 or 1 (either result).
  Otherwise ``gas`` becomes min(gas + 1, R+1), and if someone is inside the alarm may come on; it
  never goes off without a test.
* Health. A RUN component stays at ``h`` or rises to ``h + 1``. From ``h`` = 3 the other outcome is
  an uncontrolled failure, which is VIOL (P5). Other ``op`` values keep ``h``.
* Announcement and end of the permit. If ``soon`` was 1, the announced end happens: a permit that
  is not NONE becomes REVAL **or** NONE (both are outcomes: a handover asks for revalidation, the
  12 hours from issue spend the permit, and one tick cannot say which) and ``soon`` becomes 0 **or**
  1 (both are outcomes: a handover and a permit limit can fall on consecutive quarters). If ``soon``
  was 0, ``soon`` becomes 0 or 1 (any step may announce one). Everything else in this step is
  unchanged.

Last, the state reached is VIOL if someone is inside and any of these holds: the new ``gas`` is
R+1 (P3, the re-test period was missed), the new permit is not VALID (P2, expired, spent or
awaiting revalidation), or ``alarm`` was already on before this tick (P6, one-tick version: the
alarm is answered by ``EXIT``, which is always executed, so being inside one tick after an alarm
is a controller failure). An alarm that comes on during this tick is not yet a violation; the
next command must be ``EXIT``.

The permit has no clock in the state. CORP-HSE-004 clause 4.1 limits a permit to 12 hours from
its issue, and clause 5.1 records a handover on the permit without renewing the limit (ADR 0003
A12.1), so a shift length does not cover P2's limit: a permit can be spent in the middle of a
shift, and a revalidation does not move that. The abstraction stays finite by over-approximating:
the environment may end a permit at any announced step, which includes every step at which the
plant's limit falls (A_soon). Hot work, work at height and the safety officer of ADR 0001
section 2 are not in the abstraction.

Pump pair game (``PumpPairModel``) and conveyor game (``ConveyorModel``)
------------------------------------------------------------------------
A pump, or the conveyor, is a pair ``(op, h)`` with ``op`` in RUN, STOP, LOCKED, ISO, WORK, FAIL
(24 states) and ``h`` the hidden bin. Neither has a permit, a gas test or an alarm in the game. The
commands of one asset are, in this order: NOOP, START, STOP, LOCK, PROVE, BEGIN, COMPLETE, UNISO;
the plant commands ISSUE, REVAL, TEST, ENTER, EXIT, CLOSE and DERATE are the abstract NOOP.

Per asset, for the executed command ``e`` (a refused one behaves as NOOP; only STOP and NOOP are
never refused):

* START: STOP becomes RUN; in LOCKED, ISO or WORK it is VIOL. STOP: RUN becomes STOP.
* LOCK: STOP or FAIL becomes LOCKED. PROVE: LOCKED becomes ISO. UNISO: ISO becomes STOP.
* BEGIN: ISO becomes WORK; in RUN, STOP, LOCKED or FAIL it is VIOL (work on an asset whose
  isolation is not proven, P1). COMPLETE: WORK becomes ISO and ``h`` becomes 0 or stays as it was
  (both are outcomes, as for the crusher: ADR 0003 A11).
* Then a RUN asset stays at ``h`` or rises to ``h + 1``; from ``h`` = 3 the other outcome is FAIL
  (a failure is not VIOL by itself).

The pump pair is a state ``(op1, h1, op2, h2)`` (576 raw states). Its observable part is
``(op1, op2)``, the hidden part ``(h1, h2)``, the initial state ``(RUN, 0, RUN, 0)``. A command is
the string ``"a1|a2"`` (64 of them) where ``a1`` and ``a2`` are commands of one pump; each pump's
command is refused or executed on its own. The outcomes of the pair are all combinations of the
two pumps' outcomes. A combination is VIOL when either pump hit a VIOL rule, or when both pumps end
in LOCKED, ISO, WORK or FAIL (P4: no pump available; only RUN and STOP are available, a locked pump
cannot be started until the lock is removed, ADR 0003 A12.2).

The conveyor is a state ``(op, h)`` (24 raw states), observable ``(op,)``, hidden ``h``, initial
``(RUN, 0)``, with the 8 commands above. It has no pair rule: its only VIOL rules are START in
LOCKED, ISO or WORK and BEGIN in RUN, STOP, LOCKED or FAIL (ADR 0003 A12.3: P1 reads CV-101).
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from functools import cache
from itertools import product
from typing import Protocol, cast

from closed_loop.core.types import N_BINS, Cmd, Op, Permit

VIOL = "VIOL"

CrusherState = tuple[str, int, str, int, int, int, int]  # (op, h, permit, occ, gas, alarm, soon)
PairState = tuple[str, int, str, int]  # (op1, h1, op2, h2)
AssetState = tuple[str, int]  # (op, h) of a pump or the conveyor
PumpOutcome = tuple[str, int] | str  # (op, h), or VIOL
ABSTRACTION_VERSION = "2"  # 1: ADR 0003 A9 to A11; 2: A12 (permit limit, locked pump, conveyor)

HMAX = N_BINS - 1

CRUSHER_OPS: tuple[str, ...] = tuple(op.value for op in Op)
PERMITS: tuple[str, ...] = tuple(p.value for p in Permit)
CRUSHER_COMMANDS: tuple[str, ...] = tuple(c.value for c in Cmd if c is not Cmd.DERATE)

# A pump and the conveyor share one alphabet and one set of per-asset rules (ADR 0003 A12.2, A12.3).
PUMP_OPS: tuple[str, ...] = tuple(op.value for op in Op)
PUMP_COMMANDS: tuple[str, ...] = (
    "NOOP", "START", "STOP", "LOCK", "PROVE", "BEGIN", "COMPLETE", "UNISO",
)  # fmt: skip
PUMP_UNAVAILABLE: frozenset[str] = frozenset({"LOCKED", "ISO", "WORK", "FAIL"})
# The feasibility spike's pump: five ops (no LOCKED), one ISOLATE in place of LOCK and PROVE.
_SPIKE_PUMP_OPS: tuple[str, ...] = tuple(op.value for op in Op if op is not Op.LOCKED)
_SPIKE_PUMP_COMMANDS: tuple[str, ...] = (
    "NOOP", "START", "STOP", "ISOLATE", "BEGIN", "COMPLETE", "UNISO",
)  # fmt: skip

# A_exec: the plant never refuses these two (core.types.FORCED_CMDS, by value).
FORCED: frozenset[str] = frozenset({"STOP", "EXIT"})

Outcomes = frozenset[Hashable | str]


class AbstractModel(Protocol):
    name: str

    def states(self) -> Iterable[Hashable]: ...

    def commands(self) -> Sequence[str]: ...

    def step(self, s: Hashable, c: str) -> frozenset[Hashable | str]: ...  # may contain VIOL

    def initial(self) -> Hashable: ...

    def observable(self, s: Hashable) -> Hashable: ...  # drops the hidden health bin(s)

    def health(self, s: Hashable) -> int | tuple[int, ...]: ...


class CrusherJobModel:
    """The confined-space crusher job. Rules: see the module docstring.

    ``retest_q`` is the gas re-test period in quarters (``SpecParams.retest_q``); 8 builds the
    stale-spec shield (the 2-hour interval of the superseded revision).

    ``spike_semantics=True`` reproduces the transition relation of the ADR 0001 feasibility spike,
    which differs in six places: an alarm that comes on while someone is inside is itself VIOL,
    an occupied state with permit NONE is not VIOL, ``COMPLETE`` always renews the component,
    ``ISSUE`` works in ISO only, a handover turns a permit into REVAL and never into NONE, and
    ``soon`` is 0 after an announced step (the last three are ADR 0003 A12.1). The first makes
    ``ENTER`` unsafe in every state (an environment move can always raise the alarm), so no shield
    built on it can ever admit an entry. It exists only so that the difference can be measured;
    never build a shield from it.
    """

    def __init__(self, retest_q: int = 2, *, spike_semantics: bool = False) -> None:
        if isinstance(retest_q, bool) or not isinstance(retest_q, int) or retest_q < 1:
            raise ValueError(f"retest_q must be an integer of at least 1, got {retest_q!r}")
        self.retest_q = retest_q
        self.spike_semantics = spike_semantics
        self._stale = retest_q + 1
        self._commands = CRUSHER_COMMANDS
        self._command_set = frozenset(CRUSHER_COMMANDS)
        self.name = f"crusher_job[retest_q={retest_q}]"

    def parameters(self) -> Mapping[str, str]:
        return {
            "kind": "crusher_job",
            "retest_q": str(self.retest_q),
            "spike_semantics": str(self.spike_semantics).lower(),
        }

    def states(self) -> Iterable[Hashable]:
        yield from product(
            CRUSHER_OPS,
            range(N_BINS),
            PERMITS,
            (0, 1),
            range(1, self._stale + 1),
            (0, 1),
            (0, 1),
        )

    def commands(self) -> Sequence[str]:
        return self._commands

    def initial(self) -> Hashable:
        return ("RUN", 0, "NONE", 0, self._stale, 0, 0)

    def observable(self, s: Hashable) -> Hashable:
        op, _h, permit, occ, gas, alarm, soon = cast(CrusherState, s)
        return (op, permit, occ, gas, alarm, soon)

    def health(self, s: Hashable) -> int:
        return cast(CrusherState, s)[1]

    def step(self, s: Hashable, c: str) -> frozenset[Hashable | str]:
        if c not in self._command_set:
            raise ValueError(f"unknown crusher command {c!r}")
        out: set[Hashable | str] = set()
        refusable = c not in FORCED and c != "NOOP"
        for refused in (False, True) if refusable else (False,):
            out |= self._execute(s, "NOOP" if refused else c)
        return frozenset(out)

    def _execute(self, s: Hashable, e: str) -> set[Hashable | str]:
        """Outcomes when command ``e`` is executed: its effect, then the environment's move."""
        op, h, permit, occ, gas, alarm, soon = cast(CrusherState, s)
        stale = self._stale
        tested = False
        healths_at_start: tuple[int, ...] = (h,)

        if e == "START":
            if op == "STOP":
                op = "RUN"
            elif op in ("LOCKED", "ISO", "WORK"):
                return {VIOL}
        elif e == "STOP":
            if op == "RUN":
                op = "STOP"
        elif e == "LOCK":
            if op in ("STOP", "FAIL"):
                op = "LOCKED"
        elif e == "PROVE":
            if op == "LOCKED":
                op = "ISO"
        elif e == "ISSUE":
            # a permit spent at its limit is replaced in WORK when nobody is inside (A12.1); the
            # spike: ISO only
            if permit == "NONE" and (
                op == "ISO" or (op == "WORK" and occ == 0 and not self.spike_semantics)
            ):
                permit = "VALID"
        elif e == "BEGIN":
            if op == "ISO" and permit == "VALID":
                op = "WORK"
            elif op in ("RUN", "STOP", "LOCKED", "FAIL") or (op == "ISO" and permit != "VALID"):
                return {VIOL}
        elif e == "TEST":
            tested = op == "WORK"
        elif e == "ENTER":
            if occ == 0:
                if op == "WORK" and permit == "VALID" and gas == 1 and alarm == 0:
                    occ = 1
                else:
                    return {VIOL}
        elif e == "EXIT":
            occ = 0
            gas = stale
        elif e == "COMPLETE":
            if occ == 1:
                return {VIOL}
            if op == "WORK":
                op = "ISO"
                healths_at_start = (0,) if self.spike_semantics else tuple(dict.fromkeys((0, h)))
        elif e == "CLOSE":
            if op == "WORK":
                return {VIOL}
            if op == "ISO" and permit != "NONE":
                permit = "NONE"
        elif e == "UNISO":
            if op == "ISO":
                if permit != "NONE":
                    return {VIOL}
                op = "STOP"

        out: set[Hashable | str] = set()
        holders = (0, 1) if e == "REVAL" and permit == "REVAL" else (0,)
        for holder, h0 in product(holders, healths_at_start):
            permit_now = "VALID" if holder else permit
            for result in (0, 1) if tested else (0,):
                for onset in (0, 1) if (occ == 1 and not tested) else (0,):
                    alarm_next = result if tested else int(bool(alarm or onset))
                    healths = [(h0, op, False)]
                    if op == "RUN":
                        healths.append((h0 + 1, op, False) if h0 < HMAX else (h0, "FAIL", True))
                    for h_next, op_next, failed in healths:
                        if failed:
                            out.add(VIOL)  # P5: uncontrolled failure while running
                            continue
                        # an announced end: the permit is asked to be revalidated or is spent
                        if soon == 1 and permit_now != "NONE":
                            permits_next: tuple[str, ...] = (
                                ("REVAL",) if self.spike_semantics else ("REVAL", "NONE")
                            )
                        else:
                            permits_next = (permit_now,)
                        # after an announced step soon may be 1 again (A12.1); the spike: 0
                        soons_next = (0,) if soon == 1 and self.spike_semantics else (0, 1)
                        for permit_next, soon_next in product(permits_next, soons_next):
                            gas_next = 1 if tested else min(gas + 1, stale)
                            if occ == 1 and self._occupied_violation(
                                gas_next, permit_next, alarm, alarm_next
                            ):
                                out.add(VIOL)
                            else:
                                out.add(
                                    (
                                        op_next,
                                        h_next,
                                        permit_next,
                                        occ,
                                        gas_next,
                                        alarm_next,
                                        soon_next,
                                    )
                                )
        return out

    def _occupied_violation(
        self, gas_next: int, permit_next: str, alarm_before: int, alarm_next: int
    ) -> bool:
        """P3, P2 and P6 for a state with someone inside."""
        if gas_next == self._stale:
            return True
        if self.spike_semantics:
            return permit_next == "REVAL" or alarm_next == 1
        return permit_next != "VALID" or alarm_before == 1


@cache
def _pump_outcomes(op: str, h: int, cmd: str, spike: bool = False) -> frozenset[PumpOutcome]:
    """One pump's (or the conveyor's) outcomes under ``cmd``: ``(op, h)`` pairs and maybe VIOL.

    ``cmd`` is one of ``PUMP_COMMANDS``; ``ISOLATE`` belongs to the spike's alphabet. ``spike``
    makes ``COMPLETE`` always renew, as the spike did (ADR 0003 A11).
    """
    out: set[PumpOutcome] = set()
    for refused in (False, True) if cmd not in FORCED and cmd != "NOOP" else (False,):
        e = "NOOP" if refused else cmd
        op_now = op
        h_choices: tuple[int, ...] = (h,)
        if e == "START":
            if op_now == "STOP":
                op_now = "RUN"
            elif op_now in ("LOCKED", "ISO", "WORK"):
                out.add(VIOL)
                continue
        elif e == "STOP":
            if op_now == "RUN":
                op_now = "STOP"
        elif e == "LOCK":
            if op_now in ("STOP", "FAIL"):
                op_now = "LOCKED"
        elif e == "PROVE":
            if op_now == "LOCKED":
                op_now = "ISO"
        elif e == "ISOLATE":
            if op_now in ("STOP", "FAIL"):
                op_now = "ISO"
        elif e == "BEGIN":
            if op_now == "ISO":
                op_now = "WORK"
            elif op_now in ("RUN", "STOP", "LOCKED", "FAIL"):
                out.add(VIOL)
                continue
        elif e == "COMPLETE":
            if op_now == "WORK":
                op_now = "ISO"
                h_choices = (0,) if spike else tuple(dict.fromkeys((0, h)))
        elif e == "UNISO":
            if op_now == "ISO":
                op_now = "STOP"
        for h_now in h_choices:
            if op_now == "RUN":
                out.add((op_now, h_now))
                out.add((op_now, h_now + 1) if h_now < HMAX else ("FAIL", h_now))
            else:
                out.add((op_now, h_now))
    return frozenset(out)


class PumpPairModel:
    """The duty pump pair: at least one pump must stay available (rules: module docstring).

    ``spike_semantics=True`` is the spike's relation: five ops (no ``LOCKED``), ``ISOLATE`` in
    place of ``LOCK`` and ``PROVE``, and ``COMPLETE`` always renews a pump. It exists only so a
    test can show how the production game differs; never build a shield from it.
    """

    def __init__(self, *, spike_semantics: bool = False) -> None:
        self.spike_semantics = spike_semantics
        self.name = "pump_pair"
        self._ops = _SPIKE_PUMP_OPS if spike_semantics else PUMP_OPS
        alphabet = _SPIKE_PUMP_COMMANDS if spike_semantics else PUMP_COMMANDS
        self._commands: tuple[str, ...] = tuple(f"{a}|{b}" for a in alphabet for b in alphabet)
        self._split = {c: (c.split("|")[0], c.split("|")[1]) for c in self._commands}

    def parameters(self) -> Mapping[str, str]:
        return {"kind": "pump_pair", "spike_semantics": str(self.spike_semantics).lower()}

    def states(self) -> Iterable[Hashable]:
        yield from product(self._ops, range(N_BINS), self._ops, range(N_BINS))

    def commands(self) -> Sequence[str]:
        return self._commands

    def initial(self) -> Hashable:
        return ("RUN", 0, "RUN", 0)

    def observable(self, s: Hashable) -> Hashable:
        pair = cast(PairState, s)
        return (pair[0], pair[2])

    def health(self, s: Hashable) -> tuple[int, int]:
        pair = cast(PairState, s)
        return (pair[1], pair[3])

    def step(self, s: Hashable, c: str) -> frozenset[Hashable | str]:
        if c not in self._split:
            raise ValueError(f"unknown pump-pair command {c!r}")
        op1, h1, op2, h2 = cast(PairState, s)
        a1, a2 = self._split[c]
        out: set[Hashable | str] = set()
        spike = self.spike_semantics
        for t1, t2 in product(
            _pump_outcomes(op1, h1, a1, spike), _pump_outcomes(op2, h2, a2, spike)
        ):
            if isinstance(t1, str) or isinstance(t2, str):  # VIOL from either pump
                out.add(VIOL)
                continue
            if t1[0] in PUMP_UNAVAILABLE and t2[0] in PUMP_UNAVAILABLE:  # P4
                out.add(VIOL)
            else:
                out.add(t1 + t2)
        return frozenset(out)


class ConveyorModel:
    """The belt conveyor CV-101: P1 only, no pair rule (rules: module docstring; ADR 0003 A12.3).

    Its states are ``(op, h)`` and its per-asset rules are the pump's. The failure of a running
    conveyor is not a violation of any property, so only a controller that energises it under a
    lock or starts work on it without proof can lose.
    """

    def __init__(self) -> None:
        self.name = "conveyor"
        self._command_set = frozenset(PUMP_COMMANDS)

    def parameters(self) -> Mapping[str, str]:
        return {"kind": "conveyor"}

    def states(self) -> Iterable[Hashable]:
        yield from product(PUMP_OPS, range(N_BINS))

    def commands(self) -> Sequence[str]:
        return PUMP_COMMANDS

    def initial(self) -> Hashable:
        return ("RUN", 0)

    def observable(self, s: Hashable) -> Hashable:
        return (cast(AssetState, s)[0],)

    def health(self, s: Hashable) -> int:
        return cast(AssetState, s)[1]

    def step(self, s: Hashable, c: str) -> frozenset[Hashable | str]:
        if c not in self._command_set:
            raise ValueError(f"unknown conveyor command {c!r}")
        op, h = cast(AssetState, s)
        return frozenset(_pump_outcomes(op, h, c))


def model_from_parameters(params: Mapping[str, str]) -> AbstractModel:
    """Rebuild a model from ``parameters()``; the shield runtime uses it to read a table's meta."""
    kind = params.get("kind")
    if kind == "crusher_job":
        return CrusherJobModel(
            int(params["retest_q"]), spike_semantics=params.get("spike_semantics") == "true"
        )
    if kind == "pump_pair":
        return PumpPairModel(spike_semantics=params.get("spike_semantics") == "true")
    if kind == "conveyor":
        return ConveyorModel()
    raise ValueError(f"unknown model kind {kind!r}")
