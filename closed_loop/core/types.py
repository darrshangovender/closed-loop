"""Frozen vocabulary shared by every area: fidelities, plant states, commands and health bins.

Interface text: docs/adr/0002-interfaces.md, section 3.1. Member order of ``Op`` and ``Permit`` is
part of the interface: ``Trace`` stores their positions in ``list(Op)`` and ``list(Permit)``.
"""

import math
from dataclasses import dataclass
from enum import StrEnum


class Fidelity(StrEnum):
    F0 = "F0"
    F1 = "F1"
    F1D = "F1+D"
    F2 = "F2"


class Op(StrEnum):
    RUN = "RUN"
    STOP = "STOP"
    LOCKED = "LOCKED"
    ISO = "ISO"
    WORK = "WORK"
    FAIL = "FAIL"


class Permit(StrEnum):
    NONE = "NONE"
    VALID = "VALID"
    REVAL = "REVAL"


class Cmd(StrEnum):
    NOOP = "NOOP"
    START = "START"
    STOP = "STOP"
    LOCK = "LOCK"
    PROVE = "PROVE"
    ISSUE = "ISSUE"
    REVAL = "REVAL"
    BEGIN = "BEGIN"
    TEST = "TEST"
    ENTER = "ENTER"
    EXIT = "EXIT"
    COMPLETE = "COMPLETE"
    CLOSE = "CLOSE"
    UNISO = "UNISO"
    DERATE = "DERATE"  # F1+D only; F2 masks it


# Assumption A_exec: the plant never refuses these two.
FORCED_CMDS: frozenset[Cmd] = frozenset({Cmd.STOP, Cmd.EXIT})


class PlanCmd(StrEnum):
    INSPECT = "INSPECT"
    ORDER = "ORDER"
    RAISE_WO = "RAISE_WO"


N_BINS = 4  # health bins 0..3, 3 = critical (0001 section 4.1)
BIN_EDGES_DAYS: tuple[float, float, float] = (30.0, 10.0, 3.0)


@dataclass(frozen=True)
class Command:
    asset_id: str
    cmd: Cmd | PlanCmd
    arg: str | None = None  # part number, failure mode, target shift index ...
    source: str = ""  # proposer name, for audit

    def __post_init__(self) -> None:
        # Both are StrEnums, so a plain string such as "BEGIN" compares equal to a member and the
        # plant's handler table would accept it, while the shield's ``isinstance(c.cmd, Cmd)`` test
        # would not gate it: a way past the proof's boundary (P12g, G6). Nothing builds one today.
        if not isinstance(self.cmd, Cmd | PlanCmd):
            raise TypeError(
                f"Command.cmd must be a Cmd or a PlanCmd, not {type(self.cmd).__name__}: "
                f"{self.cmd!r}"
            )


def bin_of_rul(rul_days: float) -> int:
    """Health bin of a remaining useful life in days: 0 (healthy) to 3 (critical).

    Bin 0 is ``rul > 30``, bin 1 is ``10 < rul <= 30``, bin 2 is ``3 <= rul <= 10`` and bin 3 is
    ``rul < 3``. The interface text gives the bins as "> 30", "10-30", "3-10" and "< 3", which
    leaves 10 on two bins; it goes to the more critical one (bin 2), as does anything at or below
    zero. Exactly 30 is bin 1 and exactly 3 is bin 2, following the strict "> 30" and "< 3".
    NaN is refused: every comparison with it is false, so it would otherwise drop silently to bin 3.
    """
    if math.isnan(rul_days):
        raise ValueError("rul_days is NaN")
    e0, e1, e2 = BIN_EDGES_DAYS
    if rul_days > e0:
        return 0
    if rul_days > e1:
        return 1
    if rul_days >= e2:
        return 2
    return 3


def bins_of_interval(lo: float, hi: float) -> tuple[int, int]:
    """Lowest and highest health bin that the RUL interval ``[lo, hi]`` touches.

    ``bin_of_rul`` falls as RUL rises, so the lowest bin comes from ``hi`` and the highest from
    ``lo``. ``lo`` may be negative (a bound below failure) and ``hi`` may be infinite.
    """
    if math.isnan(lo) or math.isnan(hi):
        raise ValueError("interval bound is NaN")
    if lo > hi:
        raise ValueError(f"empty interval: lo={lo} > hi={hi}")
    return bin_of_rul(hi), bin_of_rul(lo)
