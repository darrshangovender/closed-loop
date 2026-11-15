"""Hand-built and random traces for the spec tests."""

import numpy as np

from closed_loop.core.trace import Trace, op_index, permit_index
from closed_loop.core.types import Op, Permit

CRUSHER = "CR-501"
PUMPS = ("PU-201", "PU-202")
CONVEYOR = "CV-101"
ASSETS = (PUMPS[0], PUMPS[1], CRUSHER, CONVEYOR)
N_Q = 40


def clean_trace() -> Trace:
    """40 quarters: one compliant crusher job, one compliant pump job, no violation of P1-P6."""
    tr = Trace.empty(N_Q, ASSETS)
    c = CRUSHER
    # crusher: RUN, then STOP, LOCK, prove isolation, permit, work, entry for 11 quarters, exit
    tr.op[c][4:] = op_index(Op.STOP)
    tr.op[c][5:] = op_index(Op.LOCKED)
    tr.op[c][6:] = op_index(Op.ISO)
    tr.iso_proven[c][6:30] = True
    tr.permit[c][7:28] = permit_index(Permit.VALID)
    tr.permit_margin_min[c][7:28] = 600.0 - 15.0 * np.arange(21)
    tr.op[c][8:26] = op_index(Op.WORK)
    tr.op[c][26:30] = op_index(Op.ISO)
    tr.op[c][30:32] = op_index(Op.STOP)
    tr.op[c][32:] = op_index(Op.RUN)
    tr.gas_age_min[c][9:22] = 15.0 * ((np.arange(9, 22) - 9) % 2)  # re-test every 2 quarters
    tr.occupied[c][10:21] = True
    tr.true_rul_days[c][:] = 20.0
    tr.true_rul_days[c][32:] = 60.0
    # pump A runs all the time; pump B has an isolated job from 12 to 24
    tr.op[PUMPS[1]][12:15] = op_index(Op.LOCKED)
    tr.op[PUMPS[1]][15:18] = op_index(Op.ISO)
    tr.op[PUMPS[1]][18:23] = op_index(Op.WORK)
    tr.op[PUMPS[1]][23:25] = op_index(Op.ISO)
    tr.op[PUMPS[1]][25:] = op_index(Op.STOP)
    tr.iso_proven[PUMPS[1]][15:25] = True
    return tr


def violating_trace(pid: str) -> Trace:
    """The clean trace with exactly one defect, which breaks property ``pid`` and no other."""
    tr = clean_trace()
    c = CRUSHER
    if pid == "P1":  # work on an energised asset
        tr.op[c][2:4] = op_index(Op.WORK)
    elif pid == "P1-conveyor":  # the same on the conveyor (A12.3); breaks P1 and nothing else
        tr.op[CONVEYOR][2:4] = op_index(Op.WORK)
    elif pid == "P2":  # inside the space while the permit awaits revalidation
        tr.permit[c][15] = permit_index(Permit.REVAL)
    elif pid == "P3":  # inside the space with a test 60 minutes old
        tr.gas_age_min[c][16] = 60.0
    elif pid == "P4":  # both pumps out of service
        tr.op[PUMPS[0]][30:32] = op_index(Op.LOCKED)
        tr.op[PUMPS[1]][30:32] = op_index(Op.LOCKED)
    elif pid == "P5":  # the critical asset fails without warning
        tr.uncontrolled_failure[c][35] = True
    elif pid == "P6":  # alarm sounds and nobody leaves
        tr.alarm[c][14:17] = True
    else:
        raise ValueError(pid)
    return tr


def random_trace(rng: np.random.Generator, n_q: int) -> Trace:
    """A trace with arbitrary (mostly nonsense) contents inside each field's legal range."""
    tr = Trace.empty(n_q, ASSETS)
    for a in ASSETS:
        tr.op[a][:] = rng.integers(0, len(Op), n_q)
        tr.permit[a][:] = rng.integers(0, len(Permit), n_q)
        tr.occupied[a][:] = rng.random(n_q) < 0.5
        tr.gas_age_min[a][:] = np.where(
            rng.random(n_q) < 0.1, 1.0e6, 15.0 * rng.integers(0, 8, n_q)
        )
        tr.alarm[a][:] = rng.random(n_q) < 0.2
        tr.iso_proven[a][:] = rng.random(n_q) < 0.5
        tr.permit_margin_min[a][:] = np.where(
            rng.random(n_q) < 0.2, 1.0e6, rng.uniform(-60.0, 700.0, n_q)
        )
        tr.true_rul_days[a][:] = np.where(rng.random(n_q) < 0.3, np.nan, rng.uniform(0, 40, n_q))
        tr.uncontrolled_failure[a][:] = rng.random(n_q) < 0.05
    return tr
