"""The abstraction map for the F1 plant and the conformance test (packet P5c). SYNTHETIC.

The proof of the shield is a proof about the finite games of ``abstraction``. It reaches the plant
only through (a) the map from what the plant shows to an abstract state, and (b) the assumption that
the plant moves the way the games say. ``F1AbstractionMap`` is (a); ``conformance`` measures how far
(b) is false on sampled traces. The result is a **measured soundness gap on sampled traces, not a
proof** (ADR 0001, section 4.7).

The map
-------
* The crusher state is ``(op, h, permit, occ, gas, alarm, soon)`` from the ``AssetView``. ``h`` is
  the health bin the caller gives (``Trace.health_bin``, the bin of the *rate* RUL: ADR 0003 A10,
  which is the abstraction's ``h``; the realised RUL is not used). ``gas`` is
  ``min(gas_age_q, R+1)`` with ``R`` the re-test period in quarters. **The view is what a proposer
  sees at the start of a quarter**: a test that passed in the previous quarter's step has
  ``gas_age_q == 1`` there, and that is the abstraction's ``gas == 1`` ("a test came out of the
  previous tick"). (The test stub of P5a used ``gas_age_q + 1`` and its own convention; this is
  the plant's.) The abstraction is one quarter stricter than the plant's P3 rule, on purpose (ADR
  0003 and the P5b note): a test two quarters old is stale to the abstraction and still fresh to
  the plant.
* ``soon`` is the plant's ``view.soon``: the next quarter is a handover, or the permit reaches its
  12 hours from issue at the next quarter (ADR 0003 A12.1). The trace holds neither the plant's
  ``limit_q`` nor the view, so it is rebuilt from the rows: a handover every ``SHIFT_Q`` quarters,
  and, for a row whose permit is not NONE, the permit's limit as the first row of the unbroken run
  of such rows plus ``permit_max_q``. (The old reconstruction assumed a fixed shift pattern alone.)
* The pump pair state is ``(op1, h1, op2, h2)`` and the conveyor state ``(op, h)``, each ``op`` as
  the plant shows it: a locked pump is ``LOCKED`` and not available (ADR 0003 A12.2).
* Commands. Crusher commands map to the abstract command of the same name. For a pump and for the
  conveyor, the commands of their alphabet (NOOP, START, STOP, LOCK, PROVE, BEGIN, COMPLETE, UNISO)
  map to themselves, and every command they have no name for (ISSUE, REVAL, TEST, ENTER, EXIT,
  CLOSE, DERATE) maps to ``NOOP``: none of them changes ``op``. Planning commands (INSPECT, ORDER,
  RAISE_WO) are outside the proof and map to ``NOOP``.

The test
--------
For each trace and each quarter ``q >= 1`` the state at the end of step ``q - 1`` is mapped, the
commands issued at ``q`` are mapped, the state at the end of step ``q`` is mapped, and the last is
looked up in ``step(previous, command)``. Quarter 0 has no recorded predecessor and is skipped.

* A tick is **tested** when neither the previous nor the new concrete state violates a property.
  A tested tick is *matched* when the new abstract state is one of the non-sink outcomes, else
  *unmatched*. An unmatched tick is attributed to the variables in which the new abstract state
  differs from the closest outcome (fewest differing variables, ties by name), so a tick with one
  wrong variable counts once, under it. ``by_variable`` counts ticks per variable (a tick that
  differs in two variables counts under both, and once in ``unmatched``).
* A tick whose new concrete state **starts** a violation is an *edge*. The abstraction is sound for
  it only if ``VIOL`` is among the outcomes; otherwise it is a *missed violation*.
* A tick that begins inside a violation is not compared (the sink is absorbing: nothing follows
  from it in the games). It is counted in ``ticks_in_violation``.
* A tick with two or more commands for one component is outside the games (one controller move per
  tick) and is counted in ``multi_command_ticks``, not compared.
* ``unmatched_when_safe`` counts the unmatched ticks whose outcome set has no ``VIOL``: the
  transitions a shield could admit that the plant does not follow. Those bear on the proof; an
  unmatched tick with ``VIOL`` among its outcomes is one the shield would never admit.

The concrete violations used for edges are the English properties of ADR 0001 section 5 read off
the trace arrays (P1: working with the isolation unproven, or energised while the work is open, on
the crusher, the pumps and the conveyor; P2: occupied without a valid permit; P3: occupied with the
last passing test older than the re-test period, or entering on one older than the break window;
P4: both pumps unavailable, which a locked pump is; P5: an uncontrolled failure of the crusher; P6:
still inside after an alarm). They are not read from the
plant's own violation list, so a slip in either would show.

A_phys is checked on every component (the crusher, both pumps and the conveyor): a bin that rises
by more than one in a tick (``bin_skip``), and an uncontrolled failure from a bin below the top
(``failure_below_top``).

Nothing here is tuned. If a number is large, the abstraction or the plant is wrong or too coarse,
and the report says where.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from closed_loop.core.results import ResultRecord
from closed_loop.core.timegrid import QUARTER_MIN, SHIFT_Q
from closed_loop.core.trace import Trace, permit_index
from closed_loop.core.types import Cmd, Command, Fidelity, Op, Permit
from closed_loop.shield.abstraction import (
    CRUSHER_COMMANDS,
    PUMP_COMMANDS,
    VIOL,
    AbstractModel,
)
from closed_loop.shield.runtime import CONVEYOR_ID, CRUSHER_ID, PUMP_IDS, AbstractionMap
from closed_loop.stats.intervals import wilson

__all__ = [
    "A_PHYS_KINDS",
    "CONVEYOR_VARIABLES",
    "CRUSHER_VARIABLES",
    "PUMP_VARIABLES",
    "ConformanceReport",
    "F1AbstractionMap",
    "conformance",
    "result_records",
]

CRUSHER_VARIABLES: tuple[str, ...] = ("op", "h", "permit", "occ", "gas", "alarm", "soon")
PUMP_VARIABLES: tuple[str, ...] = ("op1", "h1", "op2", "h2")
CONVEYOR_VARIABLES: tuple[str, ...] = ("op", "h")
A_PHYS_KINDS: tuple[str, ...] = ("bin_skip", "failure_below_top")

PERMIT_MAX_H = 12.0  # CORP-HSE-004 clause 4.1, the default of ``SpecParams``
_VIEW_AGE_CAP = 99  # the plant caps ``AssetView.gas_age_q`` here
_TOP_BIN = 3
_BIN_MOVE_KINDS = ("fall", "rise_while_not_running", "completion_renewed", "completion_unrenewed")
_EXAMPLES = 60  # distinct unmatched transitions kept in the report
_MULTI = -2  # command marker: more than one command for the component in the quarter
_NONE = -1  # command marker: no command (NOOP)


class F1AbstractionMap:
    """Concrete to abstract for the F1 plant (``runtime.AbstractionMap``). See the module docstring.

    ``retest_q`` is the re-test period of the model the table was built for (``R``).
    """

    def __init__(self, retest_q: int = 2) -> None:
        if isinstance(retest_q, bool) or not isinstance(retest_q, int) or retest_q < 1:
            raise ValueError(f"retest_q must be an integer of at least 1, got {retest_q!r}")
        self.retest_q = retest_q
        self.stale = retest_q + 1

    def gas(self, gas_age_q: int) -> int:
        return min(max(gas_age_q, 1), self.stale)

    def crusher_state(self, view: Any, health_bin: int) -> Hashable:
        return (
            view.op.value,
            health_bin,
            view.permit.value,
            int(view.occupied),
            self.gas(view.gas_age_q),
            int(view.alarm),
            int(view.soon),
        )

    def pump_pair_state(self, a: Any, b: Any, bins: tuple[int, int]) -> Hashable:
        return (a.op.value, bins[0], b.op.value, bins[1])

    def conveyor_state(self, view: Any, health_bin: int) -> Hashable:
        return (view.op.value, health_bin)

    def command(self, cmd: Command) -> str:
        if not isinstance(cmd.cmd, Cmd):
            return "NOOP"  # INSPECT, ORDER, RAISE_WO: outside the proof
        name = cmd.cmd.value
        if cmd.asset_id in PUMP_IDS or cmd.asset_id == CONVEYOR_ID:
            return name if name in PUMP_COMMANDS else "NOOP"
        return name if name in CRUSHER_COMMANDS else "NOOP"


@dataclass(frozen=True)
class ConformanceReport:
    """How well the traces follow the games. The first five fields are the frozen interface.

    ``ticks`` is the number of tested ticks (both games together); ``unmatched`` how many of them
    the games cannot produce; ``by_variable`` the unmatched ticks per variable, keyed
    ``"crusher.gas"`` and so on; ``wilson`` the 95% interval of ``unmatched / ticks``. The rest is
    additive (see the module docstring).
    """

    ticks: int
    unmatched: int
    by_variable: Mapping[str, int]
    a_phys_violations: int
    wilson: tuple[float, float]
    episodes: int = 0
    ticks_total: int = 0
    ticks_in_violation: int = 0
    multi_command_ticks: int = 0
    violation_edges: int = 0
    missed_violations: int = 0
    unmatched_when_safe: int = 0
    by_variable_when_safe: Mapping[str, int] = field(default_factory=dict)
    ticks_by_model: Mapping[str, int] = field(default_factory=dict)
    unmatched_by_model: Mapping[str, int] = field(default_factory=dict)
    a_phys_ticks: int = 0
    a_phys_by_kind: Mapping[str, int] = field(default_factory=dict)
    a_phys_wilson: tuple[float, float] = (0.0, 1.0)
    # component-ticks where the health bin moved in a way the games cannot: "fall" (not a renewal),
    # "rise_while_not_running"; and the completed jobs seen: "completion_renewed" (bin back to 0)
    # and "completion_unrenewed" (bin unchanged: a wrong-mode job, ADR 0003 A11)
    bin_moves: Mapping[str, int] = field(default_factory=dict)
    # (game, previous, command, new, ticks, VIOL among the outcomes), most frequent first
    unmatched_examples: tuple[tuple[str, Hashable, str, Hashable, int, bool], ...] = ()

    def variable_rate(self, variable: str, *, when_safe: bool = False) -> tuple[int, int]:
        """``(ticks unmatched in this variable, ticks tested for its model)``."""
        model = variable.split(".")[0]
        counts = self.by_variable_when_safe if when_safe else self.by_variable
        return counts.get(variable, 0), self.ticks_by_model.get(model, 0)


# ------------------------------------------------------------------------------------------
# reading the trace
# ------------------------------------------------------------------------------------------


class _Outcomes:
    """``model.step`` with a cache: the same few thousand (state, command) pairs recur."""

    def __init__(self, model: AbstractModel) -> None:
        self.model = model
        self.cache: dict[tuple[Hashable, str], frozenset[Hashable | str]] = {}

    def __call__(self, s: Hashable, c: str) -> frozenset[Hashable | str]:
        hit = self.cache.get((s, c))
        if hit is None:
            hit = self.cache[(s, c)] = frozenset(self.model.step(s, c))
        return hit


@dataclass
class _Tally:
    """Counts for one game, pooled over traces."""

    ticks_total: int = 0
    in_violation: int = 0
    multi: int = 0
    edges: int = 0
    triples: Counter[tuple[Hashable, str, Hashable]] = field(default_factory=Counter)
    edge_triples: Counter[tuple[Hashable, str]] = field(default_factory=Counter)


def _command_ids(
    tr: Trace, mapper: AbstractionMap, assets: Sequence[str], names: Sequence[str]
) -> NDArray[np.int64]:
    """Per quarter: the index into ``names`` of the joint abstract command, -1 none, -2 several.

    ``assets`` is the crusher alone or the two pumps; a joint command is ``"a1|a2"`` for pumps.
    """
    per: dict[int, dict[str, list[str]]] = {}
    for q, command, _executed, _overridden in tr.commands:
        if command.asset_id not in assets or not isinstance(command.cmd, Cmd):
            continue
        if command.cmd is Cmd.NOOP:
            continue
        per.setdefault(q, {}).setdefault(command.asset_id, []).append(mapper.command(command))
    index = {n: i for i, n in enumerate(names)}
    out = np.full(tr.n_q, _NONE, dtype=np.int64)
    for q, by_asset in per.items():
        if q >= tr.n_q:
            continue
        if any(len(v) > 1 for v in by_asset.values()):
            out[q] = _MULTI
            continue
        parts = [by_asset[a][0] if a in by_asset else "NOOP" for a in assets]
        joint = "|".join(parts) if len(assets) > 1 else parts[0]
        out[q] = index[joint]
    return out


def _run_age(held: NDArray[np.bool_]) -> NDArray[np.int64]:
    """Rows since the current unbroken run of ``held`` began (0 on its first row, 0 outside)."""
    idx = np.arange(held.shape[0])
    started = held & ~np.concatenate([[False], held[:-1]])
    last_start = np.maximum.accumulate(np.where(started, idx, -1))
    return np.asarray(np.where(held, idx - last_start, 0), dtype=np.int64)


def _soon(tr: Trace, crusher: str, permit_max_q: int) -> NDArray[np.int64]:
    """The ``soon`` the plant's view shows at the decision after row ``r`` (end of step ``r``).

    The view at decision quarter ``q = r + 1`` says ``soon`` when ``q + 1`` starts a shift, or when
    the permit held at row ``r`` reaches its limit at ``q + 1``. A permit's limit is its issue plus
    ``permit_max_q``, and the issue is the first row of the unbroken run of rows that hold a permit
    (ADR 0003 A12.1: a revalidation does not move it).
    """
    n = tr.n_q
    handover = (np.arange(n) + 2) % SHIFT_Q == 0
    held = tr.permit[crusher] != permit_index(Permit.NONE)
    limit_next = held & (_run_age(held) + 2 >= permit_max_q)
    return np.asarray(handover | limit_next, dtype=np.int64)


def _crusher_rows(
    tr: Trace, mapper: AbstractionMap, crusher: str, permit_max_q: int
) -> tuple[NDArray[np.int64], list[Hashable]]:
    """Abstract state id of every quarter's end-of-step state, and the list of distinct states."""
    ops, permits = list(Op), list(Permit)
    age_min = tr.gas_age_min[crusher].astype(np.float64)
    # what a proposer would see at the next decision quarter: one more quarter since the test
    age_view = np.clip(np.floor(age_min / QUARTER_MIN + 0.5).astype(np.int64) + 1, 1, _VIEW_AGE_CAP)
    key = tr.op[crusher].astype(np.int64)
    for arr, base in (
        (tr.permit[crusher].astype(np.int64), len(permits)),
        (tr.occupied[crusher].astype(np.int64), 2),
        (age_view, _VIEW_AGE_CAP + 1),
        (tr.alarm[crusher].astype(np.int64), 2),
        (_soon(tr, crusher, permit_max_q), 2),
        (tr.health_bin[crusher].astype(np.int64), 4),
    ):
        key = key * base + arr
    unique, inverse = np.unique(key, return_inverse=True)
    states: list[Hashable] = []
    for k in unique.tolist():
        k, h = divmod(k, 4)
        k, soon = divmod(k, 2)
        k, alarm = divmod(k, 2)
        k, age = divmod(k, _VIEW_AGE_CAP + 1)
        k, occ = divmod(k, 2)
        op, permit = divmod(k, len(permits))
        view = SimpleNamespace(
            op=ops[op],
            permit=permits[permit],
            occupied=bool(occ),
            gas_age_q=age,
            alarm=bool(alarm),
            soon=bool(soon),
        )
        states.append(mapper.crusher_state(view, h))
    return inverse.astype(np.int64), states


def _pump_rows(
    tr: Trace, mapper: AbstractionMap, pumps: Sequence[str]
) -> tuple[NDArray[np.int64], list[Hashable]]:
    ops = list(Op)
    a, b = pumps
    key = tr.op[a].astype(np.int64)
    for arr, base in (
        (tr.op[b].astype(np.int64), len(ops)),
        (tr.health_bin[a].astype(np.int64), 4),
        (tr.health_bin[b].astype(np.int64), 4),
    ):
        key = key * base + arr
    unique, inverse = np.unique(key, return_inverse=True)
    states: list[Hashable] = []
    for k in unique.tolist():
        k, hb = divmod(k, 4)
        k, ha = divmod(k, 4)
        oa, ob = divmod(k, len(ops))
        states.append(
            mapper.pump_pair_state(
                SimpleNamespace(op=ops[oa]), SimpleNamespace(op=ops[ob]), (ha, hb)
            )
        )
    return inverse.astype(np.int64), states


def _conveyor_rows(
    tr: Trace, mapper: AbstractionMap, conveyor: str
) -> tuple[NDArray[np.int64], list[Hashable]]:
    ops = list(Op)
    key = tr.op[conveyor].astype(np.int64) * 4 + tr.health_bin[conveyor].astype(np.int64)
    unique, inverse = np.unique(key, return_inverse=True)
    states: list[Hashable] = []
    for k in unique.tolist():
        op, h = divmod(k, 4)
        states.append(mapper.conveyor_state(SimpleNamespace(op=ops[op]), h))
    return inverse.astype(np.int64), states


def _p1(tr: Trace, asset: str) -> NDArray[np.bool_]:
    """P1: working with the isolation unproven, or energised while the work is open."""
    work, run = list(Op).index(Op.WORK), list(Op).index(Op.RUN)
    bad = (tr.op[asset] == work) & ~tr.iso_proven[asset]
    bad[1:] |= (tr.op[asset][:-1] == work) & (tr.op[asset][1:] == run)
    return np.asarray(bad)


def _crusher_violation(tr: Trace, crusher: str, retest_q: int, break_q: int) -> NDArray[np.bool_]:
    """P1, P2, P3, P5 and P6 of the English text, read off the arrays."""
    occ = tr.occupied[crusher]
    valid = tr.permit[crusher] == list(Permit).index(Permit.VALID)
    age_min = tr.gas_age_min[crusher]
    stale = age_min > np.float32(retest_q * QUARTER_MIN)
    # P3 also at the moment of entry: a test older than the break window is not a fresh one
    entering = np.zeros(tr.n_q, dtype=np.bool_)
    entering[1:] = occ[1:] & ~occ[:-1] & (age_min[1:] > np.float32(break_q * QUARTER_MIN))
    bad = _p1(tr, crusher) | (occ & ~valid) | (occ & stale) | entering
    bad = bad | tr.uncontrolled_failure[crusher]
    inside_after_alarm = np.zeros(tr.n_q, dtype=np.bool_)
    inside_after_alarm[1:] = occ[1:] & occ[:-1] & tr.alarm[crusher][:-1]
    return np.asarray(bad | inside_after_alarm)


def _pump_violation(tr: Trace, pumps: Sequence[str]) -> NDArray[np.bool_]:
    """P1 for each pump and P4 (both pumps unavailable: locked, isolated, in work or failed)."""
    gone = [
        np.isin(tr.op[p], [list(Op).index(o) for o in (Op.LOCKED, Op.ISO, Op.WORK, Op.FAIL)])
        for p in pumps
    ]
    return np.asarray(_p1(tr, pumps[0]) | _p1(tr, pumps[1]) | (gone[0] & gone[1]))


def _pooled(
    ids: NDArray[np.int64],
    states: list[Hashable],
    cmd_ids: NDArray[np.int64],
    names: Sequence[str],
    mask: NDArray[np.bool_],
    post: bool,
    noop: str,
) -> Counter[tuple[Hashable, ...]]:
    """Counts of ``(previous, command, new)``, or ``(previous, command)``, over the masked ticks."""
    out: Counter[tuple[Hashable, ...]] = Counter()
    if not mask.any():
        return out
    k, n_cmd = len(states), len(names) + 1
    a, c, b = ids[:-1][mask], cmd_ids[1:][mask] + 1, ids[1:][mask]  # c == 0 is NOOP
    unique, counts = np.unique((a * n_cmd + c) * k + b, return_counts=True)
    for t, count in zip(unique.tolist(), counts.tolist(), strict=True):
        rest, bi = divmod(t, k)
        ai, ci = divmod(rest, n_cmd)
        name = noop if ci == 0 else names[ci - 1]
        key = (states[ai], name, states[bi]) if post else (states[ai], name)
        out[key] += count
    return out


def _accumulate(
    tally: _Tally,
    ids: NDArray[np.int64],
    states: list[Hashable],
    cmd_ids: NDArray[np.int64],
    names: Sequence[str],
    violating: NDArray[np.bool_],
    noop: str,
) -> None:
    n = ids.shape[0]
    if n < 2:
        return
    tally.ticks_total += n - 1
    multi = cmd_ids[1:] == _MULTI
    pre_bad, post_bad = violating[:-1], violating[1:]
    tally.multi += int(multi.sum())
    tally.in_violation += int((pre_bad & ~multi).sum())
    live = ~pre_bad & ~multi
    edge, tested = live & post_bad, live & ~post_bad
    tally.edges += int(edge.sum())
    tally.triples.update(_pooled(ids, states, cmd_ids, names, tested, True, noop))  # type: ignore[arg-type]
    tally.edge_triples.update(_pooled(ids, states, cmd_ids, names, edge, False, noop))  # type: ignore[arg-type]


def _bin_moves(tr: Trace, assets: Sequence[str]) -> Counter[str]:
    """Bin moves outside the games (see ``ConformanceReport.bin_moves``), summed over components."""
    ops = list(Op)
    run, work, iso = ops.index(Op.RUN), ops.index(Op.WORK), ops.index(Op.ISO)
    out: Counter[str] = Counter()
    for a in assets:
        h = tr.health_bin[a].astype(np.int64)
        if h.shape[0] < 2:
            continue
        op = tr.op[a]
        completed = (op[:-1] == work) & (op[1:] == iso)
        before, after = h[:-1], h[1:]
        out["completion_renewed"] += int((completed & (before > 0) & (after == 0)).sum())
        out["completion_unrenewed"] += int((completed & (before > 0) & (after == before)).sum())
        out["fall"] += int(((after < before) & ~(completed & (after == 0))).sum())
        out["rise_while_not_running"] += int(
            ((after > before) & (op[:-1] != run) & (op[1:] != run)).sum()
        )
    return out


def _a_phys(tr: Trace, assets: Sequence[str]) -> tuple[int, int, int, int]:
    """``(component-ticks, bin skips, failures below the top, ticks with either)``."""
    ticks = skips = early_failures = either = 0
    for a in assets:
        h = tr.health_bin[a].astype(np.int64)
        if h.shape[0] < 2:
            continue
        ticks += h.shape[0] - 1
        skip = (h[1:] - h[:-1]) > 1
        early = tr.uncontrolled_failure[a][1:] & (h[:-1] < _TOP_BIN)
        skips += int(skip.sum())
        early_failures += int(early.sum())
        either += int((skip | early).sum())
    return ticks, skips, early_failures, either


# ------------------------------------------------------------------------------------------
# the test
# ------------------------------------------------------------------------------------------


def conformance(
    traces: Sequence[Trace],
    model_map: AbstractionMap,
    models: Mapping[str, AbstractModel],
    *,
    crusher_id: str = CRUSHER_ID,
    pump_ids: tuple[str, str] = PUMP_IDS,
    conveyor_id: str = CONVEYOR_ID,
    break_q: int = 1,
    permit_max_h: float = PERMIT_MAX_H,
) -> ConformanceReport:
    """Test the traces against the games.

    ``models`` has the keys ``"crusher"`` and ``"pumps"`` and, for the conveyor game of ADR 0003
    A12.3, ``"conveyor"`` (without it the conveyor is not tested). ``permit_max_h`` is the permit
    limit the plant was run with (the reconstruction of ``soon`` needs it).
    """
    crusher_model, pump_model = models["crusher"], models["pumps"]
    conveyor_model = models.get("conveyor")
    retest_q = int(getattr(crusher_model, "retest_q", 2))
    permit_max_q = max(int(permit_max_h * 60 / QUARTER_MIN), 1)
    tallies = {"crusher": _Tally(), "pumps": _Tally()}
    if conveyor_model is not None:
        tallies["conveyor"] = _Tally()
    phys_ticks = skips = early_failures = a_phys = 0
    moves: Counter[str] = Counter()
    crusher_names = tuple(crusher_model.commands())
    pump_names = tuple(pump_model.commands())
    assets = (
        (crusher_id, *pump_ids) if conveyor_model is None else (crusher_id, *pump_ids, conveyor_id)
    )
    for tr in traces:
        ids, states = _crusher_rows(tr, model_map, crusher_id, permit_max_q)
        cmds = _command_ids(tr, model_map, (crusher_id,), crusher_names)
        violating = _crusher_violation(tr, crusher_id, retest_q, break_q)
        _accumulate(tallies["crusher"], ids, states, cmds, crusher_names, violating, "NOOP")
        ids, states = _pump_rows(tr, model_map, pump_ids)
        cmds = _command_ids(tr, model_map, pump_ids, pump_names)
        violating = _pump_violation(tr, pump_ids)
        _accumulate(tallies["pumps"], ids, states, cmds, pump_names, violating, "NOOP|NOOP")
        if conveyor_model is not None:
            ids, states = _conveyor_rows(tr, model_map, conveyor_id)
            names = tuple(conveyor_model.commands())
            cmds = _command_ids(tr, model_map, (conveyor_id,), names)
            _accumulate(tallies["conveyor"], ids, states, cmds, names, _p1(tr, conveyor_id), "NOOP")
        moves.update(_bin_moves(tr, assets))
        ticks, skip, early, either = _a_phys(tr, assets)
        phys_ticks += ticks
        skips += skip
        early_failures += early
        a_phys += either

    models_by_key: dict[str, tuple[AbstractModel, tuple[str, ...], str]] = {
        "crusher": (crusher_model, CRUSHER_VARIABLES, "crusher"),
        "pumps": (pump_model, PUMP_VARIABLES, "pumps"),
    }
    if conveyor_model is not None:
        models_by_key["conveyor"] = (conveyor_model, CONVEYOR_VARIABLES, "conveyor")
    by_var: Counter[str] = Counter()
    by_var_safe: Counter[str] = Counter()
    ticks_by_model: dict[str, int] = {}
    unmatched_by_model: dict[str, int] = {}
    unmatched = unmatched_safe = missed = 0
    examples: list[tuple[str, Hashable, str, Hashable, int, bool]] = []
    for key, tally in tallies.items():
        model, variables, prefix = models_by_key[key]
        outcomes = _Outcomes(model)
        tested = bad = 0
        for (s, c, new), count in tally.triples.items():
            tested += count
            out = outcomes(s, c)
            if new in out:
                continue
            bad += count
            names = _closest_difference(variables, new, out)
            safe = VIOL not in out
            examples.append((key, s, c, new, count, not safe))
            unmatched_safe += count if safe else 0
            for v in names:
                by_var[f"{prefix}.{v}"] += count
                if safe:
                    by_var_safe[f"{prefix}.{v}"] += count
        for (s, c), count in tally.edge_triples.items():
            if VIOL not in outcomes(s, c):
                missed += count
        ticks_by_model[key] = tested
        unmatched_by_model[key] = bad
        unmatched += bad

    ticks = sum(ticks_by_model.values())
    in_violation = sum(t.in_violation for t in tallies.values())
    ticks_total = sum(t.ticks_total for t in tallies.values())
    return ConformanceReport(
        ticks=ticks,
        unmatched=unmatched,
        by_variable=dict(sorted(by_var.items())),
        a_phys_violations=a_phys,
        wilson=wilson(unmatched, ticks),
        episodes=len(traces),
        ticks_total=ticks_total,
        ticks_in_violation=in_violation,
        multi_command_ticks=sum(t.multi for t in tallies.values()),
        violation_edges=sum(t.edges for t in tallies.values()),
        missed_violations=missed,
        unmatched_when_safe=unmatched_safe,
        by_variable_when_safe=dict(sorted(by_var_safe.items())),
        ticks_by_model=ticks_by_model,
        unmatched_by_model=unmatched_by_model,
        a_phys_ticks=phys_ticks,
        a_phys_by_kind={"bin_skip": skips, "failure_below_top": early_failures},
        a_phys_wilson=wilson(a_phys, phys_ticks),
        bin_moves={k: moves.get(k, 0) for k in _BIN_MOVE_KINDS},
        unmatched_examples=tuple(
            sorted(examples, key=lambda e: (-e[4], e[0], repr(e)))[:_EXAMPLES]
        ),
    )


def _closest_difference(
    variables: Sequence[str], t: Hashable, outcomes: frozenset[Hashable | str]
) -> tuple[str, ...]:
    """The variables in which ``t`` differs from the nearest non-sink outcome."""
    best: tuple[str, ...] | None = None
    mine = cast("tuple[Hashable, ...]", t)
    for o in sorted((o for o in outcomes if o != VIOL), key=repr):
        theirs = cast("tuple[Hashable, ...]", o)
        diff = tuple(v for v, x, y in zip(variables, mine, theirs, strict=True) if x != y)
        if best is None or len(diff) < len(best):
            best = diff
    return best if best is not None else ("state",)


# ------------------------------------------------------------------------------------------
# records
# ------------------------------------------------------------------------------------------


def result_records(
    report: ConformanceReport, *, measured_on: str, script: str, fidelity: Fidelity = Fidelity.F1
) -> list[ResultRecord]:
    """One ``ResultRecord`` per published number, each with its denominator and a Wilson interval.

    ``measured_on`` is stated by the caller (which proposer, shield on or off, how many episodes).
    """

    def record(name: str, k: int, n: int, **extra: float | int | str) -> ResultRecord:
        lo, hi = wilson(k, n)
        return ResultRecord(
            name=name,
            value=(k / n) if n else 0.0,
            n=n,
            fidelity=fidelity,
            measured_on=measured_on,
            script=script,
            ci_lo=lo,
            ci_hi=hi,
            ci_kind="wilson-0.95",
            extra={"k": k, **extra},
        )

    out = [
        record("conformance.unmatched", report.unmatched, report.ticks, episodes=report.episodes)
    ]
    out.append(record("conformance.unmatched_when_safe", report.unmatched_when_safe, report.ticks))
    for model, n in sorted(report.ticks_by_model.items()):
        out.append(record(f"conformance.{model}.unmatched", report.unmatched_by_model[model], n))
    for variable in sorted(report.by_variable):
        k, n = report.variable_rate(variable)
        out.append(record(f"conformance.unmatched.{variable}", k, n))
    for variable in sorted(report.by_variable_when_safe):
        k, n = report.variable_rate(variable, when_safe=True)
        out.append(record(f"conformance.unmatched_when_safe.{variable}", k, n))
    out.append(
        record("conformance.missed_violations", report.missed_violations, report.violation_edges)
    )
    out.append(record("conformance.a_phys", report.a_phys_violations, report.a_phys_ticks))
    for kind in A_PHYS_KINDS:
        out.append(
            record(
                f"conformance.a_phys.{kind}",
                report.a_phys_by_kind.get(kind, 0),
                report.a_phys_ticks,
            )
        )
    for kind in ("fall", "rise_while_not_running"):
        out.append(
            record(
                f"conformance.bin_move.{kind}", report.bin_moves.get(kind, 0), report.a_phys_ticks
            )
        )
    return out
