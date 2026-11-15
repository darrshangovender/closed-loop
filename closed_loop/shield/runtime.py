"""The runtime shield: filters a proposer's commands through the tables and substitutes.

What it enforces. For the crusher job, the pump pair and the conveyor CV-101 (ADR 0003 A12.3),
every command that leaves it is in the table's allowed set for the observable abstract state and
the perception set the plant reports. If the proposer's command (or its silence, which is a NOOP)
is not allowed, the first allowed command of the table's priority order replaces it and the
override is counted. For the pump pair, whose command is a joint one, the replacement is the
allowed joint command that changes the fewest pumps' commands, priority order breaking ties, so a
pump that did nothing wrong is left alone. Anything addressed to another asset, and every planning
command (INSPECT, ORDER, RAISE_WO), passes through untouched and is outside the proof.

What that is worth. The guarantee is the table's: finite abstraction only, under the assumptions
the abstraction names (A_exec, A_phys, A_iso, A_dec), with the abstraction map correct (A_map) and
with the perception contract C1 holding. C1 is read per life (ADR 0003, A6): the set must contain
the true bin at every tick of a life, which a per-step 99% set does not deliver. A wider set costs
throughput, not safety, so when the coverage monitor has fired the sets are widened to the whole
range and the unconditional part of the proof still stands.

One command per component per quarter: the abstraction has one controller move per tick, so a
second command to the same component in the same quarter is dropped (counted as an override).
The abstract command names of a pump and of the conveyor are the plant's (LOCK and PROVE map to
themselves since ADR 0003 A12.2), so a replacement is the plant command of the same name.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from typing import Any, Protocol

from closed_loop.core.types import N_BINS, Cmd, Command, PlanCmd
from closed_loop.shield.abstraction import AbstractModel, model_from_parameters
from closed_loop.shield.table import ShieldTable

CRUSHER_ID = "CR-501"
PUMP_IDS: tuple[str, str] = ("PU-201", "PU-202")
CONVEYOR_ID = "CV-101"
SOURCE = "shield"


class AbstractionMap(Protocol):
    """Concrete to abstract. ``view`` is the plant's ``AssetView`` (typed loosely: the plant
    package is built in parallel with this one)."""

    def crusher_state(self, view: Any, health_bin: int) -> Hashable: ...

    def pump_pair_state(self, a: Any, b: Any, bins: tuple[int, int]) -> Hashable: ...

    def conveyor_state(self, view: Any, health_bin: int) -> Hashable: ...

    def command(self, cmd: Command) -> str: ...  # a plant command without an abstract name -> NOOP


class _View(Protocol):
    @property
    def set_lo(self) -> int: ...

    @property
    def set_hi(self) -> int: ...


class _Observation(Protocol):
    @property
    def assets(self) -> Mapping[str, Any]: ...

    @property
    def monitor_fired(self) -> bool: ...


class _Proposer(Protocol):
    name: str

    def reset(self, seed: int) -> None: ...

    def propose(self, obs: Any) -> list[Command]: ...


def _concrete(asset_id: str, abstract: str) -> Command:
    """The plant command of an abstract name (every abstract name is a plant command's name)."""
    return Command(asset_id, Cmd(abstract), source=SOURCE)


class ShieldedProposer:
    """Wraps a proposer; see the module docstring.

    Counters, per decision (the crusher is one decision per quarter, the pump pair is one, the
    conveyor is one):
    ``admitted`` counts decisions in which the inner proposer issued a command for the component
    and it was allowed as it stood; ``overrides`` counts decisions in which the shield replaced or
    dropped something, including a forced command put in place of silence. ``uncovered`` counts
    overrides made with an empty allowed set (no command known safe), where the shield falls back
    to the first element of the priority order; a non-zero value means the contract or an
    assumption had already failed.

    ``monitor_fired`` is called each quarter; when it is true every perception set is widened to
    the whole range. If it is None, the observation's own ``monitor_fired`` flag is used.
    ``crusher_model``, ``pumps_model`` and ``conveyor_model`` default to the models recorded in the
    tables' meta; they only supply ``observable`` (to drop the hidden bin from the mapped state).

    ``conveyor`` is required: a shield without it would leave CV-101 outside the proof, which is
    the hole ADR 0003 A12.3 closes (the plant recorded P1 events there that M-VIOL did not count).
    """

    def __init__(
        self,
        inner: _Proposer,
        crusher: ShieldTable,
        pumps: ShieldTable,
        mapper: AbstractionMap,
        monitor_fired: Callable[[], bool] | None = None,
        *,
        conveyor: ShieldTable,
        crusher_id: str = CRUSHER_ID,
        pump_ids: tuple[str, str] = PUMP_IDS,
        conveyor_id: str = CONVEYOR_ID,
        crusher_model: AbstractModel | None = None,
        pumps_model: AbstractModel | None = None,
        conveyor_model: AbstractModel | None = None,
    ) -> None:
        self.inner = inner
        self.crusher = crusher
        self.pumps = pumps
        self.conveyor = conveyor
        self.mapper = mapper
        self.monitor_fired = monitor_fired
        self.crusher_id = crusher_id
        self.pump_ids = pump_ids
        self.conveyor_id = conveyor_id
        self._crusher_model = crusher_model or model_from_parameters(crusher.model_parameters())
        self._pumps_model = pumps_model or model_from_parameters(pumps.model_parameters())
        self._conveyor_model = conveyor_model or model_from_parameters(conveyor.model_parameters())
        self.name = f"shield({inner.name})"
        self.overrides = 0
        self.admitted = 0
        self.uncovered = 0

    def reset(self, seed: int | None = None) -> None:
        """Clear the counters; with a seed, reset the inner proposer too."""
        self.overrides = 0
        self.admitted = 0
        self.uncovered = 0
        if seed is not None:
            self.inner.reset(seed)

    def propose(self, obs: _Observation) -> list[Command]:
        proposed = self.inner.propose(obs)
        fired = self.monitor_fired() if self.monitor_fired is not None else obs.monitor_fired
        gated = {self.crusher_id, *self.pump_ids, self.conveyor_id}
        # A command on a gated asset passes only as a PlanCmd (maintenance planning is outside the
        # shield) or through the gate as a Cmd. Anything else is dropped and counted: the gate does
        # not rest on ``Command.__post_init__`` (P12g, G6).
        out = [c for c in proposed if c.asset_id not in gated or isinstance(c.cmd, PlanCmd)]
        mine = [c for c in proposed if c.asset_id in gated and isinstance(c.cmd, Cmd)]
        self.overrides += sum(
            1 for c in proposed if c.asset_id in gated and not isinstance(c.cmd, Cmd | PlanCmd)
        )
        out.extend(self._crusher(obs, [c for c in mine if c.asset_id == self.crusher_id], fired))
        out.extend(self._pumps(obs, [c for c in mine if c.asset_id in self.pump_ids], fired))
        out.extend(self._conveyor(obs, [c for c in mine if c.asset_id == self.conveyor_id], fired))
        return out

    # ------------------------------------------------------------------ crusher

    def _crusher(self, obs: _Observation, cmds: list[Command], fired: bool) -> list[Command]:
        view = obs.assets.get(self.crusher_id)
        if view is None:
            raise ValueError(f"the observation has no view of {self.crusher_id}")
        bins = _health_set(view, fired)
        state = self.mapper.crusher_state(view, bins[0])
        allowed = self.crusher.allowed(self._crusher_model.observable(state), bins)
        return self._single(self.crusher, self.crusher_id, allowed, cmds)

    # ------------------------------------------------------------------ conveyor

    def _conveyor(self, obs: _Observation, cmds: list[Command], fired: bool) -> list[Command]:
        view = obs.assets.get(self.conveyor_id)
        if view is None:
            raise ValueError(f"the observation has no view of {self.conveyor_id}")
        bins = _health_set(view, fired)
        state = self.mapper.conveyor_state(view, bins[0])
        allowed = self.conveyor.allowed(self._conveyor_model.observable(state), bins)
        return self._single(self.conveyor, self.conveyor_id, allowed, cmds)

    def _single(
        self, table: ShieldTable, asset_id: str, allowed: frozenset[str], cmds: list[Command]
    ) -> list[Command]:
        """The gate of a component that takes one command a quarter (the crusher, the conveyor)."""
        if len(cmds) > 1:
            self.overrides += len(cmds) - 1  # surplus commands are dropped, not gated
        chosen = cmds[0] if cmds else None
        proposed = self.mapper.command(chosen) if chosen is not None else "NOOP"
        if proposed in allowed:
            if chosen is None:
                return []
            self.admitted += 1
            return [chosen]
        replacement = self._substitute(table, allowed, proposed)
        if replacement == "NOOP":
            return []
        return [_concrete(asset_id, replacement)]

    # ------------------------------------------------------------------ pump pair

    def _pumps(self, obs: _Observation, cmds: list[Command], fired: bool) -> list[Command]:
        a_id, b_id = self.pump_ids
        a, b = obs.assets.get(a_id), obs.assets.get(b_id)
        if a is None or b is None:
            raise ValueError(f"the observation has no view of {a_id} or {b_id}")
        sets = (_health_set(a, fired), _health_set(b, fired))
        state = self.mapper.pump_pair_state(a, b, (sets[0][0], sets[1][0]))
        allowed = self.pumps.allowed(self._pumps_model.observable(state), sets[0] + sets[1])
        chosen: dict[str, Command] = {}
        for c in cmds:
            if c.asset_id in chosen:
                self.overrides += 1  # one command per pump per quarter
            else:
                chosen[c.asset_id] = c
        parts = [self.mapper.command(chosen[i]) if i in chosen else "NOOP" for i in (a_id, b_id)]
        joint = "|".join(parts)
        if joint in allowed:
            self.admitted += bool(chosen)
            return list(chosen.values())
        replacement = self._substitute(self.pumps, allowed, joint).split("|")
        out: list[Command] = []
        for pump_id, part, old in zip((a_id, b_id), replacement, parts, strict=True):
            if part == old and pump_id in chosen:
                out.append(chosen[pump_id])  # this pump's own command survived
            elif part != "NOOP":
                out.append(_concrete(pump_id, part))
        return out

    # ------------------------------------------------------------------ shared

    def _substitute(self, table: ShieldTable, allowed: frozenset[str], proposed: str) -> str:
        """The allowed command that changes the fewest components of ``proposed``, ties broken by
        the table's priority order. For the crusher (one component) that is plain priority; for
        the pump pair it leaves a pump alone when only the other one needs a different command."""
        self.overrides += 1
        if not allowed:
            self.uncovered += 1
            return table.priority[0]
        rank = {c: i for i, c in enumerate(table.priority)}
        want = proposed.split("|")

        def cost(c: str) -> tuple[int, int]:
            return sum(a != b for a, b in zip(c.split("|"), want, strict=True)), rank[c]

        return min(allowed, key=cost)


def _health_set(view: _View, fired: bool) -> tuple[int, int]:
    """The perception set to quantify over: the whole range when the monitor has fired or the
    reported set is not an interval of bins (the conservative reading of a bad set)."""
    lo, hi = view.set_lo, view.set_hi
    if fired or not (0 <= lo <= hi < N_BINS):
        return (0, N_BINS - 1)
    return (lo, hi)
