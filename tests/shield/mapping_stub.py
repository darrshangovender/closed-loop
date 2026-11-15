"""Test-only stand-ins for the plant side of the shield runtime.

P5c writes the real ``AbstractionMap`` for the F1 plant; the plant package did not exist when P5a
was built. These classes carry only what ``ShieldedProposer`` reads, so the runtime can be tested
end to end against the abstract models themselves. Since ADR 0003 A12 the stand-in plant has the
conveyor CV-101 as well, and a pump has the commands LOCK and PROVE under their own names.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from closed_loop.core.types import Cmd, Command
from closed_loop.shield.abstraction import (
    CRUSHER_COMMANDS,
    PUMP_COMMANDS,
    VIOL,
    ConveyorModel,
    CrusherJobModel,
    PumpPairModel,
)

CRUSHER = "CR-501"
PUMPS = ("PU-201", "PU-202")
CONVEYOR = "CV-101"


@dataclass(frozen=True)
class StubView:
    asset_id: str
    op: str
    permit: str = "NONE"
    occupied: bool = False
    gas_age_q: int = 2  # quarters since the last test; 2 with retest_q = 2 is the stale value 3
    alarm: bool = False
    soon: bool = False
    set_lo: int = 0
    set_hi: int = 0


@dataclass(frozen=True)
class StubObservation:
    assets: Mapping[str, StubView]
    monitor_fired: bool = False


def observation(
    crusher: StubView,
    a: StubView,
    b: StubView,
    *,
    fired: bool = False,
    conveyor: StubView | None = None,
) -> StubObservation:
    belt = conveyor if conveyor is not None else StubView(CONVEYOR, "RUN")
    return StubObservation({CRUSHER: crusher, PUMPS[0]: a, PUMPS[1]: b, CONVEYOR: belt}, fired)


def quiet_pump(asset_id: str, op: str = "RUN", lo: int = 0, hi: int = 0) -> StubView:
    return StubView(asset_id, op, set_lo=lo, set_hi=hi)


class StubMapper:
    """Concrete to abstract for the stub views (the plant's command names are the abstract ones)."""

    def __init__(self, retest_q: int = 2) -> None:
        self.stale = retest_q + 1

    def crusher_state(self, view: StubView, health_bin: int) -> tuple[Any, ...]:
        gas = min(view.gas_age_q + 1, self.stale)
        return (
            view.op,
            health_bin,
            view.permit,
            int(view.occupied),
            gas,
            int(view.alarm),
            int(view.soon),
        )

    def pump_pair_state(
        self, a: StubView, b: StubView, bins: tuple[int, int]
    ) -> tuple[str, int, str, int]:
        return (a.op, bins[0], b.op, bins[1])

    def conveyor_state(self, view: StubView, health_bin: int) -> tuple[str, int]:
        return (view.op, health_bin)

    def command(self, cmd: Command) -> str:
        name = str(cmd.cmd.value)
        if cmd.asset_id in PUMPS or cmd.asset_id == CONVEYOR:
            return name if name in PUMP_COMMANDS else "NOOP"
        return name


@dataclass
class ScriptedProposer:
    """Returns the same commands every call and remembers resets."""

    commands: Sequence[Command] = ()
    name: str = "scripted"
    resets: list[int] = field(default_factory=list)

    def reset(self, seed: int) -> None:
        self.resets.append(seed)

    def propose(self, obs: Any) -> list[Command]:
        return list(self.commands)


class RandomProposer:
    """Hostile: a random command for each of the four assets, every quarter."""

    name = "random"

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def reset(self, seed: int) -> None:
        pass

    def propose(self, obs: Any) -> list[Command]:
        out = [Command(CRUSHER, Cmd(CRUSHER_COMMANDS[int(self.rng.integers(14))]))]
        for pump in (*PUMPS, CONVEYOR):
            out.append(
                Command(pump, Cmd(PUMP_COMMANDS[int(self.rng.integers(len(PUMP_COMMANDS)))]))
            )
        return out


class AbstractPlant:
    """The two abstract models as a plant: random (adversarial-in-expectation) environment moves.

    ``observe`` reports a random perception set around the true bin of each component, so the
    contract C1 holds by construction; ``apply`` runs the commands through the same map.
    """

    def __init__(self, rng: np.random.Generator, mapper: StubMapper | None = None) -> None:
        self.rng = rng
        self.mapper = mapper or StubMapper()
        self.crusher_model = CrusherJobModel()
        self.pump_model = PumpPairModel()
        self.conveyor_model = ConveyorModel()
        self.crusher = self.crusher_model.initial()
        self.pumps = self.pump_model.initial()
        self.conveyor = self.conveyor_model.initial()
        self.violations = 0
        self._cache: dict[tuple[Any, str, str], list[Any]] = {}

    def _set_around(self, h: int) -> tuple[int, int]:
        return int(self.rng.integers(0, h + 1)), int(self.rng.integers(h, 4))

    def observe(self) -> StubObservation:
        op, h, permit, occ, gas, alarm, soon = self.crusher  # type: ignore[misc]
        lo, hi = self._set_around(h)
        crusher = StubView(CRUSHER, op, permit, bool(occ), gas - 1, bool(alarm), bool(soon), lo, hi)
        o1, h1, o2, h2 = self.pumps  # type: ignore[misc]
        (lo1, hi1), (lo2, hi2) = self._set_around(h1), self._set_around(h2)
        oc, hc = self.conveyor  # type: ignore[misc]
        loc, hic = self._set_around(hc)
        return observation(
            crusher,
            StubView(PUMPS[0], o1, set_lo=lo1, set_hi=hi1),
            StubView(PUMPS[1], o2, set_lo=lo2, set_hi=hi2),
            conveyor=StubView(CONVEYOR, oc, set_lo=loc, set_hi=hic),
        )

    def _move(self, kind: str, model: Any, state: Any, cmd: str) -> Any:
        key = (state, cmd, kind)
        if key not in self._cache:
            self._cache[key] = sorted(model.step(state, cmd), key=repr)
        options = self._cache[key]
        return options[int(self.rng.integers(len(options)))]

    def apply(self, commands: Sequence[Command]) -> None:
        by_asset: dict[str, str] = {}
        for c in commands:
            by_asset.setdefault(c.asset_id, self.mapper.command(c))
        nxt = self._move("c", self.crusher_model, self.crusher, by_asset.get(CRUSHER, "NOOP"))
        joint = "|".join(by_asset.get(p, "NOOP") for p in PUMPS)
        nxt_p = self._move("p", self.pump_model, self.pumps, joint)
        nxt_v = self._move("v", self.conveyor_model, self.conveyor, by_asset.get(CONVEYOR, "NOOP"))
        if VIOL in (nxt, nxt_p, nxt_v):
            self.violations += 1
            self.crusher, self.pumps = self.crusher_model.initial(), self.pump_model.initial()
            self.conveyor = self.conveyor_model.initial()
        else:
            self.crusher, self.pumps, self.conveyor = nxt, nxt_p, nxt_v
