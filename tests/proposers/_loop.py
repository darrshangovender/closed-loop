"""A bare episode loop for the proposer tests: plant, optional perception, one proposer.

It is not the episode loop of P12a (arms, monitors, spec evaluation). It does what the tests need:
step the SYNTHETIC plant with a proposer's commands, optionally feed the plant a perception output
every day, and keep the plant's own violation records and the trace.

The perception wiring uses the estimator and a conformal provider from ``closed_loop.perception``.
The estimator is told the renewal of a component by reading the plant's life id from
``plant.truth``: that is the simulation side of the loop acting as the maintenance system that knows
when a part was fitted, not a proposer reading truth. The day's interval reaches the proposer from
the next quarter.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache

from closed_loop.core.trace import Trace
from closed_loop.core.types import Command
from closed_loop.perception import TheilSenEstimator, development_data, health_set
from closed_loop.perception.conformal import DevelopmentData
from closed_loop.plant import PlantParams, make_plant
from closed_loop.plant.plant import F1Plant, Observation, StepResult
from closed_loop.plant.scenario import Scenario
from closed_loop.proposers.base import Proposer

DEV_SEEDS = tuple(range(1000, 1008))


@cache
def dev_data() -> DevelopmentData:
    return development_data(PlantParams(), DEV_SEEDS)


@dataclass
class Episode:
    seed: int
    violations: list[tuple[str, str, int]] = field(default_factory=list)
    issued: list[tuple[int, Command]] = field(default_factory=list)
    executed: list[tuple[int, Command]] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    trace: Trace | None = None
    plant: F1Plant | None = None

    def violated(self, *props: str) -> int:
        return sum(1 for p, _a, _q in self.violations if p in props)


def run_episode(
    proposer: Proposer,
    seed: int,
    *,
    perception: bool = False,
    scenario: Scenario | None = None,
    params: PlantParams | None = None,
    keep_observations: bool = False,
    horizon_q: int | None = None,
    wrap: Callable[[F1Plant, Proposer], Proposer] | None = None,
    each_quarter: Callable[[F1Plant], None] | None = None,
) -> Episode:
    """Run one episode of ``proposer`` on the plant; ``horizon_q`` shortens it (default: all).

    ``wrap`` builds the proposer actually run from the plant and ``proposer`` (a shield or an
    interlock); ``each_quarter`` is called with the plant before every decision (the simulation
    side of the loop, for example a perception layer that is handed the truth).
    """
    p = params if params is not None else PlantParams()
    plant = make_plant(p)
    obs = plant.reset(seed, scenario if scenario is not None else Scenario())
    active = wrap(plant, proposer) if wrap is not None else proposer
    active.reset(seed)
    ep = Episode(seed=seed, plant=plant)
    estimator = TheilSenEstimator(p) if perception else None
    provider = dev_data().provider("V3", alpha=0.10, seed=seed) if perception else None
    last_day = -1
    n = plant.params.horizon_q if horizon_q is None else horizon_q
    for _ in range(n):
        if estimator is not None and provider is not None and obs.day != last_day:
            last_day = obs.day
            truth_life = plant.truth.life_id
            for aid, view in obs.assets.items():
                out = estimator.update(aid, obs.day, view.hi_last, truth_life[aid])
                iv = provider.predict(out, 0.90)
                lo, hi = health_set(iv)
                plant.set_perception(
                    aid, set_lo=lo, set_hi=hi, rul_lo=iv.lo, rul_hi=iv.hi, prior_only=iv.prior
                )
        if each_quarter is not None:
            each_quarter(plant)
        if keep_observations:
            ep.observations.append(obs)
        commands = active.propose(obs)
        ep.issued.extend((obs.quarter, c) for c in commands)
        res: StepResult = plant.step(commands)
        ep.executed.extend((obs.quarter, c) for c in res.executed)
        ep.violations.extend(res.violations)
        obs = res.obs
        if res.done:
            break
    ep.trace = plant.trace()
    return ep
