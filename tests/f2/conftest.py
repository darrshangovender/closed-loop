"""Shared F2 fixtures. The sibling pipeline is slow (a run is 2 to 3 s on a quiet machine and the
import is 30 to 45 s), so every run an F2 test needs is made once per session here.

Everything is SYNTHETIC. These tests run the real asset-health, maintenance-agent and plant-copilot
code; nothing in this folder mocks a sibling.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import pytest

from closed_loop.core.rng import rng_for
from closed_loop.eval.context import ContextSpec, PreparedContext, prepare_context
from closed_loop.eval.episode import scenario_for
from closed_loop.f2._siblings import load
from closed_loop.f2.agent import (
    build_orchestrator,
    new_mock_cmms,
    new_registry,
    register_approver,
)
from closed_loop.f2.alert_source import AssetHealthAlertSource, RunCache, aware
from closed_loop.f2.cmms import PlantCMMS, inventory_of
from closed_loop.f2.human import SimHumanApprover
from closed_loop.f2.render import LifeLatent, RenderedTelemetry, render_life, sim_time
from closed_loop.f2.splice import run_segments
from closed_loop.plant.params import PARTS, PlantParams

SEED = 3  # the spike's third seed; the alert of life 1 is open before the replacement
ASSET_MAP = {"PU-201": "PMP-01"}


@pytest.fixture(scope="session")
def params() -> PlantParams:
    return PlantParams()


def pump_lives(params: PlantParams, seed: int = SEED) -> list[RenderedTelemetry]:
    """Pump PU-201: impeller wear until a planned replacement at day 40, then a bearing life.

    The asset is stopped for six hours (restart at day 40.25). Life 1 would have failed at day 52;
    life 2 starts its degradation 25 days after the restart.
    """
    first = LifeLatent("PU-201", 0, "IMPELLER_WEAR", 24.0, 52.0, 2.0, 48.0, 0.0, 40.0)
    second = LifeLatent("PU-201", 1, "BEARING_DEFECT", 65.25, 95.25, 3.0, 60.0, 40.25, None, True)
    return [render_life(first, params, seed, 0), render_life(second, params, seed, 1)]


@pytest.fixture(scope="session")
def lives(params: PlantParams) -> list[RenderedTelemetry]:
    return pump_lives(params)


@pytest.fixture(scope="session")
def spliced_run(lives: list[RenderedTelemetry]):
    return run_segments(lives)


@pytest.fixture(scope="session")
def life1_run(lives: list[RenderedTelemetry]):
    """The pipeline on life 1 alone, up to the stop: the single-life reference."""
    return run_segments(lives[:1])


@pytest.fixture(scope="session")
def pump_cache(lives: list[RenderedTelemetry]) -> RunCache:
    """The pipeline's run on life 1 of PU-201 (an alert is open by day 39.5), made once."""
    cache = RunCache()
    cache.set_history("PU-201", lives[:1])
    return cache


# ----------------------------------------------------------------------------- the agent side


@dataclass
class World:
    """One episode's agent side: clock, alert source, registry, CMMS, job queue, orchestrator."""

    params: PlantParams
    now: datetime
    stock: dict[str, int]
    source: Any = None
    queued: list[tuple[str, Any]] = field(default_factory=list)
    keys: dict[str, str] = field(default_factory=dict)
    registry: Any = None
    mock: Any = None
    cmms: Any = None
    orch: Any = None

    def clock(self) -> datetime:
        return aware(self.now)

    def advance(self, **delta: float) -> None:
        self.now = self.now + timedelta(**delta)

    def human(self, approver_id: str = "planner1", **kw: Any) -> SimHumanApprover:
        roles = kw.pop("roles", ("PLANNER",))
        rng = kw.pop("rng", rng_for(1, "approver", approver_id))
        params = kw.pop("params", self.params)
        return SimHumanApprover(
            approver_id, self.keys[approver_id], self.clock, params, rng, roles=roles, **kw
        )


def make_world(
    params: PlantParams,
    source_factory: Callable[[Callable[[], datetime]], Any],
    *,
    now: datetime,
    stock: dict[str, int] | None = None,
    queue: Callable[[Any, str], None] | None = None,
) -> World:
    """Wire the real orchestrator to ``source_factory(clock)`` and a ``PlantCMMS``.

    ``queue(draft, wo_number)`` receives each released order; the default records it in
    ``World.queued``.
    """
    world = World(
        params=params,
        now=now,
        stock=dict(stock) if stock is not None else {p: s.stock for p, s in PARTS.items()},
    )
    world.registry = new_registry()
    world.keys["planner1"] = register_approver(
        world.registry, "planner1", "Planner One", ["PLANNER"]
    )
    world.keys["officer1"] = register_approver(
        world.registry, "officer1", "Officer One", ["SAFETY_OFFICER"]
    )
    world.source = source_factory(world.clock)
    world.mock = new_mock_cmms(world.registry, world.clock)
    world.cmms = PlantCMMS(
        world.mock,
        inventory_of(lambda: world.stock),
        queue if queue is not None else (lambda draft, wo: world.queued.append((wo, draft))),
    )
    world.orch = build_orchestrator(world.source, world.cmms, world.registry, world.clock)
    return world


@pytest.fixture
def pump_world(params: PlantParams, pump_cache: RunCache) -> World:
    """The asset-health pump alert of day 39.5 (impeller wear, life 1) wired to the agent."""
    return make_world(
        params,
        lambda clock: AssetHealthAlertSource(pump_cache, clock, ASSET_MAP),
        now=sim_time(39.5),
    )


@pytest.fixture
def fixture_world(params: PlantParams) -> World:
    """The sibling's own bundled alerts, re-dated to the simulation date, on the same wiring."""
    now = sim_time(39.5)
    synthetic = load("maintenance_agent.adapters.synthetic")
    return make_world(
        params,
        lambda clock: synthetic.SyntheticAlertSource.from_fixtures(today=aware(now).date()),
        now=now,
    )


# ----------------------------------------------------------------------------- P11c: F2 episodes

EPISODE_SEED = 1023  # a development seed: crusher and conveyor bearings fail inside the episode
SMALL_SPEC = ContextSpec(
    conformal_seeds=(2000, 2040),
    proxy_seeds=(2000, 2040),
    diagnoser_seeds=(2300, 2314),
    world_model_seeds=(1000, 1020),
)


@pytest.fixture(scope="session")
def small_context() -> PreparedContext:
    """The small fitted context of the eval tests (same spec; the f2 folder cannot see theirs)."""
    return prepare_context(SMALL_SPEC)


@dataclass(frozen=True)
class Timed:
    """One F2 episode with the wall-clock and CPU seconds the call took."""

    record: Any
    wall_s: float
    cpu_s: float


@pytest.fixture(scope="session")
def f2_episodes(small_context: PreparedContext) -> dict[str, Timed]:
    """The three F2 arms on one development seed, run once per session."""
    from closed_loop.f2.episode import F2_ARMS, simulate_f2_episode

    out: dict[str, Timed] = {}
    for name, arm in F2_ARMS.items():
        wall, cpu = time.perf_counter(), time.process_time()
        record = simulate_f2_episode(
            arm, EPISODE_SEED, scenario_for("nominal"), context=small_context
        )
        out[name] = Timed(record, time.perf_counter() - wall, time.process_time() - cpu)
    return out
