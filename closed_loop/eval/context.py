"""What an episode needs that is fitted once per process, not once per episode (packet P12a).

An episode needs four fitted things and two derived ones:

* the conformal development scores (``DevelopmentData``) from ``calib`` seeds, for every interval
  provider V1 to V7 and the per-life band behind the shield's default set;
* the proxy monitor's band (``ProxyBand``), from the same seeds;
* the diagnosers (``scm`` and ``corr``, 0003 A3), fitted on labelled first lives of ``calib`` seeds
  of the unshifted plant (the closed loop's plant has no load-to-mode tilt);
* the planner's world model, fitted on ``dev`` seeds by ``planning.fitting`` (its own rule);
* the specification of the Rev 3 clause (the monitor every arm is scored against) and of the Rev 2
  clause (the stale-spec arm's shield), both from the checked-in clause cache, no sibling needed;
* the shield tables for both specs, as JSON text (``ShieldTable`` holds read-only mappings, which
  do not pickle; the tables are rebuilt from the text on first use in each process).

``PreparedContext`` is a frozen dataclass of arrays, numbers and fitted objects, so it **pickles**:
a runner builds it once and hands it to ``spawn`` workers through an initializer. A worker that is
not handed one rebuilds the default one on first use, which is deterministic (same seeds, same
fits), at a cost of about half a minute. ``ContextSpec`` says what it was built from.

Seed roles (0001 section 11): the conformal pools, the proxy band and the diagnosers use ``calib``
seeds, the world model uses ``dev`` seeds; ``require_calibration_seeds`` refuses anything else, so a
search seed or a derived held-out seed cannot calibrate or train anything.

Proposer factories in ``arms.py`` take no argument (``Arm.proposer`` is ``Callable[[], Proposer]``,
frozen in 0002 3.13). They find the context through ``current_context()``: the one installed by
``use_context`` (``run_episode`` does this around an episode) or, failing that, the process default.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import date
from hashlib import sha256
from typing import TYPE_CHECKING, Final

from closed_loop.audit.seeds import CALIB_SEEDS
from closed_loop.causal.diagnoser import Diagnoser, make_diagnoser
from closed_loop.causal.twin import passive_world
from closed_loop.perception.conformal import (
    DevelopmentData,
    development_data,
    require_calibration_seeds,
)
from closed_loop.perception.monitor import PROXY_ALPHA, ProxyBand, proxy_development_scores
from closed_loop.plant.params import ASSET_CLASS, ASSET_IDS, PlantParams
from closed_loop.shield.abstraction import ConveyorModel, CrusherJobModel, PumpPairModel
from closed_loop.shield.conformance import F1AbstractionMap
from closed_loop.shield.game import solve
from closed_loop.shield.table import ShieldTable, build_shield
from closed_loop.spec.clauses import spec_for
from closed_loop.spec.properties import SpecParams

if TYPE_CHECKING:
    from closed_loop.causal.records import LifeRecord
    from closed_loop.planning.worldmodel import PlantWorldModel

__all__ = [
    "CLAUSE_AS_OF",
    "STALE_AS_OF",
    "ContextSpec",
    "PreparedContext",
    "ShieldKit",
    "current_context",
    "default_context",
    "install_context",
    "prepare_context",
    "shield_kit",
    "use_context",
]

CLAUSE_AS_OF: Final[date] = date(2026, 10, 1)  # CORP-HSE-002 Rev 3: 30 minutes, 2 quarters
STALE_AS_OF: Final[date] = date(2026, 1, 1)  # Rev 2: 2 hours, 8 quarters
SHIELD_WIDTH: Final[int] = 4  # perception sets of 1 to 4 bins: every set the contract can produce


@dataclass(frozen=True)
class ContextSpec:
    """What a context is built from. Ranges are ``(first, stop)`` seeds, stop excluded."""

    conformal_seeds: tuple[int, int] = (2000, 2300)  # calib: 1,200 first lives (P4b)
    proxy_seeds: tuple[int, int] = (2000, 2300)  # calib: the one-day-ahead residuals (P4c)
    diagnoser_seeds: tuple[int, int] = (2300, 2375)  # calib: 75 seeds x 4 lives = 300 lives
    world_model_seeds: tuple[int, int] = (1000, 1100)  # dev: the planner's own fitting rule
    diagnosers: tuple[str, ...] = ("scm", "corr")

    def seeds(self, attr: str) -> range:
        lo, hi = getattr(self, attr)
        return range(lo, hi)


@dataclass(frozen=True)
class PreparedContext:
    """The fitted state shared by every episode of a process; see the module docstring."""

    spec: ContextSpec
    params: PlantParams
    dev: DevelopmentData
    proxy_band: ProxyBand
    diagnosers: Mapping[str, Diagnoser]
    world_fit: Mapping[str, float]
    spec_clause: SpecParams
    spec_stale: SpecParams
    shield_json: Mapping[str, str]  # "crusher:clause", "crusher:stale", "pumps", "conveyor"
    built_s: float = 0.0
    timings_s: Mapping[str, float] = field(default_factory=dict)

    def diagnoser(self, kind: str) -> Diagnoser:
        """A fitted ``scm`` or ``corr`` (shared: both are stateless after ``fit``) or a fresh,
        unbound ``oracle``, which the episode loop binds to the plant's truth."""
        if kind == "oracle":
            return make_diagnoser("oracle")
        try:
            return self.diagnosers[kind]
        except KeyError:
            raise KeyError(
                f"diagnoser {kind!r} was not prepared; this context holds {sorted(self.diagnosers)}"
            ) from None

    def world_models(self, mismatch: float = 0.0) -> dict[str, PlantWorldModel]:
        """One planner world model per asset, from the fit on ``dev`` seeds."""
        from closed_loop.planning.worldmodel import PlantWorldModel

        return {
            aid: PlantWorldModel(self.world_fit, mismatch, asset_class=ASSET_CLASS[aid])
            for aid in ASSET_IDS
        }

    def spec_for_shield(self, kind: str) -> SpecParams:
        if kind == "clause":
            return self.spec_clause
        if kind == "stale":
            return self.spec_stale
        raise ValueError(f"Arm.spec must be 'clause' or 'stale', got {kind!r}")


@dataclass(frozen=True)
class ShieldKit:
    """The tables and the map for one specification, built once per process."""

    crusher: ShieldTable
    pumps: ShieldTable
    conveyor: ShieldTable
    mapper: F1AbstractionMap
    crusher_model: CrusherJobModel
    pumps_model: PumpPairModel
    conveyor_model: ConveyorModel


def _solve_json(retest_q: int) -> tuple[str, str, str]:
    crusher = CrusherJobModel(retest_q)
    pumps = PumpPairModel()
    conveyor = ConveyorModel()
    return (
        build_shield(crusher, solve(crusher), SHIELD_WIDTH).to_json(),
        build_shield(pumps, solve(pumps), SHIELD_WIDTH).to_json(),
        build_shield(conveyor, solve(conveyor), SHIELD_WIDTH).to_json(),
    )


def _training_lives(params: PlantParams, seeds: range) -> list[LifeRecord]:
    lives: list[LifeRecord] = []
    for seed in seeds:
        lives.extend(passive_world(seed, params))
    return lives


def prepare_context(
    spec: ContextSpec | None = None, params: PlantParams | None = None
) -> PreparedContext:
    """Fit everything an episode needs. Deterministic; about half a minute for the default spec."""
    spec = spec if spec is not None else ContextSpec()
    params = params if params is not None else PlantParams()
    timings: dict[str, float] = {}
    t_all = time.perf_counter()

    def lap(label: str, t0: float) -> None:
        timings[label] = round(time.perf_counter() - t0, 3)

    for attr in ("conformal_seeds", "proxy_seeds", "diagnoser_seeds"):
        seeds = spec.seeds(attr)
        if not (seeds.start in CALIB_SEEDS and seeds.stop - 1 in CALIB_SEEDS):
            raise ValueError(f"{attr} must lie in the calib role 2000-2999, got {seeds}")
        require_calibration_seeds(seeds)

    t0 = time.perf_counter()
    dev = development_data(params, spec.seeds("conformal_seeds"))
    lap("conformal_pools", t0)

    t0 = time.perf_counter()
    band = ProxyBand.fit(proxy_development_scores(params, spec.seeds("proxy_seeds")), PROXY_ALPHA)
    lap("proxy_band", t0)

    t0 = time.perf_counter()
    fitted: dict[str, Diagnoser] = {}
    if spec.diagnosers:
        lives = _training_lives(params, spec.seeds("diagnoser_seeds"))
        for kind in spec.diagnosers:
            d = make_diagnoser(kind)
            d.fit(lives)
            fitted[kind] = d
    lap("diagnosers", t0)

    t0 = time.perf_counter()
    from closed_loop.planning.fitting import fit_world_model

    world = dict(fit_world_model(tuple(spec.seeds("world_model_seeds"))))
    lap("world_model", t0)

    t0 = time.perf_counter()
    spec_clause = spec_for(CLAUSE_AS_OF)
    spec_stale = spec_for(STALE_AS_OF)
    crusher_clause, pumps, conveyor = _solve_json(spec_clause.retest_q)
    crusher_stale, _, _ = _solve_json(spec_stale.retest_q)
    lap("shield_tables", t0)

    return PreparedContext(
        spec=spec,
        params=params,
        dev=dev,
        proxy_band=band,
        diagnosers=fitted,
        world_fit=world,
        spec_clause=spec_clause,
        spec_stale=spec_stale,
        shield_json={
            "crusher:clause": crusher_clause,
            "crusher:stale": crusher_stale,
            "pumps": pumps,
            "conveyor": conveyor,
        },
        built_s=round(time.perf_counter() - t_all, 3),
        timings_s=timings,
    )


# ----------------------------------------------------------------------------- per-process state

_PROCESS: dict[tuple[ContextSpec, PlantParams], PreparedContext] = {}
_CURRENT: ContextVar[PreparedContext | None] = ContextVar("closed_loop_context", default=None)


def default_context() -> PreparedContext:
    """The process-wide context of the default spec, built on first use."""
    key = (ContextSpec(), PlantParams())
    if key not in _PROCESS:
        _PROCESS[key] = prepare_context(*key)
    return _PROCESS[key]


def install_context(ctx: PreparedContext) -> None:
    """Make ``ctx`` the process default (the initializer of a ``spawn`` worker calls this)."""
    _PROCESS[(ContextSpec(), PlantParams())] = ctx


def current_context() -> PreparedContext:
    """The context of the running episode, else the process default."""
    ctx = _CURRENT.get()
    return ctx if ctx is not None else default_context()


@contextmanager
def use_context(ctx: PreparedContext) -> Iterator[PreparedContext]:
    token = _CURRENT.set(ctx)
    try:
        yield ctx
    finally:
        _CURRENT.reset(token)


# Tables are rebuilt from the JSON text once per process and kept, keyed by the text's hash.
_KITS: dict[tuple[str, str, int], ShieldKit] = {}


def shield_kit(ctx: PreparedContext, kind: str) -> ShieldKit:
    """Tables, map and models for the shield built from specification ``kind`` (clause or stale)."""
    retest_q = ctx.spec_for_shield(kind).retest_q
    crusher_text = ctx.shield_json[f"crusher:{kind}"]
    pumps_text = ctx.shield_json["pumps"]
    conveyor_text = ctx.shield_json["conveyor"]
    key = (
        sha256(crusher_text.encode()).hexdigest(),
        sha256((pumps_text + conveyor_text).encode()).hexdigest(),
        retest_q,
    )
    if key not in _KITS:
        _KITS[key] = ShieldKit(
            crusher=ShieldTable.from_json(crusher_text),
            pumps=ShieldTable.from_json(pumps_text),
            conveyor=ShieldTable.from_json(conveyor_text),
            mapper=F1AbstractionMap(retest_q),
            crusher_model=CrusherJobModel(retest_q),
            pumps_model=PumpPairModel(),
            conveyor_model=ConveyorModel(),
        )
    return _KITS[key]
