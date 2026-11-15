"""Arms of the F1 experiment: every row of the pre-registration's section 4 (packet P12a).

An ``Arm`` is data: which proposer, whether the shield or the hand-written interlock stands between
it and the plant, which perception variant feeds it, which specification the shield was built from,
and under which conditions the pre-registration runs it. ``ARMS`` is the registry; ``arm(name)``
looks one up.

Frozen in 0002 section 3.13: ``name``, ``proposer``, ``shield``, ``monitor``, ``perception``,
``screen``, ``spec``, ``planner_mismatch``. Added by 0003: ``diagnoser`` (A3), ``interlock`` (A4),
``shield_set`` (A6). Added here, all with defaults, because the registry has to say them somewhere:

``conditions``
    the conditions the arm runs under (``"nominal"``, ``"shifted"``): the "Conditions" column of the
    pre-registration, as data.
``monitor_mode``
    which coverage monitor acts when ``monitor`` is true (``opt``, ``pess``, ``inf``, ``proxy`` or
    ``audit``). Every episode also runs all of them passively; only this one can widen the
    perception sets.
``tuned``
    the names in ``eval.tuned`` the arm reads, so a result can say which of its numbers are
    provisional.

Pickling under ``spawn``. ``Arm.proposer`` is a ``functools.partial`` of a module-level function
with plain arguments, never a lambda or a closure, so an ``Arm`` (and the whole registry) pickles
and an unpickled copy builds the same proposer in a fresh interpreter. The factories take no
argument, as the frozen type says; they find the fitted state they need (diagnosers, the planner's
world model) through ``eval.context.current_context()`` and the tuned numbers through
``eval.tuned.current()``.

Module-level names that are decisions for the owner, not engineering: ``PRIMARY_VARIANT`` and
``MONITOR_MODE`` (read from ``benchmarks/tuned.json`` when it names them, else the provisional
``"V3"`` and ``"proxy"``),
``CARELESS_P_SKIP`` and ``PLANNER_PARTICLES`` (prereg section 2: ``N = 256``, fixed on ``dev``).

P12b added one arm and one constant to this file: ``B3-S`` (the bound rule behind the shield, the
comparison row C-PLAN needs for "P1 against B3 with shield", prereg section 7) and
``PLANNER_PARTICLES`` (the planner's particle count was the class default of 128 while the
pre-registration fixes 256).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Final

from closed_loop.eval.context import current_context
from closed_loop.eval.tuned import current as current_tuned
from closed_loop.perception.conformal import ALL_VARIANTS
from closed_loop.perception.contract import SHIELD_SETS
from closed_loop.perception.monitor import MODES
from closed_loop.planning.planner import PlanConstraints
from closed_loop.planning.proposer import PlannerProposer
from closed_loop.plant.notes import cached_scan
from closed_loop.plant.plant import Observation
from closed_loop.proposers import Careless, Gullible, make_baseline
from closed_loop.proposers.base import Proposer

__all__ = [
    "ARMS",
    "CARELESS_P_SKIP",
    "CONDITIONS",
    "MONITOR_MODE",
    "PLANNER_PARTICLES",
    "PRIMARY_VARIANT",
    "Arm",
    "NoInspectPlanner",
    "arm",
    "arms_for",
]

# The pre-registered primary perception variant of the planner arms (prereg section 7: chosen on
# ``dev`` among V2, V3, V4, V5, V2L and V3L by oracle lower-bound coverage under ``shifted``, ties
# broken by the narrower mean set). ``scripts/tune.py`` makes the choice and writes it to
# ``benchmarks/tuned.json``; until that file names one, "V3" is PROVISIONAL (the variant the hostile
# arms use, and a per-step one, so that P1-step's per-step 99% set means something).
PRIMARY_VARIANT: Final[str] = current_tuned().primary_variant

# The planner's particle count (prereg section 2: ``N = 256``, fixed on ``dev``, 0001 section 11).
PLANNER_PARTICLES: Final[int] = 256

# The coverage-monitor mode that acts in arm P2 and decides C-MON (0003 A6: "the mode chosen on
# dev"). Read from ``benchmarks/tuned.json`` when it names one, else PROVISIONAL "proxy" (the only
# mode that is not starved by the policy's own censoring and whose false-alarm rate on dev was low,
# P4c: 1 to 2 of 100 episodes; it tests the degradation path, not C1). ``scripts/tune.py`` chooses
# it by the rule of prereg section 3: the highest detection rate minus false-alarm rate on ``dev``,
# ties to the lower false-alarm rate; ``--freeze`` refuses a ``tuned.json`` that does not name one.
MONITOR_MODE: Final[str] = current_tuned().monitor_mode

# The Careless fitter's probability of an out-of-order step (P6b: 0.05; the pre-registration names
# no value, so this is the packet's choice and the same for every Careless arm).
CARELESS_P_SKIP: Final[float] = 0.05

CONDITIONS: Final[tuple[str, ...]] = ("nominal", "shifted")
_NOMINAL: Final[tuple[str, ...]] = ("nominal",)
_PERCEPTIONS: Final[frozenset[str]] = frozenset({*ALL_VARIANTS, "none"})
_MONITOR_MODES: Final[frozenset[str]] = frozenset({*MODES, "audit"})
_DIAGNOSERS: Final[frozenset[str]] = frozenset({"scm", "corr", "oracle"})


@dataclass(frozen=True)
class Arm:
    name: str
    proposer: Callable[[], Proposer]
    shield: bool
    monitor: bool
    perception: str = "V3"
    screen: bool = False
    spec: str = "clause"
    planner_mismatch: float = 0.0
    diagnoser: str = "scm"  # 0003 A3
    interlock: bool = False  # 0003 A4
    shield_set: str = "life95"  # 0003 A6
    # ---- added by P12a (defaults; see the module docstring)
    conditions: tuple[str, ...] = CONDITIONS
    monitor_mode: str = MONITOR_MODE
    tuned: tuple[str, ...] = ()
    # ---- added by P12e (0003 A18, R8): the operator stops the running pump beside a LOCK on the
    # other (a shield arm only); False is the arm as it was before R8, kept so that the pump
    # deadlock stays a generated number (``P1-nopair``)
    operator_pair: bool = True

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("an arm needs a name")
        if self.perception not in _PERCEPTIONS:
            raise ValueError(f"{self.name}: perception must be one of {sorted(_PERCEPTIONS)}")
        if self.spec not in ("clause", "stale"):
            raise ValueError(f"{self.name}: spec must be 'clause' or 'stale', got {self.spec!r}")
        if self.shield_set not in SHIELD_SETS:
            raise ValueError(f"{self.name}: shield_set must be one of {SHIELD_SETS}")
        if self.diagnoser not in _DIAGNOSERS:
            raise ValueError(f"{self.name}: diagnoser must be one of {sorted(_DIAGNOSERS)}")
        if self.shield and self.interlock:
            raise ValueError(f"{self.name}: the shield and the interlock are alternatives (A4)")
        if self.monitor_mode not in _MONITOR_MODES:
            raise ValueError(f"{self.name}: monitor_mode must be one of {sorted(_MONITOR_MODES)}")
        if not self.conditions or any(c not in CONDITIONS for c in self.conditions):
            raise ValueError(f"{self.name}: conditions must be a non-empty subset of {CONDITIONS}")
        if self.planner_mismatch < 0.0:
            raise ValueError(f"{self.name}: planner_mismatch cannot be negative")
        if (self.shield or self.monitor) and self.perception == "none":
            raise ValueError(f"{self.name}: the shield and the monitor need a perception layer")


# ----------------------------------------------------------------------------- proposer factories
# Module-level and argument-free once bound with ``partial``: they pickle by reference.


def make_run_to_failure() -> Proposer:
    return make_baseline("B0")


def make_calendar() -> Proposer:
    return make_baseline("B1", fraction=current_tuned().b1_fraction)


def make_fixed_limit() -> Proposer:
    return make_baseline("B2", limit=current_tuned().b2_limit)


def make_bound_rule(diagnoser: str | None = "scm") -> Proposer:
    """B3 with the tuned ``k``; ``diagnoser`` is the name of a fitted one from the context."""
    fitted = None if diagnoser is None else current_context().diagnoser(diagnoser)
    return make_baseline("B3", k=current_tuned().k_b3, diagnoser=fitted)


def make_careless(p_skip: float = CARELESS_P_SKIP, diagnoser: str | None = "scm") -> Proposer:
    return Careless(make_bound_rule(diagnoser), p_skip)


def make_gullible(screen: bool, diagnoser: str | None = "scm") -> Proposer:
    return Gullible(make_bound_rule(diagnoser), screen, cached_scan)


class NoInspectPlanner(PlannerProposer):
    """The planner with INSPECT disabled (prereg section 8): no inspection is ever started.

    It uses the planner's own limit, ``PlanConstraints.inspect_slots = 0``. The planner's
    assessment may still prefer an inspection; since none is started the asset simply runs on, which
    is what "information-gathering removed" means for the closed loop. A public switch in
    ``planning`` would be cleaner; the planner's files are not P12a's.
    """

    name = "planner-noinspect"

    def _constraints(
        self, obs: Observation, stalled: frozenset[str] = frozenset()
    ) -> PlanConstraints:
        return dataclasses.replace(super()._constraints(obs, stalled), inspect_slots=0)


def make_planner(
    diagnoser: str | None = "scm", mismatch: float = 0.0, inspect: bool = True
) -> Proposer:
    """The belief planner (P8b) on the context's world model, with the tuned ``delta_plan``."""
    ctx = current_context()
    fitted = None if diagnoser is None else ctx.diagnoser(diagnoser)
    cls = PlannerProposer if inspect else NoInspectPlanner
    return cls(
        model=ctx.world_models(mismatch),
        params=ctx.params,
        n_particles=PLANNER_PARTICLES,
        delta_plan=current_tuned().delta_plan,
        mismatch=mismatch,
        diagnoser=fitted,
    )


# ----------------------------------------------------------------------------- the registry


def _registry() -> dict[str, Arm]:
    arms: list[Arm] = []

    def add(**kw: object) -> None:
        arms.append(Arm(**kw))  # type: ignore[arg-type]

    # ---- B0, B1, B2: no perception layer (prereg section 4)
    add(name="B0", proposer=make_run_to_failure, shield=False, monitor=False, perception="none")
    add(
        name="B1",
        proposer=make_calendar,
        shield=False,
        monitor=False,
        perception="none",
        tuned=("b1_fraction",),
    )
    add(
        name="B2",
        proposer=make_fixed_limit,
        shield=False,
        monitor=False,
        perception="none",
        tuned=("b2_limit",),
    )

    # ---- B3 under each perception variant (the registered V1 to V7, then the life-unit forms)
    for variant in ("V1", "V2", "V3", "V4", "V5", "V6", "V7", "V2L", "V3L", "V7L"):
        add(
            name=f"B3-{variant}",
            proposer=partial(make_bound_rule, "scm"),
            shield=False,
            monitor=False,
            perception=variant,
            tuned=("k_b3",),
        )
    # the secondary reading of 0003 A6: V4's audit lives feed a mode-opt monitor of their own
    add(
        name="B3-V4-audit",
        proposer=partial(make_bound_rule, "scm"),
        shield=False,
        monitor=True,
        perception="V4",
        monitor_mode="audit",
        tuned=("k_b3",),
    )

    # ---- B3 behind the shield, under the primary variant: the "B3 with shield" of C-PLAN
    add(
        name="B3-S",
        proposer=partial(make_bound_rule, "scm"),
        shield=True,
        monitor=False,
        perception=PRIMARY_VARIANT,
        tuned=("k_b3", "primary_variant"),
    )

    # ---- the belief planner and its variants
    planner = {"perception": PRIMARY_VARIANT, "tuned": ("delta_plan", "primary_variant")}
    monitored = {**planner, "tuned": ("delta_plan", "primary_variant", "monitor_mode")}
    add(name="P0", proposer=make_planner, shield=False, monitor=False, **planner)
    add(name="P1", proposer=make_planner, shield=True, monitor=False, **planner)
    add(
        name="P2",
        proposer=make_planner,
        shield=True,
        monitor=True,
        monitor_mode=MONITOR_MODE,
        **monitored,
    )
    add(
        name="P1-corr",
        proposer=partial(make_planner, "corr"),
        shield=True,
        monitor=False,
        diagnoser="corr",
        **planner,
    )
    add(
        name="P1-oracle",
        proposer=partial(make_planner, "oracle"),
        shield=True,
        monitor=False,
        diagnoser="oracle",
        **planner,
    )
    add(name="P1-I", proposer=make_planner, shield=False, monitor=False, interlock=True, **planner)
    add(
        name="P1-nopair",
        proposer=make_planner,
        shield=True,
        monitor=False,
        operator_pair=False,
        **planner,
    )
    add(
        name="P1-step",
        proposer=make_planner,
        shield=True,
        monitor=False,
        shield_set="step99",
        **planner,
    )

    # ---- hostile proposers, nominal condition only
    hostile = {"perception": "V3", "conditions": _NOMINAL, "tuned": ("k_b3",)}
    careless = partial(make_careless, CARELESS_P_SKIP)
    add(name="H-C0", proposer=careless, shield=False, monitor=False, **hostile)
    add(name="H-C1", proposer=careless, shield=True, monitor=False, **hostile)
    add(name="H-C-I", proposer=careless, shield=False, monitor=False, interlock=True, **hostile)
    add(
        name="H-G0",
        proposer=partial(make_gullible, False),
        shield=False,
        monitor=False,
        **hostile,
    )
    add(
        name="H-G1",
        proposer=partial(make_gullible, False),
        shield=True,
        monitor=False,
        **hostile,
    )
    add(
        name="H-G-I",
        proposer=partial(make_gullible, False),
        shield=False,
        monitor=False,
        interlock=True,
        **hostile,
    )
    add(
        name="H-G2",
        proposer=partial(make_gullible, True),
        shield=False,
        monitor=False,
        screen=True,
        **hostile,
    )

    # ---- spec provenance: a shield from the superseded 2-hour clause against the 30-minute monitor
    add(name="S-stale", proposer=careless, shield=True, monitor=False, spec="stale", **hostile)
    add(name="S-clause", proposer=careless, shield=True, monitor=False, spec="clause", **hostile)

    # ---- registered ablations (prereg section 8), on the planner without the shield so that the
    # planner's own cost and failures are what moves
    add(
        name="P0-noinspect",
        proposer=partial(make_planner, "scm", 0.0, False),
        shield=False,
        monitor=False,
        **planner,
    )
    for pct in (10, 25):
        add(
            name=f"P0-mis{pct}",
            proposer=partial(make_planner, "scm", pct / 100.0),
            shield=False,
            monitor=False,
            planner_mismatch=pct / 100.0,
            **planner,
        )

    registry: dict[str, Arm] = {}
    for a in arms:
        if a.name in registry:
            raise ValueError(f"duplicate arm name {a.name!r}")
        registry[a.name] = a
    return registry


ARMS: Final[Mapping[str, Arm]] = _registry()


def arm(name: str) -> Arm:
    """The registered arm called ``name``."""
    try:
        return ARMS[name]
    except KeyError:
        raise KeyError(f"unknown arm {name!r}; the registry holds {sorted(ARMS)}") from None


def arms_for(condition: str) -> tuple[Arm, ...]:
    """The arms the registry runs under ``condition``, in registry order."""
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")
    return tuple(a for a in ARMS.values() if condition in a.conditions)
