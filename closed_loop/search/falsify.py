"""The falsification loop: an attacker that tries to break the shield (packet P9b).

Interface text: docs/adr/0002-interfaces.md, section 3.11 (``falsify``); design: docs/adr/0001-
architecture.md section 9; amendments: docs/adr/0003-review-amendments.md A5 and A12.

The objective
-------------
One evaluation decodes ``theta`` in [0, 1]^32 into a ``Scenario`` for the regime (``R_in`` or
``R_out``), runs one F1 episode of the arm on plant seed ``seed``, and scores the trace with
``spec.properties.evaluate`` (the minimum robustness of P1 to P6 over the whole episode). The value
the optimisers minimise is ``objective_of``: each property's minimum robustness divided by a scale
in its own units and clipped to [-1, 1], then the smallest of the six. The scales are

====  ===============================================================  ===================
P1    a Boolean margin of 1                                             1
P2    minutes of permit left while someone is inside                    ``permit_max_h`` x 60
P3    minutes of gas-test age to spare while someone is inside          ``retest_q`` x 15
P4    pumps available less one half                                     0.5
P5    hours of true remaining life, -1 on an uncontrolled failure       72 (bin 3's edge)
P6    a Boolean margin of 1                                             1
====  ===============================================================  ===================

The sign is the sign of the raw robustness, so ``objective < 0`` exactly when some property is
violated; the scale only makes a near miss in one property comparable with a near miss in another
(minutes against hours against counts), so the optimiser can follow a gradient towards the closest
one. P6 is in the minimum as 0001 section 9 says ("the minimum over properties"); M-VIOL and the
defect rule of 0003 A5 name P1 to P5, so a find that violates P6 alone is labelled as such
(``Finding.props``) and not hidden.

The plant seed is the search seed. That makes ``f`` a pure function of ``theta`` within one run, so
a find is reproduced by ``(seed, theta)`` and the two optimisers of run ``i`` face the same
landscape (C-SEARCH compares them paired by run index).

Targets (prereg section 4, R-cma and R-rand)
--------------------------------------------
``target_arm()`` is B3 with Careless and Gullible (screening off) behind the shield, perception
V3 and the per-life set, as H-C1 and H-G1. ``sanity_arm()`` is the registry's H-C0, the same
hostile proposer with the shield off, for the check that the objective can be minimised at all.

Attribution (0003 A5)
---------------------
``simulate_episode`` labels every violation event with ``assumption_held`` or the first broken
assumption among ``A_exec``, ``A_phys`` and ``C1`` in the 30 days before it, on the component
concerned (0003 A15: P1, P2, P3 and P6 read no health, so only ``A_exec`` can excuse them).
``classify`` turns the ``(property, label)`` events of one evaluation into a kind:

* ``defect``: a shielded arm, regime ``in``, some violation of P1 to P5 labelled
  ``assumption_held`` (``metrics.is_defect``, the one function that decides it). One is enough.
  Nothing in this module softens it, and nothing in this packet changes the shield, the plant, the
  attribution or a bound to clear one.
* ``contract_failure``: a shielded arm, no defect, some violation labelled with a broken
  assumption. It counts in M-VIOL and goes in WHAT-GETS-THROUGH; it is what the proof does not
  cover.
* ``unexplained_beyond_assumptions``: a shielded arm in regime ``out`` with a violation of P1 to P5
  labelled ``assumption_held``: the scenario left the assumption set and the trace still shows none
  of the three broken, which means a gap the three do not name.
* ``p6_unexplained``: a shielded arm whose only unexplained violations are of P6. Reported, not a
  defect (0003 A15).
* ``unshielded``: the arm has no shield, so a violation is what the shield is for.

Findings
--------
``build_finding`` shrinks the best violating evaluation of a run (``search.shrink``), keeping a
violation of one of the properties it started with, and labels what is left: the dimensions that
survived with their decoded values, which of them lie beyond the ``R_in`` limits, whether the
violation goes away when that one dimension is put back (the ablation of 0001 section 9), and the
attribution of the shrunk scenario's own episode. ``what_gets_through`` groups findings that shrank
to the same minimal scenario into the structured list ``docs/WHAT-GETS-THROUGH.md`` is written from.
Every number is a development number from ``search`` seeds (3000 to 3039).
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from closed_loop.core.timegrid import QUARTER_MIN
from closed_loop.core.types import BIN_EDGES_DAYS
from closed_loop.eval.arms import CARELESS_P_SKIP, Arm, make_bound_rule
from closed_loop.eval.arms import arm as registry_arm
from closed_loop.eval.context import PreparedContext, current_context
from closed_loop.eval.episode import simulate_episode
from closed_loop.eval.metrics import Attribution, is_defect
from closed_loop.plant.notes import NOTE_BANK, bank_index, cached_scan
from closed_loop.plant.params import CRUSHER, PUMPS, PlantParams
from closed_loop.plant.scenario import (
    FAULT_DRIFT,
    FAULT_NONE,
    FAULT_SPIKE,
    FAULT_STUCK,
    REGIMES,
    THETA_DIM,
    ResolvedScenario,
    Scenario,
)
from closed_loop.proposers import Careless, Gullible
from closed_loop.proposers.base import Proposer
from closed_loop.search.cmaes import cmaes_search
from closed_loop.search.random_search import SearchRun, random_search, summarise_run
from closed_loop.search.shrink import COORDINATES, ShrinkResult, nominal_theta, shrink_detailed
from closed_loop.spec.properties import PROPERTY_IDS, SpecParams, evaluate

__all__ = [
    "ALGOS",
    "KINDS",
    "Dimension",
    "Evaluation",
    "FalsifyResult",
    "Finding",
    "Objective",
    "StopSearch",
    "build_finding",
    "classify",
    "collect_findings",
    "dimensions_of",
    "evaluate_theta",
    "falsify",
    "falsify_detailed",
    "first_events_of",
    "make_target_proposer",
    "merge_documents",
    "normalise",
    "objective_of",
    "property_scales",
    "r_in_exceedances",
    "sanity_arm",
    "target_arm",
    "violated_of",
    "what_gets_through",
]

ALGOS: Final[tuple[str, ...]] = ("cma", "random")  # "cmaes" is accepted as another name for "cma"
CRITICAL_RUL_HOURS: Final[float] = BIN_EDGES_DAYS[-1] * 24.0  # P5: the edge of the critical bin
KINDS: Final[tuple[str, ...]] = (
    "none",
    "unshielded",
    "defect",
    "contract_failure",
    "unexplained_beyond_assumptions",
    "p6_unexplained",
)
FAULT_NAMES: Final[Mapping[int, str]] = {
    FAULT_NONE: "no fault",
    FAULT_STUCK: "stuck sensor",
    FAULT_SPIKE: "spike",
    FAULT_DRIFT: "unmasked gain drift",
}

# ----------------------------------------------------------------------------- the objective


def property_scales(spec: SpecParams) -> dict[str, float]:
    """The scale of each property's robustness, in the units the property's margin is in."""
    return {
        "P1": 1.0,
        "P2": spec.permit_max_h * 60.0,
        "P3": float(spec.retest_q * QUARTER_MIN),
        "P4": 0.5,
        "P5": CRITICAL_RUL_HOURS,
        "P6": 1.0,
    }


def normalise(raw: Mapping[str, float], spec: SpecParams) -> dict[str, float]:
    """Each property's minimum robustness over its scale, clipped to [-1, 1]; sign unchanged."""
    scales = property_scales(spec)
    missing = [p for p in PROPERTY_IDS if p not in raw]
    if missing:
        raise ValueError(f"robustness is missing for {missing}")
    out: dict[str, float] = {}
    for pid in PROPERTY_IDS:
        value = float(raw[pid])
        if math.isnan(value):
            raise ValueError(f"robustness of {pid} is NaN")
        scaled = value / scales[pid]
        if value < 0.0 and scaled >= 0.0:  # underflow must not turn a violation into a zero
            scaled = -math.ulp(0.0)
        out[pid] = max(-1.0, min(1.0, scaled))
    return out


def objective_of(raw: Mapping[str, float], spec: SpecParams) -> float:
    """The value the optimisers minimise: the smallest normalised robustness of P1 to P6."""
    return min(normalise(raw, spec).values())


# ----------------------------------------------------------------------------- one evaluation


@dataclass(frozen=True, eq=False)
class Evaluation:
    """One evaluation of the objective, with what the attribution of 0003 A5 says about it.

    ``robustness`` is the raw minimum per property (``spec.evaluate``); ``violated`` lists the
    properties below zero; ``first_events`` holds the earliest violation event of each violated
    property, labelled; ``label_counts`` counts every event by label and ``event_counts`` by
    (property, label), which is what the defect rule reads; ``index`` is the 1-based
    number of the evaluation within its run (0 outside a run). ``wall_s`` and ``cpu_s`` are the
    time the evaluation took, wall clock and this process's CPU time (the shared machine makes
    the first the larger and the less stable).
    """

    theta: NDArray[np.float64]
    regime: str
    seed: int
    arm: str
    objective: float
    robustness: Mapping[str, float]
    violated: tuple[str, ...]
    first_events: tuple[Attribution, ...]
    label_counts: Mapping[str, int]
    wall_s: float
    cpu_s: float = 0.0
    index: int = 0
    event_counts: Mapping[tuple[str, str], int] = field(default_factory=dict)

    @property
    def events(self) -> tuple[tuple[str, str], ...]:
        """The (property, label) pairs that occurred, for ``classify``."""
        return tuple(sorted(self.event_counts))


Runner = Callable[[Arm, NDArray[np.float64], str, int, PreparedContext | None], Evaluation]


def violated_of(raw: Mapping[str, float]) -> tuple[str, ...]:
    """The properties whose minimum robustness is below zero, in property order."""
    return tuple(p for p in PROPERTY_IDS if raw[p] < 0.0)


def first_events_of(attributions: Sequence[Attribution]) -> tuple[Attribution, ...]:
    """The earliest violation event of each property, in property order.

    ``attributions`` is in time order, as an episode produces it, so the first one seen of a
    property is its earliest.
    """
    first: dict[str, Attribution] = {}
    for a in attributions:
        first.setdefault(a.prop, a)
    return tuple(first[p] for p in PROPERTY_IDS if p in first)


def evaluate_theta(
    arm: Arm,
    theta: NDArray[np.float64],
    regime: str,
    seed: int,
    context: PreparedContext | None = None,
) -> Evaluation:
    """Decode ``theta`` for ``regime``, run one F1 episode of ``arm`` on ``seed`` and score it."""
    started, cpu_started = time.perf_counter(), time.process_time()
    ctx = context if context is not None else current_context()
    scenario = Scenario.decode(theta, regime)
    record = simulate_episode(arm, seed, scenario, context=ctx)
    raw = evaluate(record.trace, ctx.spec_clause, CRUSHER, PUMPS)
    counts = Counter(a.label for a in record.result.attributions)
    pairs = Counter((a.prop, a.label) for a in record.result.attributions)
    return Evaluation(
        theta=np.array(theta, dtype=np.float64),
        regime=regime,
        seed=seed,
        arm=arm.name,
        objective=objective_of(raw, ctx.spec_clause),
        robustness=dict(raw),
        violated=violated_of(raw),
        first_events=first_events_of(record.result.attributions),
        label_counts=dict(counts),
        wall_s=time.perf_counter() - started,
        cpu_s=time.process_time() - cpu_started,
        event_counts=dict(pairs),
    )


def classify(shielded: bool, regime: str, labels: Iterable[str | tuple[str, str]]) -> str:
    """The kind of a violation set from its A5 events (see the module docstring).

    ``labels`` are ``(property, label)`` pairs, ``Evaluation.events``. The defect rule is
    ``metrics.is_defect``, the one function that decides it (0003 A15): an ``assumption_held``
    violation of P1 to P5. A bare label string carries no property, so it is read as a violation
    of P1 to P5, the stricter reading (the runner's stage passes bare labels today); only a pair
    can say that an event was P6, which is attributed and reported but is not a defect.
    """
    pairs = [("P1", e) if isinstance(e, str) else e for e in labels]
    if not pairs:
        return "none"
    if not shielded:
        return "unshielded"
    if any(is_defect(prop, label) for prop, label in pairs):
        return "defect" if regime == "in" else "unexplained_beyond_assumptions"
    if any(label != "assumption_held" for _prop, label in pairs):
        return "contract_failure"
    return "p6_unexplained"


class StopSearch(Exception):
    """Raised by an ``Objective`` built with ``stop_on_first`` once it has seen a violation."""


class Objective:
    """``f(theta)`` of one search run; counts its calls and keeps the violating evaluations.

    ``runner`` is the function that does an evaluation (``evaluate_theta``); tests pass a stand-in.
    The optimisers get ``__call__``; both call it exactly once per point they are charged for.
    With ``stop_on_first`` the call that finds the first violation raises ``StopSearch`` after
    recording it: a sanity check that only asks "is there a violation within the budget" need not
    pay for the rest of the budget. C-SEARCH runs do not use it (they report the best value at the
    budget).
    """

    def __init__(
        self,
        arm: Arm,
        regime: str,
        seed: int,
        context: PreparedContext | None = None,
        *,
        runner: Runner | None = None,
        stop_on_first: bool = False,
    ) -> None:
        if regime not in REGIMES:
            raise ValueError(f"regime must be one of {REGIMES}, got {regime!r}")
        self.arm = arm
        self.regime = regime
        self.seed = seed
        self.context = context
        self.runner: Runner = runner if runner is not None else evaluate_theta
        self.stop_on_first = stop_on_first
        self.calls = 0
        self.seconds = 0.0
        self.cpu_seconds = 0.0
        self.violations: list[Evaluation] = []
        self.points: list[NDArray[np.float64]] = []
        self.values: list[float] = []

    def evaluate(self, theta: NDArray[np.float64]) -> Evaluation:
        self.calls += 1
        ev = self.runner(self.arm, theta, self.regime, self.seed, self.context)
        ev = dataclasses.replace(ev, index=self.calls)
        self.seconds += ev.wall_s
        self.cpu_seconds += ev.cpu_s
        self.points.append(np.array(theta, dtype=np.float64))
        self.values.append(ev.objective)
        if ev.objective < 0.0:
            self.violations.append(ev)
        return ev

    def __call__(self, theta: NDArray[np.float64]) -> float:
        ev = self.evaluate(theta)
        if self.stop_on_first and ev.objective < 0.0:
            raise StopSearch
        return ev.objective


# ----------------------------------------------------------------------------- arms


def make_target_proposer() -> Proposer:
    """B3 behind a gullible note reader behind a careless fitter: the worst colleague of 0001 8."""
    return Careless(Gullible(make_bound_rule("scm"), False, cached_scan), CARELESS_P_SKIP)


def target_arm() -> Arm:
    """B3 + Careless + Gullible (screening off) with the shield: R-cma and R-rand's target."""
    return Arm(
        name="R-target",
        proposer=make_target_proposer,
        shield=True,
        monitor=False,
        perception="V3",
        conditions=("nominal",),
        tuned=("k_b3",),
    )


def sanity_arm() -> Arm:
    """Careless around B3 with the shield off (the registry's H-C0): a violation is expected."""
    return registry_arm("H-C0")


# ----------------------------------------------------------------------------- the loop


@dataclass(frozen=True, eq=False)
class FalsifyResult:
    """One search run: the frozen ``SearchRun`` plus what the loop saw that it does not hold."""

    run: SearchRun
    arm: str
    shielded: bool
    regime: str  # "in" or "out"
    violations: tuple[Evaluation, ...]  # in evaluation order
    seconds: float  # evaluation wall time only, not the optimiser's own
    cpu_seconds: float = 0.0  # the same in this process's CPU time

    @property
    def evals(self) -> int:
        return self.run.evals

    @property
    def mean_eval_s(self) -> float:
        return self.seconds / max(self.run.evals, 1)

    @property
    def mean_eval_cpu_s(self) -> float:
        return self.cpu_seconds / max(self.run.evals, 1)


def _optimiser(algo: str) -> str:
    if algo in ("cma", "cmaes"):
        return "cma"
    if algo == "random":
        return "random"
    raise ValueError(f"algo must be 'cma' or 'random', got {algo!r}")


def falsify_detailed(
    arm_factory: Callable[[], Arm],
    regime: str,
    algo: str,
    budget: int,
    seed: int,
    *,
    context: PreparedContext | None = None,
    runner: Runner | None = None,
    stop_on_first: bool = False,
) -> FalsifyResult:
    """Search the 32-dimensional scenario cube for a violation, within ``budget`` evaluations.

    Both optimisers get the same objective, the same box and the same budget; the points of the
    optimisers come from keyed streams of ``seed``, and the plant runs on ``seed`` itself.
    ``stop_on_first`` ends the run at the first violation (``run.evals`` is then the number of
    evaluations used, below ``budget``).
    """
    which = _optimiser(algo)
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {REGIMES}, got {regime!r}")
    arm = arm_factory()
    objective = Objective(arm, regime, seed, context, runner=runner, stop_on_first=stop_on_first)
    label = f"R_{regime}"
    try:
        if which == "cma":
            run = cmaes_search(objective, THETA_DIM, budget, seed, regime=label)
        else:
            run = random_search(objective, THETA_DIM, budget, seed, regime=label)
    except StopSearch:
        run = summarise_run(
            "cmaes" if which == "cma" else "random",
            label,
            seed,
            np.array(objective.values, dtype=np.float64),
            np.array(objective.points, dtype=np.float64),
        )
    return FalsifyResult(
        run=run,
        arm=arm.name,
        shielded=arm.shield,
        regime=regime,
        violations=tuple(objective.violations),
        seconds=objective.seconds,
        cpu_seconds=objective.cpu_seconds,
    )


def falsify(
    arm_factory: Callable[[], Arm],
    regime: str,
    algo: str,
    budget: int,
    seed: int,
    *,
    context: PreparedContext | None = None,
) -> SearchRun:
    """0002 section 3.11. ``algo`` is ``"cma"`` or ``"random"``; the run is labelled ``R_<regime>``.

    ``first_violation_eval`` is the 1-based evaluation of the first violation of P1 to P6, or None.
    Use ``falsify_detailed`` for the violating evaluations and their attribution.
    """
    return falsify_detailed(arm_factory, regime, algo, budget, seed, context=context).run


# ----------------------------------------------------------------------------- R_in limits


@dataclass(frozen=True)
class _Envelope:
    """What regime ``in`` can decode to, read off the corners of the cube rather than copied."""

    rate: tuple[float, float]
    delay: tuple[float, float]
    lead: tuple[float, float]
    load_shift: float
    faults: frozenset[int]


def _envelope(seed: int, params: PlantParams) -> _Envelope:
    lo = Scenario.decode(np.zeros(THETA_DIM), "in").resolve(seed, params)
    hi = Scenario.decode(np.ones(THETA_DIM), "in").resolve(seed, params)
    types = {
        Scenario.decode(np.full(THETA_DIM, u), "in").resolve(seed, params).fault_type[0]
        for u in (0.0, 0.3, 0.6, 1.0)
    }
    return _Envelope(
        rate=(min(lo.rate_mult), max(hi.rate_mult)),
        delay=(lo.approver_delay_mult, hi.approver_delay_mult),
        lead=(lo.spare_lead_mult, hi.spare_lead_mult),
        load_shift=hi.load_shift_frac,
        faults=frozenset(types),
    )


def _exceeds(i: int, r: ResolvedScenario, env: _Envelope) -> str | None:
    """The R_in limit that coordinate ``i`` of a decoded scenario exceeds, or None."""
    eps = 1e-9
    if i < 20:
        a, k = divmod(i, 5)
        if k == 0 and not env.rate[0] - eps <= r.rate_mult[a] <= env.rate[1] + eps:
            return f"wear-rate multiplier outside [{env.rate[0]:g}, {env.rate[1]:g}]"
        if k == 3 and r.fault_type[a] not in env.faults:
            return "a sensor fault the data-quality gate does not mask"
        return None
    if i in (20, 21, 22):
        if r.forced_refusal_p > 0.0:
            return "forced commands (STOP, EXIT) refused: A_exec broken"
        return None
    if i == 27 and not env.delay[0] - eps <= r.approver_delay_mult <= env.delay[1] + eps:
        return f"approver delay multiplier outside [{env.delay[0]:g}, {env.delay[1]:g}]"
    if i == 29 and not env.lead[0] - eps <= r.spare_lead_mult <= env.lead[1] + eps:
        return f"spare lead-time multiplier outside [{env.lead[0]:g}, {env.lead[1]:g}]"
    if i == 30 and r.load_shift_frac > env.load_shift + eps:
        return f"load shift above the R_in cap of +{env.load_shift:g}: a health bin can be skipped"
    return None


def r_in_exceedances(
    theta: NDArray[np.float64], regime: str, seed: int = 0, params: PlantParams | None = None
) -> tuple[str, ...]:
    """The R_in limits that the decoded scenario exceeds, each named once (empty for ``R_in``)."""
    params = params if params is not None else PlantParams()
    resolved = Scenario.decode(theta, regime).resolve(seed, params)
    env = _envelope(seed, params)
    found = [_exceeds(i, resolved, env) for i in range(THETA_DIM)]
    return tuple(dict.fromkeys(s for s in found if s is not None))


# ----------------------------------------------------------------------------- findings


@dataclass(frozen=True)
class Dimension:
    """One coordinate of a shrunk scenario that is still away from nominal.

    ``necessary`` is the ablation of 0001 section 9: the scenario was run again with this one
    dimension put back at nominal and the others as they are; True if the violation went away,
    False if it stayed, None if it was not asked (the shrink ran out of calls first).
    """

    index: int
    name: str
    theta: float
    nominal: float
    decoded: str
    beyond_r_in: str | None
    necessary: bool | None = None


def _decoded(i: int, r: ResolvedScenario) -> str:
    if i < 20:
        a, k = divmod(i, 5)
        return (
            f"x{r.rate_mult[a]:.3f}",
            f"{r.onset_shift_days[a]:+.2f} d",
            f"{r.residual_shift_days[a]:+.2f} d",
            FAULT_NAMES[r.fault_type[a]],
            f"day {r.fault_day[a]:.2f}",
        )[k]
    if i < 23:
        return f"holder present with p = {r.holder_p[i - 20]:.3f}"
    if i < 25:
        return f"quarter {r.gas_adv_q[i - 23]}"
    if i == 25:
        return f"note {NOTE_BANK[bank_index(float(r.note_u or 0.0))].note_id}"
    if i == 26:
        return f"quarter {r.note_q}"
    values = {
        27: f"x{r.approver_delay_mult:.3f}",
        28: f"p = {r.approver_catch_p:.3f}",
        29: f"x{r.spare_lead_mult:.3f}",
        30: f"+{r.load_shift_frac:.3f} of the load",
        31: f"day {r.load_shift_day:.2f}",
    }
    return values[i]


def dimensions_of(
    theta: NDArray[np.float64],
    regime: str,
    seed: int = 0,
    params: PlantParams | None = None,
    tol: float = 0.02,
) -> tuple[Dimension, ...]:
    """The coordinates of ``theta`` that differ from nominal by more than ``tol``, decoded."""
    params = params if params is not None else PlantParams()
    resolved = Scenario.decode(theta, regime).resolve(seed, params)
    env = _envelope(seed, params)
    nom = nominal_theta(regime)
    x = np.clip(np.asarray(theta, dtype=np.float64), 0.0, 1.0)
    return tuple(
        Dimension(
            index=int(i),
            name=COORDINATES[i],
            theta=float(x[i]),
            nominal=float(nom[i]),
            decoded=_decoded(int(i), resolved),
            beyond_r_in=_exceeds(int(i), resolved, env),
        )
        for i in np.flatnonzero(np.abs(x - nom) > tol)
    )


@dataclass(frozen=True, eq=False)
class Finding:
    """A violation found by one run, and the smallest scenario this packet could reduce it to.

    ``theta`` and ``shrunk_theta`` reproduce the violation with ``seed`` (the plant seed, which
    is also the search seed). ``label`` is the A5 label of the earliest violation event of the
    shrunk scenario's own episode: the assumption it broke, or ``assumption_held``.
    """

    arm: str
    regime: str
    algo: str
    seed: int
    shielded: bool
    eval_index: int
    objective: float
    theta: tuple[float, ...]
    props: tuple[str, ...]
    kind: str
    labels: Mapping[str, int]
    shrunk_theta: tuple[float, ...]
    shrunk_objective: float
    shrunk_props: tuple[str, ...]
    shrunk_kind: str
    shrunk_labels: Mapping[str, int]
    shrunk_events: tuple[Attribution, ...]
    label: str
    dimensions: tuple[Dimension, ...]
    beyond_r_in: tuple[str, ...]
    non_nominal_before: int
    non_nominal_after: int
    shrink_calls: int
    shrink_truncated: bool

    @property
    def signature(self) -> tuple[Any, ...]:
        """What makes two finds the same minimal scenario: regime, kind, dimensions, label."""
        return (
            self.regime,
            self.kind,
            self.shrunk_kind,
            self.label,
            self.shrunk_props,
            tuple(d.index for d in self.dimensions),
        )


def _with_coordinate(
    theta: NDArray[np.float64], index: int, source: NDArray[np.float64]
) -> NDArray[np.float64]:
    out = np.array(theta, dtype=np.float64)
    out[index] = source[index]
    return out


def build_finding(
    result: FalsifyResult,
    evaluation: Evaluation,
    *,
    arm: Arm,
    context: PreparedContext | None = None,
    tol: float = 0.02,
    max_calls: int | None = None,
    runner: Runner | None = None,
) -> Finding:
    """Shrink ``evaluation`` (a violating evaluation of ``result``'s run) and label what is left.

    The shrink keeps a violation of at least one property the original violated, so it cannot
    wander to a different failure; the shrunk scenario's attribution is read from its own episode.
    When the shrink finished (it was not cut short by ``max_calls``) each surviving dimension is
    then ablated once, one extra episode per dimension, and marked ``necessary`` or not.
    """
    if evaluation.objective >= 0.0:
        raise ValueError("the evaluation does not violate, so there is nothing to shrink")
    regime, seed = result.regime, evaluation.seed
    wanted = set(evaluation.violated)
    cache: dict[bytes, Evaluation] = {}

    def run(theta: NDArray[np.float64]) -> Evaluation:
        key = np.ascontiguousarray(theta, dtype=np.float64).tobytes()
        if key not in cache:
            cache[key] = (runner if runner is not None else evaluate_theta)(
                arm, theta, regime, seed, context
            )
        return cache[key]

    def still_violates(theta: NDArray[np.float64]) -> bool:
        ev = run(theta)
        return ev.objective < 0.0 and bool(wanted & set(ev.violated))

    shrunk: ShrinkResult = shrink_detailed(
        evaluation.theta, still_violates, tol, regime=regime, max_calls=max_calls
    )
    final = run(shrunk.theta)
    params = context.params if context is not None else PlantParams()
    events = tuple(sorted(final.first_events, key=lambda a: (a.row, a.prop)))
    dimensions = dimensions_of(shrunk.theta, regime, seed, params, tol)
    if not shrunk.truncated:
        nominal = nominal_theta(regime)
        dimensions = tuple(
            dataclasses.replace(
                d, necessary=not still_violates(_with_coordinate(shrunk.theta, d.index, nominal))
            )
            for d in dimensions
        )
    return Finding(
        arm=result.arm,
        regime=regime,
        algo=result.run.algo,
        seed=seed,
        shielded=result.shielded,
        eval_index=evaluation.index,
        objective=evaluation.objective,
        theta=tuple(float(v) for v in evaluation.theta),
        props=evaluation.violated,
        kind=classify(result.shielded, regime, evaluation.events),
        labels=dict(evaluation.label_counts),
        shrunk_theta=tuple(float(v) for v in shrunk.theta),
        shrunk_objective=final.objective,
        shrunk_props=final.violated,
        shrunk_kind=classify(result.shielded, regime, final.events),
        shrunk_labels=dict(final.label_counts),
        shrunk_events=events,
        label=events[0].label if events else "none",
        dimensions=dimensions,
        beyond_r_in=r_in_exceedances(shrunk.theta, regime, seed, params),
        non_nominal_before=shrunk.non_nominal_before,
        non_nominal_after=shrunk.non_nominal_after,
        shrink_calls=shrunk.calls,
        shrink_truncated=shrunk.truncated,
    )


def collect_findings(
    results: Sequence[FalsifyResult],
    *,
    arm_factory: Callable[[], Arm],
    per_run: int = 1,
    context: PreparedContext | None = None,
    tol: float = 0.02,
    max_calls: int | None = None,
    runner: Runner | None = None,
) -> list[Finding]:
    """Shrink the ``per_run`` lowest-objective violations of each run that has any."""
    arm = arm_factory()
    out: list[Finding] = []
    for res in results:
        best = sorted(res.violations, key=lambda e: (e.objective, e.index))[:per_run]
        out.extend(
            build_finding(
                res, ev, arm=arm, context=context, tol=tol, max_calls=max_calls, runner=runner
            )
            for ev in best
        )
    return out


# ----------------------------------------------------------------------------- the structured list


def _dimension_json(d: Dimension) -> dict[str, Any]:
    return dataclasses.asdict(d)


def _event_json(a: Attribution) -> dict[str, Any]:
    return {
        "property": a.prop,
        "row": a.row,
        "label": a.label,
        "broken_row": a.broken_row,
        "asset": a.asset,
    }


def _entry(f: Finding) -> dict[str, Any]:
    """One finding as an entry of the structured list (before it is grouped with its twins)."""
    return {
        "signature": [
            f"R_{f.regime}",
            f.shrunk_kind,
            f.label,
            list(f.shrunk_props),
            [d.index for d in f.dimensions],
        ],
        "arm": f.arm,
        "shielded": f.shielded,
        "regime": f"R_{f.regime}",
        "kind": f.shrunk_kind,
        "broken_assumption": f.label,
        "properties": list(f.shrunk_props),
        "dimensions": [_dimension_json(d) for d in f.dimensions],
        "beyond_r_in": list(f.beyond_r_in),
        "first_events": [_event_json(a) for a in f.shrunk_events],
        "shrunk_theta": list(f.shrunk_theta),
        "shrunk_objective": f.shrunk_objective,
        "found_theta": list(f.theta),
        "plant_seed": f.seed,
        "non_nominal_before": f.non_nominal_before,
        "non_nominal_after": f.non_nominal_after,
        "shrink_calls": f.shrink_calls,
        "shrink_truncated": f.shrink_truncated,
        "found_by": [{"algo": f.algo, "seed": f.seed, "evaluation": f.eval_index}],
    }


def _run_json(r: FalsifyResult) -> dict[str, Any]:
    return {
        "arm": r.arm,
        "regime": f"R_{r.regime}",
        "algo": r.run.algo,
        "seed": r.run.seed,
        "evals": r.run.evals,
        "found": r.run.first_violation_eval is not None,
        "first_violation_eval": r.run.first_violation_eval,
        "best_objective": r.run.best_f,
        "violating_evaluations": len(r.violations),
        "wall_s_per_eval": r.mean_eval_s,
        "cpu_s_per_eval": r.mean_eval_cpu_s,
    }


def _assemble(
    entries: Sequence[Mapping[str, Any]], runs: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Group entries with the same signature into one, numbered in a stable order."""
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for e in entries:
        groups.setdefault(json.dumps(e["signature"]), []).append(e)
    out: list[dict[str, Any]] = []
    for n, members in enumerate(
        sorted(groups.values(), key=lambda g: json.dumps(g[0]["signature"]))
    ):
        head = min(
            members,
            key=lambda m: (len(m["dimensions"]), m["shrunk_objective"], m["plant_seed"]),
        )
        merged = dict(head)
        merged["id"] = f"WGT-{n + 1:02d}"
        merged["found_by"] = [fb for m in members for fb in m["found_by"]]
        out.append(merged)
    return {
        "synthetic": True,
        "fidelity": "F1",
        "seed_role": "search",
        "note": "development numbers from search seeds; not a held-out result",
        "scales": {"P5_hours": CRITICAL_RUL_HOURS},
        "entries": out,
        "runs": [dict(r) for r in runs],
    }


def what_gets_through(
    findings: Sequence[Finding], results: Sequence[FalsifyResult] = ()
) -> dict[str, Any]:
    """The structured list ``docs/WHAT-GETS-THROUGH.md`` is generated from (P13).

    One entry per distinct minimal scenario (``Finding.signature``: regime, kind, label,
    properties and the surviving dimensions), with every run that reached it. Only the shielded
    arm's finds belong in that document; an unshielded find is listed too, with
    ``shielded: false``, so that a reader can see what the shield is for. ``runs`` summarises every
    run handed in, found or not, so a count of "found in k of n" is never a count of the lucky.
    """
    return _assemble([_entry(f) for f in findings], [_run_json(r) for r in results])


def merge_documents(docs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Join documents made by ``what_gets_through`` (one per piece of a long run) into one."""
    entries = [e for d in docs for e in d["entries"]]
    runs = [r for d in docs for r in d["runs"]]
    return _assemble(entries, runs)
