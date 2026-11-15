"""Mechanised checks of the integration mismatches and sibling defects, and the F1-F2 estimator
mismatch (packet P11c). Everything is SYNTHETIC.

Each ``check_*`` function runs the real sibling code (never a stand-in) and returns a ``Check``: its
id, whether the mismatch or defect is observable in the installed siblings today, and one line of
what was observed. ``docs/FINDINGS.md`` lists the same ids; ``tests/f2/test_mismatch.py`` asserts
that every check holds. A check that stops holding means a sibling has changed, and the finding must
be re-read before anyone trusts the table.

Ids: ``M1`` to ``M21`` are the mismatches found when the three repositories were wired together,
``S1`` to ``S12`` are defects found in the siblings on the way. None is fixed here: the siblings are
read and run, never modified.

``estimator_mismatch`` measures the other kind of difference between fidelities: how far the
streaming estimator of the fast simulator (F1) and asset-health's pipeline (F2) are from the true
remaining life on the same lives.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib.metadata
import inspect
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from closed_loop.core.results import ResultRecord
from closed_loop.core.types import Fidelity
from closed_loop.f2._siblings import load
from closed_loop.f2.agent import (
    build_orchestrator,
    new_mock_cmms,
    new_registry,
    register_approver,
    run_id_for,
)
from closed_loop.f2.alert_source import (
    FEATURE_SIGNAL,
    HYPOTHESIS_MODE,
    AssetHealthAlertSource,
    RunCache,
    agent_assets,
    aware,
)
from closed_loop.f2.cmms import PlantCMMS, inventory_of
from closed_loop.f2.human import SimHumanApprover
from closed_loop.f2.render import (
    ASSET_TYPE,
    LifeLatent,
    RenderedTelemetry,
    asset_health_id,
    render_life,
    sim_time,
)
from closed_loop.f2.splice import run_segments
from closed_loop.perception.estimator import STATUS_ESTIMATED, TheilSenEstimator
from closed_loop.plant.observe import simulate_lives
from closed_loop.plant.params import ASSET_CLASS, ASSET_IDS, PARTS, PlantParams
from closed_loop.plant.scenario import Scenario

__all__ = [
    "CHECKS",
    "Check",
    "LifeMismatch",
    "dev_lives",
    "estimator_mismatch",
    "mismatch_record",
    "run_check",
    "run_checks",
]

SCRIPT = "closed_loop.f2.mismatch"
_PUMP = "PU-201"


@dataclass(frozen=True)
class Check:
    """The outcome of one mechanised check."""

    id: str
    holds: bool
    evidence: str


CHECKS: dict[str, Callable[[], Check]] = {}


def _register(check_id: str) -> Callable[[Callable[[], Check]], Callable[[], Check]]:
    def wrap(fn: Callable[[], Check]) -> Callable[[], Check]:
        CHECKS[check_id] = fn
        return fn

    return wrap


def run_check(check_id: str) -> Check:
    """Run one check by id (``"M12"``, ``"S7"``, ...)."""
    try:
        fn = CHECKS[check_id]
    except KeyError:
        raise KeyError(f"no check {check_id!r}; known: {sorted(CHECKS, key=_order)}") from None
    return fn()


def run_checks(ids: Sequence[str] | None = None) -> list[Check]:
    return [run_check(i) for i in (ids if ids is not None else sorted(CHECKS, key=_order))]


def _order(check_id: str) -> tuple[str, int]:
    return check_id[0], int(check_id[1:])


# ----------------------------------------------------------------------------- shared worlds


@dataclass(frozen=True)
class _World:
    now: list[datetime]
    stock: dict[str, int]
    registry: Any
    keys: dict[str, str]
    source: Any
    mock: Any
    cmms: Any
    orch: Any
    queued: list[tuple[str, Any]]

    def clock(self) -> datetime:
        return aware(self.now[0])

    def advance(self, **delta: float) -> None:
        self.now[0] = self.now[0] + timedelta(**delta)

    def human(self, approver_id: str, rng: np.random.Generator, params: PlantParams) -> Any:
        return SimHumanApprover(
            approver_id, self.keys[approver_id], self.clock, params, rng, roles=("PLANNER",)
        )


def _world(source_factory: Callable[[Callable[[], datetime]], Any], now: datetime) -> _World:
    clock_now = [now]

    def clock() -> datetime:
        return aware(clock_now[0])

    registry = new_registry()
    keys = {"planner1": register_approver(registry, "planner1", "Planner One", ["PLANNER"])}
    stock = {p: s.stock for p, s in PARTS.items()}
    queued: list[tuple[str, Any]] = []
    source = source_factory(clock)
    mock = new_mock_cmms(registry, clock)
    cmms = PlantCMMS(
        mock, inventory_of(lambda: stock), lambda draft, wo: queued.append((wo, draft))
    )
    orch = build_orchestrator(source, cmms, registry, clock)
    return _World(clock_now, stock, registry, keys, source, mock, cmms, orch, queued)


@dataclass(frozen=True)
class _Reference:
    params: PlantParams
    lives: tuple[RenderedTelemetry, ...]
    cache: RunCache  # life 1 of PU-201 alone: an impeller-wear alert is open by day 39.5


@cache
def _reference(seed: int = 3) -> _Reference:
    """Pump PU-201: impeller wear (onset 24, failure 52), replaced at day 40, then a bearing
    life."""
    params = PlantParams()
    first = LifeLatent(_PUMP, 0, "IMPELLER_WEAR", 24.0, 52.0, 2.0, 48.0, 0.0, 40.0)
    second = LifeLatent(_PUMP, 1, "BEARING_DEFECT", 65.25, 95.25, 3.0, 60.0, 40.25, None, True)
    lives = (render_life(first, params, seed, 0), render_life(second, params, seed, 1))
    cache_ = RunCache()
    cache_.set_history(_PUMP, [lives[0]])
    return _Reference(params, lives, cache_)


def _pump_world() -> _World:
    ref = _reference()
    return _world(
        lambda clock: AssetHealthAlertSource(ref.cache, clock, {_PUMP: asset_health_id(_PUMP)}),
        sim_time(39.5),
    )


def _fixed(source: Any) -> Callable[[Callable[[], datetime]], Any]:
    return lambda _clock: source


def _fixture_world(today: date | None = None) -> _World:
    synthetic = load("maintenance_agent.adapters.synthetic")
    return _world(
        lambda clock: synthetic.SyntheticAlertSource.from_fixtures(today=today), sim_time(39.5)
    )


def _ok(check_id: str, holds: bool, evidence: str) -> Check:
    return Check(check_id, bool(holds), evidence)


def _package_sources(package: str) -> dict[str, ast.Module]:
    """The parsed modules of an installed sibling package, by dotted name."""
    mod = load(package)
    root = Path(next(iter(mod.__path__)))
    out: dict[str, ast.Module] = {}
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).with_suffix("")
        name = ".".join((package, *rel.parts)).removesuffix(".__init__")
        out[name] = ast.parse(path.read_text(encoding="utf-8"))
    return out


# ----------------------------------------------------------------------------- M1 to M21


@_register("M1")
def check_m1() -> Check:
    fleet = load("asset_health.simulate.fleet")
    synthetic = load("maintenance_agent.adapters.synthetic")
    models = load("maintenance_agent.models")
    ah_ids = sorted(asset_health_id(a) for a in ASSET_IDS)
    data = synthetic.load_fixture("alerts.json")
    agent_ids = sorted({models.Alert.model_validate(a).asset.asset_id for a in data["alerts"]})
    prefixes = sorted(str(p) for p in fleet.ID_PREFIX.values())
    shared = set(ah_ids) & set(agent_ids)
    prefixed = [a for a in agent_ids if a.startswith(tuple(prefixes))]
    return _ok(
        "M1",
        not shared and not prefixed and bool(ah_ids) and bool(agent_ids),
        f"asset-health ids {ah_ids} (prefixes {prefixes}); agent ids {agent_ids}; "
        f"shared {sorted(shared)}",
    )


@_register("M2")
def check_m2() -> Check:
    ah = load("asset_health.models")
    requirements = load("maintenance_agent.requirements")
    types = sorted(t.value for t in ah.AssetType)
    classes = sorted(requirements.CLASS_PERMITS)
    pairs = sorted({(ASSET_TYPE[a], ASSET_CLASS[a]) for a in ASSET_IDS})
    same_name = [t for t in types if t.upper() in classes]
    mapped_types = {t for t, _ in pairs}
    mapped_classes = {c for _, c in pairs}
    return _ok(
        "M2",
        len(types) == 5
        and len(classes) == 6
        and len(pairs) == 3
        and len(same_name) == 1
        and not set(types) <= mapped_types,
        f"{len(types)} asset-health types, {len(classes)} agent classes, {len(pairs)} pairs "
        f"matched by judgement {pairs}, exact name matches {same_name}; unmatched types "
        f"{sorted(set(types) - mapped_types)}, unmatched classes "
        f"{sorted(set(classes) - mapped_classes)}",
    )


@_register("M3")
def check_m3() -> Check:
    ah = load("asset_health.models")
    ag = load("maintenance_agent.models")
    requirements = load("maintenance_agent.requirements")
    hypotheses = {h.value for h in ah.FaultHypothesis}
    agent_modes = {m.value for m in ag.FailureMode}
    produced = {m for m in HYPOTHESIS_MODE.values() if m}
    never_made = sorted(agent_modes - produced)
    try:
        ag.FailureMode("OVERHEATING")
        enum_rejects = False
    except ValueError:
        enum_rejects = True
    unknown = requirements.requirement_for("CRUSHER", "NOT_A_MODE", False)
    return _ok(
        "M3",
        "overheating" in hypotheses and enum_rejects and not unknown.known and len(never_made) == 6,
        f"asset-health hypotheses {sorted(hypotheses)}; the agent's mode enum rejects OVERHEATING; "
        f"agent modes asset-health never produces {never_made}; an unknown class/mode pair is "
        f"known={unknown.known}",
    )


@_register("M4")
def check_m4() -> Check:
    ah = load("asset_health.models")
    ag = load("maintenance_agent.models")
    estimate = ah.RULEstimate(
        asset_id="PMP-01",
        as_of=datetime(2026, 11, 1),
        status=ah.RULStatus.NOT_DEGRADING,
    )
    no_interval = estimate.p10_days is None and estimate.p50_days is None
    try:
        ag.RULInterval(as_of=date(2026, 11, 1), lower_days=None, median_days=None, upper_days=None)
        rejected = False
    except ValueError:
        rejected = True
    return _ok(
        "M4",
        no_interval and rejected,
        f"a {estimate.status.value} estimate carries p10={estimate.p10_days}, "
        f"p50={estimate.p50_days}, p90={estimate.p90_days}; the agent's RULInterval refuses "
        f"missing bounds: {rejected}",
    )


@_register("M5")
def check_m5() -> Check:
    config = load("asset_health.config")
    ag = load("maintenance_agent.models")
    coverage = float(config.RULConfig().target_coverage)
    default = float(ag.RULInterval.model_fields["confidence"].default)
    quantiles = tuple(config.RULConfig().quantiles)
    readers: list[str] = []
    for name, tree in _package_sources("maintenance_agent").items():
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "confidence":
                text = ast.unparse(node).lower()
                if "rul" in text or "interval" in text:
                    readers.append(f"{name}:{node.lineno}")
    return _ok(
        "M5",
        coverage == 0.8 and default == 0.8 and quantiles == (0.1, 0.5, 0.9) and not readers,
        f"asset-health interval quantiles {quantiles} (an {coverage:.0%} interval); the agent's "
        f"RULInterval.confidence defaults to {default}; reads of it outside its definition: "
        f"{readers or 'none'}",
    )


@_register("M6")
def check_m6() -> Check:
    ag = load("maintenance_agent.models")
    fields = sorted(ag.Alert.model_fields)
    names = [f for f in fields if "hypothes" in f or f in ("mode", "failure_mode")]
    world = _fixture_world()
    alert_id = world.source.list_alert_ids()[0]
    base = world.source.get_alert(alert_id)
    other = base.model_copy(
        update={"summary": "a different summary naming bearing wear", "anomaly_score": 0.123}
    )
    changed = world.source.with_alert(other)
    hashes = []
    for source in (world.source, changed):
        w = _world(_fixed(source), sim_time(39.5))
        w.now[0] = datetime(2026, 10, 12, 8, 0)  # the fixtures' own date
        hashes.append(w.orch.run(alert_id, run_id="m6-run").draft_hash)
    return _ok(
        "M6",
        not names and hashes[0] == hashes[1],
        f"Alert fields {fields}; none names a hypothesis; the same alert with another summary and "
        f"score drafts the same work order (draft hash {hashes[0][:12]} = {hashes[1][:12]})",
    )


@_register("M7")
def check_m7() -> Check:
    ah = load("asset_health.models")
    ag = load("maintenance_agent.models")
    ah_fields = sorted(ah.Evidence.model_fields)
    required = sorted(
        n for n, f in ag.EvidenceItem.model_fields.items() if f.is_required() and n != "comment"
    )
    return _ok(
        "M7",
        "alarm" not in ah_fields and "alarm" in required and "baseline" in required,
        f"asset-health Evidence fields {ah_fields} (no alarm level); the agent's EvidenceItem "
        f"requires {required}",
    )


@_register("M8")
def check_m8() -> Check:
    ref = _reference()
    ag = load("maintenance_agent.models")
    instant = aware(sim_time(39.5))
    analysis = ref.cache.analysis(_PUMP, instant)
    order = analysis.work_orders[0]
    lower = float(order.rul.p10_days)
    agent_date = ag.policy_latest_date(instant.date(), lower)
    asset_health_date = order.latest_safe_date
    grid = {
        x: (ag.policy_latest_date(date(2026, 11, 9), x) - date(2026, 11, 9)).days
        for x in (0.5, 2.0, 10.0)
    }
    return _ok(
        "M8",
        asset_health_date is not None
        and asset_health_date != agent_date
        and grid == {0.5: 0, 2.0: 2, 10.0: 9},
        f"for a lower bound of {lower:.1f} d asset-health's latest safe date is "
        f"{asset_health_date} and the agent's policy date is {agent_date}; the agent's rule gives "
        f"offsets {grid} days for lower bounds 0.5, 2 and 10",
    )


@_register("M9")
def check_m9() -> Check:
    synthetic = load("maintenance_agent.adapters.synthetic")
    fixture_date = synthetic.FIXTURE_DATE
    outcome: dict[int, bool] = {}
    for lag in (3, 4):
        world = _fixture_world()
        world.now[0] = datetime(
            fixture_date.year, fixture_date.month, fixture_date.day, 8
        ) + timedelta(days=lag)
        alert_id = world.source.list_alert_ids()[0]
        state = world.orch.run(alert_id, run_id=f"m9-run-{lag}")
        outcome[lag] = "V10" in state.verification.rule_ids()
    return _ok(
        "M9",
        outcome == {3: False, 4: True},
        f"bundled alerts are dated {fixture_date}; a check 3 days later raises V10: {outcome[3]}, "
        f"4 days later: {outcome[4]}",
    )


@_register("M10")
def check_m10() -> Check:
    synthetic = load("maintenance_agent.adapters.synthetic")
    src = synthetic.SyntheticAlertSource.from_fixtures()
    moved = synthetic.SyntheticAlertSource.from_fixtures(
        today=synthetic.FIXTURE_DATE + timedelta(days=5)
    )
    alert_id = src.list_alert_ids()[0]
    delta = (moved.get_alert(alert_id).rul.as_of - src.get_alert(alert_id).rul.as_of).days
    builders = sorted(
        name
        for name, obj in vars(synthetic).items()
        if inspect.isfunction(obj) and "alert" in name.lower() and not name.startswith("_")
    )
    return _ok(
        "M10",
        delta == 5 and not builders,
        f"redating the bundled alerts by 5 days moves rul.as_of by {delta}; public alert builders "
        f"in the module: {builders or 'none'}",
    )


@_register("M11")
def check_m11() -> Check:
    ag = load("maintenance_agent.models")
    lowered = ag.check_identifier("Demo-Run", "run id")
    try:
        ag.check_identifier("x" * 65, "run id")
        long_ok = True
    except ValueError:
        long_ok = False
    world = _pump_world()
    alert_id = world.source.raise_alert(_PUMP).alert_id
    assert alert_id is not None
    run_id = run_id_for(alert_id)
    world.orch.run(alert_id, run_id=run_id)
    human = world.human("planner1", np.random.default_rng(1), PlantParams())
    state = None
    for _ in range(12):
        state = human.act(world.orch, run_id)
        if state.status.value == "COMPLETED":
            break
        world.advance(hours=1)
    assert state is not None and state.status.value == "COMPLETED"
    world.orch.run(alert_id, run_id="second-run")
    second = None
    human2 = world.human("planner1", np.random.default_rng(5), PlantParams())
    for _ in range(12):
        second = human2.act(world.orch, "second-run")
        if second.status.value != "AWAITING_APPROVAL":
            break
        world.advance(hours=1)
    assert second is not None
    return _ok(
        "M11",
        lowered == "demo-run" and not long_ok and second.escalation == "duplicate_work_order",
        f"run id 'Demo-Run' is stored as {lowered!r}; 65 characters refused; a second work order "
        f"for an alert that already has one ends {second.status.value} "
        f"({second.escalation})",
    )


@_register("M12")
def check_m12() -> Check:
    wiring = load("maintenance_agent.wiring")
    params = sorted(inspect.signature(wiring.build_system).parameters)
    return _ok(
        "M12",
        "cmms" not in params and "cmms_clock" in params,
        f"build_system takes {params}: a clock for its own CMMS, no CMMS",
    )


@_register("M13")
def check_m13() -> Check:
    tree = _package_sources("maintenance_agent")["maintenance_agent.cmms"]
    writes: list[str] = []
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef):
            for node in ast.walk(fn):
                targets = (
                    node.targets
                    if isinstance(node, ast.Assign)
                    else [node.target]
                    if isinstance(node, ast.AugAssign | ast.AnnAssign)
                    else []
                )
                for t in targets:
                    if isinstance(t, ast.Attribute) and t.attr in ("_parts", "stock_qty"):
                        writes.append(fn.name)
    return _ok(
        "M13",
        set(writes) <= {"__init__"} and bool(writes),
        f"MockCMMS writes its parts table or any stock quantity only in {sorted(set(writes))}: a "
        "release leaves stock unchanged",
    )


@_register("M14")
def check_m14() -> Check:
    ports = load("maintenance_agent.ports")
    methods = sorted(n for n in dir(ports.CMMS) if not n.startswith("_"))
    scheduling = [
        n for n in methods if any(w in n for w in ("schedule", "assign", "complete", "status"))
    ]
    return _ok(
        "M14",
        methods
        == [
            "draft_work_order",
            "find_release",
            "find_release_for",
            "get_part",
            "release_work_order",
        ]
        and not scheduling,
        f"the CMMS protocol offers {methods}: nothing schedules, assigns or reports completion",
    )


@_register("M15")
def check_m15() -> Check:
    ah = load("asset_health.models")
    ag = load("maintenance_agent.models")
    ah_fields = sorted(ah.AssetSpec.model_fields)
    parts_like = [f for f in ah_fields if "part" in f or "bom" in f]
    bom = sorted(next(iter(agent_assets().values())).bom.values())
    return _ok(
        "M15",
        not parts_like and "bom" in ag.AssetInfo.model_fields and bool(bom),
        f"asset-health AssetSpec fields {ah_fields} hold no parts; the agent's AssetInfo carries "
        f"a bill of materials (one asset: {bom})",
    )


@_register("M16")
def check_m16() -> Check:
    importers: list[str] = []
    for name, tree in _package_sources("maintenance_agent").items():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[-1] == "gate":
                importers.append(name)
            elif (
                isinstance(node, ast.ImportFrom)
                and node.level
                and any(a.name == "gate" for a in node.names)
            ):
                importers.append(name)
            elif isinstance(node, ast.Import) and any(a.name.endswith(".gate") for a in node.names):
                importers.append(name)
    core = {"orchestrator", "verifier", "cmms", "approval"}
    leaked = sorted(i for i in set(importers) if i.rsplit(".", 1)[-1] in core)
    return _ok(
        "M16",
        not leaked,
        f"modules of the agent package that import the signing half: {sorted(set(importers))}; "
        f"none of the checking core ({sorted(core)}) does, so a simulated human has to live "
        "outside",
    )


@_register("M17")
def check_m17() -> Check:
    requirements = load("maintenance_agent.requirements")
    roles = {
        "crusher liner repair": requirements.required_role_for("CRUSHER", "LINER_WEAR", False),
        "crusher inspection": requirements.required_role_for("CRUSHER", None, True),
        "pump impeller repair": requirements.required_role_for(
            "SLURRY_PUMP", "IMPELLER_WEAR", False
        ),
    }
    return _ok(
        "M17",
        roles
        == {
            "crusher liner repair": "SAFETY_OFFICER",
            "crusher inspection": "PLANNER",
            "pump impeller repair": "PLANNER",
        },
        f"required signing role: {roles}",
    )


@_register("M18")
def check_m18() -> Check:
    synthetic = load("maintenance_agent.adapters.synthetic")
    text = json.dumps(synthetic.load_fixture("procedures.json"), ensure_ascii=False).lower()
    retest = [w for w in ("re-test", "retest", "gas test", "gas-test") if w in text]
    minutes = sorted({s.strip() for s in text.split('"') if "minutes" in s})
    return _ok(
        "M18",
        not retest and any("60 minutes" in m for m in minutes),
        f"gas re-test wording in the agent's procedures: {retest or 'none'}; the only minutes "
        f"text: {[m[:70] for m in minutes]}",
    )


@_register("M19")
def check_m19() -> Check:
    from closed_loop.spec.clauses import spec_from_copilot

    models = load("plant_copilot.models")
    rev3 = spec_from_copilot(date(2026, 10, 1))
    rev2 = spec_from_copilot(date(2026, 1, 1))
    fields = {
        cls.__name__: sorted(f.name for f in dataclasses.fields(cls))
        for cls in (models.SearchHit, models.Citation, models.Claim, models.Answer)
    }
    numeric = [
        f"{c}.{n}"
        for c, names in fields.items()
        for n in names
        if any(w in n for w in ("minute", "interval", "period", "quarter", "value"))
    ]
    return _ok(
        "M19",
        rev3.retest_q == 2 and rev2.retest_q == 8 and not numeric,
        f"the gas re-test period read from the verbatim clause is {rev3.retest_q} quarters under "
        f"the October 2026 revision and {rev2.retest_q} under the January 2026 one; the answer "
        f"types {sorted(fields)} have no field for it",
    )


@_register("M20")
def check_m20() -> Check:
    config = load("asset_health.config")
    fleet = load("asset_health.simulate.fleet")
    try:
        config.SimConfig(days=60)
        short_ok = True
    except ValueError:
        short_ok = False
    params = sorted(inspect.signature(fleet.simulate_asset).parameters)
    ref = _reference()
    run = run_segments(list(ref.lives))
    join = ref.lives[1].start
    fused = run.frame["fused"]
    window = fused.loc[join : join + timedelta(days=22)]
    return _ok(
        "M20",
        not short_ok
        and not {"stop", "derate", "stops"} & set(params)
        and bool(window.isna().all()),
        f"SimConfig(days=60) is refused: {not short_ok}; simulate_asset takes {params}; after a "
        f"repair record the fused score is empty for the next 22 days ({len(window)} hourly rows "
        f"all empty: {bool(window.isna().all())})",
    )


@_register("M21")
def check_m21() -> Check:
    load("asset_health.pipeline")  # the module that pulls scikit-learn in
    requires = importlib.metadata.requires("asset-health") or []
    core = sorted(
        {
            r.split(";")[0].split(">")[0].split("=")[0].split(" ")[0].lower()
            for r in requires
            if "extra" not in r
        }
    )
    lock_path = Path(__file__).resolve().parents[2] / "integration" / "siblings.lock"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    refs = {k: v["ref"] for k, v in lock["siblings"].items()}
    return _ok(
        "M21",
        "scikit-learn" in core
        and "pandas" in core
        and "sklearn" in sys.modules
        and all(r is None for r in refs.values()),
        f"asset-health requires {core}; importing its pipeline loads scikit-learn: "
        f"{'sklearn' in sys.modules}; the three siblings carry no tag or commit (refs {refs}); "
        "that the package names are unclaimed on a public index was recorded online on 31 October "
        "2026 and cannot be re-checked offline",
    )


# ----------------------------------------------------------------------------- S1 to S12


@cache
def _s1() -> tuple[float, float, bool, bool]:
    """One worn impeller, cut at day 38 and restarted at day 38.25, with and without a repair
    record and with no renewal. The life would fail only at day 110, so the record runs on."""
    params = PlantParams()
    seed = 3
    base = LifeLatent(_PUMP, 0, "IMPELLER_WEAR", 20.0, 110.0, 2.0, 48.0)
    before = render_life(dataclasses.replace(base, end_day=38.0), params, seed, 0)
    plain = dataclasses.replace(base, start_day=38.25)
    after_plain = render_life(plain, params, seed, 1)
    after_repair = render_life(dataclasses.replace(plain, repair_at_start=True), params, seed, 1)
    window = slice(sim_time(62.0), sim_time(80.0))
    with_record = run_segments([before, after_repair])
    without = run_segments([before, after_plain])
    instant = aware(sim_time(72.0))
    cache_a, cache_b = RunCache(), RunCache()
    cache_a.set_history(_PUMP, [before, after_repair])
    cache_b.set_history(_PUMP, [before, after_plain])
    return (
        float(with_record.frame["fused"].loc[window].mean()),
        float(without.frame["fused"].loc[window].mean()),
        bool(cache_a.analysis(_PUMP, instant).work_orders),
        bool(cache_b.analysis(_PUMP, instant).work_orders),
    )


@_register("S1")
def check_s1() -> Check:
    mean_with, mean_without, alert_with, alert_without = _s1()
    return _ok(
        "S1",
        alert_without and not alert_with and mean_with < mean_without,
        f"the same worn impeller, with a repair record at day 38.25 and no renewal: mean fused "
        f"score over days 62 to 80 is {mean_with:.2f} with the record and {mean_without:.2f} "
        f"without it; an alert is open at day 72 without the record: {alert_without}, with it: "
        f"{alert_with}",
    )


@_register("S2")
def check_s2() -> Check:
    fleet = load("asset_health.simulate.fleet")
    models = load("asset_health.models")
    config = load("asset_health.config")
    catalog = load("asset_health.catalog")
    spec = catalog.spec_from_profile(
        catalog.PROFILES[models.AssetType.SLURRY_PUMP],
        asset_id="PMP-01",
        name="pump 1",
        baseline_scale=1.0,
    )
    cfg = config.SimConfig(seed=1, start=datetime(2026, 10, 1), days=61, fleet={spec.asset_type: 1})
    failure = models.FailureEvent(
        asset_id="PMP-01",
        mode=models.FailureMode.IMPELLER_WEAR,
        onset=datetime(2026, 10, 20),
        failure=datetime(2026, 11, 20),
        shape_k=2.0,
        amplitude=1.0,
        downtime_hours=48.0,
    )
    clean_a, latent_a, _ = fleet.simulate_asset(np.random.default_rng(11), cfg, spec, None, [])
    clean_b, latent_b, _ = fleet.simulate_asset(np.random.default_rng(11), cfg, spec, failure, [])
    before = 18 * 24  # rows before the failure's onset (day 19)
    regime_equal = bool(
        np.allclose(latent_a["load"].iloc[:before], latent_b["load"].iloc[:before])
        and np.allclose(
            latent_a["speed_ratio"].iloc[:before], latent_b["speed_ratio"].iloc[:before]
        )
    )
    noise_differs = not bool(
        np.allclose(clean_a["vib_rms"].iloc[:before], clean_b["vib_rms"].iloc[:before])
    )
    params = sorted(inspect.signature(fleet.simulate_asset).parameters)
    return _ok(
        "S2",
        regime_equal and noise_differs and not {"stop", "derate"} & set(params),
        f"simulate_asset takes {params}; with the same generator, before the failure's onset the "
        f"operating regime is identical with and without a failure ({regime_equal}) while the "
        f"measured vibration differs ({noise_differs}): the failure's own draws shift the noise "
        "that follows, because regime and noise share one random stream",
    )


@_register("S3")
def check_s3() -> Check:
    faults = load("asset_health.simulate.faults")
    config = load("asset_health.config")
    pd = load("pandas")
    index = pd.date_range("2026-10-01", periods=24 * 30, freq="h")
    frame = pd.DataFrame({"vib_rms": np.ones(len(index))}, index=index)
    cfg = config.DataFaultConfig(
        dropout_per_channel_day=0.0,
        stuck_per_channel_day=0.0,
        sentinel_prob=0.0,
        glitch_prob=0.0,
        missing_row_prob=0.0,
        comm_outage_per_day=0.0,
        clock_drift_asset_fraction=1.0,
        clock_steps_max=0,
        clock_drift_s_per_day=(100.0, 100.0),
    )

    def offsets(piece: Any) -> Any:
        dirty, _ = faults.apply_data_faults(np.random.default_rng(3), piece, "PMP-01", cfg, 3600.0)
        reported = dirty["timestamp"].reset_index(drop=True)
        return (reported - piece.index.to_series().reset_index(drop=True)).dt.total_seconds()

    half = len(frame) // 2
    at_join = float(offsets(frame).iloc[half])
    at_start = float(offsets(frame.iloc[half:]).iloc[0])
    return _ok(
        "S3",
        abs(at_join - at_start) > 60.0,
        f"a logger clock drifting at 100 s/day reads {at_join:.0f} s off 15 days into one record, "
        f"but {at_start:.0f} s off at the first row of a record that starts there: each record "
        "restarts the drift",
    )


@_register("S4")
def check_s4() -> Check:
    models = load("asset_health.models")
    kinds = sorted(models.DataFaultEvent.model_fields["kind"].annotation.__args__)
    try:
        models.DataFaultEvent(
            asset_id="PMP-01",
            kind="gain_drift",
            start=datetime(2026, 10, 1),
            end=datetime(2026, 10, 2),
        )
        accepted = True
    except ValueError:
        accepted = False
    return _ok(
        "S4",
        not accepted and not any("gain" in k for k in kinds),
        f"asset-health's data-fault kinds are {kinds}; 'gain_drift' is refused: {not accepted}",
    )


@_register("S5")
def check_s5() -> Check:
    config = load("asset_health.config")
    ref = _reference()
    fused = ref.cache.run(_PUMP).frame["fused"].dropna()
    clip = float(config.DetectConfig().clip)
    threshold = float(config.PolicyConfig().alert_threshold)
    peak = float(fused.max())
    return _ok(
        "S5",
        peak > 1.0 and peak <= clip + 1e-9 and threshold == 1.0,
        f"fused score of the reference pump peaks at {peak:.2f}; the clip is {clip} and the alert "
        f"threshold {threshold}: the score is not a value in [0, 1]",
    )


@_register("S6")
def check_s6() -> Check:
    ah = load("asset_health.models")
    params = PlantParams()
    seed = 5
    base = LifeLatent(_PUMP, 0, "IMPELLER_WEAR", 24.0, 52.0, 2.0, 48.0)
    piece = render_life(base, params, seed, 0)
    cache_ = RunCache()
    cache_.set_history(_PUMP, [piece])
    counts: list[int] = []
    statuses: dict[str, int] = {}
    interval_free = None
    for day in range(30, 52):
        analysis = cache_.analysis(_PUMP, aware(sim_time(float(day))))
        for wo in analysis.work_orders:
            counts.append(len(wo.evidence))
            status = wo.rul.status.value if wo.rul is not None else "no_rul"
            statuses[status] = statuses.get(status, 0) + 1
            if status != ah.RULStatus.ESTIMATED.value and interval_free is None:
                interval_free = (
                    day,
                    wo.severity.value,
                    wo.rul is not None and wo.rul.p50_days is None,
                )
    return _ok(
        "S6",
        bool(counts) and max(counts) <= 4 and interval_free is not None,
        f"work orders over days 30 to 51 of the reference pump hold {min(counts, default=0)} to "
        f"{max(counts, default=0)} evidence items (at most 4); RUL status counts {statuses}; first "
        f"work order without an interval: {interval_free}",
    )


@_register("S7")
def check_s7() -> Check:
    verifier = load("maintenance_agent.verifier")
    synthetic = load("maintenance_agent.adapters.synthetic")
    src = synthetic.SyntheticAlertSource.from_fixtures()
    alert_id = src.list_alert_ids()[0]
    alert = src.get_alert(alert_id)
    evidence = src.get_evidence(alert_id)
    base = verifier.alert_fingerprint(alert, evidence)
    confidence = alert.model_copy(deep=True)
    confidence.rul.confidence = 0.9
    other = alert.model_copy(update={"anomaly_score": 0.01, "summary": "something else"})
    return _ok(
        "S7",
        verifier.alert_fingerprint(confidence, evidence) != base
        and verifier.alert_fingerprint(other, evidence) == base,
        "changing rul.confidence alone changes the alert fingerprint (which makes V14 block a "
        "draft); changing the anomaly score and the summary does not",
    )


@_register("S8")
def check_s8() -> Check:
    wiring = load("maintenance_agent.wiring")
    sig = inspect.signature(wiring.build_system)
    keyword_only = [
        n for n, p in sig.parameters.items() if p.kind is inspect.Parameter.KEYWORD_ONLY
    ]
    return _ok(
        "S8",
        "cmms" not in sig.parameters and len(keyword_only) == len(sig.parameters),
        f"build_system is keyword-only with {sorted(sig.parameters)}: the CMMS cannot be passed "
        "in (the same defect as M12, seen from the signature)",
    )


@_register("S9")
def check_s9() -> Check:
    tree = _package_sources("maintenance_agent")["maintenance_agent.cmms"]
    draft = next(
        n
        for c in ast.walk(tree)
        if isinstance(c, ast.ClassDef) and c.name == "MockCMMS"
        for n in c.body
        if isinstance(n, ast.FunctionDef) and n.name == "draft_work_order"
    )
    uses_table = any(isinstance(n, ast.Attribute) and n.attr == "_parts" for n in ast.walk(draft))
    calls_get_part = any(
        isinstance(n, ast.Attribute) and n.attr == "get_part" for n in ast.walk(draft)
    )
    return _ok(
        "S9",
        uses_table and not calls_get_part,
        f"MockCMMS.draft_work_order reads its private parts table ({uses_table}) and does not go "
        f"through get_part ({not calls_get_part}), so a replaced get_part never reaches the dry "
        "run",
    )


@_register("S10")
def check_s10() -> Check:
    world = _pump_world()  # 18:00 on the simulation day of the alert
    alert_id = world.source.raise_alert(_PUMP).alert_id
    assert alert_id is not None
    run_id = run_id_for(alert_id)
    state = world.orch.run(alert_id, run_id=run_id)
    human = world.human("planner1", np.random.default_rng(1), PlantParams())
    approval = None
    for _ in range(11):  # looks every half hour until midnight
        state = world.orch.refresh_verification(run_id)
        approval = human.review(state)
        if approval is not None:
            break
        world.advance(minutes=30)
    assert approval is not None, "the simulated approver did not sign before midnight"
    signed_on = world.clock().date()
    now = world.now[0]
    world.now[0] = datetime(now.year, now.month, now.day) + timedelta(days=1, minutes=5)
    after = world.orch.submit_approval(run_id, approval)
    return _ok(
        "S10",
        after.status.value == "AWAITING_APPROVAL" and not world.queued,
        f"an approval signed on {signed_on} and submitted after midnight leaves the run "
        f"{after.status.value} with nothing released: the verification digest covers the date",
    )


@_register("S11")
def check_s11() -> Check:
    rules = load("maintenance_agent.planners.rules")
    world = _pump_world()
    alert_id = world.source.raise_alert(_PUMP).alert_id
    assert alert_id is not None
    state = world.orch.run(alert_id, run_id=run_id_for(alert_id))
    draft = state.draft
    playbook = rules.Playbook.from_fixtures()
    weights = playbook.modes
    mapped = set(FEATURE_SIGNAL.values())
    ceiling = {
        mode: round(max((w for s, w in sig.items() if s in mapped), default=0.0), 2)
        for mode, sig in weights.items()
        if mode in ("LINER_WEAR", "IMPELLER_WEAR", "BEARING_DEFECT")
    }
    confidence = float(draft["hypotheses"][0]["confidence"])
    return _ok(
        "S11",
        draft["inspection_first"] is True and confidence < playbook.min_confidence,
        f"the draft made from the real asset-health evidence of the reference pump is "
        f"inspection-first with confidence {confidence:.2f} against a minimum of "
        f"{playbook.min_confidence}; the heaviest single signal a mapped channel can reach per "
        f"mode is {ceiling}",
    )


@_register("S12")
def check_s12() -> Check:
    copilot = load("plant_copilot.copilot")
    real = copilot.load_corpus
    calls = [0]

    def counted(*args: Any, **kwargs: Any) -> Any:
        calls[0] += 1
        return real(*args, **kwargs)

    copilot.load_corpus = counted
    seconds = []
    try:
        for _ in range(2):
            t0 = time.perf_counter()
            copilot.PlantCopilot(as_of=date(2026, 10, 1))
            seconds.append(time.perf_counter() - t0)
    finally:
        copilot.load_corpus = real
    return _ok(
        "S12",
        calls[0] == 2,
        f"two PlantCopilot constructions read the corpus {calls[0]} times "
        f"({seconds[0]:.2f} s and {seconds[1]:.2f} s on this machine)",
    )


# ----------------------------------------------------------------------------- estimator mismatch


@dataclass(frozen=True)
class LifeMismatch:
    """One life: the mean absolute error of each estimator's point RUL against the truth."""

    seed: int
    asset_id: str
    mode: str
    days: int  # checkpoints on which both estimators gave a trend estimate
    mae_f1: float
    mae_f2: float

    @property
    def diff(self) -> float:
        return self.mae_f2 - self.mae_f1


def dev_lives(n: int = 20, first_seed: int = 1000) -> list[tuple[int, str]]:
    """The first ``n`` (seed, asset) pairs, from development seed ``first_seed`` up, whose first
    life fails inside the episode when nobody acts (a failing life has a truth to be wrong
    about)."""
    params = PlantParams()
    out: list[tuple[int, str]] = []
    seed = first_seed
    while len(out) < n:
        if not 1000 <= seed <= 1999:
            raise ValueError("dev lives come from the development seeds 1000 to 1999")
        table = simulate_lives(params, seed, Scenario())
        for aid in ASSET_IDS:
            if table.failure_day[aid] < table.n_days and len(out) < n:
                out.append((seed, aid))
        seed += 1
    return out


def estimator_mismatch(lives: Sequence[tuple[int, str]]) -> list[LifeMismatch]:
    """F1 against F2 on the first life of each (seed, asset), in the world where nobody acts.

    F1 is the streaming Theil-Sen estimator on the plant's own health-index readings; F2 is
    asset-health's ``estimate_rul`` on telemetry its generator writes from the same life. Both are
    scored on the days on which both report a trend estimate, against the realised remaining life.
    """
    params = PlantParams()
    pd = load("pandas")
    health = load("asset_health.rul.health_index")
    estimate = load("asset_health.rul.estimate")
    tables = {s: simulate_lives(params, s, Scenario()) for s in {s for s, _ in lives}}
    out: list[LifeMismatch] = []
    for seed, aid in lives:
        table = tables[seed]
        life = table.lives[aid]
        piece = render_life(
            LifeLatent(
                aid,
                0,
                life.mode,
                float(table.onset_day[aid]),
                float(table.failure_day[aid]),
                float(life.shape),
                float(life.downtime_hours),
            ),
            params,
            seed,
            0,
        )
        cache_ = RunCache()
        cache_.set_history(aid, [piece])
        run = cache_.run(aid)
        hi = health.health_index(run.z24, cache_.analysis_cfg.rul.top_k)
        f1 = TheilSenEstimator(params)
        last = min(table.n_days, int(np.ceil(table.failure_day[aid])))
        e1: list[float] = []
        e2: list[float] = []
        for day in range(last):
            h = table.hi[aid][day]
            o1 = f1.update(aid, day, None if np.isnan(h) else float(h), 0)
            est = estimate.estimate_rul(
                piece.ah_asset_id,
                hi,
                pd.Timestamp(sim_time(float(day)).replace(tzinfo=None)),
                cache_.rul_model,
                cache_.analysis_cfg.rul,
                piece.spec.asset_type,
            )
            if o1.status == STATUS_ESTIMATED and est.status.value == "estimated":
                truth = float(table.true_rul_days[aid][day])
                e1.append(abs(o1.m_days - truth))
                e2.append(abs(float(est.p50_days) - truth))
        if e1:
            out.append(
                LifeMismatch(
                    seed, aid, str(life.mode), len(e1), float(np.mean(e1)), float(np.mean(e2))
                )
            )
    return out


def mismatch_record(rows: Sequence[LifeMismatch], cap_days: float = 30.0) -> ResultRecord:
    """The mean of ``mae_f2 - mae_f1`` over lives (days) as a ``ResultRecord``, with a betting
    confidence sequence on the per-life differences clipped to ``[-cap_days, cap_days]``."""
    from closed_loop.eval.stages import value_record

    if not rows:
        raise ValueError("no life had a day on which both estimators reported a trend")
    diffs = [r.diff for r in rows]
    where = (
        f"{len(rows)} first lives that fail inside the episode (seeds "
        f"{min(r.seed for r in rows)} to {max(r.seed for r in rows)}, development role), nobody "
        "acting; mean absolute error of the point remaining-useful-life estimate against the "
        "realised remaining life on the days both estimators report a trend, F2 (asset-health "
        "pipeline on rendered telemetry) minus F1 (streaming Theil-Sen estimator)"
    )
    rec = value_record(
        "f2.estimator_mismatch.mae_diff_days",
        diffs,
        lo=-cap_days,
        hi=cap_days,
        fidelity=Fidelity.F2,
        measured_on=where,
        extra={
            "mean_mae_f1": float(np.mean([r.mae_f1 for r in rows])),
            "mean_mae_f2": float(np.mean([r.mae_f2 for r in rows])),
            "lives_f2_worse": sum(1 for d in diffs if d > 0),
            "checkpoints": int(sum(r.days for r in rows)),
        },
    )
    return ResultRecord(
        name=rec.name,
        value=rec.value,
        n=rec.n,
        fidelity=rec.fidelity,
        measured_on=rec.measured_on,
        script=SCRIPT,
        ci_lo=rec.ci_lo,
        ci_hi=rec.ci_hi,
        ci_kind=rec.ci_kind,
        extra=rec.extra,
    )
