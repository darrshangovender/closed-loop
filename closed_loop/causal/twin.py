"""Observational lives under an operator policy, and twin-world ground truth (packet P7).

*Twin worlds.* The plant keys every exogenous stream (onset, duration, shape, noise, load, upsets,
data-quality events) through ``rng_for(root_seed, stream, *ids)``, so ``F1Plant.twin`` replays the
same world with a ``do()`` intervention added and nothing else changes: a replacement or a derate in
one world cannot shift the noise of another asset, of a later life or of a later event
(``tests/plant/test_twin.py`` shows it). ``twin_truth`` uses that to answer a counterfactual query
exactly: abduction is the shared seed, action is the injected command, prediction is the plant's own
dynamics. No operator acts in a twin; it holds only the commands of the query.

*Operator policies* (``run_world``, ``generate_observational``). The observational data are one
episode per seed of the F1+D plant in which an operator derates assets:

``pi_obs``
    the confounded policy of 0001 section 7. Each day the operator looks at an asset with
    probability ``p_check`` and derates it if its health index, averaged over the last
    ``smooth_days`` readings, is above ``theta_hi``. Who gets derated, and when, depends on how
    fast the degradation shows, which is also what drives the failure time.
``randomised``
    a coin flip at the first alert, independent of everything (the unconfounded control).
``none``
    nobody acts.

Policy randomness comes from its own keyed streams (``"operator"``, ``"operator_coin"``), so it
never touches the plant's streams. Every number here is SYNTHETIC; the operator parameters are
assumptions (A).

*Causal-study plant.* ``causal_params`` returns the F1+D configuration: derating enabled, every
first component designed to fail inside the horizon (``p_fail_in_episode = 1``) and a 150-day
horizon, so that nearly every life is seen to failure even after a derate has postponed it, and
commands are never refused (``p_refuse = 0``: there are no crews or permits in this study, and a
refused ``do()`` would make the intervention time random).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from closed_loop.causal.records import AlertLabel, LifeRecord
from closed_loop.core.rng import rng_for
from closed_loop.core.timegrid import DAY_Q
from closed_loop.core.types import Cmd, Command, Fidelity, Op, PlanCmd
from closed_loop.plant import DO_REPLACE, F1Plant, PlantParams, Scenario, make_plant
from closed_loop.plant.observe import DailyObs, blind_until_day
from closed_loop.plant.params import ASSET_IDS

POLICIES = ("pi_obs", "randomised", "none")
CAUSAL_HORIZON_DAYS = 150
MAX_TRUTH_DAY = 400.0  # the plant knows failure times up to horizon + 210 days of extra load


def causal_params(
    base: PlantParams | None = None, *, tilt: float | None = None, **overrides: object
) -> PlantParams:
    """The F1+D plant of the causal study (see the module docstring).

    ``tilt`` sets ``mode_load_tilt`` (the lever of C-DIAG); other fields are replaced by keyword.
    """
    p = base if base is not None else PlantParams()
    changes: dict[str, object] = {
        "derate_enabled": True,
        "p_fail_in_episode": 1.0,
        "horizon_q": CAUSAL_HORIZON_DAYS * DAY_Q,
        "p_refuse": 0.0,
    }
    if tilt is not None:
        changes["mode_load_tilt"] = tilt
    changes.update(overrides)
    out = dataclasses.replace(p, **changes)  # type: ignore[arg-type]
    out.validate()
    return out


@dataclass(frozen=True)
class OperatorConfig:
    """Assumptions about the operator (A)."""

    theta_hi: float = 3.0  # smoothed health index at which a checked asset is derated
    smooth_days: int = 3
    p_check: float = 0.2  # chance the operator looks at a given asset on a given day
    p_coin: float = 0.5  # the randomised policy derates this share of assets at their first alert


# ------------------------------------------------------------------------------------ one world
@dataclass
class _Collector:
    days: list[int]
    hi: list[float | None]
    sig: list[tuple[float, float, float]]
    load: list[float]
    derated: list[bool]
    fail_q: int | None = None
    derate_start: float | None = None
    alerts: list[AlertLabel] | None = None

    def add(self, reading: DailyObs, derated: bool) -> None:
        self.days.append(reading.day)
        self.hi.append(reading.hi)
        self.sig.append((reading.sig_bearing, reading.sig_proc, reading.sig_hyd))
        self.load.append(reading.load)
        self.derated.append(derated)


def run_world(
    seed: int,
    params: PlantParams,
    policy: str = "pi_obs",
    scenario: Scenario | None = None,
    operator: OperatorConfig | None = None,
) -> list[LifeRecord]:
    """One episode of the F1+D plant under ``policy``; the four first lives as ``LifeRecord``s."""
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
    op_cfg = operator if operator is not None else OperatorConfig()
    if policy != "none" and not params.derate_enabled:
        raise ValueError("a derating operator needs derate_enabled (use causal_params)")
    plant = make_plant(params, fidelity=Fidelity.F1D if params.derate_enabled else Fidelity.F1)
    obs = plant.reset(seed, scenario if scenario is not None else Scenario())
    n_days = params.horizon_days
    look = {aid: rng_for(seed, "operator", aid).random(n_days) for aid in ASSET_IDS}
    coin = {aid: float(rng_for(seed, "operator_coin", aid).random()) for aid in ASSET_IDS}
    col = {aid: _Collector([], [], [], [], []) for aid in ASSET_IDS}
    for c in col.values():
        c.alerts = []
    recent: dict[str, list[float]] = {aid: [] for aid in ASSET_IDS}
    day = -1
    while True:
        cmds: list[Command] = []
        if obs.day != day:
            day = obs.day
            for aid in ASSET_IDS:
                view = obs.assets[aid]
                reading = view.last_obs
                if reading is None or col[aid].fail_q is not None:
                    continue
                col[aid].add(reading, view.derated)
                if reading.hi is not None:
                    recent[aid].append(reading.hi)
                if view.derated or view.op is not Op.RUN:
                    continue
                first_alert = f"{aid}-L0-A1" in obs.alerts
                if _operator_derates(
                    policy, op_cfg, params, recent[aid], reading, look[aid][day], first_alert
                ):
                    if policy == "pi_obs" or coin[aid] < op_cfg.p_coin:
                        cmds.append(Command(aid, Cmd.DERATE, source="operator"))
        q_now = obs.quarter
        result = plant.step(cmds)
        for done in result.executed:
            if done.cmd is Cmd.DERATE and col[done.asset_id].derate_start is None:
                col[done.asset_id].derate_start = q_now / DAY_Q
        obs = result.obs
        for aid in ASSET_IDS:
            if col[aid].fail_q is None and obs.assets[aid].op is Op.FAIL:
                col[aid].fail_q = obs.quarter  # the state at the start of this quarter is FAIL
        for alert_id in obs.alerts:
            t = plant.alert_truth(alert_id)
            level = int(alert_id.rsplit("-A", 1)[1])
            lab = AlertLabel(alert_id, t.day, level, t.cause, t.mode)
            col[t.asset_id].alerts.append(lab)  # type: ignore[union-attr]
        if result.done:
            break
    return [_record(aid, col[aid], seed, n_days, plant.truth.life_mode[aid]) for aid in ASSET_IDS]


def _operator_derates(
    policy: str,
    cfg: OperatorConfig,
    params: PlantParams,
    recent: list[float],
    reading: DailyObs,
    u_look: float,
    first_alert: bool,
) -> bool:
    if policy == "none" or reading.hi is None:
        return False
    if policy == "pi_obs":
        if reading.day < blind_until_day(0.0, params) or u_look >= cfg.p_check:
            return False
        smooth = float(np.mean(recent[-cfg.smooth_days :]))
        return smooth > cfg.theta_hi
    return first_alert  # randomised: act at the first alert (the caller applies the coin)


def _record(aid: str, c: _Collector, seed: int, n_days: int, mode: str) -> LifeRecord:
    days = np.array(c.days, dtype=np.float64)
    fail_day = None if c.fail_q is None else c.fail_q / DAY_Q
    if fail_day is not None:
        keep = days * DAY_Q <= c.fail_q  # type: ignore[operator]
    else:
        keep = np.ones(days.size, dtype=bool)
    days = days[keep]
    hi_all = [h for h, k in zip(c.hi, keep, strict=True) if k]
    has_hi = np.array([h is not None for h in hi_all], dtype=bool)
    sig = np.array(c.sig, dtype=np.float64)[keep].T
    derated = np.array(c.derated, dtype=bool)[keep]
    labels = tuple(a for a in (c.alerts or []) if fail_day is None or a.day <= fail_day)
    return LifeRecord(
        asset_id=aid,
        hi_days=days[has_hi],
        hi=np.array([h for h in hi_all if h is not None], dtype=np.float64),
        derate_days=days[derated],
        fail_day=fail_day,
        censored_day=None if fail_day is not None else float(n_days),
        days=days,
        sig=sig,
        load=np.array(c.load, dtype=np.float64)[keep],
        derate_start=c.derate_start,
        mode=mode,
        seed=seed,
        alerts=labels,
    )


def generate_observational(
    n: int,
    seed: int,
    policy: str = "pi_obs",
    *,
    params: PlantParams | None = None,
    scenario: Scenario | None = None,
    operator: OperatorConfig | None = None,
) -> list[LifeRecord]:
    """``n`` first lives from consecutive seeds ``seed, seed + 1, ...`` (four lives per seed).

    Frozen text of 0002 section 3.9; the keywords are additive. Pass the study's plant with
    ``params=causal_params(...)``; the default is ``causal_params()``.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    p = params if params is not None else causal_params()
    out: list[LifeRecord] = []
    s = seed
    while len(out) < n:
        out.extend(run_world(s, p, policy, scenario, operator))
        s += 1
    return out[:n]


# ------------------------------------------------------------------------------- twin worlds
@dataclass(frozen=True)
class Query:
    """A counterfactual question about one life.

    ``kind`` is ``"nothing"`` (no intervention), ``"derate"`` (derate from ``day`` on, retaining
    ``load_cap`` of the load: the plant's derate keeps ``1 - derate_fraction``, 0.7) or
    ``"replace"`` (swap the component at ``day``).
    """

    kind: str
    asset_id: str
    day: float = 0.0
    load_cap: float = 0.7

    def __post_init__(self) -> None:
        if self.kind not in ("nothing", "derate", "replace"):
            raise ValueError(f"unknown query kind {self.kind!r}")
        if self.asset_id not in ASSET_IDS:
            raise ValueError(f"unknown asset {self.asset_id!r}")
        if self.day < 0.0:
            raise ValueError("day must be non-negative")
        if not 0.0 < self.load_cap <= 1.0:
            raise ValueError("load_cap must be in (0, 1]")


@dataclass(frozen=True)
class TruthAnswer:
    """The first uncontrolled failure in the twin world of a query.

    ``failure_day`` is the day of the first failure (the old component's if it failed before a
    replacement, otherwise the component that is installed then), capped at ``MAX_TRUTH_DAY``.
    ``applied_day`` is when the intervention took effect: the query's day, unless the plant
    refused the command and it was repeated each quarter until executed (only when ``p_refuse``
    is above zero); ``None`` for a query with no intervention.
    """

    failure_day: float
    query: Query
    applied_day: float | None = None


def twin_truth(
    life_seed: int,
    query: Query,
    *,
    params: PlantParams | None = None,
    scenario: Scenario | None = None,
) -> TruthAnswer:
    """Rerun the F1+D plant of seed ``life_seed`` with the intervention of ``query``."""
    return twin_truths(life_seed, [query], params=params, scenario=scenario)[0]


def twin_truths(
    life_seed: int,
    queries: Sequence[Query],
    *,
    params: PlantParams | None = None,
    scenario: Scenario | None = None,
) -> list[TruthAnswer]:
    """Answers to several queries on one seed, sharing twin worlds where that is exact.

    Interventions on different assets do not interact (the assets share load processes but
    nothing the intervention changes), so all queries of one kind run in one twin world. Each
    answer is read from the plant's truth right after its own intervention quarter.
    """
    p = params if params is not None else causal_params()
    sc = scenario if scenario is not None else Scenario()
    cap_ok = 1.0 - p.derate_fraction
    for qy in queries:
        if qy.kind != "nothing" and qy.day * DAY_Q >= p.horizon_q:
            raise ValueError(f"query day {qy.day} is past the end of the episode")
        if qy.kind == "derate" and not math.isclose(qy.load_cap, cap_ok, rel_tol=1e-9):
            raise ValueError(
                f"the plant derates to {cap_ok:.3f} of the load; the query asks for {qy.load_cap}"
            )
    answers: dict[int, TruthAnswer] = {}
    idx = list(range(len(queries)))

    base = make_plant(p, fidelity=Fidelity.F1D)
    base.reset(life_seed, sc)
    nothing = base.truth.failure_time_q
    for i in idx:
        if queries[i].kind == "nothing":
            answers[i] = TruthAnswer(_cap(nothing[queries[i].asset_id] / DAY_Q), queries[i])

    for kind in ("derate", "replace"):
        # Two interventions of one kind on the same asset cannot share a world (a second derate
        # would find the asset derated already), so the k-th such query goes to the k-th world.
        worlds: list[list[int]] = []
        seen: dict[str, int] = {}
        for i in (j for j in idx if queries[j].kind == kind):
            k = seen.get(queries[i].asset_id, 0)
            seen[queries[i].asset_id] = k + 1
            while len(worlds) <= k:
                worlds.append([])
            worlds[k].append(i)
        for group in worlds:
            _run_world_group(base, p, kind, queries, group, nothing, answers)
    return [answers[i] for i in idx]


def _run_world_group(
    base: F1Plant,
    p: PlantParams,
    kind: str,
    queries: Sequence[Query],
    group: list[int],
    nothing: Mapping[str, float],
    answers: dict[int, TruthAnswer],
) -> None:
    """Run one twin world with the interventions of ``group`` and read each answer."""
    twin = base.twin([])
    waiting = sorted(group, key=lambda i: queries[i].day)
    while waiting:
        q = twin.quarter
        due = [i for i in waiting if round(queries[i].day * DAY_Q) <= q]
        cmds = [_do_command(queries[i]) for i in due]
        if q >= p.horizon_q:
            raise RuntimeError("an intervention could not be applied before the episode ended")
        res = twin.step(cmds)
        done = {(c.asset_id, c.source) for c in res.executed}
        for i in due:
            qy = queries[i]
            cmd = _do_command(qy)
            if (cmd.asset_id, cmd.source) not in done:
                continue  # refused: try again next quarter
            waiting.remove(i)
            t_old = nothing[qy.asset_id] / DAY_Q
            if kind == "replace" and t_old * DAY_Q <= q:
                fail = t_old  # the old component had failed before the swap
            else:
                fail = twin.truth.failure_time_q[qy.asset_id] / DAY_Q
            answers[i] = TruthAnswer(_cap(fail), qy, q / DAY_Q)


def _do_command(qy: Query) -> Command:
    if qy.kind == "derate":
        return Command(qy.asset_id, Cmd.DERATE, source="do:derate")
    return Command(qy.asset_id, PlanCmd.RAISE_WO, "any", source=DO_REPLACE)


def _cap(day: float) -> float:
    return float(min(day, MAX_TRUTH_DAY))


# ------------------------------------------------------------------ the passive world, fast
def passive_world(
    seed: int, params: PlantParams, scenario: Scenario | None = None
) -> list[LifeRecord]:
    """The four first lives of a world where nobody acts, without stepping the plant.

    Equal to ``run_world(seed, params, "none", scenario)`` (``tests/causal/test_twin.py`` compares
    them): the readings, failure times and alert rules come from the same plant functions
    (``simulate_lives`` and ``alert_cause``), only the quarter-by-quarter stepping is skipped. About
    eight times faster, which is what the diagnosis study needs at scale.
    """
    from closed_loop.plant.degradation import build_physical
    from closed_loop.plant.observe import alert_cause, build_obs_streams, simulate_lives

    sc = scenario if scenario is not None else Scenario()
    table = simulate_lives(params, seed, sc)
    phys = build_physical(params, seed, sc.resolve(seed, params))
    levels = (params.hi_alert, 0.5 * (params.hi_alert + params.hi_fail))
    blind = blind_until_day(0.0, params)
    n_days = params.horizon_days
    out = []
    for aid in ASSET_IDS:
        streams = build_obs_streams(params, seed, aid, phys)
        hi = table.hi[aid]
        fail_day = table.failure_day[aid] if table.failure_day[aid] < n_days else None
        life = table.lives[aid]
        fail_q = None if fail_day is None else round(fail_day * DAY_Q)
        last = n_days - 1 if fail_q is None else min(fail_q // DAY_Q, n_days - 1)
        days = np.arange(last + 1, dtype=np.float64)
        labels = []
        lvl = 0
        for d in range(n_days):
            h = hi[d]
            if np.isnan(h) or lvl >= len(levels) or h < levels[lvl] or d < blind:
                continue
            lvl += 1
            cause = alert_cause(
                params, streams, d, float(table.severity[aid][d]), float(table.load_true[aid][d])
            )
            if fail_day is None or d <= fail_day:
                labels.append(AlertLabel(f"{aid}-L0-A{lvl}", d, lvl, cause, life.mode))
        h_keep = hi[: last + 1]
        has_hi = ~np.isnan(h_keep)
        out.append(
            LifeRecord(
                asset_id=aid,
                hi_days=days[has_hi],
                hi=h_keep[has_hi],
                derate_days=np.zeros(0),
                fail_day=fail_day,
                censored_day=None if fail_day is not None else float(n_days),
                days=days,
                sig=table.sig[aid][:, : last + 1],
                load=table.load[aid][: last + 1],
                derate_start=None,
                mode=life.mode,
                seed=seed,
                alerts=tuple(labels),
            )
        )
    return out
