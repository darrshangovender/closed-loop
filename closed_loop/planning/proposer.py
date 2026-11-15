"""``PlannerProposer``: the belief planner as a controller of the SYNTHETIC plant (packet P8b).

It is a ``Proposer`` (``proposers.base``, docs/adr/0002 section 3.8) built from three parts:

* a ``BeliefTracker`` that folds the day's health-index reading, the observed load and any
  inspection result into a particle belief for every asset;
* a ``BeliefPlanner`` that chooses, once a day, which assets get a work order now, which get an
  inspection and which spares are ordered (``planner.py``);
* the plant's own job path: every open work order is carried out by ``plant.jobs.JobRunner``, one
  command per quarter, exactly as for the baselines. A plan is therefore a *job choice* and the
  shield, when there is one, sits below the same commands for every controller.

Decisions the interface text leaves open:

* **Diagnosis is in the loop (docs/adr/0003 A3).** ``diagnoser`` works as it does for B3: a name
  (``"scm"``, ``"corr"``, ``"oracle"``) or a ``Diagnoser``. The work order carries the diagnosed
  mode and a diagnosis of ``sensor_fault`` or ``process_upset`` raises none. A named diagnoser is
  not fitted: call ``fit_diagnoser``. ``diagnoser=None`` uses the most probable mode under the
  belief. An asset that has already failed is repaired whatever the diagnosis says.
* A completed work order is taken to have renewed the component (the controller cannot tell a wrong
  repair from a right one): the belief starts a new life and the day's reading is not used until the
  next day boundary. Work on the two pumps is never isolated at the same time: the first command
  that would isolate a pump (``PROVE``) is dropped while the other is isolated, in work or failed.
* The planner runs at the first quarter of every day, when the new reading arrives, and again
  whenever an asset is down with no work order and the plant's picture of the assets has changed.
  A work order the plant refused is raised again for up to two hours.
* ``mismatch`` is the wear-rate error of the world model (``Arm.planner_mismatch``; 0001 section 8).
* The world model is fitted on development seeds only (``planning.fitting``), on first use.

The proposer reads the ``Observation`` and its own belief. It imports no truth type; the test
``tests/planning/test_planner_truth_free.py`` replays an episode with the plant's truth poisoned.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from functools import cache
from types import MappingProxyType
from typing import TYPE_CHECKING

from closed_loop.core.timegrid import DAY_Q
from closed_loop.core.types import Cmd, Command, Op, PlanCmd
from closed_loop.planning.belief import BeliefTracker, ModelSource
from closed_loop.planning.planner import (
    BeliefPlanner,
    PlanConstraints,
    PlanCosts,
    PlanDecision,
    mean_life_days,
)
from closed_loop.planning.worldmodel import PlantWorldModel, WorldModel
from closed_loop.plant.jobs import JobRunner
from closed_loop.plant.params import (
    ASSET_CLASS,
    ASSET_IDS,
    CONFINED_JOBS,
    MODES_BY_CLASS,
    PUMPS,
    PlantParams,
)
from closed_loop.plant.plant import AssetView, Observation
from closed_loop.proposers.stall import StallWatch

if TYPE_CHECKING:
    from closed_loop.causal.diagnoser import Diagnoser, Diagnosis
    from closed_loop.causal.records import HistoryRecorder, LifeRecord

__all__ = ["PlannerProposer", "default_models"]

DEV_FIT_SEEDS: tuple[int, ...] = tuple(range(1000, 1100))
_RETRY_Q = 8  # a refused work order is raised again for two hours


@cache
def _fitted(seeds: tuple[int, ...]) -> Mapping[str, float]:
    from closed_loop.planning.fitting import fit_world_model  # reads the simulator, dev seeds only

    return MappingProxyType(fit_world_model(seeds))


def default_models(
    mismatch: float = 0.0, fit_seeds: Sequence[int] = DEV_FIT_SEEDS
) -> dict[str, PlantWorldModel]:
    """One ``PlantWorldModel`` per asset, fitted on development seeds (cached per process)."""
    fitted = _fitted(tuple(int(s) for s in fit_seeds))
    return {
        aid: PlantWorldModel(fitted, mismatch, asset_class=ASSET_CLASS[aid]) for aid in ASSET_IDS
    }


class PlannerProposer:
    """The belief planner as a ``Proposer`` (see the module docstring)."""

    name = "planner"

    def __init__(
        self,
        model: ModelSource | None = None,
        *,
        params: PlantParams | None = None,
        n_particles: int = 128,
        horizon_days: int = 14,
        delta_plan: float = 0.02,
        seed: int = 0,
        mismatch: float = 0.0,
        diagnoser: Diagnoser | str | None = "scm",
        runner: JobRunner | None = None,
        cooldown_q: int = DAY_Q,
        inspect_cooldown_days: int = 2,
        order_retry_days: int = 30,
        full_evaluation: bool = False,
        lookahead_readings: int | None = None,
    ) -> None:
        if cooldown_q < 0 or inspect_cooldown_days < 0 or order_retry_days < 1:
            raise ValueError("cooldown_q and inspect_cooldown_days >= 0, order_retry_days >= 1")
        self.params = params if params is not None else PlantParams()
        self.runner = runner if runner is not None else JobRunner(source=self.name)
        self.n_particles = n_particles
        self.horizon_days = horizon_days
        self.delta_plan = delta_plan
        self.cooldown_q = cooldown_q
        self.inspect_cooldown_days = inspect_cooldown_days
        self.order_retry_days = order_retry_days
        self.full_evaluation = full_evaluation
        self.lookahead_readings = lookahead_readings
        self.fast_path = True  # tests switch it off to show it changes nothing
        self.mismatch = mismatch
        self._models: ModelSource = model if model is not None else default_models(mismatch)
        self._costs = self._plant_costs()
        self._diagnoser: Diagnoser | None
        if isinstance(diagnoser, str):
            from closed_loop.causal.diagnoser import make_diagnoser

            self._diagnoser = make_diagnoser(diagnoser)
        else:
            self._diagnoser = diagnoser
        self.tracker: BeliefTracker
        self.planner: BeliefPlanner
        self.reset(seed)

    # ------------------------------------------------------------------------------------ setup
    def _model_of(self, asset_id: str) -> WorldModel:
        src = self._models
        if isinstance(src, Mapping):
            return src[asset_id]
        if callable(src) and not hasattr(src, "failed"):
            return src(asset_id)
        return src  # type: ignore[return-value]

    def _plant_costs(self) -> dict[str, PlanCosts]:
        """Per asset: the plant's costs, with a day of life worth a job over the mean life."""
        p = self.params
        # Mean approval latency of the log-normal delay, in days: the time an asset can still fail
        # after a work order has been raised.
        lead = p.approver_median_h * math.exp(0.5 * p.approver_sigma**2) / 24.0
        by_model: dict[int, float] = {}
        out: dict[str, PlanCosts] = {}
        for aid in ASSET_IDS:
            model = self._model_of(aid)
            if id(model) not in by_model:
                by_model[id(model)] = mean_life_days(model)
            out[aid] = PlanCosts.for_plant(
                planned_job_cost=p.planned_job_cost,
                unplanned_cost_ratio=p.unplanned_cost_ratio,
                inspect_cost=p.inspect_cost,
                mean_life_days=by_model[id(model)],
                lead_days=lead,
            )
        return out

    @property
    def diagnoser(self) -> Diagnoser | None:
        return self._diagnoser

    def fit_diagnoser(self, lives: Sequence[LifeRecord]) -> None:
        if self._diagnoser is None:
            raise ValueError("this planner has no diagnoser")
        self._diagnoser.fit(lives)

    # ----------------------------------------------------------------------------- Proposer
    def reset(self, seed: int) -> None:
        self.seed = seed
        self.tracker = BeliefTracker(self._models, n_particles=self.n_particles, seed=seed)
        self.planner = BeliefPlanner(
            self._models,
            n_particles=self.n_particles,
            horizon_days=self.horizon_days,
            delta_plan=self.delta_plan,
            seed=seed,
            costs=self._costs,
            full_evaluation=self.full_evaluation,
            lookahead_readings=self.lookahead_readings,
        )
        self._day = -1
        self._stall = StallWatch()
        self._planned_stalled: frozenset[str] = frozenset()
        self._seen_assets: Mapping[str, AssetView] | None = None
        self._planned_assets: Mapping[str, AssetView] | None = None
        self._active = True
        self._jobs_done: dict[str, int] = {}
        self._renewed_day: dict[str, int] = {}
        self._last_raise: dict[str, int] = {}
        self._pending: dict[str, tuple[int, Command]] = {}
        self._ordered: dict[str, int] = {}
        self._inspected_day: dict[str, int] = {}
        self._inspection_seen: dict[str, tuple[int, float]] = {}
        self._recorder: HistoryRecorder | None = None
        self._diagnosed: dict[str, tuple[int, Diagnosis]] = {}
        self.diagnoses: list[Diagnosis] = []
        self.decisions: list[tuple[int, PlanDecision]] = []

    def propose(self, obs: Observation) -> list[Command]:
        if (
            self.fast_path
            and obs.assets is self._seen_assets
            and obs.day == self._day
            and not self._active
        ):
            # A quiet quarter: the plant handed back the very mapping of the last one, so no asset
            # view changed, no work order is open or being raised again, nothing is down.
            return []
        out: list[Command] = []
        for aid, view in obs.assets.items():
            self._look(aid, view, obs)
        self._stall.observe(obs)
        stalled = self._stall.stalled_assets(obs)
        new_day = obs.day != self._day
        if new_day:
            self._day = obs.day
            self._daily(obs)
        self._inspections(obs)
        proving: set[str] = set()
        for view in obs.assets.values():
            if view.job is None:
                continue
            cmd = self.runner.next_command(view, view.job, obs.quarter)
            if cmd is not None and self._allowed(cmd, obs, proving):
                out.append(cmd)
        down = any(
            (v.op is Op.FAIL or v.asset_id in stalled) and not v.job_open
            for v in obs.assets.values()
        )
        if new_day or (
            down and (obs.assets is not self._planned_assets or stalled != self._planned_stalled)
        ):
            self._planned_assets = obs.assets
            self._planned_stalled = stalled
            out.extend(self._plan(obs, stalled))
        out.extend(self._retry(obs, out))
        self._seen_assets = obs.assets
        # an asset standing stopped with no work order keeps the planner awake: its clock runs
        self._active = bool(self._pending) or any(
            v.job_open or v.op is Op.FAIL or v.op is Op.STOP for v in obs.assets.values()
        )
        return out

    # ------------------------------------------------------------------------------ bookkeeping
    def _look(self, aid: str, view: AssetView, obs: Observation) -> None:
        """A completed work order (``jobs_done`` moved) is taken as a renewal."""
        seen = self._jobs_done.setdefault(aid, view.jobs_done)
        if view.jobs_done == seen:
            return
        self._jobs_done[aid] = view.jobs_done
        self._renewed_day[aid] = obs.day
        self.tracker.reset_life(aid, obs.day)
        if self._recorder is not None:
            self._recorder.reset_life(aid, obs.day)
        self._diagnosed.pop(aid, None)
        self._pending.pop(aid, None)

    def _stale(self, aid: str, view: AssetView) -> bool:
        """The reading was taken before the last completed work order: it is the old part's."""
        return view.last_obs is not None and view.last_obs.day <= self._renewed_day.get(aid, -1)

    def _daily(self, obs: Observation) -> None:
        for aid, view in obs.assets.items():
            reading = view.last_obs
            if reading is None or self._stale(aid, view):
                continue
            self.tracker.update(aid, reading.day, reading.hi, False, load=reading.load)
        if self._diagnoser is not None:
            if self._recorder is None:
                from closed_loop.causal.records import HistoryRecorder

                self._recorder = HistoryRecorder()
            kept = {a: v for a, v in obs.assets.items() if not self._stale(a, v)}
            self._recorder.record(dataclasses.replace(obs, assets=MappingProxyType(kept)))

    def _inspections(self, obs: Observation) -> None:
        """An inspection result is a reading taken by a person: it enters the belief sharper."""
        for aid, view in obs.assets.items():
            ins = view.inspection
            if ins is None or self._inspection_seen.get(aid) == (ins.day, ins.hi):
                continue
            self._inspection_seen[aid] = (ins.day, ins.hi)
            if ins.day <= self._renewed_day.get(aid, -1):
                continue
            load = view.last_obs.load if view.last_obs is not None else None
            day = max(ins.day, self.tracker.day(aid))
            self.tracker.update(aid, day, ins.hi, True, load=load)
            self._inspected_day[aid] = obs.day

    def _allowed(self, cmd: Command, obs: Observation, proving: set[str]) -> bool:
        """Never isolate both pumps: drop a ``PROVE`` while the other pump is down."""
        if cmd.cmd is not Cmd.PROVE or cmd.asset_id not in PUMPS:
            return True
        other = PUMPS[1] if cmd.asset_id == PUMPS[0] else PUMPS[0]
        view = obs.assets.get(other)
        if other in proving or (view is not None and view.op in (Op.ISO, Op.WORK, Op.FAIL)):
            return False
        proving.add(cmd.asset_id)
        return True

    # ---------------------------------------------------------------------------------- planning
    def _constraints(
        self, obs: Observation, stalled: frozenset[str] = frozenset()
    ) -> PlanConstraints:
        opened = [v for v in obs.assets.values() if v.job_open]
        part_for = {
            (aid, m.mode): m.part_no for aid in obs.assets for m in MODES_BY_CLASS[ASSET_CLASS[aid]]
        }
        busy = {
            aid
            for aid in obs.assets
            if obs.day <= self._renewed_day.get(aid, -1)
            or (aid in self._last_raise and obs.quarter - self._last_raise[aid] < self.cooldown_q)
        }
        return PlanConstraints(
            max_open_jobs=self.params.crew_per_shift,
            pump_pair=PUMPS,
            confined_jobs=frozenset(
                k for k in part_for if (ASSET_CLASS[k[0]], k[1]) in CONFINED_JOBS
            ),
            confined_open=sum(1 for v in opened if v.job is not None and v.job.confined),
            part_for=part_for,
            on_order=frozenset(
                p for p, d in self._ordered.items() if obs.day - d < self.order_retry_days
            ),
            busy=frozenset(busy),
            no_inspect=frozenset(
                a
                for a, d in self._inspected_day.items()
                if obs.day - d < self.inspect_cooldown_days
            ),
            inspect_slots=max(obs.crew_free, 0),
            stalled=stalled,
        )

    def _mode_of(self, obs: Observation, aid: str, stalled: bool = False) -> str | None:
        view = obs.assets[aid]
        if self._diagnoser is None or self._recorder is None:
            return self.planner.map_mode(self.tracker, aid)
        history = self._recorder.history(aid)
        if history.n == 0:
            return self.planner.map_mode(self.tracker, aid)
        cached = self._diagnosed.get(aid)
        if cached is None or cached[0] != obs.day:
            cached = (obs.day, self._diagnoser.diagnose(aid, obs.day, history))
            self._diagnosed[aid] = cached
            self.diagnoses.append(cached[1])
        diagnosis = cached[1]
        if diagnosis.cause == "degradation" or view.op is Op.FAIL or stalled:
            # a failed or stalled asset is repaired whatever the diagnosis says
            return diagnosis.mode
        return None

    def _plan(self, obs: Observation, stalled: frozenset[str] = frozenset()) -> list[Command]:
        decision = self.planner.decide(
            obs,
            self.tracker,
            self._constraints(obs, stalled),
            mode_of=lambda aid: self._mode_of(obs, aid, aid in stalled),
        )
        self.decisions.append((obs.quarter, decision))
        for cmd in decision.commands:
            if cmd.cmd is PlanCmd.RAISE_WO:
                self._last_raise[cmd.asset_id] = obs.quarter
                self._pending[cmd.asset_id] = (obs.quarter, cmd)
            elif cmd.cmd is PlanCmd.ORDER and cmd.arg is not None:
                self._ordered[cmd.arg] = obs.day
        return list(decision.commands)

    def _retry(self, obs: Observation, issued: list[Command]) -> list[Command]:
        """Raise again a work order the plant has not accepted, for a couple of hours."""
        fresh = {c.asset_id for c in issued if c.cmd is PlanCmd.RAISE_WO}
        out: list[Command] = []
        for aid, (q0, cmd) in list(self._pending.items()):
            view = obs.assets.get(aid)
            if view is None or view.job_open or obs.quarter - q0 > _RETRY_Q:
                del self._pending[aid]
            elif aid not in fresh and obs.quarter > q0:
                out.append(cmd)
        return out
