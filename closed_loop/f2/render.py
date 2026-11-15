"""Render one plant component life as asset-health telemetry (packet P11a). Everything is SYNTHETIC.

The F2 plant is the owned simulator of ``closed_loop.plant``. What asset-health sees is telemetry
that ``asset_health.simulate.fleet.simulate_asset`` writes from the plant's latent failure event
(onset, failure, shape, mode). The rendering follows four rules, each settled by the spike
``docs/adr/spikes/splice_lives.py`` and its output:

* **One full-horizon record per life, cut afterwards.** The generator draws its regime and noise
  sequentially, so a record of another length would redraw every row. A life is rendered once for
  the whole episode from a keyed stream and then cut by time, so what the controller has already
  seen never depends on when the policy chooses to stop (the causal caching rule, 0001 section 3).
* **Calendar alignment.** Records start at midnight on ``SIM_EPOCH`` so the generator's diurnal,
  weekly and seasonal phases line up with the calendar in every life. The plant's quarter 0 is
  06:00 (``PLANT_START``).
* **The logger clock is not rendered** (``DataFaultConfig.clock_drift_asset_fraction = 0``). A clock
  model cannot be continued across a splice (each record would draw its own drift), and rows are
  cut by reported timestamp, which is only safe while that stays within seconds of the true slot.
  The pipeline's retime path is therefore not exercised by F2 (a limitation, recorded).
* **No planned services.** The plant has no lubrication rounds, so the maintenance log carries
  repair records only, at the joins.

The generator offers no hook for a stop or for a different severity curve, so two adaptations are
made in the open and counted as findings: a stop is a gap of missing rows (the pipeline treats it as
it treats a communications outage), and the plant's power-law severity ``x^k`` is mapped to the
generator's ``(e^(kappa u) - 1) / (e^kappa - 1)`` by a least-squares choice of kappa
(``asset_health_shape``). Neither is the plant's wear clock: a stopped asset does not age in the
plant and does age on the generator's wall clock.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any

import numpy as np

from closed_loop.core.rng import rng_for
from closed_loop.f2._siblings import load
from closed_loop.plant.params import ASSET_CLASS, ASSET_IDS, MODES_BY_CLASS, PlantParams

# Midnight: the generator's phases (diurnal, weekly, seasonal) are those of the calendar.
SIM_EPOCH = datetime(2026, 10, 1, 0, 0, 0)
# Quarter 0 of the plant (06:00, 0002 section 3.2).
PLANT_START = SIM_EPOCH + timedelta(hours=6)

# Plant asset -> asset-health asset type (0001, section 10.2).
ASSET_TYPE: dict[str, str] = {
    "CR-501": "primary_crusher",
    "CV-101": "conveyor_pulley",
    "PU-201": "slurry_pump",
    "PU-202": "slurry_pump",
}
# Plant (maintenance-agent) failure mode -> asset-health failure mode (0001, section 10.3, item 3).
FAILURE_MODE: dict[str, str] = {
    "LINER_WEAR": "liner_wear",
    "IMPELLER_WEAR": "impeller_wear",
    "BEARING_DEFECT": "bearing_spall",
}
_HALF_SLOT = timedelta(minutes=30)


class RenderError(ValueError):
    """A life cannot be rendered as asset-health telemetry."""


def sim_time(day: float) -> datetime:
    """Wall-clock instant of plant day ``day`` (day 0 starts at ``PLANT_START``)."""
    return PLANT_START + timedelta(days=day)


def sim_day(instant: datetime) -> float:
    """Plant day of a wall-clock instant."""
    return (instant - PLANT_START) / timedelta(days=1)


def _ceil_hour(instant: datetime) -> datetime:
    floor = instant.replace(minute=0, second=0, microsecond=0)
    return floor if instant == floor else floor + timedelta(hours=1)


def asset_health_id(asset_id: str) -> str:
    """The asset-health id of a plant asset: ``ID_PREFIX[type]`` and a two-digit ordinal.

    ``CR-501`` is ``CRU-01``, ``CV-101`` is ``PUL-01``, ``PU-201`` and ``PU-202`` are ``PMP-01`` and
    ``PMP-02``. This answers the open item of 0001 section 15 (3): the prefix table has entries for
    the crusher and the pulley (asset_health/simulate/fleet.py, ``ID_PREFIX``).
    """
    if asset_id not in ASSET_TYPE:
        raise RenderError(f"unknown plant asset {asset_id!r}; known: {sorted(ASSET_TYPE)}")
    models = load("asset_health.models")
    fleet = load("asset_health.simulate.fleet")
    atype = models.AssetType(ASSET_TYPE[asset_id])
    same_type = [a for a in ASSET_IDS if ASSET_TYPE[a] == ASSET_TYPE[asset_id]]
    return f"{fleet.ID_PREFIX[atype]}-{same_type.index(asset_id) + 1:02d}"


@lru_cache(maxsize=256)
def asset_health_shape(plant_shape: float) -> float:
    """The generator's severity exponent whose curve is closest to the plant's ``x^k``.

    The plant's severity is ``u^k`` (0001 section 2 says this is the asset-health curve; the
    generator's ``severity_curve`` is in fact ``(exp(kappa u) - 1) / (exp(kappa) - 1)``). The two
    families differ, so kappa is chosen by least squares over ``u`` in [0, 1].
    """
    if plant_shape <= 0:
        raise RenderError(f"shape must be positive, got {plant_shape}")
    u = np.linspace(0.0, 1.0, 201)
    target = u**plant_shape

    def best(grid: np.ndarray) -> float:
        curves = np.expm1(np.outer(grid, u)) / np.expm1(grid)[:, None]
        return float(grid[np.argmin(np.mean((curves - target) ** 2, axis=1))])

    coarse = best(np.linspace(0.05, 25.0, 2000))
    return best(np.linspace(max(coarse - 0.05, 0.01), coarse + 0.05, 2001))


@dataclass(frozen=True)
class LifeLatent:
    """The sim-only description of one component life, as the plant knows it.

    Days are plant days (day 0 starts at ``PLANT_START``). ``onset_day`` and ``failure_day`` are the
    wall-clock days on which the wear clock reaches the onset and the failure in the world where
    nobody acts; the failure may lie beyond the horizon. ``start_day`` and ``end_day`` choose the
    piece of the record the pipeline is shown: the asset runs from ``start_day`` (a restart after a
    job) until it stops at ``end_day`` (a replacement, a job, or the failure; ``None`` is the end of
    the episode). ``repair_at_start`` writes a CMMS repair record at ``start_day``: always true for
    a new life, and true for a job that completed on the same component (a wrong-mode job, whose
    record the pipeline cannot tell from a real repair).
    """

    asset_id: str
    life_id: int
    mode: str
    onset_day: float
    failure_day: float
    shape: float
    downtime_hours: float = 48.0
    start_day: float = 0.0
    end_day: float | None = None
    repair_at_start: bool = False

    def __post_init__(self) -> None:
        if self.asset_id not in ASSET_CLASS:
            raise RenderError(f"unknown plant asset {self.asset_id!r}")
        modes = {m.mode for m in MODES_BY_CLASS[ASSET_CLASS[self.asset_id]]}
        if self.mode not in modes:
            raise RenderError(
                f"{self.asset_id} ({ASSET_CLASS[self.asset_id]}) has no failure mode {self.mode!r};"
                f" it has {sorted(modes)}"
            )
        if self.life_id < 0:
            raise RenderError("life_id must be non-negative")
        if not self.failure_day > self.onset_day:
            raise RenderError("failure_day must be after onset_day")
        if self.shape <= 0 or self.downtime_hours <= 0:
            raise RenderError("shape and downtime_hours must be positive")
        if self.start_day < 0:
            raise RenderError("start_day must be non-negative")
        if self.end_day is not None and not self.end_day > self.start_day:
            raise RenderError("end_day must be after start_day")
        if not self.stop_day > self.start_day:
            raise RenderError("the piece would start after the life has failed")

    @property
    def stop_day(self) -> float:
        """Last day this piece can run: the planned stop, never past the failure."""
        return self.failure_day if self.end_day is None else min(self.end_day, self.failure_day)


@dataclass(frozen=True, eq=False)
class RenderedTelemetry:
    """One piece of one life: the rows an asset-health historian export would hold.

    ``frame`` is the dirty export (reported timestamps, data faults) cut to ``[start, stop)``.
    ``failure`` is the generator's ground truth for the whole life and is never given to the
    pipeline. ``events`` are the data faults the generator injected (also ground truth).
    """

    latent: LifeLatent
    segment: int
    ah_asset_id: str
    spec: Any
    frame: Any
    start: datetime
    stop: datetime
    failure: Any
    events: tuple[Any, ...]

    @property
    def n_rows(self) -> int:
        return int(len(self.frame))

    @property
    def life_id(self) -> int:
        return self.latent.life_id


@dataclass(frozen=True, eq=False)
class _Record:
    spec: Any
    dirty: Any
    failure: Any
    events: tuple[Any, ...]


@lru_cache(maxsize=256)
def _spec_for(seed: int, asset_id: str) -> Any:
    """The asset-health ``AssetSpec`` of a plant asset: one per asset and episode seed."""
    models = load("asset_health.models")
    catalog = load("asset_health.catalog")
    atype = models.AssetType(ASSET_TYPE[asset_id])
    profile = catalog.PROFILES[atype]
    scale = float(np.exp(rng_for(seed, "f2.spec", asset_id).normal(0.0, 0.22)))
    ordinal = [a for a in ASSET_IDS if ASSET_TYPE[a] == ASSET_TYPE[asset_id]].index(asset_id) + 1
    return catalog.spec_from_profile(
        profile,
        asset_id=asset_health_id(asset_id),
        name=profile.typical_names[0].format(n=ordinal),
        baseline_scale=scale,
    )


@lru_cache(maxsize=256)
def _full_record(
    seed: int,
    asset_id: str,
    life_id: int,
    mode: str,
    onset_day: float,
    failure_day: float,
    shape: float,
    downtime_hours: float,
    horizon_days: int,
) -> _Record:
    """The whole-episode record of one life. Depends on nothing about where it will be cut."""
    models = load("asset_health.models")
    config = load("asset_health.config")
    fleet = load("asset_health.simulate.fleet")
    faults = load("asset_health.simulate.faults")
    spec = _spec_for(seed, asset_id)
    fmode = models.FailureMode(FAILURE_MODE[mode])
    draw = rng_for(seed, "f2.life", asset_id, life_id)
    amplitude = float(np.clip(np.exp(draw.normal(0.0, 0.3)), 0.55, 1.8))
    outer = bool(draw.random() < 0.6)
    race = ("outer" if outer else "inner") if fmode == models.FailureMode.BEARING_SPALL else None
    failure = models.FailureEvent(
        asset_id=spec.asset_id,
        mode=fmode,
        race=race,
        onset=sim_time(onset_day),
        failure=sim_time(failure_day),
        shape_k=asset_health_shape(round(shape, 6)),
        amplitude=amplitude,
        downtime_hours=downtime_hours,
    )
    cfg = config.SimConfig(
        seed=seed,
        start=SIM_EPOCH,
        days=max(horizon_days + 1, 61),  # SimConfig needs at least baseline_days + 40
        fleet={spec.asset_type: 1},
        faults=config.DataFaultConfig(clock_drift_asset_fraction=0.0, clock_steps_max=0),
    )
    stream = rng_for(seed, "f2.telemetry", asset_id, life_id)
    clean, _latent, _repair = fleet.simulate_asset(stream, cfg, spec, failure, [])
    dirty, events = faults.apply_data_faults(
        stream, clean, spec.asset_id, cfg.faults, cfg.sample_interval_s
    )
    return _Record(spec, dirty, failure, tuple(events))


def render_life(
    latent: LifeLatent, params: PlantParams, seed: int, segment: int
) -> RenderedTelemetry:
    """The telemetry of one piece of one life.

    Deterministic in ``(seed, latent)``. The record of a life depends on its identity, onset,
    failure, shape and mode only; ``start_day`` and ``end_day`` merely cut it, so cutting the same
    life earlier gives a prefix of the longer piece (tested). ``segment`` is the position of the
    piece in the asset's history (0 for the first); it labels the piece and is checked by
    ``splice``, it does not enter the random streams.
    """
    pd = load("pandas")
    if segment < 0:
        raise RenderError("segment must be non-negative")
    horizon = params.horizon_days
    rec = _full_record(
        seed,
        latent.asset_id,
        latent.life_id,
        latent.mode,
        round(latent.onset_day, 9),
        round(latent.failure_day, 9),
        round(latent.shape, 9),
        latent.downtime_hours,
        horizon,
    )
    start = _ceil_hour(sim_time(latent.start_day))
    stop = _ceil_hour(sim_time(min(latent.stop_day, float(horizon))))
    stamps = rec.dirty["timestamp"]
    keep = (stamps >= pd.Timestamp(start - _HALF_SLOT)) & (stamps < pd.Timestamp(stop - _HALF_SLOT))
    return RenderedTelemetry(
        latent=latent,
        segment=segment,
        ah_asset_id=rec.spec.asset_id,
        spec=rec.spec,
        frame=rec.dirty.loc[keep].reset_index(drop=True),
        start=start,
        stop=stop,
        failure=rec.failure,
        events=rec.events,
    )
