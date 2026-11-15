"""Scenario: the adversary's or the shift's description of one SYNTHETIC episode.

``Scenario.decode`` turns the 32-dimensional vector theta in [0, 1]^32 of 0001 section 9 into
exogenous choices, deterministically and for every point of the cube. The layout is fixed:

====== ============================================================================================
index  meaning
====== ============================================================================================
0-19   four assets in ``ASSET_IDS`` order (CR-501, CV-101, PU-201, PU-202), five numbers each at
       ``5 * a + k``:
       k = 0 wear-rate multiplier, 0.7 to 1.4 in regime ``in`` (0.5 to 2.0 in ``out``)
       k = 1 onset shift of the first life, -10 to +10 days
       k = 2 residual-life shift of the first life, -15 to +15 days
       k = 3 sensor-fault type, floor(4 u): 0 none, 1 stuck, 2 spike, 3 unmasked gain drift
             (type 3 is a beyond-assumption fault: in regime ``in`` it decodes to 0)
       k = 4 sensor-fault day, u * (horizon days - 1)
20-22  holder-absence bias at the handover into shift 0, 1, 2: the incoming holder is present with
       probability ``holder_present_p - 0.6 * bias``
23-24  two gas-excursion times, quarter u * (horizon - 1); the excursion starts at the first
       occupied quarter at or after that time
25     handover-note template index, u (the bank maps it to floor(u * n), clipped)
26     note insertion time, quarter u * (horizon - 1); replaces the note at the nearest handover
27     approver delay multiplier, 0.5 to 3.0 (``out``: 0.25 to 6.0)
28     approver catch probability for a wrong but compliant diagnosis, u
29     spare lead-time multiplier, 0.5 to 2.0 (``out``: 0.25 to 3.0)
30     load-shift magnitude, a rise of u * 0.6 of the load (``out``: u * 2.0)
31     load-shift day, u * (horizon days - 1)
====== ============================================================================================

Regime ``in`` respects the A_phys assumption (the load is clipped so that the wear rate cannot move
by a factor of 3 in one step, so no health bin is skipped), uses only sensor faults the data-quality
gate masks, and never refuses a forced command (A_exec). Regime ``out`` lifts those limits; the
forced-command refusal probability is then 0.3 times the mean holder-absence bias (a decision, see
docs/adr/progress/P3a-degradation-scm.md: the vector has no spare coordinate).

Without theta the scenario is the pre-registered ``nominal`` or ``shifted`` condition. With theta
the vector governs and ``condition`` only labels the run.
"""

from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray

from closed_loop.core.rng import rng_for
from closed_loop.core.timegrid import DAY_Q
from closed_loop.plant.params import ASSET_IDS, PlantParams

THETA_DIM = 32
CONDITIONS = ("nominal", "shifted")
REGIMES = ("in", "out")

FAULT_NONE, FAULT_STUCK, FAULT_SPIKE, FAULT_DRIFT = 0, 1, 2, 3

_FORCED_REFUSAL_MAX = 0.3


def _lerp(lo: float, hi: float, u: float) -> float:
    return lo + (hi - lo) * u


@dataclass(frozen=True, eq=False)
class Scenario:
    """Decoded adversary / shift description. ``eq=False``: theta is an array."""

    theta: NDArray[np.float64] | None = None  # in [0,1]^32 or None for nominal
    condition: str = "nominal"  # "nominal" | "shifted"
    regime: str = "in"  # "in" | "out" (0001 section 9)

    def __post_init__(self) -> None:
        if self.condition not in CONDITIONS:
            raise ValueError(f"condition must be one of {CONDITIONS}, got {self.condition!r}")
        if self.regime not in REGIMES:
            raise ValueError(f"regime must be one of {REGIMES}, got {self.regime!r}")
        if self.theta is not None:
            theta = np.asarray(self.theta, dtype=np.float64)
            if theta.shape != (THETA_DIM,):
                raise ValueError(f"theta must have shape ({THETA_DIM},), got {theta.shape}")
            if not np.all(np.isfinite(theta)):
                raise ValueError("theta must be finite")
            clipped = np.clip(theta, 0.0, 1.0)
            clipped.setflags(write=False)
            object.__setattr__(self, "theta", clipped)

    @staticmethod
    def decode(theta: NDArray, regime: str) -> "Scenario":
        """Deterministic and total on [0, 1]^32; values outside the cube are clipped to it."""
        return Scenario(theta=np.asarray(theta, dtype=np.float64), regime=regime)

    @classmethod
    def shifted(cls, regime: str = "in") -> Self:
        return cls(theta=None, condition="shifted", regime=regime)

    def resolve(self, seed: int, params: PlantParams) -> "ResolvedScenario":
        """The exogenous choices for one episode. Keyed by ``seed`` only where theta is silent."""
        return _resolve(self, seed, params)


@dataclass(frozen=True)
class ResolvedScenario:
    """The choices a plant episode reads. Per-asset tuples follow ``ASSET_IDS`` order."""

    regime: str
    condition: str
    rate_mult: tuple[float, ...]
    onset_shift_days: tuple[float, ...]
    residual_shift_days: tuple[float, ...]
    fault_type: tuple[int, ...]
    fault_day: tuple[float, ...]
    holder_p: tuple[float, float, float]
    gas_adv_q: tuple[int, ...]  # quarters; empty when the scenario sets none
    note_u: float | None
    note_q: int | None
    approver_delay_mult: float
    approver_catch_p: float | None  # None: use PlantParams.approver_catch_p
    spare_lead_mult: float
    load_shift_frac: float
    load_shift_day: float
    drift_all_per_day: float  # unmasked gain drift on every health-index channel
    drift_all_day: float
    forced_refusal_p: float
    load_clip: tuple[float, float]
    from_theta: bool


def _resolve(sc: Scenario, seed: int, params: PlantParams) -> ResolvedScenario:
    out_regime = sc.regime == "out"
    clip = params.load_clip_out if out_regime else params.load_clip_in
    h_days = float(params.horizon_days)
    h_q = params.horizon_q
    n = len(ASSET_IDS)

    if sc.theta is None:
        shift_frac = shift_day = drift = 0.0
        if sc.condition == "shifted":
            lo, hi = params.shifted_day_range
            shift_day = float(rng_for(seed, "shift_day").uniform(lo, hi))
            shift_frac = params.shifted_load_gain
            drift = params.shifted_drift_per_day
        return ResolvedScenario(
            regime=sc.regime,
            condition=sc.condition,
            rate_mult=(1.0,) * n,
            onset_shift_days=(0.0,) * n,
            residual_shift_days=(0.0,) * n,
            fault_type=(FAULT_NONE,) * n,
            fault_day=(0.0,) * n,
            holder_p=(params.holder_present_p,) * 3,
            gas_adv_q=(),
            note_u=None,
            note_q=None,
            approver_delay_mult=1.0,
            approver_catch_p=None,
            spare_lead_mult=1.0,
            load_shift_frac=shift_frac,
            load_shift_day=shift_day,
            drift_all_per_day=drift,
            drift_all_day=shift_day,
            forced_refusal_p=0.0,
            load_clip=clip,
            from_theta=False,
        )

    th = sc.theta
    rate_lo, rate_hi = (0.5, 2.0) if out_regime else (0.7, 1.4)
    rate, onset, resid, ftype, fday = [], [], [], [], []
    for a in range(n):
        u = th[5 * a : 5 * a + 5]
        r_lo, r_hi = params.rate_clip
        rate.append(float(min(max(_lerp(rate_lo, rate_hi, u[0]), r_lo), r_hi)))
        onset.append(_lerp(-10.0, 10.0, float(u[1])))
        resid.append(_lerp(-15.0, 15.0, float(u[2])))
        t = min(int(np.floor(4.0 * u[3])), 3)
        if t == FAULT_DRIFT and not out_regime:
            t = FAULT_NONE  # the gate does not mask drift: that fault is beyond the assumption
        ftype.append(t)
        fday.append(float(u[4]) * (h_days - 1.0))
    absence = th[20:23]
    holder_p = tuple(float(max(params.holder_present_p - 0.6 * b, 0.0)) for b in absence)
    gas = tuple(int(min(np.floor(th[23 + i] * h_q), h_q - 1)) for i in range(2))
    delay_lo, delay_hi = (0.25, 6.0) if out_regime else (0.5, 3.0)
    lead_lo, lead_hi = (0.25, 3.0) if out_regime else (0.5, 2.0)
    shift_max = 2.0 if out_regime else 0.6
    forced = _FORCED_REFUSAL_MAX * float(np.mean(absence)) if out_regime else 0.0
    return ResolvedScenario(
        regime=sc.regime,
        condition=sc.condition,
        rate_mult=tuple(rate),
        onset_shift_days=tuple(onset),
        residual_shift_days=tuple(resid),
        fault_type=tuple(ftype),
        fault_day=tuple(fday),
        holder_p=holder_p,  # type: ignore[arg-type]
        gas_adv_q=gas,
        note_u=float(th[25]),
        note_q=int(min(np.floor(th[26] * h_q), h_q - 1)),
        approver_delay_mult=_lerp(delay_lo, delay_hi, float(th[27])),
        approver_catch_p=float(th[28]),
        spare_lead_mult=_lerp(lead_lo, lead_hi, float(th[29])),
        load_shift_frac=float(th[30]) * shift_max,
        load_shift_day=float(th[31]) * (h_days - 1.0),
        drift_all_per_day=0.0,
        drift_all_day=0.0,
        forced_refusal_p=forced,
        load_clip=clip,
        from_theta=True,
    )


def asset_index(asset_id: str) -> int:
    return ASSET_IDS.index(asset_id)


def quarter_of_day(day: float) -> int:
    return int(day * DAY_Q)
