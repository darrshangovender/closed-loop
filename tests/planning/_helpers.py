"""Shared fixtures for the planning tests. SYNTHETIC data only."""

from __future__ import annotations

import numpy as np

from closed_loop.planning.worldmodel import PlantWorldModel, TabularWorldModel

# A hand-written fitted mapping (no simulator): the pump class with the catalogue-like ranges.
FITTED: dict[str, float] = {
    "hi_base": 1.0,
    "hi_fail": 9.0,
    "hi_noise_sd": 0.3,
    "hi_load_coef": 2.0,
    "beta_load": 0.8,
    "horizon_days": 90.0,
    "first_p_in": 0.6,
    "first_onset_min": 24.0,
    "first_fail_hi": 160.0,
    "renew_onset_lo": 24.0,
    "renew_onset_hi": 60.0,
    "SLURRY_PUMP.IMPELLER_WEAR.weight": 0.7,
    "SLURRY_PUMP.IMPELLER_WEAR.dur_lo": 40.0,
    "SLURRY_PUMP.IMPELLER_WEAR.dur_hi": 75.0,
    "SLURRY_PUMP.IMPELLER_WEAR.k_lo": 1.2,
    "SLURRY_PUMP.IMPELLER_WEAR.k_hi": 3.0,
    "SLURRY_PUMP.BEARING_DEFECT.weight": 0.3,
    "SLURRY_PUMP.BEARING_DEFECT.dur_lo": 20.0,
    "SLURRY_PUMP.BEARING_DEFECT.dur_hi": 45.0,
    "SLURRY_PUMP.BEARING_DEFECT.k_lo": 2.0,
    "SLURRY_PUMP.BEARING_DEFECT.k_hi": 4.5,
}


def plant_model(mismatch: float = 0.0, **overrides: float) -> PlantWorldModel:
    return PlantWorldModel({**FITTED, **overrides}, mismatch, asset_class="SLURRY_PUMP")


def quiet_model(mismatch: float = 0.0) -> PlantWorldModel:
    """No rate drift, no rate spread and no load wander: the dynamics are deterministic."""
    return plant_model(mismatch, rate_sd=0.0, rate_walk_sd=0.0, load_sd=0.0)


COST = {"continue": 0.0, "inspect": 0.05, "replace": 1.0, "failure": 8.0}


def small_tabular(inspect_sharper: bool = True) -> TabularWorldModel:
    """Four levels (0-2 working, 3 failed and absorbing), three actions, three symbols."""
    cont = np.array(
        [
            [0.8, 0.2, 0.0, 0.0],
            [0.0, 0.8, 0.2, 0.0],
            [0.0, 0.0, 0.8, 0.2],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    repl = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (4, 1))
    T = np.stack([cont, cont, repl])
    O = np.array(  # noqa: E741
        [
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.3, 0.6],
            [0.05, 0.05, 0.9],
        ]
    )
    O_ins = None
    if inspect_sharper:
        O_ins = np.array(
            [
                [0.9, 0.08, 0.02],
                [0.05, 0.9, 0.05],
                [0.02, 0.18, 0.8],
                [0.01, 0.01, 0.98],
            ]
        )
    return TabularWorldModel(T, O, COST, O_inspect=O_ins)
