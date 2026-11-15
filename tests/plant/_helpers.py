"""Shared helpers for the plant tests. SYNTHETIC data only."""

import numpy as np

from closed_loop.plant.params import PlantParams


def quiet_params(**overrides: object) -> PlantParams:
    """Parameters with every noise source and data-quality event switched off.

    Only the structural equations remain, so monotonicity and exact-slope tests can be exact.
    """
    base: dict[str, object] = {
        "hi_noise_sd": 0.0,
        "sig_noise_sd": 0.0,
        "load_noise_sd": 0.0,
        "load_level_sd": 0.0,
        "load_obs_sd": 0.0,
        "upset_p_per_circuit": 0.0,
        "p_gain_fault": 0.0,
        "p_stuck_day": 0.0,
        "p_spike_day": 0.0,
    }
    base.update(overrides)
    return PlantParams(**base)  # type: ignore[arg-type]


def theta_of(**coords: float) -> np.ndarray:
    """A theta vector at 0.5 with the named coordinates set: ``theta_of(c3=0.9)`` sets index 3.

    The sensor-fault type coordinates (3, 8, 13, 18) default to 0.1, which is "no fault", so a test
    that sets one asset's fault does not also fault the others.
    """
    theta = np.full(32, 0.5)
    theta[[3, 8, 13, 18]] = 0.1
    for name, value in coords.items():
        theta[int(name.lstrip("c"))] = value
    return theta


_PER_ASSET = (
    "op", "permit", "occupied", "gas_age_min", "alarm", "iso_proven", "permit_margin_min",
    "true_rul_days", "health_bin", "set_lo", "set_hi", "uncontrolled_failure",
)  # fmt: skip


def asset_arrays_equal(a, b, asset_id: str, fields: tuple[str, ...] = _PER_ASSET) -> bool:
    """True when every per-asset trace array of ``asset_id`` is bit-identical in the two traces."""
    n = min(a.n_q, b.n_q)
    return all(
        getattr(a, f)[asset_id][:n].tobytes() == getattr(b, f)[asset_id][:n].tobytes()
        for f in fields
    )


def trace_digest(tr) -> str:
    """SHA-256 of every array of a Trace and of its command log, for cross-process comparison."""
    import hashlib

    h = hashlib.sha256()
    h.update(str((tr.n_q, tr.asset_ids)).encode())
    for aid in tr.asset_ids:
        for f in _PER_ASSET:
            h.update(getattr(tr, f)[aid].tobytes())
    h.update(tr.throughput.tobytes())
    for q, cmd, ok, over in tr.commands:
        h.update(f"{q}|{cmd.asset_id}|{cmd.cmd}|{cmd.arg}|{cmd.source}|{ok}|{over}".encode())
    return h.hexdigest()
