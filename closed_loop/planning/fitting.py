"""Fit the parameters of ``PlantWorldModel`` on development seeds (packet P8a).

This is the one module of the planning package that touches the simulator's truth: it runs
``simulate_lives`` on **development** seeds (1000-1999) and reads the sim-only severity and
realised load, as any tuning on ``dev`` may. It is offline: the belief tracker and the world model
never import it, and the planner's runtime path does not call it.

What is fitted, and how honest the word is. The observation relation ``hi = a + b s + c (L - 1)``
and
the load sensitivity ``beta`` are estimated by least squares; where the plant's relation is exact
(``beta``, the load coefficient) the estimate equals the generating value to rounding, so those are
*estimated* rather than *independent* parameters. What departs from the generating values on
purpose is the model structure around them: ranges are sample minima and maxima over a finite set
of lives, the per-asset rate spread and load wander are summary statistics, and the ``mismatch``
knob of ``PlantWorldModel`` puts a deliberate wear-rate error on top.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from closed_loop.plant import PlantParams, Scenario
from closed_loop.plant.observe import simulate_lives
from closed_loop.plant.params import ASSET_CLASS, ASSET_IDS

DEV_RANGE = range(1000, 2000)


def fit_world_model(
    seeds: Iterable[int],
    params: PlantParams | None = None,
    scenario: Scenario | None = None,
) -> dict[str, float]:
    """Fitted parameters for ``PlantWorldModel`` from the first lives of ``seeds`` (dev seeds)."""
    params = params if params is not None else PlantParams()
    scenario = scenario if scenario is not None else Scenario()
    seed_list = [int(s) for s in seeds]
    if not seed_list:
        raise ValueError("fit_world_model needs at least one seed")
    if any(s not in DEV_RANGE for s in seed_list):
        raise ValueError("fit_world_model may only use development seeds (1000-1999)")
    tables = [simulate_lives(params, s, scenario) for s in seed_list]

    # ---- observation model: hi = a + b * severity + c * (load - 1), robust to unmasked drift
    ys, cols = [], []
    for t in tables:
        for aid in ASSET_IDS:
            hi, sev, load = t.hi[aid], t.severity[aid], t.load_true[aid][: t.n_days]
            ok = np.isfinite(hi)
            ys.append(hi[ok])
            cols.append(np.column_stack([np.ones(ok.sum()), sev[ok], load[ok] - 1.0]))
    y = np.concatenate(ys)
    x = np.vstack(cols)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    res = y - x @ coef
    keep = np.abs(res) <= 3.0 * res.std()
    coef, *_ = np.linalg.lstsq(x[keep], y[keep], rcond=None)
    sd = float((y[keep] - x[keep] @ coef).std(ddof=3))

    # ---- load sensitivity: clock speed per day from the sim-only progress, against the load
    dl, log_rho = [], []
    for t in tables:
        for aid in ASSET_IDS:
            life = t.lives[aid]
            sev = t.severity[aid]
            prog = np.where(sev > 0.0, np.power(np.clip(sev, 1e-12, 1.0), 1.0 / life.shape), 0.0)
            inside = (prog[:-1] > 0.0) & (prog[1:] < 1.0) & (prog[1:] > prog[:-1])
            rho = life.duration * (prog[1:] - prog[:-1])
            idx = np.nonzero(inside)[0]
            dl.append(t.load_true[aid][idx] - 1.0)
            log_rho.append(np.log(rho[idx]))
    dl_all = np.concatenate(dl)
    lr_all = np.concatenate(log_rho)
    if dl_all.size >= 10:
        beta, icpt = np.polyfit(dl_all, lr_all, 1)
        rate_sd = float(np.std(lr_all - (beta * dl_all + icpt)))
    else:
        beta, rate_sd = params.beta_load, 0.0

    # ---- load wander when the load is not observed
    wander = [
        t.load_true[aid][: t.n_days] - np.mean(t.load_true[aid][: t.n_days])
        for t in tables
        for aid in ASSET_IDS
    ]
    load_sd = float(np.std(np.concatenate(wander)))

    # ---- lives: failure-mode weights and ranges per class, first-life timing
    fitted: dict[str, float] = {
        "hi_base": float(coef[0]),
        "hi_fail": float(coef[0] + coef[1]),
        "hi_load_coef": float(coef[2]),
        "hi_noise_sd": sd,
        "beta_load": float(beta),
        "rate_sd": max(rate_sd, 0.02),
        "load_sd": load_sd,
    }
    horizon = float(params.horizon_days)
    firsts = [(aid, t) for t in tables for aid in ASSET_IDS]
    fail_days = np.array([t.failure_day[aid] for aid, t in firsts])
    onsets = np.array([t.onset_day[aid] for aid, t in firsts])
    fitted["horizon_days"] = horizon
    fitted["first_p_in"] = float(np.mean(fail_days < horizon))
    fitted["first_onset_min"] = float(onsets.min())
    fitted["first_fail_hi"] = float(fail_days.max())
    renew_lo, renew_hi = params.renewal_onset_days
    fitted["renew_onset_lo"], fitted["renew_onset_hi"] = float(renew_lo), float(renew_hi)

    for cls in sorted(set(ASSET_CLASS.values())):
        lives = [t.lives[aid] for aid, t in firsts if ASSET_CLASS[aid] == cls]
        for mode in sorted({lf.mode for lf in lives}):
            sub = [lf for lf in lives if lf.mode == mode]
            key = f"{cls}.{mode}"
            fitted[f"{key}.weight"] = len(sub) / len(lives)
            durs = np.array([lf.duration for lf in sub])
            ks = np.array([lf.shape for lf in sub])
            fitted[f"{key}.dur_lo"], fitted[f"{key}.dur_hi"] = float(durs.min()), float(durs.max())
            fitted[f"{key}.k_lo"], fitted[f"{key}.k_hi"] = float(ks.min()), float(ks.max())
    return fitted
