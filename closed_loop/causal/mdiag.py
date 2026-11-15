"""M-DIAG: per-alert correct cause and mode when the load-to-mode correlation is reversed.

The metric of 0003 A3 and the pre-registration (section 5.1): for every alert the plant raises on a
held-out life, 1 if both the diagnosed ``cause`` and the diagnosed ``mode`` equal the truth, else 0;
the cell value is the mean over alerts. The truth is the plant's own label of the alert
(``AlertTruth``): the largest contributor to the health index on the alert day, and the failure mode
of the component. The mode is scored on every alert, also on the few whose cause is a sensor fault
or a process upset, where the mode is the life's latent mode and no reading can show it before
wear begins; that reading of the text is the literal one, and ``breakdown`` separates the cases so
the effect of it can be seen.

*How the shift is built.* ``PlantParams.mode_load_tilt`` tilts the failure mode by the asset's fixed
load level (heavy assets get wear modes when the tilt is positive). A diagnoser is trained on lives
from a plant with one tilt and tested on lives from a plant with the other sign (``reversed``) or
the same (``unshifted``). The two diagnosers see the same lives and the same labelled alerts. No
decision rule is applied in this module: the win and loss thresholds of 0003 A3 belong to the
pre-registered run. It returns the numbers, with confidence sequences for the paired per-alert
difference ``scm - corr``.

*Seeds.* ``collect_lives`` takes a ``role``: ``dev`` (1000-1999, the default: the development
study), ``calib`` (2000-2999: the final training set, prereg section 3) or ``heldout`` (the derived
64-bit seeds of the registered run, which are none of the fixed ranges). A seed outside its role is
refused, so a held-out seed cannot train anything and a development seed cannot pose as a test.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from closed_loop.causal.diagnoser import Diagnoser, Diagnosis, OracleDiagnoser
from closed_loop.causal.records import LifeRecord, ObservationHistory
from closed_loop.causal.twin import passive_world
from closed_loop.core.results import ResultRecord
from closed_loop.core.types import Fidelity
from closed_loop.plant import PlantParams
from closed_loop.plant.params import ASSET_CLASS
from closed_loop.stats import betting_cs, paired_cs

SCRIPT = "closed_loop.causal.mdiag"
DEV_SEEDS = range(1000, 2000)
# The log-odds of a wear mode against a bearing mode rise by 2 T x level, and the per-asset load
# level has a standard deviation of 0.08, so T = 8 is a slope of 1.3 per standard deviation of
# load level: a heavily loaded asset is wear-prone (0.89) and a lightly loaded one is not (0.39).
# An assumption (A), fixed before any run and not tuned; the study script also sweeps stronger
# values, labelled exploratory.
DEFAULT_TILT = 8.0


@dataclass(frozen=True, eq=False)
class AlertCase:
    """One alert with the history a controller would hold when it was raised."""

    seed: int
    asset_id: str
    day: int
    level: int
    truth_cause: str
    truth_mode: str
    history: ObservationHistory
    life: int = -1  # the index of the life among those the cases came from (-1: its own life)

    @property
    def asset_class(self) -> str:
        return ASSET_CLASS[self.asset_id]


def diagnosis_params(tilt: float, base: PlantParams | None = None) -> PlantParams:
    """The F1 plant of the diagnosis study with the load-to-mode tilt set."""
    p = dataclasses.replace(base if base is not None else PlantParams(), mode_load_tilt=tilt)
    p.validate()
    return p


_FIXED_ROLES = (DEV_SEEDS, range(2000, 3000), range(3000, 4000))
ROLES = ("dev", "calib", "heldout")


def _check_role(seeds: Sequence[int], role: str) -> None:
    if role == "dev":
        bad = [s for s in seeds if s not in DEV_SEEDS]
    elif role == "calib":
        bad = [s for s in seeds if s not in range(2000, 3000)]
    elif role == "heldout":
        bad = [s for s in seeds if any(s in r for r in _FIXED_ROLES)]
    else:
        raise ValueError(f"role must be one of {ROLES}, got {role!r}")
    if bad and role == "dev":
        raise ValueError(f"M-DIAG uses development seeds (1000-1999) only, got {bad[:3]}")
    if bad:
        raise ValueError(f"M-DIAG seeds of role {role!r}: {bad[:3]} belong to another role")


def collect_lives(
    seeds: Sequence[int], params: PlantParams, *, role: str = "dev"
) -> list[LifeRecord]:
    """First lives of the four assets for each seed, with nobody acting, labelled by post-mortem.

    Uses ``twin.passive_world``, which equals the stepped plant episode (and is tested to).
    ``role`` is the seed role the seeds must belong to (see the module docstring).
    """
    _check_role(seeds, role)
    out: list[LifeRecord] = []
    for s in seeds:
        out.extend(passive_world(s, params))
    return out


def alert_cases(lives: Sequence[LifeRecord]) -> list[AlertCase]:
    """Every labelled alert of ``lives`` with the history up to its day."""
    cases = []
    for index, lf in enumerate(lives):
        for a in lf.alerts:
            cases.append(
                AlertCase(
                    lf.seed if lf.seed is not None else -1,
                    lf.asset_id,
                    a.day,
                    a.level,
                    a.cause,
                    a.mode,
                    lf.history_at(a.day),
                    index,
                )
            )
    return cases


@dataclass(frozen=True, eq=False)
class Outcome:
    """What one diagnoser said about a list of alert cases."""

    name: str
    cause: NDArray[np.str_]
    mode: NDArray[np.str_]
    correct: NDArray[np.bool_]  # cause and mode both right
    cause_ok: NDArray[np.bool_]
    mode_ok: NDArray[np.bool_]


def _truth_of(case: AlertCase) -> Callable[[str, int], tuple[str, str]]:
    def lookup(asset_id: str, day: int) -> tuple[str, str]:
        return case.truth_mode, case.truth_cause

    return lookup


def evaluate(diagnoser: Diagnoser, cases: Sequence[AlertCase]) -> Outcome:
    """Run ``diagnoser`` on every case. An oracle is bound to each case's own truth in turn."""
    out: list[Diagnosis] = []
    for k in cases:
        if isinstance(diagnoser, OracleDiagnoser):
            diagnoser.bind(_truth_of(k))
        out.append(diagnoser.diagnose(k.asset_id, k.day, k.history))
    cause = np.array([d.cause for d in out])
    mode = np.array([d.mode for d in out])
    t_cause = np.array([k.truth_cause for k in cases])
    t_mode = np.array([k.truth_mode for k in cases])
    c_ok, m_ok = cause == t_cause, mode == t_mode
    return Outcome(diagnoser.name, cause, mode, c_ok & m_ok, c_ok, m_ok)


def life_means(cases: Sequence[AlertCase], values: NDArray[np.bool_]) -> NDArray[np.float64]:
    """The mean of ``values`` over the alerts of each life, one number per life that has an alert,
    in the order the lives first appear in ``cases`` (pre-registration 5.1, M-DIAG).

    Alerts in one life share a mode and a cause, so they are not independent: the unit of the
    confidence sequence and of the paired difference is the life.
    """
    order: dict[int, list[float]] = {}
    for position, (case, value) in enumerate(zip(cases, values, strict=True)):
        key = case.life if case.life >= 0 else -1 - position  # an unnumbered case is its own life
        order.setdefault(key, []).append(float(value))
    return np.array([float(np.mean(v)) for v in order.values()], dtype=np.float64)


def breakdown(cases: Sequence[AlertCase], outcomes: Mapping[str, Outcome]) -> dict[str, object]:
    """Accuracies by truth cause, asset class and alert level, and cause / mode alone."""
    truth = np.array([k.truth_cause for k in cases])
    cls = np.array([k.asset_class for k in cases])
    lvl = np.array([k.level for k in cases])
    out: dict[str, object] = {"n_alerts": len(cases)}
    out["n_by_cause"] = {c: int((truth == c).sum()) for c in sorted(set(truth))}
    for name, o in outcomes.items():
        deg = truth == "degradation"
        out[name] = {
            "both": float(o.correct.mean()),
            "cause_only": float(o.cause_ok.mean()),
            "mode_only": float(o.mode_ok.mean()),
            "mode_on_degradation": float(o.mode_ok[deg].mean()) if deg.any() else None,
            "both_on_degradation": float(o.correct[deg].mean()) if deg.any() else None,
            "by_cause": {c: float(o.correct[truth == c].mean()) for c in sorted(set(truth))},
            "by_class": {c: float(o.correct[cls == c].mean()) for c in sorted(set(cls))},
            "by_level": {int(v): float(o.correct[lvl == v].mean()) for v in sorted(set(lvl))},
        }
    return out


def records(
    setting: str,
    cases: Sequence[AlertCase],
    outcomes: Mapping[str, Outcome],
    *,
    alpha: float = 0.05,
    measured_on: str,
    n_lives: int | None = None,
) -> dict[str, ResultRecord]:
    """Accuracy of each diagnoser and the paired difference ``scm - corr``, as result records.

    The unit is the life: a life's value is the mean of its alerts' scores, ``n`` is the number of
    lives with an alert, and ``n_lives`` (the lives that were run, if the caller knows) gives the
    count of lives excluded for having no alert. ``extra["alerts"]`` is the number of alerts.
    """
    recs: dict[str, ResultRecord] = {}
    n_alerts = len(cases)
    n = int(life_means(cases, np.ones(n_alerts, dtype=bool)).size)
    excluded = {} if n_lives is None else {"lives_without_alert": n_lives - n}
    for name, o in outcomes.items():
        x = life_means(cases, o.correct)
        low, up = betting_cs(x, alpha)
        recs[name] = ResultRecord(
            name=f"M-DIAG.{name}.{setting}",
            value=float(x.mean()),
            n=n,
            fidelity=Fidelity.F1,
            measured_on=measured_on,
            script=SCRIPT,
            ci_lo=float(low[-1]),
            ci_hi=float(up[-1]),
            ci_kind=f"confseq-{alpha:g}",
            extra={
                "cause_only": float(life_means(cases, o.cause_ok).mean()),
                "mode_only": float(life_means(cases, o.mode_ok).mean()),
                "alerts": n_alerts,
                "alert_accuracy": float(o.correct.mean()),
                **excluded,
            },
        )
    if "scm" in outcomes and "corr" in outcomes:
        a = life_means(cases, outcomes["scm"].correct)
        b = life_means(cases, outcomes["corr"].correct)
        lo, hi = paired_cs(a, b, 0.0, 1.0, alpha)
        recs["scm-corr"] = ResultRecord(
            name=f"M-DIAG.scm-corr.{setting}",
            value=float(np.mean(a - b)),
            n=n,
            fidelity=Fidelity.F1,
            measured_on=measured_on,
            script=SCRIPT,
            ci_lo=lo,
            ci_hi=hi,
            ci_kind=f"confseq-{alpha:g}",
            extra={
                "scm_only_right": int(np.sum(a > b)),  # lives where the SCM scored higher
                "corr_only_right": int(np.sum(a < b)),
                "alerts": n_alerts,
                **excluded,
            },
        )
    return recs
