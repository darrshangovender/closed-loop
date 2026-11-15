"""The perception contract: sets of health bins for the shield, and how well they hold (P4c).

Interface text: docs/adr/0002-interfaces.md, section 3.6; docs/adr/0003 A6 and A10;
docs/adr/0001-architecture.md, section 6.4. Everything here is SYNTHETIC.

``health_set`` turns a RUL interval into the pair ``(lowest, highest)`` of health bins it touches,
clipped to the range 0 to 3, and always includes the bin of the point estimate.

Two sources of the set (docs/adr/0003 A6), chosen by ``Arm.shield_set``:

``step99``  the arm's own perception variant at ``alpha = 0.01``: a per-checkpoint 99% interval.
            It misses one checkpoint in a hundred, which over a 90-day life is not 1% per life.
``life95``  the V6 per-life band at ``alpha_life = 0.05``: the probability that a life leaves the
            band at *any* checkpoint is at most 5%. C1 as the proof states it holds "at every
            tick", which is a per-life event, so this is the reading the proof uses (default).

Two truths (A10). Coverage reads the realised RUL ``true_rul_days``; C1 reads ``health_bin``, the
bin of the rate RUL, which is the abstraction's ``h``. ``contract_report`` computes the per-life
any-miss rate under both readings for both sources, side by side. A gap between the two readings is
a finding, not an error: it says a set can cover what the estimator was trained on and still miss
what the shield needs, after a load step moved the rate RUL and the realised RUL differently.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import astuple, dataclass
from typing import Any

from closed_loop.core.results import ResultRecord
from closed_loop.core.types import N_BINS, Fidelity, bins_of_interval
from closed_loop.perception.conformal import (
    ALPHA_LIFE,
    DevelopmentData,
    IntervalProvider,
)
from closed_loop.perception.coverage import select
from closed_loop.perception.labels import Checkpoint, RulInterval
from closed_loop.stats.intervals import wilson

__all__ = [
    "ALPHA_STEP",
    "SHIELD_SETS",
    "ContractCounts",
    "ContractRow",
    "bin_in_set",
    "contract_counts",
    "contract_report",
    "contract_row",
    "health_set",
    "make_shield_provider",
    "nominal_miss_level",
    "report_records",
    "shield_set_alpha",
]

ALPHA_STEP = 0.01  # miss level of the per-step set (level 0.99)
SHIELD_SETS = ("step99", "life95")


def health_set(interval: RulInterval) -> tuple[int, int]:
    """Lowest and highest health bin touched by ``interval`` and by its point estimate.

    The bin of ``m`` is included even if the interval, narrowed by an adaptive method, does not
    reach it. Bounds beyond the range (negative ``lo``, infinite ``hi``) clip to 0 and 3.
    """
    lo = min(interval.lo, interval.m)
    hi = max(interval.hi, interval.m)
    low_bin, high_bin = bins_of_interval(lo, hi)
    return max(low_bin, 0), min(high_bin, N_BINS - 1)


def bin_in_set(health_bin: int, bins: tuple[int, int]) -> bool:
    """Whether the true bin lies in the set (contract C1 at one tick)."""
    return bins[0] <= health_bin <= bins[1]


def _check_set(shield_set: str) -> None:
    if shield_set not in SHIELD_SETS:
        raise ValueError(f"shield_set must be one of {SHIELD_SETS}, got {shield_set!r}")


def shield_set_alpha(shield_set: str) -> float:
    """Miss level of the source: 0.01 for ``step99``, ``alpha_life`` (0.05) for ``life95``."""
    _check_set(shield_set)
    return ALPHA_STEP if shield_set == "step99" else ALPHA_LIFE


def nominal_miss_level(shield_set: str) -> float:
    """``p0`` for the coverage monitor of this source (docs/adr/0003 A6): the nominal miss level of
    the set being monitored. For ``life95`` it is ``alpha_life``, an upper bound on the mean of the
    per-life fraction of missed checkpoints."""
    return shield_set_alpha(shield_set)


def make_shield_provider(
    shield_set: str,
    *,
    variant: str,
    development: DevelopmentData,
    seed: int,
    **kw: Any,
) -> IntervalProvider:
    """The interval provider behind a shield set.

    ``step99`` is ``variant`` (the arm's perception, ``V1`` to ``V5`` or ``V7``) at alpha 0.01 and
    ``life95`` is the V6 band at ``alpha_life`` whatever the variant. Its level is
    ``1 - shield_set_alpha(shield_set)``.
    """
    _check_set(shield_set)
    if shield_set == "life95":
        return development.provider("V6", alpha=ALPHA_LIFE, seed=seed, **kw)
    return development.provider(variant, alpha=ALPHA_STEP, seed=seed, **kw)


# ----------------------------------------------------------------------------- the report


@dataclass(frozen=True)
class ContractCounts:
    """Counts for one source on one population; add the counts of episodes to pool them."""

    lives: int = 0
    lives_miss_rul: int = 0  # a life with a checkpoint whose realised RUL is outside [lo, hi]
    lives_miss_bin: int = 0  # a life with a checkpoint whose health bin is outside the set
    checkpoints: int = 0
    miss_rul: int = 0
    miss_bin: int = 0
    width_sum: int = 0  # sum of set widths in bins (hi - lo + 1)

    def __add__(self, other: ContractCounts) -> ContractCounts:
        return ContractCounts(*(a + b for a, b in zip(astuple(self), astuple(other), strict=True)))


def contract_counts(checkpoints: Sequence[Checkpoint], population: str = "all") -> ContractCounts:
    """Counts of the two readings for the checkpoints of ONE episode (a life is
    ``(asset_id, life_id)``). Every checkpoint needs both sim-only truths."""
    cps = select(checkpoints, population)
    lives: dict[tuple[str, int], tuple[bool, bool]] = {}
    miss_rul = miss_bin = width = 0
    for cp in cps:
        if cp.true_rul is None or cp.health_bin is None:
            raise ValueError(
                f"checkpoint ({cp.asset_id}, day {cp.day}) lacks true_rul or health_bin: the "
                "contract report needs both truths"
            )
        bins = health_set(RulInterval(cp.lo, cp.m, cp.hi, cp.level, cp.prior))
        m_rul = not (cp.lo <= cp.true_rul <= cp.hi)
        m_bin = not bin_in_set(cp.health_bin, bins)
        miss_rul += m_rul
        miss_bin += m_bin
        width += bins[1] - bins[0] + 1
        had_rul, had_bin = lives.get((cp.asset_id, cp.life_id), (False, False))
        lives[(cp.asset_id, cp.life_id)] = (had_rul or m_rul, had_bin or m_bin)
    return ContractCounts(
        lives=len(lives),
        lives_miss_rul=sum(a for a, _ in lives.values()),
        lives_miss_bin=sum(b for _, b in lives.values()),
        checkpoints=len(cps),
        miss_rul=miss_rul,
        miss_bin=miss_bin,
        width_sum=width,
    )


@dataclass(frozen=True)
class ContractRow:
    """The report for one shield-set source: both readings, side by side."""

    shield_set: str
    counts: ContractCounts

    @staticmethod
    def _rate(k: int, n: int) -> float:
        return k / n if n else math.nan

    @property
    def any_miss_rul(self) -> float:
        """Per-life any-miss rate, coverage reading (realised RUL, A10)."""
        return self._rate(self.counts.lives_miss_rul, self.counts.lives)

    @property
    def any_miss_bin(self) -> float:
        """Per-life any-miss rate, C1 reading (health bin of the rate RUL, A10)."""
        return self._rate(self.counts.lives_miss_bin, self.counts.lives)

    @property
    def gap(self) -> float:
        """C1 reading minus coverage reading: positive when the set misses what the shield needs
        more often than it misses what the estimator was trained on."""
        return self.any_miss_bin - self.any_miss_rul

    @property
    def step_miss_rul(self) -> float:
        return self._rate(self.counts.miss_rul, self.counts.checkpoints)

    @property
    def step_miss_bin(self) -> float:
        return self._rate(self.counts.miss_bin, self.counts.checkpoints)

    @property
    def mean_width(self) -> float:
        """Mean number of health bins in the set."""
        return self._rate(self.counts.width_sum, self.counts.checkpoints)

    def wilson_rul(self) -> tuple[float, float]:
        return wilson(self.counts.lives_miss_rul, self.counts.lives)

    def wilson_bin(self) -> tuple[float, float]:
        return wilson(self.counts.lives_miss_bin, self.counts.lives)


def contract_row(
    shield_set: str, checkpoints: Sequence[Checkpoint], population: str = "all"
) -> ContractRow:
    _check_set(shield_set)
    return ContractRow(shield_set, contract_counts(checkpoints, population))


def contract_report(
    step99: Sequence[Checkpoint] | ContractCounts,
    life95: Sequence[Checkpoint] | ContractCounts,
    population: str = "all",
) -> Mapping[str, ContractRow]:
    """Both sources, both readings. Each argument is the checkpoints of one episode, or the
    pooled ``ContractCounts`` of many."""
    rows: dict[str, ContractRow] = {}
    for name, data in (("step99", step99), ("life95", life95)):
        counts = data if isinstance(data, ContractCounts) else contract_counts(data, population)
        rows[name] = ContractRow(name, counts)
    return rows


def report_records(
    report: Mapping[str, ContractRow], *, measured_on: str, script: str, fidelity: Fidelity
) -> list[ResultRecord]:
    """``ResultRecord``s for the per-life any-miss rates (Wilson 95% intervals) of every source and
    reading. Rows with no life are left out."""
    out: list[ResultRecord] = []
    for name, row in report.items():
        n = row.counts.lives
        if n == 0:
            continue
        for reading, rate, (lo, hi) in (
            ("rul", row.any_miss_rul, row.wilson_rul()),
            ("bin", row.any_miss_bin, row.wilson_bin()),
        ):
            out.append(
                ResultRecord(
                    name=f"contract.{name}.any_miss_{reading}",
                    value=rate,
                    n=n,
                    fidelity=fidelity,
                    measured_on=measured_on,
                    script=script,
                    ci_lo=lo,
                    ci_hi=hi,
                    ci_kind="wilson-0.95",
                    extra={
                        "step_miss": row.step_miss_rul if reading == "rul" else row.step_miss_bin,
                        "checkpoints": row.counts.checkpoints,
                        "mean_width_bins": row.mean_width,
                    },
                )
            )
    return out
