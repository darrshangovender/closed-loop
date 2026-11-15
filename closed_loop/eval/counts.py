"""Every count, target and registered level of the benchmark, in one place (packet P12b).

The pre-registration (sections 3, 6 and 7) fixes maximum counts per cell, the batch size, the
half-width targets of sequential stopping and the levels of the confidence sequences; 0001 section
13 says how ``--quick`` shrinks them. All of it is here, so that the single pre-freeze adjustment of
``n_max`` that 0002 section 5 allows (after the timing review) is a change to this file and to
nothing else. The numbers marked (A) in the pre-registration are assumptions until that review.

Nothing here runs anything. ``Counts.quick()`` is the ``--quick`` reading: every count divided by
25 with a floor of 4, except the counts 0001 section 13 states outright (F2 limited to 3
episodes and 3 replays), the search cut, the F0 cut and the cap on the F1 cells described at
``Counts.quick``.

The maxima are those of the pre-registration after 0003 A14: 500 for rule arms, perception variants
and hostile arms, 300 for planner arms, 25 F2 replays and 200 evaluations per search run.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Final

__all__ = [
    "ALPHA_DESCRIPTIVE",
    "ALPHA_PERC",
    "ALPHA_PRIMARY",
    "FULL",
    "N_PRIMARY_CLAIMS",
    "QUICK",
    "QUICK_CELL_CAP",
    "QUICK_DIVISOR",
    "QUICK_FIRST_DEV_SEED",
    "QUICK_FLOORS",
    "QUICK_MIN",
    "MON_ARM",
    "QUICK_SEARCH_BUDGET",
    "QUICK_SEARCH_RUNS",
    "Counts",
    "StopRule",
    "stop_rule_for",
]

N_PRIMARY_CLAIMS: Final[int] = 9  # prereg section 6: Bonferroni over the nine primary claims
ALPHA_DESCRIPTIVE: Final[float] = 0.05  # secondary intervals, "descriptive" (prereg section 6)
ALPHA_PRIMARY: Final[float] = 0.05 / N_PRIMARY_CLAIMS  # each primary claim is judged at this level
ALPHA_PERC: Final[float] = 0.05 / (2 * N_PRIMARY_CLAIMS)  # C-PERC has two verdicts

QUICK_DIVISOR: Final[int] = 25  # 0001 section 13: "all counts divided by 25 (minimum 4)"
QUICK_MIN: Final[int] = 4
# --quick draws its episodes from ``dev`` seeds, and the planner's world model is fitted on dev
# seeds 1000 to 1099 (``ContextSpec.world_model_seeds``): start above them so that no quick
# episode is in-sample for the planner.
QUICK_FIRST_DEV_SEED: Final[int] = 1100
# A cut the floor of 4 cannot make: the correlational effect model needs 9 training lives that
# raised an alert (``causal.correlational``), and about 60% of lives do, so 12 lives (300 / 25)
# fail to fit. 40 lives fit.
QUICK_FLOORS: Final[dict[str, int]] = {"causal_train_lives": 40}
# The cut that brings ``--quick`` towards the 8 minutes of 0002 P12b: 500 / 25 = 20 episodes per
# cell ran well over that with two workers on the shared laptop (the measured times are in
# docs/adr/progress/P12b-runner.md), so the F1 cells are capped at 12 (three batches of 4). The
# cap is on the cells only: the conformance traces, the exact-DP beliefs and the causal study
# keep their divided counts.
QUICK_CELL_CAP: Final[int] = 12
# The search is a stage 0001 section 13 gives its own quick count (30 evaluations). An
# evaluation is a whole F1 episode of the shielded hostile arm, 2.5 to 3.4 s under load (P9b), so
# 4 runs of 30 in each of four cells would be about 28 CPU-minutes against the 8 minutes of the
# whole run. Two runs of 15 is a little over one generation of CMA-ES (population 14): enough to
# take every code path, far too few to say anything about CMA-ES.
QUICK_SEARCH_RUNS: Final[int] = 2
QUICK_SEARCH_BUDGET: Final[int] = 15
# The arm whose cells carry C-MON (the planner with the shield and passive monitors, 0003 A15):
# fixed at 700 unshifted and 300 shifted episodes with no early stop, because at the claim level
# the zero-event upper bound reaches the false-alarm limit of 0.02 only at about 600 episodes.
MON_ARM: Final[str] = "P1"
_CELL_COUNTS: Final[tuple[str, ...]] = (
    "n_max_perception",
    "n_max_policy",
    "n_max_planner",
    "n_max_hostile",
    "open_loop_seeds",
    "mon_nominal",
    "mon_shifted",
)


@dataclass(frozen=True)
class Counts:
    """The counts of one run. Defaults are the pre-registered maxima (prereg section 3)."""

    batch: int = 50  # episodes per batch; stopping and the commit-charge check act at its boundary
    # ---- F1 cells (prereg section 3 after 0003 A14; sequential stopping usually uses fewer)
    n_max_perception: int = 500  # B3 under each perception variant, nominal and shifted
    n_max_policy: int = 500  # the rule arms: B0, B1, B2 and B3-S
    n_max_planner: int = 300  # the planner arms and their ablations (0003 A14)
    mon_nominal: int = 700  # C-MON, ``MON_ARM`` unshifted: fixed, no early stop (0003 A15)
    mon_shifted: int = 300  # C-MON, ``MON_ARM`` shifted: fixed, no early stop
    n_max_hostile: int = 500  # hostile-proposer and specification arms, nominal only
    open_loop_seeds: int = 500  # open-loop perception: seeds (four first lives each)
    # ---- causal study (prereg section 3: 600 lives, 300 train and 300 test)
    causal_train_lives: int = 300  # effect models: observational training lives
    causal_test_seeds: int = 75  # effect models: test seeds, about four lives each
    mdiag_train_seeds: int = 300  # M-DIAG: training seeds (calib), four lives each
    mdiag_test_seeds: int = 300  # M-DIAG: test seeds, reversed and unreversed
    # ---- conformance and exact DP (0002 P5c, 0001 section 8)
    conformance_episodes: int = 200
    exact_dp_beliefs: int = 50
    # ---- search (prereg section 3 and M-FIND: 10 runs per regime and algorithm, 200 evaluations)
    # ``search_budget`` is the one place the number lives: the stage, the censoring of M-FIND and
    # the scale of the evaluations difference of C-SEARCH all read it
    search_runs: int = 10
    search_budget: int = 200
    # ---- F2 (prereg section 3: 60 episodes per arm, up to 25 replays)
    f2_episodes: int = 60
    f2_replays: int = 25
    # ---- F0 (0001 section 13 row 1): the widths of perception set the checker walks, the game
    # of the superseded clause, and the sabotage suite (a pytest run of about two minutes)
    f0_widths: tuple[int, ...] = (1, 2, 3, 4)
    f0_stale: bool = True
    f0_sabotage: bool = True
    # ---- operational limits, not statistics
    episode_timeout_s: float = 900.0  # a longer episode counts as failed (worst value)

    def n_max(self, group: str) -> int:
        """The cell maximum of a group: ``perception``, ``policy`` (the rule arms), ``planner``
        or ``hostile``."""
        try:
            return {
                "perception": self.n_max_perception,
                "policy": self.n_max_policy,
                "planner": self.n_max_planner,
                "hostile": self.n_max_hostile,
            }[group]
        except KeyError:
            raise ValueError(f"unknown cell group {group!r}") from None

    def cell_limit(self, arm_name: str, condition: str, group: str) -> int:
        """The episodes of one cell: ``MON_ARM`` has its own two fixed numbers (C-MON), every other
        cell the maximum of its group."""
        if arm_name == MON_ARM:
            return self.mon_shifted if condition == "shifted" else self.mon_nominal
        return self.n_max(group)

    @property
    def largest_cell(self) -> int:
        """How many F1 seeds a run needs: the largest ``n_max`` (cells share a prefix of them)."""
        return max(
            self.n_max_perception,
            self.n_max_policy,
            self.n_max_planner,
            self.n_max_hostile,
            self.open_loop_seeds,
            self.mon_nominal,
            self.mon_shifted,
            self.conformance_episodes,
        )

    def quick(self) -> Counts:
        """The ``--quick`` counts: each divided by 25 with a floor of 4 (0001 section 13), but
        F2 limited to 3 episodes and 3 replays, and the search cut as ``QUICK_SEARCH_*`` says."""
        scaled = {
            f.name: max(QUICK_FLOORS.get(f.name, QUICK_MIN), getattr(self, f.name) // QUICK_DIVISOR)
            for f in fields(self)
            if isinstance(getattr(self, f.name), int)
            and not isinstance(getattr(self, f.name), bool)
        }
        scaled.update(
            f2_episodes=3,
            f2_replays=3,
            search_budget=QUICK_SEARCH_BUDGET,
            search_runs=QUICK_SEARCH_RUNS,
        )
        for name in _CELL_COUNTS:
            scaled[name] = min(scaled[name], QUICK_CELL_CAP)
        # F0 is exhaustive, so it cannot be scaled; the quick run keeps the widest sets (width 4)
        # and drops the sabotage suite and the stale-specification game
        return replace(self, **scaled, f0_widths=(4,), f0_stale=False, f0_sabotage=False)


FULL: Final[Counts] = Counts()
QUICK: Final[Counts] = FULL.quick()


@dataclass(frozen=True)
class StopRule:
    """Sequential stopping of one cell (prereg section 7).

    A cell stops at the first batch boundary where the half-width of the confidence sequence of
    ``metric`` is at most ``target``, or at ``n_max``. With ``zero_event_runs_on`` a proportion cell
    that has seen no event never stops early: "a 'no violations' claim needs the data". A ``fixed``
    cell never stops early (the C-MON cells).
    """

    metric: str
    target: float
    zero_event_runs_on: bool = False
    fixed: bool = False


# Targets, prereg section 7 after 0003 A15: 0.01 for M-COV and M-COST (half the smallest decision
# margin, 0.02, so that a stopped cell can still resolve its claim); 0.05 for proportions where the
# decision margin is 0.05, otherwise the margin; zero-event cells run to n_max; the C-MON cells do
# not stop at all.
_COVERAGE: Final[StopRule] = StopRule("m_cov", 0.01)
_COST: Final[StopRule] = StopRule("m_cost", 0.01)
_VIOLATIONS: Final[StopRule] = StopRule("m_viol_p13", 0.05, zero_event_runs_on=True)
_MONITOR: Final[StopRule] = StopRule("m_cost", 0.0, fixed=True)


def stop_rule_for(arm_name: str) -> StopRule:
    """The pre-registered primary metric and target of an arm's cells.

    Perception cells (B3 under a variant) are judged on coverage, hostile and specification arms on
    the P1-P3 violation proportion, every other arm on M-COST, and the C-MON arm does not stop. By
    name, so that this module needs none of the heavy ones.
    """
    if arm_name == MON_ARM:
        return _MONITOR
    if arm_name.startswith("B3-V"):
        return _COVERAGE
    if arm_name.startswith(("H-", "S-")):
        return _VIOLATIONS
    return _COST
