"""P11c: one test per mismatch and sibling defect of docs/FINDINGS.md, run against the installed
siblings, and the F1-F2 estimator mismatch for 20 development lives.

A failing test here means a sibling has changed (or a claim in FINDINGS.md was wrong): read the
finding again before trusting the table. Nothing is mocked; every check runs the real sibling code.
"""

from __future__ import annotations

import math

import pytest

from closed_loop.core.results import ResultRecord
from closed_loop.core.types import Fidelity
from closed_loop.f2.mismatch import (
    CHECKS,
    Check,
    LifeMismatch,
    dev_lives,
    estimator_mismatch,
    mismatch_record,
    run_check,
    run_checks,
)

pytestmark = pytest.mark.f2

MISMATCHES = [f"M{i}" for i in range(1, 22)]
DEFECTS = [f"S{i}" for i in range(1, 13)]


def test_there_is_a_check_for_each_of_the_33_findings_and_for_nothing_else() -> None:
    assert sorted(CHECKS) == sorted(MISMATCHES + DEFECTS)


@pytest.mark.parametrize("check_id", MISMATCHES + DEFECTS)
def test_the_finding_holds_in_the_installed_siblings(check_id: str) -> None:
    check = run_check(check_id)
    assert isinstance(check, Check) and check.id == check_id
    assert check.holds, check.evidence
    assert len(check.evidence) > 20  # it says what was observed


def test_an_unknown_id_names_the_known_ones() -> None:
    with pytest.raises(KeyError, match="M1"):
        run_check("M99")


def test_run_checks_returns_them_in_numerical_order() -> None:
    out = run_checks(["S2", "M10", "M2"])
    assert [c.id for c in out] == ["S2", "M10", "M2"]


@pytest.fixture(scope="module")
def lives() -> list[tuple[int, str]]:
    return dev_lives(20)


@pytest.fixture(scope="module")
def rows(lives: list[tuple[int, str]]) -> list[LifeMismatch]:
    return estimator_mismatch(lives)


class TestEstimatorMismatchOnTwentyLives:
    def test_the_lives_are_twenty_failing_first_lives_on_development_seeds(
        self, lives: list[tuple[int, str]]
    ) -> None:
        assert len(lives) == len(set(lives)) == 20
        assert all(1000 <= seed <= 1999 for seed, _ in lives)
        assert lives == sorted(
            lives, key=lambda x: (x[0], ["CR-501", "CV-101", "PU-201", "PU-202"].index(x[1]))
        )

    def test_every_life_has_days_on_which_both_estimators_gave_a_trend(
        self, rows: list[LifeMismatch]
    ) -> None:
        assert len(rows) == 20
        assert all(r.days >= 5 and r.mae_f1 > 0.0 and r.mae_f2 > 0.0 for r in rows)
        assert all(math.isfinite(r.diff) for r in rows)

    def test_the_result_record_is_valid_labelled_f2_and_says_what_it_was_measured_on(
        self, rows: list[LifeMismatch]
    ) -> None:
        rec = mismatch_record(rows)
        assert isinstance(rec, ResultRecord)
        assert rec.name == "f2.estimator_mismatch.mae_diff_days"
        assert rec.fidelity is Fidelity.F2 and rec.n == 20
        assert "development" in rec.measured_on and "20 first lives" in rec.measured_on
        assert rec.script == "closed_loop.f2.mismatch"
        assert rec.value == pytest.approx(sum(r.diff for r in rows) / 20)
        assert (
            rec.ci_lo is not None and rec.ci_hi is not None and rec.ci_lo <= rec.value <= rec.ci_hi
        )
        assert rec.extra["checkpoints"] == sum(r.days for r in rows)
        assert rec.extra["lives_f2_worse"] == sum(1 for r in rows if r.diff > 0)

    def test_the_difference_is_f2_minus_f1(self) -> None:
        row = LifeMismatch(1000, "CR-501", "LINER_WEAR", 10, 4.0, 7.0)
        assert row.diff == 3.0

    def test_an_empty_set_of_lives_is_refused_rather_than_averaged(self) -> None:
        with pytest.raises(ValueError, match="no life"):
            mismatch_record([])

    def test_held_out_seeds_are_refused(self) -> None:
        with pytest.raises(ValueError, match="development seeds"):
            dev_lives(5, first_seed=2000)
